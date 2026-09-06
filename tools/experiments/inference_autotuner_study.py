#!/usr/bin/env python3
"""Run reproducible TrackerKit batch-policy experiments on real hardware.

This is a research harness, not an autotuner implementation.  Each case runs
the existing equivalence runner in a fresh process and isolated output/config
directories.  The harness records:

* the full-run timing reported by ``tools/equivalence/runner.py``;
* the authoritative forward inference span tree emitted by TrackerKit;
* peak total GPU memory, utilization, power, and temperature sampled through
  ``nvidia-smi``; and
* peak RSS for the runner process tree.

The matrix is JSON so the exact experiment can be reviewed and repeated.  A
case may override ordinary TrackerKit config fields under ``config`` and the
machine-global advanced settings under ``advanced``.  ``HYDRA_CONFIG_DIR`` is
pointed at a case-private directory, so the user's real advanced config is
never read or changed.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

REPO_ROOT = Path(__file__).resolve().parents[2]
RUNNER = REPO_ROOT / "tools" / "equivalence" / "runner.py"


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        payload = json.load(stream)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return payload


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def _slug(text: str) -> str:
    clean = "".join(char if char.isalnum() else "-" for char in text.lower())
    return "-".join(part for part in clean.split("-") if part)


def materialize_case_config(
    base: dict[str, Any],
    overrides: dict[str, Any],
    *,
    frames: int | None,
) -> dict[str, Any]:
    """Return a benchmark config without mutating the fixture config."""

    result = json.loads(json.dumps(base))
    resolved_overrides = dict(overrides)
    cnn_batch_size = resolved_overrides.pop("cnn_batch_size", None)
    result.update(resolved_overrides)
    if cnn_batch_size is not None:
        for classifier in result.get("cnn_classifiers", []):
            classifier["batch_size"] = int(cnn_batch_size)
    result["use_cached_detections"] = False
    result["enable_backward_tracking"] = False
    result["enable_postprocessing"] = False
    result["video_output_enabled"] = False
    result["final_media_export_videos_enabled"] = False
    result["enable_dataset_generation"] = False
    result["enable_individual_dataset"] = False
    result["enable_individual_image_save"] = False
    result["enable_confidence_density_map"] = False
    result["enable_profiling"] = True
    if frames is not None:
        start = int(result.get("start_frame", 0) or 0)
        result["end_frame"] = start + max(1, int(frames)) - 1
    return result


def flatten_spans(node: dict[str, Any], prefix: str = "") -> dict[str, dict[str, Any]]:
    """Flatten a profiler span tree by full path."""

    name = str(node.get("name", ""))
    path = f"{prefix}/{name}" if prefix else name
    output = {
        path: {
            key: node.get(key)
            for key in ("total_s", "self_s", "n_calls", "units", "max_s")
        }
    }
    for child in node.get("children", []):
        output.update(flatten_spans(child, path))
    return output


def _number(text: str) -> float:
    return float(text.strip().split()[0])


@dataclass
class ResourceSamples:
    gpu_memory_mib: list[float] = field(default_factory=list)
    gpu_util_percent: list[float] = field(default_factory=list)
    gpu_power_w: list[float] = field(default_factory=list)
    gpu_temperature_c: list[float] = field(default_factory=list)
    tree_rss_bytes: list[int] = field(default_factory=list)

    def summary(self) -> dict[str, Any]:
        def peak(values: Iterable[float]) -> float | None:
            values = list(values)
            return round(max(values), 3) if values else None

        def median(values: Iterable[float]) -> float | None:
            values = list(values)
            return round(statistics.median(values), 3) if values else None

        return {
            "sample_count": max(len(self.gpu_memory_mib), len(self.tree_rss_bytes), 0),
            "gpu_memory_peak_mib": peak(self.gpu_memory_mib),
            "gpu_util_median_percent": median(self.gpu_util_percent),
            "gpu_util_peak_percent": peak(self.gpu_util_percent),
            "gpu_power_peak_w": peak(self.gpu_power_w),
            "gpu_temperature_peak_c": peak(self.gpu_temperature_c),
            "tree_rss_peak_bytes": max(self.tree_rss_bytes, default=None),
        }


def _sample_gpu(samples: ResourceSamples) -> None:
    try:
        raw = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=memory.used,utilization.gpu,power.draw,temperature.gpu",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            timeout=2,
            stderr=subprocess.DEVNULL,
        )
        first = raw.splitlines()[0].split(",")
        samples.gpu_memory_mib.append(_number(first[0]))
        samples.gpu_util_percent.append(_number(first[1]))
        samples.gpu_power_w.append(_number(first[2]))
        samples.gpu_temperature_c.append(_number(first[3]))
    except (OSError, subprocess.SubprocessError, ValueError, IndexError):
        pass


def _sample_tree_rss(pid: int, samples: ResourceSamples) -> None:
    try:
        import psutil

        root = psutil.Process(pid)
        processes = [root, *root.children(recursive=True)]
        samples.tree_rss_bytes.append(
            sum(
                process.memory_info().rss
                for process in processes
                if process.is_running()
            )
        )
    except Exception:
        pass


def _monitor(pid: int, stop: threading.Event, samples: ResourceSamples) -> None:
    while not stop.wait(0.1):
        _sample_gpu(samples)
        _sample_tree_rss(pid, samples)
    _sample_gpu(samples)
    _sample_tree_rss(pid, samples)


def _profile_payload(outdir: Path) -> dict[str, Any] | None:
    candidates = sorted(outdir.glob("*_logs/tracking_profile_forward.json"))
    if not candidates:
        return None
    return _read_json(candidates[-1])


def _tail(path: Path, lines: int = 80) -> str:
    try:
        return "\n".join(path.read_text(encoding="utf-8").splitlines()[-lines:])
    except OSError:
        return ""


def run_case(
    case: dict[str, Any],
    *,
    matrix_path: Path,
    output_root: Path,
    repeat: int,
    frames: int | None,
) -> dict[str, Any]:
    label = str(case["label"])
    base_dir = matrix_path.parent
    config_path = (base_dir / str(case["config_path"])).resolve()
    video_path = (base_dir / str(case["video_path"])).resolve()
    skeleton_raw = case.get("skeleton_path")
    skeleton_path = (base_dir / str(skeleton_raw)).resolve() if skeleton_raw else None
    runtime = str(case.get("runtime", "gpu"))
    case_dir = output_root / _slug(label) / f"repeat-{repeat:02d}"
    outdir = case_dir / "run"
    config_dir = case_dir / "config-home"
    case_dir.mkdir(parents=True, exist_ok=True)
    config_dir.mkdir(parents=True, exist_ok=True)

    config = materialize_case_config(
        _read_json(config_path),
        dict(case.get("config", {})),
        frames=frames,
    )
    generated_config = case_dir / "input_config.json"
    _atomic_json(generated_config, config)
    _atomic_json(config_dir / "advanced_config.json", dict(case.get("advanced", {})))

    command = [
        sys.executable,
        str(RUNNER),
        "--orig-config",
        str(generated_config),
        "--video",
        str(video_path),
        "--outdir",
        str(outdir),
        "--runtime",
        runtime,
        "--label",
        label,
    ]
    if skeleton_path is not None:
        command.extend(("--skeleton", str(skeleton_path)))

    environment = dict(os.environ)
    environment.update(
        {
            "PYTHONPATH": str(REPO_ROOT / "src"),
            "HYDRA_CONFIG_DIR": str(config_dir),
            "KMP_DUPLICATE_LIB_OK": "TRUE",
            "QT_QPA_PLATFORM": "offscreen",
            "HYDRA_PROFILE": "1",
            "YOLO_AUTOINSTALL": "false",
            "ULTRALYTICS_SKIP_REQUIREMENTS_CHECKS": "1",
        }
    )

    log_path = case_dir / "runner.log"
    started = time.time_ns()
    samples = ResourceSamples()
    with log_path.open("w", encoding="utf-8") as log_stream:
        process = subprocess.Popen(
            command,
            cwd=REPO_ROOT,
            env=environment,
            stdout=log_stream,
            stderr=subprocess.STDOUT,
            text=True,
        )
        stop = threading.Event()
        monitor = threading.Thread(
            target=_monitor, args=(process.pid, stop, samples), daemon=True
        )
        monitor.start()
        returncode = process.wait()
        stop.set()
        monitor.join(timeout=5)
    finished = time.time_ns()

    meta_path = outdir / "meta.json"
    meta = _read_json(meta_path) if meta_path.exists() else None
    profile = _profile_payload(outdir)
    result: dict[str, Any] = {
        "label": label,
        "repeat": repeat,
        "returncode": returncode,
        "started_at_unix_ns": started,
        "finished_at_unix_ns": finished,
        "wall_seconds": round((finished - started) / 1e9, 6),
        "runtime": runtime,
        "config_overrides": dict(case.get("config", {})),
        "advanced_overrides": dict(case.get("advanced", {})),
        "resource_samples": samples.summary(),
        "output_dir": str(outdir),
        "log_path": str(log_path),
        "meta": meta,
    }
    if profile is not None:
        result["forward_profile"] = {
            key: profile.get(key)
            for key in (
                "total_frames",
                "wall_clock_s",
                "measured_s",
                "avg_fps",
                "avg_frame_ms",
                "phases",
                "config",
            )
        }
        result["spans"] = flatten_spans(profile["spans"])
    if returncode != 0:
        result["error_tail"] = _tail(log_path)
    _atomic_json(case_dir / "result.json", result)
    return result


def _selected_cases(matrix: dict[str, Any], labels: set[str]) -> list[dict[str, Any]]:
    cases = matrix.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ValueError("Matrix must contain a non-empty 'cases' list")
    selected = [case for case in cases if not labels or str(case["label"]) in labels]
    if labels:
        missing = labels - {str(case["label"]) for case in selected}
        if missing:
            raise ValueError(f"Unknown case labels: {sorted(missing)}")
    return selected


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--frames", type=int, default=None)
    parser.add_argument("--label", action="append", default=[])
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    matrix_path = args.matrix.expanduser().resolve()
    matrix = _read_json(matrix_path)
    cases = _selected_cases(matrix, set(args.label))
    if args.dry_run:
        print(json.dumps({"cases": cases}, indent=2, sort_keys=True))
        return 0

    output_root = args.output_root.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    manifest = {
        "matrix_path": str(matrix_path),
        "matrix": matrix,
        "frames_override": args.frames,
        "repeats": args.repeats,
        "python": sys.version,
        "started_at_unix_ns": time.time_ns(),
        "results": [],
    }
    manifest_path = output_root / "study_results.json"
    _atomic_json(manifest_path, manifest)

    failures = 0
    total = len(cases) * max(1, args.repeats)
    index = 0
    for case in cases:
        for repeat in range(1, max(1, args.repeats) + 1):
            index += 1
            print(f"[{index}/{total}] {case['label']} repeat={repeat}", flush=True)
            result = run_case(
                case,
                matrix_path=matrix_path,
                output_root=output_root,
                repeat=repeat,
                frames=args.frames,
            )
            manifest["results"].append(result)
            _atomic_json(manifest_path, manifest)
            if result["returncode"] != 0:
                failures += 1
                print(result.get("error_tail", "case failed"), file=sys.stderr)

    manifest["finished_at_unix_ns"] = time.time_ns()
    _atomic_json(manifest_path, manifest)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
