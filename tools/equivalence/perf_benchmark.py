"""CUDA performance benchmark for the HYDRA Suite inference pipeline.

Measures frames/second across a matrix of:
  pipeline_depth ∈ {1, 2, 4}  (sync / double-buffer / deep-prefetch)
  nvdec           ∈ {on, off}  (hardware decode vs. cv2 CPU decode)
  trt             ∈ {on, off}  (TensorRT/ONNX vs. PyTorch)

For each combination the script:
  1. Builds a fresh InferenceConfig with the requested settings.
  2. Runs InferenceRunner.run_batch_pass over the clip in a fresh temp dir
     (so no detection-cache reuse skews timing).
  3. Repeats --repeats times after --warmup warm-up runs, reports median fps.

Gate (exit code):
  Best accelerated config (NVDEC on + TRT on + depth≥2) must be faster than
  the unaccelerated baseline (NVDEC off + TRT off + depth=1).
  If --baseline-fps is supplied, the best config must also meet or exceed it.

Unavailable combos (e.g. NVDEC on a CPU-only box, or TRT without a .pt file
to auto-export from) are skipped and logged; the gate considers only the
combos that actually ran.

Usage
-----
python tools/equivalence/perf_benchmark.py \\
    --video <path> \\
    --config <path> \\
    --depths 1,2,4 \\
    --nvdec on,off \\
    --trt on,off \\
    --warmup 1 \\
    --repeats 3

Dry-run (builds config matrix, no execution):
    ... --dry-run

Help:
    python tools/equivalence/perf_benchmark.py --help
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import statistics
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

# Conda/torch builds often link libomp twice; without this, OpenMP aborts the
# process ("OMP Error #15"). Must be set before torch is imported.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
# Headless Qt (some transitive imports pull in Qt)
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s %(name)s: %(message)s",
)
log = logging.getLogger("perf_benchmark")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "--video",
        metavar="PATH",
        help="Path to an H.264/H.265 video clip to benchmark. "
        "Required unless --dry-run is given.",
    )
    ap.add_argument(
        "--config",
        metavar="PATH",
        help="Path to a hydra_suite InferenceConfig JSON. "
        "The script overrides pipeline_depth, compute_runtime, and "
        "auto_export/nvdec settings per combo. "
        "Required unless --dry-run is given.",
    )
    ap.add_argument(
        "--depths",
        default="1,2,4",
        metavar="INT[,INT...]",
        help="Comma-separated pipeline_depth values to benchmark. "
        "depth=1 → synchronous; depth=2 → double-buffer; "
        "depth>2 → deep-prefetch queue of depth-1 windows. "
        "Default: 1,2,4",
    )
    ap.add_argument(
        "--nvdec",
        default="on,off",
        metavar="on|off[,...]",
        help="Comma-separated NVDEC on/off toggle(s). "
        "NVDEC requires a CUDA box + PyNvVideoCodec + H.264/H.265 input. "
        "Default: on,off",
    )
    ap.add_argument(
        "--trt",
        default="on,off",
        metavar="on|off[,...]",
        help="Comma-separated TRT/ONNX on/off toggle(s). "
        "TRT requires a CUDA box and a .pt model to auto-export from. "
        "Default: on,off",
    )
    ap.add_argument(
        "--baseline-fps",
        type=float,
        default=None,
        metavar="FLOAT",
        help="Known legacy production throughput (frames/sec). "
        "If provided, the gate also requires the best config to meet or "
        "exceed this value.",
    )
    ap.add_argument(
        "--warmup",
        type=int,
        default=1,
        metavar="N",
        help="Number of warm-up runs per combo (results discarded). Default: 1",
    )
    ap.add_argument(
        "--repeats",
        type=int,
        default=3,
        metavar="N",
        help="Number of timed repeats per combo. Median fps is reported. Default: 3",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Build and print the config matrix without executing any runs. "
        "Exits 0. Useful on boxes without a real video/model.",
    )
    return ap.parse_args(argv)


# ---------------------------------------------------------------------------
# Config matrix
# ---------------------------------------------------------------------------


@dataclass
class ComboSpec:
    depth: int
    nvdec: bool  # request NVDEC (honoured only when available)
    trt: bool  # request TensorRT/ONNX-CUDA (honoured only when available)

    @property
    def label(self) -> str:
        nvdec_s = "nvdec=on" if self.nvdec else "nvdec=off"
        trt_s = "trt=on" if self.trt else "trt=off"
        return f"depth={self.depth}  {nvdec_s}  {trt_s}"

    @property
    def is_baseline(self) -> bool:
        """True for the fully unaccelerated combo (depth=1, nvdec=off, trt=off)."""
        return self.depth == 1 and not self.nvdec and not self.trt

    @property
    def is_best_accelerated(self) -> bool:
        """True for the target accelerated combo (depth≥2, nvdec=on, trt=on)."""
        return self.depth >= 2 and self.nvdec and self.trt


def _build_combo_matrix(
    depths: list[int], nvdec_flags: list[bool], trt_flags: list[bool]
) -> list[ComboSpec]:
    combos: list[ComboSpec] = []
    for d in depths:
        for n in nvdec_flags:
            for t in trt_flags:
                combos.append(ComboSpec(depth=d, nvdec=n, trt=t))
    return combos


def _parse_bool_list(s: str) -> list[bool]:
    parts = [p.strip().lower() for p in s.split(",")]
    result: list[bool] = []
    for p in parts:
        if p in ("on", "true", "1", "yes"):
            result.append(True)
        elif p in ("off", "false", "0", "no"):
            result.append(False)
        else:
            raise argparse.ArgumentTypeError(f"Expected on/off values, got: {p!r}")
    return result


def _parse_int_list(s: str) -> list[int]:
    return [int(p.strip()) for p in s.split(",")]


# ---------------------------------------------------------------------------
# Runtime / config wiring
# ---------------------------------------------------------------------------


def _patch_combo_into_config(
    base_cfg_dict: dict,
    combo: ComboSpec,
) -> dict:
    """Return a modified copy of base_cfg_dict with combo settings applied.

    - pipeline_depth → combo.depth
    - When trt=on: set all compute_runtime fields to "tensorrt", auto_export=True
    - When trt=off (and not CUDA): set to "cpu"
    - When trt=off but NVDEC=on: set to "cuda" (NVDEC + PyTorch CUDA, no TRT)
    - nvdec is set via runtime.use_nvdec, which is derived automatically from
      RuntimeContext.from_config() when compute_runtime is a CUDA-group runtime.
      We don't need to set it explicitly; make_frame_source reads runtime.use_nvdec.
    """
    import copy

    cfg = copy.deepcopy(base_cfg_dict)
    cfg["pipeline_depth"] = combo.depth

    if combo.trt:
        target_runtime = "tensorrt"
    elif combo.nvdec:
        # NVDEC requires a CUDA runtime (NVDEC falls back to CPU if unavailable)
        target_runtime = "cuda"
    else:
        target_runtime = "cpu"

    # Patch all runtime fields that InferenceConfig._collect_all_runtimes() reads.
    # We don't validate runtime consistency here; InferenceConfig does it.
    _set_runtime_fields(cfg, target_runtime)
    return cfg


def _set_runtime_fields(cfg: dict, runtime: str) -> None:
    """Recursively set all compute_runtime fields to *runtime*."""
    # OBB
    obb = cfg.get("obb", {})
    if obb.get("direct"):
        obb["direct"]["compute_runtime"] = runtime
        obb["direct"]["auto_export"] = True
    if obb.get("sequential"):
        obb["sequential"]["detect_compute_runtime"] = runtime
        obb["sequential"]["obb_compute_runtime"] = runtime
        obb["sequential"]["auto_export"] = True
    # HeadTail
    if cfg.get("headtail"):
        cfg["headtail"]["compute_runtime"] = runtime
    # CNN phases
    for phase in cfg.get("cnn_phases", []):
        phase["compute_runtime"] = runtime
    # Pose
    pose = cfg.get("pose")
    if pose:
        if pose.get("yolo"):
            pose["yolo"]["compute_runtime"] = runtime
        if pose.get("sleap"):
            pose["sleap"]["compute_runtime"] = runtime


# ---------------------------------------------------------------------------
# Benchmark result
# ---------------------------------------------------------------------------


@dataclass
class RunResult:
    combo: ComboSpec
    status: str  # "ok", "skipped:<reason>", "error:<msg>"
    fps_values: list[float]  # one per timed repeat (empty if skipped/error)
    median_fps: float | None
    skip_reason: str = ""
    error_msg: str = ""

    @property
    def ok(self) -> bool:
        return self.status == "ok"

    @property
    def skipped(self) -> bool:
        return self.status.startswith("skipped")

    @property
    def errored(self) -> bool:
        return self.status.startswith("error")


# ---------------------------------------------------------------------------
# Execution
# ---------------------------------------------------------------------------


def _run_combo(
    combo: ComboSpec,
    base_cfg_dict: dict,
    video_path: Path,
    warmup: int,
    repeats: int,
    progress_cb: Callable | None = None,
) -> RunResult:
    """Time one combo; return a RunResult."""
    from hydra_suite.core.inference.config import InferenceConfig, InferenceConfigError
    from hydra_suite.core.inference.runner import InferenceRunner

    patched = _patch_combo_into_config(base_cfg_dict, combo)

    # Validate: mixed CUDA/CPU runtimes would raise inside from_json equivalent.
    try:
        cfg = _dict_to_inference_config(patched)
    except (InferenceConfigError, ValueError) as exc:
        return RunResult(
            combo=combo,
            status=f"skipped:config_invalid",
            fps_values=[],
            median_fps=None,
            skip_reason=str(exc),
        )

    # Quick availability check for CUDA/TRT combos.
    if combo.nvdec or combo.trt:
        try:
            import torch

            if not torch.cuda.is_available():
                return RunResult(
                    combo=combo,
                    status="skipped:cuda_unavailable",
                    fps_values=[],
                    median_fps=None,
                    skip_reason="CUDA not available on this box",
                )
        except ImportError:
            return RunResult(
                combo=combo,
                status="skipped:torch_missing",
                fps_values=[],
                median_fps=None,
                skip_reason="torch not installed",
            )

    fps_values: list[float] = []
    total_runs = warmup + repeats

    for run_idx in range(total_runs):
        is_warmup = run_idx < warmup
        label = "warmup" if is_warmup else f"repeat {run_idx - warmup + 1}/{repeats}"
        log.info("[%s] %s …", combo.label, label)

        cache_tmp = Path(tempfile.mkdtemp(prefix="hydra_bench_"))
        try:
            runner = InferenceRunner(cfg, cache_dir=cache_tmp, video_path=video_path)
            t0 = time.perf_counter()
            try:
                runner.run_batch_pass(video_path, progress_cb=progress_cb)
            except Exception as exc:
                runner.close()
                shutil.rmtree(cache_tmp, ignore_errors=True)
                return RunResult(
                    combo=combo,
                    status=f"error:{type(exc).__name__}",
                    fps_values=fps_values,
                    median_fps=statistics.median(fps_values) if fps_values else None,
                    error_msg=str(exc),
                )
            elapsed = time.perf_counter() - t0
            runner.close()
        finally:
            shutil.rmtree(cache_tmp, ignore_errors=True)

        # Infer frame count from the video.
        n_frames = _video_frame_count(video_path)
        fps = n_frames / elapsed if elapsed > 0 else 0.0

        if not is_warmup:
            fps_values.append(fps)
            log.info(
                "[%s] %s: %.1f fps (%.2fs for %d frames)",
                combo.label,
                label,
                fps,
                elapsed,
                n_frames,
            )

    median_fps = statistics.median(fps_values) if fps_values else None
    return RunResult(
        combo=combo,
        status="ok",
        fps_values=fps_values,
        median_fps=median_fps,
    )


def _video_frame_count(video_path: Path) -> int:
    import cv2

    cap = cv2.VideoCapture(str(video_path))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if cap.isOpened() else 0
    cap.release()
    return max(n, 1)


def _dict_to_inference_config(cfg_dict: dict):
    """Convert a raw dict to an InferenceConfig, mirroring config.py's _dict_to_config."""
    from hydra_suite.core.inference.config import _dict_to_config

    cfg = _dict_to_config(cfg_dict)
    cfg._validate_runtime_consistency()
    return cfg


# ---------------------------------------------------------------------------
# Table + gate
# ---------------------------------------------------------------------------

_COL_LABEL = 38
_COL_FPS = 10
_COL_SPEEDUP = 10
_COL_STATUS = 28


def _print_table(results: list[RunResult], baseline_fps_arg: float | None) -> None:
    hdr = (
        f"{'Config':<{_COL_LABEL}} "
        f"{'Median fps':>{_COL_FPS}} "
        f"{'Speedup':>{_COL_SPEEDUP}} "
        f"{'Status':<{_COL_STATUS}}"
    )
    sep = "-" * len(hdr)
    print()
    print(sep)
    print(hdr)
    print(sep)

    baseline_median = _get_baseline_median(results)

    for r in results:
        if r.ok and r.median_fps is not None:
            fps_str = f"{r.median_fps:.1f}"
            if baseline_median is not None and baseline_median > 0:
                speedup = r.median_fps / baseline_median
                speedup_str = f"{speedup:.2f}x"
            else:
                speedup_str = "N/A"
            status_str = "OK"
        elif r.skipped:
            fps_str = "—"
            speedup_str = "—"
            status_str = f"SKIPPED ({r.skip_reason[:24]})"
        else:
            fps_str = "ERR"
            speedup_str = "—"
            status_str = f"ERROR ({r.error_msg[:22]})"

        marker = ""
        if r.combo.is_baseline:
            marker = " [baseline]"
        elif r.combo.is_best_accelerated and r.ok:
            marker = " [best]"

        label = r.combo.label + marker
        print(
            f"{label:<{_COL_LABEL}} "
            f"{fps_str:>{_COL_FPS}} "
            f"{speedup_str:>{_COL_SPEEDUP}} "
            f"{status_str:<{_COL_STATUS}}"
        )

    print(sep)

    if baseline_fps_arg is not None:
        print(f"  Legacy baseline target:  {baseline_fps_arg:.1f} fps (--baseline-fps)")
    if baseline_median is not None:
        print(
            f"  Unaccelerated baseline:  {baseline_median:.1f} fps (depth=1, nvdec=off, trt=off)"
        )
    print()


def _get_baseline_median(results: list[RunResult]) -> float | None:
    for r in results:
        if r.combo.is_baseline and r.ok and r.median_fps is not None:
            return r.median_fps
    return None


def _pick_best_accelerated(results: list[RunResult]) -> RunResult | None:
    """Return the best-accelerated result, or fall back to best of any accelerated combo."""
    # Exact match first
    for r in results:
        if r.combo.is_best_accelerated and r.ok and r.median_fps is not None:
            return r
    # Fallback: any combo with depth>=2 that ran successfully, pick highest fps
    candidates = [
        r for r in results if r.ok and r.median_fps is not None and r.combo.depth >= 2
    ]
    if candidates:
        return max(candidates, key=lambda r: r.median_fps or 0.0)
    return None


def _evaluate_gate(
    results: list[RunResult],
    baseline_fps_arg: float | None,
) -> int:
    """Evaluate the performance gate. Returns 0 (pass) or 1 (fail)."""
    baseline_median = _get_baseline_median(results)
    best = _pick_best_accelerated(results)

    # If baseline didn't run, we can't compute the gate.
    if baseline_median is None:
        log.warning(
            "GATE: unaccelerated baseline (depth=1, nvdec=off, trt=off) did not run "
            "or errored — cannot evaluate speedup gate. "
            "Ensure nvdec=off and trt=off are in --nvdec and --trt, and depth=1 is "
            "in --depths."
        )
        return 0  # Can't fail what we didn't measure

    if best is None:
        print(
            "GATE SKIP: No accelerated combo ran successfully — cannot evaluate gate."
        )
        return 0

    speedup = (best.median_fps or 0.0) / baseline_median if baseline_median > 0 else 0.0

    passed = True
    messages: list[str] = []

    if speedup > 1.0:
        messages.append(
            f"GATE PASS: best accelerated ({best.combo.label}) = "
            f"{best.median_fps:.1f} fps, speedup = {speedup:.2f}x > 1.0 vs. baseline."
        )
    else:
        passed = False
        messages.append(
            f"GATE FAIL: best accelerated ({best.combo.label}) = "
            f"{best.median_fps:.1f} fps, speedup = {speedup:.2f}x <= 1.0 vs. baseline "
            f"({baseline_median:.1f} fps). Accelerations are NOT improving throughput."
        )

    if baseline_fps_arg is not None:
        if (best.median_fps or 0.0) >= baseline_fps_arg:
            messages.append(
                f"GATE PASS: best accelerated ({best.median_fps:.1f} fps) >= "
                f"--baseline-fps ({baseline_fps_arg:.1f} fps)."
            )
        else:
            passed = False
            messages.append(
                f"GATE FAIL: best accelerated ({best.median_fps:.1f} fps) < "
                f"--baseline-fps ({baseline_fps_arg:.1f} fps). "
                f"Pipeline has not reached legacy production throughput."
            )

    for msg in messages:
        print(msg)

    return 0 if passed else 1


# ---------------------------------------------------------------------------
# Dry-run
# ---------------------------------------------------------------------------


def _print_dry_run(combos: list[ComboSpec], base_cfg_dict: dict) -> None:
    print("\n=== DRY RUN: config matrix (no execution) ===\n")
    print(f"{'#':<4} {'Config':<45} {'target_runtime':<16} {'valid_config'}")
    print("-" * 80)
    for i, combo in enumerate(combos, 1):
        from hydra_suite.core.inference.config import InferenceConfigError

        patched = _patch_combo_into_config(base_cfg_dict, combo)
        try:
            _dict_to_inference_config(patched)
            valid = "OK"
        except (InferenceConfigError, ValueError, KeyError) as exc:
            valid = f"ERROR: {exc}"

        if combo.trt:
            rt = "tensorrt"
        elif combo.nvdec:
            rt = "cuda"
        else:
            rt = "cpu"

        print(f"{i:<4} {combo.label:<45} {rt:<16} {valid}")
    print("\nNo runs executed (--dry-run). Script exits 0.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    depths = _parse_int_list(args.depths)
    nvdec_flags = _parse_bool_list(args.nvdec)
    trt_flags = _parse_bool_list(args.trt)

    combos = _build_combo_matrix(depths, nvdec_flags, trt_flags)

    if args.dry_run:
        # Dry-run: still need a minimal config dict to show config construction.
        # If --config was given, load it; else use a minimal stub.
        if args.config:
            with open(args.config) as fh:
                base_cfg_dict = json.load(fh)
        else:
            base_cfg_dict = _minimal_stub_config()
        _print_dry_run(combos, base_cfg_dict)
        return 0

    # Real run: both --video and --config are required.
    if not args.video:
        log.error("--video is required unless --dry-run is given.")
        return 2
    if not args.config:
        log.error("--config is required unless --dry-run is given.")
        return 2

    video_path = Path(args.video).resolve()
    if not video_path.exists():
        log.error("Video not found: %s", video_path)
        return 2

    with open(args.config) as fh:
        base_cfg_dict = json.load(fh)

    print(f"\nBenchmark: {video_path.name}")
    print(f"Config:    {args.config}")
    print(f"Combos:    {len(combos)}")
    print(f"Warmup:    {args.warmup}  Repeats: {args.repeats}")
    print()

    results: list[RunResult] = []
    for combo in combos:
        log.info("--- Running: %s ---", combo.label)
        result = _run_combo(
            combo,
            base_cfg_dict,
            video_path,
            warmup=args.warmup,
            repeats=args.repeats,
        )
        results.append(result)

    _print_table(results, args.baseline_fps)
    rc = _evaluate_gate(results, args.baseline_fps)
    return rc


def _minimal_stub_config() -> dict:
    """Return a minimal InferenceConfig dict for dry-run without a real config file."""
    return {
        "obb": {
            "mode": "direct",
            "direct": {
                "model_path": "/stub.pt",
                "compute_runtime": "cpu",
                "auto_export": True,
            },
            "sequential": None,
            "target_classes": [],
            "max_detections": 20,
            "raw_detection_cap": 0,
            "min_object_size": 0.0,
            "max_object_size": None,
            "min_aspect_ratio": 0.0,
            "max_aspect_ratio": None,
            "confidence_threshold": 0.25,
            "iou_threshold": 0.7,
        },
        "headtail": None,
        "cnn_phases": [],
        "pose": None,
        "apriltag": {"enabled": False},
        "detection_batch_size": 1,
        "pipeline_depth": 2,
        "realtime": False,
        "use_cache": True,
        "cache_dir": None,
    }


if __name__ == "__main__":
    raise SystemExit(main())
