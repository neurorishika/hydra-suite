#!/usr/bin/env python
"""Whole-pipeline benchmark across runtime tiers.

Runs the full InferenceRunner pipeline described by a config file at each of the
three tiers (cpu / gpu / gpu_fast) and reports end-to-end per-frame timing.
Unlike the retired tools/benchmark_models.py, it does NOT benchmark individual
models or arbitrary runtime strings — tiers are the user-facing choice now.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import time
from pathlib import Path

import numpy as np

from hydra_suite.core.inference.config import build_inference_config_from_params
from hydra_suite.core.inference.runner import InferenceRunner
from hydra_suite.runtime.resolver import available_tiers, detect_platform, tier_label


# Test seam: a tiny stand-in config so unit tests need no real models.
@dataclasses.dataclass
class _Cfg:
    runtime_tier: str = "cpu"


def _synthetic_frame(frame_size: tuple[int, int]) -> np.ndarray:
    h, w = frame_size
    return np.zeros((h, w, 3), dtype=np.uint8)


def run_pipeline_benchmark(
    config_params: dict,
    *,
    tiers: list[str],
    iterations: int,
    warmup: int,
    frame_size: tuple[int, int],
    compile_timing: bool,
) -> list[dict]:
    """Time the full pipeline per tier. Returns one row dict per tier."""
    rows: list[dict] = []
    frame = _synthetic_frame(frame_size)
    for tier in tiers:
        base = build_inference_config_from_params(config_params)
        cfg = dataclasses.replace(base, runtime_tier=tier)
        row: dict = {"tier": tier}
        t_build0 = time.perf_counter()
        try:
            runner = InferenceRunner(cfg)
        except Exception as e:  # noqa: BLE001 - report, don't crash the whole sweep
            row["error"] = f"{type(e).__name__}: {e}"
            rows.append(row)
            continue
        if compile_timing:
            row["build_s"] = time.perf_counter() - t_build0
        try:
            for _ in range(max(0, warmup)):
                runner.run_realtime(frame)
            samples = []
            for _ in range(max(1, iterations)):
                t0 = time.perf_counter()
                runner.run_realtime(frame)
                samples.append((time.perf_counter() - t0) * 1000.0)
            row["mean_ms"] = float(np.mean(samples))
            row["p50_ms"] = float(np.percentile(samples, 50))
            row["p95_ms"] = float(np.percentile(samples, 95))
        except Exception as e:  # noqa: BLE001
            row["error"] = f"{type(e).__name__}: {e}"
        finally:
            runner.close()
        rows.append(row)
    return rows


def _load_config_params(path: Path) -> dict:
    return json.loads(path.read_text())


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "config", type=Path, help="Path to a JSON config file (params dict)."
    )
    parser.add_argument(
        "--tiers",
        nargs="*",
        default=None,
        help="Tiers to benchmark (default: all available on this platform).",
    )
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--frame-size", type=int, nargs=2, default=[2000, 2000])
    parser.add_argument(
        "--compile-timing",
        action="store_true",
        help="Also report per-tier runner build time (engine build / export).",
    )
    parser.add_argument("--output-json", type=Path, default=None)
    args = parser.parse_args(argv)

    platform = detect_platform()
    tiers = args.tiers or available_tiers(platform)
    params = _load_config_params(args.config)
    rows = run_pipeline_benchmark(
        params,
        tiers=tiers,
        iterations=args.iterations,
        warmup=args.warmup,
        frame_size=tuple(args.frame_size),
        compile_timing=args.compile_timing,
    )
    for r in rows:
        label = tier_label(r["tier"], platform)
        if "error" in r:
            print(f"{label:22s}  ERROR: {r['error']}")
        else:
            extra = f"  build={r['build_s']:.1f}s" if "build_s" in r else ""
            print(
                f"{label:22s}  mean={r['mean_ms']:.2f}ms  p95={r['p95_ms']:.2f}ms{extra}"
            )
    if args.output_json:
        args.output_json.write_text(json.dumps(rows, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
