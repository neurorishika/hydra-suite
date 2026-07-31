"""Optimization + runtime microbenchmark for the HYDRA Suite inference stages.

Answers two questions empirically, per device:

  (A) RUNTIME VALUE  — for the same real model, how do native torch
      (cpu/mps/cuda) vs onnx_* vs tensorrt compare in latency? Decides whether
      onnx_cpu/onnx_cuda/onnx_coreml/tensorrt earn their keep or are just
      confusing user options.

  (B) EXACT-WIN HEADROOM — how much do the *determinism-preserving* torch
      optimizations buy on this hardware:
        - no_grad -> inference_mode
        - contiguous -> channels_last (NHWC)
        - pageable -> pinned + non_blocking H2D copy
      Measured at the torch level on a representative conv net (efficientnet_b0)
      and on a YOLO-OBB forward, since these timings depend on architecture +
      hardware, not on trained weights.

This script NEVER mutates repo state and skips any runtime/device that is
unavailable on the box (logged as SKIP). Run the same script on the Mac (MPS)
and on the CUDA box; diff the tables.

Usage:
    PYTHONPATH=src python tools/equivalence/opt_microbench.py \
        --classifier "/path/to/efficientnet_b0.pth" \
        --obb "/path/to/yolo-obb.pt" \
        --batch 64 --warmup 3 --repeats 20 --json out.json
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import torch  # noqa: E402


# ─────────────────────────────────────────────────────────────────────────────
# helpers
# ─────────────────────────────────────────────────────────────────────────────
def _sync(device: str) -> None:
    if device == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize()
    elif device == "mps" and torch.backends.mps.is_available():
        torch.mps.synchronize()


def _time_ms(fn, device: str, warmup: int, repeats: int) -> dict:
    for _ in range(warmup):
        fn()
    _sync(device)
    samples = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        _sync(device)
        samples.append((time.perf_counter() - t0) * 1000.0)
    return {
        "median_ms": round(statistics.median(samples), 3),
        "p10_ms": (
            round(statistics.quantiles(samples, n=10)[0], 3) if repeats >= 10 else None
        ),
        "min_ms": round(min(samples), 3),
    }


def _torch_devices() -> list[str]:
    devs = ["cpu"]
    if torch.cuda.is_available():
        devs.append("cuda")
    if torch.backends.mps.is_available():
        devs.append("mps")
    return devs


# ─────────────────────────────────────────────────────────────────────────────
# (A) runtime value — real classifier across runtimes
# ─────────────────────────────────────────────────────────────────────────────
def bench_classifier_runtimes(
    model_path: str, batch: int, warmup: int, repeats: int
) -> list[dict]:
    from hydra_suite.core.identity.classification.backend import ClassifierBackend

    rows = []
    # Candidate runtimes filtered by what the box can do.
    cands = ["cpu"]
    if torch.cuda.is_available():
        cands += ["cuda", "onnx_cuda", "tensorrt"]
    if torch.backends.mps.is_available():
        cands += ["mps", "onnx_coreml"]
    cands += ["onnx_cpu"]

    # Need the model input size to build representative crops.
    try:
        probe = ClassifierBackend(model_path, compute_runtime="cpu")
        probe._ensure_loaded()
        h, w = probe._metadata.input_size
        probe.close()
    except Exception as exc:  # noqa: BLE001
        return [{"stage": "classifier", "error": f"probe failed: {exc}"}]

    crops = [np.random.randint(0, 256, (h, w, 3), dtype=np.uint8) for _ in range(batch)]

    for rt in cands:
        row = {"stage": "classifier", "runtime": rt, "batch": batch, "input": [h, w]}
        try:
            be = ClassifierBackend(model_path, compute_runtime=rt)
            be._ensure_loaded()
            active = getattr(be, "_active_execution_backend", "?")
            row["active_backend"] = active
            dev = (
                "cuda"
                if rt in ("cuda", "onnx_cuda", "tensorrt")
                else ("mps" if rt in ("mps", "onnx_coreml") else "cpu")
            )
            stats = _time_ms(
                lambda be=be: be.predict_batch(crops), dev, warmup, repeats
            )
            row.update(stats)
            row["per_crop_ms"] = round(stats["median_ms"] / batch, 4)
            be.close()
        except Exception as exc:  # noqa: BLE001
            row["error"] = f"{type(exc).__name__}: {exc}"[:200]
        rows.append(row)
    return rows


# ─────────────────────────────────────────────────────────────────────────────
# (A) runtime value — real OBB across runtimes
# ─────────────────────────────────────────────────────────────────────────────
def bench_obb_runtimes(
    model_path: str, warmup: int, repeats: int, imgsz: int
) -> list[dict]:
    from hydra_suite.core.inference.runtime_artifacts import load_obb_executor

    rows = []
    cands = ["cpu"]
    if torch.cuda.is_available():
        cands += ["cuda", "onnx_cuda", "tensorrt"]
    if torch.backends.mps.is_available():
        cands += ["mps"]
    cands += ["onnx_cpu"]

    frame = np.random.randint(0, 256, (imgsz, imgsz, 3), dtype=np.uint8)
    for rt in cands:
        row = {"stage": "obb", "runtime": rt, "imgsz": imgsz}
        try:
            exe = load_obb_executor(
                model_path, compute_runtime=rt, auto_export=True, max_det=100
            )
            dev = (
                "cuda"
                if rt in ("cuda", "onnx_cuda", "tensorrt")
                else ("mps" if rt == "mps" else "cpu")
            )

            # Pass a LIST of frames: ultralytics tolerates a bare array, but the
            # direct ONNX/TRT executor expects a list (a bare HWC array is
            # iterated row-wise -> "expected 3, got 2" unpack error).
            def _call(exe=exe):
                exe.predict([frame], conf=0.2, iou=0.5, imgsz=imgsz, verbose=False)

            stats = _time_ms(_call, dev, warmup, repeats)
            row.update(stats)
        except Exception as exc:  # noqa: BLE001
            row["error"] = f"{type(exc).__name__}: {exc}"[:200]
        rows.append(row)
    return rows


# ─────────────────────────────────────────────────────────────────────────────
# (B) exact-win headroom — torch-level, architecture-driven
# ─────────────────────────────────────────────────────────────────────────────
def bench_exact_wins(
    batch: int, warmup: int, repeats: int, input_hw: int = 224
) -> list[dict]:
    import torchvision  # noqa: F401
    from torchvision.models import efficientnet_b0

    rows = []
    model = efficientnet_b0(weights=None).eval()

    for device in _torch_devices():
        dev = torch.device(device)
        m = model.to(dev)
        x_cpu = torch.rand(batch, 3, input_hw, input_hw)

        # baseline: no_grad, contiguous, pageable (current production behaviour)
        x = x_cpu.to(dev)
        with torch.no_grad():

            def _base(m=m, x=x):
                with torch.no_grad():
                    return m(x)

            rows.append(
                {
                    "stage": "exact",
                    "device": device,
                    "variant": "no_grad+contig",
                    **_time_ms(_base, device, warmup, repeats),
                }
            )

            # inference_mode instead of no_grad
            def _infmode(m=m, x=x):
                with torch.inference_mode():
                    return m(x)

            rows.append(
                {
                    "stage": "exact",
                    "device": device,
                    "variant": "inference_mode+contig",
                    **_time_ms(_infmode, device, warmup, repeats),
                }
            )

            # channels_last
            m_cl = m.to(memory_format=torch.channels_last)
            x_cl = x.to(memory_format=torch.channels_last)

            def _cl(m_cl=m_cl, x_cl=x_cl):
                with torch.inference_mode():
                    return m_cl(x_cl)

            rows.append(
                {
                    "stage": "exact",
                    "device": device,
                    "variant": "inference_mode+channels_last",
                    **_time_ms(_cl, device, warmup, repeats),
                }
            )
            m_cl = m_cl.to(memory_format=torch.contiguous_format)  # restore

        # H2D copy cost: pageable vs pinned+non_blocking (cuda only meaningful)
        if device == "cuda":
            pageable = x_cpu
            pinned = x_cpu.pin_memory()

            def _h2d_pageable(pageable=pageable, dev=dev):
                pageable.to(dev, non_blocking=False)

            def _h2d_pinned(pinned=pinned, dev=dev):
                pinned.to(dev, non_blocking=True)

            rows.append(
                {
                    "stage": "h2d",
                    "device": device,
                    "variant": "pageable",
                    **_time_ms(_h2d_pageable, device, warmup, repeats),
                }
            )
            rows.append(
                {
                    "stage": "h2d",
                    "device": device,
                    "variant": "pinned+non_blocking",
                    **_time_ms(_h2d_pinned, device, warmup, repeats),
                }
            )
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--classifier", default=None, help="path to a real classifier .pth")
    ap.add_argument("--obb", default=None, help="path to a real OBB .pt")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--imgsz", type=int, default=1024)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--repeats", type=int, default=20)
    ap.add_argument("--json", default=None)
    ap.add_argument("--skip-exact", action="store_true")
    args = ap.parse_args()

    out = {
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "mps_available": torch.backends.mps.is_available(),
        "cuda_device": (
            torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
        ),
        "results": [],
    }

    if not args.skip_exact:
        print("== (B) exact-win headroom ==", flush=True)
        out["results"] += bench_exact_wins(args.batch, args.warmup, args.repeats)
    if args.classifier:
        print("== (A) classifier runtimes ==", flush=True)
        out["results"] += bench_classifier_runtimes(
            args.classifier, args.batch, args.warmup, args.repeats
        )
    if args.obb:
        print("== (A) obb runtimes ==", flush=True)
        out["results"] += bench_obb_runtimes(
            args.obb, args.warmup, args.repeats, args.imgsz
        )

    # pretty print
    for r in out["results"]:
        print(json.dumps(r), flush=True)
    if args.json:
        Path(args.json).write_text(json.dumps(out, indent=2))
        print(f"\nwrote {args.json}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
