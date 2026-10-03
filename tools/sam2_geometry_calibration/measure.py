"""Measure SAM2 geometry calibration on a real polygon source (headless).

Sets the evidence for ``IOU_FLOOR`` / ``FALLBACK_CEIL`` in
``core/inference/sam2/calibration.py`` and answers whether owner-tile SAHI
beats full-frame SAM2 on small animals. Usage::

    python tools/sam2_geometry_calibration/measure.py \
        --source ~/geom_sahi_data/merged_source_20260914 \
        --variant sam2.1-hiera-base_plus --budget 30 --device cuda \
        --out /tmp/geom_cal_base_plus.json
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import time
from dataclasses import asdict
from pathlib import Path


def _fractions(text: str):
    out = []
    for item in text.split(","):
        item = item.strip()
        out.append(None if item in {"full", "none", ""} else float(item))
    return tuple(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True)
    ap.add_argument("--variant", default="sam2.1-hiera-base_plus")
    ap.add_argument("--budget", type=int, default=30)
    ap.add_argument("--fractions", default="0.02,0.03,0.05,0.1,0.2,full")
    ap.add_argument("--overlap", type=float, default=0.5)
    ap.add_argument("--device", default=None)
    ap.add_argument("--reference-body-px", type=float, default=0.0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    from hydra_suite.core.inference.sam2.calibration import (
        calibrate_geometry,
        recommend_geometry,
    )
    from hydra_suite.core.inference.sam2.executor import Sam2SegmentExecutor
    from hydra_suite.detectkit.gui.models import OBBSource
    from hydra_suite.detectkit.jobs.calibration_frames import (
        measure_median_body_px,
        stratified_calibration_frames,
    )

    source = OBBSource(
        path=str(Path(args.source).expanduser()),
        name=Path(args.source).name,
        level="polygon",
    )
    frames = stratified_calibration_frames(
        [source], budget=args.budget, polygon_only=True
    )
    body_px = args.reference_body_px or measure_median_body_px([source])[0]
    print(f"{len(frames)} polygon frames, median body {body_px:.1f} px", flush=True)

    t0 = time.perf_counter()
    executor = Sam2SegmentExecutor.from_variant(args.variant, args.device)
    load_s = time.perf_counter() - t0
    points = calibrate_geometry(
        executor,
        [(path, [rec.points for rec in records]) for path, records in frames],
        reference_body_px=body_px,
        tile_fractions=_fractions(args.fractions),
        overlap=args.overlap,
        progress=lambda pct, msg: print(f"[{pct:3d}%] {msg}", flush=True),
    )
    best, reason = recommend_geometry(points)

    header = (
        f"{'fraction':>9} {'tile_px':>7} {'tiles/f':>7} {'s/frame':>8} "
        f"{'medIoU':>7} {'p10IoU':>7} {'fallbk':>7} {'seam':>6} {'n':>5}"
    )
    print(header)
    for p in points:
        frac = "full" if p.tile_fraction is None else f"{p.tile_fraction:.3f}"
        print(
            f"{frac:>9} {str(p.tile_px or '-'):>7} {p.owned_tiles_per_frame:7.2f} "
            f"{p.seconds_per_frame:8.3f} {p.median_iou:7.3f} {p.p10_iou:7.3f} "
            f"{p.fallback_rate:7.3f} {p.seam_fallback_rate:6.3f} {p.n_instances:5d}"
        )
    if best is None:
        print("recommended: REFUSED --", reason)
    elif best.tile_fraction is None:
        print("recommended: full frame")
    else:
        print(f"recommended: tile fraction {best.tile_fraction:g}")

    try:
        sha = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True
        ).stdout.strip()
    except OSError:  # pragma: no cover
        sha = ""
    gpu = ""
    try:
        import torch

        if torch.cuda.is_available():
            gpu = torch.cuda.get_device_name(0)
    except Exception:  # pragma: no cover
        pass
    Path(args.out).write_text(
        json.dumps(
            {
                "source": str(source.path),
                "variant": args.variant,
                "frames": len(frames),
                "median_body_px": body_px,
                "overlap": args.overlap,
                "model_load_seconds": load_s,
                "git_sha": sha,
                "host": platform.node(),
                "gpu": gpu,
                "points": [asdict(p) for p in points],
                "recommended_fraction": None if best is None else best.tile_fraction,
                "reason": reason,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
