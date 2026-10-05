#!/usr/bin/env python
"""Render a grid of SAM3 augmented tiles with their polygons (dev tool).

Column 0 is the unaugmented tile; columns 1..epochs apply the train-time
augmenter (recommended profile + rotate) for successive epoch seeds. Polygons
are green, crowd polygons red. Use it to eyeball that polygons stay on the
animals and that rotate borders are gray.

    python tools/sam3_augmentation_preview.py --dataset <sam3 dataset dir> \
        --out /tmp/sam3_aug_preview.png [--n 6] [--epochs 3] [--seed 0]
"""

from __future__ import annotations

import argparse
import dataclasses
import sys
from pathlib import Path

import cv2
import numpy as np

from hydra_suite.training.contracts import Sam3LoraParams
from hydra_suite.training.sam3_lora.augment import (
    make_tile_augmenter,
    recommended_sam3_augmentation,
)
from hydra_suite.training.sam3_lora.dataloader import build_descriptors

GREEN = (0, 220, 0)
RED = (0, 0, 255)


def _render(tile_bgr, instances, cell: int) -> np.ndarray:
    """Draw polygons (edge -> index shifted) then downscale to `cell` px."""
    h, w = tile_bgr.shape[:2]
    s = cell / max(h, w)
    out = cv2.resize(tile_bgr, (max(1, round(w * s)), max(1, round(h * s))))
    for poly, crowd in instances:
        pts = np.asarray(poly, dtype=np.float64) * s - 0.5
        cv2.polylines(
            out,
            [np.round(pts).astype(np.int32).reshape(-1, 1, 2)],
            True,
            RED if crowd else GREEN,
            1,
            cv2.LINE_AA,
        )
    canvas = np.full((cell, cell, 3), 40, np.uint8)
    canvas[: out.shape[0], : out.shape[1]] = out
    return canvas


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dataset", required=True, help="SAM3 derived dataset dir")
    ap.add_argument("--out", required=True, help="output PNG path")
    ap.add_argument("--n", type=int, default=6, help="number of tiles (rows)")
    ap.add_argument("--epochs", type=int, default=3, help="augmented columns")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cell", type=int, default=320, help="cell size in px")
    ap.add_argument("--rotate", type=float, default=15.0)
    args = ap.parse_args(argv)

    params = Sam3LoraParams(prompt="x", num_negatives=0)
    descs = build_descriptors(args.dataset, params, "train")[: args.n]
    if not descs:
        print("no descriptors found", file=sys.stderr)
        return 1
    profile = dataclasses.replace(recommended_sam3_augmentation(), rotate=args.rotate)
    augs = [
        make_tile_augmenter(profile, epoch_seed=args.seed + e, min_area_ratio=0.1)
        for e in range(args.epochs)
    ]

    rows = []
    for d in descs:
        img = cv2.imread(str(d.image_path))
        if img is None:
            print(f"cannot read {d.image_path}", file=sys.stderr)
            return 1
        inst = [
            (np.asarray(i.polygon, dtype=np.float32).reshape(-1, 2), bool(i.is_crowd))
            for i in d.instances
        ]
        cells = [_render(img, inst, args.cell)]
        for aug in augs:
            a_img, a_inst = aug(img.copy(), list(inst), d.image_id)
            cells.append(_render(a_img, a_inst, args.cell))
        rows.append(np.hstack(cells))
    grid = np.vstack(rows)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(args.out, grid)
    print(f"wrote {args.out} ({grid.shape[1]}x{grid.shape[0]})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
