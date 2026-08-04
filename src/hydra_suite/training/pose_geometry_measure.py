"""Measure a PoseKit label set and suggest a ViTPose input geometry.

Pure measurement: no Qt, and no imports from any app layer -- Training must not
depend on PoseKit. That rule is a benefit here rather than a cost: PoseKit's own
`load_yolo_pose_label` parses only the FIRST line of a label file, so it silently
under-counts multi-animal frames. This module parses every line.

Sizing targets the bare keypoint extent on purpose. Inference already pads by
PADDING_FACTOR = 1.25 inside `box2cs` before warping, so the model sees more than
the animal; the `detail` multiplier is for operator preference, not to compensate
for that padding.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np
from PIL import Image

SIZE_MULTIPLE = 32
MIN_SIZE = 64
# ViT attention cost tracks token count: 192x256 is 192 tokens, 256x256 is 256,
# 384x384 is 576, 512x512 is 1024. Cap the SUGGESTION so the tool never quietly
# proposes a model that trains several times slower; typing a larger value by
# hand stays available.
MAX_SUGGESTED_SIZE = 384


@dataclass(frozen=True)
class PoseSizeStats:
    sample_count: int  # instances measured, not files read
    frames_scanned: int
    frames_skipped: int
    median_aspect: float  # width / height of the keypoint bounding box
    median_long_px: float
    p90_long_px: float
    suggested_hw: List[int]  # [H, W]
    clamped: bool


def _snap(value: float) -> Tuple[int, bool]:
    """Round to the nearest multiple of 32 and clamp; report whether capped."""
    snapped = int(round(value / SIZE_MULTIPLE)) * SIZE_MULTIPLE
    clamped = False
    if snapped > MAX_SUGGESTED_SIZE:
        snapped, clamped = MAX_SUGGESTED_SIZE, True
    if snapped < MIN_SIZE:
        snapped = MIN_SIZE
    return snapped, clamped


def _instance_extents(
    text: str, num_keypoints: int, w_px: int, h_px: int
) -> List[Tuple[float, float]]:
    """Per-instance (width, height) in pixels of the VISIBLE keypoints' box.

    Invisible keypoints (v == 0) are excluded: they carry no position
    information, and counting them at their stored coordinates would bias every
    box toward the origin.
    """
    out: List[Tuple[float, float]] = []
    need = 5 + 3 * num_keypoints
    for line in text.splitlines():
        parts = line.split()
        if len(parts) < need:
            continue
        try:
            vals = [float(v) for v in parts[1:need]]
        except ValueError:
            continue
        xs: List[float] = []
        ys: List[float] = []
        for i in range(num_keypoints):
            x, y, v = vals[4 + 3 * i], vals[5 + 3 * i], vals[6 + 3 * i]
            if v > 0:
                xs.append(x * w_px)
                ys.append(y * h_px)
        if len(xs) < 2:
            continue
        bw = max(xs) - min(xs)
        bh = max(ys) - min(ys)
        if bw <= 0 or bh <= 0:
            continue
        out.append((bw, bh))
    return out


def measure_pose_geometry(
    image_paths: Sequence[Path],
    labels_dir: Path,
    num_keypoints: int,
    *,
    detail: float = 1.0,
    max_images: int = 500,
    seed: int = 0,
) -> PoseSizeStats:
    """Suggest a ViTPose input geometry from a PoseKit label store."""
    if detail <= 0:
        raise ValueError(f"detail must be positive; got {detail}")
    labels_dir = Path(labels_dir)

    labelled: List[Tuple[Path, Path]] = []
    for raw in image_paths:
        img_path = Path(raw)
        label_path = labels_dir / f"{img_path.stem}.txt"
        try:
            if label_path.exists() and label_path.stat().st_size > 0:
                labelled.append((img_path, label_path))
        except OSError:
            continue
    if not labelled:
        raise ValueError(f"no labelled frames found under {labels_dir}")

    if len(labelled) > max_images:
        labelled = random.Random(seed).sample(labelled, max_images)
        labelled.sort()  # order-independent of the sample draw

    widths: List[float] = []
    heights: List[float] = []
    scanned = 0
    skipped = 0
    for img_path, label_path in labelled:
        scanned += 1
        try:
            with Image.open(img_path) as im:
                w_px, h_px = im.size
            text = label_path.read_text(encoding="utf-8")
        except Exception:
            skipped += 1
            continue
        extents = _instance_extents(text, num_keypoints, int(w_px), int(h_px))
        if not extents:
            skipped += 1
            continue
        for bw, bh in extents:
            widths.append(bw)
            heights.append(bh)

    if not widths:
        raise ValueError(
            f"found {scanned} labelled frame(s) under {labels_dir} but none held a "
            "usable instance (an instance needs at least 2 visible keypoints)"
        )

    w_arr = np.asarray(widths, dtype=np.float64)
    h_arr = np.asarray(heights, dtype=np.float64)
    long_arr = np.maximum(w_arr, h_arr)
    median_aspect = float(np.median(w_arr / h_arr))
    median_long = float(np.median(long_arr))
    p90_long = float(np.percentile(long_arr, 90))

    # Reconstruct a coherent (W, H) from one length and one aspect. Taking
    # independent medians of width and height could describe an animal that
    # does not exist in the data.
    long_side = median_long * detail
    if median_aspect >= 1.0:
        raw_w, raw_h = long_side, long_side / median_aspect
    else:
        raw_h, raw_w = long_side, long_side * median_aspect

    snapped_w, clamped_w = _snap(raw_w)
    snapped_h, clamped_h = _snap(raw_h)
    return PoseSizeStats(
        sample_count=len(widths),
        frames_scanned=scanned,
        frames_skipped=skipped,
        median_aspect=median_aspect,
        median_long_px=median_long,
        p90_long_px=p90_long,
        suggested_hw=[snapped_h, snapped_w],
        clamped=clamped_w or clamped_h,
    )
