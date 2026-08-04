"""Top-down crop sampling from existing tracking output.

We take crop centres and headings straight from a completed
`*_tracking_final.csv` rather than re-running a detector, so this probe costs
nothing but video seeks.

Heading convention: `Theta` is measured in image coordinates (x right, y DOWN),
so forward is `(cos t, sin t)`. "Upright" means forward points to `(0, -1)`,
which is a point rotation by `-pi/2 - t`. `cv2.getRotationMatrix2D` rotates
points by `-angle_deg` in this y-down frame, hence `angle_deg = deg(t) + 90`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class CropSample:
    frame_id: int
    track_id: int
    cx: float
    cy: float
    theta: float


def select_samples(csv_path: Path, n: int) -> list[CropSample]:
    """Pick `n` (frame, track) pairs spread evenly over the tracked range.

    Deterministic: frames are taken at evenly spaced positions through the
    sorted unique active frames, and the k-th sample takes the k-th distinct
    track present in its frame (wrapping), so track IDs vary without any RNG.
    """
    df = pd.read_csv(
        csv_path,
        usecols=["TrajectoryID", "X", "Y", "Theta", "FrameID", "State"],
    )
    df = df[df["State"] == "active"]
    if df.empty:
        raise ValueError(f"{csv_path}: no active rows to sample")

    frames = np.sort(df["FrameID"].unique())
    if len(frames) < n:
        picks = frames
    else:
        idx = np.linspace(0, len(frames) - 1, n).round().astype(int)
        picks = frames[np.unique(idx)]

    samples: list[CropSample] = []
    for k, frame_id in enumerate(picks):
        rows = df[df["FrameID"] == frame_id].sort_values("TrajectoryID")
        row = rows.iloc[k % len(rows)]
        samples.append(
            CropSample(
                frame_id=int(frame_id),
                track_id=int(row["TrajectoryID"]),
                cx=float(row["X"]),
                cy=float(row["Y"]),
                theta=float(row["Theta"]),
            )
        )
    return samples


def crop_matrix(
    cx: float,
    cy: float,
    theta: float,
    side_px: float,
    out_px: int,
    rotate: bool,
) -> np.ndarray:
    """2x3 affine taking the source square of `side_px` centred on (cx, cy) to
    an `out_px` x `out_px` crop, optionally rotating the heading to point up."""
    angle_deg = math.degrees(theta) + 90.0 if rotate else 0.0
    scale = out_px / side_px
    m = cv2.getRotationMatrix2D((cx, cy), angle_deg, scale)
    # getRotationMatrix2D pins (cx, cy); shift it to the crop centre.
    m[0, 2] += out_px / 2.0 - cx
    m[1, 2] += out_px / 2.0 - cy
    return m.astype(np.float32)


def warp_crop(frame_bgr: np.ndarray, matrix: np.ndarray, out_px: int) -> np.ndarray:
    return cv2.warpAffine(frame_bgr, matrix, (out_px, out_px), flags=cv2.INTER_LINEAR)
