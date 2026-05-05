"""Shared overlay palette and detection helpers for RefineKit."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, Dict, Optional, Set, Tuple

import cv2
import numpy as np
import pandas as pd

from hydra_suite.data.detection_cache import DetectionCache

TAB20_RGB = [
    (31, 119, 180),
    (174, 199, 232),
    (255, 127, 14),
    (255, 187, 120),
    (44, 160, 44),
    (152, 223, 138),
    (214, 39, 40),
    (255, 152, 150),
    (148, 103, 189),
    (197, 176, 213),
    (140, 86, 75),
    (196, 156, 148),
    (227, 119, 194),
    (247, 182, 210),
    (127, 127, 127),
    (199, 199, 199),
    (188, 189, 34),
    (219, 219, 141),
    (23, 190, 207),
    (158, 218, 229),
]
TAB20_BGR = [(b, g, r) for r, g, b in TAB20_RGB]

# The requested PiYG "pink" track colour corresponds to the pink side of the map.
MAIN_TRACK_RGB = (196, 26, 124)
MAIN_TRACK_BGR = (124, 26, 196)


def tab20_rgb(track_id: int) -> Tuple[int, int, int]:
    return TAB20_RGB[track_id % len(TAB20_RGB)]


def tab20_bgr(track_id: int) -> Tuple[int, int, int]:
    return TAB20_BGR[track_id % len(TAB20_BGR)]


def overlay_scale_from_shape(
    shape: Tuple[int, ...],
    scale: float = 1.0,
) -> Tuple[float, int, int]:
    """Return font scale, marker radius, and line thickness for an image shape."""
    h, w = shape[:2]
    ref = min(h, w)
    font_scale = max(0.15, (ref / 1800.0) * scale)
    radius = max(2, int((ref / 160.0) * scale))
    thickness = max(1, int((ref / 800.0) * scale))
    return font_scale, radius, thickness


def review_overlay_style_from_shape(
    shape: Tuple[int, ...],
    scale: float = 1.0,
) -> Tuple[float, int, int, int]:
    """Return normalized text, marker, line, and outline sizes for review overlays."""
    font_scale, radius, thickness = overlay_scale_from_shape(shape, scale)
    font_scale = max(0.3, font_scale * 1.8)
    radius = max(4, int(round(radius * 1.8)))
    thickness = max(2, int(round(thickness * 2.0)))
    outline_thickness = max(thickness + 1, int(round(thickness * 1.5)))
    return font_scale, radius, thickness, outline_thickness


def _frame_luminance_bgr(img: np.ndarray) -> float:
    if img.size == 0:
        return 0.0
    return float(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).mean())


def contrast_binary_bgr(
    img: np.ndarray,
    light_on_dark: Tuple[int, int, int] = (255, 255, 255),
    dark_on_light: Tuple[int, int, int] = (0, 0, 0),
    threshold: float = 140.0,
) -> Tuple[int, int, int]:
    """Return white on dark frames and black on light frames."""
    return dark_on_light if _frame_luminance_bgr(img) >= threshold else light_on_dark


def context_gray_bgr(
    img: np.ndarray,
    light_on_dark: Tuple[int, int, int] = (190, 190, 190),
    dark_on_light: Tuple[int, int, int] = (70, 70, 70),
    threshold: float = 140.0,
) -> Tuple[int, int, int]:
    """Return a frame-contrast gray for non-focus tracks."""
    return dark_on_light if _frame_luminance_bgr(img) >= threshold else light_on_dark


def discover_detection_cache(video_path: str) -> Optional[Path]:
    """Find the detection cache file saved next to a tracked video."""
    vp = Path(video_path).expanduser()
    stem = vp.stem
    parent = vp.parent

    cache_dir = parent / f"{stem}_caches"
    if cache_dir.is_dir():
        hits = sorted(cache_dir.glob(f"{stem}_detection_cache_*.npz"))
        if hits:
            return hits[-1]

    hits = sorted(parent.glob(f"{stem}_detection_cache_*.npz"))
    return hits[-1] if hits else None


def load_resize_factor(video_path: str) -> float:
    """Read the saved resize factor for a tracked video, defaulting to 1.0."""
    vp = Path(video_path).expanduser()
    cfg_path = vp.parent / f"{vp.stem}_config.json"
    if cfg_path.is_file():
        try:
            with open(cfg_path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            return float(data.get("resize_factor", 1.0))
        except (json.JSONDecodeError, ValueError, OSError):
            return 1.0
    return 1.0


class FrameDetections:
    """Read-only detection cache wrapper that exposes OBB/ellipse geometry."""

    def __init__(self, cache: DetectionCache, inv_resize: float) -> None:
        self._cache = cache
        self._inv_resize = inv_resize

    def get(self, frame_idx: int):
        try:
            (
                meas,
                _sizes,
                shapes,
                _conf,
                obb,
                _ids,
                _hints,
                _heading_confidences,
                _dmask,
                _affines,
                _canvas_dims,
                _m_inverse,
            ) = self._cache.get_frame(frame_idx)
        except Exception:
            return None
        if not meas:
            return None

        scale = self._inv_resize
        meas_arr = np.array(meas, dtype=np.float32)
        meas_arr[:, 0] *= scale
        meas_arr[:, 1] *= scale

        shapes_arr = (
            np.array(shapes, dtype=np.float32)
            if shapes
            else np.empty((0, 2), dtype=np.float32)
        )
        semi_axes = np.empty((len(meas_arr), 2), dtype=np.float32)
        for idx in range(len(meas_arr)):
            if idx < len(shapes_arr) and shapes_arr.ndim == 2:
                area, aspect_ratio = shapes_arr[idx]
                aspect_ratio = max(aspect_ratio, 0.01)
                minor = np.sqrt(4.0 * abs(area) / (np.pi * aspect_ratio))
                major = aspect_ratio * minor
                semi_axes[idx] = [major * 0.5 * scale, minor * 0.5 * scale]
            else:
                semi_axes[idx] = [8.0, 4.0]

        obb_scaled = []
        if obb:
            for corners in obb:
                obb_scaled.append(np.asarray(corners, dtype=np.float32) * scale)

        return meas_arr, semi_axes, obb_scaled


def load_frame_detections(video_path: str) -> Optional[FrameDetections]:
    """Open a video's detection cache if one exists and is readable."""
    cache_path = discover_detection_cache(video_path)
    if cache_path is None:
        return None
    try:
        cache = DetectionCache(cache_path, mode="r")
        if not cache.is_compatible():
            return None
        inv_resize = 1.0 / load_resize_factor(video_path)
        return FrameDetections(cache, inv_resize)
    except Exception:
        return None


def build_detection_track_map(
    df: pd.DataFrame,
    frame_start: int,
    frame_end: int,
    track_ids: Optional[Set[int]] = None,
    track_resolver: Optional[Callable[[pd.Series], Optional[int]]] = None,
) -> Dict[int, Dict[int, int]]:
    """Return a per-frame map from detection index to track id."""
    if "DetectionID" not in df.columns:
        return {}

    sub = df[df["FrameID"].between(frame_start, frame_end)]
    if track_ids is not None:
        sub = sub[sub["TrajectoryID"].isin(track_ids)]
    sub = sub[sub["DetectionID"].notna()]

    det_map: Dict[int, Dict[int, int]] = {}
    for _, row in sub.iterrows():
        if track_resolver is None:
            if pd.isna(row.get("TrajectoryID")):
                continue
            track_id = int(row["TrajectoryID"])
        else:
            track_id = track_resolver(row)
            if track_id is None:
                continue
        frame_idx = int(row["FrameID"])
        det_idx = int(row["DetectionID"]) % 10000
        det_map.setdefault(frame_idx, {})[det_idx] = track_id
    return det_map


def draw_detections(
    img: np.ndarray,
    dets: FrameDetections,
    frame_idx: int,
    ox: float,
    oy: float,
    colors_by_index: Dict[int, Tuple[int, int, int]],
    thickness: int = 1,
    alpha: float = 0.45,
) -> None:
    """Draw detection OBBs or ellipses for a frame using per-detection colours."""
    if not colors_by_index:
        return

    result = dets.get(frame_idx)
    if result is None:
        return

    meas_arr, semi_axes, obb_corners = result
    overlay = img.copy()

    if obb_corners:
        for idx, corners in enumerate(obb_corners):
            color = colors_by_index.get(idx)
            if color is None:
                continue
            pts = corners.copy()
            pts[:, 0] -= ox
            pts[:, 1] -= oy
            ipts = pts.astype(np.int32).reshape((-1, 1, 2))
            cv2.polylines(overlay, [ipts], True, color, thickness, cv2.LINE_AA)
    else:
        for idx in range(len(meas_arr)):
            color = colors_by_index.get(idx)
            if color is None:
                continue
            cx = int(round(meas_arr[idx, 0] - ox))
            cy = int(round(meas_arr[idx, 1] - oy))
            theta = meas_arr[idx, 2]
            major = int(round(semi_axes[idx, 0]))
            minor = int(round(semi_axes[idx, 1]))
            if major < 2 or minor < 2:
                continue
            cv2.ellipse(
                overlay,
                (cx, cy),
                (major, minor),
                -np.degrees(theta),
                0,
                360,
                color,
                thickness,
                cv2.LINE_AA,
            )

    cv2.addWeighted(overlay, alpha, img, 1.0 - alpha, 0, img)
