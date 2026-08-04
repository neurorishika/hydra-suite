"""Skeleton overlays and contact sheets for eyeballing checkpoint output."""

from __future__ import annotations

import cv2
import numpy as np

from .skeleton import SkeletonSpec

FONT = cv2.FONT_HERSHEY_SIMPLEX
BANNER_H = 24


def draw_pose(
    crop_bgr: np.ndarray,
    coords: np.ndarray,
    conf: np.ndarray,
    spec: SkeletonSpec,
    conf_thr: float = 0.2,
) -> np.ndarray:
    """Overlay edges and keypoints. Marker radius scales with confidence so a
    hesitant keypoint reads as small rather than as a confident mistake."""
    out = crop_bgr.copy()
    ok = conf >= conf_thr

    for (a, b), color in zip(spec.skeleton_edges, spec.edge_colors_bgr):
        if not (ok[a] and ok[b]):
            continue
        pa = (int(round(coords[a, 0])), int(round(coords[a, 1])))
        pb = (int(round(coords[b, 0])), int(round(coords[b, 1])))
        cv2.line(out, pa, pb, color, 1, cv2.LINE_AA)

    for i, color in enumerate(spec.keypoint_colors_bgr):
        if not ok[i]:
            continue
        radius = 2 + int(round(3 * min(float(conf[i]), 1.0)))
        center = (int(round(coords[i, 0])), int(round(coords[i, 1])))
        cv2.circle(out, center, radius, color, -1, cv2.LINE_AA)

    return out


def label_tile(tile_bgr: np.ndarray, text: str) -> np.ndarray:
    h, w = tile_bgr.shape[:2]
    banner = np.zeros((BANNER_H, w, 3), dtype=np.uint8)
    cv2.putText(banner, text, (4, 17), FONT, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    return np.vstack([banner, tile_bgr])


def contact_sheet(tiles: list[np.ndarray], cols: int = 4, pad: int = 8) -> np.ndarray:
    if not tiles:
        raise ValueError("contact_sheet needs at least one tile")
    th, tw = tiles[0].shape[:2]
    rows = (len(tiles) + cols - 1) // cols
    sheet = np.zeros(
        (rows * th + (rows + 1) * pad, cols * tw + (cols + 1) * pad, 3),
        dtype=np.uint8,
    )
    for i, tile in enumerate(tiles):
        r, c = divmod(i, cols)
        y = pad + r * (th + pad)
        x = pad + c * (tw + pad)
        sheet[y : y + th, x : x + tw] = tile
    return sheet


def confidence_table(conf: np.ndarray, spec: SkeletonSpec) -> str:
    """conf: (N, K) peak heatmap values. A keypoint the model is guessing at
    shows up immediately as a low median row."""
    lines = [f"{'keypoint':<22} {'median':>8} {'min':>8} {'max':>8}"]
    med = np.median(conf, axis=0)
    lo = conf.min(axis=0)
    hi = conf.max(axis=0)
    for i, name in enumerate(spec.keypoint_names):
        lines.append(f"{name:<22} {med[i]:>8.3f} {lo[i]:>8.3f} {hi[i]:>8.3f}")
    return "\n".join(lines)
