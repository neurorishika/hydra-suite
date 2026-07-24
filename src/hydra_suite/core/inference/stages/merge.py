"""Cross-tile duplicate merge for sliced OBB inference.

When a frame is cut into overlapping tiles and each tile is detected
separately, one real animal straddling a tile boundary produces TWO
truncated detections. This module deduplicates those cross-tile duplicates
via either suppression (nms: keep the higher-confidence raw box) or
merging (nmm/greedy_nmm: union the duplicates into one larger OBB).

The cv2 backend here is the correctness oracle; a later task adds a `gpu`
backend with the same policy x metric semantics for large detection counts.
"""

from __future__ import annotations

import cv2
import numpy as np

from ..result import OBBResult
from .obb import _corners_from_xywhr, _empty_obb_result, _normalize_obb_geometry


def _hull(corners: np.ndarray) -> tuple[np.ndarray, float]:
    p = cv2.convexHull(np.asarray(corners, dtype=np.float32)).reshape(-1, 2)
    return p, float(abs(cv2.contourArea(p)))


def _pair_overlap(
    hull_a: np.ndarray, area_a: float, hull_b: np.ndarray, area_b: float, metric: str
) -> float:
    """IoU or IoS of two convex corner polygons (cv2 intersection area)."""
    if area_a <= 1e-9 or area_b <= 1e-9:
        return 0.0
    try:
        inter, _ = cv2.intersectConvexConvex(hull_a, hull_b)
        inter = float(max(0.0, inter))
    except Exception:
        inter = 0.0
    if metric == "ios":
        denom = min(area_a, area_b)
    else:  # iou
        denom = area_a + area_b - inter
    return float(inter / denom) if denom > 1e-9 else 0.0


def band_membership(
    corners: np.ndarray, tiles: list[tuple[int, int, int, int]]
) -> np.ndarray:
    """(D,) bool: True where a detection's AABB touches >= 2 tiles (overlap band).

    A detection inside a single tile's exclusive region cannot have a cross-tile
    duplicate, so only band members need the O(n^2) merge.
    """
    n = corners.shape[0]
    if n == 0:
        return np.zeros(0, dtype=bool)
    amin = corners.min(axis=1)  # (D, 2)
    amax = corners.max(axis=1)
    counts = np.zeros(n, dtype=np.int32)
    for tx0, ty0, tx1, ty1 in tiles:
        hit = (
            (amax[:, 0] > tx0)
            & (amin[:, 0] < tx1)
            & (amax[:, 1] > ty0)
            & (amin[:, 1] < ty1)
        )
        counts += hit.astype(np.int32)
    return counts >= 2


def _union_obb(members: OBBResult, idxs: list[int], frame_idx: int) -> tuple:
    """Union the member corners into one OBB via cv2.minAreaRect.

    Returns (cx, cy, w, h, angle_rad, conf, cls) renormalized through the shared
    geometry pipeline so the merged box is indistinguishable from a native OBB.
    """
    pts = members.corners[idxs].reshape(-1, 2).astype(np.float32)
    (cx, cy), (w, h), angle_deg = cv2.minAreaRect(pts)
    ang, _, _ = _normalize_obb_geometry(
        np.array([w], np.float32),
        np.array([h], np.float32),
        np.array([np.deg2rad(angle_deg)], np.float32),
    )
    conf = float(members.confidences[idxs].max())
    top = idxs[int(np.argmax(members.confidences[idxs]))]
    cls = int(members.class_ids_or_zeros[top])
    return float(cx), float(cy), float(w), float(h), float(ang[0]), conf, cls


def merge_obb_detections(
    result: OBBResult,
    *,
    policy: str,
    metric: str,
    threshold: float,
    backend: str,
    overlap_bands: "np.ndarray | None" = None,
    runtime=None,
) -> OBBResult:
    """Merge cross-tile duplicate detections. cv2 backend is the oracle.

    ``overlap_bands`` (D,) bool restricts the quadratic stage to band members;
    exclusive-region detections pass through untouched. None => all considered.
    """
    n = result.num_detections
    if n <= 1:
        return result
    if backend == "gpu":
        from .merge_gpu import merge_obb_detections_gpu  # lazy; Task 5

        return merge_obb_detections_gpu(
            result, policy=policy, metric=metric, threshold=threshold, runtime=runtime
        )

    if overlap_bands is None:
        band_idx = np.arange(n)
        passthrough_idx = np.array([], dtype=int)
    else:
        band_idx = np.where(overlap_bands)[0]
        passthrough_idx = np.where(~overlap_bands)[0]
    if band_idx.size <= 1:
        return result

    # confidence-descending order over band members.
    order = band_idx[np.argsort(result.confidences[band_idx])[::-1]]
    hulls: dict[int, tuple[np.ndarray, float]] = {}

    def hull(i: int) -> tuple[np.ndarray, float]:
        c = hulls.get(i)
        if c is None:
            c = _hull(result.corners[i])
            hulls[i] = c
        return c

    consumed = np.zeros(n, dtype=bool)
    merged_rows: list[tuple] = []  # unioned OBBs
    keep_single: list[int] = []  # nms survivors / lone members
    for i in order:
        if consumed[i]:
            continue
        group = [int(i)]
        pi, ai = hull(i)
        for j in order:
            if j == i or consumed[j]:
                continue
            pj, aj = hull(j)
            if _pair_overlap(pi, ai, pj, aj, metric) >= threshold:
                consumed[j] = True
                group.append(int(j))
        consumed[i] = True
        if policy == "nms" or len(group) == 1:
            keep_single.append(int(i))  # highest-conf member of the group
        else:  # nmm / greedy_nmm -> union
            merged_rows.append(_union_obb(result, group, result.frame_idx))

    return _assemble(result, keep_single, passthrough_idx.tolist(), merged_rows)


def _assemble(
    src: OBBResult,
    keep_single: list[int],
    passthrough: list[int],
    merged_rows: list[tuple],
) -> OBBResult:
    """Concatenate nms/lone survivors + passthrough + unioned rows into one OBBResult."""
    keep = sorted(keep_single + passthrough)
    cxs, cys, ws, hs, angs, confs, clss = [], [], [], [], [], [], []
    for i in keep:
        cxs.append(src.centroids[i, 0])
        cys.append(src.centroids[i, 1])
        # recover w,h from sizes/shapes is lossy; reuse stored corners' minAreaRect.
        (mcx, mcy), (mw, mh), mdeg = cv2.minAreaRect(src.corners[i].astype(np.float32))
        ws.append(mw)
        hs.append(mh)
        angs.append(src.angles[i])
        confs.append(src.confidences[i])
        clss.append(int(src.class_ids_or_zeros[i]))
    for cx, cy, w, h, ang, conf, cls in merged_rows:
        cxs.append(cx)
        cys.append(cy)
        ws.append(w)
        hs.append(h)
        angs.append(ang)
        confs.append(conf)
        clss.append(cls)
    if not cxs:
        return _empty_obb_result(src.frame_idx)
    cx = np.asarray(cxs, np.float32)
    cy = np.asarray(cys, np.float32)
    w = np.asarray(ws, np.float32)
    h = np.asarray(hs, np.float32)
    ang = np.asarray(angs, np.float32)
    ang_fixed, sizes, aspect = _normalize_obb_geometry(w, h, ang)
    corners = _corners_from_xywhr(cx, cy, w, h, ang_fixed)
    m = len(cxs)
    return OBBResult(
        frame_idx=src.frame_idx,
        centroids=np.stack([cx, cy], axis=1),
        angles=ang_fixed,
        sizes=sizes,
        shapes=np.stack([sizes, aspect], axis=1),
        confidences=np.asarray(confs, np.float32),
        corners=corners,
        detection_ids=OBBResult.make_detection_ids(src.frame_idx, m),
        class_ids=np.asarray(clss, np.int64),
    )
