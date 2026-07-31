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

import logging

import cv2
import numpy as np

from ..result import OBBResult
from .obb import _corners_from_xywhr, _empty_obb_result, _normalize_obb_geometry

logger = logging.getLogger(__name__)


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
    except cv2.error as exc:
        logger.debug(
            "cv2.intersectConvexConvex failed for a hull pair; treating as no "
            "overlap: %s",
            exc,
        )
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

    Returns the RAW, self-consistent ``(cx, cy, w, h, angle_rad, conf, cls)``
    straight off ``cv2.minAreaRect`` -- ``w`` is the extent along the
    ``angle_rad`` axis and ``h`` the extent perpendicular to it, the same
    convention ``_union_via_kernel`` (the gpu backend) returns.

    It must NOT pre-canonicalize the angle here: ``_assemble`` feeds this exact
    ``(w, h, angle)`` triple through ``_normalize_obb_geometry`` itself. Doing
    it in both places applies the ``w < h`` +90-degree major-axis correction
    TWICE, which lands the reported angle back on the MINOR axis and emits a box
    rotated 90 degrees from the true footprint. Area is invariant under that
    swap, so it is invisible to ``sizes`` and shows up only as visibly
    cross-wise boxes on the sliced (SAHI) path -- and only for the subset of
    detections that actually reach a multi-member union in a tile overlap band.
    """
    pts = members.corners[idxs].reshape(-1, 2).astype(np.float32)
    (cx, cy), (w, h), angle_deg = cv2.minAreaRect(pts)
    conf = float(members.confidences[idxs].max())
    top = idxs[int(np.argmax(members.confidences[idxs]))]
    cls = int(members.class_ids_or_zeros[top])
    return (
        float(cx),
        float(cy),
        float(w),
        float(h),
        float(np.deg2rad(angle_deg)),
        conf,
        cls,
    )


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

    # The band split is backend-INDEPENDENT and therefore happens before
    # dispatch: both backends must consider the same candidate set, or they
    # stop being oracle/implementation of one another. (Finding I1: the gpu
    # backend used to be dispatched before this split and never received
    # ``overlap_bands``, so two genuinely distinct touching animals inside one
    # tile's exclusive region -- ios >= threshold, but with no possible
    # cross-tile duplicate -- were unioned under ``gpu`` and left alone under
    # ``cv2``.)
    if overlap_bands is None:
        band_idx = np.arange(n)
        passthrough_idx = np.array([], dtype=int)
    else:
        band_idx = np.where(overlap_bands)[0]
        passthrough_idx = np.where(~overlap_bands)[0]
    if band_idx.size <= 1:
        return result

    if backend == "gpu":
        from .merge_gpu import merge_obb_detections_gpu  # lazy; Task 5

        return merge_obb_detections_gpu(
            result,
            policy=policy,
            metric=metric,
            threshold=threshold,
            runtime=runtime,
            band_idx=band_idx,
            passthrough_idx=passthrough_idx,
        )

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
    """Concatenate nms/lone survivors + passthrough + unioned rows into one OBBResult.

    Kept-single and passthrough detections were never modified, so their fields
    are copied straight through by array indexing -- NO geometry round-trip
    through ``cv2.minAreaRect``. Recovering (w, h) from ``minAreaRect`` on the
    stored corners and pairing it with the ORIGINAL ``src.angles[i]`` is unsound:
    minAreaRect can return the same physical box parametrized with w/h swapped
    (angle offset by 90 degrees), which silently rotates non-square boxes.  Only
    ``merged_rows`` (from ``_union_obb``, which pairs w/h with the angle from the
    SAME minAreaRect call) legitimately need synthesized geometry.
    """
    keep = np.asarray(sorted(keep_single + passthrough), dtype=np.int64)
    n_keep = keep.size
    n_merged = len(merged_rows)
    if n_keep == 0 and n_merged == 0:
        return _empty_obb_result(src.frame_idx)

    centroids = src.centroids[keep]
    angles = src.angles[keep]
    sizes = src.sizes[keep]
    shapes = src.shapes[keep]
    confidences = src.confidences[keep]
    corners = src.corners[keep]
    class_ids = src.class_ids_or_zeros[keep]

    if merged_rows:
        cx = np.asarray([r[0] for r in merged_rows], np.float32)
        cy = np.asarray([r[1] for r in merged_rows], np.float32)
        w = np.asarray([r[2] for r in merged_rows], np.float32)
        h = np.asarray([r[3] for r in merged_rows], np.float32)
        ang = np.asarray([r[4] for r in merged_rows], np.float32)
        ang_fixed, m_sizes, m_aspect = _normalize_obb_geometry(w, h, ang)
        m_corners = _corners_from_xywhr(cx, cy, w, h, ang_fixed)
        m_confs = np.asarray([r[5] for r in merged_rows], np.float32)
        m_cls = np.asarray([r[6] for r in merged_rows], np.int64)

        centroids = np.concatenate([centroids, np.stack([cx, cy], axis=1)], axis=0)
        angles = np.concatenate([angles, ang_fixed], axis=0)
        sizes = np.concatenate([sizes, m_sizes], axis=0)
        shapes = np.concatenate([shapes, np.stack([m_sizes, m_aspect], axis=1)], axis=0)
        confidences = np.concatenate([confidences, m_confs], axis=0)
        corners = np.concatenate([corners, m_corners], axis=0)
        class_ids = np.concatenate([class_ids, m_cls], axis=0)

    m = n_keep + n_merged
    return OBBResult(
        frame_idx=src.frame_idx,
        centroids=centroids.astype(np.float32, copy=False),
        angles=angles.astype(np.float32, copy=False),
        sizes=sizes.astype(np.float32, copy=False),
        shapes=shapes.astype(np.float32, copy=False),
        confidences=confidences.astype(np.float32, copy=False),
        corners=corners.astype(np.float32, copy=False),
        detection_ids=OBBResult.make_detection_ids(src.frame_idx, m),
        class_ids=class_ids.astype(np.int64, copy=False),
    )
