"""GPU-native merge backend: pairwise matrix on device, greedy grouping on CPU,
union via a direct angle-search kernel over the member corner points.

Matches the ``cv2`` backend (``stages/merge.py``, the correctness oracle)
within tolerance. Only the ``(N, N)`` float overlap matrix is synced to host
memory; the O(n) grouping bookkeeping that follows is pure index arithmetic,
and any union geometry that needs to go back through the shared
``_normalize_obb_geometry`` / ``_corners_from_xywhr`` pipeline is computed on
device first.

Convention hazard (see ``stages/merge.py`` docstring on ``_assemble``): pairing
a ``(w, h, angle)`` triple with a mismatched angle convention silently rotates
non-square boxes 90 degrees while leaving area untouched, which lets an
area-only test pass over a broken box. Two defenses here:

1. Kept-single survivors (``policy="nms"`` or a lone group) are copied through
   by array indexing from ``result`` -- NEVER geometry-recomputed -- exactly
   like ``merge.py``'s own ``_assemble`` treats its keep/passthrough rows.
2. ``_union_via_kernel`` returns ``uw`` as the extent along the *same* axis as
   the returned ``angle`` and ``uh`` as the extent perpendicular to it, which
   is the exact convention ``_normalize_obb_geometry`` assumes of its
   ``(w, h, angle)`` input (see that function's docstring: angle describes the
   ``w``-axis before the major/minor swap is applied).
"""

from __future__ import annotations

import math

import numpy as np
import torch

from ....utils.rotated_iou import pairwise_obb_overlap
from ..result import OBBResult
from .merge import _assemble

__all__ = ["merge_obb_detections_gpu"]


def _greedy_groups(matrix: np.ndarray, order: np.ndarray, threshold: float) -> list:
    """Greedy grouping on a synced (N,N) overlap matrix (pure index bookkeeping).

    Mirrors ``merge.py``'s own confidence-descending greedy loop: walk
    detections highest-confidence first, absorb any not-yet-consumed detection
    whose overlap with the current anchor meets ``threshold`` into its group,
    and never revisit a consumed index. No geometry crosses this function.
    """
    n = matrix.shape[0]
    consumed = np.zeros(n, dtype=bool)
    groups: list[list[int]] = []
    for i in order:
        if consumed[i]:
            continue
        group = [int(i)]
        consumed[i] = True
        for j in order:
            if j == i or consumed[j]:
                continue
            if matrix[i, j] >= threshold:
                consumed[j] = True
                group.append(int(j))
        groups.append(group)
    return groups


def _union_via_kernel(
    pts: torch.Tensor, device: "torch.device | str", num_angles: int = 64
) -> tuple[float, float, float, float, float]:
    """Tightest rotated rect over member corner points -- exact, no rasterization.

    Same angle-projection idea as ``utils/obb_from_mask.rotated_rect_from_masks``,
    but applied DIRECTLY to the corner point set instead of a rasterized mask,
    so there is no grid-quantization error (rasterizing to a small pixel grid
    would cost a few px of accuracy for no benefit -- we already have exact
    points here).

    Projects all points onto ``num_angles`` candidate axes at once, takes the
    axis whose bounding extent has minimum area, and reconstructs the rect
    centered on that extent's midpoint. Fully vectorized: one
    ``(num_angles, P)`` matmul-shaped broadcast, no Python loop over angles or
    points.

    Returns ``(cx, cy, w, h, angle_rad)`` where ``w`` is the extent along the
    ``angle_rad`` axis and ``h`` is the extent along the perpendicular axis --
    the self-consistent convention ``_normalize_obb_geometry`` requires of its
    ``(w, h, angle)`` triple (see module docstring).
    """
    angles = torch.linspace(
        0.0, float(torch.pi), num_angles + 1, device=device, dtype=torch.float32
    )[
        :num_angles
    ]  # (A,)
    cos_a, sin_a = torch.cos(angles), torch.sin(angles)
    # Project every point onto every candidate axis and its perpendicular.
    u = pts[:, 0][None, :] * cos_a[:, None] + pts[:, 1][None, :] * sin_a[:, None]
    v = -pts[:, 0][None, :] * sin_a[:, None] + pts[:, 1][None, :] * cos_a[:, None]
    umin, umax = u.min(dim=1).values, u.max(dim=1).values  # (A,)
    vmin, vmax = v.min(dim=1).values, v.max(dim=1).values
    w_all, h_all = umax - umin, vmax - vmin
    best = int(torch.argmin(w_all * h_all))
    ang = float(angles[best])
    uw, uh = float(w_all[best]), float(h_all[best])
    # Centre in the rotated (u, v) frame -> back to (x, y) frame coords.
    uc = (umin[best] + umax[best]) * 0.5
    vc = (vmin[best] + vmax[best]) * 0.5
    cos_b, sin_b = math.cos(ang), math.sin(ang)
    ucx = float(uc) * cos_b - float(vc) * sin_b
    ucy = float(uc) * sin_b + float(vc) * cos_b
    return ucx, ucy, uw, uh, ang


def merge_obb_detections_gpu(
    result: OBBResult, *, policy: str, metric: str, threshold: float, runtime
) -> OBBResult:
    """GPU-native merge: pairwise matrix on device, greedy grouping on CPU,
    union via the shared angle-search kernel. Matches cv2 within tolerance.

    ``policy="nms"`` (or any singleton group) keeps the top-confidence member
    verbatim via array indexing -- no geometry is recomputed for it.
    ``policy="nmm"``/``"greedy_nmm"`` unions each multi-member group via
    ``_union_via_kernel`` and renormalizes through the same shared geometry
    pipeline ``merge.py``'s cv2 path uses, so the output contract (corner
    ordering, angle convention, sizes/aspect) is identical either way.
    """
    n = result.num_detections
    if n <= 1:
        return result

    device = getattr(runtime, "device", "cpu") if runtime is not None else "cpu"
    corners_t = torch.as_tensor(result.corners, dtype=torch.float32, device=device)
    matrix = pairwise_obb_overlap(corners_t, metric=metric).detach().cpu().numpy()
    order = np.argsort(result.confidences)[::-1]
    groups = _greedy_groups(matrix, order, threshold)

    keep_single: list[int] = []
    merged_rows: list[tuple] = []
    for group in groups:
        if policy == "nms" or len(group) == 1:
            top = group[int(np.argmax(result.confidences[group]))]
            keep_single.append(top)
            continue
        pts = torch.as_tensor(
            result.corners[group].reshape(-1, 2), dtype=torch.float32, device=device
        )
        ucx, ucy, uw, uh, uang = _union_via_kernel(pts, device)
        conf = float(result.confidences[group].max())
        top = group[int(np.argmax(result.confidences[group]))]
        cls = int(result.class_ids_or_zeros[top])
        merged_rows.append((ucx, ucy, uw, uh, uang, conf, cls))

    return _assemble(result, keep_single, [], merged_rows)
