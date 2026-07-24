"""Torch-only, fully batched pairwise rotated-box IoU/IoS kernel.

Computes an ``(N, N)`` overlap matrix for a batch of oriented rectangles
given as raw corner coordinates ``(N, 4, 2)`` -- no ``(cx, cy, w, h, angle)``
round-trip, so this cannot reproduce the ``cv2.minAreaRect`` angle-convention
bug that hit Task 3 (there, ``(w, h, angle)`` was paired with a differently
conventioned stored angle and silently rotated boxes 90 degrees; area was
invariant under the swap so tests never caught it). Corners carry no such
ambiguity.

Why this exists (perf, not elegance)
-------------------------------------
This kernel backs an optional ``gpu`` merge backend used only on the
native-CUDA SAHI sliced-inference path, where ``cv2.intersectConvexConvex``
would otherwise be called once per candidate pair inside a Python loop. At
~200 detections in a band that is ~40k pairs, each round-tripping through
host memory for cv2 -- exactly the cost this kernel is meant to eliminate by
staying entirely on-device and computing all pairs at once.

Algorithm (Sutherland-Hodgman polygon clipping, batched across all N^2 pairs)
------------------------------------------------------------------------------
1. **Orient**: every input quad is made counter-clockwise via a batched
   shoelace signed-area test; clockwise quads are reversed with
   ``torch.where`` (no Python branch), since clipping below assumes a
   consistent winding order for both the subject and the clip polygon.
2. **Broadcast**: the subject polygon is broadcast/padded to
   ``(N, N, K, 2)`` (``K`` starts at 4, padded to 8 to hold the clipped
   result) and the clip polygon to ``(N, N, 4, 2)``, so that pair ``(i, j)``
   is "subject i clipped against clip j" for every ``(i, j)`` simultaneously.
3. **Clip**: 4 sequential Sutherland-Hodgman passes, one per edge of the
   (quadrilateral, so exactly 4) clip polygon. This is the ONLY Python loop
   in the module and it is fixed-length (4 iterations, independent of N);
   every per-vertex/per-pair computation inside each iteration (inside-test
   cross products, edge-intersection points, output compaction) is a single
   batched tensor op over all ``(N, N, K)`` vertex slots at once. Because
   convex-quad-vs-convex-quad clipping can yield at most 8 vertices, the
   working buffer is a fixed ``(N, N, 8, 2)`` tensor plus a companion
   ``(N, N, 8)`` validity mask, rather than a variable-length list per pair
   (which would force a Python loop over pairs to build).
4. **Area**: a masked shoelace reduction over the padded, masked clipped
   polygon gives the ``(N, N)`` intersection area in one batched sum.
   Pairs whose clip result has fewer than 3 valid vertices (no true overlap,
   or a degenerate/collinear input polygon) are forced to zero area.
5. **Metric**: ``iou = inter / (area_i + area_j - inter)`` or
   ``ios = inter / min(area_i, area_j)``, with zero/near-zero denominators
   guarded by ``torch.where`` (never a Python ``if`` on a tensor value) so
   degenerate (zero-area) boxes yield 0.0 instead of NaN/Inf. The diagonal is
   set to exactly 1.0 by convention (a box's overlap with itself), overriding
   whatever the clip machinery computes there (which itself works fine).
"""

from __future__ import annotations

import torch

_EPS = 1e-9
_INSIDE_EPS = -1e-6
_MAX_CLIPPED_VERTS = 8  # convex quad ∩ convex quad has at most 8 vertices


def _signed_area(poly: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
    """Batched shoelace signed area over the last two dims.

    ``poly`` is ``(..., K, 2)``; ``mask`` (optional) is ``(..., K)`` bool,
    marking valid vertices, which are assumed to occupy a CONTIGUOUS PREFIX
    (guaranteed by ``_clip_one_edge``'s valid-first compaction). The true
    polygon successor of the LAST valid vertex is vertex 0, not the next
    PHYSICAL slot (which is padding) -- a plain ``torch.roll`` gets this
    wraparound term wrong (either using a bogus padding vertex, or dropping
    the closing term entirely), so it is corrected explicitly below via the
    same "wrap to slot 0" fix used in ``_clip_one_edge``.
    """
    x = poly[..., 0]
    y = poly[..., 1]
    if mask is not None:
        x = torch.where(mask, x, torch.zeros_like(x))
        y = torch.where(mask, y, torch.zeros_like(y))
    x_next = torch.roll(x, shifts=-1, dims=-1)
    y_next = torch.roll(y, shifts=-1, dims=-1)
    if mask is not None:
        next_mask_raw = torch.roll(mask, shifts=-1, dims=-1)
        wrap = mask & (~next_mask_raw)  # this vertex is the last valid one
        x_next = torch.where(wrap, x[..., 0:1].expand_as(x_next), x_next)
        y_next = torch.where(wrap, y[..., 0:1].expand_as(y_next), y_next)
    cross = x * y_next - x_next * y
    if mask is not None:
        # A term is only meaningful if its OWN vertex is valid -- the
        # wrap fix above already ensures the last valid vertex's successor
        # is the correct (valid) vertex 0, so gating on `mask` alone (rather
        # than `mask & next_mask`) is now sufficient and correct.
        cross = torch.where(mask, cross, torch.zeros_like(cross))
    return 0.5 * cross.sum(dim=-1)


def _make_ccw(corners: torch.Tensor) -> torch.Tensor:
    """Flip any clockwise-wound quad to counter-clockwise via ``torch.where``."""
    area = _signed_area(corners)  # (N,)
    is_cw = area < 0
    flipped = torch.flip(corners, dims=[-2])
    return torch.where(is_cw[:, None, None], flipped, corners)


def _clip_one_edge(
    subject: torch.Tensor,
    subject_mask: torch.Tensor,
    edge_a: torch.Tensor,
    edge_b: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """One Sutherland-Hodgman pass: clip ``subject`` against the half-plane
    to the LEFT of directed edge ``edge_a -> edge_b`` (CCW clip polygon
    convention: "left of every edge" == inside).

    ``subject``: (..., 8, 2) padded vertex buffer.
    ``subject_mask``: (..., 8) bool, valid-vertex flags.
    ``edge_a``, ``edge_b``: (..., 2) broadcastable against ``subject``.

    Returns the new padded ``(..., 8, 2)`` buffer and ``(..., 8)`` mask.
    Every op here is batched over all leading dims (pairs) and all 8 vertex
    slots at once -- there is no loop over vertices or pairs.
    """
    K = subject.shape[-2]
    edge_vec = (edge_b - edge_a).unsqueeze(-2)  # (..., 1, 2)
    a = edge_a.unsqueeze(-2)  # (..., 1, 2)

    rel = subject - a  # (..., K, 2)
    # z-component of edge_vec x rel: positive => left of (inside) the edge.
    cross = edge_vec[..., 0] * rel[..., 1] - edge_vec[..., 1] * rel[..., 0]  # (..., K)
    inside = cross >= _INSIDE_EPS
    inside = inside & subject_mask

    curr = subject
    curr_inside = inside
    # NOTE: `subject` is (..., K, 2) so its vertex axis is dims=-2, but
    # `inside`/`subject_mask` are (..., K) -- one fewer dim -- so their
    # vertex axis is dims=-1. Using the wrong axis here would silently roll
    # the pair-batch dim instead of the vertex dim for the mask/inside
    # tensors.
    nxt_raw = torch.roll(subject, shifts=-1, dims=-2)
    nxt_inside_raw = torch.roll(inside, shifts=-1, dims=-1)
    nxt_mask_raw = torch.roll(subject_mask, shifts=-1, dims=-1)

    # Valid vertices are always a contiguous prefix (enforced by the
    # valid-first compaction at the end of the previous pass, or by
    # construction for the very first call). The physical `torch.roll`
    # above is only correct for a vertex that is NOT the last valid one --
    # for the LAST valid vertex it wraps to the next PHYSICAL slot, which is
    # padding, not the true polygon successor (vertex 0). Detect that case
    # ("wrap": this vertex is valid but its physical successor is not) and
    # override next/next-mask/next-inside to point at slot 0 instead, which
    # is always the true first valid vertex whenever curr is valid at all
    # (since the prefix has length >= 1). This restores the correct closed
    # polygon topology despite the fixed-size padded buffer.
    wrap = subject_mask & (~nxt_mask_raw)
    first_pt = subject[..., 0:1, :].expand_as(nxt_raw)
    first_mask = subject_mask[..., 0:1].expand_as(nxt_mask_raw)
    first_inside = inside[..., 0:1].expand_as(nxt_inside_raw)
    nxt = torch.where(wrap.unsqueeze(-1), first_pt, nxt_raw)
    nxt_mask = torch.where(wrap, first_mask, nxt_mask_raw)
    nxt_inside = torch.where(wrap, first_inside, nxt_inside_raw)

    # An edge (curr -> next) only "exists" (for intersection purposes) if
    # curr is itself a valid vertex; `nxt_mask` is now always True whenever
    # `subject_mask` is True (either the literal next vertex, or -- via the
    # wrap fix above -- vertex 0), so this reduces to `subject_mask`, kept
    # explicit for clarity/defensiveness.
    edge_exists = subject_mask & nxt_mask

    # Edge-line intersection with the clip line, parametrized:
    # P = curr + t * (next - curr), solved from the same cross-product form.
    d = nxt - curr  # (..., K, 2)
    rel_curr = curr - a
    denom = edge_vec[..., 0] * d[..., 1] - edge_vec[..., 1] * d[..., 0]  # (..., K)
    numer = edge_vec[..., 0] * rel_curr[..., 1] - edge_vec[..., 1] * rel_curr[..., 0]
    safe_denom = torch.where(denom.abs() < _EPS, torch.ones_like(denom), denom)
    t = torch.where(denom.abs() < _EPS, torch.zeros_like(denom), -numer / safe_denom)
    t = t.clamp(0.0, 1.0)
    intersection = curr + t.unsqueeze(-1) * d  # (..., K, 2)

    # Sutherland-Hodgman per-edge output rule:
    #   inside & next_inside      -> emit next
    #   inside & !next_inside     -> emit intersection
    #   !inside & next_inside     -> emit intersection, then next (2 outputs)
    #   !inside & !next_inside    -> emit nothing
    # A single fixed-size (K) output slot per input slot can hold at most one
    # of these emissions; the rare "2 outputs" case is handled by giving the
    # INTERSECTION its own slot (this function is called once per input
    # vertex position, contributing up to 2 output vertices via two stacked
    # candidate slots below), keeping the whole thing loop-free.
    emit_next_as_curr_output = curr_inside & nxt_inside  # emit `next`
    emit_isect_as_curr_output = curr_inside & (~nxt_inside)  # emit `intersection`
    emit_isect_then_next = (~curr_inside) & nxt_inside  # emit isect AND next

    valid_curr_edge = edge_exists

    # First candidate slot (aligned with input position k): either `next`,
    # `intersection`, or nothing -- and additionally `intersection` when the
    # edge crosses from outside->inside (the "isect_then_next" case also
    # starts with an intersection point).
    first_is_next = emit_next_as_curr_output & valid_curr_edge
    first_is_isect = (
        emit_isect_as_curr_output | emit_isect_then_next
    ) & valid_curr_edge
    first_point = torch.where(first_is_next.unsqueeze(-1), nxt, intersection)
    first_valid = first_is_next | first_is_isect

    # Second candidate slot (only populated for the outside->inside case):
    # the `next` vertex, following its intersection point.
    second_valid = emit_isect_then_next & valid_curr_edge
    second_point = nxt

    # Interleave: 2*K output slots, position 2k = first, 2k+1 = second.
    out_points = torch.stack([first_point, second_point], dim=-2)  # (..., K, 2, 2)
    out_valid = torch.stack([first_valid, second_valid], dim=-1)  # (..., K, 2)
    out_points = out_points.reshape(*subject.shape[:-2], K * 2, 2)
    out_valid = out_valid.reshape(*subject_mask.shape[:-1], K * 2)

    # Compact the (K*2) sparse buffer down to a fixed _MAX_CLIPPED_VERTS
    # buffer via a stable valid-first sort -- this is a batched op (torch
    # sorts all leading "pair" dims simultaneously), not a Python loop.
    sort_key = (~out_valid).to(out_points.dtype)  # 0 for valid, 1 for invalid
    order = torch.argsort(sort_key, dim=-1, stable=True)
    compact_points = torch.gather(
        out_points, -2, order.unsqueeze(-1).expand(*order.shape, 2)
    )
    compact_valid = torch.gather(out_valid, -1, order)

    compact_points = compact_points[..., :_MAX_CLIPPED_VERTS, :]
    compact_valid = compact_valid[..., :_MAX_CLIPPED_VERTS]
    compact_points = torch.where(
        compact_valid.unsqueeze(-1), compact_points, torch.zeros_like(compact_points)
    )
    return compact_points, compact_valid


def _clip_polygons(
    subject: torch.Tensor, clip: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Clip padded ``subject`` (..., K, 2) against convex quad ``clip`` (..., 4, 2).

    Both leading dims are the same broadcastable "pair" shape. Runs exactly
    4 Sutherland-Hodgman passes (one per clip edge) -- the only Python loop
    in this module, fixed at 4 iterations regardless of N.
    """
    K = subject.shape[-2]
    mask = torch.ones(subject.shape[:-1], dtype=torch.bool, device=subject.device)
    # Pad subject up to _MAX_CLIPPED_VERTS so the buffer size is stable across
    # the 4 passes (each pass can only shrink or hold steady the valid count
    # for a convex clip, since clipping a convex polygon against a half-plane
    # never increases the vertex count by more than 1).
    if K < _MAX_CLIPPED_VERTS:
        pad_pts = torch.zeros(
            *subject.shape[:-2],
            _MAX_CLIPPED_VERTS - K,
            2,
            device=subject.device,
            dtype=subject.dtype,
        )
        pad_mask = torch.zeros(
            *mask.shape[:-1],
            _MAX_CLIPPED_VERTS - K,
            dtype=torch.bool,
            device=subject.device,
        )
        subject = torch.cat([subject, pad_pts], dim=-2)
        mask = torch.cat([mask, pad_mask], dim=-1)

    for edge_idx in range(4):
        edge_a = clip[..., edge_idx, :]
        edge_b = clip[..., (edge_idx + 1) % 4, :]
        subject, mask = _clip_one_edge(subject, mask, edge_a, edge_b)
    return subject, mask


def pairwise_obb_overlap(corners: torch.Tensor, metric: str = "iou") -> torch.Tensor:
    """Pairwise oriented-box overlap matrix, fully batched and torch-only.

    Parameters
    ----------
    corners:
        ``(N, 4, 2)`` box corners, any winding order, any device/dtype.
    metric:
        ``"iou"`` (intersection over union) or ``"ios"`` (intersection over
        the smaller box's area).

    Returns
    -------
    torch.Tensor
        ``(N, N)`` overlap matrix, same device/dtype as ``corners``.
        Symmetric, diagonal exactly 1.0, degenerate/zero-area boxes yield
        0.0 (never NaN) off-diagonal.
    """
    if metric not in ("iou", "ios"):
        raise ValueError(f"metric must be 'iou' or 'ios', got {metric!r}")

    n = corners.shape[0]
    device = corners.device
    dtype = corners.dtype
    if n == 0:
        return torch.zeros((0, 0), device=device, dtype=dtype)

    corners = corners.to(torch.float32)
    ccw = _make_ccw(corners)  # (N, 4, 2)
    areas = _signed_area(ccw).abs()  # (N,), CCW so already positive, abs for safety

    # Broadcast subject i vs clip j for every (i, j) pair at once.
    subject = ccw.unsqueeze(1).expand(n, n, 4, 2)  # (N, N, 4, 2): dim0=i (subject)
    clip = ccw.unsqueeze(0).expand(n, n, 4, 2)  # (N, N, 4, 2): dim1=j (clip)

    clipped_pts, clipped_mask = _clip_polygons(subject, clip)  # (N, N, 8, 2), (N, N, 8)

    n_valid = clipped_mask.sum(dim=-1)  # (N, N)
    inter_area = _signed_area(clipped_pts, clipped_mask).abs()
    inter_area = torch.where(n_valid >= 3, inter_area, torch.zeros_like(inter_area))

    area_i = areas.unsqueeze(1).expand(n, n)
    area_j = areas.unsqueeze(0).expand(n, n)

    if metric == "iou":
        denom = area_i + area_j - inter_area
    else:
        denom = torch.minimum(area_i, area_j)
    denom_safe = torch.where(denom > _EPS, denom, torch.ones_like(denom))
    overlap = torch.where(
        denom > _EPS, inter_area / denom_safe, torch.zeros_like(denom)
    )

    eye = torch.eye(n, device=device, dtype=torch.bool)
    overlap = torch.where(eye, torch.ones_like(overlap), overlap)

    # Numerical clipping noise can push slightly outside [0, 1] or break exact
    # symmetry by float epsilon; enforce both invariants explicitly.
    overlap = overlap.clamp(0.0, 1.0)
    overlap = 0.5 * (overlap + overlap.transpose(0, 1))

    return overlap.to(dtype)
