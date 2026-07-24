"""Device-tensor extraction hook for the sliced OBB path.

This module owns ONLY the ``runtime.tensor_on_cuda`` half of the sliced
pipeline: turning per-tile ultralytics results into ``_RawOBBTensors`` without
a device sync, remapping them into frame space by pure translation, and
deciding when a cross-tile merge (the one sync point) is unavoidable.

Tiling, tile-job building, letterboxing and chunked prediction all live in
``slicing.py`` and are shared by every path -- this file used to duplicate them
and the two copies drifted (overlap gate, cap ordering, merge backend), which
is what let finding C1 (tier dispatch keyed off the wrong flag) hide.

``torch`` is imported at module scope here, so ``slicing.py`` keeps importing
this module lazily (function-level) -- CPU/MPS installs never pay for it.
"""

from __future__ import annotations

import math

import torch

from .slicing import SlicePlan, tiles_overlap


def _remap_raw(raw, x0: int, y0: int):
    """Return a copy of a ``_RawOBBTensors`` shifted by ``(x0, y0)`` on-device.

    PURE TRANSLATION only: ``xywhr[:, 2:5]`` (w, h, angle), ``conf`` and ``cls``
    are untouched. A translation cannot change size or orientation -- see the
    Task 3 bug this guards against (pairing ``cv2.minAreaRect``'s (w, h, angle)
    with a differently-conventioned stored angle silently rotated boxes 90
    degrees; area stayed invariant so every test passed).
    """
    from .obb import _RawOBBTensors

    if raw.xywhr.shape[0] == 0:
        return raw
    xywhr = raw.xywhr.clone()
    xywhr[:, 0] += x0
    xywhr[:, 1] += y0
    corners = raw.corners.clone()
    corners[..., 0] += x0
    corners[..., 1] += y0
    return _RawOBBTensors(
        frame_idx=raw.frame_idx,
        xywhr=xywhr,
        corners=corners,
        conf=raw.conf,
        cls=raw.cls,
    )


def _concat_raw(parts, frame_idx: int):
    """Concatenate per-tile ``_RawOBBTensors`` for one frame, entirely on-device."""
    from .obb import _RawOBBTensors

    non_empty = [p for p in parts if p.xywhr.shape[0] > 0]
    if not non_empty:
        dev = parts[0].xywhr.device if parts else torch.device("cpu")
        return _RawOBBTensors(
            frame_idx=frame_idx,
            xywhr=torch.zeros((0, 5), dtype=torch.float32, device=dev),
            corners=torch.zeros((0, 4, 2), dtype=torch.float32, device=dev),
            conf=torch.zeros(0, dtype=torch.float32, device=dev),
            cls=torch.zeros(0, dtype=torch.float32, device=dev),
        )
    return _RawOBBTensors(
        frame_idx=frame_idx,
        xywhr=torch.cat([p.xywhr for p in non_empty], dim=0),
        corners=torch.cat([p.corners for p in non_empty], dim=0),
        conf=torch.cat([p.conf for p in non_empty], dim=0),
        cls=torch.cat(
            [
                (
                    p.cls
                    if p.cls is not None
                    else torch.zeros(p.xywhr.shape[0], device=p.xywhr.device)
                )
                for p in non_empty
            ],
            dim=0,
        ),
    )


def extract_raw_tile(result, frame_idx: int, config, runtime):
    """Extract one tile's detections as ``_RawOBBTensors`` (zero ``.cpu()``)."""
    from .obb import (
        _extract_raw_tensors,
        _extract_raw_tensors_from_boxes,
        _extract_raw_tensors_from_masks,
    )

    model_task = config.direct.model_task
    if model_task == "detect":
        return _extract_raw_tensors_from_boxes(
            result,
            frame_idx,
            math.radians(config.direct.fixed_angle_deg),
            runtime.device,
        )
    if model_task == "segment":
        return _extract_raw_tensors_from_masks(
            result,
            frame_idx,
            runtime.device,
            config.raw_detection_cap,
            num_angles=config.direct.seg_num_angles,
            crop_size=config.direct.seg_crop_size,
            pad_ratio=config.direct.seg_pad_ratio,
            mask_threshold=config.direct.seg_mask_threshold,
        )
    return _extract_raw_tensors(result, frame_idx, runtime.device)


def assemble_raw_frames(
    jobs: list[tuple[int, int, int]],
    results: list,
    n_frames: int,
    plan: SlicePlan,
    config,
    runtime,
):
    """Per-frame ``_RawOBBTensors`` (or merged ``OBBResult``) from tile results.

    Whether cross-tile dedup is needed is decided by ``tiles_overlap(plan.tiles)``
    -- a geometry predicate over the tile boxes, never by
    ``slice_cfg.overlap_*_ratio`` (``get_slice_bboxes`` flushes the last tile in
    each axis to the frame edge, so tiles genuinely overlap even at a
    configured ratio of 0.0; gating on the config ratio silently double-counts
    detections -- the exact bug Task 6 shipped and had to fix twice).

    When no two planned tiles intersect, every frame's tiles are concatenated
    and returned as a ``_RawOBBTensors`` with zero device syncs. Otherwise each
    frame is materialized once (the only sync point) and merged, with the merge
    restricted to overlap-band members (``overlap_bands``); exclusive-region
    detections pass straight through.
    """
    from .merge import band_membership, merge_obb_detections
    from .obb import _apply_raw_detection_cap, materialize_tensors

    slice_cfg = config.direct.slice
    per_frame: dict[int, list] = {fi: [] for fi in range(n_frames)}
    for (fi, x0, y0), res in zip(jobs, results):
        raw = extract_raw_tile(res, fi, config, runtime)
        per_frame[fi].append(_remap_raw(raw, max(0, x0), max(0, y0)))

    any_overlap = tiles_overlap(plan.tiles)

    out = []
    for fi in range(n_frames):
        concat = _concat_raw(per_frame[fi], fi)
        if not any_overlap or concat.xywhr.shape[0] <= 1:
            out.append(concat)  # preserve _RawOBBTensors end-to-end
            continue
        # Overlap possible: materialize for the cross-tile merge (this is the
        # only sync point). materialize_tensors applies the raw detection cap,
        # so the O(n^2) merge input is bounded exactly as on the host path.
        materialized = materialize_tensors(concat, config.raw_detection_cap)
        bands = band_membership(materialized.corners, plan.tiles)
        merged = merge_obb_detections(
            materialized,
            policy=slice_cfg.merge_policy,
            metric=slice_cfg.merge_metric,
            threshold=slice_cfg.merge_threshold,
            backend=slice_cfg.merge_backend,
            overlap_bands=bands,
            runtime=runtime,
        )
        out.append(_apply_raw_detection_cap(merged, config.raw_detection_cap))
    return out
