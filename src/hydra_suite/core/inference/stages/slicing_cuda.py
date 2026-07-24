from __future__ import annotations

import math

import torch

from .slicing import plan_slices, tiles_overlap


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


def run_direct_sliced_cuda(frames, model, config, runtime, imgsz):
    """Native-cuda sliced path: preserve ``_RawOBBTensors``; band-only sync when merging.

    Mirrors ``run_direct_sliced``'s non-cuda tiling/job-building logic, but
    tiles are on-device views (no numpy copy), the model is called once on a
    single GPU-letterboxed batch (mirroring ``_run_direct``'s CUDA-tensor
    branch in ``obb.py``), and every tile's detections are extracted via the
    zero-``.cpu()`` raw-tensor extractors and remapped by pure translation.

    Whether cross-tile dedup is needed is decided by ``tiles_overlap(plan.tiles)``
    -- a geometry predicate over the tile boxes, never by
    ``slice_cfg.overlap_*_ratio`` (``get_slice_bboxes`` flushes the last tile in
    each axis to the frame edge, so tiles genuinely overlap even at a
    configured ratio of 0.0; gating on the config ratio silently double-counts
    detections -- the exact bug Task 6 shipped and had to fix twice).

    When no two planned tiles intersect, every frame's tiles are concatenated
    and returned as a ``_RawOBBTensors`` with zero device syncs. Otherwise each
    frame is materialized once (the only sync point) and merged via the
    existing cv2 merge pipeline, which itself restricts the O(n^2) hull/IoU
    work to band members (``overlap_bands``) and passes exclusive-region
    detections straight through.
    """
    from .merge import band_membership, merge_obb_detections
    from .obb import (
        _apply_raw_detection_cap,
        _extract_raw_tensors,
        _extract_raw_tensors_from_boxes,
        _extract_raw_tensors_from_masks,
        _gpu_letterbox_batch,
        _invert_letterbox_on_result,
        materialize_tensors,
    )

    slice_cfg = config.direct.slice
    model_task = config.direct.model_task
    frame_hw = (int(frames[0].shape[0]), int(frames[0].shape[1]))
    plan = plan_slices(
        frame_hw,
        slice_cfg,
        imgsz,
        None,
        ref_object_px=slice_cfg.reference_body_px,
    )

    # Tile every frame on-device (zero-copy views), collect tiles + provenance.
    jobs, tiles = [], []
    for fi, frame in enumerate(frames):
        for x0, y0, x1, y1 in plan.tiles:
            jobs.append((fi, x0, y0))
            tiles.append(frame[y0:y1, x0:x1])
        if plan.full_frame:
            jobs.append((fi, 0, 0))
            tiles.append(frame)

    batched, lb_params = _gpu_letterbox_batch(tiles, imgsz)
    results = model.predict(
        batched,
        conf=config.direct.confidence_floor,
        iou=1.0,
        classes=config.target_classes or None,
        verbose=False,
        device=runtime.device,
    )

    # Invert the letterbox so extract functions see tile-local coordinates,
    # exactly as ``_run_direct``'s own CUDA-tensor branch does. When every
    # tile is exactly ``imgsz`` x ``imgsz`` (the common auto_model case) this
    # is r=1, no pad -> a true no-op, skipped entirely so no result tensor is
    # touched. Real letterboxing only kicks in for a custom tile size that
    # differs from imgsz, or the (rare) full-frame pass.
    for tile_img, res, (r, pad_left, pad_top) in zip(tiles, results, lb_params):
        if r != 1.0 or pad_left != 0.0 or pad_top != 0.0:
            _invert_letterbox_on_result(
                res,
                r,
                pad_left,
                pad_top,
                orig_shape=(int(tile_img.shape[0]), int(tile_img.shape[1])),
            )

    per_frame = {fi: [] for fi in range(len(frames))}
    for job, res in zip(jobs, results):
        fi, x0, y0 = job
        if model_task == "detect":
            raw = _extract_raw_tensors_from_boxes(
                res, fi, math.radians(config.direct.fixed_angle_deg), runtime.device
            )
        elif model_task == "segment":
            raw = _extract_raw_tensors_from_masks(
                res,
                fi,
                runtime.device,
                config.raw_detection_cap,
                num_angles=config.direct.seg_num_angles,
                crop_size=config.direct.seg_crop_size,
                pad_ratio=config.direct.seg_pad_ratio,
                mask_threshold=config.direct.seg_mask_threshold,
            )
        else:
            raw = _extract_raw_tensors(res, fi, runtime.device)
        per_frame[fi].append(_remap_raw(raw, x0, y0))

    # Whether ANY two planned tiles actually intersect. This is a pure predicate
    # over the tile boxes -- no detection data, no device sync, computed once.
    #
    # It MUST NOT be derived from slice_cfg.overlap_*_ratio. get_slice_bboxes
    # flushes the last tile in each axis to the frame edge, so tiles genuinely
    # overlap even at a configured ratio of 0.0 (a 300px frame with 256px tiles
    # yields [0,256) and [44,300) -- 212px of real overlap). Gating on the config
    # ratio skips dedup while tiles overlap, silently double-counting detections.
    # This is the exact bug Task 6 shipped and had to fix; do not reintroduce it.
    any_overlap = tiles_overlap(plan.tiles)

    out = []
    for fi in range(len(frames)):
        concat = _concat_raw(per_frame[fi], fi)
        if not any_overlap or concat.xywhr.shape[0] <= 1:
            out.append(concat)  # preserve _RawOBBTensors end-to-end
            continue
        # overlap possible: materialize for the cross-tile merge (this is the
        # only sync point). band_membership restricts the O(n^2) hull/IoU
        # work to band members; exclusive-region detections pass through.
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
