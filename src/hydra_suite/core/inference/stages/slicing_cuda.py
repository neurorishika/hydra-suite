"""Device-tensor extraction hook for the sliced OBB path.

This module owns ONLY the ``runtime.tensor_on_cuda`` half of the sliced
pipeline: turning per-tile ultralytics results into ``_RawOBBTensors`` without
a device sync, and remapping them into frame space by pure translation. The
per-frame merge-or-passthrough decision (whether a cross-tile merge -- the
one sync point -- is unavoidable) is delegated to
``obb.merge_per_frame(..., "overlap_band_nms", ...)`` (Task 8), which owns
that decision for both the raw and numpy universes so the two never drift
again the way they did before (overlap gate, cap ordering, merge backend --
finding C1).

Tiling, tile-job building, letterboxing and chunked prediction all live in
``slicing.py`` and are shared by every path.

``torch`` is imported at module scope here, so ``slicing.py`` keeps importing
this module lazily (function-level) -- CPU/MPS installs never pay for it.
"""

from __future__ import annotations

import torch

from .slicing import SlicePlan


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


def assemble_raw_frames(
    jobs: list[tuple[int, int, int]],
    results: list,
    n_frames: int,
    plan: SlicePlan,
    config,
    runtime,
):
    """Per-frame ``_RawOBBTensors`` (or merged ``OBBResult``) from tile results.

    Extracts each tile's raw device tensors (no device sync), remaps them into
    frame space by pure translation, then delegates the per-frame
    merge-or-passthrough decision to
    ``obb.merge_per_frame(..., "overlap_band_nms", ...)`` -- see that
    function's docstring (and ``_merge_raw_overlap_band_nms``) for the
    ``tiles_overlap`` gate / materialize / cap-ordering contract this used to
    implement inline.
    """
    from .obb import extract_with_transform, merge_per_frame
    from .regions import Affine

    per_frame: dict[int, list] = {fi: [] for fi in range(n_frames)}
    for (fi, x0, y0), res in zip(jobs, results):
        per_frame[fi].append(
            extract_with_transform(
                res,
                fi,
                config.direct.model_task,
                Affine(offset=(float(max(0, x0)), float(max(0, y0)))),
                config,
                runtime,
            )
        )

    return [
        merge_per_frame(per_frame[fi], "overlap_band_nms", plan, config, runtime)
        for fi in range(n_frames)
    ]
