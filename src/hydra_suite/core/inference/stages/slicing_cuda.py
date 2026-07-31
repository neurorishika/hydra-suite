"""Device-tensor concat hook for the raw (``tensor_on_cuda``) merge path.

``_concat_raw`` is the on-device (no sync) concatenation of per-region
``_RawOBBTensors`` that ``obb.merge_per_frame``'s raw branches
(``"plain"`` and ``_merge_raw_overlap_band_nms``, Task 8) delegate to. Per-tile
extraction (turning ultralytics results into ``_RawOBBTensors`` and remapping
by translation) and the per-frame merge-or-passthrough decision both now live
in ``obb.extract_with_transform`` / ``obb.merge_per_frame``, driven by
``run_obb``'s shared plan/execute/extract/merge loop (Task 9) -- this module
no longer assembles frames itself.

Tiling, tile-job building, letterboxing and chunked prediction all live in
``slicing.py`` and are shared by every path.

``torch`` is imported at module scope here, so callers keep importing this
module lazily (function-level) -- CPU/MPS installs never pay for it.
"""

from __future__ import annotations

import torch


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
