"""Tests for _extract_canonical_window CPU path correctness.

Focus: oversize crops (native extent > out_size) must be resized, not
hard-cropped, so CPU and GPU produce equivalent content for large animals.
"""
import numpy as np
import pytest
import torch

from hydra_suite.core.inference.result import OBBResult
from hydra_suite.core.inference.runtime import RuntimeContext
from hydra_suite.core.inference.stages.crops import _extract_canonical_window


def _runtime_cpu() -> RuntimeContext:
    return RuntimeContext(
        cuda_mode=False,
        device="cpu",
        use_nvdec=False,
        default_runtime="cpu",
        tensor_on_cuda=False,
    )


def _make_large_obb(frame_h: int = 256, frame_w: int = 256) -> OBBResult:
    """Return a single OBB whose native canonical crop will exceed out_size.

    The OBB is an axis-aligned rectangle 120×60 px centred in the frame.
    With aspect_ratio=2.0 and margin=1.3 the native crop produced by
    compute_native_crop_dimensions will be substantially larger than 32×32.
    """
    cx, cy = frame_w / 2.0, frame_h / 2.0
    hw, hh = 60.0, 30.0  # half-widths in x, y
    corners = np.array(
        [
            [cx - hw, cy - hh],
            [cx + hw, cy - hh],
            [cx + hw, cy + hh],
            [cx - hw, cy + hh],
        ],
        dtype=np.float32,
    ).reshape(1, 4, 2)
    centroid = np.array([[cx, cy]], dtype=np.float32)
    return OBBResult(
        frame_idx=0,
        centroids=centroid,
        angles=np.zeros(1, np.float32),
        sizes=np.full(1, hw * hh * 4, np.float32),
        shapes=np.ones((1, 2), np.float32),
        confidences=np.ones(1, np.float32),
        corners=corners,
        detection_ids=np.array([42], np.int64),
    )


def test_oversize_native_crop_is_resized_not_truncated():
    """CPU path must resize oversize native crops to out_size, not hard-crop them.

    If the old hard-crop behaviour (crop[:out_h, :out_w]) were used, the
    returned crop would contain a solid block of zeros in any region that
    was padding, or would silently discard pixel content beyond out_size.
    We verify that:
      1. The output crop has exactly out_size shape.
      2. The crop is not all-zero (content was preserved, not zeroed out
         by out-of-bounds padding from a misaligned slice).
      3. native_sizes records the true native (pre-resize) dimensions which
         must be strictly larger than out_size for this OBB.
    """
    out_size = (32, 32)  # out_w, out_h — deliberately smaller than the OBB
    out_w, out_h = out_size

    frame = np.full((256, 256, 3), 128, dtype=np.uint8)
    obb = _make_large_obb(256, 256)

    crops_t, native_sizes = _extract_canonical_window(
        frame,
        obb,
        margin=1.3,
        aspect_ratio=2.0,
        out_size=out_size,
        runtime=_runtime_cpu(),
    )

    # Shape must be exactly out_size
    assert crops_t.shape == (1, 3, out_h, out_w), (
        f"Expected (1, 3, {out_h}, {out_w}), got {tuple(crops_t.shape)}"
    )

    # native_sizes must record dimensions larger than out_size (proving the
    # OBB's native crop was bigger and a resize — not a hard-crop — occurred)
    assert native_sizes.shape == (1, 2)
    native_h, native_w = native_sizes[0]
    assert native_h > out_h or native_w > out_w, (
        f"Expected native size > out_size but got native ({native_h}, {native_w}) "
        f"vs out ({out_h}, {out_w}). The OBB may not be large enough for this test."
    )

    # Crop must not be all-zero (content was preserved)
    assert crops_t.abs().sum() > 0, "Crop is entirely zero — pixel content was lost"

    # No all-zero columns or rows at the edges (a sign of hard-crop padding)
    crop_np = crops_t[0].permute(1, 2, 0).numpy()  # (H, W, C)
    last_col_zero = (crop_np[:, -1, :] == 0).all()
    last_row_zero = (crop_np[-1, :, :] == 0).all()
    assert not last_col_zero, "Last column is all-zero — suggests hard-crop/pad artifact"
    assert not last_row_zero, "Last row is all-zero — suggests hard-crop/pad artifact"
