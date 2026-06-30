"""Tests for the live canonical pose-crop builder ``extract_canonical_crops_batch``.

The live pose path warps each detection to its NATIVE extent and pads (never
resizes) to the window-wide max canvas, recording each crop's true native
``[h, w]`` in ``native_sizes`` so ``run_pose_batch`` can slice back to native.
This test verifies the native-extent contract for an oversize OBB.
"""

import numpy as np

from hydra_suite.core.inference.result import OBBResult
from hydra_suite.core.inference.runtime import RuntimeContext
from hydra_suite.core.inference.stages.crops import extract_canonical_crops_batch


def _runtime_cpu() -> RuntimeContext:
    return RuntimeContext(
        cuda_mode=False,
        device="cpu",
        use_nvdec=False,
        default_runtime="cpu",
        tensor_on_cuda=False,
    )


def _make_large_obb(frame_h: int = 256, frame_w: int = 256) -> OBBResult:
    """Return a single OBB whose native canonical crop is substantial (120x60 px)."""
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


def test_native_extent_crop_preserves_content_and_records_native_size():
    """The live builder must warp to native extent (no resize) and record it.

    Verifies that:
      1. native_sizes records the true native crop dimensions (matching
         compute_native_crop_dimensions for the OBB).
      2. The crop content is preserved within its native extent (not zeroed).
      3. The padded crop tensor is at least as large as the native extent.
    """
    from hydra_suite.core.canonicalization.crop import compute_native_crop_dimensions

    frame = np.full((256, 256, 3), 128, dtype=np.uint8)
    obb = _make_large_obb(256, 256)
    ar, mg = 2.0, 1.3

    batch = extract_canonical_crops_batch([frame], [obb], ar, mg, _runtime_cpu())

    assert batch.native_sizes.shape == (1, 2)
    native_h, native_w = int(batch.native_sizes[0, 0]), int(batch.native_sizes[0, 1])

    pad = max(0.0, mg - 1.0)
    cw, ch = compute_native_crop_dimensions(obb.corners[0], ar, pad)
    assert (native_h, native_w) == (
        int(ch),
        int(cw),
    ), "native_sizes must record the true native crop dimensions"

    # Padded crop tensor must accommodate the native extent.
    assert batch.crops.shape[0] == 1
    assert batch.crops.shape[2] >= native_h
    assert batch.crops.shape[3] >= native_w

    # The native region must contain preserved content (a uniform 128 frame
    # warps to non-zero pixels), not be zeroed out.
    native_region = batch.crops[0, :, :native_h, :native_w]
    assert native_region.abs().sum() > 0, "native crop region is entirely zero"
