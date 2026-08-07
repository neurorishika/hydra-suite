"""Tests for the live canonical pose-crop builder ``extract_canonical_crops_batch``.

The live pose path now warps every detection onto ``geometry``'s FIXED canvas
(Layer 1: rotation + translation only, no scale) regardless of the OBB's
native pixel extent, so ``native_sizes`` rows are uniformly
``[canvas_h, canvas_w]`` -- there is nothing left for ``run_pose_batch`` to
slice back to a smaller native region.
"""

import numpy as np

from hydra_suite.core.canonicalization.geometry import CanonicalGeometry
from hydra_suite.core.inference.result import OBBResult
from hydra_suite.core.inference.runtime import RuntimeContext
from hydra_suite.core.inference.stages.crops import extract_canonical_crops_batch


def _runtime_cpu() -> RuntimeContext:
    return RuntimeContext(
        cuda_mode=False,
        device="cpu",
        use_nvdec=False,
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


def test_fixed_canvas_crop_preserves_content_and_records_canvas_size():
    """The live builder must warp onto the fixed canonical canvas and record it.

    Verifies that:
      1. native_sizes records the geometry's fixed canvas dimensions (every
         crop is already that size -- nothing to slice back to).
      2. The crop tensor is exactly the fixed canvas size.
      3. The crop content is preserved (not zeroed).
    """
    frame = np.full((256, 256, 3), 128, dtype=np.uint8)
    obb = _make_large_obb(256, 256)
    geometry = CanonicalGeometry.from_reference(
        reference_body_px=40.0, aspect_ratio=2.0, margin=1.3
    )

    batch = extract_canonical_crops_batch([frame], [obb], geometry, _runtime_cpu())

    assert batch.native_sizes.shape == (1, 2)
    native_h, native_w = int(batch.native_sizes[0, 0]), int(batch.native_sizes[0, 1])
    assert (native_h, native_w) == (
        geometry.canvas_h,
        geometry.canvas_w,
    ), "native_sizes must record the fixed canonical canvas dimensions"

    # Crop tensor is exactly the fixed canvas size -- no padding to reconcile.
    assert batch.crops.shape[0] == 1
    assert batch.crops.shape[2] == geometry.canvas_h
    assert batch.crops.shape[3] == geometry.canvas_w

    # The crop must contain preserved content (a uniform 128 frame warps to
    # non-zero pixels), not be zeroed out.
    assert batch.crops[0].abs().sum() > 0, "canonical crop is entirely zero"
