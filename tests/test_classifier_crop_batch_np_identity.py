"""The uint8 classifier crop batch is BYTE-identical to the float32 round trip.

``run_headtail_batch`` / ``run_cnn_batch`` used to build a ``(N, C, H, W)``
float32 ``[0, 1]`` :class:`CropBatch` and immediately quantise it back to HWC
uint8 for ``predict_batch``. :func:`extract_classifier_crops_batch_np` skips
that detour. These tests pin the two claims that make the skip legal:

1. ``uint8 -> /255 (float32) -> *255 -> clip -> astype(uint8)`` is the identity
   for every one of the 256 byte values, so nothing was ever changed by it;
2. on real warped crops the two builders agree exactly, row for row, along with
   every metadata field.
"""

import numpy as np
import torch

from hydra_suite.core.canonicalization.geometry import CanonicalGeometry
from hydra_suite.core.inference.result import OBBResult
from hydra_suite.core.inference.stages.crops import (
    extract_classifier_crops_batch,
    extract_classifier_crops_batch_np,
)

_GEOM = CanonicalGeometry(canvas_wh=(64, 32), margin=1.3, aspect_ratio=2.0)


def _obb(frame_idx: int, n: int) -> OBBResult:
    cx = np.linspace(20, 40, n).astype(np.float32)
    corners = np.stack(
        [
            np.stack([cx - 6, np.full(n, 14, np.float32)], -1),
            np.stack([cx + 6, np.full(n, 14, np.float32)], -1),
            np.stack([cx + 6, np.full(n, 26, np.float32)], -1),
            np.stack([cx - 6, np.full(n, 26, np.float32)], -1),
        ],
        axis=1,
    ).astype(np.float32)
    return OBBResult(
        frame_idx=frame_idx,
        centroids=np.stack([cx, np.full(n, 20, np.float32)], -1),
        angles=np.zeros(n, np.float32),
        sizes=np.full(n, 100, np.float32),
        shapes=np.ones((n, 2), np.float32),
        confidences=np.ones(n, np.float32),
        corners=corners,
        detection_ids=np.array([frame_idx * 10000 + s for s in range(n)], np.int64),
    )


def test_uint8_float32_round_trip_is_the_identity_for_every_byte():
    """The exact arithmetic the removed detour performed, on all 256 values."""
    src = np.arange(256, dtype=np.uint8).reshape(1, 16, 16, 1).repeat(3, axis=3)
    tensor = torch.from_numpy(np.ascontiguousarray(src)).permute(0, 3, 1, 2)
    tensor = tensor.float() / 255.0
    hwc = np.ascontiguousarray(tensor.permute(0, 2, 3, 1).cpu().numpy())
    back = (hwc * 255.0).clip(0, 255).astype(np.uint8)
    assert np.array_equal(back, src)


def test_numpy_batch_is_byte_identical_to_tensor_batch():
    rng = np.random.default_rng(0)
    frames = [
        rng.integers(0, 256, (64, 64, 3), dtype=np.uint8),
        rng.integers(0, 256, (64, 64, 3), dtype=np.uint8),
    ]
    obbs = [_obb(0, 3), _obb(1, 2)]

    tensor_batch = extract_classifier_crops_batch(frames, obbs, _GEOM)
    unpacked = np.ascontiguousarray(
        tensor_batch.crops.permute(0, 2, 3, 1).cpu().numpy()
    )
    reference = (unpacked * 255.0).clip(0, 255).astype(np.uint8)

    np_batch = extract_classifier_crops_batch_np(frames, obbs, _GEOM)

    assert len(np_batch.crops) == reference.shape[0] == 5
    for i, crop in enumerate(np_batch.crops):
        assert crop.dtype == np.uint8
        assert crop.shape == (_GEOM.canvas_h, _GEOM.canvas_w, 3)
        assert np.array_equal(crop, reference[i]), f"crop {i} diverged"

    assert np.array_equal(np_batch.detection_ids, tensor_batch.detection_ids)
    assert np.array_equal(np_batch.frame_index, tensor_batch.frame_index)
    assert np.array_equal(np_batch.native_sizes, tensor_batch.native_sizes)
    assert np_batch.frames() == tensor_batch.frames()
    for frame_idx in np_batch.frames():
        assert np.array_equal(
            np_batch.select_frame(frame_idx), tensor_batch.select_frame(frame_idx)
        )


def test_numpy_batch_handles_empty_window():
    obb = _obb(0, 0)
    batch = extract_classifier_crops_batch_np(
        [np.zeros((8, 8, 3), np.uint8)], [obb], _GEOM
    )
    assert batch.crops == []
    assert batch.detection_ids.shape == (0,)
    assert batch.native_sizes.shape == (0, 2)
