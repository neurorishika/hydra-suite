"""Regression guard: the exported-SLEAP GPU path must scale float [0,1] canonical
crops to uint8 [0,255] — NOT floor them to a black image.

Bug: `SleapExportedBackend.predict_batch_cuda`'s ONNX fallback did
`c.clamp(0,255).byte()` on float32 [0,1] crops, flooring every pixel to 0. The
downstream `_prepare_export_crop` also casts to uint8, so the model saw a black
image and SLEAP returned zero-confidence keypoints (valid_mask=0) in the full
pipeline. Fix: scale [0,1] → [0,255] before the uint8 cast (pass [0,255] through).
"""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from hydra_suite.core.identity.pose.backends.sleap import (
    SleapExportedBackend,
    SleapServiceBackend,
)


def _capture_cpu_crops(crops):
    be = SleapExportedBackend.__new__(SleapExportedBackend)
    be._runner = object()  # not a _DirectTensorRTEngine → ONNX fallback path
    captured = {}
    be.predict_batch = lambda c: captured.setdefault("crops", c) or []  # type: ignore
    be.predict_batch_cuda(crops)
    return captured["crops"]


def test_float_unit_crops_scaled_to_uint8_not_black():
    crops = [torch.full((3, 8, 8), 0.5, dtype=torch.float32) for _ in range(4)]
    got = _capture_cpu_crops(crops)
    assert len(got) == 4
    arr = np.asarray(got[0])
    assert arr.shape == (8, 8, 3)
    assert arr.dtype == np.uint8
    assert int(arr.max()) == 127  # 0.5 * 255 -> 127, NOT 0 (the bug)
    assert int(arr.min()) == 127


def test_already_255_range_passed_through():
    # A crop already in [0,255] must not be re-scaled to all-white.
    crops = [torch.full((3, 8, 8), 200.0, dtype=torch.float32) for _ in range(2)]
    got = _capture_cpu_crops(crops)
    arr = np.asarray(got[0])
    assert arr.dtype == np.uint8
    assert int(arr.max()) == 200  # unchanged, not clamped to 255


def test_service_to_uint8_image_scales_unit_floats():
    # The native path's shared helper must also scale [0,1] floats (guards the
    # convention the fix relies on).
    out = SleapServiceBackend._to_uint8_image(np.full((8, 8, 3), 0.5, dtype=np.float32))
    assert out.dtype == np.uint8
    assert int(out.max()) == 127
