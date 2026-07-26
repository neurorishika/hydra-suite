import numpy as np
import pytest

from hydra_suite.core.inference.result import OBBResult


def _empty_result():
    return OBBResult(
        frame_idx=0,
        centroids=np.zeros((0, 2), np.float32),
        angles=np.zeros((0,), np.float32),
        sizes=np.zeros((0,), np.float32),
        shapes=np.zeros((0, 2), np.float32),
        confidences=np.zeros((0,), np.float32),
        corners=np.zeros((0, 4, 2), np.float32),
        detection_ids=np.zeros((0,), np.int64),
    )


def test_polygons_defaults_none():
    assert _empty_result().polygons is None


def test_polygons_settable():
    r = _empty_result()
    r.polygons = [np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]], np.float32)]
    assert len(r.polygons) == 1


# ── Task 13: extractors emit native contours into OBBResult.polygons ────────


class _FakeBoxes:
    def __init__(self, xyxy, conf):
        import torch

        self.xyxy = torch.tensor(xyxy, dtype=torch.float32)
        self.conf = torch.tensor(conf, dtype=torch.float32)


class _FakeDetectResult:
    def __init__(self, xyxy, conf):
        self.boxes = _FakeBoxes(xyxy, conf)


def test_detect_extractor_emits_quad_polygons():
    pytest.importorskip("torch")
    from hydra_suite.core.inference.stages.obb import _extract_obb_from_boxes

    res = _FakeDetectResult([[10.0, 20.0, 30.0, 60.0]], [0.9])
    out = _extract_obb_from_boxes(
        res, frame_idx=0, fixed_angle_rad=0.0, emit_native_geometry=True
    )
    assert out.polygons is not None
    assert out.polygons[0].shape == (4, 2)


def test_detect_extractor_default_no_polygons():
    pytest.importorskip("torch")
    from hydra_suite.core.inference.stages.obb import _extract_obb_from_boxes

    res = _FakeDetectResult([[10.0, 20.0, 30.0, 60.0]], [0.9])
    out = _extract_obb_from_boxes(res, frame_idx=0, fixed_angle_rad=0.0)
    assert out.polygons is None
