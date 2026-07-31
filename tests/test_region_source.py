import numpy as np
import pytest

torch = pytest.importorskip("torch")

from hydra_suite.core.inference.stages import obb as m
from hydra_suite.core.inference.stages.regions import Affine


def test_affine_identity_and_translate_only():
    assert Affine().offset == (0.0, 0.0) and Affine().scale == (1.0, 1.0)
    assert Affine(offset=(5.0, 6.0)).is_translate_only
    assert not Affine(scale=(2.0, 1.0)).is_translate_only


class _FakeBoxes:
    def __init__(self, xyxy, conf):
        self.xyxy = torch.tensor(xyxy, dtype=torch.float32)
        self.conf = torch.tensor(conf, dtype=torch.float32)


class _FakeDetResult:
    def __init__(self, xyxy, conf):
        self.boxes = _FakeBoxes(xyxy, conf)


def test_extract_boxes_offset_scale_maps_to_frame():
    res = _FakeDetResult([[10.0, 20.0, 30.0, 60.0]], [0.9])  # cx=20,cy=40,w=20,h=40
    out = m._extract_obb_from_boxes(
        res, 0, 0.0, offset=(100.0, 200.0), scale=(2.0, 2.0)
    )
    assert out.centroids[0][0] == pytest.approx(140.0)  # 20*2+100
    assert out.centroids[0][1] == pytest.approx(280.0)  # 40*2+200


def test_extract_boxes_default_identity_byte_identical():
    res = _FakeDetResult([[10.0, 20.0, 30.0, 60.0]], [0.9])
    a = m._extract_obb_from_boxes(res, 0, 0.0)
    b = m._extract_obb_from_boxes(res, 0, 0.0, offset=(0.0, 0.0), scale=(1.0, 1.0))
    assert np.array_equal(a.centroids, b.centroids) and np.array_equal(
        a.corners, b.corners
    )
