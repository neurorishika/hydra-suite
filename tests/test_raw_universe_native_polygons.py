"""Native polygon emission must work in the raw (device-resident) universe.

`extract_with_transform` routes to a device-resident "raw universe" when the
runtime reports `tensor_on_cuda` and the affine is translate-only. That branch
called `_extract_raw_tensors_from_masks`, which had no `emit_native_geometry`
parameter at all -- so a caller that asked for native polygons (only
polygon-level active-learning export does) silently got none, on every CUDA
host, while CPU/MPS returned them correctly.

These tests pin the contract in the raw universe and pin that both universes
agree. They construct the raw branch with `tensor_on_cuda=True` and
`device="cpu"`, the established trick in `test_region_source.py`, so they run
without a GPU.
"""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from hydra_suite.core.inference.stages import obb as m
from hydra_suite.core.inference.stages.obb import _RawOBBTensors
from hydra_suite.core.inference.stages.regions import Affine


class _FakeMasks:
    def __init__(self, data, xy):
        self.data = data
        self.xy = xy

    def __getitem__(self, idx):
        order = np.asarray(idx, dtype=np.int64)
        return _FakeMasks(self.data[order], [self.xy[int(i)] for i in order])


class _FakeBoxes:
    def __init__(self, xyxy, conf):
        self.xyxy = torch.tensor(xyxy, dtype=torch.float32)
        self.conf = torch.tensor(conf, dtype=torch.float32)


class _FakeSegResult:
    """`n` square masks in an 80x80 crop, each a distinct 10x10 block."""

    def __init__(self, confs=(0.9,)):
        n = len(confs)
        data = np.zeros((n, 80, 80), dtype=np.float32)
        xy, boxes = [], []
        for i in range(n):
            x0 = 10 + i * 15
            data[i, 30:40, x0 : x0 + 10] = 1.0
            xy.append(
                np.array([[x0, 30], [x0 + 10, 30], [x0 + 10, 40], [x0, 40]], np.float32)
            )
            boxes.append([x0, 30, x0 + 10, 40])
        self.masks = _FakeMasks(torch.tensor(data), xy)
        self.boxes = _FakeBoxes(boxes, list(confs))
        self.orig_shape = (80, 80)


class _RawRuntime:
    """Raw universe: device-resident branch, but allocating on CPU."""

    tensor_on_cuda = True
    device = "cpu"


class _Cfg:
    class direct:
        fixed_angle_deg = 0.0
        seg_num_angles = 24
        seg_crop_size = 64
        seg_pad_ratio = 0.15
        seg_mask_threshold = 0.5

    raw_detection_cap = 0
    emit_native_geometry = True


# ── the headline regression ────────────────────────────────────────────────


def test_extract_with_transform_raw_universe_emits_native_polygons():
    """The bug: raw universe silently dropped the native-geometry request."""
    out = m.extract_with_transform(
        _FakeSegResult((0.9, 0.8)), 0, "segment", Affine(), _Cfg, _RawRuntime()
    )
    assert out.polygons is not None, "raw universe dropped the polygon request"
    assert len(out.polygons) == out.xywhr.shape[0]
    for poly in out.polygons:
        assert np.asarray(poly).reshape(-1, 2).shape[0] >= 3


def test_extract_with_transform_raw_universe_no_polygons_when_not_requested():
    class _Off(_Cfg):
        emit_native_geometry = False

    out = m.extract_with_transform(
        _FakeSegResult((0.9,)), 0, "segment", Affine(), _Off, _RawRuntime()
    )
    assert out.polygons is None


def test_raw_and_numpy_universes_agree_on_polygons():
    """Both universes must return the same native contours for one result."""

    class _NumpyRuntime:
        tensor_on_cuda = False
        device = "cpu"

    raw = m.extract_with_transform(
        _FakeSegResult((0.9, 0.8)), 0, "segment", Affine(), _Cfg, _RawRuntime()
    )
    numpy_out = m.extract_with_transform(
        _FakeSegResult((0.9, 0.8)), 0, "segment", Affine(), _Cfg, _NumpyRuntime()
    )
    assert numpy_out.polygons is not None and raw.polygons is not None
    assert len(raw.polygons) == len(numpy_out.polygons)
    for a, b in zip(raw.polygons, numpy_out.polygons):
        np.testing.assert_allclose(np.asarray(a), np.asarray(b))


# ── each hop the polygons must survive ─────────────────────────────────────


def _raw_with_polygons(n=2, offset=0.0):
    polys = [
        np.array([[0, 0], [1, 0], [1, 1]], np.float32) + offset + i for i in range(n)
    ]
    return _RawOBBTensors(
        frame_idx=0,
        xywhr=torch.tensor([[10.0 + i, 20.0, 5.0, 6.0, 0.3] for i in range(n)]),
        corners=torch.zeros((n, 4, 2)),
        conf=torch.tensor([0.9 - 0.1 * i for i in range(n)]),
        cls=torch.zeros(n),
        polygons=polys,
    )


def test_translate_raw_shifts_polygons():
    out = m._translate_raw(_raw_with_polygons(2), (100.0, 200.0))
    assert out.polygons is not None
    np.testing.assert_allclose(
        np.asarray(out.polygons[0]),
        np.array([[100, 200], [101, 200], [101, 201]], np.float32),
    )


def test_translate_raw_zero_offset_keeps_polygons():
    raw = _raw_with_polygons(2)
    assert m._translate_raw(raw, (0.0, 0.0)).polygons is not None


def test_filter_valid_raw_rows_subsets_polygons():
    raw = _raw_with_polygons(3)
    bad = raw.xywhr.clone()
    bad[1, 2] = 0.0  # non-positive width -> row 1 dropped
    raw = raw._replace(xywhr=bad)
    out = m._filter_valid_raw_rows(raw)
    assert out.xywhr.shape[0] == 2
    assert out.polygons is not None and len(out.polygons) == 2
    np.testing.assert_allclose(np.asarray(out.polygons[1]), np.asarray(raw.polygons[2]))


def test_materialize_tensors_carries_polygons():
    out = m.materialize_tensors(_raw_with_polygons(2))
    assert out.polygons is not None and len(out.polygons) == out.num_detections


def test_materialize_tensors_drops_polygons_with_invalid_rows():
    raw = _raw_with_polygons(3)
    bad = raw.xywhr.clone()
    bad[0, 3] = float("nan")
    raw = raw._replace(xywhr=bad)
    out = m.materialize_tensors(raw)
    assert out.num_detections == 2
    assert out.polygons is not None and len(out.polygons) == 2
    np.testing.assert_allclose(np.asarray(out.polygons[0]), np.asarray(raw.polygons[1]))


def test_concat_raw_concatenates_polygons():
    from hydra_suite.core.inference.stages.slicing_cuda import _concat_raw

    out = _concat_raw([_raw_with_polygons(2), _raw_with_polygons(1, offset=50.0)], 0)
    assert out.polygons is not None and len(out.polygons) == 3
    np.testing.assert_allclose(
        np.asarray(out.polygons[2]),
        np.array([[50, 50], [51, 50], [51, 51]], np.float32),
    )


def test_filter_from_tensors_subsets_polygons():
    from hydra_suite.core.inference.config import OBBConfig
    from hydra_suite.core.inference.stages.filtering import filter_from_tensors

    class _Rt:
        tensor_on_cuda = True
        device = "cpu"

    cfg = OBBConfig(confidence_threshold=0.85)  # keeps row 0 only (0.9 vs 0.8)
    out = filter_from_tensors(_raw_with_polygons(2), cfg, None, _Rt())
    assert out.num_detections == 1
    assert out.polygons is not None and len(out.polygons) == 1
