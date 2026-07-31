from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from hydra_suite.core.inference.config import SliceConfig
from hydra_suite.core.inference.stages import obb as m
from hydra_suite.core.inference.stages.regions import (
    Affine,
    Grid,
    Stage1Proposals,
    WholeFrame,
    select_region_source,
)


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


def test_extract_with_transform_numpy_detect(monkeypatch):
    from hydra_suite.core.inference.stages.regions import Affine

    class _Rt:  # numpy universe
        tensor_on_cuda = False
        device = "cpu"

    class _Cfg:
        class direct:
            fixed_angle_deg = 0.0
            seg_num_angles = 24
            seg_crop_size = 64
            seg_pad_ratio = 0.15
            seg_mask_threshold = 0.5

        raw_detection_cap = 0
        emit_native_geometry = False

    res = _FakeDetResult([[10.0, 20.0, 30.0, 60.0]], [0.9])
    out = m.extract_with_transform(
        res, 0, "detect", Affine(offset=(100.0, 0.0)), _Cfg, _Rt()
    )
    assert out.centroids[0][0] == pytest.approx(120.0)  # 20 + 100


def test_translate_raw_offsets_only():
    from hydra_suite.core.inference.stages.obb import _RawOBBTensors, _translate_raw

    raw = _RawOBBTensors(
        frame_idx=0,
        xywhr=torch.tensor([[10.0, 20.0, 5.0, 6.0, 0.3]]),
        corners=torch.zeros((1, 4, 2)),
        conf=torch.tensor([0.9]),
        cls=torch.tensor([0.0]),
    )
    out = _translate_raw(raw, (100.0, 200.0))
    assert out.xywhr[0, 0].item() == pytest.approx(110.0)
    assert out.xywhr[0, 1].item() == pytest.approx(220.0)
    assert out.xywhr[0, 2].item() == pytest.approx(5.0)  # w unchanged


def test_extract_with_transform_scaled_affine_forces_numpy_under_cuda():
    """Task 7 / A1: scale != 1 forces the numpy branch even when
    tensor_on_cuda=True -- the raw universe is translate-only (invariant);
    a scaled affine (e.g. sequential stage-2 crop resize) must NOT go raw.
    Proven by NOT raising (the raw branch's defensive assert would fire on a
    non-translate-only affine) and by the offset+scale actually being applied
    via the numpy extractor.
    """

    class _Rt:
        tensor_on_cuda = True
        device = "cpu"  # avoid real CUDA allocation on non-CUDA test machines

    class _Cfg:
        class direct:
            fixed_angle_deg = 0.0
            seg_num_angles = 24
            seg_crop_size = 64
            seg_pad_ratio = 0.15
            seg_mask_threshold = 0.5

        raw_detection_cap = 0
        emit_native_geometry = False

    res = _FakeDetResult([[10.0, 20.0, 30.0, 60.0]], [0.9])  # cx=20,cy=40,w=20,h=40
    out = m.extract_with_transform(
        res, 0, "detect", Affine(offset=(0.0, 0.0), scale=(2.0, 1.0)), _Cfg, _Rt()
    )
    # A numpy OBBResult (has `.centroids`), NOT a `_RawOBBTensors` (no `.centroids`).
    assert hasattr(out, "centroids")
    assert out.centroids[0][0] == pytest.approx(40.0)  # cx=20 * scale.x=2


def test_extract_with_transform_raw_translate_only_stays_raw():
    """Companion: an actually translate-only affine under tensor_on_cuda=True
    still takes the raw branch (A1 must not regress the existing raw path)."""

    class _Rt:
        tensor_on_cuda = True
        device = "cpu"  # avoid real CUDA allocation on non-CUDA test machines

    class _Cfg:
        class direct:
            model_task = "obb"
            fixed_angle_deg = 0.0

        raw_detection_cap = 0

    class _FakeObb:
        def __len__(self):
            return 0

    class _FakeResult:
        obb = _FakeObb()

    out = m.extract_with_transform(
        _FakeResult(), 0, "obb", Affine(offset=(5.0, 5.0)), _Cfg, _Rt()
    )
    assert not hasattr(out, "centroids")  # _RawOBBTensors, not OBBResult


# --- Task 4: RegionSource planners ------------------------------------------


def test_whole_frame_plan():
    frames = [
        np.zeros((10, 10, 3), dtype=np.uint8),
        np.zeros((10, 10, 3), dtype=np.uint8),
    ]
    src = WholeFrame()
    planned = src.plan(frames, models=None, config=None, runtime=None)
    assert len(planned) == 2
    for fi, regions in enumerate(planned):
        assert len(regions) == 1
        assert regions[0].affine == Affine.IDENTITY
        assert regions[0].frame_idx == fi
        assert regions[0].image is frames[fi]
    assert src.merge_policy == "plain"
    assert src.device_residency == "on_device_capable"


def test_grid_plan_tiles_translate_only():
    frame = np.zeros((64, 64, 3), dtype=np.uint8)
    frames = [frame]
    slice_cfg = SliceConfig(
        enabled=True,
        geometry_mode="custom",
        slice_width=32,
        slice_height=32,
        overlap_width_ratio=0.0,
        overlap_height_ratio=0.0,
        perform_standard_pred=False,
    )
    config = SimpleNamespace(direct=SimpleNamespace(slice=slice_cfg))
    models = SimpleNamespace(direct_model=SimpleNamespace(imgsz=32))
    src = Grid()
    planned = src.plan(frames, models, config, runtime=None)
    assert len(planned) == 1
    regions = planned[0]
    assert len(regions) == 4
    expected_offsets = {(0.0, 0.0), (32.0, 0.0), (0.0, 32.0), (32.0, 32.0)}
    seen = set()
    for r in regions:
        assert r.affine.is_translate_only
        assert r.affine.scale == (1.0, 1.0)
        seen.add(r.affine.offset)
        assert r.image.shape == (32, 32, 3)
        assert r.frame_idx == 0
    assert seen == expected_offsets
    assert src.merge_policy == "overlap_band_nms"
    assert src.device_residency == "on_device_capable"


def test_grid_plan_full_frame_appended():
    frame = np.zeros((64, 64, 3), dtype=np.uint8)
    frames = [frame]
    slice_cfg = SliceConfig(
        enabled=True,
        geometry_mode="custom",
        slice_width=32,
        slice_height=32,
        overlap_width_ratio=0.0,
        overlap_height_ratio=0.0,
        perform_standard_pred=True,
    )
    config = SimpleNamespace(direct=SimpleNamespace(slice=slice_cfg))
    models = SimpleNamespace(direct_model=SimpleNamespace(imgsz=32))
    planned = Grid().plan(frames, models, config, runtime=None)
    regions = planned[0]
    assert len(regions) == 5  # 4 tiles + 1 full frame
    assert regions[-1].affine == Affine.IDENTITY
    assert regions[-1].image.shape == (64, 64, 3)


class _FakeBoxesXY:
    def __init__(self, xyxy):
        self.xyxy = torch.tensor(xyxy, dtype=torch.float32)

    def __len__(self):
        return len(self.xyxy)


class _FakeStage1Result:
    def __init__(self, xyxy):
        self.boxes = _FakeBoxesXY(xyxy) if xyxy else None


class _FakeDetectModel:
    def __init__(self, per_frame_boxes):
        self._per_frame_boxes = per_frame_boxes

    def predict(self, frames, **kwargs):
        return [_FakeStage1Result(b) for b in self._per_frame_boxes]


def _fake_sequential_config(stage2_image_size=16):
    return SimpleNamespace(
        detect_image_size=0,
        detect_confidence_threshold=0.1,
        crop_pad_ratio=0.0,
        min_crop_size_px=0.0,
        enforce_square_crop=False,
        stage2_image_size=stage2_image_size,
    )


def test_stage1_proposals_plan_offset_scale():
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    frames = [frame]
    boxes = [[10.0, 10.0, 30.0, 30.0]]  # square box, 20x20
    detect_model = _FakeDetectModel([boxes])
    seq = _fake_sequential_config(stage2_image_size=16)
    config = SimpleNamespace(sequential=seq, target_classes=[])
    models = SimpleNamespace(detect_model=detect_model)
    runtime = SimpleNamespace(device="cpu")

    src = Stage1Proposals()
    planned = src.plan(frames, models, config, runtime)
    assert len(planned) == 1
    regions = planned[0]
    assert len(regions) == 1
    r = regions[0]
    # crop is arr[10:30, 10:30] -> 20x20 orig, resized to 16x16 -> scale 20/16
    assert r.affine.offset == (10.0, 10.0)
    assert r.affine.scale[0] == pytest.approx(20 / 16)
    assert r.affine.scale[1] == pytest.approx(20 / 16)
    assert r.image.shape[:2] == (16, 16)
    assert r.frame_idx == 0
    assert src.merge_policy == "plain"
    assert src.device_residency == "cpu_crop_boundary"


def test_stage1_proposals_plan_empty_boxes():
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    detect_model = _FakeDetectModel([[]])
    seq = _fake_sequential_config()
    config = SimpleNamespace(sequential=seq, target_classes=[])
    models = SimpleNamespace(detect_model=detect_model)
    runtime = SimpleNamespace(device="cpu")

    planned = Stage1Proposals().plan([frame], models, config, runtime)
    assert planned == [[]]


# --- Task 5: _run_direct routes through extract_with_transform -------------


class _FakeDirectModel:
    """Stands in for an ultralytics YOLO model in direct mode."""

    def __init__(self, results):
        self._results = results
        self.calls = []

    def predict(self, frames, **kwargs):
        self.calls.append((frames, kwargs))
        return self._results


def test_run_direct_routes_through_extract_with_transform(monkeypatch):
    frames = [
        np.zeros((10, 10, 3), dtype=np.uint8),
        np.zeros((10, 10, 3), dtype=np.uint8),
    ]
    fake_results = [object(), object()]
    model = _FakeDirectModel(fake_results)

    config = SimpleNamespace(
        direct=SimpleNamespace(
            confidence_floor=1e-3,
            model_task="obb",
        ),
        target_classes=[],
        raw_detection_cap=5,
        emit_native_geometry=False,
    )
    runtime = SimpleNamespace(tensor_on_cuda=False, device="cpu")

    calls = []

    def _fake_extract_with_transform(result, frame_idx, task, affine, cfg, rt):
        calls.append((result, frame_idx, task, affine, cfg, rt))
        return f"extracted-{frame_idx}"

    cap_calls = []

    def _fake_apply_cap(extracted, cap):
        cap_calls.append((extracted, cap))
        return f"capped-{extracted}"

    monkeypatch.setattr(m, "extract_with_transform", _fake_extract_with_transform)
    monkeypatch.setattr(m, "_apply_raw_detection_cap", _fake_apply_cap)
    monkeypatch.setattr(m, "_frames_are_cuda_tensors", lambda frames: False)

    out = m._run_direct(frames, model, config, runtime)

    assert len(calls) == 2
    for idx, call in enumerate(calls):
        result, frame_idx, task, affine, cfg, rt = call
        assert result is fake_results[idx]
        assert frame_idx == idx
        assert task == "obb"
        assert affine == Affine.IDENTITY
        assert cfg is config
        assert rt is runtime

    # numpy universe (tensor_on_cuda=False): every extracted result gets the
    # outer raw-detection cap applied.
    assert cap_calls == [
        ("extracted-0", config.raw_detection_cap),
        ("extracted-1", config.raw_detection_cap),
    ]
    assert out == ["capped-extracted-0", "capped-extracted-1"]


def test_run_direct_raw_universe_skips_outer_cap(monkeypatch):
    frames = [np.zeros((10, 10, 3), dtype=np.uint8)]
    fake_results = [object()]
    model = _FakeDirectModel(fake_results)

    config = SimpleNamespace(
        direct=SimpleNamespace(confidence_floor=1e-3, model_task="obb"),
        target_classes=[],
        raw_detection_cap=5,
        emit_native_geometry=False,
    )
    runtime = SimpleNamespace(tensor_on_cuda=True, device="cuda")

    monkeypatch.setattr(m, "extract_with_transform", lambda *a, **kw: "raw-extracted")
    cap_calls = []
    monkeypatch.setattr(
        m,
        "_apply_raw_detection_cap",
        lambda extracted, cap: cap_calls.append((extracted, cap)),
    )
    monkeypatch.setattr(m, "_frames_are_cuda_tensors", lambda frames: False)

    out = m._run_direct(frames, model, config, runtime)

    assert cap_calls == []  # raw universe defers the cap to materialize_tensors
    assert out == ["raw-extracted"]


def test_select_region_source_dispatch():
    direct_cfg = SimpleNamespace(
        mode="direct", direct=SimpleNamespace(slice=SliceConfig(enabled=False))
    )
    sliced_cfg = SimpleNamespace(
        mode="direct", direct=SimpleNamespace(slice=SliceConfig(enabled=True))
    )
    seq_cfg = SimpleNamespace(mode="sequential", direct=None)

    assert isinstance(select_region_source(direct_cfg), WholeFrame)
    assert isinstance(select_region_source(sliced_cfg), Grid)
    assert isinstance(select_region_source(seq_cfg), Stage1Proposals)


# --- Task 7: _run_sequential stage-2 routes through extract_with_transform --


class _Seq2S1Boxes:
    """Stage-1 boxes stand-in: one detection, enough for build_crops to run
    (build_crops itself is monkeypatched below, so xyxy content is unused)."""

    def __len__(self):
        return 1

    xyxy = torch.tensor([[10.0, 10.0, 30.0, 30.0]])


class _Seq2S1Result:
    boxes = _Seq2S1Boxes()


class _Seq2FakeDetectModel:
    def predict(self, *a, **kw):
        return [_Seq2S1Result()]


class _Seq2FakeStage2Model:
    """Stage-2 stand-in: returns one dummy result object per crop."""

    def predict(self, batch, **kw):
        return [object() for _ in batch]


def test_run_sequential_routes_stage2_through_extract_with_transform(monkeypatch):
    """Task 7: _run_sequential's stage-2 dispatch calls extract_with_transform
    (not the old inline if/else), passing seg_source=seq and the per-crop
    offset/scale computed from the stage-1 crop geometry."""
    from hydra_suite.core.inference.config import OBBConfig, OBBSequentialConfig
    from hydra_suite.core.inference.runtime import RuntimeContext

    seq = OBBSequentialConfig(
        detect_model_path="d.pt",
        obb_model_path="s.pt",
        stage2_task="segment",
        stage2_image_size=80,
    )
    cfg = OBBConfig(mode="sequential", sequential=seq)
    runtime = RuntimeContext(cuda_mode=False, device="cpu", use_nvdec=False)

    calls = []

    def _spy_extract_with_transform(
        result, frame_idx, task, affine, cfg_arg, rt_arg, seg_source=None
    ):
        calls.append((result, frame_idx, task, affine, cfg_arg, rt_arg, seg_source))
        return m._empty_obb_result(frame_idx)

    monkeypatch.setattr(m, "extract_with_transform", _spy_extract_with_transform)
    # One 80x80 crop offset at (20, 30) -> matches _FakeSegResult-style geometry
    # used elsewhere; scale (1.0, 1.0) since stage2_image_size == crop size.
    monkeypatch.setattr(
        m,
        "build_crops",
        lambda *a, **kw: ([np.zeros((80, 80, 3), np.uint8)], [(20.0, 30.0)]),
    )

    frame = np.zeros((200, 200, 3), np.uint8)
    models = m.OBBModels(
        mode="sequential",
        detect_model=_Seq2FakeDetectModel(),
        obb_model=_Seq2FakeStage2Model(),
    )

    m._run_sequential([frame], models, cfg, runtime)

    assert len(calls) == 1
    result, frame_idx, task, affine, cfg_arg, rt_arg, seg_source = calls[0]
    assert frame_idx == 0
    assert task == "segment"
    assert affine.offset == (20.0, 30.0)
    assert affine.scale == (1.0, 1.0)  # 80/80
    assert cfg_arg is cfg
    assert rt_arg is runtime
    assert seg_source is seq  # A2: seg params sourced from the sequential config
