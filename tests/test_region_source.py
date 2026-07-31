from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from hydra_suite.core.inference.config import SliceConfig
from hydra_suite.core.inference.result import OBBResult
from hydra_suite.core.inference.stages import obb as m
from hydra_suite.core.inference.stages import regions
from hydra_suite.core.inference.stages.obb import _RawOBBTensors
from hydra_suite.core.inference.stages.regions import (
    Affine,
    Grid,
    Stage1Proposals,
    WholeFrame,
    select_region_source,
)
from hydra_suite.utils.slice_geometry import SlicePlan


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


def test_extract_with_transform_force_numpy_overrides_raw_on_gpu_tier():
    """Task 7 fix: sequential's force_numpy=True must force the numpy branch
    EVEN for a pixel-exact crop (translate-only affine, scale==(1,1)) on the
    gpu-native tier -- matching A.5's always-numpy sequential behavior.
    Companion assertion: force_numpy=False (direct/grid default) with the
    same translate-only affine + tensor_on_cuda=True still takes the raw
    branch, so direct/grid behavior is unchanged.
    """

    class _Rt:
        tensor_on_cuda = True
        device = "cpu"  # avoid real CUDA allocation on non-CUDA test machines

    class _Cfg:
        class direct:
            model_task = "obb"
            fixed_angle_deg = 0.0

        raw_detection_cap = 0
        emit_native_geometry = False

    class _FakeObb:
        def __len__(self):
            return 0

    class _FakeResult:
        obb = _FakeObb()

    affine = Affine(offset=(5.0, 0.0), scale=(1.0, 1.0))
    assert affine.is_translate_only

    out_forced = m.extract_with_transform(
        _FakeResult(), 0, "obb", affine, _Cfg, _Rt(), force_numpy=True
    )
    # numpy branch returns an OBBResult (has `.centroids`), not _RawOBBTensors.
    assert hasattr(out_forced, "centroids")

    out_default = m.extract_with_transform(
        _FakeResult(), 0, "obb", affine, _Cfg, _Rt(), force_numpy=False
    )
    # raw branch preserved for direct/grid callers (default force_numpy=False).
    assert not hasattr(out_default, "centroids")


# --- Task 4: RegionSource planners ------------------------------------------


def test_whole_frame_plan():
    frames = [
        np.zeros((10, 10, 3), dtype=np.uint8),
        np.zeros((10, 10, 3), dtype=np.uint8),
    ]
    src = WholeFrame()
    planned = src.plan(frames, models=None, config=None, runtime=None)
    assert len(planned) == 2
    for fi, frame_regions in enumerate(planned):
        assert len(frame_regions) == 1
        assert frame_regions[0].affine == Affine.IDENTITY
        assert frame_regions[0].frame_idx == fi
        assert frame_regions[0].image is frames[fi]
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


# --- Task 5/9: WholeFrame.execute (verbatim retired _run_direct predict) ---


class _FakeDirectModel:
    """Stands in for an ultralytics YOLO model in direct mode."""

    def __init__(self, results):
        self._results = results
        self.calls = []

    def predict(self, frames, **kwargs):
        self.calls.append((frames, kwargs))
        return self._results


def test_whole_frame_execute_calls_predict_once_per_batch_and_wraps_per_frame():
    frames = [
        np.zeros((10, 10, 3), dtype=np.uint8),
        np.zeros((10, 10, 3), dtype=np.uint8),
    ]
    fake_results = [object(), object()]
    model = _FakeDirectModel(fake_results)

    config = SimpleNamespace(
        direct=SimpleNamespace(confidence_floor=1e-3, model_task="obb"),
        target_classes=[],
    )
    models = SimpleNamespace(direct_model=model)
    runtime = SimpleNamespace(tensor_on_cuda=False, device="cpu")

    src = WholeFrame()
    planned = src.plan(frames, models, config, runtime)
    out = src.execute(planned, models, config, runtime)

    assert len(model.calls) == 1  # single batched predict call
    called_frames, kwargs = model.calls[0]
    assert called_frames == frames
    assert kwargs["conf"] == 1e-3
    assert kwargs["iou"] == 1.0
    assert out == [[fake_results[0]], [fake_results[1]]]


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


# --- Task 7/9: Stage1Proposals.execute (verbatim retired _run_sequential
# stage-2 predict loop) --------------------------------------------------


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


def test_stage1_proposals_execute_calls_stage2_predict_per_frame_batch(monkeypatch):
    """Stage1Proposals.execute runs stage-2 predict per region, batched per
    frame by seq.stage2_batch_size, in the same order as the planned regions
    (mirrors the retired _run_sequential's stage-2 loop)."""
    monkeypatch.setattr(
        m,
        "build_crops",
        lambda *a, **kw: ([np.zeros((80, 80, 3), np.uint8)], [(20.0, 30.0)]),
    )

    seq = SimpleNamespace(
        detect_image_size=0,
        detect_confidence_threshold=0.1,
        stage2_image_size=80,
        stage2_batch_size=0,
        obb_confidence_threshold=0.2,
        stage2_task="segment",
    )
    config = SimpleNamespace(sequential=seq, target_classes=[])
    models = SimpleNamespace(
        detect_model=_Seq2FakeDetectModel(), obb_model=_Seq2FakeStage2Model()
    )
    runtime = SimpleNamespace(device="cpu")

    frame = np.zeros((200, 200, 3), np.uint8)
    src = Stage1Proposals()
    planned = src.plan([frame], models, config, runtime)
    assert len(planned[0]) == 1
    assert planned[0][0].affine.offset == (20.0, 30.0)
    assert planned[0][0].affine.scale == (1.0, 1.0)  # 80/80

    out = src.execute(planned, models, config, runtime)
    assert len(out) == 1
    assert len(out[0]) == 1  # one stage-2 result per planned crop

    assert src.task(config) == "segment"
    assert src.seg_source(config) is seq
    assert src.force_numpy is True


def test_stage1_proposals_execute_zero_regions_returns_empty_list():
    src = Stage1Proposals()
    config = SimpleNamespace(sequential=SimpleNamespace(stage2_batch_size=0))
    out = src.execute([[]], models=None, config=config, runtime=None)
    assert out == [[]]


# --- Task 8: merge_per_frame (numpy/raw x plain/overlap_band_nms) -----------


def _make_obb_result(frame_idx, centroids, confidences=None, half=5.0):
    centroids = np.asarray(centroids, dtype=np.float32).reshape(-1, 2)
    n = centroids.shape[0]
    if confidences is None:
        confidences = np.full(n, 0.9, dtype=np.float32)
    confidences = np.asarray(confidences, dtype=np.float32)
    corners = np.zeros((n, 4, 2), dtype=np.float32)
    for i, (cx, cy) in enumerate(centroids):
        corners[i] = [
            [cx - half, cy - half],
            [cx + half, cy - half],
            [cx + half, cy + half],
            [cx - half, cy + half],
        ]
    return OBBResult(
        frame_idx=frame_idx,
        centroids=centroids,
        angles=np.zeros(n, dtype=np.float32),
        sizes=np.full(n, (2 * half) ** 2, dtype=np.float32),
        shapes=np.stack([np.full(n, (2 * half) ** 2), np.ones(n)], axis=1).astype(
            np.float32
        ),
        confidences=confidences,
        corners=corners,
        detection_ids=OBBResult.make_detection_ids(frame_idx, n),
        class_ids=np.zeros(n, dtype=np.int64),
    )


def _make_raw(frame_idx, cx, cy, conf, w=10.0, h=10.0, angle=0.0):
    return _RawOBBTensors(
        frame_idx=frame_idx,
        xywhr=torch.tensor([[cx, cy, w, h, angle]], dtype=torch.float32),
        corners=torch.zeros((1, 4, 2), dtype=torch.float32),
        conf=torch.tensor([conf], dtype=torch.float32),
        cls=torch.tensor([0.0], dtype=torch.float32),
    )


def test_merge_per_frame_rejects_unknown_policy():
    part = _make_obb_result(0, [(1.0, 1.0)])
    with pytest.raises(ValueError):
        m.merge_per_frame([part], "bogus", None, SimpleNamespace(), runtime=None)


def test_merge_per_frame_rejects_empty_parts():
    with pytest.raises(ValueError):
        m.merge_per_frame([], "plain", None, SimpleNamespace(), runtime=None)


def test_merge_per_frame_plain_numpy_concat_and_cap():
    """numpy + plain == _apply_raw_detection_cap(merge_obb_results(...))."""
    part_a = _make_obb_result(0, [(10.0, 10.0)], confidences=[0.5])
    part_b = _make_obb_result(0, [(90.0, 90.0)], confidences=[0.9])
    config = SimpleNamespace(raw_detection_cap=0)
    out = m.merge_per_frame([part_a, part_b], "plain", None, config, runtime=None)
    assert out.num_detections == 2
    assert sorted(out.confidences.tolist()) == pytest.approx([0.5, 0.9])

    # cap=1 truncates to the highest-confidence detection.
    config_capped = SimpleNamespace(raw_detection_cap=1)
    out_capped = m.merge_per_frame(
        [part_a, part_b], "plain", None, config_capped, runtime=None
    )
    assert out_capped.num_detections == 1
    assert out_capped.confidences[0] == pytest.approx(0.9)


def test_merge_per_frame_plain_raw_concat_no_cap():
    """raw + plain concatenates on-device with NO cap applied.

    Matches today's precedent for a non-merged raw frame: `_run_direct`'s raw
    branch applies no cap at extraction time, and
    `assemble_raw_frames`'s non-overlapping passthrough returns the plain
    concat untouched -- the cap is applied later, exactly once, wherever the
    raw tensors are eventually materialized (`materialize_tensors` always
    caps at the end).
    """
    part_a = _make_raw(2, 10.0, 10.0, 0.5)
    part_b = _make_raw(2, 90.0, 90.0, 0.9)
    # cap=1 would truncate if (incorrectly) applied here.
    config = SimpleNamespace(raw_detection_cap=1)
    out = m.merge_per_frame([part_a, part_b], "plain", None, config, runtime=None)
    assert isinstance(out, _RawOBBTensors)
    assert out.xywhr.shape[0] == 2
    assert out.frame_idx == 2


def _overlapping_two_tile_plan():
    return SlicePlan(
        tiles=[(0, 0, 64, 64), (32, 32, 96, 96)],
        full_frame=False,
        slice_wh=(64, 64),
        frame_wh=(96, 96),
    )


def _disjoint_two_tile_plan():
    return SlicePlan(
        tiles=[(0, 0, 64, 64), (64, 0, 128, 64)],
        full_frame=False,
        slice_wh=(64, 64),
        frame_wh=(128, 64),
    )


def _grid_slice_cfg(**kw):
    kw.setdefault("merge_policy", "nmm")
    kw.setdefault("merge_metric", "ios")
    kw.setdefault("merge_threshold", 0.1)
    kw.setdefault("merge_backend", "cv2")
    return SliceConfig(enabled=True, **kw)


def test_merge_per_frame_overlap_band_nms_numpy_dedups_cross_tile_duplicate():
    """numpy + overlap_band_nms == retired `_merge_frame_obb_results`: always
    attempts band-membership + merge (no `tiles_overlap` gate on this path),
    so a genuine cross-tile duplicate touching >=2 tiles collapses to one."""
    dup = (50.0, 50.0)
    part_a = _make_obb_result(0, [dup], confidences=[0.6])
    part_b = _make_obb_result(0, [dup], confidences=[0.9])
    plan = _overlapping_two_tile_plan()
    config = SimpleNamespace(
        direct=SimpleNamespace(slice=_grid_slice_cfg()), raw_detection_cap=0
    )
    out = m.merge_per_frame(
        [part_a, part_b], "overlap_band_nms", plan, config, runtime=None
    )
    assert out.num_detections == 1


def test_merge_per_frame_overlap_band_nms_numpy_leaves_non_band_detections_alone():
    """Two detections that never touch >=2 tiles are outside the overlap band
    -- merge_obb_detections passes them straight through untouched."""
    part_a = _make_obb_result(0, [(5.0, 5.0)], confidences=[0.6])
    part_b = _make_obb_result(0, [(120.0, 5.0)], confidences=[0.9])
    plan = _disjoint_two_tile_plan()
    config = SimpleNamespace(
        direct=SimpleNamespace(slice=_grid_slice_cfg()), raw_detection_cap=0
    )
    out = m.merge_per_frame(
        [part_a, part_b], "overlap_band_nms", plan, config, runtime=None
    )
    assert out.num_detections == 2


def test_merge_per_frame_overlap_band_nms_raw_skips_merge_when_tiles_disjoint():
    """raw + overlap_band_nms: gated by `tiles_overlap(plan.tiles)` (geometry),
    never `overlap_*_ratio`. Disjoint tiles -> plain on-device concat
    passthrough, `_RawOBBTensors` preserved end-to-end, no cap applied."""
    part_a = _make_raw(0, 10.0, 10.0, 0.5)
    part_b = _make_raw(0, 90.0, 10.0, 0.9)
    plan = _disjoint_two_tile_plan()
    # cap=1 would truncate if the (incorrect) skip-branch capped.
    config = SimpleNamespace(
        direct=SimpleNamespace(slice=_grid_slice_cfg()), raw_detection_cap=1
    )
    out = m.merge_per_frame(
        [part_a, part_b], "overlap_band_nms", plan, config, runtime=None
    )
    assert isinstance(out, _RawOBBTensors)
    assert out.xywhr.shape[0] == 2


def test_merge_per_frame_overlap_band_nms_raw_materializes_and_merges_when_tiles_overlap():
    """raw + overlap_band_nms: overlapping tiles + a genuine cross-tile
    duplicate -> materialize (the only sync point) + band NMS/NMM collapses
    it to one, and the result is a materialized OBBResult (cv2 backend used
    here since SliceConfig.merge_backend defaults to "cv2"; the "gpu" backend
    is exercised on real CUDA hardware only)."""
    dup = (50.0, 50.0)
    part_a = _make_raw(0, dup[0], dup[1], 0.6)
    part_b = _make_raw(0, dup[0], dup[1], 0.9)
    plan = _overlapping_two_tile_plan()
    config = SimpleNamespace(
        direct=SimpleNamespace(slice=_grid_slice_cfg()), raw_detection_cap=0
    )
    out = m.merge_per_frame(
        [part_a, part_b], "overlap_band_nms", plan, config, runtime=None
    )
    assert isinstance(out, OBBResult)  # materialized, no longer _RawOBBTensors
    assert out.num_detections == 1


# --- Task 6/9: Grid.execute (verbatim retired run_direct_sliced tile predict) --


class _FakeTileModel:
    def __init__(self):
        self.calls = []
        self._next_id = 0

    def predict(self, images, **kwargs):
        self.calls.append((list(images), kwargs))
        out = [f"res-{self._next_id + i}" for i in range(len(images))]
        self._next_id += len(images)
        return out


def test_grid_execute_regroups_flat_predict_results_per_frame():
    """Grid.execute must (1) issue ONE chunked predict call over the flattened
    tile list (verbatim `_predict_tiles`) and (2) regroup results back into
    per-frame lists aligned with `Grid.plan`'s region order."""
    frame = np.zeros((64, 64, 3), dtype=np.uint8)
    frames = [frame, frame]
    slice_cfg = SliceConfig(
        enabled=True,
        geometry_mode="custom",
        slice_width=32,
        slice_height=32,
        overlap_width_ratio=0.0,
        overlap_height_ratio=0.0,
        perform_standard_pred=False,
    )
    config = SimpleNamespace(
        direct=SimpleNamespace(
            slice=slice_cfg, confidence_floor=1e-3, model_task="obb"
        ),
        target_classes=[],
    )
    model = _FakeTileModel()
    models = SimpleNamespace(direct_model=model)
    runtime = SimpleNamespace(device="cpu")

    src = Grid()
    planned = src.plan(frames, models, config, runtime)
    assert [len(r) for r in planned] == [4, 4]

    out = src.execute(planned, models, config, runtime)

    # chunk_size == plan.jobs_per_frame (4): 8 tiles total -> 2 chunks of 4,
    # matching the retired `run_direct_sliced`'s `chunk_size = plan.jobs_per_frame`.
    assert len(model.calls) == 2
    for images, kwargs in model.calls:
        assert len(images) == 4
        assert kwargs["conf"] == 1e-3
        assert kwargs["iou"] == 1.0

    assert [len(r) for r in out] == [4, 4]
    assert out[0] == ["res-0", "res-1", "res-2", "res-3"]
    assert out[1] == ["res-4", "res-5", "res-6", "res-7"]

    # merge_plan is the memoized SlicePlan from `plan()` -- same object, used
    # by `run_obb`'s shared merge step for overlap-band NMS tile geometry.
    assert src.merge_plan(0) is src._plan


# --- Task 9: run_obb == select_region_source -> plan -> execute -> extract ->
# merge, for every mode ------------------------------------------------------


def _spy_extract_and_merge(monkeypatch, calls):
    def _extract(
        result, frame_idx, task, affine, cfg, rt, seg_source=None, force_numpy=False
    ):
        calls["extract"].append(
            (result, frame_idx, task, affine, cfg, rt, seg_source, force_numpy)
        )
        return f"extracted-{frame_idx}-{result}"

    def _merge(parts, policy, plan, cfg, rt):
        calls["merge"].append((tuple(parts), policy, plan, cfg, rt))
        return f"merged-{parts}"

    monkeypatch.setattr(m, "extract_with_transform", _extract)
    monkeypatch.setattr(m, "merge_per_frame", _merge)


def test_run_obb_whole_frame_mode_routes_plan_execute_extract_merge(monkeypatch):
    frames = [np.zeros((10, 10, 3), dtype=np.uint8)]
    config = SimpleNamespace(
        mode="direct",
        direct=SimpleNamespace(slice=SliceConfig(enabled=False), model_task="obb"),
    )
    models = SimpleNamespace(mode="direct")
    runtime = SimpleNamespace()

    calls = {"plan": [], "execute": [], "extract": [], "merge": []}

    def _plan(self, fr, mo, cfg, rt, roi_mask=None):
        calls["plan"].append((fr, mo, cfg, rt, roi_mask))
        return [[regions.Region(image="img0", affine=Affine.IDENTITY, frame_idx=0)]]

    def _execute(self, per_frame_regions, mo, cfg, rt):
        calls["execute"].append((per_frame_regions, mo, cfg, rt))
        return [["res0"]]

    monkeypatch.setattr(regions.WholeFrame, "plan", _plan)
    monkeypatch.setattr(regions.WholeFrame, "execute", _execute)
    _spy_extract_and_merge(monkeypatch, calls)

    out = m.run_obb(frames, models, config, runtime)

    assert len(calls["plan"]) == 1
    assert len(calls["execute"]) == 1
    assert calls["extract"] == [
        ("res0", 0, "obb", Affine.IDENTITY, config, runtime, None, False)
    ]
    assert calls["merge"] == [(("extracted-0-res0",), "plain", None, config, runtime)]
    assert out == ["merged-['extracted-0-res0']"]


def test_run_obb_grid_mode_routes_plan_execute_extract_merge(monkeypatch):
    frames = [np.zeros((64, 64, 3), dtype=np.uint8)]
    config = SimpleNamespace(
        mode="direct",
        direct=SimpleNamespace(slice=SliceConfig(enabled=True), model_task="obb"),
    )
    models = SimpleNamespace(mode="direct")
    runtime = SimpleNamespace()
    roi_mask = np.ones((64, 64), dtype=bool)

    calls = {"plan": [], "execute": [], "extract": [], "merge": []}
    fake_plan = object()

    def _plan(self, fr, mo, cfg, rt, roi_mask=None):
        calls["plan"].append((fr, mo, cfg, rt, roi_mask))
        self._plan = fake_plan
        return [
            [
                regions.Region(
                    image="tile0", affine=Affine(offset=(0.0, 0.0)), frame_idx=0
                ),
                regions.Region(
                    image="tile1", affine=Affine(offset=(32.0, 0.0)), frame_idx=0
                ),
            ]
        ]

    def _execute(self, per_frame_regions, mo, cfg, rt):
        calls["execute"].append((per_frame_regions, mo, cfg, rt))
        return [["res-t0", "res-t1"]]

    monkeypatch.setattr(regions.Grid, "plan", _plan)
    monkeypatch.setattr(regions.Grid, "execute", _execute)
    _spy_extract_and_merge(monkeypatch, calls)

    out = m.run_obb(frames, models, config, runtime, roi_mask=roi_mask)

    # roi_mask is threaded straight through to Grid.plan.
    assert calls["plan"][0][-1] is roi_mask
    assert len(calls["extract"]) == 2
    for call, expected_affine in zip(
        calls["extract"], [Affine(offset=(0.0, 0.0)), Affine(offset=(32.0, 0.0))]
    ):
        _, frame_idx, task, affine, cfg, rt, seg_source, force_numpy = call
        assert frame_idx == 0
        assert task == "obb"
        assert affine == expected_affine
        assert seg_source is None
        assert force_numpy is False
    # Grid's merge_policy + merge_plan (the memoized SlicePlan) flow through.
    assert calls["merge"][0][1] == "overlap_band_nms"
    assert calls["merge"][0][2] is fake_plan
    assert out == ["merged-['extracted-0-res-t0', 'extracted-0-res-t1']"]


def test_run_obb_sequential_mode_routes_plan_execute_extract_merge(monkeypatch):
    frames = [np.zeros((100, 100, 3), dtype=np.uint8)]
    seq = SimpleNamespace(stage2_task="segment")
    config = SimpleNamespace(mode="sequential", direct=None, sequential=seq)
    models = SimpleNamespace(mode="sequential")
    runtime = SimpleNamespace()

    calls = {"plan": [], "execute": [], "extract": [], "merge": []}

    def _plan(self, fr, mo, cfg, rt, roi_mask=None):
        calls["plan"].append((fr, mo, cfg, rt, roi_mask))
        return [
            [
                regions.Region(
                    image="crop0",
                    affine=Affine(offset=(5.0, 6.0), scale=(2.0, 2.0)),
                    frame_idx=0,
                )
            ]
        ]

    def _execute(self, per_frame_regions, mo, cfg, rt):
        calls["execute"].append((per_frame_regions, mo, cfg, rt))
        return [["res-crop0"]]

    monkeypatch.setattr(regions.Stage1Proposals, "plan", _plan)
    monkeypatch.setattr(regions.Stage1Proposals, "execute", _execute)
    _spy_extract_and_merge(monkeypatch, calls)

    out = m.run_obb(frames, models, config, runtime)

    assert len(calls["extract"]) == 1
    _, frame_idx, task, affine, cfg, rt, seg_source, force_numpy = calls["extract"][0]
    assert frame_idx == 0
    assert task == "segment"  # from config.sequential.stage2_task
    assert affine == Affine(offset=(5.0, 6.0), scale=(2.0, 2.0))
    assert seg_source is seq  # Stage1Proposals.seg_source == config.sequential
    assert force_numpy is True  # cpu_crop_boundary invariant (spec S5.2)
    assert calls["merge"][0][1] == "plain"
    assert calls["merge"][0][2] is None
    assert out == ["merged-['extracted-0-res-crop0']"]


def test_run_obb_short_circuits_zero_region_frames_without_extract_or_merge(
    monkeypatch,
):
    """A frame with zero planned regions (e.g. sequential stage-1 found
    nothing) must bypass extract_with_transform/merge_per_frame entirely and
    get `_empty_obb_result` directly -- mirrors the retired `_run_sequential`'s
    early-continue for empty stage-1/crops."""
    frames = [np.zeros((10, 10, 3), dtype=np.uint8)]
    seq = SimpleNamespace(stage2_task="obb")
    config = SimpleNamespace(mode="sequential", direct=None, sequential=seq)
    models = SimpleNamespace(mode="sequential")
    runtime = SimpleNamespace()

    calls = {"extract": [], "merge": []}
    _spy_extract_and_merge(monkeypatch, calls)

    monkeypatch.setattr(regions.Stage1Proposals, "plan", lambda self, *a, **kw: [[]])
    monkeypatch.setattr(regions.Stage1Proposals, "execute", lambda self, *a, **kw: [[]])

    out = m.run_obb(frames, models, config, runtime)

    assert calls["extract"] == []
    assert calls["merge"] == []
    assert len(out) == 1
    assert out[0].frame_idx == 0
    assert out[0].num_detections == 0
