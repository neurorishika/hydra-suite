import numpy as np
import pytest

torch = pytest.importorskip("torch")

from hydra_suite.core.inference.stages import obb as obb_stage


class _FakeMasks:
    def __init__(self, data, xy):
        self.data = data
        self.xy = xy


class _FakeBoxes:
    def __init__(self, xyxy, conf):
        self.xyxy = torch.tensor(xyxy, dtype=torch.float32)
        self.conf = torch.tensor(conf, dtype=torch.float32)


class _FakeSegResult:
    """A minimal segment result: one square mask centered in an 80x80 crop."""

    def __init__(self):
        m = np.zeros((1, 80, 80), dtype=np.float32)
        m[0, 30:50, 30:50] = 1.0  # 20x20 block centered at (40,40)
        self.masks = _FakeMasks(
            torch.tensor(m),
            [np.array([[30, 30], [50, 30], [50, 50], [30, 50]], np.float32)],
        )
        self.boxes = _FakeBoxes([[30, 30, 50, 50]], [0.9])
        self.orig_shape = (80, 80)


def _centroid(res, **kw):
    out = obb_stage._extract_obb_from_masks(res, frame_idx=0, **kw)
    assert out.num_detections == 1
    return out.centroids[0]


def test_mask_extract_identity_default():
    cx, cy = _centroid(_FakeSegResult())
    assert cx == pytest.approx(40.0, abs=1.5) and cy == pytest.approx(40.0, abs=1.5)


def test_mask_extract_offset_scale_maps_to_frame():
    # scale x2 then offset (+100,+200): crop centroid (40,40) -> (40*2+100, 40*2+200)
    cx, cy = _centroid(_FakeSegResult(), offset=(100.0, 200.0), scale=(2.0, 2.0))
    assert cx == pytest.approx(180.0, abs=3.0) and cy == pytest.approx(280.0, abs=3.0)


def test_mask_extract_offset_scale_contours_in_frame_space():
    out = obb_stage._extract_obb_from_masks(
        _FakeSegResult(),
        frame_idx=0,
        offset=(100.0, 200.0),
        scale=(2.0, 2.0),
        emit_native_geometry=True,
    )
    poly = out.polygons[0]
    # contour x in [30,50]*2+100 = [160,200]; y in [30,50]*2+200 = [260,300]
    assert poly[:, 0].min() == pytest.approx(160.0, abs=2.0)
    assert poly[:, 1].max() == pytest.approx(300.0, abs=2.0)


from hydra_suite.core.inference.config import (  # noqa: E402
    OBBSequentialConfig,
    build_inference_config_from_params,
)


def test_sequential_config_stage2_task_defaults_obb():
    c = OBBSequentialConfig(detect_model_path="d.pt", obb_model_path="s.pt")
    assert c.stage2_task == "obb"
    assert (
        c.seg_num_angles,
        c.seg_crop_size,
        c.seg_pad_ratio,
        c.seg_mask_threshold,
    ) == (
        24,
        64,
        0.15,
        0.5,
    )


def test_sequential_config_from_params_threads_stage2_task():
    params = {
        "DETECTION_METHOD": "yolo_obb",
        "YOLO_OBB_MODE": "sequential",
        "YOLO_DETECT_MODEL_PATH": "d.pt",
        "YOLO_CROP_OBB_MODEL_PATH": "s.pt",
        "YOLO_SEQ_STAGE2_TASK": "segment",
        "RUNTIME_TIER": "cpu",
    }
    cfg = build_inference_config_from_params(params)
    assert cfg.obb.sequential.stage2_task == "segment"


def test_sequential_config_from_params_coerces_bad_stage2_task():
    params = {
        "DETECTION_METHOD": "yolo_obb",
        "YOLO_OBB_MODE": "sequential",
        "YOLO_DETECT_MODEL_PATH": "d.pt",
        "YOLO_CROP_OBB_MODEL_PATH": "s.pt",
        "YOLO_SEQ_STAGE2_TASK": "banana",
        "RUNTIME_TIER": "cpu",
    }
    cfg = build_inference_config_from_params(params)
    assert cfg.obb.sequential.stage2_task == "obb"


from hydra_suite.core.inference.config import OBBConfig  # noqa: E402
from hydra_suite.core.inference.runtime import RuntimeContext  # noqa: E402


class _FakeStage2SegModel:
    """Stage-2 stand-in: returns one _FakeSegResult per crop, ignoring inputs."""

    def __init__(self):
        self.calls = 0

    def predict(self, batch, **kw):
        self.calls += 1
        return [_FakeSegResult() for _ in batch]


class _S1Boxes:
    """Stage-1 boxes stand-in: one detection, enough for build_crops to be
    called (build_crops itself is monkeypatched, so xyxy content is unused)."""

    def __len__(self):
        return 1

    xyxy = torch.tensor([[10.0, 10.0, 30.0, 30.0]])


class _S1Result:
    boxes = _S1Boxes()


class _FakeDetectModel:
    def predict(self, *a, **kw):
        return [_S1Result()]


def test_run_sequential_segment_dispatch(monkeypatch):
    """With stage2_task="segment", stage-2 extraction routes through
    _extract_obb_from_masks (not extract_obb_result), using the same
    per-crop offset/scale threaded through the direct path."""
    seq = OBBSequentialConfig(
        detect_model_path="d.pt",
        obb_model_path="s.pt",
        stage2_task="segment",
        stage2_image_size=80,  # matches the 80x80 fake crop -> scale (1.0, 1.0)
    )
    cfg = OBBConfig(mode="sequential", sequential=seq)
    runtime = RuntimeContext(cuda_mode=False, device="cpu", use_nvdec=False)

    called = {"masks": 0, "obb": 0}
    real_masks = obb_stage._extract_obb_from_masks
    real_obb = obb_stage.extract_obb_result

    def mask_spy(*a, **kw):
        called["masks"] += 1
        return real_masks(*a, **kw)

    def obb_spy(*a, **kw):
        called["obb"] += 1
        return real_obb(*a, **kw)

    monkeypatch.setattr(obb_stage, "_extract_obb_from_masks", mask_spy)
    monkeypatch.setattr(obb_stage, "extract_obb_result", obb_spy)
    # _FakeSegResult's orig_shape is (80, 80); build one matching 80x80 crop.
    monkeypatch.setattr(
        obb_stage,
        "iter_crops",
        lambda *a, **kw: iter([(np.zeros((80, 80, 3), np.uint8), (20.0, 30.0))]),
    )

    frame = np.zeros((200, 200, 3), np.uint8)
    models = obb_stage.OBBModels(
        mode="sequential",
        detect_model=_FakeDetectModel(),
        obb_model=_FakeStage2SegModel(),
    )

    out = obb_stage.run_obb([frame], models, cfg, runtime)

    assert called["masks"] == 1
    assert called["obb"] == 0
    assert out[0].num_detections >= 1
    # Offset (20, 30) threaded through: crop centroid (40,40) at scale 1.0
    # (stage2_image_size == 80 by default -> orig_w/orig_h == stage2 size).
    cx, cy = out[0].centroids[0]
    assert cx == pytest.approx(60.0, abs=3.0) and cy == pytest.approx(70.0, abs=3.0)


def test_assert_task_matches_checkpoint_raises_on_mismatch():
    from hydra_suite.core.inference.stages import obb as m

    class _Ckpt:
        task = "segment"

    with pytest.raises(ValueError, match="task"):
        m._assert_task_matches_checkpoint(_Ckpt(), "obb", "s.pt")


def test_assert_task_matches_checkpoint_ok_on_match():
    from hydra_suite.core.inference.stages import obb as m

    class _Ckpt:
        task = "segment"

    m._assert_task_matches_checkpoint(_Ckpt(), "segment", "s.pt")  # no raise


def test_project_roundtrips_seq_crop_segment(tmp_path):
    from hydra_suite.detectkit.gui.models import DetectKitProject

    p = DetectKitProject()
    p.role_seq_crop_segment = True
    p.imgsz_seq_crop_segment = 192
    p.model_seq_crop_segment = "custom-seg.pt"
    dest = tmp_path / "p.json"
    p.save(dest)
    q = DetectKitProject.load(dest)
    assert q.role_seq_crop_segment is True
    assert q.imgsz_seq_crop_segment == 192
    assert q.model_seq_crop_segment == "custom-seg.pt"
