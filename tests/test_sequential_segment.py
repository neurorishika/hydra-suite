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
