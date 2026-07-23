import tempfile

import pytest

from hydra_suite.core.inference.config import (
    CNNConfig,
    HeadTailConfig,
    InferenceConfig,
    OBBConfig,
    OBBDirectConfig,
    OBBSequentialConfig,
)


def _minimal_cpu_config() -> InferenceConfig:
    return InferenceConfig(
        obb=OBBConfig(
            mode="direct",
            direct=OBBDirectConfig(model_path="/tmp/obb.pt"),
        ),
        runtime_tier="cpu",
    )


def _minimal_cuda_config() -> InferenceConfig:
    return InferenceConfig(
        obb=OBBConfig(
            mode="direct",
            direct=OBBDirectConfig(model_path="/tmp/obb.pt"),
        ),
        runtime_tier="gpu",
    )


def test_from_json_round_trip():
    config = _minimal_cpu_config()
    with tempfile.NamedTemporaryFile(suffix=".json", mode="w", delete=False) as f:
        config.to_json(f.name)
        path = f.name
    loaded = InferenceConfig.from_json(path)
    assert loaded.obb.mode == "direct"
    assert loaded.obb.direct.model_path == "/tmp/obb.pt"
    assert loaded.runtime_tier == "cpu"


def test_round_trip_with_headtail():
    config = InferenceConfig(
        obb=OBBConfig(mode="direct", direct=OBBDirectConfig(model_path="/m.pt")),
        headtail=HeadTailConfig(model_path="/ht.pt"),
        detection_batch_size=4,
        runtime_tier="cpu",
    )
    with tempfile.NamedTemporaryFile(suffix=".json", mode="w", delete=False) as f:
        config.to_json(f.name)
        path = f.name
    loaded = InferenceConfig.from_json(path)
    assert loaded.headtail.model_path == "/ht.pt"
    assert loaded.detection_batch_size == 4


def test_round_trip_with_cnn_phases():
    config = InferenceConfig(
        obb=OBBConfig(mode="direct", direct=OBBDirectConfig(model_path="/m.pt")),
        cnn_phases=[
            CNNConfig(label="identity", model_path="/cnn.pt"),
        ],
        runtime_tier="cpu",
    )
    with tempfile.NamedTemporaryFile(suffix=".json", mode="w", delete=False) as f:
        config.to_json(f.name)
        path = f.name
    loaded = InferenceConfig.from_json(path)
    assert len(loaded.cnn_phases) == 1
    assert loaded.cnn_phases[0].label == "identity"


def test_sequential_config_round_trip():
    config = InferenceConfig(
        obb=OBBConfig(
            mode="sequential",
            sequential=OBBSequentialConfig(
                detect_model_path="/detect.pt",
                obb_model_path="/obb.pt",
                detect_confidence_threshold=0.1,
                obb_confidence_threshold=0.05,
            ),
        ),
        runtime_tier="gpu_fast",
    )
    with tempfile.NamedTemporaryFile(suffix=".json", mode="w", delete=False) as f:
        config.to_json(f.name)
        path = f.name
    loaded = InferenceConfig.from_json(path)
    assert loaded.obb.sequential.detect_model_path == "/detect.pt"
    assert loaded.obb.sequential.obb_model_path == "/obb.pt"
    assert loaded.obb.sequential.detect_confidence_threshold == pytest.approx(0.1)
    assert loaded.runtime_tier == "gpu_fast"


def test_obb_direct_config_model_task_round_trips(tmp_path):
    config = InferenceConfig(
        obb=OBBConfig(
            mode="direct",
            direct=OBBDirectConfig(
                model_path="yolo26s-seg.pt",
                model_task="segment",
                fixed_angle_deg=0.0,
            ),
        )
    )
    path = tmp_path / "cfg.json"
    config.to_json(str(path))
    loaded = InferenceConfig.from_json(str(path))

    assert loaded.obb.direct.model_task == "segment"
    assert loaded.obb.direct.fixed_angle_deg == 0.0


def test_obb_direct_config_model_task_defaults_to_obb():
    direct = OBBDirectConfig(model_path="yolo26s-obb.pt")
    assert direct.model_task == "obb"
    assert direct.fixed_angle_deg == 0.0


def test_obb_direct_config_seg_kernel_params_round_trip(tmp_path):
    config = InferenceConfig(
        obb=OBBConfig(
            mode="direct",
            direct=OBBDirectConfig(
                model_path="yolo26s-seg.pt",
                model_task="segment",
                seg_num_angles=48,
                seg_crop_size=128,
                seg_pad_ratio=0.25,
                seg_mask_threshold=0.35,
            ),
        )
    )
    path = tmp_path / "cfg.json"
    config.to_json(str(path))
    loaded = InferenceConfig.from_json(str(path))

    assert loaded.obb.direct.seg_num_angles == 48
    assert loaded.obb.direct.seg_crop_size == 128
    assert loaded.obb.direct.seg_pad_ratio == pytest.approx(0.25)
    assert loaded.obb.direct.seg_mask_threshold == pytest.approx(0.35)


def test_obb_direct_config_seg_kernel_params_default_to_kernel_defaults():
    direct = OBBDirectConfig(model_path="yolo26s-seg.pt")
    assert direct.seg_num_angles == 24
    assert direct.seg_crop_size == 64
    assert direct.seg_pad_ratio == pytest.approx(0.15)
    assert direct.seg_mask_threshold == pytest.approx(0.5)
