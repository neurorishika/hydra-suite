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
            direct=OBBDirectConfig(model_path="/tmp/obb.pt", compute_runtime="cpu"),
        )
    )


def _minimal_cuda_config() -> InferenceConfig:
    return InferenceConfig(
        obb=OBBConfig(
            mode="direct",
            direct=OBBDirectConfig(model_path="/tmp/obb.pt", compute_runtime="cuda"),
        )
    )


def test_from_json_round_trip():
    config = _minimal_cpu_config()
    with tempfile.NamedTemporaryFile(suffix=".json", mode="w", delete=False) as f:
        config.to_json(f.name)
        path = f.name
    loaded = InferenceConfig.from_json(path)
    assert loaded.obb.mode == "direct"
    assert loaded.obb.direct.model_path == "/tmp/obb.pt"
    assert loaded.obb.direct.compute_runtime == "cpu"


def test_round_trip_with_headtail():
    config = InferenceConfig(
        obb=OBBConfig(mode="direct", direct=OBBDirectConfig(model_path="/m.pt")),
        headtail=HeadTailConfig(model_path="/ht.pt", compute_runtime="cpu"),
        detection_batch_size=4,
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
            CNNConfig(label="identity", model_path="/cnn.pt", compute_runtime="cpu"),
        ],
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
                detect_compute_runtime="cuda",
                obb_compute_runtime="tensorrt",
                detect_confidence_threshold=0.1,
                obb_confidence_threshold=0.05,
            ),
        )
    )
    with tempfile.NamedTemporaryFile(suffix=".json", mode="w", delete=False) as f:
        config.to_json(f.name)
        path = f.name
    loaded = InferenceConfig.from_json(path)
    assert loaded.obb.sequential.detect_compute_runtime == "cuda"
    assert loaded.obb.sequential.obb_compute_runtime == "tensorrt"
    assert loaded.obb.sequential.detect_confidence_threshold == pytest.approx(0.1)
