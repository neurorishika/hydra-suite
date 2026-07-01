from hydra_suite.core.inference.config import _dict_to_config, migrate_runtime_to_tier

_MINIMAL_OBB = {
    "mode": "direct",
    "direct": {"model_path": "/tmp/obb.pt", "compute_runtime": "cpu"},
}


def test_pose_yolo_tensorrt_migrates_to_gpu_fast():
    d = {
        "obb": _MINIMAL_OBB,
        "pose": {
            "yolo": {"model_path": "/tmp/pose.pt", "compute_runtime": "tensorrt"},
        },
    }
    config = _dict_to_config(d)
    assert (
        config.runtime_tier == "gpu_fast"
    ), f"Expected 'gpu_fast', got {config.runtime_tier!r}"


def test_pose_sleap_onnx_cuda_migrates_to_gpu_fast():
    d = {
        "obb": _MINIMAL_OBB,
        "pose": {
            "sleap": {"model_path": "/tmp/sleap.pt", "compute_runtime": "onnx_cuda"},
        },
    }
    config = _dict_to_config(d)
    assert (
        config.runtime_tier == "gpu_fast"
    ), f"Expected 'gpu_fast', got {config.runtime_tier!r}"


def test_cpu_maps_to_cpu():
    assert migrate_runtime_to_tier({"cpu"}) == "cpu"


def test_cuda_and_mps_map_to_gpu():
    assert migrate_runtime_to_tier({"cuda"}) == "gpu"
    assert migrate_runtime_to_tier({"mps"}) == "gpu"


def test_onnx_and_tensorrt_map_to_gpu_fast():
    for rt in ("onnx_cpu", "onnx_cuda", "onnx_coreml", "tensorrt"):
        assert migrate_runtime_to_tier({rt}) == "gpu_fast"


def test_mixed_takes_highest_tier():
    assert migrate_runtime_to_tier({"cpu", "cuda", "tensorrt"}) == "gpu_fast"
    assert migrate_runtime_to_tier({"cpu", "mps"}) == "gpu"


def test_empty_defaults_to_gpu():
    assert migrate_runtime_to_tier(set()) == "gpu"
