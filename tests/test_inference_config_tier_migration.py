from hydra_suite.core.inference.config import migrate_runtime_to_tier


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
