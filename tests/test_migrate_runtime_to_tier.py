from hydra_suite.core.inference.config import migrate_runtime_to_tier


def test_tensorrt_cuda_flavor_maps_to_gpu_fast():
    assert migrate_runtime_to_tier({"tensorrt_cuda"}) == "gpu_fast"


def test_coreml_flavor_maps_to_gpu_fast():
    assert migrate_runtime_to_tier({"coreml"}) == "gpu_fast"


def test_onnx_mps_flavor_maps_to_gpu_fast():
    assert migrate_runtime_to_tier({"onnx_mps"}) == "gpu_fast"


def test_plain_gpu_and_cpu_flavors_unchanged():
    assert migrate_runtime_to_tier({"cuda"}) == "gpu"
    assert migrate_runtime_to_tier({"mps"}) == "gpu"
    assert migrate_runtime_to_tier({"cpu"}) == "cpu"
    assert migrate_runtime_to_tier(set()) == "gpu"
    # mixed set takes highest tier
    assert migrate_runtime_to_tier({"cpu", "coreml"}) == "gpu_fast"
