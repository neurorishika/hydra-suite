from hydra_suite.runtime.compute_runtime import available_tiers, tier_label
from hydra_suite.runtime.resolver import PlatformInfo


def test_cuda_host_tiers_and_labels():
    p = PlatformInfo(has_cuda=True, has_mps=False)
    assert available_tiers(p) == ["cpu", "gpu", "gpu_fast"]
    assert tier_label("gpu", p) == "GPU (CUDA)"
    assert tier_label("gpu_fast", p) == "GPU-Fast (TensorRT)"


def test_mac_host_tiers_and_labels():
    p = PlatformInfo(has_cuda=False, has_mps=True)
    assert available_tiers(p) == ["cpu", "gpu", "gpu_fast"]
    assert tier_label("gpu", p) == "GPU (Metal)"
    assert tier_label("gpu_fast", p) == "GPU-Fast (CoreML)"


def test_cpu_only_host_only_cpu():
    p = PlatformInfo(has_cuda=False, has_mps=False)
    assert available_tiers(p) == ["cpu"]
