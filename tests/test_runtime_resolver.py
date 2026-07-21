from hydra_suite.runtime.resolver import PlatformInfo, ResolvedBackend, RuntimeResolver

CUDA = PlatformInfo(has_cuda=True, has_mps=False)
MAC = PlatformInfo(has_cuda=False, has_mps=True)
CPU_ONLY = PlatformInfo(has_cuda=False, has_mps=False)


def test_cpu_tier_always_torch_cpu():
    r = RuntimeResolver("cpu", CUDA)
    assert r.resolve("obb") == ResolvedBackend("torch", "cpu", False)


def test_gpu_tier_cuda_is_native_torch():
    r = RuntimeResolver("gpu", CUDA)
    assert r.resolve("cnn") == ResolvedBackend("torch", "cuda", False)


def test_gpu_tier_mac_is_native_mps():
    r = RuntimeResolver("gpu", MAC)
    assert r.resolve("cnn") == ResolvedBackend("torch", "mps", False)


def test_gpu_fast_cuda_with_artifact_is_tensorrt():
    r = RuntimeResolver("gpu_fast", CUDA)
    assert r.resolve("obb", artifact_available=lambda: True) == ResolvedBackend(
        "tensorrt", "cuda", False
    )


def test_gpu_fast_cuda_without_artifact_falls_back_to_native_cuda():
    r = RuntimeResolver("gpu_fast", CUDA)
    assert r.resolve("cnn", artifact_available=lambda: False) == ResolvedBackend(
        "torch", "cuda", True
    )


def test_gpu_fast_mac_with_artifact_is_coreml():
    r = RuntimeResolver("gpu_fast", MAC)
    assert r.resolve("obb", artifact_available=lambda: True) == ResolvedBackend(
        "coreml", "mps", False
    )


def test_gpu_fast_mac_without_artifact_falls_back_to_native_mps():
    r = RuntimeResolver("gpu_fast", MAC)
    assert r.resolve("cnn", artifact_available=lambda: False) == ResolvedBackend(
        "torch", "mps", True
    )


def test_gpu_tier_on_cpu_only_host_degrades_to_cpu():
    r = RuntimeResolver("gpu", CPU_ONLY)
    assert r.resolve("obb") == ResolvedBackend("torch", "cpu", True)


def test_available_tiers_cpu_only():
    platform = PlatformInfo(has_cuda=False, has_mps=False)
    from hydra_suite.runtime.resolver import available_tiers

    assert available_tiers(platform) == ["cpu"]


def test_available_tiers_with_accelerator():
    platform = PlatformInfo(has_cuda=True, has_mps=False)
    from hydra_suite.runtime.resolver import available_tiers

    assert available_tiers(platform) == ["cpu", "gpu", "gpu_fast"]


def test_tier_label_cuda_platform():
    platform = PlatformInfo(has_cuda=True, has_mps=False)
    from hydra_suite.runtime.resolver import tier_label

    assert tier_label("cpu", platform) == "CPU"
    assert tier_label("gpu", platform) == "GPU (CUDA)"
    assert tier_label("gpu_fast", platform) == "GPU-Fast (TensorRT)"


def test_tier_label_mps_platform():
    platform = PlatformInfo(has_cuda=False, has_mps=True)
    from hydra_suite.runtime.resolver import tier_label

    assert tier_label("gpu", platform) == "GPU (Metal)"
    assert tier_label("gpu_fast", platform) == "GPU-Fast (CoreML)"
