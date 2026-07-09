"""Tests for posekit runtime-tier selector (Task 7 — Phase 2)."""

from hydra_suite.runtime.resolver import PlatformInfo, available_tiers, tier_label


def test_posekit_tier_labels_mac():
    """Task-brief guard: tier labels on Apple Silicon match expected strings."""
    p = PlatformInfo(has_cuda=False, has_mps=True)
    assert [tier_label(t, p) for t in available_tiers(p)] == [
        "CPU",
        "GPU (Metal)",
        "GPU-Fast (CoreML)",
    ]


def test_posekit_tier_labels_cpu_only():
    p = PlatformInfo(has_cuda=False, has_mps=False)
    assert available_tiers(p) == ["cpu"]
    assert [tier_label(t, p) for t in available_tiers(p)] == ["CPU"]


def test_posekit_tier_labels_cuda():
    p = PlatformInfo(has_cuda=True, has_mps=False)
    assert [tier_label(t, p) for t in available_tiers(p)] == [
        "CPU",
        "GPU (CUDA)",
        "GPU-Fast (TensorRT)",
    ]


def test_tier_to_canonical_runtime_mps():
    from hydra_suite.posekit.gui.runtimes import tier_to_canonical_runtime

    p = PlatformInfo(has_cuda=False, has_mps=True)
    assert tier_to_canonical_runtime("cpu", p) == "cpu"
    assert tier_to_canonical_runtime("gpu", p) == "mps"
    # Apple GPU-Fast resolves to native CoreML, not ONNX Runtime's CoreML EP.
    # ("onnx_coreml" was the pre-fix behavior and is measurably slower/
    # different from the native .mlpackage path — see Task 6.)
    assert tier_to_canonical_runtime("gpu_fast", p) == "coreml"


def test_posekit_gpu_fast_apple_resolves_to_coreml():
    """Task 6 guard: PoseKit resolves Apple GPU-Fast identically to the main tracker."""
    from hydra_suite.posekit.gui.runtimes import tier_to_canonical_runtime

    platform = PlatformInfo(has_cuda=False, has_mps=True)
    assert tier_to_canonical_runtime("gpu_fast", platform) == "coreml"


def test_tier_to_canonical_runtime_cuda():
    from hydra_suite.posekit.gui.runtimes import tier_to_canonical_runtime

    p = PlatformInfo(has_cuda=True, has_mps=False)
    assert tier_to_canonical_runtime("cpu", p) == "cpu"
    assert tier_to_canonical_runtime("gpu", p) == "cuda"
    assert tier_to_canonical_runtime("gpu_fast", p) == "tensorrt"


def test_tier_to_canonical_runtime_cpu_only():
    from hydra_suite.posekit.gui.runtimes import tier_to_canonical_runtime

    p = PlatformInfo(has_cuda=False, has_mps=False)
    assert tier_to_canonical_runtime("cpu", p) == "cpu"
    assert tier_to_canonical_runtime("gpu", p) == "cpu"
    assert tier_to_canonical_runtime("gpu_fast", p) == "cpu"


def test_canonical_runtime_to_tier():
    from hydra_suite.posekit.gui.runtimes import canonical_runtime_to_tier

    assert canonical_runtime_to_tier("cpu") == "cpu"
    assert canonical_runtime_to_tier("mps") == "gpu"
    assert canonical_runtime_to_tier("cuda") == "gpu"
    assert canonical_runtime_to_tier("tensorrt") == "gpu_fast"
    assert canonical_runtime_to_tier("coreml") == "gpu_fast"
    assert canonical_runtime_to_tier("onnx_coreml") == "gpu_fast"
    assert canonical_runtime_to_tier("onnx_cpu") == "gpu_fast"
    assert canonical_runtime_to_tier("onnx_cuda") == "gpu_fast"


def test_posekit_config_runtime_tier_default():
    from hydra_suite.posekit.config.schemas import PoseKitConfig

    cfg = PoseKitConfig()
    assert cfg.runtime_tier == "gpu"


def test_posekit_config_runtime_tier_round_trip():
    from hydra_suite.posekit.config.schemas import PoseKitConfig

    cfg = PoseKitConfig(runtime_tier="gpu_fast")
    restored = PoseKitConfig.from_dict(cfg.to_dict())
    assert restored.runtime_tier == "gpu_fast"


def test_posekit_config_runtime_tier_migration_from_legacy():
    """Legacy ``compute_runtime`` strings are migrated to the tier field."""
    from hydra_suite.posekit.config.schemas import PoseKitConfig

    # mps → gpu
    cfg = PoseKitConfig.from_dict({"compute_runtime": "mps"})
    assert cfg.runtime_tier == "gpu"

    # tensorrt → gpu_fast
    cfg = PoseKitConfig.from_dict({"pred_runtime": "tensorrt"})
    assert cfg.runtime_tier == "gpu_fast"

    # cpu → cpu
    cfg = PoseKitConfig.from_dict({"compute_runtime": "cpu"})
    assert cfg.runtime_tier == "cpu"
