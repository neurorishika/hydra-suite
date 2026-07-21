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


def test_posekit_config_runtime_tier_default():
    from hydra_suite.posekit.config.schemas import PoseKitConfig

    cfg = PoseKitConfig()
    assert cfg.runtime_tier == "gpu"


def test_posekit_config_runtime_tier_round_trip():
    from hydra_suite.posekit.config.schemas import PoseKitConfig

    cfg = PoseKitConfig(runtime_tier="gpu_fast")
    restored = PoseKitConfig.from_dict(cfg.to_dict())
    assert restored.runtime_tier == "gpu_fast"


def test_posekit_config_ignores_legacy_runtime_strings_clean_break():
    """Clean break (Runtime Gen-2, FT7b): ``PoseKitConfig.from_dict`` no longer
    migrates legacy ``compute_runtime`` / ``pred_runtime`` strings in-schema —
    those old files are migrated by scripts/migrate_runtime_config.py. A dict
    that carries only legacy string keys (no ``runtime_tier``) defaults to
    ``"gpu"``."""
    from hydra_suite.posekit.config.schemas import PoseKitConfig

    assert PoseKitConfig.from_dict({"compute_runtime": "mps"}).runtime_tier == "gpu"
    assert PoseKitConfig.from_dict({"pred_runtime": "tensorrt"}).runtime_tier == "gpu"
    assert PoseKitConfig.from_dict({"compute_runtime": "cpu"}).runtime_tier == "gpu"
    # An explicit runtime_tier is always honored.
    assert PoseKitConfig.from_dict({"runtime_tier": "cpu"}).runtime_tier == "cpu"
