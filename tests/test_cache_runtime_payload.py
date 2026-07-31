"""Runtime Gen-2 (FT2): the individual-properties extractor hash is a PERSISTED
cache key derived from ``RUNTIME_TIER`` (not the retired ``COMPUTE_RUNTIME`` /
``POSE_SLEAP_DEVICE`` params). These tests pin the tier->string derivation and,
crucially, prove the key is STABLE for a given (tier, platform) so the Gen-2
branch lands with no net property-cache invalidation.
"""

from __future__ import annotations

import pytest

from hydra_suite.runtime import resolver as resolver_mod
from tests.helpers.module_loader import load_src_module

mod = load_src_module(
    "hydra_suite/core/identity/properties/cache.py",
    "cache_runtime_payload_under_test",
)


def _patch_platform(monkeypatch, *, has_cuda: bool, has_mps: bool) -> None:
    monkeypatch.setattr(
        resolver_mod,
        "detect_platform",
        lambda: resolver_mod.PlatformInfo(has_cuda=has_cuda, has_mps=has_mps),
    )


@pytest.mark.parametrize(
    "tier, has_cuda, has_mps, expected_runtime, expected_sleap_device",
    [
        ("cpu", False, False, "cpu", "cpu"),
        ("cpu", True, False, "cpu", "cpu"),  # cpu tier ignores the accelerator
        ("gpu", True, False, "cuda", "cuda:0"),
        ("gpu", False, True, "mps", "mps"),
        ("gpu", False, False, "cpu", "cpu"),  # gpu degrades to cpu
        ("gpu_fast", True, False, "tensorrt", "cuda:0"),
        ("gpu_fast", False, True, "coreml", "mps"),
        ("gpu_fast", False, False, "cpu", "cpu"),
    ],
)
def test_tier_derives_runtime_and_sleap_device(
    monkeypatch, tier, has_cuda, has_mps, expected_runtime, expected_sleap_device
) -> None:
    _patch_platform(monkeypatch, has_cuda=has_cuda, has_mps=has_mps)
    resolved = mod._resolve_tier_backend({"RUNTIME_TIER": tier}, stage="obb")
    assert mod._runtime_string_from_resolved(resolved) == expected_runtime
    assert mod._sleap_device_from_resolved(resolved) == expected_sleap_device


def test_extractor_hash_is_stable_for_a_given_tier(monkeypatch) -> None:
    """Same tier + platform must yield a byte-identical extractor hash."""
    _patch_platform(monkeypatch, has_cuda=True, has_mps=False)
    params = {
        "RUNTIME_TIER": "gpu_fast",
        "ENABLE_POSE_EXTRACTOR": True,
        "POSE_MODEL_TYPE": "sleap",
        "POSE_SLEAP_ENV": "sleap",
    }
    h1 = mod.compute_extractor_hash(dict(params))
    h2 = mod.compute_extractor_hash(dict(params))
    assert h1 == h2


def test_extractor_hash_ignores_retired_compute_runtime_param(monkeypatch) -> None:
    """The retired COMPUTE_RUNTIME / POSE_SLEAP_DEVICE params no longer feed the key."""
    _patch_platform(monkeypatch, has_cuda=False, has_mps=False)
    base = {"RUNTIME_TIER": "cpu", "POSE_MODEL_TYPE": "sleap"}
    ref = mod.compute_extractor_hash(dict(base))
    polluted = dict(base)
    polluted["COMPUTE_RUNTIME"] = "onnx_cuda"
    polluted["compute_runtime"] = "tensorrt"
    polluted["POSE_SLEAP_DEVICE"] = "cuda:7"
    assert mod.compute_extractor_hash(polluted) == ref


def test_extractor_hash_tracks_tier_when_platform_distinguishes(monkeypatch) -> None:
    """On an accelerator host, cpu vs gpu_fast tiers must key differently."""
    _patch_platform(monkeypatch, has_cuda=True, has_mps=False)
    cpu_hash = mod.compute_extractor_hash({"RUNTIME_TIER": "cpu"})
    fast_hash = mod.compute_extractor_hash({"RUNTIME_TIER": "gpu_fast"})
    assert cpu_hash != fast_hash
