"""Regression guard: the tracking worker threads `runtime_tier` into the
`InferenceConfig` it builds.

Runtime Gen-2 (FT5): `RUNTIME_TIER` is the sole runtime knob in
`build_inference_config_from_params`. The legacy per-stage `COMPUTE_RUNTIME`
param no longer influences the tier; an absent or invalid `RUNTIME_TIER`
defaults to "cpu" (no silent migration).

The builder is a pure function of `params` (module-level in
`hydra_suite.core.inference.config`), so it is called directly without
constructing a QThread / QApplication.
"""

import pytest

from hydra_suite.core.inference.config import build_inference_config_from_params

_BUILD = build_inference_config_from_params


def _tier(params: dict) -> str:
    params.setdefault("YOLO_OBB_DIRECT_MODEL_PATH", "yolo26s-obb.pt")
    return _BUILD(params).runtime_tier


@pytest.mark.parametrize("tier", ["cpu", "gpu", "gpu_fast"])
def test_explicit_runtime_tier_is_honored(tier):
    assert _tier({"RUNTIME_TIER": tier, "COMPUTE_RUNTIME": "cpu"}) == tier


@pytest.mark.parametrize("legacy", ["cpu", "mps", "cuda", "tensorrt", "onnx_cuda"])
def test_legacy_compute_runtime_no_longer_influences_tier(legacy):
    # Runtime Gen-2: COMPUTE_RUNTIME is inert; without an explicit RUNTIME_TIER
    # the builder defaults the tier to "cpu" (no legacy migration).
    assert _tier({"COMPUTE_RUNTIME": legacy}) == "cpu"


def test_missing_tier_defaults_to_cpu():
    assert _tier({}) == "cpu"


def test_invalid_explicit_tier_defaults_to_cpu():
    # A bogus RUNTIME_TIER is not trusted and falls back to "cpu".
    assert _tier({"RUNTIME_TIER": "bogus", "COMPUTE_RUNTIME": "cuda"}) == "cpu"
