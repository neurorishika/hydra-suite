"""Regression guard: the tracking worker threads `runtime_tier` into the
`InferenceConfig` it builds.

This was previously a silent gap — `build_inference_config_from_params` built
the config without `runtime_tier`, so it defaulted to "gpu" and GUI/CLI tier
selection never changed the inference device. These tests assert the tier is
honored (explicit `RUNTIME_TIER`) and correctly derived from the legacy
`COMPUTE_RUNTIME` when no explicit tier is given.

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


@pytest.mark.parametrize(
    "legacy,expected",
    [
        ("cpu", "cpu"),
        ("mps", "gpu"),
        ("cuda", "gpu"),
        ("tensorrt", "gpu_fast"),
        ("onnx_cuda", "gpu_fast"),
        ("onnx_coreml", "gpu_fast"),
    ],
)
def test_tier_migrated_from_legacy_compute_runtime_when_no_explicit_tier(
    legacy, expected
):
    assert _tier({"COMPUTE_RUNTIME": legacy}) == expected


def test_invalid_explicit_tier_falls_back_to_migration():
    # A bogus RUNTIME_TIER must not be trusted; fall back to migrating the
    # legacy compute_runtime instead.
    assert _tier({"RUNTIME_TIER": "bogus", "COMPUTE_RUNTIME": "cuda"}) == "gpu"
