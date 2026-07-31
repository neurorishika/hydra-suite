import pytest

from hydra_suite.core.inference.config import _dict_to_config, migrate_runtime_to_tier

_MINIMAL_OBB = {
    "mode": "direct",
    "direct": {"model_path": "/tmp/obb.pt"},
}


def test_missing_runtime_tier_raises():
    """Runtime Gen-2: a config dict with no runtime_tier is a loud error."""
    d = {"obb": _MINIMAL_OBB}
    with pytest.raises(ValueError) as excinfo:
        _dict_to_config(d)
    msg = str(excinfo.value)
    assert "runtime_tier" in msg
    assert "migrate_runtime_config" in msg


def test_explicit_runtime_tier_is_preserved():
    d = {"obb": _MINIMAL_OBB, "runtime_tier": "gpu_fast"}
    config = _dict_to_config(d)
    assert config.runtime_tier == "gpu_fast"


# ── migrate_runtime_to_tier is retained for other callers (public api.py entry
#    points, worker fallbacks, cli_config, schemas); its unit contract stays. ──


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
