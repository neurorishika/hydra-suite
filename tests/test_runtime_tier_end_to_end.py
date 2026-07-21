"""End-to-end tests for the runtime_tier config contract (Runtime Gen-2).

FT5 made ``runtime_tier`` the sole source of truth: legacy configs that carry
only per-stage ``compute_runtime`` strings and no top-level ``runtime_tier`` are
now a LOUD error at load time (no silent migration). These tests write minimal
configs to a JSON file, load them via ``InferenceConfig.from_json``, and assert
the new behavior — an explicit tier is preserved, a missing tier raises.
"""

import json

import pytest

from hydra_suite.core.inference.config import InferenceConfig, _dict_to_config


def _write_and_load(tmp_path, payload: dict) -> InferenceConfig:
    """Write *payload* to a temp JSON file and load via InferenceConfig.from_json."""
    p = tmp_path / "config.json"
    p.write_text(json.dumps(payload))
    return InferenceConfig.from_json(str(p))


# ---------------------------------------------------------------------------
# Missing runtime_tier is a loud error (clean break — no silent migration)
# ---------------------------------------------------------------------------


def test_missing_runtime_tier_raises_on_load(tmp_path):
    payload = {
        "obb": {"mode": "direct", "direct": {"model_path": "m.pt"}},
        "pipeline_depth": 2,
    }
    with pytest.raises(ValueError) as excinfo:
        _write_and_load(tmp_path, payload)
    msg = str(excinfo.value)
    assert "runtime_tier" in msg
    assert "migrate_runtime_config" in msg


def test_missing_runtime_tier_raises_via_dict_to_config():
    payload = {"obb": {"mode": "direct", "direct": {"model_path": "m.pt"}}}
    with pytest.raises(ValueError, match="migrate_runtime_config"):
        _dict_to_config(payload)


# ---------------------------------------------------------------------------
# Explicit runtime_tier is preserved verbatim
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("tier", ["cpu", "gpu", "gpu_fast"])
def test_explicit_runtime_tier_is_preserved(tmp_path, tier):
    payload = {
        "obb": {"mode": "direct", "direct": {"model_path": "m.pt"}},
        "runtime_tier": tier,
    }
    cfg = _write_and_load(tmp_path, payload)
    assert cfg.runtime_tier == tier


def test_explicit_runtime_tier_preserved_with_sequential(tmp_path):
    payload = {
        "obb": {
            "mode": "sequential",
            "sequential": {
                "detect_model_path": "det.pt",
                "obb_model_path": "obb.pt",
            },
        },
        "runtime_tier": "gpu_fast",
    }
    cfg = _write_and_load(tmp_path, payload)
    assert cfg.runtime_tier == "gpu_fast"
