"""Tests for scripts/migrate_runtime_config.py.

The script is deliberately self-contained (does not import from
``hydra_suite``), so it is loaded directly from ``scripts/`` rather than via
a package import.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SCRIPT_PATH = _REPO_ROOT / "scripts" / "migrate_runtime_config.py"

_spec = importlib.util.spec_from_file_location("migrate_runtime_config", _SCRIPT_PATH)
assert _spec is not None and _spec.loader is not None
migrate_runtime_config = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = migrate_runtime_config
_spec.loader.exec_module(migrate_runtime_config)

migrate_config_dict = migrate_runtime_config.migrate_config_dict
main = migrate_runtime_config.main


def test_flat_preset_migrates_to_gpu_and_strips_legacy_keys():
    original = {
        "compute_runtime": "mps",
        "pose_runtime_flavor": "mps",
        "pose_sleap_device": "mps",
        "preset_name": "obiroi",
    }
    result = migrate_config_dict(original)

    assert result["runtime_tier"] == "gpu"
    assert "compute_runtime" not in result
    assert "pose_runtime_flavor" not in result
    assert "pose_sleap_device" not in result
    assert result["preset_name"] == "obiroi"
    # Input must not be mutated.
    assert "compute_runtime" in original


def test_nested_obb_direct_cuda_migrates_to_gpu():
    original = {"obb": {"direct": {"compute_runtime": "cuda"}}}
    result = migrate_config_dict(original)

    assert result["runtime_tier"] == "gpu"
    assert "compute_runtime" not in result["obb"]["direct"]


def test_nested_pose_sleap_tensorrt_migrates_to_gpu_fast():
    original = {"pose": {"sleap": {"compute_runtime": "tensorrt"}}}
    result = migrate_config_dict(original)

    assert result["runtime_tier"] == "gpu_fast"
    assert "compute_runtime" not in result["pose"]["sleap"]


def test_nested_obb_direct_cpu_migrates_to_cpu():
    original = {"obb": {"direct": {"compute_runtime": "cpu"}}}
    result = migrate_config_dict(original)

    assert result["runtime_tier"] == "cpu"


def test_existing_valid_tier_is_idempotent():
    original = {"runtime_tier": "gpu_fast", "obb": {}}
    result = migrate_config_dict(original)

    assert result["runtime_tier"] == "gpu_fast"


def test_empty_dict_defaults_to_gpu():
    # A config with no runtime string at all maps to "gpu", matching the legacy
    # load path (migrate_runtime_to_tier defaulted an empty set to "gpu"), so a
    # runtime-less config keeps its old effective tier after migration.
    assert migrate_config_dict({})["runtime_tier"] == "gpu"


def test_onnx_and_alias_normalization_maps_to_gpu_fast():
    original = {"pose_runtime_flavor": "onnx_mps"}
    assert migrate_config_dict(original)["runtime_tier"] == "gpu_fast"

    original2 = {"headtail": {"compute_runtime": "cuda:0"}}
    assert migrate_config_dict(original2)["runtime_tier"] == "gpu"

    original3 = {"pose": {"yolo": {"compute_runtime": "trt"}}}
    assert migrate_config_dict(original3)["runtime_tier"] == "gpu_fast"


def test_cnn_phases_list_contributes_runtime():
    original = {
        "cnn_phases": [{"compute_runtime": "cpu"}, {"compute_runtime": "onnx_cuda"}]
    }
    assert migrate_config_dict(original)["runtime_tier"] == "gpu_fast"


BUNDLED_CONFIG = (
    _REPO_ROOT / "src" / "hydra_suite" / "resources" / "configs" / "ooceraea_biroi.json"
)


@pytest.mark.skipif(not BUNDLED_CONFIG.exists(), reason="bundled config not present")
def test_bundled_config_round_trips(tmp_path):
    data = json.loads(BUNDLED_CONFIG.read_text())
    result = migrate_config_dict(data)

    assert result["runtime_tier"] == "gpu"
    for legacy_key in (
        "compute_runtime",
        "pose_runtime_flavor",
        "pose_sleap_device",
    ):
        assert legacy_key not in result
    # Non-runtime keys survive untouched.
    assert result["preset_name"] == data["preset_name"]
    assert result["yolo_model_path"] == data["yolo_model_path"]


def test_main_cli_writes_bak_and_migrates_file(tmp_path):
    target = tmp_path / "preset.json"
    target.write_text(json.dumps({"compute_runtime": "cuda"}))

    exit_code = main([str(target)])

    assert exit_code == 0
    migrated = json.loads(target.read_text())
    assert migrated["runtime_tier"] == "gpu"
    assert "compute_runtime" not in migrated

    bak = target.with_suffix(".json.bak")
    assert bak.exists()
    assert json.loads(bak.read_text()) == {"compute_runtime": "cuda"}


def test_main_does_not_overwrite_existing_bak(tmp_path):
    target = tmp_path / "preset.json"
    target.write_text(json.dumps({"compute_runtime": "cuda"}))
    bak = target.with_suffix(".json.bak")
    bak.write_text("SENTINEL")

    main([str(target)])

    assert bak.read_text() == "SENTINEL"


def test_main_no_args_returns_error(capsys):
    exit_code = main([])
    assert exit_code == 1
