import json

import pytest

from hydra_suite.trackerkit.cli_config import apply_sahi_profile_override


def _cfg_with_model(tmp_path, profiles):
    model = tmp_path / "m.pt"
    model.write_text("stub", encoding="utf-8")
    (tmp_path / "m.pt.slice_meta.json").write_text(
        json.dumps(
            {
                "schema_version": 2,
                "training_geometry": {"geometry_mode": "auto_model"},
                "primary_profile_id": "",
                "profiles": profiles,
            }
        ),
        encoding="utf-8",
    )
    return {"yolo_obb_direct_model_path": str(model)}


PROFILES = [
    {"id": "p-abc", "name": "High recall", "settings": {"enabled": True}},
    {"id": "p-def", "name": "Balanced", "settings": {"enabled": True}},
]


def test_override_by_name(tmp_path):
    cfg = _cfg_with_model(tmp_path, PROFILES)
    assert apply_sahi_profile_override(cfg, "Balanced")["slice_profile_id"] == "p-def"


def test_override_by_id(tmp_path):
    cfg = _cfg_with_model(tmp_path, PROFILES)
    assert apply_sahi_profile_override(cfg, "p-abc")["slice_profile_id"] == "p-abc"


def test_override_clears_stale_snapshot(tmp_path):
    cfg = _cfg_with_model(tmp_path, PROFILES)
    cfg["slice_profile_settings"] = {"overlap": 0.9}
    out = apply_sahi_profile_override(cfg, "Balanced")
    assert "slice_profile_settings" not in out


def test_unknown_profile_is_a_hard_error(tmp_path):
    cfg = _cfg_with_model(tmp_path, PROFILES)
    with pytest.raises(ValueError) as excinfo:
        apply_sahi_profile_override(cfg, "Nope")
    assert "Nope" in str(excinfo.value)
    assert "High recall" in str(excinfo.value)


def test_missing_sidecar_is_a_hard_error(tmp_path):
    model = tmp_path / "bare.pt"
    model.write_text("stub", encoding="utf-8")
    with pytest.raises(ValueError):
        apply_sahi_profile_override(
            {"yolo_obb_direct_model_path": str(model)}, "Balanced"
        )


def test_sequential_mode_is_a_hard_error(tmp_path):
    cfg = _cfg_with_model(tmp_path, PROFILES)
    cfg["yolo_obb_mode"] = "sequential"
    with pytest.raises(ValueError) as excinfo:
        apply_sahi_profile_override(cfg, "Balanced")
    assert "sequential" in str(excinfo.value)


def test_training_keyword_is_accepted(tmp_path):
    cfg = _cfg_with_model(tmp_path, PROFILES)
    out = apply_sahi_profile_override(cfg, "__training__")
    assert out["slice_profile_id"] == "__training__"


def test_original_cfg_is_not_mutated(tmp_path):
    cfg = _cfg_with_model(tmp_path, PROFILES)
    apply_sahi_profile_override(cfg, "Balanced")
    assert "slice_profile_id" not in cfg
