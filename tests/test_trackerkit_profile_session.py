import hashlib
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from hydra_suite.core.inference.slice_meta import profile_evidence_state


def _profile(digest):
    return {
        "id": "a",
        "name": "Balanced",
        "settings": {},
        "measurement": {"checkpoint_fingerprint": digest},
    }


def test_replaced_weights_invalidate_profile_evidence(tmp_path):
    checkpoint = tmp_path / "m.pt"
    checkpoint.write_bytes(b"weights")
    digest = "sha256:" + hashlib.sha256(b"weights").hexdigest()
    fresh, reason = profile_evidence_state(_profile(digest), checkpoint_path=checkpoint)
    assert fresh is True and reason == ""
    checkpoint.write_bytes(b"retrained")
    fresh, reason = profile_evidence_state(_profile(digest), checkpoint_path=checkpoint)
    assert fresh is False and "weights changed" in reason


def test_missing_provenance_is_not_fatal(tmp_path):
    checkpoint = tmp_path / "m.pt"
    checkpoint.write_bytes(b"weights")
    fresh, reason = profile_evidence_state(
        {"id": "a", "name": "n", "settings": {}, "measurement": {}},
        checkpoint_path=checkpoint,
    )
    assert fresh is True and reason == ""


def test_unreadable_checkpoint_is_not_fatal(tmp_path):
    fresh, reason = profile_evidence_state(
        _profile("sha256:deadbeef"), checkpoint_path=tmp_path / "absent.pt"
    )
    assert fresh is True and reason == ""


def test_unknown_saved_profile_falls_back_visibly(monkeypatch):
    from tests.test_main_window_config_persistence import _make_main_window

    window = _make_main_window(monkeypatch)
    panel = window._detection_panel
    panel._slice_meta = {
        "schema_version": 2,
        "training_geometry": {"geometry_mode": "auto_object", "imgsz": 640},
        "primary_profile_id": "",
        "profiles": [],
    }
    window.advanced_config["slice_profile_id"] = "gone-1234"
    panel._apply_slice_meta_values("gone-1234")
    status = panel.slice_profile_status_text()
    assert "Training geometry" in status and "no longer" in status
    window.close()


def test_session_restores_the_saved_profile_not_a_changed_primary(monkeypatch):
    from hydra_suite.core.inference.slice_meta import upsert_slice_profile
    from tests.test_main_window_config_persistence import _make_main_window

    meta = upsert_slice_profile(
        {"geometry_mode": "auto_object", "imgsz": 640},
        name="Balanced",
        settings={
            "enabled": True,
            "geometry_mode": "auto_object",
            "overlap": 0.2,
            "object_tile_fraction": 0.4,
        },
        primary=True,
    )
    meta = upsert_slice_profile(
        meta,
        name="Fast scan",
        settings={
            "enabled": True,
            "geometry_mode": "auto_object",
            "overlap": 0.1,
            "object_tile_fraction": 0.6,
        },
        primary=True,  # primary later moved to Fast scan
    )
    balanced = next(p for p in meta["profiles"] if p["name"] == "Balanced")

    window = _make_main_window(monkeypatch)
    panel = window._detection_panel
    panel._slice_meta = meta
    window.advanced_config["slice_profile_id"] = balanced["id"]
    panel._apply_slice_meta_values(balanced["id"])
    assert window.advanced_config["slice_object_tile_fraction"] == 0.4
    assert window.advanced_config["slice_profile_id"] == balanced["id"]
    window.close()


def test_switching_models_does_not_carry_profile_settings_over(monkeypatch, tmp_path):
    from tests.test_main_window_config_persistence import _make_main_window

    window = _make_main_window(monkeypatch)
    panel = window._detection_panel
    window.advanced_config["slice_profile_id"] = "stale-1234"
    panel.apply_slice_meta_for_model(str(tmp_path / "other.pt"))  # no sidecar
    assert window.advanced_config.get("slice_profile_id", "") in ("", "__training__")
    window.close()


def test_deleted_profile_falls_back_to_saved_effective_settings(monkeypatch, tmp_path):
    """The third restore rung: the named profile is gone but its settings survive."""
    from hydra_suite.core.inference.slice_meta import (
        sidecar_path,
        upsert_slice_profile,
        write_slice_meta,
    )
    from tests.test_main_window_config_persistence import _make_main_window

    model_path = tmp_path / "model.pt"
    model_path.write_bytes(b"weights")
    meta = upsert_slice_profile(
        {"geometry_mode": "auto_object", "imgsz": 640},
        name="High recall",
        settings={
            "enabled": True,
            "geometry_mode": "auto_object",
            "overlap": 0.35,
            "object_tile_fraction": 0.5,
        },
    )
    high_recall = next(p for p in meta["profiles"] if p["name"] == "High recall")
    # The profile the user chose is later deleted from the sidecar entirely.
    meta_without_profile = {
        "schema_version": 2,
        "training_geometry": meta["training_geometry"],
        "primary_profile_id": "",
        "profiles": [],
    }
    write_slice_meta(model_path, meta_without_profile)
    assert sidecar_path(model_path).exists()

    window = _make_main_window(monkeypatch)
    panel = window._detection_panel
    saved_settings = {
        "enabled": True,
        "geometry_mode": "auto_object",
        "overlap": 0.35,
        "object_tile_fraction": 0.5,
        "trained_body_px": 0.0,
        "slice_width": 0,
        "slice_height": 0,
        "confidence_threshold": None,
        "merge_policy": None,
        "merge_metric": None,
        "merge_threshold": None,
        "merge_backend": None,
    }
    window.advanced_config["slice_profile_id"] = high_recall["id"]
    window.advanced_config["_slice_profile_saved_settings"] = saved_settings
    # This is the panel's first-ever resolution for *this* model (mirroring
    # a session restore's first call for the restored model) -- not a
    # user-driven switch away from whatever model happened to be selected
    # by default when the window was constructed.
    panel._slice_meta_model_path = None
    panel.apply_slice_meta_for_model(str(model_path))

    assert window.advanced_config["slice_object_tile_fraction"] == 0.5
    assert window.advanced_config["slice_overlap"] == 0.35
    status = panel.slice_profile_status_text()
    assert "no longer" in status
    assert "saved settings" in status
    window.close()


def test_profile_session_round_trip_through_save_and_restore(
    monkeypatch, tmp_path
) -> None:
    """Real save/restore path: the chosen profile and its settings return identical."""
    from hydra_suite.core.inference.slice_meta import (
        upsert_slice_profile,
        write_slice_meta,
    )
    from tests.test_main_window_config_persistence import (
        _make_main_window,
        _seed_trackerkit_model_repository,
        _select_first_model_with_suffix,
    )

    models_root = _seed_trackerkit_model_repository(tmp_path, monkeypatch)
    model_path = models_root / "obb" / "direct_keep.pt"

    meta = upsert_slice_profile(
        {"geometry_mode": "auto_object", "imgsz": 640},
        name="High recall",
        settings={
            "enabled": True,
            "geometry_mode": "auto_object",
            "overlap": 0.3,
            "object_tile_fraction": 0.45,
        },
        primary=True,
    )
    high_recall = next(p for p in meta["profiles"] if p["name"] == "High recall")
    write_slice_meta(model_path, meta)

    window = _make_main_window(monkeypatch)
    window._detection_panel.combo_detection_method.setCurrentIndex(1)  # YOLO OBB
    _select_first_model_with_suffix(
        window._detection_panel.combo_yolo_model, "direct_keep.pt"
    )
    window._detection_panel.combo_slice_geometry.setCurrentText("auto_object")
    window._detection_panel._apply_slice_meta_values(high_recall["id"])
    assert window.advanced_config["slice_profile_id"] == high_recall["id"]
    assert window.advanced_config["slice_object_tile_fraction"] == 0.45

    config_path = tmp_path / "profile_roundtrip.json"
    assert window.save_config(preset_mode=True, preset_path=str(config_path))
    window.close()

    reloaded = _make_main_window(monkeypatch)
    reloaded._load_config_from_file(str(config_path), preset_mode=True)

    assert reloaded.advanced_config["slice_profile_id"] == high_recall["id"]
    for key in (
        "slice_overlap",
        "slice_object_tile_fraction",
        "slice_trained_body_px",
        "slice_width",
        "slice_height",
    ):
        assert reloaded.advanced_config[key] == window.advanced_config[key], key
    reloaded.close()


def test_unknown_saved_profile_with_primary_names_the_applied_profile(monkeypatch):
    from tests.test_main_window_config_persistence import _make_main_window

    window = _make_main_window(monkeypatch)
    panel = window._detection_panel
    panel._slice_meta = {
        "schema_version": 2,
        "training_geometry": {"geometry_mode": "auto_object", "imgsz": 640},
        "primary_profile_id": "balanced",
        "profiles": [
            {
                "id": "balanced",
                "name": "Balanced",
                "settings": {"enabled": True, "geometry_mode": "auto_object"},
                "measurement": {},
            }
        ],
    }
    window.advanced_config["slice_profile_id"] = "gone-1234"
    panel._apply_slice_meta_values("gone-1234")
    status = panel.slice_profile_status_text()
    assert "no longer" in status
    assert "Balanced" in status
    assert "Training geometry" not in status
    window.close()
