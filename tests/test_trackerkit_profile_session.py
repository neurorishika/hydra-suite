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
