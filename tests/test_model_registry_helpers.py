"""Tests for registry v2-root format: flat-root is no longer supported."""

from __future__ import annotations

import json


def test_iter_registry_entries_flat_root(tmp_path, monkeypatch):
    """Phase 7: flat-root registries yield nothing (no longer accepted)."""
    from hydra_suite.training import model_publish

    reg_path = tmp_path / "model_registry.json"
    reg_path.write_text(
        json.dumps(
            {
                "classification/identity/a.pth": {
                    "arch": "tinyclassifier",
                    "usage_role": "cnn_identity",
                },
                "classification/identity/b.pth": {
                    "arch": "yolo",
                    "usage_role": "cnn_identity",
                },
            }
        )
    )
    monkeypatch.setattr(model_publish, "_registry_path", lambda: reg_path)
    entries = dict(model_publish.iter_registry_entries())
    assert entries == {}


def test_iter_registry_entries_v2_root(tmp_path, monkeypatch):
    """iter_registry_entries yields entries from v2 root format."""
    from hydra_suite.training import model_publish

    reg_path = tmp_path / "model_registry.json"
    reg_path.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "entries": {
                    "classification/identity/a.pth": {
                        "schema_version": 2,
                        "arch": "resnet18",
                        "usage_role": "cnn_identity",
                    },
                },
            }
        )
    )
    monkeypatch.setattr(model_publish, "_registry_path", lambda: reg_path)
    entries = dict(model_publish.iter_registry_entries())
    assert len(entries) == 1
    assert entries["classification/identity/a.pth"]["arch"] == "resnet18"


def test_save_registry_writes_v2_root(tmp_path, monkeypatch):
    """save_model_registry writes the v2 root format ({schema_version, entries})."""
    from hydra_suite.training import model_publish

    reg_path = tmp_path / "model_registry.json"
    monkeypatch.setattr(model_publish, "_registry_path", lambda: reg_path)

    model_publish.save_model_registry({"a.pth": {"arch": "x"}})
    import json as _json

    data = _json.loads(reg_path.read_text())
    assert data["schema_version"] == 2
    assert data["entries"] == {"a.pth": {"arch": "x"}}


def test_load_registry_rejects_flat_root_after_flip(tmp_path, monkeypatch):
    """Phase 7: flat-root registries are no longer accepted on load."""
    from hydra_suite.training import model_publish

    reg_path = tmp_path / "model_registry.json"
    reg_path.write_text('{"a.pth": {"arch": "x"}}')
    monkeypatch.setattr(model_publish, "_registry_path", lambda: reg_path)

    entries = list(model_publish.iter_registry_entries())
    assert entries == []


def test_load_registry_empty_on_missing_file(tmp_path, monkeypatch):
    from hydra_suite.training import model_publish

    reg_path = tmp_path / "nonexistent.json"
    monkeypatch.setattr(model_publish, "_registry_path", lambda: reg_path)
    assert model_publish.load_model_registry() == {}
    assert dict(model_publish.iter_registry_entries()) == {}
