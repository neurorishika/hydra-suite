"""Tests for registry compatibility helpers: accept both flat-root and v2-root formats."""

from __future__ import annotations

import json


def test_iter_registry_entries_flat_root(tmp_path, monkeypatch):
    """iter_registry_entries yields (key, metadata) pairs from flat-root format."""
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
    assert len(entries) == 2
    assert entries["classification/identity/a.pth"]["arch"] == "tinyclassifier"


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


def test_save_registry_preserves_format_during_phase_1(tmp_path, monkeypatch):
    """save_model_registry writes flat-root format during phases 1–6."""
    from hydra_suite.training import model_publish

    reg_path = tmp_path / "model_registry.json"
    monkeypatch.setattr(model_publish, "_registry_path", lambda: reg_path)

    model_publish.save_model_registry({"a.pth": {"arch": "x"}})
    data = json.loads(reg_path.read_text())
    # Phase 1 still writes the current flat-root dict format
    assert data == {"a.pth": {"arch": "x"}}


def test_load_registry_empty_on_missing_file(tmp_path, monkeypatch):
    from hydra_suite.training import model_publish

    reg_path = tmp_path / "nonexistent.json"
    monkeypatch.setattr(model_publish, "_registry_path", lambda: reg_path)
    assert model_publish.load_model_registry() == {}
    assert dict(model_publish.iter_registry_entries()) == {}
