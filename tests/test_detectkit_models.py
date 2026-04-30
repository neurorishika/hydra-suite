"""Tests for DetectKitProject model — active_model_path field."""

from __future__ import annotations


def test_project_has_active_model_path_field():
    from hydra_suite.detectkit.gui.models import DetectKitProject

    proj = DetectKitProject()
    assert hasattr(
        proj, "active_model_path"
    ), "DetectKitProject must have active_model_path"
    assert proj.active_model_path == "", "Default must be empty string"


def test_project_active_model_path_persists(tmp_path):
    from hydra_suite.detectkit.gui.models import DetectKitProject

    proj = DetectKitProject(project_dir=tmp_path)
    proj.active_model_path = "/some/model.pt"
    save_path = tmp_path / "project.json"
    proj.save(save_path)

    loaded = DetectKitProject.load(save_path)
    assert loaded.active_model_path == "/some/model.pt"


def test_project_load_missing_active_model_path_defaults(tmp_path):
    """Old project files without active_model_path should load without error."""
    import json

    from hydra_suite.detectkit.gui.models import DetectKitProject

    # Write a minimal project JSON without active_model_path
    data = {"version": 1, "class_names": ["ant"]}
    save_path = tmp_path / "project.json"
    save_path.write_text(json.dumps(data), encoding="utf-8")

    proj = DetectKitProject.load(save_path)
    assert proj.active_model_path == ""


def test_project_to_dict_includes_active_model_path():
    from hydra_suite.detectkit.gui.models import DetectKitProject

    proj = DetectKitProject()
    proj.active_model_path = "weights/best.pt"
    d = proj.to_dict()
    assert "active_model_path" in d
    assert d["active_model_path"] == "weights/best.pt"
    assert d["active_model_path"] == "weights/best.pt"


def test_project_training_history_persists(tmp_path):
    from hydra_suite.detectkit.gui.models import DetectKitProject

    proj = DetectKitProject(project_dir=tmp_path)
    proj.training_history = [{"run_id": "run_001", "project_model_path": "model.pt"}]
    save_path = tmp_path / "project.json"
    proj.save(save_path)

    loaded = DetectKitProject.load(save_path)
    assert loaded.training_history == [
        {"run_id": "run_001", "project_model_path": "model.pt"}
    ]
