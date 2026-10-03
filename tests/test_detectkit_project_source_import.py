"""Regression checks for importing all sources from another DetectKit project."""

from pathlib import Path

import pytest

from hydra_suite.detectkit.gui.models import OBBSource
from hydra_suite.detectkit.gui.project import create_project, save_project
from hydra_suite.detectkit.gui.project_source_import import project_sources_to_import


def _source(root: Path, *, classes: str = "ant\n") -> Path:
    (root / "images").mkdir(parents=True)
    (root / "labels").mkdir()
    (root / "images" / "frame.jpg").write_bytes(b"image")
    (root / "labels" / "frame.txt").write_text("0 0.5 0.5 0.4 0.2\n")
    (root / "classes.txt").write_text(classes)
    return root


def test_project_sources_to_import_skips_existing_and_keeps_all_new(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("HYDRA_DATA_DIR", str(tmp_path / "user-data"))
    source_project = create_project(tmp_path / "from", class_names=["ant"])
    roots = [_source(tmp_path / f"dataset-{index}") for index in range(3)]
    source_project.sources = [
        OBBSource(path=str(root), name=root.name) for root in roots
    ]
    save_project(source_project)
    destination = create_project(tmp_path / "to", class_names=["ant"])
    destination.sources = [OBBSource(path=str(roots[0]), name="existing")]

    assert (
        project_sources_to_import(destination, source_project.project_dir) == roots[1:]
    )


def test_project_sources_to_import_rejects_incompatible_classes(tmp_path, monkeypatch):
    monkeypatch.setenv("HYDRA_DATA_DIR", str(tmp_path / "user-data"))
    source_project = create_project(tmp_path / "from", class_names=["bee"])
    source_project.sources = [
        OBBSource(path=str(_source(tmp_path / "dataset", classes="bee\n")))
    ]
    save_project(source_project)
    destination = create_project(tmp_path / "to", class_names=["ant"])

    with pytest.raises(ValueError, match="Project classes must match"):
        project_sources_to_import(destination, source_project.project_dir)
    assert destination.sources == []


def test_project_sources_to_import_rejects_missing_source(tmp_path, monkeypatch):
    monkeypatch.setenv("HYDRA_DATA_DIR", str(tmp_path / "user-data"))
    source_project = create_project(tmp_path / "from", class_names=["ant"])
    source_project.sources = [OBBSource(path=str(tmp_path / "missing"))]
    save_project(source_project)
    destination = create_project(tmp_path / "to", class_names=["ant"])

    with pytest.raises(ValueError, match="missing"):
        project_sources_to_import(destination, source_project.project_dir)
