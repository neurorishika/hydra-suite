"""Tests for DetectKit project model and persistence."""

from __future__ import annotations

import json
from pathlib import Path

from hydra_suite.detectkit.gui.models import DetectKitProject, OBBSource
from hydra_suite.detectkit.gui.project import (
    create_project,
    default_project_parent_dir,
    detectkit_artifact_paths,
    legacy_project_file_path,
    open_project,
    project_exists,
    project_file_path,
)


def test_project_roundtrip(tmp_path: Path):
    proj = DetectKitProject(
        project_dir=tmp_path,
        class_names=["ant", "bee"],
        sources=[
            OBBSource(path=str(tmp_path / "ds1"), name="ds1"),
            OBBSource(path=str(tmp_path / "ds2"), name="ds2"),
        ],
    )
    proj_file = tmp_path / "detectkit_project.json"
    proj.save(proj_file)
    assert proj_file.exists()

    loaded = DetectKitProject.load(proj_file)
    assert loaded.class_name == "ant"
    assert loaded.class_names == ["ant", "bee"]
    assert len(loaded.sources) == 2
    assert loaded.sources[0].name == "ds1"


def test_project_loads_legacy_single_class_field(tmp_path: Path):
    proj_file = tmp_path / "detectkit_project.json"
    proj_file.write_text(
        json.dumps(
            {
                "version": 1,
                "project_dir": str(tmp_path),
                "class_name": "ant",
                "sources": [],
            }
        ),
        encoding="utf-8",
    )

    loaded = DetectKitProject.load(proj_file)

    assert loaded.class_name == "ant"
    assert loaded.class_names == ["ant"]


def test_project_defaults():
    proj = DetectKitProject(project_dir=Path("/tmp/test"))
    assert proj.class_name == "object"
    assert proj.class_names == ["object"]
    assert proj.sources == []
    assert proj.split_train == 0.8
    assert proj.split_val == 0.2
    assert proj.seed == 42


def test_obb_source_roundtrip():
    src = OBBSource(
        path="/data/obb_ds",
        name="my_dataset",
        original_path="/mnt/original/obb_ds",
        source_kind="yolo_detect",
        imported=True,
    )
    d = src.to_dict()
    restored = OBBSource.from_dict(d)
    assert restored.path == "/data/obb_ds"
    assert restored.name == "my_dataset"
    assert restored.original_path == "/mnt/original/obb_ds"
    assert restored.source_kind == "yolo_detect"
    assert restored.imported is True


def test_default_project_parent_dir_uses_hydra_projects_root(
    tmp_path: Path, monkeypatch
):
    monkeypatch.setenv("HYDRA_PROJECTS_DIR", str(tmp_path / "hydra-projects"))

    assert default_project_parent_dir() == tmp_path / "hydra-projects" / "DetectKit"


def test_create_project_uses_bundle_layout(tmp_path: Path) -> None:
    proj = create_project(tmp_path, class_names=["ant", "bee"])
    artifact_paths = detectkit_artifact_paths(tmp_path)

    assert proj.project_dir == tmp_path.resolve()
    assert (tmp_path / "hydra_project.json").exists()
    assert (tmp_path / "state").is_dir()
    assert (tmp_path / "artifacts").is_dir()
    assert (tmp_path / "history").is_dir()
    assert artifact_paths["training_runs"].is_dir()
    assert artifact_paths["evaluation"].is_dir()
    assert artifact_paths["exports"].is_dir()
    assert project_file_path(tmp_path).exists()
    assert not legacy_project_file_path(tmp_path).exists()
    assert project_exists(tmp_path) is True


def test_open_project_migrates_legacy_root_file_to_bundle(tmp_path: Path) -> None:
    legacy_path = legacy_project_file_path(tmp_path)
    legacy_path.write_text(
        json.dumps(
            {
                "version": 1,
                "project_dir": str(tmp_path),
                "class_names": ["ant", "bee"],
                "sources": [{"path": "/data/ds1", "name": "ds1"}],
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    loaded = open_project(tmp_path)

    assert loaded is not None
    assert loaded.class_names == ["ant", "bee"]
    assert loaded.sources[0].name == "ds1"
    assert (tmp_path / "hydra_project.json").exists()
    assert project_file_path(tmp_path).exists()
    assert not legacy_path.exists()
    assert (tmp_path / "history" / "legacy_detectkit_project.json").exists()


def test_open_project_reads_bundle_manifest(tmp_path: Path) -> None:
    created = create_project(tmp_path, class_names=["ant", "bee"])

    loaded = open_project(tmp_path)

    assert loaded is not None
    assert loaded.project_dir == created.project_dir
    assert loaded.class_names == ["ant", "bee"]


def test_open_project_recovers_from_malformed_manifest_using_legacy_file(
    tmp_path: Path,
) -> None:
    (tmp_path / "hydra_project.json").write_text("{bad-manifest", encoding="utf-8")
    legacy_path = legacy_project_file_path(tmp_path)
    legacy_path.write_text(
        json.dumps(
            {
                "version": 1,
                "project_dir": str(tmp_path),
                "class_names": ["ant"],
                "sources": [],
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    loaded = open_project(tmp_path)

    assert loaded is not None
    assert loaded.class_names == ["ant"]
    assert project_file_path(tmp_path).exists()
    assert (tmp_path / "history" / "legacy_detectkit_project.json").exists()
