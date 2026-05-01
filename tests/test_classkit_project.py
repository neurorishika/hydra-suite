"""Tests for ClassKit project-path helpers."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from hydra_suite.classkit.core.store.db import ClassKitDB
from hydra_suite.classkit.gui.project import (
    classkit_config_path,
    classkit_db_path,
    classkit_project_is_portable,
    classkit_project_linked_image_count,
    classkit_scheme_path,
    default_project_parent_dir,
    prepare_project_directory,
    project_exists,
)
from hydra_suite.data.project_bundle import (
    export_project_bundle_archive,
    import_project_bundle_archive,
)


def test_default_project_parent_dir_uses_hydra_projects_root(
    tmp_path: Path, monkeypatch
):
    monkeypatch.setenv("HYDRA_PROJECTS_DIR", str(tmp_path / "hydra-projects"))

    assert default_project_parent_dir() == tmp_path / "hydra-projects" / "ClassKit"


def test_prepare_project_directory_creates_bundle_layout(tmp_path: Path) -> None:
    db_path = prepare_project_directory(tmp_path)

    assert db_path == classkit_db_path(tmp_path)
    assert (tmp_path / "hydra_project.json").exists()
    assert classkit_config_path(tmp_path).parent.is_dir()
    assert classkit_scheme_path(tmp_path).parent.is_dir()
    assert (tmp_path / "artifacts" / "models").is_dir()
    assert (tmp_path / "artifacts" / "exports").is_dir()
    assert project_exists(tmp_path) is True


def test_prepare_project_directory_migrates_legacy_root_files(tmp_path: Path) -> None:
    (tmp_path / "classkit.db").write_text("db", encoding="utf-8")
    (tmp_path / "project.json").write_text(
        json.dumps({"name": "colony", "classes": ["a", "b"]}, indent=2),
        encoding="utf-8",
    )
    (tmp_path / "scheme.json").write_text(
        json.dumps({"name": "scheme"}, indent=2),
        encoding="utf-8",
    )

    db_path = prepare_project_directory(tmp_path)

    assert db_path == classkit_db_path(tmp_path)
    assert classkit_db_path(tmp_path).read_text(encoding="utf-8") == "db"
    assert (
        json.loads(classkit_config_path(tmp_path).read_text(encoding="utf-8"))["name"]
        == "colony"
    )
    assert (
        json.loads(classkit_scheme_path(tmp_path).read_text(encoding="utf-8"))["name"]
        == "scheme"
    )
    assert not (tmp_path / "classkit.db").exists()
    assert not (tmp_path / "project.json").exists()
    assert not (tmp_path / "scheme.json").exists()


def test_prepare_project_directory_recovers_from_malformed_manifest(
    tmp_path: Path,
) -> None:
    (tmp_path / "hydra_project.json").write_text("{bad-manifest", encoding="utf-8")
    (tmp_path / "classkit.db").write_text("db", encoding="utf-8")

    db_path = prepare_project_directory(tmp_path)

    assert db_path == classkit_db_path(tmp_path)
    assert classkit_db_path(tmp_path).read_text(encoding="utf-8") == "db"
    assert (tmp_path / "hydra_project.json").exists()


def test_classkit_project_portability_helpers_count_external_images(
    tmp_path: Path,
) -> None:
    project_dir = tmp_path / "classkit_project"
    prepare_project_directory(project_dir)
    local_image = project_dir / "artifacts" / "imported_sources" / "a.png"
    local_image.parent.mkdir(parents=True, exist_ok=True)
    local_image.write_bytes(b"png")
    external_image = tmp_path / "external" / "b.png"
    external_image.parent.mkdir(parents=True, exist_ok=True)
    external_image.write_bytes(b"png")

    assert (
        classkit_project_linked_image_count(
            project_dir,
            [local_image, external_image],
        )
        == 1
    )
    assert (
        classkit_project_is_portable(project_dir, [local_image, external_image])
        is False
    )
    assert classkit_project_is_portable(project_dir, [local_image]) is True


def test_classkit_db_materializes_linked_images_into_bundle(tmp_path: Path) -> None:
    project_dir = tmp_path / "classkit_project"
    db_path = prepare_project_directory(project_dir)
    external_dir = tmp_path / "external_source"
    external_dir.mkdir(parents=True, exist_ok=True)
    external_image = external_dir / "frame_a.png"
    external_image.write_bytes(b"png")

    db = ClassKitDB(db_path)
    db.add_images(
        [external_image],
        metadata_by_path={
            str(external_image.resolve()): {
                "source_root": str(external_dir.resolve()),
            }
        },
    )

    assert db.materialize_linked_images() == 1

    image_paths = db.get_all_image_paths()
    assert len(image_paths) == 1
    materialized_path = Path(image_paths[0])
    assert materialized_path.exists()
    assert materialized_path.is_relative_to(project_dir.resolve())

    metadata = db.get_image_metadata_by_path()[image_paths[0]]
    assert metadata["source_root"] == str(external_dir.resolve())
    assert Path(metadata["standardized_source_dir"]).is_relative_to(
        project_dir.resolve()
    )


def test_classkit_db_relocates_bundle_owned_paths_after_archive_import(
    tmp_path: Path,
) -> None:
    project_dir = tmp_path / "classkit_project"
    db_path = prepare_project_directory(project_dir)
    imported_root = project_dir / "artifacts" / "imported_sources" / "source-a"
    imported_image = imported_root / "images" / "000001.png"
    imported_image.parent.mkdir(parents=True, exist_ok=True)
    imported_image.write_bytes(b"fake-png")

    db = ClassKitDB(db_path)
    image_key = str(imported_image.resolve())
    db.add_images(
        [imported_image],
        metadata_by_path={
            image_key: {
                "standardized_source_dir": str(imported_root.resolve()),
                "source_root": "/external/source-a",
            }
        },
    )
    db.save_embeddings(
        np.zeros((1, 2), dtype=np.float32),
        model_name="toy",
        image_paths=[imported_image],
    )

    archive_path = export_project_bundle_archive(project_dir, tmp_path / "classkit.zip")
    restored_dir = import_project_bundle_archive(
        archive_path,
        tmp_path / "restored_classkit_project",
        expected_kit="classkit",
    )

    restored_db = ClassKitDB(classkit_db_path(restored_dir))

    assert restored_db.relocate_project_owned_paths() > 0

    expected_image_path = str(
        (
            restored_dir
            / "artifacts"
            / "imported_sources"
            / "source-a"
            / "images"
            / "000001.png"
        ).resolve()
    )
    assert restored_db.get_all_image_paths() == [expected_image_path]

    metadata = restored_db.get_image_metadata_by_path()[expected_image_path]
    assert metadata["standardized_source_dir"] == str(
        (restored_dir / "artifacts" / "imported_sources" / "source-a").resolve()
    )
    assert restored_db.get_most_recent_embeddings() is not None


def test_relocate_rebases_classkit_runs_model_artifacts(tmp_path: Path) -> None:
    """Model checkpoints under .classkit_runs/ rebase after a project move."""
    original_dir = tmp_path / "original_project"
    db_path = prepare_project_directory(original_dir)
    run_dir = original_dir / ".classkit_runs" / "flat_custom_20260101_000000"
    run_dir.mkdir(parents=True)
    artifact = run_dir / "best.pth"
    artifact.write_bytes(b"fake-checkpoint")

    db = ClassKitDB(db_path)
    db.save_model_cache(
        mode="flat_custom",
        artifact_paths=[str(artifact.resolve())],
        class_names=["a", "b"],
        num_classes=2,
    )

    moved_dir = tmp_path / "moved_project"
    original_dir.rename(moved_dir)

    moved_db = ClassKitDB(classkit_db_path(moved_dir))
    assert moved_db.relocate_project_owned_paths() > 0

    entries = moved_db.list_model_caches()
    assert len(entries) == 1
    rebased = Path(entries[0]["artifact_paths"][0])
    assert (
        rebased
        == (
            moved_dir / ".classkit_runs" / "flat_custom_20260101_000000" / "best.pth"
        ).resolve()
    )
    assert rebased.exists()
