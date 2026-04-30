"""Tests for shared project-bundle manifest loading behavior."""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

import pytest

from hydra_suite.data.project_bundle import (
    ProjectBundleManifest,
    export_project_bundle_archive,
    import_project_bundle_archive,
    load_project_bundle_archive_manifest,
    load_project_bundle_manifest,
    save_project_bundle_manifest,
)


def test_load_project_bundle_manifest_returns_none_for_malformed_json(
    tmp_path: Path,
) -> None:
    (tmp_path / "hydra_project.json").write_text("{not-json", encoding="utf-8")

    assert load_project_bundle_manifest(tmp_path) is None


def test_load_project_bundle_manifest_returns_none_for_unsupported_version(
    tmp_path: Path,
) -> None:
    save_project_bundle_manifest(
        tmp_path,
        ProjectBundleManifest(
            kit="detectkit",
            state_path="state/detectkit_project.json",
            bundle_version=2,
        ),
    )

    assert load_project_bundle_manifest(tmp_path) is None


def test_load_project_bundle_manifest_strict_raises_for_invalid_manifest(
    tmp_path: Path,
) -> None:
    (tmp_path / "hydra_project.json").write_text(json.dumps(["bad"]), encoding="utf-8")

    with pytest.raises(ValueError):
        load_project_bundle_manifest(tmp_path, strict=True)


def test_project_bundle_archive_round_trip(tmp_path: Path) -> None:
    project_dir = tmp_path / "classkit_project"
    save_project_bundle_manifest(
        project_dir,
        ProjectBundleManifest(
            kit="classkit",
            state_path="state/project.json",
            database_path="state/classkit.db",
        ),
    )
    (project_dir / "state").mkdir(parents=True, exist_ok=True)
    (project_dir / "artifacts" / "models").mkdir(parents=True, exist_ok=True)
    (project_dir / "state" / "project.json").write_text(
        json.dumps({"classes": ["ant"]}),
        encoding="utf-8",
    )
    (project_dir / "state" / "classkit.db").write_text("db", encoding="utf-8")
    (project_dir / "artifacts" / "models" / "latest.pt").write_bytes(b"weights")

    archive_path = export_project_bundle_archive(project_dir, tmp_path / "project.zip")

    manifest = load_project_bundle_archive_manifest(archive_path, strict=True)
    assert manifest.kit == "classkit"

    restored_dir = import_project_bundle_archive(
        archive_path,
        tmp_path / "restored_project",
        expected_kit="classkit",
    )
    assert restored_dir == (tmp_path / "restored_project").resolve()
    assert (restored_dir / "hydra_project.json").exists()
    assert (restored_dir / "state" / "project.json").exists()
    assert (restored_dir / "artifacts" / "models" / "latest.pt").exists()


def test_project_bundle_archive_import_rejects_zip_slip_entries(tmp_path: Path) -> None:
    archive_path = tmp_path / "bad.zip"
    with zipfile.ZipFile(
        archive_path, "w", compression=zipfile.ZIP_DEFLATED
    ) as archive:
        archive.writestr(
            "hydra_project.json",
            json.dumps({"kit": "classkit", "state_path": "state/project.json"}),
        )
        archive.writestr("../escape.txt", "nope")

    with pytest.raises(ValueError):
        import_project_bundle_archive(
            archive_path,
            tmp_path / "restored",
            expected_kit="classkit",
        )
