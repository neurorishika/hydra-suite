"""Removing a source deletes its project-owned imported copy, and nothing else."""

from __future__ import annotations

from pathlib import Path

from hydra_suite.detectkit.gui.source_removal import (
    IMPORTED_SOURCES_RELDIR,
    delete_imported_copy,
    owned_imported_copy,
)


def _imported(project: Path, name: str) -> Path:
    path = project / IMPORTED_SOURCES_RELDIR / name
    (path / "images").mkdir(parents=True)
    (path / "images" / "a.jpg").write_bytes(b"x")
    return path


def test_deletes_imported_copy(tmp_path: Path):
    copy = _imported(tmp_path, "ds-abc")
    assert delete_imported_copy(copy, tmp_path)
    assert not copy.exists()
    assert (tmp_path / IMPORTED_SOURCES_RELDIR).is_dir()


def test_linked_source_outside_project_is_never_deleted(tmp_path: Path):
    linked = tmp_path / "user_data"
    (linked / "images").mkdir(parents=True)
    assert not delete_imported_copy(linked, tmp_path / "project")
    assert linked.exists()


def test_imported_root_itself_and_nested_paths_are_refused(tmp_path: Path):
    copy = _imported(tmp_path, "ds-abc")
    assert owned_imported_copy(tmp_path / IMPORTED_SOURCES_RELDIR, tmp_path) is None
    assert owned_imported_copy(copy / "images", tmp_path) is None
    assert owned_imported_copy(tmp_path, tmp_path) is None


def test_escape_via_dotdot_is_refused(tmp_path: Path):
    outside = tmp_path / "outside"
    outside.mkdir()
    sneaky = (
        tmp_path / "proj" / IMPORTED_SOURCES_RELDIR / ".." / ".." / ".." / "outside"
    )
    (tmp_path / "proj" / IMPORTED_SOURCES_RELDIR).mkdir(parents=True)
    assert not delete_imported_copy(sneaky, tmp_path / "proj")
    assert outside.exists()


def test_copy_still_used_by_another_source_is_kept(tmp_path: Path):
    copy = _imported(tmp_path, "ds-abc")
    assert not delete_imported_copy(copy, tmp_path, [copy])
    assert not delete_imported_copy(copy, tmp_path, [copy / "images"])
    assert copy.exists()


def test_missing_project_dir_or_path_is_a_noop(tmp_path: Path):
    copy = _imported(tmp_path, "ds-abc")
    assert not delete_imported_copy(copy, None)
    assert not delete_imported_copy("", tmp_path)
    assert copy.exists()
