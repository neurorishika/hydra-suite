"""Tests for PoseKit bundle persistence and legacy migration helpers."""

from __future__ import annotations

import json
from pathlib import Path

from hydra_suite.posekit.gui.constants import (
    DEFAULT_DATASET_IMAGES_DIR,
    DEFAULT_POSEKIT_PROJECT_DIR,
    DEFAULT_PROJECT_NAME,
)
from hydra_suite.posekit.gui.models import Project
from hydra_suite.posekit.gui.project import (
    add_source_to_project,
    canonical_project_path,
    find_project,
    open_project_from_path,
    remove_source_from_project,
    save_project_state,
)


def _make_posekit_project(dataset_root: Path) -> Project:
    images_dir = dataset_root / DEFAULT_DATASET_IMAGES_DIR
    out_root = dataset_root / DEFAULT_POSEKIT_PROJECT_DIR
    labels_dir = out_root / "labels"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)
    return Project(
        images_dir=images_dir,
        out_root=out_root,
        labels_dir=labels_dir,
        project_path=out_root / DEFAULT_PROJECT_NAME,
        class_names=["ant"],
        keypoint_names=["head", "tail"],
        skeleton_edges=[(0, 1)],
    )


def test_posekit_save_project_state_uses_bundle_layout(tmp_path: Path) -> None:
    dataset_root = tmp_path / "dataset"
    project = _make_posekit_project(dataset_root)

    saved_path = save_project_state(project)

    assert saved_path == canonical_project_path(project.out_root)
    assert saved_path.exists()
    assert (project.out_root / "hydra_project.json").exists()
    assert (project.out_root / "artifacts" / "cache").is_dir()
    assert (project.out_root / "artifacts" / "exports").is_dir()


def test_posekit_find_project_prefers_bundle_state_path(tmp_path: Path) -> None:
    dataset_root = tmp_path / "dataset"
    project = _make_posekit_project(dataset_root)
    save_project_state(project)

    found = find_project(dataset_root)

    assert found == canonical_project_path(project.out_root)


def test_posekit_open_project_from_manifest_loads_bundled_state(tmp_path: Path) -> None:
    dataset_root = tmp_path / "dataset"
    project = _make_posekit_project(dataset_root)
    save_project_state(project)

    loaded = open_project_from_path(project.out_root / "hydra_project.json")

    assert loaded is not None
    assert loaded.project_path == canonical_project_path(project.out_root)
    assert loaded.class_names == ["ant"]


def test_posekit_bundle_round_trip_preserves_last_source_id(tmp_path: Path) -> None:
    dataset_root = tmp_path / "dataset"
    project = _make_posekit_project(dataset_root)
    project.last_source_id = "source_3"
    save_project_state(project)

    loaded = open_project_from_path(project.out_root / "hydra_project.json")

    assert loaded is not None
    assert loaded.last_source_id == "source_3"


def test_posekit_add_and_remove_source_updates_empty_project_state(
    tmp_path: Path,
) -> None:
    project_root = tmp_path / "pose_project"
    images_dir = project_root / DEFAULT_DATASET_IMAGES_DIR
    labels_dir = project_root / "labels"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)
    project = Project(
        images_dir=images_dir,
        out_root=project_root,
        labels_dir=labels_dir,
        project_path=project_root / DEFAULT_PROJECT_NAME,
        class_names=["ant"],
        keypoint_names=["head", "tail"],
        skeleton_edges=[(0, 1)],
    )

    dataset_root = tmp_path / "source_a"
    source_images = dataset_root / DEFAULT_DATASET_IMAGES_DIR
    source_images.mkdir(parents=True, exist_ok=True)
    (source_images / "frame_0.png").write_bytes(b"png")

    added = add_source_to_project(project, dataset_root, "colony a")

    assert project.sources[0].source_id == added.source_id
    assert project.last_source_id == added.source_id
    assert project.images_dir == added.images_dir

    removed = remove_source_from_project(project, added.source_id)

    assert removed is True
    assert project.sources == []
    assert project.last_source_id == ""
    assert project.images_dir == project.out_root / DEFAULT_DATASET_IMAGES_DIR


def test_posekit_open_project_from_legacy_file_migrates_to_bundle(
    tmp_path: Path,
) -> None:
    dataset_root = tmp_path / "dataset"
    project = _make_posekit_project(dataset_root)
    legacy_path = project.out_root / DEFAULT_PROJECT_NAME
    legacy_path.parent.mkdir(parents=True, exist_ok=True)
    legacy_path.write_text(json.dumps(project.to_json(), indent=2), encoding="utf-8")

    loaded = open_project_from_path(legacy_path)

    assert loaded is not None
    assert loaded.project_path == canonical_project_path(project.out_root)
    assert canonical_project_path(project.out_root).exists()
    assert not legacy_path.exists()
    assert (project.out_root / "history" / f"legacy_{DEFAULT_PROJECT_NAME}").exists()


def test_posekit_open_project_from_manifest_recovers_using_legacy_file(
    tmp_path: Path,
) -> None:
    dataset_root = tmp_path / "dataset"
    project = _make_posekit_project(dataset_root)
    legacy_path = project.out_root / DEFAULT_PROJECT_NAME
    legacy_path.parent.mkdir(parents=True, exist_ok=True)
    legacy_path.write_text(json.dumps(project.to_json(), indent=2), encoding="utf-8")
    (project.out_root / "hydra_project.json").write_text(
        "{bad-manifest", encoding="utf-8"
    )

    loaded = open_project_from_path(project.out_root / "hydra_project.json")

    assert loaded is not None
    assert loaded.project_path == canonical_project_path(project.out_root)
    assert canonical_project_path(project.out_root).exists()
    assert (project.out_root / "history" / f"legacy_{DEFAULT_PROJECT_NAME}").exists()


def test_cli_project_standalone_bundle_is_not_relocated(tmp_path: Path) -> None:
    """A bundle beside a separate images folder opens where it is."""
    from hydra_suite.posekit.gui.main import dataset_dir_for_cli_project

    crops = tmp_path / "crops"
    (crops / DEFAULT_DATASET_IMAGES_DIR).mkdir(parents=True)
    bundle = tmp_path / "pose_project"
    state = bundle / "state" / DEFAULT_PROJECT_NAME
    state.parent.mkdir(parents=True)
    state.write_text("{}")
    for selected in (state, bundle / "hydra_project.json", bundle):
        assert dataset_dir_for_cli_project(selected, None) is None


def test_cli_project_dataset_anchored_project_uses_dataset(tmp_path: Path) -> None:
    from hydra_suite.posekit.gui.main import dataset_dir_for_cli_project

    proj = _make_posekit_project(tmp_path / "ds")
    path = save_project_state(proj)
    assert dataset_dir_for_cli_project(path, None) == (tmp_path / "ds").resolve()
    explicit = tmp_path / "other"
    assert dataset_dir_for_cli_project(path, explicit) == explicit


def test_cli_project_legacy_layouts_still_repair(tmp_path: Path) -> None:
    from hydra_suite.posekit.gui.main import dataset_dir_for_cli_project

    ds = tmp_path / "ds"
    (ds / DEFAULT_DATASET_IMAGES_DIR).mkdir(parents=True)
    (ds / "labels").mkdir()
    for legacy in (ds / DEFAULT_PROJECT_NAME, ds / "labels" / DEFAULT_PROJECT_NAME):
        legacy.write_text("{}")
        assert dataset_dir_for_cli_project(legacy, None) == ds


def test_cli_project_symlinked_posekit_project_dir(tmp_path: Path) -> None:
    from hydra_suite.posekit.gui.main import dataset_dir_for_cli_project

    ds = tmp_path / "ds"
    (ds / DEFAULT_DATASET_IMAGES_DIR).mkdir(parents=True)
    real = tmp_path / "elsewhere" / "proj"
    (real / "state").mkdir(parents=True)
    (real / "state" / DEFAULT_PROJECT_NAME).write_text("{}")
    (ds / DEFAULT_POSEKIT_PROJECT_DIR).symlink_to(real, target_is_directory=True)
    selected = ds / DEFAULT_POSEKIT_PROJECT_DIR / "state" / DEFAULT_PROJECT_NAME
    assert dataset_dir_for_cli_project(selected, None) == ds
