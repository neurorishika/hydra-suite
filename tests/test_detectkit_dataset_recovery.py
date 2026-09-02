"""Regression coverage for recoverable DetectKit dataset mutations."""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest


def _source(root: Path, name: str = "source") -> tuple[Path, Path, Path]:
    source = root / name
    image = source / "images" / "train" / "frame.jpg"
    label = source / "labels" / "train" / "frame.txt"
    image.parent.mkdir(parents=True)
    label.parent.mkdir(parents=True)
    image.write_bytes(b"image-bytes")
    label.write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")
    (source / "classes.txt").write_text("ant\n", encoding="utf-8")
    return source, image, label


def test_remove_images_stages_image_and_label_then_restores_them(tmp_path: Path):
    from hydra_suite.detectkit.gui.dataset_recovery import (
        latest_dataset_recovery,
        remove_images_with_recovery,
        undo_latest_dataset_recovery,
    )

    project_dir = tmp_path / "project"
    project_dir.mkdir()
    source, image, label = _source(project_dir)

    operation = remove_images_with_recovery(project_dir, source, [image])

    assert operation.action == "remove_images"
    assert operation.item_count == 1
    assert not image.exists()
    assert not label.exists()
    assert operation.manifest_path.is_relative_to(
        project_dir / "artifacts" / "recovery"
    )
    assert latest_dataset_recovery(project_dir).operation_id == operation.operation_id

    restored = undo_latest_dataset_recovery(project_dir)

    assert restored.operation_id == operation.operation_id
    assert image.read_bytes() == b"image-bytes"
    assert label.read_text(encoding="utf-8") == "0 0.5 0.5 0.2 0.2\n"
    assert latest_dataset_recovery(project_dir) is None


def test_clear_labels_snapshots_exact_bytes_and_undo_restores_them(tmp_path: Path):
    from hydra_suite.detectkit.gui.dataset_recovery import (
        clear_labels_with_recovery,
        undo_latest_dataset_recovery,
    )

    project_dir = tmp_path / "project"
    project_dir.mkdir()
    _source_dir, _image, label = _source(project_dir)
    original = b"0 0.123456789 0.5 0.2 0.2\r\n"
    label.write_bytes(original)

    operation = clear_labels_with_recovery(project_dir, [label])

    assert operation.action == "clear_labels"
    assert operation.item_count == 1
    assert label.read_bytes() == b""

    undo_latest_dataset_recovery(project_dir)

    assert label.read_bytes() == original


def test_linked_source_recovery_payload_is_owned_by_project(tmp_path: Path):
    from hydra_suite.detectkit.gui.dataset_recovery import remove_images_with_recovery

    project_dir = tmp_path / "project"
    project_dir.mkdir()
    linked_source, image, _label = _source(tmp_path / "external")

    operation = remove_images_with_recovery(project_dir, linked_source, [image])

    assert not image.exists()
    assert operation.manifest_path.is_relative_to(
        project_dir / "artifacts" / "recovery"
    )
    assert all(
        entry.recovery_path.is_relative_to(project_dir / "artifacts" / "recovery")
        for entry in operation.entries
    )


def test_undo_refuses_to_overwrite_a_recreated_image(tmp_path: Path):
    from hydra_suite.detectkit.gui.dataset_recovery import (
        DatasetRecoveryError,
        latest_dataset_recovery,
        remove_images_with_recovery,
        undo_latest_dataset_recovery,
    )

    project_dir = tmp_path / "project"
    project_dir.mkdir()
    source, image, _label = _source(project_dir)
    operation = remove_images_with_recovery(project_dir, source, [image])
    image.parent.mkdir(parents=True, exist_ok=True)
    image.write_bytes(b"new-file")

    with pytest.raises(DatasetRecoveryError, match="already exists"):
        undo_latest_dataset_recovery(project_dir)

    assert image.read_bytes() == b"new-file"
    assert latest_dataset_recovery(project_dir).operation_id == operation.operation_id


def test_undo_refuses_to_overwrite_labels_edited_after_clear(tmp_path: Path):
    from hydra_suite.detectkit.gui.dataset_recovery import (
        DatasetRecoveryError,
        clear_labels_with_recovery,
        latest_dataset_recovery,
        undo_latest_dataset_recovery,
    )

    project_dir = tmp_path / "project"
    project_dir.mkdir()
    _source_dir, _image, label = _source(project_dir)
    operation = clear_labels_with_recovery(project_dir, [label])
    label.write_text("0 0.9 0.9 0.1 0.1\n", encoding="utf-8")

    with pytest.raises(DatasetRecoveryError, match="changed since"):
        undo_latest_dataset_recovery(project_dir)

    assert label.read_text(encoding="utf-8") == "0 0.9 0.9 0.1 0.1\n"
    assert latest_dataset_recovery(project_dir).operation_id == operation.operation_id


def test_recovery_deduplicates_paths_before_mutating(tmp_path: Path):
    from hydra_suite.detectkit.gui.dataset_recovery import clear_labels_with_recovery

    project_dir = tmp_path / "project"
    project_dir.mkdir()
    _source_dir, _image, label = _source(project_dir)

    operation = clear_labels_with_recovery(project_dir, [label, label])

    assert operation.item_count == 1
    assert len(operation.entries) == 1


def test_project_owned_recovery_survives_project_folder_move(tmp_path: Path):
    from hydra_suite.detectkit.gui.dataset_recovery import (
        clear_labels_with_recovery,
        undo_latest_dataset_recovery,
    )

    project_dir = tmp_path / "project"
    project_dir.mkdir()
    _source_dir, _image, label = _source(project_dir)
    original = label.read_bytes()
    clear_labels_with_recovery(project_dir, [label])

    moved_project = tmp_path / "moved-project"
    shutil.move(project_dir, moved_project)
    moved_label = moved_project / "source" / "labels" / "train" / "frame.txt"

    undo_latest_dataset_recovery(moved_project)

    assert moved_label.read_bytes() == original


def test_recovery_operations_form_a_last_in_first_out_undo_stack(tmp_path: Path):
    from hydra_suite.detectkit.gui.dataset_recovery import (
        clear_labels_with_recovery,
        latest_dataset_recovery,
        remove_images_with_recovery,
        undo_latest_dataset_recovery,
    )

    project_dir = tmp_path / "project"
    project_dir.mkdir()
    source, image, label = _source(project_dir)
    original_label = label.read_bytes()
    first = clear_labels_with_recovery(project_dir, [label])
    second = remove_images_with_recovery(project_dir, source, [image])

    assert latest_dataset_recovery(project_dir).operation_id == second.operation_id
    undo_latest_dataset_recovery(project_dir)
    assert image.exists()
    assert label.read_bytes() == b""
    assert latest_dataset_recovery(project_dir).operation_id == first.operation_id

    undo_latest_dataset_recovery(project_dir)

    assert label.read_bytes() == original_label
    assert latest_dataset_recovery(project_dir) is None


def test_failed_rollback_keeps_stranded_image_available_to_undo(
    tmp_path: Path, monkeypatch
):
    import hydra_suite.detectkit.gui.dataset_recovery as recovery

    project_dir = tmp_path / "project"
    project_dir.mkdir()
    source, first_image, _first_label = _source(project_dir)
    second_image = source / "images" / "train" / "second.jpg"
    second_image.write_bytes(b"second-image")

    real_move = recovery.shutil.move
    move_calls = 0

    def _fail_operation_and_rollback(src, dst):
        nonlocal move_calls
        move_calls += 1
        if move_calls in {2, 3}:
            raise OSError("simulated move failure")
        return real_move(src, dst)

    monkeypatch.setattr(recovery.shutil, "move", _fail_operation_and_rollback)

    with pytest.raises(recovery.DatasetRecoveryError, match="Could not stage"):
        recovery.remove_images_with_recovery(
            project_dir, source, [first_image, second_image]
        )

    retained = recovery.latest_dataset_recovery(project_dir)
    assert retained is not None
    assert retained.item_count == 1
    assert not first_image.exists()
    assert second_image.read_bytes() == b"second-image"

    recovery.undo_latest_dataset_recovery(project_dir)

    assert first_image.read_bytes() == b"image-bytes"
