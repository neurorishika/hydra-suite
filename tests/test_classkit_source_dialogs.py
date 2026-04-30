"""Tests for ClassKit source-folder selection dialogs."""

from __future__ import annotations

import gc
import os
import sys
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytest.importorskip("PySide6")

from PySide6.QtWidgets import QApplication, QMessageBox  # noqa: E402

from hydra_suite.classkit.gui.dialogs.add_source import AddSourceDialog  # noqa: E402
from hydra_suite.classkit.gui.dialogs.source_manager import (  # noqa: E402
    SourceManagerDialog,
)
from hydra_suite.widgets.dialogs import (  # noqa: E402
    HYDRA_DIALOG_MUTED_TEXT_COLOR,
    HYDRA_DIALOG_STYLE,
)


@pytest.fixture()
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    return app


@pytest.fixture(autouse=True)
def cleanup_qt_widgets(qapp):
    yield
    for widget in list(qapp.topLevelWidgets()):
        widget.close()
        widget.deleteLater()
    qapp.processEvents()
    gc.collect()


def _capture_warning(
    monkeypatch: pytest.MonkeyPatch, module_path: str
) -> list[tuple[str, str]]:
    warnings: list[tuple[str, str]] = []

    def _warning(_parent, title: str, text: str, *args, **kwargs):
        warnings.append((title, text))
        return QMessageBox.Ok

    monkeypatch.setattr(f"{module_path}.QMessageBox.warning", _warning)
    return warnings


def test_add_source_dialog_rejects_folder_without_images_subdir(
    qapp, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    source_root = tmp_path / "dataset_root"
    source_root.mkdir()

    monkeypatch.setattr(
        "hydra_suite.classkit.gui.dialogs.add_source.QFileDialog.getExistingDirectory",
        lambda *_args, **_kwargs: str(source_root),
    )
    warnings = _capture_warning(
        monkeypatch, "hydra_suite.classkit.gui.dialogs.add_source"
    )

    dialog = AddSourceDialog()
    dialog._browse()

    assert dialog.sources == []
    assert warnings == [
        (
            "Invalid Source Folder",
            "Selected source folder must contain compatible .jpg / .jpeg / .png files, "
            "an images/ subdirectory, a supported COCO or YOLO OBB/detect dataset root, "
            "or a train/val class-folder dataset root.\n\n"
            f"{source_root}",
        )
    ]


def test_add_source_dialog_uses_images_subdir_when_present(
    qapp, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    source_root = tmp_path / "dataset_root"
    images_dir = source_root / "images"
    images_dir.mkdir(parents=True)
    (images_dir / "frame001.jpg").write_bytes(b"not-an-image")

    monkeypatch.setattr(
        "hydra_suite.classkit.gui.dialogs.add_source.QFileDialog.getExistingDirectory",
        lambda *_args, **_kwargs: str(source_root),
    )

    dialog = AddSourceDialog()
    dialog._browse()

    assert dialog.sources == [
        (source_root.resolve(), source_root.resolve(), source_root.name)
    ]


def test_add_source_dialog_standardizes_flat_image_folder_when_user_accepts(
    qapp, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    source_root = tmp_path / "dataset_root"
    source_root.mkdir()
    image_path = source_root / "frame001.jpg"
    image_path.write_bytes(b"not-an-image")

    monkeypatch.setattr(
        "hydra_suite.classkit.gui.dialogs.add_source.QFileDialog.getExistingDirectory",
        lambda *_args, **_kwargs: str(source_root),
    )
    dialog = AddSourceDialog()
    dialog._browse()

    assert dialog.sources == [
        (source_root.resolve(), source_root.resolve(), source_root.name)
    ]
    assert (source_root / "images").exists() is False


def test_add_source_dialog_accepts_train_val_class_folder_dataset(
    qapp, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    source_root = tmp_path / "dataset_root"
    ant_train = source_root / "train" / "ant"
    bee_val = source_root / "val" / "bee"
    ant_train.mkdir(parents=True)
    bee_val.mkdir(parents=True)
    (ant_train / "frame001.jpg").write_bytes(b"ant-image")
    (bee_val / "frame002.png").write_bytes(b"bee-image")

    monkeypatch.setattr(
        "hydra_suite.classkit.gui.dialogs.add_source.QFileDialog.getExistingDirectory",
        lambda *_args, **_kwargs: str(source_root),
    )

    dialog = AddSourceDialog()
    dialog._browse()

    assert dialog.sources == [
        (source_root.resolve(), source_root.resolve(), source_root.name)
    ]


def test_source_manager_dialog_rejects_folder_without_images_subdir(
    qapp, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    source_root = tmp_path / "dataset_root"
    source_root.mkdir()

    monkeypatch.setattr(
        SourceManagerDialog,
        "_load_existing_sources",
        lambda self: None,
    )
    monkeypatch.setattr(
        "hydra_suite.classkit.gui.dialogs.source_manager.QFileDialog.getExistingDirectory",
        lambda *_args, **_kwargs: str(source_root),
    )
    warnings = _capture_warning(
        monkeypatch, "hydra_suite.classkit.gui.dialogs.source_manager"
    )

    dialog = SourceManagerDialog(db_path=tmp_path / "classkit.db")
    dialog._browse_add()

    assert dialog.folders_to_add == []
    assert warnings == [
        (
            "Invalid Source Folder",
            "Selected source folder must contain compatible .jpg / .jpeg / .png files, "
            "an images/ subdirectory, a supported COCO or YOLO OBB/detect dataset root, "
            "or a train/val class-folder dataset root.\n\n"
            f"{source_root}",
        )
    ]


def test_source_manager_dialog_accepts_coco_dataset_root(
    qapp, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    source_root = tmp_path / "coco_dataset"
    images_dir = source_root / "images"
    images_dir.mkdir(parents=True)
    (images_dir / "frame001.jpg").write_bytes(b"image-bytes")
    (source_root / "annotations.json").write_text(
        """
        {
          "images": [{"id": 1, "file_name": "frame001.jpg"}],
          "annotations": [{"id": 1, "image_id": 1, "category_id": 0}],
          "categories": [{"id": 0, "name": "ant"}]
        }
        """.strip(),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        SourceManagerDialog, "_load_existing_sources", lambda self: None
    )
    monkeypatch.setattr(
        "hydra_suite.classkit.gui.dialogs.source_manager.QFileDialog.getExistingDirectory",
        lambda *_args, **_kwargs: str(source_root),
    )

    dialog = SourceManagerDialog(db_path=tmp_path / "classkit.db")
    dialog._browse_add()

    assert dialog.folders_to_add == [source_root.resolve()]


def test_source_manager_dialog_accepts_yolo_obb_dataset_root(
    qapp, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    source_root = tmp_path / "obb_dataset"
    train_images = source_root / "images" / "train"
    train_labels = source_root / "labels" / "train"
    train_images.mkdir(parents=True)
    train_labels.mkdir(parents=True)
    (train_images / "frame001.jpg").write_bytes(b"image-bytes")
    (train_labels / "frame001.txt").write_text(
        "0 0.1 0.1 0.2 0.1 0.2 0.2 0.1 0.2\n",
        encoding="utf-8",
    )
    (source_root / "dataset.yaml").write_text(
        "train: images/train\nnames:\n  0: ant\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(
        SourceManagerDialog, "_load_existing_sources", lambda self: None
    )
    monkeypatch.setattr(
        "hydra_suite.classkit.gui.dialogs.source_manager.QFileDialog.getExistingDirectory",
        lambda *_args, **_kwargs: str(source_root),
    )

    dialog = SourceManagerDialog(db_path=tmp_path / "classkit.db")
    dialog._browse_add()

    assert dialog.folders_to_add == [source_root.resolve()]


def test_source_manager_dialog_standardizes_flat_image_folder_when_user_accepts(
    qapp, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    source_root = tmp_path / "dataset_root"
    source_root.mkdir()
    image_path = source_root / "frame001.jpg"
    image_path.write_bytes(b"not-an-image")

    monkeypatch.setattr(
        SourceManagerDialog, "_load_existing_sources", lambda self: None
    )
    monkeypatch.setattr(
        "hydra_suite.classkit.gui.dialogs.source_manager.QFileDialog.getExistingDirectory",
        lambda *_args, **_kwargs: str(source_root),
    )
    dialog = SourceManagerDialog(db_path=tmp_path / "classkit.db")
    dialog._browse_add()

    assert dialog.folders_to_add == [source_root.resolve()]
    assert (source_root / "images").exists() is False


def test_source_manager_dialog_accepts_train_val_class_folder_dataset(
    qapp, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    source_root = tmp_path / "dataset_root"
    ant_train = source_root / "train" / "ant"
    bee_val = source_root / "val" / "bee"
    ant_train.mkdir(parents=True)
    bee_val.mkdir(parents=True)
    (ant_train / "frame001.jpg").write_bytes(b"ant-image")
    (bee_val / "frame002.png").write_bytes(b"bee-image")

    monkeypatch.setattr(
        SourceManagerDialog, "_load_existing_sources", lambda self: None
    )
    monkeypatch.setattr(
        "hydra_suite.classkit.gui.dialogs.source_manager.QFileDialog.getExistingDirectory",
        lambda *_args, **_kwargs: str(source_root),
    )

    dialog = SourceManagerDialog(db_path=tmp_path / "classkit.db")
    dialog._browse_add()

    assert dialog.folders_to_add == [source_root.resolve()]


def test_source_dialogs_use_shared_hydra_theme(
    qapp, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr(
        SourceManagerDialog, "_load_existing_sources", lambda self: None
    )

    add_dialog = AddSourceDialog()
    manager_dialog = SourceManagerDialog(db_path=tmp_path / "classkit.db")

    assert add_dialog.styleSheet() == HYDRA_DIALOG_STYLE
    assert manager_dialog.styleSheet() == HYDRA_DIALOG_STYLE
    assert HYDRA_DIALOG_MUTED_TEXT_COLOR in manager_dialog._summary.styleSheet()
