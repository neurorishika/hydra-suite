"""Tests for DetectKit external-source standardization."""

from __future__ import annotations

import json
from pathlib import Path

from hydra_suite.detectkit.gui.source_import import (
    IMPORT_MODE_LINKED,
    inspect_detectkit_source,
    materialize_detectkit_source,
)


def _write_fake_image(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"fake-image")


def test_inspect_detectkit_source_accepts_existing_canonical_root(tmp_path: Path):
    (tmp_path / "images").mkdir()
    (tmp_path / "labels").mkdir()
    (tmp_path / "classes.txt").write_text("ant\n", encoding="utf-8")

    inspection = inspect_detectkit_source(tmp_path)

    assert inspection.source_kind == "detectkit"
    assert inspection.requires_import is False
    assert inspection.discovered_labels == ["ant"]


def test_materialize_detectkit_source_converts_yolo_detect_boxes(tmp_path: Path):
    source_root = tmp_path / "yolo_detect"
    (source_root / "images").mkdir(parents=True)
    (source_root / "labels").mkdir(parents=True)
    _write_fake_image(source_root / "images" / "frame001.jpg")
    (source_root / "labels" / "frame001.txt").write_text(
        "0 0.5 0.5 0.4 0.2\n",
        encoding="utf-8",
    )
    (source_root / "dataset.yaml").write_text(
        "train: images\nnames:\n  0: ant\n",
        encoding="utf-8",
    )

    project_dir = tmp_path / "project"
    materialized = materialize_detectkit_source(source_root, project_dir)

    assert materialized.imported is True
    assert materialized.source_kind == "yolo_detect"
    assert materialized.source_root == source_root.resolve()

    classes_txt = (materialized.canonical_path / "classes.txt").read_text(
        encoding="utf-8"
    )
    label_text = (materialized.canonical_path / "labels" / "frame001.txt").read_text(
        encoding="utf-8"
    )
    fields = label_text.strip().split()

    assert classes_txt == "ant\n"
    assert len(fields) == 9
    assert fields[0] == "0"


def test_materialize_detectkit_source_converts_coco_bbox_annotations(tmp_path: Path):
    source_root = tmp_path / "coco"
    _write_fake_image(source_root / "images" / "sample.jpg")
    (source_root / "annotations").mkdir(parents=True, exist_ok=True)
    (source_root / "annotations" / "instances.json").write_text(
        json.dumps(
            {
                "images": [
                    {
                        "id": 7,
                        "file_name": "sample.jpg",
                        "width": 100,
                        "height": 50,
                    }
                ],
                "annotations": [
                    {
                        "id": 1,
                        "image_id": 7,
                        "category_id": 5,
                        "bbox": [10, 5, 40, 20],
                    }
                ],
                "categories": [{"id": 5, "name": "ant"}],
            }
        ),
        encoding="utf-8",
    )

    materialized = materialize_detectkit_source(source_root, tmp_path / "project")

    assert materialized.imported is True
    assert materialized.source_kind == "coco"
    assert (materialized.canonical_path / "images" / "sample.jpg").exists()
    assert (materialized.canonical_path / "classes.txt").read_text(
        encoding="utf-8"
    ) == "ant\n"

    fields = (
        (materialized.canonical_path / "labels" / "sample.txt")
        .read_text(encoding="utf-8")
        .strip()
        .split()
    )
    assert len(fields) == 9
    assert fields[0] == "0"


def test_materialize_detectkit_source_can_link_and_normalize_in_place(tmp_path: Path):
    source_root = tmp_path / "linked_yolo_detect"
    (source_root / "images").mkdir(parents=True)
    (source_root / "labels").mkdir(parents=True)
    _write_fake_image(source_root / "images" / "frame001.jpg")
    (source_root / "labels" / "frame001.txt").write_text(
        "0 0.5 0.5 0.4 0.2\n",
        encoding="utf-8",
    )
    (source_root / "dataset.yaml").write_text(
        "train: images\nnames:\n  0: ant\n",
        encoding="utf-8",
    )

    materialized = materialize_detectkit_source(
        source_root,
        tmp_path / "project",
        import_mode=IMPORT_MODE_LINKED,
    )

    assert materialized.imported is False
    assert materialized.canonical_path == source_root.resolve()
    assert (source_root / "classes.txt").read_text(encoding="utf-8") == "ant\n"
    fields = (
        (source_root / "labels" / "frame001.txt")
        .read_text(encoding="utf-8")
        .strip()
        .split()
    )
    assert len(fields) == 9
    assert fields[0] == "0"
