from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("PySide6")


def test_build_source_import_plan_for_yolo_obb_root(tmp_path: Path) -> None:
    from hydra_suite.classkit.core.data.source_import import build_source_import_plan

    source_root = tmp_path / "obb_dataset"
    train_images = source_root / "images" / "train"
    train_labels = source_root / "labels" / "train"
    train_images.mkdir(parents=True)
    train_labels.mkdir(parents=True)
    image_path = train_images / "frame001.jpg"
    image_path.write_bytes(b"image-bytes")
    (train_labels / "frame001.txt").write_text(
        "0 0.1 0.1 0.2 0.1 0.2 0.2 0.1 0.2\n",
        encoding="utf-8",
    )
    (source_root / "dataset.yaml").write_text(
        "train: images/train\nnames:\n  0: ant\n",
        encoding="utf-8",
    )

    plan = build_source_import_plan(source_root)

    assert plan.source_kind == "yolo_obb"
    assert plan.image_paths == [image_path.resolve()]
    assert plan.discovered_labels == ["ant"]
    assert plan.label_updates[str(image_path.resolve())] == ("ant", 1.0)
    assert plan.metadata_by_path[str(image_path.resolve())]["source_kind"] == "yolo_obb"


def test_ingest_worker_imports_coco_labels_into_db(tmp_path: Path) -> None:
    from hydra_suite.classkit.core.store.db import ClassKitDB
    from hydra_suite.classkit.jobs.task_workers import IngestWorker

    source_root = tmp_path / "coco_dataset"
    images_dir = source_root / "images"
    images_dir.mkdir(parents=True)
    image_path = images_dir / "frame001.jpg"
    image_path.write_bytes(b"image-bytes")
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

    db_path = tmp_path / "classkit.db"
    ClassKitDB(db_path)

    worker = IngestWorker(source_root, db_path, project_classes=["ant"])
    worker.run()

    db = ClassKitDB(db_path)
    assert db.count_images() == 1
    assert db.get_all_labels() == ["ant"]

    stored_paths = db.get_all_image_paths()
    assert len(stored_paths) == 1
    stored_path = Path(stored_paths[0])
    assert stored_path.exists() is True
    assert source_root not in stored_path.parents

    status = db.get_label_review_status_by_path()[str(stored_path)]
    assert status["label"] == "ant"
    assert status["label_source"] == "import"
    assert status["verified"] is True

    sources = db.get_source_folders()
    assert sources == [{"folder": str(source_root.resolve()), "count": 1}]


def test_build_source_import_plan_for_train_val_class_folders(tmp_path: Path) -> None:
    from hydra_suite.classkit.core.data.source_import import build_source_import_plan

    source_root = tmp_path / "folder_labels"
    ant_train = source_root / "train" / "ant"
    bee_val = source_root / "val" / "bee"
    ant_train.mkdir(parents=True)
    bee_val.mkdir(parents=True)
    ant_image = ant_train / "frame001.jpg"
    bee_image = bee_val / "frame002.png"
    ant_image.write_bytes(b"ant-image")
    bee_image.write_bytes(b"bee-image")

    plan = build_source_import_plan(source_root)

    assert plan.source_kind == "class_folders"
    assert plan.discovered_labels == ["ant", "bee"]
    assert plan.label_updates[str(ant_image.resolve())] == ("ant", 1.0)
    assert plan.label_updates[str(bee_image.resolve())] == ("bee", 1.0)
    assert plan.metadata_by_path[str(ant_image.resolve())]["split"] == "train"
    assert plan.metadata_by_path[str(bee_image.resolve())]["split"] == "val"


def test_ingest_worker_can_import_images_without_mismatched_labels(
    tmp_path: Path,
) -> None:
    from hydra_suite.classkit.core.store.db import ClassKitDB
    from hydra_suite.classkit.jobs.task_workers import IngestWorker

    source_root = tmp_path / "folder_labels"
    ant_train = source_root / "train" / "ant"
    ant_train.mkdir(parents=True)
    image_path = ant_train / "frame001.jpg"
    image_path.write_bytes(b"ant-image")

    db_path = tmp_path / "classkit.db"
    ClassKitDB(db_path)

    worker = IngestWorker(
        source_root,
        db_path,
        project_classes=["bee"],
        import_labels=False,
    )
    worker.run()

    db = ClassKitDB(db_path)
    assert db.count_images() == 1
    assert db.get_all_labels() == [None]
    stored_path = Path(db.get_all_image_paths()[0])
    assert stored_path.exists() is True
    assert source_root not in stored_path.parents
