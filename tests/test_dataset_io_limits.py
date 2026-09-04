"""Adversarial caps for dataset discovery and metadata."""

from __future__ import annotations

import random
from pathlib import Path

import pytest

from hydra_suite.training.class_mapping import read_classes_txt
from hydra_suite.training.dataset_inspector import (
    DatasetInspection,
    DatasetItem,
    DatasetItemStore,
    inspect_obb_or_detect_dataset,
    split_items_for_training,
)
from hydra_suite.training.dataset_io import (
    DatasetIOLimits,
    DatasetLimitError,
    iter_bounded_text_lines,
    sorted_file_index,
)


def test_recursive_discovery_caps_depth_before_index_escape(tmp_path):
    root = tmp_path / "images"
    nested = root
    for index in range(4):
        nested = nested / str(index)
        nested.mkdir(parents=True)
    (nested / "image.jpg").write_bytes(b"image")
    with pytest.raises(DatasetLimitError, match="depth"):
        with sorted_file_index(
            root, suffixes={".jpg"}, limits=DatasetIOLimits(max_depth=2)
        ):
            pass


def test_recursive_discovery_caps_aggregate_path_index_bytes(tmp_path):
    root = tmp_path / "images"
    root.mkdir()
    (root / "first.jpg").write_bytes(b"image")
    (root / "second.jpg").write_bytes(b"image")
    with pytest.raises(DatasetLimitError, match="pathname index"):
        with sorted_file_index(
            root,
            suffixes={".jpg"},
            limits=DatasetIOLimits(max_path_index_bytes=12),
        ):
            pass


def test_bounded_line_reader_rejects_oversized_line(tmp_path):
    label = tmp_path / "label.txt"
    label.write_text("x" * 65, encoding="utf-8")
    with pytest.raises(DatasetLimitError, match="Line 1"):
        list(iter_bounded_text_lines(label, limits=DatasetIOLimits(max_line_bytes=64)))


def test_class_file_cardinality_is_explicitly_bounded(tmp_path):
    (tmp_path / "classes.txt").write_text(
        "\n".join(f"class-{index}" for index in range(4097)), encoding="utf-8"
    )
    with pytest.raises(DatasetLimitError, match="4096"):
        read_classes_txt(tmp_path)


def test_inspector_keeps_discovered_paths_in_disk_backed_sequence(tmp_path):
    images = tmp_path / "images"
    labels = tmp_path / "labels"
    images.mkdir()
    labels.mkdir()
    for index in range(3):
        (images / f"frame-{index}.jpg").write_bytes(b"image")
        (labels / f"frame-{index}.txt").write_text(
            "0 0.5 0.5 0.2 0.2\n", encoding="utf-8"
        )
    inspection = inspect_obb_or_detect_dataset(tmp_path)
    assert isinstance(inspection.splits["all"], DatasetItemStore)
    assert [Path(item.image_path).name for item in inspection.splits["all"]] == [
        "frame-0.jpg",
        "frame-1.jpg",
        "frame-2.jpg",
    ]


def test_disk_backed_indexes_use_admitted_filesystem(tmp_path, monkeypatch):
    index_root = tmp_path / "admitted-indexes"
    monkeypatch.setenv("HYDRA_DATASET_INDEX_DIR", str(index_root))
    images = tmp_path / "images"
    images.mkdir()
    (images / "frame.jpg").write_bytes(b"image")

    with sorted_file_index(images, suffixes={".jpg"}):
        live_indexes = list(index_root.glob("*.sqlite3"))
        assert len(live_indexes) == 1

    assert not list(index_root.glob("*.sqlite3"))


def test_disk_backed_split_matches_legacy_shuffle_and_boundaries():
    items = [DatasetItem(str(index), str(index), "all") for index in range(20)]
    expected = list(items)
    random.Random(17).shuffle(expected)

    split = split_items_for_training(
        DatasetInspection(root_dir="fixture", splits={"all": items}),
        (0.6, 0.2, 0.2),
        seed=17,
    )

    actual = [*split["train"], *split["val"], *split["test"]]
    assert [item.image_path for item in actual] == [
        item.image_path for item in expected
    ]
    assert [len(split[name]) for name in ("train", "val", "test")] == [12, 4, 4]
    assert all(item.split == name for name, values in split.items() for item in values)
