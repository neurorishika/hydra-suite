"""Adversarial caps for dataset discovery and metadata."""

from __future__ import annotations

import pytest

from hydra_suite.training.class_mapping import read_classes_txt
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
