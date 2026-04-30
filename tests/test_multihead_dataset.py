"""Unit tests for the multi-factor composite-folder dataset."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from hydra_suite.training.multihead_dataset import MultiFactorImageFolder


def _write_image(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = (np.random.rand(8, 8, 3) * 255).astype(np.uint8)
    Image.fromarray(arr).save(path)


def test_dataset_yields_per_factor_label_tuples(tmp_path):
    root = tmp_path / "train"
    for cls in ("red__blue", "red__green", "yellow__green"):
        for i in range(2):
            _write_image(root / cls / f"{cls}_{i}.png")

    tf = transforms.Compose([transforms.Resize((8, 8)), transforms.ToTensor()])
    ds = MultiFactorImageFolder(
        str(root),
        class_names_per_factor=[["red", "yellow"], ["blue", "green"]],
        delimiter="__",
        transform=tf,
    )
    assert len(ds) == 6
    img, labels = ds[0]
    assert img.shape == (3, 8, 8)
    assert isinstance(labels, torch.LongTensor)
    assert labels.shape == (2,)


def test_dataset_rejects_unknown_factor_label(tmp_path):
    root = tmp_path / "train"
    _write_image(root / "purple__blue" / "x.png")  # purple not declared
    tf = transforms.Compose([transforms.Resize((8, 8)), transforms.ToTensor()])
    try:
        MultiFactorImageFolder(
            str(root),
            class_names_per_factor=[["red", "yellow"], ["blue", "green"]],
            delimiter="__",
            transform=tf,
        )
    except ValueError as exc:
        assert "purple" in str(exc)
    else:
        raise AssertionError("expected ValueError for unknown factor label")


def test_dataset_rejects_wrong_factor_count(tmp_path):
    root = tmp_path / "train"
    _write_image(root / "red" / "x.png")  # only one factor in folder name
    tf = transforms.Compose([transforms.Resize((8, 8)), transforms.ToTensor()])
    try:
        MultiFactorImageFolder(
            str(root),
            class_names_per_factor=[["red", "yellow"], ["blue", "green"]],
            delimiter="__",
            transform=tf,
        )
    except ValueError as exc:
        assert "factor count" in str(exc).lower() or "expected 2" in str(exc).lower()
    else:
        raise AssertionError("expected ValueError for wrong factor count")
