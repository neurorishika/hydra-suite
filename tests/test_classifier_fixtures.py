"""Test fixtures — build minimal classifier artifacts for backend tests.

Artifacts are cached under pytest's tmp_path_factory fixture so the first run
pays the build cost and subsequent runs reuse the files.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest


@pytest.fixture(scope="session")
def fixtures_dir(tmp_path_factory) -> Path:
    return tmp_path_factory.mktemp("classifier_fixtures", numbered=False)


@pytest.fixture(scope="session")
def tiny_flat_headtail(fixtures_dir: Path) -> Path:
    """TinyClassifier v2 checkpoint with head-tail labels."""
    import torch

    path = fixtures_dir / "tiny_flat_headtail.pth"
    if path.exists():
        return path
    from hydra_suite.training.tiny_model import _build_tiny_classifier_class

    TinyClassifier = _build_tiny_classifier_class()
    model = TinyClassifier(n_classes=5, hidden_layers=1, hidden_dim=32, dropout=0.1)
    ckpt: dict[str, Any] = {
        "schema_version": 2,
        "arch": "tinyclassifier",
        "input_size": [64, 64],
        "factor_names": ["flat"],
        "class_names_per_factor": [["up", "down", "left", "right", "unknown"]],
        "class_names": ["up", "down", "left", "right", "unknown"],
        "num_classes": 5,
        "monochrome": False,
        "model_state_dict": model.state_dict(),
        "hidden_layers": 1,
        "hidden_dim": 32,
        "dropout": 0.1,
    }
    torch.save(ckpt, str(path))
    return path


@pytest.fixture(scope="session")
def tiny_flat_subset(fixtures_dir: Path) -> Path:
    """TinyClassifier v2 checkpoint with only {left, right}."""
    import torch

    path = fixtures_dir / "tiny_flat_subset.pth"
    if path.exists():
        return path
    from hydra_suite.training.tiny_model import _build_tiny_classifier_class

    TinyClassifier = _build_tiny_classifier_class()
    model = TinyClassifier(n_classes=2, hidden_layers=1, hidden_dim=32, dropout=0.0)
    ckpt: dict[str, Any] = {
        "schema_version": 2,
        "arch": "tinyclassifier",
        "input_size": [64, 64],
        "factor_names": ["flat"],
        "class_names_per_factor": [["left", "right"]],
        "class_names": ["left", "right"],
        "num_classes": 2,
        "monochrome": False,
        "model_state_dict": model.state_dict(),
        "hidden_layers": 1,
        "hidden_dim": 32,
        "dropout": 0.0,
    }
    torch.save(ckpt, str(path))
    return path


@pytest.fixture(scope="session")
def tiny_flat_nonsquare(fixtures_dir: Path) -> Path:
    """TinyClassifier v2 with non-square input size to catch [H, W] serialization bugs."""
    import torch

    path = fixtures_dir / "tiny_flat_nonsquare.pth"
    if path.exists():
        return path
    from hydra_suite.training.tiny_model import _build_tiny_classifier_class

    TinyClassifier = _build_tiny_classifier_class()
    model = TinyClassifier(n_classes=3, hidden_layers=1, hidden_dim=16, dropout=0.0)
    ckpt: dict[str, Any] = {
        "schema_version": 2,
        "arch": "tinyclassifier",
        "input_size": [256, 192],  # H=256, W=192
        "factor_names": ["flat"],
        "class_names_per_factor": [["a", "b", "c"]],
        "class_names": ["a", "b", "c"],
        "num_classes": 3,
        "monochrome": False,
        "model_state_dict": model.state_dict(),
        "hidden_layers": 1,
        "hidden_dim": 16,
        "dropout": 0.0,
    }
    torch.save(ckpt, str(path))
    return path


@pytest.fixture(scope="session")
def yolo_flat_headtail(fixtures_dir: Path) -> Path:
    """YOLO classify flat model mapping 5 head-tail classes.

    Uses the smallest available ultralytics model. The classifier head is
    replaced with a 5-output Linear layer so that inference genuinely emits
    5-class probability vectors — we skip actual training since only the
    shape/metadata contracts are under test.
    """
    path = fixtures_dir / "yolo_flat_headtail.pt"
    if path.exists():
        return path
    pytest.importorskip("ultralytics")
    import torch
    from ultralytics import YOLO

    model = YOLO("yolov8n-cls.pt")  # pretrained download on first run
    # Replace the final Linear layer so inference outputs 5 classes, not 1000
    old_linear = model.model.model[9].linear
    model.model.model[9].linear = torch.nn.Linear(old_linear.in_features, 5)
    model.model.names = {0: "up", 1: "down", 2: "left", 3: "right", 4: "unknown"}
    model.save(str(path))
    return path


@pytest.fixture(scope="session")
def torchvision_flat_identity(fixtures_dir: Path) -> Path:
    """resnet18 v2 flat checkpoint with 3 identity classes."""
    path = fixtures_dir / "torchvision_flat_identity.pth"
    if path.exists():
        return path
    from hydra_suite.training.torchvision_model import (
        build_torchvision_classifier,
        save_torchvision_checkpoint,
    )

    model = build_torchvision_classifier("resnet18", num_classes=3, trainable_layers=-1)
    save_torchvision_checkpoint(
        model=model,
        backbone="resnet18",
        class_names=["antA", "antB", "antC"],
        factor_names=["flat"],
        input_size=(64, 64),
        best_val_acc=None,
        history={},
        trainable_layers=-1,
        backbone_lr_scale=1.0,
        monochrome=False,
        path=str(path),
    )
    return path
