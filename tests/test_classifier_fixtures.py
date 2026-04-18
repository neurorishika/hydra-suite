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
