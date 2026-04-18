"""Verify ClassKit save paths produce v2 classifier artifacts consumable by ClassifierBackend."""

from __future__ import annotations

import pytest


def _build_tiny_state(n_classes: int = 3):
    from hydra_suite.training.tiny_model import _build_tiny_classifier_class

    TinyClassifier = _build_tiny_classifier_class()
    model = TinyClassifier(
        n_classes=n_classes, hidden_layers=1, hidden_dim=16, dropout=0.0
    )
    return model


def test_tiny_checkpoint_v2_flat_schema(tmp_path):
    """_save_tiny_checkpoint writes schema_version=2 and required v2 fields."""
    pytest.importorskip("cv2")
    pytest.importorskip("torch")
    import torch

    from hydra_suite.training.runner import _save_tiny_checkpoint

    model = _build_tiny_state(3)
    out_path = tmp_path / "tiny.pth"
    _save_tiny_checkpoint(
        model=model,
        save_path=str(out_path),
        class_names=["a", "b", "c"],
        input_size=(96, 64),  # H=96, W=64
        monochrome=True,
        hidden_layers=1,
        hidden_dim=16,
        dropout=0.0,
        best_val_acc=0.5,
        history={"loss": [1.0]},
    )

    ckpt = torch.load(str(out_path), map_location="cpu", weights_only=False)
    assert ckpt["schema_version"] == 2
    assert ckpt["arch"] == "tinyclassifier"
    assert ckpt["input_size"] == [96, 64]
    assert ckpt["factor_names"] == ["flat"]
    assert ckpt["class_names_per_factor"] == [["a", "b", "c"]]
    assert ckpt["class_names"] == ["a", "b", "c"]
    assert ckpt["monochrome"] is True
    assert ckpt["num_classes"] == 3


def test_tiny_checkpoint_v2_consumable_by_backend(tmp_path):
    """A v2 tiny flat checkpoint round-trips through ClassifierBackend cleanly."""
    pytest.importorskip("cv2")
    pytest.importorskip("torch")
    from hydra_suite.core.identity.classification.backend import ClassifierBackend
    from hydra_suite.training.runner import _save_tiny_checkpoint

    model = _build_tiny_state(3)
    out_path = tmp_path / "tiny.pth"
    _save_tiny_checkpoint(
        model=model,
        save_path=str(out_path),
        class_names=["left", "right", "unknown"],
        input_size=(64, 64),
        monochrome=False,
        hidden_layers=1,
        hidden_dim=16,
        dropout=0.0,
        best_val_acc=None,
        history={},
    )
    backend = ClassifierBackend(str(out_path))
    meta = backend.metadata
    assert meta.class_names_per_factor == [["left", "right", "unknown"]]
    assert meta.factor_names == ["flat"]
    assert meta.input_size == (64, 64)
    backend.close()
