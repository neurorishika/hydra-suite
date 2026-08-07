"""Round-trip test: checkpoint writers carry calibration artifact keys.

Task 3 of the Phase-2 calibration slice only plumbs three keys
(``calibration_temperature``, ``calibration_signature``, ``calibration_ece``)
through the two CNN-checkpoint save functions. The actual fit calls land in a
later task; here we just prove the keys survive a save/load round-trip with
their defaults (``None``) and with explicit values.
"""

import torch

from hydra_suite.training.runner import _save_tiny_checkpoint
from hydra_suite.training.torchvision_model import save_torchvision_checkpoint


def test_torchvision_checkpoint_carries_calibration(tmp_path):
    model = torch.nn.Linear(4, 3)
    p = tmp_path / "m.pth"
    save_torchvision_checkpoint(
        model=model,
        backbone="resnet18",
        class_names=["a", "b", "c"],
        factor_names=["flat"],
        input_size=(64, 64),
        best_val_acc=0.9,
        history=[],
        trainable_layers=0,
        backbone_lr_scale=1.0,
        monochrome=False,
        path=str(p),
        calibration_temperature=[1.7],
        calibration_signature="abc123",
        calibration_ece=[0.04],
    )
    ckpt = torch.load(str(p), map_location="cpu", weights_only=False)
    assert ckpt["calibration_temperature"] == [1.7]
    assert ckpt["calibration_signature"] == "abc123"
    assert ckpt["calibration_ece"] == [0.04]


def test_torchvision_checkpoint_calibration_defaults_none(tmp_path):
    model = torch.nn.Linear(4, 3)
    p = tmp_path / "m.pth"
    save_torchvision_checkpoint(
        model=model,
        backbone="resnet18",
        class_names=["a", "b", "c"],
        factor_names=["flat"],
        input_size=(64, 64),
        best_val_acc=0.9,
        history=[],
        trainable_layers=0,
        backbone_lr_scale=1.0,
        monochrome=False,
        path=str(p),
    )
    ckpt = torch.load(str(p), map_location="cpu", weights_only=False)
    assert ckpt["calibration_temperature"] is None
    assert ckpt["calibration_signature"] is None
    assert ckpt["calibration_ece"] is None


def test_tiny_checkpoint_carries_calibration(tmp_path):
    from hydra_suite.training.tiny_model import _build_tiny_classifier_class

    TinyClassifier = _build_tiny_classifier_class()
    model = TinyClassifier(n_classes=3, hidden_layers=1, hidden_dim=8, dropout=0.0)
    p = tmp_path / "tiny.pth"
    _save_tiny_checkpoint(
        model=model,
        save_path=str(p),
        class_names=["a", "b", "c"],
        input_size=(32, 32),
        monochrome=False,
        hidden_layers=1,
        hidden_dim=8,
        dropout=0.0,
        best_val_acc=0.8,
        history=[],
        calibration_temperature=[2.1],
        calibration_signature="def456",
        calibration_ece=[0.02],
    )
    ckpt = torch.load(str(p), map_location="cpu", weights_only=False)
    assert ckpt["calibration_temperature"] == [2.1]
    assert ckpt["calibration_signature"] == "def456"
    assert ckpt["calibration_ece"] == [0.02]


def test_tiny_checkpoint_calibration_defaults_none(tmp_path):
    from hydra_suite.training.tiny_model import _build_tiny_classifier_class

    TinyClassifier = _build_tiny_classifier_class()
    model = TinyClassifier(n_classes=3, hidden_layers=1, hidden_dim=8, dropout=0.0)
    p = tmp_path / "tiny.pth"
    _save_tiny_checkpoint(
        model=model,
        save_path=str(p),
        class_names=["a", "b", "c"],
        input_size=(32, 32),
        monochrome=False,
        hidden_layers=1,
        hidden_dim=8,
        dropout=0.0,
        best_val_acc=0.8,
        history=[],
    )
    ckpt = torch.load(str(p), map_location="cpu", weights_only=False)
    assert ckpt["calibration_temperature"] is None
    assert ckpt["calibration_signature"] is None
    assert ckpt["calibration_ece"] is None
