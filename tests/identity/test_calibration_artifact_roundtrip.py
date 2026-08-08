"""Round-trip test: checkpoint writers carry calibration artifact keys.

Task 3 of the Phase-2 calibration slice only plumbs three keys
(``calibration_temperature``, ``calibration_signature``, ``calibration_ece``)
through the two CNN-checkpoint save functions. The actual fit calls land in a
later task; here we just prove the keys survive a save/load round-trip with
their defaults (``None``) and with explicit values.

Task 5 extends this: for each artifact form (tiny .pth, torchvision .pth,
YOLO .v2meta.json sidecar, .multihead.json manifest) prove the same three
keys survive a full ``ClassifierBackend(path).metadata`` parse, plus the
``calibration_status`` transitions.
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


def test_torchvision_metadata_surfaces_calibration(tmp_path):
    """ClassifierBackend.metadata parses calibration_* off a torchvision .pth."""
    from hydra_suite.core.individual.classification.backend import ClassifierBackend
    from hydra_suite.runtime.resolver import ResolvedBackend

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
    backend = ClassifierBackend(str(p), ResolvedBackend("torch", "cpu", False))
    try:
        meta = backend.metadata
    finally:
        backend.close()
    assert meta.calibration_temperature == (1.7,)
    assert meta.calibration_signature == "abc123"
    assert meta.calibration_ece == (0.04,)


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


def test_tiny_metadata_surfaces_calibration(tmp_path):
    """ClassifierBackend.metadata parses calibration_* off a tiny .pth."""
    from hydra_suite.core.individual.classification.backend import ClassifierBackend
    from hydra_suite.runtime.resolver import ResolvedBackend
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
    backend = ClassifierBackend(str(p), ResolvedBackend("torch", "cpu", False))
    try:
        meta = backend.metadata
    finally:
        backend.close()
    assert meta.calibration_temperature == (2.1,)
    assert meta.calibration_signature == "def456"
    assert meta.calibration_ece == (0.02,)


def test_yolo_sidecar_surfaces_calibration(tmp_path):
    """ClassifierBackend parses calibration_* off a .v2meta.json sidecar.

    Exercised at the sidecar-loader level (``_YoloFlatLoader.parse_metadata``)
    rather than via a real YOLO .pt checkpoint, since instantiating an actual
    ultralytics YOLO model is expensive and unnecessary to prove the sidecar
    plumbing.
    """
    import json

    from hydra_suite.core.individual.classification.backend import _YoloFlatLoader

    pt_path = tmp_path / "flat.pt"
    pt_path.write_bytes(b"")  # loader only needs the sidecar to exist alongside
    sidecar_path = pt_path.with_suffix(".v2meta.json")
    sidecar_path.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "arch": "yolo",
                "factor_names": ["flat"],
                "class_names_per_factor": [["a", "b", "c"]],
                "input_size": [224, 224],
                "monochrome": False,
                "calibration_temperature": [1.3],
                "calibration_signature": "yolo123",
                "calibration_ece": [0.05],
            }
        ),
        encoding="utf-8",
    )
    meta = _YoloFlatLoader.parse_metadata(str(pt_path))
    assert meta.calibration_temperature == (1.3,)
    assert meta.calibration_signature == "yolo123"
    assert meta.calibration_ece == (0.05,)


def test_multihead_manifest_roundtrips_calibration(tmp_path):
    """write_classifier_multihead_manifest carries calibration_* into the
    payload, and ClassifierBackend parses it back out."""
    from hydra_suite.core.individual.classification.backend import ClassifierBackend
    from hydra_suite.runtime.resolver import ResolvedBackend
    from hydra_suite.training.model_publish import write_classifier_multihead_manifest

    model_a = torch.nn.Linear(4, 2)
    model_b = torch.nn.Linear(4, 2)
    p_a = tmp_path / "factor_a.pth"
    p_b = tmp_path / "factor_b.pth"
    save_torchvision_checkpoint(
        model=model_a,
        backbone="resnet18",
        class_names=["x", "y"],
        factor_names=["flat"],
        input_size=(64, 64),
        best_val_acc=0.9,
        history=[],
        trainable_layers=0,
        backbone_lr_scale=1.0,
        monochrome=False,
        path=str(p_a),
    )
    save_torchvision_checkpoint(
        model=model_b,
        backbone="resnet18",
        class_names=["p", "q"],
        factor_names=["flat"],
        input_size=(64, 64),
        best_val_acc=0.9,
        history=[],
        trainable_layers=0,
        backbone_lr_scale=1.0,
        monochrome=False,
        path=str(p_b),
    )
    manifest_path = tmp_path / "bundle.multihead.json"
    write_classifier_multihead_manifest(
        manifest_path,
        factor_entries=[
            {"factor": "color", "path": p_a, "class_names": ["x", "y"]},
            {"factor": "size", "path": p_b, "class_names": ["p", "q"]},
        ],
        input_size=(64, 64),
        monochrome=False,
        calibration_temperature=[1.1, 1.2],
        calibration_signature="bundleSig",
        calibration_ece=[0.01, 0.02],
    )
    backend = ClassifierBackend(
        str(manifest_path), ResolvedBackend("torch", "cpu", False)
    )
    try:
        meta = backend.metadata
    finally:
        backend.close()
    assert meta.calibration_temperature == (1.1, 1.2)
    assert meta.calibration_signature == "bundleSig"
    assert meta.calibration_ece == (0.01, 0.02)


def test_calibration_status_transitions():
    from hydra_suite.core.individual.classification.backend import (
        ClassifierMetadata,
        calibration_status,
    )

    base = dict(
        arch="resnet18",
        input_size=(64, 64),
        is_multihead=False,
        factor_names=["flat"],
        class_names_per_factor=[["a", "b"]],
        monochrome=False,
        recommended_confidence_threshold=None,
        source_path="x",
    )
    uncal = ClassifierMetadata(
        **base,
        calibration_temperature=None,
        calibration_signature=None,
        calibration_ece=None,
    )
    cal = ClassifierMetadata(
        **base,
        calibration_temperature=(1.5,),
        calibration_signature="sigA",
        calibration_ece=(0.03,),
    )
    assert calibration_status(uncal, "sigA") == "uncalibrated"
    assert calibration_status(cal, "sigA") == "calibrated"
    assert calibration_status(cal, "sigB") == "stale"
