"""Verify ClassKit save paths produce v2 classifier artifacts consumable by ClassifierBackend."""

from __future__ import annotations

from pathlib import Path

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


def test_torchvision_checkpoint_v2_flat_schema(tmp_path):
    pytest.importorskip("cv2")
    pytest.importorskip("torch")
    import torch

    from hydra_suite.training.torchvision_model import (
        build_torchvision_classifier,
        save_torchvision_checkpoint,
    )

    model = build_torchvision_classifier(
        "tinyclassifier", num_classes=4, trainable_layers=-1
    )
    out_path = tmp_path / "tv_flat.pth"
    save_torchvision_checkpoint(
        model=model,
        backbone="tinyclassifier",
        class_names=["a", "b", "c", "d"],
        factor_names=["flat"],
        input_size=(128, 96),
        best_val_acc=0.75,
        history={},
        trainable_layers=-1,
        backbone_lr_scale=1.0,
        monochrome=True,
        path=str(out_path),
    )

    ckpt = torch.load(str(out_path), map_location="cpu", weights_only=False)
    assert ckpt["schema_version"] == 2
    assert ckpt["input_size"] == [128, 96]
    assert ckpt["factor_names"] == ["flat"]
    assert ckpt["class_names_per_factor"] == [["a", "b", "c", "d"]]
    assert ckpt["class_names"] == ["a", "b", "c", "d"]
    assert ckpt["monochrome"] is True


def test_torchvision_checkpoint_v2_multihead_schema(tmp_path):
    """Multi-head torchvision writes class_names_per_factor and omits flat class_names."""
    pytest.importorskip("cv2")
    pytest.importorskip("torch")
    import torch

    from hydra_suite.training.torchvision_model import (
        build_torchvision_classifier,
        save_torchvision_checkpoint,
    )

    model = build_torchvision_classifier(
        "tinyclassifier", num_classes=5, trainable_layers=-1
    )
    out_path = tmp_path / "tv_multi.pth"
    save_torchvision_checkpoint(
        model=model,
        backbone="tinyclassifier",
        class_names=[],  # ignored in multi-head path
        class_names_per_factor=[["red", "green", "blue"], ["square", "circle"]],
        factor_names=["color", "shape"],
        input_size=(224, 224),
        best_val_acc=None,
        history={},
        trainable_layers=-1,
        backbone_lr_scale=1.0,
        monochrome=False,
        path=str(out_path),
    )

    ckpt = torch.load(str(out_path), map_location="cpu", weights_only=False)
    assert ckpt["schema_version"] == 2
    assert ckpt["factor_names"] == ["color", "shape"]
    assert ckpt["class_names_per_factor"] == [
        ["red", "green", "blue"],
        ["square", "circle"],
    ]
    assert "class_names" not in ckpt


def test_yolo_flat_publish_writes_sidecar(tmp_path, monkeypatch):
    """publish_trained_model writes a {stem}.v2meta.json sidecar for YOLO .pt artifacts."""
    import json

    from hydra_suite.training import model_publish
    from hydra_suite.training.contracts import TrainingRole

    src = tmp_path / "best.pt"
    src.write_bytes(b"fake yolo weights")

    models_root = tmp_path / "models"
    monkeypatch.setattr(model_publish, "get_models_root", lambda: models_root)

    key, dst = model_publish.publish_trained_model(
        role=TrainingRole.CLASSIFY_FLAT_YOLO,
        artifact_path=str(src),
        size="small",
        species="ant",
        model_info="id",
        trained_from_run_id="run1",
        dataset_fingerprint="abc123",
        base_model="yolov8n-cls",
        scheme_name="scheme",
        classifier_v2_meta={
            "arch": "yolo",
            "input_size": [224, 224],
            "factor_names": ["flat"],
            "class_names_per_factor": [["antA", "antB", "antC"]],
            "monochrome": False,
        },
    )

    sidecar = Path(dst).with_suffix(".v2meta.json")
    assert sidecar.exists(), "v2 sidecar manifest was not written"
    data = json.loads(sidecar.read_text())
    assert data["schema_version"] == 2
    assert data["arch"] == "yolo"
    assert data["factor_names"] == ["flat"]
    assert data["class_names_per_factor"] == [["antA", "antB", "antC"]]
    assert data["input_size"] == [224, 224]
    assert data["monochrome"] is False


def test_multihead_yolo_manifest_emission(tmp_path):
    """emit_yolo_multihead_manifest writes a manifest that references per-factor .pt paths."""
    import json

    from hydra_suite.training.runner import emit_yolo_multihead_manifest

    factor_dirs = []
    for fname, classes in [("color", ["r", "g", "b"]), ("shape", ["sq", "ci"])]:
        fdir = tmp_path / fname
        fdir.mkdir()
        pt = fdir / "best.pt"
        pt.write_bytes(b"fake")
        factor_dirs.append((fname, pt, classes))

    manifest_path = tmp_path / "bundle.multihead.json"
    emit_yolo_multihead_manifest(
        manifest_path=str(manifest_path),
        factors=factor_dirs,
        input_size=(224, 224),
        monochrome=False,
    )

    data = json.loads(manifest_path.read_text())
    assert data["schema_version"] == 2
    assert data["kind"] == "yolo_multihead_bundle"
    assert data["factor_names"] == ["color", "shape"]
    assert data["input_size"] == [224, 224]
    assert data["monochrome"] is False
    assert len(data["factor_models"]) == 2
    assert data["factor_models"][0]["factor"] == "color"
    assert data["factor_models"][0]["class_names"] == ["r", "g", "b"]
    # Paths are relative to the manifest location
    assert data["factor_models"][0]["path"] == "color/best.pt"
