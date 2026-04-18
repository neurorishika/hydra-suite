"""Tests for the shared classifier backend."""

from __future__ import annotations

import numpy as np
import pytest


def test_error_hierarchy_importable():
    """ClassifierError hierarchy exports from the errors module with correct inheritance."""
    from hydra_suite.core.identity.classification.errors import (
        ClassifierConfigError,
        ClassifierError,
        ClassifierFormatError,
        ClassifierRuntimeError,
        HeadTailFormatError,
    )

    assert issubclass(ClassifierFormatError, ClassifierError)
    assert issubclass(ClassifierRuntimeError, ClassifierError)
    assert issubclass(ClassifierConfigError, ClassifierError)
    assert issubclass(HeadTailFormatError, ClassifierFormatError)

    # Instantiable with a message
    err = ClassifierFormatError("bad")
    assert str(err) == "bad"


def test_classifier_metadata_fields():
    """ClassifierMetadata is frozen and exposes canonical fields."""
    from hydra_suite.core.identity.classification.backend import ClassifierMetadata

    meta = ClassifierMetadata(
        arch="tinyclassifier",
        input_size=(224, 224),
        is_multihead=False,
        factor_names=["flat"],
        class_names_per_factor=[["a", "b"]],
        monochrome=False,
        source_path="/tmp/model.pth",
    )
    assert meta.arch == "tinyclassifier"
    assert meta.input_size == (224, 224)
    assert meta.is_multihead is False
    assert meta.factor_names == ["flat"]
    assert meta.class_names_per_factor == [["a", "b"]]

    # Frozen
    import dataclasses

    with pytest.raises(dataclasses.FrozenInstanceError):
        meta.arch = "yolo"


def test_backend_parses_tiny_flat_metadata(tiny_flat_headtail):
    """ClassifierBackend exposes metadata for a v2 tiny flat checkpoint without loading weights."""
    from hydra_suite.core.identity.classification.backend import ClassifierBackend

    backend = ClassifierBackend(str(tiny_flat_headtail), compute_runtime="cpu")
    meta = backend.metadata
    assert meta.arch == "tinyclassifier"
    assert meta.input_size == (64, 64)
    assert meta.is_multihead is False
    assert meta.factor_names == ["flat"]
    assert meta.class_names_per_factor == [["up", "down", "left", "right", "unknown"]]
    assert meta.monochrome is False
    assert meta.source_path == str(tiny_flat_headtail)
    backend.close()


def test_backend_tiny_flat_predict_batch_shape(tiny_flat_headtail):
    """predict_batch returns per-crop per-factor probability vectors with correct shape."""
    from hydra_suite.core.identity.classification.backend import ClassifierBackend

    backend = ClassifierBackend(str(tiny_flat_headtail), compute_runtime="cpu")
    crops = [np.zeros((32, 32, 3), dtype=np.uint8) for _ in range(3)]
    out = backend.predict_batch(crops)
    assert isinstance(out, list) and len(out) == 3
    for per_crop in out:
        assert isinstance(per_crop, list) and len(per_crop) == 1  # K=1
        probs = per_crop[0]
        assert probs.shape == (5,)
        assert np.isfinite(probs).all()
        assert abs(probs.sum() - 1.0) < 1e-5
    backend.close()


def test_backend_non_square_input_size_roundtrip(tiny_flat_nonsquare):
    """[H, W] serialization in checkpoint is preserved as (H, W) in memory."""
    from hydra_suite.core.identity.classification.backend import ClassifierBackend

    backend = ClassifierBackend(str(tiny_flat_nonsquare), compute_runtime="cpu")
    assert backend.metadata.input_size == (256, 192)
    # predict_batch preprocesses to (H=256, W=192). Just smoke-test that it does not raise.
    crops = [np.zeros((100, 100, 3), dtype=np.uint8)]
    out = backend.predict_batch(crops)
    assert out[0][0].shape == (3,)
    backend.close()


def test_backend_parses_yolo_flat_metadata(yolo_flat_headtail):
    """YOLO classify .pt exposes 5-class flat metadata via backend."""
    pytest.importorskip("ultralytics")
    from hydra_suite.core.identity.classification.backend import ClassifierBackend

    backend = ClassifierBackend(str(yolo_flat_headtail), compute_runtime="cpu")
    meta = backend.metadata
    assert meta.arch == "yolo"
    assert meta.is_multihead is False
    assert meta.factor_names == ["flat"]
    assert len(meta.class_names_per_factor[0]) == 5
    assert set(meta.class_names_per_factor[0]) == {
        "up",
        "down",
        "left",
        "right",
        "unknown",
    }
    backend.close()


def test_backend_yolo_flat_predict_batch_shape(yolo_flat_headtail):
    pytest.importorskip("ultralytics")
    from hydra_suite.core.identity.classification.backend import ClassifierBackend

    backend = ClassifierBackend(str(yolo_flat_headtail), compute_runtime="cpu")
    crops = [np.zeros((64, 64, 3), dtype=np.uint8) for _ in range(2)]
    out = backend.predict_batch(crops)
    assert len(out) == 2
    for per_crop in out:
        assert len(per_crop) == 1
        probs = per_crop[0]
        assert probs.shape == (5,)
        assert abs(probs.sum() - 1.0) < 1e-3
    backend.close()


def test_backend_torchvision_flat_metadata_and_inference(torchvision_flat_identity):
    """ClassifierBackend loads a torchvision flat v2 checkpoint and returns per-factor probs."""
    from hydra_suite.core.identity.classification.backend import ClassifierBackend

    backend = ClassifierBackend(str(torchvision_flat_identity), compute_runtime="cpu")
    meta = backend.metadata
    assert meta.arch == "resnet18"
    assert meta.input_size == (64, 64)
    assert meta.is_multihead is False
    assert meta.class_names_per_factor == [["antA", "antB", "antC"]]

    crops = [np.zeros((32, 32, 3), dtype=np.uint8) for _ in range(2)]
    out = backend.predict_batch(crops)
    assert len(out) == 2
    for per_crop in out:
        assert len(per_crop) == 1
        assert per_crop[0].shape == (3,)
    backend.close()


def test_backend_tiny_multi_metadata_and_inference(tiny_multi_identity):
    """ClassifierBackend parses multi-head metadata and splits logits per factor."""
    from hydra_suite.core.identity.classification.backend import ClassifierBackend

    backend = ClassifierBackend(str(tiny_multi_identity), compute_runtime="cpu")
    meta = backend.metadata
    assert meta.is_multihead is True
    assert meta.factor_names == ["color", "shape"]
    assert meta.class_names_per_factor == [["r", "g", "b"], ["sq", "ci"]]

    crops = [np.zeros((32, 32, 3), dtype=np.uint8) for _ in range(2)]
    out = backend.predict_batch(crops)
    assert len(out) == 2
    for per_crop in out:
        assert len(per_crop) == 2  # K=2
        assert per_crop[0].shape == (3,)
        assert per_crop[1].shape == (2,)
        assert abs(per_crop[0].sum() - 1.0) < 1e-5
        assert abs(per_crop[1].sum() - 1.0) < 1e-5
    backend.close()
