"""Tests for the shared classifier backend."""

from __future__ import annotations

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
