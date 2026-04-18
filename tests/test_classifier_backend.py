"""Tests for the shared classifier backend."""

from __future__ import annotations


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
