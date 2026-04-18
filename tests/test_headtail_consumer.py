"""Tests for the head-tail consumer (rewritten to wrap ClassifierBackend)."""

from __future__ import annotations

import numpy as np
import pytest


def test_normalize_label_accepts_canonical_tokens():
    from hydra_suite.core.identity.classification.headtail import (
        HEADTAIL_CANONICAL_LABELS,
        normalize_headtail_label,
    )

    for token in ["up", "UP", "u", "N", "north"]:
        assert normalize_headtail_label(token) == "up"
    for token in ["down", "d", "south"]:
        assert normalize_headtail_label(token) == "down"
    for token in ["left", "l", "west"]:
        assert normalize_headtail_label(token) == "left"
    for token in ["right", "r", "east"]:
        assert normalize_headtail_label(token) == "right"
    for token in ["unknown", "?", "none", "na"]:
        assert normalize_headtail_label(token) == "unknown"

    assert HEADTAIL_CANONICAL_LABELS == frozenset(
        {"up", "down", "left", "right", "unknown"}
    )


def test_normalize_label_rejects_unknown_token():
    from hydra_suite.core.identity.classification.headtail import (
        normalize_headtail_label,
    )

    with pytest.raises(ValueError):
        normalize_headtail_label("cat")


def test_validate_headtail_labels_accepts_subset():
    from hydra_suite.core.identity.classification.headtail import (
        validate_headtail_labels,
    )

    normalized = validate_headtail_labels(["L", "R"])
    assert normalized == ["left", "right"]


def test_validate_headtail_labels_rejects_out_of_set():
    from hydra_suite.core.identity.classification.errors import HeadTailFormatError
    from hydra_suite.core.identity.classification.headtail import (
        validate_headtail_labels,
    )

    with pytest.raises(HeadTailFormatError):
        validate_headtail_labels(["cat", "dog"])


def test_headtail_accepts_flat_tiny_five_class(tiny_flat_headtail):
    from hydra_suite.core.identity.classification.headtail import HeadTailAnalyzer

    analyzer = HeadTailAnalyzer(
        model_path=str(tiny_flat_headtail), compute_runtime="cpu"
    )
    assert analyzer.is_loaded()
    # Expect the normalized label set in order of the checkpoint.
    assert analyzer.canonical_labels == ("up", "down", "left", "right", "unknown")


def test_headtail_accepts_flat_tiny_subset(tiny_flat_subset):
    from hydra_suite.core.identity.classification.headtail import HeadTailAnalyzer

    analyzer = HeadTailAnalyzer(model_path=str(tiny_flat_subset), compute_runtime="cpu")
    assert analyzer.canonical_labels == ("left", "right")


def test_headtail_rejects_multi_head(tiny_multi_identity):
    from hydra_suite.core.identity.classification.errors import HeadTailFormatError
    from hydra_suite.core.identity.classification.headtail import HeadTailAnalyzer

    with pytest.raises(HeadTailFormatError):
        HeadTailAnalyzer(model_path=str(tiny_multi_identity), compute_runtime="cpu")


def test_headtail_rejects_non_headtail_labels(torchvision_flat_identity):
    from hydra_suite.core.identity.classification.errors import HeadTailFormatError
    from hydra_suite.core.identity.classification.headtail import HeadTailAnalyzer

    with pytest.raises(HeadTailFormatError):
        HeadTailAnalyzer(
            model_path=str(torchvision_flat_identity), compute_runtime="cpu"
        )


def test_headtail_predict_labels_returns_normalized(tiny_flat_headtail):
    """predict_labels returns (canonical_label, confidence) tuples per crop."""
    from hydra_suite.core.identity.classification.headtail import HeadTailAnalyzer

    analyzer = HeadTailAnalyzer(
        model_path=str(tiny_flat_headtail), compute_runtime="cpu"
    )
    crops = [np.zeros((32, 32, 3), dtype=np.uint8) for _ in range(2)]
    out = analyzer.predict_labels(crops)
    assert len(out) == 2
    for label, conf in out:
        assert label in HeadTailAnalyzer.valid_output_labels()
        assert 0.0 <= conf <= 1.0


def test_headtail_load_failure_surfaces_as_exception(tmp_path):
    """Invalid checkpoints raise ClassifierError, not silent warnings."""
    from hydra_suite.core.identity.classification.errors import ClassifierFormatError
    from hydra_suite.core.identity.classification.headtail import HeadTailAnalyzer

    bad = tmp_path / "not_a_checkpoint.pth"
    bad.write_bytes(b"garbage")
    with pytest.raises(ClassifierFormatError):
        HeadTailAnalyzer(model_path=str(bad), compute_runtime="cpu")
