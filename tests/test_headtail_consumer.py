"""Tests for the head-tail consumer (rewritten to wrap ClassifierBackend)."""

from __future__ import annotations

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
