"""Tests for the head-tail consumer (rewritten to wrap ClassifierBackend)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from hydra_suite.runtime.resolver import ResolvedBackend


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


def test_validate_class_names_strict_accepts_four_label_subset():
    from hydra_suite.core.identity.classification.headtail import HeadTailAnalyzer

    normalized = HeadTailAnalyzer._validate_class_names(
        ["left", "right", "unknown", "up"],
        strict=True,
        source="checkpoint",
    )
    assert normalized == ["left", "right", "unknown", "up"]


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
        model_path=str(tiny_flat_headtail),
        resolved=ResolvedBackend("torch", "cpu", False),
    )
    assert analyzer.is_loaded()
    # Expect the normalized label set in order of the checkpoint.
    assert analyzer.canonical_labels == ("up", "down", "left", "right", "unknown")


def test_headtail_accepts_flat_tiny_subset(tiny_flat_subset):
    from hydra_suite.core.identity.classification.headtail import HeadTailAnalyzer

    analyzer = HeadTailAnalyzer(
        model_path=str(tiny_flat_subset),
        resolved=ResolvedBackend("torch", "cpu", False),
    )
    assert analyzer.canonical_labels == ("left", "right")


def test_headtail_accepts_legacy_flat_torchvision_subset(
    legacy_torchvision_flat_headtail,
):
    from hydra_suite.core.identity.classification.errors import ClassifierFormatError
    from hydra_suite.core.identity.classification.headtail import HeadTailAnalyzer

    with pytest.raises(ClassifierFormatError):
        HeadTailAnalyzer(
            model_path=str(legacy_torchvision_flat_headtail),
            resolved=ResolvedBackend("torch", "cpu", False),
        )


def test_headtail_rejects_multi_head(tiny_multi_identity):
    from hydra_suite.core.identity.classification.errors import HeadTailFormatError
    from hydra_suite.core.identity.classification.headtail import HeadTailAnalyzer

    with pytest.raises(HeadTailFormatError):
        HeadTailAnalyzer(
            model_path=str(tiny_multi_identity),
            resolved=ResolvedBackend("torch", "cpu", False),
        )


def test_headtail_rejects_non_headtail_labels(torchvision_flat_identity):
    from hydra_suite.core.identity.classification.errors import HeadTailFormatError
    from hydra_suite.core.identity.classification.headtail import HeadTailAnalyzer

    with pytest.raises(HeadTailFormatError):
        HeadTailAnalyzer(
            model_path=str(torchvision_flat_identity),
            resolved=ResolvedBackend("torch", "cpu", False),
        )


def test_headtail_predict_labels_returns_normalized(tiny_flat_headtail):
    """predict_labels returns (canonical_label, confidence) tuples per crop."""
    from hydra_suite.core.identity.classification.headtail import HeadTailAnalyzer

    analyzer = HeadTailAnalyzer(
        model_path=str(tiny_flat_headtail),
        resolved=ResolvedBackend("torch", "cpu", False),
    )
    crops = [np.zeros((32, 32, 3), dtype=np.uint8) for _ in range(2)]
    out = analyzer.predict_labels(crops)
    assert len(out) == 2
    for label, conf in out:
        assert label in HeadTailAnalyzer.valid_output_labels()
        assert 0.0 <= conf <= 1.0


def test_heading_for_direction_matches_canonical_crop_offsets():
    from hydra_suite.core.identity.classification.headtail import heading_for_direction

    axis = 0.25
    assert heading_for_direction(axis, "right") == pytest.approx(axis)
    assert heading_for_direction(axis, "left") == pytest.approx(
        (axis + math.pi) % (2.0 * math.pi)
    )
    assert heading_for_direction(axis, "up") == pytest.approx(
        (axis - (math.pi / 2.0)) % (2.0 * math.pi)
    )
    assert heading_for_direction(axis, "down") == pytest.approx(
        (axis + (math.pi / 2.0)) % (2.0 * math.pi)
    )
    assert heading_for_direction(axis, "unknown") is None


def test_scatter_backend_v2_uses_canonical_direction_offsets():
    from hydra_suite.core.identity.classification.headtail import HeadTailAnalyzer

    analyzer = HeadTailAnalyzer.__new__(HeadTailAnalyzer)
    analyzer._canonical_labels = ("left", "right", "up", "down", "unknown")
    analyzer._conf_threshold = 0.5

    cls_results = [
        [np.array([0.9, 0.01, 0.01, 0.01, 0.07], dtype=np.float32)],
        [np.array([0.01, 0.9, 0.01, 0.01, 0.07], dtype=np.float32)],
        [np.array([0.01, 0.01, 0.9, 0.01, 0.07], dtype=np.float32)],
        [np.array([0.01, 0.01, 0.01, 0.9, 0.07], dtype=np.float32)],
    ]
    axis = 0.4
    all_meta = [(0, idx, axis, np.eye(2, 3, dtype=np.float32)) for idx in range(4)]
    results = [[(float("nan"), 0.0, 0)] * 4]

    analyzer._scatter_backend_v2(cls_results, all_meta, results)

    assert results[0][0][0] == pytest.approx((axis + math.pi) % (2.0 * math.pi))
    assert results[0][1][0] == pytest.approx(axis)
    assert results[0][2][0] == pytest.approx((axis - (math.pi / 2.0)) % (2.0 * math.pi))
    assert results[0][3][0] == pytest.approx((axis + (math.pi / 2.0)) % (2.0 * math.pi))
    assert all(entry[2] == 1 for entry in results[0])


def test_headtail_load_failure_surfaces_as_exception(tmp_path):
    """Invalid checkpoints raise ClassifierError, not silent warnings."""
    from hydra_suite.core.identity.classification.errors import ClassifierFormatError
    from hydra_suite.core.identity.classification.headtail import HeadTailAnalyzer

    bad = tmp_path / "not_a_checkpoint.pth"
    bad.write_bytes(b"garbage")
    with pytest.raises(ClassifierFormatError):
        HeadTailAnalyzer(
            model_path=str(bad), resolved=ResolvedBackend("torch", "cpu", False)
        )


def test_headtail_predict_chunks_large_crop_batches():
    from hydra_suite.core.identity.classification.headtail import HeadTailAnalyzer

    calls: list[int] = []

    class FakeBackend:
        def predict_batch(self, crops):
            calls.append(len(crops))
            return [[np.array([1.0], dtype=np.float32)] for _ in crops]

    analyzer = HeadTailAnalyzer.__new__(HeadTailAnalyzer)
    analyzer._backend_obj = FakeBackend()
    analyzer._batch_size = 2

    crops = [np.zeros((8, 8, 3), dtype=np.uint8) for _ in range(5)]
    result = analyzer._predict(crops)

    assert len(result) == 5
    assert calls == [2, 2, 1]
