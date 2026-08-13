"""Tests for MAT CNN identity method."""

from __future__ import annotations

import pytest

from hydra_suite.runtime.resolver import ResolvedBackend

# ---------------------------------------------------------------------------
# CNNIdentityConfig tests
# ---------------------------------------------------------------------------


def test_cnn_identity_config_defaults():
    from hydra_suite.core.individual.classification.cnn import CNNIdentityConfig

    cfg = CNNIdentityConfig()
    assert cfg.model_path == ""
    assert cfg.confidence == 0.5
    assert cfg.label == ""
    assert cfg.batch_size == 64
    assert cfg.match_bonus == 0.5
    assert cfg.mismatch_penalty == 1.0
    assert cfg.window == 10
    assert cfg.scoring_mode == "atomic"


def test_cnn_identity_config_custom():
    from hydra_suite.core.individual.classification.cnn import CNNIdentityConfig

    cfg = CNNIdentityConfig(model_path="/tmp/model.pth", confidence=0.8, window=5)
    assert cfg.model_path == "/tmp/model.pth"
    assert cfg.confidence == 0.8
    assert cfg.window == 5


# ---------------------------------------------------------------------------
# ClassPrediction tests
# ---------------------------------------------------------------------------


def test_class_prediction_fields():
    from hydra_suite.core.individual.classification.cnn import ClassPrediction

    p = ClassPrediction(
        det_index=2,
        factor_names=("flat",),
        class_names=("antA",),
        confidences=(0.9,),
    )
    assert p.class_name == "antA"
    assert p.confidence == 0.9
    assert p.det_index == 2


def test_class_prediction_none_class_name():
    from hydra_suite.core.individual.classification.cnn import ClassPrediction

    p = ClassPrediction(
        det_index=1,
        factor_names=("flat",),
        class_names=(None,),
        confidences=(0.4,),
    )
    assert p.class_name is None
    assert p.confidence == 0.4


# ---------------------------------------------------------------------------
# CNNIdentityBackend (mocked) tests
# ---------------------------------------------------------------------------


def test_backend_predict_batch_cardinality(tiny_flat_headtail):
    """predict_batch() must return exactly one ClassPrediction per input crop."""
    import numpy as np

    from hydra_suite.core.individual.classification.cnn import (
        CNNIdentityBackend,
        CNNIdentityConfig,
    )

    cfg = CNNIdentityConfig(model_path=str(tiny_flat_headtail), confidence=0.0)
    crops = [np.zeros((64, 64, 3), dtype=np.uint8) for _ in range(3)]
    backend = CNNIdentityBackend(
        cfg,
        model_path=str(tiny_flat_headtail),
        resolved=ResolvedBackend("torch", "cpu", False),
    )
    results = backend.predict_batch(crops)
    backend.close()

    assert len(results) == len(crops)
    for p in results:
        assert p.factor_names == ("flat",)


def test_backend_below_confidence_returns_none_class(tiny_flat_headtail):
    """Predictions below confidence threshold return class_name=None."""
    import numpy as np

    from hydra_suite.core.individual.classification.cnn import (
        CNNIdentityBackend,
        CNNIdentityConfig,
    )

    # With confidence=0.999, random tiny-model weights should never exceed it.
    cfg = CNNIdentityConfig(model_path=str(tiny_flat_headtail), confidence=0.999)
    crops = [np.zeros((64, 64, 3), dtype=np.uint8)]
    backend = CNNIdentityBackend(
        cfg,
        model_path=str(tiny_flat_headtail),
        resolved=ResolvedBackend("torch", "cpu", False),
    )
    results = backend.predict_batch(crops)
    backend.close()

    assert len(results) == 1
    assert results[0].class_name is None


# ---------------------------------------------------------------------------
# ClassPrediction multi-factor tests
# ---------------------------------------------------------------------------


def test_class_prediction_multi_factor_shape():
    """ClassPrediction exposes factor_names, class_names, confidences as tuples."""
    from hydra_suite.core.individual.classification.cnn import ClassPrediction

    p = ClassPrediction(
        det_index=0,
        factor_names=("color", "shape"),
        class_names=("red", None),
        confidences=(0.9, 0.4),
    )
    assert p.factor_names == ("color", "shape")
    assert p.class_names == ("red", None)
    assert p.confidences == (0.9, 0.4)
    assert p.is_unknown == (False, False)


def test_class_prediction_flat_convenience_accessors():
    """Flat (K=1) predictions expose class_name / confidence shortcuts."""
    from hydra_suite.core.individual.classification.cnn import ClassPrediction

    p = ClassPrediction(
        det_index=3,
        factor_names=("flat",),
        class_names=("antA",),
        confidences=(0.75,),
    )
    assert p.class_name == "antA"
    assert p.confidence == 0.75

    q = ClassPrediction(
        det_index=3,
        factor_names=("flat",),
        class_names=(None,),
        confidences=(0.2,),
    )
    assert q.class_name is None
    assert q.confidence == 0.2


def test_class_prediction_flat_accessors_error_on_multi_factor():
    from hydra_suite.core.individual.classification.cnn import ClassPrediction

    p = ClassPrediction(
        det_index=0,
        factor_names=("a", "b"),
        class_names=("x", "y"),
        confidences=(0.5, 0.5),
    )
    with pytest.raises(ValueError):
        _ = p.class_name
    with pytest.raises(ValueError):
        _ = p.confidence


# ---------------------------------------------------------------------------
# CNNIdentityBackend (real model) tests
# ---------------------------------------------------------------------------


def test_cnn_identity_backend_flat_predict(tiny_flat_headtail):
    """CNNIdentityBackend returns one ClassPrediction per crop for a flat tiny model."""
    from hydra_suite.core.individual.classification.cnn import (
        CNNIdentityBackend,
        CNNIdentityConfig,
    )

    cfg = CNNIdentityConfig(model_path=str(tiny_flat_headtail), confidence=0.0)
    backend = CNNIdentityBackend(
        cfg,
        model_path=str(tiny_flat_headtail),
        resolved=ResolvedBackend("torch", "cpu", False),
    )
    import numpy as _np

    crops = [_np.zeros((32, 32, 3), dtype=_np.uint8) for _ in range(3)]
    preds = backend.predict_batch(crops)
    assert len(preds) == 3
    for i, p in enumerate(preds):
        assert p.det_index == i
        assert p.factor_names == ("flat",)
        assert len(p.class_names) == 1
        assert len(p.confidences) == 1
    backend.close()


def test_cnn_identity_backend_multihead_predict(tiny_multi_identity):
    from hydra_suite.core.individual.classification.cnn import (
        CNNIdentityBackend,
        CNNIdentityConfig,
    )

    cfg = CNNIdentityConfig(
        model_path=str(tiny_multi_identity),
        confidence=0.0,
        scoring_mode="per_head_average",
    )
    backend = CNNIdentityBackend(
        cfg,
        model_path=str(tiny_multi_identity),
        resolved=ResolvedBackend("torch", "cpu", False),
    )
    import numpy as _np

    crops = [_np.zeros((32, 32, 3), dtype=_np.uint8) for _ in range(2)]
    preds = backend.predict_batch(crops)
    assert len(preds) == 2
    for p in preds:
        assert p.factor_names == ("color", "shape")
        assert len(p.class_names) == 2
        assert len(p.confidences) == 2
    backend.close()


def test_cnn_identity_backend_rejects_multihead_without_scoring_mode(
    tiny_multi_identity,
):
    from hydra_suite.core.individual.classification.cnn import (
        CNNIdentityBackend,
        CNNIdentityConfig,
    )
    from hydra_suite.core.individual.classification.errors import ClassifierConfigError

    # Default scoring_mode == "atomic" is permissible for flat; for multi-head the
    # registry would have stored the mode explicitly. Construct with an explicit
    # empty string to simulate a missing value and assert the backend rejects it.
    cfg = CNNIdentityConfig(model_path=str(tiny_multi_identity), scoring_mode="")
    with pytest.raises(ClassifierConfigError):
        CNNIdentityBackend(
            cfg,
            model_path=str(tiny_multi_identity),
            resolved=ResolvedBackend("torch", "cpu", False),
        )


def test_cnn_identity_backend_per_factor_threshold(tiny_multi_identity):
    """Per-factor confidence threshold: a below-threshold head reports None."""
    from hydra_suite.core.individual.classification.cnn import (
        CNNIdentityBackend,
        CNNIdentityConfig,
    )

    # Set threshold high so random weights never meet it.
    cfg = CNNIdentityConfig(
        model_path=str(tiny_multi_identity),
        confidence=0.999,
        scoring_mode="atomic",
    )
    backend = CNNIdentityBackend(
        cfg,
        model_path=str(tiny_multi_identity),
        resolved=ResolvedBackend("torch", "cpu", False),
    )
    import numpy as _np

    preds = backend.predict_batch([_np.zeros((32, 32, 3), dtype=_np.uint8)])
    p = preds[0]
    # Each head below threshold -> None
    for name in p.class_names:
        assert name is None
    backend.close()
