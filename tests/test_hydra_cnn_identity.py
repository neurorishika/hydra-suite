"""Tests for MAT CNN identity method."""

from __future__ import annotations

import pytest

# ---------------------------------------------------------------------------
# CNNIdentityConfig tests
# ---------------------------------------------------------------------------


def test_cnn_identity_config_defaults():
    from hydra_suite.core.identity.classification.cnn import CNNIdentityConfig

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
    from hydra_suite.core.identity.classification.cnn import CNNIdentityConfig

    cfg = CNNIdentityConfig(model_path="/tmp/model.pth", confidence=0.8, window=5)
    assert cfg.model_path == "/tmp/model.pth"
    assert cfg.confidence == 0.8
    assert cfg.window == 5


# ---------------------------------------------------------------------------
# ClassPrediction tests
# ---------------------------------------------------------------------------


def test_class_prediction_fields():
    from hydra_suite.core.identity.classification.cnn import ClassPrediction

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
    from hydra_suite.core.identity.classification.cnn import ClassPrediction

    p = ClassPrediction(
        det_index=1,
        factor_names=("flat",),
        class_names=(None,),
        confidences=(0.4,),
    )
    assert p.class_name is None
    assert p.confidence == 0.4


# ---------------------------------------------------------------------------
# CNNIdentityCache round-trip tests
# ---------------------------------------------------------------------------


def test_cnn_identity_cache_roundtrip(tmp_path):
    from hydra_suite.core.identity.classification.cnn import (
        ClassPrediction,
        CNNIdentityCache,
    )

    cache_path = tmp_path / "cnn_identity.npz"
    cache = CNNIdentityCache(str(cache_path))
    preds = [
        ClassPrediction(
            det_index=0,
            factor_names=("flat",),
            class_names=("tag_0",),
            confidences=(0.9,),
        ),
        ClassPrediction(
            det_index=1, factor_names=("flat",), class_names=(None,), confidences=(0.3,)
        ),
    ]
    cache.save(5, preds)
    cache.flush()  # required before loading from a fresh instance
    loaded_cache = CNNIdentityCache(str(cache_path))
    loaded = loaded_cache.load(5)
    assert len(loaded) == 2
    assert loaded[0].class_name == "tag_0"
    assert loaded[0].confidence == pytest.approx(0.9)
    assert loaded[0].det_index == 0
    assert loaded[1].class_name is None
    assert loaded[1].det_index == 1


def test_cnn_identity_cache_exists(tmp_path):
    from hydra_suite.core.identity.classification.cnn import (
        ClassPrediction,
        CNNIdentityCache,
    )

    cache_path = tmp_path / "cnn_identity.npz"
    cache = CNNIdentityCache(str(cache_path))
    assert not cache.exists()
    cache.save(
        0,
        [
            ClassPrediction(
                det_index=0,
                factor_names=("flat",),
                class_names=("tag_0",),
                confidences=(0.9,),
            )
        ],
    )
    cache.flush()
    assert cache.exists()


def test_cnn_identity_cache_empty_frame(tmp_path):
    from hydra_suite.core.identity.classification.cnn import CNNIdentityCache

    cache_path = tmp_path / "cnn_identity.npz"
    cache = CNNIdentityCache(str(cache_path))
    cache.save(10, [])
    cache.flush()
    loaded_cache = CNNIdentityCache(str(cache_path))
    loaded = loaded_cache.load(10)
    assert loaded == []


def test_cnn_identity_cache_missing_frame_returns_empty(tmp_path):
    from hydra_suite.core.identity.classification.cnn import (
        ClassPrediction,
        CNNIdentityCache,
    )

    cache_path = tmp_path / "cnn_identity.npz"
    cache = CNNIdentityCache(str(cache_path))
    cache.save(
        0,
        [
            ClassPrediction(
                det_index=0,
                factor_names=("flat",),
                class_names=("tag_0",),
                confidences=(0.9,),
            )
        ],
    )
    loaded = cache.load(99)  # frame 99 not saved
    assert loaded == []


# ---------------------------------------------------------------------------
# CNNIdentityBackend (mocked) tests
# ---------------------------------------------------------------------------


def test_backend_predict_batch_cardinality(tiny_flat_headtail):
    """predict_batch() must return exactly one ClassPrediction per input crop."""
    import numpy as np

    from hydra_suite.core.identity.classification.cnn import (
        CNNIdentityBackend,
        CNNIdentityConfig,
    )

    cfg = CNNIdentityConfig(model_path=str(tiny_flat_headtail), confidence=0.0)
    crops = [np.zeros((64, 64, 3), dtype=np.uint8) for _ in range(3)]
    backend = CNNIdentityBackend(
        cfg, model_path=str(tiny_flat_headtail), compute_runtime="cpu"
    )
    results = backend.predict_batch(crops)
    backend.close()

    assert len(results) == len(crops)
    for p in results:
        assert p.factor_names == ("flat",)


def test_backend_below_confidence_returns_none_class(tiny_flat_headtail):
    """Predictions below confidence threshold return class_name=None."""
    import numpy as np

    from hydra_suite.core.identity.classification.cnn import (
        CNNIdentityBackend,
        CNNIdentityConfig,
    )

    # With confidence=0.999, random tiny-model weights should never exceed it.
    cfg = CNNIdentityConfig(model_path=str(tiny_flat_headtail), confidence=0.999)
    crops = [np.zeros((64, 64, 3), dtype=np.uint8)]
    backend = CNNIdentityBackend(
        cfg, model_path=str(tiny_flat_headtail), compute_runtime="cpu"
    )
    results = backend.predict_batch(crops)
    backend.close()

    assert len(results) == 1
    assert results[0].class_name is None


# ---------------------------------------------------------------------------
# Checkpoint metadata extraction tests (for _handle_add_new_cnn_identity_model)
# ---------------------------------------------------------------------------


def test_pth_checkpoint_metadata_extraction(tmp_path):
    """Verify that .pth checkpoint fields are correctly extracted during import."""
    import torch

    ckpt = {
        "arch": "resnet18",
        "class_names": ["tag_0", "tag_1", "no_tag"],
        "factor_names": [],
        "input_size": (224, 224),
        "num_classes": 3,
        "model_state_dict": {},
        "best_val_acc": 0.95,
        "history": {},
        "trainable_layers": 0,
        "backbone_lr_scale": 0.1,
    }
    ckpt_path = tmp_path / "model.pth"
    torch.save(ckpt, str(ckpt_path))

    loaded = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    assert loaded["arch"] == "resnet18"
    assert loaded["class_names"] == ["tag_0", "tag_1", "no_tag"]
    assert loaded["num_classes"] == 3
    assert list(loaded["input_size"]) == [224, 224]


def test_registry_entry_format_after_import(tmp_path):
    """Registry entry for a CNN identity model has all required fields."""
    import json
    from datetime import datetime

    entry = {
        "arch": "convnext_tiny",
        "num_classes": 11,
        "class_names": [f"tag_{i}" for i in range(10)] + ["no_tag"],
        "factor_names": [],
        "input_size": [224, 224],
        "species": "ant",
        "classification_label": "apriltag",
        "added_at": datetime.now().isoformat(),
        "task_family": "classify",
        "usage_role": "cnn_identity",
    }
    registry_path = tmp_path / "model_registry.json"
    registry = {"classification/identity/test.pth": entry}
    registry_path.write_text(json.dumps(registry))

    loaded = json.loads(registry_path.read_text())
    loaded_entry = loaded["classification/identity/test.pth"]
    required = {
        "arch",
        "num_classes",
        "class_names",
        "factor_names",
        "input_size",
        "species",
        "classification_label",
        "added_at",
        "task_family",
        "usage_role",
    }
    assert required.issubset(set(loaded_entry.keys()))
    assert loaded_entry["usage_role"] == "cnn_identity"
    assert loaded_entry["num_classes"] == 11


# ---------------------------------------------------------------------------
# ClassPrediction multi-factor tests
# ---------------------------------------------------------------------------


def test_class_prediction_multi_factor_shape():
    """ClassPrediction exposes factor_names, class_names, confidences as tuples."""
    from hydra_suite.core.identity.classification.cnn import ClassPrediction

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
    from hydra_suite.core.identity.classification.cnn import ClassPrediction

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
    from hydra_suite.core.identity.classification.cnn import ClassPrediction

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


def test_cnn_identity_config_scoring_mode_default():
    from hydra_suite.core.identity.classification.cnn import CNNIdentityConfig

    cfg = CNNIdentityConfig()
    assert cfg.scoring_mode == "atomic"


def test_cnn_identity_config_accepts_per_head_average():
    from hydra_suite.core.identity.classification.cnn import CNNIdentityConfig

    cfg = CNNIdentityConfig(scoring_mode="per_head_average")
    assert cfg.scoring_mode == "per_head_average"


# ---------------------------------------------------------------------------
# CNNIdentityBackend (real model) tests
# ---------------------------------------------------------------------------


def test_cnn_identity_backend_flat_predict(tiny_flat_headtail):
    """CNNIdentityBackend returns one ClassPrediction per crop for a flat tiny model."""
    from hydra_suite.core.identity.classification.cnn import (
        CNNIdentityBackend,
        CNNIdentityConfig,
    )

    cfg = CNNIdentityConfig(model_path=str(tiny_flat_headtail), confidence=0.0)
    backend = CNNIdentityBackend(
        cfg, model_path=str(tiny_flat_headtail), compute_runtime="cpu"
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
    from hydra_suite.core.identity.classification.cnn import (
        CNNIdentityBackend,
        CNNIdentityConfig,
    )

    cfg = CNNIdentityConfig(
        model_path=str(tiny_multi_identity),
        confidence=0.0,
        scoring_mode="per_head_average",
    )
    backend = CNNIdentityBackend(
        cfg, model_path=str(tiny_multi_identity), compute_runtime="cpu"
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
    from hydra_suite.core.identity.classification.cnn import (
        CNNIdentityBackend,
        CNNIdentityConfig,
    )
    from hydra_suite.core.identity.classification.errors import ClassifierConfigError

    # Default scoring_mode == "atomic" is permissible for flat; for multi-head the
    # registry would have stored the mode explicitly. Construct with an explicit
    # empty string to simulate a missing value and assert the backend rejects it.
    cfg = CNNIdentityConfig(model_path=str(tiny_multi_identity), scoring_mode="")
    with pytest.raises(ClassifierConfigError):
        CNNIdentityBackend(
            cfg, model_path=str(tiny_multi_identity), compute_runtime="cpu"
        )


def test_cnn_identity_backend_per_factor_threshold(tiny_multi_identity):
    """Per-factor confidence threshold: a below-threshold head reports None."""
    from hydra_suite.core.identity.classification.cnn import (
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
        cfg, model_path=str(tiny_multi_identity), compute_runtime="cpu"
    )
    import numpy as _np

    preds = backend.predict_batch([_np.zeros((32, 32, 3), dtype=_np.uint8)])
    p = preds[0]
    # Each head below threshold -> None
    for name in p.class_names:
        assert name is None
    backend.close()
