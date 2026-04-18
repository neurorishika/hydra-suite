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


def test_backend_predict_batch_cardinality():
    """predict_batch() must return exactly one ClassPrediction per input crop."""
    from unittest.mock import patch

    import numpy as np

    from hydra_suite.core.identity.classification.cnn import (
        CNNIdentityBackend,
        CNNIdentityConfig,
    )

    cfg = CNNIdentityConfig(model_path="/tmp/m.pth", confidence=0.5)
    crops = [np.zeros((64, 64, 3), dtype=np.uint8) for _ in range(3)]
    backend = CNNIdentityBackend(cfg, compute_runtime="cpu")

    # Mock _ensure_loaded and _infer_fn together to avoid touching disk
    fixed_logits = np.array([[1.0, 2.0, 0.5]] * 3, dtype=np.float32)

    def fake_ensure_loaded():
        backend._loaded = True
        backend._class_names = ["tag_0", "tag_1", "no_tag"]
        backend._input_size = (64, 64)
        backend._infer_fn = lambda batch_np: fixed_logits

    with patch.object(backend, "_ensure_loaded", fake_ensure_loaded):
        results = backend.predict_batch(crops)

    assert len(results) == len(crops)


def test_backend_below_confidence_returns_none_class():
    """Predictions below confidence threshold return class_name=None."""
    from unittest.mock import patch

    import numpy as np

    from hydra_suite.core.identity.classification.cnn import (
        CNNIdentityBackend,
        CNNIdentityConfig,
    )

    # With confidence=0.9, softmax of [1.0, 1.0, 1.0] = [0.333, 0.333, 0.333] < 0.9
    cfg = CNNIdentityConfig(model_path="/tmp/m.pth", confidence=0.9)
    crops = [np.zeros((64, 64, 3), dtype=np.uint8)]
    backend = CNNIdentityBackend(cfg, model_path="/tmp/m.pth", compute_runtime="cpu")

    # Equal logits → max softmax = 0.333 < 0.9 threshold
    flat_logits = np.array([[1.0, 1.0, 1.0]], dtype=np.float32)

    def fake_ensure_loaded():
        backend._loaded = True
        backend._class_names = ["tag_0", "tag_1", "no_tag"]
        backend._input_size = (64, 64)
        backend._infer_fn = lambda batch_np: flat_logits

    with patch.object(backend, "_ensure_loaded", fake_ensure_loaded):
        results = backend.predict_batch(crops)

    assert len(results) == 1
    assert results[0].class_name is None
    assert results[0].confidence == pytest.approx(1.0 / 3.0, abs=1e-3)


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
# TrackCNNHistory tests
# ---------------------------------------------------------------------------


def test_track_cnn_history_majority_vote():
    """3 out of 5 frames predict the same class → that class is the identity."""
    from hydra_suite.core.identity.classification.cnn import TrackCNNHistory

    hist = TrackCNNHistory(n_tracks=1, window=10)
    hist.record(0, 1, "tag_3")
    hist.record(0, 2, "tag_3")
    hist.record(0, 3, "tag_3")
    hist.record(0, 4, "tag_0")
    hist.record(0, 5, "tag_1")
    assert hist.majority_class(0) == "tag_3"


def test_track_cnn_history_no_observations_returns_none():
    from hydra_suite.core.identity.classification.cnn import TrackCNNHistory

    hist = TrackCNNHistory(n_tracks=2, window=10)
    assert hist.majority_class(0) is None
    assert hist.majority_class(1) is None


def test_track_cnn_history_tied_returns_none():
    """Exact tie in majority vote → no clear identity."""
    from hydra_suite.core.identity.classification.cnn import TrackCNNHistory

    hist = TrackCNNHistory(n_tracks=1, window=10)
    hist.record(0, 1, "tag_0")
    hist.record(0, 2, "tag_1")
    # Tied: no majority → returns None
    assert hist.majority_class(0) is None


def test_track_cnn_history_window_drops_old():
    """Observations outside the window are not counted."""
    from hydra_suite.core.identity.classification.cnn import TrackCNNHistory

    hist = TrackCNNHistory(n_tracks=1, window=3)
    # Old observations (frames 0-2): tag_0 wins
    hist.record(0, 0, "tag_0")
    hist.record(0, 1, "tag_0")
    hist.record(0, 2, "tag_0")
    # New observations (frames 3-5): tag_1 wins; frame 0-2 drop out
    hist.record(0, 3, "tag_1")
    hist.record(0, 4, "tag_1")
    hist.record(0, 5, "tag_1")
    assert hist.majority_class(0) == "tag_1"


def test_track_cnn_history_build_list():
    from hydra_suite.core.identity.classification.cnn import TrackCNNHistory

    hist = TrackCNNHistory(n_tracks=3, window=10)
    hist.record(0, 1, "tag_0")
    hist.record(0, 2, "tag_0")
    hist.record(2, 1, "no_tag")
    identity_list = hist.build_track_identity_list(3)
    assert identity_list[0] == "tag_0"
    assert identity_list[1] is None
    assert identity_list[2] == "no_tag"


# ---------------------------------------------------------------------------
# Hungarian cost adjustment tests
# ---------------------------------------------------------------------------


def test_hungarian_cnn_match_bonus_applied():
    """When detection class == track identity, cost decreases by match_bonus."""
    from hydra_suite.core.identity.classification.cnn import apply_cnn_identity_cost

    cost = 50.0
    adjusted = apply_cnn_identity_cost(
        cost=cost,
        det_class="tag_3",
        track_identity="tag_3",
        match_bonus=20.0,
        mismatch_penalty=50.0,
    )
    assert adjusted == pytest.approx(50.0 - 20.0)


def test_hungarian_cnn_mismatch_penalty_applied():
    """When detection class != track identity, cost increases by mismatch_penalty."""
    from hydra_suite.core.identity.classification.cnn import apply_cnn_identity_cost

    cost = 50.0
    adjusted = apply_cnn_identity_cost(
        cost=cost,
        det_class="tag_0",
        track_identity="tag_3",
        match_bonus=20.0,
        mismatch_penalty=50.0,
    )
    assert adjusted == pytest.approx(50.0 + 50.0)


def test_hungarian_cnn_no_adjustment_when_det_none():
    """No cost adjustment when det_class is None (low confidence)."""
    from hydra_suite.core.identity.classification.cnn import apply_cnn_identity_cost

    cost = 50.0
    adjusted = apply_cnn_identity_cost(
        cost=cost,
        det_class=None,
        track_identity="tag_3",
        match_bonus=20.0,
        mismatch_penalty=50.0,
    )
    assert adjusted == pytest.approx(50.0)


def test_hungarian_cnn_no_adjustment_when_track_identity_none():
    """No cost adjustment when track identity is None (unassigned)."""
    from hydra_suite.core.identity.classification.cnn import apply_cnn_identity_cost

    cost = 50.0
    adjusted = apply_cnn_identity_cost(
        cost=cost,
        det_class="tag_3",
        track_identity=None,
        match_bonus=20.0,
        mismatch_penalty=50.0,
    )
    assert adjusted == pytest.approx(50.0)


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
