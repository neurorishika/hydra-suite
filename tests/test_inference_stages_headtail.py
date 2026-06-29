import math
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch

from hydra_suite.core.inference.config import HeadTailConfig
from hydra_suite.core.inference.result import OBBResult
from hydra_suite.core.inference.runtime import RuntimeContext
from hydra_suite.core.inference.stages.headtail import (
    HeadTailModel,
    _label_to_heading_offset,
    run_headtail,
)


def _cpu_rt():
    return RuntimeContext(
        cuda_mode=False,
        device="cpu",
        use_nvdec=False,
        default_runtime="cpu",
        tensor_on_cuda=False,
    )


def _obb(n: int = 2) -> OBBResult:
    return OBBResult(
        frame_idx=0,
        centroids=np.zeros((n, 2), dtype=np.float32),
        angles=np.zeros(n, dtype=np.float32),
        sizes=np.ones(n, dtype=np.float32) * 400.0,
        shapes=np.ones((n, 2), dtype=np.float32),
        confidences=np.ones(n, dtype=np.float32) * 0.9,
        corners=np.zeros((n, 4, 2), dtype=np.float32),
        detection_ids=OBBResult.make_detection_ids(0, n),
    )


def test_label_to_heading_right():
    assert _label_to_heading_offset("right") == pytest.approx(0.0)


def test_label_to_heading_left():
    assert _label_to_heading_offset("left") == pytest.approx(math.pi)


def test_label_to_heading_up():
    assert _label_to_heading_offset("up") == pytest.approx(-math.pi / 2)


def test_label_to_heading_down():
    assert _label_to_heading_offset("down") == pytest.approx(math.pi / 2)


def test_label_to_heading_unknown_returns_none():
    assert _label_to_heading_offset("unknown") is None


def test_label_to_heading_normalizes_aliases():
    # H7: non-canonical-but-known labels resolve via alias normalization
    # instead of silently becoming undirected.
    assert _label_to_heading_offset("head_left") == pytest.approx(math.pi)
    assert _label_to_heading_offset("north") == pytest.approx(-math.pi / 2)
    assert _label_to_heading_offset("n") == pytest.approx(-math.pi / 2)
    assert _label_to_heading_offset("E") == pytest.approx(0.0)
    assert _label_to_heading_offset("south") == pytest.approx(math.pi / 2)


def test_label_to_heading_unrecognized_token_returns_none():
    # H7: a label outside the alias map leaves the detection undirected.
    assert _label_to_heading_offset("diagonal") is None


def _patch_backend(monkeypatch, *, is_multihead, labels, factor_names=("direction",)):
    """Patch ClassifierBackend used by load_headtail_model with a fake whose
    metadata reports the given multi-head flag and labels."""
    import hydra_suite.core.identity.classification.backend as backend_mod

    meta = MagicMock()
    meta.is_multihead = is_multihead
    meta.factor_names = list(factor_names)
    meta.class_names_per_factor = [list(labels)]
    meta.input_size = (64, 64)

    fake_backend = MagicMock()
    fake_backend.metadata = meta

    def _ctor(*_args, **_kwargs):
        return fake_backend

    monkeypatch.setattr(backend_mod, "ClassifierBackend", _ctor)
    return fake_backend


def test_load_headtail_rejects_multihead(monkeypatch):
    # H7: a multi-head artifact must raise rather than silently use factor 0.
    from hydra_suite.core.identity.classification.errors import HeadTailFormatError
    from hydra_suite.core.inference.stages.headtail import load_headtail_model

    fake_backend = _patch_backend(
        monkeypatch,
        is_multihead=True,
        labels=["right", "left"],
        factor_names=("direction", "species"),
    )
    config = HeadTailConfig(model_path="/ht.pt")
    with pytest.raises(HeadTailFormatError):
        load_headtail_model(config, _cpu_rt())
    fake_backend.close.assert_called_once()


def test_load_headtail_normalizes_labels(monkeypatch):
    # H7: aliased checkpoint labels are normalized to the canonical set.
    from hydra_suite.core.inference.stages.headtail import load_headtail_model

    _patch_backend(
        monkeypatch,
        is_multihead=False,
        labels=["head_right", "head_left", "head_up", "head_down"],
    )
    config = HeadTailConfig(model_path="/ht.pt")
    model = load_headtail_model(config, _cpu_rt())
    assert model.class_names == ["right", "left", "up", "down"]


def test_load_headtail_rejects_noncanonical_labels(monkeypatch):
    # H7: labels outside the canonical head-tail set fail loudly at load.
    from hydra_suite.core.identity.classification.errors import HeadTailFormatError
    from hydra_suite.core.inference.stages.headtail import load_headtail_model

    _patch_backend(
        monkeypatch,
        is_multihead=False,
        labels=["antA", "antB"],
    )
    config = HeadTailConfig(model_path="/ht.pt")
    with pytest.raises(HeadTailFormatError):
        load_headtail_model(config, _cpu_rt())


def test_run_headtail_empty_crops_returns_nan_hints():
    config = HeadTailConfig(model_path="/ht.pt")
    mock_backend = MagicMock()
    model = HeadTailModel(
        backend=mock_backend,
        input_size=(64, 64),
        class_names=["right", "left", "up", "down", "unknown"],
    )
    # Backend yields no predictions -> hints stay NaN, still sized to n.
    mock_backend.predict_batch.return_value = []
    frame = np.zeros((128, 128, 3), dtype=np.uint8)
    result = run_headtail(frame, _obb(n=2), model, config, _cpu_rt())
    assert len(result.heading_hints) == 2
    assert all(math.isnan(h) for h in result.heading_hints)
    assert all(m == 0 for m in result.directed_mask)
    # Per Correction 15: canonical_affines is None when not provided by this stage
    assert result.canonical_affines is None


def test_run_headtail_confident_prediction():
    config = HeadTailConfig(model_path="/ht.pt", confidence_threshold=0.5)
    mock_backend = MagicMock()
    mock_backend.predict_batch.return_value = [
        [np.array([0.9, 0.1, 0.0, 0.0, 0.0])],
        [np.array([0.1, 0.8, 0.0, 0.0, 0.1])],
    ]
    model = HeadTailModel(
        backend=mock_backend,
        input_size=(64, 64),
        class_names=["right", "left", "up", "down", "unknown"],
    )
    crops = torch.zeros((2, 3, 64, 64))
    result = run_headtail(crops, _obb(n=2), model, config, _cpu_rt())
    assert result.directed_mask[0] == 1
    assert result.directed_mask[1] == 1
    assert result.heading_hints[0] == pytest.approx(0.0)
    assert result.heading_hints[1] == pytest.approx(math.pi)
    # Inference path also returns None for canonical_affines (affines belong to crops stage)
    assert result.canonical_affines is None


def test_run_headtail_below_threshold_not_directed():
    config = HeadTailConfig(model_path="/ht.pt", confidence_threshold=0.9)
    mock_backend = MagicMock()
    mock_backend.predict_batch.return_value = [
        [np.array([0.6, 0.4, 0.0, 0.0, 0.0])],
    ]
    model = HeadTailModel(
        backend=mock_backend,
        input_size=(64, 64),
        class_names=["right", "left", "up", "down", "unknown"],
    )
    crops = torch.zeros((1, 3, 64, 64))
    result = run_headtail(crops, _obb(n=1), model, config, _cpu_rt())
    assert result.directed_mask[0] == 0


def test_run_headtail_unknown_label_not_directed():
    config = HeadTailConfig(model_path="/ht.pt", confidence_threshold=0.5)
    mock_backend = MagicMock()
    mock_backend.predict_batch.return_value = [
        [np.array([0.0, 0.0, 0.0, 0.0, 0.95])],  # unknown, high conf
    ]
    model = HeadTailModel(
        backend=mock_backend,
        input_size=(64, 64),
        class_names=["right", "left", "up", "down", "unknown"],
    )
    crops = torch.zeros((1, 3, 64, 64))
    result = run_headtail(crops, _obb(n=1), model, config, _cpu_rt())
    assert result.directed_mask[0] == 0
    assert math.isnan(result.heading_hints[0])
