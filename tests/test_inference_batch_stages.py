"""Tests for batch-native head-tail / CNN / pose stages and scatter assembly.

Fakes align to the REAL backend predict_batch contract:
  - predict_batch(crops) -> list-per-detection of list-per-factor of np.ndarray
  - e.g. flat headtail: [[np.array([p_right, p_left, p_up, p_down])], ...]
"""
from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from hydra_suite.core.inference.config import CNNConfig, HeadTailConfig
from hydra_suite.core.inference.result import CropBatch, FrameResult, OBBResult
from hydra_suite.core.inference.runtime import RuntimeContext
from hydra_suite.core.inference.stages.assemble import scatter
from hydra_suite.core.inference.stages.cnn import CNNModel, run_cnn, run_cnn_batch
from hydra_suite.core.inference.stages.headtail import (
    HeadTailModel,
    run_headtail,
    run_headtail_batch,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _cpu_rt():
    return RuntimeContext(
        cuda_mode=False,
        device="cpu",
        use_nvdec=False,
        tensor_on_cuda=False,
        default_runtime="cpu",
    )


def _obb(frame_idx: int, n: int) -> OBBResult:
    return OBBResult(
        frame_idx=frame_idx,
        centroids=np.zeros((n, 2), np.float32),
        angles=np.zeros(n, np.float32),
        sizes=np.ones(n, np.float32),
        shapes=np.ones((n, 2), np.float32),
        confidences=np.ones(n, np.float32),
        corners=np.zeros((n, 4, 2), np.float32),
        detection_ids=np.array(
            [frame_idx * 10000 + s for s in range(n)], np.int64
        ),
    )


def _make_batch(obb0: OBBResult, obb1: OBBResult) -> CropBatch:
    """Build a 2-frame CropBatch with obb0 at frame 0, obb1 at frame 1."""
    n0, n1 = obb0.num_detections, obb1.num_detections
    n_total = n0 + n1
    crops = torch.zeros(n_total, 3, 16, 16)
    det_ids = np.concatenate([obb0.detection_ids, obb1.detection_ids])
    frame_idx = np.array([0] * n0 + [1] * n1, np.int64)
    native_sizes = np.array([[16, 16]] * n_total, np.int64)
    return CropBatch(
        crops=crops,
        detection_ids=det_ids,
        frame_index=frame_idx,
        obb_by_frame={0: obb0, 1: obb1},
        native_sizes=native_sizes,
    )


# ---------------------------------------------------------------------------
# Fake models with REAL backend contract
# ---------------------------------------------------------------------------

# predict_batch returns: list[list[np.ndarray]]  (detections × factors)
# Each element is a list of factor-probability arrays.
# For a flat 4-class headtail model: [[np.array([p0,p1,p2,p3])], ...]


class _FakeHTBackend:
    """Returns prob-array with "up" (index 2) winning at 0.9 for every crop."""

    def predict_batch(self, crops):
        # up is index 2 in [right, left, up, down]
        return [
            [np.array([0.05, 0.05, 0.9, 0.0], dtype=np.float32)]
            for _ in crops
        ]


class _FakeHTModel:
    backend = _FakeHTBackend()
    input_size = (16, 16)
    class_names = ["right", "left", "up", "down"]


class _FakeCNNBackend:
    """Returns uniform probabilities over 3 classes for every crop."""

    def predict_batch(self, crops):
        probs = np.array([1 / 3, 1 / 3, 1 / 3], dtype=np.float32)
        return [[probs.copy()] for _ in crops]


def _fake_cnn_model():
    return CNNModel(
        backend=_FakeCNNBackend(),
        input_size=(16, 16),
        factor_names=["identity"],
        factor_class_names=[["a", "b", "c"]],
    )


def _fake_ht_model():
    return _FakeHTModel()


# ---------------------------------------------------------------------------
# run_headtail_batch tests
# ---------------------------------------------------------------------------


def test_run_headtail_batch_keys_by_frame_and_matches_counts():
    obb0 = _obb(0, 2)
    obb1 = _obb(1, 1)
    batch = _make_batch(obb0, obb1)
    out = run_headtail_batch(batch, _fake_ht_model(), config=HeadTailConfig(model_path="/ht.pt"), runtime=_cpu_rt())
    assert set(out) == {0, 1}
    assert out[0].heading_hints.shape[0] == 2
    assert out[1].heading_hints.shape[0] == 1


def test_run_headtail_batch_heading_values():
    """Up label (index 2) with axis=0 -> heading = 0 + (-pi/2) mod 2pi = 3pi/2."""
    obb0 = _obb(0, 1)
    obb1 = _obb(1, 1)
    batch = _make_batch(obb0, obb1)
    config = HeadTailConfig(model_path="/ht.pt", confidence_threshold=0.5)
    out = run_headtail_batch(batch, _fake_ht_model(), config=config, runtime=_cpu_rt())
    expected = (0.0 + (-math.pi / 2)) % (2 * math.pi)
    assert out[0].heading_hints[0] == pytest.approx(expected)
    assert out[0].directed_mask[0] == 1
    assert out[1].heading_hints[0] == pytest.approx(expected)


def test_run_headtail_batch_empty_frame():
    """Frames with 0 detections get an empty HeadTailResult."""
    obb0 = _obb(0, 0)
    obb1 = _obb(1, 2)
    batch = CropBatch(
        crops=torch.zeros(2, 3, 16, 16),
        detection_ids=obb1.detection_ids.copy(),
        frame_index=np.array([1, 1], np.int64),
        obb_by_frame={0: obb0, 1: obb1},
        native_sizes=np.array([[16, 16]] * 2, np.int64),
    )
    config = HeadTailConfig(model_path="/ht.pt", confidence_threshold=0.5)
    out = run_headtail_batch(batch, _fake_ht_model(), config=config, runtime=_cpu_rt())
    assert out[0].heading_hints.shape[0] == 0
    assert out[1].heading_hints.shape[0] == 2


# ---------------------------------------------------------------------------
# Equivalence: run_headtail_batch == run_headtail per frame
# ---------------------------------------------------------------------------


def test_run_headtail_batch_equivalent_to_per_frame():
    """Batch path must produce the same HeadTailResult as per-frame calls.

    Uses a fake model that returns deterministic results regardless of input,
    so the equivalence test proves structure/assembly rather than crop numerics.
    """
    obb0 = _obb(0, 2)
    obb1 = _obb(1, 3)
    batch = _make_batch(obb0, obb1)
    config = HeadTailConfig(model_path="/ht.pt", confidence_threshold=0.5)
    model = _fake_ht_model()
    rt = _cpu_rt()

    batch_out = run_headtail_batch(batch, model, config=config, runtime=rt)

    # Per-frame: pass a dummy frame (model ignores pixel content in this fake)
    dummy_frame = np.zeros((32, 32, 3), dtype=np.uint8)
    pf0 = run_headtail(dummy_frame, obb0, model, config, rt)
    pf1 = run_headtail(dummy_frame, obb1, model, config, rt)

    np.testing.assert_array_equal(batch_out[0].directed_mask, pf0.directed_mask)
    np.testing.assert_array_equal(batch_out[1].directed_mask, pf1.directed_mask)
    np.testing.assert_array_almost_equal(
        batch_out[0].heading_hints, pf0.heading_hints
    )
    np.testing.assert_array_almost_equal(
        batch_out[1].heading_hints, pf1.heading_hints
    )
    np.testing.assert_array_almost_equal(
        batch_out[0].heading_confidences, pf0.heading_confidences
    )
    np.testing.assert_array_almost_equal(
        batch_out[1].heading_confidences, pf1.heading_confidences
    )


# ---------------------------------------------------------------------------
# run_cnn_batch tests
# ---------------------------------------------------------------------------


def test_run_cnn_batch_keys_by_frame_and_counts():
    obb0 = _obb(0, 2)
    obb1 = _obb(1, 1)
    batch = _make_batch(obb0, obb1)
    config = CNNConfig(label="identity", model_path="/cnn.pt")
    out = run_cnn_batch(batch, _fake_cnn_model(), config=config, runtime=_cpu_rt())
    assert set(out) == {0, 1}
    assert len(out[0].predictions) == 2
    assert len(out[1].predictions) == 1


def test_run_cnn_batch_probabilities():
    obb0 = _obb(0, 1)
    obb1 = _obb(1, 1)
    batch = _make_batch(obb0, obb1)
    config = CNNConfig(label="identity", model_path="/cnn.pt")
    out = run_cnn_batch(batch, _fake_cnn_model(), config=config, runtime=_cpu_rt())
    probs = out[0].predictions[0].factors[0].raw_probabilities
    np.testing.assert_array_almost_equal(
        probs, np.array([1 / 3, 1 / 3, 1 / 3], dtype=np.float32)
    )


def test_run_cnn_batch_equivalent_to_per_frame():
    """Batch path must produce same CNNResult structure as per-frame calls."""
    obb0 = _obb(0, 2)
    obb1 = _obb(1, 3)
    batch = _make_batch(obb0, obb1)
    config = CNNConfig(label="identity", model_path="/cnn.pt")
    model = _fake_cnn_model()
    rt = _cpu_rt()

    batch_out = run_cnn_batch(batch, model, config=config, runtime=rt)

    dummy_frame = np.zeros((32, 32, 3), dtype=np.uint8)
    pf0 = run_cnn(dummy_frame, obb0, model, config, rt)
    pf1 = run_cnn(dummy_frame, obb1, model, config, rt)

    assert len(batch_out[0].predictions) == len(pf0.predictions)
    assert len(batch_out[1].predictions) == len(pf1.predictions)
    for bp, pp in zip(batch_out[0].predictions, pf0.predictions):
        np.testing.assert_array_almost_equal(
            bp.factors[0].raw_probabilities, pp.factors[0].raw_probabilities
        )
    for bp, pp in zip(batch_out[1].predictions, pf1.predictions):
        np.testing.assert_array_almost_equal(
            bp.factors[0].raw_probabilities, pp.factors[0].raw_probabilities
        )


# ---------------------------------------------------------------------------
# scatter / assemble tests
# ---------------------------------------------------------------------------


def test_scatter_builds_one_frame_result_per_frame():
    obb0 = _obb(0, 2)
    obb1 = _obb(1, 1)
    obb_by_frame = {0: obb0, 1: obb1}
    out = scatter(obb_by_frame, headtail=None, cnns=None, pose=None, apriltag=None, config=None)
    assert len(out) == 2
    assert all(isinstance(r, FrameResult) for r in out)
    frame_idxs = [r.frame_idx for r in out]
    assert frame_idxs == [0, 1]


def test_scatter_resolved_headings_obb_fallback():
    """Without headtail/pose, resolved_headings == obb.angles."""
    obb = _obb(0, 2)
    obb.angles[:] = np.array([1.0, 2.0], dtype=np.float32)
    out = scatter({0: obb}, headtail=None, cnns=None, pose=None, apriltag=None, config=None)
    np.testing.assert_array_almost_equal(out[0].resolved_headings, obb.angles)


def test_scatter_headtail_overrides_obb():
    """headtail directed detections override OBB axis in resolved_headings."""
    from hydra_suite.core.inference.result import HeadTailResult

    obb = _obb(0, 1)
    obb.angles[:] = np.array([0.5], dtype=np.float32)
    ht = HeadTailResult(
        heading_hints=np.array([1.5], np.float32),
        heading_confidences=np.array([0.9], np.float32),
        directed_mask=np.array([1], np.uint8),
        canonical_affines=None,
    )
    out = scatter({0: obb}, headtail={0: ht}, cnns=None, pose=None, apriltag=None, config=None)
    assert out[0].resolved_headings[0] == pytest.approx(1.5)


def test_scatter_empty_obb_by_frame():
    out = scatter({}, headtail=None, cnns=None, pose=None, apriltag=None, config=None)
    assert out == []


def test_scatter_sorted_frame_order():
    obb3 = _obb(3, 1)
    obb1 = _obb(1, 1)
    obb0 = _obb(0, 1)
    out = scatter(
        {3: obb3, 1: obb1, 0: obb0},
        headtail=None, cnns=None, pose=None, apriltag=None, config=None
    )
    assert [r.frame_idx for r in out] == [0, 1, 3]


def test_scatter_cnn_passed_through():
    from hydra_suite.core.inference.result import CNNDetectionPrediction, CNNFactorPrediction, CNNResult

    obb = _obb(0, 1)
    cnn_r = CNNResult(
        label="identity",
        predictions=[
            CNNDetectionPrediction(
                det_index=0,
                factors=[
                    CNNFactorPrediction(
                        factor_name="identity",
                        class_names=["a"],
                        raw_probabilities=np.array([1.0], np.float32),
                    )
                ],
            )
        ],
    )
    out = scatter(
        {0: obb},
        headtail=None,
        cnns={0: [cnn_r]},
        pose=None,
        apriltag=None,
        config=None,
    )
    assert out[0].cnn == [cnn_r]
