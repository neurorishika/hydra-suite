"""Tests for batch-native head-tail / CNN / pose stages and scatter assembly.

Fakes align to the REAL backend predict_batch contract:
  - predict_batch(crops) -> list-per-detection of list-per-factor of np.ndarray
  - e.g. flat headtail: [[np.array([p_right, p_left, p_up, p_down])], ...]

The equivalence tests use crop-content-sensitive fake backends: each crop's output
is keyed by its pixel mean so that if the batch path feeds different crops than
the per-frame path (or mis-splits them), the test fails.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from hydra_suite.core.inference.config import CNNConfig, HeadTailConfig
from hydra_suite.core.inference.result import FrameResult, OBBResult
from hydra_suite.core.inference.runtime import RuntimeContext
from hydra_suite.core.inference.stages.assemble import scatter
from hydra_suite.core.inference.stages.cnn import CNNModel, run_cnn, run_cnn_batch
from hydra_suite.core.inference.stages.headtail import run_headtail, run_headtail_batch

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _cpu_rt():
    return RuntimeContext(
        cuda_mode=False,
        device="cpu",
        use_nvdec=False,
        tensor_on_cuda=False,
    )


def _obb(frame_idx: int, n: int, distinct: bool = False) -> OBBResult:
    """Build an OBBResult with non-degenerate corners so warpAffine produces real crops.

    When distinct=True, each detection's box is placed at a distinct x-offset so
    the warped crop has a different pixel mean — letting content-sensitive backends
    distinguish them.
    """
    if n == 0:
        return OBBResult(
            frame_idx=frame_idx,
            centroids=np.zeros((0, 2), np.float32),
            angles=np.zeros(0, np.float32),
            sizes=np.zeros(0, np.float32),
            shapes=np.zeros((0, 2), np.float32),
            confidences=np.zeros(0, np.float32),
            corners=np.zeros((0, 4, 2), np.float32),
            detection_ids=np.zeros(0, np.int64),
        )

    # Place each detection at a different x-center so crops differ when distinct=True
    cx = np.linspace(20.0, 20.0 + 20.0 * (n - 1), n).astype(np.float32)
    cy = np.full(n, 30.0, np.float32)
    half_w, half_h = 8.0, 4.0  # non-square so axis is unambiguous
    corners = np.stack(
        [
            np.stack([cx - half_w, cy - half_h], -1),
            np.stack([cx + half_w, cy - half_h], -1),
            np.stack([cx + half_w, cy + half_h], -1),
            np.stack([cx - half_w, cy + half_h], -1),
        ],
        axis=1,
    ).astype(np.float32)
    return OBBResult(
        frame_idx=frame_idx,
        centroids=np.stack([cx, cy], -1),
        angles=np.zeros(n, np.float32),
        sizes=np.full(n, 64.0, np.float32),
        shapes=np.ones((n, 2), np.float32),
        confidences=np.ones(n, np.float32),
        corners=corners,
        detection_ids=np.array([frame_idx * 10000 + s for s in range(n)], np.int64),
    )


def _frame_for_obb(obb: OBBResult, value: int = 128) -> np.ndarray:
    """Build a 100×100 frame with a distinct fill value per detection, for content sensitivity."""
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    for i in range(obb.num_detections):
        cx = int(obb.centroids[i, 0])
        cy = int(obb.centroids[i, 1])
        r = 6
        y0, y1 = max(0, cy - r), min(100, cy + r)
        x0, x1 = max(0, cx - r), min(100, cx + r)
        # unique fill per detection: value + detection slot index
        fill = min(255, value + i * 30)
        frame[y0:y1, x0:x1] = fill
    return frame


# ---------------------------------------------------------------------------
# Fake models: content-INSENSITIVE (structure/assembly tests)
# ---------------------------------------------------------------------------

# predict_batch returns: list[list[np.ndarray]]  (detections × factors)
# For a flat 4-class headtail model: [[np.array([p0,p1,p2,p3])], ...]


class _FakeHTBackend:
    """Returns prob-array with "up" (index 2) winning at 0.9 for every crop."""

    def predict_batch(self, crops):
        # up is index 2 in [right, left, up, down]
        return [[np.array([0.05, 0.05, 0.9, 0.0], dtype=np.float32)] for _ in crops]


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
# Fake models: content-SENSITIVE (equivalence tests)
# These backends key their output on each crop's pixel mean so that if the
# batch path feeds different crops or mis-splits them, deep-equal checks fail.
# ---------------------------------------------------------------------------


class _ContentHTBackend:
    """Head-tail backend whose output encodes the crop's pixel mean as confidence.

    Output: [[np.array([mean, 1-mean, 0, 0])]] so the "right" direction
    wins with confidence = mean. Two crops with different means get different
    confidences, making the test sensitive to crop content.
    """

    def predict_batch(self, crops):
        results = []
        for crop in crops:
            m = float(np.mean(crop)) / 255.0
            # right wins if m > 0.5, left wins otherwise; conf = max(m, 1-m)
            probs = np.array([m, 1.0 - m, 0.0, 0.0], dtype=np.float32)
            results.append([probs])
        return results


class _ContentCNNBackend:
    """CNN backend whose output encodes the crop's pixel mean.

    Output: [[np.array([mean, (1-mean)/2, (1-mean)/2])]]
    Two crops with different means get different raw_probabilities.
    """

    def predict_batch(self, crops):
        results = []
        for crop in crops:
            m = float(np.mean(crop)) / 255.0
            probs = np.array([m, (1.0 - m) / 2.0, (1.0 - m) / 2.0], dtype=np.float32)
            results.append([probs])
        return results


def _content_ht_model():
    model = _FakeHTModel.__new__(_FakeHTModel)
    model.backend = _ContentHTBackend()
    model.input_size = (16, 16)
    model.class_names = ["right", "left", "up", "down"]
    return model


def _content_cnn_model():
    return CNNModel(
        backend=_ContentCNNBackend(),
        input_size=(16, 16),
        factor_names=["identity"],
        factor_class_names=[["a", "b", "c"]],
    )


# ---------------------------------------------------------------------------
# run_headtail_batch tests (structure / assembly)
# ---------------------------------------------------------------------------


def test_run_headtail_batch_keys_by_frame_and_matches_counts():
    obb0 = _obb(0, 2)
    obb1 = _obb(1, 1)
    frames = [np.zeros((100, 100, 3), np.uint8), np.zeros((100, 100, 3), np.uint8)]
    out = run_headtail_batch(
        frames,
        [obb0, obb1],
        _fake_ht_model(),
        config=HeadTailConfig(model_path="/ht.pt"),
        runtime=_cpu_rt(),
    )
    assert set(out) == {0, 1}
    assert out[0].heading_hints.shape[0] == 2
    assert out[1].heading_hints.shape[0] == 1


def test_run_headtail_batch_heading_values():
    """Up label (index 2) with axis=0 -> heading = 0 + (-pi/2) mod 2pi = 3pi/2."""
    obb0 = _obb(0, 1)
    obb1 = _obb(1, 1)
    frames = [np.zeros((100, 100, 3), np.uint8), np.zeros((100, 100, 3), np.uint8)]
    config = HeadTailConfig(model_path="/ht.pt", confidence_threshold=0.5)
    out = run_headtail_batch(
        frames, [obb0, obb1], _fake_ht_model(), config=config, runtime=_cpu_rt()
    )
    expected = (0.0 + (-math.pi / 2)) % (2 * math.pi)
    assert out[0].heading_hints[0] == pytest.approx(expected)
    assert out[0].directed_mask[0] == 1
    assert out[1].heading_hints[0] == pytest.approx(expected)


def test_run_headtail_batch_empty_frame():
    """Frames with 0 detections get an empty HeadTailResult."""
    obb0 = _obb(0, 0)
    obb1 = _obb(1, 2)
    frames = [np.zeros((100, 100, 3), np.uint8), np.zeros((100, 100, 3), np.uint8)]
    config = HeadTailConfig(model_path="/ht.pt", confidence_threshold=0.5)
    out = run_headtail_batch(
        frames, [obb0, obb1], _fake_ht_model(), config=config, runtime=_cpu_rt()
    )
    assert out[0].heading_hints.shape[0] == 0
    assert out[1].heading_hints.shape[0] == 2


# ---------------------------------------------------------------------------
# Equivalence: run_headtail_batch == run_headtail per frame
# Uses crop-content-sensitive backend to catch crop-path divergence.
# ---------------------------------------------------------------------------


def test_run_headtail_batch_equivalent_to_per_frame():
    """Batch path must produce per-frame HeadTailResult deep-equal to per-frame calls.

    Uses _ContentHTBackend whose output encodes each crop's pixel mean.
    If the batch path feeds different crops (e.g. canonical instead of classifier
    crops, or mis-splits the window), confidences will differ and the test fails.
    """
    obb0 = _obb(0, 2, distinct=True)
    obb1 = _obb(1, 3, distinct=True)
    # Give frames distinct pixel values so crops differ across frames
    f0 = _frame_for_obb(obb0, value=60)
    f1 = _frame_for_obb(obb1, value=160)
    config = HeadTailConfig(model_path="/ht.pt", confidence_threshold=0.0)
    model = _content_ht_model()
    rt = _cpu_rt()

    batch_out = run_headtail_batch(
        [f0, f1], [obb0, obb1], model, config=config, runtime=rt
    )

    pf0 = run_headtail(f0, obb0, model, config, rt)
    pf1 = run_headtail(f1, obb1, model, config, rt)

    np.testing.assert_array_equal(batch_out[0].directed_mask, pf0.directed_mask)
    np.testing.assert_array_equal(batch_out[1].directed_mask, pf1.directed_mask)
    np.testing.assert_array_almost_equal(batch_out[0].heading_hints, pf0.heading_hints)
    np.testing.assert_array_almost_equal(batch_out[1].heading_hints, pf1.heading_hints)
    np.testing.assert_array_almost_equal(
        batch_out[0].heading_confidences, pf0.heading_confidences
    )
    np.testing.assert_array_almost_equal(
        batch_out[1].heading_confidences, pf1.heading_confidences
    )


# ---------------------------------------------------------------------------
# run_cnn_batch tests (structure / assembly)
# ---------------------------------------------------------------------------


def test_run_cnn_batch_keys_by_frame_and_counts():
    obb0 = _obb(0, 2)
    obb1 = _obb(1, 1)
    frames = [np.zeros((100, 100, 3), np.uint8), np.zeros((100, 100, 3), np.uint8)]
    config = CNNConfig(label="identity", model_path="/cnn.pt")
    out = run_cnn_batch(
        frames, [obb0, obb1], _fake_cnn_model(), config=config, runtime=_cpu_rt()
    )
    assert set(out) == {0, 1}
    assert len(out[0].predictions) == 2
    assert len(out[1].predictions) == 1


def test_run_cnn_batch_probabilities():
    obb0 = _obb(0, 1)
    obb1 = _obb(1, 1)
    frames = [np.zeros((100, 100, 3), np.uint8), np.zeros((100, 100, 3), np.uint8)]
    config = CNNConfig(label="identity", model_path="/cnn.pt")
    out = run_cnn_batch(
        frames, [obb0, obb1], _fake_cnn_model(), config=config, runtime=_cpu_rt()
    )
    probs = out[0].predictions[0].factors[0].raw_probabilities
    np.testing.assert_array_almost_equal(
        probs, np.array([1 / 3, 1 / 3, 1 / 3], dtype=np.float32)
    )


# ---------------------------------------------------------------------------
# Equivalence: run_cnn_batch == run_cnn per frame
# Uses crop-content-sensitive backend to catch crop-path divergence.
# ---------------------------------------------------------------------------


def test_run_cnn_batch_equivalent_to_per_frame():
    """Batch path must produce per-frame CNNResult deep-equal to per-frame calls.

    Uses _ContentCNNBackend whose output encodes each crop's pixel mean.
    If the batch path feeds different crops or mis-splits them, raw_probabilities
    will differ and the test fails.
    """
    obb0 = _obb(0, 2, distinct=True)
    obb1 = _obb(1, 3, distinct=True)
    f0 = _frame_for_obb(obb0, value=60)
    f1 = _frame_for_obb(obb1, value=160)
    config = CNNConfig(label="identity", model_path="/cnn.pt")
    model = _content_cnn_model()
    rt = _cpu_rt()

    batch_out = run_cnn_batch([f0, f1], [obb0, obb1], model, config=config, runtime=rt)

    pf0 = run_cnn(f0, obb0, model, config, rt)
    pf1 = run_cnn(f1, obb1, model, config, rt)

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
    out = scatter(
        obb_by_frame, headtail=None, cnns=None, pose=None, apriltag=None, config=None
    )
    assert len(out) == 2
    assert all(isinstance(r, FrameResult) for r in out)
    frame_idxs = [r.frame_idx for r in out]
    assert frame_idxs == [0, 1]


def test_scatter_resolved_headings_obb_fallback():
    """Without headtail/pose, resolved_headings == obb.angles."""
    obb = _obb(0, 2)
    obb.angles[:] = np.array([1.0, 2.0], dtype=np.float32)
    out = scatter(
        {0: obb}, headtail=None, cnns=None, pose=None, apriltag=None, config=None
    )
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
    out = scatter(
        {0: obb}, headtail={0: ht}, cnns=None, pose=None, apriltag=None, config=None
    )
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
        headtail=None,
        cnns=None,
        pose=None,
        apriltag=None,
        config=None,
    )
    assert [r.frame_idx for r in out] == [0, 1, 3]


def test_scatter_cnn_passed_through():
    from hydra_suite.core.inference.result import (
        CNNDetectionPrediction,
        CNNFactorPrediction,
        CNNResult,
    )

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
