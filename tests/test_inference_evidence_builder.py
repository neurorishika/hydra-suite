"""Tests for IdentityEvidenceBuilder (Task 16).

Uses the corrected Phase 1 result types (CNNResult.predictions list, OBBResult
with detection_ids). See "Phase 5 IdentityEvidenceBuilder corrections" in plan.
"""

from unittest.mock import MagicMock

import numpy as np

from hydra_suite.core.inference.config import CNNConfig
from hydra_suite.core.inference.result import (
    AprilTagResult,
    CNNDetectionPrediction,
    CNNFactorPrediction,
    CNNResult,
    FrameResult,
    OBBResult,
)


def _make_frame_result(
    n_dets: int = 3,
    raw_probs_per_det: (
        list | None
    ) = None,  # list[np.ndarray] one (num_classes,) per det
    factor_name: str = "identity",
    class_names: list | None = None,
    apriltag_det_indices: list | None = None,
    apriltag_tag_ids: list | None = None,
) -> FrameResult:
    rng = np.random.default_rng(0)
    _class_names = class_names or ["ant_A", "ant_B", "ant_C"]
    n_classes = len(_class_names)
    w = rng.uniform(10, 50, n_dets).astype(np.float32)
    h = rng.uniform(20, 80, n_dets).astype(np.float32)
    obb = OBBResult(
        frame_idx=0,
        centroids=rng.uniform(0, 640, (n_dets, 2)).astype(np.float32),
        angles=rng.uniform(0, np.pi, n_dets).astype(np.float32),
        sizes=(w * h).astype(np.float32),
        shapes=np.stack([w * h, h / w], axis=1).astype(np.float32),
        confidences=np.full(n_dets, 0.9, dtype=np.float32),
        corners=rng.uniform(0, 640, (n_dets, 4, 2)).astype(np.float32),
        detection_ids=OBBResult.make_detection_ids(0, n_dets),
    )
    probs_list = raw_probs_per_det or [
        rng.dirichlet(np.ones(n_classes)).astype(np.float32) for _ in range(n_dets)
    ]
    predictions = [
        CNNDetectionPrediction(
            det_index=i,
            factors=[
                CNNFactorPrediction(
                    factor_name=factor_name,
                    class_names=_class_names,
                    raw_probabilities=probs_list[i],
                )
            ],
        )
        for i in range(n_dets)
    ]
    cnn = CNNResult(label="identity", predictions=predictions)

    apriltag = None
    if apriltag_det_indices is not None:
        n_tags = len(apriltag_det_indices)
        apriltag = AprilTagResult(
            tag_ids=apriltag_tag_ids or list(range(n_tags)),
            det_indices=apriltag_det_indices,
            centers=rng.uniform(0, 640, (n_tags, 2)).astype(np.float32),
            corners=rng.uniform(0, 640, (n_tags, 4, 2)).astype(np.float32),
        )

    return FrameResult(
        frame_idx=0,
        obb=obb,
        filtered_indices=list(range(n_dets)),
        headtail=None,
        cnn=[cnn],
        pose=None,
        apriltag=apriltag,
        resolved_headings=np.zeros(n_dets, dtype=np.float32),
    )


def _make_builder(label: str = "identity", temperature: float = 1.0):
    from hydra_suite.core.tracking.identity.evidence import IdentityEvidenceBuilder

    cfg = CNNConfig(
        label=label, model_path="/tmp/model.pt", calibration_temperature=temperature
    )
    catalog = MagicMock()
    catalog.get_label.return_value = None
    return IdentityEvidenceBuilder(cfg, catalog, phase_index=0)


def test_evidence_builder_produces_one_detection_per_obb():
    builder = _make_builder()
    frame = _make_frame_result(n_dets=3)
    evidence = builder.build(frame)
    assert len(evidence.detections) == 3


def test_evidence_builder_calibration_temperature_one_is_identity():
    # t=1 should not change winner
    rng = np.random.default_rng(42)
    probs = [rng.dirichlet([5, 1, 1]).astype(np.float32) for _ in range(2)]
    frame = _make_frame_result(n_dets=2, raw_probs_per_det=probs)
    builder = _make_builder(temperature=1.0)
    evidence = builder.build(frame)
    for det in evidence.detections:
        assert det.cnn_factors is not None
        f = det.cnn_factors[0]
        assert f.winning_class == "ant_A"  # highest prior always wins


def test_evidence_builder_high_temperature_flattens_distribution():
    # t >> 1 makes distribution uniform; very high temperature should reduce confidence
    probs = [np.array([0.99, 0.005, 0.005], dtype=np.float32)]
    frame = _make_frame_result(n_dets=1, raw_probs_per_det=probs)
    hot_builder = _make_builder(temperature=100.0)
    cold_builder = _make_builder(temperature=0.1)
    hot_ev = hot_builder.build(frame)
    cold_ev = cold_builder.build(frame)
    assert (
        hot_ev.detections[0].cnn_factors[0].confidence
        < cold_ev.detections[0].cnn_factors[0].confidence
    )


def test_apriltag_overrides_cnn_when_both_present():
    from hydra_suite.core.tracking.identity.evidence import IdentityEvidenceBuilder

    cfg = CNNConfig(label="identity", model_path="/tmp/model.pt")
    catalog = MagicMock()
    catalog.get_label.return_value = "ant_A"  # tag maps to ant_A
    builder = IdentityEvidenceBuilder(cfg, catalog, phase_index=0)

    frame = _make_frame_result(n_dets=2, apriltag_det_indices=[0], apriltag_tag_ids=[7])
    evidence = builder.build(frame)

    det0 = evidence.detections[0]
    assert det0.is_authoritative is True
    assert det0.apriltag_label == "ant_A"
    assert det0.apriltag_tag_id == 7
    assert det0.resolved_label == "ant_A"

    det1 = evidence.detections[1]
    assert det1.is_authoritative is False
    assert det1.apriltag_label is None


def test_no_cnn_result_gives_none_factors():
    from hydra_suite.core.tracking.identity.evidence import IdentityEvidenceBuilder

    cfg = CNNConfig(label="identity", model_path="/tmp/model.pt")
    catalog = MagicMock()
    catalog.get_label.return_value = None
    builder = IdentityEvidenceBuilder(cfg, catalog, phase_index=5)  # index out of range

    frame = _make_frame_result(n_dets=2)
    evidence = builder.build(frame)
    for det in evidence.detections:
        assert det.cnn_factors is None
        assert det.resolved_label is None


def test_evidence_phase_label_matches_config():
    builder = _make_builder(label="my_phase")
    frame = _make_frame_result(n_dets=1)
    evidence = builder.build(frame)
    assert evidence.phase_label == "my_phase"


def test_full_calibrated_distribution_preserved():
    """Correction 17: full probability vector must be passed through, not just top-1."""
    probs = [np.array([0.1, 0.2, 0.7], dtype=np.float32)]
    frame = _make_frame_result(n_dets=1, raw_probs_per_det=probs)
    builder = _make_builder(temperature=1.0)
    evidence = builder.build(frame)

    factor = evidence.detections[0].cnn_factors[0]
    # Must have 3 elements — full distribution preserved
    assert len(factor.calibrated_probabilities) == 3
    # Must sum to 1
    np.testing.assert_allclose(factor.calibrated_probabilities.sum(), 1.0, atol=1e-5)
    # Winner is ant_C (index 2)
    assert factor.winning_class == "ant_C"
