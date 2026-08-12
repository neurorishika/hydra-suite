"""Tests for `IdentityEvidenceStage` (Identity Phase 3, Task 3).

The stage is the inference-time producer that turns raw CNN/AprilTag cache
reads for one frame into `list[IdentityEvidence]`, using the shared
`EvidenceBuilder` (CNN) and `catalog.apriltag_log_prior` (tags). Purely
additive: nothing in the runner calls this yet (Task 4).
"""

from __future__ import annotations

import numpy as np

from hydra_suite.core.individual.identity.catalog import IdentityCatalog
from hydra_suite.core.individual.identity.evidence import EvidenceSource
from hydra_suite.core.individual.identity.evidence_builder import EvidenceBuilder
from hydra_suite.core.inference.result import (
    AprilTagResult,
    CNNDetectionPrediction,
    CNNFactorPrediction,
)
from hydra_suite.core.inference.stages.identity_evidence import IdentityEvidenceStage


def _make_builder(source_name: str = "cnn0") -> tuple[IdentityCatalog, EvidenceBuilder]:
    catalog = IdentityCatalog.from_labels(["white", "black", "brown"])
    builder = EvidenceBuilder(
        catalog,
        source_name,
        [["white", "black", "brown"]],
        runtime_signature="cpu",
    )
    return catalog, builder


def test_cnn_dets_get_from_cnn_evidence_matching_direct_builder_call():
    catalog, builder = _make_builder()
    stage = IdentityEvidenceStage(
        catalog=catalog,
        cnn_builders={"cnn0": builder},
        tag_to_label={},
    )

    # Frame has 3 detection slots (det_index 0..2), stable det_ids below.
    det_ids = [100, 101, 102]

    predictions = [
        CNNDetectionPrediction(
            det_index=0,
            factors=[
                CNNFactorPrediction(
                    factor_name="color",
                    class_names=["white", "black", "brown"],
                    raw_probabilities=np.array([0.7, 0.2, 0.1]),
                )
            ],
        ),
        CNNDetectionPrediction(
            det_index=1,
            factors=[
                CNNFactorPrediction(
                    factor_name="color",
                    class_names=["white", "black", "brown"],
                    raw_probabilities=np.array([0.1, 0.1, 0.8]),
                )
            ],
        ),
    ]
    # det_index 2 has no CNN prediction at all this frame.

    evidences = stage.evidences_for_frame(
        frame_idx=5,
        det_ids=det_ids,
        cnn_reads={"cnn0": predictions},
        tag_read=None,
    )

    assert len(evidences) == 2
    by_det = {e.detection_id: e for e in evidences}
    assert set(by_det) == {100, 101}

    # Directly call the builder to get the expected log_probs.
    direct = builder.build_frame_evidences(
        5,
        [100, 101],
        [
            [np.array([0.7, 0.2, 0.1])],
            [np.array([0.1, 0.1, 0.8])],
        ],
    )
    direct_by_det = {e.detection_id: e for e in direct}

    for det_id in (100, 101):
        got = by_det[det_id]
        expected = direct_by_det[det_id]
        assert got.source == EvidenceSource.CNN
        assert got.source_name == "cnn0"
        assert np.array_equal(got.log_probs, expected.log_probs)
        assert np.array_equal(got.observed_mask, expected.observed_mask)

    # det 102 (no CNN prediction, no tag) is absent, not a `missing()` placeholder.
    assert 102 not in by_det


def test_tagged_dets_get_from_apriltag_evidence_matching_catalog_prior():
    catalog, builder = _make_builder()
    tag_to_label = {7: "black", 9: "brown"}
    stage = IdentityEvidenceStage(
        catalog=catalog,
        cnn_builders={"cnn0": builder},
        tag_to_label=tag_to_label,
    )

    det_ids = [200, 201, 202]

    tag_read = AprilTagResult(
        tag_ids=[7, 9],
        det_indices=[0, 2],
        centers=np.zeros((2, 2)),
        corners=np.zeros((2, 4, 2)),
    )

    evidences = stage.evidences_for_frame(
        frame_idx=8,
        det_ids=det_ids,
        cnn_reads={},
        tag_read=tag_read,
    )

    assert len(evidences) == 2
    by_det = {e.detection_id: e for e in evidences}
    assert set(by_det) == {200, 202}

    expected_7 = catalog.apriltag_log_prior(7, tag_to_label)
    expected_9 = catalog.apriltag_log_prior(9, tag_to_label)

    assert by_det[200].source == EvidenceSource.APRILTAG
    assert by_det[200].source_name == "apriltag"
    assert np.array_equal(by_det[200].log_probs, expected_7)

    assert by_det[202].source == EvidenceSource.APRILTAG
    assert np.array_equal(by_det[202].log_probs, expected_9)

    # det 201 has neither a CNN nor a tag observation -> absent.
    assert 201 not in by_det


def test_cnn_and_tag_evidence_merge_per_frame_and_preserve_ordering():
    catalog, builder = _make_builder()
    tag_to_label = {3: "white"}
    stage = IdentityEvidenceStage(
        catalog=catalog,
        cnn_builders={"cnn0": builder},
        tag_to_label=tag_to_label,
    )

    det_ids = [300, 301]

    predictions = [
        CNNDetectionPrediction(
            det_index=0,
            factors=[
                CNNFactorPrediction(
                    factor_name="color",
                    class_names=["white", "black", "brown"],
                    raw_probabilities=np.array([0.5, 0.3, 0.2]),
                )
            ],
        ),
    ]
    tag_read = AprilTagResult(
        tag_ids=[3],
        det_indices=[1],
        centers=np.zeros((1, 2)),
        corners=np.zeros((1, 4, 2)),
    )

    evidences = stage.evidences_for_frame(
        frame_idx=1,
        det_ids=det_ids,
        cnn_reads={"cnn0": predictions},
        tag_read=tag_read,
    )

    assert [e.source for e in evidences] == [
        EvidenceSource.CNN,
        EvidenceSource.APRILTAG,
    ]
    assert [e.detection_id for e in evidences] == [300, 301]


def test_gap_factor_compaction_matches_builder_contract():
    """A multi-factor CNN read with an empty (`class_names=[]`) middle factor:
    per_det_factor_probs handed to the builder must skip the gap, aligned
    only to the non-empty factors -- mirroring `EvidenceBuilder`'s own
    documented compaction contract.
    """
    catalog = IdentityCatalog.from_labels(["a_c", "a_d", "b_c", "b_d"])
    class_labels_per_factor = [["a", "b"], [], ["c", "d"]]
    builder = EvidenceBuilder(
        catalog,
        "cnn_multi",
        class_labels_per_factor,
        runtime_signature="cpu",
    )
    stage = IdentityEvidenceStage(
        catalog=catalog,
        cnn_builders={"cnn_multi": builder},
        tag_to_label={},
    )

    det_ids = [400]
    predictions = [
        CNNDetectionPrediction(
            det_index=0,
            factors=[
                CNNFactorPrediction(
                    factor_name="f0",
                    class_names=["a", "b"],
                    raw_probabilities=np.array([0.6, 0.4]),
                ),
                CNNFactorPrediction(
                    factor_name="f1_gap",
                    class_names=[],
                    raw_probabilities=np.array([]),
                ),
                CNNFactorPrediction(
                    factor_name="f2",
                    class_names=["c", "d"],
                    raw_probabilities=np.array([0.3, 0.7]),
                ),
            ],
        ),
    ]

    evidences = stage.evidences_for_frame(
        frame_idx=2,
        det_ids=det_ids,
        cnn_reads={"cnn_multi": predictions},
        tag_read=None,
    )

    expected = builder.build_frame_evidences(
        2,
        [400],
        [[np.array([0.6, 0.4]), np.array([0.3, 0.7])]],
    )

    assert len(evidences) == 1
    assert np.array_equal(evidences[0].log_probs, expected[0].log_probs)
    assert np.array_equal(evidences[0].observed_mask, expected[0].observed_mask)
