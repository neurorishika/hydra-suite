"""Tests for StreamingAnalysisPayload.from_frame_result() (Task 17g).

Verifies that the new classmethod correctly converts a FrameResult produced
by InferenceRunner.run_realtime() into a StreamingAnalysisPayload compatible
with the legacy identity analysis workers.
"""

from __future__ import annotations

import numpy as np

from hydra_suite.core.inference.result import FrameResult, HeadTailResult, OBBResult
from hydra_suite.core.tracking.ingest.streaming_payload import StreamingAnalysisPayload


def _make_obb(n: int = 3) -> OBBResult:
    return OBBResult(
        frame_idx=5,
        centroids=np.zeros((n, 2), dtype=np.float32),
        angles=np.full(n, 0.1, dtype=np.float32),
        sizes=np.ones(n, dtype=np.float32),
        shapes=np.ones((n, 2), dtype=np.float32),
        confidences=np.ones(n, dtype=np.float32),
        corners=np.zeros((n, 4, 2), dtype=np.float32),
        detection_ids=OBBResult.make_detection_ids(5, n),
    )


def _make_headtail(n: int = 3) -> HeadTailResult:
    return HeadTailResult(
        heading_hints=np.full(n, 0.5, dtype=np.float32),
        heading_confidences=np.full(n, 0.9, dtype=np.float32),
        directed_mask=np.ones(n, dtype=np.uint8),
        canonical_affines=np.eye(2, 3, dtype=np.float32)[np.newaxis].repeat(n, axis=0),
    )


def _make_frame_result(n: int = 3, with_headtail: bool = True) -> FrameResult:
    obb = _make_obb(n)
    ht = _make_headtail(n) if with_headtail else None
    return FrameResult(
        frame_idx=5,
        obb=obb,
        filtered_indices=list(range(n)),
        headtail=ht,
        cnn=[],
        pose=None,
        apriltag=None,
        resolved_headings=np.zeros(n, dtype=np.float32),
    )


def test_from_frame_result_basic():
    """from_frame_result should produce a StreamingAnalysisPayload."""
    fr = _make_frame_result()
    payload = StreamingAnalysisPayload.from_frame_result(fr)
    assert isinstance(payload, StreamingAnalysisPayload)


def test_from_frame_result_frame_idx():
    """Payload frame_idx should match OBB frame_idx."""
    fr = _make_frame_result()
    payload = StreamingAnalysisPayload.from_frame_result(fr)
    assert payload.frame_idx == 5


def test_from_frame_result_detection_ids():
    """detection_ids should be copied from the OBBResult."""
    fr = _make_frame_result(n=4)
    payload = StreamingAnalysisPayload.from_frame_result(fr)
    np.testing.assert_array_equal(payload.detection_ids, fr.obb.detection_ids)


def test_from_frame_result_headtail_populated():
    """Heading arrays should be populated from HeadTailResult when present."""
    fr = _make_frame_result(with_headtail=True)
    payload = StreamingAnalysisPayload.from_frame_result(fr)
    np.testing.assert_allclose(payload.headtail_heading, fr.headtail.heading_hints)
    np.testing.assert_allclose(
        payload.headtail_confidence, fr.headtail.heading_confidences
    )
    np.testing.assert_array_equal(
        payload.headtail_directed, fr.headtail.directed_mask.astype(bool)
    )


def test_from_frame_result_no_headtail():
    """Heading arrays should be zeros when headtail is None."""
    n = 3
    fr = _make_frame_result(n=n, with_headtail=False)
    payload = StreamingAnalysisPayload.from_frame_result(fr)
    np.testing.assert_array_equal(payload.headtail_heading, np.zeros(n))
    np.testing.assert_array_equal(payload.headtail_confidence, np.zeros(n))
    assert not payload.headtail_directed.any()


def test_from_frame_result_corners_shape():
    """obb_corners should be (D, 4, 2)."""
    n = 2
    fr = _make_frame_result(n=n)
    payload = StreamingAnalysisPayload.from_frame_result(fr)
    assert payload.obb_corners.shape == (n, 4, 2)


def test_from_frame_result_runtime_family_propagated():
    """runtime_family kwarg should be stored on the payload."""
    fr = _make_frame_result()
    payload = StreamingAnalysisPayload.from_frame_result(fr, runtime_family="mps")
    assert payload.runtime_family == "mps"


def test_frame_result_streaming_payload_field_exists():
    """FrameResult.streaming_payload field should exist and default to None."""
    fr = _make_frame_result()
    assert hasattr(fr, "streaming_payload")
    assert fr.streaming_payload is None
