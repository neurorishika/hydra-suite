from types import SimpleNamespace

import numpy as np
import pandas as pd

from hydra_suite.data.al.escalation import LabelRecord
from hydra_suite.data.dataset_generation import (
    _detect_records_for_frames,
    _select_records_for_frame,
)
from hydra_suite.utils.geometry import obb_corners_from_dims
from hydra_suite.utils.geometry_levels import GeometryLevel


def _detector_record(cx, cy):
    return LabelRecord(
        class_id=0,
        confidence=0.9,
        points=obb_corners_from_dims(cx, cy, 44.0, 16.0, 0.0),
        level=GeometryLevel.OBB,
    )


def _rows(entries):
    """entries: list of (x, y, state)."""
    return pd.DataFrame(
        [
            {"FrameID": 0, "TrackID": i, "X": x, "Y": y, "Theta": 0.0, "State": s}
            for i, (x, y, s) in enumerate(entries)
        ]
    )


PARAMS = {"REFERENCE_BODY_SIZE": 20.0}


def test_matched_rows_are_exported():
    rows = _rows([(100.0, 100.0, "tracked"), (300.0, 300.0, "tracked")])
    dets = [_detector_record(101.0, 99.0), _detector_record(299.0, 301.0)]
    records, drops = _select_records_for_frame(rows, dets, PARAMS, 1.0)
    assert len(records) == 2
    assert drops == {"lost": 0, "unmatched": 0}


def test_lost_rows_are_dropped_and_counted():
    rows = _rows([(100.0, 100.0, "tracked"), (300.0, 300.0, "lost")])
    dets = [_detector_record(101.0, 99.0), _detector_record(299.0, 301.0)]
    records, drops = _select_records_for_frame(rows, dets, PARAMS, 1.0)
    assert len(records) == 1
    assert drops["lost"] == 1


def test_unmatched_rows_are_dropped_not_fabricated():
    """The legacy exporter invented a ref*2.2 x ref*0.8 box here."""
    rows = _rows([(100.0, 100.0, "tracked"), (2000.0, 2000.0, "tracked")])
    dets = [_detector_record(101.0, 99.0)]
    records, drops = _select_records_for_frame(rows, dets, PARAMS, 1.0)
    assert len(records) == 1
    assert drops["unmatched"] == 1


def test_two_rows_cannot_bind_the_same_detection():
    """Greedy nearest-centre matching emitted duplicate identical boxes here."""
    rows = _rows([(100.0, 100.0, "tracked"), (104.0, 101.0, "tracked")])
    dets = [_detector_record(102.0, 100.0)]
    records, drops = _select_records_for_frame(rows, dets, PARAMS, 1.0)
    assert len(records) == 1
    assert drops["unmatched"] == 1


def test_match_radius_scales_with_reference_body_size():
    """A 120 px offset matches for a large animal and not for a small one."""
    rows = _rows([(100.0, 100.0, "tracked")])
    dets = [_detector_record(220.0, 100.0)]

    _small, small_drops = _select_records_for_frame(
        rows, dets, {"REFERENCE_BODY_SIZE": 8.0}, 1.0
    )
    large, large_drops = _select_records_for_frame(
        rows, dets, {"REFERENCE_BODY_SIZE": 90.0}, 1.0
    )
    assert small_drops["unmatched"] == 1
    assert len(large) == 1
    assert large_drops["unmatched"] == 0


def test_nan_positions_are_dropped_as_unmatched():
    rows = _rows([(float("nan"), float("nan"), "tracked")])
    records, drops = _select_records_for_frame(
        rows, [_detector_record(100, 100)], PARAMS, 1.0
    )
    assert records == []
    assert drops["unmatched"] == 1


def test_radius_gate_is_part_of_the_assignment_not_a_post_filter():
    """A global-min-cost assignment can strand a row on an out-of-radius
    detection even though a fully in-radius perfect matching exists --
    filtering by radius AFTER solving misses it. The optimizer must be
    steered away from out-of-radius pairs before solving.

    This configuration was found by search: max_distance == 10.0 (from
    REFERENCE_BODY_SIZE = 10.0 / 2.2). The naive "solve raw distances, then
    filter" approach assigns row1 to det2 at cost ~10.216 (over-radius, so it
    would have been dropped), even though the fully in-radius perfect
    matching row0-det0 / row1-det1 / row2-det2 (all costs < 10.0) exists and
    is what a radius-aware assignment must find.
    """
    rows = _rows(
        [
            (5.08946833, -2.12965478, "tracked"),
            (6.66017136, 5.9763788, "tracked"),
            (6.82463976, 2.20762837, "tracked"),
        ]
    )
    dets = [
        _detector_record(14.6906931, -1.90484266),
        _detector_record(0.442393544, -0.950975386),
        _detector_record(14.9494817, 0.00505408915),
    ]
    records, drops = _select_records_for_frame(
        rows, dets, {"REFERENCE_BODY_SIZE": 10.0 / 2.2}, 1.0
    )
    assert len(records) == 3
    assert drops["unmatched"] == 0


def _fake_obb_result(cx, cy, w=44.0, h=16.0, theta=0.0):
    """A minimal OBBResult-alike carrying one detection at (cx, cy) in
    whatever pixel space the caller's fake image was in (resized space for
    this test)."""
    from hydra_suite.core.inference.result import OBBResult

    corners = obb_corners_from_dims(cx, cy, w, h, theta).reshape(1, 4, 2)
    return OBBResult(
        frame_idx=0,
        centroids=np.array([[cx, cy]], dtype=np.float32),
        angles=np.array([theta], dtype=np.float32),
        sizes=np.array([w * h], dtype=np.float32),
        shapes=np.array([[w * h, w / h]], dtype=np.float32),
        confidences=np.array([0.9], dtype=np.float32),
        corners=corners,
        detection_ids=OBBResult.make_detection_ids(0, 1),
        class_ids=np.array([0], dtype=np.int64),
    )


class _FakeRunner:
    """Stands in for InferenceRunner: `detect_batch_raw` returns one canned
    OBBResult (in resized-frame space) per requested frame, regardless of
    image content -- the point of this test is coordinate scaling, not
    detection itself.

    `detection_source="bgsub"` makes `filter_for_source` the identity, so
    the canned result passes through unfiltered exactly like the pre-cache
    `detect_batch` mock this replaces.
    """

    def __init__(self, result_by_frame, cache_dir=None):
        self._result_by_frame = result_by_frame
        self.config = SimpleNamespace(detection_source="bgsub")
        self.cache_dir = cache_dir
        self._roi_mask = None

    def detect_batch_raw(self, frames, frame_indices=None, roi_mask=None):
        return [self._result_by_frame[fid] for fid in frame_indices]


def test_detector_points_are_scaled_back_to_original_frame_space(tmp_path):
    """Regression: detection runs on the RESIZE_FACTOR-scaled frame, so raw
    obb corners come back in resized-frame space. The deleted legacy
    `_measurements_to_detections` used to scale them back by 1/resize_factor;
    the rewrite dropped that step, silently comparing resized-space detector
    geometry against original-space CSV rows (and mislabeling with the
    wrong-space geometry for any row that did survive by fluke).

    A detection at (55, 30) in a RESIZE_FACTOR=0.5 detection frame should
    come back as a LabelRecord centered at (110, 60) in original-frame space
    -- and a CSV row (already original-space) at (110, 60) should then bind
    to it under normal-sized radii.
    """
    resize_factor = 0.5
    obb_result = _fake_obb_result(55.0, 30.0)
    runner = _FakeRunner({0: obb_result}, cache_dir=tmp_path / "cache")

    frames = {0: np.zeros((3, 3, 3), dtype=np.uint8)}
    records_by_frame, stats = _detect_records_for_frames(
        runner, frames, {"RESIZE_FACTOR": resize_factor}, GeometryLevel.OBB
    )
    assert stats["detection_failed"] == 0

    records = records_by_frame[0]
    assert len(records) == 1
    center = records[0].points.mean(axis=0)
    np.testing.assert_allclose(center, [110.0, 60.0], atol=1e-3)

    # And the original-space CSV row now matches it under strict selection.
    rows = _rows([(110.0, 60.0, "tracked")])
    matched, drops = _select_records_for_frame(rows, records, PARAMS, 1.0)
    assert len(matched) == 1
    assert drops == {"lost": 0, "unmatched": 0}


def test_no_rows_yields_no_records():
    records, drops = _select_records_for_frame(
        None, [_detector_record(100, 100)], PARAMS, 1.0
    )
    assert records == []
    assert drops == {"lost": 0, "unmatched": 0}
