import pandas as pd

from hydra_suite.data.al.escalation import LabelRecord
from hydra_suite.data.dataset_generation import _select_records_for_frame
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


def test_no_rows_yields_no_records():
    records, drops = _select_records_for_frame(
        None, [_detector_record(100, 100)], PARAMS, 1.0
    )
    assert records == []
    assert drops == {"lost": 0, "unmatched": 0}
