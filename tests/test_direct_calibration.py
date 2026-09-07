from __future__ import annotations

import numpy as np
import pytest

from hydra_suite.core.inference.direct_calibration import (
    CalibrationDetection,
    match_frame,
    score_frames,
)


def _box(x0, y0, x1, y1, *, class_id=0, confidence=1.0):
    return CalibrationDetection(
        class_id=class_id,
        confidence=confidence,
        polygon_px=np.asarray(
            [[x0, y0], [x1, y0], [x1, y1], [x0, y1]], dtype=np.float32
        ),
    )


def test_score_is_class_aware_and_one_to_one_with_duplicate_reporting():
    labels = [_box(0, 0, 10, 10), _box(20, 0, 30, 10, class_id=1)]
    predictions = [
        _box(0, 0, 10, 10),
        _box(1, 0, 11, 10),  # same class and object: cross-tile duplicate
        _box(20, 0, 30, 10, class_id=0),  # wrong class
    ]
    score = match_frame(predictions, labels)
    assert score.matched == 1
    assert score.missed == 1
    assert score.extra == 2
    assert score.duplicate == 1


def test_frame_aggregate_reports_precision_recall_f1_and_localization_quality():
    score = score_frames(
        [
            ([_box(0, 0, 10, 10)], [_box(0, 0, 10, 10)]),
            ([_box(30, 30, 40, 40)], [_box(0, 0, 10, 10)]),
        ]
    )
    assert score.frames == 2
    assert score.matched == score.missed == score.extra == 1
    assert score.precision == score.recall == score.f1 == 0.5
    assert score.mean_iou == 0.5


from hydra_suite.core.inference.direct_calibration import (
    MIN_MATCHED_INSTANCES,
    RECOMMENDATION_RULE,
    CalibrationScore,
    DirectCalibrationPoint,
    recommend_balanced,
)


def _point(
    label,
    seconds,
    *,
    matched=200,
    missed=10,
    extra=10,
    mean_iou=0.8,
    mean_quality=0.7,
    failed="",
    tiles=9,
    confidence=0.35,
):
    """Build a scored point with SELF-CONSISTENT precision/recall/F1.

    The pre-D8 fixture set precision = recall = f1 = a hand-passed number
    independent of the counts. That was harmless while F1 was the target;
    under a recall-first rule it would let a test assert on a recall the
    counts do not support, so the rates are derived from the counts here.
    """
    precision = matched / (matched + extra) if matched + extra else 0.0
    recall = matched / (matched + missed) if matched + missed else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if precision + recall else 0.0
    score = CalibrationScore(
        frames=20,
        matched=matched,
        missed=missed,
        extra=extra,
        precision=precision,
        recall=recall,
        f1=f1,
        duplicate=0,
        mean_iou=mean_iou,
        mean_quality=mean_quality,
    )
    return DirectCalibrationPoint(
        label=label,
        enabled=True,
        geometry_mode="auto_object",
        tile_width=640,
        tile_height=640,
        overlap=0.2,
        object_tile_fraction=0.4,
        max_detections=64,
        tiles_per_frame=tiles,
        seconds_per_frame=seconds,
        confidence=confidence,
        merge_policy="greedy_nmm",
        merge_metric="ios",
        merge_threshold=0.5,
        merge_backend="cv2",
        score=score,
        failed_reason=failed,
    )


def test_recommendation_takes_the_cheapest_point_clearing_every_floor():
    best, reason = recommend_balanced(
        [
            _point("slow", 2.0, missed=8, extra=8),
            _point("fast", 0.4, missed=9, extra=9),
            _point("low_recall", 0.1, matched=100, missed=100, extra=1),
        ]
    )
    assert best.label == "fast"
    assert RECOMMENDATION_RULE in reason


def test_f1_no_longer_influences_selection():
    """D8's load-bearing assertion. ``lower_f1_faster`` has a strictly worse
    F1 (many more extras) but clears every floor and is cheaper, so under
    the retired F1-tolerance rule -- 0.741 vs 0.952, far outside 0.01 -- the
    high-F1 point won. Recall-first must now pick the cheap one. F1 is still
    reported on both points, and still appears in the explanation.
    """
    high_f1 = _point("high_f1_slow", 2.0, matched=200, missed=10, extra=10)
    lower_f1 = _point("lower_f1_faster", 0.4, matched=200, missed=10, extra=130)
    assert lower_f1.score.f1 < high_f1.score.f1 - 0.01
    assert lower_f1.score.recall == high_f1.score.recall
    best, reason = recommend_balanced([high_f1, lower_f1])
    assert best.label == "lower_f1_faster"
    assert "F1" in reason  # retired as a target, still reported


def test_recall_below_the_floor_is_refused_outright():
    best, reason = recommend_balanced(
        [_point("precise_but_blind", 0.1, matched=100, missed=100, extra=0)]
    )
    assert best is None
    assert "recall" in reason


def test_failed_and_undersampled_points_are_never_recommended():
    best, reason = recommend_balanced(
        [
            _point("broken", 0.1, failed="tile budget exceeded"),
            _point("thin", 0.1, matched=MIN_MATCHED_INSTANCES - 1, missed=1, extra=1),
        ]
    )
    assert best is None
    assert "matched instances" in reason


def test_poor_match_quality_is_excluded_even_at_perfect_recall():
    best, reason = recommend_balanced(
        [_point("mistargeted", 0.1, missed=0, mean_quality=0.2)]
    )
    assert best is None
    assert "quality" in reason
    assert "Mistargeted" in reason
    assert "never measured" not in reason


def test_never_measured_quality_is_not_reported_as_mistargeting():
    """Finding 4: a profile with no ``mean_quality`` (pre-D8) must refuse
    with an honest "never measured" reason, not the "mistargeted" claim --
    that claim asserts something about detection geometry that was never
    actually checked for this profile."""
    best, reason = recommend_balanced(
        [_point("unmeasured", 0.1, missed=0, mean_quality=None)]
    )
    assert best is None
    assert "never measured" in reason
    assert "Mistargeted" not in reason
    assert "probably covering the wrong thing" not in reason


def test_never_measured_and_genuinely_bad_quality_are_distinguishable():
    """A genuine mean_quality of 0.0 (measured, and bad) must still be
    reported as mistargeting -- distinct from an absent measurement."""
    best, reason = recommend_balanced(
        [_point("measured_zero", 0.1, missed=0, mean_quality=0.0)]
    )
    assert best is None
    assert "Mistargeted" in reason
    assert "never measured" not in reason


def test_a_qualifying_point_beats_a_higher_quality_slower_one():
    """Cost is the objective among survivors; quality is a FLOOR, not a
    ranking term -- otherwise the floors would be applied twice."""
    best, _reason = recommend_balanced(
        [
            _point("pristine_slow", 2.0, mean_quality=0.95),
            _point("adequate_fast", 0.5, mean_quality=0.40),
        ]
    )
    assert best.label == "adequate_fast"


def test_equal_speed_ties_break_on_fewer_tiles_then_higher_confidence():
    best, _reason = recommend_balanced(
        [
            _point("many_tiles", 1.0, tiles=16, confidence=0.9),
            _point("few_tiles", 1.0, tiles=4, confidence=0.2),
        ]
    )
    assert best.label == "few_tiles"
    best, _reason = recommend_balanced(
        [
            _point("timid", 1.0, tiles=4, confidence=0.2),
            _point("confident", 1.0, tiles=4, confidence=0.8),
        ]
    )
    assert best.label == "confident"


def test_empty_input_refuses_rather_than_raising():
    best, reason = recommend_balanced([])
    assert best is None and RECOMMENDATION_RULE in reason


def test_axis_aligned_matching_counts_crowded_boxes_one_to_one():
    import numpy as np

    from hydra_suite.core.inference.direct_calibration import (
        CalibrationDetection,
        match_frame,
    )

    def box(x, y, w=10, h=10):
        return np.array([[x, y], [x + w, y], [x + w, y + h], [x, y + h]], np.float32)

    labels = [CalibrationDetection(0, box(0, 0)), CalibrationDetection(0, box(30, 0))]
    predictions = [
        CalibrationDetection(0, box(0, 0)),
        CalibrationDetection(0, box(1, 1)),  # duplicate on label 0
        CalibrationDetection(0, box(100, 100)),  # extra
    ]
    score = match_frame(predictions, labels, task="detect")
    assert score.matched == 1
    assert score.missed == 1
    assert score.extra == 2
    assert score.duplicate == 1


def test_segment_polygons_match_on_mask_overlap():
    import numpy as np

    from hydra_suite.core.inference.direct_calibration import (
        CalibrationDetection,
        match_frame,
    )

    triangle = np.array([[0, 0], [20, 0], [10, 20]], np.float32)
    score = match_frame(
        [CalibrationDetection(0, triangle)],
        [CalibrationDetection(0, triangle)],
        task="segment",
    )
    assert score.matched == 1 and score.mean_iou > 0.99


def test_rotated_prediction_is_scored_as_its_aabb_under_detect():
    """A detect model cannot express rotation; scoring must not credit it."""
    import numpy as np

    from hydra_suite.core.inference.direct_calibration import (
        CalibrationDetection,
        match_frame,
    )

    rotated = np.array([[10, 0], [20, 10], [10, 20], [0, 10]], np.float32)
    aabb = np.array([[0, 0], [20, 0], [20, 20], [0, 20]], np.float32)
    obb_score = match_frame(
        [CalibrationDetection(0, rotated)], [CalibrationDetection(0, aabb)], task="obb"
    )
    detect_score = match_frame(
        [CalibrationDetection(0, rotated)],
        [CalibrationDetection(0, aabb)],
        task="detect",
    )
    assert detect_score.mean_iou > obb_score.mean_iou


def test_area_band_rejects_an_oversized_mistargeted_prediction():
    """D9: containment alone credits a blob that swallows a whole label.

    Without a band the huge prediction contains the label's representative
    point, so it earns the recall credit; with a band fitted from the
    labels themselves it is inadmissible and the label is honestly a miss.
    """
    from hydra_suite.core.inference.direct_calibration import fit_calibration_area_band

    labels = [_box(0, 0, 10, 10), _box(30, 0, 40, 10)]
    blob = [_box(0, 0, 40, 10)]  # 4x a single label: spans both animals
    band = fit_calibration_area_band([labels])

    unbanded = match_frame(blob, labels)
    assert unbanded.matched == 1

    banded = match_frame(blob, labels, area_band=band)
    assert banded.matched == 0
    assert banded.missed == 2
    assert banded.extra == 1


def test_area_band_still_admits_the_extent_convention_overshoot():
    """The band must not undo D7: a correct silhouette traced ~1.7x the
    labelled body core is still well inside HIGH_MULTIPLIER."""
    from hydra_suite.core.inference.direct_calibration import fit_calibration_area_band

    labels = [_box(0, 0, 20, 20)]
    inflated = [_box(-3, -3, 23, 23)]
    band = fit_calibration_area_band([labels])
    assert match_frame(inflated, labels, area_band=band).matched == 1


def test_area_band_is_pooled_over_every_labelled_frame():
    """The prior is a property of the whole label set, not of one frame."""
    from hydra_suite.core.inference.direct_calibration import fit_calibration_area_band

    small = [_box(0, 0, 10, 10)]
    large = [_box(0, 0, 40, 40)]
    band = fit_calibration_area_band([small, large])
    assert band is not None
    assert band.n_labels == 2
    # Ceiling anchored to the LARGEST label, floor to the smallest.
    assert band.max_px2 >= 2.5 * 1600.0
    assert band.min_px2 <= 0.3 * 100.0


def test_unfittable_labels_yield_no_band_and_admit_everything():
    from hydra_suite.core.inference.direct_calibration import fit_calibration_area_band

    assert fit_calibration_area_band([]) is None
    assert fit_calibration_area_band([[]]) is None


# --------------------------------------------------------------------------
# D8 quality-floor aggregation: the floor must gate the quantity it was
# calibrated on (semantic/calibration.py pools per MATCHED PAIR across
# frames; the direct path used to average PER-FRAME MEANS, which weights a
# 1-match frame the same as a 16-match frame and injected a hard 0.0 for
# every zero-match frame).
# --------------------------------------------------------------------------
def _quality_frames():
    """Two frames with DIFFERENT match counts and different qualities."""
    return [
        (
            [_box(0, 0, 10, 10), _box(20, 0, 30, 10)],
            [_box(0, 0, 10, 10), _box(20, 0, 30, 10)],
        ),
        ([_box(0, 30, 13, 43)], [_box(0, 30, 10, 40)]),
    ]


def test_mean_quality_pools_per_matched_pair_not_per_frame():
    frames = _quality_frames()
    per_frame = [match_frame(p, ls) for p, ls in frames]
    assert [s.matched for s in per_frame] == [2, 1]
    pooled = sum(s.mean_quality * s.matched for s in per_frame) / sum(
        s.matched for s in per_frame
    )
    naive = sum(s.mean_quality for s in per_frame) / len(per_frame)
    assert abs(pooled - naive) > 1e-6, "fixture must separate the two aggregations"
    assert score_frames(frames).mean_quality == pytest.approx(pooled)


def test_a_zero_match_frame_contributes_no_quality_sample():
    """Absence of evidence is not evidence of bad geometry.

    A frame where nothing matched used to inject a hard 0.0 into the mean,
    dragging a good configuration under MIN_MEAN_QUALITY on arithmetic
    rather than on geometry.
    """
    matched_frame = ([_box(0, 30, 13, 43)], [_box(0, 30, 10, 40)])
    empty_frame = ([_box(500, 500, 510, 510)], [_box(0, 0, 10, 10)])
    only = match_frame(*matched_frame)
    assert only.matched == 1
    assert match_frame(*empty_frame).matched == 0
    both = score_frames([matched_frame, empty_frame])
    assert both.missed == 1  # the zero-match frame still costs recall
    assert both.mean_quality == pytest.approx(only.mean_quality)


def test_matching_nothing_anywhere_scores_zero_quality_not_a_vacuous_pass():
    """An empty sample set must not sneak past the quality floor."""
    score = score_frames(
        [
            ([_box(500, 500, 510, 510)], [_box(0, 0, 10, 10)]),
            ([_box(600, 600, 610, 610)], [_box(0, 0, 10, 10)]),
        ]
    )
    assert score.matched == 0
    assert score.mean_quality == 0.0
    assert score.mean_quality is not None  # None means NEVER MEASURED


def test_an_empty_quality_sample_cannot_pass_the_recommender():
    """The companion to the pooling change, at the recommender.

    Pooling per matched pair means a configuration with NO matched pairs has
    an EMPTY quality sample, reported as 0.0. That must never become a
    vacuous pass, and it must not be reported as a geometry claim either:
    with nothing matched there is no recall, so the RECALL floor refuses
    first and the "Mistargeted" verdict is never reached. Evidence-poor but
    genuinely matching configurations are caught by the MATCHED-INSTANCES
    floor instead.
    """
    for label, matched, missed in (
        ("matched_nothing", 0, 200),
        ("nothing_labelled_either", 0, 0),
    ):
        best, reason = recommend_balanced(
            [_point(label, 0.1, matched=matched, missed=missed, mean_quality=0.0)]
        )
        assert best is None, label
        assert "recall" in reason
        assert "Mistargeted" not in reason

    thin, thin_reason = recommend_balanced(
        [
            _point(
                "well_targeted_but_tiny",
                0.1,
                matched=MIN_MATCHED_INSTANCES - 1,
                missed=0,
                extra=0,
                mean_quality=0.7,
            )
        ]
    )
    assert thin is None
    assert "matched instances" in thin_reason
