"""Well-formedness checks for the frozen direct-calibration before-gate corpus.

Task 3 of docs/superpowers/plans/2026-09-06-direct-path-scoring-unification.md.

IMPORTANT: this test suite intentionally does NOT assert ``before.json``'s
scores against live code. ``before.json`` is frozen historical evidence of
what the legacy scoring rules produced on this corpus at the commit recorded
inside it -- Task 4 changes the scoring rules on purpose, so an equivalence
test here would either fail post-swap or force someone to keep a dead legacy
scorer alive just to satisfy it. See
``tests/data/direct_calibration_golden/generate_before.py`` module docstring.

What IS asserted: that the corpus loads, is internally consistent, and
covers the cases the D7/D8/D9 rulings are about, and that the committed
``before.json`` is structurally well-formed and matches the corpus it claims
to score (same case names, same task labels).
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

DATA_DIR = Path(__file__).parent / "data" / "direct_calibration_golden"
CORPUS_PATH = DATA_DIR / "corpus.json"
BEFORE_PATH = DATA_DIR / "before.json"

VALID_TASKS = {"detect", "obb", "segment"}


@pytest.fixture(scope="module")
def corpus():
    return json.loads(CORPUS_PATH.read_text())


@pytest.fixture(scope="module")
def before():
    return json.loads(BEFORE_PATH.read_text())


def _polygon_is_finite(polygon):
    return all(math.isfinite(coordinate) for point in polygon for coordinate in point)


def test_corpus_file_exists_and_parses(corpus):
    assert corpus["seed"] == 20260906
    assert isinstance(corpus["cases"], list) and corpus["cases"]


def test_every_case_has_required_fields_and_valid_task(corpus):
    for case in corpus["cases"]:
        assert {"name", "task", "description", "frames"} <= set(case)
        assert case["task"] in VALID_TASKS
        assert isinstance(case["name"], str) and case["name"]
        assert isinstance(case["frames"], list) and case["frames"]


def test_every_frame_has_labels_and_predictions_lists(corpus):
    for case in corpus["cases"]:
        for frame in case["frames"]:
            assert "labels" in frame and "predictions" in frame
            assert isinstance(frame["labels"], list)
            assert isinstance(frame["predictions"], list)


def test_every_detection_record_has_required_keys(corpus):
    for case in corpus["cases"]:
        for frame in case["frames"]:
            for record in frame["labels"] + frame["predictions"]:
                assert {"class_id", "polygon_px", "confidence"} <= set(record)
                assert isinstance(record["class_id"], int)
                assert isinstance(record["polygon_px"], list)


def test_all_three_tasks_are_covered(corpus):
    tasks_seen = {case["task"] for case in corpus["cases"]}
    assert tasks_seen == VALID_TASKS


def test_ruling_cases_are_present_for_every_task(corpus):
    names = {case["name"] for case in corpus["cases"]}
    for task in VALID_TASKS:
        assert f"d7_extent_convention_inflation_{task}" in names
        assert f"d7_nonconvex_rotated_{task}" in names
        assert f"rotated_task_divergence_{task}" in names
        assert f"nonconvex_centroid_outside_{task}" in names
        assert f"d9_blob_spans_two_animals_{task}" in names
        assert f"dense_cluster_label_steal_{task}" in names
        assert f"degenerate_geometry_{task}" in names
        assert f"bulk_easy_matches_{task}" in names


def _shoelace_area(polygon):
    xs = [p[0] for p in polygon]
    ys = [p[1] for p in polygon]
    n = len(polygon)
    total = 0.0
    for i in range(n):
        j = (i + 1) % n
        total += xs[i] * ys[j] - xs[j] * ys[i]
    return abs(total) / 2.0


def test_d7_nonconvex_rotated_case_preserves_the_17x_area_ratio_and_is_non_convex(
    corpus,
):
    """Rotation is area- and IoU-preserving; this guards against the
    rotated/non-convex headline variant silently drifting from the 1.7x
    ratio the plain D7 case also asserts, and confirms it really is
    non-convex geometry (not merely a rotated rectangle).
    """

    def _is_convex(polygon):
        n = len(polygon)
        if n < 4:
            return True
        signs = []
        for i in range(n):
            ax, ay = polygon[i]
            bx, by = polygon[(i + 1) % n]
            cx, cy = polygon[(i + 2) % n]
            cross = (bx - ax) * (cy - by) - (by - ay) * (cx - bx)
            signs.append(cross > 0)
        return all(signs) or not any(signs)

    case = next(c for c in corpus["cases"] if c["name"] == "d7_nonconvex_rotated_obb")
    frame = case["frames"][0]
    label_area = _shoelace_area(frame["labels"][0]["polygon_px"])
    pred_polygon = frame["predictions"][0]["polygon_px"]
    pred_area = _shoelace_area(pred_polygon)
    # Coordinates are rounded to 3 decimals when the rotated corpus is
    # generated, so a tiny (<0.01%) rasterization-free rounding drift is
    # expected here -- this is exact-geometry shoelace area, not IoU.
    assert label_area == pytest.approx(400.0, rel=1e-4)
    assert pred_area == pytest.approx(680.0, rel=1e-4)
    assert pred_area / label_area == pytest.approx(1.7, rel=1e-4)
    assert not _is_convex(pred_polygon)


def test_rotated_task_divergence_case_geometry_is_not_axis_aligned(corpus):
    """Sanity guard: the case that is supposed to make detect diverge from
    obb/segment must actually contain rotated (non-axis-aligned) edges,
    else AABB reduction would be a no-op and the case would prove nothing.
    """
    for task in VALID_TASKS:
        case = next(
            c for c in corpus["cases"] if c["name"] == f"rotated_task_divergence_{task}"
        )
        frame = case["frames"][0]
        for record in frame["labels"] + frame["predictions"]:
            polygon = record["polygon_px"]
            xs = sorted({round(p[0], 3) for p in polygon})
            ys = sorted({round(p[1], 3) for p in polygon})
            # An axis-aligned rectangle has exactly 2 distinct x's and 2
            # distinct y's; a rotated one has 4 of each.
            assert len(xs) > 2 and len(ys) > 2


def test_nonconvex_centroid_outside_case_defeats_naive_centroids(corpus):
    """The load-bearing assertion for Finding 2: recompute, straight from
    the committed corpus geometry, that BOTH the naive vertex mean AND the
    ``cv2.moments`` area centroid of the horseshoe silhouette fall outside
    the polygon, while ``representative_point`` (the function D7's
    containment matcher actually calls) returns a point that is inside.
    This is checked against the live function, not frozen -- it is a
    property of the shared, unmoved ``match_geometry`` module (Task 1),
    not of the legacy direct-calibration scorer Task 4 is about to change.
    """
    import cv2
    import numpy as np

    from hydra_suite.core.inference.match_geometry import (
        _contains,
        representative_point,
    )

    case = next(
        c for c in corpus["cases"] if c["name"] == "nonconvex_centroid_outside_obb"
    )
    frame = case["frames"][0]
    silhouette = np.asarray(frame["predictions"][0]["polygon_px"], dtype=np.float64)
    label = np.asarray(frame["labels"][0]["polygon_px"], dtype=np.float64)

    vertex_mean = silhouette.mean(axis=0)
    assert not _contains(silhouette, vertex_mean)

    moments = cv2.moments(silhouette.astype(np.float32).reshape(-1, 1, 2))
    area_centroid = np.array(
        [moments["m10"] / moments["m00"], moments["m01"] / moments["m00"]]
    )
    assert not _contains(silhouette, area_centroid)

    rep_point = representative_point(silhouette)
    assert _contains(silhouette, rep_point)

    # The label was placed at the verified representative point so a
    # future containment matcher can find it; confirm that placement.
    label_center = label.mean(axis=0)
    assert label_center == pytest.approx(rep_point, abs=1.0)


def test_d7_case_geometry_matches_its_documented_area_ratio(corpus):
    """The D7 case's own docstring claims: label area 400, prediction area
    680 (1.7x), IoU ~= 0.421. Recompute the areas straight from the corpus
    geometry (axis-aligned rectangles, so width*height is exact) as a
    guard against the hand-written case silently drifting from its claim.
    """

    def _rect_area(polygon):
        xs = [p[0] for p in polygon]
        ys = [p[1] for p in polygon]
        return (max(xs) - min(xs)) * (max(ys) - min(ys))

    case = next(
        c for c in corpus["cases"] if c["name"] == "d7_extent_convention_inflation_obb"
    )
    frame = case["frames"][0]
    label_area = _rect_area(frame["labels"][0]["polygon_px"])
    pred_area = _rect_area(frame["predictions"][0]["polygon_px"])
    assert label_area == pytest.approx(400.0)
    assert pred_area == pytest.approx(680.0)
    assert pred_area / label_area == pytest.approx(1.7, rel=1e-6)


def test_degenerate_case_contains_non_finite_and_sub_triangle_geometry(corpus):
    case = next(
        c for c in corpus["cases"] if c["task"] == "obb" and "degenerate" in c["name"]
    )
    frame = case["frames"][0]
    labels = frame["labels"]
    predictions = frame["predictions"]
    assert any(len(record["polygon_px"]) < 3 for record in labels)
    assert any(not _polygon_is_finite(record["polygon_px"]) for record in predictions)
    assert any(_polygon_is_finite(record["polygon_px"]) for record in predictions)


def test_bulk_cases_clear_min_matched_instances_floor(corpus):
    # MIN_MATCHED_INSTANCES in core/inference/direct_calibration.py is 60;
    # duplicated here as a literal (not imported) because this test must
    # keep working even after Task 4 changes the constant's role -- it is
    # checking the CORPUS's volume, not the live threshold.
    min_matched_instances = 60
    for task in VALID_TASKS:
        case = next(
            c for c in corpus["cases"] if c["name"] == f"bulk_easy_matches_{task}"
        )
        total_instances = sum(len(frame["labels"]) for frame in case["frames"])
        assert total_instances >= min_matched_instances


def test_before_json_exists_and_is_well_formed(before):
    assert isinstance(before["generated_at_commit"], str)
    assert len(before["generated_at_commit"]) == 40
    assert "scoring_rules" in before
    assert before["scoring_rules"]["iou_threshold"] == 0.5
    assert before["scoring_rules"]["area_band"] is None
    assert isinstance(before["cases"], list) and before["cases"]
    assert "recommend_balanced_demo" in before


def test_before_json_case_names_match_corpus_case_names(corpus, before):
    corpus_names = {case["name"] for case in corpus["cases"]}
    before_names = {case["name"] for case in before["cases"]}
    assert corpus_names == before_names


def test_before_json_aggregate_scores_are_internally_consistent(before):
    for case in before["cases"]:
        aggregate = case["aggregate"]
        matched = aggregate["matched"]
        missed = aggregate["missed"]
        extra = aggregate["extra"]
        expected_precision = matched / (matched + extra) if matched + extra else 0.0
        expected_recall = matched / (matched + missed) if matched + missed else 0.0
        assert aggregate["precision"] == pytest.approx(expected_precision)
        assert aggregate["recall"] == pytest.approx(expected_recall)
        if expected_precision + expected_recall:
            expected_f1 = (
                2
                * expected_precision
                * expected_recall
                / (expected_precision + expected_recall)
            )
        else:
            expected_f1 = 0.0
        assert aggregate["f1"] == pytest.approx(expected_f1)


def test_before_json_demonstrates_the_d7_simultaneous_miss_and_extra(before):
    """The single most important case: under the legacy hard-IoU gate, a
    correct silhouette with an inflated extent must score as BOTH a miss
    and an extra in the same frame (not a partial credit of any kind).
    """
    for task in VALID_TASKS:
        case = next(
            c
            for c in before["cases"]
            if c["name"] == f"d7_extent_convention_inflation_{task}"
        )
        aggregate = case["aggregate"]
        assert aggregate["matched"] == 0
        assert aggregate["missed"] == 1
        assert aggregate["extra"] == 1


def test_before_json_recommend_balanced_demo_excludes_ineligible_points(before):
    demo = before["recommend_balanced_demo"]
    assert demo["chosen_label"] not in ("synthetic_undersampled", "synthetic_failed")
    assert demo["chosen_label"] is not None


def test_before_json_recommend_balanced_demo_decided_by_the_speed_tie_break(before):
    """Finding 2's fix: the synthetic sweep must put >=2 points inside
    F1_TOLERANCE=0.01 of the best F1 at different speeds, so the winner is
    actually decided by ``recommend_balanced``'s fastest-within-tolerance
    rule rather than by one point dominating on every axis.
    """
    demo = before["recommend_balanced_demo"]
    assert "synthetic_best_slow" in demo["candidate_labels"]
    assert "synthetic_near_best_faster" in demo["candidate_labels"]
    # The faster, slightly-lower-F1 point must be the one chosen -- proof
    # the tie-break, not raw F1 dominance, decided the outcome.
    assert demo["chosen_label"] == "synthetic_near_best_faster"
    assert demo["chosen_f1"] == pytest.approx(0.9937106918238994, rel=1e-6)
    assert demo["chosen_f1"] < 1.0 - 1e-9  # strictly not the best-F1 point
    assert demo["chosen_seconds_per_frame"] == pytest.approx(0.5)


def test_before_json_task_suffixes_produce_different_aggregates(before):
    """Finding 1's load-bearing check: before this fix-round, detect/obb/
    segment produced byte-identical aggregates everywhere because every
    case was an axis-aligned rectangle (AABB reduction is the identity
    transform on those). The ``rotated_task_divergence`` case must now
    make detect's aggregate genuinely differ from obb's and segment's --
    not just a different IoU number on the same match/miss outcome, but a
    different outcome: detect scores a match, obb/segment score a
    miss+extra, on IDENTICAL input geometry (task is the only variable).
    """
    aggregates = {
        case["task"]: case["aggregate"]
        for case in before["cases"]
        if case["name"].startswith("rotated_task_divergence_")
    }
    assert aggregates["detect"] != aggregates["obb"]
    assert aggregates["obb"] == aggregates["segment"]  # both use full polygon IoU
    assert aggregates["detect"]["matched"] == 1
    assert aggregates["detect"]["missed"] == 0
    assert aggregates["detect"]["extra"] == 0
    assert aggregates["obb"]["matched"] == 0
    assert aggregates["obb"]["missed"] == 1
    assert aggregates["obb"]["extra"] == 1
