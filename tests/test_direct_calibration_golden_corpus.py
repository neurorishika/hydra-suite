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
        assert f"d9_blob_spans_two_animals_{task}" in names
        assert f"dense_cluster_label_steal_{task}" in names
        assert f"degenerate_geometry_{task}" in names
        assert f"bulk_easy_matches_{task}" in names


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
