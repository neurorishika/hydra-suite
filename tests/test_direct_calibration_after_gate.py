"""Standing characterization gate for the direct-calibration after-scores.

Task 5 of docs/superpowers/plans/2026-09-06-direct-path-scoring-unification.md.

Re-scoring the frozen corpus with CURRENT code must reproduce the committed
``after.json``. This is the standing half of the before/after gate: any future
change to the direct path's matcher, shape prior, or recommender will fail
here and must be accounted for by regenerating ``after.json`` deliberately and
saying why in ``tests/data/direct_calibration_golden/ATTRIBUTION.md``.

``before.json`` deliberately gets NO standing assertion -- it is frozen
historical evidence of the legacy rules and asserting it would require keeping
a dead scorer alive. See that file's own generator docstring.

The scorer is LOADED from ``generate_after.py`` rather than reimplemented, so
this test cannot drift from the artifact-generation path.

Assertions about ``rotated_task_divergence`` target VALUES (mean_iou /
mean_quality), never match COUNTS: under the D7 containment matcher both
``detect`` and ``obb`` now match, and the detect-vs-obb divergence migrated
from the outcome into the numbers. The AABB reduction's liveness is asserted
separately and independently by
``tests/test_direct_calibration.py::test_rotated_prediction_is_scored_as_its_aabb_under_detect``.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

DATA_DIR = Path(__file__).parent / "data" / "direct_calibration_golden"
CORPUS_PATH = DATA_DIR / "corpus.json"
AFTER_PATH = DATA_DIR / "after.json"
GENERATOR_PATH = DATA_DIR / "generate_after.py"

# Provenance keys depend on WHERE the checkout lives and WHICH commit is
# checked out, so they are excluded from the value comparison. They are
# asserted separately for well-formedness.
VOLATILE_KEYS = {"generated_at_commit", "hydra_suite_package_root"}

VALID_TASKS = {"detect", "obb", "segment"}


@pytest.fixture(scope="module")
def generator():
    spec = importlib.util.spec_from_file_location(
        "direct_calibration_generate_after", GENERATOR_PATH
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def committed():
    return json.loads(AFTER_PATH.read_text())


@pytest.fixture(scope="module")
def recomputed(generator):
    corpus = json.loads(CORPUS_PATH.read_text())
    payload = generator.build_payload(corpus, use_area_band=True, stage="after")
    # Round-trip through JSON so float formatting matches the committed file.
    return json.loads(json.dumps(payload, sort_keys=True))


def _strip_volatile(payload: dict) -> dict:
    return {k: v for k, v in payload.items() if k not in VOLATILE_KEYS}


def test_current_code_reproduces_the_committed_after_scores(recomputed, committed):
    """The whole point of the gate: live code still produces after.json."""
    assert _strip_volatile(recomputed) == _strip_volatile(committed)


def test_after_json_provenance_is_well_formed(committed):
    assert isinstance(committed["generated_at_commit"], str)
    assert len(committed["generated_at_commit"]) == 40
    assert committed["hydra_suite_package_root"].endswith("hydra_suite")
    assert committed["stage"] == "after"


def test_after_json_case_names_match_corpus_case_names(committed):
    corpus = json.loads(CORPUS_PATH.read_text())
    assert {c["name"] for c in corpus["cases"]} == {
        c["name"] for c in committed["cases"]
    }


def test_after_json_aggregate_scores_are_internally_consistent(committed):
    for case in committed["cases"]:
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


def test_d7_extent_convention_case_is_no_longer_a_simultaneous_miss_and_extra(
    committed,
):
    """D7's headline: a correct silhouette with a 1.7x extent convention was
    scored as BOTH a miss and an extra under the legacy hard IoU gate (see
    before.json). It is now a single match -- at an IoU the old gate would
    have rejected, which is the point.
    """
    for task in VALID_TASKS:
        case = next(
            c
            for c in committed["cases"]
            if c["name"] == f"d7_extent_convention_inflation_{task}"
        )
        aggregate = case["aggregate"]
        assert aggregate["matched"] == 1
        assert aggregate["missed"] == 0
        assert aggregate["extra"] == 0
        assert aggregate["mean_iou"] < 0.5  # the legacy gate would have refused it
        assert aggregate["mean_quality"] > 0.35


def test_d9_band_rejects_the_grossly_oversized_nonconvex_silhouette(committed):
    """D9's real corpus evidence. The D7 matcher CREDITS this case (the
    label's representative point is inside the horseshoe) -- see
    stage_a_d7_matcher.json -- and D9's area band takes it back, because the
    silhouette is ~25x the labelled body area.

    NOTE for the next reader: ``d9_blob_spans_two_animals_*`` is misnamed and
    does NOT exercise D9 (its band admits the blob). See ATTRIBUTION.md.
    """
    for task in ("obb", "segment"):
        case = next(
            c
            for c in committed["cases"]
            if c["name"] == f"nonconvex_centroid_outside_{task}"
        )
        assert case["aggregate"]["matched"] == 0
        band = case["area_band"]
        assert band is not None and band["n"] == 1
        assert band["max"] == pytest.approx(360.0)


def test_misnamed_d9_blob_case_is_admitted_by_its_own_band(committed):
    """Guard the ATTRIBUTION.md claim that this case's NAME is misleading:
    its band admits the prediction, so it demonstrates nothing about D9. If
    a future change makes the band reject it, this test fails and the
    document must be corrected rather than silently going stale.
    """
    for task in VALID_TASKS:
        case = next(
            c
            for c in committed["cases"]
            if c["name"] == f"d9_blob_spans_two_animals_{task}"
        )
        band = case["area_band"]
        assert band is not None and band["n"] == 2
        assert band["median"] == pytest.approx(400.0)
        assert band["max"] == pytest.approx(1000.0)
        # The blob is 624 px^2 -- comfortably inside a 1000 px^2 ceiling.
        assert band["max"] > 624.0
        # And it is still counted as a match, exactly as before the swap.
        assert case["aggregate"]["matched"] == 1


def test_rotated_task_divergence_survives_in_values_not_counts(committed):
    """Under the D7 containment matcher BOTH detect and obb now match, so
    the old count-based divergence assertion (before.json's) would no longer
    catch a regression. The divergence migrated into the numbers: detect
    scores higher because both polygons are reduced to their axis-aligned
    bounding boxes first, which is IoU-inflating on rotated geometry.
    """
    aggregates = {
        case["task"]: case["aggregate"]
        for case in committed["cases"]
        if case["name"].startswith("rotated_task_divergence_")
    }
    assert aggregates["detect"]["matched"] == 1
    assert aggregates["obb"]["matched"] == 1
    # obb and segment both score full-polygon geometry -- identical.
    assert aggregates["obb"] == aggregates["segment"]
    # detect diverges on VALUES.
    assert aggregates["detect"]["mean_iou"] > aggregates["obb"]["mean_iou"]
    assert aggregates["detect"]["mean_quality"] > aggregates["obb"]["mean_quality"]
    assert aggregates["detect"]["mean_iou"] == pytest.approx(0.5111, abs=1e-3)
    assert aggregates["obb"]["mean_iou"] == pytest.approx(0.4643, abs=1e-3)


def test_d8_recommendation_is_recall_first_and_stamped_with_its_rule_id(committed):
    demo = committed["recommend_balanced_demo"]
    assert demo["rule_id"] == "recall-first-quality-floors-v1"
    assert demo["chosen_label"] not in ("synthetic_undersampled", "synthetic_failed")
    assert demo["chosen_recall"] >= 0.90
    assert demo["chosen_mean_quality"] >= 0.35
    # F1 is still REPORTED -- retired as a target, not deleted as a number.
    assert demo["chosen_f1"] is not None
    assert "reported, not optimised" in demo["explanation"]


def test_staged_attribution_payloads_are_from_distinct_intended_commits():
    """The failure mode of the whole attribution exercise is scoring HEAD four
    times (an editable install shadowing PYTHONPATH). Each staged payload
    records the sha of the tree its hydra_suite came from; assert they are the
    three ruling commits, so a smeared artifact cannot pass unnoticed.
    """
    expected = {
        "stage_a_d7_matcher.json": "5700e7ec",
        "stage_b_d7_d9.json": "8c7d2230",
        "stage_c_d7_d9_d8.json": "546b9989",
    }
    shas = set()
    for filename, prefix in expected.items():
        payload = json.loads((DATA_DIR / filename).read_text())
        assert payload["generated_at_commit"].startswith(prefix), filename
        shas.add(payload["generated_at_commit"])
    assert len(shas) == 3


def test_staged_payloads_isolate_each_ruling():
    """The attribution claims, asserted against the committed payloads:
    D9 changes no recommendation, D8 changes no case score.
    """
    stage_a = json.loads((DATA_DIR / "stage_a_d7_matcher.json").read_text())
    stage_b = json.loads((DATA_DIR / "stage_b_d7_d9.json").read_text())
    stage_c = json.loads((DATA_DIR / "stage_c_d7_d9_d8.json").read_text())
    after = json.loads(AFTER_PATH.read_text())

    # Stage (a) is band-free -- that is what makes it D7-only.
    assert all(case["area_band"] is None for case in stage_a["cases"])
    assert all(case["area_band"] is not None for case in stage_b["cases"])

    def aggregates(payload):
        return [case["aggregate"] for case in payload["cases"]]

    # D8 (b -> c) touches the recommender only.
    assert aggregates(stage_b) == aggregates(stage_c)
    assert (
        stage_b["recommend_balanced_demo"]["rule_id"]
        != stage_c["recommend_balanced_demo"]["rule_id"]
    )
    # The persistence-only follow-up (c -> after) changes no behaviour.
    assert aggregates(stage_c) == aggregates(after)
    assert stage_c["recommend_balanced_demo"] == after["recommend_balanced_demo"]
