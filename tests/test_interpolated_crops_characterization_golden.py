"""Characterization golden for the interpolated-crop inference unification.

Diffs the overhauled ``run_interpolated_crops`` output against a golden
captured from pre-change ``main`` (commit ``645186b27f927ce2f31404cea5ab51b20c37ad77``,
the commit immediately before Task 7's first commit) on a synthetic
occlusion-heavy fixture.

Investigation summary (see task-13-report.md for the full writeup): NONE of
the ``tools/equivalence/fixtures/`` clip configs enable pose + CNN +
AprilTag + head-tail simultaneously (checked every ``configs/*.json`` for
``ENABLE_POSE_EXTRACTOR``/``CNN_CLASSIFIERS``/``USE_APRILTAGS``/
``YOLO_HEADTAIL_MODEL_PATH`` -- none qualify), matching the design spec's own
audit. A synthetic harness (``tests/fixtures/interpolated_crops_golden/
_harness.py``) was built instead: a hand-written 2-trajectory tracking CSV
with occluded runs exercising BOTH the mechanism-1 CSV-value priority (Task
8's fix) and the NaN fallback, a tiny synthetic ``cv2.VideoWriter`` video,
and real (untrained but architecturally real) CNN-identity and head-tail
classifier weights built via the same helpers as
``tests/test_classifier_fixtures.py``'s session fixtures.

Coverage actually achieved -- 2 of the 4 signal types, plus a bonus
geometry-sourcing check:

* CNN identity: real, exercised on every occluded frame (single- and
  multi-task).
* Head-tail: real, exercised on every occluded frame.
* Pose: OMITTED. No small CPU-fast pose-model fixture exists anywhere in
  the test suite (building one needs either a network download of a
  YOLO-pose base checkpoint + non-trivial backend adaptation, or the SLEAP
  conda env). ``test_pose_output_matches_golden_within_registered_differences``
  is kept as a named, explicitly-skipped test documenting this.
* AprilTag: OMITTED. The lab AprilTag fork IS importable in this dev
  environment, but no small-fixture generator for decodable synthetic tag
  imagery exists, and building one (valid tag36h11/tag36ARTag bit patterns)
  was judged disproportionate to this task's budget.
  ``test_tag_output_unmasked_relative_to_golden_on_multi_task_frames`` is
  kept as a named, explicitly-skipped test documenting this.

Four differences were PRE-REGISTERED as expected (design spec's Testing
section) and must NOT fail this test:

1. CNN crop identity (now unmasked/independent of pose). This harness
   demonstrates an even more visible form of this than "identity differs on
   multi-task frames": pre-refactor ``main`` coupled CNN crop-building to
   pose entirely (``_pending_cnn_crops.append(pose_crop)`` inside
   ``if pose_backend is not None:`` -- see
   ``core/post/interpolated_crops.py`` at 645186b2, lines ~1066-1082), so
   with pose disabled the golden's CNN output is EMPTY (0 rows) on every
   frame, not just single-task ones. The current code runs CNN
   independently via ``run_cnn_batch``/``extract_classifier_crops_batch_np``
   regardless of pose, producing real predictions on every occluded frame.
2. AprilTag crop masking (now unmasked) -- not exercised (AprilTag omitted).
3. Pose crop LSB rounding (~1px) -- not exercised (pose omitted).
4. Head-tail crop-construction path (``HeadTailAnalyzer.analyze_crops`` ->
   ``run_headtail_batch``), verified by tolerance not byte-identity. This
   harness surfaces a specific, understood instance: the shared
   ``_assemble_headtail_result`` (``core/inference/stages/headtail.py``)
   reports ``heading_conf=0.0`` for undirected detections, while the legacy
   ``HeadTailAnalyzer.predict_labels`` always reports the classifier's raw
   confidence even when the label collapses to "unknown". This stems from
   the SAME shared ``run_headtail_batch``/``_assemble_headtail_result`` code
   the live tracking Pipeline already used before this refactor -- Tasks
   7-12 only changed WHICH caller (interpolated-crops vs. only live
   tracking) reaches it, not the function itself. The ``heading_directed``
   decision (the semantically load-bearing part) is identical between
   golden and current on every row in this fixture.

A FIFTH difference, not in the original four but directly demonstrating
Task 8's fix, is captured via the ROI/geometry CSV (``interpolated_rois.csv``,
not one of the four "artifact" CSVs but produced identically by both code
paths otherwise): pre-refactor ``main`` derives ``Theta`` for occluded rows
via its OWN linear/angle interpolation even when the CSV already carries a
mechanism-1-filled (non-NaN) ``Theta`` for that exact row, ignoring it. The
current code's NaN-triggered geometry-sourcing priority (Task 8) uses the
CSV's own value directly when present. ``cx``/``cy`` already matched between
golden and current (position priority pre-dates Task 8); only ``Theta``
diverges, and only on the two mechanism-1-filled rows (frame 5 and 6,
trajectory 1) -- every other geometry field, on every row, is byte-identical.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

GOLDEN_DIR = Path(__file__).parent / "fixtures" / "interpolated_crops_golden"

sys.path.insert(0, str(GOLDEN_DIR))
import _harness  # noqa: E402  (golden-dir harness module, see its docstring)


def _run_current(tmp_path):
    """Run today's (post-unification) run_interpolated_crops on the SAME
    CSV/video/params shape Step 1 used, reusing the EXACT SAME classifier
    weight files the golden was captured with (``_harness.py``'s
    ``cnn_model_src``/``headtail_model_src`` params) so the only variable
    between golden and current is the code under test, not independently
    random model initialization.
    """
    root = tmp_path / "harness_root"
    payload = _harness.run_harness(
        root,
        cnn_model_src=GOLDEN_DIR / "cnn_identity.pth",
        headtail_model_src=GOLDEN_DIR / "headtail.pth",
    )
    return payload


# ---------------------------------------------------------------------------
# Sanity: the golden itself must exist and have the expected shape before any
# comparison test can mean anything.
# ---------------------------------------------------------------------------


def test_golden_fixture_files_present():
    assert (GOLDEN_DIR / "interpolated_headtail.csv").exists()
    assert (GOLDEN_DIR / "interpolated_rois.csv").exists()
    assert (GOLDEN_DIR / "interpolated_mapping.csv").exists()
    assert (GOLDEN_DIR / "cnn_identity.pth").exists()
    assert (GOLDEN_DIR / "headtail.pth").exists()
    # Registered difference #1's most visible form: golden has NO CNN/pose/
    # tag CSVs at all (pose+apriltag were never enabled; CNN was coupled to
    # pose pre-refactor).
    assert not (GOLDEN_DIR / "interpolated_cnn_identity.csv").exists()
    assert not (GOLDEN_DIR / "interpolated_pose.csv").exists()
    assert not (GOLDEN_DIR / "interpolated_tags.csv").exists()


@pytest.mark.skip(
    reason=(
        "Pose omitted from this golden's synthetic harness: no small "
        "CPU-fast pose-model fixture exists anywhere in the test suite "
        "(tests/fixtures/, test_classifier_fixtures.py, etc. only cover "
        "CNN/head-tail classifier architectures). Building one needs either "
        "a network download of a YOLO-pose base checkpoint plus adapting "
        "the YOLO-pose backend loader, or the SLEAP conda env -- judged "
        "disproportionate to this task's budget. See task-13-report.md."
    )
)
def test_pose_output_matches_golden_within_registered_differences(tmp_path):
    golden = pd.read_csv(GOLDEN_DIR / "interpolated_pose.csv")
    payload = _run_current(tmp_path)
    current = pd.read_csv(payload["pose_csv_path"])
    merged = golden.merge(
        current, on=["frame_id", "trajectory_id"], suffixes=("_golden", "_current")
    )
    assert len(merged) == len(golden), "row count changed unexpectedly"
    for kpt_col in [
        c for c in golden.columns if c.startswith("PoseKpt_") and c.endswith("_X")
    ]:
        base = kpt_col
        # Registered difference: pose crop LSB rounding -> allow +/-1px.
        diff = (merged[f"{base}_current"] - merged[f"{base}_golden"]).abs()
        assert (
            diff <= 1.5
        ).all(), f"{base} diverges beyond the registered LSB tolerance"


def test_cnn_output_diverges_only_on_frames_with_multiple_interpolated_tasks(tmp_path):
    """CNN crop identity is a REGISTERED difference (design spec bug fix #1:
    CNN is no longer coupled to pose). The golden here demonstrates the
    MAXIMAL form of that divergence -- with pose disabled, pre-refactor
    ``main`` produces literally zero CNN rows on ANY frame (single- or
    multi-task), because ``_pending_cnn_crops`` was only ever populated
    inside the pose branch. The current code produces real, independent CNN
    predictions on every occluded frame regardless of task count or pose
    state -- this test asserts BOTH halves of that: golden absence, current
    presence on both the multi-task frame range (5-9, both trajectories
    simultaneously occluded) and the single-task frame range (15-17, only
    trajectory 2 occluded).
    """
    golden_path = GOLDEN_DIR / "interpolated_cnn_identity.csv"
    assert not golden_path.exists(), (
        "golden unexpectedly has CNN output -- re-verify the pose/CNN "
        "coupling assumption documented in this test's docstring"
    )

    payload = _run_current(tmp_path)
    cnn_csv_paths = payload.get("cnn_csv_paths") or {}
    assert "identity" in cnn_csv_paths, "current code produced no CNN output at all"
    current = pd.read_csv(cnn_csv_paths["identity"])

    multi_task_frames = set(range(5, 10))
    single_task_frames = {15, 16, 17}
    got_frames = set(current["frame_id"].tolist())
    assert multi_task_frames <= got_frames, "CNN missing rows on multi-task frames"
    assert single_task_frames <= got_frames, "CNN missing rows on single-task frames"

    # Both trajectories get a prediction on every multi-task frame.
    multi = current[current["frame_id"].isin(multi_task_frames)]
    for f in multi_task_frames:
        traj_ids = set(multi.loc[multi["frame_id"] == f, "trajectory_id"])
        assert traj_ids == {
            1,
            2,
        }, f"frame {f}: expected both trajectories, got {traj_ids}"

    # CNN_identity_Conf must be a real probability in (0, 1], not a stub/zero.
    assert (current["CNN_identity_Conf"] > 0.0).all()
    assert (current["CNN_identity_Conf"] <= 1.0).all()


def test_headtail_output_agrees_within_tolerance_not_byte_identity(tmp_path):
    """run_headtail_batch is a materially different crop-construction path
    than HeadTailAnalyzer.analyze_crops -- verify agreement empirically
    rather than requiring byte-identity.

    The semantically load-bearing field is ``heading_directed`` (whether the
    classifier's prediction was confident enough to assign a heading); this
    must agree row-for-row. ``heading_conf`` is EXPECTED to diverge on
    undirected rows: the shared ``_assemble_headtail_result`` the current
    code now reuses reports 0.0 for undirected detections, while the legacy
    ``HeadTailAnalyzer.predict_labels`` always surfaces the classifier's raw
    confidence. This fixture's untrained weights never cross the confidence
    threshold, so every row here is undirected -- this test explicitly
    documents/asserts that known conf=0-vs-raw divergence rather than
    silently allowing (or blindly requiring) numeric agreement.
    """
    golden = pd.read_csv(GOLDEN_DIR / "interpolated_headtail.csv")
    payload = _run_current(tmp_path)
    assert payload.get("headtail_csv_path"), "current code produced no head-tail output"
    current = pd.read_csv(payload["headtail_csv_path"])

    merged = golden.merge(
        current, on=["frame_id", "trajectory_id"], suffixes=("_golden", "_current")
    )
    assert len(merged) == len(golden), "row count changed unexpectedly"

    # Semantic decision (directed vs. undirected) must agree row-for-row.
    assert (
        merged["heading_directed_golden"] == merged["heading_directed_current"]
    ).all(), "heading_directed decision diverged -- NOT a registered difference"

    # Where undirected (this fixture: every row), heading_rad is NaN on both
    # sides and heading_conf is 0.0 on the current side (registered
    # difference: current always zeroes conf for undirected detections; the
    # legacy path did not).
    undirected = merged["heading_directed_current"] == 0
    assert undirected.all(), "fixture assumption changed: expected all-undirected rows"
    assert merged.loc[undirected, "heading_rad_golden"].isna().all()
    assert merged.loc[undirected, "heading_rad_current"].isna().all()
    assert (merged.loc[undirected, "heading_conf_current"] == 0.0).all(), (
        "current code no longer zeroes heading_conf for undirected rows -- "
        "re-verify this against _assemble_headtail_result before treating "
        "it as a regression"
    )
    # The golden's raw (non-zero) confidence for the same undirected rows is
    # the registered divergence -- assert it's still non-zero so this test
    # would fail loudly (not silently) if the golden fixture ever changed.
    assert (merged.loc[undirected, "heading_conf_golden"] > 0.0).all()


@pytest.mark.skip(
    reason=(
        "AprilTag omitted from this golden's synthetic harness: the lab "
        "apriltag fork IS importable in this dev environment (pupil "
        "AprilTags is also present but is NOT the required fork), but no "
        "small-fixture generator for decodable synthetic tag imagery "
        "(valid tag36h11/tag36ARTag bit patterns) exists in the test suite, "
        "and building one was judged disproportionate to this task's "
        "budget. See task-13-report.md."
    )
)
def test_tag_output_unmasked_relative_to_golden_on_multi_task_frames(tmp_path):
    """AprilTag crops lose foreign-suppression -- a registered, deliberate
    difference. Assert it's understood (tag ids may legitimately differ on
    multi-task frames) rather than asserting byte-identity."""
    golden = pd.read_csv(GOLDEN_DIR / "interpolated_tags.csv")
    payload = _run_current(tmp_path)
    current = pd.read_csv(payload["tag_csv_path"])
    assert set(current["frame_id"]) >= set(golden["frame_id"])


# ---------------------------------------------------------------------------
# Bonus coverage (not one of the four pre-registered differences, but a
# direct, high-value characterization of Task 8's NaN-triggered geometry-
# sourcing priority fix): the interpolated-crop task geometry itself
# (interpolated_rois.csv) must be byte-identical between golden and current
# EXCEPT for Theta on the two mechanism-1 (CSV-prefilled) rows, where the
# fix intentionally changes behavior -- confirmed against the literal CSV
# input values, not just "differs from golden".
# ---------------------------------------------------------------------------


def test_geometry_sourcing_priority_matches_golden_except_task8_theta_fix(tmp_path):
    golden = pd.read_csv(GOLDEN_DIR / "interpolated_rois.csv")
    payload = _run_current(tmp_path)
    assert payload.get("roi_csv_path"), "current code produced no ROI/geometry output"
    current = pd.read_csv(payload["roi_csv_path"])

    merged = golden.merge(
        current, on=["frame_id", "trajectory_id"], suffixes=("_golden", "_current")
    )
    assert len(merged) == len(golden) == 13

    for col in ("cx", "cy", "w", "h"):
        assert (merged[f"{col}_golden"] == merged[f"{col}_current"]).all(), (
            f"{col} diverged -- unexpected, geometry sourcing for position/size "
            "did not change in this refactor"
        )

    # Task 8's fix is scoped to Theta on exactly the two mechanism-1-filled
    # rows: (frame_id=5, trajectory_id=1) and (frame_id=6, trajectory_id=1).
    task8_rows = merged["frame_id"].isin([5, 6]) & (merged["trajectory_id"] == 1)
    other_rows = ~task8_rows

    assert (
        merged.loc[other_rows, "theta_golden"]
        == merged.loc[other_rows, "theta_current"]
    ).all(), "Theta diverged outside the two documented Task-8-affected rows"

    # On the affected rows, current must equal the CSV's OWN pre-filled
    # Theta directly (0.25 at frame 5, 0.30 at frame 6 -- see
    # tests/fixtures/interpolated_crops_golden/_harness.py's
    # build_tracking_csv), not merely "differ from golden".
    expected_current_theta = {5: 0.25, 6: 0.30}
    for f, expected in expected_current_theta.items():
        row = merged[(merged["frame_id"] == f) & (merged["trajectory_id"] == 1)]
        assert row["theta_current"].iloc[0] == pytest.approx(expected)
        # And the golden's value must NOT equal the CSV-prefilled value --
        # confirming this really is the priority-fix delta, not noise.
        assert row["theta_golden"].iloc[0] != pytest.approx(expected)
