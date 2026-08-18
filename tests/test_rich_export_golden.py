"""Value-level golden test for the rich Debug ``_with_individual.csv`` export.

Goldens under ``tests/goldens/rich_export/`` are byte-for-byte snapshots of the
rich (Debug-mode) trajectory export captured on the **Part-1-complete**
pre-vectorization tree (commit ``f940a2ef``, before the Part-2 postproc
vectorization slices), via ``tools/equivalence/runner.py`` against the
equivalence-harness fixture clips (see ``tools/equivalence/fixtures/``) --
never hand-written. ``DEBUG_MODE`` defaults to ``True`` when a config omits
``debug_mode`` (see ``engine_params.py``), which is the case for these
fixture configs, so an unmodified run of the harness naturally emits
``<stem>_tracking_final_with_individual.csv`` (the rich export) rather than
the clean ``<stem>_tracks.csv`` that ``test_user_mode_golden.py`` covers.

This is the byte-identical oracle for Tasks 5-8 (the postproc vectorization
slices): if those slices are truly behavior-preserving, re-running the SAME
harness invocation on the current (HEAD) tree must reproduce these goldens
byte-for-byte.

This test is heavy (real SLEAP/pose + tracking pipeline runs, ~2-6 minutes
per clip) and requires the `hydra-mps`/`hydra-cuda` conda env (the SLEAP
service shells out to `conda run -n sleap`) plus the equivalence fixtures.
It mirrors the gating already used by ``tests/test_classifier_integration_smoke.py``
(``pytest.mark.slow``) and ``tests/test_gui_session_cutover_equivalence.py``
(skip when fixtures are absent). It additionally guards against colliding
with a **foreign** tracker/equivalence run on a shared box: if one is
detected, the test skips rather than launching a competing pipeline run.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
FIXTURE_CLIPS_DIR = REPO / "tools/equivalence/fixtures/clips"
FIXTURE_CONFIGS_DIR = REPO / "tools/equivalence/fixtures/configs"
SKELETON_FILE = REPO / "tools/equivalence/fixtures/ooceraea_biroi.json"
RUNNER_SCRIPT = REPO / "tools/equivalence/runner.py"
GOLDEN_DIR = REPO / "tests/goldens/rich_export"

# Both clips need a skeleton override (portable fixture configs leave
# pose_skeleton_file blank) -- see tools/equivalence/runner.py build_config().
CLIPS = ["ant_pose_headtail", "ant_cnn_identity"]

_FOREIGN_TRACKER_RE = re.compile(r"trackerkit|headless_tracking|run_matrix\.sh")


def _foreign_tracker_running() -> bool:
    """True if some OTHER process (not this test) is already using the tracker.

    Best-effort ``pgrep`` scan. Never raises -- if ``pgrep`` is unavailable or
    errors, conservatively reports "not running" so this only ever skips on a
    genuine positive signal, matching the box-sharing convention documented in
    CLAUDE.md (never collide with a foreign sleap/hydra run).
    """
    try:
        out = subprocess.run(
            ["pgrep", "-fl", "trackerkit|headless_tracking|run_matrix.sh"],
            capture_output=True,
            text=True,
            timeout=10,
        ).stdout
    except Exception:
        return False
    this_pid = str(__import__("os").getpid())
    for line in out.splitlines():
        line = line.strip()
        if not line or line.startswith(this_pid + " "):
            continue
        if _FOREIGN_TRACKER_RE.search(line):
            return True
    return False


def _current_worktree_commit() -> str | None:
    """Short SHA of this worktree's checked-out commit, or None if undeterminable."""
    try:
        out = subprocess.run(
            ["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=10,
        )
    except Exception:
        return None
    if out.returncode != 0:
        return None
    sha = out.stdout.strip()
    return sha or None


def _run_rich_export(clip_name: str, outdir: Path) -> Path:
    """Drive tools/equivalence/runner.py for *clip_name*; return the rich CSV path.

    CRITICAL: this repo has `hydra_suite` pip-installed editable pointing at the
    MAIN repo root, so a bare subprocess (no explicit PYTHONPATH) would silently
    import MAIN's src instead of THIS worktree's HEAD -- comparing the wrong tree
    against the golden and giving a false pass/fail. We therefore (a) force
    PYTHONPATH to this worktree's src ahead of anything inherited from the
    caller's environment, and (b) verify after the fact -- via the runner's
    logged `meta.json` (git_commit field) -- that the run actually executed
    against this worktree's checked-out commit, skipping loudly if that can't
    be confirmed rather than trusting the run silently.
    """
    video = FIXTURE_CLIPS_DIR / f"{clip_name}.mp4"
    config = FIXTURE_CONFIGS_DIR / f"{clip_name}.json"
    outdir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        str(RUNNER_SCRIPT),
        "--orig-config",
        str(config),
        "--video",
        str(video),
        "--outdir",
        str(outdir),
        "--runtime",
        "mps",
        "--label",
        "rich_export_golden_test",
        "--skeleton",
        str(SKELETON_FILE),
    ]
    env = os.environ.copy()
    worktree_src = str(REPO / "src")
    existing_pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = (
        worktree_src + os.pathsep + existing_pythonpath
        if existing_pythonpath
        else worktree_src
    )
    result = subprocess.run(
        cmd, cwd=REPO, env=env, capture_output=True, text=True, timeout=1800
    )
    assert result.returncode == 0, (
        f"runner.py failed for {clip_name} (rc={result.returncode}).\n"
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
    rich_csv = outdir / f"{clip_name}_tracking_final_with_individual.csv"
    assert rich_csv.exists(), (
        f"Expected rich export CSV missing: {rich_csv}\n"
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )

    # Provenance guard: confirm the subprocess actually imported THIS worktree's
    # HEAD, not some other hydra_suite install picked up from the environment.
    meta_path = outdir / "meta.json"
    expected_commit = _current_worktree_commit()
    if expected_commit is None:
        pytest.skip(
            "Could not determine this worktree's HEAD commit via `git rev-parse` "
            "-- refusing to trust the runner's src provenance silently."
        )
    if not meta_path.exists():
        pytest.skip(
            f"runner.py did not write {meta_path} -- cannot verify which "
            "hydra_suite src the subprocess actually imported."
        )
    meta = json.loads(meta_path.read_text())
    run_commit = (meta.get("git_commit") or "")[: len(expected_commit)]
    if not run_commit:
        pytest.skip(f"{meta_path} has no git_commit field -- cannot verify provenance.")
    assert run_commit == expected_commit, (
        f"Provenance mismatch for {clip_name}: runner.py ran against commit "
        f"{meta.get('git_commit')!r} (src={meta.get('hydra_suite_file')!r}) but "
        f"this worktree's HEAD is {expected_commit!r}. The subprocess likely "
        "imported the wrong hydra_suite install (e.g. the main repo's editable "
        "install) instead of this worktree's src -- fix PYTHONPATH, don't trust "
        "this run."
    )
    return rich_csv


def _fixtures_available(clip_name: str) -> bool:
    """Fixture presence check for ONE clip (not the whole CLIPS list), so a
    missing fixture for one clip doesn't spuriously skip the other's case."""
    if not (RUNNER_SCRIPT.exists() and SKELETON_FILE.exists()):
        return False
    if not (FIXTURE_CLIPS_DIR / f"{clip_name}.mp4").exists():
        return False
    if not (FIXTURE_CONFIGS_DIR / f"{clip_name}.json").exists():
        return False
    return True


@pytest.mark.slow
@pytest.mark.timeout(900)
@pytest.mark.parametrize("clip_name", CLIPS)
def test_rich_export_matches_golden_byte_for_byte(clip_name, tmp_path):
    # 900s comfortably exceeds the observed ~304s/clip (even under box
    # contention) while staying below the internal subprocess timeout=1800
    # cap in _run_rich_export -- overrides the repo-wide pytest.ini
    # --timeout=300, which would otherwise kill this heavy pipeline run
    # before it ever reaches the byte comparison.
    golden_path = GOLDEN_DIR / f"{clip_name}_with_individual.csv"
    if not golden_path.exists():
        pytest.fail(
            f"Golden missing: {golden_path}. Capture it per "
            "tests/goldens/rich_export (see test_rich_export_golden.py docstring)."
        )
    if not _fixtures_available(clip_name):
        pytest.skip(
            f"{clip_name} equivalence fixtures missing "
            "(run tools/equivalence/fixtures/fetch_fixtures.sh)."
        )
    if _foreign_tracker_running():
        pytest.skip(
            "A foreign trackerkit/headless_tracking/run_matrix.sh process is "
            "already running on this box -- refusing to launch a competing "
            "tracker run (see CLAUDE.md box-sharing convention)."
        )

    outdir = tmp_path / clip_name
    produced_path = _run_rich_export(clip_name, outdir)

    golden_text = golden_path.read_text()
    produced_text = produced_path.read_text()

    golden_lines = golden_text.splitlines()
    produced_lines = produced_text.splitlines()
    assert len(golden_lines) > 1, f"Golden {golden_path} is empty (header-only)."
    assert (
        len(produced_lines) > 1
    ), f"Produced CSV {produced_path} is empty (header-only)."

    assert produced_text == golden_text, (
        f"Rich export for {clip_name} diverged from the committed pre-vectorization "
        f"golden ({golden_path}). Row counts: golden={len(golden_lines)} "
        f"produced={len(produced_lines)}. This localizes a real behavior change in "
        "the postproc vectorization slices (Tasks 5-8) -- do not edit the golden "
        "to silence this failure."
    )


def test_rich_export_goldens_are_nonempty_with_full_schema():
    """Lightweight, always-on companion check: goldens exist, are non-empty, and
    carry the expected rich schema (pose triples, identity block, quality/temporal
    columns). Runs in the default (non-slow) suite -- does not touch the tracker.
    """
    expected_pose_cols = {
        "PoseKpt_left_antenna_tip_X",
        "PoseKpt_left_antenna_tip_Y",
        "PoseKpt_left_antenna_tip_Conf",
        "PoseKpt_tip_of_gaster_X",
        "PoseKpt_tip_of_gaster_Y",
        "PoseKpt_tip_of_gaster_Conf",
    }
    expected_quality_cols = {
        "PoseQualityScore",
        "PoseQualityState",
        "PoseQualityFlags",
        "PoseMeanConf",
        "PoseValidFraction",
    }
    expected_temporal_cols = {
        "FrameID",
        "TrajectoryID",
        "HeadingMethod",
        "HeadingResolved",
    }
    expected_identity_cols = {
        "IdentityEvidenceSources",
        "IdentityEvidenceConflictFlag",
        "IdentityFinalLabel",
        "IdentityFinalConfidence",
    }

    for clip_name in CLIPS:
        golden_path = GOLDEN_DIR / f"{clip_name}_with_individual.csv"
        assert golden_path.exists(), f"Golden missing: {golden_path}"
        lines = golden_path.read_text().splitlines()
        assert len(lines) > 1, f"Golden {golden_path} has no data rows."
        header = lines[0].split(",")
        cols = set(header)
        missing_pose = expected_pose_cols - cols
        missing_quality = expected_quality_cols - cols
        missing_temporal = expected_temporal_cols - cols
        missing_identity = expected_identity_cols - cols
        assert not missing_pose, f"{clip_name}: missing pose columns {missing_pose}"
        assert (
            not missing_quality
        ), f"{clip_name}: missing quality columns {missing_quality}"
        assert (
            not missing_temporal
        ), f"{clip_name}: missing temporal columns {missing_temporal}"
        assert (
            not missing_identity
        ), f"{clip_name}: missing identity columns {missing_identity}"
