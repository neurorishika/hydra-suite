"""An equivalence harness must never report success for a run that crashed.

Tracking writes its CSV header before it produces any rows, so a crashed run
leaves a well-formed, existent, EMPTY file behind. Comparing two of those
satisfies every criterion vacuously -- no positions means pos_p99 is 0, no
angles means theta_mean is 0, nothing to match means unmatched is 0 -- and the
harness printed "EQUIVALENT". That happened for real: three CUDA clips OOMed on
2026-08-06 and were reported as passing.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

HARNESS = Path(__file__).resolve().parents[1] / "tools" / "equivalence"
HEADER = (
    "TrackID,TrajectoryID,Index,X,Y,Theta,FrameID,State,DetectionConfidence,"
    "AssignmentConfidence,PositionUncertainty,DetectionID\n"
)
ROW = "0,0,0,10.0,10.0,0.0,0,tracked,0.9,0.9,1.0,0\n"


def _run(a: Path, b: Path) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(HARNESS / "compare.py"), str(a), str(b)],
        capture_output=True,
        text=True,
    )


def test_header_only_csvs_are_not_equivalent(tmp_path):
    a, b = tmp_path / "a.csv", tmp_path / "b.csv"
    a.write_text(HEADER)
    b.write_text(HEADER)

    res = _run(a, b)
    assert "EQUIVALENT ✅" not in res.stdout, "a crashed run reported as passing"
    assert "NO DATA ❌" in res.stdout
    assert res.returncode == 2


def test_one_empty_side_is_not_equivalent(tmp_path):
    """The asymmetric case: one tree ran, the other crashed."""
    a, b = tmp_path / "a.csv", tmp_path / "b.csv"
    a.write_text(HEADER + ROW)
    b.write_text(HEADER)

    res = _run(a, b)
    assert "EQUIVALENT ✅" not in res.stdout
    assert "NO DATA ❌" in res.stdout
    assert res.returncode == 2


def test_real_identical_data_still_passes(tmp_path):
    """The guard must not swallow genuine equivalence."""
    a, b = tmp_path / "a.csv", tmp_path / "b.csv"
    a.write_text(HEADER + ROW)
    b.write_text(HEADER + ROW)

    res = _run(a, b)
    assert "EQUIVALENT ✅" in res.stdout
    assert res.returncode == 0


def _cmp_case(tmp_path, make_a: bool, make_b: bool) -> subprocess.CompletedProcess:
    """Drive run_matrix.sh's cmp() in isolation for the three existence cases."""
    a, b = tmp_path / "a.csv", tmp_path / "b.csv"
    if make_a:
        a.write_text(HEADER + ROW)
    if make_b:
        b.write_text(HEADER + ROW)
    script = f"""
    set -uo pipefail
    WT={HARNESS.parent}
    FAILED_CLIPS=()
    note_failure() {{ FAILED_CLIPS+=("$1: $2"); }}
    has_rows() {{ [ -f "$1" ] || return 1; [ "$(wc -l < "$1")" -gt 1 ] || return 1; }}
    {_CMP_BODY}
    cmp "{a}" "{b}" "TEST" "clip"
    echo "FAILURES=${{#FAILED_CLIPS[@]}}"
    """
    return subprocess.run(["bash", "-c", script], capture_output=True, text=True)


_CMP_BODY = (
    (HARNESS / "run_matrix.sh")
    .read_text()
    .split("cmp() {  # a b title clip", 1)[1]
    .split("\n# Performance gate", 1)[0]
)
_CMP_BODY = "cmp() {  # a b title clip" + _CMP_BODY


def test_absent_on_both_sides_is_not_a_failure(tmp_path):
    """A clip config that emits no forward CSV must not be flagged."""
    res = _cmp_case(tmp_path, make_a=False, make_b=False)
    assert "FAILURES=0" in res.stdout, res.stdout
    assert "not produced by either tree" in res.stdout


def test_absent_on_one_side_is_a_failure(tmp_path):
    """Trees disagreeing about what they produced IS a real difference."""
    res = _cmp_case(tmp_path, make_a=True, make_b=False)
    assert "FAILURES=1" in res.stdout, res.stdout
    assert "MISSING ON ONE SIDE ONLY" in res.stdout
