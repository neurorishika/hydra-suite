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
