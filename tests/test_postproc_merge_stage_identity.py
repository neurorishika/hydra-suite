"""Regression test: merge-stage identity logic must read the Realtime family.

`core/post/processing.py` runs at the MERGE stage (via `resolve_trajectories`,
called from `core/post/merge.py`), which executes BEFORE the offline solver
and the realtime->Final mirror populate any `IdentityFinal*` column. Phase 6
Task 5 wrongly migrated the merge-stage identity logic
(`_committed_identity_disagrees`/`_committed_identity_agrees`,
`resolve_simultaneous_identity_conflicts`, and friends) to read
`C.FINAL_*` columns -- which do not exist yet at merge time -- silently
killing the identity-aware merge/relink/conflict logic.

This test builds DataFrames/rows with ONLY the Realtime family present (no
Final columns at all) and asserts the identity-aware logic still fires. Run
against the buggy (Final-reading) code, every assertion here fails/is
vacuously false; against the fixed (Realtime-reading) code, they pass.
"""

from __future__ import annotations

import types

import pandas as pd

from hydra_suite.core.individual.identity import columns as C
from tests.helpers.module_loader import load_src_module


def _scipy_stub() -> dict[str, object]:
    # See tests/test_identity_conflict_resolution.py for why only
    # `scipy.interpolate` is stubbed (not the top-level `scipy` module):
    # processing.py's `C` import transitively needs the real `scipy.optimize`
    # (via hydra_suite.core.assigners.hungarian) and numba needs the real
    # top-level `scipy` for its own version check.
    interp_ns = types.SimpleNamespace(
        CubicSpline=object,
        UnivariateSpline=object,
        interp1d=object,
    )
    return {
        "scipy.interpolate": interp_ns,
    }


mod = load_src_module(
    "hydra_suite/core/post/processing.py",
    "processing_merge_stage_under_test",
    stubs=_scipy_stub(),
)

resolve_simultaneous_identity_conflicts = mod.resolve_simultaneous_identity_conflicts
_committed_identity_disagrees = mod._committed_identity_disagrees
_committed_identity_agrees = mod._committed_identity_agrees
_IDENTITY_LABEL_COL = mod._IDENTITY_LABEL_COL
_IDENTITY_ID_COL = mod._IDENTITY_ID_COL
_IDENTITY_CONF_COL = mod._IDENTITY_CONF_COL
_IDENTITY_CONFLICT_COL = mod._IDENTITY_CONFLICT_COL


def test_identity_columns_are_realtime_family_at_merge() -> None:
    """The module-level column constants must point at Realtime, not Final --
    Final doesn't exist yet at merge time."""
    assert _IDENTITY_LABEL_COL == C.REALTIME_LABEL
    assert _IDENTITY_ID_COL == C.REALTIME_ID
    assert _IDENTITY_CONF_COL == C.REALTIME_CONFIDENCE


def test_committed_identity_disagrees_uses_realtime_only_columns() -> None:
    """Two committed realtime rows with different labels disagree, even with
    zero Final columns present anywhere."""
    r1 = {C.REALTIME_COMMITTED: 1, C.REALTIME_LABEL: "antA"}
    r2 = {C.REALTIME_COMMITTED: 1, C.REALTIME_LABEL: "antB"}
    assert _committed_identity_disagrees(r1, r2) is True
    assert _committed_identity_agrees(r1, r2) is False


def test_committed_identity_agrees_uses_realtime_only_columns() -> None:
    """Two committed realtime rows with the same label agree, even with zero
    Final columns present anywhere."""
    r1 = {C.REALTIME_COMMITTED: 1, C.REALTIME_LABEL: "antA"}
    r2 = {C.REALTIME_COMMITTED: 1, C.REALTIME_LABEL: "antA"}
    assert _committed_identity_agrees(r1, r2) is True
    assert _committed_identity_disagrees(r1, r2) is False


def test_uncommitted_realtime_row_never_disagrees_or_agrees() -> None:
    """A row whose IdentityRealtimeCommitted flag is 0 (not committed) never
    participates in agreement/disagreement, regardless of its label."""
    committed = {C.REALTIME_COMMITTED: 1, C.REALTIME_LABEL: "antA"}
    uncommitted = {C.REALTIME_COMMITTED: 0, C.REALTIME_LABEL: "antB"}
    assert _committed_identity_disagrees(committed, uncommitted) is False
    assert _committed_identity_agrees(committed, uncommitted) is False


def _make_realtime_only_traj(
    frames: list[int], label: str | None, conf: float, tag_votes: int, source: str
) -> pd.DataFrame:
    """A merge-stage trajectory carrying ONLY Realtime-family identity
    columns -- no IdentityFinal* column exists anywhere in the frame, which
    is the true state of the world at merge time."""
    rows = []
    for f in frames:
        rows.append(
            {
                "FrameID": f,
                "X": float(f),
                "Y": 0.0,
                C.REALTIME_LABEL: label,
                C.REALTIME_CONFIDENCE: conf if label is not None else float("nan"),
                C.REALTIME_ID: 0 if label is not None else float("nan"),
                C.REALTIME_COMMITTED: 1 if label is not None else 0,
                "TagVotes": tag_votes,
                "_source": source,
            }
        )
    return pd.DataFrame(rows)


def test_resolve_conflicts_fires_on_realtime_only_columns() -> None:
    """With ONLY Realtime columns present (no Final columns anywhere),
    `resolve_simultaneous_identity_conflicts` must still detect the
    overlapping same-label claim and strip the loser -- proving the
    identity-aware merge logic is alive, not silently dead."""
    strong = _make_realtime_only_traj(
        [1, 2, 3], label="ant_1", conf=0.9, tag_votes=10, source="forward"
    )
    weak = _make_realtime_only_traj(
        [2, 3, 4], label="ant_1", conf=0.5, tag_votes=0, source="backward"
    )
    assert C.FINAL_LABEL not in strong.columns
    assert C.FINAL_LABEL not in weak.columns

    result = resolve_simultaneous_identity_conflicts([strong.copy(), weak.copy()])

    winner, loser = result[0], result[1]
    assert winner[C.REALTIME_LABEL].iloc[0] == "ant_1"
    assert pd.isna(loser[C.REALTIME_LABEL].iloc[0])
    assert pd.isna(loser[C.REALTIME_ID].iloc[0])
    assert float(loser[C.REALTIME_CONFIDENCE].iloc[0]) == 0.0
    assert bool(loser[_IDENTITY_CONFLICT_COL].iloc[0])
