"""Identity Phase 6 Task 3 — the headline provenance invariant.

The offline fragment solver (``run_fragment_solver`` /
``solve_global_assignment``) must write ONLY the ``IdentityFinal*`` column
family. It must NEVER mutate an ``IdentityRealtime*`` column -- those are
owned exclusively by the online (Kalman-time) decoder. This test builds a
trajectories df with POPULATED ``IdentityRealtime*`` columns (as if the
realtime decoder had actually run) plus a real ``IdentityEvidenceCache``,
runs the offline solver over it, and asserts:

1. Every ``IdentityRealtime*`` column is byte-identical before/after (no
   clobber).
2. Every solver-assigned (non-unknown) row gets
   ``IdentityFinalSource == "offline"`` and a non-empty ``IdentityFinalLabel``.
3. ``IdentityFinalSource`` only ever takes the values ``"offline"`` or ``""``
   (this solver never claims ``"realtime"``/``"tag"`` provenance).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from hydra_suite.core.individual.identity import columns as C
from hydra_suite.core.individual.identity.cache import IdentityEvidenceCache
from hydra_suite.core.individual.identity.catalog import IdentityCatalog
from hydra_suite.core.individual.identity.evidence import IdentityEvidence
from hydra_suite.core.individual.identity.offline import run_fragment_solver

_CATALOG_LABELS = ("unknown", "ant_a", "ant_b", "ant_c")


def _confident_log_probs(favor_label: str) -> np.ndarray:
    probs = np.full(len(_CATALOG_LABELS), 0.02 / (len(_CATALOG_LABELS) - 1))
    probs[_CATALOG_LABELS.index(favor_label)] = 0.98
    probs /= probs.sum()
    return np.log(probs)


def _build_realtime_populated_df() -> pd.DataFrame:
    """A tracking-output df with POPULATED IdentityRealtime* columns (as if
    the realtime decoder actually ran and committed labels) -- deliberately
    seeded with WRONG/stale labels relative to the cache evidence below, so
    a clobber (offline overwriting realtime) or a leak (offline reading
    realtime as truth) would both be visible.
    """
    n = 30
    rows = []
    for f in range(n):
        rows.append(
            {
                "TrajectoryID": 1,
                "FrameID": f,
                "DetectionID": f,
                "X": 0.0,
                "Y": 0.0,
                C.REALTIME_LABEL: "ant_a",
                C.REALTIME_CONFIDENCE: 0.55,
                C.REALTIME_ID: 1.0,
            }
        )
    for f in range(n):
        rows.append(
            {
                "TrajectoryID": 2,
                "FrameID": f,
                "DetectionID": 1000 + f,
                "X": 500.0,
                "Y": 500.0,
                C.REALTIME_LABEL: "ant_a",
                C.REALTIME_CONFIDENCE: 0.55,
                C.REALTIME_ID: 1.0,
            }
        )
    df = pd.DataFrame(rows)
    return df


def _write_cache(tmp_path, df: pd.DataFrame) -> str:
    """Traj 1 -> confident "ant_c", traj 2 -> confident "ant_b" -- both
    deliberately DIFFERENT from the seeded (wrong) IdentityRealtimeLabel
    values above, so the offline result is distinguishable from a leaked
    realtime value.
    """
    path = tmp_path / "evidence_cache.npz"
    cache = IdentityEvidenceCache(path, catalog_labels=_CATALOG_LABELS, mode="w")
    by_frame: dict[int, list[IdentityEvidence]] = {}
    for _, row in df.iterrows():
        frame_idx = int(row["FrameID"])
        det_id = int(row["DetectionID"])
        favor = "ant_c" if row["TrajectoryID"] == 1 else "ant_b"
        ev = IdentityEvidence.from_cnn(
            frame_idx, det_id, "cnn_identity", _confident_log_probs(favor)
        )
        by_frame.setdefault(frame_idx, []).append(ev)
    for frame_idx, evidences in by_frame.items():
        cache.save_frame(frame_idx, evidences)
    cache.flush()
    return str(path)


def _params() -> dict:
    return {
        "ENABLE_IDENTITY_FRAGMENT_SOLVER": True,
        "CNN_CLASSIFIERS": [
            {
                "unique_identifier": True,
                "factor_names": ["identity"],
                "class_names_per_factor": [["ant_a", "ant_b", "ant_c"]],
            }
        ],
        "TAG_IDENTITY_LABELS": [],
    }


def test_offline_never_mutates_realtime_columns(tmp_path):
    df = _build_realtime_populated_df()
    cache_path = _write_cache(tmp_path, df)
    cache = IdentityEvidenceCache(cache_path, mode="r")
    catalog = IdentityCatalog.from_labels(list(_CATALOG_LABELS[1:]))

    realtime_cols_before = {
        col: df[col].copy() for col in df.columns if col.startswith("IdentityRealtime")
    }
    assert (
        realtime_cols_before
    ), "fixture must actually populate IdentityRealtime* columns"

    out = run_fragment_solver(df, catalog, _params(), cache=cache)

    # (a) IdentityRealtime* columns are byte-identical before/after -- the
    # offline solver never clobbered them.
    for col, before in realtime_cols_before.items():
        assert col in out.columns
        pd.testing.assert_series_equal(
            out[col], before, check_names=False, check_dtype=False
        )

    # (b) Every solver-assigned row has IdentityFinalSource == "offline" and
    # a non-empty IdentityFinalLabel.
    assigned_mask = out[C.FINAL_LABEL].notna() & (out[C.FINAL_LABEL] != "unknown")
    assert (
        assigned_mask.any()
    ), "expected the solver to confidently assign at least one row"
    assert out.loc[assigned_mask, C.FINAL_SOURCE].eq("offline").all()

    # (c) IdentityFinalSource only ever takes "offline" or "" from this
    # solver -- it never claims "realtime"/"tag" provenance.
    assert out[C.FINAL_SOURCE].isin(["offline", ""]).all()

    # Sanity: the offline result reflects the CACHE evidence (ant_c/ant_b),
    # not the seeded (wrong) realtime labels (ant_a) -- proving it wasn't
    # silently derived from IdentityRealtimeLabel.
    label_t1 = out.loc[out["X"] == 0.0, C.FINAL_LABEL].iloc[0]
    label_t2 = out.loc[out["X"] == 500.0, C.FINAL_LABEL].iloc[0]
    assert label_t1 == "ant_c"
    assert label_t2 == "ant_b"
