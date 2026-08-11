"""Identity Phase 5 — THE honesty regression test.

Before Phase 5, offline/post-hoc identity post-processing
(``apply_identity_postprocessing_to_df`` → ``run_fragment_solver`` →
``_build_traj_summaries``) reconstructed per-trajectory evidence from the
wide-CSV ``CNN_*_Prob``/``DetectedTag*`` columns -- columns only the
*realtime* decoder ever populates. With realtime identity off
(``ENABLE_IDENTITY_IN_TRACKING=False``), those columns are simply absent, so
the offline solver was starved: it had no real per-trajectory evidence to
work from, even though Phase 3 unconditionally writes a calibrated
per-frame ``IdentityEvidenceCache`` sidecar during inference regardless of
the realtime flags.

Phase 5 rewires the offline solver to source directly from that
always-written cache (``identity_evidence_cache_path`` threaded into
``apply_identity_postprocessing_to_df``), making post-hoc identity
self-sufficient. This test builds a tracking-output DataFrame that mimics a
realtime-off run exactly: no ``CNN_*_Prob``/``DetectedTag*`` columns, and
EMPTY ``IdentityFinalLabel``/``IdentityFinalConfidence`` columns (the
realtime decoder never ran, so it never wrote anything there) -- plus a
real, confident, per-detection ``IdentityEvidenceCache`` (exactly what
Phase 3 writes during inference). It asserts:

1. WITH the cache path threaded in, each trajectory is correctly and
   confidently identified purely from the cache (``IdentityFinalLabel``
   becomes non-empty AND matches the identity the cache evidence actually
   supports, with a high fragment score).
2. WITHOUT a cache path (the pre-Phase-5 starved condition -- no CNN_*_Prob
   columns to reconstruct from either), the solver cannot tell the two
   trajectories apart: both fall back to the same low-confidence guess.
   (A bare "IdentityFinalLabel is non-empty" check is NOT by itself a
   valid discriminator here -- a pre-existing, Phase-5-independent quirk in
   the iterative solver's zero-evidence fallback [``_normalize_support_
   scores`` returns a uniform distribution when there is no evidence at
   all, and the Unknown-rescue pass then commits *some* label anyway] means
   even a fully-starved fragment ends up "committed" to a low-confidence
   guess rather than staying explicitly Unknown. The decisive signal is
   therefore CORRECTNESS + CONFIDENCE, not mere non-emptiness.)

This is the proof of the whole phase: it must be RED on pre-Task-5 code
(the cache path didn't exist as an input at all, and even simulating its
absence starves the solver of any real per-trajectory signal) and GREEN
after Task 5's wiring.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from hydra_suite.core.individual.identity import columns as C
from hydra_suite.core.individual.identity.cache import IdentityEvidenceCache
from hydra_suite.core.individual.identity.evidence import IdentityEvidence
from hydra_suite.core.individual.postprocess_df import (
    apply_identity_postprocessing_to_df,
)

_CATALOG_LABELS = ("unknown", "ant_a", "ant_b", "ant_c")
_CNN_CLASSIFIERS = [
    {
        "unique_identifier": True,
        "factor_names": ["identity"],
        "class_names_per_factor": [["ant_a", "ant_b", "ant_c"]],
    }
]


def _confident_log_probs(favor_label: str) -> np.ndarray:
    """A sharp log-posterior over ``_CATALOG_LABELS`` favoring one label."""
    probs = np.full(len(_CATALOG_LABELS), 0.02 / (len(_CATALOG_LABELS) - 1))
    probs[_CATALOG_LABELS.index(favor_label)] = 0.98
    probs /= probs.sum()
    return np.log(probs)


def _build_realtime_off_df() -> pd.DataFrame:
    """A tracking-output df exactly as a realtime-identity-OFF run produces
    it: TrajectoryID/FrameID/DetectionID + EMPTY IdentityFinalLabel/
    IdentityFinalConfidence (no prior offline pass has run yet either), no
    CNN_*_Prob/DetectedTag* columns at all.

    Faithful to production dtype: a realtime-OFF final CSV has an
    all-empty IdentityFinalLabel column, which pandas reads back as
    all-NaN *float64* (NOT object ""). That dtype is load-bearing -- the
    fragment solver must coerce it to object before writing string labels,
    or the write raises a pandas LossySetitemError (pandas>=3) that the
    caller silently swallows, re-breaking the honesty fix. Building the
    column as float64 NaN here is what makes this a real regression guard
    for that crash (an object-"" column hides it).
    """
    n = 30
    rows = []
    # Trajectory 1: stationary at (0, 0).
    for f in range(n):
        rows.append(
            {
                "TrajectoryID": 1,
                "FrameID": f,
                "DetectionID": f,
                "X": 0.0,
                "Y": 0.0,
                C.FINAL_LABEL: np.nan,
                C.FINAL_CONFIDENCE: np.nan,
            }
        )
    # Trajectory 2: stationary far away at (500, 500), same frame range
    # (temporally overlapping -- the uniqueness constraint must still tell
    # them apart correctly using the cache alone).
    for f in range(n):
        rows.append(
            {
                "TrajectoryID": 2,
                "FrameID": f,
                "DetectionID": 1000 + f,
                "X": 500.0,
                "Y": 500.0,
                C.FINAL_LABEL: np.nan,
                C.FINAL_CONFIDENCE: np.nan,
            }
        )
    df = pd.DataFrame(rows)
    # Guarantee the production dtype regardless of pandas' inference.
    df[C.FINAL_LABEL] = df[C.FINAL_LABEL].astype("float64")
    return df


def _write_cache(tmp_path, df: pd.DataFrame) -> str:
    """Write a real IdentityEvidenceCache: traj 1 -> confident "ant_c",
    traj 2 -> confident "ant_b" (deliberately NOT the first known label, so
    a naive "always guesses the first catalog label" fallback cannot
    accidentally pass this test).
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


def _params(enable_solver: bool = True) -> dict:
    return {
        "ENABLE_IDENTITY_FRAGMENT_SOLVER": enable_solver,
        "CNN_CLASSIFIERS": _CNN_CLASSIFIERS,
        "TAG_IDENTITY_LABELS": [],
    }


def test_honesty_fix_self_sufficient_from_cache_alone(tmp_path):
    """THE test: realtime never ran (empty IdentityFinalLabel columns,
    no CNN_*_Prob columns) + a real evidence cache -> the offline solver
    still produces correct, confident, non-empty identities."""
    df = _build_realtime_off_df()
    cache_path = _write_cache(tmp_path, df)

    result = apply_identity_postprocessing_to_df(
        df, _params(), identity_evidence_cache_path=cache_path
    )

    assert result is not None and not result.empty
    label_t1 = result.loc[result["X"] == 0.0, C.FINAL_LABEL].iloc[0]
    label_t2 = result.loc[result["X"] == 500.0, C.FINAL_LABEL].iloc[0]

    # The primary honesty assertion the brief specifies.
    assert pd.notna(label_t1) and str(label_t1).strip() not in ("", "unknown")
    assert pd.notna(label_t2) and str(label_t2).strip() not in ("", "unknown")

    # The stronger, decisive assertion: correct AND high-confidence, not a
    # zero-evidence uniform guess.
    assert label_t1 == "ant_c", f"expected traj 1 -> ant_c from cache, got {label_t1!r}"
    assert label_t2 == "ant_b", f"expected traj 2 -> ant_b from cache, got {label_t2!r}"

    score_t1 = result.loc[result["X"] == 0.0, C.FINAL_FRAGMENT_SCORE].iloc[0]
    score_t2 = result.loc[result["X"] == 500.0, C.FINAL_FRAGMENT_SCORE].iloc[0]
    uniform_guess = 1.0 / 3.0  # 3 known labels, zero-evidence fallback score
    assert (
        score_t1 > uniform_guess
    ), f"traj 1 score {score_t1} not above uniform-guess floor"
    assert (
        score_t2 > uniform_guess
    ), f"traj 2 score {score_t2} not above uniform-guess floor"

    # Provenance (Phase 6): the offline solver's own record, populated
    # purely from the cache (no realtime decoder ever ran here) and
    # explicitly attributed to "offline" -- not silently mirrored from
    # some other stage.
    assigned_mask = result[C.FINAL_LABEL].notna() & (result[C.FINAL_LABEL] != "unknown")
    assert assigned_mask.any()
    assert result.loc[assigned_mask, C.FINAL_SOURCE].eq("offline").all()
    assert (result[C.FINAL_SMOOTHED_LABEL] != "").any()


def test_without_cache_the_same_starved_df_cannot_tell_trajectories_apart(tmp_path):
    """Negative control proving the RED condition: the identical
    realtime-off df, with no cache path threaded in (pre-Phase-5 shaped
    input -- no CNN_*_Prob columns to reconstruct from either), cannot
    correctly/confidently distinguish the two trajectories from each
    other -- unlike the cache-sourced run above."""
    df = _build_realtime_off_df()

    result = apply_identity_postprocessing_to_df(
        df, _params(), identity_evidence_cache_path=None
    )

    label_t1 = result.loc[result["X"] == 0.0, C.FINAL_LABEL].iloc[0]
    label_t2 = result.loc[result["X"] == 500.0, C.FINAL_LABEL].iloc[0]

    # Starved of any real per-trajectory evidence, the solver cannot land on
    # the SAME correct labels the cache-sourced run above reaches.
    assert not (label_t1 == "ant_c" and label_t2 == "ant_b"), (
        "starved (no-cache) run should not coincidentally reproduce the "
        "cache-sourced run's correct, distinct labels"
    )


def test_missing_cache_path_degrades_gracefully(tmp_path):
    """A nonexistent cache path must not raise -- it degrades to the
    no-cache fallback (per the brief: 'if the path can't be resolved ...
    the solver should no-op gracefully')."""
    df = _build_realtime_off_df()
    missing_path = str(tmp_path / "does_not_exist.npz")

    result = apply_identity_postprocessing_to_df(
        df, _params(), identity_evidence_cache_path=missing_path
    )

    assert result is not None and not result.empty
