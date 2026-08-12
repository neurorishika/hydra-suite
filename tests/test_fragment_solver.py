import numpy as np
import pandas as pd

from hydra_suite.core.individual.identity import substrate
from hydra_suite.core.individual.identity.cache import IdentityEvidenceCache
from hydra_suite.core.individual.identity.catalog import IdentityCatalog
from hydra_suite.core.individual.identity.evidence import IdentityEvidence
from hydra_suite.core.individual.identity.offline import (
    _build_traj_summaries,
    _fragment_stability,
    _iterative_assign,
    detect_identity_changepoints,
    run_fragment_solver,
    solve_global_assignment,
    split_trajectories_at_changepoints,
)


def _write_cache(tmp_path, catalog_labels, evidences_by_frame):
    """Write a real IdentityEvidenceCache (mirrors tests/identity's helper)."""
    path = tmp_path / "evidence_cache.npz"
    cache = IdentityEvidenceCache(path, catalog_labels=catalog_labels, mode="w")
    for frame_idx, evidences in evidences_by_frame.items():
        cache.save_frame(frame_idx, evidences)
    cache.flush()
    return IdentityEvidenceCache(path, mode="r")


def _catalog_log_probs(catalog, known_probs: dict) -> np.ndarray:
    """Build a full-catalog normalized log-probability vector (unknown at
    index 0) from a ``{label: probability}`` mapping over the known labels
    -- the same per-frame shape a real ``IdentityEvidenceCache`` load
    produces (see ``load_trajectory_evidence``/``_evidence_dicts_for_
    fragment``), and the shape ``solve_global_assignment``'s
    ``evidence_by_traj`` entries carry.
    """
    known_labels = list(catalog.labels[1:])
    vals = np.array(
        [max(1e-6, float(known_probs.get(label, 1e-6))) for label in known_labels],
        dtype=np.float64,
    )
    probs = np.concatenate([[1e-6], vals])
    probs /= probs.sum()
    return np.log(probs)


def _evidence_by_traj_const(df, catalog, probs_by_traj: dict, tag_by_traj: dict = None):
    """Build ``{TrajectoryID: [(FrameID, catalog_log_probs), ...]}`` -- the
    exact shape ``solve_global_assignment``'s ``evidence_by_traj`` parameter
    accepts (what a real ``IdentityEvidenceCache`` load + fuse produces via
    ``load_trajectory_evidence`` in production) -- from a constant
    per-trajectory ``{label: prob}`` evidence mapping applied to every row's
    FrameID.

    Phase 7 Task 5 deleted the ``CNN_*_Prob``/``DetectedTag*`` CSV-column
    reconstruction fallback these tests used to rely on inside
    ``solve_global_assignment``/``_build_traj_summaries``; this helper
    constructs the same evidence directly, at the level those functions
    actually consume it (cache-sourced ``evidence_by_traj``), rather than
    round-tripping through on-disk cache files or hand-built wide columns.

    ``tag_by_traj`` (optional): ``{TrajectoryID: {label: prob}}`` tag
    evidence, fused with the CNN evidence via the same
    ``substrate.fuse_log_evidence`` the real cache-loading path uses to
    combine multiple sources for one detection (mirrors
    ``load_trajectory_evidence``'s per-detection fusion loop), instead of
    being kept as a separate additive ``TagLogEvidence`` term the way the
    deleted fallback modeled it -- post-Task-5 the solver only ever sees one
    fused evidence vector per detection, exactly like a real cache.
    """
    tag_by_traj = tag_by_traj or {}
    result = {}
    for traj_id, grp in df.groupby("TrajectoryID", sort=False):
        cnn_log = _catalog_log_probs(catalog, probs_by_traj.get(traj_id, {}))
        if traj_id in tag_by_traj:
            tag_log = _catalog_log_probs(catalog, tag_by_traj[traj_id])
            fused = substrate.fuse_log_evidence(catalog.uniform_log_prior(), cnn_log)
            fused = substrate.fuse_log_evidence(fused, tag_log)
        else:
            fused = cnn_log
        sequence = [(int(f), fused) for f in sorted(grp["FrameID"].tolist())]
        result[traj_id] = sequence
    return result


def _binary_probs(label: str, high: float = 0.95, low: float = 0.05) -> dict:
    """``{"blue": high, "green": low}`` if ``label == "blue"`` else swapped --
    matches the ``0.95 if label == "blue" else 0.05`` convention
    ``_make_two_fragment_df`` uses for its ``CNN_test_*_Prob`` columns."""
    return (
        {"blue": high, "green": low}
        if label == "blue"
        else {"blue": low, "green": high}
    )


def test_fragment_solver_imports():
    from hydra_suite.core.individual.identity.offline import (
        detect_identity_changepoints,
        run_fragment_solver,
        solve_global_assignment,
        split_trajectories_at_changepoints,
    )

    assert callable(run_fragment_solver)
    assert callable(detect_identity_changepoints)
    assert callable(split_trajectories_at_changepoints)
    assert callable(solve_global_assignment)


def test_solve_global_assignment_no_cache_evidence_does_not_reconstruct_from_csv():
    """Phase 7 Task 5: with no ``evidence_by_traj`` supplied, the presence of
    legacy ``CNN_*_Prob`` wide-CSV columns in ``df`` must NOT be reconstructed
    into identity evidence -- the offline solver is cache-only.

    Fixture: two trajectories with a HIGH-confidence (0.9) online label each
    (deliberately high, so the online-label prior alone -- with no CNN
    evidence at all -- favors keeping it, isolating the CNN-columns effect)
    and ``CNN_test_blue_Prob``/``CNN_test_green_Prob`` columns that strongly
    favor the OPPOSITE label (traj 1 online "blue" but CNN says green; traj
    2 online "green" but CNN says blue). Pre-Task-5, calling
    ``solve_global_assignment`` on this df WITHOUT an explicit
    ``evidence_by_traj`` reconstructs that CNN evidence from the columns and
    corrects the "swap" (t1->green, t2->blue, confirmed against the
    pre-Task-5 module directly). Post-Task-5, with no reconstruction
    available, the CNN columns are inert: the solver has nothing but the
    (self-reinforcing, high-confidence) online-label prior to go on, so it
    must retain the online labels instead (t1 stays "blue", t2 stays
    "green") -- NOT reproduce the CNN-driven swap-correction.

    (An unrelated pre-existing quirk in ``_normalize_support_scores``'s
    zero-evidence branch -- documented in
    ``tests/identity/test_honesty_fix.py`` -- means *some* label still gets
    committed even with literally no evidence at all; that quirk is why this
    test checks for the specific swap-corrected outcome rather than
    "unknown"/empty, mirroring that test's own approach.)

    This is RED against pre-Task-5 code (produces t1="green", t2="blue" via
    CSV reconstruction) and GREEN after Task 5's deletion (t1="blue",
    t2="green" -- the online labels, unperturbed by the now-inert columns).
    """
    catalog = _make_catalog()
    n = 30
    df = pd.DataFrame(
        {
            "TrajectoryID": [1] * n + [2] * n,
            "FrameID": list(range(n)) + list(range(n, 2 * n)),
            "X": [float(i) for i in range(2 * n)],
            "Y": [0.0] * (2 * n),
            "IdentityFinalLabel": ["blue"] * n + ["green"] * n,
            "IdentityFinalConfidence": [0.9] * (2 * n),
            "CNN_test_blue_Prob": [0.1] * n + [0.9] * n,
            "CNN_test_green_Prob": [0.9] * n + [0.1] * n,
        }
    )
    assert any(
        str(c).startswith("CNN_") and str(c).endswith("_Prob") for c in df.columns
    )
    params = {
        "ONLINE_PRIOR_WEIGHT": 0.1,
        "ASSIGNMENT_MARGIN_THRESHOLD": 0.05,
        "FRAGMENT_CNN_WEIGHT": 0.7,
        "MAX_VELOCITY_BREAK": 50.0,
    }

    result = solve_global_assignment(df, catalog, params)  # no evidence_by_traj

    label_t1 = result[result["TrajectoryID"] == 1]["IdentityFinalLabel"].iloc[0]
    label_t2 = result[result["TrajectoryID"] == 2]["IdentityFinalLabel"].iloc[0]
    assert not (label_t1 == "green" and label_t2 == "blue"), (
        f"CNN_*_Prob columns were reconstructed into evidence despite no "
        f"evidence_by_traj being supplied (t1={label_t1!r}, t2={label_t2!r})"
    )
    # The decisive, precise assertion: with the CNN columns inert, the
    # high-confidence online-label prior alone determines the outcome, so
    # both trajectories retain their (un-"corrected") online labels.
    assert label_t1 == "blue" and label_t2 == "green", (
        f"expected the online labels to survive untouched (t1='blue', "
        f"t2='green'), got t1={label_t1!r}, t2={label_t2!r}"
    )


def _smoothed_by_traj_from_prob_cols(df, catalog):
    """Adapter: build the new smoothed_by_traj input from a df's CNN_*_Prob
    columns, mirroring what Task 5's wiring will hand to
    detect_identity_changepoints (a stand-in for real forward-backward
    smoothing, since these tests only care about the changepoint-detection
    behavior, not smoothing itself).
    """
    known_labels = list(catalog.labels[1:])
    prob_cols = []
    for label in known_labels:
        suffix = f"_{label}_Prob"
        for col in df.columns:
            if str(col).endswith(suffix):
                prob_cols.append(col)
                break
    if not prob_cols:
        return {}

    result = {}
    for traj_id, grp in df.groupby("TrajectoryID", sort=False):
        grp_sorted = grp.sort_values("FrameID")
        sequence = []
        for frame_id, row in zip(grp_sorted["FrameID"], grp_sorted[prob_cols].values):
            known_probs = np.clip(row.astype(np.float64), 1e-6, None)
            probs = np.concatenate([[1e-6], known_probs])
            probs /= probs.sum()
            sequence.append((int(frame_id), np.log(probs)))
        result[traj_id] = sequence
    return result


def _make_catalog():
    return IdentityCatalog.from_labels(["blue", "green"])


def _make_df_with_prob_cols(n_frames=60, swap_at=30):
    """60-frame single-trajectory DataFrame with a clear CNN swap at frame 30."""
    frames = list(range(n_frames))
    traj_ids = [1] * n_frames
    blue_probs = [0.9] * swap_at + [0.1] * (n_frames - swap_at)
    green_probs = [0.1] * swap_at + [0.9] * (n_frames - swap_at)
    return pd.DataFrame(
        {
            "TrajectoryID": traj_ids,
            "FrameID": frames,
            "X": [float(i) for i in range(n_frames)],
            "Y": [0.0] * n_frames,
            "IdentityFinalLabel": ["blue"] * n_frames,
            "IdentityFinalConfidence": [0.8] * n_frames,
            "CNN_test_blue_Prob": blue_probs,
            "CNN_test_green_Prob": green_probs,
        }
    )


def test_changepoint_detects_clear_swap():
    df = _make_df_with_prob_cols(n_frames=60, swap_at=30)
    catalog = _make_catalog()
    smoothed_by_traj = _smoothed_by_traj_from_prob_cols(df, catalog)
    result = detect_identity_changepoints(
        smoothed_by_traj,
        catalog,
        {"CHANGEPOINT_PENALTY": 2.0, "MIN_FRAGMENT_FRAMES": 5},
    )
    # Trajectory 1 should have exactly one split near frame 30.
    splits = result.get(1, [])
    assert len(splits) == 1
    assert 27 <= splits[0] <= 32, f"expected split near 29-30, got {splits[0]}"


def test_changepoint_no_split_when_stable():
    df = _make_df_with_prob_cols(n_frames=60, swap_at=60)  # no swap
    catalog = _make_catalog()
    smoothed_by_traj = _smoothed_by_traj_from_prob_cols(df, catalog)
    result = detect_identity_changepoints(
        smoothed_by_traj,
        catalog,
        {"CHANGEPOINT_PENALTY": 2.0, "MIN_FRAGMENT_FRAMES": 5},
    )
    assert result.get(1, []) == [], "stable trajectory should have no splits"


def test_changepoint_no_cnn_columns_returns_empty():
    df = pd.DataFrame(
        {
            "TrajectoryID": [1] * 10,
            "FrameID": list(range(10)),
            "IdentityFinalLabel": ["blue"] * 10,
            "IdentityFinalConfidence": [0.8] * 10,
        }
    )
    catalog = _make_catalog()
    smoothed_by_traj = _smoothed_by_traj_from_prob_cols(df, catalog)
    result = detect_identity_changepoints(smoothed_by_traj, catalog, {})
    assert result == {}, "no CNN columns should produce empty changepoint dict"


def _make_two_trajectory_df():
    """Two trajectories: traj 1 has strong green CNN but online label 'blue';
    traj 2 has strong blue CNN but online label 'green'. Non-overlapping in time."""
    n = 30
    frames_t1 = list(range(0, n))
    frames_t2 = list(range(n, 2 * n))
    return pd.DataFrame(
        {
            "TrajectoryID": [1] * n + [2] * n,
            "FrameID": frames_t1 + frames_t2,
            "X": [float(i) for i in range(2 * n)],
            "Y": [0.0] * (2 * n),
            "IdentityFinalLabel": ["blue"] * n + ["green"] * n,
            "IdentityFinalConfidence": [0.3] * (2 * n),
            "CNN_test_blue_Prob": [0.1] * n + [0.9] * n,
            "CNN_test_green_Prob": [0.9] * n + [0.1] * n,
        }
    )


def _two_traj_probs():
    """Cache-sourced evidence mirroring ``_make_two_trajectory_df``'s
    ``CNN_test_blue_Prob``/``CNN_test_green_Prob`` columns: traj 1 favors
    green, traj 2 favors blue (an online-label swap the solver must fix)."""
    return {1: {"blue": 0.1, "green": 0.9}, 2: {"blue": 0.9, "green": 0.1}}


def test_solve_global_assignment_corrects_swap():
    catalog = _make_catalog()
    df = _make_two_trajectory_df()
    params = {
        "ONLINE_PRIOR_WEIGHT": 0.1,
        "ASSIGNMENT_MARGIN_THRESHOLD": 0.05,
        "FRAGMENT_CNN_WEIGHT": 0.7,
        "MAX_VELOCITY_BREAK": 50.0,
    }
    evidence_by_traj = _evidence_by_traj_const(df, catalog, _two_traj_probs())
    result = solve_global_assignment(
        df, catalog, params, evidence_by_traj=evidence_by_traj
    )
    assert "IdentityFinalLabel" in result.columns
    label_t1 = result[result["TrajectoryID"] == 1]["IdentityFinalLabel"].iloc[0]
    label_t2 = result[result["TrajectoryID"] == 2]["IdentityFinalLabel"].iloc[0]
    assert label_t1 == "green", f"expected green for traj 1, got {label_t1}"
    assert label_t2 == "blue", f"expected blue for traj 2, got {label_t2}"


def test_solve_global_assignment_uniform_labels_per_trajectory():
    """Every row within a trajectory must get the same assigned label."""
    catalog = _make_catalog()
    df = _make_two_trajectory_df()
    params = {
        "ONLINE_PRIOR_WEIGHT": 0.1,
        "ASSIGNMENT_MARGIN_THRESHOLD": 0.05,
        "FRAGMENT_CNN_WEIGHT": 0.7,
        "MAX_VELOCITY_BREAK": 50.0,
    }
    evidence_by_traj = _evidence_by_traj_const(df, catalog, _two_traj_probs())
    result = solve_global_assignment(
        df, catalog, params, evidence_by_traj=evidence_by_traj
    )
    for tid in result["TrajectoryID"].unique():
        labels = result[result["TrajectoryID"] == tid]["IdentityFinalLabel"].unique()
        assert len(labels) == 1, f"trajectory {tid} has mixed labels: {labels}"


def test_solve_global_assignment_keeps_online_label_when_margin_too_small():
    catalog = _make_catalog()
    n = 30
    df = pd.DataFrame(
        {
            "TrajectoryID": [1] * n,
            "FrameID": list(range(n)),
            "X": [float(i) for i in range(n)],
            "Y": [0.0] * n,
            "IdentityFinalLabel": ["blue"] * n,
            "IdentityFinalConfidence": [0.9] * n,
        }
    )
    params = {
        "ONLINE_PRIOR_WEIGHT": 0.25,
        "ASSIGNMENT_MARGIN_THRESHOLD": 0.20,
        "FRAGMENT_CNN_WEIGHT": 0.7,
        "MAX_VELOCITY_BREAK": 50.0,
    }
    evidence_by_traj = _evidence_by_traj_const(
        df, catalog, {1: {"blue": 0.52, "green": 0.48}}
    )
    result = solve_global_assignment(
        df, catalog, params, evidence_by_traj=evidence_by_traj
    )
    assert result.iloc[0]["IdentityFinalLabel"] == "blue"


def test_solve_global_assignment_combines_multiple_cnn_phases():
    """Pre-Task-5 this exercised the deleted multi-head partial-column
    reconstruction (``_build_cnn_label_specs``): two separate CNN phase
    heads (color/side-style factors) whose per-phase probabilities were
    multiplied together to reconstruct one composite label's probability.
    That reconstruction mechanism no longer exists in production -- a real
    ``IdentityEvidenceCache`` load already hands the solver ONE fused
    per-detection posterior (``load_trajectory_evidence`` fuses every
    source before the solver ever sees it; see
    ``smoothing.load_trajectory_evidence``). This test now supplies that
    already-fused composite probability directly (the same numeric product
    the deleted reconstruction used to compute: 0.7*0.1=0.07 for "blue",
    0.3*0.9=0.27 for "green"), preserving the outcome-level assertion
    (evidence favors "green" enough to override the "blue" online label)
    without re-testing the deleted per-phase-column parsing mechanism.
    """
    catalog = _make_catalog()
    n = 20
    df = pd.DataFrame(
        {
            "TrajectoryID": [1] * n,
            "FrameID": list(range(n)),
            "X": [float(i) for i in range(n)],
            "Y": [0.0] * n,
            "IdentityFinalLabel": ["blue"] * n,
            "IdentityFinalConfidence": [0.2] * n,
        }
    )
    params = {
        "ONLINE_PRIOR_WEIGHT": 0.05,
        "ASSIGNMENT_MARGIN_THRESHOLD": 0.01,
        "FRAGMENT_CNN_WEIGHT": 0.9,
        "MAX_VELOCITY_BREAK": 50.0,
    }
    evidence_by_traj = _evidence_by_traj_const(
        df, catalog, {1: {"blue": 0.7 * 0.1, "green": 0.3 * 0.9}}
    )

    result = solve_global_assignment(
        df, catalog, params, evidence_by_traj=evidence_by_traj
    )

    assert result.iloc[0]["IdentityFinalLabel"] == "green"


def test_solve_global_assignment_reconstructs_multihead_label_probabilities():
    """See ``test_solve_global_assignment_combines_multiple_cnn_phases``'s
    docstring: the deleted multi-head CSV reconstruction (``color`` x
    ``side`` factor product -> composite label) has no production
    equivalent post-Task-5; this supplies the same composite probability
    (0.9*0.9=0.81 for "red_left", 0.1*0.1=0.01 for "blue_right") directly as
    cache-sourced evidence.
    """
    catalog = IdentityCatalog.from_labels(["red_left", "blue_right"])
    n = 20
    df = pd.DataFrame(
        {
            "TrajectoryID": [1] * n,
            "FrameID": list(range(n)),
            "X": [float(i) for i in range(n)],
            "Y": [0.0] * n,
            "IdentityFinalLabel": ["blue_right"] * n,
            "IdentityFinalConfidence": [0.2] * n,
        }
    )
    params = {
        "ONLINE_PRIOR_WEIGHT": 0.05,
        "ASSIGNMENT_MARGIN_THRESHOLD": 0.01,
        "FRAGMENT_CNN_WEIGHT": 0.95,
        "MAX_VELOCITY_BREAK": 50.0,
    }
    evidence_by_traj = _evidence_by_traj_const(
        df,
        catalog,
        {1: {"red_left": 0.9 * 0.9, "blue_right": 0.1 * 0.1}},
    )

    result = solve_global_assignment(
        df, catalog, params, evidence_by_traj=evidence_by_traj
    )

    assert result.iloc[0]["IdentityFinalLabel"] == "red_left"


def test_run_fragment_solver_returns_dataframe():
    df = _make_df_with_prob_cols(n_frames=60, swap_at=30)
    catalog = _make_catalog()
    result = run_fragment_solver(df, catalog, {})
    assert isinstance(result, pd.DataFrame)
    assert len(result) == len(df)
    assert "IdentityFinalLabel" in result.columns


def test_run_fragment_solver_empty_df():
    catalog = _make_catalog()
    result = run_fragment_solver(pd.DataFrame(), catalog, {})
    assert isinstance(result, pd.DataFrame)


def test_run_fragment_solver_pelt_disabled_by_default():
    """With PELT disabled (default), TrajectoryIDs are unchanged."""
    df = _make_df_with_prob_cols(n_frames=60, swap_at=30)
    catalog = _make_catalog()
    result = run_fragment_solver(df, catalog, {})
    assert sorted(result["TrajectoryID"].unique()) == sorted(
        df["TrajectoryID"].unique()
    )


def test_run_fragment_solver_pelt_enabled_splits_trajectory(tmp_path):
    """With PELT enabled and a real identity-evidence cache (Task 5's
    self-sufficient wiring), a clear CNN swap sourced from the cache -- NOT
    the CNN_*_Prob CSV columns -- produces two distinct TrajectoryIDs.

    This is the end-to-end proof that run_fragment_solver's PELT dispatch
    now feeds detect_identity_changepoints the cache-sourced, smoothed
    per-frame posterior (Task 2/3's new signature) instead of the retired
    DataFrame-based reconstruction.
    """
    n_frames = 60
    swap_at = 30
    catalog = _make_catalog()
    df = _make_df_with_prob_cols(n_frames=n_frames, swap_at=swap_at)
    # Cache join key -- one detection per frame.
    df["DetectionID"] = df["FrameID"]
    # Drop the CNN_*_Prob columns: the whole point is that the split comes
    # from the cache, not these columns (which a realtime-off run would
    # never have populated anyway).
    df = df.drop(columns=["CNN_test_blue_Prob", "CNN_test_green_Prob"])

    def _lp(favor_blue: bool) -> np.ndarray:
        probs = (
            np.array([0.02, 0.96, 0.02]) if favor_blue else np.array([0.02, 0.02, 0.96])
        )
        return np.log(probs)

    evidences_by_frame = {
        f: [IdentityEvidence.from_cnn(f, f, "cnn_test", _lp(f < swap_at))]
        for f in range(n_frames)
    }
    cache = _write_cache(tmp_path, catalog.labels, evidences_by_frame)

    result = run_fragment_solver(
        df,
        catalog,
        {"ENABLE_PELT_SPLITTING": True, "CHANGEPOINT_PENALTY": 2.0},
        cache=cache,
    )
    assert isinstance(result, pd.DataFrame)
    assert (
        result["TrajectoryID"].nunique() == 2
    ), f"expected 2 trajectories after PELT split, got {result['TrajectoryID'].nunique()}"
    assert "OriginalTrajectoryID" in result.columns
    assert (result["OriginalTrajectoryID"] == 1).all()
    # The cache-sourced pipeline also writes the raw per-frame smoothed decode.
    assert "IdentityFinalSmoothedLabel" in result.columns
    assert (result["IdentityFinalSmoothedLabel"] != "").any()
    # And the fragment solver's own committed decision, independent of any
    # (absent) realtime IdentityRealtimeLabel input.
    assert "IdentityFinalLabel" in result.columns
    assert (result["IdentityFinalLabel"] != "unknown").any()


def test_run_fragment_solver_smoothing_disabled_still_produces_labels(tmp_path):
    """Phase 6 Task 7: IDENTITY_ENABLE_SMOOTHING=False must skip the
    forward-backward smoothing pass but still run PELT/assignment off the
    raw (unsmoothed) cache evidence, producing Final labels -- and default
    (True/absent) must preserve the pre-Task-7 behavior of always smoothing
    when a cache is present.
    """
    n_frames = 60
    swap_at = 30
    catalog = _make_catalog()
    df = _make_df_with_prob_cols(n_frames=n_frames, swap_at=swap_at)
    df["DetectionID"] = df["FrameID"]
    df = df.drop(columns=["CNN_test_blue_Prob", "CNN_test_green_Prob"])

    def _lp(favor_blue: bool) -> np.ndarray:
        probs = (
            np.array([0.02, 0.96, 0.02]) if favor_blue else np.array([0.02, 0.02, 0.96])
        )
        return np.log(probs)

    evidences_by_frame = {
        f: [IdentityEvidence.from_cnn(f, f, "cnn_test", _lp(f < swap_at))]
        for f in range(n_frames)
    }

    cache_unsmoothed = _write_cache(tmp_path, catalog.labels, evidences_by_frame)
    result_unsmoothed = run_fragment_solver(
        df,
        catalog,
        {"IDENTITY_ENABLE_SMOOTHING": False},
        cache=cache_unsmoothed,
    )
    assert isinstance(result_unsmoothed, pd.DataFrame)
    assert "IdentityFinalLabel" in result_unsmoothed.columns
    assert (result_unsmoothed["IdentityFinalLabel"] != "unknown").any()

    # Default (absent) and explicit True must behave identically to
    # pre-Task-7 (always-smoothed) behavior.
    cache_default = _write_cache(tmp_path, catalog.labels, evidences_by_frame)
    result_default = run_fragment_solver(df, catalog, {}, cache=cache_default)
    cache_true = _write_cache(tmp_path, catalog.labels, evidences_by_frame)
    result_true = run_fragment_solver(
        df, catalog, {"IDENTITY_ENABLE_SMOOTHING": True}, cache=cache_true
    )
    assert (
        result_default["IdentityFinalSmoothedLabel"].tolist()
        == result_true["IdentityFinalSmoothedLabel"].tolist()
    )
    assert "IdentityFinalLabel" in result_default.columns
    assert (result_default["IdentityFinalLabel"] != "unknown").any()


def test_split_trajectories_produces_two_ids_on_changepoint():
    """A 60-frame trajectory split at frame 29 becomes two trajectories."""
    df = _make_df_with_prob_cols(n_frames=60, swap_at=30)
    changepoints = {1: [29]}
    result = split_trajectories_at_changepoints(df, changepoints, {})
    ids = result["TrajectoryID"].unique()
    assert len(ids) == 2, f"expected 2 trajectories, got {len(ids)}"
    for tid in ids:
        seg = result[result["TrajectoryID"] == tid]
        frames = sorted(seg["FrameID"].values)
        assert frames == list(
            range(frames[0], frames[-1] + 1)
        ), f"trajectory {tid} has non-contiguous frames: {frames}"


def test_split_trajectories_preserves_original_id():
    df = _make_df_with_prob_cols(n_frames=60, swap_at=30)
    changepoints = {1: [29]}
    result = split_trajectories_at_changepoints(df, changepoints, {})
    assert "OriginalTrajectoryID" in result.columns
    assert (
        result["OriginalTrajectoryID"] == 1
    ).all(), "all rows should reference original trajectory 1"


def test_split_trajectories_no_changepoints_unchanged():
    df = _make_df_with_prob_cols(n_frames=40, swap_at=40)
    result = split_trajectories_at_changepoints(df, {}, {})
    assert sorted(result["TrajectoryID"].unique()) == sorted(
        df["TrajectoryID"].unique()
    )
    assert len(result) == len(df)


def test_split_trajectories_drops_short_segments():
    """A split that would produce a segment shorter than MIN_FRAGMENT_FRAMES is dropped."""
    df = _make_df_with_prob_cols(n_frames=20, swap_at=20)
    changepoints = {1: [2]}
    result = split_trajectories_at_changepoints(
        df, changepoints, {"MIN_FRAGMENT_FRAMES": 5}
    )
    assert len(result["TrajectoryID"].unique()) == 1
    assert len(result) == 17


def test_split_trajectories_preserves_all_columns():
    df = _make_df_with_prob_cols(n_frames=60, swap_at=30)
    changepoints = {1: [29]}
    result = split_trajectories_at_changepoints(df, changepoints, {})
    original_cols = set(df.columns)
    assert original_cols.issubset(
        set(result.columns)
    ), f"missing columns: {original_cols - set(result.columns)}"


def test_split_trajectories_multiple_trajectories_independent():
    """Splitting traj 1 does not affect traj 2."""
    df1 = _make_df_with_prob_cols(n_frames=60, swap_at=30)
    df2 = _make_df_with_prob_cols(n_frames=40, swap_at=40)
    df2 = df2.copy()
    df2["TrajectoryID"] = 2
    df2["FrameID"] = list(range(40))
    combined = pd.concat([df1, df2], ignore_index=True)
    changepoints = {1: [29]}
    result = split_trajectories_at_changepoints(combined, changepoints, {})
    traj2_rows = result[result["OriginalTrajectoryID"] == 2]
    assert len(traj2_rows) == 40


def test_long_consistent_track_beats_short_confident_fragment():
    """A short high-confidence fragment must NOT displace a long id-consistent track.

    Scenario (mirrors the reported bug):
    - Traj 1 (large, 100 frames): consistently labeled "blue" by online tracker
      (conf=0.80), moderate CNN for "blue" — the "real" identity holder.
    - Traj 2 (small, 5 frames): also labeled "blue" online (mislabeled) with very
      high CNN + tag evidence, but at a spatially inconsistent position.  Both
      overlap in time, so the MILP uniqueness constraint forces a choice.

    Before the fix (additive length bonus only) the small fragment won due to its
    CNN + tag + prior advantage.  The multiplicative length weighting (default 0.60)
    must discount the small fragment enough that the large track retains "blue".

    Note: the small fragment may also retain "blue" via the online-label fallback
    (dual-assignment is a separate known issue); the critical assertion is that the
    large track is not displaced.

    Phase 7 Task 5: the deleted CSV fallback kept CNN and DetectedTag evidence
    as two separate additive terms (``FRAGMENT_CNN_WEIGHT`` /
    ``FRAGMENT_TAG_WEIGHT``). A real cache instead fuses every source into
    ONE per-detection posterior before the solver ever sees it (see
    ``smoothing.load_trajectory_evidence``), so traj 2's very-high CNN +
    tag evidence is folded together here via ``substrate.fuse_log_evidence``
    (mirroring that same fusion) into a single evidence vector, passed as
    ``evidence_by_traj``. The tag confidence value (0.9999) matches what the
    deleted ``_trajectory_tag_evidence`` helper computed for a
    ``DetectedTagLabel`` with no ``DetectedTagConf``/``DetectedTagHamming``
    column present (``conf_vals = ones(n)`` clipped to ``1 - 1e-4``).
    """
    catalog = _make_catalog()

    n_large = 100
    n_small = 5
    small_start = 40  # sits in the middle of the large track's time range

    rows = []
    for f in range(n_large):
        rows.append(
            {
                "TrajectoryID": 1,
                "FrameID": f,
                "X": float(f),
                "Y": 0.0,
                # "id consistent" — online tracker labeled it "blue" throughout
                "IdentityFinalLabel": "blue",
                "IdentityFinalConfidence": 0.80,
            }
        )
    for f in range(small_start, small_start + n_small):
        rows.append(
            {
                "TrajectoryID": 2,
                "FrameID": f,
                "X": 500.0,  # far from traj 1's position — spatially inconsistent
                "Y": 500.0,
                "IdentityFinalLabel": "blue",
                "IdentityFinalConfidence": 0.95,
            }
        )

    df = pd.DataFrame(rows)

    params = {
        "FRAGMENT_CNN_WEIGHT": 0.40,
        "ONLINE_PRIOR_WEIGHT": 0.25,
        "FRAGMENT_TAG_WEIGHT": 0.15,
        "FRAGMENT_LENGTH_WEIGHT": 0.60,
        "SPATIAL_NO_NEIGHBOR_SCORE": 0.3,
        "FRAGMENT_SPATIAL_VETO_THRESHOLD": 0.05,
        "MAX_VELOCITY_BREAK": 50.0,
        "ASSIGNMENT_MARGIN_THRESHOLD": 0.05,
    }
    evidence_by_traj = _evidence_by_traj_const(
        df,
        catalog,
        probs_by_traj={
            1: {"blue": 0.70, "green": 0.30},
            2: {"blue": 0.99, "green": 0.01},
        },
        tag_by_traj={2: {"blue": 0.9999, "green": 0.0001}},
    )
    result = solve_global_assignment(
        df, catalog, params, evidence_by_traj=evidence_by_traj
    )

    label_large = result[result["TrajectoryID"] == 1]["IdentityFinalLabel"].iloc[0]

    assert (
        label_large == "blue"
    ), f"long id-consistent track should retain 'blue', got '{label_large}'"


# === Iterative-solver-specific tests ===


def test_fragment_stability_clean_vs_jittery():
    """A consistent high-margin fragment scores higher than a jittery one."""
    n = 20
    n_labels = 2
    clean = np.zeros((n, n_labels))
    clean[:, 0] = 0.9
    clean[:, 1] = 0.1

    jittery = np.zeros((n, n_labels))
    # Alternate top-1 between labels with a small margin.
    for i in range(n):
        if i % 2 == 0:
            jittery[i] = [0.55, 0.45]
        else:
            jittery[i] = [0.45, 0.55]

    s_clean = _fragment_stability(clean)
    s_jittery = _fragment_stability(jittery)
    assert s_clean > s_jittery, f"clean={s_clean}, jittery={s_jittery}"
    # Clean: agreement=1.0, margin=0.8 → 0.8.
    assert abs(s_clean - 0.8) < 1e-6
    # Jittery: agreement=0.5, margin=0.1 → 0.05.
    assert abs(s_jittery - 0.05) < 1e-6


def test_fragment_stability_no_evidence_returns_zero():
    arr = np.full((10, 2), np.nan)
    assert _fragment_stability(arr) == 0.0


def test_per_row_probs_extracts_direct_columns():
    """Pre-Task-5 this exercised the deleted ``_trajectory_per_row_probs``
    CSV-column extraction. Stability is now computed from cache-sourced
    ``evidence_by_traj`` (``_evidence_dicts_for_fragment``); this supplies a
    consistent (per-frame-identical) "blue"-favoring evidence sequence,
    which should still yield high Stability."""
    catalog = _make_catalog()
    df = _make_df_with_prob_cols(n_frames=10, swap_at=10)  # no swap: constant blue
    evidence_by_traj = _evidence_by_traj_const(
        df, catalog, {1: {"blue": 0.9, "green": 0.1}}
    )
    summaries = _build_traj_summaries(df, catalog, evidence_by_traj)
    assert len(summaries) == 1
    # The single fragment is consistent "blue" → high stability.
    assert float(summaries.iloc[0]["Stability"]) > 0.5


def test_iterative_solver_resolves_spurious_blocking_fragment():
    """The reported pathology: a spurious 5-frame fragment with confident-wrong
    label, sandwiched by long correctly-labeled fragments of a different
    identity.  The iterative solver must relabel (or Unknown) the spurious
    fragment so the long fragment that was blocked from its correct label can
    keep it.
    """
    catalog = _make_catalog()
    rows = []
    # Long blue track (correctly online-labeled) frames 0-99 at X=0..99.
    for f in range(100):
        rows.append(
            {
                "TrajectoryID": 1,
                "FrameID": f,
                "X": float(f),
                "Y": 0.0,
                "IdentityFinalLabel": "blue",
                "IdentityFinalConfidence": 0.85,
            }
        )
    # Long green track (correctly online-labeled) frames 0-99 at X=0..99 but Y=200.
    for f in range(100):
        rows.append(
            {
                "TrajectoryID": 2,
                "FrameID": f,
                "X": float(f),
                "Y": 200.0,
                "IdentityFinalLabel": "green",
                "IdentityFinalConfidence": 0.85,
            }
        )
    # Spurious 5-frame fragment claiming "green" but spatially aligned with the
    # blue track (Y=0). It should NOT keep "green" — the iterative solver should
    # see the massive Y jump between this fragment and the actual green track.
    for k, f in enumerate(range(40, 45)):
        rows.append(
            {
                "TrajectoryID": 3,
                "FrameID": f,
                "X": float(f),
                "Y": 0.0,
                "IdentityFinalLabel": "green",
                "IdentityFinalConfidence": 0.92,
            }
        )

    df = pd.DataFrame(rows)
    params = {
        "FRAGMENT_CNN_WEIGHT": 0.40,
        "FRAGMENT_TAG_WEIGHT": 0.0,
        "ONLINE_PRIOR_WEIGHT": 0.25,
        "FRAGMENT_LENGTH_WEIGHT": 0.60,
        "SPATIAL_NO_NEIGHBOR_SCORE": 0.3,
        "FRAGMENT_SPATIAL_VETO_THRESHOLD": 0.05,
        "MAX_VELOCITY_BREAK": 50.0,
        "ASSIGNMENT_MARGIN_THRESHOLD": 0.01,
    }
    evidence_by_traj = _evidence_by_traj_const(
        df,
        catalog,
        {
            1: {"blue": 0.80, "green": 0.20},
            2: {"blue": 0.20, "green": 0.80},
            3: {"blue": 0.05, "green": 0.95},
        },
    )
    result = solve_global_assignment(
        df, catalog, params, evidence_by_traj=evidence_by_traj
    )

    label_t1 = result[result["TrajectoryID"] == 1]["IdentityFinalLabel"].iloc[0]
    label_t2 = result[result["TrajectoryID"] == 2]["IdentityFinalLabel"].iloc[0]
    label_t3 = result[result["TrajectoryID"] == 3]["IdentityFinalLabel"].iloc[0]

    # The two long anchor tracks must keep their correct online labels.
    assert label_t1 == "blue", f"long blue track lost label, got {label_t1!r}"
    assert label_t2 == "green", f"long green track lost label, got {label_t2!r}"
    # The spurious fragment must NOT remain "green" (would imply impossible jump).
    # It should either be relabeled "blue" or stay Unknown.
    assert (
        label_t3 != "green"
    ), f"spurious fragment should not retain colliding 'green' label; got {label_t3!r}"


def test_iterative_solver_unknown_promotion_when_feasible():
    """An Unknown fragment with strong CNN evidence and no spatial conflict is
    promoted to its top label by the iterative solver."""
    catalog = _make_catalog()
    rows = []
    # Long blue anchor.
    for f in range(50):
        rows.append(
            {
                "TrajectoryID": 1,
                "FrameID": f,
                "X": float(f),
                "Y": 0.0,
                "IdentityFinalLabel": "blue",
                "IdentityFinalConfidence": 0.9,
            }
        )
    # Unknown fragment with strong green CNN, far from blue.
    for f in range(60, 80):
        rows.append(
            {
                "TrajectoryID": 2,
                "FrameID": f,
                "X": 500.0,
                "Y": 500.0,
                "IdentityFinalLabel": "unknown",
                "IdentityFinalConfidence": 0.0,
            }
        )

    df = pd.DataFrame(rows)
    params = {
        "FRAGMENT_CNN_WEIGHT": 0.6,
        "ONLINE_PRIOR_WEIGHT": 0.1,
        "FRAGMENT_LENGTH_WEIGHT": 0.6,
        "ASSIGNMENT_MARGIN_THRESHOLD": 0.01,
    }
    evidence_by_traj = _evidence_by_traj_const(
        df,
        catalog,
        {1: {"blue": 0.9, "green": 0.1}, 2: {"blue": 0.05, "green": 0.95}},
    )
    result = solve_global_assignment(
        df, catalog, params, evidence_by_traj=evidence_by_traj
    )
    label_t2 = result[result["TrajectoryID"] == 2]["IdentityFinalLabel"].iloc[0]
    assert label_t2 == "green", f"expected Unknown→green promotion, got {label_t2!r}"


def test_iterative_solver_monotone_gate_blocks_marginal_flips():
    """A flip whose evidence delta is below ASSIGNMENT_MARGIN_THRESHOLD must be
    rejected; the online label survives."""
    catalog = _make_catalog()
    n = 30
    df = pd.DataFrame(
        {
            "TrajectoryID": [1] * n,
            "FrameID": list(range(n)),
            "X": [float(i) for i in range(n)],
            "Y": [0.0] * n,
            "IdentityFinalLabel": ["blue"] * n,
            "IdentityFinalConfidence": [0.9] * n,
        }
    )
    params = {
        "FRAGMENT_CNN_WEIGHT": 0.7,
        "ONLINE_PRIOR_WEIGHT": 0.25,
        # Aggressive monotone gate: any plausible flip must clear 50% of the unit
        # objective.
        "ASSIGNMENT_MARGIN_THRESHOLD": 0.50,
        "FRAGMENT_LENGTH_WEIGHT": 0.6,
    }
    # CNN evidence very near 50/50 — minimal margin.
    evidence_by_traj = _evidence_by_traj_const(
        df, catalog, {1: {"blue": 0.51, "green": 0.49}}
    )
    result = solve_global_assignment(
        df, catalog, params, evidence_by_traj=evidence_by_traj
    )
    assert result.iloc[0]["IdentityFinalLabel"] == "blue"


def test_long_gap_does_not_excuse_implausible_distance():
    """Two same-identity trajectories at very distant positions, separated by a
    LONG temporal gap, must not both retain the same identity.

    Reported pathology: looking at a single identity label across the run, the
    points "flicker" between two distant locations.  The animal cannot occupy
    two arena-spanning positions simultaneously, so one of the two trajectories
    must be a misassignment; the solver must veto and relabel one of them
    (relabel to the other known identity if evidence supports it, otherwise
    Unknown).

    Why the linear ``velocity = dist / gap`` check fails to catch this:
    over a large gap (e.g. 1000 frames), ``dist = 5000`` gives velocity 5
    px/frame — well below the 50 px/frame ``MAX_VELOCITY_BREAK`` cap — so the
    spatial veto never fires even though the implied displacement is
    physically unreasonable for a freely-moving animal.

    The fix: clamp the gap denominator with ``MAX_BRIDGE_GAP_FRAMES``.  Beyond
    that window we have no evidence of the animal's path, so we cap the gap
    when checking whether the same-identity bridge is plausible.
    """
    catalog = _make_catalog()
    rows = []
    # Anchor identity "blue" sits at (0..49, Y=0), frames 0-49.  Strong CNN.
    for f in range(50):
        rows.append(
            {
                "TrajectoryID": 1,
                "FrameID": f,
                "X": float(f),
                "Y": 0.0,
                "IdentityFinalLabel": "blue",
                "IdentityFinalConfidence": 0.9,
            }
        )
    # Two short fragments online-labeled "green" at distant positions, separated
    # by a 1000-frame gap.  Linear velocity = 7071 / 991 ≈ 7.1 px/frame —
    # below the 50 px/frame cap, so the existing hard veto misses this.
    for f in range(50, 60):
        rows.append(
            {
                "TrajectoryID": 2,
                "FrameID": f,
                "X": 0.0,
                "Y": 0.0,
                "IdentityFinalLabel": "green",
                "IdentityFinalConfidence": 0.9,
            }
        )
    for f in range(1050, 1060):
        rows.append(
            {
                "TrajectoryID": 3,
                "FrameID": f,
                "X": 5000.0,
                "Y": 5000.0,
                "IdentityFinalLabel": "green",
                "IdentityFinalConfidence": 0.9,
            }
        )

    df = pd.DataFrame(rows)
    params = {
        "FRAGMENT_CNN_WEIGHT": 0.6,
        "ONLINE_PRIOR_WEIGHT": 0.1,
        "FRAGMENT_LENGTH_WEIGHT": 0.6,
        "ASSIGNMENT_MARGIN_THRESHOLD": 0.01,
        "MAX_VELOCITY_BREAK": 50.0,
    }
    evidence_by_traj = _evidence_by_traj_const(
        df,
        catalog,
        {
            1: {"blue": 0.9, "green": 0.1},
            2: {"blue": 0.05, "green": 0.95},
            3: {"blue": 0.05, "green": 0.95},
        },
    )
    result = solve_global_assignment(
        df, catalog, params, evidence_by_traj=evidence_by_traj
    )

    label_t2 = result[result["TrajectoryID"] == 2]["IdentityFinalLabel"].iloc[0]
    label_t3 = result[result["TrajectoryID"] == 3]["IdentityFinalLabel"].iloc[0]
    # Both cannot be "green": they would imply a single animal occupying
    # (0,0) and (5000,5000) at frames separated by 1000 — an implausible
    # bridge regardless of how large the gap is.
    assert not (label_t2 == "green" and label_t3 == "green"), (
        f"both distant fragments retained 'green' (t2={label_t2!r}, "
        f"t3={label_t3!r}); long gap should not excuse a 7000px jump"
    )


def test_iterative_solver_breaks_label_lockin_via_swap():
    """Greedy lock-in case: two long anchors are mutually mislabeled by the
    online tracker (each holds the *other's* identity), and they sit far apart
    spatially.  Without a swap move the iterative solver gets stuck — neither
    can flip to its true label, because the other is currently sitting on it
    and the spatial veto fires.

    Setup (positions chosen so the spatial-veto fires for both flips):
    - Track A (frames 0–99, position x=0..99 y=0): online-labeled "blue" but
      CNN strongly says green.
    - Track B (frames 200–299, position x=0..99 y=3000): online-labeled
      "green" but CNN strongly says blue.

    Distance A↔B is ~3000 px; clamped gap is 30 frames; implied velocity is
    100 px/frame, well above the 50 px/frame ``MAX_VELOCITY_BREAK`` cap, so
    each flip would be vetoed in isolation.  The swap move resolves the
    deadlock by exchanging both labels in a single atomic step.
    """
    catalog = _make_catalog()
    rows = []
    for f in range(100):
        rows.append(
            {
                "TrajectoryID": 1,
                "FrameID": f,
                "X": float(f),
                "Y": 0.0,
                "IdentityFinalLabel": "blue",
                "IdentityFinalConfidence": 0.5,
            }
        )
    for f in range(200, 300):
        rows.append(
            {
                "TrajectoryID": 2,
                "FrameID": f,
                "X": float(f - 200),
                "Y": 3000.0,
                "IdentityFinalLabel": "green",
                "IdentityFinalConfidence": 0.5,
            }
        )

    df = pd.DataFrame(rows)
    params = {
        "FRAGMENT_CNN_WEIGHT": 0.7,
        "ONLINE_PRIOR_WEIGHT": 0.05,
        "FRAGMENT_LENGTH_WEIGHT": 0.6,
        "ASSIGNMENT_MARGIN_THRESHOLD": 0.01,
        "MAX_VELOCITY_BREAK": 50.0,
    }
    evidence_by_traj = _evidence_by_traj_const(
        df,
        catalog,
        {1: {"blue": 0.05, "green": 0.95}, 2: {"blue": 0.95, "green": 0.05}},
    )
    result = solve_global_assignment(
        df, catalog, params, evidence_by_traj=evidence_by_traj
    )

    label_t1 = result[result["TrajectoryID"] == 1]["IdentityFinalLabel"].iloc[0]
    label_t2 = result[result["TrajectoryID"] == 2]["IdentityFinalLabel"].iloc[0]
    assert label_t1 == "green", (
        f"swap move should let track 1 reach 'green' despite track 2 sitting "
        f"on it; got {label_t1!r}"
    )
    assert label_t2 == "blue", (
        f"swap move should let track 2 reach 'blue' in the same exchange; "
        f"got {label_t2!r}"
    )


def test_iterative_solver_returns_assignments_dict():
    """Direct call to _iterative_assign returns one entry per fragment."""
    catalog = _make_catalog()
    df = _make_two_trajectory_df()
    evidence_by_traj = _evidence_by_traj_const(df, catalog, _two_traj_probs())
    summaries = _build_traj_summaries(df, catalog, evidence_by_traj).reset_index(
        drop=True
    )
    out = _iterative_assign(
        summaries,
        list(catalog.labels[1:]),
        {
            "FRAGMENT_CNN_WEIGHT": 0.7,
            "ONLINE_PRIOR_WEIGHT": 0.1,
            "ASSIGNMENT_MARGIN_THRESHOLD": 0.01,
        },
    )
    assert set(out.keys()) == set(range(len(summaries)))
    # Each value is either a known label or None (Unknown).
    for v in out.values():
        assert v is None or v in {"blue", "green"}


# === substrate.solve_unique_assignment base-assignment tests (Task 4) ===


def _make_two_fragment_df(t1_range, t2_range, label="blue", x_offset=0.0):
    """Two single-identity-favoring trajectories over the given frame ranges,
    both strongly favoring ``label`` with matching online labels."""
    rows = []
    for f in t1_range:
        rows.append(
            {
                "TrajectoryID": 1,
                "FrameID": f,
                "X": float(f),
                "Y": 0.0,
                "IdentityFinalLabel": label,
                "IdentityFinalConfidence": 0.9,
            }
        )
    for f in t2_range:
        rows.append(
            {
                "TrajectoryID": 2,
                "FrameID": f,
                "X": float(f) + x_offset,
                "Y": 0.0,
                "IdentityFinalLabel": label,
                "IdentityFinalConfidence": 0.9,
            }
        )
    return pd.DataFrame(rows)


def test_substrate_solver_overlapping_fragments_get_injective_assignment():
    """Two time-overlapping fragments both favoring the same identity must not
    both keep it — the substrate solver enforces uniqueness within the
    overlap group; the loser is reassigned or Unknown."""
    catalog = _make_catalog()
    df = _make_two_fragment_df(range(0, 50), range(20, 70), label="blue")
    params = {
        "FRAGMENT_CNN_WEIGHT": 0.6,
        "ONLINE_PRIOR_WEIGHT": 0.25,
        "ASSIGNMENT_MARGIN_THRESHOLD": 0.01,
    }
    probs = _binary_probs("blue")
    evidence_by_traj = _evidence_by_traj_const(df, catalog, {1: probs, 2: probs})
    result = solve_global_assignment(
        df, catalog, params, evidence_by_traj=evidence_by_traj
    )
    label_t1 = result[result["TrajectoryID"] == 1]["IdentityFinalLabel"].iloc[0]
    label_t2 = result[result["TrajectoryID"] == 2]["IdentityFinalLabel"].iloc[0]
    labels = {label_t1, label_t2}
    assert not (
        label_t1 == "blue" and label_t2 == "blue"
    ), f"overlapping fragments both got 'blue' (t1={label_t1!r}, t2={label_t2!r})"
    # One of the two must still be blue (the injective winner).
    assert "blue" in labels


def test_substrate_solver_nonoverlapping_fragments_may_share_label():
    """Two fragments that never share a frame may both legitimately be
    assigned the same identity — the substrate solver's uniqueness
    constraint must be scoped to the temporal-overlap group, not global."""
    catalog = _make_catalog()
    df = _make_two_fragment_df(range(0, 50), range(100, 150), label="blue")
    params = {
        "FRAGMENT_CNN_WEIGHT": 0.6,
        "ONLINE_PRIOR_WEIGHT": 0.25,
        "ASSIGNMENT_MARGIN_THRESHOLD": 0.01,
    }
    probs = _binary_probs("blue")
    evidence_by_traj = _evidence_by_traj_const(df, catalog, {1: probs, 2: probs})
    result = solve_global_assignment(
        df, catalog, params, evidence_by_traj=evidence_by_traj
    )
    label_t1 = result[result["TrajectoryID"] == 1]["IdentityFinalLabel"].iloc[0]
    label_t2 = result[result["TrajectoryID"] == 2]["IdentityFinalLabel"].iloc[0]
    assert (
        label_t1 == "blue" and label_t2 == "blue"
    ), f"non-overlapping fragments should both be able to keep 'blue', got t1={label_t1!r}, t2={label_t2!r}"


def test_iterative_assign_routes_through_substrate_solve_unique_assignment(
    monkeypatch,
):
    """`_iterative_assign`'s base assignment is literally produced by
    `substrate.solve_unique_assignment` — not a re-implementation."""
    from hydra_suite.core.individual.identity import offline as offline_mod
    from hydra_suite.core.individual.identity import substrate as substrate_mod

    calls = []
    original = substrate_mod.solve_unique_assignment

    def _spy(posterior_probs, num_known, display_threshold, **kwargs):
        calls.append((len(posterior_probs), num_known, display_threshold))
        return original(posterior_probs, num_known, display_threshold, **kwargs)

    monkeypatch.setattr(offline_mod.substrate, "solve_unique_assignment", _spy)

    catalog = _make_catalog()
    df = _make_two_trajectory_df()
    evidence_by_traj = _evidence_by_traj_const(df, catalog, _two_traj_probs())
    summaries = _build_traj_summaries(df, catalog, evidence_by_traj).reset_index(
        drop=True
    )
    _iterative_assign(
        summaries,
        list(catalog.labels[1:]),
        {"FRAGMENT_CNN_WEIGHT": 0.7, "ONLINE_PRIOR_WEIGHT": 0.1},
    )
    assert calls, "solve_unique_assignment was never called by _iterative_assign"
