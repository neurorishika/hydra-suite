import numpy as np
import pandas as pd

from hydra_suite.core.individual.identity import columns as C
from hydra_suite.core.post.processing import resolve_trajectories


def _traj(traj_id, x0, n=30, arena_id=0):
    # NOTE: production trajectory DataFrames use the capitalized schema
    # (TrajectoryID/FrameID/X/Y/Theta) -- see tests/test_post_should_stop.py
    # ``_make_traj_df``. ``arena_id`` itself is the lowercase raw-CSV column
    # per the Task 6 contract.
    return pd.DataFrame(
        {
            "FrameID": np.arange(n),
            "X": np.full(n, float(x0)),
            "Y": np.full(n, 10.0),
            "Theta": np.zeros(n),
            "TrajectoryID": traj_id,
            "arena_id": arena_id,
        }
    )


PARAMS = {
    "AGREEMENT_DISTANCE": 15.0,
    "MIN_OVERLAP_FRAMES": 5,
    "MIN_TRAJECTORY_LENGTH": 5,
}


def test_uniform_slot_arena_matches_ungrouped_result():
    """Single-arena parity: grouping must be a no-op when there is one arena."""
    fwd = [_traj(0, 10.0), _traj(1, 200.0)]
    bwd = [_traj(0, 10.0), _traj(1, 200.0)]
    ungrouped = resolve_trajectories(fwd, bwd, PARAMS)
    grouped = resolve_trajectories(
        fwd, bwd, PARAMS, slot_arena=np.zeros(2, dtype=np.int32)
    )
    assert len(ungrouped) == len(grouped)
    for a, b in zip(ungrouped, grouped):
        pd.testing.assert_frame_equal(
            a.reset_index(drop=True), b.reset_index(drop=True)
        )


def test_spatially_coincident_trajectories_in_different_arenas_never_merge():
    """Two arenas whose tracks sit at identical coordinates must stay separate.

    Without arena grouping these are perfect merge candidates -- this is the
    exact cross-arena contamination the feature must prevent.
    """
    fwd = [_traj(0, 50.0, arena_id=0), _traj(1, 50.0, arena_id=1)]
    bwd = [_traj(0, 50.0, arena_id=0), _traj(1, 50.0, arena_id=1)]
    out = resolve_trajectories(
        fwd, bwd, PARAMS, slot_arena=np.array([0, 1], dtype=np.int32)
    )
    arenas = [int(df["arena_id"].iloc[0]) for df in out]
    assert sorted(arenas) == [0, 1], "each arena must retain its own trajectory"


def test_trajectory_ids_are_globally_unique_after_grouping():
    fwd = [_traj(0, 10.0, arena_id=0), _traj(1, 10.0, arena_id=1)]
    bwd = [_traj(0, 10.0, arena_id=0), _traj(1, 10.0, arena_id=1)]
    out = resolve_trajectories(
        fwd, bwd, PARAMS, slot_arena=np.array([0, 1], dtype=np.int32)
    )
    ids = [int(df["TrajectoryID"].iloc[0]) for df in out]
    assert len(ids) == len(set(ids))


def test_arena_column_survives_resolution():
    fwd = [_traj(0, 10.0, arena_id=3)]
    bwd = [_traj(0, 10.0, arena_id=3)]
    out = resolve_trajectories(
        fwd, bwd, PARAMS, slot_arena=np.array([3], dtype=np.int32)
    )
    assert all("arena_id" in df.columns for df in out)


def test_identity_sort_key_is_arena_scoped():
    """`sort_trajectories_by_identity` renumbers globally; with labels repeating
    per arena, arena 0's ids must not depend on arena 1's contents."""
    from hydra_suite.core.post.identity_postprocess import sort_trajectories_by_identity

    df = pd.DataFrame(
        {
            "FrameID": [0, 1, 0, 1, 0, 1],
            "TrajectoryID": [10, 10, 20, 20, 30, 30],
            "arena_id": [0, 0, 1, 1, 0, 0],
            "IdentityFinalLabel": ["antA"] * 2 + ["antA"] * 2 + ["antB"] * 2,
        }
    )
    out = sort_trajectories_by_identity(df)
    per_arena = out.groupby("arena_id")["TrajectoryID"].apply(lambda s: sorted(set(s)))
    # Every arena's ids form one contiguous run -> no interleaving across arenas.
    for ids in per_arena:
        assert ids == list(range(min(ids), min(ids) + len(ids)))


def test_relink_never_joins_same_label_across_arenas():
    """Relink matches fragments via UniqueIdentityKey, which repeats across arenas.

    Two arenas each hold one short fragment carrying the same identity key and
    stationary position, separated by a plausible occlusion gap -- a genuine
    relink candidate (verified below via the raw, arena-blind function).
    ``relink_trajectories_with_pose_by_arena`` (the function actually wired
    into ``rich_export.relink_and_export_rich_csv``) must not merge them.

    Note: the real call site (``rich_export.py:398``) does not invoke
    ``relink_trajectories_with_pose`` directly -- it goes through the
    arena-scoped wrapper, since ``relink_trajectories_with_pose`` itself is
    intentionally left arena-blind (only its consumer is grouped).
    """
    from hydra_suite.core.post.processing import (
        relink_trajectories_with_pose,
        relink_trajectories_with_pose_by_arena,
    )

    params = {
        "MAX_OCCLUSION_GAP": 30,
        "AGREEMENT_DISTANCE": 15.0,
        "MAX_VELOCITY_BREAK": 100.0,
    }

    def _fragment(traj_id, arena_id, frame0, n=9):
        return pd.DataFrame(
            {
                "FrameID": np.arange(frame0, frame0 + n),
                "TrajectoryID": traj_id,
                "arena_id": arena_id,
                "UniqueIdentityKey": "cnn=antA",
                "X": 10.0,
                "Y": 10.0,
                "Theta": 0.0,
            }
        )

    df = pd.concat([_fragment(0, 0, 0), _fragment(1, 1, 10)], ignore_index=True)

    # Sanity check: confirm this fixture really is a merge candidate when
    # arena-blind, so the assertion below is meaningful rather than vacuous.
    ungrouped = relink_trajectories_with_pose(df, params)
    assert (
        ungrouped["TrajectoryID"].nunique() == 1
    ), "fixture must be a genuine cross-arena merge candidate for this test to mean anything"

    grouped = relink_trajectories_with_pose_by_arena(df, params)
    assert grouped.groupby("TrajectoryID")["arena_id"].nunique().max() == 1
    assert grouped["TrajectoryID"].nunique() == 2


# ---------------------------------------------------------------------------
# postprocess_df.py:544 -- offline identity uniqueness (run_fragment_solver)
# ---------------------------------------------------------------------------
#
# The fragment solver performs an INJECTIVE fragment -> label assignment
# within each temporal-overlap connected component (see
# `_base_assignment_via_substrate` in `core/individual/identity/offline.py`).
# With labels repeating per arena, four temporally-overlapping fragments (two
# per arena) that all favor the same catalog label "ant_a" must yield ONE
# "ant_a" winner PER ARENA (2 total) when grouped -- not one winner for the
# whole file (1 total), which is what an arena-blind global solve forces.
# This mirrors tests/identity/test_honesty_fix.py's cache-driven pattern.

_ARENA_CATALOG_LABELS = ("unknown", "ant_a", "ant_b")
_ARENA_CNN_CLASSIFIERS = [
    {
        "unique_identifier": True,
        "factor_names": ["identity"],
        "class_names_per_factor": [["ant_a", "ant_b"]],
    }
]


def _arena_confident_log_probs(favor_label: str) -> np.ndarray:
    probs = np.full(len(_ARENA_CATALOG_LABELS), 0.02 / (len(_ARENA_CATALOG_LABELS) - 1))
    probs[_ARENA_CATALOG_LABELS.index(favor_label)] = 0.98
    probs /= probs.sum()
    return np.log(probs)


def _arena_fragment_solver_df():
    """4 trajectories (2 per arena, temporally overlapping within + across
    arenas) all favoring catalog label "ant_a"."""
    n = 20
    rows = []
    specs = [
        # (traj_id, arena_id, x)
        (1, 0, 0.0),
        (2, 0, 100.0),
        (3, 1, 0.0),
        (4, 1, 100.0),
    ]
    det_id = 0
    for traj_id, arena_id, x in specs:
        for f in range(n):
            rows.append(
                {
                    "TrajectoryID": traj_id,
                    "arena_id": arena_id,
                    "FrameID": f,
                    "DetectionID": det_id,
                    "X": x,
                    "Y": 0.0,
                    C.FINAL_LABEL: np.nan,
                    C.FINAL_CONFIDENCE: np.nan,
                }
            )
            det_id += 1
    df = pd.DataFrame(rows)
    df[C.FINAL_LABEL] = df[C.FINAL_LABEL].astype("float64")
    return df


def _write_arena_fragment_cache(tmp_path, df):
    from hydra_suite.core.individual.identity.cache import IdentityEvidenceCache
    from hydra_suite.core.individual.identity.evidence import IdentityEvidence

    path = tmp_path / "arena_evidence_cache.npz"
    cache = IdentityEvidenceCache(path, catalog_labels=_ARENA_CATALOG_LABELS, mode="w")
    by_frame: dict = {}
    for _, row in df.iterrows():
        frame_idx = int(row["FrameID"])
        det_id = int(row["DetectionID"])
        ev = IdentityEvidence.from_cnn(
            frame_idx, det_id, "cnn_identity", _arena_confident_log_probs("ant_a")
        )
        by_frame.setdefault(frame_idx, []).append(ev)
    for frame_idx, evidences in by_frame.items():
        cache.save_frame(frame_idx, evidences)
    cache.flush()
    return str(path)


def test_fragment_solver_uniqueness_is_scoped_per_arena(tmp_path):
    from hydra_suite.core.individual.postprocess_df import (
        apply_identity_postprocessing_to_df,
    )

    df = _arena_fragment_solver_df()
    cache_path = _write_arena_fragment_cache(tmp_path, df)
    params = {
        "ENABLE_IDENTITY_FRAGMENT_SOLVER": True,
        "CNN_CLASSIFIERS": _ARENA_CNN_CLASSIFIERS,
        "TAG_IDENTITY_LABELS": [],
    }

    result = apply_identity_postprocessing_to_df(
        df, params, identity_evidence_cache_path=cache_path
    )

    assert result is not None and not result.empty
    per_traj_label = result.groupby("TrajectoryID")[C.FINAL_LABEL].first()
    per_traj_arena = result.groupby("TrajectoryID")["arena_id"].first()
    winners_by_arena = (
        per_traj_arena[per_traj_label == "ant_a"].value_counts().to_dict()
    )
    # Arena-scoped uniqueness: exactly one "ant_a" winner in EACH arena (2
    # total) -- an arena-blind global solve would force all 4 fragments
    # (which all overlap in frame range) to compete for one shared "ant_a"
    # slot, yielding only 1 winner total across the whole file.
    assert winners_by_arena == {
        0: 1,
        1: 1,
    }, f"expected exactly one ant_a winner per arena, got {winners_by_arena}"


def test_fragment_solver_forwards_catalog_spec_on_every_per_arena_call(tmp_path):
    """Omitting ``catalog_spec`` on a per-arena call silently degrades a
    cross-product (2-model) catalog to exact-label matching, which floors
    every phase label and fabricates certainty on "unknown" (see
    ``run_fragment_solver``'s docstring and
    ``tests/identity/test_offline_phase_remap.py::
    test_fragment_solver_two_models_resolves_a_real_identity``, the oracle
    this mirrors). One trajectory per arena, each with consistent
    thorax=red + abdomen=circle evidence, must resolve to "red_circle" in
    BOTH arenas when catalog_spec is correctly forwarded on every per-arena
    solver call.
    """
    from hydra_suite.core.individual.identity.cache import IdentityEvidenceCache
    from hydra_suite.core.individual.identity.catalog import IdentityCatalog
    from hydra_suite.core.individual.identity.evidence import IdentityEvidence
    from hydra_suite.core.individual.identity.resolve import resolve_catalog_spec
    from hydra_suite.core.individual.postprocess_df import (
        apply_identity_postprocessing_to_df,
    )

    thorax_cfg = {
        "label": "thorax",
        "unique_identifier": True,
        "class_names_per_factor": [["red", "blue"]],
        "factor_names": ["dot"],
    }
    abdomen_cfg = {
        "label": "abdomen",
        "unique_identifier": True,
        "class_names_per_factor": [["square", "circle"]],
        "factor_names": ["shape"],
    }
    spec = resolve_catalog_spec([thorax_cfg, abdomen_cfg], [])
    catalog = IdentityCatalog.from_spec(spec)

    thorax_lp = np.log(np.array([0.02, 0.96, 0.02]))  # favors "red"
    abdomen_lp = np.log(np.array([0.02, 0.02, 0.96]))  # favors "circle"

    cache_path = tmp_path / "cross_product_cache.npz"
    cache = IdentityEvidenceCache(
        cache_path,
        catalog_labels=catalog.labels,
        mode="w",
        catalog_labels_by_source={
            "thorax": ("unknown", "red", "blue"),
            "abdomen": ("unknown", "square", "circle"),
        },
    )
    for f in range(4):
        cache.save_frame(
            f,
            [
                IdentityEvidence.from_cnn(f, 5, "thorax", thorax_lp),
                IdentityEvidence.from_cnn(f, 5, "abdomen", abdomen_lp),
                IdentityEvidence.from_cnn(f, 6, "thorax", thorax_lp),
                IdentityEvidence.from_cnn(f, 6, "abdomen", abdomen_lp),
            ],
        )
    cache.flush()

    df = pd.DataFrame(
        {
            "TrajectoryID": [1] * 4 + [2] * 4,
            "arena_id": [0] * 4 + [1] * 4,
            "FrameID": list(range(4)) * 2,
            "DetectionID": [5] * 4 + [6] * 4,
            "X": [1.0] * 8,
            "Y": [1.0] * 8,
            C.FINAL_LABEL: np.nan,
            C.FINAL_CONFIDENCE: np.nan,
        }
    )
    df[C.FINAL_LABEL] = df[C.FINAL_LABEL].astype("float64")

    params = {
        "ENABLE_IDENTITY_FRAGMENT_SOLVER": True,
        "CNN_CLASSIFIERS": [thorax_cfg, abdomen_cfg],
        "TAG_IDENTITY_LABELS": [],
    }
    result = apply_identity_postprocessing_to_df(
        df, params, identity_evidence_cache_path=str(cache_path)
    )

    # Group by arena_id, not TrajectoryID: sort_trajectories_by_identity (the
    # next stage in the pipeline) renumbers TrajectoryID, so arena_id is the
    # only stable handle back to "which input trajectory".
    labels = result.groupby("arena_id")[C.FINAL_LABEL].first().to_dict()
    assert labels == {0: "red_circle", 1: "red_circle"}, (
        "catalog_spec must be forwarded on every per-arena fragment-solver "
        f"call, else the cross-product catalog degrades to exact-label "
        f"matching and floors every phase label to unknown; got {labels}"
    )


# ---------------------------------------------------------------------------
# processing.py:757 -- process_trajectories_from_csv
# ---------------------------------------------------------------------------
#
# process_trajectories_from_csv breaks/splits ONE trajectory (one raw
# TrajectoryID value) at a time -- it never compares two different
# TrajectoryID values against each other, so it cannot "merge" across
# arenas the way resolve_trajectories or relink can. Its real arena
# contamination risk is different: `df[df["TrajectoryID"] == traj_id]`
# collects ALL rows sharing that raw id value, so if two DIFFERENT arenas
# happen to reuse the same raw TrajectoryID (a real possibility -- track ids
# are allocated per Kalman track slot, and slots are arena-scoped), an
# ungrouped pass silently splices two physically distant animals' frames
# into one "trajectory" before velocity-break detection ever sees them as
# separate. Grouping by arena_id before iterating TrajectoryID values
# prevents this; _renumber_concatenated then keeps the per-arena-local ids
# dense and globally unique afterward.


def _write_raw_csv_with_colliding_traj_ids(path):
    """Two arenas that both reuse raw TrajectoryID 0, at physically distant,
    internally-consistent positions -- a genuine collision risk if the two
    arenas' rows were ever processed as a single fragment."""
    n = 20
    rows = []
    for f in range(n):
        rows.append(
            {
                "TrajectoryID": 0,
                "arena_id": 0,
                "FrameID": f,
                "X": 10.0,
                "Y": 10.0,
                "Theta": 0.0,
                "State": "active",
            }
        )
    for f in range(n):
        rows.append(
            {
                "TrajectoryID": 0,
                "arena_id": 1,
                "FrameID": f,
                "X": 500.0,
                "Y": 500.0,
                "Theta": 0.0,
                "State": "active",
            }
        )
    pd.DataFrame(rows).to_csv(path, index=False)


def test_process_trajectories_from_csv_keeps_arenas_with_colliding_raw_ids_separate(
    tmp_path,
):
    from hydra_suite.core.post.processing import process_trajectories_from_csv

    csv_path = tmp_path / "raw_colliding.csv"
    _write_raw_csv_with_colliding_traj_ids(csv_path)

    params = {
        "MIN_TRAJECTORY_LENGTH": 5,
        "MAX_VELOCITY_BREAK": 100.0,
        "MAX_OCCLUSION_GAP": 30,
    }
    result, stats = process_trajectories_from_csv(str(csv_path), params)

    assert result is not None and not result.empty
    # No cross-arena splicing: every output trajectory's X values must come
    # from exactly one of the two source positions, never a mix.
    for traj_id, group in result.groupby("TrajectoryID"):
        xs = set(group["X"].round(1).unique())
        assert xs in (
            {10.0},
            {500.0},
        ), f"trajectory {traj_id} mixes positions from both arenas: {xs}"
    # Both arenas survived as distinct trajectories with globally unique ids
    # and no dropped rows.
    assert result["TrajectoryID"].nunique() == 2
    assert len(result) == 40
    assert set(result["arena_id"].unique()) == {0, 1}
    assert result.groupby("TrajectoryID")["arena_id"].nunique().max() == 1
