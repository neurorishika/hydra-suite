from __future__ import annotations

import numpy as np
import pandas as pd

from hydra_suite.core.identity.cache import IdentityEvidenceCache
from hydra_suite.core.identity.catalog import IdentityCatalog
from hydra_suite.core.identity.classification.cnn import ClassPrediction
from hydra_suite.core.identity.evidence import IdentityEvidence
from hydra_suite.core.identity.offline import (
    run_identity_residual_assignment,
    smooth_trajectory_identity_posteriors,
    solve_fragment_identity_assignment,
    split_mixed_identity_trajectories,
)
from hydra_suite.core.tracking.evidence_emitter import IdentityEvidenceEmitter


def _log_probs(*values: float) -> np.ndarray:
    probs = np.asarray(values, dtype=np.float64)
    probs /= probs.sum()
    return np.log(np.clip(probs, 1e-300, None))


def test_identity_evidence_emitter_uses_factor_posteriors(tmp_path) -> None:
    cache_path = tmp_path / "evidence.npz"
    emitter = IdentityEvidenceEmitter(
        cache_path=cache_path,
        source_name="cnn_identity",
        class_labels_per_factor=[["mouse1", "mouse2"], ["red", "blue"]],
        runtime_signature="cpu",
    )

    preds = [
        ClassPrediction(
            det_index=420000,
            factor_names=("identity", "coat"),
            class_names=("mouse1", "red"),
            confidences=(0.91, 0.82),
        )
    ]
    posteriors = [[np.array([0.91, 0.09]), np.array([0.82, 0.18])]]

    emitter.emit_frame(42, preds, posteriors=posteriors)
    emitter.flush()

    cache = IdentityEvidenceCache(cache_path, mode="r")
    try:
        assert cache.catalog_labels == ("unknown", "mouse1", "mouse2", "red", "blue")
        frame = cache.load_frame(42)
        assert len(frame) == 1
        evidence = frame[0]
        probs = np.exp(evidence.log_probs)
        probs /= probs.sum()
        assert evidence.detection_id == 420000
        assert probs[1] > probs[2]
        assert probs[3] > probs[4]
        assert evidence.observed_mask is not None
        assert bool(evidence.observed_mask[1]) is True
        assert bool(evidence.observed_mask[3]) is True
    finally:
        cache.close()


def test_identity_evidence_emitter_maps_slot_indices_to_stable_detection_ids(
    tmp_path,
) -> None:
    cache_path = tmp_path / "evidence_ids.npz"
    emitter = IdentityEvidenceEmitter(
        cache_path=cache_path,
        source_name="cnn_identity",
        class_labels_per_factor=[["mouse1", "mouse2"]],
        runtime_signature="cpu",
    )

    preds = [
        ClassPrediction(
            det_index=0,
            factor_names=("identity",),
            class_names=("mouse2",),
            confidences=(0.88,),
        )
    ]

    emitter.emit_frame(42, preds, detection_ids=[420123])
    emitter.flush()

    cache = IdentityEvidenceCache(cache_path, mode="r")
    try:
        frame = cache.load_frame(42)
        assert len(frame) == 1
        assert frame[0].detection_id == 420123
    finally:
        cache.close()


def test_offline_smoothing_uses_detection_id_not_frame_global(tmp_path) -> None:
    cache_path = tmp_path / "per_detection.npz"
    writer = IdentityEvidenceCache(
        cache_path,
        catalog_labels=("unknown", "mouse1", "mouse2"),
        mode="w",
    )
    writer.save_frame(
        7,
        [
            IdentityEvidence.from_cnn(7, 70000, "cnn", _log_probs(0.01, 0.97, 0.02)),
            IdentityEvidence.from_cnn(7, 70001, "cnn", _log_probs(0.01, 0.02, 0.97)),
        ],
    )
    writer.flush()

    cache = IdentityEvidenceCache(cache_path, mode="r")
    catalog = IdentityCatalog.from_labels(["mouse1", "mouse2"])
    trajectories = pd.DataFrame(
        [
            {"TrajectoryID": 1, "FrameID": 7, "DetectionID": 70000},
            {"TrajectoryID": 2, "FrameID": 7, "DetectionID": 70001},
        ]
    )

    try:
        smoothed = smooth_trajectory_identity_posteriors(
            trajectories,
            cache,
            catalog,
            {},
        )
    finally:
        cache.close()

    by_traj = dict(zip(smoothed["TrajectoryID"], smoothed["IdentitySmoothedLabel"]))
    assert by_traj[1] == "mouse1"
    assert by_traj[2] == "mouse2"


def test_offline_smoothing_merges_multiple_evidence_sidecars(tmp_path) -> None:
    cache_a_path = tmp_path / "cache_a.npz"
    cache_b_path = tmp_path / "cache_b.npz"

    cache_a_writer = IdentityEvidenceCache(
        cache_a_path,
        catalog_labels=("unknown", "mouse1"),
        mode="w",
    )
    cache_a_writer.save_frame(
        11,
        [IdentityEvidence.from_cnn(11, 110000, "cnn_a", _log_probs(0.01, 0.99))],
    )
    cache_a_writer.flush()

    cache_b_writer = IdentityEvidenceCache(
        cache_b_path,
        catalog_labels=("unknown", "mouse2"),
        mode="w",
    )
    cache_b_writer.save_frame(
        11,
        [IdentityEvidence.from_cnn(11, 110001, "cnn_b", _log_probs(0.01, 0.99))],
    )
    cache_b_writer.flush()

    cache_a = IdentityEvidenceCache(cache_a_path, mode="r")
    cache_b = IdentityEvidenceCache(cache_b_path, mode="r")
    catalog = IdentityCatalog.from_labels(["mouse1", "mouse2"])
    trajectories = pd.DataFrame(
        [
            {"TrajectoryID": 1, "FrameID": 11, "DetectionID": 110000},
            {"TrajectoryID": 2, "FrameID": 11, "DetectionID": 110001},
        ]
    )

    try:
        smoothed = smooth_trajectory_identity_posteriors(
            trajectories,
            [cache_a, cache_b],
            catalog,
            {},
        )
    finally:
        cache_a.close()
        cache_b.close()

    by_traj = dict(zip(smoothed["TrajectoryID"], smoothed["IdentitySmoothedLabel"]))
    assert by_traj[1] == "mouse1"
    assert by_traj[2] == "mouse2"


def test_residual_fragment_assignment_uses_alternative_label() -> None:
    catalog = IdentityCatalog.from_labels(["mouse1", "mouse2"])
    fragments = pd.DataFrame(
        [
            {
                "FragmentID": 0,
                "TrajectoryID": 1,
                "StartFrame": 0,
                "EndFrame": 10,
                "DominantLabel": "mouse1",
                "FragmentConf": 0.95,
                "FragmentLength": 11,
                "LabelScores": {"mouse1": 0.95},
            },
            {
                "FragmentID": 1,
                "TrajectoryID": 2,
                "StartFrame": 5,
                "EndFrame": 12,
                "DominantLabel": "mouse1",
                "FragmentConf": 0.90,
                "FragmentLength": 8,
                "LabelScores": {"mouse1": 0.90, "mouse2": 0.86},
            },
        ]
    )

    assigned = solve_fragment_identity_assignment(fragments, catalog, {})
    assert (
        assigned.loc[assigned["FragmentID"] == 1, "AssignedLabel"].iloc[0] == "mouse2"
    )

    residual = run_identity_residual_assignment(
        assigned,
        {"IDENTITY_OFFLINE_AMBIGUITY_MARGIN": 0.1},
        catalog=catalog,
    )
    assert (
        residual.loc[residual["FragmentID"] == 1, "AssignedLabel"].iloc[0] == "mouse2"
    )


def test_exact_fragment_assignment_beats_greedy_choice() -> None:
    catalog = IdentityCatalog.from_labels(["mouse1", "mouse2"])
    fragments = pd.DataFrame(
        [
            {
                "FragmentID": 0,
                "TrajectoryID": 1,
                "StartFrame": 0,
                "EndFrame": 10,
                "DominantLabel": "mouse1",
                "FragmentConf": 0.90,
                "FragmentLength": 11,
                "LabelScores": {"mouse1": 0.90, "mouse2": 0.89},
            },
            {
                "FragmentID": 1,
                "TrajectoryID": 2,
                "StartFrame": 0,
                "EndFrame": 10,
                "DominantLabel": "mouse1",
                "FragmentConf": 0.88,
                "FragmentLength": 11,
                "LabelScores": {"mouse1": 0.88},
            },
        ]
    )

    solved = solve_fragment_identity_assignment(
        fragments,
        catalog,
        {"IDENTITY_OFFLINE_AMBIGUITY_MARGIN": 0.02},
    )

    assert solved.loc[solved["FragmentID"] == 0, "AssignedLabel"].iloc[0] == "mouse2"
    assert solved.loc[solved["FragmentID"] == 1, "AssignedLabel"].iloc[0] == "mouse1"


def test_global_fragment_assignment_uses_ilp_when_legacy_exact_limit_is_low() -> None:
    catalog = IdentityCatalog.from_labels(["mouse1", "mouse2"])
    fragments = pd.DataFrame(
        [
            {
                "FragmentID": 0,
                "TrajectoryID": 1,
                "StartFrame": 0,
                "EndFrame": 10,
                "DominantLabel": "mouse1",
                "FragmentConf": 0.90,
                "FragmentLength": 11,
                "LabelScores": {"mouse1": 0.90, "mouse2": 0.89},
            },
            {
                "FragmentID": 1,
                "TrajectoryID": 2,
                "StartFrame": 0,
                "EndFrame": 10,
                "DominantLabel": "mouse1",
                "FragmentConf": 0.88,
                "FragmentLength": 11,
                "LabelScores": {"mouse1": 0.88},
            },
        ]
    )

    solved = solve_fragment_identity_assignment(
        fragments,
        catalog,
        {
            "IDENTITY_OFFLINE_AMBIGUITY_MARGIN": 0.02,
            "IDENTITY_OFFLINE_EXACT_MAX_COMPONENT": 1,
        },
    )

    assert solved.loc[solved["FragmentID"] == 0, "AssignedLabel"].iloc[0] == "mouse2"
    assert solved.loc[solved["FragmentID"] == 1, "AssignedLabel"].iloc[0] == "mouse1"


def test_split_mixed_identity_trajectories_cuts_sustained_posterior_switch() -> None:
    smoothed = pd.DataFrame(
        [
            {
                "TrajectoryID": 7,
                "FrameID": frame_idx,
                "DetectionID": 70000 + frame_idx,
                "IdentitySmoothedLabel": "mouse1" if frame_idx < 5 else "mouse2",
                "IdentitySmoothedConf": 0.96 if frame_idx < 5 else 0.95,
                "IdentitySmoothedMargin": 0.70 if frame_idx < 5 else 0.72,
            }
            for frame_idx in range(10)
        ]
    )

    split_df = split_mixed_identity_trajectories(
        smoothed,
        {
            "IDENTITY_OFFLINE_SPLIT_TRAJECTORIES": True,
            "IDENTITY_OFFLINE_SPLIT_MIN_CONF": 0.8,
            "IDENTITY_OFFLINE_SPLIT_MIN_MARGIN": 0.4,
            "IDENTITY_OFFLINE_SPLIT_MIN_FRAMES": 3,
            "IDENTITY_OFFLINE_SPLIT_MAX_BRIDGE_FRAMES": 1,
        },
    )

    assert split_df["TrajectoryID"].nunique() == 2
    assert set(split_df["OriginalTrajectoryID"].tolist()) == {7}
    first_segment = split_df[split_df["TrajectoryID"] == 7]
    second_segment = split_df[split_df["TrajectoryID"] != 7]
    assert list(first_segment["FrameID"]) == [0, 1, 2, 3, 4]
    assert list(second_segment["FrameID"]) == [5, 6, 7, 8, 9]


def test_split_mixed_identity_trajectories_is_disabled_by_default() -> None:
    smoothed = pd.DataFrame(
        [
            {
                "TrajectoryID": 7,
                "FrameID": frame_idx,
                "DetectionID": 70000 + frame_idx,
                "IdentitySmoothedLabel": "mouse1" if frame_idx < 5 else "mouse2",
                "IdentitySmoothedConf": 0.96 if frame_idx < 5 else 0.95,
                "IdentitySmoothedMargin": 0.70 if frame_idx < 5 else 0.72,
            }
            for frame_idx in range(10)
        ]
    )

    out = split_mixed_identity_trajectories(
        smoothed,
        {
            "IDENTITY_OFFLINE_SPLIT_MIN_CONF": 0.8,
            "IDENTITY_OFFLINE_SPLIT_MIN_MARGIN": 0.4,
            "IDENTITY_OFFLINE_SPLIT_MIN_FRAMES": 3,
            "IDENTITY_OFFLINE_SPLIT_MAX_BRIDGE_FRAMES": 1,
        },
    )

    assert out["TrajectoryID"].nunique() == 1
    assert (
        "OriginalTrajectoryID" not in out.columns
        or out["OriginalTrajectoryID"].isna().all()
    )
