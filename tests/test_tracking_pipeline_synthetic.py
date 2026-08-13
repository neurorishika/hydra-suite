from __future__ import annotations

import numpy as np
import pandas as pd

from tests.helpers.module_loader import load_src_module, make_cv2_stub
from tests.helpers.postproc_runner import run_interpolate, run_resolve_trajectories

kalman_mod = load_src_module(
    "hydra_suite/core/filters/kalman.py",
    "kalman_pipeline_under_test",
    stubs={"cv2": make_cv2_stub()},
)
kalman_mod.NUMBA_AVAILABLE = False
assigner_mod = load_src_module(
    "hydra_suite/core/assigners/hungarian.py",
    "assigner_pipeline_under_test",
)


def _compute_cost_matrix_numba_py(
    N,
    M,
    meas_pos,
    meas_ori,
    pred_pos,
    pred_ori,
    shapes_area,
    shapes_asp,
    prev_areas,
    prev_asps,
    S_inv_batch,
    use_maha,
    Wp,
    Wo,
    Wa,
    Wasp,
    per_track_gates,
    meas_ori_directed,
):
    cost = np.zeros((N, M), dtype=np.float32)
    for i in range(N):
        inv_S_pos = S_inv_batch[i, :2, :2]
        gate_i = per_track_gates[i]
        for j in range(M):
            diff = meas_pos[j] - pred_pos[i]
            if use_maha:
                pos_dist = float(np.sqrt(diff @ inv_S_pos @ diff))
            else:
                pos_dist = float(np.linalg.norm(diff))
            if pos_dist > gate_i:
                cost[i, j] = 1e6
                continue
            odiff = abs(pred_ori[i] - meas_ori[j])
            if odiff > np.pi:
                odiff = 2 * np.pi - odiff
            if meas_ori_directed[j] == 0:
                odiff = min(odiff, np.pi - odiff)
            area_diff = abs(shapes_area[j] - prev_areas[i])
            asp_diff = abs(shapes_asp[j] - prev_asps[i])
            cost[i, j] = Wp * pos_dist + Wo * odiff + Wa * area_diff + Wasp * asp_diff
    return cost


assigner_mod._compute_cost_matrix_numba = _compute_cost_matrix_numba_py

KalmanFilterManager = kalman_mod.KalmanFilterManager
TrackAssigner = assigner_mod.TrackAssigner


def _kalman_params() -> dict:
    return {
        "REFERENCE_BODY_SIZE": 20.0,
        "KALMAN_MATURITY_AGE": 3,
        "KALMAN_INITIAL_VELOCITY_RETENTION": 0.2,
        "KALMAN_MAX_VELOCITY_MULTIPLIER": 3.0,
        "KALMAN_NOISE_COVARIANCE": 0.03,
        "KALMAN_LONGITUDINAL_NOISE_MULTIPLIER": 5.0,
        "KALMAN_LATERAL_NOISE_MULTIPLIER": 0.1,
        "KALMAN_MEASUREMENT_NOISE_COVARIANCE": 0.1,
        "KALMAN_DAMPING": 0.95,
    }


def _assigner_params() -> dict:
    return {
        "USE_MAHALANOBIS": True,
        "W_POSITION": 1.0,
        "W_ORIENTATION": 0.2,
        "W_AREA": 0.02,
        "W_ASPECT": 0.02,
        "MAX_DISTANCE_THRESHOLD": 80.0,
        "CONTINUITY_THRESHOLD": 2,
        "ENABLE_GREEDY_ASSIGNMENT": False,
        "ENABLE_SPATIAL_OPTIMIZATION": False,
        "MIN_RESPAWN_DISTANCE": 20.0,
    }


def test_synthetic_tracking_pipeline_regression() -> None:
    n_tracks = 2
    kf = KalmanFilterManager(n_tracks, _kalman_params())
    assigner = TrackAssigner(_assigner_params())

    # Synthetic detections for two animals over 8 frames.
    frames = []
    for t in range(8):
        frames.append(
            [
                np.array(
                    [50.0 + 3 * t, 60.0 + 1.2 * t, 0.1 + 0.01 * t], dtype=np.float32
                ),
                np.array(
                    [140.0 - 2.5 * t, 90.0 + 0.8 * t, 1.3 + 0.02 * t], dtype=np.float32
                ),
            ]
        )

    # Initialize Kalman tracks from frame 0.
    for i in range(n_tracks):
        m = frames[0][i]
        kf.initialize_filter(
            i, np.array([m[0], m[1], m[2], 0.0, 0.0], dtype=np.float32)
        )

    track_states = ["active"] * n_tracks
    continuity = [5] * n_tracks
    # Local row->trajectory-id map. Trajectory-id lifecycle was moved out of
    # TrackAssigner.assign_tracks (no more trajectory_ids/next_trajectory_id
    # params; it returns (rows, cols, free_dets, identity_rejoin_pairs)).
    trajectory_ids = [0, 1]
    traj_rows = {0: [], 1: []}

    for frame_id, measurements in enumerate(frames):
        preds = kf.predict()
        shapes = [(22.0, 1.3), (21.0, 1.2)]
        last_shape_info = [(22.0, 1.3), (21.0, 1.2)]

        cost, spatial_candidates = assigner.compute_cost_matrix(
            N=n_tracks,
            measurements=measurements,
            predictions=preds,
            shapes=shapes,
            kf_manager=kf,
            last_shape_info=last_shape_info,
        )

        rows, cols, _, _ = assigner.assign_tracks(
            cost=cost,
            N=n_tracks,
            M=len(measurements),
            meas=measurements,
            track_states=track_states,
            tracking_continuity=continuity,
            kf_manager=kf,
            spatial_candidates=spatial_candidates,
        )

        for r, c in zip(rows, cols):
            kf.correct(r, measurements[c])
            tid = trajectory_ids[r]
            traj_rows[tid].append(
                {
                    "TrajectoryID": tid,
                    "FrameID": frame_id,
                    "X": float(measurements[c][0]),
                    "Y": float(measurements[c][1]),
                    "Theta": float(measurements[c][2]),
                    "State": "active",
                }
            )

    forward_df = pd.concat(
        [pd.DataFrame(v) for v in traj_rows.values()], ignore_index=True
    )
    resolved = run_resolve_trajectories(
        {"forward": forward_df, "backward": pd.DataFrame()},
        {
            "AGREEMENT_DISTANCE": 15.0,
            "MIN_OVERLAP_FRAMES": 2,
            "MIN_TRAJECTORY_LENGTH": 2,
        },
    )
    interpolated = run_interpolate(resolved, {"method": "linear", "max_gap": 2})

    assert not resolved.empty
    assert not interpolated.empty
    assert resolved["TrajectoryID"].nunique() == 2
    assert resolved.groupby("TrajectoryID")["FrameID"].is_monotonic_increasing.all()
    assert interpolated["FrameID"].min() == 0
    assert interpolated["FrameID"].max() == 7
