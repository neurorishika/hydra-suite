# src/hydra_suite/core/post/processing.py
"""
Trajectory post-processing utilities for cleaning and refining tracking data.

Optimizations:
- NumPy vectorization for distance calculations
- Numba JIT compilation for inner loops (if available)
- Parallel processing for independent trajectory operations
"""

import logging
import math
import re

import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline, UnivariateSpline, interp1d

# Import Numba from gpu_utils (handles availability detection).
# Fallback keeps post-processing importable in lightweight test environments.
try:
    from hydra_suite.utils.gpu_utils import NUMBA_AVAILABLE, njit, prange
except Exception:
    NUMBA_AVAILABLE = False

    def njit(*args, **kwargs):
        """CPU fallback for numba.njit when Numba is unavailable; returns the function unchanged."""

        def decorator(func):
            return func

        return decorator

    def prange(*args):
        """CPU fallback for numba.prange when Numba is unavailable; delegates to built-in range."""
        return range(*args)


logger = logging.getLogger(__name__)

_POSE_KPT_COL_RE = re.compile(r"^PoseKpt_(.+)_(X|Y|Conf)$")

# Create jit decorator based on availability
if NUMBA_AVAILABLE:
    from numba import jit
else:
    # Create no-op decorator when Numba not available
    def jit(*args, **kwargs):
        """CPU fallback for numba.jit when Numba is unavailable; returns the function unchanged."""

        def decorator(func):
            return func

        return decorator


# ============================================================================
# NUMBA-ACCELERATED CORE FUNCTIONS
# ============================================================================


@jit(nopython=True, cache=True)
def _compute_pairwise_distances_numba(x1, y1, frames1, x2, y2, frames2, threshold):
    """
    Compute agreeing frame count between two trajectories using Numba.

    Returns (agreeing_count, common_count)
    """
    agreeing = 0
    common = 0

    # Create frame lookup for trajectory 2
    for i in range(len(frames1)):
        f1 = frames1[i]
        x1_val = x1[i]
        y1_val = y1[i]

        # Check if NaN
        if np.isnan(x1_val) or np.isnan(y1_val):
            continue

        # Find matching frame in trajectory 2
        for j in range(len(frames2)):
            if frames2[j] == f1:
                x2_val = x2[j]
                y2_val = y2[j]

                if np.isnan(x2_val) or np.isnan(y2_val):
                    break

                common += 1
                dist = np.sqrt((x1_val - x2_val) ** 2 + (y1_val - y2_val) ** 2)
                if dist <= threshold:
                    agreeing += 1
                break

    return agreeing, common


@jit(nopython=True, cache=True)
def _compute_all_merge_candidates_numba(
    fwd_x_list,
    fwd_y_list,
    fwd_frames_list,
    fwd_starts,
    fwd_ends,
    bwd_x_list,
    bwd_y_list,
    bwd_frames_list,
    bwd_starts,
    bwd_ends,
    threshold,
    min_overlap,
):
    """
    Find all merge candidates between forward and backward trajectories.

    NOTE: Not using parallel=True due to race conditions with candidate_count.
    The sequential version is still fast due to Numba JIT.

    Returns arrays of (forward_idx, backward_idx, agreeing_count, common_count)
    """
    n_fwd = len(fwd_starts)
    n_bwd = len(bwd_starts)

    # Pre-allocate maximum possible candidates
    max_candidates = n_fwd * n_bwd
    candidates_fi = np.empty(max_candidates, dtype=np.int32)
    candidates_bi = np.empty(max_candidates, dtype=np.int32)
    candidates_agreeing = np.empty(max_candidates, dtype=np.int32)
    candidates_common = np.empty(max_candidates, dtype=np.int32)

    candidate_count = 0

    for fi in range(n_fwd):
        f_start, f_end = fwd_starts[fi], fwd_ends[fi]
        fwd_x = fwd_x_list[f_start:f_end]
        fwd_y = fwd_y_list[f_start:f_end]
        fwd_frames = fwd_frames_list[f_start:f_end]

        for bi in range(n_bwd):
            b_start, b_end = bwd_starts[bi], bwd_ends[bi]
            bwd_x = bwd_x_list[b_start:b_end]
            bwd_y = bwd_y_list[b_start:b_end]
            bwd_frames = bwd_frames_list[b_start:b_end]

            agreeing, common = _compute_pairwise_distances_numba(
                fwd_x, fwd_y, fwd_frames, bwd_x, bwd_y, bwd_frames, threshold
            )

            if agreeing >= min_overlap:
                candidates_fi[candidate_count] = fi
                candidates_bi[candidate_count] = bi
                candidates_agreeing[candidate_count] = agreeing
                candidates_common[candidate_count] = common
                candidate_count += 1

    return (
        candidates_fi[:candidate_count],
        candidates_bi[:candidate_count],
        candidates_agreeing[:candidate_count],
        candidates_common[:candidate_count],
    )


def _prepare_trajectory_arrays(traj_dfs):
    """
    Convert list of trajectory DataFrames to flat NumPy arrays for Numba.

    Returns: (x_array, y_array, frames_array, start_indices, end_indices)
    """
    if not traj_dfs:
        return (
            np.array([], dtype=np.float64),
            np.array([], dtype=np.float64),
            np.array([], dtype=np.int64),
            np.array([], dtype=np.int64),
            np.array([], dtype=np.int64),
        )

    # Calculate total size
    total_size = sum(len(df) for df in traj_dfs)

    # Pre-allocate arrays
    x_array = np.empty(total_size, dtype=np.float64)
    y_array = np.empty(total_size, dtype=np.float64)
    frames_array = np.empty(total_size, dtype=np.int64)
    starts = np.empty(len(traj_dfs), dtype=np.int64)
    ends = np.empty(len(traj_dfs), dtype=np.int64)

    offset = 0
    for i, df in enumerate(traj_dfs):
        n = len(df)
        starts[i] = offset
        ends[i] = offset + n

        x_array[offset : offset + n] = df["X"].values
        y_array[offset : offset + n] = df["Y"].values
        frames_array[offset : offset + n] = df["FrameID"].values.astype(np.int64)

        offset += n

    return x_array, y_array, frames_array, starts, ends


def _build_frame_lookup(traj_df, require_valid_x=False):
    """
    Build FrameID -> row dictionary lookup for a trajectory DataFrame.

    Preserves existing duplicate handling semantics: later duplicate FrameID rows
    overwrite earlier rows.
    """
    if traj_df is None or traj_df.empty:
        return {}

    columns = list(traj_df.columns)
    col_arrays = {col: traj_df[col].to_numpy(copy=False) for col in columns}
    frames = col_arrays["FrameID"]
    n_rows = len(traj_df)

    if require_valid_x and "X" in col_arrays:
        valid_positions = np.flatnonzero(~pd.isna(col_arrays["X"]))
    else:
        valid_positions = range(n_rows)

    lookup = {}
    for i in valid_positions:
        frame_id = frames[i]
        if frame_id in lookup:
            logger.warning(
                "_build_frame_lookup: duplicate FrameID %s; later row overwrites earlier.",
                frame_id,
            )
        lookup[frame_id] = {col: col_arrays[col][i] for col in columns}
    return lookup


def _find_merge_candidates_python(
    forward_dfs, backward_dfs, agreement_distance, min_overlap
):
    """
    Pure Python fallback for finding merge candidates.
    Used when Numba is not available or for small trajectory counts.

    Optimized with vectorized NumPy operations where possible.
    """
    merge_candidates = []

    for fi, fwd in enumerate(forward_dfs):
        # Extract arrays for vectorized operations
        fwd_frames = fwd["FrameID"].values
        fwd_x = fwd["X"].values
        fwd_y = fwd["Y"].values
        fwd_frame_set = set(fwd_frames)
        fwd_frame_to_idx = {f: i for i, f in enumerate(fwd_frames)}

        for bi, bwd in enumerate(backward_dfs):
            bwd_frames = bwd["FrameID"].values
            bwd_frame_set = set(bwd_frames)
            common_frames = fwd_frame_set.intersection(bwd_frame_set)

            if len(common_frames) < min_overlap:
                continue

            # Build index mappings for fast lookup
            bwd_frame_to_idx = {f: i for i, f in enumerate(bwd_frames)}

            bwd_x = bwd["X"].values
            bwd_y = bwd["Y"].values

            # Count agreeing frames using vectorized operations.
            common_frames_list = list(common_frames)
            fi_indices = np.fromiter(
                (fwd_frame_to_idx[f] for f in common_frames_list), dtype=np.int64
            )
            bi_indices = np.fromiter(
                (bwd_frame_to_idx[f] for f in common_frames_list), dtype=np.int64
            )

            fx = fwd_x[fi_indices]
            fy = fwd_y[fi_indices]
            bx = bwd_x[bi_indices]
            by = bwd_y[bi_indices]

            # Skip pairs where either trajectory has NaN X.
            valid_mask = (~np.isnan(fx)) & (~np.isnan(bx))
            if not np.any(valid_mask):
                continue

            dx = fx[valid_mask] - bx[valid_mask]
            dy = fy[valid_mask] - by[valid_mask]
            dist = np.sqrt(dx * dx + dy * dy)
            agreeing_frames = int(np.count_nonzero(dist <= agreement_distance))

            if agreeing_frames >= min_overlap:
                identity_agreeing = 0
                if (
                    "IdentityCommitted" in fwd.columns
                    and "IdentityAssignedLabel" in fwd.columns
                    and "IdentityCommitted" in bwd.columns
                    and "IdentityAssignedLabel" in bwd.columns
                ):
                    fwd_c = fwd["IdentityCommitted"].values
                    bwd_c = bwd["IdentityCommitted"].values
                    fwd_lbl = fwd["IdentityAssignedLabel"].values
                    bwd_lbl = bwd["IdentityAssignedLabel"].values
                    valid_indices = np.where(valid_mask)[0]
                    for k_valid, d in zip(valid_indices, dist):
                        if d > agreement_distance:
                            continue
                        fi_k = fi_indices[k_valid]
                        bi_k = bi_indices[k_valid]
                        if fwd_c[fi_k] != 1 or bwd_c[bi_k] != 1:
                            continue
                        l1 = str(fwd_lbl[fi_k] or "")
                        l2 = str(bwd_lbl[bi_k] or "")
                        if l1 and l2 and l1 == l2:
                            identity_agreeing += 1
                merge_candidates.append(
                    (fi, bi, agreeing_frames, len(common_frames), identity_agreeing)
                )

    return merge_candidates


def _enrich_candidates_with_identity(
    candidates, forward_dfs, backward_dfs, agreement_distance
):
    """Append identity-agreement count to Numba-produced (fi, bi, agreeing, common) tuples."""
    enriched = []
    for fi, bi, agreeing, total_common in candidates:
        fwd = forward_dfs[fi]
        bwd = backward_dfs[bi]
        identity_agreeing = 0
        if (
            "IdentityCommitted" in fwd.columns
            and "IdentityAssignedLabel" in fwd.columns
            and "IdentityCommitted" in bwd.columns
            and "IdentityAssignedLabel" in bwd.columns
        ):
            fwd_map = {f: i for i, f in enumerate(fwd["FrameID"].values)}
            bwd_map = {f: i for i, f in enumerate(bwd["FrameID"].values)}
            fwd_x = fwd["X"].values
            fwd_y = fwd["Y"].values
            bwd_x = bwd["X"].values
            bwd_y = bwd["Y"].values
            fwd_c = fwd["IdentityCommitted"].values
            bwd_c = bwd["IdentityCommitted"].values
            fwd_lbl = fwd["IdentityAssignedLabel"].values
            bwd_lbl = bwd["IdentityAssignedLabel"].values
            for f, fi_k in fwd_map.items():
                bi_k = bwd_map.get(f)
                if bi_k is None:
                    continue
                if pd.isna(fwd_x[fi_k]) or pd.isna(bwd_x[bi_k]):
                    continue
                dx = fwd_x[fi_k] - bwd_x[bi_k]
                dy = fwd_y[fi_k] - bwd_y[bi_k]
                if np.sqrt(dx * dx + dy * dy) > agreement_distance:
                    continue
                if fwd_c[fi_k] != 1 or bwd_c[bi_k] != 1:
                    continue
                l1 = str(fwd_lbl[fi_k] or "")
                l2 = str(bwd_lbl[bi_k] or "")
                if l1 and l2 and l1 == l2:
                    identity_agreeing += 1
        enriched.append((fi, bi, agreeing, total_common, identity_agreeing))
    return enriched


# ============================================================================
# TRAJECTORY PROCESSING FUNCTIONS
# ============================================================================


def _compute_velocity_zscore_breaks(
    traj_df,
    zscore_threshold=3.0,
    window_size=10,
    min_velocity_threshold=2.0,
    std_regularization=0.5,
    active_velocity_threshold=0.5,
):
    """
    Identify trajectory break points based on velocity z-scores.

    Detects sudden, statistically significant changes in velocity that often indicate
    identity swaps. For each point, calculates the z-score of its velocity relative to
    the rolling mean and std of recent past velocities.

    CRITICAL SAFEGUARDS to avoid false positives:
    1. Minimum velocity threshold - prevents breaking when animal starts moving from rest
    2. Regularized std - prevents extreme z-scores from low-variability periods
    3. Active velocity filtering - excludes near-stationary periods from statistics

    Args:
        traj_df (pd.DataFrame): Trajectory with 'Velocity' column already computed
        zscore_threshold (float): Z-score threshold for breaking (default: 3.0)
        window_size (int): Number of past velocities to use for statistics (default: 10)
        min_velocity_threshold (float): Minimum velocity to consider for z-score breaking (pixels/frame)
        std_regularization (float): Regularization added to std to prevent extreme z-scores (pixels/frame)
        active_velocity_threshold (float): Minimum velocity to be considered "active" (pixels/frame)

    Returns:
        list: Indices where velocity z-score exceeds threshold
    """
    if len(traj_df) < window_size + 2:
        return []  # Not enough data for z-score analysis

    velocities = traj_df["Velocity"].values
    break_indices = []

    # Start from window_size+1 to ensure we have enough history
    for i in range(window_size + 1, len(velocities)):
        current_vel = velocities[i]

        # Skip if current velocity is NaN (happens at trajectory start or after gaps)
        if pd.isna(current_vel):
            continue

        # SAFEGUARD 1: Skip if current velocity is too low
        # This prevents false breaks when animal transitions from rest to normal movement
        if current_vel < min_velocity_threshold:
            continue

        # Get past velocities within the window (excluding NaN values)
        past_velocities = velocities[max(0, i - window_size) : i]
        past_velocities = past_velocities[~pd.isna(past_velocities)]

        # Need at least 3 past velocities for meaningful statistics
        if len(past_velocities) < 3:
            continue

        # SAFEGUARD 3: Calculate statistics using only "active" velocities
        # This prevents stationary periods from creating artificially low baseline
        active_velocities = past_velocities[
            past_velocities >= active_velocity_threshold
        ]

        # If too few active velocities, fall back to all velocities
        if len(active_velocities) >= 3:
            mean_vel = np.mean(active_velocities)
            std_vel = np.std(active_velocities, ddof=1)
        else:
            mean_vel = np.mean(past_velocities)
            std_vel = np.std(past_velocities, ddof=1)

        # SAFEGUARD 2: Regularize std to prevent extreme z-scores
        # When std is very small (consistent low movement), regularization prevents
        # normal movement from triggering breaks
        regularized_std = std_vel + std_regularization

        # Calculate z-score with regularized std
        zscore = (current_vel - mean_vel) / regularized_std

        # Break if z-score exceeds threshold (only positive - sudden acceleration)
        # This now requires BOTH unusually high velocity relative to history
        # AND sufficient absolute velocity (via min_velocity_threshold check above)
        if zscore > zscore_threshold:
            break_indices.append(traj_df.index[i])

    return break_indices


def _split_segment_at_indices(segment, split_indices, min_len, new_traj_id_start):
    """Split a DataFrame segment at the given indices, keeping sub-segments >= min_len.

    Returns (sub_segments_list, next_traj_id).
    """
    result = []
    new_traj_id = new_traj_id_start

    split_indices = sorted(split_indices)
    subseg_start_idx = segment.index[0]

    for split_idx in split_indices:
        subseg = segment.loc[subseg_start_idx : split_idx - 1].copy()
        if len(subseg) >= min_len:
            subseg["TrajectoryID"] = new_traj_id
            new_traj_id += 1
            result.append(subseg)
        subseg_start_idx = split_idx

    # Add the last subsegment
    last_subseg = segment.loc[subseg_start_idx:].copy()
    if len(last_subseg) >= min_len:
        last_subseg["TrajectoryID"] = new_traj_id
        new_traj_id += 1
        result.append(last_subseg)

    return result, new_traj_id


def _split_segments_at_occlusion_gaps(
    segments, max_occlusion_gap, min_len, new_traj_id, stats
):
    """Split trajectory segments at long occlusion gaps.

    Returns (final_segments, new_traj_id).
    """
    final_segments = []
    for segment in segments:
        is_occluded = segment["State"].isin(["occluded", "lost"])
        state_changes = is_occluded.ne(is_occluded.shift()).cumsum()
        runs = segment.groupby(state_changes)

        split_indices = []
        for _run_id, run_group in runs:
            if (
                is_occluded.loc[run_group.index[0]]
                and len(run_group) > max_occlusion_gap
            ):
                # Split AFTER the occlusion gap rather than at its start.
                # This keeps the occluded frames attached to the preceding
                # segment (where _clean_trajectory strips them as trailing NaN
                # rows) and ensures the following segment starts with the
                # first active detection frame.  Splitting at the gap start
                # was creating tiny segments whose rows are all occluded/NaN
                # and that would be discarded by min_len, over-counting
                # stats["broken_occlusion"].
                run_end_pos = segment.index.get_loc(run_group.index[-1])
                if run_end_pos + 1 < len(segment):
                    next_frame_idx = segment.index[run_end_pos + 1]
                    split_indices.append(next_frame_idx)
                    stats["broken_occlusion"] += 1

        if not split_indices:
            final_segments.append(segment)
        else:
            sub_segs, new_traj_id = _split_segment_at_indices(
                segment, split_indices, min_len, new_traj_id
            )
            final_segments.extend(sub_segs)

    return final_segments, new_traj_id


def _split_segments_at_spatial_jumps(
    segments, max_vel_break, min_len, new_traj_id, stats
):
    """Split segments at large spatial discontinuities that span NaN gaps.

    This is the complement to ``_break_trajectory_at_velocity``.  That function
    uses ``diff()`` on the DataFrame, which produces NaN velocity wherever a row
    has NaN X/Y (occluded/lost state), so it cannot detect teleportation across
    an occlusion gap.  This function explicitly iterates over only valid (non-NaN)
    positions and checks ``frame_gap > 1`` — i.e. it *only* fires across gaps,
    never on consecutive frames.  The two functions together cover all cases.

    Returns (final_segments, new_traj_id).
    """
    final_segments = []
    for segment in segments:
        valid_mask = segment["X"].notna()
        if valid_mask.sum() < 2:
            final_segments.append(segment)
            continue

        valid_indices = segment[valid_mask].index.tolist()

        split_indices = []
        for i in range(1, len(valid_indices)):
            curr_idx = valid_indices[i]
            prev_idx = valid_indices[i - 1]

            curr_row = segment.loc[curr_idx]
            prev_row = segment.loc[prev_idx]

            frame_gap = curr_row["FrameID"] - prev_row["FrameID"]
            if frame_gap > 1:
                distance = np.sqrt(
                    (curr_row["X"] - prev_row["X"]) ** 2
                    + (curr_row["Y"] - prev_row["Y"]) ** 2
                )
                velocity = distance / frame_gap
                if velocity > max_vel_break:
                    split_indices.append(curr_idx)
                    stats["broken_spatial_gap"] += 1

        if not split_indices:
            final_segments.append(segment)
        else:
            sub_segs, new_traj_id = _split_segment_at_indices(
                segment, split_indices, min_len, new_traj_id
            )
            final_segments.extend(sub_segs)

    return final_segments, new_traj_id


def _break_trajectory_at_velocity(
    traj_df,
    min_len,
    max_vel_break,
    max_vel_zscore,
    vel_zscore_window,
    new_traj_id,
    stats,
):
    """Break a single trajectory at velocity-based break points.

    Operates on frame-to-frame velocity between consecutively tracked positions.
    NaN rows (occluded/lost state, inserted before this function is called) cause
    ``diff()`` to produce NaN velocity at gap boundaries — those are invisible here
    and are handled downstream by ``_split_segments_at_spatial_jumps`` instead.

    Returns (segments_list, new_traj_id).
    """
    temp_cols = ["FrameDiff", "DistDiff", "Velocity", "MaxAllowedDist"]

    traj_df["FrameDiff"] = traj_df["FrameID"].diff()
    traj_df["DistDiff"] = np.sqrt(traj_df["X"].diff() ** 2 + traj_df["Y"].diff() ** 2)
    traj_df["Velocity"] = traj_df["DistDiff"] / traj_df["FrameDiff"]
    traj_df["MaxAllowedDist"] = max_vel_break * traj_df["FrameDiff"]

    break_indices = traj_df[traj_df["Velocity"] > max_vel_break].index.tolist()
    stats["broken_velocity"] += len(break_indices)

    if max_vel_zscore > 0:
        zscore_break_indices = _compute_velocity_zscore_breaks(
            traj_df, zscore_threshold=max_vel_zscore, window_size=vel_zscore_window
        )
        stats["broken_velocity_zscore"] += len(zscore_break_indices)
        break_indices = sorted(set(break_indices + zscore_break_indices))

    segments = []
    segment_start_idx = 0
    for break_idx in break_indices:
        segment = traj_df.iloc[segment_start_idx:break_idx].copy()
        if len(segment) >= min_len:
            segment["TrajectoryID"] = new_traj_id
            new_traj_id += 1
            segment = segment.drop(columns=temp_cols, errors="ignore")
            segments.append(segment)
        segment_start_idx = break_idx

    last_segment = traj_df.iloc[segment_start_idx:].copy()
    if len(last_segment) >= min_len:
        last_segment["TrajectoryID"] = new_traj_id
        new_traj_id += 1
        last_segment = last_segment.drop(columns=temp_cols, errors="ignore")
        segments.append(last_segment)

    return segments, new_traj_id


def _clean_and_reassign_trajectories(result_df, min_len):
    """Clean trajectories and reassign sequential IDs. Returns (result_df, final_count)."""
    if "State" not in result_df.columns:
        return result_df, result_df["TrajectoryID"].nunique()

    cleaned_traj_list = []
    for traj_id in result_df["TrajectoryID"].unique():
        traj_df = result_df[result_df["TrajectoryID"] == traj_id]
        cleaned_traj = _clean_trajectory(traj_df)
        if cleaned_traj is not None and len(cleaned_traj) >= min_len:
            cleaned_traj_list.append(cleaned_traj)

    if cleaned_traj_list:
        result_df = pd.concat(cleaned_traj_list, ignore_index=True)
        old_ids = result_df["TrajectoryID"].unique()
        id_mapping = {old_id: new_id for new_id, old_id in enumerate(old_ids)}
        result_df["TrajectoryID"] = result_df["TrajectoryID"].map(id_mapping)
        return result_df, len(old_ids)
    else:
        return pd.DataFrame(), 0


def process_trajectories_from_csv(csv_path: object, params: object) -> object:
    """
    Cleans and refines trajectory data from CSV file, preserving all columns including confidence metrics.

    This function performs several steps:
    - Removes trajectories that are too short.
    - Breaks trajectories at points of impossibly high velocity or large jumps,
      which often indicate identity switches or tracking errors.
    - Preserves all columns from the input CSV (including confidence metrics)

    Args:
        csv_path (str): Path to the raw CSV file from tracking
        params (dict): The dictionary of tracking parameters.

    Returns:
        tuple: (final_trajectories_df, statistics_dict)
    """
    if pd is None:
        logger.warning("pandas is not available. Skipping trajectory post-processing.")
        return None, {}

    min_len = params.get("MIN_TRAJECTORY_LENGTH", 10)
    max_vel_break = params.get("MAX_VELOCITY_BREAK", 100.0)
    max_occlusion_gap = params.get("MAX_OCCLUSION_GAP", 30)
    max_vel_zscore = params.get("MAX_VELOCITY_ZSCORE", 0.0)
    vel_zscore_window = params.get("VELOCITY_ZSCORE_WINDOW", 10)

    try:
        df = pd.read_csv(csv_path)
        logger.info(
            f"Loaded {len(df)} rows from {csv_path} with columns: {list(df.columns)}"
        )

        columns_to_drop = ["TrackID", "Index"]
        df = df.drop(
            columns=[col for col in columns_to_drop if col in df.columns],
            errors="ignore",
        )
        if columns_to_drop:
            logger.info(
                f"Dropped columns: {[col for col in columns_to_drop if col in df.columns]}"
            )
    except Exception as e:
        logger.error(f"Failed to read CSV {csv_path}: {e}")
        return None, {}

    # Set X, Y, Theta to NaN for occluded/lost states to enable proper interpolation
    if "State" in df.columns:
        occluded_mask = df["State"].isin(["occluded", "lost"])
        if occluded_mask.any():
            logger.info(
                f"Setting X, Y, Theta to NaN for {occluded_mask.sum()} occluded/lost detections"
            )
            df.loc[occluded_mask, ["X", "Y", "Theta"]] = np.nan

    stats = {
        "original_count": df["TrajectoryID"].nunique(),
        "removed_short": 0,
        "broken_velocity": 0,
        "broken_velocity_zscore": 0,
        "broken_occlusion": 0,
        "broken_spatial_gap": 0,
        "final_count": 0,
    }

    cleaned_segments = []
    new_traj_id = 0

    for traj_id in df["TrajectoryID"].unique():
        traj_df = (
            df[df["TrajectoryID"] == traj_id]
            .sort_values("FrameID")
            .reset_index(drop=True)
        )
        if len(traj_df) < min_len:
            stats["removed_short"] += 1
            continue

        segs, new_traj_id = _break_trajectory_at_velocity(
            traj_df,
            min_len,
            max_vel_break,
            max_vel_zscore,
            vel_zscore_window,
            new_traj_id,
            stats,
        )
        cleaned_segments.extend(segs)

    # Further split segments based on long occlusion gaps
    if "State" in df.columns and max_occlusion_gap > 0:
        cleaned_segments, new_traj_id = _split_segments_at_occlusion_gaps(
            cleaned_segments, max_occlusion_gap, min_len, new_traj_id, stats
        )

    # Split segments on large spatial discontinuities across NaN gaps
    cleaned_segments, new_traj_id = _split_segments_at_spatial_jumps(
        cleaned_segments, max_vel_break, min_len, new_traj_id, stats
    )

    stats["final_count"] = len(cleaned_segments)

    if cleaned_segments:
        result_df = pd.concat(cleaned_segments, ignore_index=True)
        result_df, stats["final_count"] = _clean_and_reassign_trajectories(
            result_df, min_len
        )
    else:
        result_df = pd.DataFrame()

    logger.info(f"Post-processing stats: {stats}")
    return result_df, stats


def process_trajectories(trajectories_full: object, params: object) -> object:
    """
    Cleans and refines raw trajectory data.

    This function performs several steps:
    - Removes trajectories that are too short.
    - Breaks trajectories at points of impossibly high velocity or large jumps,
      which often indicate identity switches or tracking errors.

    Args:
        trajectories_full (list of lists): The raw trajectory data from the tracker.
        params (dict): The dictionary of tracking parameters.

    Returns:
        tuple: (final_trajectories, statistics_dict)
    """
    if pd is None:
        logger.warning("pandas is not available. Skipping trajectory post-processing.")
        return trajectories_full, {}

    min_len = params.get("MIN_TRAJECTORY_LENGTH", 10)
    max_vel_break = params.get("MAX_VELOCITY_BREAK", 100.0)
    # MAX_DISTANCE_BREAK is derived from MAX_VELOCITY_BREAK * frame_diff (computed per-point)
    max_vel_zscore = params.get("MAX_VELOCITY_ZSCORE", 0.0)  # 0.0 means disabled
    vel_zscore_window = params.get("VELOCITY_ZSCORE_WINDOW", 10)
    vel_zscore_min_vel = params.get("VELOCITY_ZSCORE_MIN_VELOCITY", 2.0)  # pixels/frame

    all_data = []
    # Note: Using original track_id as TrajectoryID for this initial dataframe.
    # A more advanced version could use the persistent trajectory_ids from the tracker.
    for track_id, traj in enumerate(trajectories_full):
        for point in traj:
            x, y, theta, frame = point
            all_data.append(
                {
                    "TrajectoryID": track_id,
                    "X": int(x),
                    "Y": int(y),
                    "Theta": theta,
                    "FrameID": int(frame),
                }
            )

    if not all_data:
        return [], {
            "original_count": 0,
            "removed_short": 0,
            "broken_velocity": 0,
            "broken_velocity_zscore": 0,
            "final_count": 0,
        }

    df = pd.DataFrame(all_data)

    stats = {
        "original_count": len([t for t in trajectories_full if t]),
        "removed_short": 0,
        "broken_velocity": 0,
        "broken_velocity_zscore": 0,
        "final_count": 0,
    }

    cleaned_segments = []
    for traj_id in df["TrajectoryID"].unique():
        traj_df = (
            df[df["TrajectoryID"] == traj_id]
            .sort_values("FrameID")
            .reset_index(drop=True)
        )
        if len(traj_df) < min_len:
            stats["removed_short"] += 1
            continue

        # Calculate frame difference, distance, and velocity between consecutive points
        traj_df["FrameDiff"] = traj_df["FrameID"].diff()
        traj_df["DistDiff"] = np.sqrt(
            traj_df["X"].diff() ** 2 + traj_df["Y"].diff() ** 2
        )
        traj_df["Velocity"] = traj_df["DistDiff"] / traj_df["FrameDiff"]

        # Identify break points: velocity exceeds threshold (handles variable frame gaps)
        break_indices = traj_df[traj_df["Velocity"] > max_vel_break].index.tolist()

        stats["broken_velocity"] += len(break_indices)

        # Add z-score based velocity breaks if enabled
        if max_vel_zscore > 0:
            # Calculate body-size-relative thresholds
            # std_regularization and active_threshold are set to ~25% and ~12.5% of min_velocity
            zscore_break_indices = _compute_velocity_zscore_breaks(
                traj_df,
                zscore_threshold=max_vel_zscore,
                window_size=vel_zscore_window,
                min_velocity_threshold=vel_zscore_min_vel,
                std_regularization=vel_zscore_min_vel * 0.25,
                active_velocity_threshold=vel_zscore_min_vel * 0.125,
            )
            stats["broken_velocity_zscore"] += len(zscore_break_indices)
            # Combine break indices and remove duplicates
            break_indices = sorted(set(break_indices + zscore_break_indices))

        # Create segments based on break points
        segment_start_idx = 0
        for break_idx in break_indices:
            segment = traj_df.iloc[segment_start_idx:break_idx]
            if len(segment) >= min_len:
                cleaned_segments.append(segment)
            segment_start_idx = break_idx

        # Add the last segment
        last_segment = traj_df.iloc[segment_start_idx:]
        if len(last_segment) >= min_len:
            cleaned_segments.append(last_segment)

    # Drop helper columns — they are not part of the output contract and
    # keeping them in cleaned_segments leaks transient velocity state.
    _temp_cols = ["FrameDiff", "DistDiff", "Velocity"]
    cleaned_segments = [
        seg.drop(columns=[c for c in _temp_cols if c in seg.columns])
        for seg in cleaned_segments
    ]

    stats["final_count"] = len(cleaned_segments)

    # Convert dataframe segments back to list of tuples format
    final_trajectories = [
        [tuple(row) for row in seg_df[["X", "Y", "Theta", "FrameID"]].to_numpy()]
        for seg_df in cleaned_segments
    ]

    logger.info(f"Post-processing stats: {stats}")
    return final_trajectories, stats


def _drop_source_column(traj_list):
    for traj in traj_list:
        if "_source" in traj.columns:
            traj.drop(columns=["_source"], inplace=True)


def _find_merge_candidates(forward_dfs, backward_dfs, agreement_distance, min_overlap):
    if NUMBA_AVAILABLE and len(forward_dfs) > 1 and len(backward_dfs) > 1:
        logger.debug("Using Numba-accelerated merge candidate search")
        fwd_x, fwd_y, fwd_frames, fwd_starts, fwd_ends = _prepare_trajectory_arrays(
            forward_dfs
        )
        bwd_x, bwd_y, bwd_frames, bwd_starts, bwd_ends = _prepare_trajectory_arrays(
            backward_dfs
        )
        try:
            fi_arr, bi_arr, agreeing_arr, common_arr = (
                _compute_all_merge_candidates_numba(
                    fwd_x,
                    fwd_y,
                    fwd_frames,
                    fwd_starts,
                    fwd_ends,
                    bwd_x,
                    bwd_y,
                    bwd_frames,
                    bwd_starts,
                    bwd_ends,
                    agreement_distance,
                    min_overlap,
                )
            )
            candidates = list(zip(fi_arr, bi_arr, agreeing_arr, common_arr))
            return _enrich_candidates_with_identity(
                candidates, forward_dfs, backward_dfs, agreement_distance
            )
        except Exception as e:
            logger.warning(f"Numba acceleration failed, falling back to Python: {e}")
    return _find_merge_candidates_python(
        forward_dfs, backward_dfs, agreement_distance, min_overlap
    )


def _apply_merge_candidates(
    merge_candidates,
    forward_dfs,
    backward_dfs,
    agreement_distance,
    min_length,
    identity_disagree_min_run=5,
    identity_drives_splits: bool = True,
):
    used_forward = set()
    used_backward = set()
    result_trajectories = []
    merge_candidates.sort(key=lambda x: (-x[2], -x[4]))
    for fi, bi, agreeing, total_common, identity_agreeing in merge_candidates:
        if fi in used_forward or bi in used_backward:
            continue
        used_forward.add(fi)
        used_backward.add(bi)
        logger.debug(
            f"Merging forward_{fi} with backward_{bi}: "
            f"{agreeing}/{total_common} agreeing frames "
            f"({identity_agreeing} identity-confirmed)"
        )
        merged_segments = _conservative_merge(
            forward_dfs[fi],
            backward_dfs[bi],
            agreement_distance,
            min_length,
            identity_disagree_min_run=identity_disagree_min_run,
            identity_drives_splits=identity_drives_splits,
        )
        result_trajectories.extend(merged_segments)
    for fi, fwd in enumerate(forward_dfs):
        if fi not in used_forward:
            result_trajectories.append(fwd.copy())
    for bi, bwd in enumerate(backward_dfs):
        if bi not in used_backward:
            result_trajectories.append(bwd.copy())
    return result_trajectories


def _reassign_trajectory_ids(result_trajectories):
    for new_id, traj in enumerate(result_trajectories):
        traj["TrajectoryID"] = new_id
        if "_source" in traj.columns:
            traj.drop(columns=["_source"], inplace=True)


def resolve_trajectories(
    forward_trajs: object, backward_trajs: object, params: object = None
) -> object:
    """
    Merges forward and backward trajectories using conservative consensus-based merging.

    This function prioritizes identity confidence over trajectory completeness:
    1. Only considers trajectory pairs as merge candidates if they have sufficient
       overlapping frames where positions agree (within AGREEMENT_DISTANCE)
    2. Merges only the agreeing segments - disagreeing frames cause trajectory splits
    3. Results in more trajectory fragments but higher confidence in identity

    Algorithm:
    - For each forward/backward pair, count frames where both have valid positions
      within AGREEMENT_DISTANCE of each other
    - If count >= MIN_OVERLAP_FRAMES, they are merge candidates
    - During merge: agreeing frames are averaged, disagreeing frames cause splits
      into separate trajectory segments

    Args:
        forward_trajs (list): List of forward trajectory DataFrames or lists of tuples
        backward_trajs (list): List of backward trajectory DataFrames or lists of tuples
        params (dict, optional): Parameters for merging thresholds

    Returns:
        list: Final merged trajectories as list of DataFrames
    """

    if not forward_trajs and not backward_trajs:
        return []

    # Default parameters
    if params is None:
        params = {}

    # Get parameters - AGREEMENT_DISTANCE should be ~0.5 * body_size (in scaled pixels)
    AGREEMENT_DISTANCE = params.get("AGREEMENT_DISTANCE", 15.0)
    MIN_OVERLAP_FRAMES = params.get("MIN_OVERLAP_FRAMES", 5)
    MIN_LENGTH = params.get("MIN_TRAJECTORY_LENGTH", 5)
    IDENTITY_DISAGREE_MIN_RUN = int(params.get("IDENTITY_DISAGREE_MIN_RUN", 5))
    # Whether identity labels are allowed to influence trajectory structure
    # (splits, stitch vetos).  When False (typically because the user set the
    # identity weight to 0), post-processing makes structural decisions on
    # geometry alone — labels still flow through the output, they just no
    # longer cause splits or block stitches.
    IDENTITY_GATES_TRAJECTORY_STRUCTURE = bool(
        params.get("IDENTITY_GATES_TRAJECTORY_STRUCTURE", True)
    )

    logger.info(
        f"Starting conservative trajectory resolution with {len(forward_trajs)} forward "
        f"and {len(backward_trajs)} backward trajectories"
    )
    logger.info(
        f"Parameters: AGREEMENT_DISTANCE={AGREEMENT_DISTANCE:.2f}px, "
        f"MIN_OVERLAP_FRAMES={MIN_OVERLAP_FRAMES}, MIN_LENGTH={MIN_LENGTH}"
    )

    # Convert and prepare forward trajectories
    forward_dfs = []
    for i, traj in enumerate(forward_trajs):
        df = _convert_trajectory_to_dataframe(traj, f"forward_{i}")
        if len(df) >= MIN_LENGTH:
            df["Theta"] = df["Theta"] % (2 * np.pi)
            df["_source"] = "forward"
            forward_dfs.append(df)

    # Convert and prepare backward trajectories
    backward_dfs = []
    for i, traj in enumerate(backward_trajs):
        df = _convert_trajectory_to_dataframe(traj, f"backward_{i}")
        if len(df) >= MIN_LENGTH:
            # No frame adjustment or extra theta flipping needed.
            # Backward fallback correction is now applied at tracking write-time.
            df["_source"] = "backward"
            backward_dfs.append(df)

    # Clean trajectories
    forward_dfs = _clean_trajectories(forward_dfs, MIN_LENGTH)
    backward_dfs = _clean_trajectories(backward_dfs, MIN_LENGTH)

    logger.info(
        f"After cleaning: {len(forward_dfs)} forward, {len(backward_dfs)} backward"
    )

    if not forward_dfs and not backward_dfs:
        logger.warning("No valid trajectories found for merging")
        return []

    # If only one direction has trajectories, return those
    if not forward_dfs:
        _drop_source_column(backward_dfs)
        return backward_dfs
    if not backward_dfs:
        _drop_source_column(forward_dfs)
        return forward_dfs

    # Find merge candidates based on overlap counting
    merge_candidates = _find_merge_candidates(
        forward_dfs, backward_dfs, AGREEMENT_DISTANCE, MIN_OVERLAP_FRAMES
    )

    logger.info(f"Found {len(merge_candidates)} merge candidates")

    # Now merge candidates using conservative strategy
    result_trajectories = _apply_merge_candidates(
        merge_candidates,
        forward_dfs,
        backward_dfs,
        AGREEMENT_DISTANCE,
        MIN_LENGTH,
        identity_disagree_min_run=IDENTITY_DISAGREE_MIN_RUN,
        identity_drives_splits=IDENTITY_GATES_TRAJECTORY_STRUCTURE,
    )

    # Note: Filtering by MIN_LENGTH is deferred until after stitching
    # to allow small fragments to be reconnected.

    # CRITICAL: Remove duplicate trajectories that are spatially contained within others
    # This can happen when a forward trajectory matches multiple backward trajectories,
    # and the "unused" ones cover the same physical location
    result_trajectories = _remove_spatially_redundant_trajectories(
        result_trajectories, AGREEMENT_DISTANCE, MIN_OVERLAP_FRAMES
    )

    # CRITICAL: Merge overlapping trajectories that agree spatially
    # This handles fragments that overlap in the middle (neither contains the other)
    result_trajectories = _merge_overlapping_agreeing_trajectories(
        result_trajectories,
        AGREEMENT_DISTANCE,
        MIN_OVERLAP_FRAMES,
        MIN_LENGTH,
        identity_disagree_min_run=IDENTITY_DISAGREE_MIN_RUN,
        identity_drives_splits=IDENTITY_GATES_TRAJECTORY_STRUCTURE,
    )

    # NEW: Stitch consecutive fragments that are spatially close
    # This fixes tracking breaks during turns or fast movements.
    # Use 2× agreement distance as the base spatial gate; the motion-aware
    # scorer in `_stitch_broken_trajectory_fragments` then tightens via:
    #   • motion-predicted residual (uses end_velocity × Δframes),
    #   • heading mismatch veto when both endpoints are moving,
    #   • density tightening when ≥2 other fragments are near the gap midpoint,
    #   • margin test (reject when runner-up is comparable),
    #   • symmetric NN test (A→B accepted only if B also prefers A).
    _stitch_gap = int(params.get("STITCH_MAX_GAP_FRAMES", 3))
    _max_vel_break = float(params.get("MAX_VELOCITY_BREAK", 100.0))
    _single_option_margin = float(params.get("STITCH_SINGLE_OPTION_MARGIN", 0.5))
    _heading_gate_deg = float(params.get("STITCH_HEADING_GATE_DEG", 60.0))
    _heading_gate_rad = float(np.deg2rad(_heading_gate_deg))
    _density_tighten_factor = float(params.get("STITCH_DENSITY_TIGHTEN_FACTOR", 0.5))
    result_trajectories = _stitch_broken_trajectory_fragments(
        result_trajectories,
        AGREEMENT_DISTANCE * 2,
        max_gap=_stitch_gap,
        max_vel_break=_max_vel_break,
        heading_gate_rad=_heading_gate_rad,
        single_option_margin=_single_option_margin,
        density_tighten_factor=_density_tighten_factor,
        identity_gates_stitching=IDENTITY_GATES_TRAJECTORY_STRUCTURE,
    )

    # FINAL DEDUPLICATION: Run a second redundancy pass after all merging and stitching.
    # _merge_overlapping_agreeing_trajectories can produce new disagree-source fragments
    # that partially overlap with the main merged trajectories.  Stitching can also
    # lengthen trajectories, making some surviving fragments newly redundant (>70%).
    # This pass catches any duplicates that slipped through the first pass.
    result_trajectories = _remove_spatially_redundant_trajectories(
        result_trajectories, AGREEMENT_DISTANCE, MIN_OVERLAP_FRAMES
    )

    # FINAL CLEANING: Now that stitching is done, remove trajectories that are still too short
    result_trajectories = [t for t in result_trajectories if len(t) >= MIN_LENGTH]
    result_trajectories = _clean_trajectories(result_trajectories, MIN_LENGTH)

    # Reassign trajectory IDs and remove internal columns
    # Enforce: only one trajectory may claim a given identity at any frame.
    result_trajectories = resolve_simultaneous_identity_conflicts(
        result_trajectories, params
    )

    _reassign_trajectory_ids(result_trajectories)

    logger.info(f"Final result: {len(result_trajectories)} trajectories")

    return result_trajectories


# ---------------------------------------------------------------------------
# Identity conflict arbitration
# ---------------------------------------------------------------------------

_IDENTITY_LABEL_COL = "IdentityAssignedLabel"
_IDENTITY_ID_COL = "IdentityAssignedID"
_IDENTITY_CONF_COL = "IdentityAssignedConfidence"
_IDENTITY_SLOT_COL = "IdentitySlotLockLabel"
_IDENTITY_CONFLICT_COL = "IdentityConflictResolved"


_CLAIM_TAG_WEIGHT = 1.5


def _claim_features(df: pd.DataFrame, modal_label: str) -> tuple:
    """Return the per-trajectory features feeding ``_claim_score``.

    Components (in declaration order):
    - tag_votes: sum of AprilTag agreements
    - agreement: fraction of labelled rows whose label matches ``modal_label``
    - mean_conf: mean of ``IdentityAssignedConfidence`` over labelled rows
    - frame_count: number of rows in the trajectory
    - is_forward: 1 when any row was contributed by the forward pass
    """
    tag_votes = float(df["TagVotes"].sum()) if "TagVotes" in df.columns else 0.0

    label_series = (
        df[_IDENTITY_LABEL_COL]
        if _IDENTITY_LABEL_COL in df.columns
        else pd.Series(dtype=object)
    )
    valid_label_mask = label_series.notna() & (
        label_series.astype(str).str.strip() != ""
    )
    valid_count = int(valid_label_mask.sum())
    if valid_count > 0:
        match = (label_series.astype(str) == modal_label) & valid_label_mask
        agreement = float(match.sum()) / float(valid_count)
    else:
        agreement = 0.0

    mean_conf = 0.0
    if _IDENTITY_CONF_COL in df.columns:
        conf_series = df.loc[valid_label_mask, _IDENTITY_CONF_COL].dropna()
        if not conf_series.empty:
            candidate = float(conf_series.mean())
            if np.isfinite(candidate):
                mean_conf = max(0.0, min(1.0, candidate))

    frame_count = int(len(df))
    is_forward = int(
        "_source" in df.columns and bool((df["_source"] == "forward").any())
    )
    return (tag_votes, agreement, mean_conf, frame_count, is_forward)


def _claim_score(
    features: tuple,
    max_frame_count: int,
    max_tag_votes: float,
    tag_weight: float = _CLAIM_TAG_WEIGHT,
) -> float:
    """Combine claim features into a scalar score (higher = stronger claim).

    Mirrors the iterative fragment-solver objective:
    ``agreement × mean_conf × length_factor`` is the unary trajectory-quality
    term; the tag-vote sum is a separate additive bonus so a heavily
    tag-confirmed claim dominates over noisy long ones. Use
    ``(_claim_score, is_forward)`` as the lexicographic ordering key — the
    forward-pass flag breaks exact ties.
    """
    tag_votes, agreement, mean_conf, frame_count, _ = features
    if max_frame_count <= 0:
        length_factor = 1.0
    else:
        log_max = math.log1p(max_frame_count)
        length_factor = math.log1p(frame_count) / log_max if log_max > 0 else 1.0
    if max_tag_votes <= 0:
        tag_norm = 0.0
    else:
        tag_norm = tag_votes / max_tag_votes
    return float(agreement * mean_conf * length_factor + tag_weight * tag_norm)


def _strip_identity_columns(df: pd.DataFrame) -> pd.DataFrame:
    if _IDENTITY_LABEL_COL in df.columns:
        df[_IDENTITY_LABEL_COL] = np.nan
    if _IDENTITY_ID_COL in df.columns:
        df[_IDENTITY_ID_COL] = np.nan
    if _IDENTITY_CONF_COL in df.columns:
        df[_IDENTITY_CONF_COL] = 0.0
    if _IDENTITY_SLOT_COL in df.columns:
        df[_IDENTITY_SLOT_COL] = np.nan
    df[_IDENTITY_CONFLICT_COL] = True
    return df


def resolve_simultaneous_identity_conflicts(
    result_dfs: list,
    params: dict | None = None,
) -> list:
    """Demote the weaker of two tracks that simultaneously claim the same identity.

    Enforces the physical constraint that a given identity can belong to at most
    one trajectory at any point in time.  For each pair of trajectories with the
    same majority ``IdentityAssignedLabel`` and at least one shared frame, the
    lower-scoring one has its identity columns cleared and
    ``IdentityConflictResolved`` set to ``True``.

    Scoring follows the same shape as the iterative fragment solver: a unary
    quality term ``agreement × mean_conf × length_factor`` plus an additive
    AprilTag bonus, with the forward-pass flag as the lex tiebreaker. A long
    track with consistent labels and a clear margin therefore wins over a
    short, jittery, or low-confidence one — the loser is the one that gets
    cleared to Unknown.
    """
    if not result_dfs:
        return result_dfs

    labeled: list[tuple] = []
    for idx, df in enumerate(result_dfs):
        if _IDENTITY_LABEL_COL not in df.columns or "FrameID" not in df.columns:
            continue
        valid = df[_IDENTITY_LABEL_COL].dropna()
        valid = valid[valid.astype(str).str.strip() != ""]
        if valid.empty:
            continue
        label = str(valid.mode().iloc[0])
        frames = frozenset(df["FrameID"].dropna().astype(int).tolist())
        features = _claim_features(df, label)
        labeled.append((idx, label, frames, features))

    if not labeled:
        return result_dfs

    max_frame_count = max(features[3] for _, _, _, features in labeled)
    max_tag_votes = max(features[0] for _, _, _, features in labeled)

    scored: list[tuple] = []
    for idx, label, frames, features in labeled:
        score = _claim_score(features, max_frame_count, max_tag_votes)
        is_forward = features[4]
        scored.append((idx, label, frames, (score, is_forward)))

    by_label: dict[str, list] = {}
    for item in scored:
        by_label.setdefault(item[1], []).append(item)

    loser_indices: set[int] = set()
    for candidates in by_label.values():
        if len(candidates) < 2:
            continue
        for i in range(len(candidates)):
            for j in range(i + 1, len(candidates)):
                idx_a, _, frames_a, score_a = candidates[i]
                idx_b, _, frames_b, score_b = candidates[j]
                if idx_a in loser_indices or idx_b in loser_indices:
                    continue
                if frames_a.isdisjoint(frames_b):
                    continue
                loser = idx_b if score_a >= score_b else idx_a
                loser_indices.add(loser)
                logger.debug(
                    "Identity conflict on label '%s': trajectory index %d wins over %d",
                    candidates[i][1],
                    idx_a if score_a >= score_b else idx_b,
                    loser,
                )

    for idx in loser_indices:
        result_dfs[idx] = _strip_identity_columns(result_dfs[idx].copy())

    return result_dfs


def _committed_identity_disagrees(r1: dict, r2: dict) -> bool:
    """True when both rows carry committed, non-empty identity labels that differ."""
    if r1.get("IdentityCommitted") != 1 or r2.get("IdentityCommitted") != 1:
        return False
    l1 = r1.get("IdentityAssignedLabel") or ""
    l2 = r2.get("IdentityAssignedLabel") or ""
    return bool(l1) and bool(l2) and l1 != l2


def _committed_identity_agrees(r1: dict, r2: dict) -> bool:
    """True when both rows carry committed, non-empty identity labels that are equal."""
    if r1.get("IdentityCommitted") != 1 or r2.get("IdentityCommitted") != 1:
        return False
    l1 = r1.get("IdentityAssignedLabel") or ""
    l2 = r2.get("IdentityAssignedLabel") or ""
    return bool(l1) and bool(l2) and l1 == l2


def _compute_identity_disagree_frames(
    t1_by_frame: dict,
    t2_by_frame: dict,
    agreement_distance: float,
    min_run: int,
    identity_drives_splits: bool = True,
) -> frozenset:
    """Return the set of frames that belong to a sustained identity-disagree run.

    A frame qualifies when forward and backward positions agree spatially AND both
    carry committed but conflicting identity labels.  Isolated disagreements shorter
    than ``min_run`` consecutive frames are discarded to suppress single-frame blips.
    Gap tolerance between consecutive candidate frames is 2 frames.

    When ``identity_drives_splits`` is False, identity is treated as advisory only
    and the function short-circuits to an empty set so post-processing structure
    is decided on geometry alone.
    """
    if not identity_drives_splits:
        return frozenset()
    common = sorted(set(t1_by_frame.keys()).intersection(t2_by_frame.keys()))
    candidates: list[int] = []
    for frame in common:
        r1, r2 = t1_by_frame[frame], t2_by_frame[frame]
        x1, y1 = r1.get("X"), r1.get("Y")
        x2, y2 = r2.get("X"), r2.get("Y")
        if pd.isna(x1) or pd.isna(x2):
            continue
        dist = np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
        if dist <= agreement_distance and _committed_identity_disagrees(r1, r2):
            candidates.append(frame)

    if not candidates:
        return frozenset()
    if min_run <= 1:
        return frozenset(candidates)

    result: set[int] = set()
    run_start = 0
    for i in range(1, len(candidates) + 1):
        if i == len(candidates) or candidates[i] - candidates[i - 1] > 2:
            if i - run_start >= min_run:
                result.update(candidates[run_start:i])
            run_start = i
    return frozenset(result)


def _classify_frame_pair(
    t1_by_frame,
    t2_by_frame,
    frame,
    agreement_distance,
    identity_disagree_frames=frozenset(),
):
    in_t1 = frame in t1_by_frame
    in_t2 = frame in t2_by_frame
    if in_t1 and in_t2:
        r1 = t1_by_frame[frame]
        r2 = t2_by_frame[frame]
        x1_valid = not pd.isna(r1.get("X"))
        x2_valid = not pd.isna(r2.get("X"))
        if not x1_valid and not x2_valid:
            return "agree", _average_trajectory_rows(r1, r2)
        if not x1_valid:
            return "t2_only", r2
        if not x2_valid:
            return "t1_only", r1
        dx = r1["X"] - r2["X"]
        dy = r1["Y"] - r2["Y"]
        dist = np.sqrt(dx * dx + dy * dy)
        if dist <= agreement_distance:
            if frame in identity_disagree_frames:
                return "disagree", (r1, r2)
            return "agree", _average_trajectory_rows(r1, r2)
        return "disagree", (r1, r2)
    if in_t1:
        return "t1_only", t1_by_frame[frame]
    return "t2_only", t2_by_frame[frame]


def _advance_merged_state(
    classification,
    data,
    current_segment,
    split_t1_segment,
    split_t2_segment,
    result_segments_rows,
    short_segments_rows,
    min_length,
):
    def _append_segment(seg):
        if len(seg) >= min_length:
            result_segments_rows.append(seg)
        elif seg:
            short_segments_rows.append(seg)

    if classification in ("agree", "t1_only", "t2_only"):
        current_segment.append(data)
        return "merged", current_segment, split_t1_segment, split_t2_segment
    # disagree: end merged segment, start split
    _append_segment(current_segment)
    r1, r2 = data
    return "split", [], [r1.copy()], [r2.copy()]


def _advance_split_state(
    classification,
    data,
    current_segment,
    split_t1_segment,
    split_t2_segment,
    result_segments_rows,
    short_segments_rows,
    min_length,
):
    def _append_segment(seg):
        if len(seg) >= min_length:
            result_segments_rows.append(seg)
        elif seg:
            short_segments_rows.append(seg)

    if classification == "agree":
        _append_segment(split_t1_segment)
        _append_segment(split_t2_segment)
        return "merged", [data], [], []
    if classification == "disagree":
        r1, r2 = data
        split_t1_segment.append(r1.copy())
        split_t2_segment.append(r2.copy())
    elif classification == "t1_only":
        split_t1_segment.append(data.copy())
    elif classification == "t2_only":
        split_t2_segment.append(data.copy())
    return "split", current_segment, split_t1_segment, split_t2_segment


def _finalize_state_machine_segments(
    state,
    current_segment,
    split_t1_segment,
    split_t2_segment,
    result_segments_rows,
    short_segments_rows,
    min_length,
):
    def _append_segment(seg):
        if len(seg) >= min_length:
            result_segments_rows.append(seg)
        elif seg:
            short_segments_rows.append(seg)

    if state == "merged":
        _append_segment(current_segment)
    elif state == "split":
        _append_segment(split_t1_segment)
        _append_segment(split_t2_segment)


def _build_final_segments_from_rows(
    result_segments_rows, short_segments_rows, agreement_distance
):
    max_spatial_jump = agreement_distance * 5
    final_segments = []
    for seg_rows in result_segments_rows:
        if not seg_rows:
            continue
        sub_segments_rows = _split_rows_into_segments(
            seg_rows, max_gap=5, max_spatial_jump=max_spatial_jump
        )
        final_segments.extend(pd.DataFrame(seg) for seg in sub_segments_rows if seg)
    if short_segments_rows:
        n_short = sum(len(s) for s in short_segments_rows)
        logger.debug(
            f"Conservative merge: preserving {len(short_segments_rows)} short "
            f"segment(s) ({n_short} total frames) for downstream stitching"
        )
        for seg_rows in short_segments_rows:
            if seg_rows:
                final_segments.append(pd.DataFrame(seg_rows))
    return final_segments


def _conservative_merge(
    traj1,
    traj2,
    agreement_distance,
    min_length,
    identity_disagree_min_run=5,
    identity_drives_splits: bool = True,
):
    """
    Conservatively merge two trajectories.

    - Agreeing frames (both exist, distance <= threshold): Average positions
    - Disagreeing frames (both exist, distance > threshold): Split into separate segments
    - Agreeing frames where both have committed but conflicting identity: also split
    - Unique frames: Keep from whichever trajectory has them

    Returns list of trajectory DataFrames (may be more than input if splits occur).
    """
    # Build frame dictionaries (faster than per-row iloc/to_dict loops)
    t1_by_frame = _build_frame_lookup(traj1, require_valid_x=False)
    t2_by_frame = _build_frame_lookup(traj2, require_valid_x=False)

    frames1 = set(t1_by_frame.keys())
    frames2 = set(t2_by_frame.keys())
    all_frames = sorted(frames1.union(frames2))

    identity_disagree_frames = _compute_identity_disagree_frames(
        t1_by_frame,
        t2_by_frame,
        agreement_distance,
        identity_disagree_min_run,
        identity_drives_splits=identity_drives_splits,
    )

    # Build trajectory segments using state machine
    # State: "merged" = building single merged segment
    #        "split" = building two parallel segments (after disagreement)
    result_segments_rows = []
    # Short segments (<min_length) are kept separately so they can be stitched
    # later rather than being silently discarded.  Without this, detections
    # that fall into brief agree/split transitions vanish from the output.
    short_segments_rows = []
    state = "merged"
    current_segment = []
    split_t1_segment = []
    split_t2_segment = []

    for frame in all_frames:
        classification, data = _classify_frame_pair(
            t1_by_frame,
            t2_by_frame,
            frame,
            agreement_distance,
            identity_disagree_frames,
        )
        if state == "merged":
            state, current_segment, split_t1_segment, split_t2_segment = (
                _advance_merged_state(
                    classification,
                    data,
                    current_segment,
                    split_t1_segment,
                    split_t2_segment,
                    result_segments_rows,
                    short_segments_rows,
                    min_length,
                )
            )
        else:
            state, current_segment, split_t1_segment, split_t2_segment = (
                _advance_split_state(
                    classification,
                    data,
                    current_segment,
                    split_t1_segment,
                    split_t2_segment,
                    result_segments_rows,
                    short_segments_rows,
                    min_length,
                )
            )

    # Finalize remaining segments
    _finalize_state_machine_segments(
        state,
        current_segment,
        split_t1_segment,
        split_t2_segment,
        result_segments_rows,
        short_segments_rows,
        min_length,
    )

    return _build_final_segments_from_rows(
        result_segments_rows, short_segments_rows, agreement_distance
    )


def _merge_col(r1, r2, col, merge_fn):
    """Merge a column from two row dicts using merge_fn, handling NaN."""
    if col not in r1 or col not in r2:
        return None
    v1, v2 = r1.get(col), r2.get(col)
    if pd.isna(v1):
        return v2
    if pd.isna(v2):
        return v1
    return merge_fn(v1, v2)


def _average_trajectory_rows(r1, r2):
    """
    Average two trajectory row dictionaries.
    For positions: average X, Y. For theta: circular mean.
    For confidence: take max. For uncertainty: take min.
    For state: prefer 'active'.
    """
    result = r1.copy()

    # Average positions
    if not pd.isna(r1.get("X")) and not pd.isna(r2.get("X")):
        result["X"] = (r1["X"] + r2["X"]) / 2
        result["Y"] = (r1["Y"] + r2["Y"]) / 2
    elif pd.isna(r1.get("X")):
        result["X"] = r2.get("X")
        result["Y"] = r2.get("Y")

    # Average theta (circular)
    merged_theta = _merge_col(
        r1, r2, "Theta", lambda t1, t2: _merge_angle_mean(float(t1), float(t2))
    )
    if merged_theta is not None:
        result["Theta"] = merged_theta

    # For confidence metrics, take the max (more informative)
    for col in ["DetectionConfidence", "AssignmentConfidence"]:
        merged = _merge_col(r1, r2, col, max)
        if merged is not None:
            result[col] = merged

    # For uncertainty, take the min (lower = more confident)
    merged_unc = _merge_col(r1, r2, "PositionUncertainty", min)
    if merged_unc is not None:
        result["PositionUncertainty"] = merged_unc

    # State: prefer "active" over "occluded"/"lost"
    if "State" in r1 and "State" in r2:
        s1, s2 = r1.get("State", "active"), r2.get("State", "active")
        result["State"] = "active" if (s1 == "active" or s2 == "active") else s1

    return result


def _find_agreeing_frames(a_by_frame, b_by_frame, agreement_distance):
    """Return set of frames where B's position agrees with A within distance."""
    agreeing = set()
    for frame, (bx, by) in b_by_frame.items():
        if frame in a_by_frame:
            ax, ay = a_by_frame[frame]
            dist = np.sqrt((bx - ax) ** 2 + (by - ay) ** 2)
            if dist <= agreement_distance:
                agreeing.add(frame)
    return agreeing


def _trim_or_remove_trajectory(
    idx_b,
    trajectories,
    agreeing_frame_set,
    agreeing_frames,
    total_b_frames,
    agreement_ratio,
    trimmed_replacements,
    redundant_indices,
):
    """Trim a partially redundant trajectory to unique frames, or remove it."""
    traj_b = trajectories[idx_b]
    unique_mask = ~traj_b["FrameID"].isin(agreeing_frame_set)
    trimmed = traj_b.loc[unique_mask].copy()
    if not trimmed.empty:
        trimmed_replacements[idx_b] = trimmed
        logger.debug(
            f"Trimming partially redundant trajectory: "
            f"{agreeing_frames}/{total_b_frames} "
            f"({agreement_ratio:.1%}) redundant, "
            f"preserving {len(trimmed)} unique frames"
        )
    else:
        redundant_indices.add(idx_b)


def _remove_spatially_redundant_trajectories(
    trajectories, agreement_distance, min_overlap
):
    """
    Remove trajectories that are spatially redundant (covered by another trajectory).

    A trajectory B is considered redundant relative to trajectory A if:
    - B has significant frame overlap with A
    - At those overlapping frames, B's positions are within agreement_distance of A's

    When redundancy is detected, we keep the LONGER trajectory and remove the shorter one.
    This handles cases where an "unused" trajectory is actually a subset of a merged one.

    OPTIMIZED: Uses NumPy arrays instead of iterrows() for speed.
    """
    if not trajectories:
        return trajectories

    # Sort by length (longest first) - longer trajectories are preferred
    sorted_trajs = sorted(enumerate(trajectories), key=lambda x: -len(x[1]))

    # Track which trajectories to keep
    redundant_indices = set()

    # Pre-extract arrays for all trajectories
    traj_arrays = []
    for idx, traj in sorted_trajs:
        frames = traj["FrameID"].values
        x = traj["X"].values
        y = traj["Y"].values
        # Create frame -> position lookup using dict comprehension on arrays
        valid_mask = ~np.isnan(x)
        frame_to_pos = {
            frames[i]: (x[i], y[i]) for i in range(len(frames)) if valid_mask[i]
        }
        traj_arrays.append((idx, frame_to_pos, np.sum(valid_mask)))

    trimmed_replacements = {}  # idx -> trimmed DataFrame (or None to remove)

    for i, (idx_a, a_by_frame, _) in enumerate(traj_arrays):
        if idx_a in redundant_indices:
            continue

        for idx_b, b_by_frame, total_b_frames in traj_arrays[i + 1 :]:
            if idx_b in redundant_indices or total_b_frames == 0:
                continue

            agreeing_frame_set = _find_agreeing_frames(
                a_by_frame, b_by_frame, agreement_distance
            )
            agreeing_frames = len(agreeing_frame_set)

            if agreeing_frames < min(min_overlap, total_b_frames):
                continue

            agreement_ratio = agreeing_frames / total_b_frames
            if agreement_ratio < 0.7:
                continue

            if agreement_ratio >= 0.95:
                redundant_indices.add(idx_b)
                logger.debug(
                    f"Removing fully redundant trajectory: "
                    f"{agreeing_frames}/{total_b_frames} "
                    f"({agreement_ratio:.1%}) frames agree"
                )
            else:
                _trim_or_remove_trajectory(
                    idx_b,
                    trajectories,
                    agreeing_frame_set,
                    agreeing_frames,
                    total_b_frames,
                    agreement_ratio,
                    trimmed_replacements,
                    redundant_indices,
                )

    # Build result: replace trimmed, skip removed
    result = []
    for i, t in enumerate(trajectories):
        if i in redundant_indices:
            continue
        if i in trimmed_replacements:
            result.append(trimmed_replacements[i])
        else:
            result.append(t)

    n_removed = len(redundant_indices)
    n_trimmed = len(trimmed_replacements)
    if n_removed or n_trimmed:
        logger.info(
            f"Redundancy pass: removed {n_removed}, trimmed {n_trimmed} trajectories"
        )

    return result


def _get_last_position(segment):
    """Get the (X, Y) of the last frame in a segment (list of row dicts)."""
    if not segment:
        return None, None
    last = segment[-1]
    return last.get("X"), last.get("Y")


def _is_spatially_continuous(
    segment, new_row, threshold, has_detection_id, check_detection_id=True
):
    """
    Check if new_row is spatially close to the end of segment.

    If both have valid DetectionID and they match consecutive detections,
    spatial continuity is enforced more strictly. If DetectionID suggests
    a jump in detection sequence, we're more lenient with spatial distance.
    """
    if not segment:
        return True
    last_x, last_y = _get_last_position(segment)
    if last_x is None or pd.isna(last_x):
        return True
    new_x, new_y = new_row.get("X"), new_row.get("Y")
    if new_x is None or pd.isna(new_x):
        return True
    dist = np.sqrt((new_x - last_x) ** 2 + (new_y - last_y) ** 2)

    # If DetectionID is available and both rows have it, use it to inform continuity
    if has_detection_id and check_detection_id:
        last_det_id = segment[-1].get("DetectionID")
        new_det_id = new_row.get("DetectionID")

        # If both have valid DetectionIDs (not NaN)
        if (
            last_det_id is not None
            and not pd.isna(last_det_id)
            and new_det_id is not None
            and not pd.isna(new_det_id)
        ):
            # If DetectionIDs differ, it's a different detection - be lenient with spatial distance
            # This handles cases where tracking switched between detections
            if last_det_id != new_det_id:
                # Allow larger spatial jumps when DetectionID changes
                return dist <= threshold * 2

    return dist <= threshold


def _check_detection_id_match(ra, rb, has_detection_id):
    """Check whether two row dicts share a valid DetectionID."""
    if not has_detection_id:
        return False
    det_a = ra.get("DetectionID")
    det_b = rb.get("DetectionID")
    return (
        det_a is not None
        and not pd.isna(det_a)
        and det_b is not None
        and not pd.isna(det_b)
        and det_a == det_b
    )


def _count_overlap_agreement(
    common_frames, lookup_a, lookup_b, has_detection_id, agreement_distance
):
    """Count agreeing frames and DetectionID matches in the overlap between two trajectories."""
    agreeing = 0
    detection_id_matches = 0
    for frame in common_frames:
        ra, rb = lookup_a[frame], lookup_b[frame]

        # Check DetectionID match first (strongest evidence)
        if _check_detection_id_match(ra, rb, has_detection_id):
            detection_id_matches += 1
            agreeing += 1
            continue

        # Fall back to spatial agreement
        dist = np.sqrt((ra["X"] - rb["X"]) ** 2 + (ra["Y"] - rb["Y"]) ** 2)
        if dist <= agreement_distance:
            agreeing += 1

    return agreeing, detection_id_matches


def _classify_overlap_frames(
    all_frames,
    lookup_a,
    lookup_b,
    has_detection_id,
    agreement_distance,
    identity_disagree_frames=frozenset(),
):
    """Classify each frame of the union of two trajectories as agree/disagree/single-source."""
    frame_classifications = {}
    for frame in all_frames:
        in_a = frame in lookup_a
        in_b = frame in lookup_b

        if in_a and in_b:
            ra, rb = lookup_a[frame], lookup_b[frame]

            if _check_detection_id_match(ra, rb, has_detection_id):
                merged = _average_trajectory_rows(ra, rb)
                frame_classifications[frame] = ("agree_detid", merged)
            else:
                dist = np.sqrt((ra["X"] - rb["X"]) ** 2 + (ra["Y"] - rb["Y"]) ** 2)
                if dist <= agreement_distance:
                    if frame in identity_disagree_frames:
                        frame_classifications[frame] = ("disagree", (ra, rb))
                    else:
                        merged = _average_trajectory_rows(ra, rb)
                        frame_classifications[frame] = ("agree", merged)
                else:
                    frame_classifications[frame] = ("disagree", (ra, rb))
        elif in_a:
            frame_classifications[frame] = ("a_only", lookup_a[frame])
        else:
            frame_classifications[frame] = ("b_only", lookup_b[frame])

    return frame_classifications


def _handle_disagree_frame(
    ra,
    rb,
    current_segment,
    max_spatial_jump,
    has_detection_id,
    result_segments,
    disagree_a_observations,
    disagree_b_observations,
    min_length,
):
    """Handle a frame where both trajectories exist but disagree spatially.

    Mutates current_segment, result_segments, disagree_a_observations,
    disagree_b_observations in place. Returns the (possibly replaced) current_segment.
    """
    state_a = ra.get("State", "active")
    state_b = rb.get("State", "active")
    lenient_threshold = max_spatial_jump * 2

    # If one is active and the other is occluded/lost, prefer the active one
    if state_a == "active" and state_b in ("occluded", "lost"):
        current_segment = _append_or_split_segment(
            current_segment,
            ra.copy(),
            lenient_threshold,
            has_detection_id,
            result_segments,
            min_length,
        )
    elif state_b == "active" and state_a in ("occluded", "lost"):
        current_segment = _append_or_split_segment(
            current_segment,
            rb.copy(),
            lenient_threshold,
            has_detection_id,
            result_segments,
            min_length,
        )
    else:
        # Both active or both occluded - genuine disagreement
        current_segment = _handle_genuine_disagree(
            ra,
            rb,
            current_segment,
            lenient_threshold,
            has_detection_id,
            result_segments,
            disagree_a_observations,
            disagree_b_observations,
            min_length,
        )

    return current_segment


def _append_or_split_segment(
    current_segment, row, threshold, has_detection_id, result_segments, min_length
):
    """Append row to current_segment if spatially continuous, else split.

    Returns the (possibly replaced) current_segment.
    """
    if _is_spatially_continuous(
        current_segment,
        row,
        threshold,
        has_detection_id,
        check_detection_id=False,
    ):
        current_segment.append(row)
    else:
        if len(current_segment) >= min_length:
            result_segments.append(pd.DataFrame(current_segment))
        current_segment = [row]
    return current_segment


def _handle_genuine_disagree(
    ra,
    rb,
    current_segment,
    lenient_threshold,
    has_detection_id,
    result_segments,
    disagree_a_observations,
    disagree_b_observations,
    min_length,
):
    """Handle genuine disagreement where both observations are active (or both occluded).

    Returns the (possibly replaced) current_segment.
    """
    a_continuous = _is_spatially_continuous(
        current_segment,
        ra,
        lenient_threshold,
        has_detection_id,
        check_detection_id=False,
    )
    b_continuous = _is_spatially_continuous(
        current_segment,
        rb,
        lenient_threshold,
        has_detection_id,
        check_detection_id=False,
    )

    if a_continuous and not b_continuous:
        current_segment.append(ra.copy())
        disagree_b_observations.append(rb.copy())
    elif b_continuous and not a_continuous:
        current_segment.append(rb.copy())
        disagree_a_observations.append(ra.copy())
    elif a_continuous and b_continuous:
        # Both could continue - prefer the one minimizing spatial jump
        current_segment = _pick_closer_observation(
            current_segment, ra, rb, disagree_a_observations, disagree_b_observations
        )
    else:
        # Neither continuous - save current and preserve both separately
        if len(current_segment) >= min_length:
            result_segments.append(pd.DataFrame(current_segment))
        current_segment = []
        disagree_a_observations.append(ra.copy())
        disagree_b_observations.append(rb.copy())

    return current_segment


def _pick_closer_observation(
    current_segment, ra, rb, disagree_a_observations, disagree_b_observations
):
    """When both observations are spatially continuous, pick the one closer to the segment end.

    Returns the (unchanged) current_segment after appending the chosen observation.
    """
    last_x, last_y = _get_last_position(current_segment)

    if last_x is None or pd.isna(last_x):
        current_segment.append(ra.copy())
        disagree_b_observations.append(rb.copy())
        return current_segment

    xa, ya = ra.get("X"), ra.get("Y")
    xb, yb = rb.get("X"), rb.get("Y")

    dist_a = float("inf")
    dist_b = float("inf")

    if not pd.isna(xa):
        dist_a = np.sqrt((xa - last_x) ** 2 + (ya - last_y) ** 2)
    if not pd.isna(xb):
        dist_b = np.sqrt((xb - last_x) ** 2 + (yb - last_y) ** 2)

    if dist_a <= dist_b:
        current_segment.append(ra.copy())
        disagree_b_observations.append(rb.copy())
    else:
        current_segment.append(rb.copy())
        disagree_a_observations.append(ra.copy())

    return current_segment


def _build_segments_from_classifications(
    all_frames, frame_classifications, max_spatial_jump, has_detection_id, min_length
):
    """Build merged trajectory segments from classified frames using a state machine.

    Returns (result_segments, disagree_a_observations, disagree_b_observations).
    """
    result_segments = []
    current_segment = []
    disagree_a_observations = []
    disagree_b_observations = []

    for frame in all_frames:
        classification, data = frame_classifications[frame]

        if classification == "agree_detid":
            # DetectionID match - ALWAYS continuous (same physical detection)
            current_segment.append(data)

        elif classification == "agree":
            if _is_spatially_continuous(
                current_segment, data, max_spatial_jump, has_detection_id
            ):
                current_segment.append(data)
            else:
                if len(current_segment) >= min_length:
                    result_segments.append(pd.DataFrame(current_segment))
                current_segment = [data]

        elif classification in ("a_only", "b_only"):
            # Single-source frame: use the same max_spatial_jump threshold
            if _is_spatially_continuous(
                current_segment,
                data,
                max_spatial_jump,
                has_detection_id,
                check_detection_id=False,
            ):
                current_segment.append(data)
            else:
                if len(current_segment) >= min_length:
                    result_segments.append(pd.DataFrame(current_segment))
                current_segment = [data]

        elif classification == "disagree":
            ra, rb = data
            current_segment = _handle_disagree_frame(
                ra,
                rb,
                current_segment,
                max_spatial_jump,
                has_detection_id,
                result_segments,
                disagree_a_observations,
                disagree_b_observations,
                min_length,
            )

    # Finalize remaining segment
    if len(current_segment) >= min_length:
        result_segments.append(pd.DataFrame(current_segment))

    return result_segments, disagree_a_observations, disagree_b_observations


def _build_disagree_trajectory(
    obs_list, max_spatial_jump, has_detection_id, min_length
):
    """Build trajectory segments from a list of disagree observations.

    Groups temporally and spatially close observations into segments.
    """
    if not obs_list:
        return []

    segments = []
    obs_list.sort(key=lambda x: x.get("FrameID", 0))

    current_seg = []
    for obs in obs_list:
        if not current_seg:
            current_seg = [obs]
            continue

        last_frame = current_seg[-1].get("FrameID", 0)
        curr_frame = obs.get("FrameID", 0)

        # Skip if same frame (shouldn't happen but safety check)
        if curr_frame == last_frame:
            continue

        # If frames are close in time and space, continue segment
        if curr_frame - last_frame <= 5:
            if _is_spatially_continuous(
                current_seg,
                obs,
                max_spatial_jump * 3,
                has_detection_id,
                check_detection_id=False,
            ):
                current_seg.append(obs)
            else:
                if len(current_seg) >= min_length:
                    segments.append(pd.DataFrame(current_seg))
                current_seg = [obs]
        else:
            if len(current_seg) >= min_length:
                segments.append(pd.DataFrame(current_seg))
            current_seg = [obs]

    if len(current_seg) >= min_length:
        segments.append(pd.DataFrame(current_seg))

    return segments


def _try_merge_trajectory_pair(
    i,
    j,
    trajectories,
    traj_lookups,
    traj_frame_sets,
    traj_bounds,
    has_detection_id,
    agreement_distance,
    min_overlap,
    min_length,
    max_spatial_jump,
    identity_disagree_min_run=5,
    identity_drives_splits: bool = True,
):
    """Try to merge trajectory pair (i, j). Returns merged segments list, or None if not mergeable."""
    lookup_a = traj_lookups[i]
    frames_a = traj_frame_sets[i]
    start_a, end_a = traj_bounds[i]

    lookup_b = traj_lookups[j]
    frames_b = traj_frame_sets[j]
    start_b, end_b = traj_bounds[j]

    # Fast temporal pruning
    if end_a < start_b or end_b < start_a:
        return None

    common_frames = frames_a.intersection(frames_b)
    if len(common_frames) < min_overlap:
        return None

    agreeing, detection_id_matches = _count_overlap_agreement(
        common_frames, lookup_a, lookup_b, has_detection_id, agreement_distance
    )

    required_matches = (
        min_overlap if detection_id_matches < 2 else max(2, min_overlap // 2)
    )
    if agreeing < required_matches:
        return None

    # Require majority agreement (waived for strong DetectionID evidence)
    if detection_id_matches < 2:
        agreement_ratio = agreeing / len(common_frames)
        if agreement_ratio < 0.5:
            return None

    if detection_id_matches > 0:
        logger.debug(
            f"Found {detection_id_matches} DetectionID matches out of "
            f"{len(common_frames)} overlapping frames"
        )

    all_frames = sorted(frames_a.union(frames_b))
    identity_disagree_frames = _compute_identity_disagree_frames(
        lookup_a,
        lookup_b,
        agreement_distance,
        identity_disagree_min_run,
        identity_drives_splits=identity_drives_splits,
    )
    frame_classifications = _classify_overlap_frames(
        all_frames,
        lookup_a,
        lookup_b,
        has_detection_id,
        agreement_distance,
        identity_disagree_frames,
    )

    result_segments, disagree_a_obs, disagree_b_obs = (
        _build_segments_from_classifications(
            all_frames,
            frame_classifications,
            max_spatial_jump,
            has_detection_id,
            min_length,
        )
    )

    disagree_segs_a = _build_disagree_trajectory(
        disagree_a_obs, max_spatial_jump, has_detection_id, min_length
    )
    disagree_segs_b = _build_disagree_trajectory(
        disagree_b_obs, max_spatial_jump, has_detection_id, min_length
    )
    result_segments.extend(disagree_segs_a)
    result_segments.extend(disagree_segs_b)

    if not result_segments:
        return None

    total_merged_frames = sum(len(seg) for seg in result_segments)
    disagree_count = len(disagree_a_obs) + len(disagree_b_obs)
    logger.debug(
        f"Merged trajectories {i} ({len(lookup_a)} frames) + {j} ({len(lookup_b)} frames) "
        f"→ {len(result_segments)} segments with {total_merged_frames} total frames "
        f"(DetectionID matches: {detection_id_matches}, disagree preserved: {disagree_count})"
    )

    return result_segments


def _merge_overlapping_agreeing_trajectories(
    trajectories,
    agreement_distance,
    min_overlap,
    min_length,
    identity_disagree_min_run=5,
    identity_drives_splits: bool = True,
):
    """
    Merge trajectories that overlap in time and agree spatially or share DetectionIDs.

    ENHANCED with DetectionID support for deterministic merging:
    - Trajectories sharing DetectionIDs are guaranteed to be the same animal
    - DetectionID matches override spatial discontinuity checks
    - Falls back to spatial agreement for frames without DetectionID

    CONSERVATIVE approach: Only merge the frames that actually agree.
    Non-overlapping parts that are spatially distant become separate trajectories.

    Uses iterative processing until no more merges are possible.
    """
    if not trajectories:
        return trajectories

    has_detection_id = (
        "DetectionID" in trajectories[0].columns if trajectories else False
    )
    max_spatial_jump = agreement_distance * 5

    max_iterations = 50
    iteration = 0

    while iteration < max_iterations:
        iteration += 1
        merged_any = False
        used = set()
        new_trajectories = []

        # Build lookups and frame bounds once per iteration for faster pair pruning.
        traj_lookups = []
        traj_frame_sets = []
        traj_bounds = []
        for traj in trajectories:
            lookup = _build_frame_lookup(traj, require_valid_x=True)
            frame_set = set(lookup.keys())
            traj_lookups.append(lookup)
            traj_frame_sets.append(frame_set)
            if frame_set:
                traj_bounds.append((min(frame_set), max(frame_set)))
            else:
                traj_bounds.append((np.inf, -np.inf))

        for i in range(len(trajectories)):
            if i in used:
                continue

            traj_a = trajectories[i]

            for j in range(i + 1, len(trajectories)):
                if j in used:
                    continue

                result_segments = _try_merge_trajectory_pair(
                    i,
                    j,
                    trajectories,
                    traj_lookups,
                    traj_frame_sets,
                    traj_bounds,
                    has_detection_id,
                    agreement_distance,
                    min_overlap,
                    min_length,
                    max_spatial_jump,
                    identity_disagree_min_run=identity_disagree_min_run,
                    identity_drives_splits=identity_drives_splits,
                )

                if result_segments is not None:
                    used.add(i)
                    used.add(j)
                    new_trajectories.extend(result_segments)
                    merged_any = True
                    break

            # If trajectory i wasn't merged with anything, keep it.
            # Do NOT filter by min_length here — short fragments must survive
            # so that downstream stitching can reconnect them.  The final
            # min_length filter in resolve_trajectories handles cleanup.
            if i not in used:
                new_trajectories.append(traj_a)
                used.add(i)

        trajectories = new_trajectories

        if not merged_any:
            break

    if iteration > 1:
        logger.info(f"Processed overlapping trajectories in {iteration} iterations")

    return trajectories


def _last_committed_label(df: pd.DataFrame) -> str:
    """Return the most recent committed identity label in df, or empty string."""
    if (
        "IdentityCommitted" not in df.columns
        or "IdentityAssignedLabel" not in df.columns
    ):
        return ""
    committed = df[df["IdentityCommitted"] == 1]
    if committed.empty:
        return ""
    lbl = committed["IdentityAssignedLabel"].iloc[-1]
    return str(lbl) if lbl and not pd.isna(lbl) else ""


def _first_committed_label(df: pd.DataFrame) -> str:
    """Return the earliest committed identity label in df, or empty string."""
    if (
        "IdentityCommitted" not in df.columns
        or "IdentityAssignedLabel" not in df.columns
    ):
        return ""
    committed = df[df["IdentityCommitted"] == 1]
    if committed.empty:
        return ""
    lbl = committed["IdentityAssignedLabel"].iloc[0]
    return str(lbl) if lbl and not pd.isna(lbl) else ""


def _endpoint_window_velocity(df: pd.DataFrame, at_start: bool, k: int = 3):
    """Estimate end-of-fragment velocity (px/frame) from the last (or first) k valid rows.

    Uses only rows with finite X/Y so NaN gaps inside the fragment do not corrupt the
    estimate. Returns (vx, vy, theta) where theta is the row's reported heading at the
    selected endpoint (used for heading-consistency gating).
    """
    if df is None or df.empty:
        return 0.0, 0.0, 0.0
    valid = df[df["X"].notna() & df["Y"].notna()]
    if valid.empty:
        return 0.0, 0.0, 0.0
    if at_start:
        window = valid.head(k)
        anchor = window.iloc[0]
    else:
        window = valid.tail(k)
        anchor = window.iloc[-1]
    if len(window) < 2:
        theta = (
            float(anchor["Theta"]) if np.isfinite(anchor.get("Theta", np.nan)) else 0.0
        )
        return 0.0, 0.0, theta
    r0 = window.iloc[0]
    r1 = window.iloc[-1]
    dt = max(1.0, float(r1["FrameID"]) - float(r0["FrameID"]))
    vx = (float(r1["X"]) - float(r0["X"])) / dt
    vy = (float(r1["Y"]) - float(r0["Y"])) / dt
    theta = float(anchor["Theta"]) if np.isfinite(anchor.get("Theta", np.nan)) else 0.0
    return vx, vy, theta


def _build_stitch_info(trajectories):
    """Build start/end frame, position, motion predictors, and identity for each fragment.

    Motion predictors (``end_velocity``, ``start_velocity`` and corresponding speeds/headings)
    are estimated from a short window of valid rows so the basic stitcher can do
    motion-aware gating without pose features. Identity labels are cached for the
    identity-conflict gate.
    """
    traj_info = {}
    for idx, df in enumerate(trajectories):
        if df.empty:
            continue
        sx = df["X"].iat[0]
        sy = df["Y"].iat[0]
        ex = df["X"].iat[-1]
        ey = df["Y"].iat[-1]
        s_vx, s_vy, s_theta = _endpoint_window_velocity(df, at_start=True)
        e_vx, e_vy, e_theta = _endpoint_window_velocity(df, at_start=False)
        traj_info[idx] = {
            "idx": idx,
            "start_frame": int(df["FrameID"].iat[0]),
            "end_frame": int(df["FrameID"].iat[-1]),
            "start_pos": (sx, sy),
            "end_pos": (ex, ey),
            "start_velocity": (s_vx, s_vy),
            "end_velocity": (e_vx, e_vy),
            "start_speed": float(np.hypot(s_vx, s_vy)),
            "end_speed": float(np.hypot(e_vx, e_vy)),
            "start_theta": s_theta,
            "end_theta": e_theta,
            "end_identity": _last_committed_label(df),
            "start_identity": _first_committed_label(df),
            "df": df,
        }
    sorted_indices = sorted(traj_info.keys(), key=lambda i: traj_info[i]["start_frame"])
    return traj_info, sorted_indices


def _build_stitch_density_index(traj_info):
    """Pre-index per-frame fragment positions for the density neighbour count.

    Returns a dict {frame: list of (idx, x, y)} covering the union of frames present
    in any fragment. Built once per stitch iteration so the inner loop stays cheap.
    """
    index: dict[int, list[tuple[int, float, float]]] = {}
    for idx, info in traj_info.items():
        df = info["df"]
        for frame_v, x_v, y_v in zip(
            df["FrameID"].to_numpy(), df["X"].to_numpy(), df["Y"].to_numpy()
        ):
            if pd.isna(x_v) or pd.isna(y_v):
                continue
            f = int(frame_v)
            index.setdefault(f, []).append((idx, float(x_v), float(y_v)))
    return index


def _count_stitch_neighbors(
    density_index,
    midpoint,
    frame_lo,
    frame_hi,
    radius,
    exclude_a,
    exclude_b,
):
    """Count distinct fragments other than ``exclude_a``/``exclude_b`` within ``radius``
    of ``midpoint`` over the inclusive frame range [frame_lo, frame_hi]."""
    if radius <= 0.0 or not density_index:
        return 0
    mx, my = midpoint
    seen = set()
    r2 = float(radius) ** 2
    for f in range(int(frame_lo), int(frame_hi) + 1):
        rows = density_index.get(f)
        if not rows:
            continue
        for idx, x_v, y_v in rows:
            if idx in (exclude_a, exclude_b) or idx in seen:
                continue
            dx = x_v - mx
            dy = y_v - my
            if dx * dx + dy * dy <= r2:
                seen.add(idx)
    return len(seen)


def _score_stitch_pair(
    info_a,
    info_b,
    agreement_distance,
    max_vel_break,
    max_gap,
    heading_gate,
    min_motion_speed,
    identity_gates_stitching,
    density_neighbors,
    density_tighten_factor,
):
    """Conservative motion-aware score for a candidate stitch ``info_a → info_b``.

    Returns ``(score, identity_match)`` if the pair passes every gate, else ``None``.
    Lower scores are better. The score is dimensionless: residual relative to a
    body-length, plus small heading and gap-length penalties so ties favour
    short, kinematically-consistent stitches.

    Gates applied (any failure → reject):
      • frame-gap ∈ [1, max_gap]
      • raw end→start displacement ≤ max_vel_break × min(Δframes, 4)  (absolute envelope)
      • |predicted_pos − start_pos| ≤ effective_agreement_distance
        where effective = agreement × (density_tighten_factor if ≥2 neighbours else 1)
      • heading mismatch ≤ heading_gate, only when both endpoints exceed min_motion_speed
      • identity-conflict gate (when both endpoints carry committed labels)
    """
    ax, ay = info_a["end_pos"]
    bx, by = info_b["start_pos"]
    if pd.isna(ax) or pd.isna(ay) or pd.isna(bx) or pd.isna(by):
        return None

    gap = int(info_b["start_frame"] - info_a["end_frame"])
    if gap < 1 or gap > max_gap:
        return None

    end_id = info_a.get("end_identity", "")
    start_id = info_b.get("start_identity", "")
    if identity_gates_stitching and end_id and start_id and end_id != start_id:
        return None

    delta_frames = float(gap)
    raw_jump = float(np.hypot(bx - ax, by - ay))
    raw_jump_limit = max(
        agreement_distance * 2.0,
        max_vel_break * float(min(delta_frames, 4.0)),
    )
    if raw_jump > raw_jump_limit:
        return None

    e_vx, e_vy = info_a["end_velocity"]
    predicted_x = float(ax) + e_vx * delta_frames
    predicted_y = float(ay) + e_vy * delta_frames
    residual = float(np.hypot(predicted_x - bx, predicted_y - by))

    effective_agreement = agreement_distance
    if density_neighbors >= 2:
        effective_agreement = agreement_distance * float(density_tighten_factor)

    if residual > effective_agreement:
        return None

    heading_norm = 0.0
    if (
        info_a["end_speed"] > min_motion_speed
        and info_b["start_speed"] > min_motion_speed
    ):
        heading_diff = _link_orientation_diff(
            info_a["end_theta"], info_b["start_theta"]
        )
        if heading_diff > heading_gate:
            return None
        heading_norm = float(heading_diff / np.pi)

    score = (
        residual / max(effective_agreement, 1e-6)
        + 0.25 * (delta_frames / max(float(max_gap), 1.0))
        + 0.35 * heading_norm
    )
    identity_match = bool(end_id and start_id and end_id == start_id)
    return float(score), identity_match


def _find_best_stitch_candidate(
    info_a,
    sorted_indices,
    idx_ptr,
    traj_info,
    used,
    agreement_distance,
    max_gap,
    *,
    max_vel_break,
    heading_gate,
    min_motion_speed,
    single_option_margin,
    density_index,
    density_radius,
    density_tighten_factor,
    best_src_for_dst,
    identity_gates_stitching: bool = True,
):
    """Pick a stitch partner for ``info_a``, applying conservative gates.

    Conservatism layers (in addition to every gate enforced by
    ``_score_stitch_pair``):
      • **Margin test** — best.score must be ≤ single_option_margin × second_best.score,
        otherwise the stitch is too ambiguous and we leave the fragments split.
      • **Symmetric NN** — accepted only if ``info_a`` is also the best predecessor for
        the chosen ``info_b`` under the same scorer (precomputed in
        ``best_src_for_dst``).

    Identity-tied scores nudge the best toward identity-matching candidates as a
    cheap tie-breaker but never override the gates above.
    """
    candidates = []
    for idx_next in range(idx_ptr + 1, len(sorted_indices)):
        idx_b = sorted_indices[idx_next]
        if idx_b in used:
            continue
        info_b = traj_info[idx_b]
        gap = info_b["start_frame"] - info_a["end_frame"]
        if gap > max_gap:
            break

        midpoint = (
            0.5 * (float(info_a["end_pos"][0]) + float(info_b["start_pos"][0])),
            0.5 * (float(info_a["end_pos"][1]) + float(info_b["start_pos"][1])),
        )
        neighbors = _count_stitch_neighbors(
            density_index,
            midpoint,
            info_a["end_frame"],
            info_b["start_frame"],
            density_radius,
            info_a["idx"],
            idx_b,
        )

        result = _score_stitch_pair(
            info_a,
            info_b,
            agreement_distance,
            max_vel_break,
            max_gap,
            heading_gate,
            min_motion_speed,
            identity_gates_stitching,
            neighbors,
            density_tighten_factor,
        )
        if result is None:
            continue
        score, identity_match = result
        # Identity-tied ties: nudge identity-matching candidates slightly ahead.
        adj_score = score - (1e-3 if identity_match else 0.0)
        candidates.append((adj_score, idx_b))

    if not candidates:
        return -1, float("inf")

    candidates.sort(key=lambda item: item[0])
    best_score, best_idx_b = candidates[0]

    # Margin test: only stitch if best is clearly better than runner-up.
    if len(candidates) >= 2:
        second_score = candidates[1][0]
        # second_score is also dimensionless — guard against zero so the inequality
        # behaves sanely when both scores are tiny.
        denom = max(second_score, 1e-6)
        if (best_score / denom) > float(single_option_margin):
            return -1, float("inf")

    # Symmetric NN test: ``info_a`` must also be the best predecessor of ``best_idx_b``.
    if best_src_for_dst.get(best_idx_b, -1) != info_a["idx"]:
        return -1, float("inf")

    return best_idx_b, best_score


def _build_best_src_for_dst(
    sorted_indices,
    traj_info,
    used,
    agreement_distance,
    max_gap,
    max_vel_break,
    heading_gate,
    min_motion_speed,
    density_index,
    density_radius,
    density_tighten_factor,
    identity_gates_stitching,
):
    """For each ``dst`` fragment, return the ``src`` whose stitch score is lowest.

    Used by the symmetric-NN check in ``_find_best_stitch_candidate`` so that a
    candidate ``A → B`` is only accepted when ``A`` is also the best predecessor
    for ``B`` under the same conservative scorer.
    """
    best_src: dict[int, tuple[float, int]] = {}
    n = len(sorted_indices)
    for i in range(n):
        idx_a = sorted_indices[i]
        if idx_a in used:
            continue
        info_a = traj_info[idx_a]
        for j in range(i + 1, n):
            idx_b = sorted_indices[j]
            if idx_b in used:
                continue
            info_b = traj_info[idx_b]
            gap = info_b["start_frame"] - info_a["end_frame"]
            if gap > max_gap:
                break
            midpoint = (
                0.5 * (float(info_a["end_pos"][0]) + float(info_b["start_pos"][0])),
                0.5 * (float(info_a["end_pos"][1]) + float(info_b["start_pos"][1])),
            )
            neighbors = _count_stitch_neighbors(
                density_index,
                midpoint,
                info_a["end_frame"],
                info_b["start_frame"],
                density_radius,
                info_a["idx"],
                idx_b,
            )
            result = _score_stitch_pair(
                info_a,
                info_b,
                agreement_distance,
                max_vel_break,
                max_gap,
                heading_gate,
                min_motion_speed,
                identity_gates_stitching,
                neighbors,
                density_tighten_factor,
            )
            if result is None:
                continue
            score, identity_match = result
            adj = score - (1e-3 if identity_match else 0.0)
            existing = best_src.get(idx_b)
            if existing is None or adj < existing[0]:
                best_src[idx_b] = (adj, idx_a)
    return {dst: src for dst, (_, src) in best_src.items()}


def _stitch_broken_trajectory_fragments(
    trajectories,
    agreement_distance,
    max_gap=2,
    *,
    max_vel_break: float = 100.0,
    heading_gate_rad: float = float(np.pi / 3.0),
    single_option_margin: float = 0.5,
    density_radius_multiplier: float = 5.0,
    density_tighten_factor: float = 0.5,
    min_motion_speed: float = 1e-3,
    identity_gates_stitching: bool = True,
):
    """Stitch consecutive fragments that are likely the same track.

    The matcher applies, in order: a kinematic gate (motion-predicted residual
    plus an absolute jump cap), a heading-consistency gate, an identity-conflict
    gate, a density-aware tightening of the spatial gate, a margin test against
    the runner-up, and a symmetric nearest-neighbour check. All gates can only
    *reject* a stitch — they never relax the existing radius — so accepting the
    output is at least as conservative as the legacy behaviour and reconnects
    additional fragments where motion prediction lands on the right successor.
    """
    if not trajectories:
        return trajectories

    max_iterations = 20
    current_trajectories = trajectories
    iteration = 0
    density_radius = float(agreement_distance) * float(density_radius_multiplier)

    for iteration in range(1, max_iterations + 1):
        merged_any = False
        used: set[int] = set()
        new_trajectories: list[pd.DataFrame] = []
        traj_info, sorted_indices = _build_stitch_info(current_trajectories)
        density_index = _build_stitch_density_index(traj_info)
        best_src_for_dst = _build_best_src_for_dst(
            sorted_indices,
            traj_info,
            used,
            agreement_distance,
            max_gap,
            max_vel_break,
            heading_gate_rad,
            min_motion_speed,
            density_index,
            density_radius,
            density_tighten_factor,
            identity_gates_stitching,
        )

        idx_ptr = 0
        while idx_ptr < len(sorted_indices):
            idx_a = sorted_indices[idx_ptr]
            if idx_a in used:
                idx_ptr += 1
                continue

            info_a = traj_info[idx_a]
            best_idx_b, _best_score = _find_best_stitch_candidate(
                info_a,
                sorted_indices,
                idx_ptr,
                traj_info,
                used,
                agreement_distance,
                max_gap,
                max_vel_break=max_vel_break,
                heading_gate=heading_gate_rad,
                min_motion_speed=min_motion_speed,
                single_option_margin=single_option_margin,
                density_index=density_index,
                density_radius=density_radius,
                density_tighten_factor=density_tighten_factor,
                best_src_for_dst=best_src_for_dst,
                identity_gates_stitching=identity_gates_stitching,
            )

            if best_idx_b != -1:
                merged = pd.concat(
                    [info_a["df"], traj_info[best_idx_b]["df"]], ignore_index=True
                )
                new_trajectories.append(merged)
                used.add(idx_a)
                used.add(best_idx_b)
                merged_any = True
            else:
                new_trajectories.append(info_a["df"])
                used.add(idx_a)

            idx_ptr += 1

        current_trajectories = new_trajectories
        if not merged_any:
            break

    if iteration > 1:
        logger.info(f"Stitched broken fragments in {iteration} iterations")

    return current_trajectories


def _convert_trajectory_to_dataframe(traj, traj_id):
    """Convert trajectory from list of tuples to DataFrame format."""
    if isinstance(traj, pd.DataFrame):
        df = traj.copy()
        if "TrajectoryID" not in df.columns:
            df["TrajectoryID"] = traj_id
        # Ensure required columns exist
        if "State" not in df.columns:
            df["State"] = "active"
        return df
    else:
        # Convert from list of tuples format
        data = []
        for point in traj:
            if len(point) >= 4:
                x, y, theta, frame = point[:4]
                data.append(
                    {
                        "TrajectoryID": traj_id,
                        "X": int(x) if not np.isnan(x) else x,
                        "Y": int(y) if not np.isnan(y) else y,
                        "Theta": theta,
                        "FrameID": int(frame),
                        "State": "active",
                    }
                )
        return pd.DataFrame(data)


def _angle_diff_rad(theta_a: float, theta_b: float) -> float:
    """Shortest signed angular difference (theta_a - theta_b), in [-pi, pi]."""
    return float(np.arctan2(np.sin(theta_a - theta_b), np.cos(theta_a - theta_b)))


def _merge_angle_mean(theta_a: float, theta_b: float) -> float:
    """
    Average two angles for trajectory merge while handling 180-degree OBB ambiguity.

    During forward/backward merge, one branch can report theta and the other theta+pi
    for the same body axis. Plain circular mean of such pairs yields orthogonal
    artifacts (~+/-90 deg). We collapse this ambiguity before averaging.
    """
    two_pi = 2.0 * np.pi
    a = float(theta_a) % two_pi
    b = float(theta_b) % two_pi

    direct_delta = abs(_angle_diff_rad(b, a))
    flipped_b = (b + np.pi) % two_pi
    flipped_delta = abs(_angle_diff_rad(flipped_b, a))
    if flipped_delta + 1e-12 < direct_delta:
        b = flipped_b

    return (a + 0.5 * _angle_diff_rad(b, a)) % two_pi


def _clean_trajectory(traj_df):
    """
    Clean a trajectory DataFrame by:
    1. Removing leading 'lost' and 'occluded' states
    2. Removing trailing 'lost' and 'occluded' states
    3. Returning None if no valid detections remain

    Args:
        traj_df: DataFrame with trajectory data

    Returns:
        Cleaned DataFrame or None if trajectory has no valid detections
    """
    if traj_df.empty:
        return None

    # Check if State column exists
    if "State" not in traj_df.columns:
        return traj_df

    state_values = traj_df["State"].to_numpy(copy=False)
    invalid_mask = (state_values == "occluded") | (state_values == "lost")

    # Fast-path: no leading/trailing invalid states to trim.
    if not invalid_mask.any():
        return traj_df

    valid_positions = np.flatnonzero(~invalid_mask)
    if len(valid_positions) == 0:
        return None

    first_valid_pos = int(valid_positions[0])
    last_valid_pos = int(valid_positions[-1])

    if first_valid_pos == 0 and last_valid_pos == len(traj_df) - 1:
        return traj_df

    # Trim trajectory to valid detection range
    return traj_df.iloc[first_valid_pos : last_valid_pos + 1].copy()


def _clean_trajectories(traj_list, min_length=5):
    """
    Clean a list of trajectory DataFrames by removing useless ones.

    Args:
        traj_list: List of trajectory DataFrames
        min_length: Minimum length for valid trajectories

    Returns:
        List of cleaned trajectories
    """
    cleaned = []
    for traj in traj_list:
        cleaned_traj = _clean_trajectory(traj)
        if cleaned_traj is not None and len(cleaned_traj) >= min_length:
            cleaned.append(cleaned_traj)
    return cleaned


def _split_rows_into_segments(rows, max_gap=5, max_spatial_jump=50.0):
    """
    Split list-of-dict rows into continuous segments based on FrameID gaps/spatial jumps.
    Used to avoid DataFrame round-trips in tight loops.
    """
    if not rows:
        return []

    segments = []
    current_segment = []

    for row in rows:
        should_split = False
        if current_segment:
            prev_row = current_segment[-1]
            frame_gap = row["FrameID"] - prev_row["FrameID"]

            if frame_gap > max_gap:
                should_split = True

            if not should_split and frame_gap <= max_gap:
                prev_x, prev_y = prev_row.get("X"), prev_row.get("Y")
                curr_x, curr_y = row.get("X"), row.get("Y")

                if (
                    not pd.isna(prev_x)
                    and not pd.isna(curr_x)
                    and not pd.isna(prev_y)
                    and not pd.isna(curr_y)
                ):
                    dx = curr_x - prev_x
                    dy = curr_y - prev_y
                    dist = np.sqrt(dx * dx + dy * dy)
                    if dist > max_spatial_jump:
                        should_split = True

        if not current_segment or not should_split:
            current_segment.append(row)
        else:
            segments.append(current_segment)
            current_segment = [row]

    if current_segment:
        segments.append(current_segment)

    return segments


def interpolate_trajectories(
    trajectories_df: object,
    method: object = "linear",
    max_gap: object = 10,
    heading_flip_max_burst: int = 5,
    directed_heading_posthoc: bool = False,
) -> object:
    """
    Interpolate missing values in trajectories using various methods.

    Args:
        trajectories_df: DataFrame with trajectory data (must have X, Y, Theta, FrameID columns)
        method: Interpolation method - 'linear', 'cubic', 'spline', or 'none'
        max_gap: Maximum gap size to interpolate (frames). Larger gaps are left as NaN.
        heading_flip_max_burst: Maximum length of an isolated heading-flip burst to
            correct in post-processing.  Longer segments are assumed genuine.
            Ignored when *directed_heading_posthoc* is True.
        directed_heading_posthoc: When True (head-tail or pose model was used),
            apply global heading consistency via dynamic programming instead of
            the local burst-based flip correction.  This resolves the minimum
            number of per-frame flips needed to make the entire track
            consistently directed.

    Returns:
        DataFrame with interpolated values
    """
    if trajectories_df is None or trajectories_df.empty:
        return trajectories_df

    if method == "none" or method is None:
        return trajectories_df

    logger.info(f"Interpolating trajectories using {method} method (max_gap={max_gap})")

    # Pre-compute confidence columns once from the input DataFrame instead of
    # recomputing (and mutating) the growing result_df on every iteration.
    confidence_cols = [
        col
        for col in trajectories_df.columns
        if col.lower().endswith("confidence") or col == "PositionUncertainty"
    ]

    # Accumulate per-trajectory results in a list and concat once at the end.
    # The old pattern (filter + concat inside the loop) is O(T × n); this is O(n).
    interpolated_parts: list = []

    for traj_id, traj_data in trajectories_df.groupby("TrajectoryID", sort=False):
        traj_data = traj_data.sort_values("FrameID").reset_index(drop=True)

        # Remove duplicate FrameIDs (keep first occurrence)
        if traj_data["FrameID"].duplicated().any():
            n_duplicates = traj_data["FrameID"].duplicated().sum()
            logger.warning(
                f"Trajectory {traj_id} has {n_duplicates} duplicate FrameID(s). "
                f"Keeping first occurrence of each frame."
            )
            traj_data = traj_data.drop_duplicates(
                subset="FrameID", keep="first"
            ).reset_index(drop=True)

        min_frame = traj_data["FrameID"].min()
        max_frame = traj_data["FrameID"].max()
        has_missing_frames = len(traj_data) < max_frame - min_frame + 1
        has_value_gaps = any(
            col in traj_data.columns and traj_data[col].isna().any()
            for col in ["X", "Y", "Theta"]
        )

        # Fast-path: skip only when the trajectory is frame-complete and the
        # interpolated columns already have no NaN gaps. TrackerKit writes
        # contiguous occluded rows with NaN X/Y/Theta, and those still need to
        # go through interpolation.
        if not has_missing_frames and not has_value_gaps:
            # Still fix heading flips even without gaps
            if "Theta" in traj_data.columns:
                if directed_heading_posthoc:
                    traj_data["Theta"] = _fix_heading_globally(
                        traj_data["Theta"].values
                    )
                else:
                    traj_data["Theta"] = _fix_heading_flips(
                        traj_data["Theta"].values, max_burst=heading_flip_max_burst
                    )
            interpolated_parts.append(traj_data)
            continue

        if has_missing_frames:
            all_frames = np.arange(min_frame, max_frame + 1)
            traj_data = traj_data.set_index("FrameID").reindex(all_frames).reset_index()
            traj_data["TrajectoryID"] = traj_id

        if "State" in traj_data.columns:
            traj_data["State"] = traj_data["State"].fillna("occluded")
        else:
            traj_data["State"] = "occluded"

        for col in confidence_cols:
            if col not in traj_data.columns:
                traj_data[col] = np.nan

        for col in ["X", "Y"]:
            traj_data[col] = _interpolate_column(
                traj_data["FrameID"].values,
                traj_data[col].values,
                method=method,
                max_gap=max_gap,
            )

        # Fix heading ambiguities before interpolation so that sin/cos
        # interpolation doesn't smooth across an artificial discontinuity.
        if directed_heading_posthoc:
            traj_data["Theta"] = _fix_heading_globally(traj_data["Theta"].values)
        else:
            traj_data["Theta"] = _fix_heading_flips(
                traj_data["Theta"].values, max_burst=heading_flip_max_burst
            )

        traj_data["Theta"] = _interpolate_angle(
            traj_data["FrameID"].values,
            traj_data["Theta"].values,
            method=method,
            max_gap=max_gap,
        )

        interpolated_parts.append(traj_data)

    result_df = pd.concat(interpolated_parts, ignore_index=True)
    result_df = result_df.sort_values(["TrajectoryID", "FrameID"]).reset_index(
        drop=True
    )

    logger.info("Interpolation complete")
    return result_df


def _link_orientation_diff(theta_a, theta_b):
    diff = abs(float(theta_a) - float(theta_b))
    if diff > np.pi:
        diff = 2 * np.pi - diff
    return float(max(0.0, diff))


def _relink_pose_labels(df: pd.DataFrame) -> list[str]:
    labels = []
    for col in df.columns:
        match = _POSE_KPT_COL_RE.match(str(col))
        if match and match.group(2) == "X":
            labels.append(match.group(1))
    return labels


def _normalize_pose_keypoints_window(
    window_df: pd.DataFrame, pose_labels: list[str], min_valid_conf: float
):
    from hydra_suite.core.identity.pose.quality import (
        normalize_pose_keypoints_for_relink,
    )

    return normalize_pose_keypoints_for_relink(window_df, pose_labels, min_valid_conf)


def _pose_paired_distance_relink(det_pose, track_pose, min_shared: int = 3):
    if det_pose is None or track_pose is None:
        return None
    det_arr = np.asarray(det_pose, dtype=np.float32)
    track_arr = np.asarray(track_pose, dtype=np.float32)
    if det_arr.shape != track_arr.shape or det_arr.ndim != 2 or det_arr.shape[1] < 2:
        return None

    dists = []
    for kp_idx in range(len(det_arr)):
        det_valid = np.isfinite(det_arr[kp_idx, 0]) and np.isfinite(det_arr[kp_idx, 1])
        track_valid = np.isfinite(track_arr[kp_idx, 0]) and np.isfinite(
            track_arr[kp_idx, 1]
        )
        if not (det_valid and track_valid):
            continue
        dist = float(np.linalg.norm(det_arr[kp_idx, :2] - track_arr[kp_idx, :2]))
        if np.isfinite(dist):
            dists.append(dist)

    if len(dists) < min_shared:
        return None

    dists_arr = np.asarray(dists, dtype=np.float32)
    med = float(np.median(dists_arr))
    abs_dev = np.abs(dists_arr - med)
    mad = float(np.median(abs_dev))
    if mad > 1e-6:
        keep = abs_dev <= (2.5 * mad)
        filtered = dists_arr[keep]
        if len(filtered) >= min_shared:
            dists_arr = filtered
    if len(dists_arr) >= 5:
        cutoff = max(1, int(np.floor(len(dists_arr) * 0.2)))
        dists_arr = (
            np.sort(dists_arr)[:-cutoff] if cutoff < len(dists_arr) else dists_arr
        )
    return float(np.mean(dists_arr))


def _get_window_quality(window_df: pd.DataFrame, fallback_visibility: float) -> float:
    """Extract mean PoseQualityScore from a window df, falling back to visibility."""
    if (
        window_df is not None
        and not window_df.empty
        and "PoseQualityScore" in window_df.columns
    ):
        scores = window_df["PoseQualityScore"].dropna()
        if not scores.empty:
            return float(scores.mean())
    return float(fallback_visibility)


def _build_relink_fragment_summaries(
    trajectories_df: pd.DataFrame, params: dict
) -> tuple[list[dict], dict[int, pd.DataFrame]]:
    pose_labels = _relink_pose_labels(trajectories_df)
    min_pose_conf = float(params.get("POSE_MIN_KPT_CONF_VALID", 0.2))
    fragments = []
    fragment_frames = {}

    for traj_id in sorted(trajectories_df["TrajectoryID"].dropna().unique()):
        frag_df = (
            trajectories_df[trajectories_df["TrajectoryID"] == traj_id]
            .copy()
            .sort_values("FrameID", kind="stable")
            .reset_index(drop=True)
        )
        if frag_df.empty:
            continue
        valid_pos = frag_df["X"].notna() & frag_df["Y"].notna()
        valid_df = frag_df.loc[valid_pos].copy()
        if valid_df.empty:
            continue

        start_row = valid_df.iloc[0]
        end_row = valid_df.iloc[-1]
        start_frame = int(start_row["FrameID"])
        end_frame = int(end_row["FrameID"])

        start_window = valid_df.head(3)
        end_window = valid_df.tail(3)

        start_velocity = np.zeros(2, dtype=np.float32)
        end_velocity = np.zeros(2, dtype=np.float32)
        start_speed = 0.0
        end_speed = 0.0

        if len(start_window) >= 2:
            r0 = start_window.iloc[0]
            r1 = start_window.iloc[1]
            dt = max(1.0, float(r1["FrameID"]) - float(r0["FrameID"]))
            start_velocity = np.asarray(
                [
                    (float(r1["X"]) - float(r0["X"])) / dt,
                    (float(r1["Y"]) - float(r0["Y"])) / dt,
                ],
                dtype=np.float32,
            )
            start_speed = float(np.linalg.norm(start_velocity))

        if len(end_window) >= 2:
            r0 = end_window.iloc[-2]
            r1 = end_window.iloc[-1]
            dt = max(1.0, float(r1["FrameID"]) - float(r0["FrameID"]))
            end_velocity = np.asarray(
                [
                    (float(r1["X"]) - float(r0["X"])) / dt,
                    (float(r1["Y"]) - float(r0["Y"])) / dt,
                ],
                dtype=np.float32,
            )
            end_speed = float(np.linalg.norm(end_velocity))

        start_pose, start_vis = _normalize_pose_keypoints_window(
            start_window, pose_labels, min_pose_conf
        )
        end_pose, end_vis = _normalize_pose_keypoints_window(
            end_window, pose_labels, min_pose_conf
        )

        start_quality = _get_window_quality(start_window, start_vis)
        end_quality = _get_window_quality(end_window, end_vis)

        traj_id_int = int(traj_id)
        fragments.append(
            {
                "traj_id": traj_id_int,
                "start_frame": start_frame,
                "end_frame": end_frame,
                "identity_sources": _fragment_unique_identity_sources(frag_df),
                "start_pos": np.asarray(
                    [float(start_row["X"]), float(start_row["Y"])], dtype=np.float32
                ),
                "end_pos": np.asarray(
                    [float(end_row["X"]), float(end_row["Y"])], dtype=np.float32
                ),
                "start_theta": (
                    float(start_row["Theta"])
                    if np.isfinite(start_row["Theta"])
                    else 0.0
                ),
                "end_theta": (
                    float(end_row["Theta"]) if np.isfinite(end_row["Theta"]) else 0.0
                ),
                "start_velocity": start_velocity,
                "end_velocity": end_velocity,
                "start_speed": start_speed,
                "end_speed": end_speed,
                "start_pose": start_pose,
                "end_pose": end_pose,
                "start_pose_visibility": start_vis,
                "end_pose_visibility": end_vis,
                "start_quality": start_quality,
                "end_quality": end_quality,
            }
        )
        fragment_frames[traj_id_int] = frag_df

    return fragments, fragment_frames


def _parse_unique_identity_key(identity_key) -> dict[str, str]:
    token = str(identity_key).strip() if identity_key is not None else ""
    if not token or token.lower() == "nan":
        return {}
    parsed = {}
    for item in token.split("|"):
        if "=" not in item:
            continue
        source, value = item.split("=", 1)
        source = str(source).strip()
        value = str(value).strip()
        if source and value:
            parsed[source] = value
    return parsed


def _fragment_unique_identity_sources(frag_df: pd.DataFrame) -> dict[str, str]:
    if "UniqueIdentityKey" not in frag_df.columns:
        return {}
    keys = [
        str(value).strip()
        for value in frag_df["UniqueIdentityKey"].tolist()
        if str(value).strip() and str(value).strip().lower() != "nan"
    ]
    if not keys:
        return {}
    dominant_key = max(set(keys), key=keys.count)
    return _parse_unique_identity_key(dominant_key)


def _score_relink_candidate(
    src,
    dst,
    max_gap,
    max_vel_break,
    agreement_distance,
    pose_max_distance,
    min_pose_quality,
    heading_gate,
    min_motion_speed,
):
    """Score a single (src, dst) relink candidate. Returns (score, src_id, dst_id) or None if rejected."""
    src_identity = dict(src.get("identity_sources") or {})
    dst_identity = dict(dst.get("identity_sources") or {})
    if src_identity and dst_identity:
        from hydra_suite.core.post.identity_postprocess import identity_sources_conflict

        if identity_sources_conflict(src_identity, dst_identity):
            return None

    gap = int(dst["start_frame"] - src["end_frame"] - 1)
    if gap < 1 or gap > max_gap:
        return None

    delta_frames = gap + 1

    raw_jump = float(np.linalg.norm(dst["start_pos"] - src["end_pos"]))
    # Keep relinking conservative over long occlusion gaps: the destination must
    # remain within a small absolute motion envelope, not merely a large
    # time-scaled envelope.
    raw_jump_limit = max(
        agreement_distance * 2.0,
        max_vel_break * float(min(delta_frames, 4)),
    )
    if raw_jump > raw_jump_limit:
        return None

    predicted_pos = src["end_pos"] + src["end_velocity"] * float(delta_frames)
    distance = float(np.linalg.norm(predicted_pos - dst["start_pos"]))
    if distance > agreement_distance:
        return None

    heading_norm = 0.0
    if src["end_speed"] > min_motion_speed and dst["start_speed"] > min_motion_speed:
        heading_diff = _link_orientation_diff(src["end_theta"], dst["start_theta"])
        if heading_diff > heading_gate:
            return None
        heading_norm = float(heading_diff / np.pi)

    pose_norm = 0.0
    src_end_pose = (
        src["end_pose"] if src.get("end_quality", 1.0) >= min_pose_quality else None
    )
    dst_start_pose = (
        dst["start_pose"] if dst.get("start_quality", 1.0) >= min_pose_quality else None
    )
    if src_end_pose is not None and dst_start_pose is not None:
        pose_dist = _pose_paired_distance_relink(src_end_pose, dst_start_pose)
        if pose_dist is not None:
            if pose_dist > pose_max_distance:
                return None
            pose_norm = float(pose_dist / pose_max_distance)

    score = (
        float(distance / agreement_distance)
        + 0.25 * (float(gap) / float(max_gap if max_gap > 0 else 1))
        + 0.35 * heading_norm
        + 0.75 * pose_norm
    )
    return (score, int(src["traj_id"]), int(dst["traj_id"]))


def _greedy_match_candidates(candidates):
    """Greedily match relink candidates (one outgoing per src, one incoming per dst).

    Returns (outgoing, incoming) dicts mapping src_id -> dst_id and dst_id -> src_id.
    """
    candidates.sort(key=lambda item: (item[0], item[1], item[2]))
    outgoing = {}
    incoming = {}
    for _, src_id, dst_id in candidates:
        if src_id in outgoing or dst_id in incoming:
            continue
        outgoing[src_id] = dst_id
        incoming[dst_id] = src_id
    return outgoing, incoming


def _build_relink_chains(outgoing, incoming, fragment_frames):
    """Build ordered chains of fragment IDs from greedy matching results."""
    ordered_ids = sorted(fragment_frames)
    visited = set()
    chains = []
    for traj_id in ordered_ids:
        if traj_id in incoming or traj_id in visited:
            continue
        chain = [traj_id]
        visited.add(traj_id)
        while chain[-1] in outgoing:
            nxt = outgoing[chain[-1]]
            if nxt in visited:
                break
            chain.append(nxt)
            visited.add(nxt)
        chains.append(chain)
    for traj_id in ordered_ids:
        if traj_id not in visited:
            chains.append([traj_id])
            visited.add(traj_id)
    return chains


def _assemble_relinked_chains(chains, fragment_frames):
    """Concatenate fragment DataFrames along each chain and assign new trajectory IDs."""
    relinked_parts = []
    for new_id, chain in enumerate(chains):
        part = pd.concat(
            [fragment_frames[traj_id].copy() for traj_id in chain], ignore_index=True
        )
        part = part.sort_values("FrameID", kind="stable")
        if part["FrameID"].duplicated().any():
            logger.warning(
                "Relinking chain %s produced duplicate FrameID rows; keeping first occurrence.",
                chain,
            )
            part = part.drop_duplicates(subset="FrameID", keep="first")
        part = part.reset_index(drop=True)
        part["TrajectoryID"] = new_id
        relinked_parts.append(part)
    return relinked_parts


def relink_trajectories_with_pose(
    trajectories_df: pd.DataFrame, params: dict
) -> pd.DataFrame:
    """Greedily relink short trajectory fragments using motion and optional pose continuity."""
    if trajectories_df is None or trajectories_df.empty:
        return trajectories_df
    if (
        "TrajectoryID" not in trajectories_df.columns
        or "FrameID" not in trajectories_df.columns
    ):
        return trajectories_df
    if not bool(params.get("ENABLE_TRACKLET_RELINKING", True)):
        return trajectories_df.copy()

    fragments, fragment_frames = _build_relink_fragment_summaries(
        trajectories_df, params
    )
    if len(fragments) < 2:
        return trajectories_df.copy()

    max_gap = int(max(0, params.get("MAX_OCCLUSION_GAP", 30)))
    max_vel_break = float(max(1e-6, params.get("MAX_VELOCITY_BREAK", 100.0)))
    agreement_distance = float(max(1e-6, params.get("AGREEMENT_DISTANCE", 15.0)))
    pose_max_distance = float(max(1e-6, params.get("RELINK_POSE_MAX_DISTANCE", 0.45)))
    min_pose_quality = float(max(0.0, params.get("RELINK_MIN_POSE_QUALITY", 0.4)))
    heading_gate = float(np.pi / 3.0)
    min_motion_speed = 1e-3

    candidates = []
    for src in fragments:
        for dst in fragments:
            if src["traj_id"] == dst["traj_id"]:
                continue
            if dst["start_frame"] <= src["end_frame"]:
                continue

            result = _score_relink_candidate(
                src,
                dst,
                max_gap,
                max_vel_break,
                agreement_distance,
                pose_max_distance,
                min_pose_quality,
                heading_gate,
                min_motion_speed,
            )
            if result is not None:
                candidates.append(result)

    if not candidates:
        return trajectories_df.copy()

    outgoing, incoming = _greedy_match_candidates(candidates)

    if not outgoing:
        return trajectories_df.copy()

    chains = _build_relink_chains(outgoing, incoming, fragment_frames)
    relinked_parts = _assemble_relinked_chains(chains, fragment_frames)

    result_df = pd.concat(relinked_parts, ignore_index=True)
    result_df = result_df.sort_values(["TrajectoryID", "FrameID"], kind="stable")
    result_df = result_df.reset_index(drop=True)
    logger.info(
        "Tracklet relinking collapsed %d fragments into %d trajectories.",
        len(fragments),
        result_df["TrajectoryID"].nunique(),
    )
    return result_df


def _make_interp_func(method, valid_frames, valid_values, n_valid):
    """Create an interpolation function for the given method, or None."""

    def _linear():
        return interp1d(
            valid_frames,
            valid_values,
            kind="linear",
            bounds_error=False,
            fill_value=np.nan,
        )

    if method == "linear":
        return _linear()
    if method == "cubic":
        if n_valid < 4:
            return _linear()
        return CubicSpline(
            valid_frames, valid_values, bc_type="natural", extrapolate=False
        )
    if method == "spline":
        if n_valid < 4:
            return _linear()
        return UnivariateSpline(valid_frames, valid_values, s=None, k=3)
    return None


def _interpolate_column(frames, values, method="linear", max_gap=10):
    """Interpolate a single column with gap limit."""
    valid_mask = ~np.isnan(values)
    valid_indices = np.where(valid_mask)[0]

    if len(valid_indices) < 2:
        return values

    try:
        interp_func = _make_interp_func(
            method,
            frames[valid_indices],
            values[valid_indices],
            len(valid_indices),
        )
    except Exception as e:
        logger.warning(f"Interpolation failed: {e}, keeping original values")
        return values

    if interp_func is None:
        return values

    result = values.copy()

    # Build a single boolean mask for all within-limit gaps, then call
    # interp_func once (vectorised, much faster than per-gap calls).
    fill_mask = np.zeros(len(frames), dtype=bool)
    for i in range(len(valid_indices) - 1):
        gap_size = valid_indices[i + 1] - valid_indices[i] - 1
        if 0 < gap_size <= max_gap:
            fill_mask[valid_indices[i] + 1 : valid_indices[i + 1]] = True

    if fill_mask.any():
        try:
            result[fill_mask] = interp_func(frames[fill_mask])
        except Exception:
            pass

    return result


def _circ_diff(a, b):
    """Absolute circular difference in [0, pi]."""
    two_pi = 2.0 * np.pi
    d = abs(a - b) % two_pi
    return min(d, two_pi - d)


def _find_prev_valid(result, i):
    """Find the last non-NaN index before i, or -1."""
    prev_idx = i - 1
    while prev_idx >= 0 and np.isnan(result[prev_idx]):
        prev_idx -= 1
    return prev_idx


def _find_next_valid(result, start, n):
    """Find the first non-NaN index at or after start, or n."""
    idx = start
    while idx < n and np.isnan(result[idx]):
        idx += 1
    return idx


def _measure_flip_burst(result, start, n, prev_val, max_burst):
    """Find the end of a contiguous flipped burst starting at 'start'."""
    burst_end = start + 1
    while burst_end < n and burst_end - start < max_burst + 1:
        if np.isnan(result[burst_end]):
            burst_end += 1
            continue
        if _circ_diff(result[burst_end], prev_val) > np.pi / 2:
            burst_end += 1
        else:
            break
    return burst_end


def _fix_heading_flips(theta: np.ndarray, max_burst: int = 5) -> np.ndarray:
    """Detect and correct isolated 180-degree heading flips in a trajectory.

    Scans the theta array for contiguous bursts of frames where the heading
    jumped ~180° relative to the surrounding values and then returned.  Such
    bursts are corrected by adding π (mod 2π).

    Parameters
    ----------
    theta : np.ndarray
        Heading values in radians.  May contain NaN for missing frames.
    max_burst : int
        Maximum length (in frames) of a flip burst to correct.  Longer
        segments are assumed to be genuine orientation changes.

    Returns
    -------
    np.ndarray
        Corrected heading array (copy).
    """
    two_pi = 2.0 * np.pi
    result = theta.copy()
    n = len(result)
    if n < 3:
        return result

    # Iterative passes — a single pass may leave residual flips when bursts
    # are adjacent.  Three passes is sufficient for typical data.
    for _pass in range(3):
        changed = False
        i = 0
        while i < n:
            if np.isnan(result[i]):
                i += 1
                continue

            prev_idx = _find_prev_valid(result, i)
            if prev_idx < 0:
                i += 1
                continue

            prev_val = result[prev_idx]
            if _circ_diff(result[i], prev_val) <= np.pi / 2:
                i += 1
                continue

            # Found a potential flip burst starting at i
            burst_end = _measure_flip_burst(result, i, n, prev_val, max_burst)
            burst_len = burst_end - i

            if burst_len <= max_burst:
                post_idx = _find_next_valid(result, burst_end, n)
                if post_idx < n and _circ_diff(result[post_idx], prev_val) <= np.pi / 2:
                    for j in range(i, burst_end):
                        if not np.isnan(result[j]):
                            result[j] = (result[j] + np.pi) % two_pi
                    changed = True
                    i = burst_end
                    continue

            i = burst_end if burst_len > max_burst else i + 1

        if not changed:
            break

    return result


def _fix_heading_globally(theta: np.ndarray) -> np.ndarray:
    """Globally correct 180-degree heading ambiguities using dynamic programming.

    For each frame the heading can be kept as-is (state 0) or flipped by π
    (state 1).  Dynamic programming finds the assignment that minimises the
    total circular angular variation across the entire track — equivalent to
    the minimum number of flips needed to make the trajectory consistently
    directed.

    This is appropriate when a head-tail or pose model provides directed
    headings, because those models independently assign a direction to every
    frame and therefore produce isolated 180° errors that are better resolved
    globally than frame-by-frame with a hysteresis filter.

    Parameters
    ----------
    theta : np.ndarray
        Heading values in radians.  May contain NaN for missing frames.

    Returns
    -------
    np.ndarray
        Corrected heading array (copy).
    """
    two_pi = 2.0 * np.pi
    result = theta.copy()
    n = len(result)

    valid_idx = [i for i in range(n) if not np.isnan(result[i])]
    if len(valid_idx) < 2:
        return result

    INF = float("inf")

    # dp[s] = minimum total angular variation to reach the current valid frame
    # in state s (s=0: original, s=1: flipped by pi).
    dp = [0.0, 0.0]
    # parent[j][s] = state chosen at the previous valid frame that leads to the
    # optimal cost dp[s] at valid frame j.
    parent = [[None, None] for _ in range(len(valid_idx))]

    for vi in range(1, len(valid_idx)):
        prev_i = valid_idx[vi - 1]
        curr_i = valid_idx[vi]
        prev_orig = result[prev_i]
        curr_orig = result[curr_i]
        prev_flip = (prev_orig + np.pi) % two_pi
        curr_flip = (curr_orig + np.pi) % two_pi

        new_dp = [INF, INF]
        new_parent = [None, None]

        # Effective headings for each (prev_state, curr_state) pair:
        # prev_state x curr_state → angular cost
        for curr_s, curr_val in enumerate((curr_orig, curr_flip)):
            best_cost = INF
            best_prev_s = 0
            for prev_s, prev_val in enumerate((prev_orig, prev_flip)):
                diff = abs(curr_val - prev_val)
                if diff > np.pi:
                    diff = two_pi - diff
                cost = dp[prev_s] + diff
                if cost < best_cost:
                    best_cost = cost
                    best_prev_s = prev_s
            new_dp[curr_s] = best_cost
            new_parent[curr_s] = best_prev_s

        dp = new_dp
        parent[vi] = new_parent

    # Back-track to find the optimal state sequence.
    states = [None] * len(valid_idx)
    states[-1] = 0 if dp[0] <= dp[1] else 1
    for vi in range(len(valid_idx) - 1, 0, -1):
        states[vi - 1] = parent[vi][states[vi]]

    # Apply flips.
    for vi, i in enumerate(valid_idx):
        if states[vi] == 1:
            result[i] = (result[i] + np.pi) % two_pi

    return result


def _interpolate_angle(frames, angles, method="linear", max_gap=10):
    """
    Interpolate angles using circular interpolation to handle wraparound.
    """
    # Find valid (non-NaN) indices
    valid_mask = ~np.isnan(angles)
    valid_indices = np.where(valid_mask)[0]

    if len(valid_indices) < 2:
        return angles

    # Convert to Cartesian coordinates for interpolation
    sin_values = np.sin(angles)
    cos_values = np.cos(angles)

    # Interpolate sin and cos separately
    sin_interp = _interpolate_column(frames, sin_values, method=method, max_gap=max_gap)
    cos_interp = _interpolate_column(frames, cos_values, method=method, max_gap=max_gap)

    # Convert back to angles
    result = np.arctan2(sin_interp, cos_interp) % (2 * np.pi)

    # Preserve original NaN values where both sin and cos are NaN
    nan_mask = np.isnan(sin_interp) & np.isnan(cos_interp)
    result[nan_mask] = np.nan

    return result
