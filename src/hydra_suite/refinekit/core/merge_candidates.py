"""Merge and swap candidate scoring for fragment proofreading.

Builds a directed graph of merge hypotheses: for each dying track,
ranks candidate continuations by spatial distance, heading agreement,
Kalman-extrapolated distance, and gap size.

Also builds swap hypotheses: for each dying track, finds alive tracks
that are nearby at the time of death and whose post-swap trajectory
continues the dead track's motion.

This is a pure Python module with no Qt dependency.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class TrackSegment:
    """Trajectory endpoint summary for merge candidate ranking."""

    track_id: int
    frame_birth: int
    frame_death: int
    pos_birth: Tuple[float, float]
    pos_death: Tuple[float, float]
    heading_birth: float
    heading_death: float
    vel_death: Tuple[float, float]
    n_active_frames: int
    is_alive_at_end: bool


@dataclass
class MergeCandidate:
    """Directed merge hypothesis: source -> target."""

    source_id: int
    target_id: int
    gap_frames: int
    spatial_dist: float
    predicted_dist: float
    heading_agreement: float
    overlap_frames: int  # noqa: DC01  (dataclass field)
    score: float


@dataclass
class SwapCandidate:
    """Swap hypothesis: source's continuation is target's post-swap trajectory.

    Accepting a swap relabels *target*'s rows from *swap_frame* onward
    to *source_id*, making *source* live again.  *target* is truncated
    to ``[target_birth, swap_frame - 1]`` and becomes a new dead
    fragment that re-enters the candidate pool.
    """

    source_id: int
    target_id: int
    swap_frame: int
    min_distance: float
    heading_continuity: float
    velocity_continuity: float  # noqa: DC01  (dataclass field)
    prediction_match: float  # noqa: DC01  (dataclass field)
    score: float


@dataclass(frozen=True)
class MergeSearchTuning:
    """Active merge-search parameters for RefineKit candidate generation."""

    max_dist: float
    max_pred_dist: float
    max_dist_cap: float
    required_reductions: int


# ---------------------------------------------------------------------------
# Scoring weights (tunable)
# ---------------------------------------------------------------------------

W_DIST = 0.35
W_PRED = 0.25
W_GAP = 0.15
W_HEAD = 0.15
W_OVERLAP = 0.10

MAX_DIST = 200.0
MAX_PRED_DIST = 120.0
MAX_GAP = 60
MAX_OVERLAP = 0

MIN_SCORE = 0.05


# ---------------------------------------------------------------------------
# Swap scoring weights (tunable)
# ---------------------------------------------------------------------------

SW_PROX = 0.25  # proximity at swap point
SW_HEAD = 0.25  # heading continuity
SW_PRED = 0.30  # Kalman-predicted trajectory match
SW_TIME = 0.10  # prefer swap near the death frame
SW_SPEED = 0.10  # speed magnitude continuity

MAX_SWAP_DIST = 120.0  # max pixels between tracks at swap point
SWAP_SEARCH_WINDOW = 20  # frames before/after death to search
MIN_PRE_SWAP_FRAMES = 10  # target must be alive this long before swap
MIN_SWAP_SCORE = 0.10


# ---------------------------------------------------------------------------
# Kalman extrapolation (Tier 1 — plain CV fallback)
# ---------------------------------------------------------------------------


def predict_position(
    pos: Tuple[float, float],
    vel: Tuple[float, float],
    n_frames: int,
    damping: float = 0.95,
) -> Tuple[float, float]:
    """Constant-velocity extrapolation with per-frame damping."""
    x, y = pos
    vx, vy = vel
    for _ in range(n_frames):
        x += vx
        y += vy
        vx *= damping
        vy *= damping
    return (x, y)


# ---------------------------------------------------------------------------
# Heading / velocity helpers
# ---------------------------------------------------------------------------


def _finite_diff_heading(xs: np.ndarray, ys: np.ndarray) -> float:
    """Estimate heading from a short tail of (x, y) positions."""
    if len(xs) < 2:
        return 0.0
    dx = float(xs[-1] - xs[0])
    dy = float(ys[-1] - ys[0])
    if abs(dx) < 1e-9 and abs(dy) < 1e-9:
        return 0.0
    return math.atan2(dy, dx)


def _finite_diff_velocity(xs: np.ndarray, ys: np.ndarray) -> Tuple[float, float]:
    """Estimate velocity from a short tail of (x, y) positions."""
    if len(xs) < 2:
        return (0.0, 0.0)
    n = min(5, len(xs))
    dx = float(xs[-1] - xs[-n]) / max(n - 1, 1)
    dy = float(ys[-1] - ys[-n]) / max(n - 1, 1)
    return (dx, dy)


# ---------------------------------------------------------------------------
# Segment extraction
# ---------------------------------------------------------------------------


def extract_segments(
    df: pd.DataFrame,
    last_frame: int,
) -> List[TrackSegment]:
    """Build :class:`TrackSegment` list from trajectory DataFrame."""
    segments: List[TrackSegment] = []
    for tid, grp in df.groupby("TrajectoryID"):
        active = grp.dropna(subset=["X", "Y"])
        if active.empty:
            continue

        xs = active["X"].values
        ys = active["Y"].values

        frame_birth = int(active["FrameID"].iloc[0])
        frame_death = int(active["FrameID"].iloc[-1])

        pos_birth = (float(xs[0]), float(ys[0]))
        pos_death = (float(xs[-1]), float(ys[-1]))

        # Heading at birth (from first few points)
        n_head = min(5, len(xs))
        heading_birth = _finite_diff_heading(xs[:n_head], ys[:n_head])

        # Heading at death (from last few points)
        heading_death = _finite_diff_heading(xs[-n_head:], ys[-n_head:])

        # Use Theta column if available
        if "Theta" in active.columns:
            theta_vals = active["Theta"].dropna()
            if not theta_vals.empty:
                heading_birth = float(theta_vals.iloc[0])
                heading_death = float(theta_vals.iloc[-1])

        vel_death = _finite_diff_velocity(xs, ys)

        segments.append(
            TrackSegment(
                track_id=int(tid),
                frame_birth=frame_birth,
                frame_death=frame_death,
                pos_birth=pos_birth,
                pos_death=pos_death,
                heading_birth=heading_birth,
                heading_death=heading_death,
                vel_death=vel_death,
                n_active_frames=len(active),
                is_alive_at_end=(frame_death >= last_frame),
            )
        )

    return segments


# ---------------------------------------------------------------------------
# Candidate scoring
# ---------------------------------------------------------------------------


def _score_candidate(
    source: TrackSegment,
    target: TrackSegment,
    max_gap: int,
    max_overlap: int,
    max_dist: float,
    max_pred_dist: float,
) -> Optional[MergeCandidate]:
    """Score a single source->target merge hypothesis.

    Returns ``None`` if the candidate fails hard gating.
    """
    gap = target.frame_birth - source.frame_death
    overlap = max(0, -gap)

    # Merging overlapping fragments would create simultaneous duplicate IDs.
    if overlap > 0:
        return None

    # Spatial distance at junction
    dx = target.pos_birth[0] - source.pos_death[0]
    dy = target.pos_birth[1] - source.pos_death[1]
    spatial_dist = math.sqrt(dx * dx + dy * dy)
    if spatial_dist > max_dist:
        return None

    # Kalman-predicted distance
    n_pred = max(1, abs(gap))
    pred_x, pred_y = predict_position(source.pos_death, source.vel_death, n_pred)
    pdx = target.pos_birth[0] - pred_x
    pdy = target.pos_birth[1] - pred_y
    predicted_dist = math.sqrt(pdx * pdx + pdy * pdy)

    # Heading agreement: cos(θ_death_A − θ_birth_B)
    heading_agreement = math.cos(source.heading_death - target.heading_birth)

    # Composite score
    score = (
        W_DIST * (1.0 - min(spatial_dist / max_dist, 1.0))
        + W_PRED * (1.0 - min(predicted_dist / max_pred_dist, 1.0))
        + W_GAP * (1.0 - min(abs(gap) / max(max_gap, 1), 1.0))
        + W_HEAD * max(0.0, heading_agreement)
        - W_OVERLAP * (overlap / max(max_overlap, 1))
    )
    score = max(0.0, min(1.0, score))

    return MergeCandidate(
        source_id=source.track_id,
        target_id=target.track_id,
        gap_frames=gap,
        spatial_dist=spatial_dist,
        predicted_dist=predicted_dist,
        heading_agreement=heading_agreement,
        overlap_frames=overlap,
        score=score,
    )


def build_candidates(
    segments: List[TrackSegment],
    *,
    max_gap: int = MAX_GAP,
    max_overlap: int = MAX_OVERLAP,
    max_dist: float = MAX_DIST,
    max_pred_dist: float = MAX_PRED_DIST,
    min_score: float = MIN_SCORE,
) -> Dict[int, List[MergeCandidate]]:
    """For each dying track, return ranked list of merge candidates.

    Returns a dict mapping ``source_id`` to a list of
    :class:`MergeCandidate` sorted by score (descending).
    """
    # Sort targets by birth frame for windowed search
    targets_by_birth = sorted(segments, key=lambda s: s.frame_birth)

    candidates: Dict[int, List[MergeCandidate]] = {}

    for source in segments:
        if source.is_alive_at_end:
            continue  # Track survives to end — no merge needed

        source_cands: List[MergeCandidate] = []
        window_start = source.frame_death - max_overlap
        window_end = source.frame_death + max_gap

        for target in targets_by_birth:
            if target.track_id == source.track_id:
                continue
            if target.frame_birth < window_start:
                continue
            if target.frame_birth > window_end:
                break  # Past window — all remaining are later

            mc = _score_candidate(
                source, target, max_gap, max_overlap, max_dist, max_pred_dist
            )
            if mc is not None and mc.score >= min_score:
                source_cands.append(mc)

        if source_cands:
            source_cands.sort(key=lambda c: c.score, reverse=True)
            candidates[source.track_id] = source_cands

    return candidates


def compute_required_reductions(
    segments: List[TrackSegment],
    max_animals: Optional[int],
) -> int:
    """Return how many fragment merges are needed to reach the target count."""
    if max_animals is None:
        return 0
    return max(0, len(segments) - max(1, int(max_animals)))


def compute_merge_distance_cap(
    segments: List[TrackSegment],
    *,
    max_gap: int = MAX_GAP,
) -> float:
    """Return the farthest forward-only merge distance still plausible in time."""
    max_dist = MAX_DIST
    targets_by_birth = sorted(segments, key=lambda s: s.frame_birth)

    for source in segments:
        if source.is_alive_at_end:
            continue
        window_end = source.frame_death + max_gap
        for target in targets_by_birth:
            if target.track_id == source.track_id:
                continue
            if target.frame_birth <= source.frame_death:
                continue
            if target.frame_birth > window_end:
                break
            dx = target.pos_birth[0] - source.pos_death[0]
            dy = target.pos_birth[1] - source.pos_death[1]
            max_dist = max(max_dist, math.sqrt(dx * dx + dy * dy))

    return float(max_dist)


def next_merge_search_distance(
    current_max_dist: float,
    max_dist_cap: float,
    *,
    expansion_step: float = 50.0,
) -> float:
    """Return the next merge-radius step, clamped to the computed cap."""
    if current_max_dist >= max_dist_cap:
        return float(current_max_dist)
    return float(min(max_dist_cap, current_max_dist + expansion_step))


def build_candidates_for_target_count(
    segments: List[TrackSegment],
    *,
    max_animals: Optional[int],
    max_gap: int = MAX_GAP,
    max_overlap: int = MAX_OVERLAP,
    start_max_dist: float = MAX_DIST,
    start_max_pred_dist: float = MAX_PRED_DIST,
    min_score: float = MIN_SCORE,
    expansion_step: float = 50.0,
) -> Tuple[Dict[int, List[MergeCandidate]], MergeSearchTuning]:
    """Build merge candidates and auto-expand distance until target count is reachable."""
    required_reductions = compute_required_reductions(segments, max_animals)
    max_dist_cap = compute_merge_distance_cap(segments, max_gap=max_gap)
    current_max_dist = float(max(start_max_dist, MAX_DIST))
    current_max_pred_dist = float(max(start_max_pred_dist, MAX_PRED_DIST))

    candidates = build_candidates(
        segments,
        max_gap=max_gap,
        max_overlap=max_overlap,
        max_dist=current_max_dist,
        max_pred_dist=current_max_pred_dist,
        min_score=min_score,
    )

    while required_reductions > 0 and len(candidates) < required_reductions:
        next_max_dist = next_merge_search_distance(
            current_max_dist,
            max_dist_cap,
            expansion_step=expansion_step,
        )
        if next_max_dist <= current_max_dist:
            break
        scale = next_max_dist / max(current_max_dist, 1.0)
        current_max_dist = next_max_dist
        current_max_pred_dist *= scale
        candidates = build_candidates(
            segments,
            max_gap=max_gap,
            max_overlap=max_overlap,
            max_dist=current_max_dist,
            max_pred_dist=current_max_pred_dist,
            min_score=min_score,
        )

    tuning = MergeSearchTuning(
        max_dist=current_max_dist,
        max_pred_dist=current_max_pred_dist,
        max_dist_cap=max_dist_cap,
        required_reductions=required_reductions,
    )
    return candidates, tuning


# ---------------------------------------------------------------------------
# Swap candidate scoring
# ---------------------------------------------------------------------------


def _score_swap_candidate(
    df: pd.DataFrame,
    source: TrackSegment,
    target: TrackSegment,
    swap_window: int,
    max_swap_dist: float,
) -> Optional[SwapCandidate]:
    """Score a single swap hypothesis (source dies, target carries its ID).

    Returns ``None`` if the candidate fails hard gating.
    """
    D = source.frame_death
    search_start = D
    search_end = min(target.frame_death, D + swap_window)

    # Get positions of both tracks in the search window
    src_rows = df[
        (df["TrajectoryID"] == source.track_id)
        & df["FrameID"].between(search_start, search_end)
    ].dropna(subset=["X", "Y"])
    tgt_rows = df[
        (df["TrajectoryID"] == target.track_id)
        & df["FrameID"].between(search_start, search_end)
    ].dropna(subset=["X", "Y"])

    if src_rows.empty or tgt_rows.empty:
        return None

    # Find minimum distance across common frames.
    # Deduplicate by FrameID first (keep first occurrence per frame) so that
    # the aligned arrays always have matching shapes.
    src_pos = src_rows.drop_duplicates("FrameID").set_index("FrameID")[["X", "Y"]]
    tgt_pos = tgt_rows.drop_duplicates("FrameID").set_index("FrameID")[["X", "Y"]]
    common = src_pos.index.intersection(tgt_pos.index)

    if len(common) == 0:
        return None

    dx = src_pos.loc[common, "X"].values - tgt_pos.loc[common, "X"].values
    dy = src_pos.loc[common, "Y"].values - tgt_pos.loc[common, "Y"].values
    dists = np.sqrt(dx * dx + dy * dy)

    min_idx = int(np.argmin(dists))
    min_dist = float(dists[min_idx])
    swap_frame = int(common[min_idx])

    if min_dist > max_swap_dist:
        return None

    # --- Heading continuity: source heading at death vs target post-swap ---
    post_tgt = (
        df[(df["TrajectoryID"] == target.track_id) & (df["FrameID"] > swap_frame)]
        .dropna(subset=["X", "Y"])
        .head(10)
    )
    if len(post_tgt) < 2:
        return None
    tgt_post_heading = _finite_diff_heading(post_tgt["X"].values, post_tgt["Y"].values)
    heading_cont = math.cos(source.heading_death - tgt_post_heading)

    # --- Velocity / speed continuity ---
    tgt_post_vel = _finite_diff_velocity(post_tgt["X"].values, post_tgt["Y"].values)
    src_speed = math.sqrt(source.vel_death[0] ** 2 + source.vel_death[1] ** 2)
    tgt_speed = math.sqrt(tgt_post_vel[0] ** 2 + tgt_post_vel[1] ** 2)
    max_speed = max(src_speed, tgt_speed, 1.0)
    speed_cont = 1.0 - abs(src_speed - tgt_speed) / max_speed

    # --- Prediction match: extrapolate source into post-swap, compare ---
    n_pred = max(1, swap_frame - D + 5)
    pred_x, pred_y = predict_position(source.pos_death, source.vel_death, n_pred)
    if len(post_tgt) >= 5:
        actual_x = float(post_tgt["X"].iloc[4])
        actual_y = float(post_tgt["Y"].iloc[4])
    else:
        actual_x = float(post_tgt["X"].iloc[-1])
        actual_y = float(post_tgt["Y"].iloc[-1])
    pred_dist = math.sqrt((pred_x - actual_x) ** 2 + (pred_y - actual_y) ** 2)
    pred_match = max(0.0, 1.0 - pred_dist / (max_swap_dist * 2))

    # --- Time penalty: prefer swap close to death frame ---
    time_score = 1.0 - abs(swap_frame - D) / max(swap_window, 1)

    # --- Composite score ---
    score = (
        SW_PROX * (1.0 - min(min_dist / max_swap_dist, 1.0))
        + SW_HEAD * max(0.0, heading_cont)
        + SW_PRED * pred_match
        + SW_TIME * max(0.0, time_score)
        + SW_SPEED * max(0.0, speed_cont)
    )
    score = max(0.0, min(1.0, score))

    return SwapCandidate(
        source_id=source.track_id,
        target_id=target.track_id,
        swap_frame=swap_frame,
        min_distance=min_dist,
        heading_continuity=heading_cont,
        velocity_continuity=speed_cont,
        prediction_match=pred_match,
        score=score,
    )


def build_swap_candidates(
    df: pd.DataFrame,
    segments: List[TrackSegment],
    *,
    max_swap_dist: float = MAX_SWAP_DIST,
    swap_window: int = SWAP_SEARCH_WINDOW,
    min_pre_swap: int = MIN_PRE_SWAP_FRAMES,
    min_score: float = MIN_SWAP_SCORE,
) -> Dict[int, List[SwapCandidate]]:
    """For each dying track, find alive tracks that may have swapped identity.

    Returns a dict mapping ``source_id`` to a list of
    :class:`SwapCandidate` sorted by score (descending).

    Only considers targets that were alive well before the source died
    (at least *min_pre_swap* frames), so that newly-born tracks are left
    to the merge-candidate pathway.
    """
    candidates: Dict[int, List[SwapCandidate]] = {}

    for source in segments:
        if source.is_alive_at_end:
            continue

        D = source.frame_death
        source_cands: List[SwapCandidate] = []

        for target in segments:
            if target.track_id == source.track_id:
                continue
            # Target must have been alive well before source died
            if target.frame_birth > D - min_pre_swap:
                continue
            # Target must still be alive after source dies
            if target.frame_death <= D:
                continue

            sc = _score_swap_candidate(
                df,
                source,
                target,
                swap_window,
                max_swap_dist,
            )
            if sc is not None and sc.score >= min_score:
                source_cands.append(sc)

        if source_cands:
            source_cands.sort(key=lambda c: c.score, reverse=True)
            candidates[source.track_id] = source_cands

    return candidates


# ---------------------------------------------------------------------------
# Graph update after merge
# ---------------------------------------------------------------------------


def _purge_stale_candidates(
    candidates: Dict[int, List[MergeCandidate]],
    source_id: int,
    target_id: int,
) -> Dict[int, List[MergeCandidate]]:
    """Remove old source/target entries and stale target references."""
    new_candidates = {
        k: v for k, v in candidates.items() if k not in (source_id, target_id)
    }
    for k in list(new_candidates.keys()):
        new_candidates[k] = [c for c in new_candidates[k] if c.target_id != target_id]
        if not new_candidates[k]:
            del new_candidates[k]
    return new_candidates


def _recompute_candidates_for_merged(
    merged: "TrackSegment",
    new_segments: List["TrackSegment"],
    new_candidates: Dict[int, List[MergeCandidate]],
) -> None:
    """Compute new candidate links from *merged* segment's death point."""
    if merged.is_alive_at_end:
        return
    targets_by_birth = sorted(new_segments, key=lambda s: s.frame_birth)
    merged_cands: List[MergeCandidate] = []
    window_start = merged.frame_death - MAX_OVERLAP
    window_end = merged.frame_death + MAX_GAP

    for t in targets_by_birth:
        if t.track_id == merged.track_id:
            continue
        if t.frame_birth < window_start:
            continue
        if t.frame_birth > window_end:
            break
        mc = _score_candidate(merged, t, MAX_GAP, MAX_OVERLAP, MAX_DIST, MAX_PRED_DIST)
        if mc is not None and mc.score >= MIN_SCORE:
            merged_cands.append(mc)

    if merged_cands:
        merged_cands.sort(key=lambda c: c.score, reverse=True)
        new_candidates[merged.track_id] = merged_cands


def update_after_merge(
    segments: List[TrackSegment],
    candidates: Dict[int, List[MergeCandidate]],
    source_id: int,
    target_id: int,
) -> Tuple[List[TrackSegment], Dict[int, List[MergeCandidate]]]:
    """Update segment list and candidate graph after accepting a merge.

    Collapses *source* and *target* into a single segment that keeps
    the source's birth and the target's death.  Recomputes candidates
    from the merged segment's new death point.
    """
    seg_by_id: Dict[int, TrackSegment] = {s.track_id: s for s in segments}

    source = seg_by_id.get(source_id)
    target = seg_by_id.get(target_id)
    if source is None or target is None:
        return segments, candidates

    merged = TrackSegment(
        track_id=source.track_id,
        frame_birth=source.frame_birth,
        frame_death=target.frame_death,
        pos_birth=source.pos_birth,
        pos_death=target.pos_death,
        heading_birth=source.heading_birth,
        heading_death=target.heading_death,
        vel_death=target.vel_death,
        n_active_frames=source.n_active_frames + target.n_active_frames,
        is_alive_at_end=target.is_alive_at_end,
    )

    new_segments = [s for s in segments if s.track_id not in (source_id, target_id)]
    new_segments.append(merged)

    new_candidates = _purge_stale_candidates(candidates, source_id, target_id)
    _recompute_candidates_for_merged(merged, new_segments, new_candidates)

    return new_segments, new_candidates
