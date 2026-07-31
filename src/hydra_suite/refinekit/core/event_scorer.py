"""Suspicion event scorer for RefineKit.

Scores trajectory data for identity anomalies: pairwise swaps, flicker
(swap + immediate swap-back), track fragmentation, absorption (two become
one), phantom tracks, and multi-way shuffles.

This is a pure Python module with no Qt dependency.
"""

from __future__ import annotations

import itertools
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from hydra_suite.core.tracking.confidence.confidence_density import DensityRegion
from hydra_suite.refinekit.core.event_types import EventType, SuspicionEvent

# ---------------------------------------------------------------------------
# Signal weights  (pairwise swap / flicker scoring)
# ---------------------------------------------------------------------------

W_CROSSING = 1.0
W_PROXIMITY = 0.45
W_HEADING = 0.5
W_POSE_DROP = 0.3
W_REGION_DISCOUNT = 0.4
W_BOUNDARY_BONUS = 0.25

# ---------------------------------------------------------------------------
# Thresholds
# ---------------------------------------------------------------------------

_FLICKER_MAX_GAP = 40  # max frames between two crossings to call flicker
_FRAG_MAX_GAP = 20  # max frame gap between consecutive fragments
_FRAG_MAX_DIST = 80.0  # max pixel distance at fragment boundary
_FRAG_MAX_OVERLAP = 10  # max negative-gap (overlap) frames allowed
_PHANTOM_MAX_ACTIVE = 15  # max active frames for a phantom
_PHANTOM_MAX_SPAN = 50  # max total frame span for a phantom
_ABSORB_VANISH_FRAMES = 5  # min consecutive NaN frames to treat as vanished
_ABSORB_PROXIMITY = 60.0  # max distance at absorption point
_MULTI_PROXIMITY = 60.0  # proximity threshold for multi-shuffle cluster
_MULTI_WINDOW = 10  # frame window to detect simultaneous close approach


# ---------------------------------------------------------------------------
# Backward compat aliases (deprecated — prefer SuspicionEvent)
# ---------------------------------------------------------------------------

SwapSuspicionEvent = SuspicionEvent


# ---------------------------------------------------------------------------
# EventScorer
# ---------------------------------------------------------------------------


class EventScorer:
    """Score trajectory data for all suspicious event types.

    Parameters
    ----------
    regions:
        Density regions from the confidence map (may be empty).
    min_score:
        Floor for including events in output.
    approach_distance:
        Pixel distance at which two tracks are considered interacting.
    crossing_window:
        Frames before/after closest approach to check for sign flip.
    heading_reversal_deg:
        Minimum heading change (degrees) for heading signal.
    """

    def __init__(
        self,
        regions: Optional[List[DensityRegion]] = None,
        min_score: float = 0.10,
        approach_distance: float = 60.0,
        crossing_window: int = 15,
        heading_reversal_deg: float = 120.0,
    ) -> None:
        self.regions = regions or []
        self.min_score = min_score
        self.approach_distance = approach_distance
        self.crossing_window = crossing_window
        self.heading_reversal_rad = np.deg2rad(heading_reversal_deg)
        # Reviewed regions: list of (frame_start, frame_end, set_of_track_ids)
        # Events that fall entirely within a reviewed region are penalised.
        self._reviewed_regions: List[Tuple[int, int, set]] = []

    # ------------------------------------------------------------------
    # Reviewed-region management
    # ------------------------------------------------------------------

    def add_reviewed_region(
        self,
        frame_start: int,
        frame_end: int,
        track_ids: List[int],
    ) -> None:
        """Register a region the user has already proofread."""
        self._reviewed_regions.append((frame_start, frame_end, set(track_ids)))

    def _apply_review_discount(
        self, events: List[SuspicionEvent]
    ) -> List[SuspicionEvent]:
        """Penalise events whose range falls inside a reviewed region."""
        if not self._reviewed_regions:
            return events
        _DISCOUNT = 0.6
        for ev in events:
            for rstart, rend, rtracks in self._reviewed_regions:
                # Event is inside the reviewed window AND involves only reviewed tracks
                if (
                    ev.frame_range[0] >= rstart
                    and ev.frame_range[1] <= rend
                    and set(ev.involved_tracks) <= rtracks
                ):
                    ev.score = max(ev.score * (1.0 - _DISCOUNT), 0.0)
                    break
        return events

    # ==================================================================
    # Public API
    # ==================================================================

    def score_all(
        self,
        df: pd.DataFrame,
        min_score: Optional[float] = None,
    ) -> List[SuspicionEvent]:
        """Run every detector and return events sorted by score descending."""
        threshold = min_score if min_score is not None else self.min_score

        track_data = self._index_tracks(df)

        events: List[SuspicionEvent] = []
        events.extend(self._detect_pairwise(df, track_data, threshold))
        events.extend(self._detect_fragmentation(df, track_data, threshold))
        events.extend(self._detect_phantoms(df, track_data, threshold))
        events.extend(self._detect_absorption(df, track_data, threshold))

        # Post-process: promote swap-pairs to flicker, detect multi-shuffle
        events = self._promote_flickers(events)
        events = self._detect_multi_shuffle(events)

        # Penalise events in regions the user has already proofread
        events = self._apply_review_discount(events)

        events.sort(key=lambda e: e.score, reverse=True)
        return events

    def score_local(
        self,
        df: pd.DataFrame,
        affected_tracks: List[int],
        frame_range: Tuple[int, int],
        context_frames: int = 50,
        min_score: Optional[float] = None,
    ) -> List[SuspicionEvent]:
        """Score only events involving *affected_tracks* near *frame_range*.

        Much faster than :meth:`score_all` for incremental updates after an
        edit — only pairs that include at least one affected track are
        evaluated, and only frames within ``[frame_range[0] - context_frames,
        frame_range[1] + context_frames]`` are considered for the
        fragmentation, phantom, and absorption detectors.

        Returns events sorted by score descending.
        """
        threshold = min_score if min_score is not None else self.min_score
        affected_set = {int(t) for t in affected_tracks}

        # Restrict dataframe to the temporal neighbourhood for non-pairwise
        # detectors (pairwise still uses the full df for accurate distances).
        f_start = frame_range[0] - context_frames
        f_end = frame_range[1] + context_frames

        track_data = self._index_tracks(df)

        events: List[SuspicionEvent] = []

        # Pairwise: only pairs that include at least one affected track
        events.extend(
            self._detect_pairwise_local(df, track_data, affected_set, threshold)
        )

        # Structural detectors: only for affected tracks, limited frame window
        local_df = df[df["FrameID"].between(f_start, f_end)]
        local_track_data = {
            tid: tdf for tid, tdf in track_data.items() if tid in affected_set
        }
        events.extend(self._detect_fragmentation(local_df, local_track_data, threshold))
        events.extend(self._detect_phantoms(local_df, local_track_data, threshold))
        events.extend(self._detect_absorption(local_df, local_track_data, threshold))

        events = self._promote_flickers(events)
        events = self._detect_multi_shuffle(events)
        events = self._apply_review_discount(events)
        events.sort(key=lambda e: e.score, reverse=True)
        return events

    # Backward compat alias
    def score(
        self,
        df: pd.DataFrame,
        min_score: Optional[float] = None,
    ) -> List[SuspicionEvent]:
        """Alias for :meth:`score_all` (backward compatibility)."""
        return self.score_all(df, min_score)

    # ==================================================================
    # Track indexing
    # ==================================================================

    @staticmethod
    def _index_tracks(df: pd.DataFrame) -> Dict[int, pd.DataFrame]:
        track_ids = sorted(df["TrajectoryID"].unique())
        return {
            int(tid): df[df["TrajectoryID"] == tid]
            .sort_values("FrameID")
            .drop_duplicates(subset="FrameID", keep="last")
            for tid in track_ids
        }

    # ==================================================================
    # Pairwise swap detector  (same logic as old SwapScorer)
    # ==================================================================

    def _detect_pairwise(
        self,
        df: pd.DataFrame,
        track_data: Dict[int, pd.DataFrame],
        threshold: float,
    ) -> List[SuspicionEvent]:
        events: List[SuspicionEvent] = []

        for id_a, id_b in itertools.combinations(track_data, 2):
            ev = self._score_pair(df, track_data, id_a, id_b, threshold)
            if ev is not None:
                events.append(ev)

        return events

    # ==================================================================
    # Pairwise swap detector — local variant (affected tracks only)
    # ==================================================================

    def _detect_pairwise_local(
        self,
        df: pd.DataFrame,
        track_data: Dict[int, pd.DataFrame],
        affected: set,
        threshold: float,
    ) -> List[SuspicionEvent]:
        """Like ``_detect_pairwise`` but only evaluates pairs where at least
        one track is in *affected*.  Drastically reduces the combinatorial
        cost for incremental updates.
        """
        events: List[SuspicionEvent] = []
        all_ids = sorted(track_data.keys())

        for i, id_a in enumerate(all_ids):
            for id_b in all_ids[i + 1 :]:
                if id_a not in affected and id_b not in affected:
                    continue
                # Delegate to the shared pairwise logic (same as _detect_pairwise body)
                ev = self._score_pair(df, track_data, id_a, id_b, threshold)
                if ev is not None:
                    events.append(ev)

        return events

    def _collect_signals(self, a_idx, b_idx, df_a, df_b, common, distances, peak_frame):
        """Collect all signal components for a pair. Returns (signals, cr, pr, hd, pq_term) or None."""
        signals: List[str] = []

        cr = self._crossing_signal(a_idx, b_idx, common, distances)
        if cr > 0:
            signals.append("Cr")

        pr = self._proximity_signal(distances)
        if pr > 0 and cr == 0:
            signals.append("Pr")
        elif cr > 0:
            pr = 0.0

        hd = self._heading_signal(df_a, df_b, common)
        if hd > 0:
            signals.append("Hd")

        pq = self._pose_quality_signal(df_a, df_b, peak_frame)
        pq_term = 0.0
        if pq > 0 and (cr > 0 or pr > 0):
            signals.append("Pq")
            pq_term = W_POSE_DROP * pq

        if not signals:
            return None
        return signals, cr, pr, hd, pq_term

    def _score_pair(
        self,
        df: pd.DataFrame,
        track_data: Dict[int, pd.DataFrame],
        id_a: int,
        id_b: int,
        threshold: float,
    ) -> Optional[SuspicionEvent]:
        """Score a single pair and return a SuspicionEvent or None."""
        df_a = track_data[id_a]
        df_b = track_data[id_b]

        common = np.intersect1d(df_a["FrameID"].values, df_b["FrameID"].values)
        if len(common) < 3:
            return None

        a_idx = df_a.set_index("FrameID").loc[common]
        b_idx = df_b.set_index("FrameID").loc[common]

        dx = a_idx["X"].values - b_idx["X"].values
        dy = a_idx["Y"].values - b_idx["Y"].values
        distances = np.sqrt(dx**2 + dy**2)

        if float(distances.min()) >= self.approach_distance:
            return None

        peak_i = int(np.argmin(distances))
        peak_frame = int(common[peak_i])

        result = self._collect_signals(
            a_idx, b_idx, df_a, df_b, common, distances, peak_frame
        )
        if result is None:
            return None
        signals, cr, pr, hd, pq_term = result

        cx = float((a_idx["X"].values[peak_i] + b_idx["X"].values[peak_i]) / 2)
        cy = float((a_idx["Y"].values[peak_i] + b_idx["Y"].values[peak_i]) / 2)
        region_label, is_boundary = self._get_region_context(cx, cy, peak_frame)

        region_discount = (
            W_REGION_DISCOUNT
            if region_label != "open_field" and not is_boundary
            else 0.0
        )
        boundary_bonus = W_BOUNDARY_BONUS if is_boundary else 0.0

        raw = (
            W_CROSSING * cr
            + W_PROXIMITY * pr
            + W_HEADING * hd
            + pq_term
            - region_discount
            + boundary_bonus
        )
        score_val = float(np.clip(raw, 0.0, 1.0))
        if score_val < threshold:
            return None

        close_mask = distances < self.approach_distance
        close_frames = common[close_mask]
        if len(close_frames) == 0:
            close_frames = common
        frame_range = (int(close_frames.min()), int(close_frames.max()))

        return SuspicionEvent(
            event_type=EventType.SWAP,
            involved_tracks=[int(id_a), int(id_b)],
            frame_peak=peak_frame,
            frame_range=frame_range,
            score=score_val,
            signals=signals,
            region_label=region_label,
            region_boundary=is_boundary,
        )

    # ==================================================================
    # Flicker promotion  (pair of swaps on same pair within short window)
    # ==================================================================

    @staticmethod
    def _promote_flickers(events: List[SuspicionEvent]) -> List[SuspicionEvent]:
        """Merge pairs of SWAP events on the same track-pair into FLICKER."""
        swap_events = [e for e in events if e.event_type == EventType.SWAP]
        non_swap = [e for e in events if e.event_type != EventType.SWAP]

        # Group swaps by sorted track pair
        by_pair: Dict[Tuple[int, int], List[SuspicionEvent]] = {}
        for ev in swap_events:
            key = (min(ev.involved_tracks), max(ev.involved_tracks))
            by_pair.setdefault(key, []).append(ev)

        kept: List[SuspicionEvent] = []
        consumed = set()

        for key, pair_events in by_pair.items():
            pair_events.sort(key=lambda e: e.frame_peak)
            for i in range(len(pair_events)):
                if id(pair_events[i]) in consumed:
                    continue
                for j in range(i + 1, len(pair_events)):
                    if id(pair_events[j]) in consumed:
                        continue
                    gap = pair_events[j].frame_peak - pair_events[i].frame_peak
                    if gap <= _FLICKER_MAX_GAP:
                        # Merge into flicker
                        merged = SuspicionEvent(
                            event_type=EventType.FLICKER,
                            involved_tracks=list(key),
                            frame_peak=pair_events[i].frame_peak,
                            frame_range=(
                                min(
                                    pair_events[i].frame_range[0],
                                    pair_events[j].frame_range[0],
                                ),
                                max(
                                    pair_events[i].frame_range[1],
                                    pair_events[j].frame_range[1],
                                ),
                            ),
                            score=max(pair_events[i].score, pair_events[j].score),
                            signals=sorted(
                                set(pair_events[i].signals + pair_events[j].signals)
                            ),
                            region_label=pair_events[i].region_label,
                            region_boundary=(
                                pair_events[i].region_boundary
                                or pair_events[j].region_boundary
                            ),
                        )
                        kept.append(merged)
                        consumed.add(id(pair_events[i]))
                        consumed.add(id(pair_events[j]))
                        break

            # Keep un-consumed swaps
            for ev in pair_events:
                if id(ev) not in consumed:
                    kept.append(ev)

        return non_swap + kept

    # ==================================================================
    # Multi-shuffle detection  (3+ tracks in proximity cluster)
    # ==================================================================

    @staticmethod
    def _build_cluster(ev_i, pairwise, consumed, i):
        """Grow a cluster from event *ev_i*, absorbing overlapping events."""
        cluster = [ev_i]
        consumed.add(i)
        tracks_in = set(ev_i.involved_tracks)
        changed = True
        while changed:
            changed = False
            for j, ev_j in enumerate(pairwise):
                if j in consumed:
                    continue
                if (
                    ev_j.frame_range[0] <= ev_i.frame_range[1] + _MULTI_WINDOW
                    and ev_j.frame_range[1] >= ev_i.frame_range[0] - _MULTI_WINDOW
                    and tracks_in & set(ev_j.involved_tracks)
                ):
                    cluster.append(ev_j)
                    consumed.add(j)
                    tracks_in.update(ev_j.involved_tracks)
                    changed = True
        return cluster, tracks_in

    @staticmethod
    def _cluster_to_event(cluster, tracks_in):
        """Create a MULTI_SHUFFLE event from a cluster of pairwise events."""
        return SuspicionEvent(
            event_type=EventType.MULTI_SHUFFLE,
            involved_tracks=sorted(tracks_in),
            frame_peak=cluster[0].frame_peak,
            frame_range=(
                min(e.frame_range[0] for e in cluster),
                max(e.frame_range[1] for e in cluster),
            ),
            score=max(e.score for e in cluster),
            signals=sorted({s for e in cluster for s in e.signals}),
            region_label=cluster[0].region_label,
            region_boundary=any(e.region_boundary for e in cluster),
        )

    @staticmethod
    def _detect_multi_shuffle(events: List[SuspicionEvent]) -> List[SuspicionEvent]:
        """Promote overlapping swap/flicker events into MULTI_SHUFFLE."""
        pairwise = [
            e for e in events if e.event_type in (EventType.SWAP, EventType.FLICKER)
        ]
        others = [
            e for e in events if e.event_type not in (EventType.SWAP, EventType.FLICKER)
        ]
        if len(pairwise) < 2:
            return events

        consumed = set()

        for i, ev_i in enumerate(pairwise):
            if i in consumed:
                continue
            cluster, tracks_in = EventScorer._build_cluster(ev_i, pairwise, consumed, i)
            if len(cluster) >= 2 and len(tracks_in) >= 3:
                others.append(EventScorer._cluster_to_event(cluster, tracks_in))
            else:
                others.extend(cluster)

        for i, ev in enumerate(pairwise):
            if i not in consumed:
                others.append(ev)

        return others

    # ==================================================================
    # Fragmentation detector
    # ==================================================================

    @staticmethod
    def _compute_track_spans(
        tids, track_data
    ) -> Dict[int, Tuple[int, int, float, float, float, float]]:
        """Compute span for each track: (first_frame, last_frame, last_x, last_y, first_x, first_y)."""
        spans: Dict[int, Tuple[int, int, float, float, float, float]] = {}
        for tid in tids:
            tdf = track_data[tid]
            active = tdf.dropna(subset=["X", "Y"])
            if active.empty:
                continue
            first_row = active.iloc[0]
            last_row = active.iloc[-1]
            spans[tid] = (
                int(first_row["FrameID"]),
                int(last_row["FrameID"]),
                float(last_row["X"]),
                float(last_row["Y"]),
                float(first_row["X"]),
                float(first_row["Y"]),
            )
        return spans

    @staticmethod
    def _score_fragment(gap, dist):
        """Score a candidate fragment connection given gap and distance."""
        overlap = max(0, -gap)
        effective_gap = abs(gap)
        raw_score = (
            0.7 * (1.0 - dist / _FRAG_MAX_DIST)
            + 0.2 * (1.0 - effective_gap / _FRAG_MAX_GAP)
            - 0.1 * (overlap / max(_FRAG_MAX_OVERLAP, 1))
        )
        return float(np.clip(raw_score, 0.0, 1.0))

    def _detect_fragmentation(
        self,
        df: pd.DataFrame,
        track_data: Dict[int, pd.DataFrame],
        threshold: float,
    ) -> List[SuspicionEvent]:
        """Find trajectory fragments that are likely the same animal."""
        events: List[SuspicionEvent] = []
        tids = sorted(track_data.keys())
        spans = self._compute_track_spans(tids, track_data)

        seen = set()
        for tid_a in tids:
            if tid_a not in spans:
                continue
            a_start, a_end, a_last_x, a_last_y, _, _ = spans[tid_a]
            for tid_b in tids:
                if tid_b <= tid_a or tid_b not in spans:
                    continue
                pair_key = (tid_a, tid_b)
                if pair_key in seen:
                    continue

                b_start, b_end, _, _, b_first_x, b_first_y = spans[tid_b]

                # Forward direction: a -> b
                self._check_fragment_direction(
                    a_end,
                    b_start,
                    a_last_x,
                    a_last_y,
                    b_first_x,
                    b_first_y,
                    tid_a,
                    tid_b,
                    a_start,
                    b_end,
                    threshold,
                    seen,
                    pair_key,
                    events,
                )

                # Reverse direction: b -> a
                if pair_key not in seen:
                    _, _, b_last_x, b_last_y, _, _ = spans[tid_b]
                    a_first_x, a_first_y = spans[tid_a][4], spans[tid_a][5]
                    self._check_fragment_direction(
                        b_end,
                        a_start,
                        b_last_x,
                        b_last_y,
                        a_first_x,
                        a_first_y,
                        tid_b,
                        tid_a,
                        b_start,
                        a_end,
                        threshold,
                        seen,
                        pair_key,
                        events,
                    )

        return events

    def _check_fragment_direction(
        self,
        end_frame,
        start_frame,
        end_x,
        end_y,
        start_x,
        start_y,
        tid_src,
        tid_dst,
        src_start,
        dst_end,
        threshold,
        seen,
        pair_key,
        events,
    ):
        """Check one direction of fragment connection and append event if scored."""
        gap = start_frame - end_frame
        if not (-_FRAG_MAX_OVERLAP <= gap <= _FRAG_MAX_GAP):
            return
        dist = np.sqrt((end_x - start_x) ** 2 + (end_y - start_y) ** 2)
        if dist >= _FRAG_MAX_DIST:
            return
        score_val = self._score_fragment(gap, dist)
        if score_val >= threshold:
            seen.add(pair_key)
            events.append(
                SuspicionEvent(
                    event_type=EventType.FRAGMENTATION,
                    involved_tracks=[int(tid_src), int(tid_dst)],
                    frame_peak=end_frame,
                    frame_range=(
                        max(end_frame - 5, src_start),
                        min(start_frame + 5, dst_end),
                    ),
                    score=score_val,
                    signals=["Frag"],
                    region_label="open_field",
                )
            )

    # ==================================================================
    # Phantom detector
    # ==================================================================

    def _detect_phantoms(
        self,
        df: pd.DataFrame,
        track_data: Dict[int, pd.DataFrame],
        threshold: float,
    ) -> List[SuspicionEvent]:
        """Flag very short trajectories as potential phantoms."""
        events: List[SuspicionEvent] = []

        for tid, tdf in track_data.items():
            active = tdf.dropna(subset=["X", "Y"])
            n_active = len(active)
            if n_active == 0:
                continue

            total_span = int(tdf["FrameID"].max()) - int(tdf["FrameID"].min()) + 1
            if n_active > _PHANTOM_MAX_ACTIVE or total_span > _PHANTOM_MAX_SPAN:
                continue

            # Score inversely with length
            score_val = float(
                np.clip(
                    0.6 * (1.0 - n_active / _PHANTOM_MAX_ACTIVE)
                    + 0.3 * (1.0 - total_span / _PHANTOM_MAX_SPAN),
                    0.0,
                    1.0,
                )
            )
            if score_val < threshold:
                continue

            first_frame = int(tdf["FrameID"].min())
            last_frame = int(tdf["FrameID"].max())
            mid = (first_frame + last_frame) // 2

            events.append(
                SuspicionEvent(
                    event_type=EventType.PHANTOM,
                    involved_tracks=[int(tid)],
                    frame_peak=mid,
                    frame_range=(first_frame, last_frame),
                    score=score_val,
                    signals=["Ph"],
                )
            )

        return events

    # ==================================================================
    # Absorption detector
    # ==================================================================

    def _detect_absorption(
        self,
        df: pd.DataFrame,
        track_data: Dict[int, pd.DataFrame],
        threshold: float,
    ) -> List[SuspicionEvent]:
        """Detect when one track vanishes while near another (merge)."""
        events: List[SuspicionEvent] = []

        for tid_a, tdf_a in track_data.items():
            active_a = tdf_a.dropna(subset=["X", "Y"])
            if len(active_a) < 5:
                continue

            # Find runs of NaN (vanish) in this track
            frames_a = tdf_a["FrameID"].values
            x_a = tdf_a["X"].values
            vanish_start = None

            for i in range(len(x_a)):
                if np.isnan(x_a[i]):
                    if vanish_start is None:
                        vanish_start = i
                else:
                    if vanish_start is not None:
                        vanish_len = i - vanish_start
                        if vanish_len >= _ABSORB_VANISH_FRAMES:
                            # Track vanished — check if another was nearby
                            self._check_absorption_at(
                                track_data,
                                tid_a,
                                tdf_a,
                                int(frames_a[vanish_start]),
                                int(frames_a[i - 1]),
                                events,
                                threshold,
                            )
                        vanish_start = None

            # Handle trailing vanish
            if vanish_start is not None:
                vanish_len = len(x_a) - vanish_start
                if vanish_len >= _ABSORB_VANISH_FRAMES:
                    self._check_absorption_at(
                        track_data,
                        tid_a,
                        tdf_a,
                        int(frames_a[vanish_start]),
                        int(frames_a[-1]),
                        events,
                        threshold,
                    )

        return events

    def _check_absorption_at(
        self,
        track_data: Dict[int, pd.DataFrame],
        vanished_tid: int,
        vanished_tdf: pd.DataFrame,
        vanish_frame_start: int,
        vanish_frame_end: int,
        events: List[SuspicionEvent],
        threshold: float,
    ) -> None:
        """Check if the vanished track was near an active track at vanish time."""
        # Get last known position before vanishing
        before = vanished_tdf[(vanished_tdf["FrameID"] < vanish_frame_start)].dropna(
            subset=["X", "Y"]
        )
        if before.empty:
            return
        last_row = before.iloc[-1]
        vx, vy = float(last_row["X"]), float(last_row["Y"])

        for tid_b, tdf_b in track_data.items():
            if tid_b == vanished_tid:
                continue
            # Check if track_b was active and nearby at the vanish frame
            near = tdf_b[
                (tdf_b["FrameID"] >= vanish_frame_start - 2)
                & (tdf_b["FrameID"] <= vanish_frame_start + 2)
            ].dropna(subset=["X", "Y"])
            if near.empty:
                continue
            bx = float(near["X"].mean())
            by = float(near["Y"].mean())
            dist = np.sqrt((vx - bx) ** 2 + (vy - by) ** 2)
            if dist < _ABSORB_PROXIMITY:
                score_val = float(
                    np.clip(
                        0.6 * (1.0 - dist / _ABSORB_PROXIMITY) + 0.3,
                        0.0,
                        1.0,
                    )
                )
                if score_val >= threshold:
                    events.append(
                        SuspicionEvent(
                            event_type=EventType.ABSORPTION,
                            involved_tracks=[int(vanished_tid), int(tid_b)],
                            frame_peak=vanish_frame_start,
                            frame_range=(
                                max(vanish_frame_start - 10, 0),
                                vanish_frame_end + 10,
                            ),
                            score=score_val,
                            signals=["Ab"],
                        )
                    )
                    return  # one absorber per vanish event

    # ==================================================================
    # Pairwise signal helpers (unchanged from old SwapScorer)
    # ==================================================================

    def _crossing_signal(
        self,
        a_indexed: pd.DataFrame,
        b_indexed: pd.DataFrame,
        common: np.ndarray,
        distances: np.ndarray,
    ) -> float:
        peak_idx = int(np.argmin(distances))
        min_dist = float(distances[peak_idx])
        if min_dist >= self.approach_distance:
            return 0.0

        rel_x = a_indexed["X"].values - b_indexed["X"].values
        w = self.crossing_window
        before_start = max(0, peak_idx - w)
        after_end = min(len(common), peak_idx + w + 1)

        if peak_idx - before_start < 1 or after_end - peak_idx < 2:
            return 0.0

        rel_before = rel_x[before_start:peak_idx]
        rel_after = rel_x[peak_idx + 1 : after_end]
        if len(rel_before) == 0 or len(rel_after) == 0:
            return 0.0

        sign_before = np.sign(np.median(rel_before))
        sign_after = np.sign(np.median(rel_after))
        if sign_before == 0 or sign_after == 0:
            return 0.0

        if sign_before != sign_after:
            strength = 1.0 - (min_dist / self.approach_distance)
            return float(np.clip(strength, 0.0, 1.0))
        return 0.0

    def _proximity_signal(self, distances: np.ndarray) -> float:
        min_dist = float(distances.min())
        if min_dist >= self.approach_distance:
            return 0.0
        strength = 0.6 * (1.0 - min_dist / self.approach_distance)
        return float(np.clip(strength, 0.0, 1.0))

    def _heading_signal(
        self,
        df_a: pd.DataFrame,
        df_b: pd.DataFrame,
        common: np.ndarray,
    ) -> float:
        max_diff = 0.0
        for tdf in (df_a, df_b):
            tdf_common = tdf[tdf["FrameID"].isin(common)].sort_values("FrameID")
            if len(tdf_common) < 2:
                continue
            thetas = tdf_common["Theta"].values
            diffs = np.abs(np.arctan2(np.sin(np.diff(thetas)), np.cos(np.diff(thetas))))
            if len(diffs) > 0:
                max_diff = max(max_diff, float(diffs.max()))

        if max_diff >= self.heading_reversal_rad:
            return float(min(max_diff / np.pi, 1.0))
        return 0.0

    def _pose_quality_signal(
        self,
        df_a: pd.DataFrame,
        df_b: pd.DataFrame,
        peak_frame: int,
    ) -> float:
        if "PoseQualityScore" not in df_a.columns:
            return 0.0
        window = 10
        z_threshold = 2.5
        max_drop = 0.0
        for tdf in (df_a, df_b):
            if "PoseQualityScore" not in tdf.columns:
                continue
            scores = tdf["PoseQualityScore"].dropna()
            if len(scores) < 5:
                continue
            overall_mean = float(scores.mean())
            overall_std = float(scores.std())
            if overall_std < 1e-8:
                continue
            near_peak = tdf[
                (tdf["FrameID"] >= peak_frame - window)
                & (tdf["FrameID"] <= peak_frame + window)
            ]
            if len(near_peak) == 0:
                continue
            local_mean = float(near_peak["PoseQualityScore"].mean())
            z_score = (overall_mean - local_mean) / overall_std
            if z_score > z_threshold:
                drop = min(z_score / 5.0, 1.0)
                max_drop = max(max_drop, drop)
        return max_drop

    def _get_region_context(
        self,
        cx: float,
        cy: float,
        frame: int,
    ) -> Tuple[str, bool]:
        for region in self.regions:
            if region.contains(frame, cx, cy):
                return region.label, region.is_boundary_frame(frame)
        return "open_field", False


# ---------------------------------------------------------------------------
# Backward compat: old name
# ---------------------------------------------------------------------------

SwapScorer = EventScorer
