"""Global identity fragment solver.

Identity post-processing pipeline:
1. PELT changepoint detection on per-trajectory smoothed identity posteriors.
2. Fragment building from detected changepoints.
3. Mass-first seeding + doubt-ordered refinement (``_iterative_assign``):
   fragments are first seeded in descending evidence-mass order (duration x
   top support) so long, high-confidence tracks claim their label before
   short, noisy fragments get a turn (2026-08-27 identity-final-consistency:
   replaces a dead component-Hungarian base step whose temporal-overlap
   connected components collapsed to one giant component on any
   multi-animal clip). Refinement then walks fragments in order of doubt
   score (low CNN stability × short length × poor spatial fit + Unknown
   bonus), evaluates the top-K candidate labels for each, and commits a
   flip -- or a multi-blocker displacement of the label's current
   occupant(s) -- only if it strictly increases (by at least
   ``ASSIGNMENT_MARGIN_THRESHOLD``, floored at 1e-3) the exact objective
   (sum of evidence × spatial × length) over every fragment the move
   touches. Iterates to a fixed point. Long fragments with stable
   per-frame CNN agreement settle to the bottom of the queue and act as
   anchors; short or jittery fragments yield to the schedule formed by them.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from hydra_suite.core.individual.identity import columns as C
from hydra_suite.core.individual.identity.catalog import IdentityCatalog
from hydra_suite.core.individual.identity.smoothing import (
    load_trajectory_evidence,
    smooth_trajectory_posteriors,
    smoothed_label_and_conf,
)

log = logging.getLogger(__name__)

_LABEL_COL = C.FINAL_LABEL
_CONF_COL = C.FINAL_CONFIDENCE
_UNKNOWN_VALUES = frozenset({"", "unknown"})


def _fragment_stability(per_row_probs: np.ndarray) -> float:
    """Combined agreement × mean-margin stability score in [0, 1].

    Stability is high for fragments whose per-frame argmax is consistently the
    same label (high agreement) *and* whose top-1 / top-2 separation is wide
    (high mean margin). A long fragment with jittery per-frame predictions or
    a small margin scores low even though it has many rows; a short fragment
    that is internally consistent and confident scores high.

    Returns 0.0 when no per-frame evidence is present. Used by
    ``_evidence_dicts_for_fragment`` on cache-sourced per-frame catalog
    posteriors (Task 5: the only evidence source; no CSV-column fallback).
    """
    if per_row_probs.size == 0:
        return 0.0
    valid_mask = np.isfinite(per_row_probs).any(axis=1)
    if not valid_mask.any():
        return 0.0
    valid = per_row_probs[valid_mask]
    valid = np.where(np.isfinite(valid), valid, 0.0)
    n = valid.shape[0]
    n_labels = valid.shape[1]

    top1_idx = np.argmax(valid, axis=1)
    if n_labels >= 2:
        sorted_desc = np.sort(valid, axis=1)[:, ::-1]
        top1 = sorted_desc[:, 0]
        top2 = sorted_desc[:, 1]
    else:
        top1 = valid[:, 0]
        top2 = np.zeros(n, dtype=np.float64)

    counts = np.bincount(top1_idx, minlength=n_labels)
    agreement = float(counts.max()) / float(n)
    margin = float(np.mean(top1 - top2))
    return float(agreement * max(0.0, margin))


def _normalize_support_scores(
    known_labels: list[str],
    log_scores: dict[str, float],
) -> dict[str, float]:
    """Convert log-supports into a normalized per-label score distribution."""
    raw = np.array(
        [
            math.exp(float(log_scores[label])) if label in log_scores else 0.0
            for label in known_labels
        ],
        dtype=np.float64,
    )
    raw[~np.isfinite(raw)] = 0.0
    total = float(raw.sum())
    if total <= 1e-12:
        if not known_labels:
            return {}
        uniform = 1.0 / len(known_labels)
        return {label: uniform for label in known_labels}
    raw /= total
    return {label: float(raw[idx]) for idx, label in enumerate(known_labels)}


def _build_prior_log_scores(
    known_labels: list[str],
    online_label: str,
    online_confidence: float,
) -> dict[str, float]:
    """Build a soft prior over labels from the online label/confidence pair."""
    if online_label not in known_labels or not np.isfinite(online_confidence):
        return {label: 0.0 for label in known_labels}

    conf = float(np.clip(online_confidence, 1e-4, 1.0 - 1e-4))
    n_labels = len(known_labels)
    if n_labels <= 1:
        return {known_labels[0]: math.log(conf)} if known_labels else {}
    other = max((1.0 - conf) / (n_labels - 1), 1e-6)
    return {
        label: math.log(conf if label == online_label else other)
        for label in known_labels
    }


def _combined_support(
    frag_row: pd.Series, known_labels: list[str], params: dict[str, Any]
) -> dict[str, float]:
    """Normalised per-label support from the *informative* sources of one
    fragment, blended convexly: ``Σ w_s·log_s / Σ w_s``. With one informative
    source this is that source's geometric-mean posterior itself -- the
    ``FRAGMENT_*_WEIGHT`` knobs are relative source weights, never a softmax
    temperature (the 2026-08-27 audit found ``cnn_w=0.1`` flattening 0.99999
    evidence to 0.2 support).

    A source only counts as "present" when it is actually informative, not
    merely non-empty: e.g. an online-label prior for a label that is not in
    ``known_labels`` carries no information and must be excluded from the
    blend rather than included as an all-zero vector that would otherwise
    dilute a genuinely informative source's weighted average.
    """
    cnn_w = float(params.get("FRAGMENT_CNN_WEIGHT", 0.40))
    tag_w = float(params.get("FRAGMENT_TAG_WEIGHT", 0.15))
    prior_w = float(params.get("ONLINE_PRIOR_WEIGHT", 0.25))
    cnn_log = frag_row.get("CNNLogEvidence") or {}
    tag_log = frag_row.get("TagLogEvidence") or {}
    online_lbl = str(frag_row.get("OnlineLabel", "unknown"))
    online_conf = float(frag_row.get("OnlineConfidence", 0.0))
    sources: list[tuple[float, dict[str, float]]] = []
    if cnn_log and cnn_w > 0:
        sources.append((cnn_w, cnn_log))
    if tag_log and tag_w > 0:
        sources.append((tag_w, tag_log))
    if prior_w > 0 and online_lbl in known_labels and np.isfinite(online_conf):
        sources.append(
            (prior_w, _build_prior_log_scores(known_labels, online_lbl, online_conf))
        )
    if not sources:
        return _normalize_support_scores(known_labels, {})
    total_w = sum(w for w, _ in sources)
    combined = {
        label: sum(w * float(src.get(label, 0.0)) for w, src in sources) / total_w
        for label in known_labels
    }
    return _normalize_support_scores(known_labels, combined)


def detect_identity_changepoints(
    smoothed_by_traj: dict[Any, list[tuple[int, np.ndarray]]],
    catalog: IdentityCatalog,
    params: dict[str, Any],
) -> dict[Any, list[int]]:
    """Return {traj_id: [split_frame_indices]} using PELT on the smoothed
    per-frame identity posterior (Phase 5 Task 2's forward-backward
    smoothing output), not the ``CNN_*_Prob`` CSV columns.

    Each split_frame_index is the *inclusive end* (last FrameID) of a segment.
    ``build_fragments`` treats these as inclusive boundaries: segment k spans
    FrameIDs [split_indices[k-1]+1, split_indices[k]].
    Trajectories with no evidence or fewer than min_fragment_frames*2
    rows are returned with no splits.

    PELT model is read from params["PELT_MODEL"] (l1 / l2 / rbf; default rbf).

    Args:
        smoothed_by_traj: ``{TrajectoryID: [(FrameID, smoothed_log_probs), ...]}``,
            e.g. ``zip(frame_ids, smooth_trajectory_posteriors(...))`` per
            trajectory (see ``identity/smoothing.py``). Each ``smoothed_log_probs``
            is a normalized log-posterior over the full catalog (``unknown`` at
            index 0, known labels at 1..N). Sequences need not be pre-sorted by
            FrameID; this function sorts defensively.
        catalog: the identity catalog the smoothed posteriors are indexed
            against (used only to size the known-label slice of the signal).
        params: see module docstring; keys ``CHANGEPOINT_PENALTY``,
            ``MIN_FRAGMENT_FRAMES``, ``PELT_MODEL``.
    """
    try:
        import ruptures as rpt
    except ImportError:
        log.warning(
            "ruptures not installed; changepoint detection skipped — install ruptures>=1.1"
        )
        return {}

    penalty = float(params.get("CHANGEPOINT_PENALTY", 3.0))
    min_frames = int(params.get("MIN_FRAGMENT_FRAMES", 5))
    pelt_model = str(params.get("PELT_MODEL", "rbf")).lower()
    if pelt_model not in ("l1", "l2", "rbf"):
        pelt_model = "rbf"

    if len(catalog.labels) <= 1:
        return {}

    result: dict[Any, list[int]] = {}

    for traj_id, sequence in smoothed_by_traj.items():
        if len(sequence) < min_frames * 2:
            continue

        ordered = sorted(sequence, key=lambda item: item[0])
        frame_ids = np.array([frame_id for frame_id, _ in ordered])
        log_probs = np.stack([lp for _, lp in ordered]).astype(np.float64)
        # Guard against NaN/inf leaking in from upstream evidence/smoothing
        # (e.g. a degenerate all-zero-mass fusion) before the softmax below --
        # real cache-sourced wiring (Task 5) must never let a non-finite value
        # propagate into ruptures.
        log_probs = np.nan_to_num(log_probs, nan=-700.0, posinf=700.0, neginf=-700.0)

        # Known-label probabilities (exp of the smoothed log-posterior),
        # dropping the leading `unknown` column -- mirrors the old CNN_*_Prob
        # column set (one column per known identity label).
        probs = np.exp(log_probs - log_probs.max(axis=1, keepdims=True))
        probs /= np.clip(probs.sum(axis=1, keepdims=True), 1e-300, None)
        signal = probs[:, 1:]

        # The signal is a probability simplex slice in [0, 1]; the penalty is in
        # those units. Per-trajectory z-scoring made the penalty's units
        # trajectory-dependent and inflated float noise on constant posteriors
        # into unit-variance "signal" (660 splits on 128 tracks, 2026-08-27).

        try:
            splits = (
                rpt.Pelt(model=pelt_model, min_size=min_frames, jump=1)
                .fit(signal)
                .predict(pen=penalty)
            )
        except Exception as exc:
            log.warning("PELT failed for traj %s: %s", traj_id, exc)
            continue

        # ruptures returns end-of-segment indices (1-indexed frame position in
        # `ordered`). Convert to FrameID values (drop the final sentinel which
        # equals len).
        split_frames = [
            int(frame_ids[s - 1]) for s in splits[:-1] if s < len(frame_ids)
        ]
        if split_frames:
            result[traj_id] = split_frames

    return result


def split_trajectories_at_changepoints(
    df: pd.DataFrame,
    changepoints: dict[Any, list[int]],
    params: dict[str, Any],
) -> pd.DataFrame:
    """Split trajectories at PELT-detected changepoints, assigning new TrajectoryIDs.

    Each value in changepoints is a list of FrameID values that are the
    *inclusive end* of a segment (same convention as detect_identity_changepoints).
    Sub-segments shorter than MIN_FRAGMENT_FRAMES rows are merged into their
    neighbour (never dropped).
    OriginalTrajectoryID is set to the pre-split TrajectoryID on all rows.
    Trajectories with no changepoints pass through unchanged.
    """
    min_frames = int(params.get("MIN_FRAGMENT_FRAMES", 5))

    to_split = {tid: sorted(sfs) for tid, sfs in changepoints.items() if sfs}
    if not to_split:
        out = df.copy()
        if "OriginalTrajectoryID" not in out.columns:
            out["OriginalTrajectoryID"] = out["TrajectoryID"]
        return out

    out = df.copy()
    if "OriginalTrajectoryID" not in out.columns:
        out["OriginalTrajectoryID"] = out["TrajectoryID"]

    next_id = int(out["TrajectoryID"].max()) + 1

    unchanged = out[~out["TrajectoryID"].isin(to_split)].copy()
    parts: list[pd.DataFrame] = [unchanged]

    for traj_id, split_frames in to_split.items():
        grp = out[out["TrajectoryID"] == traj_id].sort_values("FrameID")
        if grp.empty:
            continue

        first_frame = int(grp["FrameID"].min())
        last_frame = int(grp["FrameID"].max())

        # Build inclusive (start, end) boundaries from split_frames.
        boundaries: list[tuple[int, int]] = []
        prev = first_frame
        for sf in split_frames:
            if prev <= sf < last_frame:
                boundaries.append((prev, sf))
                prev = sf + 1
        boundaries.append((prev, last_frame))

        segments: list[pd.DataFrame] = []
        for start_f, end_f in boundaries:
            seg = grp[(grp["FrameID"] >= start_f) & (grp["FrameID"] <= end_f)]
            if seg.empty:
                continue
            if len(seg) < min_frames and segments:
                segments[-1] = pd.concat([segments[-1], seg])  # fold into previous
            elif len(seg) < min_frames:
                segments.append(seg)  # leading remnant: fold into next
                continue
            elif segments and len(segments[-1]) < min_frames:
                segments[-1] = pd.concat([segments[-1], seg])
            else:
                segments.append(seg)
        for seg in segments:
            seg = seg.copy()
            seg["TrajectoryID"] = next_id
            seg["OriginalTrajectoryID"] = traj_id
            next_id += 1
            parts.append(seg)

    result = pd.concat(parts, ignore_index=True)
    result = result.sort_values(["TrajectoryID", "FrameID"], kind="stable").reset_index(
        drop=True
    )
    return result


def _spatial_score_for_fragment(
    frag: pd.Series,
    identity: str,
    schedule: dict[str, list[dict]],
    max_velocity: float,
    no_neighbor_score: float = 0.3,
    max_bridge_gap: int = 30,
) -> tuple[float, bool]:
    """Velocity-based spatial continuity score against nearest neighboring segment.

    Returns (score, has_neighbors).

    Scoring:
    - Effective gap = min(actual gap, ``max_bridge_gap``). Clamping prevents
      arbitrarily long temporal gaps from excusing arbitrarily large spatial
      jumps: beyond ``max_bridge_gap`` frames we have no evidence of the
      animal's path, so the bridge must still be explainable as if the gap
      were no longer than this window.
    - Implied velocity = dist / effective_gap (pixels per frame).
    - Hard veto: if velocity > max_velocity for any neighbor, return (0.0, True)
      immediately — physically implausible jump, caller should mark ineligible.
    - Otherwise: score = exp(-2 * (velocity / max_velocity)^2).
      This is a velocity-space Gaussian where max_velocity is one sigma; scores
      range from ~1.0 (stationary) through ~0.61 (half max) to ~0.14 (at max).
    - When has_neighbors is False the spatial score cannot be trusted;
      the caller should use evidence-only scoring.
    """
    t0 = int(frag["StartFrame"])
    t1 = int(frag["EndFrame"])
    x0, y0 = float(frag["StartX"]), float(frag["StartY"])
    x1, y1 = float(frag["EndX"]), float(frag["EndY"])
    segs = schedule.get(identity, [])
    term_scores: list[float] = []
    cap = max(1, int(max_bridge_gap))

    prior = max(
        (s for s in segs if s["end_frame"] < t0),
        key=lambda s: s["end_frame"],
        default=None,
    )
    if prior and all(
        math.isfinite(v) for v in [x0, y0, prior["end_X"], prior["end_Y"]]
    ):
        gap = max(1, t0 - prior["end_frame"])
        effective_gap = min(gap, cap)
        dist = math.hypot(x0 - prior["end_X"], y0 - prior["end_Y"])
        velocity = dist / effective_gap
        if velocity > max_velocity:
            return 0.0, True  # physically implausible — hard veto
        term_scores.append(math.exp(-2.0 * (velocity / max_velocity) ** 2))

    following = min(
        (s for s in segs if s["start_frame"] > t1),
        key=lambda s: s["start_frame"],
        default=None,
    )
    if following and all(
        math.isfinite(v) for v in [x1, y1, following["start_X"], following["start_Y"]]
    ):
        gap = max(1, following["start_frame"] - t1)
        effective_gap = min(gap, cap)
        dist = math.hypot(x1 - following["start_X"], y1 - following["start_Y"])
        velocity = dist / effective_gap
        if velocity > max_velocity:
            return 0.0, True  # physically implausible — hard veto
        term_scores.append(math.exp(-2.0 * (velocity / max_velocity) ** 2))

    if term_scores:
        return float(np.mean(term_scores)), True
    return no_neighbor_score, False


def _seg_from_row(row: pd.Series) -> dict:
    return {
        "start_frame": int(row["StartFrame"]),
        "end_frame": int(row["EndFrame"]),
        "start_X": float(row["StartX"]),
        "start_Y": float(row["StartY"]),
        "end_X": float(row["EndX"]),
        "end_Y": float(row["EndY"]),
    }


def _evidence_mass(duration: float, top_support: float) -> float:
    """Descending-mass seeding key: duration x top per-label support.

    A long, confidently-evidenced fragment is seeded (and rescued) before a
    short, weakly-evidenced one, so it claims its label first rather than
    losing a race to whichever short fragment the old component-Hungarian
    step happened to grant a slot to.
    """
    return float(duration) * float(top_support)


def _iterative_assign(
    frags: pd.DataFrame,
    known_labels: list[str],
    params: dict[str, Any],
) -> dict[int, str | None]:
    """Mass-first seeding + doubt-ordered refinement with exact-objective
    multi-blocker displacement.

    Replaces the old component-Hungarian base assignment
    (``_base_assignment_via_substrate``, deleted): on any multi-animal clip
    the temporal-overlap connected components collapse to one giant
    component (every fragment overlaps some other fragment transitively),
    so the base step degenerated into every fragment competing for every
    label as if simultaneously visible, gated at a display threshold
    nothing could clear. This function never routes through that solver.

    Algorithm:
    1. **Mass-first seeding**: fragments are visited in descending
       ``_evidence_mass`` (duration x top support) order and each greedily
       claims its best free, unvetoed label. A label is a *candidate* for a
       fragment at all only if its normalised support clears
       ``FRAGMENT_MIN_SUPPORT`` (an absolute posterior floor) and it ranks
       in the fragment's top ``FRAGMENT_TOP_K`` labels by support. Seeding
       long/confident fragments first means a 700-frame 0.999-evidence
       track claims its label before a swarm of short noisy fragments ever
       gets a turn — the inverse of the old doubt-ordered-only walk, which
       let short fragments grab labels first via the (broken) base step.
    2. **Doubt-ordered refinement**: fragments are revisited in descending
       doubt (low stability x short length x poor spatial fit, plus an
       Unknown bonus). A direct flip to a free/better label is accepted
       when it clears ``ASSIGNMENT_MARGIN_THRESHOLD``. When every
       candidate is blocked, ``_try_displacement`` evaluates evicting the
       blocker(s) (up to ``FRAGMENT_MAX_BLOCKERS``) and re-homing them to
       their own best remaining label.
    3. **Unknown rescue**: any fragment still unassigned after refinement
       (e.g. it never won its label during seeding) gets one more direct or
       displacing attempt, again in descending mass order.

    Termination argument (multi-blocker displacement): define the exact
    objective as ``sum(_score(j, current[j]) for j touched)`` over the
    finite set of fragments whose assignment or scoring could be affected
    by a move (the mover, every blocker it displaces, and everyone
    scheduled under the label(s) involved). ``_try_displacement`` accepts a
    move iff this exact objective, evaluated over that touched set both
    before and after, rises by at least ``monotone_eps`` (floored at
    ``1e-3`` so a caller cannot request ASSIGNMENT_MARGIN_THRESHOLD=0 and
    accept infinitesimal or floating-point-noise "improvements"). Every
    per-fragment score is bounded in ``[0, 1]`` (``_score`` returns
    ``evidence * spatial_factor * length_factor`` — a product of three
    terms each in ``[0, 1]``), so the whole-schedule objective
    ``sum(_score(i, current[i]) for i in range(n))`` is bounded above by
    ``n``. Since every accepted move strictly increases that bounded
    monotone quantity by a fixed floor, only finitely many
    (``<= n / monotone_eps``) moves can be accepted in total — the
    refinement pass cannot cycle or loop forever on its own. Simple flips
    (no blocker) are gated the same way as single-fragment moves within
    that same bound. ``FRAGMENT_MAX_PASSES`` remains a hard cap regardless,
    since a pass can also do no-op work (recomputing without flipping).

    Returns ``{frag_index: assigned_label_or_None}`` (None means Unknown).
    """
    length_w = min(1.0, max(0.0, float(params.get("FRAGMENT_LENGTH_WEIGHT", 0.60))))
    max_vel = float(params.get("MAX_VELOCITY_BREAK", 50.0))
    max_bridge_gap = max(1, int(params.get("MAX_BRIDGE_GAP_FRAMES", 30)))
    no_neighbor_score = float(params.get("SPATIAL_NO_NEIGHBOR_SCORE", 0.3))
    spatial_veto = float(params.get("FRAGMENT_SPATIAL_VETO_THRESHOLD", 0.05))
    monotone_eps = max(1e-3, float(params.get("ASSIGNMENT_MARGIN_THRESHOLD", 0.10)))
    top_k = max(1, int(params.get("FRAGMENT_TOP_K", 3)))
    max_passes = max(1, int(params.get("FRAGMENT_MAX_PASSES", 10)))
    min_support = float(params.get("FRAGMENT_MIN_SUPPORT", 0.5))
    max_blockers = max(1, int(params.get("FRAGMENT_MAX_BLOCKERS", 4)))
    unknown_doubt_bonus = float(params.get("FRAGMENT_UNKNOWN_DOUBT_BONUS", 0.5))

    n = len(frags)
    if n == 0:
        return {}

    rows = [frags.iloc[i] for i in range(n)]
    durations = np.array(
        [max(1, int(r["EndFrame"]) - int(r["StartFrame"]) + 1) for r in rows],
        dtype=np.float64,
    )
    log_max = math.log1p(float(durations.max()))
    length_scales = (
        np.log1p(durations) / log_max
        if log_max > 1e-9
        else np.ones(n, dtype=np.float64)
    )
    length_factors = 1.0 - length_w * (1.0 - length_scales)
    supports = [_combined_support(r, known_labels, params) for r in rows]
    stabilities = np.array(
        [float(r.get("Stability", 0.0)) for r in rows], dtype=np.float64
    )
    segs = [_seg_from_row(r) for r in rows]
    candidates_of: list[list[str]] = [
        [
            lbl
            for lbl in sorted(known_labels, key=lambda lbl: -supports[i].get(lbl, 0.0))[
                :top_k
            ]
            if supports[i].get(lbl, 0.0) >= min_support
        ]
        for i in range(n)
    ]

    current: list[str | None] = [None] * n
    schedule: dict[str, list[int]] = {lbl: [] for lbl in known_labels}

    def _overlaps(i: int, j: int) -> bool:
        return int(rows[i]["StartFrame"]) <= int(rows[j]["EndFrame"]) and int(
            rows[j]["StartFrame"]
        ) <= int(rows[i]["EndFrame"])

    def _blockers(i: int, label: str) -> list[int]:
        return [j for j in schedule[label] if j != i and _overlaps(i, j)]

    def _score(i: int, label: str | None) -> float:
        """Score of fragment i under ``label`` given the CURRENT schedule (i
        excluded). Returns 0.0 when unassigned, blocked (temporal-overlap
        collision), or spatially vetoed."""
        if label is None:
            return 0.0
        if _blockers(i, label):
            return 0.0
        sched = {label: [segs[j] for j in schedule[label] if j != i]}
        spatial_s, has_nb = _spatial_score_for_fragment(
            rows[i], label, sched, max_vel, no_neighbor_score, max_bridge_gap
        )
        if has_nb and spatial_s < spatial_veto:
            return 0.0
        evidence = float(supports[i].get(label, 0.0))
        raw = evidence * spatial_s if has_nb else evidence
        return float(raw * length_factors[i])

    def _commit(i: int, label: str | None) -> None:
        cur = current[i]
        if cur is not None:
            schedule[cur].remove(i)
        if label is not None:
            schedule[label].append(i)
        current[i] = label

    def _objective(touched: set[int]) -> float:
        return sum(_score(j, current[j]) for j in touched)

    def _affected(label_a: str | None, label_b: str | None, extra) -> set[int]:
        s = set(extra)
        for lbl in (label_a, label_b):
            if lbl is not None:
                s.update(schedule[lbl])
        return s

    def _best_alternative(j: int, exclude: str | None) -> tuple[float, str | None]:
        best_s, best_l = 0.0, None
        for c in candidates_of[j]:
            if c == exclude:
                continue
            s = _score(j, c)
            if s > best_s:
                best_s, best_l = s, c
        return best_s, best_l

    def _try_displacement(i: int, c: str) -> bool:
        """Tentatively give ``c`` to fragment i, evicting and re-homing its
        blocker(s). Accept iff the exact objective over every affected
        fragment (i, each evicted blocker, and every fragment scheduled
        under any label touched by the move) rises by >= monotone_eps.
        Reverts to the pre-move state otherwise. See the docstring above
        for the termination argument this gate is load-bearing for.
        """
        blockers = _blockers(i, c)
        if not blockers or len(blockers) > max_blockers:
            return False
        before_assign = {j: current[j] for j in blockers}
        before_assign[i] = current[i]
        touched = _affected(current[i], c, blockers)
        for j in blockers:
            if current[j] is not None:
                touched.update(schedule[current[j]])
        j_before = _objective(touched)

        for j in blockers:
            _commit(j, None)
        _commit(i, c)
        new_labels: dict[int, str | None] = {}
        for j in blockers:
            _, alt = _best_alternative(j, c)
            new_labels[j] = alt
            _commit(j, alt)
            if alt is not None:
                touched.update(schedule[alt])
        j_after = _objective(touched)

        if j_after - j_before >= monotone_eps and _score(i, c) > 0.0:
            return True

        # Revert: undo the re-homing, then restore the original assignment.
        for j in new_labels:
            _commit(j, None)
        _commit(i, None)
        for j, lbl in before_assign.items():
            _commit(j, lbl)
        return False

    # --- 1. mass-first seeding ---
    order = sorted(
        range(n),
        key=lambda i: -_evidence_mass(
            durations[i], max(supports[i].values()) if supports[i] else 0.0
        ),
    )
    for i in order:
        for c in candidates_of[i]:
            if _score(i, c) > 0.0:
                _commit(i, c)
                break

    # --- 2. doubt-ordered refinement ---
    def _doubt(i: int) -> float:
        s_norm = 1.0 - float(stabilities[i])
        l_norm = 1.0 - float(length_scales[i])
        if current[i] is None:
            return s_norm * l_norm + unknown_doubt_bonus
        sched = {current[i]: [segs[j] for j in schedule[current[i]] if j != i]}
        spatial_s, has_nb = _spatial_score_for_fragment(
            rows[i], current[i], sched, max_vel, no_neighbor_score, max_bridge_gap
        )
        fit = float(spatial_s) if has_nb else no_neighbor_score
        return s_norm * l_norm * (1.0 - fit)

    for pass_idx in range(max_passes):
        flips = 0
        for i in sorted(range(n), key=lambda i: -_doubt(i)):
            cur = current[i]
            cur_s = _score(i, cur) if cur is not None else 0.0
            best_l, best_s = cur, cur_s
            for c in candidates_of[i]:
                if c == cur:
                    continue
                s = _score(i, c)
                if s > 0.0 and s - cur_s >= monotone_eps and s > best_s:
                    best_l, best_s = c, s
            if best_l != cur:
                _commit(i, best_l)
                flips += 1
                continue
            for c in candidates_of[i]:
                if c != cur and _score(i, c) == 0.0 and _try_displacement(i, c):
                    flips += 1
                    break
        log.debug("iterative fragment solver pass %d: %d flips", pass_idx + 1, flips)
        if flips == 0:
            break
    else:
        log.warning(
            "Iterative fragment solver hit FRAGMENT_MAX_PASSES (%d) without convergence.",
            max_passes,
        )

    # --- 3. unknown rescue by descending mass (may displace) ---
    for i in sorted(
        (i for i in range(n) if current[i] is None),
        key=lambda i: -_evidence_mass(
            durations[i], max(supports[i].values()) if supports[i] else 0.0
        ),
    ):
        placed = False
        for c in candidates_of[i]:
            if _score(i, c) > 0.0:
                _commit(i, c)
                placed = True
                break
        if not placed:
            for c in candidates_of[i]:
                if _try_displacement(i, c):
                    break

    return {i: current[i] for i in range(n)}


def _evidence_dicts_for_fragment(
    known_labels: list[str],
    sequence: list[tuple[int, np.ndarray]],
) -> tuple[dict[str, float], dict[str, float], float]:
    """Build ``(MeanCNNProbs, CNNLogEvidence, Stability)`` for one fragment from
    cache-sourced (smoothed) per-frame catalog posteriors instead of the
    reconstructed ``CNN_*_Prob`` CSV columns.

    ``sequence`` is this fragment's slice of a Task-5-resolved
    ``evidence_by_traj`` sequence (already restricted to this fragment's own
    FrameID set by the caller): ``[(FrameID, catalog_log_probs), ...]``, each
    ``catalog_log_probs`` a normalized log-posterior over the full catalog
    (unknown at index 0). Rows with no matched cache evidence contribute
    nothing (frames simply absent from ``sequence``) -- "no evidence, no
    belief", matching Task 1/2's join semantics.

    ``CNNLogEvidence``'s convention (mean of per-row log-probabilities, i.e.
    a geometric mean, not the log of an arithmetic mean) is preserved from
    the CSV-based reconstruction so ``_iterative_assign``'s downstream
    weighted blend is unaffected by the evidence source.
    """
    if not sequence:
        return {}, {}, 0.0

    log_probs = np.stack([lp for _, lp in sequence]).astype(np.float64)
    known_log = log_probs[:, 1:]  # drop the leading `unknown` column
    known_probs = np.exp(known_log)

    mean_probs = {
        label: float(np.mean(known_probs[:, idx]))
        for idx, label in enumerate(known_labels)
    }
    cnn_log_scores = {
        label: float(np.mean(known_log[:, idx]))
        for idx, label in enumerate(known_labels)
    }
    stability = _fragment_stability(known_probs)
    return mean_probs, cnn_log_scores, stability


def _build_traj_summaries(
    df: pd.DataFrame,
    catalog: IdentityCatalog,
    evidence_by_traj: dict[Any, list[tuple[int, np.ndarray]]] | None = None,
) -> pd.DataFrame:
    """Build a per-trajectory summary DataFrame consumed by the iterative solver.

    Columns: TrajectoryID, StartFrame, EndFrame, StartX, StartY, EndX, EndY,
    MeanCNNProbs (dict), MeanTagProbs (dict), CNNLogEvidence (dict),
    TagLogEvidence (dict), Stability (float), OnlineLabel, OnlineConfidence.

    Identity Phase 7 Task 5 (clean-break retirement): this solver is
    CACHE-ONLY. ``evidence_by_traj`` (``{OriginalTrajectoryID/TrajectoryID:
    [(FrameID, catalog_log_probs), ...]}``, the cache-sourced + optionally
    forward-backward-smoothed sequences ``run_fragment_solver`` builds via
    ``identity/smoothing.py``) is the *only* evidence source; each
    fragment's ``MeanCNNProbs``/``CNNLogEvidence``/``Stability`` are derived
    from it, restricted to the fragment's own FrameID range. There is no
    ``CNN_*_Prob``/``DetectedTag*`` wide-CSV column reconstruction fallback
    -- the live pipeline (``postprocess_df.py``) always opens the evidence
    cache Phase 3 writes unconditionally and passes ``evidence_by_traj``, so
    that fallback was dead in production. When ``evidence_by_traj`` is
    ``None``/empty, or a specific trajectory has no matched cache evidence,
    this produces the documented no-sidecar degrade: empty evidence dicts
    (``MeanCNNProbs``/``CNNLogEvidence`` = ``{}``, ``Stability`` = 0.0) --
    "no evidence, no belief" (Task 1's convention) -- rather than a
    reconstructed guess. ``TagLogEvidence``/``MeanTagProbs`` are always
    empty now, since ``load_trajectory_evidence`` already fuses every
    source (CNN + tag) into one catalog posterior per detection -- there is
    nothing left for a separate tag term to add.

    ``OnlineLabel``/``OnlineConfidence`` are always read from the
    ``IdentityFinalLabel``/``IdentityFinalConfidence`` CSV columns when
    present (an OPTIONAL weak prior from a prior resolution pass -- never a
    required input; a fragment with no such label defaults to "unknown"/0.0
    confidence and the cache evidence alone still drives assignment, which
    is the honesty fix).
    """
    known_labels = list(catalog.labels[1:])
    has_orig_col = "OriginalTrajectoryID" in df.columns
    rows: list[dict] = []

    for traj_id, grp in df.groupby("TrajectoryID", sort=False):
        grp_sorted = grp.sort_values("FrameID").reset_index(drop=True)
        start_f = int(grp_sorted["FrameID"].iloc[0])
        end_f = int(grp_sorted["FrameID"].iloc[-1])

        valid_xy = (
            grp_sorted[grp_sorted["X"].notna() & grp_sorted["Y"].notna()].sort_values(
                "FrameID"
            )
            if "X" in grp_sorted.columns and "Y" in grp_sorted.columns
            else pd.DataFrame()
        )
        if not valid_xy.empty:
            sx = float(valid_xy.iloc[0]["X"])
            sy = float(valid_xy.iloc[0]["Y"])
            ex = float(valid_xy.iloc[-1]["X"])
            ey = float(valid_xy.iloc[-1]["Y"])
        else:
            sx = sy = ex = ey = math.nan

        fragment_evidence: list[tuple[int, np.ndarray]] = []
        if evidence_by_traj:
            orig_id = (
                grp_sorted["OriginalTrajectoryID"].iloc[0] if has_orig_col else traj_id
            )
            frame_set = {int(f) for f in grp_sorted["FrameID"]}
            fragment_evidence = [
                (f, lp) for f, lp in evidence_by_traj.get(orig_id, []) if f in frame_set
            ]

        if fragment_evidence:
            mean_probs, cnn_log_scores, stability = _evidence_dicts_for_fragment(
                known_labels, fragment_evidence
            )
        else:
            # No cache evidence for this trajectory (or none supplied at
            # all) -- "no evidence, no belief" (Task 1 convention). No
            # CSV-column reconstruction fallback.
            mean_probs, cnn_log_scores, stability = {}, {}, 0.0
        tag_probs, tag_log_scores = {}, {}

        label_col = grp_sorted.get(
            _LABEL_COL, pd.Series("unknown", index=grp_sorted.index, dtype=object)
        )
        unknown_mask = label_col.isna() | label_col.astype(str).str.strip().isin(
            _UNKNOWN_VALUES
        )
        known_rows = grp_sorted[~unknown_mask]
        if not known_rows.empty:
            online_label = str(known_rows[_LABEL_COL].astype(str).mode().iloc[0])
            if _CONF_COL in known_rows.columns:
                conf_vals = pd.to_numeric(known_rows[_CONF_COL], errors="coerce")
                online_conf = (
                    float(np.nanmean(conf_vals.values))
                    if conf_vals.notna().any()
                    else 0.0
                )
            else:
                online_conf = 0.0
        else:
            online_label = "unknown"
            online_conf = 0.0

        rows.append(
            {
                "TrajectoryID": traj_id,
                "StartFrame": start_f,
                "EndFrame": end_f,
                "StartX": sx,
                "StartY": sy,
                "EndX": ex,
                "EndY": ey,
                "MeanCNNProbs": mean_probs,
                "MeanTagProbs": tag_probs,
                "CNNLogEvidence": cnn_log_scores,
                "TagLogEvidence": tag_log_scores,
                "Stability": stability,
                "OnlineLabel": online_label,
                "OnlineConfidence": online_conf,
            }
        )

    if not rows:
        return pd.DataFrame(
            columns=[
                "TrajectoryID",
                "StartFrame",
                "EndFrame",
                "StartX",
                "StartY",
                "EndX",
                "EndY",
                "MeanCNNProbs",
                "MeanTagProbs",
                "CNNLogEvidence",
                "TagLogEvidence",
                "Stability",
                "OnlineLabel",
                "OnlineConfidence",
            ]
        )
    return pd.DataFrame(rows)


def _ensure_final_columns(out: pd.DataFrame) -> pd.DataFrame:
    """Create/coerce the ``IdentityFinal*`` columns to writable dtypes in-place.

    Shared by ``solve_global_assignment`` and the evidence-quality breaker's
    "no evidence, no belief" bypass in ``run_fragment_solver`` -- both need
    the same object-dtype coercion so string label writes don't raise a
    pandas ``LossySetitemError`` on an all-NaN float64 column. Mutates and
    returns ``out``.
    """
    if C.FINAL_LABEL not in out.columns:
        # object dtype (not the float64 a bare `np.nan` column init would get)
        # -- otherwise the string label writes below raise a pandas
        # LossySetitemError.
        out[C.FINAL_LABEL] = pd.Series(
            [np.nan] * len(out), index=out.index, dtype=object
        )
    elif out[C.FINAL_LABEL].dtype != object:
        # The honesty fix's target scenario (ENABLE_IDENTITY_IN_TRACKING off)
        # leaves this column present but all-NaN float64 (no prior pass wrote
        # strings). Writing a string label into a float64 column raises a
        # pandas LossySetitemError (pandas>=3), which the caller's broad
        # except would swallow, silently reverting the solver to "results
        # unchanged" -- i.e. exactly re-breaking the honesty fix. Coerce to
        # object so the per-trajectory writes below can land.
        out[C.FINAL_LABEL] = out[C.FINAL_LABEL].astype(object)
    if C.FINAL_SOURCE not in out.columns:
        out[C.FINAL_SOURCE] = pd.Series(
            [C.IdentityFinalSource.NONE] * len(out), index=out.index, dtype=object
        )
    elif out[C.FINAL_SOURCE].dtype != object:
        out[C.FINAL_SOURCE] = out[C.FINAL_SOURCE].astype(object)
    out[C.FINAL_SOURCE] = C.normalize_final_source_series(out[C.FINAL_SOURCE])
    if C.FINAL_ID not in out.columns:
        out[C.FINAL_ID] = np.nan
    if C.FINAL_CONFIDENCE not in out.columns:
        out[C.FINAL_CONFIDENCE] = np.nan
    if C.FINAL_FRAGMENT_SCORE not in out.columns:
        out[C.FINAL_FRAGMENT_SCORE] = np.nan
    if C.FINAL_CONFLICT_RESOLVED not in out.columns:
        out[C.FINAL_CONFLICT_RESOLVED] = False
    return out


def _write_no_evidence_outcome(
    df: pd.DataFrame, catalog: IdentityCatalog
) -> pd.DataFrame:
    """Write the "no evidence, no belief" outcome to every row of ``df``.

    Used by ``run_fragment_solver`` when the evidence-quality breaker trips:
    per the design spec (section 3.4), the solver must not commit any label
    -- not even via ``_iterative_assign``'s Unknown-rescue pass, which would
    otherwise assign a uniform-low-confidence label to every fragment even
    with zero informative evidence. Mirrors the exact per-row values
    ``solve_global_assignment`` writes when it explicitly decides a fragment
    is Unknown, applied unconditionally to every row since ALL rows share
    the same outcome when the breaker trips. Returns a modified copy of df.
    """
    out = df.copy()
    _ensure_final_columns(out)
    out[C.FINAL_LABEL] = "unknown"
    out[C.FINAL_ID] = catalog.unknown_index
    out[C.FINAL_CONFIDENCE] = 0.0
    out[C.FINAL_FRAGMENT_SCORE] = 0.0
    out[C.FINAL_SOURCE] = C.IdentityFinalSource.NONE
    return out


def solve_global_assignment(
    df: pd.DataFrame,
    catalog: IdentityCatalog,
    params: dict[str, Any],
    evidence_by_traj: dict[Any, list[tuple[int, np.ndarray]]] | None = None,
) -> pd.DataFrame:
    """Assign one identity label per trajectory via the iterative solver.

    Builds per-trajectory summaries internally, runs ``_iterative_assign`` with
    spatial continuity + CNN/tag evidence + online-label prior + per-fragment
    stability, then writes the ``IdentityFinal*`` family (``IdentityFinalLabel``,
    ``IdentityFinalID``, ``IdentityFinalConfidence``, ``IdentityFinalFragmentScore``,
    ``IdentityFinalSource``) back into every row of each trajectory -- a
    provenance-explicit record of what THIS (offline/post-hoc) solver decided.
    This function NEVER writes any ``IdentityRealtime*`` column: the offline
    solver's decision is cache-sourced when ``evidence_by_traj`` is given, so
    it is populated even when realtime identity never ran (the honesty fix's
    visible proof surface), and it never clobbers whatever the realtime
    decoder separately wrote. Returns a modified copy of df.

    ``evidence_by_traj`` (cache-sourced; the only evidence input -- Phase 7
    Task 5 deleted the ``CNN_*_Prob``/``DetectedTag*`` CSV-column
    reconstruction fallback): ``{OriginalTrajectoryID/TrajectoryID:
    [(FrameID, catalog_log_probs), ...]}`` -- see ``_build_traj_summaries``.
    When ``None``/empty (no evidence sidecar available), every trajectory
    degrades to the documented no-sidecar outcome (empty evidence; see
    ``_build_traj_summaries``), not a CSV-reconstructed guess.
    """
    known_labels = list(catalog.labels[1:])
    if not known_labels or df is None or df.empty:
        return df if df is not None else pd.DataFrame()

    traj_summaries = _build_traj_summaries(df, catalog, evidence_by_traj)
    if traj_summaries.empty:
        return df

    summaries = traj_summaries.reset_index(drop=True)
    n_trajs = len(summaries)

    assigned = _iterative_assign(summaries, known_labels, params)

    # Per-fragment final score for committed labels (recomputed with the final
    # schedule so the value reflects the converged spatial configuration).
    final_schedule: dict[str, list[dict]] = {}
    for i in range(n_trajs):
        lbl = assigned.get(i)
        if lbl is None:
            continue
        final_schedule.setdefault(lbl, []).append(_seg_from_row(summaries.iloc[i]))

    length_w = min(1.0, max(0.0, float(params.get("FRAGMENT_LENGTH_WEIGHT", 0.60))))
    max_vel = float(params.get("MAX_VELOCITY_BREAK", 50.0))
    max_bridge_gap = max(1, int(params.get("MAX_BRIDGE_GAP_FRAMES", 30)))
    no_neighbor_score = float(params.get("SPATIAL_NO_NEIGHBOR_SCORE", 0.3))

    durations = np.array(
        [
            max(1, int(r["EndFrame"]) - int(r["StartFrame"]) + 1)
            for _, r in summaries.iterrows()
        ],
        dtype=np.float64,
    )
    log_max = math.log1p(float(durations.max()))
    length_scales = (
        np.log1p(durations) / log_max
        if log_max > 1e-9
        else np.ones(n_trajs, dtype=np.float64)
    )
    length_factors = 1.0 - length_w * (1.0 - length_scales)

    assigned_scores: list[float] = []
    for i in range(n_trajs):
        lbl = assigned.get(i)
        if lbl is None or lbl not in known_labels:
            assigned_scores.append(0.0)
            continue
        support = _combined_support(summaries.iloc[i], known_labels, params)
        sched_minus_self = {
            lbl: [
                seg
                for j, seg in enumerate(final_schedule.get(lbl, []))
                if not (
                    seg["start_frame"] == int(summaries.iloc[i]["StartFrame"])
                    and seg["end_frame"] == int(summaries.iloc[i]["EndFrame"])
                )
            ]
        }
        spatial_s, has_neighbors = _spatial_score_for_fragment(
            summaries.iloc[i],
            lbl,
            sched_minus_self,
            max_vel,
            no_neighbor_score,
            max_bridge_gap,
        )
        evidence = float(support.get(lbl, 0.0))
        raw = evidence * spatial_s if has_neighbors else evidence
        assigned_scores.append(float(raw * float(length_factors[i])))

    # Write one label per trajectory back to every row -- the IdentityFinal*
    # family only. This function must NEVER write an IdentityRealtime* column.
    out = df.copy()
    _ensure_final_columns(out)

    for i in range(n_trajs):
        label = assigned.get(i)
        traj_id = summaries.iloc[i]["TrajectoryID"]
        mask = out["TrajectoryID"] == traj_id
        if label is None or label in _UNKNOWN_VALUES:
            # The solver explicitly chose Unknown for this fragment (e.g. its
            # spatial fit under every feasible label fails the veto). Record
            # the solver's decision, but with no attributable source (no
            # identity was actually resolved for this fragment).
            out.loc[mask, C.FINAL_LABEL] = "unknown"
            out.loc[mask, C.FINAL_ID] = catalog.unknown_index
            out.loc[mask, C.FINAL_CONFIDENCE] = 0.0
            out.loc[mask, C.FINAL_FRAGMENT_SCORE] = 0.0
            out.loc[mask, C.FINAL_SOURCE] = C.IdentityFinalSource.NONE
            continue
        try:
            catalog_index = catalog.index_of(label)
        except KeyError:
            catalog_index = np.nan
        out.loc[mask, C.FINAL_LABEL] = label
        out.loc[mask, C.FINAL_ID] = catalog_index
        out.loc[mask, C.FINAL_CONFIDENCE] = assigned_scores[i]
        out.loc[mask, C.FINAL_FRAGMENT_SCORE] = assigned_scores[i]
        out.loc[mask, C.FINAL_SOURCE] = C.IdentityFinalSource.OFFLINE

    return out


def _annotate_smoothed_labels(
    df: pd.DataFrame,
    smoothed_by_traj: dict[Any, list[tuple[int, np.ndarray]]],
    catalog: IdentityCatalog,
    params: dict[str, Any],
) -> pd.DataFrame:
    """Write per-row ``IdentityFinalSmoothedLabel``/``IdentityFinalSmoothedConfidence``
    from the forward-backward-smoothed per-frame posterior (Task 2), joined
    on ``(OriginalTrajectoryID or TrajectoryID, FrameID)``.

    This is a **record** of the cache-evidence forward-backward posterior,
    independent of (and written before) the fragment solver's per-fragment
    committed decision (``IdentityFinalLabel``): every row with cache
    evidence gets the argmax known label and its raw posterior, ungated by
    any display threshold. ``unknown``/0.0 means no cache evidence joined
    this row (e.g. crop-pass rows with no ``DetectionID``) -- never a
    thresholded blank.
    """
    out = df.copy()
    # object dtype -- see the FINAL_LABEL note in solve_global_assignment for
    # why a bare `""`/string column init can still land as float64 in some
    # pandas construction paths and raise LossySetitemError on write.
    out[C.FINAL_SMOOTHED_LABEL] = pd.Series(
        ["unknown"] * len(out), index=out.index, dtype=object
    )
    out[C.FINAL_SMOOTHED_CONFIDENCE] = 0.0

    id_col = (
        "OriginalTrajectoryID"
        if "OriginalTrajectoryID" in out.columns
        else "TrajectoryID"
    )

    for traj_id, sequence in smoothed_by_traj.items():
        if not sequence:
            continue
        mask = out[id_col] == traj_id
        if not mask.any():
            continue
        frame_ids = [f for f, _ in sequence]
        log_probs = [lp for _, lp in sequence]
        labels_confs = smoothed_label_and_conf(
            log_probs, catalog, display_threshold=None
        )
        by_frame = dict(zip(frame_ids, labels_confs))

        sub_frames = out.loc[mask, "FrameID"]
        for row_idx, frame_id in sub_frames.items():
            hit = by_frame.get(int(frame_id))
            if hit is None:
                continue
            label, conf = hit
            out.at[row_idx, C.FINAL_SMOOTHED_LABEL] = label
            out.at[row_idx, C.FINAL_SMOOTHED_CONFIDENCE] = conf

    return out


def merge_same_label_neighbours(
    df: pd.DataFrame, did_split: bool, label_col: str = C.FINAL_LABEL
) -> pd.DataFrame:
    """Undo solver cuts that changed nothing: consecutive fragments of the same
    ``OriginalTrajectoryID`` whose final labels agree (unknown == unknown too)
    are re-joined under the earlier fragment's TrajectoryID. Relink cannot do
    this (it rejects gap == 0), so the solver owns it.

    ``did_split`` must be True only when THIS call's ``run_fragment_solver``
    invocation actually performed a PELT split (Finding M3). The presence of
    an ``OriginalTrajectoryID`` column alone is not sufficient evidence: a
    trajectory can carry that column from a prior pass (or from
    ``split_trajectories_at_changepoints`` passing rows through unchanged
    when no changepoints were found) with no split having happened on this
    call, and merging in that case could join trajectories this call never
    split.
    """
    if (
        not did_split
        or "OriginalTrajectoryID" not in df.columns
        or label_col not in df.columns
    ):
        return df
    out = df.copy()
    spans = (
        out.groupby("TrajectoryID")
        .agg(
            orig=("OriginalTrajectoryID", "first"),
            start=("FrameID", "min"),
            end=("FrameID", "max"),
            label=(label_col, "first"),
        )
        .sort_values(["orig", "start"])
    )
    remap: dict = {}
    prev_tid = prev_orig = prev_label = None
    prev_end = None
    for tid, r in spans.iterrows():
        same = (
            prev_tid is not None
            and r.orig == prev_orig
            and r.start == prev_end + 1
            and str(r.label) == str(prev_label)
        )
        if same:
            remap[tid] = remap.get(prev_tid, prev_tid)
        else:
            prev_tid, prev_orig, prev_label = tid, r.orig, r.label
        prev_end = r.end
    if remap:
        out["TrajectoryID"] = out["TrajectoryID"].map(lambda t: remap.get(t, t))
    return out


EVIDENCE_CONF_LEVEL = 0.5  # a detection "knows" its label at this posterior
EVIDENCE_MIN_CONF_FRAC = 0.10  # <10% confident detections → source is uninformative
EVIDENCE_MIN_DIVERSITY = 0.30  # distinct labels/frame vs achievable → collapsed source


@dataclass(frozen=True)
class EvidenceQuality:
    conf_frac: float
    diversity: float
    n_frames: int
    ok: bool


def assess_evidence_quality(
    raw_evidence: dict[Any, list[tuple[int, np.ndarray]]], catalog: IdentityCatalog
) -> EvidenceQuality:
    """Cheap, source-level sanity check on RAW (unsmoothed) per-frame evidence.

    Guards against a source whose evidence is simply uninformative -- a
    badly miscalibrated classifier, wrong model loaded, mismatched
    preprocessing, etc. -- driving trajectory restructuring downstream
    (PELT splitting + fragment reassignment). Computed on the raw,
    unsmoothed per-detection posteriors so a broken source can't be masked
    by forward-backward smoothing before the check runs.

    conf_frac: fraction of (frame, detection) rows whose max KNOWN posterior
        (excluding the ``unknown`` slot) is >= ``EVIDENCE_CONF_LEVEL``.
    diversity: mean over frames of distinct argmax labels / min(#known
        labels, #detections in that frame). Low when the source keeps
        pointing at the same one or two labels regardless of which
        detection it's looking at.
    Both were ~0 / 0.3 on the 2026-08-27 failure (a mis-preprocessed
    classifier) and >0.8 / >0.7 with correct preprocessing.
    """
    per_frame: dict[int, list[int]] = {}
    conf = 0
    total = 0
    for sequence in raw_evidence.values():
        for frame_id, lp in sequence:
            p = np.exp(lp - np.logaddexp.reduce(lp))
            known = p[1:]
            total += 1
            if known.max() >= EVIDENCE_CONF_LEVEL:
                conf += 1
            per_frame.setdefault(int(frame_id), []).append(int(known.argmax()))
    if total == 0:
        return EvidenceQuality(0.0, 0.0, 0, False)
    n_known = max(1, len(catalog.labels) - 1)
    diversity = float(
        np.mean([len(set(v)) / min(n_known, len(v)) for v in per_frame.values()])
    )
    conf_frac = conf / total
    ok = conf_frac >= EVIDENCE_MIN_CONF_FRAC and diversity >= EVIDENCE_MIN_DIVERSITY
    return EvidenceQuality(conf_frac, diversity, len(per_frame), ok)


def run_fragment_solver(
    trajectories_df: pd.DataFrame,
    catalog: IdentityCatalog,
    params: dict[str, Any] | None = None,
    cache: Any = None,
    catalog_spec: Any = None,
) -> pd.DataFrame:
    """End-to-end fragment solver: cache evidence → forward-backward smoothing
    → (optional PELT split on the smoothed posterior) → iterative assign.

    Identity Phase 5 (the honesty fix) + Phase 7 Task 5 (clean-break
    retirement): when ``cache`` (an open, read-mode ``IdentityEvidenceCache``)
    is given, every trajectory's identity evidence is sourced from it -- via
    ``identity/smoothing.py``'s ``load_trajectory_evidence`` (join on
    ``(FrameID, DetectionID)``) then ``smooth_trajectory_posteriors``
    (forward-backward chaining). This makes the solver self-sufficient: it
    produces real identities even when ``ENABLE_IDENTITY_IN_TRACKING`` was
    off for the whole run, as long as Phase 3's evidence sidecar was written
    (which happens unconditionally -- the live pipeline always supplies
    ``cache``). When ``cache`` is ``None`` (or it has no evidence for any
    trajectory in ``trajectories_df``), this degrades gracefully: PELT
    splitting is skipped (nothing to split on) and
    ``solve_global_assignment`` produces the documented no-sidecar outcome
    (empty evidence per trajectory -- "no evidence, no belief") rather than
    reconstructing from ``CNN_*_Prob``/``DetectedTag*`` wide-CSV columns --
    that reconstruction fallback was dead in production (the live pipeline
    always supplies a cache) and was deleted in Phase 7 Task 5.

    Per-row ``IdentityFinalSmoothedLabel``/``IdentityFinalSmoothedConfidence``
    (the raw per-frame smoothed decode, pre-fragment-solving) are written
    whenever cache evidence was available, via ``_annotate_smoothed_labels``.

    Parameters
    ----------
    trajectories_df : post-augmentation trajectory DataFrame.
    catalog : IdentityCatalog for the run.
    params : optional overrides. Keys:
        ENABLE_PELT_SPLITTING            bool   default False
        CHANGEPOINT_PENALTY              float  default 3.0 — in probability units (raw, un-normalised signal)
        MIN_FRAGMENT_FRAMES              int    default 5
        PELT_MODEL                       str    default "rbf" (l1 / l2 / rbf)
        FRAGMENT_CNN_WEIGHT              float  default 0.40
        FRAGMENT_TAG_WEIGHT              float  default 0.15
        ONLINE_PRIOR_WEIGHT              float  default 0.25
        FRAGMENT_MIN_SUPPORT             float  default 0.5 — a label is a candidate
            for a fragment only if its normalised support ≥ this; absolute posterior
            floor.
        FRAGMENT_LENGTH_WEIGHT           float  default 0.60
            Multiplicative blend [0,1]: discounts short fragments' evidence relative
            to the longest fragment in the pool.  Prevents a tiny high-confidence
            fragment from overriding a long spatially-consistent track on CNN alone.
        SPATIAL_NO_NEIGHBOR_SCORE        float  default 0.3
        FRAGMENT_SPATIAL_VETO_THRESHOLD  float  default 0.05
            Minimum acceptable spatial score when neighbors exist; fragments below
            this are marked ineligible for that identity (spatially incompatible).
        ASSIGNMENT_MARGIN_THRESHOLD      float  default 0.10
            Minimum global-objective delta required to accept a fragment relabel
            during iterative refinement (monotone gate epsilon). Floored at 1e-3
            regardless of the configured value, so the termination argument for
            multi-blocker displacement (see ``_iterative_assign``'s docstring)
            always holds even if a caller passes 0.
        FRAGMENT_MAX_BLOCKERS            int    default 4
            Cap on the number of same-label overlapping fragments a single
            displacement move may evict and re-home in one step.
        MAX_VELOCITY_BREAK               float  default 50.0
        MAX_BRIDGE_GAP_FRAMES            int    default 30
            Cap on the temporal gap (in frames) used when computing the implied
            velocity between two same-identity segments.  Without this cap an
            arbitrarily long temporal gap would excuse an arbitrarily large
            spatial jump (``dist / gap`` shrinks with gap), so the same identity
            could be assigned to two trajectories at far-apart positions
            separated by a long pause.  Beyond this window we have no evidence
            of the animal's path; the bridge must remain plausible as if the
            gap were no longer than this many frames.
        FRAGMENT_TOP_K                   int    default 3
            Number of top-evidence candidate labels evaluated per fragment per pass.
        FRAGMENT_MAX_PASSES              int    default 10
            Hard cap on iterative-refinement passes.
        FRAGMENT_UNKNOWN_DOUBT_BONUS     float  default 0.5
            Additive doubt bonus for currently-Unknown fragments so they get
            re-evaluated early in each pass.
        IDENTITY_TRANSITION_EPSILON      float  default 0.02
            Sticky-Markov transition leak used by the forward-backward
            smoother (same knob/semantics as the realtime decoder's).
        IDENTITY_ENABLE_SMOOTHING        bool   default True
            When True (default), cache evidence is forward-backward
            smoothed via ``smooth_trajectory_posteriors`` before being used
            for changepoint detection and assignment. When False, the
            smoothing step is skipped entirely and the raw per-frame cache
            evidence is used directly instead.
    cache : an open (mode="r") IdentityEvidenceCache, or None. Required for
        the self-sufficient/cache-sourced path; when omitted the solver
        produces the no-sidecar degrade (empty evidence, no reconstruction).
    catalog_spec : the ``IdentityCatalogSpec`` ``catalog`` was built from, or
        None. Required to remap each identity model's phase-local evidence
        onto a *cross-product* catalog (two or more identity models): the
        spec carries the per-entry factor structure the phase maps are built
        from. Omitted/None falls back to exact label matching, which is
        correct for a single identity model (its phase basis IS the global
        catalog) but would floor every phase label -- and, after
        renormalization, fabricate certainty on ``unknown`` -- on a
        composite catalog.
    """
    params = params or {}

    if trajectories_df is None or trajectories_df.empty:
        return trajectories_df if trajectories_df is not None else pd.DataFrame()

    known_labels = list(catalog.labels[1:])
    if not known_labels:
        return trajectories_df

    phase_label_maps: dict[str, dict[str, list[int]]] = {}
    if catalog_spec is not None:
        from hydra_suite.core.individual.identity.phase_remap import (
            build_phase_label_maps,
        )

        phase_label_maps = build_phase_label_maps(
            catalog_spec, catalog, params.get("CNN_CLASSIFIERS") or []
        )

    smoothed_by_traj: dict[Any, list[tuple[int, np.ndarray]]] | None = None
    raw_evidence: dict[Any, list[tuple[int, np.ndarray]]] = {}
    if cache is not None:
        try:
            raw_evidence = load_trajectory_evidence(
                trajectories_df, cache, catalog, phase_label_maps
            )
        except Exception:
            log.exception(
                "fragment_solver: failed to load evidence from the identity "
                "evidence cache; proceeding without cache evidence."
            )
            raw_evidence = {}

        if raw_evidence:
            if bool(params.get("IDENTITY_ENABLE_SMOOTHING", True)):
                transition_epsilon = float(
                    params.get("IDENTITY_TRANSITION_EPSILON", 0.02)
                )
                smoothed_by_traj = {}
                for traj_id, sequence in raw_evidence.items():
                    frame_ids = [f for f, _ in sequence]
                    log_probs_list = [lp for _, lp in sequence]
                    smoothed = smooth_trajectory_posteriors(
                        log_probs_list, transition_epsilon
                    )
                    smoothed_by_traj[traj_id] = list(zip(frame_ids, smoothed))
            else:
                # Smoothing disabled: use the raw (already-normalized,
                # per-frame) cache evidence directly -- no forward-backward
                # chaining -- for changepoint detection and assignment.
                log.info(
                    "fragment_solver: IDENTITY_ENABLE_SMOOTHING is False; "
                    "using unsmoothed per-frame evidence."
                )
                smoothed_by_traj = {
                    traj_id: list(sequence)
                    for traj_id, sequence in raw_evidence.items()
                }
        else:
            log.info(
                "fragment_solver: identity evidence cache provided but no "
                "trajectory evidence matched; solver will no-op for identity "
                "(falls back to any CSV columns present)."
            )

    # Preserve the smoothed posteriors for the raw per-row annotation
    # (``IdentityFinalSmoothedLabel``/``...Confidence``) regardless of what
    # the evidence-quality breaker below decides for splitting/assignment --
    # a human proofreading the run should still see what the (uninformative)
    # source thought, even when the solver refuses to act on it.
    smoothed_for_annotation = smoothed_by_traj

    breaker_tripped = False
    if raw_evidence:
        quality = assess_evidence_quality(raw_evidence, catalog)
        if not quality.ok:
            log.error(
                "fragment_solver: identity evidence is uninformative "
                "(confident=%.1f%% of detections, diversity=%.2f over %d "
                "frames) -- refusing to split or assign identities. Check "
                "the classifier's fit_policy / preprocessing.",
                quality.conf_frac * 100,
                quality.diversity,
                quality.n_frames,
            )
            smoothed_by_traj = None
            breaker_tripped = True

    did_split = False
    if params.get("ENABLE_PELT_SPLITTING", False) and smoothed_by_traj:
        changepoints = detect_identity_changepoints(smoothed_by_traj, catalog, params)
        split_df = split_trajectories_at_changepoints(
            trajectories_df, changepoints, params
        )
        n_splits = sum(len(v) for v in changepoints.values())
        did_split = n_splits > 0
        log.info(
            "fragment_solver: PELT found %d changepoints; %d → %d trajectories after splitting.",
            n_splits,
            trajectories_df["TrajectoryID"].nunique(),
            split_df["TrajectoryID"].nunique(),
        )
    else:
        if params.get("ENABLE_PELT_SPLITTING", False):
            if breaker_tripped:
                # The ERROR log above already states the real reason
                # (evidence rejected as uninformative) -- don't say
                # "no evidence is available", which is misleading: evidence
                # WAS available, it was rejected by the breaker.
                log.info(
                    "fragment_solver: PELT splitting requested but the "
                    "evidence-quality breaker rejected the evidence "
                    "(see the ERROR above); skipping split."
                )
            else:
                log.info(
                    "fragment_solver: PELT splitting requested but no smoothed "
                    "cache evidence is available; skipping split."
                )
        split_df = trajectories_df
        log.info(
            "fragment_solver: iteratively assigning labels to %d existing trajectories.",
            trajectories_df["TrajectoryID"].nunique(),
        )

    if smoothed_for_annotation:
        split_df = _annotate_smoothed_labels(
            split_df, smoothed_for_annotation, catalog, params
        )

    if breaker_tripped:
        # Design spec section 3.4: on a tripped evidence-quality breaker, the
        # solver must not commit any label -- bypass solve_global_assignment
        # entirely so its Unknown-rescue pass (which commits *some* label to
        # every fragment even with zero informative evidence -- correct
        # behavior for the general no-cache/no-evidence path, but not what
        # the breaker is supposed to guarantee) never runs.
        solved = _write_no_evidence_outcome(split_df, catalog)
    else:
        solved = solve_global_assignment(
            split_df, catalog, params, evidence_by_traj=smoothed_by_traj
        )
    merged = merge_same_label_neighbours(solved, did_split=did_split)
    log.info(
        "fragment_solver: re-merged %d → %d trajectories after assignment.",
        solved["TrajectoryID"].nunique(),
        merged["TrajectoryID"].nunique(),
    )
    return merged
