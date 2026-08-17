"""Active-learning frame acquisition: weighted ranking with diversity guard."""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Sequence

import numpy as np

from .signals import ALSignals


@dataclass
class AcquisitionWeights:
    """Per-signal weights. Auto-normalized to sum to 1.0 at use."""

    uncertainty: float = 0.40
    nms_instability: float = 0.20
    count: float = 0.20
    crowd: float = 0.15
    edge: float = 0.05
    fragmentation: float = 0.0
    # Tracker-only extras (zero on detector-only paths)
    assignment: float = 0.0
    track_loss: float = 0.0
    position_uncertainty: float = 0.0

    def normalized(self) -> "AcquisitionWeights":
        total = sum(getattr(self, f.name) for f in fields(self))
        if total <= 0:
            return AcquisitionWeights()
        return AcquisitionWeights(
            **{f.name: getattr(self, f.name) / total for f in fields(self)}
        )


PRESETS: dict[str, AcquisitionWeights] = {
    "balanced": AcquisitionWeights(
        uncertainty=0.40,
        nms_instability=0.20,
        count=0.20,
        crowd=0.15,
        edge=0.05,
    ),
    "uncertainty_heavy": AcquisitionWeights(
        uncertainty=0.55,
        nms_instability=0.25,
        count=0.10,
        crowd=0.05,
        edge=0.05,
    ),
    "exploration_heavy": AcquisitionWeights(
        uncertainty=0.25,
        nms_instability=0.15,
        count=0.15,
        crowd=0.30,
        edge=0.15,
    ),
    "tracker_default": AcquisitionWeights(
        uncertainty=0.30,
        nms_instability=0.0,
        count=0.20,
        crowd=0.15,
        edge=0.05,
        fragmentation=0.30,
        assignment=0.15,
        track_loss=0.10,
        position_uncertainty=0.05,
    ),
}


def _channel_array(signals: Sequence[ALSignals], attr: str) -> np.ndarray:
    """Pull a signal channel into a numpy array, treating NaN as 0."""
    if attr in {"assignment", "track_loss", "position_uncertainty"}:
        vals = [s.extras.get(attr, 0.0) for s in signals]
    elif attr == "uncertainty":
        vals = [s.uncertainty_score for s in signals]
    else:
        # Map channel names to ALSignals attributes
        attr_map = {
            "count": "count_deviation",
            "nms_instability": "nms_instability",
            "crowd": "crowd_score",
            "edge": "edge_score",
            "fragmentation": "fragmentation_score",
        }
        attr_name = attr_map.get(attr, attr)
        vals = [getattr(s, attr_name) for s in signals]
    arr = np.asarray(vals, dtype=np.float64)
    arr = np.where(np.isnan(arr), 0.0, arr)
    return arr


def _composite_score(
    signals: Sequence[ALSignals],
    weights: AcquisitionWeights,
) -> np.ndarray:
    """Weighted sum of ABSOLUTE per-channel severities, in [0, 1].

    Deliberately NOT min-max normalized: a within-run rescale makes the top
    frame score ~1 however easy the video was, which both defeats `min_score`
    as a gate and makes scores incomparable across videos. A perfectly tracked
    video must be able to legitimately produce zero candidates -- that is the
    entire point of restoring absolute semantics. Do NOT reintroduce
    normalization here.
    """
    w = weights.normalized()
    channels = {
        "uncertainty": w.uncertainty,
        "nms_instability": w.nms_instability,
        "count": w.count,
        "crowd": w.crowd,
        "fragmentation": w.fragmentation,
        "edge": w.edge,
        "assignment": w.assignment,
        "track_loss": w.track_loss,
        "position_uncertainty": w.position_uncertainty,
    }
    n = len(signals)
    score = np.zeros(n, dtype=np.float64)
    for name, weight in channels.items():
        if weight <= 0:
            continue
        score += weight * np.clip(_channel_array(signals, name), 0.0, 1.0)
    return score


def explain(
    signals: Sequence[ALSignals],
    weights: AcquisitionWeights,
) -> dict[str, float]:
    """Per-channel maxima observed across `signals`, for weighted channels only.

    Used to tell the user WHY a selection came back empty under absolute
    scoring, rather than surfacing a bare "no frames met the criteria".
    """
    w = weights.normalized()
    out: dict[str, float] = {}
    for name in (
        "uncertainty",
        "nms_instability",
        "count",
        "crowd",
        "fragmentation",
        "edge",
        "assignment",
        "track_loss",
        "position_uncertainty",
    ):
        if getattr(w, name) <= 0:
            continue
        arr = _channel_array(signals, name)
        out[name] = float(arr.max()) if arr.size else 0.0
    return out


def select(
    signals: Sequence[ALSignals],
    weights: AcquisitionWeights,
    k: int,
    diversity_window: int = 30,
    probabilistic: bool = True,
    rng: np.random.Generator | None = None,
    min_score: float = 0.0,
) -> list[int]:
    """Return up to k frame_ids from `signals`, ranked by weighted composite score.

    `diversity_window` enforces minimum frame-index spacing between picks (`abs(a-b) >= diversity_window`).
    `probabilistic=True` uses rank-based sampling; False is deterministic top-K.
    `min_score` drops candidates whose composite score is below this cutoff.
    """
    if not signals or k <= 0:
        return []

    score = _composite_score(signals, weights)

    keep_mask = score >= float(min_score)
    indices = [int(i) for i in np.argsort(-score) if keep_mask[i]]
    if not indices:
        return []
    sorted_ids = [int(signals[i].frame_id) for i in indices]
    rng = rng or np.random.default_rng()

    picks: list[int] = []

    def _diverse(fid: int) -> bool:
        return all(abs(fid - p) >= diversity_window for p in picks)

    if not probabilistic:
        for fid in sorted_ids:
            if len(picks) >= k:
                break
            if _diverse(fid):
                picks.append(fid)
        return picks

    candidates = sorted_ids[:]
    while len(picks) < k and candidates:
        weights_arr = np.array([1.0 / (i + 1) for i in range(len(candidates))])
        weights_arr /= weights_arr.sum()
        chosen_idx = int(rng.choice(len(candidates), p=weights_arr))
        fid = candidates[chosen_idx]
        if _diverse(fid):
            picks.append(fid)
            # Only enforce the diversity-window pruning around accepted picks.
            candidates = [c for c in candidates if abs(c - fid) >= diversity_window]
        else:
            # Rejected: drop just this candidate and continue.
            candidates.pop(chosen_idx)
    return picks
