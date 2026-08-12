"""Task 3: PELT changepoint detection driven by the smoothed posterior.

``detect_identity_changepoints`` now accepts precomputed smoothed per-frame
posteriors (Task 2's ``smooth_trajectory_posteriors`` output, paired with
FrameIDs) instead of reading ``CNN_*_Prob`` columns off a DataFrame -- the
offline decoder is being made self-sufficient from the evidence cache
(Phase 5), and CSV probability columns are only ever populated when
``ENABLE_IDENTITY_IN_TRACKING`` was on.
"""

from __future__ import annotations

import numpy as np
import pytest

from hydra_suite.core.individual.identity.catalog import IdentityCatalog
from hydra_suite.core.individual.identity.offline import detect_identity_changepoints

ruptures = pytest.importorskip(
    "ruptures", reason="ruptures not installed; real PELT splitting cannot be exercised"
)


def _make_catalog() -> IdentityCatalog:
    return IdentityCatalog.from_labels(["blue", "green"])


def _log_probs(blue_prob: float, green_prob: float) -> np.ndarray:
    """A 3-slot (unknown, blue, green) normalized log-posterior."""
    probs = np.array([1e-6, blue_prob, green_prob], dtype=np.float64)
    probs /= probs.sum()
    return np.log(probs)


def _switching_trajectory(n_frames: int = 60, swap_at: int = 30):
    """[(FrameID, log_probs)] favoring blue for the first half, green after."""
    sequence = []
    for frame in range(n_frames):
        if frame < swap_at:
            lp = _log_probs(0.9, 0.1)
        else:
            lp = _log_probs(0.1, 0.9)
        sequence.append((frame, lp))
    return sequence


def _stable_trajectory(n_frames: int = 60):
    """[(FrameID, log_probs)] constantly favoring blue -- no regime change."""
    lp = _log_probs(0.9, 0.1)
    return [(frame, lp) for frame in range(n_frames)]


def test_changepoint_detects_smoothed_posterior_switch():
    catalog = _make_catalog()
    smoothed_by_traj = {1: _switching_trajectory(n_frames=60, swap_at=30)}
    result = detect_identity_changepoints(
        smoothed_by_traj,
        catalog,
        {"CHANGEPOINT_PENALTY": 2.0, "MIN_FRAGMENT_FRAMES": 5},
    )
    splits = result.get(1, [])
    assert len(splits) == 1
    assert 27 <= splits[0] <= 32, f"expected split near frame 29-30, got {splits[0]}"


def test_changepoint_no_split_when_smoothed_posterior_stable():
    catalog = _make_catalog()
    smoothed_by_traj = {1: _stable_trajectory(n_frames=60)}
    result = detect_identity_changepoints(
        smoothed_by_traj,
        catalog,
        {"CHANGEPOINT_PENALTY": 2.0, "MIN_FRAGMENT_FRAMES": 5},
    )
    assert result.get(1, []) == [], "stable smoothed posterior should have no splits"


def test_changepoint_empty_input_returns_empty():
    catalog = _make_catalog()
    result = detect_identity_changepoints({}, catalog, {})
    assert result == {}


def test_changepoint_short_trajectory_skipped():
    """Fewer than min_frames*2 rows: no PELT run, no split."""
    catalog = _make_catalog()
    smoothed_by_traj = {1: _switching_trajectory(n_frames=6, swap_at=3)}
    result = detect_identity_changepoints(
        smoothed_by_traj,
        catalog,
        {"CHANGEPOINT_PENALTY": 2.0, "MIN_FRAGMENT_FRAMES": 5},
    )
    assert result.get(1, []) == []


def test_changepoint_no_ruptures_returns_empty(monkeypatch):
    """Graceful fallback: ruptures import failure yields {} rather than raising."""
    import builtins

    catalog = _make_catalog()
    smoothed_by_traj = {1: _switching_trajectory(n_frames=60, swap_at=30)}

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "ruptures":
            raise ImportError("simulated missing ruptures")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    result = detect_identity_changepoints(smoothed_by_traj, catalog, {})
    assert result == {}
