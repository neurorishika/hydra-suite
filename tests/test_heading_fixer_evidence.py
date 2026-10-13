"""Directed-heading post-fix: self-consistent AND anchored to per-frame evidence.

Regression: the fixer minimised frame-to-frame angle change only, with one
whole-track majority anchor. Flipping a contiguous stretch costs nothing inside
it, so a genuine >90 deg turn let it flip everything after the turn: on Libby
X8_d16 an ant walking forward carried a heading pointing at its gaster for 595
frames, although head-tail, pose and motion all agreed on the right direction.
"""

from __future__ import annotations

import numpy as np

from hydra_suite.core.post.processing import _fix_heading_globally

PI = np.pi


def _circ(a, b):
    return np.abs(np.angle(np.exp(1j * (np.asarray(a) - np.asarray(b)))))


def _flipped(out, ref):
    return _circ(out, ref) > PI / 2


def test_sharp_turn_does_not_flip_the_following_stretch():
    rng = np.random.default_rng(0)
    theta = np.r_[np.full(1300, 3.14), np.full(600, 0.72)] + rng.normal(0, 0.02, 1900)
    out = _fix_heading_globally(theta)
    assert not _flipped(out, theta).any()


def test_short_isolated_flips_are_corrected():
    theta = np.full(200, 1.0)
    for start, length in [(20, 1), (60, 3), (120, 8)]:
        theta[start : start + length] = 1.0 + PI
    out = _fix_heading_globally(theta)
    assert not _flipped(out, np.full(200, 1.0)).any()


def test_long_consistent_evidence_is_kept_without_motion():
    theta = np.full(300, 1.0)
    theta[100:160] = 1.0 + PI  # 60 frames of consistent opposite head calls
    out = _fix_heading_globally(theta)
    assert not _flipped(out, theta).any()


def _walk_plus_x(n, speed=5.0):
    x = np.arange(n, dtype=float) * speed
    return x, np.zeros(n)


def test_medium_flip_against_clear_motion_is_corrected():
    n = 200
    x, y = _walk_plus_x(n)
    theta = np.zeros(n)
    theta[80:100] = PI  # 20 frames pointing backwards while walking forward
    kept = _fix_heading_globally(theta)  # no motion information: evidence wins
    assert _flipped(kept, np.zeros(n))[80:100].all()
    out = _fix_heading_globally(theta, x=x, y=y, body_size_px=40.0)
    assert not _flipped(out, np.zeros(n)).any()


def test_long_backward_walk_trusts_head_tail_over_motion():
    n = 400
    x, y = _walk_plus_x(n)
    theta = np.zeros(n)
    theta[100:250] = PI  # 150 frames of head calls saying it walks backwards
    out = _fix_heading_globally(theta, x=x, y=y, body_size_px=40.0)
    assert not _flipped(out, theta).any()


def test_nan_frames_are_preserved():
    theta = np.full(50, 0.5)
    theta[10:15] = np.nan
    out = _fix_heading_globally(theta)
    assert np.isnan(out[10:15]).all()
    assert not _flipped(out[~np.isnan(theta)], theta[~np.isnan(theta)]).any()


def test_interpolate_trajectories_passes_motion_to_the_heading_fix():
    import pandas as pd

    from hydra_suite.core.post.processing import (
        heading_body_size_px,
        interpolate_trajectories,
    )

    n = 200
    x, y = _walk_plus_x(n)
    theta = np.zeros(n)
    theta[80:100] = PI
    df = pd.DataFrame(
        {"TrajectoryID": 1, "FrameID": np.arange(n), "X": x, "Y": y, "Theta": theta}
    )
    body = heading_body_size_px({"REFERENCE_BODY_SIZE": 20.0}, 2.0)
    assert body == 40.0
    out = interpolate_trajectories(
        df, method="linear", directed_heading_posthoc=True, heading_body_size_px=body
    )
    assert not _flipped(out["Theta"].to_numpy(), np.zeros(n)).any()
    assert heading_body_size_px({}, 1.0) is None
