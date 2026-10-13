"""Directed-heading post-fix: self-consistent AND anchored to per-frame evidence.

Regression: the fixer minimised frame-to-frame angle change only, with one
whole-track majority anchor. Flipping a contiguous stretch costs nothing inside
it, so a genuine >90 deg turn let it flip everything after the turn: on Libby
X8_d16 an ant walking forward carried a heading pointing at its gaster for 595
frames, although head-tail, pose and motion all agreed on the right direction.
"""

from __future__ import annotations

import numpy as np

from hydra_suite.core.post.heading_fix import HeadingFixSettings
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
    theta = np.full(400, 1.0)
    theta[100:220] = 1.0 + PI  # 120 frames of consistent opposite head calls
    out = _fix_heading_globally(theta)
    assert not _flipped(out, theta).any()


def test_instant_reversal_in_a_still_animal_is_folded_in():
    # 40 frames whose head calls all reverse at once while the animal stands
    # still: physically impossible, so the shorter side follows the longer.
    n = 300
    theta = np.full(n, 1.0)
    theta[100:140] = 1.0 + PI
    x, y = np.zeros(n), np.zeros(n)
    out = _fix_heading_globally(
        theta, x=x, y=y, settings=HeadingFixSettings(body_size_px=40.0)
    )
    assert not _flipped(out, np.full(n, 1.0)).any()


def test_reversal_at_a_track_join_is_allowed():
    # A position jump of several body lengths is a join onto another animal;
    # its heading may legitimately differ by 180 degrees.
    n = 200
    theta = np.r_[np.full(100, 1.0), np.full(100, 1.0 + PI)]
    x = np.r_[np.zeros(100), np.full(100, 200.0)]
    out = _fix_heading_globally(
        theta, x=x, y=np.zeros(n), settings=HeadingFixSettings(body_size_px=40.0)
    )
    assert not _flipped(out, theta).any()


def _walk_plus_x(n, speed=5.0):
    x = np.arange(n, dtype=float) * speed
    return x, np.zeros(n)


def test_medium_flip_against_clear_motion_is_corrected():
    n = 200
    x, y = _walk_plus_x(n)
    theta = np.zeros(n)
    theta[80:100] = PI  # 20 frames pointing backwards while walking forward
    out = _fix_heading_globally(
        theta, x=x, y=y, settings=HeadingFixSettings(body_size_px=40.0)
    )
    assert not _flipped(out, np.zeros(n)).any()


def test_long_backward_walk_trusts_head_tail_over_motion():
    n = 400
    x, y = _walk_plus_x(n)
    theta = np.zeros(n)
    theta[100:250] = PI  # 150 frames of head calls saying it walks backwards
    out = _fix_heading_globally(
        theta, x=x, y=y, settings=HeadingFixSettings(body_size_px=40.0)
    )
    assert not _flipped(out, theta).any()


def test_nan_frames_are_preserved():
    theta = np.full(50, 0.5)
    theta[10:15] = np.nan
    out = _fix_heading_globally(theta)
    assert np.isnan(out[10:15]).all()
    assert not _flipped(out[~np.isnan(theta)], theta[~np.isnan(theta)]).any()


def test_interpolate_trajectories_passes_motion_to_the_heading_fix():
    import pandas as pd

    from hydra_suite.core.post.processing import interpolate_trajectories

    n = 200
    x, y = _walk_plus_x(n)
    theta = np.zeros(n)
    theta[80:100] = PI
    df = pd.DataFrame(
        {"TrajectoryID": 1, "FrameID": np.arange(n), "X": x, "Y": y, "Theta": theta}
    )
    settings = HeadingFixSettings.from_params({"REFERENCE_BODY_SIZE": 20.0}, 2.0)
    assert settings.body_size_px == 40.0
    out = interpolate_trajectories(
        df, method="linear", directed_heading_posthoc=True, heading_fix=settings
    )
    assert not _flipped(out["Theta"].to_numpy(), np.zeros(n)).any()
    assert HeadingFixSettings.from_params({}).body_size_px is None


def test_motion_term_corrects_a_long_run_against_clear_motion():
    # 80 frames reversed while walking steadily forward: too long for the
    # reversal rule alone (kept without motion), corrected once motion counts.
    n = 300
    x, y = _walk_plus_x(n)
    theta = np.zeros(n)
    theta[100:180] = PI
    kept = _fix_heading_globally(theta, x=x, y=y)  # no body size: motion off
    assert _flipped(kept, np.zeros(n))[100:180].all()
    out = _fix_heading_globally(
        theta, x=x, y=y, settings=HeadingFixSettings(body_size_px=40.0)
    )
    assert not _flipped(out, np.zeros(n)).any()


def test_reversal_rule_spans_short_gaps_in_the_head_calls():
    n = 300
    theta = np.full(n, 1.0)
    theta[100:130] = 1.0 + PI
    theta[99] = theta[130] = np.nan  # head call missing at both ends of the run
    x, y = np.zeros(n), np.zeros(n)
    out = _fix_heading_globally(
        theta, x=x, y=y, settings=HeadingFixSettings(body_size_px=40.0)
    )
    ok = ~np.isnan(theta)
    assert not _flipped(out[ok], np.full(n, 1.0)[ok]).any()


def test_missing_position_does_not_count_as_standing_still():
    n = 200
    theta = np.r_[np.full(100, 1.0), np.full(100, 1.0 + PI)]
    x = np.zeros(n)
    x[100] = np.nan  # the join frame has no position
    out = _fix_heading_globally(
        theta, x=x, y=np.zeros(n), settings=HeadingFixSettings(body_size_px=40.0)
    )
    assert not _flipped(out, theta).any()


def test_reversal_rule_relaxes_at_low_frame_rates():
    # At 3 fps a real one-frame turn can exceed 180 deg/frame worth of the max
    # turn rate, so an instant reversal is no longer impossible.
    n = 300
    theta = np.full(n, 1.0)
    theta[100:140] = 1.0 + PI
    x, y = np.zeros(n), np.zeros(n)
    folded = _fix_heading_globally(
        theta, x=x, y=y, settings=HeadingFixSettings(body_size_px=40.0, fps=25.0)
    )
    assert not _flipped(folded, np.full(n, 1.0)).any()
    kept = _fix_heading_globally(
        theta, x=x, y=y, settings=HeadingFixSettings(body_size_px=40.0, fps=3.0)
    )
    assert not _flipped(kept, theta).any()


def test_fast_running_is_not_mistaken_for_a_track_join():
    # 30 px/frame with a 40 px body is fast running, not a jump to another
    # animal, so an instant reversal there is still folded in.
    n = 300
    x = np.arange(n, dtype=float) * 30.0
    theta = np.zeros(n)
    theta[100:140] = PI
    out = _fix_heading_globally(
        theta, x=x, y=np.zeros(n), settings=HeadingFixSettings(body_size_px=40.0)
    )
    assert not _flipped(out, np.zeros(n)).any()


def test_settings_are_time_based_and_overridable():
    at10 = HeadingFixSettings(body_size_px=40.0, fps=10.0)
    at50 = HeadingFixSettings(body_size_px=40.0, fps=50.0)
    # The same durations span 5x more frames at 50 fps.
    assert at50.reversal_max_gap == 5 * at10.reversal_max_gap
    assert np.isclose(at10.obs_cost, 5 * at50.obs_cost)
    # The track-join threshold is a speed: 10 body sizes/s.
    assert at10.join_step_px == 40.0 and at50.join_step_px == 8.0
    custom = HeadingFixSettings.from_params(
        {
            "FPS": 10,
            "HEADING_FIX_OVERRIDE_SECONDS": 2.4,
            "HEADING_FIX_MAX_TURN_DEG_S": None,
        }
    )
    assert custom.override_seconds == 2.4 and custom.max_turn_deg_s == 720.0


def test_fold_window_follows_the_frame_rate():
    # 40 reversed frames: 0.8 s at 50 fps (folded) but 8 s at 5 fps (kept, it
    # exceeds the 5 s fold window) -- with the turn-rate rule still active.
    n = 400
    theta = np.full(n, 1.0)
    theta[100:140] = 1.0 + PI
    x, y = np.zeros(n), np.zeros(n)
    fast = _fix_heading_globally(
        theta, x=x, y=y, settings=HeadingFixSettings(body_size_px=40.0, fps=50.0)
    )
    assert not _flipped(fast, np.full(n, 1.0)).any()
    slow = _fix_heading_globally(
        theta, x=x, y=y, settings=HeadingFixSettings(body_size_px=40.0, fps=5.0)
    )
    assert not _flipped(slow, theta).any()
