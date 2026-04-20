"""Orientation smoothing for the tracking pipeline.

Temporal smoothing of track orientations, handling both directed (pose/head-tail)
and undirected (OBB axis) heading sources with flip detection and hysteresis.
"""

import math

from hydra_suite.utils.geometry import wrap_angle_degs


def _smooth_directed_heading(
    r,
    theta,
    speed,
    p,
    old,
    position_deques,
    orient_confidence,
    heading_flip_counters,
):
    """Smooth a directed (pose/head-tail) heading with flip hysteresis."""
    if old is None:
        if heading_flip_counters is not None:
            heading_flip_counters[r] = 0
        return theta

    flip_conf_thresh = float(p.get("DIRECTED_ORIENT_FLIP_CONFIDENCE", 0.7))
    flip_persistence = int(p.get("DIRECTED_ORIENT_FLIP_PERSISTENCE", 3))
    old_deg = math.degrees(old)
    new_deg = math.degrees(theta)
    delta = wrap_angle_degs(new_deg - old_deg)

    if abs(delta) > 90:
        flip_supported = _is_flip_motion_supported(
            r,
            speed,
            p,
            old_deg,
            new_deg,
            position_deques,
            orient_confidence,
            flip_conf_thresh,
        )
        new_deg = _apply_flip_hysteresis(
            r,
            new_deg,
            flip_supported,
            flip_persistence,
            heading_flip_counters,
        )
    else:
        if heading_flip_counters is not None:
            heading_flip_counters[r] = 0

    return math.radians(new_deg % 360.0)


def _is_flip_motion_supported(
    r,
    speed,
    p,
    old_deg,
    new_deg,
    position_deques,
    orient_confidence,
    flip_conf_thresh,
):
    """Check if track motion supports an orientation flip."""
    if speed < p["VELOCITY_THRESHOLD"] or len(position_deques[r]) != 2:
        return False
    (x1, y1, _), (x2, y2, _) = position_deques[r]
    ang_deg = math.degrees(math.atan2(y2 - y1, x2 - x1))
    motion_favors_new = abs(wrap_angle_degs(new_deg - ang_deg)) < abs(
        wrap_angle_degs(old_deg - ang_deg)
    )
    return motion_favors_new and orient_confidence >= flip_conf_thresh


def _apply_flip_hysteresis(r, new_deg, flip_supported, flip_persistence, counters):
    """Apply hysteresis counter logic, returning possibly reversed new_deg."""
    if flip_supported:
        if counters is not None:
            counters[r] += 1
            if counters[r] >= flip_persistence:
                counters[r] = 0
            else:
                new_deg = (new_deg + 180.0) % 360.0
    else:
        if counters is not None:
            counters[r] = 0
        new_deg = (new_deg + 180.0) % 360.0
    return new_deg


def smooth_orientation(
    r,
    theta,
    speed,
    p,
    orientation_last,
    position_deques,
    directed_heading=False,
    orient_confidence=1.0,
    heading_flip_counters=None,
):
    """Smooth a track's orientation over time.

    Args:
        r: Track index.
        theta: Raw measured heading (radians).
        speed: Track speed estimate.
        p: Parameter dict (needs VELOCITY_THRESHOLD, MAX_ORIENT_DELTA_STOPPED,
           INSTANT_FLIP_ORIENTATION, DIRECTED_ORIENT_SMOOTHING,
           DIRECTED_ORIENT_FLIP_CONFIDENCE, DIRECTED_ORIENT_FLIP_PERSISTENCE,
           DIRECTED_ORIENT_POSTHOC_CONSISTENCY).
        orientation_last: List of last committed theta per track (mutated).
        position_deques: Per-track deques of (x, y, frame) positions.
        directed_heading: Whether this heading comes from a directed source.
        orient_confidence: Confidence in the heading estimate [0, 1].
        heading_flip_counters: Per-track flip hysteresis counters (mutated).

    Returns:
        Smoothed theta in radians.  When DIRECTED_ORIENT_POSTHOC_CONSISTENCY is
        True and directed_heading is True, the raw theta is returned unchanged so
        that global heading consistency can be resolved in post-processing.
    """
    old = orientation_last[r]

    if directed_heading and p.get("DIRECTED_ORIENT_POSTHOC_CONSISTENCY", False):
        # Post-hoc mode: accept the raw directed heading without any online flip
        # check.  Global heading consistency will be resolved per-track in the
        # post-processing step instead.
        if old is None and heading_flip_counters is not None:
            heading_flip_counters[r] = 0
        return theta

    if directed_heading and p.get("DIRECTED_ORIENT_SMOOTHING", True):
        return _smooth_directed_heading(
            r,
            theta,
            speed,
            p,
            old,
            position_deques,
            orient_confidence,
            heading_flip_counters,
        )

    # --- Original undirected smoothing (axis-only, no direction signal) ---
    final_theta = theta
    if speed < p["VELOCITY_THRESHOLD"] and old is not None:
        old_deg, new_deg = math.degrees(old), math.degrees(theta)
        delta = wrap_angle_degs(new_deg - old_deg)
        if abs(delta) > 90:
            new_deg = (new_deg + 180) % 360
        elif abs(delta) > p["MAX_ORIENT_DELTA_STOPPED"]:
            new_deg = old_deg + math.copysign(p["MAX_ORIENT_DELTA_STOPPED"], delta)
        final_theta = math.radians(new_deg)
    elif speed >= p["VELOCITY_THRESHOLD"] and p["INSTANT_FLIP_ORIENTATION"]:
        (x1, y1, _), (x2, y2, _) = position_deques[r]
        ang = math.atan2(y2 - y1, x2 - x1)
        diff = (ang - theta + math.pi) % (2 * math.pi) - math.pi
        if abs(diff) > math.pi / 2:
            final_theta = (theta + math.pi) % (2 * math.pi)
    return final_theta
