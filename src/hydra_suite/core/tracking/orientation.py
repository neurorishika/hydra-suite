"""Orientation smoothing for the tracking pipeline.

Two modes:

1. **Directed source active** (head-tail and/or pose-override loaded for the
   run): the caller has already resolved per-detection heading via
   :func:`hydra_suite.core.identity.geometry.resolve_detection_tracking_theta`.
   High-quality directed detections become the new anchor; low-quality /
   undirected detections fall back to the OBB axis collapsed against the
   anchor.  This module simply passes that resolved heading through — no
   motion-based override, no flip hysteresis.  Global 180° ambiguities are
   resolved later by the post-processing DP pass.

2. **No directed source**: motion is the only physical signal that can
   disambiguate the 180°-symmetric OBB axis.  This module applies the
   motion-aware smoothing (stationary → clamp delta against the previous
   anchor, moving → optional motion-direction flip).
"""

import math

from hydra_suite.utils.geometry import wrap_angle_degs


def smooth_orientation(
    r,
    theta,
    speed,
    p,
    orientation_last,
    position_deques,
    directed_heading=False,
    motion_is_reversed=False,
):
    """Smooth a track's orientation over time.

    Args:
        r: Track index.
        theta: Caller-resolved heading (radians).  When a directed source is
            active for the run, the caller has already chosen between the
            directed heading (high-quality detections) and the OBB axis
            collapsed against the anchor (low-quality / undirected
            detections), so this value is trusted as-is.
        speed: Track speed estimate, used only by the undirected motion path.
        p: Parameter dict (needs VELOCITY_THRESHOLD, MAX_ORIENT_DELTA_STOPPED,
           INSTANT_FLIP_ORIENTATION, DIRECTED_ORIENT_POSTHOC_CONSISTENCY).
        orientation_last: List of last committed theta per track (mutated by
            the caller — this function does not write to it).
        position_deques: Per-track deques of (x, y, frame) positions, used by
            the motion-direction flip in the undirected path.
        directed_heading: True when this detection had a high-quality
            directed source (head-tail / pose) above its confidence gate.
            Currently unused — the caller's resolved theta already encodes
            the high-quality vs fallback decision.
        motion_is_reversed: True when ``position_deques`` records positions
            in reverse-time order (backward tracking pass).  The undirected
            motion-flip path negates the displacement vector accordingly.

    Returns:
        Smoothed theta in radians.
    """
    # When a directed source (head-tail or pose) is active for this run, trust
    # the caller's resolved heading.  Anchor-based hysteresis has already been
    # applied upstream via collapse_obb_axis_theta against orientation_last,
    # and global 180° ambiguities are resolved by the post-processing DP pass.
    if p.get("DIRECTED_ORIENT_POSTHOC_CONSISTENCY", False):
        return theta

    old = orientation_last[r]

    # --- Undirected smoothing (axis-only, motion is the only direction signal) ---
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
        dx, dy = (x2 - x1, y2 - y1)
        if motion_is_reversed:
            dx, dy = -dx, -dy
        ang = math.atan2(dy, dx)
        diff = (ang - theta + math.pi) % (2 * math.pi) - math.pi
        if abs(diff) > math.pi / 2:
            final_theta = (theta + math.pi) % (2 * math.pi)
    return final_theta
