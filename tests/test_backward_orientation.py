"""Regression tests for backward-pass orientation handling."""

import math
from collections import deque

from hydra_suite.core.identity.geometry import collapse_obb_axis_theta
from hydra_suite.core.tracking.orientation import smooth_orientation


def _simulate_backward_orientation_loop(
    num_frames: int,
    obb_axis: float,
    directed_heading: bool,
    backward_mode: bool,
    feed_output_back: bool,
):
    """Simulate the per-frame orientation update loop from worker.py.

    Returns the list of det_theta_out values (one per frame) and the list
    of orientation_last values stored *after* each frame.

    When ``feed_output_back`` is True we reproduce the current (buggy)
    behaviour where ``det_theta_out`` is written back into
    ``orientation_last``. When False we model the proposed fix where the
    internal ``theta_for_tracking`` is the next-frame reference.
    """

    # Initial seed: collapse with no reference => normalised theta_axis.
    orientation_last = collapse_obb_axis_theta(obb_axis, None)

    outputs = []
    refs = []

    for _ in range(num_frames):
        # resolve_detection_tracking_theta with directed_heading=False
        # collapses the OBB axis against the current orientation_last.
        theta_for_tracking = collapse_obb_axis_theta(obb_axis, orientation_last)

        # Output computation mirrors worker.py:3728-3730.
        det_theta_out = theta_for_tracking
        if backward_mode and not directed_heading:
            det_theta_out = (det_theta_out + math.pi) % (2 * math.pi)

        outputs.append(det_theta_out)

        # Worker's current behaviour: orientation_last := det_theta_out.
        # Proposed fix: orientation_last := theta_for_tracking.
        if feed_output_back:
            orientation_last = det_theta_out
        else:
            orientation_last = theta_for_tracking

        refs.append(orientation_last)

    return outputs, refs


def test_backward_undirected_output_oscillates_with_current_logic():
    """Document the bug: feeding det_theta_out back makes output flip 180° per frame."""
    outputs, _ = _simulate_backward_orientation_loop(
        num_frames=6,
        obb_axis=0.0,
        directed_heading=False,
        backward_mode=True,
        feed_output_back=True,
    )
    # Successive outputs should differ by ~pi (modulo 2*pi).
    for i in range(1, len(outputs)):
        delta = (outputs[i] - outputs[i - 1] + math.pi) % (2 * math.pi) - math.pi
        assert (
            abs(abs(delta) - math.pi) < 1e-6
        ), f"Frame {i}: expected |delta| ≈ pi, got {delta}. Outputs: {outputs}"


def test_backward_undirected_output_stable_with_internal_reference():
    """The proposed fix: feeding theta_for_tracking back keeps output stable."""
    outputs, _ = _simulate_backward_orientation_loop(
        num_frames=6,
        obb_axis=0.0,
        directed_heading=False,
        backward_mode=True,
        feed_output_back=False,
    )
    # All outputs should be identical (animal hasn't turned).
    for i in range(1, len(outputs)):
        delta = (outputs[i] - outputs[0] + math.pi) % (2 * math.pi) - math.pi
        assert (
            abs(delta) < 1e-6
        ), f"Frame {i}: expected stable output, got delta={delta}. Outputs: {outputs}"


def test_forward_undirected_output_stable_either_way():
    """Sanity: forward mode is stable regardless of feedback policy."""
    for feed_back in (True, False):
        outputs, _ = _simulate_backward_orientation_loop(
            num_frames=6,
            obb_axis=0.0,
            directed_heading=False,
            backward_mode=False,
            feed_output_back=feed_back,
        )
        for i in range(1, len(outputs)):
            delta = (outputs[i] - outputs[0] + math.pi) % (2 * math.pi) - math.pi
            assert abs(delta) < 1e-6, (
                f"feed_back={feed_back} frame {i}: forward mode should be stable; "
                f"got delta={delta}. Outputs: {outputs}"
            )


def test_collapse_obb_axis_theta_with_internal_reference_is_stable():
    """End-to-end-style: with the fix, the next-frame reference (theta_for_tracking)
    is bit-stable when the OBB axis is unchanged."""
    obb_axis = 0.0
    ref = collapse_obb_axis_theta(obb_axis, None)
    history = [ref]
    for _ in range(10):
        theta_for_tracking = collapse_obb_axis_theta(obb_axis, ref)
        ref = theta_for_tracking  # the fix: feed internal value back, not output
        history.append(ref)
    assert all(abs(h - history[0]) < 1e-9 for h in history), history


def _make_position_deque(prev_xy, curr_xy):
    dq = deque(maxlen=2)
    dq.append((prev_xy[0], prev_xy[1], 0))
    dq.append((curr_xy[0], curr_xy[1], 1))
    return dq


def _params(**overrides):
    base = {
        "VELOCITY_THRESHOLD": 0.5,
        "MAX_ORIENT_DELTA_STOPPED": 30.0,
        "INSTANT_FLIP_ORIENTATION": True,
        "DIRECTED_ORIENT_SMOOTHING": True,
        "DIRECTED_ORIENT_FLIP_CONFIDENCE": 0.0,
        "DIRECTED_ORIENT_FLIP_PERSISTENCE": 3,
        "DIRECTED_ORIENT_POSTHOC_CONSISTENCY": False,
    }
    base.update(overrides)
    return base


def test_undirected_instant_flip_forward_aligns_to_motion():
    """Forward: motion vector points head-first; theta opposing motion gets flipped."""
    deque_r = _make_position_deque((0.0, 0.0), (10.0, 0.0))  # motion = +x
    theta = math.pi  # heading -x, opposite to motion
    out = smooth_orientation(
        r=0,
        theta=theta,
        speed=10.0,
        p=_params(),
        orientation_last=[0.0],
        position_deques=[deque_r],
        directed_heading=False,
    )
    # Forward + opposing motion -> flip toward motion (= 0).
    assert abs(((out - 0.0 + math.pi) % (2 * math.pi)) - math.pi) < 1e-6


def test_undirected_instant_flip_backward_aligns_to_true_motion():
    """Backward: position_deque motion is reversed; with motion_is_reversed=True the
    flip should align with the *true* head direction (opposite of deque motion)."""
    deque_r = _make_position_deque((0.0, 0.0), (10.0, 0.0))  # processing motion = +x
    # In backward, true motion (head direction) is -x.
    # theta = 0 (pointing +x) opposes true head direction; should be flipped to pi.
    theta = 0.0
    out = smooth_orientation(
        r=0,
        theta=theta,
        speed=10.0,
        p=_params(),
        orientation_last=[math.pi],
        position_deques=[deque_r],
        directed_heading=False,
        motion_is_reversed=True,
    )
    assert abs(((out - math.pi + math.pi) % (2 * math.pi)) - math.pi) < 1e-6


def test_directed_smoothed_flip_motion_supported_uses_negated_motion_in_backward():
    """Directed mode: _is_flip_motion_supported must use the true (head-first) motion
    direction. We check the flip_counters side effect because both paths happen to
    return pi in this scenario (one via flip-not-supported, one via supported-but-
    blocked-by-hysteresis); the counter is what distinguishes them.

    Setup: processing-motion is +x; old heading is pi (true head); new heading is 0.
    - With motion_is_reversed=True, true motion is -x (= pi). The new heading (0) is
      farther from -x than the old heading (pi), so flip is NOT supported and the
      counter is reset to 0.
    - With motion_is_reversed=False, processing motion (+x = 0) matches new heading;
      flip IS supported and the counter increments (then hysteresis blocks the flip
      because counter < persistence).
    """
    deque_r = _make_position_deque((0.0, 0.0), (10.0, 0.0))  # processing motion = +x

    flip_counters_backward = [0]
    out_backward = smooth_orientation(
        r=0,
        theta=0.0,
        speed=10.0,
        p=_params(DIRECTED_ORIENT_FLIP_CONFIDENCE=0.0),
        orientation_last=[math.pi],
        position_deques=[deque_r],
        directed_heading=True,
        orient_confidence=1.0,
        heading_flip_counters=flip_counters_backward,
        motion_is_reversed=True,
    )
    # Output keeps old direction (pi).
    assert abs(((out_backward - math.pi + math.pi) % (2 * math.pi)) - math.pi) < 1e-6
    # Counter must NOT increment: with reversed motion, the flip is not supported.
    assert flip_counters_backward == [0], (
        f"With motion_is_reversed=True the flip should not be supported, but the "
        f"counter became {flip_counters_backward}."
    )

    flip_counters_forward = [0]
    out_forward = smooth_orientation(
        r=0,
        theta=0.0,
        speed=10.0,
        p=_params(DIRECTED_ORIENT_FLIP_CONFIDENCE=0.0),
        orientation_last=[math.pi],
        position_deques=[deque_r],
        directed_heading=True,
        orient_confidence=1.0,
        heading_flip_counters=flip_counters_forward,
        motion_is_reversed=False,
    )
    # Same pi output (hysteresis blocks the flip), but counter incremented.
    assert abs(((out_forward - math.pi + math.pi) % (2 * math.pi)) - math.pi) < 1e-6
    assert flip_counters_forward == [1], (
        f"With motion_is_reversed=False the flip should be supported and the counter "
        f"should increment to 1; got {flip_counters_forward}."
    )
