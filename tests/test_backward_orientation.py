"""Regression tests for backward-pass orientation handling."""

import math

from hydra_suite.core.identity.geometry import collapse_obb_axis_theta


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
