"""Conservative motion-aware fragment stitcher.

Covers the layered gates added to ``_stitch_broken_trajectory_fragments``:
motion prediction, raw-jump cap, heading veto, density tightening, single-option
margin test, and symmetric nearest-neighbour check. Each test isolates a single
gate so a regression in one gate fails one test rather than a generic
"stitching broke" assertion.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from hydra_suite.core.post.processing import _stitch_broken_trajectory_fragments


def _row(frame: int, x: float, y: float, theta: float = 0.0) -> dict:
    return {"FrameID": frame, "X": x, "Y": y, "Theta": theta}


def _linear_fragment(frames, x0, y0, vx, vy, theta=0.0) -> pd.DataFrame:
    """Build a fragment with constant velocity so end_velocity prediction is exact."""
    return pd.DataFrame(
        [
            _row(f, x0 + vx * (f - frames[0]), y0 + vy * (f - frames[0]), theta)
            for f in frames
        ]
    )


def test_motion_prediction_stitches_fast_fragment_across_gap() -> None:
    """A fast-moving fragment with a 4-frame gap whose successor lands exactly on
    the predicted position must stitch — the legacy raw-distance gate would have
    missed this because |end_pos - start_pos| ≫ agreement_distance."""
    a = _linear_fragment(range(0, 10), x0=0.0, y0=100.0, vx=10.0, vy=0.0)
    # Gap of 4 frames; successor starts where motion prediction lands.
    b = _linear_fragment(range(14, 24), x0=140.0, y0=100.0, vx=10.0, vy=0.0)
    # agreement_distance is small; raw |end_a - start_b| = 50 px ≫ 15
    out = _stitch_broken_trajectory_fragments(
        [a, b],
        agreement_distance=15.0,
        max_gap=5,
        max_vel_break=200.0,
    )
    assert len(out) == 1, "motion-aware gate should connect the two fragments"


def test_raw_jump_cap_blocks_implausible_displacement() -> None:
    """Even when motion prediction *says* a far-away successor is consistent,
    the absolute raw-jump cap (max_vel_break × min(Δframes, 4)) must block
    teleport-scale displacements."""
    a = _linear_fragment(range(0, 10), x0=0.0, y0=100.0, vx=10.0, vy=0.0)
    # Successor exactly on prediction at gap=20 (200px away), but max_vel_break=10
    # caps the absolute envelope to 10 × min(20, 4) = 40 px → must reject.
    b = _linear_fragment(range(30, 40), x0=300.0, y0=100.0, vx=10.0, vy=0.0)
    out = _stitch_broken_trajectory_fragments(
        [a, b],
        agreement_distance=15.0,
        max_gap=25,
        max_vel_break=10.0,
    )
    assert len(out) == 2, "raw-jump cap should reject teleport-scale stitches"


def test_heading_veto_blocks_opposing_directions() -> None:
    """Two fragments moving in opposite directions but ending/starting near each
    other must NOT stitch — heading mismatch identifies them as different
    animals even when spatial residual is tiny."""
    # Fragment a heads east at the end.
    a = _linear_fragment(range(0, 10), x0=0.0, y0=100.0, vx=5.0, vy=0.0, theta=0.0)
    # Fragment b heads west at the start; positions are spatially compatible.
    b_frames = list(range(11, 21))
    b = _linear_fragment(
        b_frames, x0=50.0, y0=100.0, vx=-5.0, vy=0.0, theta=float(np.pi)
    )
    out = _stitch_broken_trajectory_fragments(
        [a, b],
        agreement_distance=20.0,
        max_gap=5,
        max_vel_break=100.0,
        heading_gate_rad=float(np.deg2rad(60.0)),
    )
    assert len(out) == 2, "opposing headings must be vetoed"


def test_heading_veto_skipped_when_endpoints_stationary() -> None:
    """Stationary fragments should not be punished by heading checks — the gate
    is only meaningful when both endpoints are actually moving."""
    a = _linear_fragment(range(0, 10), x0=100.0, y0=100.0, vx=0.0, vy=0.0, theta=0.0)
    # Stationary too, opposite reported heading — should still stitch since both speeds are zero.
    b = _linear_fragment(
        range(11, 21), x0=100.0, y0=100.0, vx=0.0, vy=0.0, theta=float(np.pi)
    )
    out = _stitch_broken_trajectory_fragments(
        [a, b],
        agreement_distance=10.0,
        max_gap=5,
        max_vel_break=100.0,
        heading_gate_rad=float(np.deg2rad(30.0)),
    )
    assert len(out) == 1, "stationary endpoints should bypass heading veto"


def test_margin_test_rejects_ambiguous_pair() -> None:
    """When two roughly-equal candidates compete for the same predecessor, leave
    the fragments split — that's exactly the case where identity should
    arbitrate, not geometry."""
    a = _linear_fragment(range(0, 10), x0=0.0, y0=100.0, vx=5.0, vy=0.0)
    # Both successors offset symmetrically off the prediction (55, 100) so
    # neither dominates: residuals 2 and 4 ⇒ score ratio ≈ 0.63 > 0.5 ⇒ reject.
    b1 = _linear_fragment(range(11, 21), x0=55.0, y0=102.0, vx=5.0, vy=0.0)
    b2 = _linear_fragment(range(11, 21), x0=55.0, y0=104.0, vx=5.0, vy=0.0)
    out = _stitch_broken_trajectory_fragments(
        [a, b1, b2],
        agreement_distance=15.0,
        max_gap=5,
        max_vel_break=100.0,
        single_option_margin=0.5,
    )
    assert len(out) == 3, "ambiguous candidates must keep fragments split"


def test_margin_test_accepts_unambiguous_winner() -> None:
    """A clearly-better successor (much smaller residual than runner-up) should
    pass the margin test."""
    a = _linear_fragment(range(0, 10), x0=0.0, y0=100.0, vx=5.0, vy=0.0)
    # b1 lands exactly on prediction (residual ≈ 0); b2 lands far (residual ≈ 14).
    b1 = _linear_fragment(range(11, 21), x0=55.0, y0=100.0, vx=5.0, vy=0.0)
    b2 = _linear_fragment(range(11, 21), x0=55.0, y0=114.0, vx=5.0, vy=0.0)
    out = _stitch_broken_trajectory_fragments(
        [a, b1, b2],
        agreement_distance=15.0,
        max_gap=5,
        max_vel_break=100.0,
        single_option_margin=0.5,
    )
    # Expect a stitched into b1, b2 left alone → 2 trajectories.
    assert (
        len(out) == 2
    ), f"unambiguous winner should be accepted; got {len(out)} trajectories"


def test_density_tightening_blocks_stitch_in_crowded_region() -> None:
    """When 2+ other fragments are near the gap midpoint, the spatial gate
    halves — a stitch that would pass in open space is rejected in a cluster."""
    # Fragment pair: residual ~9 px after motion prediction (no movement).
    a = _linear_fragment(range(0, 10), x0=100.0, y0=100.0, vx=0.0, vy=0.0)
    b = _linear_fragment(range(11, 21), x0=109.0, y0=100.0, vx=0.0, vy=0.0)
    # Two unrelated fragments crowd the midpoint at the gap frame.
    crowd1 = _linear_fragment(range(8, 14), x0=104.0, y0=102.0, vx=0.0, vy=0.0)
    crowd2 = _linear_fragment(range(8, 14), x0=106.0, y0=98.0, vx=0.0, vy=0.0)
    # Open-space variant — residual=9 ≤ 10 → accepted.
    open_out = _stitch_broken_trajectory_fragments(
        [a.copy(), b.copy()],
        agreement_distance=10.0,
        max_gap=5,
        max_vel_break=100.0,
        density_radius_multiplier=5.0,
        density_tighten_factor=0.5,
        single_option_margin=1.0,
    )
    assert len(open_out) == 1, "open-space stitch should pass"
    # Crowded variant — effective gate halves to 5 → 9 > 5 → rejected.
    crowded_out = _stitch_broken_trajectory_fragments(
        [a.copy(), b.copy(), crowd1, crowd2],
        agreement_distance=10.0,
        max_gap=5,
        max_vel_break=100.0,
        density_radius_multiplier=5.0,
        density_tighten_factor=0.5,
        single_option_margin=1.0,
    )
    # a+b should remain split; crowd1/crowd2 are independent fragments.
    a_b_present = sum(1 for t in crowded_out if int(t["FrameID"].iat[0]) in (0,)) + sum(
        1 for t in crowded_out if int(t["FrameID"].iat[0]) == 11
    )
    assert a_b_present == 2, "density tightening should block the crowded stitch"


def test_symmetric_nn_blocks_asymmetric_grab() -> None:
    """If A's best successor is B but B's best predecessor is some other A',
    reject the stitch — neither side claims the other unambiguously."""
    # A and A' both end near the same spot; B starts there.  Both A and A'
    # see B as their best successor (low residuals), but B prefers whichever
    # of A/A' has the lower residual.  The "loser" of A/A' should NOT stitch
    # to B even though B is its best.
    a = _linear_fragment(
        range(0, 10), x0=0.0, y0=100.0, vx=5.0, vy=0.0
    )  # ends (45,100)
    a_prime = _linear_fragment(
        range(0, 10), x0=0.0, y0=102.0, vx=5.0, vy=0.0
    )  # ends (45,102), better aligned
    b = _linear_fragment(
        range(11, 21), x0=55.0, y0=102.0, vx=5.0, vy=0.0
    )  # starts (55,102)
    out = _stitch_broken_trajectory_fragments(
        [a, a_prime, b],
        agreement_distance=15.0,
        max_gap=5,
        max_vel_break=100.0,
        single_option_margin=1.0,  # disable margin gate so symmetric NN is the deciding factor
    )
    # a_prime is B's best predecessor → only a_prime stitches with b.
    # a has no other valid successor and remains alone.
    # Expected: 2 trajectories: stitched (a_prime+b), and a.
    assert len(out) == 2, f"symmetric-NN should keep loser separate; got {len(out)}"


def test_existing_signature_compatibility_with_default_kwargs() -> None:
    """Legacy callers that pass only ``agreement_distance``, ``max_gap``, and
    ``identity_gates_stitching`` must continue to work."""
    a = pd.DataFrame([_row(f, 100.0 + f, 100.0) for f in range(0, 10)])
    b = pd.DataFrame([_row(f, 100.0 + f, 100.0) for f in range(11, 20)])
    out = _stitch_broken_trajectory_fragments(
        [a, b],
        agreement_distance=20.0,
        max_gap=5,
        identity_gates_stitching=True,
    )
    # Constant-velocity fragments at the same y — should stitch with default
    # max_vel_break=100 (cap >> needed).
    assert len(out) == 1
