import math

import numpy as np
import pytest

from hydra_suite.core.canonicalization.geometry import (
    CanonicalGeometry,
    canonical_affine,
    overflow_ratio,
)


def obb(cx, cy, major, minor, theta):
    """(4,2) OBB corners for a box centred at (cx, cy), rotated by theta."""
    hw, hh = major / 2.0, minor / 2.0
    base = np.array([[-hw, -hh], [hw, -hh], [hw, hh], [-hw, hh]], dtype=np.float32)
    c, s = math.cos(theta), math.sin(theta)
    rot = np.array([[c, -s], [s, c]], dtype=np.float32)
    return (base @ rot.T + np.array([cx, cy], dtype=np.float32)).astype(np.float32)


def test_canvas_derives_major_axis_from_geometric_mean():
    # REFERENCE_BODY_SIZE is sqrt(major*minor); major = body * sqrt(ar).
    g = CanonicalGeometry.from_reference(
        reference_body_px=20.0, aspect_ratio=4.0, margin=1.5
    )
    # major = 20 * 2 = 40; canvas_w = 40 * 1.5 = 60 -> even
    assert g.canvas_w == 60
    # canvas_h = canvas_w / ar = 60 / 4 = 15, rounded up to the nearest even
    # value (the module's stated "canvas dimensions are even" invariant).
    assert g.canvas_h == 16


def test_canvas_dimensions_are_even():
    g = CanonicalGeometry.from_reference(
        reference_body_px=17.3, aspect_ratio=2.44, margin=1.37
    )
    assert g.canvas_w % 2 == 0
    assert g.canvas_h % 2 == 0


def test_affine_linear_part_is_a_pure_rotation():
    """The defining property: Layer 1 scales nothing."""
    g = CanonicalGeometry.from_reference(20.0, 2.44, 1.5)
    for ar in (1.1, 1.8, 2.44, 3.5, 6.0):
        corners = obb(100.0, 80.0, 40.0, 40.0 / ar, math.radians(37.0))
        M, _, _ = canonical_affine(corners, g)
        A = np.asarray(M)[:, :2]
        sv = np.linalg.svd(A, compute_uv=False)
        np.testing.assert_allclose(sv, [1.0, 1.0], atol=1e-6)


def test_affine_is_invariant_to_animal_size():
    """Same centre and angle, different extents -> identical affine."""
    g = CanonicalGeometry.from_reference(20.0, 2.44, 1.5)
    small = obb(100.0, 80.0, 20.0, 8.0, 0.4)
    large = obb(100.0, 80.0, 60.0, 24.0, 0.4)
    m_small, _, _ = canonical_affine(small, g)
    m_large, _, _ = canonical_affine(large, g)
    # atol relaxed from 1e-9: obb() builds corners in float32, so the two
    # centroids/angles carry ~1e-6 absolute float32 rounding noise before
    # canonical_affine ever sees them.
    np.testing.assert_allclose(m_small, m_large, atol=1e-5)


def test_centroid_maps_to_canvas_centre():
    g = CanonicalGeometry.from_reference(20.0, 2.44, 1.5)
    corners = obb(123.0, 45.0, 30.0, 12.0, 1.1)
    M, _, _ = canonical_affine(corners, g)
    centre = np.asarray(M) @ np.array([123.0, 45.0, 1.0])
    np.testing.assert_allclose(centre, [g.canvas_w / 2.0, g.canvas_h / 2.0], atol=1e-6)


def test_theta_recovers_the_major_axis_angle():
    g = CanonicalGeometry.from_reference(20.0, 2.44, 1.5)
    for deg in (0.0, 30.0, 91.0, 179.0):
        corners = obb(50.0, 50.0, 40.0, 16.0, math.radians(deg))
        _, theta, _ = canonical_affine(corners, g)
        assert math.isclose(
            math.cos(2 * theta), math.cos(2 * math.radians(deg)), abs_tol=1e-5
        )


def test_clipping_is_reported_not_absorbed():
    g = CanonicalGeometry.from_reference(20.0, 2.0, 1.3)
    fits = obb(80.0, 80.0, 20.0, 10.0, 0.0)
    _, _, clipped_small = canonical_affine(fits, g)
    assert clipped_small is False

    huge = obb(80.0, 80.0, 400.0, 200.0, 0.0)
    _, _, clipped_big = canonical_affine(huge, g)
    assert clipped_big is True
    assert overflow_ratio(huge, g) > 1.0
    assert overflow_ratio(fits, g) <= 1.0


def test_overflow_ratio_and_clipped_are_not_margin_invariant():
    """Regression guard: overflow_ratio/clipped must report ACTUAL containment.

    Before the fix, ``margin`` appeared in both the numerator (major/minor *
    margin) and the canvas denominator (canvas_w/h already baked in margin),
    so it cancelled and the reported ratio barely moved across margins even
    though the animal genuinely fit at some margins and not others. Pin the
    corrected behaviour: same detection, larger margin -> larger canvas ->
    strictly lower overflow_ratio, and clipped flips True -> False once the
    canvas is big enough.
    """
    # major=45, minor=22 fixed detection extent.
    small_margin = CanonicalGeometry.from_reference(
        reference_body_px=20.0, aspect_ratio=2.0, margin=1.3
    )
    large_margin = CanonicalGeometry.from_reference(
        reference_body_px=20.0, aspect_ratio=2.0, margin=2.0
    )
    # canvas_w/h: margin=1.3 -> (38, 20); margin=2.0 -> (58, 30).
    assert small_margin.canvas_wh == (38, 20)
    assert large_margin.canvas_wh == (58, 30)

    corners = obb(80.0, 80.0, 45.0, 22.0, 0.0)

    ratio_small_margin = overflow_ratio(corners, small_margin)
    ratio_large_margin = overflow_ratio(corners, large_margin)

    # The core property the old formula got wrong: raising the margin must
    # visibly LOWER the reported ratio for the same detection.
    assert ratio_large_margin < ratio_small_margin
    # Sanity on the actual numbers (hand-computed from the new formula:
    # max(major/canvas_w, minor/canvas_h), no margin factor).
    assert ratio_small_margin == pytest.approx(max(45.0 / 38.0, 22.0 / 20.0))
    assert ratio_large_margin == pytest.approx(max(45.0 / 58.0, 22.0 / 30.0))

    _, _, clipped_small_margin = canonical_affine(corners, small_margin)
    _, _, clipped_large_margin = canonical_affine(corners, large_margin)
    assert clipped_small_margin is True
    assert ratio_small_margin > 1.0
    assert clipped_large_margin is False
    assert ratio_large_margin <= 1.0


def test_degenerate_obb_raises():
    g = CanonicalGeometry.from_reference(20.0, 2.0, 1.3)
    degenerate = np.zeros((4, 2), dtype=np.float32)
    with pytest.raises(ValueError):
        canonical_affine(degenerate, g)


def test_round_trips_through_dict():
    g = CanonicalGeometry.from_reference(20.0, 2.44, 1.5)
    assert CanonicalGeometry.from_dict(g.to_dict()) == g
