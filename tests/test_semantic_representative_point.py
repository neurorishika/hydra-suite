"""The inside-guaranteed representative point, and the matcher gate on top.

Covers the measured defect (a curved outline whose vertex-mean is outside),
the dense-cluster protection the containment gate exists to provide, and
degenerate inputs.
"""

import numpy as np
import pytest

from hydra_suite.core.inference.semantic.calibration import (
    _contains,
    _vertex_mean,
    match_one_to_one,
    representative_point,
)
from hydra_suite.core.inference.semantic.shape_prior import fit_area_band


def _arc(cx, cy, r_in, r_out, a0, a1, n=64):
    """A crescent / arc band: the classic shape whose vertex mean and area
    centroid both fall in the hollow."""
    ang = np.linspace(a0, a1, n)
    outer = np.stack([cx + r_out * np.cos(ang), cy + r_out * np.sin(ang)], axis=1)
    inner = np.stack(
        [cx + r_in * np.cos(ang[::-1]), cy + r_in * np.sin(ang[::-1])], axis=1
    )
    return np.concatenate([outer, inner], axis=0).astype(np.float32)


def _u_shape():
    return np.array(
        [[0, 0], [10, 0], [10, 40], [30, 40], [30, 0], [40, 0], [40, 50], [0, 50]],
        dtype=np.float32,
    )


def _s_curve(n=80):
    t = np.linspace(0.0, 4.0 * np.pi, n)
    spine = np.stack([t * 6.0, 30.0 * np.sin(t)], axis=1)
    normal = np.stack([-np.gradient(spine[:, 1]), np.gradient(spine[:, 0])], axis=1)
    normal /= np.linalg.norm(normal, axis=1, keepdims=True)
    return np.concatenate([spine + 3.0 * normal, (spine - 3.0 * normal)[::-1]]).astype(
        np.float32
    )


def _thin_arc():
    return _arc(200.0, 200.0, 48.0, 51.0, 0.2, np.pi * 1.4, n=90)


def _self_touching():
    # A figure-eight-ish outline that crosses itself.
    return np.array([[0, 0], [40, 40], [0, 40], [40, 0], [20, 20]], dtype=np.float32)


_CURVED = {
    "crescent": _arc(100.0, 100.0, 20.0, 34.0, 0.0, np.pi * 1.3),
    "u_shape": _u_shape(),
    "s_curve": _s_curve(),
    "thin_arc": _thin_arc(),
    "self_touching": _self_touching(),
}


@pytest.mark.parametrize("name", sorted(_CURVED))
def test_representative_point_is_inside_for_adversarial_shapes(name):
    poly = _CURVED[name]
    assert _contains(poly, representative_point(poly))


def test_the_measured_defect_vertex_mean_is_outside_where_the_fix_is_inside():
    """The 15.9 % case, reproduced in miniature.

    A curved, elongated, densely sampled outline -- the shape of a real ant
    label -- does not contain its own vertex mean. The replacement point is
    inside, so the containment gate no longer vetoes it.
    """
    poly = _CURVED["crescent"]
    assert not _contains(poly, _vertex_mean(poly))
    assert _contains(poly, representative_point(poly))


def test_a_near_perfect_mask_over_a_curved_label_is_no_longer_a_miss():
    """The verified case from the measurement: high IoU, matching area,
    claimed by no other label, previously scored a MISS because neither
    vertex mean landed inside the other polygon."""
    label = _arc(300.0, 300.0, 20.0, 34.0, 0.0, np.pi * 1.3)
    pred = _arc(300.6, 300.4, 20.0, 34.2, 0.0, np.pi * 1.3)
    assert not _contains(label, _vertex_mean(pred))
    assert not _contains(pred, _vertex_mean(label))
    assert match_one_to_one([pred], [label]) == [(0, 0)]


def _sq(cx, cy, side=20.0):
    h = side / 2.0
    return np.array(
        [[cx - h, cy - h], [cx + h, cy - h], [cx + h, cy + h], [cx - h, cy + h]],
        dtype=np.float32,
    )


def test_dense_cluster_blob_still_cannot_steal_a_neighbours_label():
    """Widening admissibility must not let a blob earn credit.

    Provenance note, because it is easy to misread this test: the rejections
    below are done by the AREA BAND and by ``min_quality``, not by
    containment -- mutation testing confirms deleting containment leaves this
    passing. It is a regression guard on the OUTCOME (`shape_prior`'s
    mistargeting fix survives the new IoU route), not evidence for the
    containment gate. The routes themselves are pinned separately, below.

    The IoU route cannot rescue a blob for a geometric reason: IoU >= 0.5
    implies min(area) / max(area) >= 0.5, so a region spanning two labels is
    at least 2x either one and can never reach the threshold.
    """
    labels = [_sq(100, 100), _sq(130, 100), _sq(160, 100)]
    blob = _sq(130, 100, side=400.0)
    band = fit_area_band(labels)
    assert match_one_to_one([blob], labels) == []
    assert match_one_to_one([blob], labels, area_band=band) == []
    # And a moderately oversized blob claims at most one label, never two.
    medium = _sq(130, 100, side=70.0)
    assert len(match_one_to_one([medium], labels)) <= 1


def test_iou_route_does_not_admit_a_low_overlap_far_prediction():
    labels = [_sq(100, 100)]
    assert match_one_to_one([_sq(400, 400)], labels) == []
    # Touching but barely overlapping: below ADMISSIBLE_IOU and uncontained.
    assert match_one_to_one([_sq(118, 100)], labels) == []


@pytest.mark.parametrize(
    "poly",
    [
        np.zeros((0, 2), dtype=np.float32),
        np.array([[5.0, 5.0]], dtype=np.float32),
        np.array([[5.0, 5.0], [9.0, 5.0]], dtype=np.float32),
        np.array([[0, 0], [10, 0], [4, 7]], dtype=np.float32),
        np.array([[3, 3], [3, 3], [3, 3], [3, 3]], dtype=np.float32),
        np.array([[0, 0], [10, 0], [20, 0], [10, 0]], dtype=np.float32),
        np.array([[0.0, 0.0], [0.4, 0.0], [0.4, 0.2], [0.0, 0.2]], dtype=np.float32),
    ],
    ids=[
        "empty",
        "one_point",
        "two_points",
        "triangle",
        "duplicate_vertices",
        "zero_area_collinear",
        "subpixel",
    ],
)
def test_degenerate_inputs_return_a_finite_point_and_never_raise(poly):
    pt = representative_point(poly)
    assert pt.shape == (2,)
    assert np.all(np.isfinite(pt))
    if poly.shape[0] >= 3:
        # `_contains` accepts on-boundary (>= 0), which is the definition the
        # gate itself uses -- so even a zero-area shape yields an admissible
        # point rather than one the gate would immediately veto.
        assert _contains(poly, pt)


def test_property_generated_polygons_always_contain_their_representative_point():
    rng = np.random.default_rng(20260906)
    for _ in range(300):
        n = int(rng.integers(3, 40))
        ang = np.sort(rng.uniform(0.0, 2.0 * np.pi, n))
        # Wildly varying radii produce star, crescent and comb-like outlines.
        radius = rng.uniform(1.0, 60.0, n)
        cx, cy = rng.uniform(-500.0, 500.0, 2)
        poly = np.stack(
            [cx + radius * np.cos(ang), cy + radius * np.sin(ang)], axis=1
        ).astype(np.float32)
        assert _contains(poly, representative_point(poly)), poly.tolist()


# ---------------------------------------------------------------------------
# The two admissibility routes, pinned INDEPENDENTLY. Mutation testing showed
# that deleting the IoU route left every other test in the suite passing, so
# each route below is exercised with the OTHER one disabled.
# ---------------------------------------------------------------------------


def _crescent_pair():
    """A natural, uncontained, high-IoU pair.

    Found by search over the crescent family and pinned by literal parameters
    (the properties asserted in the test are the contract, not the numbers):
    two overlapping arcs whose poles of inaccessibility each land in an arm
    the other polygon does not cover, while the shared body keeps IoU well
    above ``ADMISSIBLE_IOU``. This is the geometry the IoU route exists for --
    a correct mask that traces slightly differently from its label.
    """
    a = _arc(100.0, 100.0, 17.1525, 29.5458, 4.7179, 4.7179 + 2.6925, n=48)
    b = _arc(
        101.6406,
        99.6033,
        17.1525 * 0.9271,
        29.5458 * 0.9271,
        4.7179 - 0.5150,
        4.7179 - 0.5150 + 2.6925 * 1.1429,
        n=48,
    )
    return a, b


def test_a_high_iou_pair_matches_with_neither_point_contained():
    from hydra_suite.core.inference.masks import polygon_iou
    from hydra_suite.core.inference.semantic.calibration import ADMISSIBLE_IOU

    label, pred = _crescent_pair()
    # The premise: containment cannot save this pair in EITHER direction.
    assert not _contains(label, representative_point(pred))
    assert not _contains(pred, representative_point(label))
    assert polygon_iou(pred, label) >= ADMISSIBLE_IOU
    # ... so a match here is the IoU route and nothing else.
    assert match_one_to_one([pred], [label]) == [(0, 0)]


def test_the_iou_route_boundary_is_exactly_admissible_iou(monkeypatch):
    """Pins ADMISSIBLE_IOU as a VALUE, in both directions.

    Containment is forced off, so admissibility is the IoU route alone and
    the match/no-match boundary must sit exactly at the constant. Widening
    the constant (to 0.3, say) would admit pairs this asserts are rejected --
    which matters, because IoU >= 0.5 is what bounds the area ratio to
    0.5x-2x and so makes a two-label blob geometrically unable to use this
    route at all. At 0.3 the bound is 0.3x-3.3x and a blob gets in.
    """
    from hydra_suite.core.inference.masks import polygon_iou
    from hydra_suite.core.inference.semantic import calibration

    monkeypatch.setattr(calibration, "_contains", lambda _poly, _pt: False)

    label = _arc(100.0, 100.0, 17.1525, 29.5458, 4.7179, 4.7179 + 2.6925, n=48)
    checked_above = checked_below = 0
    for scale in np.arange(0.60, 1.26, 0.02):
        pred = _arc(
            101.6406,
            99.6033,
            17.1525 * 0.9271 * scale,
            29.5458 * 0.9271 * scale,
            4.7179 - 0.5150,
            4.7179 - 0.5150 + 2.6925 * 1.1429,
            n=48,
        )
        iou = polygon_iou(pred, label)
        if abs(iou - calibration.ADMISSIBLE_IOU) < 1e-6:
            continue  # exactly on the boundary: not a meaningful assertion
        matched = bool(match_one_to_one([pred], [label]))
        if iou >= calibration.ADMISSIBLE_IOU:
            assert matched, f"IoU {iou:.4f} should be admissible"
            checked_above += 1
        else:
            assert not matched, f"IoU {iou:.4f} should NOT be admissible"
            checked_below += 1
    assert checked_above > 3 and checked_below > 3


def test_the_containment_route_still_admits_a_low_iou_curved_pair(monkeypatch):
    """Pins the containment route AND the point fix, with the IoU route off.

    Two nearly-coincident curved outlines: IoU is below ADMISSIBLE_IOU, so
    the overlap route cannot admit them, and the vertex mean of each falls in
    the OTHER's hollow, so the pre-fix matcher could not either. Only
    ``representative_point`` + containment admits this pair.

    This is the test that kills two independent mutants: deleting containment
    (which the pre-existing quality-floor tests do not catch, since they
    reject on quality regardless), and reverting the representative point to
    the vertex mean inside ``match_one_to_one`` (which every other test in
    this file survives, because the IoU route silently rescues them).
    """
    from hydra_suite.core.inference.masks import polygon_iou
    from hydra_suite.core.inference.semantic import calibration
    from hydra_suite.core.inference.semantic.shape_prior import match_quality

    label = _arc(100.0, 100.0, 23.1741, 28.3131, 2.3608, 2.3608 + 5.1467, n=48)
    pred = _arc(
        98.4076,
        97.4916,
        23.1741 * 0.9961,
        28.3131 * 0.9961,
        2.3608 - 0.0360,
        2.3608 - 0.0360 + 5.1467 * 0.9961,
        n=48,
    )
    assert polygon_iou(pred, label) < calibration.ADMISSIBLE_IOU
    assert match_quality(pred, label) > 0.1
    # The pre-fix point fails in BOTH directions -- this is the 15.9 % case.
    assert not _contains(label, _vertex_mean(pred))
    assert not _contains(pred, _vertex_mean(label))
    # The fixed point succeeds.
    assert _contains(label, representative_point(pred)) or _contains(
        pred, representative_point(label)
    )

    # Disable the IoU route outright, so only containment can admit the pair.
    # Only calibration's own reference is patched, so `match_quality`'s
    # internal overlap term is untouched.
    monkeypatch.setattr(calibration, "polygon_iou", lambda _a, _b: 0.0)
    assert match_one_to_one([pred], [label]) == [(0, 0)]


# ---------------------------------------------------------------------------
# Non-finite vertices: a GUI-reachable malformed label must NOT crash.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_a_non_finite_vertex_never_raises_and_never_matches(bad):
    poly = np.array(
        [[0.0, 0.0], [20.0, 0.0], [20.0, bad], [0.0, 20.0]], dtype=np.float64
    )
    pt = representative_point(poly)  # must not raise
    assert pt.shape == (2,)
    label = _sq(10, 10, side=20.0)
    # No crash, and a malformed polygon is simply not a find -- the same
    # outcome the pre-fix vertex mean produced silently.
    assert match_one_to_one([poly], [label]) == []
    assert match_one_to_one([label], [poly]) == []


def test_all_non_finite_polygon_is_tolerated():
    poly = np.full((6, 2), np.nan, dtype=np.float64)
    assert representative_point(poly).shape == (2,)
    assert match_one_to_one([poly], [_sq(10, 10)]) == []
