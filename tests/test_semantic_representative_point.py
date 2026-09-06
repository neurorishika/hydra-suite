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
    """A blob must not earn recall credit.

    Provenance note, because it is easy to misread this test: the rejections
    below are done by the AREA BAND and by ``min_quality``, not by
    containment -- mutation testing confirms deleting containment leaves this
    passing. It is a regression guard on the OUTCOME (`shape_prior`'s
    mistargeting fix survives the new IoU route), not evidence for the
    containment gate. The routes themselves are pinned separately, below.

    (An IoU second admissibility route briefly existed and was removed; the
    reason a blob could never have used it is recorded at the deletion site
    in ``match_one_to_one``.)
    """
    labels = [_sq(100, 100), _sq(130, 100), _sq(160, 100)]
    blob = _sq(130, 100, side=400.0)
    band = fit_area_band(labels)
    assert match_one_to_one([blob], labels) == []
    assert match_one_to_one([blob], labels, area_band=band) == []
    # And a moderately oversized blob claims at most one label, never two.
    medium = _sq(130, 100, side=70.0)
    assert len(match_one_to_one([medium], labels)) <= 1


def test_an_uncontained_prediction_is_not_admitted():
    labels = [_sq(100, 100)]
    assert match_one_to_one([_sq(400, 400)], labels) == []
    # Touching but barely overlapping, and neither point inside the other.
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
# Containment, pinned. Mutation testing drove this section: with an IoU
# second route present (since REMOVED -- see the comment in
# `match_one_to_one`), BOTH deleting containment and reverting the
# representative point to the vertex mean inside the matcher were silently
# survivable, because the IoU route rescued every case. The test below kills
# both mutants and must keep doing so.
# ---------------------------------------------------------------------------


def test_containment_admits_a_low_overlap_curved_pair_only_via_the_fixed_point():
    """Pins the containment gate AND the point fix together.

    Two nearly-coincident curved outlines whose overlap is LOW (IoU 0.48) and
    whose vertex means each fall in the other's hollow -- so the pre-fix
    matcher scored this a miss -- but whose representative points are
    contained, so the shipped matcher finds it.

    This test kills two independent mutants: deleting containment (which the
    pre-existing quality-floor tests do not catch, as they reject on quality
    regardless), and reverting the representative point to the vertex mean
    inside ``match_one_to_one``.
    """
    from hydra_suite.core.inference.masks import polygon_iou
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
    # Low overlap: nothing overlap-based could admit this pair.
    assert polygon_iou(pred, label) < 0.5
    assert match_quality(pred, label) > 0.1
    # The pre-fix point fails in BOTH directions -- this is the 15.9 % case.
    assert not _contains(label, _vertex_mean(pred))
    assert not _contains(pred, _vertex_mean(label))
    # The fixed point succeeds.
    assert _contains(label, representative_point(pred)) or _contains(
        pred, representative_point(label)
    )

    assert match_one_to_one([pred], [label]) == [(0, 0)]


def test_a_high_overlap_but_uncontained_pair_is_REJECTED():
    """Containment must still be able to say NO. Kills delete-containment.

    This is the exact pair that the retired IoU second route was added to
    admit: two overlapping arcs at IoU 0.73 whose poles each land in an arm
    the other does not cover. Quality is ~0.88, far above ``min_quality``,
    and both polygons sit inside any sane area band -- so containment is the
    ONLY thing that can reject it, and deleting the gate makes this pass.

    Keeping it as a REJECTION test is deliberate: when the IoU route was
    removed (see the comment in ``match_one_to_one``) this pair's expected
    outcome flipped from match to no-match, so the coverage moves with the
    decision instead of being dropped along with it.
    """
    from hydra_suite.core.inference.masks import polygon_iou
    from hydra_suite.core.inference.semantic.shape_prior import match_quality

    label = _arc(100.0, 100.0, 17.1525, 29.5458, 4.7179, 4.7179 + 2.6925, n=48)
    pred = _arc(
        101.6406,
        99.6033,
        17.1525 * 0.9271,
        29.5458 * 0.9271,
        4.7179 - 0.5150,
        4.7179 - 0.5150 + 2.6925 * 1.1429,
        n=48,
    )
    assert polygon_iou(pred, label) > 0.7  # high overlap ...
    assert match_quality(pred, label) > 0.5  # ... and high quality ...
    assert not _contains(label, representative_point(pred))
    assert not _contains(pred, representative_point(label))
    # ... but uncontained, so not a find.
    assert match_one_to_one([pred], [label]) == []


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


def test_one_corrupted_vertex_in_an_otherwise_valid_outline_does_not_crash():
    """The realistic malformed label, and the one that actually reaches the
    second crash entrance.

    ``cv2.pointPolygonTest`` tolerates a NaN/inf vertex and still reports the
    label's point as INSIDE this outline -- so the pair passes containment
    and goes on to ``match_quality`` -> ``polygon_iou``, which raises
    (`utils/polygon_iou.py:59`). Without the ``_is_finite`` filter in
    ``match_one_to_one`` this is a hard crash on a GUI-reachable path, not a
    silent non-match.
    """
    label = _sq(100, 100, side=20.0)
    for bad in (np.nan, np.inf, -np.inf):
        pred = np.array(
            [[95, 95], [105, 95], [105, 105], [95, 105], [bad, bad]],
            dtype=np.float64,
        )
        # Precondition: containment does NOT reject it, so the filter is the
        # only thing standing between this input and the raise.
        assert _contains(pred, representative_point(label))
        assert match_one_to_one([pred], [label]) == []
        assert match_one_to_one([label], [pred]) == []
