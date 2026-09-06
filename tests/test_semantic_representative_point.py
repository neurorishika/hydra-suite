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
    """The reason the gate exists. Widening admissibility must NOT re-open it."""
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
