"""Shape/area prior: the gate that stops SAM3 masks being mistargeted.

Calibration used to admit a match on containment alone, so an arena-sized
blob EARNED recall credit and a leg-sized fragment counted as a find. These
tests pin the band and the quality score that replace that.
"""

import numpy as np
import pytest

from hydra_suite.core.inference.semantic.shape_prior import (
    AreaBand,
    aspect_ratio,
    fit_area_band,
    in_band,
    match_quality,
    polygon_area,
)


def _sq(cx, cy, side=20.0):
    h = side / 2.0
    return np.array(
        [[cx - h, cy - h], [cx + h, cy - h], [cx + h, cy + h], [cx - h, cy + h]],
        dtype=np.float32,
    )


def _rect(cx, cy, w, h):
    return np.array(
        [
            [cx - w / 2, cy - h / 2],
            [cx + w / 2, cy - h / 2],
            [cx + w / 2, cy + h / 2],
            [cx - w / 2, cy + h / 2],
        ],
        dtype=np.float32,
    )


def test_polygon_area_matches_the_analytic_area():
    assert polygon_area(_sq(0, 0, side=20.0)) == pytest.approx(400.0, rel=1e-6)


def test_fit_returns_none_without_labels():
    assert fit_area_band([]) is None


def test_every_ground_truth_label_falls_inside_its_own_band():
    """The invariant that makes a hard gate safe.

    A band fitted from the labels must never exclude a label -- otherwise
    calibration would score the user's own ground truth as impossible.
    """
    rng = np.random.default_rng(0)
    labels = [_sq(100, 100, side=float(s)) for s in rng.uniform(12, 60, size=40)]
    band = fit_area_band(labels)
    assert band is not None
    assert all(in_band(g, band) for g in labels)


def test_band_rejects_an_arena_sized_blob_and_a_leg_sized_fragment():
    labels = [_sq(100, 100, side=20.0) for _ in range(10)]
    band = fit_area_band(labels)
    assert not in_band(_sq(500, 500, side=600.0), band)  # half the arena
    assert not in_band(_sq(100, 100, side=4.0), band)  # one leg
    assert in_band(_sq(100, 100, side=26.0), band)  # a real body, +69% area


def test_band_tolerates_the_legs_and_antennae_overshoot():
    """SAM3 traces appendages at ~1.7x the labelled body-core area.

    That overshoot is a labelling CONVENTION difference, not a bad mask, so
    the band's ceiling must sit comfortably above it.
    """
    labels = [_sq(100, 100, side=20.0) for _ in range(10)]
    band = fit_area_band(labels)
    traced = _sq(100, 100, side=float(20.0 * np.sqrt(1.7)))
    assert in_band(traced, band)


def test_no_band_admits_everything():
    assert in_band(_sq(0, 0, side=9999.0), None)


def test_aspect_ratio_is_at_least_one_and_scale_invariant():
    assert aspect_ratio(_rect(0, 0, 40, 10)) == pytest.approx(4.0, rel=1e-3)
    assert aspect_ratio(_rect(0, 0, 10, 40)) == pytest.approx(4.0, rel=1e-3)
    assert aspect_ratio(_sq(0, 0, 20.0)) == pytest.approx(1.0, rel=1e-3)


def test_quality_is_one_for_an_exact_match():
    poly = _sq(100, 100, side=20.0)
    assert match_quality(poly, poly.copy()) == pytest.approx(1.0, abs=0.02)


def test_quality_penalises_a_blob_more_than_the_appendage_overshoot():
    label = _sq(100, 100, side=20.0)
    traced = _sq(100, 100, side=26.0)  # legs included, still the right animal
    blob = _sq(100, 100, side=200.0)  # a chunk of arena
    assert match_quality(traced, label) > match_quality(blob, label)
    assert match_quality(blob, label) < 0.1


def test_quality_separates_a_weird_shape_from_a_compact_body_of_equal_area():
    """Area agreement alone cannot see this; aspect agreement can."""
    label = _sq(100, 100, side=20.0)
    compact = _sq(100, 100, side=20.0)
    spindly = _rect(100, 100, 100.0, 4.0)  # same 400 px^2, wrong shape
    assert polygon_area(spindly) == pytest.approx(polygon_area(compact), rel=1e-6)
    assert match_quality(spindly, label) < 0.5 * match_quality(compact, label)


def test_quality_is_zero_for_disjoint_polygons():
    assert match_quality(_sq(0, 0), _sq(900, 900)) == pytest.approx(0.0)


def test_band_bounds_are_ordered():
    band = fit_area_band([_sq(0, 0, side=20.0)])
    assert isinstance(band, AreaBand)
    assert 0 < band.min_px2 < band.median_px2 < band.max_px2
    assert band.n_labels == 1
