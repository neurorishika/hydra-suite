"""Tests for shared geometry-derivation math (used by DetectKit's canvas rendering)."""

from __future__ import annotations

from hydra_suite.utils.geometry_derivation import (
    axis_aligned_bbox_quad,
    min_area_rect_quad,
)


def test_min_area_rect_quad_axis_aligned_square_returns_its_own_corners():
    points = [(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)]
    box = min_area_rect_quad(points)
    assert box is not None
    assert len(box) == 4
    xs = sorted(p[0] for p in box)
    ys = sorted(p[1] for p in box)
    assert abs(xs[0] - 0.0) < 0.5 and abs(xs[-1] - 10.0) < 0.5
    assert abs(ys[0] - 0.0) < 0.5 and abs(ys[-1] - 10.0) < 0.5


def test_min_area_rect_quad_too_few_points_returns_none():
    assert min_area_rect_quad([(0.0, 0.0), (1.0, 1.0)]) is None
    assert min_area_rect_quad([]) is None


def test_axis_aligned_bbox_quad_returns_exact_bbox_corners():
    points = [(2.0, 5.0), (8.0, 3.0), (6.0, 9.0)]
    box = axis_aligned_bbox_quad(points)
    assert box is not None
    xs = sorted(p[0] for p in box)
    ys = sorted(p[1] for p in box)
    assert xs[0] == 2.0 and xs[-1] == 8.0
    assert ys[0] == 3.0 and ys[-1] == 9.0
    assert len(box) == 4


def test_axis_aligned_bbox_quad_empty_points_returns_none():
    assert axis_aligned_bbox_quad([]) is None


def test_axis_aligned_bbox_quad_of_a_single_point_is_degenerate_but_defined():
    box = axis_aligned_bbox_quad([(5.0, 5.0)])
    assert box is not None
    assert all(p == (5.0, 5.0) for p in box)
