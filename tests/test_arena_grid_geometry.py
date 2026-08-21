"""Grid generation under rotation, non-overlapping pitch floors, and extent caps."""

import math

from hydra_suite.trackerkit.arena_geometry import (
    generate_grid_shapes,
    max_grid_extent,
    min_pitch,
)


def test_zero_rotation_matches_the_unrotated_lattice():
    """Rotation must be a strict extension: 0 degrees changes nothing."""
    plain = generate_grid_shapes(2, 2, 50, 50, 100, 100, 40)
    rotated = generate_grid_shapes(2, 2, 50, 50, 100, 100, 40, rotation_deg=0.0)
    assert plain == rotated


def test_rotation_pivots_about_the_first_arena_centre():
    shapes = generate_grid_shapes(1, 2, 100, 100, 50, 50, 20, rotation_deg=90.0)
    assert (shapes[0]["params"][0], shapes[0]["params"][1]) == (100, 100)
    assert (shapes[1]["params"][0], shapes[1]["params"][1]) == (100, 150)


def test_rotation_preserves_centre_to_centre_distance():
    shapes = generate_grid_shapes(1, 2, 100, 100, 60, 60, 20, rotation_deg=37.0)
    ax, ay, _ = shapes[0]["params"]
    bx, by, _ = shapes[1]["params"]
    assert math.hypot(bx - ax, by - ay) == 60


def test_rectangle_uses_separate_width_and_height():
    shapes = generate_grid_shapes(
        1, 1, 100, 100, 200, 200, 40, shape_type="polygon", size_y=20
    )
    xs = [p[0] for p in shapes[0]["params"]]
    ys = [p[1] for p in shapes[0]["params"]]
    assert max(xs) - min(xs) == 40
    assert max(ys) - min(ys) == 20


def test_rotated_rectangle_is_a_four_point_polygon():
    shapes = generate_grid_shapes(
        1, 1, 100, 100, 200, 200, 40, shape_type="polygon", rotation_deg=30.0
    )
    assert shapes[0]["type"] == "polygon"
    assert len(shapes[0]["params"]) == 4


def test_rotated_rectangle_corners_are_actually_rotated():
    """A rotated square must not stay axis-aligned."""
    shapes = generate_grid_shapes(
        1, 1, 100, 100, 200, 200, 40, shape_type="polygon", rotation_deg=30.0
    )
    ys = sorted(p[1] for p in shapes[0]["params"])
    assert len(set(ys)) > 2


def test_circles_ignore_size_y():
    """A circle has one dimension; size_y must not silently deform it."""
    shapes = generate_grid_shapes(1, 1, 50, 50, 100, 100, 40, size_y=10)
    assert shapes[0]["params"][2] == 20


def test_min_pitch_for_circles_is_the_diameter():
    """radius/2 (the original brief) guarantees overlap; 2*radius is the floor."""
    assert min_pitch("circle", 40) == (40, 40)


def test_min_pitch_for_rectangles_is_width_and_height():
    assert min_pitch("polygon", 40, size_y=20) == (40, 20)


def test_min_pitch_grid_produces_no_overlap():
    from hydra_suite.trackerkit.arena_geometry import overlapping_arena_pairs

    px, py = min_pitch("circle", 40)
    shapes = generate_grid_shapes(3, 3, 60, 60, px, py, 40)
    assert overlapping_arena_pairs(shapes, 400, 400) == []


def test_extent_cap_keeps_every_centre_inside():
    rows, cols = max_grid_extent(50, 50, 100, 100, 400, 300)
    assert (rows, cols) == (3, 4)


def test_extent_cap_shrinks_under_rotation():
    """Rotating a wide grid pushes far centres off-frame, so the cap tightens."""
    straight = max_grid_extent(20, 20, 100, 100, 400, 400, rotation_deg=0.0)
    tilted = max_grid_extent(20, 20, 100, 100, 400, 400, rotation_deg=45.0)
    assert tilted[0] * tilted[1] < straight[0] * straight[1]


def test_extent_cap_is_at_least_one_by_one():
    """An origin outside the frame must still yield a usable minimum."""
    assert max_grid_extent(9999, 9999, 100, 100, 400, 300) == (1, 1)


def test_capped_grid_has_every_centre_in_frame():
    rows, cols = max_grid_extent(30, 30, 90, 70, 640, 480, rotation_deg=22.0)
    shapes = generate_grid_shapes(rows, cols, 30, 30, 90, 70, 20, rotation_deg=22.0)
    for shape in shapes:
        cx, cy, _ = shape["params"]
        assert 0 <= cx < 640 and 0 <= cy < 480
