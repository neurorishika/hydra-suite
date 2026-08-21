"""Hit-testing for arena shapes. Pure geometry, no Qt import."""

from hydra_suite.trackerkit.arena_geometry import (
    arena_at_point,
    point_in_shape,
    shape_centroid,
)


def _circle(cx, cy, r, arena_id=0, mode="include"):
    return {"type": "circle", "params": (cx, cy, r), "mode": mode, "arena_id": arena_id}


def _square(cx, cy, half, arena_id=0, mode="include"):
    return {
        "type": "polygon",
        "params": [
            [cx - half, cy - half],
            [cx + half, cy - half],
            [cx + half, cy + half],
            [cx - half, cy + half],
        ],
        "mode": mode,
        "arena_id": arena_id,
    }


def test_circle_centroid_is_its_centre():
    assert shape_centroid(_circle(100, 200, 30)) == (100.0, 200.0)


def test_polygon_centroid_is_vertex_mean():
    assert shape_centroid(_square(100, 200, 10)) == (100.0, 200.0)


def test_point_inside_circle():
    assert point_in_shape(_circle(100, 100, 20), 110, 100) is True


def test_point_outside_circle():
    assert point_in_shape(_circle(100, 100, 20), 130, 100) is False


def test_point_on_circle_edge_counts_as_inside():
    """Inclusive boundary matches cv2.circle's filled rasterization."""
    assert point_in_shape(_circle(100, 100, 20), 120, 100) is True


def test_point_inside_polygon():
    assert point_in_shape(_square(100, 100, 10), 100, 100) is True


def test_point_outside_polygon():
    assert point_in_shape(_square(100, 100, 10), 200, 200) is False


def test_arena_at_point_finds_the_arena():
    shapes = [_circle(50, 50, 20, arena_id=0), _circle(200, 50, 20, arena_id=1)]
    assert arena_at_point(shapes, 200, 50) == 1


def test_arena_at_point_returns_none_outside_every_arena():
    shapes = [_circle(50, 50, 20, arena_id=0)]
    assert arena_at_point(shapes, 500, 500) is None


def test_exclude_hole_is_not_part_of_the_arena():
    """A point inside an exclude zone belongs to no arena, even inside an include."""
    shapes = [
        _circle(100, 100, 50, arena_id=3),
        _circle(100, 100, 10, arena_id=3, mode="exclude"),
    ]
    assert arena_at_point(shapes, 100, 100) is None
    assert arena_at_point(shapes, 140, 100) == 3


def test_overlap_resolves_by_draw_order():
    """Last-writer-wins, matching engine_params.build_arena_labels."""
    shapes = [_circle(100, 100, 40, arena_id=0), _circle(110, 100, 40, arena_id=1)]
    assert arena_at_point(shapes, 105, 100) == 1
