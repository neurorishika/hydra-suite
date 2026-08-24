"""Overlap detection between arenas, including fast-path/brute-force agreement."""

import numpy as np
import pytest

from hydra_suite.trackerkit.arena_geometry import (
    non_contiguous_arena_ids,
    overlapping_arena_pairs,
)


def _circle(cx, cy, r, arena_id, mode="include"):
    return {"type": "circle", "params": (cx, cy, r), "mode": mode, "arena_id": arena_id}


def _square(cx, cy, half, arena_id, mode="include"):
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


def _brute_force_pairs(shapes, width, height):
    """Authoritative reference: rasterize every arena full-frame and intersect."""
    ids = sorted({int(s["arena_id"]) for s in shapes if s.get("mode") == "include"})
    import cv2

    masks = {}
    for arena_id in ids:
        canvas = np.zeros((height, width), np.uint8)
        for shape in shapes:
            if int(shape["arena_id"]) != arena_id:
                continue
            value = 255 if shape.get("mode", "include") == "include" else 0
            if shape["type"] == "circle":
                cx, cy, r = shape["params"]
                cv2.circle(canvas, (int(cx), int(cy)), int(r), value, -1)
            else:
                pts = np.asarray(shape["params"], np.int32)
                cv2.fillPoly(canvas, [pts], value)
        masks[arena_id] = canvas > 0
    out = []
    for i, a in enumerate(ids):
        for b in ids[i + 1 :]:
            if np.any(masks[a] & masks[b]):
                out.append((a, b))
    return out


def test_separate_circles_do_not_overlap():
    shapes = [_circle(50, 50, 20, 0), _circle(200, 50, 20, 1)]
    assert overlapping_arena_pairs(shapes, 400, 200) == []


def test_intersecting_circles_are_reported():
    shapes = [_circle(100, 100, 40, 0), _circle(150, 100, 40, 1)]
    assert overlapping_arena_pairs(shapes, 400, 300) == [(0, 1)]


def test_exactly_tangent_circles_do_not_overlap():
    """Centre distance == r1 + r2 touches but shares no interior pixel."""
    shapes = [_circle(100, 100, 30, 0), _circle(160, 100, 30, 1)]
    assert overlapping_arena_pairs(shapes, 400, 300) == []


def test_exclude_zone_can_resolve_an_overlap():
    """Punching the shared region out of one arena clears the conflict."""
    shapes = [
        _circle(100, 100, 40, 0),
        _circle(150, 100, 40, 1),
        _circle(150, 100, 40, 0, mode="exclude"),
    ]
    assert overlapping_arena_pairs(shapes, 400, 300) == []


def test_pairs_are_sorted_ascending():
    shapes = [_circle(100, 100, 60, 5), _circle(120, 100, 60, 2)]
    assert overlapping_arena_pairs(shapes, 400, 300) == [(2, 5)]


def test_mixed_circle_and_polygon_overlap():
    shapes = [_circle(100, 100, 30, 0), _square(115, 100, 20, 1)]
    assert overlapping_arena_pairs(shapes, 400, 300) == [(0, 1)]


def test_single_arena_never_overlaps_itself():
    shapes = [_circle(100, 100, 40, 0), _circle(110, 100, 40, 0)]
    assert overlapping_arena_pairs(shapes, 400, 300) == []


@pytest.mark.parametrize("seed", range(20))
def test_fast_path_agrees_with_brute_force(seed):
    """The analytic and bbox filters must never disagree with rasterization.

    Without this, an optimization bug would silently weaken the overlap gate
    -- the failure mode is a missed conflict, which is invisible in the UI.
    """
    rng = np.random.default_rng(seed)
    width = height = 200
    shapes = []
    for arena_id in range(5):
        cx, cy = rng.integers(20, 180, size=2)
        r = int(rng.integers(10, 45))
        if arena_id % 2:
            shapes.append(_circle(int(cx), int(cy), r, arena_id))
        else:
            shapes.append(_square(int(cx), int(cy), r, arena_id))
    assert overlapping_arena_pairs(shapes, width, height) == _brute_force_pairs(
        shapes, width, height
    )


def test_disconnected_far_apart_circles_same_arena_are_flagged():
    shapes = [_circle(50, 50, 20, 0), _circle(2850, 50, 20, 0)]
    assert non_contiguous_arena_ids(shapes, 3000, 200) == [0]


def test_touching_circles_same_arena_are_not_flagged():
    shapes = [_circle(100, 100, 30, 0), _circle(160, 100, 30, 0)]
    assert non_contiguous_arena_ids(shapes, 400, 300) == []


def test_far_apart_circles_in_different_arenas_are_not_flagged():
    """Cross-arena separation is fine -- each arena still has one piece.

    Cross-arena OVERLAP is ``overlapping_arena_pairs``'s job, not this
    function's -- distant shapes in different arenas should trip neither
    check.
    """
    shapes = [_circle(50, 50, 20, 0), _circle(2850, 50, 20, 1)]
    assert non_contiguous_arena_ids(shapes, 3000, 200) == []
    assert overlapping_arena_pairs(shapes, 3000, 200) == []


def test_exclude_that_bisects_an_arena_is_flagged():
    """A wide exclude splitting one large include region into two pieces."""
    shapes = [
        _square(100, 100, 90, 0),
        {
            "type": "polygon",
            "params": [[95, 5], [105, 5], [105, 195], [95, 195]],
            "mode": "exclude",
            "arena_id": 0,
        },
    ]
    assert non_contiguous_arena_ids(shapes, 300, 300) == [0]


def test_exclude_that_bisects_an_arena_is_flagged_regardless_of_shape_order():
    """Fix Wave 21 Finding B regression: _rasterize must paint ALL includes
    first, then ALL excludes, mirroring build_arena_labels' two-pass
    structure -- NOT whatever order the shapes happen to appear in the
    list. Reachable in the GUI: the arena panel's zone buttons let a user
    draw an exclude zone before that arena has any include shape at all.

    Before the fix, _rasterize painted shapes in list order: with the
    exclude listed first, it painted 0 onto an all-zero canvas (no
    effect), then the include painted 255 straight over that same area
    afterward -- the hole never got punched, the region read as fully
    connected, and this test would have failed (asserting [] instead of
    [0]).
    """
    shapes = [
        {
            "type": "polygon",
            "params": [[95, 5], [105, 5], [105, 195], [95, 195]],
            "mode": "exclude",
            "arena_id": 0,
        },
        _square(100, 100, 90, 0),
    ]
    assert non_contiguous_arena_ids(shapes, 300, 300) == [0]


def test_small_edge_notch_exclude_does_not_disconnect():
    """An exclude that only nibbles one edge leaves a single connected piece."""
    shapes = [
        _circle(100, 100, 80, 0),
        _circle(175, 100, 15, 0, mode="exclude"),
    ]
    assert non_contiguous_arena_ids(shapes, 300, 300) == []


def test_empty_shapes_returns_empty_list():
    assert non_contiguous_arena_ids([], 300, 300) == []
    assert non_contiguous_arena_ids(None, 300, 300) == []
