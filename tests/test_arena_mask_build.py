import numpy as np

from hydra_suite.trackerkit.engine_params import build_arena_labels, build_roi_mask


def _circle(cx, cy, r, arena_id=None, mode="include"):
    shape = {"type": "circle", "params": [cx, cy, r], "mode": mode}
    if arena_id is not None:
        shape["arena_id"] = arena_id
    return shape


def _polygon(points, arena_id=None, mode="include"):
    shape = {"type": "polygon", "params": points, "mode": mode}
    if arena_id is not None:
        shape["arena_id"] = arena_id
    return shape


def test_legacy_shapes_without_arena_id_collapse_to_one_arena():
    """Back-compat: three shapes, no arena_id -> a single arena 0."""
    shapes = [_circle(20, 20, 10), _circle(60, 20, 10), _circle(20, 60, 10)]
    labels, n_arenas = build_arena_labels(shapes, 100, 100)
    assert n_arenas == 1
    assert set(np.unique(labels)) == {0, 1}


def test_distinct_arena_ids_produce_distinct_labels():
    shapes = [_circle(20, 20, 10, arena_id=0), _circle(60, 20, 10, arena_id=1)]
    labels, n_arenas = build_arena_labels(shapes, 100, 100)
    assert n_arenas == 2
    assert labels[20, 20] == 1
    assert labels[20, 60] == 2
    assert labels[50, 50] == 0


def test_exclusion_hole_is_outside_every_arena():
    shapes = [_circle(50, 50, 30, arena_id=0), _circle(50, 50, 10, mode="exclude")]
    labels, _ = build_arena_labels(shapes, 100, 100)
    assert labels[50, 50] == 0  # inside the hole
    assert labels[50, 75] == 1  # in the annulus


def test_label_union_matches_roi_mask_exactly():
    """Invariant: (labels > 0) is pixel-identical to the existing ROI mask.

    Covers both shape types (circle + polygon), two distinct arenas, and an
    exclusion hole punched into one of them -- not just circles.
    """
    shapes = [
        _polygon([[10, 10], [50, 10], [50, 50], [10, 50]], arena_id=0),
        _circle(70, 70, 15, arena_id=1),
        _circle(30, 30, 5, mode="exclude"),
    ]
    labels, _ = build_arena_labels(shapes, 100, 100)
    roi = build_roi_mask(shapes, 100, 100)
    np.testing.assert_array_equal(labels > 0, roi > 0)


def test_unrecognized_mode_contributes_no_pixels_to_either_function():
    """A shape with an unknown `mode` string is dropped by BOTH functions.

    `build_roi_mask` only renders on `mode == "include"` / `== "exclude"`,
    silently dropping any other mode string. `build_arena_labels` must mirror
    that partition exactly, not treat "not exclude" as "include".
    """
    shapes = [
        _circle(50, 50, 20, arena_id=0),
        _circle(50, 50, 40, arena_id=1, mode="zone"),
    ]
    labels, n_arenas = build_arena_labels(shapes, 100, 100)
    roi = build_roi_mask(shapes, 100, 100)
    # The "zone" shape must not appear in either the labels or the ROI mask,
    # and must not introduce a second arena.
    assert n_arenas == 1
    np.testing.assert_array_equal(labels > 0, roi > 0)
    # A point that only the dropped "zone" circle would have covered (outside
    # the arena_id=0 circle, inside the radius-40 "zone" circle) stays 0.
    assert labels[50, 85] == 0
    assert roi[50, 85] == 0


def test_no_shapes_returns_none():
    assert build_arena_labels([], 100, 100) == (None, 1)
    assert build_arena_labels(None, 100, 100) == (None, 1)


def test_arena_ids_are_densified():
    """Sparse ids (0, 5, 9) become contiguous labels 1, 2, 3."""
    shapes = [
        _circle(20, 20, 8, arena_id=0),
        _circle(50, 20, 8, arena_id=5),
        _circle(80, 20, 8, arena_id=9),
    ]
    labels, n_arenas = build_arena_labels(shapes, 100, 100)
    assert n_arenas == 3
    assert sorted(np.unique(labels).tolist()) == [0, 1, 2, 3]
