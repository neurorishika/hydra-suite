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


def test_exclude_zone_is_scoped_per_arena_not_global():
    """Fix Wave 14, Fix 2: an exclude drawn for one arena must not erase a
    NEIGHBOURING arena's pixels, even where their raw shapes overlap.

    Three arenas. Arena 1's exclude zone is deliberately drawn so it
    geometrically overlaps arena 2's include region. Under the OLD (global)
    semantics this exclude would zero out that overlap in arena 2's label
    too; under the correct per-arena semantics, arena 2's label must be
    completely unaffected.
    """
    shapes = [
        _circle(20, 20, 15, arena_id=0),
        _circle(60, 20, 15, arena_id=1),
        _circle(60, 20, 25, arena_id=1, mode="exclude"),  # overlaps arena 2 below
        _circle(90, 20, 15, arena_id=2),
    ]
    labels, n_arenas = build_arena_labels(shapes, 120, 40)
    assert n_arenas == 3
    # Arena 0 is untouched by any exclude.
    assert labels[20, 20] == 1
    # Arena 1's own include is punched out by its own exclude (radius 25 >
    # radius 15, so the entire arena-1 circle is inside the exclude).
    assert labels[20, 60] == 0
    # Arena 2's pixels inside the overlap with arena 1's exclude (exclude
    # centred at (60, 20) r=25 reaches to x=85, arena 2 centred at (90, 20)
    # r=15 starts at x=75 -- they overlap in x in [75, 85]) must STILL be
    # labeled 3 (arena 2 + 1), not erased by arena 1's exclude.
    assert labels[20, 80] == 3
    assert labels[20, 90] == 3


def test_exclude_with_no_matching_include_arena_is_skipped():
    """An orphaned exclude (arena_id with no matching include shape) has
    nothing to scope against and must be skipped rather than crash."""
    shapes = [
        _circle(20, 20, 15, arena_id=0),
        _circle(20, 20, 5, arena_id=7, mode="exclude"),  # no include with id 7
    ]
    labels, n_arenas = build_arena_labels(shapes, 100, 100)
    assert n_arenas == 1
    # The orphaned exclude must not remove arena 0's pixels.
    assert labels[20, 20] == 1


def test_generate_combined_roi_mask_matches_build_arena_labels(qtbot=None):
    """SessionOrchestrator._generate_combined_roi_mask must produce a mask
    consistent with build_arena_labels' own `labels > 0` for the same
    shapes -- the new shared-implementation invariant (replacing the old
    'pixel-identical to build_roi_mask' one, which no longer universally
    holds now that excludes are per-arena)."""
    import os

    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from types import SimpleNamespace

    from PySide6.QtWidgets import QApplication

    from hydra_suite.trackerkit.gui.orchestrators.session import SessionOrchestrator

    QApplication.instance() or QApplication([])

    shapes = [
        _circle(20, 20, 15, arena_id=0),
        _circle(60, 20, 15, arena_id=1),
        _circle(60, 20, 25, arena_id=1, mode="exclude"),
        _circle(90, 20, 15, arena_id=2),
    ]
    mw = SimpleNamespace(roi_shapes=shapes, roi_mask=None)
    mw._invalidate_roi_cache = lambda: None

    orch = SessionOrchestrator.__new__(SessionOrchestrator)
    orch._mw = mw
    orch._generate_combined_roi_mask(height=40, width=120)

    labels, _n_arenas = build_arena_labels(shapes, 120, 40)
    expected = (labels > 0).astype(np.uint8) * 255
    np.testing.assert_array_equal(mw.roi_mask, expected)
