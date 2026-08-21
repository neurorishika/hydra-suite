import numpy as np

from hydra_suite.core.tracking.arenas import ArenaLayout


def _labels():
    """100x100: arena 0 = left half rows 0-49, arena 1 = right half rows 0-49."""
    labels = np.zeros((100, 100), np.uint16)
    labels[0:50, 0:50] = 1
    labels[0:50, 50:100] = 2
    return labels


def _labels_wide_ids():
    """100x100, vertical split at col 50: id 1 (left) vs id 1000 (right).

    Ids are numerically far apart on purpose. A clean 2x downsample (e.g.
    100x100 -> 50x50) has every destination sample point land mid-cell,
    never straddling the col-50 boundary, so INTER_NEAREST/LINEAR/AREA all
    agree there -- that fixture would not discriminate the interpolation
    mode. A non-integer-ratio resize (100 -> 33) forces some destination
    sample points to straddle the boundary; blending id 1 with id 1000
    under a linear/area filter produces some value strictly between them
    (never 0, 1, or 1000), which INTER_NEAREST can never produce.
    """
    labels = np.zeros((100, 100), np.uint16)
    labels[:, :50] = 1
    labels[:, 50:] = 1000
    return labels


def test_slot_arena_is_contiguous_blocks():
    layout = ArenaLayout(n_arenas=3, animals_per_arena=2, label_image=None)
    assert layout.max_targets == 6
    np.testing.assert_array_equal(layout.slot_arena, [0, 0, 1, 1, 2, 2])


def test_single_arena_layout_is_flagged():
    layout = ArenaLayout(n_arenas=1, animals_per_arena=4, label_image=None)
    assert layout.is_single_arena
    assert layout.max_targets == 4
    np.testing.assert_array_equal(layout.slot_arena, [0, 0, 0, 0])


def test_arena_of_points_maps_centroids():
    layout = ArenaLayout(n_arenas=2, animals_per_arena=1, label_image=_labels())
    xy = np.array([[10.0, 10.0], [80.0, 10.0], [10.0, 80.0]], dtype=np.float32)
    np.testing.assert_array_equal(layout.arena_of_points(xy), [0, 1, -1])


def test_arena_of_points_clips_out_of_frame_coordinates():
    """Mirrors filter_with_indices:300 -- coordinates are clipped, never wrapped."""
    layout = ArenaLayout(n_arenas=2, animals_per_arena=1, label_image=_labels())
    xy = np.array([[-5.0, 10.0], [500.0, 10.0]], dtype=np.float32)
    np.testing.assert_array_equal(layout.arena_of_points(xy), [0, 1])


def test_arena_of_points_without_label_image_is_all_zero():
    layout = ArenaLayout(n_arenas=1, animals_per_arena=4, label_image=None)
    xy = np.array([[10.0, 10.0], [90.0, 90.0]], dtype=np.float32)
    np.testing.assert_array_equal(layout.arena_of_points(xy), [0, 0])


def test_empty_detection_array_returns_empty():
    layout = ArenaLayout(n_arenas=2, animals_per_arena=1, label_image=_labels())
    out = layout.arena_of_points(np.zeros((0, 2), dtype=np.float32))
    assert out.shape == (0,)


def test_resize_uses_nearest_and_invents_no_labels():
    layout = ArenaLayout(
        n_arenas=2, animals_per_arena=1, label_image=_labels_wide_ids()
    )
    # Non-integer-ratio target (100 -> 33) forces destination sample points to
    # straddle the id-1/id-1000 boundary; see _labels_wide_ids for why this
    # fixture (unlike a clean 2x downsample) actually discriminates
    # INTER_NEAREST from interpolating resize modes.
    small = layout.label_image_for_size(33, 33)
    assert small.shape == (33, 33)
    assert set(np.unique(small).tolist()) <= {0, 1, 1000}


def test_resize_result_is_cached():
    layout = ArenaLayout(n_arenas=2, animals_per_arena=1, label_image=_labels())
    assert layout.label_image_for_size(50, 50) is layout.label_image_for_size(50, 50)


def test_arena_of_points_with_frame_size_matches_native_query():
    """A centroid expressed in a scaled frame's coordinates, queried with
    frame_size set, must resolve to the same arena as the equivalent
    native-resolution query without it (Task 6 passes frame.shape after
    RESIZE_FACTOR has scaled the tracking frame -- see worker.py:2062/2076)."""
    layout = ArenaLayout(n_arenas=2, animals_per_arena=1, label_image=_labels())
    # Native space (100x100): x=10 -> arena 0, x=80 -> arena 1.
    native_xy = np.array([[10.0, 10.0], [80.0, 10.0]], dtype=np.float32)
    native_result = layout.arena_of_points(native_xy)
    np.testing.assert_array_equal(native_result, [0, 1])

    # Half-resolution frame (50x50): same relative positions, scaled coords.
    scaled_xy = native_xy * 0.5
    scaled_result = layout.arena_of_points(scaled_xy, frame_size=(50, 50))
    np.testing.assert_array_equal(scaled_result, native_result)
