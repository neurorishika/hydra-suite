import numpy as np

from hydra_suite.core.tracking.arenas import ArenaLayout


def _labels():
    """100x100: arena 0 = left half rows 0-49, arena 1 = right half rows 0-49."""
    labels = np.zeros((100, 100), np.uint16)
    labels[0:50, 0:50] = 1
    labels[0:50, 50:100] = 2
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
    layout = ArenaLayout(n_arenas=2, animals_per_arena=1, label_image=_labels())
    small = layout.label_image_for_size(50, 50)
    assert small.shape == (50, 50)
    assert set(np.unique(small).tolist()) <= {0, 1, 2}


def test_resize_result_is_cached():
    layout = ArenaLayout(n_arenas=2, animals_per_arena=1, label_image=_labels())
    assert layout.label_image_for_size(50, 50) is layout.label_image_for_size(50, 50)
