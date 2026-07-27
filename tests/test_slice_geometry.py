import numpy as np

from hydra_suite.utils.slice_geometry import (
    clip_polygon_to_tile,
    get_slice_bboxes,
    plan_tiles,
    polygon_area,
    tile_size_for_mode,
    tiles_overlap,
)


def test_grid_flushes_last_tile_to_edge():
    boxes = get_slice_bboxes(1000, 1000, 640, 640, 0.2, 0.2)
    assert all(x1 <= 1000 and y1 <= 1000 for _, _, x1, y1 in boxes)
    assert any(x1 == 1000 for _, _, x1, _ in boxes)


def test_tile_size_custom_falls_back_to_imgsz():
    assert tile_size_for_mode(
        geometry_mode="custom",
        imgsz=512,
        reference_body_px=0.0,
        object_tile_fraction=0.15,
        slice_width=0,
        slice_height=0,
    ) == (512, 512)
    assert tile_size_for_mode(
        geometry_mode="custom",
        imgsz=512,
        reference_body_px=0.0,
        object_tile_fraction=0.15,
        slice_width=300,
        slice_height=200,
    ) == (300, 200)


def test_tile_size_auto_object():
    # ref=64, frac=0.16 -> 400
    assert tile_size_for_mode(
        geometry_mode="auto_object",
        imgsz=1024,
        reference_body_px=64.0,
        object_tile_fraction=0.16,
        slice_width=0,
        slice_height=0,
    ) == (400, 400)


def test_tile_size_auto_object_zero_ref_falls_back_to_auto_model():
    assert tile_size_for_mode(
        geometry_mode="auto_object",
        imgsz=1024,
        reference_body_px=0.0,
        object_tile_fraction=0.15,
        slice_width=0,
        slice_height=0,
    ) == (1024, 1024)


def test_plan_tiles_ceiling_raises():
    import pytest

    with pytest.raises(ValueError):
        plan_tiles((1080, 1920), 64, 64, 0.9, 0.9)


def test_plan_tiles_roi_gating_drops_tiles():
    mask = np.zeros((1000, 1000), dtype=bool)
    mask[:256, :256] = True
    full = plan_tiles((1000, 1000), 256, 256, 0.0, 0.0)
    gated = plan_tiles((1000, 1000), 256, 256, 0.0, 0.0, roi_mask=mask)
    assert len(gated.tiles) < len(full.tiles)


def test_tiles_overlap_true_for_flush_last_tile():
    assert tiles_overlap(get_slice_bboxes(300, 300, 256, 256, 0.0, 0.0)) is True


def test_polygon_area_unit_square():
    sq = np.array([[0, 0], [10, 0], [10, 10], [0, 10]], dtype=np.float32)
    assert abs(polygon_area(sq) - 100.0) < 1e-3


def test_clip_fully_inside_returns_same_area():
    poly = np.array([[10, 10], [30, 10], [30, 30], [10, 30]], dtype=np.float32)
    clipped = clip_polygon_to_tile(poly, (0, 0, 100, 100))
    assert clipped is not None
    assert abs(polygon_area(clipped) - 400.0) < 1e-3


def test_clip_straddling_boundary_halves_area():
    # 20x20 square centered on x=100 boundary of tile [0..100]
    poly = np.array([[90, 40], [110, 40], [110, 60], [90, 60]], dtype=np.float32)
    clipped = clip_polygon_to_tile(poly, (0, 0, 100, 200))
    assert clipped is not None
    assert abs(polygon_area(clipped) - 200.0) < 1e-3  # half of 400
    assert clipped[:, 0].max() <= 100.0 + 1e-3


def test_clip_fully_outside_returns_none():
    poly = np.array([[200, 200], [220, 200], [220, 220], [200, 220]], dtype=np.float32)
    assert clip_polygon_to_tile(poly, (0, 0, 100, 100)) is None
