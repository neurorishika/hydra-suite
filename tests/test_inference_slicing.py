import numpy as np

from hydra_suite.core.inference.config import SliceConfig
from hydra_suite.core.inference.stages.slicing import get_slice_bboxes, plan_slices


def test_grid_covers_frame_and_flushes_to_edge():
    # 1000x1000 frame, 640 tiles, 0.2 overlap -> step 512.
    boxes = get_slice_bboxes(1000, 1000, 640, 640, 0.2, 0.2)
    # every pixel covered: min corner at 0, max corner at frame edge.
    xs0 = sorted({b[0] for b in boxes})
    assert xs0[0] == 0
    # last tile flush to right edge (no runt): some tile ends exactly at 1000.
    assert max(b[2] for b in boxes) == 1000
    assert max(b[3] for b in boxes) == 1000
    # no tile exceeds the frame.
    assert all(b[2] <= 1000 and b[3] <= 1000 for b in boxes)
    # tile size preserved (last tile shifted back, not shrunk).
    assert all((b[2] - b[0]) == 640 and (b[3] - b[1]) == 640 for b in boxes)


def test_zero_overlap_tiles_are_disjoint_step_equals_size():
    boxes = get_slice_bboxes(1280, 1280, 640, 640, 0.0, 0.0)
    assert len(boxes) == 4
    assert (0, 0, 640, 640) in boxes


def test_auto_model_uses_imgsz():
    plan = plan_slices(
        (2000, 2000),
        SliceConfig(enabled=True, geometry_mode="auto_model"),
        imgsz=1024,
        roi_mask=None,
    )
    assert plan.slice_wh == (1024, 1024)


def test_custom_uses_explicit_size():
    cfg = SliceConfig(
        enabled=True, geometry_mode="custom", slice_height=512, slice_width=768
    )
    plan = plan_slices((2000, 2000), cfg, imgsz=1024, roi_mask=None)
    assert plan.slice_wh == (768, 512)  # (w, h)


def test_auto_object_sizes_from_reference():
    # ref object 64px, want it to span 0.1 of the tile -> tile ~640px.
    cfg = SliceConfig(
        enabled=True, geometry_mode="auto_object", object_tile_fraction=0.1
    )
    plan = plan_slices((4000, 4000), cfg, imgsz=1024, roi_mask=None, ref_object_px=64.0)
    assert 600 <= plan.slice_wh[0] <= 680


def test_roi_gating_drops_tiles_outside_mask():
    # ROI only in the top-left quadrant.
    mask = np.zeros((1000, 1000), dtype=np.uint8)
    mask[:400, :400] = 1
    cfg = SliceConfig(
        enabled=True, geometry_mode="custom", slice_height=256, slice_width=256
    )
    full = plan_slices(
        (1000, 1000),
        SliceConfig(
            enabled=True, geometry_mode="custom", slice_height=256, slice_width=256
        ),
        imgsz=256,
        roi_mask=None,
    )
    gated = plan_slices((1000, 1000), cfg, imgsz=256, roi_mask=mask)
    assert len(gated.tiles) < len(full.tiles)
    # every kept tile intersects the ROI.
    for x0, y0, x1, y1 in gated.tiles:
        assert mask[y0:y1, x0:x1].any()


def test_perform_standard_pred_flag():
    cfg = SliceConfig(
        enabled=True,
        geometry_mode="custom",
        slice_height=256,
        slice_width=256,
        perform_standard_pred=True,
    )
    plan = plan_slices((512, 512), cfg, imgsz=256, roi_mask=None)
    assert plan.full_frame is True


def test_roi_eliminates_every_tile_falls_back_to_full_grid():
    """When ROI mask would drop every tile, fallback to full grid."""
    # All-zero ROI mask eliminates all tiles.
    mask = np.zeros((1000, 1000), dtype=np.uint8)
    cfg = SliceConfig(
        enabled=True, geometry_mode="custom", slice_height=256, slice_width=256
    )

    # Plan with no ROI.
    full = plan_slices((1000, 1000), cfg, imgsz=256, roi_mask=None)

    # Plan with all-zero ROI (eliminates everything).
    gated = plan_slices((1000, 1000), cfg, imgsz=256, roi_mask=mask)

    # Should fall back to full grid: same tiles, same length.
    assert len(gated.tiles) == len(full.tiles)
    assert gated.tiles == full.tiles


def test_tile_larger_than_frame():
    """Tile size larger than frame produces single tile without exceeding bounds."""
    cfg = SliceConfig(
        enabled=True, geometry_mode="custom", slice_height=2000, slice_width=2000
    )
    plan = plan_slices((500, 500), cfg, imgsz=2000, roi_mask=None)

    # Should produce exactly one tile.
    assert len(plan.tiles) == 1
    x0, y0, x1, y1 = plan.tiles[0]

    # Tile should not exceed frame bounds.
    assert x0 >= 0 and y0 >= 0 and x1 <= 500 and y1 <= 500
    # Tile size is clamped to frame size.
    assert (x1 - x0) == 500 and (y1 - y0) == 500


def test_auto_object_with_zero_ref_object_px_falls_back():
    """auto_object with zero ref_object_px falls back to auto_model sizing."""
    cfg = SliceConfig(
        enabled=True, geometry_mode="auto_object", object_tile_fraction=0.1
    )
    plan = plan_slices((4000, 4000), cfg, imgsz=1024, roi_mask=None, ref_object_px=0.0)
    # Should fall back to auto_model: tile size equals imgsz.
    assert plan.slice_wh == (1024, 1024)


def test_auto_object_with_negative_ref_object_px_falls_back():
    """auto_object with negative ref_object_px falls back to auto_model sizing."""
    cfg = SliceConfig(
        enabled=True, geometry_mode="auto_object", object_tile_fraction=0.1
    )
    plan = plan_slices(
        (4000, 4000), cfg, imgsz=1024, roi_mask=None, ref_object_px=-10.0
    )
    # Should fall back to auto_model: tile size equals imgsz.
    assert plan.slice_wh == (1024, 1024)


def test_overlap_approaching_one():
    """Very high overlap (0.85) still produces finite tile count."""
    cfg = SliceConfig(
        enabled=True,
        geometry_mode="custom",
        slice_height=512,
        slice_width=512,
        overlap_height_ratio=0.85,
        overlap_width_ratio=0.85,
    )
    plan = plan_slices((1280, 1280), cfg, imgsz=512, roi_mask=None)

    # Should not crash and tile count should be finite and sane.
    assert len(plan.tiles) > 0
    assert len(plan.tiles) < 500

    # Every tile should be full size (not shrunk).
    for x0, y0, x1, y1 in plan.tiles:
        assert (x1 - x0) == 512 and (y1 - y0) == 512


def test_asymmetric_frame_and_overlap():
    """Non-square frame with different height/width overlap ratios tiles correctly."""
    cfg = SliceConfig(
        enabled=True,
        geometry_mode="custom",
        slice_height=256,
        slice_width=384,
        overlap_height_ratio=0.3,
        overlap_width_ratio=0.5,
    )
    plan = plan_slices((900, 1600), cfg, imgsz=384, roi_mask=None)

    # Should cover the entire frame: some tile reaches right edge, some reaches bottom.
    xs = sorted({b[0] for b in plan.tiles})
    ys = sorted({b[1] for b in plan.tiles})
    assert xs[0] == 0
    assert ys[0] == 0
    assert max(b[2] for b in plan.tiles) == 1600  # right edge
    assert max(b[3] for b in plan.tiles) == 900  # bottom edge

    # Every tile should be full configured size (not shrunk).
    for x0, y0, x1, y1 in plan.tiles:
        assert (x1 - x0) == 384 and (y1 - y0) == 256
