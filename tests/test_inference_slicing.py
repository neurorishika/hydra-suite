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
