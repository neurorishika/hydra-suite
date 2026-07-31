import pytest

from hydra_suite.core.inference.config import SliceConfig
from hydra_suite.core.inference.stages import slicing as stages_slicing
from hydra_suite.utils import slice_geometry as sg


def test_names_are_reexported_from_stages():
    # Existing inference tests import these from stages.slicing; they must stay.
    assert stages_slicing.get_slice_bboxes is sg.get_slice_bboxes
    assert stages_slicing.tiles_overlap is sg.tiles_overlap
    assert stages_slicing.SlicePlan is sg.SlicePlan
    assert stages_slicing.MAX_TILES_PER_FRAME == sg.MAX_TILES_PER_FRAME


@pytest.mark.parametrize(
    "frame_hw,mode,imgsz,ref,frac,sw,sh,overlap",
    [
        ((1000, 1000), "auto_model", 640, 0.0, 0.15, 0, 0, 0.2),
        ((2000, 2000), "custom", 1024, 0.0, 0.15, 512, 512, 0.2),
        ((4000, 4000), "auto_object", 1024, 64.0, 0.15, 0, 0, 0.2),
        ((900, 1600), "auto_model", 384, 0.0, 0.15, 0, 0, 0.3),
    ],
)
def test_plan_tiles_matches_plan_slices(
    frame_hw, mode, imgsz, ref, frac, sw, sh, overlap
):
    cfg = SliceConfig(
        enabled=True,
        geometry_mode=mode,
        slice_width=sw,
        slice_height=sh,
        overlap_width_ratio=overlap,
        overlap_height_ratio=overlap,
        object_tile_fraction=frac,
    )
    plan = stages_slicing.plan_slices(
        frame_hw, cfg, imgsz=imgsz, roi_mask=None, ref_object_px=ref
    )
    w, h = sg.tile_size_for_mode(
        geometry_mode=mode,
        imgsz=imgsz,
        reference_body_px=ref,
        object_tile_fraction=frac,
        slice_width=sw,
        slice_height=sh,
    )
    direct = sg.plan_tiles(
        frame_hw, w, h, overlap, overlap, full_frame=False, roi_mask=None
    )
    assert plan.tiles == direct.tiles
    assert plan.slice_wh == direct.slice_wh
