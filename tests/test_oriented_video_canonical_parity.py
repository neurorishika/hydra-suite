"""Oriented-video frames must come out of the SHARED canonical crop path.

The crop-dataset path was migrated to the canonical resampler
(`canonicalization.crop.extract_canonical_crop`, which warps via torch
`F.grid_sample` -- its docstring notes it "replaces the former OpenCV
affine-warp kernel"). `OrientedTrackVideoExporter._render_task` was left behind
on a hand-rolled `cv2.warpAffine`, so an oriented video frame is NOT the same
pixels as the canonical crop every model and every exported dataset sees for
that same detection.

Two consequences, both covered here:
  1. resampler divergence -- different pixel values for the same affine;
  2. foreign-OBB masking has no own-body protection, so an overlapping
     neighbour's OBB punches a hole through the subject's own body. The shared
     `_apply_foreign_mask_canonical` takes `own_corners` precisely to prevent
     that.
"""

import numpy as np
import pytest

from hydra_suite.core.canonicalization.crop import extract_canonical_crop
from hydra_suite.core.canonicalization.geometry import (
    CanonicalGeometry,
    canonical_affine,
)
from hydra_suite.core.individual.dataset.oriented_video import (
    FrameTask,
    OrientedTrackVideoExporter,
)

GEOMETRY = CanonicalGeometry.from_reference(
    reference_body_px=24.0, aspect_ratio=2.0, margin=1.5
)


def _obb(cx, cy, half_w, half_h):
    return np.array(
        [
            [cx - half_w, cy - half_h],
            [cx + half_w, cy - half_h],
            [cx + half_w, cy + half_h],
            [cx - half_w, cy + half_h],
        ],
        dtype=np.float32,
    )


def _textured_frame(h=200, w=240):
    """Non-uniform content so a resampler difference actually shows up."""
    yy, xx = np.mgrid[0:h, 0:w]
    frame = np.zeros((h, w, 3), dtype=np.uint8)
    frame[:, :, 0] = (xx * 7 % 256).astype(np.uint8)
    frame[:, :, 1] = (yy * 11 % 256).astype(np.uint8)
    frame[:, :, 2] = ((xx + yy) * 5 % 256).astype(np.uint8)
    return frame


def _exporter(tmp_path, **kw):
    return OrientedTrackVideoExporter(
        str(tmp_path / "ds"),
        str(tmp_path / "final.csv"),
        video_path=str(tmp_path / "in.mp4"),
        detection_cache_path=str(tmp_path / "detection.npz"),
        fps=5.0,
        geometry=GEOMETRY,
        **kw,
    )


def _task(corners, geometry, polygon_index=0):
    affine, _theta, _clip = canonical_affine(corners, geometry)
    return FrameTask(
        frame_id=0,
        trajectory_id=0,
        affine=affine,
        out_w=geometry.canvas_w,
        out_h=geometry.canvas_h,
        center_x=float(corners.mean(axis=0)[0]),
        center_y=float(corners.mean(axis=0)[1]),
        width=10.0,
        height=5.0,
        theta=0.0,
        corners=corners,
        expanded_corners=OrientedTrackVideoExporter._expand_corners(
            corners, geometry.margin - 1.0
        ),
        polygon_index=polygon_index,
    )


def test_render_warp_matches_the_shared_canonical_crop(tmp_path):
    """The warped pixels must be identical to extract_canonical_crop's."""
    frame = _textured_frame()
    corners = _obb(120.0, 100.0, 14.0, 7.0)
    task = _task(corners, GEOMETRY)

    # Background must equal the masked-out fill so the comparison isolates the
    # resampler, and the mask must cover the whole canvas.
    exporter = _exporter(tmp_path, background_color=(0, 0, 0))
    rendered = exporter._render_task(
        frame,
        task,
        [task.corners],
        (GEOMETRY.canvas_w, GEOMETRY.canvas_h),
    )
    assert rendered is not None

    expected = extract_canonical_crop(frame, task.affine, geometry=GEOMETRY)

    # Compare only inside the exporter's own expanded-OBB mask -- outside it the
    # exporter deliberately paints background (documented behaviour), so those
    # pixels are not expected to match.
    import cv2

    mask = np.zeros((GEOMETRY.canvas_h, GEOMETRY.canvas_w), np.uint8)
    poly = OrientedTrackVideoExporter._transform_polygon(
        task.expanded_corners, task.affine, task.out_w, task.out_h
    )
    cv2.fillPoly(mask, [poly], 255)
    inside = mask > 0
    assert inside.any()

    np.testing.assert_array_equal(
        rendered[inside],
        expected[inside],
    )


def test_foreign_mask_does_not_erase_the_subjects_own_body(tmp_path):
    """An overlapping neighbour must not blank out the subject's own OBB."""
    frame = _textured_frame()
    own = _obb(120.0, 100.0, 14.0, 7.0)
    # A neighbour that overlaps the subject's own OBB.
    foreign = _obb(130.0, 100.0, 14.0, 7.0)

    task = _task(own, GEOMETRY, polygon_index=0)
    exporter = _exporter(tmp_path, background_color=(0, 0, 0))

    rendered = exporter._render_task(
        frame,
        task,
        [own, foreign],
        (GEOMETRY.canvas_w, GEOMETRY.canvas_h),
        suppress_foreign_obb=True,
    )
    assert rendered is not None

    reference = extract_canonical_crop(frame, task.affine, geometry=GEOMETRY)

    import cv2

    own_mask = np.zeros((GEOMETRY.canvas_h, GEOMETRY.canvas_w), np.uint8)
    own_poly = OrientedTrackVideoExporter._transform_polygon(
        task.corners, task.affine, task.out_w, task.out_h
    )
    cv2.fillPoly(own_mask, [own_poly], 255)
    inside_own = own_mask > 0
    assert inside_own.any()

    np.testing.assert_array_equal(
        rendered[inside_own],
        reference[inside_own],
    )


@pytest.mark.parametrize("theta_deg", [0.0, 33.0, 90.0])
def test_parity_holds_for_rotated_boxes(tmp_path, theta_deg):
    """Rotation is where cv2 vs grid_sample sampling diverges most."""
    frame = _textured_frame()
    t = np.deg2rad(theta_deg)
    base = _obb(120.0, 100.0, 14.0, 7.0)
    c = base.mean(axis=0)
    rot = np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])
    corners = ((base - c) @ rot.T + c).astype(np.float32)

    task = _task(corners, GEOMETRY)
    exporter = _exporter(tmp_path, background_color=(0, 0, 0))
    rendered = exporter._render_task(
        frame, task, [task.corners], (GEOMETRY.canvas_w, GEOMETRY.canvas_h)
    )
    expected = extract_canonical_crop(frame, task.affine, geometry=GEOMETRY)

    import cv2

    mask = np.zeros((GEOMETRY.canvas_h, GEOMETRY.canvas_w), np.uint8)
    poly = OrientedTrackVideoExporter._transform_polygon(
        task.expanded_corners, task.affine, task.out_w, task.out_h
    )
    cv2.fillPoly(mask, [poly], 255)
    inside = mask > 0
    np.testing.assert_array_equal(rendered[inside], expected[inside])
