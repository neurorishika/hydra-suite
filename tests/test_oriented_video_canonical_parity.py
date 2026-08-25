"""Oriented-video frames ARE the shared canonical crop -- full canvas, same pixels.

The crop path migrated off the OpenCV affine kernel to the torch
`F.grid_sample` resampler (`canonicalization.crop.extract_canonical_crop`,
whose docstring records that it "replaces the former OpenCV affine-warp
kernel"). `OrientedTrackVideoExporter._render_task` was left behind on a
hand-rolled `cv2.warpAffine` AND on its own masking scheme, so an oriented
video frame was neither the same pixels nor the same framing as the canonical
crop every model and every exported dataset sees for that same detection.

Both are now the one shared call. In particular the exporter no longer paints
everything outside an expanded-OBB polygon with the background colour: the
frame is the full canonical canvas, exactly like a crop-dataset image, with
foreign-OBB suppression as the only optional masking.
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
BG = (233, 232, 231)


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


def _rotated(cx, cy, half_w, half_h, theta):
    base = _obb(cx, cy, half_w, half_h)
    c = base.mean(axis=0)
    rot = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    return ((base - c) @ rot.T + c).astype(np.float32)


def _textured_frame(h=200, w=240):
    """Non-uniform content so a resampler difference actually shows up.

    The pre-existing oriented-video tests render uniform-colour frames, where
    any two resamplers agree trivially -- which is why the divergence hid.
    """
    yy, xx = np.mgrid[0:h, 0:w]
    frame = np.zeros((h, w, 3), dtype=np.uint8)
    frame[:, :, 0] = (xx * 7 % 256).astype(np.uint8)
    frame[:, :, 1] = (yy * 11 % 256).astype(np.uint8)
    frame[:, :, 2] = ((xx + yy) * 5 % 256).astype(np.uint8)
    return frame


def _exporter(tmp_path, **kw):
    kw.setdefault("background_color", BG)
    return OrientedTrackVideoExporter(
        str(tmp_path / "ds"),
        str(tmp_path / "final.csv"),
        video_path=str(tmp_path / "in.mp4"),
        detection_cache_path=str(tmp_path / "detection.npz"),
        fps=5.0,
        geometry=GEOMETRY,
        **kw,
    )


def _task(corners, polygon_index=0):
    affine, _theta, _clipped = canonical_affine(corners, GEOMETRY)
    centre = corners.mean(axis=0)
    return FrameTask(
        frame_id=0,
        trajectory_id=0,
        affine=affine,
        out_w=GEOMETRY.canvas_w,
        out_h=GEOMETRY.canvas_h,
        center_x=float(centre[0]),
        center_y=float(centre[1]),
        width=10.0,
        height=5.0,
        theta=0.0,
        corners=corners,
        polygon_index=polygon_index,
    )


@pytest.mark.parametrize("theta_deg", [0.0, 17.0, 33.0, 90.0, 154.0])
def test_render_is_the_full_canonical_crop(tmp_path, theta_deg):
    """Every pixel of the frame must equal the shared canonical crop."""
    frame = _textured_frame()
    corners = _rotated(120.0, 100.0, 14.0, 7.0, np.deg2rad(theta_deg))
    task = _task(corners)

    rendered = _exporter(tmp_path)._render_task(
        frame, task, [task.corners], (GEOMETRY.canvas_w, GEOMETRY.canvas_h)
    )
    expected = extract_canonical_crop(frame, task.affine, geometry=GEOMETRY)

    assert rendered is not None
    assert (
        rendered.shape
        == expected.shape
        == (
            GEOMETRY.canvas_h,
            GEOMETRY.canvas_w,
            3,
        )
    )
    np.testing.assert_array_equal(rendered, expected)


def test_no_background_cutout_outside_the_obb(tmp_path):
    """The corners of the canvas must carry image content, not background.

    The retired expanded-OBB mask painted everything outside a rotated
    rectangle with `background_color`; full-canvas framing keeps it.
    """
    frame = _textured_frame()
    corners = _rotated(120.0, 100.0, 14.0, 7.0, np.deg2rad(30.0))
    task = _task(corners)

    rendered = _exporter(tmp_path)._render_task(
        frame, task, [task.corners], (GEOMETRY.canvas_w, GEOMETRY.canvas_h)
    )

    corners_px = np.array(
        [
            rendered[0, 0],
            rendered[0, -1],
            rendered[-1, 0],
            rendered[-1, -1],
        ]
    )
    assert not np.all(corners_px == np.array(BG)), (
        "canvas corners are still painted with the background colour -- the "
        "expanded-OBB cutout is still in effect"
    )


def test_foreign_suppression_matches_the_shared_masker(tmp_path):
    """Foreign masking must be the shared canonical masker, own body protected."""
    frame = _textured_frame()
    own = _obb(120.0, 100.0, 14.0, 7.0)
    foreign = _obb(130.0, 100.0, 14.0, 7.0)  # overlaps `own`
    task = _task(own, polygon_index=0)

    rendered = _exporter(tmp_path)._render_task(
        frame,
        task,
        [own, foreign],
        (GEOMETRY.canvas_w, GEOMETRY.canvas_h),
        suppress_foreign_obb=True,
    )
    expected = extract_canonical_crop(
        frame,
        task.affine,
        geometry=GEOMETRY,
        bg_color=BG,
        foreign_corners=[foreign],
        own_corners=own,
    )
    np.testing.assert_array_equal(rendered, expected)


def test_overlapping_neighbour_does_not_erase_the_subject(tmp_path):
    """Regression: a touching neighbour used to punch a hole in the subject."""
    frame = _textured_frame()
    own = _obb(120.0, 100.0, 14.0, 7.0)
    foreign = _obb(130.0, 100.0, 14.0, 7.0)
    task = _task(own, polygon_index=0)

    rendered = _exporter(tmp_path)._render_task(
        frame,
        task,
        [own, foreign],
        (GEOMETRY.canvas_w, GEOMETRY.canvas_h),
        suppress_foreign_obb=True,
    )
    unmasked = extract_canonical_crop(frame, task.affine, geometry=GEOMETRY)

    import cv2

    M = np.asarray(task.affine, dtype=np.float64)
    pts = (M[:, :2] @ np.asarray(own, np.float64).T + M[:, 2:]).T
    own_mask = np.zeros((GEOMETRY.canvas_h, GEOMETRY.canvas_w), np.uint8)
    cv2.fillPoly(own_mask, [pts.astype(np.int32).reshape(-1, 1, 2)], 255)
    inside_own = own_mask > 0
    assert inside_own.any()

    np.testing.assert_array_equal(rendered[inside_own], unmasked[inside_own])
