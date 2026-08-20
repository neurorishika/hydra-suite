"""GAP 3 regression: the interpolated-crops pipeline's pose/CNN/AprilTag/
head-tail path for a degenerate OBB (``canonical_affine`` raises --
zero-length edge, see ``core/canonicalization/geometry.py::_axes``) must skip
the detection loudly instead of feeding a wrongly-scaled, un-canonicalized
masked crop to any backend.

Old behavior (pre-Task 12): ``_extract_pose_crop`` fell back to
``gen._extract_obb_masked_crop`` -- an axis-aligned crop with an arbitrary
aspect ratio and no Layer 1 rigid transform -- and ``_flush_pose_batch`` fed
it straight to the backend, bypassing Layer 2 entirely (``fit_to_model_input``
assumes the source is the canonical canvas, so ``apply_fit`` would have
silently mis-scaled it even if it *had* been called). A genuinely degenerate
OBB has no salvageable animal geometry to recover, so the fix skips instead
of fitting a crop that cannot be honestly fit.

Task 12 replaced ``_extract_pose_crop``/``_process_single_task`` with a
delegation to ``synthetic_detections.filter_degenerate_tasks`` (called from
``_compute_frame_corners_and_affines``): a degenerate task never survives
into ``kept_tasks``, so it never reaches ``build_synthetic_obb_result`` or
any of the windowed pose/CNN/AprilTag/head-tail stage calls at all -- there
is no more per-task pending_crops/pending_entries machinery to guard here.
"""

from __future__ import annotations

import logging

import numpy as np

from hydra_suite.core.canonicalization.geometry import CanonicalGeometry, ClippingStats
from hydra_suite.core.post import interpolated_crops as ic
from hydra_suite.core.post.synthetic_detections import filter_degenerate_tasks

_GEOMETRY = CanonicalGeometry.from_reference(20.0, 2.0, 1.3)


def _degenerate_task(frame_id=0, traj_id=1) -> dict:
    # w=h=0 -> every ellipse-derived OBB corner coincides -> both edges are
    # zero-length -- canonical_affine's _axes() raises ValueError.
    return {
        "cx": 0.0,
        "cy": 0.0,
        "w": 0.0,
        "h": 0.0,
        "theta": 0.0,
        "frame_id": frame_id,
        "traj_id": traj_id,
        "interp_from": (0, 1),
        "interp_index": 0,
        "interp_total": 1,
    }


def _fitting_task(frame_id=0, traj_id=2) -> dict:
    return {
        "cx": 0.0,
        "cy": 0.0,
        "w": 20.0,
        "h": 10.0,
        "theta": 0.0,
        "frame_id": frame_id,
        "traj_id": traj_id,
        "interp_from": (0, 1),
        "interp_index": 0,
        "interp_total": 1,
    }


def test_filter_degenerate_tasks_drops_degenerate_obb_and_warns(caplog):
    """The stage-layer entry point (``filter_degenerate_tasks``, called from
    ``_compute_frame_corners_and_affines``) must loudly skip a degenerate OBB
    rather than let it flow into ``build_synthetic_obb_result``."""
    with caplog.at_level(logging.WARNING):
        kept = filter_degenerate_tasks(
            [_degenerate_task(), _fitting_task()], _GEOMETRY, ClippingStats()
        )

    assert len(kept) == 1
    assert kept[0]["traj_id"] == 2
    assert any("degenerate OBB" in rec.message for rec in caplog.records)


def test_process_single_frame_never_builds_a_synthetic_obb_for_a_degenerate_only_frame(
    monkeypatch,
):
    """End-to-end through the real call site: a frame whose only task is a
    degenerate OBB must never reach ``build_synthetic_obb_result`` -- nothing
    to fit, nothing to predict on, nothing to detect AprilTags/head-tail in --
    rather than arriving as an un-fit, wrongly-scaled crop.
    """

    class _FakeGen:
        background_color = (0, 0, 0)

        def save_interpolated_crop(self, **kwargs):
            raise AssertionError(
                "save_interpolated_outputs is False; save_interpolated_crop "
                "must not be called"
            )

    def _unused_build_synthetic_obb_result(*args, **kwargs):
        raise AssertionError(
            "a frame with only a degenerate-OBB task must never reach "
            "build_synthetic_obb_result"
        )

    monkeypatch.setattr(
        "hydra_suite.core.post.synthetic_detections.build_synthetic_obb_result",
        _unused_build_synthetic_obb_result,
    )

    task = _degenerate_task()
    frame_tasks = {0: [task]}
    interp_saved = 0
    interp_rows: list = []
    roi_rows: list = []
    roi_corners: list = []
    interp_tag_rows: list = []
    pending_frames: list = []
    pending_obbs: list = []
    pending_tasks_by_frame: list = []

    result = ic._process_single_frame(
        {},
        None,
        None,
        0,
        1,
        np.zeros((10, 10, 3), dtype=np.uint8),
        1,
        frame_tasks,
        _FakeGen(),
        False,  # save_interpolated_outputs
        _GEOMETRY,
        ClippingStats(),
        None,  # apriltag_model
        None,  # apriltag_cfg
        interp_saved,
        interp_rows,
        roi_rows,
        roi_corners,
        interp_tag_rows,
        pending_frames,
        pending_obbs,
        pending_tasks_by_frame,
    )

    assert result == 0
    assert pending_frames == []
    assert pending_obbs == []
    assert pending_tasks_by_frame == []
