"""GAP 3 regression: the interpolated-crops pipeline's pose/CNN fallback path
for a degenerate OBB (``canonical_affine`` raises -- zero-length edge, see
``core/canonicalization/geometry.py::_axes``) must skip the detection loudly
instead of feeding a wrongly-scaled, un-canonicalized masked crop to the
backend.

Old behavior: ``_extract_pose_crop`` fell back to
``gen._extract_obb_masked_crop`` -- an axis-aligned crop with an arbitrary
aspect ratio and no Layer 1 rigid transform -- and ``_flush_pose_batch`` fed
it straight to the backend, bypassing Layer 2 entirely (``fit_to_model_input``
assumes the source is the canonical canvas, so ``apply_fit`` would have
silently mis-scaled it even if it *had* been called). A genuinely degenerate
OBB has no salvageable animal geometry to recover, so the fix skips instead
of fitting a crop that cannot be honestly fit.
"""

from __future__ import annotations

import logging

import numpy as np

from hydra_suite.core.canonicalization.geometry import CanonicalGeometry, ClippingStats
from hydra_suite.core.post import interpolated_crops as ic

_GEOMETRY = CanonicalGeometry.from_reference(20.0, 2.0, 1.3)


def _degenerate_corners() -> np.ndarray:
    # A zero-area OBB: every corner coincides, so both edges are
    # zero-length -- canonical_affine's _axes() raises ValueError.
    return np.zeros((4, 2), dtype=np.float32)


def test_extract_pose_crop_skips_degenerate_obb_instead_of_masked_fallback(caplog):
    class _UnusedGen:
        background_color = (0, 0, 0)

        def _extract_obb_masked_crop(self, *args, **kwargs):
            raise AssertionError(
                "the un-canonicalized masked-crop fallback must not be called "
                "for a degenerate OBB -- the fix skips instead of fitting one"
            )

    def _unused_extract_canonical(*args, **kwargs):
        raise AssertionError("Layer 1 extraction must not run when _aff is None")

    with caplog.at_level(logging.WARNING):
        pose_crop, pose_crop_info = ic._extract_pose_crop(
            task_idx=0,
            frame=np.zeros((10, 10, 3), dtype=np.uint8),
            _frame_all_corners=[_degenerate_corners()],
            _aff=None,  # what _compute_frame_corners_and_affines stores when
            # canonical_affine raised for this task's corners.
            corners=_degenerate_corners(),
            gen=_UnusedGen(),
            _extract_canonical=_unused_extract_canonical,
        )

    assert pose_crop is None
    assert pose_crop_info is None
    assert any("degenerate OBB" in rec.message for rec in caplog.records)


def test_process_single_task_adds_nothing_to_pending_batches_for_degenerate_obb():
    """End-to-end through the real call site: a degenerate-OBB task must not
    reach pending_crops/pending_cnn_crops at all (nothing to fit, nothing to
    predict on), rather than arriving as an un-fit, wrongly-scaled crop.
    """

    class _FakeGen:
        background_color = (0, 0, 0)

        def save_interpolated_crop(self, **kwargs):
            return ""

        def _extract_obb_masked_crop(self, *args, **kwargs):
            # The old fallback: a real (but un-canonicalized, arbitrary-aspect)
            # crop. If this pipeline still called it for a degenerate OBB, the
            # crop below would flow into pending_crops/pending_cnn_crops --
            # exactly the wrongly-scaled-crop bug this test guards against.
            return (
                np.full((7, 13, 3), 128, dtype=np.uint8),
                {"crop_size": (13, 7), "crop_bbox": (0, 0, 13, 7)},
            )

    task = {
        "cx": 0.0,
        "cy": 0.0,
        "w": 0.0,
        "h": 0.0,
        "theta": 0.0,
        "frame_id": 0,
        "traj_id": 1,
        "interp_from": (0, 1),
        "interp_index": 0,
        "interp_total": 1,
    }
    frame_corners, frame_affines = ic._compute_frame_corners_and_affines(
        [task], _GEOMETRY, ClippingStats()
    )
    assert frame_affines == [None]  # degenerate OBB -> canonical_affine raised

    pending_crops: list = []
    pending_entries: list = []
    pending_cnn_crops: list = []
    pending_cnn_entries: list = []

    def _unused_extract_canonical(*args, **kwargs):
        raise AssertionError("must not extract Layer 1 crop for a degenerate OBB")

    ic._process_single_task(
        task,
        0,
        np.zeros((10, 10, 3), dtype=np.uint8),
        frame_corners,
        frame_affines,
        _FakeGen(),
        False,  # save_interpolated_outputs
        _unused_extract_canonical,
        [object()],  # cnn_backends: non-empty so the CNN branch is exercised
        object(),  # pose_backend: non-None so the pose branch is exercised
        0,
        [],
        [],
        [],
        pending_crops,
        pending_entries,
        pending_cnn_crops,
        pending_cnn_entries,
    )

    assert pending_crops == []
    assert pending_entries == []
    assert pending_cnn_crops == []
    assert pending_cnn_entries == []
