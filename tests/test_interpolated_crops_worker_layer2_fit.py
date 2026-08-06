"""Regression guard (Deviation B): the interpolated-crops pipeline must pre-fit
every Layer 1 canonical crop through Layer 2 (``fit_to_model_input`` /
``apply_fit``) before handing it to a pose or CNN backend -- exactly like
``core/inference/stages/pose.py`` / ``core/inference/stages/cnn.py`` do for
the tracked-frame path.

Bug: `_flush_pose_batch` fed the raw canonical (Layer 1) crop straight to
`pose_backend.predict_batch`, and `_flush_cnn_batch` fed the same raw crop to
`cnn_backend.predict_batch`, which anisotropically stretches it
(`cv2.resize(crop, (w, h))`). Both bypassed the model's true input geometry,
so interpolated-frame identity/pose came out under a different (or sheared)
geometry than tracked-frame results from the same model.
"""

from __future__ import annotations

import numpy as np

from hydra_suite.core.canonicalization.geometry import CanonicalGeometry
from hydra_suite.core.post import interpolated_crops as ic


class _NoopProfiler:
    def tick(self, *args, **kwargs) -> None:
        return None

    def tock(self, *args, **kwargs) -> None:
        return None


_GEOMETRY = CanonicalGeometry.from_reference(20.0, 2.0, 1.3)
_CANVAS_W, _CANVAS_H = _GEOMETRY.canvas_wh


def _canonical_crop() -> np.ndarray:
    return np.full((_CANVAS_H, _CANVAS_W, 3), 128, dtype=np.uint8)


def test_flush_pose_batch_hands_backend_the_models_input_size_not_the_canvas():
    captured: dict = {}

    class FakePoseBackend:
        preferred_input_size = 0  # signals "has a preferred_input_wh instead"

        @property
        def preferred_input_wh(self):
            return (96, 64)  # non-square, and != the canonical canvas size

        def predict_batch(self, crops):
            captured["shapes"] = [c.shape for c in crops]
            return [None for _ in crops]

    entries = [
        {
            "task": {"frame_id": 0, "traj_id": 1},
            "filename": "x.png",
            "crop_info": {"canonical": True, "M_forward": np.eye(2, 3)},
        }
    ]
    pending_crops = [_canonical_crop()]
    interp_pose_rows: list = []

    ic._flush_pose_batch(
        FakePoseBackend(),
        pending_crops,
        entries,
        interp_pose_rows,
        [],
        [],
        _NoopProfiler(),
        _GEOMETRY,
    )

    assert captured["shapes"] == [(64, 96, 3)]
    # The raw canvas must NOT have reached the backend unchanged.
    assert captured["shapes"][0][:2] != (_CANVAS_H, _CANVAS_W)
    # pending_crops/pending_entries are drained same as before the fix.
    assert pending_crops == []
    assert entries == []


def test_flush_cnn_batch_hands_backend_the_models_input_size_not_the_canvas():
    captured: dict = {}

    class _Meta:
        input_size = (48, 32)  # (H, W)

    class FakeCNNBackend:
        metadata = _Meta()

        def predict_batch(self, crops):
            captured["shapes"] = [c.shape for c in crops]
            return []

    pending_cnn_crops = [_canonical_crop()]
    pending_cnn_entries = [{"task": {"frame_id": 0, "traj_id": 1}}]
    interp_cnn_rows: dict = {"cnn": []}

    ic._flush_cnn_batch(
        [FakeCNNBackend()],
        ["cnn"],
        pending_cnn_crops,
        pending_cnn_entries,
        interp_cnn_rows,
        _NoopProfiler(),
        _GEOMETRY,
    )

    assert captured["shapes"] == [(48, 32, 3)]
    assert captured["shapes"][0][:2] != (_CANVAS_H, _CANVAS_W)


def test_flush_pose_batch_composes_layer2_fit_with_layer1_affine_for_backprojection():
    """A keypoint at the exact Layer2-fitted location of the canonical
    canvas's centre must round-trip back to that centre -- proving the
    Layer2 fit affine is composed with the Layer1 affine before inverting,
    not applied/ignored inconsistently.
    """
    from hydra_suite.core.canonicalization.fit import fit_affine, fit_to_model_input
    from hydra_suite.core.identity.pose.types import PoseResult as BackendPoseResult

    model_wh = (96, 64)
    fit = fit_to_model_input(_GEOMETRY.canvas_wh, model_wh)
    fit_m = fit_affine(fit)
    canvas_center = np.array([_CANVAS_W / 2.0, _CANVAS_H / 2.0, 1.0], dtype=np.float64)
    model_space_center = fit_m @ canvas_center  # where Layer2 puts it

    class FakePoseBackend:
        preferred_input_size = 0

        @property
        def preferred_input_wh(self):
            return model_wh

        def predict_batch(self, crops):
            kpts = np.array(
                [[model_space_center[0], model_space_center[1], 1.0]],
                dtype=np.float32,
            )
            return [
                BackendPoseResult(
                    keypoints=kpts,
                    mean_conf=1.0,
                    valid_fraction=1.0,
                    num_valid=1,
                    num_keypoints=1,
                )
            ]

    entries = [
        {
            "task": {"frame_id": 0, "traj_id": 1},
            "filename": "x.png",
            # Identity Layer 1 affine: canonical canvas == image coords.
            "crop_info": {"canonical": True, "M_forward": np.eye(2, 3)},
        }
    ]
    pending_crops = [_canonical_crop()]
    interp_pose_rows: list = []

    ic._flush_pose_batch(
        FakePoseBackend(),
        pending_crops,
        entries,
        interp_pose_rows,
        ["k0"],
        ["k0"],
        _NoopProfiler(),
        _GEOMETRY,
    )

    row = interp_pose_rows[0]
    assert abs(row["PoseKpt_k0_X"] - _CANVAS_W / 2.0) < 1e-3
    assert abs(row["PoseKpt_k0_Y"] - _CANVAS_H / 2.0) < 1e-3
