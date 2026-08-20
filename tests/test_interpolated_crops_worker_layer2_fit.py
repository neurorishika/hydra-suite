"""Regression guard (Deviation B): the interpolated-crops pipeline must run
pose/CNN inference over interpolated crops through the SAME Layer2-fit-aware
stage functions ``Pipeline`` uses for real detections (``run_pose_batch`` /
``run_cnn_batch``, ``core/inference/stages/pose.py`` / ``.../cnn.py``) --
not a hand-rolled raw-crop path that bypasses the model's true input
geometry.

Old bug (pre-Task 10): ``_flush_pose_batch`` fed the raw canonical (Layer 1)
crop straight to ``pose_backend.predict_batch``, and ``_flush_cnn_batch`` fed
the same raw crop to ``cnn_backend.predict_batch``, which anisotropically
stretches it (``cv2.resize(crop, (w, h))``). Both bypassed the model's true
input geometry, so interpolated-frame identity/pose came out under a
different (or sheared) geometry than tracked-frame results from the same
model.

Task 10 deleted ``_flush_pose_batch``/``_flush_cnn_batch`` outright and
replaced them with ``_flush_pose_cnn_window``, which calls
``extract_canonical_crops_batch`` then ``run_pose_batch``/``run_cnn_batch``
per CNN phase -- the exact same stage functions ``Pipeline`` calls
(``pipeline.py:367-387``), so the Layer2-fit guarantee is now structural
(shared code, not a parallel re-implementation) rather than something this
module needs to re-verify numerically. What Task 12's wiring must still
prove is that ``_flush_pose_cnn_window`` actually delegates to those shared
functions (not a resurrected raw-crop path) and stamps the resulting rows
with the ``PoseSource``/``CNN_<label>_Source`` provenance columns Task 10
added.
"""

from __future__ import annotations

import numpy as np

from hydra_suite.core.canonicalization.geometry import CanonicalGeometry
from hydra_suite.core.inference.result import (
    CNNDetectionPrediction,
    CNNFactorPrediction,
    CNNResult,
    PoseResult,
)
from hydra_suite.core.post import interpolated_crops as ic
from hydra_suite.core.post.synthetic_detections import build_synthetic_obb_result


class _NoopProfiler:
    def tick(self, *args, **kwargs) -> None:
        return None

    def tock(self, *args, **kwargs) -> None:
        return None


_GEOMETRY = CanonicalGeometry.from_reference(20.0, 2.0, 1.3)
_CANVAS_W, _CANVAS_H = _GEOMETRY.canvas_wh


def _task(frame_id=0, traj_id=1) -> dict:
    return {
        "cx": 0.0,
        "cy": 0.0,
        "w": 20.0,
        "h": 10.0,
        "theta": 0.0,
        "frame_id": frame_id,
        "traj_id": traj_id,
        "interp_index": 0,
    }


def _frame() -> np.ndarray:
    return np.zeros((64, 64, 3), dtype=np.uint8)


class _FakePoseModel:
    keypoint_names = ["k0"]


class _FakeCNNModel:
    pass


def test_flush_pose_cnn_window_delegates_to_the_shared_stage_functions(monkeypatch):
    """``_flush_pose_cnn_window`` must build crops via
    ``extract_canonical_crops_batch`` and run pose/CNN via
    ``run_pose_batch``/``run_cnn_batch`` -- the SAME functions ``Pipeline``
    calls -- rather than any hand-rolled raw-crop resize path."""
    pose_module = __import__(
        "hydra_suite.core.inference.stages.pose", fromlist=["run_pose_batch"]
    )
    cnn_module = __import__(
        "hydra_suite.core.inference.stages.cnn", fromlist=["run_cnn_batch"]
    )
    crops_stage_module = __import__(
        "hydra_suite.core.inference.stages.crops",
        fromlist=["extract_canonical_crops_batch"],
    )

    task = _task()
    obb = build_synthetic_obb_result(0, [task])

    calls: dict = {}

    def _fake_extract_canonical_crops_batch(frames, obbs, geometry, runtime, **kwargs):
        calls["extract_canonical_crops_batch"] = (frames, obbs, geometry, kwargs)
        return "FAKE_CROP_BATCH"

    def _fake_run_pose_batch(crop_batch, model, pose_cfg, runtime, geometry):
        calls["run_pose_batch"] = (crop_batch, model, pose_cfg, geometry)
        kpts = np.array([[[1.0, 2.0, 0.9]]], dtype=np.float32)  # (D=1, K=1, 3)
        return {0: PoseResult(keypoints=kpts, valid_mask=np.array([True], dtype=bool))}

    def _fake_run_cnn_batch(frames, obbs, cnn_model, cnn_cfg, runtime, geometry):
        calls["run_cnn_batch"] = (frames, obbs, cnn_model, cnn_cfg, geometry)
        return {
            0: CNNResult(
                label="identity",
                predictions=[
                    CNNDetectionPrediction(
                        det_index=0,
                        factors=[
                            CNNFactorPrediction(
                                factor_name="identity",
                                class_names=["ant_a", "ant_b"],
                                raw_probabilities=np.array(
                                    [0.2, 0.8], dtype=np.float32
                                ),
                            )
                        ],
                    )
                ],
            )
        }

    monkeypatch.setattr(
        crops_stage_module,
        "extract_canonical_crops_batch",
        _fake_extract_canonical_crops_batch,
    )
    monkeypatch.setattr(pose_module, "run_pose_batch", _fake_run_pose_batch)
    monkeypatch.setattr(cnn_module, "run_cnn_batch", _fake_run_cnn_batch)

    class _FakeCfg:
        pose = type("P", (), {"min_keypoint_confidence": 0.2})()
        cnn_phases = [object()]

    interp_pose_rows: list = []
    interp_cnn_rows: dict = {}

    ic._flush_pose_cnn_window(
        [_frame()],
        [obb],
        [[task]],
        _FakePoseModel(),
        [_FakeCNNModel()],
        ["identity"],
        _FakeCfg(),
        runtime=None,
        geometry=_GEOMETRY,
        interp_pose_rows=interp_pose_rows,
        interp_cnn_rows=interp_cnn_rows,
        profiler=_NoopProfiler(),
    )

    # The shared stage functions were actually called (proving delegation,
    # not a resurrected hand-rolled raw-crop path).
    assert "extract_canonical_crops_batch" in calls
    assert "run_pose_batch" in calls
    assert "run_cnn_batch" in calls
    assert calls["run_pose_batch"][0] == "FAKE_CROP_BATCH"

    # Rows are stamped with the provenance columns Task 10 added.
    assert len(interp_pose_rows) == 1
    pose_row = interp_pose_rows[0]
    assert pose_row["PoseSource"] == "interp"
    assert pose_row["frame_id"] == task["frame_id"]
    assert pose_row["trajectory_id"] == task["traj_id"]
    assert pose_row["PoseKpt_k0_X"] == 1.0
    assert pose_row["PoseKpt_k0_Y"] == 2.0

    assert "identity" in interp_cnn_rows
    cnn_row = interp_cnn_rows["identity"][0]
    assert cnn_row["CNN_identity_Source"] == "interp"
    # argmax over [0.2, 0.8] -> "ant_b" at confidence 0.8.
    assert cnn_row["CNN_identity_Class"] == "ant_b"
    assert abs(cnn_row["CNN_identity_Conf"] - 0.8) < 1e-6
