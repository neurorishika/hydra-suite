"""predict_pose_for_image must wire crops -> run_pose correctly. It has never
run before this fix (it imported a nonexistent symbol)."""

import types

import numpy as np

import hydra_suite.core.inference.api as api


def test_predict_pose_for_image_wires_crops_to_run_pose(monkeypatch):
    calls = {}

    fake_model = object()

    def fake_load_pose_model(cfg, runtime):
        calls["loaded"] = True
        return fake_model

    def fake_extract_canonical_crops(frame, obb, geometry, runtime, **kw):
        calls["crops_frame_shape"] = frame.shape
        calls["crops_geometry"] = geometry
        return "CROPS_TENSOR"

    def fake_run_pose(crops, obb, model, cfg, runtime, geometry):
        calls["run_pose_crops"] = crops
        calls["run_pose_model"] = model
        calls["run_pose_geometry"] = geometry
        return "POSE_RESULT"  # scalar, not a list

    monkeypatch.setattr(
        "hydra_suite.core.inference.stages.pose.load_pose_model",
        fake_load_pose_model,
    )
    monkeypatch.setattr(
        "hydra_suite.core.inference.stages.pose.run_pose", fake_run_pose
    )
    monkeypatch.setattr(
        "hydra_suite.core.inference.stages.crops.extract_canonical_crops",
        fake_extract_canonical_crops,
    )

    pose_config = types.SimpleNamespace(
        yolo=types.SimpleNamespace(compute_runtime="cpu"),
        sleap=None,
    )
    image = np.zeros((64, 32, 3), dtype=np.uint8)

    result = api.predict_pose_for_image(image, pose_config)

    assert result == "POSE_RESULT"  # scalar returned, not results[0]
    assert calls["run_pose_crops"] == "CROPS_TENSOR"  # crops, not raw [image]
    assert calls["run_pose_model"] is fake_model
    assert calls["loaded"] is True
    assert calls["crops_frame_shape"] == (64, 32, 3)
    # extract_canonical_crops and run_pose must share the SAME geometry, or
    # keypoints get decoded against the wrong crop geometry.
    assert calls["crops_geometry"] is calls["run_pose_geometry"]
    assert calls["crops_geometry"].canvas_wh == (32, 64)  # (w, h) of the image
