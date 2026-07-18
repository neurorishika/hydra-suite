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

    def fake_extract_canonical_crops(frame, obb, ar, mg, runtime, **kw):
        calls["crops_frame_shape"] = frame.shape
        return "CROPS_TENSOR"

    def fake_run_pose(crops, obb, model, cfg, runtime, ar, mg):
        calls["run_pose_crops"] = crops
        calls["run_pose_model"] = model
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
