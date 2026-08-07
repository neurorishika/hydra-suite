"""Test the ViTPose branch of PoseInferenceService.predict.

Verifies the preds-dict contract used by the PoseKit evaluation dashboard
(``evaluation.py::_score_one_frame``): ``{str(image_path): [(x, y, conf), ...]}``
with exactly ``num_kpts`` entries per image.

The real ``ViTPoseBackend.predict_batch`` (core/individual/pose/backends/vitpose.py)
returns a list of ``PoseResult`` (core/individual/pose/types.py), each carrying a
``keypoints`` ndarray of shape (K, 3) -- not a raw ndarray. The fake backend here
mirrors that exact return shape so the test exercises the real dict-shape
contract rather than an invented one.
"""

import numpy as np
import pytest

from hydra_suite.core.individual.pose.types import PoseResult
from hydra_suite.integrations.sleap.service import PoseInferenceService


class _FakeBackend:
    def predict_batch(self, crops):
        # One instance, 2 keypoints, per image -- matches PoseResult contract.
        kpts = np.array([[1.0, 2.0, 0.9], [3.0, 4.0, 0.8]], dtype=np.float32)
        return [
            PoseResult(
                keypoints=kpts,
                mean_conf=0.85,
                valid_fraction=1.0,
                num_valid=2,
                num_keypoints=2,
            )
            for _ in crops
        ]

    def close(self):
        pass


def test_vitpose_branch_returns_keypoint_dict(tmp_path, monkeypatch):
    ckpt = tmp_path / "best.pt"
    ckpt.write_bytes(b"x")
    img = tmp_path / "img0.png"
    img.write_bytes(b"x")

    captured_kwargs = {}

    def _fake_load_pose_backend(**kw):
        captured_kwargs.update(kw)
        return _FakeBackend()

    monkeypatch.setattr(
        "hydra_suite.integrations.sleap.service.load_pose_backend",
        _fake_load_pose_backend,
        raising=False,
    )
    # Stub image loading used by the vitpose branch to avoid decoding a fake PNG.
    monkeypatch.setattr(
        "hydra_suite.integrations.sleap.service._load_images_for_vitpose",
        lambda paths: [np.zeros((8, 8, 3), dtype=np.uint8) for _ in paths],
        raising=False,
    )

    svc = PoseInferenceService(tmp_path, ["a", "b"])
    preds, err = svc.predict(
        model_path=ckpt,
        image_paths=[img],
        device="cpu",
        imgsz=256,
        conf=0.0,
        batch=1,
        backend="vitpose",
        cache_predictions=False,
    )

    assert err == ""
    assert list(preds.keys()) == [str(img)]
    assert len(preds[str(img)]) == 2  # K=2 keypoints
    assert preds[str(img)][0] == pytest.approx((1.0, 2.0, 0.9))
    assert preds[str(img)][1] == pytest.approx((3.0, 4.0, 0.8))

    # Confirm the real load_pose_backend kwargs (core/inference/api.py:41) are used.
    assert captured_kwargs["backend_family"] == "vitpose"
    assert captured_kwargs["model_path"] == str(ckpt)
    assert captured_kwargs["compute_runtime"] == "cpu"
    assert captured_kwargs["vitpose_batch"] == 1


def test_vitpose_branch_rejects_missing_weights(tmp_path):
    svc = PoseInferenceService(tmp_path, ["a", "b"])
    missing = tmp_path / "nope.pt"
    preds, err = svc.predict(
        model_path=missing,
        image_paths=[tmp_path / "img0.png"],
        device="cpu",
        imgsz=256,
        conf=0.0,
        batch=1,
        backend="vitpose",
        cache_predictions=False,
    )
    assert preds is None
    assert "not found" in err.lower()
