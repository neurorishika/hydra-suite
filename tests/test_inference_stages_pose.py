import json
from unittest.mock import MagicMock

import numpy as np
import torch

from hydra_suite.core.inference.config import PoseConfig, PoseYOLOConfig
from hydra_suite.core.inference.result import OBBResult
from hydra_suite.core.inference.runtime import RuntimeContext
from hydra_suite.core.inference.stages.pose import PoseModel, load_pose_model, run_pose


def _cpu_rt():
    return RuntimeContext(
        cuda_mode=False,
        device="cpu",
        use_nvdec=False,
        default_runtime="cpu",
        tensor_on_cuda=False,
    )


def _obb(n: int) -> OBBResult:
    return OBBResult(
        frame_idx=0,
        centroids=np.array([[100.0, 100.0]] * n, dtype=np.float32),
        angles=np.zeros(n, dtype=np.float32),
        sizes=np.ones(n, dtype=np.float32) * 400.0,
        shapes=np.ones((n, 2), dtype=np.float32),
        confidences=np.ones(n, dtype=np.float32),
        corners=np.array(
            [[[80, 90], [120, 90], [120, 110], [80, 110]]] * n, dtype=np.float32
        ),
        detection_ids=OBBResult.make_detection_ids(0, n),
    )


def _mock_pose_result(n_kpts: int = 4, conf: float = 0.8) -> MagicMock:
    """Mock pose-backend result holding a `.keypoints.data` (1, K, 3) tensor accessor."""
    r = MagicMock()
    kpts = np.zeros((1, n_kpts, 3), dtype=np.float32)
    kpts[0, :, 2] = conf
    r.keypoints.data.cpu.return_value.numpy.return_value = kpts
    return r


def test_run_pose_empty_crops():
    config = PoseConfig(yolo=PoseYOLOConfig(model_path="/p.pt"))
    model = PoseModel(
        backend=MagicMock(), n_keypoints=4, keypoint_names=["a", "b", "c", "d"]
    )
    crops = torch.zeros((0, 3, 64, 64))
    result = run_pose(crops, _obb(0), model, config, _cpu_rt())
    assert result.keypoints.shape == (0, 4, 3)
    assert result.valid_mask.shape == (0,)


def test_run_pose_shape():
    config = PoseConfig(
        yolo=PoseYOLOConfig(model_path="/p.pt"),
        min_keypoint_confidence=0.5,
    )
    mock_backend = MagicMock()
    mock_backend.predict_batch.return_value = [
        _mock_pose_result(4, conf=0.8),
        _mock_pose_result(4, conf=0.8),
    ]
    model = PoseModel(
        backend=mock_backend, n_keypoints=4, keypoint_names=["a", "b", "c", "d"]
    )
    crops = torch.zeros((2, 3, 64, 64))
    result = run_pose(crops, _obb(2), model, config, _cpu_rt())
    assert result.keypoints.shape == (2, 4, 3)
    assert result.valid_mask.shape == (2,)


def test_load_pose_model_reads_canonical_skeleton_keys(tmp_path, monkeypatch):
    """Regression: load_pose_model must read 'keypoint_names'/'skeleton_edges'
    (canonical skeleton JSON keys), not the wrong 'keypoints'/'edges' keys.
    Uses the YOLO branch with a stub backend to avoid loading a real model.
    """
    skel = tmp_path / "skel.json"
    skel.write_text(
        json.dumps(
            {
                "keypoint_names": ["head", "thorax", "abdomen"],
                "skeleton_edges": [[0, 1], [1, 2]],
            }
        )
    )

    import hydra_suite.core.identity.pose.backends.yolo as yolo_mod

    captured = {}

    class _StubBackend:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(yolo_mod, "YoloNativeBackend", _StubBackend)

    config = PoseConfig(
        backend="yolo",
        yolo=PoseYOLOConfig(model_path="/p.pt"),
        skeleton_file=str(skel),
    )
    model = load_pose_model(config, _cpu_rt())

    assert model.keypoint_names == ["head", "thorax", "abdomen"]
    assert model.n_keypoints == 3
    assert captured["keypoint_names"] == ["head", "thorax", "abdomen"]


def test_run_pose_valid_mask_high_conf():
    config = PoseConfig(
        yolo=PoseYOLOConfig(model_path="/p.pt"),
        min_keypoint_confidence=0.5,
        min_valid_keypoints=2,
    )
    mock_backend = MagicMock()
    r0 = _mock_pose_result(4, conf=0.9)
    r1 = _mock_pose_result(4, conf=0.1)
    mock_backend.predict_batch.return_value = [r0, r1]
    model = PoseModel(backend=mock_backend, n_keypoints=4, keypoint_names=list("abcd"))
    crops = torch.zeros((2, 3, 64, 64))
    result = run_pose(crops, _obb(2), model, config, _cpu_rt())
    assert bool(result.valid_mask[0]) is True
    assert bool(result.valid_mask[1]) is False
