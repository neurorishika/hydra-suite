from unittest.mock import MagicMock, patch

import numpy as np

from hydra_suite.core.inference.config import (
    InferenceConfig,
    OBBConfig,
    OBBDirectConfig,
)
from hydra_suite.core.inference.result import OBBResult


def _cfg() -> InferenceConfig:
    return InferenceConfig(
        obb=OBBConfig(
            mode="direct",
            direct=OBBDirectConfig(model_path="/m.pt", compute_runtime="cpu"),
            confidence_threshold=0.5,
        ),
    )


def _make_obb(n: int = 5, conf_values=None, frame_idx: int = 0) -> OBBResult:
    rng = np.random.default_rng(42)
    conf = (
        np.array(conf_values, dtype=np.float32)
        if conf_values is not None
        else rng.uniform(0.2, 1.0, n).astype(np.float32)
    )
    n_actual = len(conf)
    w = rng.uniform(10, 50, n_actual).astype(np.float32)
    h = rng.uniform(20, 80, n_actual).astype(np.float32)
    return OBBResult(
        frame_idx=frame_idx,
        centroids=rng.uniform(0, 640, (n_actual, 2)).astype(np.float32),
        angles=rng.uniform(0, np.pi, n_actual).astype(np.float32),
        sizes=(w * h).astype(np.float32),
        shapes=np.stack([w * h, h / w], axis=1).astype(np.float32),
        confidences=conf,
        corners=rng.uniform(0, 640, (n_actual, 4, 2)).astype(np.float32),
        detection_ids=OBBResult.make_detection_ids(frame_idx, n_actual),
    )


def test_filter_with_indices_returns_correct_indices():
    from hydra_suite.core.inference.stages.filtering import filter_with_indices

    raw = _make_obb(conf_values=[0.9, 0.1, 0.8, 0.2, 0.7])
    cfg = OBBConfig(
        mode="direct",
        direct=OBBDirectConfig(model_path="/m.pt"),
        confidence_threshold=0.5,
        iou_threshold=1.0,
    )
    filtered, indices = filter_with_indices(raw, cfg)
    assert set(indices.tolist()) == {0, 2, 4}
    assert filtered.num_detections == 3


def test_filter_with_indices_subsets_detection_ids():
    from hydra_suite.core.inference.stages.filtering import filter_with_indices

    raw = _make_obb(conf_values=[0.9, 0.1, 0.8, 0.2, 0.7], frame_idx=5)
    cfg = OBBConfig(
        mode="direct",
        direct=OBBDirectConfig(model_path="/m.pt"),
        confidence_threshold=0.5,
        iou_threshold=1.0,
    )
    filtered, indices = filter_with_indices(raw, cfg)
    expected = raw.detection_ids[[0, 2, 4]]
    np.testing.assert_array_equal(filtered.detection_ids, expected)


def test_filter_with_indices_empty_result():
    from hydra_suite.core.inference.stages.filtering import filter_with_indices

    raw = _make_obb(conf_values=[0.1, 0.2, 0.1])
    cfg = OBBConfig(
        mode="direct",
        direct=OBBDirectConfig(model_path="/m.pt"),
        confidence_threshold=0.5,
        iou_threshold=1.0,
    )
    filtered, indices = filter_with_indices(raw, cfg)
    assert len(indices) == 0
    assert filtered.num_detections == 0
    assert indices.dtype == np.int32


def test_inference_runner_init_loads_models():
    from hydra_suite.core.inference.runner import InferenceRunner

    cfg = _cfg()
    mock_models = MagicMock()
    with patch(
        "hydra_suite.core.inference.runner._load_all_models",
        return_value=mock_models,
    ) as mock_load:
        runner = InferenceRunner(cfg)
    mock_load.assert_called_once_with(
        cfg, runner.runtime, cache_only=False, video_path=None
    )
    assert runner._models is mock_models


def test_inference_runner_caches_all_valid_returns_false_when_no_cache_dir():
    from hydra_suite.core.inference.runner import InferenceRunner

    cfg = _cfg()
    with patch("hydra_suite.core.inference.runner._load_all_models"):
        runner = InferenceRunner(cfg, cache_dir=None)
    assert runner.caches_all_valid() is False


def test_run_realtime_persists_detection_cache_for_backward(tmp_path):
    """Realtime forward must persist detections so a backward pass can replay them
    (regression: realtime + backward previously produced an empty backward pass)."""
    from hydra_suite.core.inference.runner import InferenceRunner, _AllModels

    cfg = InferenceConfig(
        obb=OBBConfig(
            mode="direct",
            direct=OBBDirectConfig(model_path="/m.pt", compute_runtime="cpu"),
            confidence_threshold=0.5,
            iou_threshold=1.0,  # disable NMS for a deterministic count
        ),
    )
    # obb-only: no headtail/cnn/pose/apriltag -> caches_all_valid checks detection only
    models = _AllModels(
        obb=MagicMock(), headtail=None, cnn=[], pose=None, apriltag=None
    )
    frame = np.zeros((640, 640, 3), dtype=np.uint8)

    with (
        patch(
            "hydra_suite.core.inference.runner._load_all_models", return_value=models
        ),
        patch(
            "hydra_suite.core.inference.runner.run_obb",
            side_effect=lambda frames, *a, **k: [
                _make_obb(conf_values=[0.9, 0.8, 0.1])
            ],
        ),
    ):
        writer = InferenceRunner(cfg, cache_dir=tmp_path, video_path=None)
        for fi in range(3):
            writer.run_realtime(frame, fi)
        assert writer._caches_writable is True
        writer.close()  # flush to disk

        # A fresh runner (the backward pass) must see a valid, replayable cache.
        reader = InferenceRunner(cfg, cache_dir=tmp_path, video_path=None)
        assert reader.caches_all_valid() is True
        fr = reader.load_frame(2)
        assert fr.obb.num_detections == 2  # 0.9 & 0.8 pass the 0.5 conf gate


def test_inference_runner_close_calls_model_close():
    from hydra_suite.core.inference.runner import InferenceRunner, _AllModels

    cfg = _cfg()
    mock_obb = MagicMock()
    mock_ht = MagicMock()
    mock_cnn1 = MagicMock()
    mock_pose = MagicMock()
    mock_at = MagicMock()
    mock_models = _AllModels(
        obb=mock_obb,
        headtail=mock_ht,
        cnn=[mock_cnn1],
        pose=mock_pose,
        apriltag=mock_at,
    )
    with patch(
        "hydra_suite.core.inference.runner._load_all_models",
        return_value=mock_models,
    ):
        runner = InferenceRunner(cfg)
    runner.close()
    mock_obb.close.assert_called_once()
    mock_ht.close.assert_called_once()
    mock_cnn1.close.assert_called_once()
    mock_pose.close.assert_called_once()
    mock_at.close.assert_called_once()
