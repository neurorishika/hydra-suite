import numpy as np
import pytest
import torch

from hydra_suite.core.inference.cache.base import CACHE_SCHEMA_VERSION, CacheKey
from hydra_suite.core.inference.cache.keys import (
    apriltag_cache_key,
    bgsub_detection_cache_key,
    cnn_cache_key,
    detection_cache_key,
    headtail_cache_key,
    pose_cache_key,
    video_signature,
    with_video_signature,
)
from hydra_suite.core.inference.config import (
    AprilTagConfig,
    BgSubConfig,
    CNNConfig,
    HeadTailConfig,
    OBBConfig,
    OBBDirectConfig,
    OBBSequentialConfig,
    PoseConfig,
    PoseYOLOConfig,
)
from hydra_suite.core.inference.result import OBBResult
from hydra_suite.core.inference.stages.obb import _RawOBBTensors, materialize_tensors


def _raw(n: int = 2) -> _RawOBBTensors:
    return _RawOBBTensors(
        frame_idx=3,
        xywhr=torch.tensor([[10.0, 20.0, 8.0, 4.0, 0.3]] * n),
        corners=torch.zeros(n, 4, 2),
        conf=torch.full((n,), 0.7),
    )


def _obb_direct(path="/m.pt", runtime="cpu", threshold=0.5) -> OBBConfig:
    return OBBConfig(
        mode="direct",
        direct=OBBDirectConfig(
            model_path=path,
            compute_runtime=runtime,
            confidence_threshold=threshold,
        ),
    )


def _ht_config(path="/ht.pt", aspect=1.5, margin=0.1, threshold=0.4) -> HeadTailConfig:
    return HeadTailConfig(
        model_path=path,
        compute_runtime="cpu",
        confidence_threshold=threshold,
        canonical_aspect_ratio=aspect,
        canonical_margin=margin,
    )


def _cnn_config(path="/cnn.pt", label="id", temperature=1.0) -> CNNConfig:
    return CNNConfig(
        label=label,
        model_path=path,
        compute_runtime="cpu",
        calibration_temperature=temperature,
    )


def _pose_config(path="/pose.pt", padding=0.1) -> PoseConfig:
    return PoseConfig(
        backend="yolo",
        yolo=PoseYOLOConfig(model_path=path, compute_runtime="cpu"),
        crop_padding=padding,
    )


def _at_config(family="tag36h11", decimate=1.0, blur=0.0) -> AprilTagConfig:
    return AprilTagConfig(enabled=True, tag_family=family, decimate=decimate, blur=blur)


# ---- materialize_tensors ----


def test_materialize_tensors_shape():
    raw = _raw(n=3)
    result = materialize_tensors(raw)
    assert isinstance(result, OBBResult)
    assert result.frame_idx == 3
    assert result.num_detections == 3
    assert result.centroids.shape == (3, 2)
    assert result.corners.shape == (3, 4, 2)
    assert result.confidences.shape == (3,)
    # Per Correction 14: detection_ids must be present
    assert result.detection_ids.shape == (3,)
    assert result.detection_ids[0] == 3 * 10000


def test_materialize_tensors_values():
    raw = _raw(n=1)
    result = materialize_tensors(raw)
    assert result.centroids[0, 0] == pytest.approx(10.0)
    assert result.centroids[0, 1] == pytest.approx(20.0)
    assert result.confidences[0] == pytest.approx(0.7)
    assert result.sizes[0] == pytest.approx(8.0 * 4.0)


def test_materialize_tensors_empty():
    raw = _RawOBBTensors(
        frame_idx=0,
        xywhr=torch.zeros((0, 5)),
        corners=torch.zeros((0, 4, 2)),
        conf=torch.zeros(0),
    )
    result = materialize_tensors(raw)
    assert result.num_detections == 0


# ---- CacheKey schema_version (Correction 16) ----


def test_cache_key_carries_schema_version():
    """Per Correction 16: every CacheKey is tagged with CACHE_SCHEMA_VERSION."""
    k = detection_cache_key(_obb_direct())
    assert k.schema_version == CACHE_SCHEMA_VERSION


def test_cache_key_matches_only_when_schema_version_matches():
    a = CacheKey(
        schema_version=2, model_path="/m.pt", model_mtime=12345.0, config_hash="x"
    )
    b = CacheKey(
        schema_version=2, model_path="/m.pt", model_mtime=12345.0, config_hash="x"
    )
    c = CacheKey(
        schema_version=1, model_path="/m.pt", model_mtime=12345.0, config_hash="x"
    )
    assert a.matches(b) is True
    assert a.matches(c) is False


def test_cache_key_matches_tolerates_small_mtime_diff():
    """Floating-point mtime can vary at the microsecond level — within 1ms is the same."""
    a = CacheKey(
        schema_version=2, model_path="/m.pt", model_mtime=12345.0, config_hash="x"
    )
    b = CacheKey(
        schema_version=2,
        model_path="/m.pt",
        model_mtime=12345.0001,
        config_hash="x",
    )
    assert a.matches(b) is True


# ---- detection_cache_key ----


def test_detection_key_changes_with_model_path():
    k1 = detection_cache_key(_obb_direct(path="/a.pt"))
    k2 = detection_cache_key(_obb_direct(path="/b.pt"))
    assert k1 != k2


def test_detection_key_stable_with_threshold():
    k1 = detection_cache_key(_obb_direct(threshold=0.3))
    k2 = detection_cache_key(_obb_direct(threshold=0.8))
    assert k1.model_path == k2.model_path
    assert k1.config_hash == k2.config_hash


def test_detection_key_sequential_encodes_both_models():
    cfg = OBBConfig(
        mode="sequential",
        sequential=OBBSequentialConfig(
            detect_model_path="/det.pt",
            obb_model_path="/obb.pt",
        ),
    )
    k = detection_cache_key(cfg)
    assert "/det.pt" in k.model_path and "/obb.pt" in k.model_path


# ---- bgsub_detection_cache_key ----


def test_bgsub_key_changes_with_detection_params():
    k1 = bgsub_detection_cache_key(BgSubConfig.from_params({"THRESHOLD_VALUE": 25}))
    k2 = bgsub_detection_cache_key(BgSubConfig.from_params({"THRESHOLD_VALUE": 100}))
    assert k1 != k2
    assert k1.model_path == "background_subtraction"


def test_bgsub_key_stable_for_same_params():
    params = {"THRESHOLD_VALUE": 25, "START_FRAME": 0, "END_FRAME": 499}
    assert bgsub_detection_cache_key(
        BgSubConfig.from_params(params)
    ) == bgsub_detection_cache_key(BgSubConfig.from_params(dict(params)))


def test_bgsub_key_video_bound():
    k = bgsub_detection_cache_key(BgSubConfig.from_params({"THRESHOLD_VALUE": 25}))
    assert with_video_signature(k, "111:222") != with_video_signature(k, "333:444")


# ---- video signature binding ----


def test_with_video_signature_noop_when_empty():
    k = detection_cache_key(_obb_direct())
    assert with_video_signature(k, "") == k


def test_with_video_signature_changes_key_and_differs_per_video():
    k = detection_cache_key(_obb_direct())
    k_a = with_video_signature(k, "100:111")
    k_b = with_video_signature(k, "200:222")
    # Binding a signature changes the key, and different videos yield different
    # keys — so a cache from one video is never reused for another.
    assert k_a != k
    assert k_a != k_b
    # Only config_hash is mixed; model identity fields are untouched.
    assert k_a.model_path == k.model_path
    assert k_a.model_mtime == k.model_mtime


def test_video_signature_changes_with_file_size(tmp_path):
    v = tmp_path / "clip.mp4"
    v.write_bytes(b"x" * 10)
    sig_small = video_signature(str(v))
    v.write_bytes(b"x" * 5000)  # regenerate same name, different content/size
    sig_big = video_signature(str(v))
    assert sig_small and sig_big and sig_small != sig_big


def test_video_signature_empty_for_missing_or_none():
    assert video_signature(None) == ""
    assert video_signature("/no/such/file.mp4") == ""


# ---- headtail_cache_key ----


def test_headtail_key_changes_with_model_path():
    k1 = headtail_cache_key(_ht_config(path="/a.pt"))
    k2 = headtail_cache_key(_ht_config(path="/b.pt"))
    assert k1 != k2


def test_headtail_key_stable_with_threshold():
    k1 = headtail_cache_key(_ht_config(threshold=0.3))
    k2 = headtail_cache_key(_ht_config(threshold=0.9))
    assert k1.model_path == k2.model_path
    assert k1.config_hash == k2.config_hash


def test_headtail_key_changes_with_canonical_params():
    k1 = headtail_cache_key(_ht_config(aspect=1.5, margin=0.1))
    k2 = headtail_cache_key(_ht_config(aspect=2.0, margin=0.1))
    assert k1.config_hash != k2.config_hash


# ---- cnn_cache_key ----


def test_cnn_key_stable_with_calibration_temperature():
    k1 = cnn_cache_key(_cnn_config(temperature=1.0))
    k2 = cnn_cache_key(_cnn_config(temperature=2.5))
    assert k1.model_path == k2.model_path
    assert k1.config_hash == k2.config_hash


def test_cnn_key_changes_with_model_path():
    k1 = cnn_cache_key(_cnn_config(path="/a.pt"))
    k2 = cnn_cache_key(_cnn_config(path="/b.pt"))
    assert k1 != k2


# ---- pose_cache_key ----


def test_pose_key_changes_with_crop_padding():
    k1 = pose_cache_key(_pose_config(padding=0.1))
    k2 = pose_cache_key(_pose_config(padding=0.3))
    assert k1.config_hash != k2.config_hash


# ---- apriltag_cache_key ----


def test_apriltag_key_changes_with_family():
    k1 = apriltag_cache_key(_at_config(family="tag36h11"))
    k2 = apriltag_cache_key(_at_config(family="tag25h9"))
    assert k1.config_hash != k2.config_hash


def test_apriltag_key_has_empty_model_path():
    k = apriltag_cache_key(_at_config())
    assert k.model_path == ""


# Silence unused-import warnings (np is implicitly required by OBBResult fixtures)
_ = np
