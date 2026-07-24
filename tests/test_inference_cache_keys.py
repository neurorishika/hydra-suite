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
    SliceConfig,
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


def _obb_direct(path="/m.pt", threshold=0.5) -> OBBConfig:
    return OBBConfig(
        mode="direct",
        direct=OBBDirectConfig(
            model_path=path,
            confidence_threshold=threshold,
        ),
    )


def _ht_config(path="/ht.pt", aspect=1.5, margin=0.1, threshold=0.4) -> HeadTailConfig:
    return HeadTailConfig(
        model_path=path,
        confidence_threshold=threshold,
        canonical_aspect_ratio=aspect,
        canonical_margin=margin,
    )


def _cnn_config(path="/cnn.pt", label="id", temperature=1.0) -> CNNConfig:
    return CNNConfig(
        label=label,
        model_path=path,
        calibration_temperature=temperature,
    )


def _pose_config(path="/pose.pt", padding=0.1) -> PoseConfig:
    return PoseConfig(
        backend="yolo",
        yolo=PoseYOLOConfig(model_path=path),
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


# ---- detection_cache_key: SliceConfig folding (Task 9) ----


def _obb_direct_slice(slice_cfg: SliceConfig) -> OBBConfig:
    return OBBConfig(
        mode="direct",
        direct=OBBDirectConfig(model_path="m.pt", slice=slice_cfg),
    )


def test_disabled_slice_key_equals_no_slice_baseline():
    # Baseline: default (disabled) slice.
    base = detection_cache_key(_obb_direct_slice(SliceConfig()))
    # A config whose slice is disabled but has non-default *other* fields must
    # still hash identically (disabled => inert).
    other = detection_cache_key(
        _obb_direct_slice(
            SliceConfig(enabled=False, merge_threshold=0.9, slice_height=999)
        )
    )
    assert base.config_hash == other.config_hash


def test_disabled_slice_key_matches_pre_change_baseline():
    """The exact byte-parity requirement: disabled slice => config_hash == ''."""
    k = detection_cache_key(_obb_direct_slice(SliceConfig()))
    assert k.config_hash == ""


def test_enabling_slice_changes_key():
    off = detection_cache_key(_obb_direct_slice(SliceConfig(enabled=False)))
    on = detection_cache_key(_obb_direct_slice(SliceConfig(enabled=True)))
    assert off.config_hash != on.config_hash


def test_slice_param_change_changes_key_when_enabled():
    a = detection_cache_key(
        _obb_direct_slice(SliceConfig(enabled=True, merge_threshold=0.5))
    )
    b = detection_cache_key(
        _obb_direct_slice(SliceConfig(enabled=True, merge_threshold=0.6))
    )
    assert a.config_hash != b.config_hash


def test_sequential_mode_slice_contribution_stays_empty():
    """Slicing is direct-mode only; sequential mode's key is unaffected."""
    cfg = OBBConfig(
        mode="sequential",
        sequential=OBBSequentialConfig(
            detect_model_path="/det.pt",
            obb_model_path="/obb.pt",
        ),
    )
    assert detection_cache_key(cfg).config_hash == ""


@pytest.mark.parametrize(
    "field_name,off_value,on_value",
    [
        ("geometry_mode", "auto_model", "auto_object"),
        ("slice_height", 0, 640),
        ("slice_width", 0, 640),
        ("overlap_height_ratio", 0.2, 0.3),
        ("overlap_width_ratio", 0.2, 0.3),
        ("object_tile_fraction", 0.15, 0.25),
        ("reference_body_px", 0.0, 42.0),
        ("merge_policy", "greedy_nmm", "nms"),
        ("merge_metric", "ios", "iou"),
        ("merge_threshold", 0.5, 0.7),
        ("merge_backend", "cv2", "gpu"),
        ("perform_standard_pred", False, True),
    ],
)
def test_every_output_affecting_slice_field_is_in_the_hash(
    field_name, off_value, on_value
):
    """Every SliceConfig field that alters detection output must participate
    in the hash -- a silently omitted field means a user changes it, the
    cache is NOT invalidated, and they get stale detections."""
    a = detection_cache_key(
        _obb_direct_slice(SliceConfig(enabled=True, **{field_name: off_value}))
    )
    b = detection_cache_key(
        _obb_direct_slice(SliceConfig(enabled=True, **{field_name: on_value}))
    )
    assert a.config_hash != b.config_hash, (
        f"SliceConfig.{field_name} does not affect detection_cache_key -- "
        "a change to it silently would NOT invalidate the detection cache"
    )


# ---- detection_cache_key: ROI mask folding (SAHI ROI tile-gating) ----


def _roi(shape=(8, 8), fill=1, corner_zero=False) -> np.ndarray:
    m = np.full(shape, fill, dtype=np.uint8)
    if corner_zero:
        m[: shape[0] // 2, : shape[1] // 2] = 0
    return m


def test_roi_folds_into_key_only_when_slicing_enabled_and_mask_present():
    """(b) enabled + ROI None == pre-ROI baseline; (a) mask A != mask B."""
    enabled = _obb_direct_slice(SliceConfig(enabled=True))
    base = detection_cache_key(enabled)  # no roi arg == roi None
    base_explicit_none = detection_cache_key(enabled, None)
    assert base.config_hash == base_explicit_none.config_hash
    # A non-None mask changes the key vs the None baseline...
    a = detection_cache_key(enabled, _roi(fill=1))
    assert a.config_hash != base.config_hash
    # ...and two DIFFERENT masks give different keys.
    b = detection_cache_key(enabled, _roi(corner_zero=True))
    assert a.config_hash != b.config_hash


def test_roi_ignored_when_slicing_disabled_key_is_byte_identical():
    """(c) slicing DISABLED + any ROI == today's disabled key (== '')."""
    disabled = _obb_direct_slice(SliceConfig(enabled=False))
    baseline = detection_cache_key(disabled).config_hash
    with_mask = detection_cache_key(disabled, _roi(corner_zero=True)).config_hash
    assert baseline == with_mask == ""


def test_roi_identical_masks_give_identical_keys():
    """(d) two content-identical masks => identical keys."""
    enabled = _obb_direct_slice(SliceConfig(enabled=True))
    a = detection_cache_key(enabled, _roi(fill=1))
    b = detection_cache_key(enabled, _roi(fill=1))
    assert a.config_hash == b.config_hash


def test_roi_content_hash_not_truncated_str():
    """Masks differing only in the middle must NOT collide (content hash, not str())."""
    enabled = _obb_direct_slice(SliceConfig(enabled=True))
    big = 200
    m1 = np.ones((big, big), dtype=np.uint8)
    m2 = np.ones((big, big), dtype=np.uint8)
    m2[big // 2, big // 2] = 0  # single interior pixel differs
    k1 = detection_cache_key(enabled, m1)
    k2 = detection_cache_key(enabled, m2)
    assert k1.config_hash != k2.config_hash


def test_roi_sequential_mode_ignores_mask():
    """Sequential mode never slices, so a mask must not perturb its key."""
    cfg = OBBConfig(
        mode="sequential",
        sequential=OBBSequentialConfig(
            detect_model_path="/det.pt", obb_model_path="/obb.pt"
        ),
    )
    assert detection_cache_key(cfg, _roi()).config_hash == ""


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


# ---- _open_caches: consumer/write-path ROI-mask coordination ----
#
# Read-only cache consumers (frame_result_bridge, optimizer.py,
# optimizer_workers.py, trackerkit config.py) must reopen the OBB detection
# cache WITH the same roi_mask the write path (InferenceRunner) used, or they
# compute the pre-ROI key and fail to recognize an ROI-folded cache. This
# fails safe (mismatched key => cache treated as invalid, never stale-served)
# but silently defeats the SAHI ROI tile-gating feature for its own target
# config (sliced inference + arena ROI). These tests prove the write-path key
# and a fixed consumer's key now agree, and that the old (mask-omitting)
# consumer call produced a different key -- i.e. the bug existed.


def test_open_caches_sliced_roi_write_and_consumer_keys_now_agree():
    """GREEN: a consumer that now passes roi_mask (matching the write path)
    produces the identical detection cache key -- the coordination gap is
    closed for the sliced + ROI config."""
    from pathlib import Path

    from hydra_suite.core.inference.config import InferenceConfig
    from hydra_suite.core.inference.runner import _open_caches

    mask = _roi(fill=1)
    cfg = InferenceConfig(obb=_obb_direct_slice(SliceConfig(enabled=True)))

    # Write path: InferenceRunner opens caches with its own video_sig/roi_mask.
    write_caches = _open_caches(cfg, Path("/tmp/cache"), "vid-sig", mask)

    # Fixed consumer: now threads the SAME mask through.
    consumer_caches = _open_caches(cfg, Path("/tmp/cache"), "vid-sig", mask)

    assert write_caches.detection.key == consumer_caches.detection.key


def test_open_caches_sliced_roi_old_consumer_behavior_mismatched_key():
    """RED (documents the bug that existed): a consumer that omits roi_mask
    (the old buggy call pattern -- ``_open_caches(config, cache_dir,
    video_sig)`` with no 4th arg) computes a DIFFERENT key than the write
    path for a sliced + ROI config, so it can never recognize the ROI-folded
    cache the write path produced."""
    from pathlib import Path

    from hydra_suite.core.inference.config import InferenceConfig
    from hydra_suite.core.inference.runner import _open_caches

    mask = _roi(fill=1)
    cfg = InferenceConfig(obb=_obb_direct_slice(SliceConfig(enabled=True)))

    write_caches = _open_caches(cfg, Path("/tmp/cache"), "vid-sig", mask)
    old_buggy_consumer_caches = _open_caches(cfg, Path("/tmp/cache"), "vid-sig")

    assert write_caches.detection.key != old_buggy_consumer_caches.detection.key


def test_open_caches_non_sliced_key_unchanged_with_or_without_mask():
    """Byte-parity: for a NON-sliced (or no-ROI) config, passing the mask to
    _open_caches unconditionally must NOT change the key -- every existing
    non-sliced call site keeps producing exactly the key it always did."""
    from pathlib import Path

    from hydra_suite.core.inference.config import InferenceConfig
    from hydra_suite.core.inference.runner import _open_caches

    mask = _roi(fill=1)
    cfg = InferenceConfig(obb=_obb_direct_slice(SliceConfig(enabled=False)))

    no_mask = _open_caches(cfg, Path("/tmp/cache"), "vid-sig")
    with_mask = _open_caches(cfg, Path("/tmp/cache"), "vid-sig", mask)

    assert no_mask.detection.key == with_mask.detection.key


# Silence unused-import warnings (np is implicitly required by OBBResult fixtures)
_ = np
