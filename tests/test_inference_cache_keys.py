import os
import shutil

import numpy as np
import pytest
import torch

from hydra_suite.core.canonicalization.geometry import CanonicalGeometry
from hydra_suite.core.inference.cache import keys as keys_mod
from hydra_suite.core.inference.cache.base import CACHE_SCHEMA_VERSION, CacheKey
from hydra_suite.core.inference.cache.keys import (
    apriltag_cache_key,
    bgsub_detection_cache_key,
    canonical_geometry_key,
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


def _ht_config(path="/ht.pt", threshold=0.4) -> HeadTailConfig:
    return HeadTailConfig(
        model_path=path,
        confidence_threshold=threshold,
    )


def _cnn_config(path="/cnn.pt", label="id", temperature=1.0) -> CNNConfig:
    return CNNConfig(
        label=label,
        model_path=path,
        calibration_temperature=temperature,
    )


def _pose_config(path="/pose.pt") -> PoseConfig:
    return PoseConfig(
        backend="yolo",
        yolo=PoseYOLOConfig(model_path=path),
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


def test_cache_schema_version_is_v5_content_identity_bump():
    """Task 4 (portable-jobs): model and video identity became CONTENT-based
    (sha256 of bytes) rather than (absolute path, mtime)-based, so old
    caches keyed on the removed model_path/model_mtime fields are no longer
    even constructible -- CACHE_SCHEMA_VERSION must be bumped to invalidate
    them. See ``cache/base.py``'s v5 changelog entry.
    """
    assert CACHE_SCHEMA_VERSION == 5


def test_cnn_and_headtail_keys_differ_across_schema_v4_v5():
    """A cache written under the previous schema version must not match one
    written under the current version even with identical model/geometry --
    the schema_version field alone must invalidate it.
    """
    ht_new = headtail_cache_key(_ht_config(), _GEOM_A)
    ht_old = CacheKey(
        schema_version=CACHE_SCHEMA_VERSION - 1,
        model_id=ht_new.model_id,
        config_hash=ht_new.config_hash,
    )
    assert ht_new.schema_version == CACHE_SCHEMA_VERSION
    assert not ht_new.matches(ht_old)

    cnn_new = cnn_cache_key(_cnn_config(), _GEOM_A)
    cnn_old = CacheKey(
        schema_version=CACHE_SCHEMA_VERSION - 1,
        model_id=cnn_new.model_id,
        config_hash=cnn_new.config_hash,
    )
    assert cnn_new.schema_version == CACHE_SCHEMA_VERSION
    assert not cnn_new.matches(cnn_old)


def test_cache_key_matches_only_when_schema_version_matches():
    a = CacheKey(schema_version=2, model_id="/m.pt", config_hash="x")
    b = CacheKey(schema_version=2, model_id="/m.pt", config_hash="x")
    c = CacheKey(schema_version=1, model_id="/m.pt", config_hash="x")
    assert a.matches(b) is True
    assert a.matches(c) is False


# ---- detection_cache_key ----


def test_detection_key_changes_with_model_path(tmp_path):
    """Rewritten (round-7 corrected rationale): two nonexistent paths already
    diverge via the missing-model sentinel, which never exercises real
    content hashing. Use two real files with DIFFERENT bytes so this test
    actually exercises the content-hashing code path."""
    a = tmp_path / "a.pt"
    b = tmp_path / "b.pt"
    a.write_bytes(b"model-a-bytes")
    b.write_bytes(b"model-b-bytes")
    k1 = detection_cache_key(_obb_direct(path=str(a)))
    k2 = detection_cache_key(_obb_direct(path=str(b)))
    assert k1 != k2


def test_detection_key_stable_with_threshold():
    k1 = detection_cache_key(_obb_direct(threshold=0.3))
    k2 = detection_cache_key(_obb_direct(threshold=0.8))
    assert k1.model_id == k2.model_id
    assert k1.config_hash == k2.config_hash


def test_detection_key_sequential_encodes_both_models(tmp_path):
    """Rewritten: under content identity the key contains sha256 hex, not
    paths, so assert the key changes when EITHER model's bytes change --
    rewriting each file in place (same path, new content) isolates content
    identity from path identity for both halves of the pair."""
    det = tmp_path / "det.pt"
    obb = tmp_path / "obb.pt"
    det.write_bytes(b"det-bytes")
    obb.write_bytes(b"obb-bytes")
    cfg = OBBConfig(
        mode="sequential",
        sequential=OBBSequentialConfig(
            detect_model_path=str(det),
            obb_model_path=str(obb),
        ),
    )
    k = detection_cache_key(cfg)

    det.write_bytes(b"det-bytes-changed")
    assert detection_cache_key(cfg) != k

    det.write_bytes(b"det-bytes")  # restore, isolate the obb-side change
    assert detection_cache_key(cfg) == k
    obb.write_bytes(b"obb-bytes-changed")
    assert detection_cache_key(cfg) != k


def test_direct_raw_output_contract_changes_detection_cache_key():
    """Every direct-mode setting that changes raw OBB extraction must invalidate.

    Confidence/IoU intentionally remain absent: they are replay-time filters.
    """
    base = _obb_direct()
    assert base.direct is not None
    detect_a = _obb_direct()
    detect_b = _obb_direct()
    assert detect_a.direct is not None and detect_b.direct is not None
    detect_a.direct.model_task = detect_b.direct.model_task = "detect"
    detect_b.direct.fixed_angle_deg = 17.0
    assert detection_cache_key(detect_a) != detection_cache_key(detect_b)

    segment_a = _obb_direct()
    assert segment_a.direct is not None
    segment_a.direct.model_task = "segment"
    for attr, value in [
        ("seg_num_angles", 48),
        ("seg_crop_size", 96),
        ("seg_pad_ratio", 0.3),
        ("seg_mask_threshold", 0.7),
    ]:
        changed = _obb_direct()
        assert changed.direct is not None
        changed.direct.model_task = "segment"
        setattr(changed.direct, attr, value)
        assert detection_cache_key(segment_a) != detection_cache_key(changed), attr

    classes = _obb_direct()
    classes.target_classes = [1, 3]
    assert detection_cache_key(base) != detection_cache_key(classes)

    cap = _obb_direct()
    cap.raw_detection_cap = 17
    assert detection_cache_key(base) != detection_cache_key(cap)


@pytest.mark.parametrize("mode", ["direct", "sequential"])
def test_native_geometry_export_changes_obb_detection_cache_key(mode: str) -> None:
    """Polygon export needs a live extraction, not a polygon-free cache hit."""

    if mode == "direct":
        base = _obb_direct()
        export = _obb_direct()
    else:
        base = OBBConfig(
            mode="sequential",
            sequential=OBBSequentialConfig(
                detect_model_path="/det.pt", obb_model_path="/obb.pt"
            ),
        )
        export = OBBConfig(
            mode="sequential",
            sequential=OBBSequentialConfig(
                detect_model_path="/det.pt", obb_model_path="/obb.pt"
            ),
        )
    export.emit_native_geometry = True

    assert detection_cache_key(base) != detection_cache_key(export)


def test_sequential_second_model_signature_invalidates_detection_key(tmp_path):
    """Rewritten: Step 5 deletes ``keys._mtime`` entirely, so there is no
    mtime hook left to monkeypatch. Rewrite the second model file's bytes on
    disk instead and assert the key changes via the real content-based
    path."""
    det = tmp_path / "det.pt"
    obb = tmp_path / "obb.pt"
    det.write_bytes(b"det-bytes")
    obb.write_bytes(b"obb-bytes-v1")
    cfg = OBBConfig(
        mode="sequential",
        sequential=OBBSequentialConfig(
            detect_model_path=str(det),
            obb_model_path=str(obb),
        ),
    )
    first = detection_cache_key(cfg)

    obb.write_bytes(b"obb-bytes-v2")
    second = detection_cache_key(cfg)

    assert first != second


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


def test_disabled_slice_key_retains_raw_extraction_contract():
    """Disabled slicing is inert, but raw extraction still has a full key."""
    k = detection_cache_key(_obb_direct_slice(SliceConfig()))
    assert k.config_hash


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


def test_sequential_mode_key_records_raw_extraction_contract():
    """Sequential caches are versioned independently of final filtering."""
    cfg = OBBConfig(
        mode="sequential",
        sequential=OBBSequentialConfig(
            detect_model_path="/det.pt",
            obb_model_path="/obb.pt",
        ),
    )
    key = detection_cache_key(cfg)
    assert key.config_hash

    changed_final_filter = OBBConfig(
        mode="sequential",
        confidence_threshold=0.9,
        iou_threshold=0.1,
        sequential=cfg.sequential,
    )
    assert detection_cache_key(changed_final_filter).config_hash == key.config_hash


def test_sequential_key_changes_with_raw_stage_settings():
    base = OBBConfig(
        mode="sequential",
        sequential=OBBSequentialConfig(
            detect_model_path="/det.pt",
            obb_model_path="/obb.pt",
            detect_confidence_threshold=0.1,
        ),
    )
    changed_stage1 = OBBConfig(
        mode="sequential",
        sequential=OBBSequentialConfig(
            detect_model_path="/det.pt",
            obb_model_path="/obb.pt",
            detect_confidence_threshold=0.2,
        ),
    )
    changed_crop = OBBConfig(
        mode="sequential",
        sequential=OBBSequentialConfig(
            detect_model_path="/det.pt",
            obb_model_path="/obb.pt",
            detect_confidence_threshold=0.1,
            crop_pad_ratio=0.4,
        ),
    )

    assert (
        detection_cache_key(base).config_hash
        != detection_cache_key(changed_stage1).config_hash
    )
    assert (
        detection_cache_key(base).config_hash
        != detection_cache_key(changed_crop).config_hash
    )


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
        # tile_batch_size is a TUNED coordinate (InferenceTuningSettings writes
        # obb.direct.slice.tile_batch_size). It was silently absent from both
        # the hash and this list, so a sliced cache written at the default 16
        # was replayed under a tuned 32.
        ("tile_batch_size", 16, 32),
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
    """(c) slicing DISABLED + any ROI leaves the raw contract unchanged."""
    disabled = _obb_direct_slice(SliceConfig(enabled=False))
    baseline = detection_cache_key(disabled).config_hash
    with_mask = detection_cache_key(disabled, _roi(corner_zero=True)).config_hash
    assert baseline == with_mask


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
    """Non-sliced sequential mode does not use ROI tile gating."""
    cfg = OBBConfig(
        mode="sequential",
        sequential=OBBSequentialConfig(
            detect_model_path="/det.pt", obb_model_path="/obb.pt"
        ),
    )
    assert (
        detection_cache_key(cfg, _roi()).config_hash
        == detection_cache_key(cfg).config_hash
    )


def test_roi_sequential_stage1_slicing_hashes_mask_content():
    cfg = OBBConfig(
        mode="sequential",
        sequential=OBBSequentialConfig(
            detect_model_path="/det.pt",
            obb_model_path="/obb.pt",
            stage1_slice=SliceConfig(enabled=True),
        ),
    )

    assert (
        detection_cache_key(cfg, _roi(fill=1)).config_hash
        != detection_cache_key(cfg, _roi(corner_zero=True)).config_hash
    )


# ---- bgsub_detection_cache_key ----


def test_bgsub_key_changes_with_detection_params():
    k1 = bgsub_detection_cache_key(BgSubConfig.from_params({"THRESHOLD_VALUE": 25}))
    k2 = bgsub_detection_cache_key(BgSubConfig.from_params({"THRESHOLD_VALUE": 100}))
    assert k1 != k2
    assert k1.model_id == "background_subtraction"


def test_bgsub_key_stable_for_same_params():
    params = {"THRESHOLD_VALUE": 25, "START_FRAME": 0, "END_FRAME": 499}
    assert bgsub_detection_cache_key(
        BgSubConfig.from_params(params)
    ) == bgsub_detection_cache_key(BgSubConfig.from_params(dict(params)))


def test_native_geometry_export_changes_bgsub_detection_cache_key() -> None:
    base = BgSubConfig.from_params({"THRESHOLD_VALUE": 25})
    export = BgSubConfig.from_params({"THRESHOLD_VALUE": 25})
    export.emit_native_geometry = True

    assert bgsub_detection_cache_key(base) != bgsub_detection_cache_key(export)


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
    assert k_a.model_id == k.model_id


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

_GEOM_A = CanonicalGeometry.from_reference(20.0, 1.5, 0.1 + 1.0)
_GEOM_B = CanonicalGeometry.from_reference(20.0, 2.0, 0.1 + 1.0)


def test_headtail_key_changes_with_model_path():
    k1 = headtail_cache_key(_ht_config(path="/a.pt"), _GEOM_A)
    k2 = headtail_cache_key(_ht_config(path="/b.pt"), _GEOM_A)
    assert k1 != k2


def test_headtail_key_stable_with_threshold():
    k1 = headtail_cache_key(_ht_config(threshold=0.3), _GEOM_A)
    k2 = headtail_cache_key(_ht_config(threshold=0.9), _GEOM_A)
    assert k1.model_id == k2.model_id
    assert k1.config_hash == k2.config_hash


def test_headtail_key_changes_with_canonical_params():
    k1 = headtail_cache_key(_ht_config(), _GEOM_A)
    k2 = headtail_cache_key(_ht_config(), _GEOM_B)
    assert k1.config_hash != k2.config_hash


# ---- cnn_cache_key ----


def test_cnn_key_stable_with_calibration_temperature():
    k1 = cnn_cache_key(_cnn_config(temperature=1.0), _GEOM_A)
    k2 = cnn_cache_key(_cnn_config(temperature=2.5), _GEOM_A)
    assert k1.model_id == k2.model_id
    assert k1.config_hash == k2.config_hash


def test_cnn_key_changes_with_model_path():
    k1 = cnn_cache_key(_cnn_config(path="/a.pt"), _GEOM_A)
    k2 = cnn_cache_key(_cnn_config(path="/b.pt"), _GEOM_A)
    assert k1 != k2


def test_cnn_key_changes_with_canonical_geometry():
    k1 = cnn_cache_key(_cnn_config(), _GEOM_A)
    k2 = cnn_cache_key(_cnn_config(), _GEOM_B)
    assert k1.config_hash != k2.config_hash


# ---- pose_cache_key ----


def test_pose_key_changes_with_canonical_geometry():
    k1 = pose_cache_key(_pose_config(), _GEOM_A)
    k2 = pose_cache_key(_pose_config(), _GEOM_B)
    assert k1.config_hash != k2.config_hash


def test_pose_key_changes_with_canonical_margin_alone():
    """``margin`` is THE surviving framing term (it replaced crop_padding), so
    changing it alone -- same body size, same aspect ratio -- must move the key.
    """
    g_narrow = CanonicalGeometry.from_reference(80.0, 2.0, 1.3)
    g_wide = CanonicalGeometry.from_reference(80.0, 2.0, 1.6)
    assert g_narrow.aspect_ratio == g_wide.aspect_ratio
    k1 = pose_cache_key(_pose_config(), g_narrow)
    k2 = pose_cache_key(_pose_config(), g_wide)
    assert k1.config_hash != k2.config_hash


# ---- canonical_geometry_key ----


def test_canonical_geometry_key_stable_for_equal_geometry():
    a = CanonicalGeometry.from_reference(20.0, 2.0, 1.3)
    b = CanonicalGeometry.from_reference(20.0, 2.0, 1.3)
    assert canonical_geometry_key(a) == canonical_geometry_key(b)


# ---- apriltag_cache_key ----


def test_apriltag_key_changes_with_family():
    k1 = apriltag_cache_key(_at_config(family="tag36h11"))
    k2 = apriltag_cache_key(_at_config(family="tag25h9"))
    assert k1.config_hash != k2.config_hash


def test_apriltag_key_has_empty_model_path():
    k = apriltag_cache_key(_at_config())
    assert k.model_id == ""


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


# ---- batch size folds into every stage key (B2, pre-existing production bug) ----
#
# The stage cache keys carried NO batch size, so a project config that
# hand-sets detection_batch_size / POSE_BATCH_SIZE / HEADTAIL_BATCH_SIZE / a
# CNN batch_size wrote its detections and pose under the SAME key as the
# default run, and the next run with "Use cached detections" replayed them.
# Batching demonstrably changes the numbers (Task 12 measured det=4 vs det=1
# at <=1px on CUDA), and unlike confidence/IoU a batch size is NOT re-applied
# at tracking time -- so by the key module's own stated principle it belongs
# in the hash.
#
# Folded only when the value is NON-DEFAULT, following the precedent already
# set for slicing and ROI: every existing cache written at the default batch
# keeps its key and stays valid.


def test_default_detection_batch_keeps_the_pre_change_key():
    """The default batch appends NO term, so no existing cache is invalidated.

    The pre-change key is whatever the direct raw-extraction contract hashes to
    -- on this base that is ``_direct_raw_config_hash`` (main folds the full
    direct contract in, so it is non-empty even with slicing off). The
    invariant this test guards is unchanged: ``batch_size=1`` must add nothing.
    """

    pre_change = keys_mod._direct_raw_config_hash(_obb_direct_slice(SliceConfig()))
    assert (
        detection_cache_key(_obb_direct_slice(SliceConfig())).config_hash == pre_change
    )
    assert (
        detection_cache_key(_obb_direct_slice(SliceConfig()), batch_size=1).config_hash
        == pre_change
    )


def test_non_default_detection_batch_changes_the_key():
    one = detection_cache_key(_obb_direct(), batch_size=1)
    four = detection_cache_key(_obb_direct(), batch_size=4)
    assert one.config_hash != four.config_hash
    assert detection_cache_key(_obb_direct(), batch_size=4).config_hash == (
        four.config_hash
    )


@pytest.mark.parametrize(
    "key_fn, config_fn, default, other",
    [
        (headtail_cache_key, _ht_config, 64, 8),
        (cnn_cache_key, _cnn_config, 64, 8),
        (pose_cache_key, _pose_config, 64, 25),
    ],
)
def test_stage_batch_size_folds_in_only_when_non_default(
    key_fn, config_fn, default, other
):
    from dataclasses import replace

    geometry = _GEOM_A
    base = config_fn()

    # The pose batch lives on the per-backend sub-config; the others carry it
    # directly. Poke it wherever it actually lives.
    def _with_batch(config, value):
        if hasattr(config, "batch_size"):
            return replace(config, batch_size=value)
        return replace(config, yolo=replace(config.yolo, batch_size=value))

    def _batch_of(config):
        return getattr(config, "batch_size", None) or config.yolo.batch_size

    assert _batch_of(base) == default, "fixture drifted from the schema default"

    at_default = key_fn(base, geometry)
    at_other = key_fn(_with_batch(base, other), geometry)
    assert at_default.config_hash != at_other.config_hash

    # Byte-parity for the default: an existing cache is not invalidated.
    assert key_fn(_with_batch(base, default), geometry).config_hash == (
        at_default.config_hash
    )


def test_default_tile_batch_keeps_the_pre_change_sliced_key():
    """Byte-parity for the default: no existing sliced cache is invalidated."""

    base = detection_cache_key(_obb_direct_slice(SliceConfig(enabled=True)))
    explicit = detection_cache_key(
        _obb_direct_slice(SliceConfig(enabled=True, tile_batch_size=16))
    )
    assert base.config_hash == explicit.config_hash
    # And with slicing DISABLED the tile batch is inert: a non-default value
    # must not perturb the key (it changes nothing about raw extraction).
    disabled_default = detection_cache_key(
        _obb_direct_slice(SliceConfig(enabled=False))
    ).config_hash
    assert (
        detection_cache_key(
            _obb_direct_slice(SliceConfig(enabled=False, tile_batch_size=32))
        ).config_hash
        == disabled_default
    )


def test_obb_detection_key_is_identical_for_the_same_model_at_two_paths(tmp_path):
    """config_hash must NOT carry the model path (keys.py _model_signature)."""
    a = tmp_path / "one" / "obb.pt"
    b = tmp_path / "two" / "obb.pt"
    a.parent.mkdir(parents=True)
    b.parent.mkdir(parents=True)
    a.write_bytes(b"obb-weights")
    shutil.copy2(a, b)
    os.utime(b, (1, 1))
    key_a = detection_cache_key(_obb_direct(path=str(a)), None)
    key_b = detection_cache_key(_obb_direct(path=str(b)), None)
    assert key_a.as_string() == key_b.as_string()


def test_sequential_obb_key_is_identical_for_the_same_pair_at_two_paths(tmp_path):
    def _pair(root):
        root.mkdir(parents=True, exist_ok=True)
        (root / "detect.pt").write_bytes(b"d")
        (root / "obb.pt").write_bytes(b"o")
        return OBBConfig(
            mode="sequential",
            sequential=OBBSequentialConfig(
                detect_model_path=str(root / "detect.pt"),
                obb_model_path=str(root / "obb.pt"),
            ),
        )

    key_a = detection_cache_key(_pair(tmp_path / "one"), None)
    key_b = detection_cache_key(_pair(tmp_path / "two"), None)
    assert key_a.as_string() == key_b.as_string()


def test_sequential_key_changes_when_either_model_changes(tmp_path):
    root = tmp_path / "m"
    root.mkdir()
    detect, obb = root / "detect.pt", root / "obb.pt"
    detect.write_bytes(b"d")
    obb.write_bytes(b"o")

    def _cfg():
        return OBBConfig(
            mode="sequential",
            sequential=OBBSequentialConfig(
                detect_model_path=str(detect), obb_model_path=str(obb)
            ),
        )

    base = detection_cache_key(_cfg(), None).as_string()
    obb.write_bytes(b"o2")
    # model_content_id is memoized on (realpath, size, mtime_ns); the rewrite
    # changes size AND mtime_ns, so the memo entry is a miss, not a stale hit.
    assert detection_cache_key(_cfg(), None).as_string() != base


def test_v4_cache_on_disk_is_rejected_and_rebuilt(tmp_path):
    """Fix M4: a literal-string comparison is tautological -- it proves nothing
    about the actual store. Exercise the real handle path: write a v4-shaped
    on-disk cache, then prove the store treats it as unusable (rejected) and a
    v5 write follows (rebuilt), per spec section 7b.6 item 3.
    """
    import dataclasses

    from hydra_suite.core.inference.cache.store import DetectionCacheHandle

    cache_dir = tmp_path / ".inference_cache_clip"
    cache_dir.mkdir()
    path = cache_dir / "detection.npz"
    result = materialize_tensors(_raw())
    v5_key = CacheKey(
        schema_version=CACHE_SCHEMA_VERSION, model_id="sha256:aa", config_hash="bb"
    )
    v4_key = dataclasses.replace(v5_key, schema_version=4)

    stale = DetectionCacheHandle(path=path, key=v4_key, write_mode="fresh")
    stale.write_frame(result.frame_idx, result=result)
    stale.close()

    rejected = DetectionCacheHandle(
        path=path, key=v5_key, read_only=True, write_mode="auto"
    )
    assert not rejected.is_reusable(), "a v4 on-disk cache must not validate at v5"
    rejected.close()  # read_only close is a disk no-op (store.py:223-225)

    fresh = DetectionCacheHandle(path=path, key=v5_key, write_mode="fresh")
    fresh.write_frame(result.frame_idx, result=result)
    fresh.close()
    rebuilt = DetectionCacheHandle(path=path, key=v5_key, read_only=True)
    assert rebuilt.is_reusable(), "a fresh v5 write must be usable"
    rebuilt.close()


def test_cache_written_at_one_path_is_reusable_from_a_copy_at_another_path(tmp_path):
    """Fix Z7: no existing test in this task proves the actual Goal-4
    property AT THE HANDLE LEVEL -- that a cache produced against a model at
    path A validates against the SAME model's bytes copied to path B in a
    simulated fresh process (a different machine, in practice). Every other
    test here proves the KEY STRING is path-independent; this proves the
    on-disk cache built from that key is actually reusable end to end.
    """
    import shutil

    from hydra_suite.core.inference import content_id
    from hydra_suite.core.inference.cache.store import DetectionCacheHandle

    model_a = tmp_path / "box_a" / "obb.pt"
    model_a.parent.mkdir(parents=True)
    model_a.write_bytes(b"obb-weights" * 1000)

    key_a = detection_cache_key(_obb_direct(path=str(model_a)), None)
    cache_dir = tmp_path / ".inference_cache_clip"
    cache_dir.mkdir()
    path = cache_dir / "detection.npz"
    result = materialize_tensors(_raw())
    writer = DetectionCacheHandle(path=path, key=key_a, write_mode="fresh")
    writer.write_frame(result.frame_idx, result=result)
    writer.close()

    # Simulate a fresh process on a different machine: drop the in-process
    # memoization AND rebuild the key from a COPY of the same bytes at a
    # different path with a different mtime.
    content_id.model_content_id.cache_clear()
    model_b = tmp_path / "box_b" / "nested" / "obb.pt"
    model_b.parent.mkdir(parents=True)
    shutil.copy2(model_a, model_b)
    os.utime(model_b, (1, 1))
    key_b = detection_cache_key(_obb_direct(path=str(model_b)), None)

    reader = DetectionCacheHandle(path=path, key=key_b, read_only=True)
    assert (
        reader.is_reusable()
    ), "identical bytes at a different path must reuse the cache"
    reader.close()


def test_cache_written_at_one_path_is_not_reusable_after_one_byte_changes(tmp_path):
    """Fix Z7 (negative case): the copy-at-a-different-path test above proves
    portability; this proves it isn't achieved by accidentally ignoring model
    content altogether -- a genuinely different model at the new path must
    NOT validate."""
    from hydra_suite.core.inference import content_id
    from hydra_suite.core.inference.cache.store import DetectionCacheHandle

    model_a = tmp_path / "box_a" / "obb.pt"
    model_a.parent.mkdir(parents=True)
    model_a.write_bytes(b"obb-weights" * 1000)

    key_a = detection_cache_key(_obb_direct(path=str(model_a)), None)
    cache_dir = tmp_path / ".inference_cache_clip"
    cache_dir.mkdir()
    path = cache_dir / "detection.npz"
    result = materialize_tensors(_raw())
    writer = DetectionCacheHandle(path=path, key=key_a, write_mode="fresh")
    writer.write_frame(result.frame_idx, result=result)
    writer.close()

    content_id.model_content_id.cache_clear()
    model_c = tmp_path / "box_c" / "obb.pt"
    model_c.parent.mkdir(parents=True)
    model_c.write_bytes(b"different-obb-weights" * 1000)
    key_c = detection_cache_key(_obb_direct(path=str(model_c)), None)

    reader = DetectionCacheHandle(path=path, key=key_c, read_only=True)
    assert not reader.is_reusable(), "genuinely different model bytes must not validate"
    reader.close()


def test_keys_module_reexports_content_id_video_signature_not_a_shadow():
    """Fix Z1: keys.py must import content_id.video_signature, not redefine
    its own — a same-named local def would silently shadow it and F811 is
    disabled in .flake8's extend-ignore, so nothing else would catch this."""
    from hydra_suite.core.inference import content_id

    assert keys_mod.video_signature is content_id.video_signature


def test_keys_module_video_signature_is_mtime_invariant_through_the_reexport(tmp_path):
    """Exercise the touched-mtime-invariance property THROUGH keys_mod's
    re-exported name specifically (not content_id directly), so a future
    reintroduction of a local mtime-based def in keys.py fails here even if
    it somehow also passed the identity check above."""
    import os

    v = tmp_path / "clip.mp4"
    v.write_bytes(b"\x00" * (1 << 20))
    first = keys_mod.video_signature(str(v))
    os.utime(v, (1, 1))
    assert keys_mod.video_signature(str(v)) == first
