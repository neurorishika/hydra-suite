from __future__ import annotations

import hashlib
import os
from dataclasses import replace

from ..config import (
    AprilTagConfig,
    BgSubConfig,
    CNNConfig,
    HeadTailConfig,
    OBBConfig,
    PoseConfig,
)
from .base import CACHE_SCHEMA_VERSION, CacheKey


def video_signature(path: str | None) -> str:
    """Cheap content fingerprint of a video file (size + mtime).

    Folding this into the cache keys makes a cache reusable only for the exact
    video file it was computed from. Without it, a video replaced under the same
    name (e.g. a clip regenerated with a different frame count) would pass the
    config-only key check and serve stale, truncated detections. Returns "" when
    no path is given so non-video contexts and tests keep the old behavior.
    """
    if not path:
        return ""
    try:
        st = os.stat(path)  # follows symlinks → fingerprints the real file
        return f"{st.st_size}:{st.st_mtime_ns}"
    except OSError:
        return ""


def with_video_signature(key: CacheKey, sig: str) -> CacheKey:
    """Return a copy of ``key`` whose config_hash is bound to a video signature.

    A no-op when ``sig`` is empty, so callers without a video file are unchanged.
    """
    if not sig:
        return key
    return replace(key, config_hash=_sha(f"{key.config_hash}|vid={sig}"))


def detection_cache_key(config: OBBConfig) -> CacheKey:
    if config.mode == "direct":
        assert config.direct is not None
        path = config.direct.model_path
    else:
        assert config.sequential is not None
        path = (
            f"{config.sequential.detect_model_path}|"
            f"{config.sequential.obb_model_path}"
        )
    return CacheKey(
        schema_version=CACHE_SCHEMA_VERSION,
        model_path=path,
        model_mtime=_mtime(path.split("|")[0]),
        config_hash="",  # confidence_threshold/iou excluded — re-applied at tracking time
    )


# Params that affect background-subtraction detection output. The bg-sub cache is
# reusable only when these (and the video signature) match — mirroring how the OBB
# detection cache keys on model + config.
_BGSUB_KEY_PARAMS = (
    "THRESHOLD_VALUE",
    "DARK_ON_LIGHT_BACKGROUND",
    "ENABLE_CONSERVATIVE_SPLIT",
    "ENABLE_ADAPTIVE_BACKGROUND",
    "BACKGROUND_LEARNING_RATE",
    "BACKGROUND_PRIME_FRAMES",
    "ENABLE_SIZE_FILTERING",
    "MIN_OBJECT_SIZE",
    "MAX_OBJECT_SIZE",
    "ENABLE_ASPECT_RATIO_FILTERING",
    "BRIGHTNESS",
    "CONTRAST",
    "GAMMA",
    "ENABLE_LIGHTING_STABILIZATION",
    "LIGHTING_SMOOTH_FACTOR",
    "LIGHTING_MEDIAN_WINDOW",
    "MORPH_KERNEL_SIZE",
    "DILATION_KERNEL_SIZE",
    "ENABLE_ADDITIONAL_DILATION",
    "DILATION_ITERATIONS",
    "CONSERVATIVE_KERNEL_SIZE",
    "CONSERVATIVE_ERODE_ITER",
    "REFERENCE_BODY_SIZE",
    "MIN_CONTOUR_AREA",
    "MAX_TARGETS",
    "MAX_CONTOUR_MULTIPLIER",
    "START_FRAME",
    "END_FRAME",
    "RESIZE_FACTOR",
    "BACKGROUND_CONVERGENCE_EPSILON",
    "BACKGROUND_CONVERGENCE_FRAMES",
    "BACKGROUND_CONVERGENCE_PIXEL_DELTA",
)


def bgsub_detection_cache_key(config: BgSubConfig) -> CacheKey:
    """Cache key for background-subtraction detections.

    There is no model file, so model_path is a sentinel and the
    detection-affecting parameters are hashed into config_hash. Callers should
    fold in the video signature via ``with_video_signature`` so the cache is
    bound to the source file.

    Soundness depends on deterministic priming (core/background/model.py samples
    evenly-spaced frames, not unseeded random ones) -- without that, identical
    params would legitimately produce different detections and this key would
    be a lie.
    """
    params = config.params
    payload = "|".join(f"{k}={params.get(k)}" for k in _BGSUB_KEY_PARAMS)
    return CacheKey(
        schema_version=CACHE_SCHEMA_VERSION,
        model_path="background_subtraction",
        model_mtime=0.0,
        config_hash=_sha(payload),
    )


def headtail_cache_key(config: HeadTailConfig) -> CacheKey:
    config_hash = _sha(f"{config.canonical_aspect_ratio}|{config.canonical_margin}")
    return CacheKey(
        schema_version=CACHE_SCHEMA_VERSION,
        model_path=config.model_path,
        model_mtime=_mtime(config.model_path),
        config_hash=config_hash,
    )


def cnn_cache_key(config: CNNConfig) -> CacheKey:
    return CacheKey(
        schema_version=CACHE_SCHEMA_VERSION,
        model_path=config.model_path,
        model_mtime=_mtime(config.model_path),
        config_hash="",  # calibration_temperature, scoring_mode excluded
    )


def pose_cache_key(config: PoseConfig) -> CacheKey:
    if config.backend == "yolo":
        assert config.yolo is not None
        path = config.yolo.model_path
    else:
        assert config.sleap is not None
        path = config.sleap.model_path
    config_hash = _sha(
        f"{config.crop_padding}|{config.suppress_foreign_regions}"
        f"|{config.background_color}"
    )
    return CacheKey(
        schema_version=CACHE_SCHEMA_VERSION,
        model_path=path,
        model_mtime=_mtime(path),
        config_hash=config_hash,
    )


def apriltag_cache_key(config: AprilTagConfig) -> CacheKey:
    config_hash = _sha(
        f"{config.tag_family}|{config.decimate}|{config.blur}"
        f"|{config.refine_edges}|{config.unsharp_kernel}"
        f"|{config.unsharp_sigma}|{config.unsharp_amount}"
        f"|{config.contrast_factor}"
    )
    return CacheKey(
        schema_version=CACHE_SCHEMA_VERSION,
        model_path="",
        model_mtime=0.0,
        config_hash=config_hash,
    )


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def _mtime(path: str) -> float:
    try:
        return os.path.getmtime(path)
    except OSError:
        return 0.0
