from __future__ import annotations

import hashlib
import os

from ..config import AprilTagConfig, CNNConfig, HeadTailConfig, OBBConfig, PoseConfig
from .base import CACHE_SCHEMA_VERSION, CacheKey


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
