from __future__ import annotations

import hashlib
from dataclasses import replace

import numpy as np

from hydra_suite.core.canonicalization.geometry import CanonicalGeometry

from ..config import (
    TRACKER_RAW_OBB_CONFIDENCE_FLOOR,
    AprilTagConfig,
    BgSubConfig,
    CNNConfig,
    HeadTailConfig,
    OBBConfig,
    PoseConfig,
    SliceConfig,
)

# Fix Z1: re-export (not redefine) content_id.video_signature. keys.py used to
# define its own (size, mtime_ns)-based `video_signature`; a same-named local
# `def` here would silently SHADOW this import rather than error, because
# F811 (redefinition) is disabled in .flake8's extend-ignore. Every caller
# that imports `video_signature` from `keys`/`runner` (runner.py, worker.py,
# optimizer.py, optimizer_workers.py, production_replay.py,
# autotune/session.py) depends on this being the content-based one.
from ..content_id import model_content_id
from ..content_id import video_signature as video_signature  # noqa: F401
from .base import CACHE_SCHEMA_VERSION, CacheKey


def canonical_geometry_key(geometry: CanonicalGeometry) -> str:
    """Content hash of a :class:`CanonicalGeometry` for folding into cache keys.

    Any change to the project-wide canonical geometry (canvas size, margin, or
    aspect ratio) changes what pixels every crop-consuming stage (head-tail,
    CNN, pose) actually sees, so it must invalidate their caches.
    """
    return _sha(
        f"{geometry.canvas_w}x{geometry.canvas_h}|{geometry.margin}|{geometry.aspect_ratio}"
    )


def with_video_signature(key: CacheKey, sig: str) -> CacheKey:
    """Return a copy of ``key`` whose config_hash is bound to a video signature.

    A no-op when ``sig`` is empty, so callers without a video file are unchanged.
    """
    if not sig:
        return key
    return replace(key, config_hash=_sha(f"{key.config_hash}|vid={sig}"))


def _default_field(config_type: type, name: str, fallback: int) -> int:
    """The value ``name`` carries on ``config_type`` when the user set nothing."""

    field = getattr(config_type, "__dataclass_fields__", {}).get(name)
    default = getattr(field, "default", None) if field is not None else None
    return int(default) if isinstance(default, int) else int(fallback)


def _default_batch_size(config: object) -> int:
    """The ``batch_size`` a stage config carries when the user set nothing."""

    return _default_field(type(config), "batch_size", 1)


def _batch_term(batch_size: int, default: int) -> str:
    """A key fragment for a NON-DEFAULT batch size, else the empty string.

    Batching changes the numbers a stage produces (a tuned det=4 was measured
    at <=1px from det=1 on CUDA), and unlike confidence/IoU a batch size is
    NOT re-applied at replay time -- so by this module's own stated principle
    it must invalidate the cache. Folding it in only when it is non-default
    follows the precedent set for slicing and ROI: every cache ever written at
    the default batch keeps its key and stays valid, so no user's existing
    cache is thrown away by this fix.
    """

    return "" if int(batch_size) == int(default) else f"|batch={int(batch_size)}"


def detection_cache_key(
    config: OBBConfig,
    roi_mask: np.ndarray | None = None,
    batch_size: int = 1,
) -> CacheKey:
    """Cache key for OBB detections.

    ``roi_mask`` is folded into the key ONLY when sliced inference is enabled
    (direct slicing or sequential stage-1 slicing) AND a mask is actually in use. ROI tile
    gating drops slice tiles that contain no live ROI pixel, which changes which
    *raw* detections land in the cache (final tracked results are unchanged --
    dropped tiles can only yield detections outside the ROI, which filtering
    removes anyway). Under any other condition the ROI term is omitted.
    """
    if config.mode == "direct":
        assert config.direct is not None
        direct = config.direct
        path = direct.model_path
        slice_cfg = direct.slice
        slice_hash = _direct_raw_config_hash(config)
        if slice_cfg is not None and slice_cfg.enabled and roi_mask is not None:
            # Content-hash the mask (same approach as bgsub's ROI_MASK folding,
            # via _param_repr: sha256 over a contiguous tobytes() with shape+
            # dtype). Two different masks => different keys; identical masks =>
            # identical keys.
            slice_hash = _sha(f"{slice_hash}|roi={_param_repr(roi_mask)}")
        model_id = model_content_id(path)
    else:
        assert config.sequential is not None
        slice_hash = _sequential_config_hash(config)
        stage1_slice = config.sequential.stage1_slice
        if stage1_slice.enabled and roi_mask is not None:
            slice_hash = _sha(f"{slice_hash}|roi={_param_repr(roi_mask)}")
        model_id = (
            f"{model_content_id(config.sequential.detect_model_path)}"
            f"|{model_content_id(config.sequential.obb_model_path)}"
        )
    return CacheKey(
        schema_version=CACHE_SCHEMA_VERSION,
        model_id=model_id,
        # User-facing confidence_threshold/iou stay excluded: replay reapplies
        # those filters. Every setting that changes raw extraction is folded in
        # below, including sequential's *second* model signature.
        # Slicing changes which raw detections exist, so it IS folded in (but
        # only when enabled, so existing non-sliced caches stay valid).
        # detection_batch_size is folded in the same way -- only when it is
        # not 1, so a default-batch cache keeps config_hash == "".
        config_hash=slice_hash + _batch_term(batch_size, 1),
    )


def _direct_raw_config_hash(config: OBBConfig) -> str:
    """Hash the complete direct-mode raw-extraction contract.

    The cache stores results before the replay-time confidence/IoU filters, but
    it is *not* a model-agnostic bag of boxes. Classes, raw cap, direct-task
    conversion, segment geometry, and sliced prediction settings all change
    which raw OBBs are materialized and must invalidate it.
    """

    assert config.direct is not None
    direct = config.direct
    task = str(direct.model_task)
    payload = (
        "direct-raw-v4",
        _model_signature(direct.model_path),
        direct.confidence_floor,
        direct.auto_export,
        task,
        direct.fixed_angle_deg if task == "detect" else None,
        direct.seg_num_angles if task == "segment" else None,
        direct.seg_crop_size if task == "segment" else None,
        direct.seg_pad_ratio if task == "segment" else None,
        direct.seg_mask_threshold if task == "segment" else None,
        tuple(config.target_classes),
        config.max_detections,
        config.raw_detection_cap,
        # OBBResult cache serialization intentionally omits native polygons.
        # Export therefore needs a geometry-producing live extraction rather
        # than a false cache hit from an ordinary tracking replay.
        config.emit_native_geometry,
        _slice_config_hash(direct.slice),
    )
    return _sha("|".join(map(str, payload)))


def _sequential_config_hash(config: OBBConfig) -> str:
    """Hash every sequential setting that changes cached raw detections.

    The outer OBB confidence/IoU gates remain excluded because replay reapplies
    them. Stage-1 confidence and crop/extraction geometry are not final filters:
    changing any of them changes which raw stage-2 OBBs can exist and therefore
    must invalidate the cache.
    """

    assert config.sequential is not None
    seq = config.sequential
    payload = (
        "sequential-raw-v4",
        _model_signature(seq.detect_model_path),
        _model_signature(seq.obb_model_path),
        seq.auto_export,
        TRACKER_RAW_OBB_CONFIDENCE_FLOOR,
        seq.detect_confidence_threshold,
        seq.obb_confidence_threshold,
        seq.detect_image_size,
        seq.crop_pad_ratio,
        seq.min_crop_size_px,
        seq.enforce_square_crop,
        seq.stage2_image_size,
        seq.stage2_task,
        seq.seg_num_angles,
        seq.seg_crop_size,
        seq.seg_pad_ratio,
        seq.seg_mask_threshold,
        _slice_config_hash(seq.stage1_slice),
        tuple(config.target_classes),
        config.max_detections,
        config.raw_detection_cap,
        # See the corresponding direct-mode raw contract above.
        config.emit_native_geometry,
    )
    return _sha("|".join(map(str, payload)))


def _slice_config_hash(slice_cfg: SliceConfig | None) -> str:
    """Empty string when disabled (baseline-identical key); param hash when on.

    Every field that can change which raw detections come out of the sliced
    path must be included here -- an omitted field means a user changes it,
    the cache is not invalidated, and they silently get stale detections.
    ``reference_body_px`` affects tile sizing in ``auto_object`` geometry mode
    and ``merge_backend`` selects cv2 vs gpu merge (tolerance-equivalent, not
    bit-identical), so both are included.
    """
    if slice_cfg is None or not slice_cfg.enabled:
        return ""
    payload = "|".join(
        str(x)
        for x in (
            "slice",
            slice_cfg.geometry_mode,
            slice_cfg.slice_height,
            slice_cfg.slice_width,
            slice_cfg.overlap_height_ratio,
            slice_cfg.overlap_width_ratio,
            slice_cfg.object_tile_fraction,
            slice_cfg.reference_body_px,
            slice_cfg.merge_policy,
            slice_cfg.merge_metric,
            slice_cfg.merge_threshold,
            slice_cfg.merge_backend,
            slice_cfg.perform_standard_pred,
        )
    )
    # tile_batch_size is a TUNED coordinate (InferenceTuningSettings writes
    # obb.direct.slice.tile_batch_size), and it changes the raw detections the
    # same way detection_batch_size does -- so a sliced cache written at the
    # default 16 could be replayed under a tuned 32. Folded in only when it is
    # non-default, exactly as the other batch sizes are, so every existing
    # sliced cache keeps its key.
    payload += _batch_term(
        slice_cfg.tile_batch_size,
        _default_field(type(slice_cfg), "tile_batch_size", 16),
    )
    return _sha(payload)


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
    "ROI_MASK",
)


def _param_repr(value: object) -> str:
    """Stringify a param for the cache-key payload.

    ndarrays (e.g. ROI_MASK) must hash by CONTENT: str(ndarray) truncates with
    '...' for large arrays, so two masks differing only in the middle would
    stringify identically and collide -- a silent stale-cache hit. Contiguity
    is forced before `.tobytes()` because a non-contiguous view (e.g. from
    slicing or a transpose) would otherwise hash its strided memory layout
    rather than its logical content, so two logically-identical masks could
    hash differently.
    """
    if isinstance(value, np.ndarray):
        contiguous = np.ascontiguousarray(value)
        digest = hashlib.sha256(contiguous.tobytes()).hexdigest()
        return f"ndarray:{contiguous.shape}:{contiguous.dtype}:{digest}"
    return str(value)


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
    payload = "|".join(
        [
            *(f"{k}={_param_repr(params.get(k))}" for k in _BGSUB_KEY_PARAMS),
            # As with OBB, the compact raw cache does not serialize polygons.
            # This explicit config field is often set by an export caller after
            # constructing BgSubConfig, so it cannot rely on params alone.
            f"emit_native_geometry={bool(config.emit_native_geometry)}",
        ]
    )
    return CacheKey(
        schema_version=CACHE_SCHEMA_VERSION,
        model_id="background_subtraction",
        config_hash=_sha(payload),
    )


def headtail_cache_key(config: HeadTailConfig, geometry: CanonicalGeometry) -> CacheKey:
    config_hash = canonical_geometry_key(geometry) + _batch_term(
        config.batch_size, _default_batch_size(config)
    )
    return CacheKey(
        schema_version=CACHE_SCHEMA_VERSION,
        model_id=model_content_id(config.model_path),
        config_hash=config_hash,
    )


def cnn_cache_key(config: CNNConfig, geometry: CanonicalGeometry) -> CacheKey:
    return CacheKey(
        schema_version=CACHE_SCHEMA_VERSION,
        model_id=model_content_id(config.model_path),
        # calibration_temperature, scoring_mode excluded; canonical geometry
        # IS included -- it changes what pixels the classifier actually sees.
        config_hash=canonical_geometry_key(geometry)
        + _batch_term(config.batch_size, _default_batch_size(config)),
    )


def pose_cache_key(config: PoseConfig, geometry: CanonicalGeometry) -> CacheKey:
    if config.backend == "yolo":
        assert config.yolo is not None
        backend_config = config.yolo
    elif config.backend == "vitpose":
        assert config.vitpose is not None
        backend_config = config.vitpose
    else:
        assert config.sleap is not None
        backend_config = config.sleap
    # The pose batch size lives on the per-backend sub-config, and each
    # backend has its own default (yolo 64, sleap/vitpose 4).
    path = backend_config.model_path
    # background_color was dropped from PoseConfig: it was always (0, 0, 0)
    # (never populated by from_parameters), so removing it does not change
    # any hash produced by any existing config in practice.
    config_hash = _sha(
        f"{config.suppress_foreign_regions}|{canonical_geometry_key(geometry)}"
    ) + _batch_term(backend_config.batch_size, _default_batch_size(backend_config))
    return CacheKey(
        schema_version=CACHE_SCHEMA_VERSION,
        model_id=model_content_id(path),
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
        model_id="",
        config_hash=config_hash,
    )


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def _model_signature(path: str) -> str:
    """Content signature of one model, for folding into a raw config hash.

    A sequential detector consumes two independently changing artifacts, so
    both are folded here as well as into ``model_id``. This MUST be
    content-based: it used to be f"{path}|mtime=…", which put an absolute
    path inside ``config_hash`` and made every OBB cache machine-local even
    after ``model_id`` itself became content-based.
    """
    return model_content_id(path)
