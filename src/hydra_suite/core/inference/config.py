from __future__ import annotations

import json
import logging
import math
import os
from dataclasses import asdict, dataclass, field
from typing import Any, Literal

from hydra_suite.core.canonicalization.geometry import (
    CanonicalGeometry,
    canonical_geometry_from_params,
)
from hydra_suite.core.individual.classification.errors import CalibrationRequiredError
from hydra_suite.runtime.resolver import RuntimeTier

logger = logging.getLogger(__name__)


class InferenceConfigError(ValueError):
    pass


def migrate_runtime_to_tier(runtimes: set[str]) -> RuntimeTier:
    """Map legacy per-stage runtime strings to a single pipeline tier.

    cpu -> "cpu"; cuda/mps -> "gpu"; onnx_*/tensorrt -> "gpu_fast".
    Mixed sets take the highest tier present (gpu_fast > gpu > cpu).
    Empty set defaults to "gpu" (the field default).
    """
    if not runtimes:
        return "gpu"
    # onnx_* / *_cuda / coreml entries cover both legacy-config migration and the
    # runtime-flavor strings the GUIs emit for gpu_fast (tensorrt_cuda, coreml,
    # onnx_mps).
    fast = {
        "onnx_cpu",
        "onnx_cuda",
        "onnx_coreml",
        "onnx_mps",
        "tensorrt",
        "tensorrt_cuda",
        "coreml",
    }
    gpu = {"cuda", "mps"}
    if runtimes & fast:
        return "gpu_fast"
    if runtimes & gpu:
        return "gpu"
    return "cpu"


@dataclass
class SliceConfig:
    """SAHI-style sliced inference for direct-mode OBB detection.

    All fields are inert when ``enabled`` is False — ``run_obb`` dispatches to
    the sliced path only when it is True, so the entire feature is dead code
    otherwise and output is byte-identical to the non-sliced pipeline.
    """

    enabled: bool = False
    geometry_mode: Literal["auto_model", "auto_object", "custom"] = "auto_model"
    # custom mode: explicit tile size in original-frame pixels.
    slice_height: int = 0
    slice_width: int = 0
    overlap_height_ratio: float = 0.2
    overlap_width_ratio: float = 0.2
    # auto_object mode: tile sized so a reference object spans this linear
    # fraction of the tile.
    object_tile_fraction: float = 0.15
    # Reference object size in ORIGINAL-FRAME pixels, sourced from
    # REFERENCE_BODY_SIZE * RESIZE_FACTOR. Only read in auto_object mode; 0
    # means "unknown", which falls back to auto_model sizing.
    reference_body_px: float = 0.0
    # merge across tile boundaries.
    merge_policy: Literal["nms", "nmm", "greedy_nmm"] = "greedy_nmm"
    merge_metric: Literal["iou", "ios"] = "ios"
    merge_threshold: float = 0.5
    # cv2 = default correctness oracle (all paths); gpu is honored only on the
    # native-CUDA (``gpu`` tier / torch) path -- cpu/mps/gpu_fast always use
    # cv2 (see stages/slicing.py's host-path downgrade + its logger.info).
    merge_backend: Literal["cv2", "gpu"] = "cv2"
    # extra full-frame pass in addition to tiles (catches > tile-size objects).
    perform_standard_pred: bool = False

    _GEOMETRY_MODES = ("auto_model", "auto_object", "custom")
    _MERGE_POLICIES = ("nms", "nmm", "greedy_nmm")
    _MERGE_METRICS = ("iou", "ios")
    _MERGE_BACKENDS = ("cv2", "gpu")

    def __post_init__(self) -> None:
        self._validate_choice("geometry_mode", self.geometry_mode, self._GEOMETRY_MODES)
        self._validate_choice("merge_policy", self.merge_policy, self._MERGE_POLICIES)
        self._validate_choice("merge_metric", self.merge_metric, self._MERGE_METRICS)
        self._validate_choice("merge_backend", self.merge_backend, self._MERGE_BACKENDS)

    @staticmethod
    def _validate_choice(field_name: str, value: str, allowed: tuple[str, ...]) -> None:
        if value not in allowed:
            raise InferenceConfigError(
                f"SliceConfig.{field_name} must be one of {allowed!r}, got {value!r}"
            )


@dataclass
class OBBDirectConfig:
    model_path: str
    confidence_floor: float = 1e-3
    confidence_threshold: float = 0.25
    # Auto-export the .engine (TensorRT) / .mlpackage (CoreML) artifact from a
    # .pt source on first load for the gpu_fast runtimes. When False and no
    # artifact exists, loading raises a
    # clear error instead of silently running PyTorch (parity finding H4).
    auto_export: bool = True
    # "obb": model_path is a native-OBB YOLO checkpoint (existing behaviour).
    # "detect": model_path is a plain axis-aligned YOLO detect checkpoint;
    # every detection is assigned the fixed angle below instead of a
    # model-predicted angle.
    # "segment": model_path is a YOLO instance-segmentation checkpoint; the
    # angle is derived per-detection from a GPU batched rotated-rectangle
    # search over the predicted mask (see utils/obb_from_mask.py).
    model_task: Literal["obb", "detect", "segment"] = "obb"
    # Only read when model_task == "detect". Degrees; converted to radians
    # before being folded through the same normalize/corners pipeline as
    # native-OBB angles.
    fixed_angle_deg: float = 0.0
    # The following four fields are only read when model_task == "segment";
    # they are forwarded as keyword args to
    # utils/obb_from_mask.py:rotated_rect_from_masks. Defaults match that
    # function's own kernel defaults.
    # Number of coarse candidate angles searched over [0, pi) before local
    # refinement. Linear cost: doubling this roughly doubles per-detection
    # kernel time.
    seg_num_angles: int = 24
    # Square resolution (crop_size x crop_size) the mask is resampled to
    # before the rotated-rect search. Quadratic cost: doubling this
    # roughly quadruples per-detection kernel time.
    seg_crop_size: int = 64
    # Fractional padding (of the axis-aligned box's own size) added around
    # the crop region before resampling, so a tightly-fit mask isn't clipped
    # at the crop border.
    seg_pad_ratio: float = 0.15
    # Foreground cutoff applied to the resampled soft mask before the
    # rotated-rect search treats a pixel as "inside" the object.
    seg_mask_threshold: float = 0.5
    slice: SliceConfig = field(default_factory=SliceConfig)


@dataclass
class OBBSequentialConfig:
    detect_model_path: str
    obb_model_path: str
    # See OBBDirectConfig.auto_export.
    auto_export: bool = True
    detect_confidence_threshold: float = 1e-3
    obb_confidence_threshold: float = 1e-3
    detect_image_size: int = 0
    crop_pad_ratio: float = 0.15
    min_crop_size_px: float = 64.0
    enforce_square_crop: bool = True
    stage2_image_size: int = 160
    stage2_batch_size: int | None = None
    stage2_task: Literal["obb", "segment"] = "obb"
    # Read only when stage2_task == "segment"; forwarded to _extract_obb_from_masks.
    # Defaults match OBBDirectConfig's segment defaults.
    seg_num_angles: int = 24
    seg_crop_size: int = 64
    seg_pad_ratio: float = 0.15
    seg_mask_threshold: float = 0.5
    # SAHI-style tiling for the STAGE-1 (detect) pass only (Phase C, Task 11).
    # Off by default -- `select_region_source` routes to the unchanged
    # `Stage1Proposals` source unless `stage1_slice.enabled` is True, so
    # existing sequential configs are byte-identical to before this field
    # existed. See `regions.SlicedStage1Proposals`.
    stage1_slice: SliceConfig = field(default_factory=SliceConfig)


@dataclass
class OBBConfig:
    mode: Literal["direct", "sequential"] = "direct"
    direct: OBBDirectConfig | None = None
    sequential: OBBSequentialConfig | None = None
    target_classes: list[int] = field(default_factory=list)
    max_detections: int = 20
    # Cap on RAW detections per frame, applied at OBB extraction (sorted by
    # confidence descending, top-k) BEFORE size/aspect/IoU filtering. Mirrors
    # legacy ``_obb_geometry._raw_detection_cap`` (= 2 * MAX_TARGETS). 0 disables.
    raw_detection_cap: int = 0
    min_object_size: float = 0.0
    max_object_size: float = float("inf")
    # Aspect-ratio (major/minor) gate, applied during filtering. Mirrors legacy
    # ``_obb_geometry`` aspect filtering (ref_ar * min/max multiplier). Defaults
    # (0, inf) disable the gate.
    min_aspect_ratio: float = 0.0
    max_aspect_ratio: float = float("inf")
    confidence_threshold: float = 0.25
    iou_threshold: float = 0.7  # legacy YOLO_IOU_THRESHOLD default
    # Export-only: when True, extractors additionally populate
    # OBBResult.polygons with native per-detection contours (frame pixel
    # space). Default False keeps the hot tracking path byte-identical (no
    # polygons computed, no OBBResult field change).
    emit_native_geometry: bool = False

    @staticmethod
    def from_dict(d: dict[str, Any]) -> "OBBConfig":
        """Construct OBBConfig from a dict, handling nested slice config."""
        obb_d = d
        if obb_d.get("max_object_size") is None:
            obb_d["max_object_size"] = float("inf")
        if obb_d.get("max_aspect_ratio") is None:
            obb_d["max_aspect_ratio"] = float("inf")

        direct = None
        if obb_d.get("direct"):
            direct_d = dict(obb_d["direct"])
            slice_d = direct_d.pop("slice", None)
            direct = OBBDirectConfig(**direct_d)
            if isinstance(slice_d, dict):
                direct.slice = SliceConfig(**slice_d)

        sequential = None
        if obb_d.get("sequential"):
            sequential_d = dict(obb_d["sequential"])
            stage1_slice_d = sequential_d.pop("stage1_slice", None)
            sequential = OBBSequentialConfig(**sequential_d)
            if isinstance(stage1_slice_d, dict):
                sequential.stage1_slice = SliceConfig(**stage1_slice_d)
        return OBBConfig(
            mode=obb_d["mode"],
            direct=direct,
            sequential=sequential,
            target_classes=obb_d.get("target_classes", []),
            max_detections=obb_d.get("max_detections", 20),
            raw_detection_cap=obb_d.get("raw_detection_cap", 0),
            min_object_size=obb_d.get("min_object_size", 0.0),
            max_object_size=obb_d.get("max_object_size", float("inf")),
            min_aspect_ratio=obb_d.get("min_aspect_ratio", 0.0),
            max_aspect_ratio=obb_d.get("max_aspect_ratio", float("inf")),
            confidence_threshold=obb_d.get("confidence_threshold", 0.25),
            iou_threshold=obb_d.get(
                "iou_threshold", 0.45
            ),  # legacy YOLO_IOU_THRESHOLD default
            emit_native_geometry=obb_d.get("emit_native_geometry", False),
        )


@dataclass
class BgSubConfig:
    """Background-subtraction detection.

    Unlike OBB there is no model file: the 'model' is the primed
    BackgroundModel, derived from the video itself.
    """

    threshold_value: float = 20.0
    dark_on_light_background: bool = True
    enable_adaptive_background: bool = True
    background_learning_rate: float = 0.001
    background_prime_frames: int = 30
    convergence_epsilon: float = 1e-4
    convergence_frames: int = 30
    convergence_pixel_delta: float = 5.0
    enable_conservative_split: bool = False
    morph_kernel_size: int = 5
    dilation_kernel_size: int = 3
    conservative_kernel_size: int = 3
    max_targets: int = 20
    min_contour_area: float = 5.0
    max_contour_multiplier: int = 20
    enable_size_filtering: bool = False
    min_object_size: float = 0.0
    max_object_size: float = float("inf")
    # Export-only: when True, run_bgsub additionally populates
    # OBBResult.polygons with native per-detection contours (frame pixel
    # space). Default False keeps the hot tracking path byte-identical (no
    # contours computed, no OBBResult field change).
    emit_native_geometry: bool = False
    # The raw param dict, retained for BackgroundModel/BackgroundMeasurer,
    # which still read params by legacy UPPER_SNAKE key.
    params: dict = field(default_factory=dict)

    @staticmethod
    def from_params(params: dict) -> "BgSubConfig":
        return BgSubConfig(
            threshold_value=float(params.get("THRESHOLD_VALUE", 20) or 20),
            dark_on_light_background=bool(params.get("DARK_ON_LIGHT_BACKGROUND", True)),
            enable_adaptive_background=bool(
                params.get("ENABLE_ADAPTIVE_BACKGROUND", True)
            ),
            background_learning_rate=float(
                params.get("BACKGROUND_LEARNING_RATE", 0.001) or 0.001
            ),
            background_prime_frames=int(
                params.get("BACKGROUND_PRIME_FRAMES", 30) or 30
            ),
            convergence_epsilon=float(
                params.get("BACKGROUND_CONVERGENCE_EPSILON", 1e-4) or 1e-4
            ),
            convergence_frames=int(
                params.get("BACKGROUND_CONVERGENCE_FRAMES", 30) or 30
            ),
            convergence_pixel_delta=float(
                params.get("BACKGROUND_CONVERGENCE_PIXEL_DELTA", 5.0) or 5.0
            ),
            enable_conservative_split=bool(
                params.get("ENABLE_CONSERVATIVE_SPLIT", False)
            ),
            morph_kernel_size=int(params.get("MORPH_KERNEL_SIZE", 5) or 5),
            dilation_kernel_size=int(params.get("DILATION_KERNEL_SIZE", 3) or 3),
            conservative_kernel_size=int(
                params.get("CONSERVATIVE_KERNEL_SIZE", 3) or 3
            ),
            max_targets=int(params.get("MAX_TARGETS", 20) or 20),
            min_contour_area=float(params.get("MIN_CONTOUR_AREA", 5) or 5),
            max_contour_multiplier=int(params.get("MAX_CONTOUR_MULTIPLIER", 20) or 20),
            enable_size_filtering=bool(params.get("ENABLE_SIZE_FILTERING", False)),
            min_object_size=float(params.get("MIN_OBJECT_SIZE", 0) or 0),
            max_object_size=float(params.get("MAX_OBJECT_SIZE", float("inf"))),
            params=dict(params),
        )


@dataclass
class HeadTailConfig:
    model_path: str
    confidence_threshold: float = 0.5
    candidate_confidence_threshold: float | None = None
    batch_size: int = 64


@dataclass
class CNNConfig:
    label: str
    model_path: str
    confidence_threshold: float = 0.5
    batch_size: int = 64
    scoring_mode: Literal["atomic", "per_head_average"] = "atomic"
    match_bonus: float = 0.1
    mismatch_penalty: float = 0.3
    calibration_temperature: float = 1.0


@dataclass
class PoseYOLOConfig:
    model_path: str
    confidence_threshold: float = 1e-4
    iou_threshold: float = 0.7
    max_detections_per_crop: int = 1
    batch_size: int = 64


@dataclass
class PoseSLEAPConfig:
    model_path: str
    conda_env: str = "sleap"
    batch_size: int = 4
    max_instances: int = 1


@dataclass
class PoseViTPoseConfig:
    model_path: str
    variant: str = "auto"
    num_keypoints: int = 0
    batch_size: int = 4
    # When False and no accelerated (tensorrt/coreml) artifact is available,
    # raise instead of silently falling back to native torch. See
    # OBBDirectConfig.auto_export.
    auto_export: bool = True


@dataclass
class PoseConfig:
    backend: Literal["yolo", "sleap", "vitpose"] = "yolo"
    skeleton_file: str = ""
    yolo: PoseYOLOConfig | None = None
    sleap: PoseSLEAPConfig | None = None
    vitpose: PoseViTPoseConfig | None = None
    crop_padding: float = 0.1
    suppress_foreign_regions: bool = True
    anterior_keypoints: list[str] = field(default_factory=list)
    posterior_keypoints: list[str] = field(default_factory=list)
    ignore_keypoints: list[str] = field(default_factory=list)
    min_keypoint_confidence: float = 0.2
    min_valid_keypoints: int = 1
    overrides_headtail: bool = True


@dataclass
class AprilTagConfig:
    enabled: bool = False
    tag_family: str = "tag36h11"
    threads: int = 4
    max_hamming: int = 1
    decimate: float = 1.0
    blur: float = 0.8
    refine_edges: bool = True
    decode_sharpening: float = 0.25
    unsharp_kernel: tuple[int, int] = (5, 5)
    unsharp_sigma: float = 1.0
    unsharp_amount: float = 1.5
    contrast_factor: float = 1.5
    max_tag_id: int | None = None
    crop_padding: float = 0.1


def _default_canonical_geometry() -> CanonicalGeometry:
    """Fallback canonical geometry (project-wide default body_px/aspect/margin).

    Used whenever a config is built without an explicit ``canonical`` (e.g.
    hand-built ``InferenceConfig``s in tests, or `_dict_to_config` reading an
    older on-disk JSON that predates this field).

    Routes through ``canonical_geometry_from_params`` with an empty params
    dict so the magic defaults (``REFERENCE_BODY_SIZE=20.0``,
    ``reference_aspect_ratio=2.0``, ``canonical_margin=1.3``) live in exactly
    one place: that helper's own defaults.
    """
    return canonical_geometry_from_params({})


@dataclass
class InferenceConfig:
    # Exactly one detection source must be set. OBB is the YOLO path; bgsub is
    # background subtraction. They are alternatives, not composable.
    obb: OBBConfig | None = None
    bgsub: BgSubConfig | None = None
    headtail: HeadTailConfig | None = None
    cnn_phases: list[CNNConfig] = field(default_factory=list)
    pose: PoseConfig | None = None
    apriltag: AprilTagConfig = field(default_factory=AprilTagConfig)
    # Single project-wide canonical crop geometry (Layer 1). Every crop-consuming
    # stage (headtail, cnn, pose) shares this ONE geometry instead of each
    # carrying its own aspect_ratio/margin pair.
    canonical: CanonicalGeometry = field(default_factory=_default_canonical_geometry)
    detection_batch_size: int = 1
    pipeline_depth: int = 2
    runtime_tier: RuntimeTier = "gpu"
    realtime: bool = False
    use_cache: bool = True
    cache_dir: str | None = None

    @staticmethod
    def from_json(path: str) -> "InferenceConfig":
        with open(path) as f:
            data = json.load(f)
        config = _dict_to_config(data)
        return config

    def to_json(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump(_config_to_dict(self), f, indent=2)

    def __post_init__(self) -> None:
        self._validate_pipeline_depth()
        self._validate_detection_source()

    def _validate_pipeline_depth(self) -> None:
        if self.pipeline_depth < 1:
            raise InferenceConfigError(
                f"pipeline_depth must be >= 1, got {self.pipeline_depth}"
            )

    def _validate_detection_source(self) -> None:
        if (self.obb is None) == (self.bgsub is None):
            raise InferenceConfigError(
                "InferenceConfig requires exactly one detection source: set "
                "either `obb` or `bgsub`, not both and not neither."
            )

    @property
    def detection_source(self) -> Literal["obb", "bgsub"]:
        return "obb" if self.obb is not None else "bgsub"


# ── serialization helpers ─────────────────────────────────────────────────────


def _config_to_dict(config: InferenceConfig) -> dict[str, Any]:
    d = asdict(config)
    obb = d.get("obb")
    if obb is not None:
        if obb.get("max_object_size") == float("inf"):
            obb["max_object_size"] = None
        if obb.get("max_aspect_ratio") == float("inf"):
            obb["max_aspect_ratio"] = None
    bgsub = d.get("bgsub")
    if bgsub is not None and bgsub.get("max_object_size") == float("inf"):
        bgsub["max_object_size"] = None
    return d


def _dict_to_config(d: dict[str, Any]) -> InferenceConfig:
    obb_d = d.get("obb")
    obb = OBBConfig.from_dict(obb_d) if obb_d else None

    bgsub_d = d.get("bgsub")
    bgsub = None
    if bgsub_d:
        if bgsub_d.get("max_object_size") is None:
            bgsub_d["max_object_size"] = float("inf")
        bgsub = BgSubConfig(**bgsub_d)

    ht_d = d.get("headtail")
    headtail = HeadTailConfig(**ht_d) if ht_d else None

    cnn_phases = [CNNConfig(**c) for c in d.get("cnn_phases", [])]

    raw_tier = d.get("runtime_tier")
    if raw_tier is None:
        raise ValueError(
            "Config has no 'runtime_tier'. Runtime Gen-2 requires an explicit tier "
            "(cpu/gpu/gpu_fast). Migrate legacy configs with "
            "`python scripts/migrate_runtime_config.py <file>` (added in a later task)."
        )

    pose_d = d.get("pose")
    pose = None
    if pose_d:
        yolo_d = pose_d.pop("yolo", None)
        sleap_d = pose_d.pop("sleap", None)
        vitpose_d = pose_d.pop("vitpose", None)
        # Dropped field (was always (0, 0, 0); never populated by
        # from_parameters). Pop so stale serialized configs from before this
        # change don't fail PoseConfig(**pose_d) with an unexpected kwarg.
        pose_d.pop("background_color", None)
        pose = PoseConfig(
            **pose_d,
            yolo=PoseYOLOConfig(**yolo_d) if yolo_d else None,
            sleap=PoseSLEAPConfig(**sleap_d) if sleap_d else None,
            vitpose=PoseViTPoseConfig(**vitpose_d) if vitpose_d else None,
        )

    at_d = d.get("apriltag", {})
    if isinstance(at_d.get("unsharp_kernel"), list):
        at_d["unsharp_kernel"] = tuple(at_d["unsharp_kernel"])
    apriltag = AprilTagConfig(**at_d) if at_d else AprilTagConfig()

    canonical_d = d.get("canonical")
    canonical = (
        CanonicalGeometry(
            canvas_wh=tuple(canonical_d["canvas_wh"]),
            margin=float(canonical_d["margin"]),
            aspect_ratio=float(canonical_d["aspect_ratio"]),
        )
        if canonical_d
        else _default_canonical_geometry()
    )

    return InferenceConfig(
        obb=obb,
        bgsub=bgsub,
        headtail=headtail,
        cnn_phases=cnn_phases,
        pose=pose,
        apriltag=apriltag,
        canonical=canonical,
        detection_batch_size=d.get("detection_batch_size", 1),
        pipeline_depth=d.get("pipeline_depth", 2),
        runtime_tier=raw_tier,
        realtime=d.get("realtime", False),
        use_cache=d.get("use_cache", True),
        cache_dir=d.get("cache_dir"),
    )


def _clamped_int(raw: Any, default: int, lo: int, hi: int) -> int:
    try:
        v = int(raw)
    except (TypeError, ValueError):
        return default
    return v if lo <= v <= hi else default


def _clamped_float(raw: Any, default: float, lo: float, hi: float) -> float:
    try:
        v = float(raw)
    except (TypeError, ValueError):
        return default
    return v if math.isfinite(v) and lo <= v <= hi else default


def _slice_config_from_params(
    params: dict, prefix: str, *, reference_body_px: float
) -> SliceConfig:
    """Build a ``SliceConfig`` from ``{prefix}*`` params.

    Shared by the direct-mode ``SLICE_*`` mapping and the sequential-mode
    stage-1 ``YOLO_SEQ_STAGE1_SLICE_*`` mapping (Task 11) -- same field
    semantics, different param-name prefix.
    """
    overlap = _clamped_float(params.get(f"{prefix}OVERLAP", 0.2), 0.2, 0.0, 0.9)
    _geometry_mode = (
        str(params.get(f"{prefix}GEOMETRY_MODE", "auto_model")).strip().lower()
    )
    _merge_policy = (
        str(params.get(f"{prefix}MERGE_POLICY", "greedy_nmm")).strip().lower()
    )
    _merge_metric = str(params.get(f"{prefix}MERGE_METRIC", "ios")).strip().lower()
    _merge_backend = str(params.get(f"{prefix}MERGE_BACKEND", "cv2")).strip().lower()
    return SliceConfig(
        enabled=bool(params.get(f"{prefix}ENABLED", False)),
        geometry_mode=(
            _geometry_mode
            if _geometry_mode in {"auto_model", "auto_object", "custom"}
            else "auto_model"
        ),
        slice_height=_clamped_int(params.get(f"{prefix}HEIGHT", 0), 0, 0, 8192),
        slice_width=_clamped_int(params.get(f"{prefix}WIDTH", 0), 0, 0, 8192),
        overlap_height_ratio=overlap,
        overlap_width_ratio=overlap,
        object_tile_fraction=_clamped_float(
            params.get(f"{prefix}OBJECT_TILE_FRACTION", 0.15), 0.15, 0.01, 0.9
        ),
        reference_body_px=reference_body_px,
        merge_policy=(
            _merge_policy
            if _merge_policy in {"nms", "nmm", "greedy_nmm"}
            else "greedy_nmm"
        ),
        merge_metric=(_merge_metric if _merge_metric in {"iou", "ios"} else "ios"),
        merge_threshold=_clamped_float(
            params.get(f"{prefix}MERGE_THRESHOLD", 0.5), 0.5, 0.0, 1.0
        ),
        merge_backend=(_merge_backend if _merge_backend in {"cv2", "gpu"} else "cv2"),
        perform_standard_pred=bool(params.get(f"{prefix}PERFORM_STANDARD_PRED", False)),
    )


def _resolve_cnn_temperature(cnn_cfg_dict: dict, model_path: str) -> float:
    """Resolve the calibration temperature for a CNN classifier phase.

    Params win when they specify ``calibration_temperature`` (or the legacy
    ``temperature`` key) explicitly. Otherwise, fall back to the artifact's
    own stored per-factor temperature (Task 5's ``ClassifierMetadata``);
    the flat/scalar consume path here uses the first factor's value. Any
    missing/unreadable/uncalibrated artifact defaults to ``1.0`` (today's
    behavior — no regression).
    """
    explicit = cnn_cfg_dict.get(
        "calibration_temperature", cnn_cfg_dict.get("temperature")
    )
    if explicit is not None:
        return float(explicit)
    try:
        from hydra_suite.core.individual.classification.backend import _select_loader

        metadata = _select_loader(model_path).parse_metadata(model_path)
        temps = metadata.calibration_temperature
        if temps:
            return float(temps[0])
    except Exception:
        pass
    return 1.0


def _gate_calibration(params: dict) -> None:
    """Mandatory-calibration gate for ``unique_identifier`` CNN models.

    A no-op unless ``params["IDENTITY_CALIBRATION_REQUIRED"]`` is set. When it
    is, every ``CNN_CLASSIFIERS`` entry marked ``unique_identifier`` is
    checked via a metadata-only parse (no weight load -- ``current_signature``
    is intentionally left ``None``, so ``calibration_status`` can only report
    ``"calibrated"``/``"uncalibrated"``, never ``"stale"``; staleness is a
    display concern handled elsewhere). An uncalibrated model raises
    ``CalibrationRequiredError`` naming the recalibrate action, unless
    ``params["IDENTITY_CALIBRATION_OVERRIDE"]`` is set, in which case the same
    message is logged as a warning instead.
    """
    if not params.get("IDENTITY_CALIBRATION_REQUIRED"):
        return
    override = bool(params.get("IDENTITY_CALIBRATION_OVERRIDE"))
    for cnn_cfg_dict in params.get("CNN_CLASSIFIERS", []):
        if not bool(cnn_cfg_dict.get("unique_identifier", False)):
            continue
        model_path = str(cnn_cfg_dict.get("model_path", "")).strip()
        if not model_path or not os.path.exists(model_path):
            continue
        try:
            from hydra_suite.core.individual.classification.backend import (
                _select_loader,
                calibration_status,
            )

            meta = _select_loader(model_path).parse_metadata(model_path)
            status = calibration_status(meta, None)
        except Exception:
            # A read failure means we cannot prove calibration -- treat it as
            # uncalibrated rather than silently letting an unverifiable model
            # through the gate.
            status = "uncalibrated"
        if status == "uncalibrated":
            message = (
                f"Identity model {model_path!r} is uncalibrated but calibration "
                "is required (IDENTITY_CALIBRATION_REQUIRED). Please recalibrate "
                "it via ClassKit -> Recalibrate model... before tracking, or set "
                "IDENTITY_CALIBRATION_OVERRIDE to downgrade this to a warning."
            )
            if override:
                logger.warning(message)
            else:
                raise CalibrationRequiredError(message)


def build_inference_config_from_params(params: dict) -> InferenceConfig:
    """Build an InferenceConfig from a tracking-worker params dict.

    Maps legacy YOLO/headtail/CNN/pose/AprilTag params to the structured
    InferenceConfig dataclasses consumed by InferenceRunner. Stages whose
    params are absent/disabled stay unset, so an OBB-only params dict yields
    an OBB-only config (headtail=None, cnn_phases=[], pose=None).
    """
    # Pipeline-wide compute tier drives backend/device selection in the
    # redesign. Runtime Gen-2 uses runtime_tier as the sole source of truth;
    # an absent or invalid RUNTIME_TIER defaults to "cpu".
    _raw_tier = str(params.get("RUNTIME_TIER", "") or "").strip().lower()
    runtime_tier = _raw_tier if _raw_tier in {"cpu", "gpu", "gpu_fast"} else "cpu"
    obb_mode = str(params.get("YOLO_OBB_MODE", "direct")).strip().lower()
    if obb_mode not in {"direct", "sequential"}:
        obb_mode = "direct"

    direct_model_path = str(
        params.get(
            "YOLO_OBB_DIRECT_MODEL_PATH",
            params.get("YOLO_MODEL_PATH", "yolo26s-obb.pt"),
        )
        or "yolo26s-obb.pt"
    )
    yolo_conf = float(params.get("YOLO_CONFIDENCE_THRESHOLD", 0.25))
    yolo_iou = float(params.get("YOLO_IOU_THRESHOLD", 0.7))
    min_obj = float(params.get("MIN_OBJECT_SIZE", 0.0))
    max_obj = float(params.get("MAX_OBJECT_SIZE", float("inf")) or float("inf"))
    # Detection caps mirror legacy core/detectors/_obb_geometry:
    #   * RAW cap = 2 * MAX_TARGETS, applied at OBB extraction sorted by
    #     confidence, BEFORE size/aspect/IoU filtering.
    #   * FINAL cap = MAX_TARGETS, applied AFTER filtering, keeping the
    #     LARGEST detections (filtering sorts the cap by size, not conf).
    # Setting max_detections = MAX_TARGETS (not 2*MAX_TARGETS) restores the
    # legacy post-filter count cap (`_obb_geometry:587`) the redesign dropped.
    max_targets = max(1, int(params.get("MAX_TARGETS", 8)))
    raw_cap = 2 * max_targets
    max_dets = max_targets

    # Restrict detections to specific class IDs (legacy YOLO_TARGET_CLASSES;
    # None/empty == all classes). Threaded into OBBConfig.target_classes and
    # passed to every model.predict() (legacy yolo_detector.py:489,1078,1665).
    _target_classes_raw = params.get("YOLO_TARGET_CLASSES", None)
    target_classes = (
        [int(c) for c in _target_classes_raw] if _target_classes_raw else []
    )

    # Aspect-ratio gate (major/minor), mirroring legacy _obb_geometry: only
    # applied when enabled; bounds = ref_ar * mult. These are power-user
    # settings stored under ADVANCED_CONFIG (lowercase keys), matching legacy
    # _advanced_config_value access in core/detectors/_obb_geometry.py.
    _adv = params.get("ADVANCED_CONFIG", {}) or {}
    if _adv.get("enable_aspect_ratio_filtering", False):
        ref_ar = float(_adv.get("reference_aspect_ratio", 2.0))
        min_ar = ref_ar * float(_adv.get("min_aspect_ratio_multiplier", 0.5))
        max_ar = ref_ar * float(_adv.get("max_aspect_ratio_multiplier", 2.0))
    else:
        min_ar, max_ar = 0.0, float("inf")

    # Single project-wide canonical crop geometry (Layer 1), shared by
    # head-tail, CNN, and pose crops alike -- built ONCE here regardless of
    # which of those stages end up configured below. CanonicalGeometry clamps
    # aspect_ratio/margin to >= 1.0, so this is also the single place that
    # guards against a degenerate (< 1.0) advanced-config value silently
    # reaching any classifier. Delegates to ``canonical_geometry_from_params``
    # so this is not a second copy of that derivation.
    canonical = canonical_geometry_from_params(params)

    if obb_mode == "sequential":
        detect_path = str(params.get("YOLO_DETECT_MODEL_PATH", "") or "")
        crop_path = str(params.get("YOLO_CROP_OBB_MODEL_PATH", "") or direct_model_path)

        # Stage-1 tiling (Task 11, off by default). Same reference-body-px
        # derivation as the direct-mode SLICE_* mapping below, with its own
        # SLICE_TRAINED_BODY_PX-style override key so a stage-1 tile grid can
        # be sized independently of any direct-mode grid in the same config.
        _stage1_trained_body_px = _clamped_float(
            params.get("YOLO_SEQ_STAGE1_SLICE_TRAINED_BODY_PX", 0.0), 0.0, 0.0, 8192.0
        )
        _stage1_reference_body_px = (
            _stage1_trained_body_px
            if _stage1_trained_body_px > 0
            else _clamped_float(
                float(params.get("REFERENCE_BODY_SIZE", 20.0) or 20.0)
                * float(params.get("RESIZE_FACTOR", 1.0) or 1.0),
                0.0,
                0.0,
                8192.0,
            )
        )
        stage1_slice_cfg = _slice_config_from_params(
            params,
            "YOLO_SEQ_STAGE1_SLICE_",
            reference_body_px=_stage1_reference_body_px,
        )

        # YOLO_SEQ_* keys mirror the legacy per-stage sequential-OBB knobs
        # (yolo_detector.py:_seq_*); threading them through here keeps the
        # redesign's sequential pipeline config-driven instead of silently
        # falling back to OBBSequentialConfig's dataclass defaults.
        obb_cfg = OBBConfig(
            mode="sequential",
            sequential=OBBSequentialConfig(
                detect_model_path=detect_path,
                obb_model_path=crop_path,
                detect_confidence_threshold=float(
                    params.get("YOLO_SEQ_DETECT_CONF_THRESHOLD", 0.25)
                ),
                obb_confidence_threshold=yolo_conf,
                detect_image_size=int(params.get("YOLO_SEQ_DETECT_IMGSZ", 0)),
                crop_pad_ratio=float(params.get("YOLO_SEQ_CROP_PAD_RATIO", 0.15)),
                min_crop_size_px=float(params.get("YOLO_SEQ_MIN_CROP_SIZE_PX", 64.0)),
                enforce_square_crop=bool(
                    params.get("YOLO_SEQ_ENFORCE_SQUARE_CROP", True)
                ),
                stage2_image_size=int(params.get("YOLO_SEQ_STAGE2_IMGSZ", 160)),
                stage2_batch_size=(
                    int(params["YOLO_SEQ_INDIVIDUAL_BATCH_SIZE"])
                    if params.get("YOLO_SEQ_INDIVIDUAL_BATCH_SIZE")
                    else None
                ),
                stage2_task=(
                    "segment"
                    if str(params.get("YOLO_SEQ_STAGE2_TASK", "obb")).strip().lower()
                    == "segment"
                    else "obb"
                ),
                seg_num_angles=int(params.get("YOLO_SEQ_SEG_NUM_ANGLES", 24)),
                seg_crop_size=int(params.get("YOLO_SEQ_SEG_CROP_SIZE", 64)),
                seg_pad_ratio=float(params.get("YOLO_SEQ_SEG_PAD_RATIO", 0.15)),
                seg_mask_threshold=float(
                    params.get("YOLO_SEQ_SEG_MASK_THRESHOLD", 0.5)
                ),
                stage1_slice=stage1_slice_cfg,
            ),
            target_classes=target_classes,
            confidence_threshold=yolo_conf,
            iou_threshold=yolo_iou,
            min_object_size=min_obj,
            max_object_size=max_obj,
            min_aspect_ratio=min_ar,
            max_aspect_ratio=max_ar,
            max_detections=max_dets,
            raw_detection_cap=raw_cap,
        )
    else:
        model_task = str(params.get("YOLO_OBB_DIRECT_TASK", "obb")).strip().lower()
        if model_task not in {"obb", "detect", "segment"}:
            model_task = "obb"
        fixed_angle_deg = float(params.get("YOLO_OBB_FIXED_ANGLE_DEG", 0.0) or 0.0)

        seg_num_angles = _clamped_int(
            params.get("YOLO_OBB_SEG_NUM_ANGLES", 24), 24, 4, 180
        )
        seg_crop_size = _clamped_int(
            params.get("YOLO_OBB_SEG_CROP_SIZE", 64), 64, 16, 256
        )
        seg_pad_ratio = _clamped_float(
            params.get("YOLO_OBB_SEG_PAD_RATIO", 0.15), 0.15, 0.0, 1.0
        )
        seg_mask_threshold = _clamped_float(
            params.get("YOLO_OBB_SEG_MASK_THRESHOLD", 0.5), 0.5, 0.05, 0.95
        )

        _trained_body_px = _clamped_float(
            params.get("SLICE_TRAINED_BODY_PX", 0.0), 0.0, 0.0, 8192.0
        )
        _reference_body_px = (
            _trained_body_px
            if _trained_body_px > 0
            else _clamped_float(
                float(params.get("REFERENCE_BODY_SIZE", 20.0) or 20.0)
                * float(params.get("RESIZE_FACTOR", 1.0) or 1.0),
                0.0,
                0.0,
                8192.0,
            )
        )
        # auto_object needs a real object scale or it silently degrades to
        # auto_model. Same source/scaling worker.py uses (worker.py:921).
        slice_cfg = _slice_config_from_params(
            params, "SLICE_", reference_body_px=_reference_body_px
        )

        obb_cfg = OBBConfig(
            mode="direct",
            direct=OBBDirectConfig(
                model_path=direct_model_path,
                confidence_floor=1e-3,
                confidence_threshold=yolo_conf,
                model_task=model_task,
                fixed_angle_deg=fixed_angle_deg,
                seg_num_angles=seg_num_angles,
                seg_crop_size=seg_crop_size,
                seg_pad_ratio=seg_pad_ratio,
                seg_mask_threshold=seg_mask_threshold,
                slice=slice_cfg,
            ),
            target_classes=target_classes,
            confidence_threshold=yolo_conf,
            iou_threshold=yolo_iou,
            min_object_size=min_obj,
            max_object_size=max_obj,
            min_aspect_ratio=min_ar,
            max_aspect_ratio=max_ar,
            max_detections=max_dets,
            raw_detection_cap=raw_cap,
        )

    # HeadTail
    headtail_model_path = str(params.get("YOLO_HEADTAIL_MODEL_PATH", "") or "").strip()
    headtail_cfg = None
    if headtail_model_path and os.path.exists(headtail_model_path):
        headtail_cfg = HeadTailConfig(
            model_path=headtail_model_path,
            confidence_threshold=float(params.get("YOLO_HEADTAIL_CONF_THRESHOLD", 0.5)),
            # Mirrors legacy's separate, stricter head-tail candidate gate
            # (_select_headtail_candidate_indices): detections below this
            # confidence never get classified at all (stay undirected),
            # independent of the main OBB filter's own confidence_threshold.
            candidate_confidence_threshold=float(
                params.get(
                    "YOLO_HEADTAIL_DETECT_CONF_THRESHOLD",
                    params.get("YOLO_CONFIDENCE_THRESHOLD", 0.25),
                )
            ),
            batch_size=int(params.get("HEADTAIL_BATCH_SIZE", 64)),
        )

    # CNN phases
    cnn_phases: list[CNNConfig] = []
    for cnn_cfg_dict in params.get("CNN_CLASSIFIERS", []):
        cnn_model_path = str(cnn_cfg_dict.get("model_path", "")).strip()
        if not cnn_model_path or not os.path.exists(cnn_model_path):
            continue
        cnn_label = str(cnn_cfg_dict.get("label", "cnn_identity"))
        cnn_phases.append(
            CNNConfig(
                label=cnn_label,
                model_path=cnn_model_path,
                confidence_threshold=float(cnn_cfg_dict.get("confidence", 0.5)),
                batch_size=int(cnn_cfg_dict.get("batch_size", 64)),
                scoring_mode=str(cnn_cfg_dict.get("scoring_mode", "atomic")),
                match_bonus=float(cnn_cfg_dict.get("match_bonus", 0.1)),
                mismatch_penalty=float(cnn_cfg_dict.get("mismatch_penalty", 0.3)),
                calibration_temperature=_resolve_cnn_temperature(
                    cnn_cfg_dict, cnn_model_path
                ),
            )
        )

    _gate_calibration(params)

    # Pose — supports both YOLO-pose and SLEAP backends.
    pose_cfg = None
    if bool(params.get("ENABLE_POSE_EXTRACTOR", False)):
        pose_model_type = str(params.get("POSE_MODEL_TYPE", "")).strip().lower()
        common_pose_kwargs = dict(
            skeleton_file=str(params.get("POSE_SKELETON_FILE", "") or "").strip(),
            crop_padding=float(params.get("INDIVIDUAL_CROP_PADDING", 0.1)),
            suppress_foreign_regions=bool(
                params.get("SUPPRESS_FOREIGN_OBB_REGIONS", True)
            ),
            min_keypoint_confidence=float(params.get("POSE_MIN_KPT_CONF_VALID", 0.2)),
            min_valid_keypoints=int(
                params.get("POSE_DIRECTION_MIN_VALID_KEYPOINTS", 1)
            ),
            anterior_keypoints=list(
                params.get("POSE_DIRECTION_ANTERIOR_KEYPOINTS", []) or []
            ),
            posterior_keypoints=list(
                params.get("POSE_DIRECTION_POSTERIOR_KEYPOINTS", []) or []
            ),
            ignore_keypoints=list(params.get("POSE_IGNORE_KEYPOINTS", []) or []),
            overrides_headtail=bool(params.get("POSE_OVERRIDES_HEADTAIL", True)),
        )
        sleap_model_path = str(
            params.get("POSE_SLEAP_MODEL_DIR", params.get("POSE_MODEL_DIR", "")) or ""
        ).strip()
        yolo_model_path = str(
            params.get(
                "POSE_YOLO_MODEL_DIR",
                params.get("POSE_MODEL_PATH", params.get("YOLO_POSE_MODEL_PATH", "")),
            )
            or ""
        ).strip()
        vitpose_model_path = str(
            params.get(
                "POSE_VITPOSE_MODEL_PATH",
                params.get("POSE_MODEL_PATH", params.get("POSE_MODEL_DIR", "")),
            )
            or ""
        ).strip()
        if pose_model_type == "sleap" and sleap_model_path:
            pose_cfg = PoseConfig(
                backend="sleap",
                sleap=PoseSLEAPConfig(
                    model_path=sleap_model_path,
                    batch_size=int(params.get("POSE_BATCH_SIZE", 4)),
                ),
                **common_pose_kwargs,
            )
        elif pose_model_type == "vitpose" and vitpose_model_path:
            pose_cfg = PoseConfig(
                backend="vitpose",
                vitpose=PoseViTPoseConfig(
                    model_path=vitpose_model_path,
                    batch_size=int(params.get("POSE_BATCH_SIZE", 4)),
                ),
                **common_pose_kwargs,
            )
        elif yolo_model_path and os.path.exists(yolo_model_path):
            pose_cfg = PoseConfig(
                backend="yolo",
                yolo=PoseYOLOConfig(
                    model_path=yolo_model_path,
                    confidence_threshold=float(
                        params.get("POSE_CONFIDENCE_THRESHOLD", 1e-4)
                    ),
                    iou_threshold=float(params.get("POSE_IOU_THRESHOLD", 0.7)),
                    max_detections_per_crop=1,
                    batch_size=int(params.get("POSE_BATCH_SIZE", 64)),
                ),
                **common_pose_kwargs,
            )

    # AprilTag
    apriltag_cfg = AprilTagConfig(
        enabled=bool(params.get("USE_APRILTAGS", False)),
        tag_family=str(params.get("APRILTAG_FAMILY", "tag36h11")),
        threads=int(params.get("APRILTAG_THREADS", 4)),
        max_hamming=int(params.get("APRILTAG_MAX_HAMMING", 1)),
        decimate=float(params.get("APRILTAG_DECIMATE", 1.0)),
        blur=float(params.get("APRILTAG_BLUR", 0.8)),
        crop_padding=float(params.get("INDIVIDUAL_CROP_PADDING", 0.1)),
    )

    batch_size = int(params.get("YOLO_BATCH_SIZE", params.get("BATCH_SIZE", 1)))

    return InferenceConfig(
        obb=obb_cfg,
        headtail=headtail_cfg,
        cnn_phases=cnn_phases,
        pose=pose_cfg,
        apriltag=apriltag_cfg,
        canonical=canonical,
        detection_batch_size=batch_size,
        realtime=False,
        use_cache=True,
        runtime_tier=runtime_tier,
    )


def build_obb_only_config(
    model_path: str,
    *,
    compute_runtime: str = "cpu",
    runtime_tier: str | None = None,
    confidence_threshold: float = 0.25,
    iou_threshold: float = 0.7,
    max_targets: int = 8,
    mode: str = "direct",
    model_task: str = "obb",
    emit_native_geometry: bool = False,
    extra_params: dict | None = None,
) -> InferenceConfig:
    """Detection-only InferenceConfig for one-shot / dataset OBB detection.

    ``model_task`` selects the checkpoint's head ("obb", "detect", "segment");
    it MUST match the checkpoint, which ``stages/obb.py`` verifies loudly.
    ``emit_native_geometry`` is the export-only opt-in that populates
    ``OBBResult.polygons`` with native contours (segment task only).

    ``extra_params`` supplies additional raw params keys that this helper's
    explicit arguments do not cover -- notably the ``YOLO_SEQ_*`` /
    ``YOLO_DETECT_MODEL_PATH`` / ``YOLO_CROP_OBB_MODEL_PATH`` family a
    ``mode="sequential"`` config needs. Keys set explicitly above always win.
    """
    task = str(model_task).strip().lower()
    if task not in {"obb", "detect", "segment"}:
        raise ValueError(
            f"model_task must be one of 'obb', 'detect', 'segment'; got {model_task!r}"
        )
    params: dict = {
        "DETECTION_METHOD": "yolo_obb",
        "YOLO_OBB_MODE": mode,
        "YOLO_OBB_DIRECT_MODEL_PATH": model_path,
        "YOLO_OBB_DIRECT_TASK": task,
        "COMPUTE_RUNTIME": compute_runtime,
        "YOLO_CONFIDENCE_THRESHOLD": confidence_threshold,
        "YOLO_IOU_THRESHOLD": iou_threshold,
        "MAX_TARGETS": max_targets,
    }
    if runtime_tier:
        params["RUNTIME_TIER"] = runtime_tier
    for key, value in (extra_params or {}).items():
        params.setdefault(key, value)
    cfg = build_inference_config_from_params(params)
    if emit_native_geometry:
        cfg.obb.emit_native_geometry = True
    return cfg
