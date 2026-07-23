from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from typing import Any, Literal

from hydra_suite.runtime.resolver import RuntimeTier


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
    # onnx_* entries kept for legacy-config migration only — not user-selectable.
    fast = {"onnx_cpu", "onnx_cuda", "onnx_coreml", "tensorrt"}
    gpu = {"cuda", "mps"}
    if runtimes & fast:
        return "gpu_fast"
    if runtimes & gpu:
        return "gpu"
    return "cpu"


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
    canonical_aspect_ratio: float = 2.0
    canonical_margin: float = 1.3


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


@dataclass
class PoseConfig:
    backend: Literal["yolo", "sleap", "vitpose"] = "yolo"
    skeleton_file: str = ""
    yolo: PoseYOLOConfig | None = None
    sleap: PoseSLEAPConfig | None = None
    vitpose: PoseViTPoseConfig | None = None
    crop_padding: float = 0.1
    suppress_foreign_regions: bool = True
    background_color: tuple[int, int, int] = (0, 0, 0)
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
    obb = None
    if obb_d:
        if obb_d.get("max_object_size") is None:
            obb_d["max_object_size"] = float("inf")
        if obb_d.get("max_aspect_ratio") is None:
            obb_d["max_aspect_ratio"] = float("inf")

        direct = OBBDirectConfig(**obb_d["direct"]) if obb_d.get("direct") else None
        sequential = (
            OBBSequentialConfig(**obb_d["sequential"])
            if obb_d.get("sequential")
            else None
        )
        obb = OBBConfig(
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
            iou_threshold=obb_d.get("iou_threshold", 0.45),
        )

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
        bg = pose_d.get("background_color")
        if isinstance(bg, list):
            pose_d["background_color"] = tuple(bg)
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

    return InferenceConfig(
        obb=obb,
        bgsub=bgsub,
        headtail=headtail,
        cnn_phases=cnn_phases,
        pose=pose,
        apriltag=apriltag,
        detection_batch_size=d.get("detection_batch_size", 1),
        pipeline_depth=d.get("pipeline_depth", 2),
        runtime_tier=raw_tier,
        realtime=d.get("realtime", False),
        use_cache=d.get("use_cache", True),
        cache_dir=d.get("cache_dir"),
    )


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

    if obb_mode == "sequential":
        detect_path = str(params.get("YOLO_DETECT_MODEL_PATH", "") or "")
        crop_path = str(params.get("YOLO_CROP_OBB_MODEL_PATH", "") or direct_model_path)
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
        obb_cfg = OBBConfig(
            mode="direct",
            direct=OBBDirectConfig(
                model_path=direct_model_path,
                confidence_floor=1e-3,
                confidence_threshold=yolo_conf,
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
            canonical_aspect_ratio=float(
                params.get("ADVANCED_CONFIG", {}).get("reference_aspect_ratio", 2.0)
            ),
            canonical_margin=float(
                params.get("ADVANCED_CONFIG", {}).get(
                    "yolo_headtail_canonical_margin", 1.3
                )
            ),
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
                calibration_temperature=float(
                    cnn_cfg_dict.get(
                        "calibration_temperature", cnn_cfg_dict.get("temperature", 1.0)
                    )
                ),
            )
        )

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
) -> InferenceConfig:
    """Detection-only InferenceConfig for one-shot / dataset OBB detection.

    Thin wrapper over build_inference_config_from_params with every non-OBB
    stage left disabled. Used by callers that have a model path + runtime but
    no full tracking params dict. ``runtime_tier`` is the live runtime knob
    (Runtime Gen-2); when omitted, the tier is migrated from ``compute_runtime``.
    """
    params: dict = {
        "DETECTION_METHOD": "yolo_obb",
        "YOLO_OBB_MODE": mode,
        "YOLO_OBB_DIRECT_MODEL_PATH": model_path,
        "COMPUTE_RUNTIME": compute_runtime,
        "YOLO_CONFIDENCE_THRESHOLD": confidence_threshold,
        "YOLO_IOU_THRESHOLD": iou_threshold,
        "MAX_TARGETS": max_targets,
    }
    if runtime_tier:
        params["RUNTIME_TIER"] = runtime_tier
    return build_inference_config_from_params(params)
