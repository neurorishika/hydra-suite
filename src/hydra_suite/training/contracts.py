"""Training contracts for MAT multi-role YOLO workflows."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any


class TrainingRole(str, Enum):
    """Canonical training roles supported by MAT."""

    OBB_DIRECT = "obb_direct"
    DETECT_DIRECT = "detect_direct"
    SEGMENT_DIRECT = "segment_direct"
    SEQ_DETECT = "seq_detect"
    SEQ_CROP_OBB = "seq_crop_obb"
    SEQ_CROP_SEGMENT = "seq_crop_segment"

    # ClassKit classification roles
    CLASSIFY_FLAT_YOLO = "classify_flat_yolo"
    CLASSIFY_FLAT_TINY = "classify_flat_tiny"
    CLASSIFY_MULTIHEAD_YOLO = "classify_multihead_yolo"
    CLASSIFY_MULTIHEAD_TINY = "classify_multihead_tiny"
    CLASSIFY_FLAT_CUSTOM = "classify_flat_custom"
    CLASSIFY_MULTIHEAD_CUSTOM = "classify_multihead_custom"
    CLASSIFY_MULTIHEAD_CUSTOM_SHARED = "classify_multihead_custom_shared"

    # DetectKit promptable-concept segmentation
    SEMANTIC_SAM3 = "semantic_sam3"


@dataclass(slots=True)
class SplitConfig:
    """Dataset split ratios."""

    train: float = 0.8
    val: float = 0.2
    test: float = 0.0


@dataclass(slots=True)
class SourceDataset:
    """Input dataset descriptor."""

    path: str
    source_type: str = "yolo_obb"
    name: str = ""
    # Native annotation fidelity. Training derives lower-fidelity labels for
    # a role but never invents a higher-fidelity geometry.
    level: str = "obb"


@dataclass(slots=True)
class TrainingHyperParams:
    """Generic training hyperparameters."""

    epochs: int = 100
    imgsz: int = 640
    batch: int = 16
    lr0: float = 0.01
    patience: int = 30
    workers: int = 8
    cache: bool = False


@dataclass(slots=True)
class TinyHeadTailParams:
    """Tiny classifier hyperparameters."""

    epochs: int = 50
    batch: int = 32
    lr: float = 3e-4
    weight_decay: float = 1e-2
    input_width: int = 128
    input_height: int = 64
    tiny_preset: str = "medium"
    # Architecture params
    hidden_layers: int = 1
    hidden_dim: int = 96
    dropout: float = 0.1
    # Early stopping
    patience: int = 10
    # Class-imbalance handling for tiny classifiers.
    # Modes: "none", "weighted_loss", "weighted_sampler", "both".
    class_rebalance_mode: str = "none"
    class_rebalance_power: float = 1.0
    # Label smoothing for CrossEntropyLoss in tiny multi-class training.
    label_smoothing: float = 0.0
    # Class name treated as intentionally unlabeled during supervised loss.
    ignore_label_name: str = "unknown"


@dataclass(slots=True)
class CustomCNNParams:
    """Hyperparameters for the unified Custom CNN training mode.

    Covers both TinyClassifier (backbone='tinyclassifier') and pretrained
    torchvision backbones (ConvNeXt, EfficientNet, ResNet, ViT).
    TinyClassifier-specific fields (hidden_layers, hidden_dim, dropout,
    input_width, input_height) are ignored when backbone != 'tinyclassifier'.
    """

    backbone: str = "tinyclassifier"
    fine_tune_method: str = "head_only"
    trainable_layers: int = 0  # 0=frozen, -1=all, N=last N layer groups
    backbone_lr_scale: float = 0.1  # LR multiplier for unfrozen backbone layers
    layerwise_lr_decay: float = 0.75
    gradual_unfreeze_interval: int = 5
    input_size: tuple[int, int] = (224, 224)  # (H, W) resize target for backbones
    epochs: int = 50
    batch: int = 32
    lr: float = 1e-3
    weight_decay: float = 1e-2
    patience: int = 10
    label_smoothing: float = 0.0
    class_rebalance_mode: str = "none"  # none, weighted_loss, weighted_sampler, both
    class_rebalance_power: float = 1.0
    ignore_label_name: str = "unknown"
    # TinyClassifier-specific (ignored for torchvision backbones)
    tiny_preset: str = "medium"
    hidden_layers: int = 1
    hidden_dim: int = 96
    dropout: float = 0.1
    input_width: int = 128
    input_height: int = 64
    # Shared-trunk multi-head head MLP configuration.
    head_kind: str = "flat"  # "flat" | "multihead_shared_trunk"
    head_hidden_dim: int = 256
    head_dropout: float = 0.1


@dataclass(slots=True)
class Sam3LoraParams:
    """SAM3 LoRA finetuning hyperparameters.

    Defaults are measured on the spike (see the design doc's Why section), not
    chosen by taste; where a value is a judgement call the comment says so.
    """

    prompt: str = ""  # required; there is no defensible default concept
    negative_prompts: list[str] = field(default_factory=list)
    rank: int = 16
    alpha: int = 32
    dropout: float = 0.1
    lr: float = 5e-5
    # AP75 plateaus by epoch ~9 (mean .642, sd .040); 40 buys nothing.
    epochs: int = 10
    # batch 2 OOMs at 1008 px on a 47 GB card; effective batch is batch*accum.
    batch: int = 1
    grad_accum: int = 8
    mixed_precision: str = "bf16"
    num_negatives: int = 3
    # Which submodules receive adapters. The text encoder is False as a
    # precaution against eroding prompt discrimination -- untested; the spike
    # froze it in every configuration.
    adapt_vision_encoder: bool = True
    adapt_text_encoder: bool = False
    adapt_geometry_encoder: bool = True
    adapt_detr_encoder: bool = True
    adapt_detr_decoder: bool = True
    adapt_mask_decoder: bool = True
    # Tiling, mirroring the SAHI sliced-training knobs.
    geometry_mode: str = "auto_object"  # auto_object | auto_model | custom
    object_tile_fraction: float = 0.055
    slice_width: int = 0  # custom mode only; 0 => fall back to imgsz
    slice_height: int = 0  # custom mode only
    tile_overlap: float = 0.25
    keep_empty_tiles: bool = True
    # Provenance does not survive a review (see the design's ordering
    # dependency), so the user must affirm the labels are good before SAM3
    # learns them -- including its own accepted output. Preflight refuses
    # without it; the panel writes it; the dialog gates the run on it.
    label_quality_acknowledged: bool = False
    # Which sidecar conda env to launch training in. Empty resolves to
    # `resolve_sam3_env`'s default (`DEFAULT_SAM3_ENV`, or `HYDRA_SAM3_ENV`);
    # travels with the spec so a run's env choice is recorded.
    env_name: str = ""


@dataclass(slots=True)
class AugmentationProfile:
    """Augmentation settings for training."""

    enabled: bool = True
    flipud: float = 0.0
    fliplr: float = 0.0
    rotate: float = 0.0
    hue: float = 0.0
    saturation: float = 0.0
    brightness: float = 0.0
    contrast: float = 0.0
    decode_color_sim: float = (
        0.0  # 0=off; ~0.5 recommended. P(apply) decode-color re-sim.
    )
    resample_sim: float = 0.0  # 0=off; ~0.3 recommended. P(apply) alternate resampler.
    canonical_aug: bool = False  # off by default; opt-in Moderate CanonicalAug
    # (resample-kernel swap + sub-pixel warp jitter + mild blur/JPEG degrade)
    # applied to the canonical crop before the Layer-2 letterbox. Training-only.
    canonical_aug_copies: int = 3  # extra augmented copies per image in the
    # YOLO-classify offline prefit when canonical_aug is on (0 => clean only).
    # The clean copy is always written; total = 1 + canonical_aug_copies.
    monochrome: bool = False
    args: dict[str, Any] = field(default_factory=dict)
    # Label-switching expansion rules.
    # Maps flip axis name → {source_class_name: target_class_name}.
    # When set, ExportWorker physically writes extra flipped copies with the
    # remapped label so the model is trained on both the original and its
    # mirror with the correct label — useful for directional/orientation labels.
    # Example:  {"fliplr": {"left": "right", "right": "left"},
    #            "flipud": {"up": "down", "down": "up"}}
    label_expansion: dict[str, dict[str, str]] = field(default_factory=dict)


@dataclass(slots=True)
class PublishPolicy:
    """Post-training artifact publishing policy."""

    auto_import: bool = True
    auto_select: bool = False  # noqa: DC01  (dataclass field)


@dataclass(slots=True)
class TrainingRunSpec:
    """Full run spec persisted to local registry."""

    role: TrainingRole
    source_datasets: list[SourceDataset]  # noqa: DC01  (dataclass field)
    derived_dataset_dir: str
    base_model: str
    hyperparams: TrainingHyperParams
    device: str = "auto"
    seed: int = 42
    training_space: str = "original"  # "original" or "canonical"
    resume_from: str = ""  # Path to last.pt checkpoint to resume from
    augmentation_profile: AugmentationProfile = field(
        default_factory=AugmentationProfile
    )
    publish_policy: PublishPolicy = field(default_factory=PublishPolicy)
    tiny_params: TinyHeadTailParams = field(default_factory=TinyHeadTailParams)
    custom_params: CustomCNNParams | None = None
    sam3_params: Sam3LoraParams | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize the run spec to a plain dict, converting the role enum to its string value."""
        out = asdict(self)
        out["role"] = self.role.value
        return out


@dataclass(slots=True)
class ValidationIssue:
    """Structured validation issue entry."""

    severity: str
    code: str
    message: str
    path: str = ""


@dataclass(slots=True)
class ValidationReport:
    """Validation summary for preflight checks."""

    valid: bool
    issues: list[ValidationIssue] = field(default_factory=list)
    stats: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize the validation report including all issues and collected statistics."""
        return {
            "valid": bool(self.valid),
            "issues": [asdict(i) for i in self.issues],
            "stats": dict(self.stats),
        }


@dataclass(slots=True)
class DatasetBuildResult:
    """Result of a dataset-build stage."""

    dataset_dir: str
    stats: dict[str, Any] = field(default_factory=dict)
    manifest_path: str = ""
