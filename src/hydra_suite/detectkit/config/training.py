"""Typed, portable configuration for headless DetectKit training."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

from hydra_suite.training.contracts import (
    AugmentationProfile,
    PublishPolicy,
    Sam3LoraParams,
    SourceDataset,
    SplitConfig,
    TrainingHyperParams,
    TrainingRole,
)


class TrainingPlanError(ValueError):
    """Raised when a DetectKit training plan is malformed or unsupported."""


_DETECTKIT_ROLES = {
    TrainingRole.OBB_DIRECT,
    TrainingRole.DETECT_DIRECT,
    TrainingRole.SEGMENT_DIRECT,
    TrainingRole.SEQ_DETECT,
    TrainingRole.SEQ_CROP_OBB,
    TrainingRole.SEQ_CROP_SEGMENT,
    TrainingRole.SEMANTIC_SAM3,
}


def _require_mapping(value: object, name: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise TrainingPlanError(f"'{name}' must be a JSON object")
    return dict(value)


def _construct_dataclass(cls, values: dict[str, Any], name: str):
    allowed = {item.name for item in fields(cls)}
    unknown = sorted(set(values) - allowed)
    if unknown:
        raise TrainingPlanError(f"Unknown {name} option(s): {', '.join(unknown)}")
    try:
        return cls(**values)
    except (TypeError, ValueError) as exc:
        raise TrainingPlanError(f"Invalid {name}: {exc}") from exc


def _resolve_path(value: object, base_dir: Path, name: str) -> Path:
    text = str(value or "").strip()
    if not text:
        raise TrainingPlanError(f"'{name}' must not be empty")
    path = Path(text).expanduser()
    if not path.is_absolute():
        path = base_dir / path
    return path.resolve()


def _resolve_model(value: object, base_dir: Path) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    candidate = Path(text).expanduser()
    if candidate.is_absolute():
        return str(candidate.resolve())
    if text.startswith(("./", "../", "~")) or len(candidate.parts) > 1:
        return str((base_dir / candidate).resolve())
    # Plain names such as yolo26s-obb.pt are Ultralytics model identifiers.
    return text


@dataclass(frozen=True, slots=True)
class SliceTrainingConfig:
    """SAHI dataset settings shared by the DetectKit GUI and CLI."""

    enabled: bool = False
    geometry_mode: str = "auto_object"
    object_tile_fraction: float = 0.15
    reference_body_px: float = 0.0
    slice_width: int = 0
    slice_height: int = 0
    overlap: float = 0.2
    min_area_ratio: float = 0.1
    negative_tile_fraction: float = 0.15
    target_size_fractions: tuple[float, ...] = ()
    target_sizes: tuple[float, ...] = (200.0, 300.0, 400.0)
    full_frame_mix: bool = True
    merge_threshold: float = 0.5

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "SliceTrainingConfig":
        values = _require_mapping(data, "dataset.slicing")
        for name in ("object_tile_fraction", "reference_body_px", "overlap"):
            if name in values:
                values[name] = float(values[name])
        for name in ("min_area_ratio", "negative_tile_fraction", "merge_threshold"):
            if name in values:
                values[name] = float(values[name])
        for name in ("slice_width", "slice_height"):
            if name in values:
                values[name] = int(values[name])
        if "target_size_fractions" in values:
            values["target_size_fractions"] = tuple(
                float(item) for item in values["target_size_fractions"]
            )
        if "target_sizes" in values:
            values["target_sizes"] = tuple(
                float(item) for item in values["target_sizes"]
            )
        config = _construct_dataclass(cls, values, "dataset.slicing")
        config.validate()
        return config

    def validate(self) -> None:
        if self.geometry_mode not in {"auto_object", "auto_model", "custom"}:
            raise TrainingPlanError(
                "dataset.slicing.geometry_mode must be auto_object, auto_model, or custom"
            )
        for name, value in (
            ("overlap", self.overlap),
            ("min_area_ratio", self.min_area_ratio),
            ("negative_tile_fraction", self.negative_tile_fraction),
        ):
            if not 0.0 <= float(value) <= 1.0:
                raise TrainingPlanError(
                    f"dataset.slicing.{name} must be between 0 and 1"
                )
        if any(not 0.0 < float(value) <= 1.0 for value in self.target_size_fractions):
            raise TrainingPlanError(
                "dataset.slicing.target_size_fractions values must be in (0, 1]"
            )

    def target_fractions(self) -> list[float]:
        if self.target_size_fractions:
            return [float(value) for value in self.target_size_fractions]
        return [float(value) / 640.0 for value in self.target_sizes if value > 0.0]

    def target_sizes_for(self, imgsz: int) -> list[float]:
        input_size = max(1, int(imgsz))
        return [fraction * input_size for fraction in self.target_fractions()]

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["target_size_fractions"] = list(self.target_size_fractions)
        result["target_sizes"] = list(self.target_sizes)
        return result


@dataclass(frozen=True, slots=True)
class RoleTrainingConfig:
    """Model and input size for one DetectKit training role."""

    role: TrainingRole
    base_model: str
    imgsz: int = 640

    @classmethod
    def from_dict(cls, data: object, base_dir: Path) -> "RoleTrainingConfig":
        values = _require_mapping(data, "roles[]")
        role_text = str(values.get("role", "")).strip()
        try:
            role = TrainingRole(role_text)
        except ValueError as exc:
            raise TrainingPlanError(f"Unknown training role: {role_text!r}") from exc
        if role not in _DETECTKIT_ROLES:
            raise TrainingPlanError(f"{role.value!r} is not a DetectKit training role")
        model = _resolve_model(
            values.get("model", values.get("base_model", "")), base_dir
        )
        if role is not TrainingRole.SEMANTIC_SAM3 and not model:
            raise TrainingPlanError(f"Role {role.value!r} requires a model")
        imgsz = int(values.get("imgsz", 640))
        if imgsz <= 0:
            raise TrainingPlanError(f"Role {role.value!r} imgsz must be positive")
        unknown = sorted(set(values) - {"role", "model", "base_model", "imgsz"})
        if unknown:
            raise TrainingPlanError(f"Unknown roles[] option(s): {', '.join(unknown)}")
        return cls(role=role, base_model=model or "sam3", imgsz=imgsz)

    def to_dict(self) -> dict[str, Any]:
        return {
            "role": self.role.value,
            "model": self.base_model,
            "imgsz": self.imgsz,
        }


@dataclass(frozen=True, slots=True)
class DetectTrainingPlan:
    """Complete, immutable description of a DetectKit training session."""

    workspace_root: Path
    sources: tuple[SourceDataset, ...]
    class_names: tuple[str, ...]
    roles: tuple[RoleTrainingConfig, ...]
    split: SplitConfig = field(default_factory=SplitConfig)
    seed: int = 42
    dedup: bool = True
    crop_pad_ratio: float = 0.15
    min_crop_size_px: int = 64
    enforce_square: bool = True
    slice_settings: SliceTrainingConfig = field(default_factory=SliceTrainingConfig)
    hyperparams: TrainingHyperParams = field(default_factory=TrainingHyperParams)
    device: str = "auto"
    augmentation_profile: AugmentationProfile = field(
        default_factory=AugmentationProfile
    )
    publish_policy: PublishPolicy = field(
        default_factory=lambda: PublishPolicy(auto_import=False, auto_select=False)
    )
    species: str = "species"
    model_tag: str = "train"
    sam3_params: Sam3LoraParams | None = None

    @classmethod
    def from_dict(
        cls, data: object, *, base_dir: str | Path = "."
    ) -> "DetectTrainingPlan":
        root = _require_mapping(data, "training plan")
        version = int(root.get("version", 1))
        if version != 1:
            raise TrainingPlanError(f"Unsupported training plan version: {version}")

        base = Path(base_dir).expanduser().resolve()
        workspace = _resolve_path(root.get("workspace"), base, "workspace")

        raw_sources = root.get("sources")
        if not isinstance(raw_sources, list) or not raw_sources:
            raise TrainingPlanError("'sources' must contain at least one dataset")
        sources: list[SourceDataset] = []
        for index, raw_source in enumerate(raw_sources):
            source = _require_mapping(raw_source, f"sources[{index}]")
            unknown = sorted(set(source) - {"path", "source_type", "name", "level"})
            if unknown:
                raise TrainingPlanError(
                    f"Unknown sources[{index}] option(s): {', '.join(unknown)}"
                )
            sources.append(
                SourceDataset(
                    path=str(
                        _resolve_path(
                            source.get("path"), base, f"sources[{index}].path"
                        )
                    ),
                    source_type=str(source.get("source_type", "yolo_obb")),
                    name=str(source.get("name", "")),
                    level=str(source.get("level", "obb")),
                )
            )
            if sources[-1].level not in {"aabb", "obb", "polygon"}:
                raise TrainingPlanError(
                    f"sources[{index}].level must be aabb, obb, or polygon"
                )

        raw_roles = root.get("roles")
        if not isinstance(raw_roles, list) or not raw_roles:
            raise TrainingPlanError("'roles' must contain at least one training role")
        roles = tuple(RoleTrainingConfig.from_dict(item, base) for item in raw_roles)

        dataset = _require_mapping(root.get("dataset"), "dataset")
        unknown_dataset = sorted(
            set(dataset)
            - {
                "split",
                "deduplicate",
                "crop_pad_ratio",
                "min_crop_size_px",
                "enforce_square",
                "slicing",
            }
        )
        if unknown_dataset:
            raise TrainingPlanError(
                f"Unknown dataset option(s): {', '.join(unknown_dataset)}"
            )
        split_values = _require_mapping(dataset.get("split"), "dataset.split")
        for name in ("train", "val", "test"):
            if name in split_values:
                split_values[name] = float(split_values[name])
        split = _construct_dataclass(
            SplitConfig,
            split_values,
            "dataset.split",
        )
        slicing = SliceTrainingConfig.from_dict(dataset.get("slicing"))

        training = _require_mapping(root.get("training"), "training")
        augmentation = _construct_dataclass(
            AugmentationProfile,
            _require_mapping(training.get("augmentation"), "training.augmentation"),
            "training.augmentation",
        )
        hyperparam_names = {item.name for item in fields(TrainingHyperParams)}
        hyperparam_values = {
            key: value for key, value in training.items() if key in hyperparam_names
        }
        for name in ("epochs", "imgsz", "batch", "patience", "workers"):
            if name in hyperparam_values:
                hyperparam_values[name] = int(hyperparam_values[name])
        if "lr0" in hyperparam_values:
            hyperparam_values["lr0"] = float(hyperparam_values["lr0"])
        unknown_training = sorted(
            set(training) - hyperparam_names - {"device", "seed", "augmentation"}
        )
        if unknown_training:
            raise TrainingPlanError(
                f"Unknown training option(s): {', '.join(unknown_training)}"
            )
        hyperparams = _construct_dataclass(
            TrainingHyperParams, hyperparam_values, "training"
        )
        publish_values = _require_mapping(root.get("publish"), "publish")
        publish_values.setdefault("auto_import", False)
        publish_values.setdefault("auto_select", False)
        publish = _construct_dataclass(
            PublishPolicy,
            publish_values,
            "publish",
        )
        sam3_values = root.get("sam3")
        sam3_params = (
            _construct_dataclass(
                Sam3LoraParams,
                _require_mapping(sam3_values, "sam3"),
                "sam3",
            )
            if sam3_values is not None
            else None
        )

        raw_class_names = root.get("class_names", ["object"])
        if not isinstance(raw_class_names, list):
            raise TrainingPlanError("'class_names' must be a JSON array")
        class_names = tuple(
            str(name).strip() for name in raw_class_names if str(name).strip()
        )
        known_root = {
            "version",
            "workspace",
            "sources",
            "class_names",
            "dataset",
            "training",
            "roles",
            "publish",
            "species",
            "model_tag",
            "sam3",
        }
        unknown_root = sorted(set(root) - known_root)
        if unknown_root:
            raise TrainingPlanError(
                f"Unknown training plan option(s): {', '.join(unknown_root)}"
            )

        plan = cls(
            workspace_root=workspace,
            sources=tuple(sources),
            class_names=class_names,
            roles=roles,
            split=split,
            seed=int(training.get("seed", 42)),
            dedup=bool(dataset.get("deduplicate", True)),
            crop_pad_ratio=float(dataset.get("crop_pad_ratio", 0.15)),
            min_crop_size_px=int(dataset.get("min_crop_size_px", 64)),
            enforce_square=bool(dataset.get("enforce_square", True)),
            slice_settings=slicing,
            hyperparams=hyperparams,
            device=str(training.get("device", "auto")),
            augmentation_profile=augmentation,
            publish_policy=publish,
            species=str(root.get("species", "species") or "species"),
            model_tag=str(root.get("model_tag", "train") or "train"),
            sam3_params=sam3_params,
        )
        plan.validate()
        return plan

    def validate(self) -> None:
        if not self.class_names:
            raise TrainingPlanError("'class_names' must contain at least one name")
        if len(set(self.class_names)) != len(self.class_names):
            raise TrainingPlanError("'class_names' must not contain duplicates")
        split_total = float(self.split.train + self.split.val + self.split.test)
        if abs(split_total - 1.0) > 1e-6:
            raise TrainingPlanError("dataset.split values must sum to 1.0")
        if min(self.split.train, self.split.val, self.split.test) < 0.0:
            raise TrainingPlanError("dataset.split values must not be negative")
        role_names = [role.role for role in self.roles]
        if len(set(role_names)) != len(role_names):
            raise TrainingPlanError("'roles' must not contain duplicates")
        if TrainingRole.SEMANTIC_SAM3 in role_names:
            if len(self.sources) != 1:
                raise TrainingPlanError(
                    "SAM3 concept training supports exactly one source dataset"
                )
            if self.sam3_params is None:
                raise TrainingPlanError("SAM3 training requires a 'sam3' configuration")
            if not self.sam3_params.label_quality_acknowledged:
                raise TrainingPlanError(
                    "SAM3 training requires label_quality_acknowledged=true"
                )
        if self.hyperparams.epochs <= 0:
            raise TrainingPlanError("training.epochs must be positive")
        if self.hyperparams.batch == 0 or self.hyperparams.batch < -1:
            raise TrainingPlanError("training.batch must be -1 or a positive integer")
        if self.hyperparams.imgsz <= 0:
            raise TrainingPlanError("training.imgsz must be positive")
        if self.hyperparams.patience < 0:
            raise TrainingPlanError("training.patience must not be negative")
        if self.hyperparams.workers < 0:
            raise TrainingPlanError("training.workers must not be negative")
        if self.min_crop_size_px <= 0:
            raise TrainingPlanError("dataset.min_crop_size_px must be positive")

    def preparation_request(self):
        from hydra_suite.detectkit.jobs.training import DatasetPreparationRequest

        return DatasetPreparationRequest(
            sources=self.sources,
            roles=tuple(role.role for role in self.roles),
            class_names=self.class_names,
            split=self.split,
            seed=self.seed,
            dedup=self.dedup,
            crop_pad_ratio=self.crop_pad_ratio,
            min_crop_size_px=self.min_crop_size_px,
            enforce_square=self.enforce_square,
            imgsz_by_role=tuple((role.role.value, role.imgsz) for role in self.roles),
            slice_settings=self.slice_settings,
            sam3_params=self.sam3_params,
        )

    def role_entries(self, role_dataset_dirs: dict[str, str]):
        from hydra_suite.detectkit.jobs.training import build_role_entries

        return build_role_entries(self, role_dataset_dirs)

    def to_dict(self) -> dict[str, Any]:
        training = asdict(self.hyperparams)
        training.update(
            {
                "device": self.device,
                "seed": self.seed,
                "augmentation": asdict(self.augmentation_profile),
            }
        )
        return {
            "version": 1,
            "workspace": str(self.workspace_root),
            "sources": [asdict(source) for source in self.sources],
            "class_names": list(self.class_names),
            "dataset": {
                "split": asdict(self.split),
                "deduplicate": self.dedup,
                "crop_pad_ratio": self.crop_pad_ratio,
                "min_crop_size_px": self.min_crop_size_px,
                "enforce_square": self.enforce_square,
                "slicing": self.slice_settings.to_dict(),
            },
            "training": training,
            "roles": [role.to_dict() for role in self.roles],
            "publish": asdict(self.publish_policy),
            "species": self.species,
            "model_tag": self.model_tag,
            "sam3": asdict(self.sam3_params) if self.sam3_params else None,
        }


def load_training_plan(path: str | Path) -> DetectTrainingPlan:
    """Load and validate a DetectKit JSON training plan."""

    config_path = Path(path).expanduser().resolve()
    try:
        raw = json.loads(config_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise TrainingPlanError(f"Training config not found: {config_path}") from exc
    except json.JSONDecodeError as exc:
        raise TrainingPlanError(
            f"Invalid JSON in {config_path}: line {exc.lineno}, column {exc.colno}"
        ) from exc
    except OSError as exc:
        raise TrainingPlanError(f"Could not read training config: {exc}") from exc
    try:
        return DetectTrainingPlan.from_dict(raw, base_dir=config_path.parent)
    except TrainingPlanError:
        raise
    except (TypeError, ValueError) as exc:
        raise TrainingPlanError(f"Invalid training configuration: {exc}") from exc
