"""Metadata-only resource admission for SAM3 LoRA training.

The parent process reads COCO metadata and asks ``nvidia-smi`` for a physical
CUDA identity. It never decodes a tile or imports torch/SAM3. Estimates refuse
obviously unsafe work; the independent child containment boundary remains the
hard backstop when an estimate is wrong.
"""

from __future__ import annotations

import csv
import json
import os
import shutil
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional

from hydra_suite.runtime.resource_budget import (
    AcceleratorKind,
    PhaseEstimate,
    ResourceBudget,
    ResourceObservation,
    ResourcePolicy,
    ResourceRequest,
    WorkLimits,
    evaluate_resource_request,
    probe_resources,
)
from hydra_suite.utils.sam3_constants import PREDICTOR_IMGSZ

GiB = 1024**3
MiB = 1024**2
SAM3_ESTIMATOR_VERSION = "sam3-lora-streaming-v1"
MIN_TRAIN_INSTANCES = 20

_CHECKPOINT_BYTES = int(3.5 * GiB)
_PUBLISH_DISK_BYTES = 8 * GiB
_TRAIN_HOST_FIXED_BYTES = 6 * GiB
_MODEL_LOAD_HOST_BYTES = 2 * _CHECKPOINT_BYTES
_PUBLISH_HOST_BYTES = 3 * _CHECKPOINT_BYTES
_MEASURED_BF16_DEVICE_PEAK_BYTES = 29 * GiB
_EXTRA_BATCH_DEVICE_BYTES = 18 * GiB
_DEVICE_STEADY_BYTES = 8 * GiB
_MASK_DEVICE_BYTES_PER_PIXEL = 16
_MASK_HOST_BYTES_PER_PIXEL = 5
_RUNTIME_HOST_ALLOWANCE_BYTES = 2 * GiB
_MAX_COCO_METADATA_BYTES = 128 * MiB
# Trainable parameters introduced per rank by the current injection suffixes.
# Optimizer admission budgets 16 bytes/parameter (weight, grad, Adam m/v).
_LORA_PARAMS_PER_RANK = {
    "adapt_vision_encoder": 565_248,
    "adapt_text_encoder": 245_760,
    "adapt_geometry_encoder": 13_824,
    "adapt_detr_encoder": 27_648,
    "adapt_detr_decoder": 27_648,
    "adapt_mask_decoder": 0,
}
_DEFAULT_HOST_RESERVE_GB = 8.0
_DEFAULT_HOST_RESERVE_FRACTION = 0.15
_DEFAULT_CUDA_SAFETY_FRACTION = 0.85
_PROBE_TIMEOUT_SECONDS = 5.0
_UNSET = object()


@dataclass(frozen=True)
class CudaDeviceObservation:
    """Physical CUDA identity and capacity reported by ``nvidia-smi``."""

    index: int
    uuid: str
    pci_bus_id: str
    name: str
    compute_capability: tuple[int, int]
    free_bytes: int
    total_bytes: int


@dataclass(frozen=True)
class Sam3DatasetProfile:
    """Lightweight COCO facts used by the estimator."""

    train_tiles: int
    validation_tiles: int
    train_instances: int
    max_active_instances_per_tile: int
    max_decoded_tile_bytes: int
    metadata_bytes: int
    raw_metadata_bytes: int
    polygon_count: int
    polygon_vertices: int

    @property
    def tile_count(self) -> int:
        return self.train_tiles + self.validation_tiles


@dataclass(frozen=True)
class Sam3PreflightDecision:
    """Complete auditable admission result for one proposed run."""

    admitted: bool
    request: ResourceRequest
    budget: ResourceBudget
    policy: ResourcePolicy
    dataset: Sam3DatasetProfile
    cuda_device: Optional[CudaDeviceObservation]
    free_disk_bytes: int
    refusals: tuple[str, ...]
    warnings: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-safe diagnostics for logs and run manifests."""

        return {
            "admitted": self.admitted,
            "request": asdict(self.request),
            "budget": asdict(self.budget),
            "policy": asdict(self.policy),
            "dataset": asdict(self.dataset),
            "cuda_device": asdict(self.cuda_device) if self.cuda_device else None,
            "free_disk_bytes": self.free_disk_bytes,
            "refusals": list(self.refusals),
            "warnings": list(self.warnings),
        }


def _free_disk_bytes(path: str) -> int:
    target = Path(path).expanduser().resolve()
    while not target.exists() and target != target.parent:
        target = target.parent
    return int(shutil.disk_usage(target).free)


def _load_coco(path: Path) -> tuple[dict[str, Any], int]:
    try:
        size = path.stat().st_size
        if size > _MAX_COCO_METADATA_BYTES:
            raise ValueError(
                f"COCO metadata {path} is {size} bytes; the metadata-only "
                f"preflight cap is {_MAX_COCO_METADATA_BYTES} bytes"
            )
        raw = path.read_text(encoding="utf-8")
        value = json.loads(raw)
    except FileNotFoundError:
        return {}, 0
    except OSError as exc:
        raise ValueError(f"Could not read COCO metadata {path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid COCO metadata {path}: {exc}") from exc
    return (value if isinstance(value, dict) else {}), len(raw.encode("utf-8"))


def _active_polygon_count(annotation: object) -> int:
    if not isinstance(annotation, dict):
        return 0
    segments = annotation.get("segmentation") or []
    if not isinstance(segments, list):
        return 0
    return sum(
        1
        for segment in segments
        if isinstance(segment, list) and len(segment) >= 6 and len(segment) % 2 == 0
    )


def _dataset_profile(dataset_dir: str) -> Sam3DatasetProfile:
    root = Path(dataset_dir).expanduser().resolve()
    train, train_bytes = _load_coco(root / "train" / "_annotations.coco.json")
    valid, valid_bytes = _load_coco(root / "valid" / "_annotations.coco.json")
    images = (
        train.get("images", []) if isinstance(train.get("images", []), list) else []
    )
    by_image: dict[tuple[int, int], int] = {}
    train_instances = 0
    polygon_count = 0
    polygon_vertices = 0
    decoded_bytes = 0
    for split_index, (split, is_train) in enumerate(((train, True), (valid, False))):
        split_images = split.get("images", [])
        if not isinstance(split_images, list):
            split_images = []
        for image in split_images:
            if not isinstance(image, dict):
                continue
            try:
                decoded_bytes = max(
                    decoded_bytes,
                    int(image.get("width", 0)) * int(image.get("height", 0)) * 3,
                )
            except (TypeError, ValueError):
                continue
        annotations = split.get("annotations", [])
        if not isinstance(annotations, list):
            annotations = []
        for annotation in annotations:
            if not isinstance(annotation, dict):
                continue
            polygons = _active_polygon_count(annotation)
            try:
                image_id = int(annotation["image_id"])
            except (KeyError, TypeError, ValueError):
                continue
            image_key = (split_index, image_id)
            by_image[image_key] = by_image.get(image_key, 0) + polygons
            polygon_count += polygons
            segments = annotation.get("segmentation") or []
            polygon_vertices += sum(
                len(segment) // 2
                for segment in segments
                if isinstance(segment, list)
                and len(segment) >= 6
                and len(segment) % 2 == 0
            )
            if is_train and polygons and not annotation.get("iscrowd"):
                train_instances += 1
    valid_images = valid.get("images", [])
    validation_tiles = len(valid_images) if isinstance(valid_images, list) else 0
    raw_metadata_bytes = train_bytes + valid_bytes
    metadata_bytes = (
        raw_metadata_bytes * 8
        + (len(images) + validation_tiles) * 512
        + polygon_count * 256
        + polygon_vertices * 128
    )
    return Sam3DatasetProfile(
        train_tiles=len(images),
        validation_tiles=validation_tiles,
        train_instances=train_instances,
        max_active_instances_per_tile=max(by_image.values(), default=0),
        max_decoded_tile_bytes=decoded_bytes,
        metadata_bytes=metadata_bytes,
        raw_metadata_bytes=raw_metadata_bytes,
        polygon_count=polygon_count,
        polygon_vertices=polygon_vertices,
    )


def _instance_count(dataset_dir: str) -> int:
    """Compatibility seam retained for callers/tests of the old preflight."""
    try:
        coco, _size = _load_coco(
            Path(dataset_dir).expanduser().resolve()
            / "train"
            / "_annotations.coco.json"
        )
    except ValueError:
        return 0
    annotations = coco.get("annotations", [])
    if not isinstance(annotations, list):
        return 0
    return sum(
        1
        for annotation in annotations
        if isinstance(annotation, dict) and not annotation.get("iscrowd")
    )


def _free_disk_gb(path: str) -> float:
    """Compatibility wrapper around the byte-precise disk probe."""

    return _free_disk_bytes(path) / GiB


def _parse_compute_capability(value: str) -> tuple[int, int]:
    major_text, separator, minor_text = value.strip().partition(".")
    if not separator:
        minor_text = "0"
    return int(major_text), int(minor_text)


def _visible_device_selector(device: str) -> str:
    if device not in {"auto", "cuda"} and not device.startswith("cuda:"):
        raise ValueError(
            f"SAM3 training requires an explicit CUDA runtime, not {device!r}"
        )
    if (
        "CUDA_VISIBLE_DEVICES" in os.environ
        and not os.environ["CUDA_VISIBLE_DEVICES"].strip()
    ):
        raise ValueError("CUDA_VISIBLE_DEVICES explicitly hides all devices")
    visible = [
        item.strip()
        for item in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")
        if item.strip()
    ]
    if visible:
        logical_index = 0
        if device.startswith("cuda:"):
            logical_index = int(device.partition(":")[2])
        if logical_index >= len(visible):
            raise ValueError("requested CUDA device is outside CUDA_VISIBLE_DEVICES")
        return visible[logical_index]
    if device.startswith("cuda:"):
        return device.partition(":")[2]
    return "0"


def _probe_cuda_device(device: str = "auto") -> Optional[CudaDeviceObservation]:
    """Probe one visible physical device without importing torch."""

    command = [
        "nvidia-smi",
        "--query-gpu=index,uuid,pci.bus_id,name,compute_cap,memory.total,memory.free",
        "--format=csv,noheader,nounits",
    ]
    try:
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=_PROBE_TIMEOUT_SECONDS,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if completed.returncode != 0:
        return None
    try:
        selector = _visible_device_selector(device)
    except (ValueError, IndexError):
        return None
    rows = list(csv.reader(completed.stdout.splitlines(), skipinitialspace=True))
    for row in rows:
        if len(row) != 7:
            continue
        index, uuid, pci_bus_id, name, capability, total_mib, free_mib = (
            item.strip() for item in row
        )
        if selector not in {index, uuid}:
            continue
        try:
            return CudaDeviceObservation(
                index=int(index),
                uuid=uuid,
                pci_bus_id=pci_bus_id,
                name=name,
                compute_capability=_parse_compute_capability(capability),
                free_bytes=int(float(free_mib) * MiB),
                total_bytes=int(float(total_mib) * MiB),
            )
        except ValueError:
            return None
    return None


def _resource_policy(params: Any) -> ResourcePolicy:
    return ResourcePolicy(
        reserve_host_bytes=int(
            float(getattr(params, "host_reserve_gb", _DEFAULT_HOST_RESERVE_GB)) * GiB
        ),
        reserve_host_fraction=float(
            getattr(params, "host_reserve_fraction", _DEFAULT_HOST_RESERVE_FRACTION)
        ),
        accelerator_safety_fraction=float(
            getattr(params, "cuda_safety_fraction", _DEFAULT_CUDA_SAFETY_FRACTION)
        ),
        warning_fraction=0.80,
    )


def build_resource_request(
    spec: Any, dataset: Sam3DatasetProfile, *, params: Any | None = None
) -> ResourceRequest:
    """Estimate phase peaks from streamed work limits and COCO metadata."""

    params = params if params is not None else getattr(spec, "sam3_params", None)
    if params is None:
        params = type(
            "MissingParams",
            (),
            {
                "batch": 1,
                "rank": 1,
                **{flag: False for flag in _LORA_PARAMS_PER_RANK},
            },
        )()
    batch_size = max(1, int(params.batch))
    inflight_tiles = min(batch_size, max(1, dataset.tile_count))
    image_pixels = PREDICTOR_IMGSZ * PREDICTOR_IMGSZ
    decoded_tiles = inflight_tiles * max(
        dataset.max_decoded_tile_bytes, image_pixels * 3
    )
    transformed_tiles = inflight_tiles * image_pixels * 3 * 4
    collated_images = batch_size * image_pixels * 3 * 4
    active_instances = inflight_tiles * dataset.max_active_instances_per_tile
    dense_masks_host = active_instances * image_pixels * _MASK_HOST_BYTES_PER_PIXEL
    dense_masks_device = active_instances * image_pixels * _MASK_DEVICE_BYTES_PER_PIXEL
    params_per_rank = sum(
        count
        for flag, count in _LORA_PARAMS_PER_RANK.items()
        if bool(getattr(params, flag, False))
    )
    lora_params = max(1, int(params.rank)) * params_per_rank
    lora_training_state = lora_params * 24
    lora_artifact = lora_params * 4
    default_lora_training_state = (
        16
        * (
            _LORA_PARAMS_PER_RANK["adapt_vision_encoder"]
            + _LORA_PARAMS_PER_RANK["adapt_geometry_encoder"]
            + _LORA_PARAMS_PER_RANK["adapt_detr_encoder"]
            + _LORA_PARAMS_PER_RANK["adapt_detr_decoder"]
        )
        * 24
    )
    metadata = dataset.metadata_bytes

    training_host_dynamic = (
        metadata
        + decoded_tiles
        + transformed_tiles
        + collated_images
        + dense_masks_host
    )
    training_host_peak = _TRAIN_HOST_FIXED_BYTES + training_host_dynamic
    training_device_peak = (
        _MEASURED_BF16_DEVICE_PEAK_BYTES
        + max(0, batch_size - 1) * _EXTRA_BATCH_DEVICE_BYTES
        + max(0, lora_training_state - default_lora_training_state)
        + dense_masks_device
    )
    validation_device_peak = training_device_peak
    common_allocations = (
        ("descriptor metadata", metadata),
        ("decoded tiles", decoded_tiles),
        ("transformed tile tensors", transformed_tiles),
        ("collated image tensors", collated_images),
        ("dense masks", dense_masks_host),
    )
    phases: tuple[PhaseEstimate, ...] = (
        PhaseEstimate(
            "model_load",
            host_steady_bytes=_CHECKPOINT_BYTES
            + _RUNTIME_HOST_ALLOWANCE_BYTES
            + metadata,
            host_peak_bytes=_MODEL_LOAD_HOST_BYTES
            + _RUNTIME_HOST_ALLOWANCE_BYTES
            + metadata,
            accelerator_steady_bytes=_DEVICE_STEADY_BYTES,
            accelerator_peak_bytes=_DEVICE_STEADY_BYTES,
            dominant_allocations=(
                ("base checkpoint", _CHECKPOINT_BYTES),
                ("model construction", _CHECKPOINT_BYTES),
                ("descriptor metadata", metadata),
            ),
        ),
        PhaseEstimate(
            "training",
            host_steady_bytes=_TRAIN_HOST_FIXED_BYTES + metadata,
            host_peak_bytes=training_host_peak,
            accelerator_steady_bytes=_DEVICE_STEADY_BYTES + lora_training_state,
            accelerator_peak_bytes=training_device_peak,
            disk_transient_bytes=2 * lora_artifact,
            dominant_allocations=common_allocations
            + (
                (
                    "measured BF16 model/activation envelope",
                    _MEASURED_BF16_DEVICE_PEAK_BYTES,
                ),
                ("LoRA and optimizer state", lora_training_state),
            ),
        ),
    )
    if dataset.validation_tiles:
        phases += (
            PhaseEstimate(
                "validation",
                host_steady_bytes=_TRAIN_HOST_FIXED_BYTES + metadata,
                host_peak_bytes=training_host_peak,
                accelerator_steady_bytes=_DEVICE_STEADY_BYTES + lora_training_state,
                accelerator_peak_bytes=validation_device_peak,
                dominant_allocations=common_allocations
                + (
                    (
                        "training model/activation envelope",
                        _MEASURED_BF16_DEVICE_PEAK_BYTES,
                    ),
                ),
            ),
        )
    publish_policy = getattr(spec, "publish_policy", None)
    if publish_policy is None or bool(getattr(publish_policy, "auto_import", True)):
        phases += (
            PhaseEstimate(
                "publish",
                host_steady_bytes=_CHECKPOINT_BYTES + lora_artifact + metadata,
                host_peak_bytes=_PUBLISH_HOST_BYTES + lora_artifact + metadata,
                disk_transient_bytes=_PUBLISH_DISK_BYTES,
                dominant_allocations=(
                    ("base checkpoint", _CHECKPOINT_BYTES),
                    ("merged checkpoint", _CHECKPOINT_BYTES),
                    ("serialization temporary", _CHECKPOINT_BYTES),
                    ("LoRA adapter", lora_artifact),
                    ("descriptor metadata", metadata),
                ),
            ),
        )
    return ResourceRequest(
        job_name="SAM3 LoRA training",
        phases=phases,
        limits=WorkLimits(
            batch_size=batch_size,
            workers=0,
            prefetch_batches=0,
            tiles=inflight_tiles,
            candidates=max(1, active_instances),
        ),
        estimator_version=SAM3_ESTIMATOR_VERSION,
    )


def _observe_resources(
    cuda_device: Optional[CudaDeviceObservation],
) -> ResourceObservation:
    if cuda_device is None:
        return probe_resources(AcceleratorKind.CPU)
    return probe_resources(
        AcceleratorKind.CUDA,
        accelerator_name=cuda_device.name,
        accelerator_probe=lambda: (cuda_device.free_bytes, cuda_device.total_bytes),
    )


def assess_preflight(
    spec: Any,
    *,
    cuda_device: CudaDeviceObservation | None | object = _UNSET,
    observation: Optional[ResourceObservation] = None,
    dataset: Optional[Sam3DatasetProfile] = None,
) -> Sam3PreflightDecision:
    """Return the complete initial or lease-held live admission decision."""

    params = getattr(spec, "sam3_params", None)
    if params is None:
        # Construct a harmless estimate so diagnostics remain typed.
        params = type(
            "MissingParams", (), {"batch": 1, "num_negatives": 0, "rank": 1}
        )()
    dataset = dataset or _dataset_profile(spec.derived_dataset_dir)
    request = build_resource_request(spec, dataset, params=params)
    if cuda_device is _UNSET:
        cuda_device = _probe_cuda_device(str(getattr(spec, "device", "auto")))
    assert cuda_device is None or isinstance(cuda_device, CudaDeviceObservation)
    observation = observation or _observe_resources(cuda_device)
    policy = _resource_policy(params)
    budget = evaluate_resource_request(request, observation, policy)
    free_disk = _free_disk_bytes(spec.derived_dataset_dir)
    refusals = list(budget.refusals)
    warnings = list(budget.warnings)

    if getattr(spec, "sam3_params", None) is None:
        refusals.append("SAM3 training requires a sam3_params configuration.")

    if cuda_device is None:
        refusals.append(
            "No CUDA device is available; SAM3 LoRA training requires CUDA."
        )
    elif cuda_device.compute_capability[0] < 8:
        major, minor = cuda_device.compute_capability
        refusals.append(
            "SAM3 LoRA training requires CUDA BF16 on compute capability >= 8.0; "
            f"the selected GPU reports {major}.{minor}. FP32 fallback is unsafe and disabled."
        )
    if getattr(params, "mixed_precision", None) != "bf16":
        refusals.append(
            "SAM3 LoRA training currently supports only CUDA BF16; fp16/fp32 "
            "modes fail against SAM3's BF16 activation path and are disabled."
        )
    if not any(bool(getattr(params, flag, False)) for flag in _LORA_PARAMS_PER_RANK):
        refusals.append(
            "SAM3 training requires at least one enabled adapter scope; all "
            "adapt_* flags are disabled."
        )
    prompt = str(getattr(params, "prompt", "") or "")
    if not prompt.strip():
        refusals.append(
            "Prompt is empty; SAM3 requires a text prompt to train against."
        )
    if dataset.train_instances < MIN_TRAIN_INSTANCES:
        refusals.append(
            f"Only {dataset.train_instances} labeled instances found; at least "
            f"{MIN_TRAIN_INSTANCES} are required to train."
        )
    required_disk = max(phase.disk_transient_bytes for phase in request.phases)
    if free_disk < required_disk:
        refusals.append(
            f"Only {free_disk / GiB:.1f} GiB free disk space; at least "
            f"{required_disk / GiB:.0f} GiB is required "
            "for transient publish artifacts."
        )
    if not bool(getattr(params, "label_quality_acknowledged", False)):
        refusals.append(
            "Label quality has not been acknowledged; affirm the training labels "
            "are good before SAM3 learns from them."
        )
    if getattr(spec, "resume_from", ""):
        refusals.append(
            "resume_from is set, but SAM3 LoRA training does not checkpoint "
            "optimiser state; resuming is not supported."
        )
    return Sam3PreflightDecision(
        admitted=not refusals,
        request=request,
        budget=budget,
        policy=policy,
        dataset=dataset,
        cuda_device=cuda_device,
        free_disk_bytes=free_disk,
        refusals=tuple(refusals),
        warnings=tuple(warnings),
    )


def preflight(spec: Any) -> list[str]:
    """Compatibility wrapper returning refusal strings."""

    return list(assess_preflight(spec).refusals)


def preflight_warnings(spec: Any) -> list[str]:
    """Compatibility wrapper returning warning strings."""

    return list(assess_preflight(spec).warnings)
