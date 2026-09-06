"""Metadata-only resource admission for SAM3 LoRA training.

The parent process reads COCO metadata and asks ``nvidia-smi`` for a physical
CUDA identity. It never decodes a tile or imports torch/SAM3. Estimates refuse
obviously unsafe work; the independent child containment boundary remains the
hard backstop when an estimate is wrong.
"""

from __future__ import annotations

import copy
import csv
import dataclasses
import hashlib
import json
import math
import os
import shutil
import subprocess
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, NamedTuple, Optional

from hydra_suite.runtime.memory_profiles import (
    MEASURED_SAFETY_FRACTION,
    MemoryMeasurement,
    MemoryProfileStore,
    ProfileIdentity,
    measured_envelope_bytes,
    profile_store_path,
    records_for,
)
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
from hydra_suite.training.contracts import (
    SAM3_MAX_CONFIGURED_PROMPT_BYTES,
    SAM3_MAX_NEGATIVE_PROMPT_BYTES,
    SAM3_MAX_NEGATIVE_PROMPT_COUNT,
    SAM3_MAX_NEGATIVE_QUERIES_PER_TILE,
    sam3_prompt_text_error,
)
from hydra_suite.training.device_ids import (
    CUDA_DEVICE_FORMS,
    names_several_devices,
    normalize_cuda_device,
)
from hydra_suite.utils.sam3_constants import PREDICTOR_IMGSZ

from .polygons import validated_segmentation_polygons
from .sizing import LORA_PARAMS_PER_RANK as _LORA_PARAMS_PER_RANK
from .sizing import MAX_LORA_RANK as _MAX_LORA_RANK
from .sizing import MAX_LORA_TRAINABLE_PARAMS as _MAX_LORA_TRAINABLE_PARAMS

GiB = 1024**3
MiB = 1024**2
SAM3_ESTIMATOR_VERSION = "sam3-lora-streaming-v2"
MIN_TRAIN_INSTANCES = 20

_CHECKPOINT_BYTES = int(3.5 * GiB)
_PUBLISH_DISK_BYTES = 8 * GiB
_TRAIN_HOST_FIXED_BYTES = 6 * GiB
_MODEL_LOAD_HOST_BYTES = 2 * _CHECKPOINT_BYTES
_RUNTIME_HOST_ALLOWANCE_BYTES = 2 * GiB
# The Set 3 publisher mutates the loaded state in place and releases it before
# mmap validation. Conservatively bound the one active tensor temporary by the
# entire checkpoint without claiming torch.save itself is a streaming format.
_PUBLISH_ACTIVE_TENSOR_BYTES = _CHECKPOINT_BYTES
_PUBLISH_HOST_BYTES = (
    _CHECKPOINT_BYTES + _PUBLISH_ACTIVE_TENSOR_BYTES + _RUNTIME_HOST_ALLOWANCE_BYTES
)
# MEASURED, and RE-MEASURED after the adapter surface changed.
#
# 2026-09-04, 206 adapters: 7.83 GiB steady across ~130 optimizer steps and a
# full shuffle of tiles, including the densest.
# 2026-09-05, 312 adapters (the current production surface, after SAM3's own
# attention clone was split): 7.72 GiB. Both via
# torch.cuda.max_memory_reserved on an RTX 6000 Ada, batch 1, rank 16, 1008px
# SAHI tiles, bf16, through the real `cli.run_training` on a real prepared
# dataset -- reserved, not allocated, because reserved is what the card must
# actually have free. The 314-module spike-parity surface (the same run plus
# `adapt_scoring_head`, which is NOT production-reachable) measured the same
# 7.72 GiB.
#
# So growing the surface from 206 to 312 modules did NOT move the peak: at
# rank 16 the adapters are ~11.4M parameters (~0.14 GiB of state) against a
# frozen base whose activations dominate, and the extra split projections
# reuse activations the fused kernels were already materialising.
#
# REFUTED 2026-09-06 -- the "~55% headroom, this constant remains sound"
# claim that stood here was WRONG, and wrong in the unsafe direction. The
# 7.72 GiB above came from a 30-step probe. A full run at the SAME surface,
# batch, precision and tile size reached 10.22 GiB by step 70, sat FLAT there
# for 230 steps across an epoch boundary, then rose to 11.59 and 12.99 GiB
# within ten steps -- exceeding not just this 12 GiB base but preflight's
# whole composed `training.accelerator_peak_bytes` (12.364 GiB) by 5.1%.
# The short probe under-measured by 41%.
#
# No OOM followed, but only because `accelerator_safety_fraction` 0.85
# demands 14.55 GiB free -- slack meant to absorb allocator noise, not to
# cover a wrong central estimate. The guard held for the wrong reason.
#
# Two lessons, both load-bearing:
#   * A TRUNCATED PROBE IS A LOWER BOUND, never a peak. Only a completed run
#     bounds it. Do not re-derive this constant from a short probe.
#   * FLATNESS IS NOT CONVERGENCE. 230 flat steps preceded a 1.37 GiB jump,
#     so no "stable for K steps" rule can be trusted at any plausible K.
#     Peak appears to track per-tile CONTENT (instance and mask count), so
#     the worst case surfaces only when the stream serves the densest tile.
#
# See docs/superpowers/specs/2026-09-06-sam3-training-vram-probe-audit.md.
#
# The measurements live in the profile store, dated and device-tagged
# (`runtime/memory_profiles.py`, written by
# `tools/sam3/measure_bf16_peak.py`). This constant is the coarse admission
# floor, NOT the measurement: never edit it to "the number someone measured",
# and never scale it when the surface changes -- re-run the probe.
# The previous value, 29 GiB, was inherited from the spike's very different
# setup (batch 4, rank 32, whole-image inputs) and was never a measurement of
# THIS role. At 3.7x the real figure it refused any card below ~32 GiB --
# excluding a 24 GB RTX 4090, which the measurement shows is comfortably
# sufficient. An admission gate calibrated against someone else's
# configuration rejects hardware that would in fact work.
_MEASURED_BF16_DEVICE_PEAK_BYTES = 12 * GiB
# FP32 keeps the same parameters resident but doubles every activation and
# autograd-saved tensor. Applied to the measured BF16 envelope this is a
# deliberate OVER-estimate (parameters do not double), which is the safe
# direction for an admission gate: it refuses a marginal run rather than
# admitting one that OOMs the device after minutes of setup.
_FP32_DEVICE_PEAK_MULTIPLIER = 2.0
SUPPORTED_PRECISIONS = ("bf16", "fp32")
# MEASURED, from two independent probes on two different corpora and two
# different GPU architectures, both under `expandable_segments:True`
# (>=60 optimizer steps/point, 312 adapters, rank 16, bf16). This constant is
# LOAD-BEARING ON THE SELECTION PATH, not just on admission: auto batch
# sizing resolves against `max(analytic, measured)`, and the analytic side of
# that max is this constant -- every extra item in the batch adds this many
# bytes to the requirement. It is a SECANT FROM n=1, deliberately, not a
# fitted slope and not a local marginal, because selection compares a
# candidate batch size n against the n=1 base:
#
#   Peer (courtship, RTX 4090, 78 frames / 1871 instances -> 486 train
#   tiles, 1766px):
#       b1 7.125 GiB, b2 8.434, b4 11.229, b8 18.748 GiB reserved
#       marginals 1.309 / 1.397 / 1.880 GiB (CONVEX/superlinear)
#       max secant from n=1: 1.660 GiB
#
#   Independent replication (mehek, RTX 6000 Ada, a DIFFERENT corpus --
#   ws_safe semantic_sam3):
#       b1 6.930 GiB, b2 8.238, b4 12.811 GiB reserved
#       marginals 1.308 / 2.287 GiB (also CONVEX)
#       max secant from n=1: 1.960 GiB
#
# The b1->b2 marginal agrees to three decimals across different corpora AND
# different GPU architectures (1.309 vs 1.308 GiB) -- the strongest single
# datum here.
#
# 2 GiB = the peer's max secant (1.660) + 20%. That headroom is not a generic
# safety margin: it is almost exactly consumed by the cross-corpus spread
# already observed (1.66 -> 1.96 GiB, +18%), and at batch 4 on the
# replication corpus the 2 GiB model clears the measurement by under 1%.
# Read this as validated on two corpora, with essentially no headroom
# remaining for a third. What would overturn it: a third corpus whose max
# secant from n=1 exceeds 2 GiB.
#
# Limits of this evidence: one allocator config (expandable_segments,
# enforced by `sam3_env_environ`), bf16, rank 16, the 312-adapter surface,
# two corpora, two cards. Every point above is a truncated probe and
# therefore a LOWER BOUND on the true peak.
#
# DO NOT set this from any single machine's numbers -- that is still the
# rule. The reason this value now stands is that it rests on two independent
# measurements across corpora and architectures, not that measuring from one
# box became acceptable.
_EXTRA_BATCH_DEVICE_BYTES = 2 * GiB
_DEVICE_STEADY_BYTES = 8 * GiB
_MASK_DEVICE_BYTES_PER_PIXEL = 16
_MASK_HOST_BYTES_PER_PIXEL = 5
_MAX_COCO_METADATA_BYTES = 16 * MiB
_MAX_JSON_DEPTH = 24
_MAX_JSON_VALUES = 2_000_000
_MAX_ESTIMATED_PARSED_BYTES = 96 * MiB
_MAX_NEGATIVE_QUERIES_PER_TILE = SAM3_MAX_NEGATIVE_QUERIES_PER_TILE
_MAX_NEGATIVE_PROMPT_COUNT = SAM3_MAX_NEGATIVE_PROMPT_COUNT
_MAX_NEGATIVE_PROMPT_BYTES = SAM3_MAX_NEGATIVE_PROMPT_BYTES
_OVER_LIMIT_NEGATIVE_PROMPTS: tuple[None, ...] = (None,) * (
    _MAX_NEGATIVE_PROMPT_COUNT + 1
)
# Optimizer admission budgets 16 bytes/parameter (weight, grad, Adam m/v).
_LORA_CPU_TRAINING_BYTES_PER_PARAM = 16
_LORA_RELOAD_BYTES_PER_PARAM = 4
_LORA_SERIALIZATION_BYTES_PER_PARAM = 8
_DEFAULT_HOST_RESERVE_GB = 8.0
_DEFAULT_HOST_RESERVE_FRACTION = 0.15
_MINIMUM_HOST_RESERVE_BYTES = int(_DEFAULT_HOST_RESERVE_GB * GiB)
_MINIMUM_HOST_RESERVE_FRACTION = _DEFAULT_HOST_RESERVE_FRACTION
_DEFAULT_CUDA_SAFETY_FRACTION = 0.85
_MAXIMUM_CUDA_SAFETY_FRACTION = 0.90
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
    mig_mode: str = "Disabled"


@dataclass(frozen=True)
class Sam3DatasetProfile:
    """Lightweight COCO facts used by the estimator."""

    train_tiles: int
    validation_tiles: int
    validation_present: bool
    train_instances: int
    max_active_instances_per_tile: int
    max_decoded_tile_bytes: int
    metadata_bytes: int
    raw_metadata_bytes: int
    polygon_count: int
    polygon_vertices: int
    # Defaulted so `asdict` diagnostics and every existing construction site
    # stay valid. Consumed by the auto-batch workload fingerprint, which needs
    # to tell a sparse dataset from a dense one: the max alone is a single
    # outlier tile, the p95 is what most steps actually pay for.
    p95_active_instances_per_tile: int = 0

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
    # Compatibility value: free bytes observed on the run artifact filesystem.
    free_disk_bytes: int
    artifact_free_disk_bytes: int
    publish_free_disk_bytes: Optional[int]
    containment_soft_host_bytes: int
    containment_hard_host_bytes: int
    refusals: tuple[str, ...]
    warnings: tuple[str, ...]
    # Which case of `device_requirement_bytes` produced the device peak:
    # "analytic" (no records), "measured" (records decided at/below an
    # observed rung), "max_extrapolated" (past the rungs, so max() applied),
    # or "probe_floor". Defaulted so every construction site stays valid.
    device_peak_provenance: str = "analytic"
    device_peak_analytic_bytes: int = 0
    device_peak_measured_bytes: int = 0
    device_peak_measured_extrapolated: bool = False
    device_peak_fingerprint: str = ""

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
            "artifact_free_disk_bytes": self.artifact_free_disk_bytes,
            "publish_free_disk_bytes": self.publish_free_disk_bytes,
            "containment_soft_host_bytes": self.containment_soft_host_bytes,
            "containment_hard_host_bytes": self.containment_hard_host_bytes,
            "refusals": list(self.refusals),
            "warnings": list(self.warnings),
            "device_peak_provenance": self.device_peak_provenance,
            "device_peak_analytic_bytes": int(self.device_peak_analytic_bytes),
            "device_peak_measured_bytes": int(self.device_peak_measured_bytes),
            "device_peak_measured_extrapolated": bool(
                self.device_peak_measured_extrapolated
            ),
            "device_peak_fingerprint": self.device_peak_fingerprint,
        }


# Set by `hf auth login`, or supplied directly by either environment variable.
_HF_TOKEN_ENV_VARS = ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN")


def _huggingface_token_present() -> bool:  # seam for tests
    """True when a Hugging Face credential is available to this user.

    Deliberately a LOCAL check -- no network call. Preflight must stay a
    millisecond, offline-safe gate; reaching out to huggingface.co would make
    admission depend on the network and could hang a run before it starts.

    A present token is necessary but not sufficient: the account must also
    have accepted the gated `facebook/sam3` licence. The refusal message says
    so, because the two failures are indistinguishable from here.
    """
    for name in _HF_TOKEN_ENV_VARS:
        if os.environ.get(name, "").strip():
            return True
    hf_home = os.environ.get("HF_HOME", "").strip()
    root = Path(hf_home) if hf_home else Path.home() / ".cache" / "huggingface"
    try:
        token_file = root / "token"
        return token_file.is_file() and token_file.stat().st_size > 0
    except OSError:
        return False


def _free_disk_bytes(path: str) -> int:
    target = Path(path).expanduser().resolve()
    while not target.exists() and target != target.parent:
        target = target.parent
    return int(shutil.disk_usage(target).free)


def _validate_json_materialization_bound(raw: bytes, path: Path) -> None:
    """Bound JSON object amplification before calling the stdlib decoder."""

    stack: list[int] = []
    in_string = False
    escaped = False
    scalar_open = False
    string_bytes = 0
    value_count = 0
    container_count = 0
    closing_for = {ord("{"): ord("}"), ord("["): ord("]")}
    whitespace = {ord(" "), ord("\t"), ord("\r"), ord("\n")}
    punctuation = {ord(","), ord(":"), ord("}"), ord("]")}

    for byte in raw:
        if in_string:
            string_bytes += 1
            if escaped:
                escaped = False
            elif byte == ord("\\"):
                escaped = True
            elif byte == ord('"'):
                in_string = False
            continue
        if byte == ord('"'):
            in_string = True
            scalar_open = False
            value_count += 1
        elif byte in closing_for:
            stack.append(closing_for[byte])
            container_count += 1
            value_count += 1
            scalar_open = False
            if len(stack) > _MAX_JSON_DEPTH:
                raise ValueError(
                    f"COCO metadata {path} exceeds the JSON nesting cap "
                    f"of {_MAX_JSON_DEPTH}"
                )
        elif byte in {ord("}"), ord("]")}:
            if not stack or stack.pop() != byte:
                raise ValueError(f"Invalid COCO metadata structure in {path}")
            scalar_open = False
        elif byte in whitespace or byte in punctuation:
            scalar_open = False
        elif not scalar_open:
            value_count += 1
            scalar_open = True
        if value_count > _MAX_JSON_VALUES:
            raise ValueError(
                f"COCO metadata {path} exceeds the JSON cardinality cap "
                f"of {_MAX_JSON_VALUES} values"
            )

    if in_string or stack:
        raise ValueError(f"Invalid COCO metadata structure in {path}")
    estimated_parsed_bytes = (
        2 * len(raw) + 2 * string_bytes + 64 * value_count + 72 * container_count
    )
    if estimated_parsed_bytes > _MAX_ESTIMATED_PARSED_BYTES:
        raise ValueError(
            f"COCO metadata {path} exceeds the parsed-memory cap of "
            f"{_MAX_ESTIMATED_PARSED_BYTES} bytes"
        )


def _load_coco(path: Path) -> tuple[dict[str, Any], int]:
    try:
        size = path.stat().st_size
        if size > _MAX_COCO_METADATA_BYTES:
            raise ValueError(
                f"COCO metadata {path} is {size} bytes; the metadata-only "
                f"preflight cap is {_MAX_COCO_METADATA_BYTES} bytes"
            )
        with path.open("rb") as stream:
            raw = stream.read(_MAX_COCO_METADATA_BYTES + 1)
        if len(raw) > _MAX_COCO_METADATA_BYTES:
            raise ValueError(
                f"COCO metadata {path} grew beyond the metadata-only "
                f"preflight cap of {_MAX_COCO_METADATA_BYTES} bytes"
            )
        _validate_json_materialization_bound(raw, path)
        value = json.loads(raw)
    except FileNotFoundError:
        return {}, 0
    except OSError as exc:
        raise ValueError(f"Could not read COCO metadata {path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid COCO metadata {path}: {exc}") from exc
    except UnicodeDecodeError as exc:
        raise ValueError(f"Invalid UTF-8 COCO metadata {path}: {exc}") from exc
    return (value if isinstance(value, dict) else {}), len(raw)


def _active_polygon_count(annotation: object) -> int:
    if not isinstance(annotation, dict):
        return 0
    return len(validated_segmentation_polygons(annotation.get("segmentation")))


def _percentile(values: Any, fraction: float) -> int:
    """Nearest-rank percentile over per-tile instance counts.

    Nearest-rank rather than interpolated: these are small integer counts of
    instances, and an interpolated 6.4 is not a tile that exists.
    """

    ordered = sorted(int(value) for value in values)
    if not ordered:
        return 0
    index = max(0, min(len(ordered) - 1, math.ceil(fraction * len(ordered)) - 1))
    return ordered[index]


def dataset_profile(dataset_dir: str) -> Sam3DatasetProfile:
    """Public seam: the bounded COCO facts the estimator and selection share."""

    root = Path(dataset_dir).expanduser().resolve()
    train, train_bytes = _load_coco(root / "train" / "_annotations.coco.json")
    validation_path = root / "valid" / "_annotations.coco.json"
    validation_present = validation_path.exists()
    valid, valid_bytes = _load_coco(validation_path)
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
            polygon_vertices += sum(
                len(polygon)
                for polygon in validated_segmentation_polygons(
                    annotation.get("segmentation")
                )
            )
            # TWO CARRIED CAVEATS on this count, recorded rather than changed
            # (no behaviour change intended here):
            #
            # 1. It feeds the MIN_TRAIN_INSTANCES refusal below, and that gate
            #    has LOOSENED without anyone deciding to loosen it. Lowering
            #    MIN_RETAINED_AREA_FRAC from 0.5 to 0.25 means fewer tile
            #    fragments are demoted to `iscrowd`, so fewer rows are excluded
            #    here and the same dataset now reports MORE train_instances
            #    than it did before. The threshold is unchanged; what it
            #    measures is not.
            # 2. Excluding `iscrowd` rows is "correct by accident", not by
            #    design. It happens to align with the builder's use of iscrowd
            #    to mark heavily-clipped fragments that should not be counted
            #    as trainable instances -- but nothing enforces that meaning,
            #    and a future producer using iscrowd in its COCO sense
            #    (genuine crowd regions) would silently change this gate again.
            #    If either is revisited, re-derive the threshold from what the
            #    count now means instead of adjusting the count to fit it.
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
        validation_present=validation_present,
        train_instances=train_instances,
        max_active_instances_per_tile=max(by_image.values(), default=0),
        p95_active_instances_per_tile=_percentile(by_image.values(), 0.95),
        max_decoded_tile_bytes=decoded_bytes,
        metadata_bytes=metadata_bytes,
        raw_metadata_bytes=raw_metadata_bytes,
        polygon_count=polygon_count,
        polygon_vertices=polygon_vertices,
    )


# Historical private name, kept so existing callers and tests keep working.
_dataset_profile = dataset_profile


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


def _resolved_negative_prompts(dataset_dir: str, params: Any) -> tuple[object, ...]:
    """Mirror dataloader manifest-first negative-prompt resolution safely."""

    configured = getattr(params, "negative_prompts", None)
    if configured is None:
        configured = []
    if type(configured) in (list, tuple) and len(configured) > (
        _MAX_NEGATIVE_PROMPT_COUNT
    ):
        # Preserve an over-limit cardinality signal without traversing or
        # copying the adversarial collection.
        return _OVER_LIMIT_NEGATIVE_PROMPTS
    if int(getattr(params, "num_negatives", 0)) <= 0:
        # These prompts are not queried, but they are still serialized into
        # spec.json by the parent and therefore remain subject to metadata caps.
        return tuple(configured) if type(configured) in (list, tuple) else (configured,)
    manifest_path = Path(dataset_dir).expanduser().resolve() / "build_manifest.json"
    if manifest_path.exists():
        manifest, _size = _load_coco(manifest_path)
        negatives = manifest.get("negative_prompts") or []
        if type(negatives) is list and negatives:
            if len(negatives) > _MAX_NEGATIVE_PROMPT_COUNT:
                return _OVER_LIMIT_NEGATIVE_PROMPTS
            return tuple(negatives)
        if negatives:
            return (negatives,)
    return tuple(configured) if type(configured) in (list, tuple) else (configured,)


def _free_disk_gb(path: str) -> float:
    """Compatibility wrapper around the byte-precise disk probe."""

    return _free_disk_bytes(path) / GiB


def _parse_compute_capability(value: str) -> tuple[int, int]:
    major_text, separator, minor_text = value.strip().partition(".")
    if not separator:
        minor_text = "0"
    return int(major_text), int(minor_text)


def sam3_device_form_error(device: Any) -> Optional[str]:
    """The user-facing reason a device STRING cannot name a SAM3 GPU, or None.

    `_probe_cuda_device` deliberately collapses every failure to ``None``, so
    on its own it cannot tell "this box has no GPU" from "this spelling was
    refused". Callers that report a refusal ask this first: five of six SAM3
    runs on the CUDA box died with "No CUDA device is available" while 47 GiB
    sat free, purely because their device was the bare ordinal ``"0"``.
    """

    value = str(device or "auto").strip()
    normalized = normalize_cuda_device(value)
    if normalized in {"auto", "cuda"} or normalized.startswith("cuda:"):
        return None
    return (
        f"SAM3 LoRA training requires a CUDA device; got {value!r}. "
        f"Accepted: {CUDA_DEVICE_FORMS}."
    )


def _visible_device_selector(device: str) -> str:
    # Bare ordinals ("0", "0,1") are the Ultralytics convention that DetectKit
    # plans carry, and the same plan schema feeds a YOLO role and a SAM3 role.
    # Multi-GPU forms resolve against the FIRST device (SAM3 pins one physical
    # device by UUID); `assess_preflight` warns when that narrowing happens.
    device = normalize_cuda_device(device)
    form_error = sam3_device_form_error(device)
    if form_error is not None:
        raise ValueError(form_error)
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
    logical_index = 0
    if device.startswith("cuda:"):
        try:
            logical_index = int(device.partition(":")[2])
        except ValueError as exc:
            raise ValueError(f"Invalid CUDA device selection {device!r}") from exc
        if logical_index < 0:
            raise ValueError("CUDA device index must be non-negative")
    if visible:
        if logical_index >= len(visible):
            raise ValueError("requested CUDA device is outside CUDA_VISIBLE_DEVICES")
        return visible[logical_index]
    if device.startswith("cuda:"):
        return str(logical_index)
    return "0"


def _probe_cuda_device(device: str = "auto") -> Optional[CudaDeviceObservation]:
    """Probe one visible physical device without importing torch."""

    try:
        selector = _visible_device_selector(device)
    except (ValueError, IndexError):
        return None
    # MIG needs slice-aware memory telemetry and a parent-GPU lease strategy;
    # accepting it as a full physical GPU would overstate capacity.
    if selector.upper().startswith("MIG-"):
        return None
    numeric_selector = selector.isdecimal()
    visible_tokens = [
        item.strip()
        for item in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")
        if item.strip()
    ]

    command = [
        "nvidia-smi",
        "--query-gpu=index,uuid,pci.bus_id,name,compute_cap,memory.total,memory.free,mig.mode.current",
        "--format=csv,noheader,nounits",
    ]
    try:
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=_PROBE_TIMEOUT_SECONDS,
            env={**os.environ, "CUDA_DEVICE_ORDER": "PCI_BUS_ID"},
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if completed.returncode != 0:
        return None
    rows = list(csv.reader(completed.stdout.splitlines(), skipinitialspace=True))
    valid_rows = [row for row in rows if len(row) == 8]
    if numeric_selector and visible_tokens:
        if os.environ.get("CUDA_DEVICE_ORDER", "").upper() != "PCI_BUS_ID":
            return None
        physical_ordinal = int(selector)
        valid_rows.sort(key=lambda row: row[2].strip().upper())
        if physical_ordinal >= len(valid_rows):
            return None
        candidate_rows = [valid_rows[physical_ordinal]]
    elif numeric_selector:
        physical_ordinal = int(selector)
        valid_rows.sort(key=lambda row: row[2].strip().upper())
        if physical_ordinal >= len(valid_rows):
            return None
        candidate_rows = [valid_rows[physical_ordinal]]
    else:
        # CUDA permits a unique leading GPU UUID abbreviation. Resolve it to
        # the full physical identity used by leases; ambiguity must fail closed.
        candidate_rows = [
            row for row in valid_rows if row[1].strip().startswith(selector)
        ]
        if len(candidate_rows) != 1:
            return None
    for row in candidate_rows:
        if len(row) != 8:
            continue
        index, uuid, pci_bus_id, name, capability, total_mib, free_mib, mig_mode = (
            item.strip() for item in row
        )
        if mig_mode.lower() not in {"disabled", "n/a", "[n/a]", "not supported"}:
            return None
        try:
            return CudaDeviceObservation(
                index=int(index),
                uuid=uuid,
                pci_bus_id=pci_bus_id,
                name=name,
                compute_capability=_parse_compute_capability(capability),
                free_bytes=int(float(free_mib) * MiB),
                total_bytes=int(float(total_mib) * MiB),
                mig_mode=mig_mode,
            )
        except ValueError:
            return None
    return None


def _resource_policy(params: Any) -> ResourcePolicy:
    return ResourcePolicy(
        reserve_host_bytes=max(
            _MINIMUM_HOST_RESERVE_BYTES,
            int(
                float(getattr(params, "host_reserve_gb", _DEFAULT_HOST_RESERVE_GB))
                * GiB
            ),
        ),
        reserve_host_fraction=max(
            _MINIMUM_HOST_RESERVE_FRACTION,
            float(
                getattr(
                    params,
                    "host_reserve_fraction",
                    _DEFAULT_HOST_RESERVE_FRACTION,
                )
            ),
        ),
        accelerator_safety_fraction=min(
            _MAXIMUM_CUDA_SAFETY_FRACTION,
            float(
                getattr(params, "cuda_safety_fraction", _DEFAULT_CUDA_SAFETY_FRACTION)
            ),
        ),
        warning_fraction=0.80,
    )


def _utf8_size(value: str) -> int:
    """Count UTF-8 bytes without allocating a second potentially large string."""

    total = 0
    for codepoint in map(ord, value):
        if 0xD800 <= codepoint <= 0xDFFF:
            # JSON may materialize lone surrogate escapes that cannot be
            # serialized or tokenized as UTF-8. Count them over the cap so
            # admission fails closed with the normal prompt diagnostic.
            return SAM3_MAX_CONFIGURED_PROMPT_BYTES + 1
        total += 1 + (codepoint >= 0x80) + (codepoint >= 0x800) + (codepoint >= 0x10000)
    return total


def containment_host_limits(params: Any, host_peak_bytes: int) -> tuple[int, int]:
    """Return the exact soft/hard host limits that admission must reserve."""

    soft = max(1, math.ceil(host_peak_bytes * 1.10))
    hard = max(
        soft,
        math.ceil(
            host_peak_bytes
            * float(getattr(params, "host_limit_headroom_fraction", 1.25))
        ),
    )
    return soft, hard


def _analytic_training_device_peak(
    *,
    precision_multiplier: float,
    batch_size: int,
    lora_training_state: int,
    default_lora_training_state: int,
    dense_masks_device: int,
) -> int:
    """The composed analytic device estimate, unchanged in form.

    Base envelope + per-batch term + LoRA adjustment + dense masks, each
    scaled where precision scales it. Extracted verbatim from
    `build_resource_request` so exactly one expression exists and both the
    admission gate and the batch-selection path can consult it.
    """

    return int(
        _MEASURED_BF16_DEVICE_PEAK_BYTES * precision_multiplier
        + max(0, batch_size - 1) * _EXTRA_BATCH_DEVICE_BYTES * precision_multiplier
        + max(0, lora_training_state - default_lora_training_state)
        + dense_masks_device
    )


def _lora_params_per_rank(params: Any) -> int:
    return sum(
        count
        for flag, count in _LORA_PARAMS_PER_RANK.items()
        if bool(getattr(params, flag, False))
    )


def _default_lora_training_state() -> int:
    return (
        16
        * (
            _LORA_PARAMS_PER_RANK["adapt_vision_encoder"]
            + _LORA_PARAMS_PER_RANK["adapt_geometry_encoder"]
            + _LORA_PARAMS_PER_RANK["adapt_detr_encoder"]
            + _LORA_PARAMS_PER_RANK["adapt_detr_decoder"]
        )
        * 24
    )


def analytic_device_peak_bytes(
    params: Any,
    dataset: Sam3DatasetProfile,
    *,
    batch_size: int | None = None,
) -> int:
    """The composed analytic device estimate for one (params, dataset, batch).

    Shares `_analytic_training_device_peak` with `build_resource_request`, so
    admission and batch selection read the same expression.
    """

    batch = max(1, int(batch_size if batch_size is not None else params.batch))
    image_pixels = PREDICTOR_IMGSZ * PREDICTOR_IMGSZ
    inflight_tiles = min(batch, max(1, dataset.tile_count))
    active_instances = inflight_tiles * dataset.max_active_instances_per_tile
    bounded_rank = min(_MAX_LORA_RANK, max(1, int(getattr(params, "rank", 1))))
    lora_training_state = bounded_rank * _lora_params_per_rank(params) * 24
    return _analytic_training_device_peak(
        precision_multiplier=(
            _FP32_DEVICE_PEAK_MULTIPLIER
            if getattr(params, "mixed_precision", "bf16") == "fp32"
            else 1.0
        ),
        batch_size=batch,
        lora_training_state=lora_training_state,
        default_lora_training_state=_default_lora_training_state(),
        dense_masks_device=active_instances
        * image_pixels
        * _MASK_DEVICE_BYTES_PER_PIXEL,
    )


class DeviceRequirement(NamedTuple):
    """What this configuration needs, and what decided it."""

    bytes: int
    provenance: str
    measured_bytes: int
    measured_extrapolated: bool
    decided_by_measurement: bool = False


def device_requirement_bytes(
    analytic_bytes: int,
    records: Sequence[MemoryMeasurement],
    batch_size: int,
) -> DeviceRequirement:
    """The single authority on device demand.

    Three cases, and `provenance` names which one applied:

    ==========================  =====================  ===================
    situation                   requirement            provenance
    ==========================  =====================  ===================
    no matching records         analytic estimate      ``"analytic"``
    records, batch <= top rung  measured envelope      ``"measured"``
    records, batch > top rung   max(analytic, meas.)   ``"max_extrapolated"``
    ==========================  =====================  ===================

    `measured_extrapolated` remains a SEPARATE fact -- was the measured side
    evaluated past the largest observed rung -- and is true exactly on the
    third row. `decided_by_measurement` says whether the number returned came
    off the measured side, which is what the measured-safety headroom check
    keys on; the provenance string is for humans and telemetry.

    WHY A MEASUREMENT MAY NOW DECIDE (it previously could only RAISE the
    analytic estimate). The original rule was empirical: on mehek the SAME
    surface reserved 7.34 GiB after a 2-step probe, 9.93 GiB after 60, and
    12.99 GiB by step 320 of a full run, so the probe was a lower bound and
    trusting it would have admitted a run that OOMs minutes in. That growth
    was later isolated to ALLOCATOR FRAGMENTATION, not the workload: reserved
    grew 35% while allocated grew 2%. Under
    `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` it does not happen --
    reserved is flat 6.89 -> 6.91 GiB from step 2 to 120, and a full 2430-step
    run stayed flat across eight epoch boundaries with the
    reserved-minus-allocated gap constant at ~0.49 GiB.

    Three things had to become true together, and all three now are:

    1. `sam3_env_environ()` HARDCODES that flag, so every probe child and
       every training child runs under it.
    2. The allocator config is hashed into the profile fingerprint's
       `backend` (`autobatch.sidecar_alloc_conf_hash`), so a record taken
       under the default allocator cannot match an expandable run. Stale
       records cannot decide; they are not even returned.
    3. `PROBE_STEPS` is 30, which reaches the -2.82% truncation plateau (see
       that constant), well inside the 0.80/0.85 safety fractions applied
       downstream.

    THE EXTRAPOLATION GUARD IS NOT NEGOTIABLE. Past the largest observed
    rung, `measured_envelope_bytes` is a linear FIT -- a guess, not an
    observation -- and records at n=1,2 must never authorise a measured-only
    verdict at n=8. There the old `max(analytic, measured)` still applies, and
    the analytic estimate is retained (never deleted) as the fallback for
    every workload that has never been probed.

    No `precision_multiplier` is ever applied to a measurement: precision is
    part of the profile fingerprint, so a record only matches a run of the
    same precision.
    """

    analytic = int(analytic_bytes)
    if not records:
        return DeviceRequirement(analytic, "analytic", 0, False, False)
    rungs = sorted(record.settings.batch_size for record in records)
    measured = measured_envelope_bytes(records, batch_size)
    extrapolated = batch_size > rungs[-1]
    if not extrapolated:
        # At or below the top rung the envelope is anchored by real
        # observations, so it decides -- upward OR downward.
        #
        # BETWEEN two rungs (n=3 with rungs 1, 2, 4) the envelope's fitted
        # term is still a guess, and `select_batch` scans contiguously so n=3
        # IS reachable on the auto path. There the requirement is raised to
        # the OBSERVED peak at the next rung ABOVE n, a guaranteed-safe upper
        # bound by monotonicity (true_need(3) <= peak@4). A batch that IS a
        # rung is an observation and is charged its own envelope, never the
        # next rung's -- otherwise every batch below the top would be priced
        # at the top rung. It costs nothing in
        # practice: if peak@4 fits, 4 would have been chosen anyway, and if it
        # does not, picking 3 on an unvalidated fit is exactly the guess this
        # excludes. `max` rather than a plain substitution so a steep fit can
        # still raise, never lower, the number.
        upper = (
            0
            if batch_size in rungs
            else next(
                (
                    max(
                        record.accelerator_reserved_peak_bytes
                        for record in records
                        if record.settings.batch_size == rung
                    )
                    for rung in rungs
                    if rung > batch_size
                ),
                0,
            )
        )
        requirement = max(measured, upper)
        return DeviceRequirement(requirement, "measured", requirement, False, True)
    if measured > analytic:
        return DeviceRequirement(measured, "max_extrapolated", measured, True, True)
    return DeviceRequirement(analytic, "max_extrapolated", measured, True, False)


def profile_fingerprint_key(identity: ProfileIdentity) -> str:
    """A short, stable, log-safe digest of a `ProfileIdentity`."""

    payload = json.dumps(asdict(identity), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _profile_records() -> tuple[MemoryMeasurement, ...]:
    """Every stored SAM3 memory profile (seam for tests).

    Offline and millisecond-fast by construction: one bounded local JSON
    read, never a probe. Any failure degrades to "no measurement", which
    falls back to the analytic estimate.
    """

    from .autobatch import PROFILE_SCOPE

    try:
        path = profile_store_path(PROFILE_SCOPE)
        if not path.exists():
            return ()
        return MemoryProfileStore(path).load()
    except (OSError, ValueError):
        return ()


def _workload_identity(
    spec: Any,
    cuda_device: Optional[CudaDeviceObservation],
) -> Optional[ProfileIdentity]:
    """The profile fingerprint for `spec` on `cuda_device` (seam for tests)."""

    if cuda_device is None:
        return None
    try:
        from . import autobatch

        dataset = autobatch.sam3_dataset_density_profile(spec)
        return autobatch.sam3_workload_fingerprint(
            spec, cuda_device=cuda_device, dataset=dataset
        ).identity
    except (OSError, ValueError, KeyError, TypeError, subprocess.SubprocessError):
        # Environment/IO failures mean "no measurement matched", which falls
        # back to the analytic estimate. Deliberately NOT a bare `except`: a
        # silent lookup failure lowers the requirement back to analytic while
        # a record that should have won sits in the store, so a programming
        # error here must surface rather than degrade the gate quietly.
        return None


def _stored_records(
    spec: Any,
    cuda_device: Optional[CudaDeviceObservation],
) -> tuple[MemoryMeasurement, ...]:
    """Validated stored measurements for this exact workload, or `()`.

    The store is read FIRST: with no store there is nothing to match and the
    fingerprint (which hashes an environment) is never computed, so the
    common case stays fast.
    """

    stored = _profile_records()
    if not stored:
        return ()
    identity = _workload_identity(spec, cuda_device)
    if identity is None:
        return ()
    from .autobatch import validate_probe_records

    return validate_probe_records(records_for(stored, identity), identity)


def build_resource_request(
    spec: Any,
    dataset: Sam3DatasetProfile,
    *,
    params: Any | None = None,
    negative_prompts: tuple[str, ...] = (),
    probe_floor: bool = False,
    measured_records: Sequence[MemoryMeasurement] = (),
) -> ResourceRequest:
    """Estimate phase peaks from streamed work limits and COCO metadata.

    `probe_floor=True` builds the estimate a *measurement probe* is admitted
    against instead of the one a training run is. The difference is confined
    to the device side: the inherited `_MEASURED_BF16_DEVICE_PEAK_BYTES` /
    `_EXTRA_BATCH_DEVICE_BYTES` / `_DEVICE_STEADY_BYTES` envelope is exactly
    what the probe exists to replace, so gating the probe on it would refuse
    hardware whose real batch-1 peak fits before it was ever measured.
    Instead the device requirement becomes a hard floor computed from the
    checkpoint size, the LoRA state, and ONE tile -- an under-estimate by
    construction. Beyond that floor the probe child's own contained OOM is
    the gate, and an OOM inside a contained probe is a measurement, not a
    failure.

    The host side is deliberately UNCHANGED in form and still scales with
    `batch_size`, so each candidate is admitted at its own batch.
    """

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
    params_per_rank = _lora_params_per_rank(params)
    requested_rank = int(params.rank)
    bounded_rank = min(_MAX_LORA_RANK, max(1, requested_rank))
    lora_params = bounded_rank * params_per_rank
    lora_training_state = lora_params * 24
    lora_artifact = lora_params * 4
    lora_cpu_training_state = lora_params * _LORA_CPU_TRAINING_BYTES_PER_PARAM
    lora_reload_copy = lora_params * _LORA_RELOAD_BYTES_PER_PARAM
    lora_serialization_copies = lora_params * _LORA_SERIALIZATION_BYTES_PER_PARAM
    requested_negatives = max(0, int(getattr(params, "num_negatives", 0)))
    selected_negatives = min(
        requested_negatives,
        len(negative_prompts),
        _MAX_NEGATIVE_QUERIES_PER_TILE,
    )
    prompt_pool_bytes = sum(_utf8_size(prompt) for prompt in negative_prompts)
    configured_prompt = getattr(params, "prompt", "")
    configured_prompt_bytes = _utf8_size(
        configured_prompt if type(configured_prompt) is str else ""
    )
    configured_negatives = getattr(params, "negative_prompts", ())
    if configured_negatives is None:
        configured_negatives = ()
    if (
        type(configured_negatives) in (list, tuple)
        and len(configured_negatives) <= _MAX_NEGATIVE_PROMPT_COUNT
    ):
        configured_prompt_bytes += sum(
            _utf8_size(prompt) for prompt in configured_negatives if type(prompt) is str
        )
    descriptor_query_bytes = dataset.tile_count * (48 + 8 * selected_negatives)
    metadata = (
        dataset.metadata_bytes
        + configured_prompt_bytes
        + prompt_pool_bytes
        + descriptor_query_bytes
    )

    training_host_dynamic = (
        metadata
        + decoded_tiles
        + transformed_tiles
        + collated_images
        + dense_masks_host
        + lora_cpu_training_state
    )
    training_host_peak = _TRAIN_HOST_FIXED_BYTES + training_host_dynamic
    precision_multiplier = (
        _FP32_DEVICE_PEAK_MULTIPLIER
        if getattr(params, "mixed_precision", "bf16") == "fp32"
        else 1.0
    )
    one_tile_device_bytes = image_pixels * 3 * 4 + (
        dataset.max_active_instances_per_tile
        * image_pixels
        * _MASK_DEVICE_BYTES_PER_PIXEL
    )
    if probe_floor:
        # NEVER consult a measurement here: this floor is a deliberate
        # under-estimate so the probe that PRODUCES the measurement can run.
        training_device_peak = int(
            _CHECKPOINT_BYTES
            + lora_training_state
            + one_tile_device_bytes * precision_multiplier
        )
    else:
        # ONE derivation of the analytic estimate, shared with the
        # provenance stamp and with batch selection. Duplicating the term
        # here would let `budget.accelerator_peak_bytes` and
        # `device_peak_analytic_bytes` drift apart silently.
        training_device_peak = device_requirement_bytes(
            analytic_device_peak_bytes(params, dataset, batch_size=batch_size),
            tuple(measured_records),
            batch_size,
        ).bytes
    training_device_peak = int(training_device_peak)
    device_steady_bytes = _CHECKPOINT_BYTES if probe_floor else _DEVICE_STEADY_BYTES

    device_envelope_label = (
        "probe hard floor (checkpoint + LoRA state + one tile)"
        if probe_floor
        else "measured BF16 model/activation envelope"
    )
    # `PhaseEstimate` rejects a peak below its own steady floor, and rightly
    # so. The analytic estimate can never go below the steady constants, but a
    # MEASURED envelope now decides on its own and legitimately can: a real
    # SAM3 LoRA run reserved ~6.9 GiB total, under the 8 GiB
    # `_DEVICE_STEADY_BYTES` guess for resident weights alone. The measurement
    # is the authority on the whole device envelope, so the analytic steady
    # guess yields to it -- never the other way round, which would silently
    # inflate every measured requirement back to the constant it replaced.
    #
    # Computed ONCE and shared by the training and validation phases. Clamping
    # only the training phase left `validation` (built only when the dataset
    # has a valid split, which is why no test caught it) raising `ValueError`
    # out of `PhaseEstimate` for every sub-8-GiB measurement -- an unhandled
    # traceback out of the launch instead of a refusal, on the ordinary case.
    on_device_steady_bytes = min(
        device_steady_bytes + lora_training_state, training_device_peak
    )
    device_envelope_bytes = (
        training_device_peak if probe_floor else _MEASURED_BF16_DEVICE_PEAK_BYTES
    )
    validation_device_peak = training_device_peak
    common_allocations = (
        ("descriptor metadata", metadata),
        ("configured prompt serialization", configured_prompt_bytes),
        ("negative query descriptors", descriptor_query_bytes + prompt_pool_bytes),
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
            accelerator_steady_bytes=device_steady_bytes,
            accelerator_peak_bytes=device_steady_bytes,
            dominant_allocations=(
                ("base checkpoint", _CHECKPOINT_BYTES),
                ("model construction", _CHECKPOINT_BYTES),
                ("descriptor metadata", metadata),
            ),
        ),
        PhaseEstimate(
            "training",
            host_steady_bytes=(
                _TRAIN_HOST_FIXED_BYTES + metadata + lora_cpu_training_state
            ),
            host_peak_bytes=training_host_peak,
            accelerator_steady_bytes=on_device_steady_bytes,
            accelerator_peak_bytes=training_device_peak,
            disk_transient_bytes=2 * lora_artifact,
            dominant_allocations=common_allocations
            + (
                (device_envelope_label, device_envelope_bytes),
                ("LoRA and optimizer state", lora_training_state),
                ("CPU LoRA training state", lora_cpu_training_state),
            ),
        ),
    )
    if dataset.validation_tiles:
        phases += (
            PhaseEstimate(
                "validation",
                host_steady_bytes=(
                    _TRAIN_HOST_FIXED_BYTES + metadata + lora_cpu_training_state
                ),
                host_peak_bytes=training_host_peak + lora_reload_copy,
                accelerator_steady_bytes=on_device_steady_bytes,
                accelerator_peak_bytes=validation_device_peak,
                dominant_allocations=common_allocations
                + (
                    (device_envelope_label, device_envelope_bytes),
                    ("CPU LoRA training state", lora_cpu_training_state),
                    ("LoRA reload copy", lora_reload_copy),
                ),
            ),
        )
    publish_policy = getattr(spec, "publish_policy", None)
    if publish_policy is None or bool(getattr(publish_policy, "auto_import", True)):
        phases += (
            PhaseEstimate(
                "publish",
                host_steady_bytes=_CHECKPOINT_BYTES + lora_artifact + metadata,
                host_peak_bytes=(
                    _PUBLISH_HOST_BYTES
                    + lora_artifact
                    + lora_reload_copy
                    + lora_serialization_copies
                    + metadata
                ),
                disk_transient_bytes=_PUBLISH_DISK_BYTES,
                dominant_allocations=(
                    ("base checkpoint", _CHECKPOINT_BYTES),
                    ("largest possible active tensor", _PUBLISH_ACTIVE_TENSOR_BYTES),
                    ("Torch/serialization runtime", _RUNTIME_HOST_ALLOWANCE_BYTES),
                    ("LoRA adapter", lora_artifact),
                    ("LoRA reload copy", lora_reload_copy),
                    ("LoRA serialization copies", lora_serialization_copies),
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
    run_dir: str | Path | None = None,
    models_root: str | Path | None = None,
    probe_floor: bool = False,
) -> Sam3PreflightDecision:
    """Return the complete initial or lease-held live admission decision.

    `probe_floor` swaps ONLY the device estimate for the measurement-probe
    hard floor (see `build_resource_request`). Every other admission concern
    -- leases, run-dir writability, dataset validity, prompts, publish policy,
    credentials, containment limits -- is evaluated identically, because they
    are the same concerns for a probe as for a training run. Callers should
    use `assess_probe_preflight` rather than passing this directly.
    """

    params = getattr(spec, "sam3_params", None)
    if params is None:
        # Construct a harmless estimate so diagnostics remain typed.
        params = type(
            "MissingParams", (), {"batch": 1, "num_negatives": 0, "rank": 1}
        )()
    # Resolve type and cardinality in O(1) before metadata profiling or prompt
    # traversal. Direct callers need the same bounded admission behavior as the
    # orchestrator even when a manifest would otherwise take precedence.
    configured_negatives_at_entry = getattr(params, "negative_prompts", ())
    configured_pool_shape_error = type(configured_negatives_at_entry) not in (
        list,
        tuple,
    )
    configured_pool_over_limit = (
        not configured_pool_shape_error
        and len(configured_negatives_at_entry) > _MAX_NEGATIVE_PROMPT_COUNT
    )
    dataset = dataset or dataset_profile(spec.derived_dataset_dir)
    resolved_negative_prompts = _resolved_negative_prompts(
        spec.derived_dataset_dir, params
    )
    valid_negative_prompts = tuple(
        prompt
        for prompt in resolved_negative_prompts
        if type(prompt) is str and bool(prompt.strip())
    )
    if cuda_device is _UNSET:
        cuda_device = _probe_cuda_device(str(getattr(spec, "device", "auto")))
    assert cuda_device is None or isinstance(cuda_device, CudaDeviceObservation)
    # The measurement lookup needs the physical device identity, so it must
    # follow the device probe. The probe-floor estimate deliberately ignores
    # measurements (see `build_resource_request`).
    measured_records = () if probe_floor else _stored_records(spec, cuda_device)
    request = build_resource_request(
        spec,
        dataset,
        params=params,
        negative_prompts=valid_negative_prompts,
        probe_floor=probe_floor,
        measured_records=measured_records,
    )
    if probe_floor:
        device_peak_provenance = "probe_floor"
        analytic_device_bytes = 0
        measured_device_bytes = 0
        measured_extrapolated = False
        device_peak_decided_by_measurement = False
        device_peak_requirement_bytes = 0
    else:
        analytic_device_bytes = analytic_device_peak_bytes(params, dataset)
        requirement = device_requirement_bytes(
            analytic_device_bytes,
            measured_records,
            max(1, int(getattr(params, "batch", 1))),
        )
        device_peak_provenance = requirement.provenance
        measured_device_bytes = requirement.measured_bytes
        measured_extrapolated = requirement.measured_extrapolated
        device_peak_decided_by_measurement = requirement.decided_by_measurement
        device_peak_requirement_bytes = requirement.bytes
    device_peak_fingerprint = (
        profile_fingerprint_key(measured_records[0].identity)
        if measured_records
        else ""
    )
    observation = observation or _observe_resources(cuda_device)
    policy = _resource_policy(params)
    budget = evaluate_resource_request(request, observation, policy)
    containment_soft_host_bytes, containment_hard_host_bytes = containment_host_limits(
        params, budget.host_peak_bytes
    )
    artifact_target = run_dir if run_dir is not None else spec.derived_dataset_dir
    artifact_free_disk = _free_disk_bytes(str(artifact_target))
    publish_phase = next(
        (phase for phase in request.phases if phase.name == "publish"), None
    )
    publish_target = models_root if models_root is not None else artifact_target
    if publish_phase is None:
        publish_free_disk = None
    elif str(publish_target) == str(artifact_target):
        publish_free_disk = artifact_free_disk
    else:
        publish_free_disk = _free_disk_bytes(str(publish_target))
    refusals = list(budget.refusals)
    warnings = list(budget.warnings)

    if names_several_devices(getattr(spec, "device", "auto")):
        # SAM3 pins ONE physical device by UUID, so a multi-GPU string is
        # narrowed rather than honoured. Say so in the durable record.
        warnings.append(
            f"device {str(spec.device)!r} names several GPUs; SAM3 pins a single "
            f"device and resolves it against {normalize_cuda_device(spec.device)} "
            "alone rather than overstating the capacity of the set."
        )

    if device_peak_decided_by_measurement:
        # A measurement is a raw envelope with no margin, so the measured
        # path keeps headroom explicitly. Applied ONCE, against raw free
        # device bytes -- never against `budget.usable_accelerator_bytes`,
        # which the policy has already discounted.
        #
        # Keyed on "did the measured side produce this number", not on the
        # provenance STRING: under `max_extrapolated` the measured side can
        # still win, and that number needs the same headroom it would have
        # needed under `measured`.
        # Gated on the REQUIREMENT, not on `budget.accelerator_peak_bytes`.
        # The budget peak is a max over phases and the `model_load` phase
        # carries its own analytic `_DEVICE_STEADY_BYTES` floor, which a
        # measurement can legitimately sit below (a real run reserved
        # ~6.9 GiB). Gating on the budget peak would apply this 0.80 fraction
        # to an analytic constant and then print it as "the measured device
        # requirement", which is simply false. That floor stays in the budget
        # -- `run_probe_measurement` calls `reset_peak_memory_stats` AFTER
        # `_build_model_and_loss`, so the model-load transient is genuinely
        # unmeasured and the analytic floor is the only number covering it --
        # but it is the policy's own 0.85 admission that gates it, not this.
        free_device_bytes = observation.available_accelerator_bytes
        if (
            free_device_bytes is not None
            and free_device_bytes * MEASURED_SAFETY_FRACTION
            < device_peak_requirement_bytes
        ):
            refusals.append(
                "The measured device requirement for this workload is "
                f"{device_peak_requirement_bytes / GiB:.1f} GiB"
                f"{' (extrapolated past the last measured batch)' if measured_extrapolated else ''}"
                ", but only "
                f"{free_device_bytes / GiB:.1f} GiB is free on the selected "
                f"GPU (usable at {MEASURED_SAFETY_FRACTION:.0%} safety)."
            )

    if containment_hard_host_bytes > budget.usable_host_bytes:
        refusals.append(
            "SAM3's hard containment limit, including configured headroom, "
            f"would be {containment_hard_host_bytes / GiB:.1f} GiB, but only "
            f"{budget.usable_host_bytes / GiB:.1f} GiB is usable after the "
            "host reserve. Reduce the workload or headroom; the reserve will "
            "not be exposed to MemoryMax/RLIMIT_AS."
        )

    if getattr(spec, "sam3_params", None) is None:
        refusals.append("SAM3 training requires a sam3_params configuration.")

    if cuda_device is None:
        refusals.append(
            sam3_device_form_error(getattr(spec, "device", "auto"))
            or "No CUDA device is available; SAM3 LoRA training requires CUDA."
        )
    elif cuda_device.compute_capability[0] < 8:
        major, minor = cuda_device.compute_capability
        refusals.append(
            "SAM3 LoRA training requires CUDA BF16 on compute capability >= 8.0; "
            f"the selected GPU reports {major}.{minor}. FP32 fallback is unsafe and disabled."
        )
    if not _huggingface_token_present():
        # `build_sam3_image_model` fetches the model config from the GATED
        # facebook/sam3 repo at build time, so a local checkpoint is not
        # sufficient -- every training machine needs its own credential.
        # Without this check the run dies minutes in, inside a sidecar
        # subprocess, as a bare `GatedRepoError: 401` with no hint about what
        # to do; observed on a freshly provisioned box.
        refusals.append(
            "No Hugging Face credential found. SAM3 training builds the model "
            "from the gated 'facebook/sam3' repo, which a local checkpoint "
            "does not replace. Run `hf auth login` on this machine (or set "
            f"{' / '.join(_HF_TOKEN_ENV_VARS)}), and make sure that account "
            "has accepted the licence at https://huggingface.co/facebook/sam3."
        )

    if getattr(params, "mixed_precision", None) not in SUPPORTED_PRECISIONS:
        # FP32 is supported again. The original refusal blamed "SAM3's BF16
        # activation path", which was `perflib.fused.addmm_act` -- a kernel
        # that hard-casts to bfloat16 AND refuses to run with grad enabled.
        # `perflib_compat` now replaces it with an eager, dtype-neutral
        # equivalent before the model is built, so nothing in the training
        # path requires bf16. FP16 stays out: its narrow range genuinely does
        # overflow SAM3's loss scales, and nothing here provides a GradScaler.
        refusals.append(
            "SAM3 LoRA training supports "
            f"{' or '.join(SUPPORTED_PRECISIONS)}; "
            f"{getattr(params, 'mixed_precision', None)!r} is not available."
        )
    params_per_rank = sum(
        count
        for flag, count in _LORA_PARAMS_PER_RANK.items()
        if bool(getattr(params, flag, False))
    )
    if params_per_rank <= 0:
        refusals.append(
            "SAM3 training requires at least one effective adapter scope; the "
            "selected adapt_* flags introduce no trainable LoRA parameters."
        )
    requested_rank = int(getattr(params, "rank", 0))
    if not 1 <= requested_rank <= _MAX_LORA_RANK:
        refusals.append(
            f"LoRA rank must be between 1 and {_MAX_LORA_RANK}; got "
            f"{requested_rank}. The estimator clamps unsafe ranks."
        )
    elif requested_rank * params_per_rank > _MAX_LORA_TRAINABLE_PARAMS:
        refusals.append(
            "The selected LoRA rank and adapter scopes would introduce "
            f"{requested_rank * params_per_rank:,} parameters, above the "
            f"safe cap of {_MAX_LORA_TRAINABLE_PARAMS:,}."
        )
    raw_prompt = getattr(params, "prompt", "")
    prompt = raw_prompt if type(raw_prompt) is str else ""
    if not prompt.strip():
        refusals.append(
            "Prompt is empty; SAM3 requires a text prompt to train against."
        )
    prompt_error = sam3_prompt_text_error(raw_prompt)
    if prompt_error is not None:
        refusals.append(f"Prompt {prompt_error}.")
    configured_negatives = configured_negatives_at_entry
    configured_pool_is_safe = (
        not configured_pool_shape_error and not configured_pool_over_limit
    )
    if configured_pool_shape_error:
        refusals.append("Configured negative_prompts must be a list or tuple.")
    elif configured_pool_over_limit:
        refusals.append(
            "Configured negative prompts exceed the safe metadata cap of "
            f"{_MAX_NEGATIVE_PROMPT_COUNT} entries."
        )
    if configured_pool_is_safe:
        for index, negative_prompt in enumerate(configured_negatives):
            prompt_error = sam3_prompt_text_error(negative_prompt)
            if prompt_error is not None:
                refusals.append(f"Configured negative prompt {index} {prompt_error}.")
    configured_prompt_bytes = _utf8_size(prompt)
    if configured_pool_is_safe:
        configured_prompt_bytes += sum(
            _utf8_size(value) for value in configured_negatives if type(value) is str
        )
    else:
        configured_prompt_bytes = SAM3_MAX_CONFIGURED_PROMPT_BYTES + 1
    if configured_prompt_bytes > SAM3_MAX_CONFIGURED_PROMPT_BYTES:
        refusals.append(
            "Configured prompt text exceeds the safe serialized metadata cap of "
            f"{SAM3_MAX_CONFIGURED_PROMPT_BYTES} UTF-8 bytes."
        )
    if dataset.train_instances < MIN_TRAIN_INSTANCES:
        refusals.append(
            f"Only {dataset.train_instances} labeled instances found; at least "
            f"{MIN_TRAIN_INSTANCES} are required to train."
        )
    if dataset.validation_present and dataset.validation_tiles == 0:
        refusals.append(
            "The validation split metadata exists but contains zero tiles; "
            "the child would fail during evaluation. Remove the absent split "
            "metadata or rebuild it with validation tiles."
        )
    requested_negatives = int(getattr(params, "num_negatives", 0))
    if requested_negatives < 0 or requested_negatives > _MAX_NEGATIVE_QUERIES_PER_TILE:
        refusals.append(
            "num_negatives must be between 0 and "
            f"{_MAX_NEGATIVE_QUERIES_PER_TILE}; got {requested_negatives}."
        )
    if requested_negatives > 0 and not resolved_negative_prompts:
        refusals.append(
            "num_negatives requests negative queries, but no negative prompts "
            "resolve from build_manifest.json or sam3_params.negative_prompts."
        )
    if len(valid_negative_prompts) != len(resolved_negative_prompts):
        refusals.append("Every resolved negative prompt must be a non-empty string.")
    for index, negative_prompt in enumerate(valid_negative_prompts):
        prompt_error = sam3_prompt_text_error(negative_prompt)
        if prompt_error is not None:
            refusals.append(f"Resolved negative prompt {index} {prompt_error}.")
    if len(valid_negative_prompts) > _MAX_NEGATIVE_PROMPT_COUNT:
        refusals.append(
            "Resolved negative prompts exceed the safe metadata cap of "
            f"{_MAX_NEGATIVE_PROMPT_COUNT} entries."
        )
    negative_prompt_bytes = sum(_utf8_size(prompt) for prompt in valid_negative_prompts)
    if negative_prompt_bytes > _MAX_NEGATIVE_PROMPT_BYTES:
        refusals.append(
            "Resolved negative prompt text exceeds the safe metadata cap of "
            f"{_MAX_NEGATIVE_PROMPT_BYTES} UTF-8 bytes."
        )
    artifact_required = max(
        (
            phase.disk_transient_bytes
            for phase in request.phases
            if phase.name != "publish"
        ),
        default=0,
    )
    if artifact_free_disk < artifact_required:
        refusals.append(
            f"Only {artifact_free_disk / GiB:.1f} GiB is free on the run "
            f"artifact disk filesystem; at least {artifact_required / GiB:.1f} "
            "GiB is required for transient training artifacts."
        )
    if (
        publish_phase is not None
        and publish_free_disk is not None
        and publish_free_disk < publish_phase.disk_transient_bytes
    ):
        refusals.append(
            f"Only {publish_free_disk / GiB:.1f} GiB is free on the publish "
            "disk filesystem; at least "
            f"{publish_phase.disk_transient_bytes / GiB:.1f} GiB is required."
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
        free_disk_bytes=artifact_free_disk,
        artifact_free_disk_bytes=artifact_free_disk,
        publish_free_disk_bytes=publish_free_disk,
        containment_soft_host_bytes=containment_soft_host_bytes,
        containment_hard_host_bytes=containment_hard_host_bytes,
        refusals=tuple(refusals),
        warnings=tuple(warnings),
        device_peak_provenance=device_peak_provenance,
        device_peak_analytic_bytes=int(analytic_device_bytes),
        device_peak_measured_bytes=int(measured_device_bytes),
        device_peak_measured_extrapolated=bool(measured_extrapolated),
        device_peak_fingerprint=device_peak_fingerprint,
    )


def spec_with_batch(spec: Any, batch: int) -> Any:
    """Return `spec` with `sam3_params.batch` replaced by `batch`.

    A copy, never a mutation: `spec` is the caller's requested configuration
    and several code paths (diagnostics, the cancelled-result shape, the
    provenance block's `requested` field) still need to see what was asked
    for. Falls back to a shallow copy for non-dataclass stand-ins used by
    tests and callers that pass duck-typed specs.
    """

    params = getattr(spec, "sam3_params", None)
    if params is None:
        return spec
    try:
        new_params = dataclasses.replace(params, batch=int(batch))
    except TypeError:
        new_params = copy.copy(params)
        new_params.batch = int(batch)
    try:
        return dataclasses.replace(spec, sam3_params=new_params)
    except TypeError:
        new_spec = copy.copy(spec)
        new_spec.sam3_params = new_params
        return new_spec


def assess_probe_preflight(
    spec: Any,
    *,
    batch: int,
    **kwargs: Any,
) -> Sam3PreflightDecision:
    """Admit ONE memory-probe candidate at its own batch size.

    Differs from `assess_preflight` in exactly two ways, and shares
    everything else by construction (it is the same function underneath):

    1. **Device gate.** The inherited device envelope is not used; a hard
       floor derived from checkpoint size and tile geometry is. See
       `build_resource_request(probe_floor=True)`.
    2. **Host gate.** Unchanged in form, but evaluated at `batch` rather than
       at the spec's requested value -- host demand scales with batch
       (decoded tiles, transformed tiles, collated images, dense masks), so
       every candidate must be admitted at the batch it will actually run.

    A refusal here stops the candidate ladder rather than skipping the
    candidate: host demand is monotone in batch, so no larger candidate can
    be admitted once one is refused.
    """

    return assess_preflight(spec_with_batch(spec, batch), probe_floor=True, **kwargs)


def preflight(spec: Any) -> list[str]:
    """Compatibility wrapper returning refusal strings."""

    return list(assess_preflight(spec).refusals)


def preflight_warnings(spec: Any) -> list[str]:
    """Compatibility wrapper returning warning strings."""

    return list(assess_preflight(spec).warnings)
