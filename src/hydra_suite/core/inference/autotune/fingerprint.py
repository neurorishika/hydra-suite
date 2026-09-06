"""Exact, path-free identity for reusable inference tuning profiles."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import socket
import sys
from dataclasses import asdict, dataclass
from functools import lru_cache
from importlib import metadata, resources
from pathlib import Path
from typing import Any, Iterable, Mapping

from hydra_suite import __version__ as hydra_version
from hydra_suite.runtime.resource_budget import ESTIMATOR_VERSION

TUNING_SCHEMA_VERSION = 1
SEARCH_POLICY_VERSION = "inference-coordinate-v1"


def _text(value: object, name: str, *, maximum: int = 512) -> str:
    result = str(value)
    if not result or len(result) > maximum or any(ord(c) < 0x20 for c in result):
        raise ValueError(f"{name} is invalid")
    return result


@dataclass(frozen=True, slots=True)
class SchemaFingerprint:
    tuning_schema: int = TUNING_SCHEMA_VERSION
    search_policy_version: str = SEARCH_POLICY_VERSION
    memory_estimator_version: str = ESTIMATOR_VERSION


@dataclass(frozen=True, slots=True)
class SystemFingerprint:
    host_id: str
    os_architecture: str
    cpu_model: str
    logical_cpu_count: int
    host_memory_class_bytes: int


@dataclass(frozen=True, slots=True)
class AcceleratorFingerprint:
    device_uuid: str
    model: str
    compute_capability: str
    total_vram_bytes: int


@dataclass(frozen=True, slots=True)
class SoftwareFingerprint:
    hydra_version: str
    hydra_commit: str
    python_abi: str
    backend: str
    precision: str
    driver: str
    cuda: str
    cudnn: str
    tensorrt: str
    pytorch: str
    ultralytics: str
    sleap: str


@dataclass(frozen=True, slots=True)
class ModelArtifactFingerprint:
    role: str
    sha256: str
    input_width: int = 0
    input_height: int = 0
    crop_width: int = 0
    crop_height: int = 0
    tensorrt_profile_id: str = "none"


@dataclass(frozen=True, slots=True)
class FrameFingerprint:
    width: int
    height: int
    channels: int
    pixel_format: str
    resize_factor: float
    decoder_mode: str


@dataclass(frozen=True, slots=True)
class DetectorFingerprint:
    method: str
    task: str
    target_classes: tuple[int, ...]
    confidence_threshold: float
    iou_threshold: float
    maximum_detections: int
    mode: str


@dataclass(frozen=True, slots=True)
class SliceFingerprint:
    enabled: bool
    tile_width: int
    tile_height: int
    overlap_width_ratio: float
    overlap_height_ratio: float
    full_frame: bool
    roi_gating_mode: str
    reference_body_geometry: str


@dataclass(frozen=True, slots=True)
class PipelineFingerprint:
    enabled_stages: tuple[str, ...]
    tracking_direction: str
    tracking_mode: str
    result_cache_stage_mask: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class WorkloadFingerprint:
    configured_target_count: int
    detections_p50_bucket: int
    detections_p95_bucket: int
    crops_p50_bucket: int
    crops_p95_bucket: int
    canonical_crop_geometries: tuple[str, ...]

    @classmethod
    def from_counts(
        cls,
        configured_target_count: int,
        detections: Iterable[int],
        crops: Iterable[int],
        canonical_crop_geometries: Iterable[str],
    ) -> "WorkloadFingerprint":
        det = tuple(int(v) for v in detections)
        crop = tuple(int(v) for v in crops)
        return cls(
            configured_target_count=int(configured_target_count),
            detections_p50_bucket=count_bucket(_percentile(det, 50)),
            detections_p95_bucket=count_bucket(_percentile(det, 95)),
            crops_p50_bucket=count_bucket(_percentile(crop, 50)),
            crops_p95_bucket=count_bucket(_percentile(crop, 95)),
            canonical_crop_geometries=tuple(
                sorted(map(str, canonical_crop_geometries))
            ),
        )


@dataclass(frozen=True, slots=True)
class TuningProfileKey:
    schema: SchemaFingerprint
    system: SystemFingerprint
    accelerator: AcceleratorFingerprint
    software: SoftwareFingerprint
    models: tuple[ModelArtifactFingerprint, ...]
    frame: FrameFingerprint
    detector: DetectorFingerprint
    slice: SliceFingerprint
    pipeline: PipelineFingerprint
    workload: WorkloadFingerprint

    def __post_init__(self) -> None:
        roles = [model.role for model in self.models]
        if roles != sorted(roles) or len(roles) != len(set(roles)):
            raise ValueError(
                "model fingerprints must have unique roles in sorted order"
            )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "TuningProfileKey":
        return cls(
            schema=SchemaFingerprint(**value["schema"]),
            system=SystemFingerprint(**value["system"]),
            accelerator=AcceleratorFingerprint(**value["accelerator"]),
            software=SoftwareFingerprint(**value["software"]),
            models=tuple(ModelArtifactFingerprint(**item) for item in value["models"]),
            frame=FrameFingerprint(**value["frame"]),
            detector=DetectorFingerprint(
                **{
                    **value["detector"],
                    "target_classes": tuple(value["detector"]["target_classes"]),
                }
            ),
            slice=SliceFingerprint(**value["slice"]),
            pipeline=PipelineFingerprint(
                **{
                    **value["pipeline"],
                    "enabled_stages": tuple(value["pipeline"]["enabled_stages"]),
                    "result_cache_stage_mask": tuple(
                        value["pipeline"]["result_cache_stage_mask"]
                    ),
                }
            ),
            workload=WorkloadFingerprint(
                **{
                    **value["workload"],
                    "canonical_crop_geometries": tuple(
                        value["workload"]["canonical_crop_geometries"]
                    ),
                }
            ),
        )

    @property
    def digest(self) -> str:
        encoded = json.dumps(
            self.to_dict(), sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()


def count_bucket(value: int | float) -> int:
    """Bucket density at powers of two while keeping zero distinguishable."""

    value = max(0, int(round(value)))
    if value <= 1:
        return value
    return 1 << (value - 1).bit_length()


def _percentile(values: tuple[int, ...], percentile: int) -> int:
    if not values:
        return 0
    ordered = sorted(values)
    index = round((len(ordered) - 1) * percentile / 100)
    return ordered[index]


@lru_cache(maxsize=1024)
def _digest_stat(path: str, size: int, mtime_ns: int) -> str:
    del size, mtime_ns
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@lru_cache(maxsize=128)
def _digest_directory_stat(root: str, entries: tuple[tuple[str, int, int], ...]) -> str:
    """Digest a model bundle by relative names and bytes, memoized by its stat set."""

    base = Path(root)
    digest = hashlib.sha256()
    for relative, _size, _mtime_ns in entries:
        encoded_name = relative.encode("utf-8")
        digest.update(len(encoded_name).to_bytes(8, "big"))
        digest.update(encoded_name)
        with (base / relative).open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def model_content_digest(path: str | Path) -> str:
    """Hash model bytes once per size/mtime identity; never persist its path."""

    artifact = Path(path).expanduser().resolve()
    if artifact.is_dir():
        files = tuple(
            sorted(
                (
                    str(item.relative_to(artifact)),
                    int(item.stat().st_size),
                    int(item.stat().st_mtime_ns),
                )
                for item in artifact.rglob("*")
                if item.is_file()
            )
        )
        if not files:
            raise ValueError(f"model bundle has no files: {artifact}")
        return _digest_directory_stat(str(artifact), files)
    stat = artifact.stat()
    return _digest_stat(str(artifact), int(stat.st_size), int(stat.st_mtime_ns))


def model_fingerprint(
    role: str,
    path: str | Path,
    *,
    input_size: tuple[int, int] = (0, 0),
    crop_size: tuple[int, int] = (0, 0),
    tensorrt_profile_id: str = "none",
) -> ModelArtifactFingerprint:
    return ModelArtifactFingerprint(
        role=_text(role, "role"),
        sha256=model_content_digest(path),
        input_width=int(input_size[0]),
        input_height=int(input_size[1]),
        crop_width=int(crop_size[0]),
        crop_height=int(crop_size[1]),
        tensorrt_profile_id=_text(tensorrt_profile_id, "tensorrt_profile_id"),
    )


def _package_version(name: str) -> str:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return "absent"


@lru_cache(maxsize=1)
def hydra_code_identity() -> str:
    """Return a deployable revision identity without invoking git."""

    explicit = os.environ.get("HYDRA_BUILD_COMMIT", "").strip()
    if explicit:
        return _text(explicit, "HYDRA_BUILD_COMMIT")
    try:
        direct_url = metadata.distribution("hydra-suite").read_text("direct_url.json")
        if direct_url:
            vcs = json.loads(direct_url).get("vcs_info", {})
            commit_id = str(vcs.get("commit_id", "")).strip()
            if commit_id:
                return _text(commit_id, "installed commit")
    except (
        metadata.PackageNotFoundError,
        OSError,
        ValueError,
        TypeError,
        AttributeError,
    ):
        pass

    # Editable/source deployments do not necessarily carry VCS metadata.
    # Hashing all package Python sources is a deterministic, stronger fallback:
    # any pipeline code change invalidates reuse even without a git checkout.
    digest = hashlib.sha256()

    def visit(node: Any, relative: str = "") -> None:
        children = sorted(node.iterdir(), key=lambda item: item.name)
        for child in children:
            name = f"{relative}/{child.name}" if relative else child.name
            if child.is_dir():
                visit(child, name)
            elif child.name.endswith(".py"):
                encoded = name.encode("utf-8")
                digest.update(len(encoded).to_bytes(8, "big"))
                digest.update(encoded)
                digest.update(child.read_bytes())

    try:
        visit(resources.files("hydra_suite"))
        return f"source-sha256:{digest.hexdigest()}"
    except (OSError, TypeError, AttributeError):
        # A packaged build without traversable sources still has an immutable
        # distribution version; retain an explicit namespace rather than the
        # unsafe generic "unknown" token.
        return f"version:{hydra_version}"


def _first_installed_version(names: tuple[str, ...]) -> str:
    for name in names:
        version = _package_version(name)
        if version != "absent":
            return version
    return "absent"


def default_system_fingerprint(*, total_host_bytes: int) -> SystemFingerprint:
    stable_host = hashlib.sha256(
        f"{socket.gethostname()}|{platform.node()}".encode("utf-8")
    ).hexdigest()
    # 4 GiB classes avoid invalidating a profile because firmware reports a few
    # reserved pages differently after reboot.
    memory_class = (int(total_host_bytes) // (4 * 1024**3)) * (4 * 1024**3)
    return SystemFingerprint(
        host_id=stable_host,
        os_architecture=f"{platform.system()}-{platform.release()}-{platform.machine()}",
        cpu_model=platform.processor() or platform.machine(),
        logical_cpu_count=int(os.cpu_count() or 1),
        host_memory_class_bytes=memory_class,
    )


def default_software_fingerprint(
    *,
    backend: str,
    precision: str,
    hydra_commit: str | None = None,
    driver: str = "unknown",
    cuda: str = "unknown",
    cudnn: str = "unknown",
) -> SoftwareFingerprint:
    """Collect versions without importing model frameworks before admission."""

    return SoftwareFingerprint(
        hydra_version=str(hydra_version),
        hydra_commit=str(hydra_commit or hydra_code_identity()),
        python_abi=f"{sys.implementation.name}-{sys.version_info.major}.{sys.version_info.minor}-{getattr(sys, 'abiflags', '')}",
        backend=_text(backend, "backend"),
        precision=_text(precision, "precision"),
        driver=str(driver),
        cuda=(
            str(cuda)
            if str(cuda) != "unknown"
            else _first_installed_version(
                ("nvidia-cuda-runtime-cu13", "nvidia-cuda-runtime-cu12")
            )
        ),
        cudnn=(
            str(cudnn)
            if str(cudnn) != "unknown"
            else _first_installed_version(("nvidia-cudnn-cu13", "nvidia-cudnn-cu12"))
        ),
        tensorrt=_package_version("tensorrt"),
        pytorch=_package_version("torch"),
        ultralytics=_package_version("ultralytics"),
        sleap=_package_version("sleap"),
    )
