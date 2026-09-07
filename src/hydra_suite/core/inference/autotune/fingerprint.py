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

TUNING_SCHEMA_VERSION = 2
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
    # S2: True when this bucket was never actually measured -- run 1 has no
    # detection cache, so the workload falls back to bucket(MAX_TARGETS). A
    # key built from an estimate must be re-keyed by the first real
    # production sample instead of demoted as if reality had "changed" from
    # a measurement that never happened.
    density_is_estimated: bool = False

    @classmethod
    def from_counts(
        cls,
        configured_target_count: int,
        detections: Iterable[int],
        crops: Iterable[int],
        canonical_crop_geometries: Iterable[str],
        *,
        density_is_estimated: bool = False,
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
            density_is_estimated=bool(density_is_estimated),
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
    # S3: identifies the starting point (and, folded into the same digest,
    # which fields were manually pinned) of the search that produced this
    # profile. Two projects with different baselines -- or one that pins a
    # field the other leaves free -- searched different spaces; without this
    # a cache hit can lower (or raise) a setting nobody ever measured for the
    # requesting project. Defaults to "" only for structural compatibility
    # with call sites that build a key without a baseline in scope (e.g. unit
    # fixtures); production keys always pass an explicit digest.
    baseline_digest: str = ""

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
            baseline_digest=str(value.get("baseline_digest", "")),
        )

    @property
    def digest(self) -> str:
        encoded = json.dumps(
            self.to_dict(), sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()


def compute_baseline_digest(
    baseline: Any, manual_field_names: Iterable[str] = ()
) -> str:
    """Digest a starting point plus which fields the search held fixed.

    ``baseline`` is duck-typed as an ``InferenceTuningSettings`` (or
    compatible) -- it must expose ``field_names()`` and ``value_for(name)``.
    Not importing the concrete type here avoids a fingerprint -> models
    import edge that the rest of this module doesn't otherwise need.

    Two profiles are only interchangeable if they were tuned from the same
    starting settings *and* searched the same coordinate space: a project
    that pins ``pose_batch_size`` never explored it, so its winner is not
    evidence for a project that left it free (S3).
    """

    pairs = sorted(
        (str(name), baseline.value_for(name)) for name in baseline.field_names()
    )
    manual = sorted({str(name) for name in manual_field_names})
    encoded = json.dumps(
        {"baseline": pairs, "manual_fields": manual},
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
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
def _digest_stat(path: str, size: int, mtime_ns: int, ctime_ns: int, inode: int) -> str:
    # Metadata participates only in the in-process memoization key. The
    # persistent identity remains the SHA-256 of the actual artifact bytes.
    del size, mtime_ns, ctime_ns, inode
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@lru_cache(maxsize=128)
def _digest_directory_stat(
    root: str, entries: tuple[tuple[str, int, int, int, int], ...]
) -> str:
    """Digest a model bundle by relative names and bytes, memoized by its stat set."""

    base = Path(root)
    digest = hashlib.sha256()
    for relative, _size, _mtime_ns, _ctime_ns, _inode in entries:
        encoded_name = relative.encode("utf-8")
        digest.update(len(encoded_name).to_bytes(8, "big"))
        digest.update(encoded_name)
        with (base / relative).open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def model_content_digest(path: str | Path) -> str:
    """Hash model bytes once per filesystem-change identity; never persist paths."""

    artifact = Path(path).expanduser().resolve()
    if artifact.is_dir():

        def entry_identity(item: Path) -> tuple[str, int, int, int, int]:
            stat = item.stat()
            return (
                str(item.relative_to(artifact)),
                int(stat.st_size),
                int(stat.st_mtime_ns),
                int(stat.st_ctime_ns),
                int(stat.st_ino),
            )

        files = tuple(
            sorted(
                entry_identity(item) for item in artifact.rglob("*") if item.is_file()
            )
        )
        if not files:
            raise ValueError(f"model bundle has no files: {artifact}")
        return _digest_directory_stat(str(artifact), files)
    stat = artifact.stat()
    return _digest_stat(
        str(artifact),
        int(stat.st_size),
        int(stat.st_mtime_ns),
        int(stat.st_ctime_ns),
        int(stat.st_ino),
    )


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


def _torch_cuda_version() -> str:
    """Return the CUDA toolkit version torch was built against, if any.

    ``nvidia-cuda-runtime-cu1x`` pip-metadata lookups read "absent" on a
    conda-managed CUDA install (the common case in this lab) even though
    CUDA is fully present and in use -- torch always knows its own build's
    CUDA version regardless of how CUDA itself was installed.
    """
    try:
        import torch

        version = torch.version.cuda
        return str(version) if version else "absent"
    except ImportError:
        return "absent"


def _torch_cudnn_version() -> str:
    """Return the loaded cuDNN version via torch, if any (see ``_torch_cuda_version``)."""
    try:
        import torch

        if not torch.backends.cudnn.is_available():
            return "absent"
        version = torch.backends.cudnn.version()
        return str(version) if version else "absent"
    except ImportError:
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
    # NOTE (Task 9 remediation): SoftwareFingerprint is a member of
    # TuningProfileKey (see its `software` field), so preferring
    # _torch_cuda_version()/_torch_cudnn_version() over the pip-metadata
    # fallback changes `cuda`/`cudnn` from "absent" to a real value for
    # every conda-managed CUDA install. That flips the digest of every
    # existing CUDA profile on such a box, so every validated profile
    # invalidates and re-calibrates exactly once on upgrade to this fix.
    # Desirable (the profile's software identity was always wrong before),
    # but a one-time, user-visible re-calibration cost worth knowing about.
    cuda: str = "unknown",
    cudnn: str = "unknown",
) -> SoftwareFingerprint:
    """Collect versions without importing model frameworks before admission."""

    resolved_cuda = str(cuda)
    if resolved_cuda == "unknown":
        resolved_cuda = _torch_cuda_version()
        if resolved_cuda == "absent":
            resolved_cuda = _first_installed_version(
                ("nvidia-cuda-runtime-cu13", "nvidia-cuda-runtime-cu12")
            )
    resolved_cudnn = str(cudnn)
    if resolved_cudnn == "unknown":
        resolved_cudnn = _torch_cudnn_version()
        if resolved_cudnn == "absent":
            resolved_cudnn = _first_installed_version(
                ("nvidia-cudnn-cu13", "nvidia-cudnn-cu12")
            )
    return SoftwareFingerprint(
        hydra_version=str(hydra_version),
        hydra_commit=str(hydra_commit or hydra_code_identity()),
        python_abi=f"{sys.implementation.name}-{sys.version_info.major}.{sys.version_info.minor}-{getattr(sys, 'abiflags', '')}",
        backend=_text(backend, "backend"),
        precision=_text(precision, "precision"),
        driver=str(driver),
        cuda=resolved_cuda,
        cudnn=resolved_cudnn,
        tensorrt=_package_version("tensorrt"),
        pytorch=_package_version("torch"),
        ultralytics=_package_version("ultralytics"),
        sleap=_package_version("sleap"),
    )
