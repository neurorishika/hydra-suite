"""OBB runtime-artifact loading: TensorRT/CoreML auto-export + direct executor.

This is a CLEAN port of the load + auto-export + direct-executor selection logic
from the legacy ``core/detectors/_runtime_artifacts.py`` (``_try_load_onnx_model``,
``_try_load_tensorrt_model``, ``_maybe_enable_direct_obb_executor``,
``_maybe_enable_direct_cuda_obb_executor``) and ``core/inference/direct_executors.py``
(``create_direct_obb_executor``).

It delegates to the sibling ``direct_executors`` module — the legacy code is a
mixin entangled with detector state (``self.params``, ``self.device``,
session-disable tracking, CoreML CPU-fallback bookkeeping, batch-override
machinery). Here we port only the standalone essence needed by the new
inference pipeline:

  * ``load_obb_executor(model_path, compute_runtime, *, auto_export)``
      - cpu/mps/cuda → returns a plain PyTorch (ultralytics ``YOLO``) model,
        ``.to()``-moved to the device. This keeps CPU/MPS behaviour byte-identical
        to the previous ``_load_yolo``.
      - tensorrt → loads (or auto-exports then loads) the ``.engine``
        artifact and returns a direct TRT executor wrapped in a YOLO-compatible
        adapter so the geometry-extraction stage is unchanged.
      - tensorrt with ``auto_export=False`` and a missing artifact →
        raises :class:`ArtifactExportError` (a CLEAR error, never a silent
        PyTorch fallback — that silent fallback was parity-audit finding H4).
      - coreml → exports (or reuses) a ``.mlpackage`` via ultralytics CoreML
        export with fixed ``imgsz`` (avoids the dynamic-shape E5RT failure seen
        with ``onnx_coreml``) then loads it back via ``YOLO(mlpackage_path)``
        (Apple Silicon only).

Note: onnx_* runtimes are NOT supported for OBB. The production pipeline
(tier→compute-runtime-string resolution) only emits {cpu, mps, cuda, tensorrt, coreml}.

Square-letterbox parity: the direct executors (ported in
``core/inference/direct_executors.py``) use ``LetterBox(auto=False)`` so the
model always sees a square ``imgsz×imgsz`` input — identical preprocessing for
the PyTorch-CUDA, ONNX, and TensorRT paths. The CUDA runtime here returns the
plain torch model (the existing ``run_obb`` native-CUDA path consumes its raw
tensors), matching the pre-existing pipeline contract.

The real export runs ultralytics ``model.export(...)`` on a CUDA box; the
selection logic is exercised on CPU by monkeypatching ``_load_torch_model``,
``_export_artifact`` and ``_create_direct_executor`` (see
``tests/test_inference_obb_artifacts.py``).
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import tempfile
import time
from contextlib import contextmanager
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any, Callable, Iterator

logger = logging.getLogger(__name__)

# Compute-runtime → direct-executor runtime name.
_TENSORRT_RUNTIMES: frozenset[str] = frozenset({"tensorrt"})
_TORCH_RUNTIMES: frozenset[str] = frozenset({"cpu", "mps", "cuda"})
_COREML_RUNTIMES: frozenset[str] = frozenset({"coreml"})

_DEFAULT_IMGSZ = 640
_DEFAULT_BATCH_SIZE = 1
_DEFAULT_MAX_DET = 20


class ArtifactExportError(RuntimeError):
    """Raised when a required ONNX/TRT artifact is unavailable and cannot be built.

    This is the explicit-error path that replaces the H4 silent PyTorch fallback:
    when an ``onnx_*``/``tensorrt`` runtime is requested but the artifact is
    missing and ``auto_export=False``, we raise this instead of quietly running
    PyTorch.
    """


def _sha256_file(path: Path) -> str:
    """Return a content digest for an immutable runtime-artifact source."""
    digest = sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def runtime_artifact_runtime_fingerprint(runtime: str) -> dict[str, str]:
    """Return the runtime/toolchain identity that makes an engine reusable.

    This deliberately excludes performance-search policy versions.  Engine
    artifacts have an independent lifecycle from throughput profiles, but a
    TensorRT/CoreML ABI or precision change must never reuse an old binary.
    Optional packages are represented by a stable ``"unavailable"`` value.
    """
    import platform
    import sys
    from importlib import metadata

    versions: dict[str, str] = {
        "runtime": str(runtime),
        "python_abi": getattr(sys.implementation, "cache_tag", "unknown"),
        "platform": f"{platform.system()}-{platform.machine()}",
    }
    for package in ("torch", "tensorrt", "ultralytics", "coremltools"):
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            versions[package] = "unavailable"
    try:
        import torch

        versions["cuda"] = str(torch.version.cuda or "none")
        versions["cudnn"] = str(torch.backends.cudnn.version() or "none")
        if runtime == "tensorrt" and torch.cuda.is_available():
            properties = torch.cuda.get_device_properties(0)
            versions["accelerator"] = (
                f"{properties.name}|cc={properties.major}.{properties.minor}|"
                f"vram={properties.total_memory}"
            )
        else:
            versions["accelerator"] = "not-applicable"
    except Exception:
        versions["cuda"] = "unavailable"
        versions["cudnn"] = "unavailable"
    return versions


@dataclass(frozen=True)
class RuntimeArtifactIdentity:
    """Exact immutable identity for one TensorRT or CoreML artifact.

    The identity is intentionally narrower than a throughput profile and has
    no video path, workload bucket, or tuning-policy revision.  It contains
    every property that can alter compiled runtime behavior instead.
    """

    source_digest: str
    runtime: str
    runtime_fingerprint: str
    imgsz: int
    task: str
    precision: str
    profile: tuple[tuple[str, object], ...]

    @classmethod
    def from_source(
        cls,
        *,
        source_path: Path,
        runtime: str,
        runtime_fingerprint: str | dict[str, str],
        imgsz: int,
        task: str,
        precision: str,
        profile: dict[str, object],
    ) -> "RuntimeArtifactIdentity":
        return cls(
            source_digest=_sha256_file(Path(source_path)),
            runtime=str(runtime),
            runtime_fingerprint=(
                runtime_fingerprint
                if isinstance(runtime_fingerprint, str)
                else _canonical_json(runtime_fingerprint)
            ),
            imgsz=int(imgsz),
            task=str(task),
            precision=str(precision),
            profile=tuple(sorted((str(key), value) for key, value in profile.items())),
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "schema": "runtime-artifact-v1",
            "source_digest": self.source_digest,
            "runtime": self.runtime,
            "runtime_fingerprint": self.runtime_fingerprint,
            "imgsz": self.imgsz,
            "task": self.task,
            "precision": self.precision,
            "profile": dict(self.profile),
        }

    @property
    def digest(self) -> str:
        return sha256(_canonical_json(self.as_dict()).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class RuntimeArtifactBuild:
    """The outcome of a serialized immutable runtime-artifact preparation."""

    identity: RuntimeArtifactIdentity
    path: Path
    built: bool
    prepare_seconds: float


ArtifactBuilder = Callable[[Path, Path], None]


class RuntimeArtifactStore:
    """Process-safe store for immutable TensorRT/CoreML build artifacts.

    Each identity gets a directory of its own, so the payload and freshness
    sidecar are promoted together.  Build callbacks receive a private copied
    source checkpoint and output path; this prevents Ultralytics' generic
    intermediate names from colliding across candidate/profile builds.
    """

    def __init__(self, root: Path) -> None:
        self.root = Path(root).expanduser().resolve()
        self._tmp_root = self.root / ".tmp"
        self._lock_root = self.root / ".locks"

    def path_for(self, identity: RuntimeArtifactIdentity) -> Path:
        return self.root / identity.runtime / identity.digest

    def artifact_path_for(self, identity: RuntimeArtifactIdentity) -> Path:
        return self.path_for(identity) / f"artifact{_artifact_suffix(identity.runtime)}"

    def is_ready(self, identity: RuntimeArtifactIdentity) -> bool:
        artifact_path = self.artifact_path_for(identity)
        marker = _meta_path(artifact_path)
        if not artifact_path.exists() or not marker.is_file():
            return False
        try:
            metadata = json.loads(marker.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError, TypeError):
            return False
        return metadata.get("runtime_artifact_id") == identity.digest

    def ensure(
        self,
        identity: RuntimeArtifactIdentity,
        build: ArtifactBuilder,
        *,
        source_path: Path | None = None,
    ) -> RuntimeArtifactBuild:
        """Return a ready artifact, building it once under an OS file lock."""
        started = time.perf_counter()
        source = Path(source_path) if source_path is not None else None
        if source is not None and not source.is_file():
            raise ArtifactExportError(
                f"runtime-artifact source does not exist: {source}"
            )
        self._lock_root.mkdir(parents=True, exist_ok=True)
        self._tmp_root.mkdir(parents=True, exist_ok=True)
        lock_path = self._lock_root / f"{identity.digest}.lock"
        with _exclusive_artifact_lock(lock_path):
            if self.is_ready(identity):
                return RuntimeArtifactBuild(
                    identity,
                    self.artifact_path_for(identity),
                    False,
                    time.perf_counter() - started,
                )
            if source is None:
                raise ArtifactExportError(
                    "building a new artifact requires its source path"
                )
            return self._build_locked(identity, source, build, started)

    def import_legacy(
        self,
        identity: RuntimeArtifactIdentity,
        legacy_path: Path,
        *,
        source_path: Path,
    ) -> RuntimeArtifactBuild | None:
        """Atomically migrate a verified old sidecar artifact into this store."""
        legacy_path = Path(legacy_path)
        if not legacy_path.exists():
            return None

        def copy_legacy(_private_source: Path, output: Path) -> None:
            if legacy_path.is_dir():
                shutil.copytree(legacy_path, output)
            else:
                shutil.copy2(legacy_path, output)

        return self.ensure(identity, copy_legacy, source_path=source_path)

    def _build_locked(
        self,
        identity: RuntimeArtifactIdentity,
        source: Path,
        build: ArtifactBuilder,
        started: float,
    ) -> RuntimeArtifactBuild:
        stage = Path(
            tempfile.mkdtemp(prefix=f"{identity.digest[:12]}-", dir=self._tmp_root)
        )
        os.chmod(stage, 0o700)
        try:
            private_source = stage / source.name
            shutil.copy2(source, private_source)
            artifact_path = stage / f"artifact{_artifact_suffix(identity.runtime)}"
            build(private_source, artifact_path)
            if not artifact_path.exists():
                raise ArtifactExportError(
                    f"runtime-artifact builder produced no payload for {identity.digest}"
                )
            _write_fresh_marker(
                artifact_path,
                source,
                identity.imgsz,
                batch_size=int(dict(identity.profile).get("opt_batch", 1)),
                enforce_trt_profile=identity.runtime == "tensorrt",
                runtime_artifact_id=identity.digest,
                runtime_artifact_identity=identity.as_dict(),
            )
            _fsync_artifact_tree(stage)
            destination = self.path_for(identity)
            destination.parent.mkdir(parents=True, exist_ok=True)
            if destination.exists():
                # A pre-marker crash cannot be a ready immutable artifact.  The
                # per-key lock makes this bounded cleanup safe and local.
                shutil.rmtree(destination)
            os.replace(stage, destination)
            _fsync_directory(destination.parent)
            return RuntimeArtifactBuild(
                identity,
                self.artifact_path_for(identity),
                True,
                time.perf_counter() - started,
            )
        except BaseException:
            shutil.rmtree(stage, ignore_errors=True)
            raise


@contextmanager
def _exclusive_artifact_lock(path: Path) -> Iterator[None]:
    """Acquire a blocking OS-backed lock for one artifact identity."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as handle:
        try:
            import fcntl

            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        except ImportError:  # pragma: no cover - Windows is not a TRT host
            raise RuntimeError("runtime artifact locking requires an OS file lock")
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _fsync_directory(path: Path) -> None:
    try:
        descriptor = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    except OSError:
        pass
    finally:
        os.close(descriptor)


def _fsync_artifact_tree(root: Path) -> None:
    for candidate in root.rglob("*"):
        if candidate.is_file():
            with candidate.open("rb") as handle:
                os.fsync(handle.fileno())
    _fsync_directory(root)


# ---------------------------------------------------------------------------
# Injectable hooks (monkeypatched in tests so selection logic runs on CPU).
# ---------------------------------------------------------------------------


def _load_torch_model(model_path: str) -> Any:
    """Load an ultralytics ``YOLO`` model from a ``.pt`` (or alias) path."""
    from ultralytics import YOLO

    return YOLO(model_path)


def _resolve_imgsz(pt_path: Path) -> int:
    """Resolve the square model input size from the source ``.pt`` metadata.

    Mirrors legacy ``_resolve_onnx_imgsz`` priority (model overrides/args →
    fallback 640), clamped to a sane export range.
    """
    imgsz: int | None = None
    try:
        model = _load_torch_model(str(pt_path))
        ov = getattr(model, "overrides", {}) or {}
        arg_imgsz = ov.get("imgsz") if isinstance(ov, dict) else None
        if arg_imgsz is None:
            margs = getattr(getattr(model, "model", None), "args", {}) or {}
            if isinstance(margs, dict):
                arg_imgsz = margs.get("imgsz")
        if arg_imgsz is not None:
            imgsz = int(arg_imgsz)
    except Exception:
        imgsz = None
    if imgsz is None:
        imgsz = _DEFAULT_IMGSZ
    return max(64, min(4096, int(imgsz)))


def _model_class_names(pt_path: Path) -> dict[int, str] | None:
    try:
        model = _load_torch_model(str(pt_path))
        names = getattr(model, "names", None)
        if names:
            return {int(k): str(v) for k, v in dict(names).items()}
    except Exception:
        pass
    return None


def _export_artifact(
    *,
    pt_path: Path,
    artifact_path: Path,
    runtime: str,
    imgsz: int,
    batch_size: int,
) -> Path:
    """Export a ``.pt`` model to a ``.engine``/``.mlpackage`` artifact.

    Ported from legacy ``_try_load_tensorrt_model`` export blocks. Runs
    ultralytics ``model.export(...)`` (CUDA-only for TRT) and copies the
    exported file to ``artifact_path``. ``runtime`` is the direct runtime name
    (``"tensorrt"`` or ``"coreml"``).

    This function only runs on a machine with ultralytics (and, for TRT, a CUDA
    device); the tests inject a fake in its place.
    """
    # A rebuild INVALIDATES whatever is at ``artifact_path`` right now, so the
    # freshness marker must go FIRST. ``_fresh()`` is checked once outside the
    # build lock as a fast path; leaving the marker in place would let a sibling
    # fan-out child pass that check and load an artifact that is mid-replacement.
    try:
        _meta_path(artifact_path).unlink()
    except FileNotFoundError:
        pass
    except OSError:  # pragma: no cover - read-only artifact dir
        logger.warning(
            "Could not clear the freshness marker for %s before rebuilding it",
            artifact_path,
        )

    if runtime == "coreml":
        # ultralytics names the CoreML export after the ``.pt`` it was handed and
        # writes it BESIDE that file -- which, for the real checkpoint, is
        # character-for-character ``artifact_path``. Exporting from the
        # checkpoint therefore builds the .mlpackage IN PLACE, exposing a
        # half-written directory to any concurrent reader (fan-out children all
        # load the same artifact) and bypassing the atomic installer entirely.
        # Export from a scratch COPY instead: ``out_path != artifact_path``, so
        # the finished directory is published by the same swap the TRT path uses.
        with tempfile.TemporaryDirectory(prefix="hydra-artifact-export-") as scratch:
            scratch_pt = Path(scratch) / pt_path.name
            shutil.copy2(str(pt_path), str(scratch_pt))
            out_path = _run_ultralytics_export(
                scratch_pt, runtime=runtime, imgsz=imgsz, batch_size=batch_size
            )
            # INSIDE the with-block: the scratch dir owns ``out_path``.
            _install_artifact_atomically(out_path, artifact_path)
        return artifact_path

    out_path = _run_ultralytics_export(
        pt_path, runtime=runtime, imgsz=imgsz, batch_size=batch_size
    )
    if out_path != artifact_path:
        _install_artifact_atomically(out_path, artifact_path)
    return artifact_path


def _run_ultralytics_export(
    pt_path: Path, *, runtime: str, imgsz: int, batch_size: int
) -> Path:
    """Run ultralytics' exporter on *pt_path* and return the file it produced.

    Where that file lands is ultralytics' choice (beside the source ``.pt``);
    publishing it at the artifact path is the caller's job.
    """
    from ultralytics import YOLO

    base_model = YOLO(str(pt_path))
    # CoreML does not use the CBC direct executor, so skip the raw-head override.
    if runtime != "coreml":
        # Force raw-head (end2end=False) export so the direct executor's NMS path
        # matches the TRT raw-CBC contract (legacy _yolo_runtime_export_profile).
        _force_raw_head(base_model)

    if runtime == "tensorrt":
        dynamic = int(batch_size) > 1
        logger.info(
            "Building TensorRT OBB engine (imgsz=%d, batch=%d, dynamic=%s) — "
            "one-time export...",
            imgsz,
            batch_size,
            dynamic,
        )
        export_path = base_model.export(
            format="engine",
            device="cuda:0",
            half=True,
            dynamic=dynamic,
            batch=int(batch_size),
            imgsz=imgsz,
            verbose=False,
        )
    elif runtime == "coreml":
        logger.info(
            "Exporting YOLO OBB model to CoreML .mlpackage (imgsz=%d)...", imgsz
        )
        export_path = base_model.export(
            format="coreml",
            imgsz=imgsz,
            nms=False,
        )
    else:  # pragma: no cover - guarded by callers
        raise ArtifactExportError(f"Unsupported export runtime: {runtime}")

    # Free the PyTorch export model before the ORT/TRT session is created so the
    # two runtimes don't compete for the same CUDA context (legacy note).
    del base_model
    try:
        import torch as _torch

        if _torch.cuda.is_available():
            _torch.cuda.synchronize()
            _torch.cuda.empty_cache()
    except Exception:
        pass

    out_path = Path(export_path).expanduser().resolve()
    if not out_path.exists():
        raise ArtifactExportError(f"Export produced no output file: {out_path}")
    return out_path


def _remove_path(path: Path) -> None:
    """Best-effort delete of a file or directory."""
    try:
        if path.is_dir() and not path.is_symlink():
            shutil.rmtree(str(path), ignore_errors=True)
        else:
            path.unlink()
    except (FileNotFoundError, OSError):
        pass


def _install_artifact_atomically(source: Path, artifact_path: Path) -> None:
    """Publish *source* as *artifact_path* without ever exposing a partial file.

    Concurrent fan-out children can be LOADING ``artifact_path`` while another
    rebuilds it (per-video sidecars differing in imgsz resolve to the same
    artifact filename). ``shutil.copy2`` truncates the file under the reader;
    staging beside it and renaming makes the swap atomic, so a reader sees
    either the whole old artifact or the whole new one — never a prefix.
    """
    staging = artifact_path.parent / f".{artifact_path.name}.tmp-{os.getpid()}"
    _remove_path(staging)
    try:
        if source.is_dir():
            shutil.copytree(str(source), str(staging))
        else:
            shutil.copy2(str(source), str(staging))
        if artifact_path.is_dir():
            # os.replace refuses a non-empty directory target (.mlpackage):
            # move the old one aside, swap the new one in, then delete it.
            displaced = (
                artifact_path.parent / f".{artifact_path.name}.old-{os.getpid()}"
            )
            _remove_path(displaced)
            os.rename(str(artifact_path), str(displaced))
            try:
                os.rename(str(staging), str(artifact_path))
            except BaseException:
                # The old artifact is now the ONLY copy: put it back before
                # propagating, or this "safer" publish would be the thing that
                # destroyed it.
                try:
                    os.rename(str(displaced), str(artifact_path))
                except OSError:
                    logger.error(
                        "Could not restore %s after a failed swap; it is at %s",
                        artifact_path,
                        displaced,
                    )
                raise
            _remove_path(displaced)
        else:
            os.replace(str(staging), str(artifact_path))
    finally:
        _remove_path(staging)


def _force_raw_head(base_model: Any) -> None:
    """Disable end2end on the model head so exports use the raw CBC head."""
    try:
        head = base_model.model.model[-1]
        if bool(getattr(head, "end2end", False)):
            head.end2end = False
            logger.info("Exporting OBB runtime artifact with raw head (end2end=False).")
    except Exception:
        pass


def _create_direct_executor(
    *,
    runtime: str,
    artifact_path: Path,
    imgsz: int,
    class_names: dict[int, str] | None = None,
    task: str = "obb",
) -> Any:
    """Create a direct ONNX/TRT executor (square-letterbox preprocessing).

    Delegates to the ported ``core/inference/direct_executors`` factory. The
    executors there use ``LetterBox(auto=False)`` — identical square-letterbox
    preprocessing across the PyTorch-CUDA, ONNX, and TRT paths, which is the
    parity guarantee enforced by legacy ``_maybe_enable_direct_cuda_obb_executor``.

    ``task="obb"`` (default) returns an executor that parses the model's raw
    head as OBB output (cx,cy,w,h,angle,conf) into ``Results(obb=...)``.
    ``task="detect"`` returns the plain-box variant (cx,cy,w,h,conf) into
    ``Results(boxes=...)`` -- required for the sequential pipeline's stage-1
    detect model, which is NOT an OBB model. Feeding a plain-detect model's
    output through the OBB parser silently misreads the class-score channel
    as an angle and always yields ``Results.boxes is None`` (mirrors legacy's
    separate ``_maybe_enable_direct_detect_executor``).

    ``task="segment"`` returns an executor that decodes the model's raw
    detection + mask-prototype outputs and derives OBB geometry via
    ``hydra_suite.utils.obb_from_mask.rotated_rect_from_masks`` -- a GPU-native,
    cv2-free batched rotated-rectangle search -- for treating a YOLO
    segmentation checkpoint as an OBB source.
    """
    from .direct_executors import (
        create_direct_detect_executor,
        create_direct_obb_executor,
        create_direct_segment_executor,
    )

    if task == "detect":
        factory = create_direct_detect_executor
    elif task == "segment":
        factory = create_direct_segment_executor
    else:
        factory = create_direct_obb_executor
    return factory(
        runtime=runtime,
        artifact_path=str(artifact_path),
        imgsz=int(imgsz),
        class_names=class_names,
    )


# ---------------------------------------------------------------------------
# Artifact path + freshness bookkeeping (clean port of legacy meta logic).
# ---------------------------------------------------------------------------


def _direct_runtime_name(compute_runtime: str) -> str:
    """Map a compute-runtime to the direct-executor runtime name.

    The inference-stage pipeline only ever reaches a direct executor on the
    gpu_fast tier, so this deliberately emits ``"tensorrt"`` exclusively (the
    runtime-tier contract is cpu->cpu torch, gpu->cuda/mps native torch,
    gpu_fast->tensorrt/coreml). The ``"onnx"`` branches of
    ``create_direct_{obb,detect}_executor`` in ``direct_executors`` are NOT
    reachable through this path — they exist only for the diagnostic tools
    (``tools/diag_*``, ``tools/compare_runtimes.py``). Anything other than a
    TensorRT runtime here is a programming error, hence the raise.
    """
    if compute_runtime in _TENSORRT_RUNTIMES:
        return "tensorrt"
    raise ArtifactExportError(
        f"compute_runtime {compute_runtime!r} is not a TensorRT runtime"
    )


def _artifact_suffix(runtime: str) -> str:
    if runtime == "coreml":
        return ".mlpackage"
    return ".engine"


def _artifact_path_for(
    pt_path: Path, runtime: str, batch_size: int = _DEFAULT_BATCH_SIZE
) -> Path:
    """Derive the artifact path for a ``.pt`` source + direct runtime name.

    ``batch_size`` is embedded in the filename (``_b1``, ``_b8``, ...) so a
    workflow requesting a different batch size never reuses a wrong-shaped
    cached engine. ``batch_size == 1`` exports a static batch=1 engine
    (unchanged from before); ``batch_size > 1`` exports a TensorRT engine
    with a dynamic batch profile (min=1, opt=batch_size, max=batch_size --
    see ``_export_artifact``).
    """
    pt_path = Path(pt_path)
    suffix = _artifact_suffix(runtime)
    if runtime == "coreml":
        # CoreML uses a bare stem (no batch suffix): OBB stays batch=1 on
        # CoreML permanently -- ultralytics' CoreML export hard-crashes at
        # compile time when both the batch and spatial dims are made
        # dynamic together for an OBB model (Spec 1 Phase A/B, 2026-07-04)
        # -- so there is only ever one CoreML OBB artifact.
        return pt_path.with_suffix(".mlpackage")
    return pt_path.with_name(f"{pt_path.stem}_b{int(batch_size)}{suffix}")


def _meta_path(artifact_path: Path) -> Path:
    return artifact_path.with_suffix(f"{artifact_path.suffix}.runtime_meta.json")


def tensorrt_profile_fingerprint(batch_size: int) -> str:
    """Versioned identity for the dynamic TensorRT batch profile.

    The artifact filename contains the maximum batch, but retaining this in
    metadata makes profile semantics explicit and invalidates pre-admission
    markers which may name a batch that cannot actually be issued.
    """
    profile = {
        "schema": "sahi-tile-profile-v2",
        "min_batch": 1,
        "opt_batch": max(1, int(batch_size)),
        "max_batch": max(1, int(batch_size)),
    }
    return sha256(json.dumps(profile, sort_keys=True).encode("utf-8")).hexdigest()


# Retained for internal callers and compatibility with focused artifact tests.
_trt_profile_fingerprint = tensorrt_profile_fingerprint


def _trt_profile_identity(batch_size: int) -> dict[str, object]:
    """Return the explicit TensorRT optimization-profile identity."""
    return {
        "schema": "sahi-tile-profile-v2",
        "min_batch": 1,
        "opt_batch": max(1, int(batch_size)),
        "max_batch": max(1, int(batch_size)),
    }


def _runtime_artifact_store_for(source_path: Path) -> RuntimeArtifactStore:
    """Locate the user-writable engine cache without using model directories.

    A read-only/shared model registry is a normal deployment.  The fallback is
    deliberately local only for constrained environments where the configured
    data directory itself cannot be created (for example a sandboxed test).
    """
    try:
        from hydra_suite.paths import get_data_dir

        root = get_data_dir() / "runtime-artifacts"
        root.mkdir(parents=True, exist_ok=True)
    except OSError:
        root = source_path.parent / ".hydra-runtime-artifacts"
        root.mkdir(parents=True, exist_ok=True)
    return RuntimeArtifactStore(root)


def _runtime_artifact_identity(
    source_path: Path,
    runtime: str,
    *,
    imgsz: int,
    batch_size: int,
    task: str,
) -> RuntimeArtifactIdentity:
    profile = (
        _trt_profile_identity(batch_size)
        if runtime == "tensorrt"
        else {"schema": "coreml-static-v1", "batch": 1}
    )
    return RuntimeArtifactIdentity.from_source(
        source_path=source_path,
        runtime=runtime,
        runtime_fingerprint=runtime_artifact_runtime_fingerprint(runtime),
        imgsz=imgsz,
        task=task,
        precision="fp16" if runtime == "tensorrt" else "fp32",
        profile=profile,
    )


def _prepare_runtime_artifact(
    *,
    source_path: Path,
    runtime: str,
    imgsz: int,
    batch_size: int,
    task: str,
    auto_export: bool,
) -> RuntimeArtifactBuild:
    """Find or safely prepare the exact runtime artifact for one source model."""
    identity = _runtime_artifact_identity(
        source_path,
        runtime,
        imgsz=imgsz,
        batch_size=batch_size,
        task=task,
    )
    store = _runtime_artifact_store_for(source_path)
    if store.is_ready(identity):
        return RuntimeArtifactBuild(
            identity, store.artifact_path_for(identity), False, 0.0
        )

    # Old adjacent artifacts remain a read-only compatibility input.  Once
    # imported they are no longer touched, and the new record gains the exact
    # runtime/profile identity missing from legacy sidecars.
    legacy_path = _artifact_path_for(source_path, runtime, batch_size=batch_size)
    if _artifact_is_fresh(
        legacy_path,
        source_path,
        imgsz,
        batch_size=batch_size,
        enforce_trt_profile=runtime == "tensorrt",
    ):
        migrated = store.import_legacy(identity, legacy_path, source_path=source_path)
        if migrated is not None:
            return migrated
    if not auto_export:
        raise ArtifactExportError(
            f"compute runtime {runtime!r} requested but no compatible immutable "
            f"runtime artifact exists for {source_path.name} and auto_export=False."
        )

    def build(private_source: Path, output: Path) -> None:
        _export_artifact(
            pt_path=private_source,
            artifact_path=output,
            runtime=runtime,
            imgsz=imgsz,
            batch_size=batch_size,
        )

    return store.ensure(identity, build, source_path=source_path)


def _write_fresh_marker(
    artifact_path: Path,
    source_pt: Path,
    imgsz: int,
    *,
    batch_size: int = _DEFAULT_BATCH_SIZE,
    enforce_trt_profile: bool = False,
    runtime_artifact_id: str | None = None,
    runtime_artifact_identity: dict[str, object] | None = None,
) -> None:
    """Write a freshness marker recording the source ``.pt`` mtime + build imgsz.

    Mirrors legacy ``_write_artifact_meta`` — used so a subsequent load can tell
    whether the cached artifact is still valid for the current source model.
    ``imgsz`` is recorded so a config change (e.g. a sequential OBB stage's
    ``stage2_image_size`` differing from the checkpoint's own default) forces a
    rebuild instead of silently reusing an artifact exported at the wrong input
    size (H4: no silent wrong-behavior fallback).
    """
    try:
        source_mtime_ns = source_pt.stat().st_mtime_ns
    except Exception:
        source_mtime_ns = 0
    marker = {"source_mtime_ns": source_mtime_ns, "imgsz": int(imgsz)}
    if enforce_trt_profile:
        marker["trt_profile_fingerprint"] = _trt_profile_fingerprint(batch_size)
    if runtime_artifact_id is not None:
        marker["runtime_artifact_id"] = str(runtime_artifact_id)
    if runtime_artifact_identity is not None:
        marker["runtime_artifact_identity"] = runtime_artifact_identity
    marker_path = _meta_path(artifact_path)
    with marker_path.open("w", encoding="utf-8") as handle:
        json.dump(marker, handle, sort_keys=True)
        handle.flush()
        os.fsync(handle.fileno())


def _artifact_is_fresh(
    artifact_path: Path,
    source_pt: Path,
    imgsz: int,
    *,
    batch_size: int = _DEFAULT_BATCH_SIZE,
    enforce_trt_profile: bool = False,
    runtime_artifact_id: str | None = None,
) -> bool:
    """Return True when ``artifact_path`` exists and is newer than its source.

    Clean analogue of legacy ``_artifact_is_fresh``: a cached artifact is reused
    only when it exists and was built from the current ``.pt`` (by recorded
    source mtime) at the requested ``imgsz``. Missing/stale markers or an
    imgsz mismatch force a rebuild. Handles both file artifacts (.onnx/.engine)
    and directory artifacts (.mlpackage).
    """
    if not artifact_path.exists():
        return False
    meta = _meta_path(artifact_path)
    if not meta.exists():
        return False
    try:
        data = json.loads(meta.read_text(encoding="utf-8"))
    except Exception:
        return False
    try:
        recorded = int(data.get("source_mtime_ns", -1))
        current = int(source_pt.stat().st_mtime_ns)
    except Exception:
        return False
    if recorded != current:
        return False
    if int(data.get("imgsz", -1)) != int(imgsz):
        return False
    if (
        runtime_artifact_id is not None
        and data.get("runtime_artifact_id") != runtime_artifact_id
    ):
        return False
    # Dynamic profiles are meaningful only for TensorRT.  Do not invalidate
    # CoreML/non-TRT artifacts merely because they predate this TRT marker.
    return not enforce_trt_profile or data.get(
        "trt_profile_fingerprint"
    ) == _trt_profile_fingerprint(batch_size)


# ---------------------------------------------------------------------------
# YOLO-compatible adapter around a direct executor.
# ---------------------------------------------------------------------------


class DirectExecutorAdapter:
    """Expose a direct ONNX/TRT executor through a YOLO-compatible ``predict``.

    ``stages/obb.py`` calls ``model.predict(frames, conf=, iou=, classes=,
    verbose=, device=, ...)`` and then runs ``_extract_obb_result`` on the
    returned ultralytics ``Results`` objects. The direct executors expose
    ``predict(frames, *, conf_thres, classes, max_det)`` and return the same
    ``Results`` objects, so this thin adapter translates the kwargs and keeps the
    geometry-extraction stage completely unchanged.
    """

    def __init__(
        self,
        executor: Any,
        *,
        max_det: int = _DEFAULT_MAX_DET,
        runtime_artifact_id: str | None = None,
        prepare_seconds: float = 0.0,
    ) -> None:
        self._executor = executor
        self._max_det = int(max_det)
        self.runtime_artifact_id = runtime_artifact_id
        self.runtime_artifact_prepare_seconds = float(prepare_seconds)
        # Surface class names for parity with YOLO model attribute access.
        self.names = getattr(executor, "names", None)
        # Surface the executor's fixed input size — obb.py's _resolve_imgsz()
        # reads this for the NVDEC CUDA-tensor letterbox path, since this
        # adapter has no .overrides/.model.args for it to duck-type against
        # (those are ultralytics-model-shaped attributes this class doesn't
        # have). Without this, _resolve_imgsz() always fell back to a
        # hardcoded 1024, which silently mismatches any engine built at a
        # different imgsz (e.g. 640) and crashes TensorRT's setInputShape.
        self.imgsz = getattr(executor, "imgsz", None)

    def predict(
        self,
        frames: Any,
        *,
        conf: float = 1e-3,
        iou: float = 1.0,  # noqa: ARG002 - direct executor handles NMS internally
        classes: Any = None,
        verbose: bool = False,  # noqa: ARG002 - accepted for YOLO-call parity
        device: Any = None,  # noqa: ARG002 - executor is bound to its own device
        imgsz: Any = None,  # noqa: ARG002 - executor is bound to its own imgsz
        **_ignored: Any,
    ) -> Any:
        return self._executor.predict(
            list(frames),
            conf_thres=conf,
            classes=classes,
            max_det=self._max_det,
        )


# ---------------------------------------------------------------------------
# Public entry point.
# ---------------------------------------------------------------------------


def load_obb_executor(
    model_path: str,
    compute_runtime: str,
    *,
    auto_export: bool = True,
    max_det: int = _DEFAULT_MAX_DET,
    imgsz_override: int | None = None,
    task: str = "obb",
    batch_size: int = _DEFAULT_BATCH_SIZE,
) -> Any:
    """Load the OBB executor for a model path + compute runtime.

    Parameters
    ----------
    model_path:
        Path to a ``.pt`` source checkpoint, or an explicit ``.engine``
        artifact.
    compute_runtime:
        One of ``cpu``/``mps``/``cuda`` (→ plain PyTorch model) or
        ``tensorrt`` (→ direct TRT executor, auto-exporting from ``.pt`` on
        first load when ``auto_export``) or ``coreml`` (→ CoreML mlpackage).
        onnx_* runtimes are not supported for OBB — the production pipeline's
        tier→compute-runtime-string resolution never emits them.
    auto_export:
        When True (default), missing ``.engine`` artifacts are exported
        from the source ``.pt`` on first load. When False, a missing artifact
        raises :class:`ArtifactExportError` (NO silent PyTorch fallback — H4).
    max_det:
        Max detections fed to the direct executor's NMS (ignored for the torch
        runtimes).
    imgsz_override:
        When set (>0), export/load the artifact at this input size instead of
        the checkpoint's own embedded default (``_resolve_imgsz``). Needed for
        the sequential-OBB stage-2 (crop) model: its checkpoint may have been
        trained/exported at a different size than
        ``OBBSequentialConfig.stage2_image_size``, and the pipeline always
        pre-resizes crops to that configured size before inference -- an
        artifact built at the checkpoint's own size would silently receive
        wrongly-scaled input under gpu_fast (TensorRT/ONNX), which can drop
        every detection.
    task:
        ``"obb"`` (default) parses the raw head as OBB output
        (cx,cy,w,h,angle,conf) into ``Results(obb=...)``. Use ``"detect"`` for
        the sequential pipeline's stage-1 model, which is a plain (non-OBB)
        detector -- parsing its output as OBB silently misreads the
        class-score channel as an angle and yields ``Results.boxes is None``
        for every frame. Ignored for the torch runtimes (cpu/mps/cuda), whose
        underlying ultralytics model already knows its own task.
    batch_size:
        The number of frames/crops this executor will typically be called
        with per ``predict()`` call. ``1`` (default) exports/loads a static
        batch=1 TensorRT engine (unchanged from before). ``>1`` exports a
        TensorRT engine with a dynamic batch profile (min=1, opt=batch_size,
        max=batch_size) so a single engine handles the whole configured
        window in one inference call. Ignored for cpu/mps/cuda (torch
        already batches natively) and for coreml (OBB stays batch=1
        permanently on CoreML -- see ``_load_coreml_executor``).

    Returns
    -------
    A plain ultralytics ``YOLO`` model (cpu/mps/cuda) or a
    :class:`DirectExecutorAdapter` wrapping a TRT direct executor.
    """
    runtime = str(compute_runtime).strip().lower()

    if runtime in _TORCH_RUNTIMES:
        return _load_torch_executor(model_path, runtime)

    if runtime in _TENSORRT_RUNTIMES:
        return _load_direct_executor(
            model_path,
            runtime,
            auto_export=auto_export,
            max_det=max_det,
            imgsz_override=imgsz_override,
            task=task,
            batch_size=batch_size,
        )

    if runtime in _COREML_RUNTIMES:
        return _load_coreml_executor(
            model_path,
            auto_export=auto_export,
            imgsz_override=imgsz_override,
            task=task,
        )

    raise ArtifactExportError(f"Unsupported compute_runtime: {compute_runtime!r}")


def _load_torch_executor(model_path: str, runtime: str) -> Any:
    """Load a plain PyTorch YOLO model (byte-parity with previous ``_load_yolo``)."""
    model = _load_torch_model(model_path)
    if runtime == "cuda":
        model.to("cuda:0")
    elif runtime == "mps":
        model.to("mps")
    # cpu: no .to() call (matches previous behaviour and CPU byte-parity).
    return model


class _CoreMLBatchExecutor:
    """Batch-safe adapter around a CoreML ``YOLO(mlpackage)`` executor.

    The ultralytics CoreML backend (``AutoBackend`` -> ``CoreMLBackend``) is
    STATIC batch=1: ``CoreMLBackend.forward`` infers only ``im[0]`` and returns
    a single ``Results`` no matter how many images the batch holds, which then
    raises ``IndexError`` inside ultralytics' own ``stream_inference``
    (``self.results[i]`` for ``i >= 1``). Direct-mode OBB never trips this
    because CoreML is pinned to batch=1 and the direct window feeds
    ``detection_batch_size`` (=1 in practice) frames per call, but the
    sequential stage-2 batches crops (``stage2_batch_size``) and hands the
    CoreML executor multi-crop lists -> hard crash (observed as an empty-CSV
    run under the ``gpu_fast`` tier on Apple Silicon).

    This wrapper presents the same ``predict(list, **kw)`` batch API every other
    executor exposes, but dispatches CoreML one image at a time and concatenates
    the per-image ``Results``. A single-image call is a passthrough --
    byte-identical to the previous working batch=1 path -- and every other
    attribute (``.task``, ``.names``, ...) delegates to the wrapped model.
    """

    def __init__(
        self,
        model: Any,
        *,
        runtime_artifact_id: str | None = None,
        prepare_seconds: float = 0.0,
    ) -> None:
        self._model = model
        self.runtime_artifact_id = runtime_artifact_id
        self.runtime_artifact_prepare_seconds = float(prepare_seconds)

    def __getattr__(self, name: str) -> Any:
        # Only reached when normal lookup fails, so the real ``_model`` instance
        # attribute never routes here; ``__getattribute__`` guards recursion.
        return getattr(object.__getattribute__(self, "_model"), name)

    def predict(self, source: Any, **kwargs: Any) -> Any:
        if isinstance(source, (list, tuple)) and len(source) > 1:
            results: list = []
            for image in source:
                results.extend(self._model.predict([image], **kwargs))
            return results
        return self._model.predict(source, **kwargs)


def _load_coreml_executor(
    model_path: str,
    *,
    auto_export: bool,
    imgsz_override: int | None = None,
    task: str = "obb",
) -> Any:
    """Load (or auto-export) a CoreML ``.mlpackage`` and return a YOLO model.

    Uses a fixed ``imgsz`` export (``nms=False``) so CoreML sees a static input
    shape — avoiding the E5RT dynamic-shape failure that ``onnx_coreml`` hit.
    Apple Silicon only; will raise if coremltools is absent.

    ``imgsz_override``, when set (>0), exports/loads the artifact at this
    input size instead of the checkpoint's own embedded default — mirrors
    ``_load_direct_executor``'s handling so sequential-OBB's stage-2 (crop)
    model gets the same treatment on CoreML as it does on TensorRT/ONNX (see
    ``load_obb_executor``'s ``imgsz_override`` docstring for why).
    """
    resolved = Path(model_path).expanduser().resolve()
    suffix = resolved.suffix.lower()

    if suffix == ".mlpackage":
        if not resolved.exists():
            raise ArtifactExportError(
                f"CoreML artifact not found: {resolved}. "
                "Provide a valid .mlpackage or use a .pt source with auto_export=True."
            )
        return _CoreMLBatchExecutor(_load_torch_model(str(resolved)))

    imgsz = (
        int(imgsz_override)
        if imgsz_override and imgsz_override > 0
        else _resolve_imgsz(resolved)
    )
    prepared = _prepare_runtime_artifact(
        source_path=resolved,
        runtime="coreml",
        imgsz=imgsz,
        batch_size=_DEFAULT_BATCH_SIZE,
        task=task,
        auto_export=auto_export,
    )
    artifact_path = prepared.path
    logger.info(
        "%s CoreML artifact %s (%s, prepare %.3fs)",
        "Built" if prepared.built else "Reusing",
        prepared.identity.digest[:12],
        artifact_path.name,
        prepared.prepare_seconds,
    )

    return _CoreMLBatchExecutor(
        _load_torch_model(str(artifact_path)),
        runtime_artifact_id=prepared.identity.digest,
        prepare_seconds=prepared.prepare_seconds,
    )


def _load_direct_executor(
    model_path: str,
    compute_runtime: str,
    *,
    auto_export: bool,
    max_det: int,
    imgsz_override: int | None = None,
    task: str = "obb",
    batch_size: int = _DEFAULT_BATCH_SIZE,
) -> DirectExecutorAdapter:
    """Resolve (or auto-export) an ONNX/TRT artifact and wrap a direct executor."""
    runtime = _direct_runtime_name(compute_runtime)
    resolved = Path(model_path).expanduser().resolve()
    suffix = resolved.suffix.lower()

    # 1) Explicit artifact path supplied by the user: use as-is.
    if suffix in {".onnx", ".engine", ".trt"}:
        if not resolved.exists():
            raise ArtifactExportError(
                f"{runtime} artifact not found: {resolved}. "
                f"Provide a valid {_artifact_suffix(runtime)} file or use a .pt "
                f"source with auto_export=True."
            )
        # Resolve the input size for a prebuilt artifact: explicit override wins,
        # else the JSON freshness sidecar written at export time, else fail loudly
        # rather than silently letterboxing at the wrong size (H4).
        if imgsz_override and int(imgsz_override) > 0:
            imgsz = int(imgsz_override)
        else:
            meta = _meta_path(resolved)
            recorded = None
            if meta.exists():
                try:
                    recorded = int(
                        json.loads(meta.read_text(encoding="utf-8")).get("imgsz", 0)
                    )
                except Exception:
                    recorded = None
            if recorded and recorded > 0:
                imgsz = recorded
            else:
                raise ArtifactExportError(
                    f"Cannot determine input imgsz for prebuilt artifact {resolved.name}: "
                    f"no freshness sidecar recording it and no imgsz_override given. "
                    f"Pass imgsz_override (the size the engine was built at) or supply "
                    f"the .pt source with auto_export=True."
                )
        executor = _create_direct_executor(
            runtime=runtime,
            artifact_path=resolved,
            imgsz=imgsz,
            class_names=None,
            task=task,
        )
        return DirectExecutorAdapter(executor, max_det=max_det)

    # 2) Source .pt path: locate (or build) the derived artifact.
    imgsz = (
        int(imgsz_override)
        if imgsz_override and imgsz_override > 0
        else _resolve_imgsz(resolved)
    )
    prepared = _prepare_runtime_artifact(
        source_path=resolved,
        runtime=runtime,
        imgsz=imgsz,
        batch_size=batch_size,
        task=task,
        auto_export=auto_export,
    )
    artifact_path = prepared.path
    logger.info(
        "%s %s artifact %s (%s, prepare %.3fs)",
        "Built" if prepared.built else "Reusing",
        runtime,
        prepared.identity.digest[:12],
        artifact_path.name,
        prepared.prepare_seconds,
    )

    class_names = _model_class_names(resolved)
    executor = _create_direct_executor(
        runtime=runtime,
        artifact_path=artifact_path,
        imgsz=imgsz,
        class_names=class_names,
        task=task,
    )
    return DirectExecutorAdapter(
        executor,
        max_det=max_det,
        runtime_artifact_id=prepared.identity.digest,
        prepare_seconds=prepared.prepare_seconds,
    )
