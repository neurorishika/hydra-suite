"""OBB runtime-artifact loading: ONNX/TensorRT/CoreML auto-export + direct executor.

This is a CLEAN port of the load + auto-export + direct-executor selection logic
from the legacy ``core/detectors/_runtime_artifacts.py`` (``_try_load_onnx_model``,
``_try_load_tensorrt_model``, ``_maybe_enable_direct_obb_executor``,
``_maybe_enable_direct_cuda_obb_executor``) and ``core/detectors/_direct_obb_runtime.py``
(``create_direct_obb_executor``).

It deliberately does NOT import from ``core/detectors`` — the legacy code is a
mixin entangled with detector state (``self.params``, ``self.device``,
session-disable tracking, CoreML CPU-fallback bookkeeping, batch-override
machinery). Here we port only the standalone essence needed by the new
inference pipeline:

  * ``load_obb_executor(model_path, compute_runtime, *, auto_export)``
      - cpu/mps/cuda → returns a plain PyTorch (ultralytics ``YOLO``) model,
        ``.to()``-moved to the device. This keeps CPU/MPS behaviour byte-identical
        to the previous ``_load_yolo``.
      - onnx_*/tensorrt → loads (or auto-exports then loads) the ``.onnx``/``.engine``
        artifact and returns a direct CUDA executor wrapped in a YOLO-compatible
        adapter so the geometry-extraction stage is unchanged.
      - onnx_*/tensorrt with ``auto_export=False`` and a missing artifact →
        raises :class:`ArtifactExportError` (a CLEAR error, never a silent
        PyTorch fallback — that silent fallback was parity-audit finding H4).
      - coreml → exports (or reuses) a ``.mlpackage`` via ultralytics CoreML
        export with fixed ``imgsz`` (avoids the dynamic-shape E5RT failure seen
        with ``onnx_coreml``) then loads it back via ``YOLO(mlpackage_path)``
        (Apple Silicon only).

Square-letterbox parity: the direct executors (ported in
``core/detectors/_direct_obb_runtime.py``) use ``LetterBox(auto=False)`` so the
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
import shutil
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Compute-runtime → direct-executor runtime name.
_ONNX_RUNTIMES: frozenset[str] = frozenset({"onnx_cpu", "onnx_cuda", "onnx_coreml"})
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
    """Export a ``.pt`` model to an ``.onnx``/``.engine`` artifact.

    Ported from legacy ``_try_load_onnx_model`` / ``_try_load_tensorrt_model``
    export blocks. Runs ultralytics ``model.export(...)`` (CUDA-only for TRT) and
    copies the exported file to ``artifact_path``. ``runtime`` is the direct
    runtime name (``"onnx"`` or ``"tensorrt"``).

    This function only runs on a machine with ultralytics (and, for TRT, a CUDA
    device); the tests inject a fake in its place.
    """
    from ultralytics import YOLO

    base_model = YOLO(str(pt_path))
    # CoreML does not use the CBC direct executor, so skip the raw-head override.
    if runtime != "coreml":
        # Force raw-head (end2end=False) export so the direct executor's NMS path
        # matches the TRT/ONNX raw-CBC contract (legacy _yolo_runtime_export_profile).
        _force_raw_head(base_model)

    if runtime == "onnx":
        logger.info(
            "Exporting YOLO OBB model to ONNX runtime artifact (imgsz=%d)...", imgsz
        )
        export_path = base_model.export(
            format="onnx",
            imgsz=imgsz,
            dynamic=False,
            simplify=False,
            nms=False,
            opset=17,
            batch=int(batch_size),
            verbose=False,
        )
    elif runtime == "tensorrt":
        logger.info(
            "Building TensorRT OBB engine (imgsz=%d, batch=%d) — one-time export...",
            imgsz,
            batch_size,
        )
        export_path = base_model.export(
            format="engine",
            device="cuda:0",
            half=True,
            dynamic=False,
            batch=int(batch_size),
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
    if out_path != artifact_path:
        if out_path.is_dir():
            # .mlpackage is a directory — use copytree.
            if artifact_path.exists():
                shutil.rmtree(str(artifact_path))
            shutil.copytree(str(out_path), str(artifact_path))
        else:
            shutil.copy2(str(out_path), str(artifact_path))
    return artifact_path


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
) -> Any:
    """Create a direct ONNX/TRT OBB executor (square-letterbox preprocessing).

    Delegates to the ported ``core/detectors/_direct_obb_runtime`` factory. The
    executors there use ``LetterBox(auto=False)`` — identical square-letterbox
    preprocessing across the PyTorch-CUDA, ONNX, and TRT paths, which is the
    parity guarantee enforced by legacy ``_maybe_enable_direct_cuda_obb_executor``.
    """
    from hydra_suite.core.detectors._direct_obb_runtime import (
        create_direct_obb_executor,
    )

    return create_direct_obb_executor(
        runtime=runtime,
        artifact_path=str(artifact_path),
        imgsz=int(imgsz),
        class_names=class_names,
    )


# ---------------------------------------------------------------------------
# Artifact path + freshness bookkeeping (clean port of legacy meta logic).
# ---------------------------------------------------------------------------


def _direct_runtime_name(compute_runtime: str) -> str:
    """Map a compute-runtime to the direct-executor runtime name."""
    if compute_runtime in _ONNX_RUNTIMES:
        return "onnx"
    if compute_runtime in _TENSORRT_RUNTIMES:
        return "tensorrt"
    raise ArtifactExportError(
        f"compute_runtime {compute_runtime!r} is not an ONNX/TensorRT runtime"
    )


def _artifact_suffix(runtime: str) -> str:
    if runtime == "onnx":
        return ".onnx"
    if runtime == "coreml":
        return ".mlpackage"
    return ".engine"


def _artifact_path_for(pt_path: Path, runtime: str) -> Path:
    """Derive the artifact path for a ``.pt`` source + direct runtime name."""
    pt_path = Path(pt_path)
    suffix = _artifact_suffix(runtime)
    if runtime == "coreml":
        # CoreML uses a bare stem (no batch suffix) since batch is always 1.
        return pt_path.with_suffix(".mlpackage")
    return pt_path.with_name(f"{pt_path.stem}_b{_DEFAULT_BATCH_SIZE}{suffix}")


def _meta_path(artifact_path: Path) -> Path:
    return artifact_path.with_suffix(f"{artifact_path.suffix}.runtime_meta.json")


def _write_fresh_marker(artifact_path: Path, source_pt: Path) -> None:
    """Write a freshness marker recording the source ``.pt`` mtime.

    Mirrors legacy ``_write_artifact_meta`` — used so a subsequent load can tell
    whether the cached artifact is still valid for the current source model.
    """
    try:
        source_mtime_ns = source_pt.stat().st_mtime_ns
    except Exception:
        source_mtime_ns = 0
    _meta_path(artifact_path).write_text(
        json.dumps({"source_mtime_ns": source_mtime_ns}), encoding="utf-8"
    )


def _artifact_is_fresh(artifact_path: Path, source_pt: Path) -> bool:
    """Return True when ``artifact_path`` exists and is newer than its source.

    Clean analogue of legacy ``_artifact_is_fresh``: a cached artifact is reused
    only when it exists and was built from the current ``.pt`` (by recorded
    source mtime). Missing/stale markers force a rebuild. Handles both file
    artifacts (.onnx/.engine) and directory artifacts (.mlpackage).
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
    return recorded == current


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

    def __init__(self, executor: Any, *, max_det: int = _DEFAULT_MAX_DET) -> None:
        self._executor = executor
        self._max_det = int(max_det)
        # Surface class names for parity with YOLO model attribute access.
        self.names = getattr(executor, "names", None)

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
) -> Any:
    """Load the OBB executor for a model path + compute runtime.

    Parameters
    ----------
    model_path:
        Path to a ``.pt`` source checkpoint, or an explicit ``.onnx``/``.engine``
        artifact.
    compute_runtime:
        One of ``cpu``/``mps``/``cuda`` (→ plain PyTorch model) or
        ``onnx_cpu``/``onnx_cuda``/``onnx_coreml``/``tensorrt`` (→ direct
        executor, auto-exporting from ``.pt`` on first load when ``auto_export``).
    auto_export:
        When True (default), missing ``.onnx``/``.engine`` artifacts are exported
        from the source ``.pt`` on first load. When False, a missing artifact
        raises :class:`ArtifactExportError` (NO silent PyTorch fallback — H4).
    max_det:
        Max detections fed to the direct executor's NMS (ignored for the torch
        runtimes).

    Returns
    -------
    A plain ultralytics ``YOLO`` model (cpu/mps/cuda) or a
    :class:`DirectExecutorAdapter` wrapping an ONNX/TRT direct executor.
    """
    runtime = str(compute_runtime).strip().lower()

    if runtime in _TORCH_RUNTIMES:
        return _load_torch_executor(model_path, runtime)

    if runtime in _ONNX_RUNTIMES or runtime in _TENSORRT_RUNTIMES:
        return _load_direct_executor(
            model_path, runtime, auto_export=auto_export, max_det=max_det
        )

    if runtime in _COREML_RUNTIMES:
        return _load_coreml_executor(model_path, auto_export=auto_export)

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


def _load_coreml_executor(model_path: str, *, auto_export: bool) -> Any:
    """Load (or auto-export) a CoreML ``.mlpackage`` and return a YOLO model.

    Uses a fixed ``imgsz`` export (``nms=False``) so CoreML sees a static input
    shape — avoiding the E5RT dynamic-shape failure that ``onnx_coreml`` hit.
    Apple Silicon only; will raise if coremltools is absent.
    """
    resolved = Path(model_path).expanduser().resolve()
    suffix = resolved.suffix.lower()

    if suffix == ".mlpackage":
        if not resolved.exists():
            raise ArtifactExportError(
                f"CoreML artifact not found: {resolved}. "
                "Provide a valid .mlpackage or use a .pt source with auto_export=True."
            )
        return _load_torch_model(str(resolved))

    artifact_path = _artifact_path_for(resolved, "coreml")

    if _artifact_is_fresh(artifact_path, resolved):
        logger.info("Reusing cached CoreML artifact: %s", artifact_path.name)
    else:
        if not auto_export:
            raise ArtifactExportError(
                "compute_runtime='coreml' requested but no fresh .mlpackage exists "
                f"for {resolved.name} and auto_export=False. "
                "Provide a prebuilt .mlpackage or enable auto_export."
            )
        imgsz = _resolve_imgsz(resolved)
        _export_artifact(
            pt_path=resolved,
            artifact_path=artifact_path,
            runtime="coreml",
            imgsz=imgsz,
            batch_size=_DEFAULT_BATCH_SIZE,
        )
        _write_fresh_marker(artifact_path, resolved)
        logger.info("Exported CoreML artifact: %s", artifact_path)

    return _load_torch_model(str(artifact_path))


def _load_direct_executor(
    model_path: str,
    compute_runtime: str,
    *,
    auto_export: bool,
    max_det: int,
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
        imgsz = _DEFAULT_IMGSZ
        executor = _create_direct_executor(
            runtime=runtime, artifact_path=resolved, imgsz=imgsz, class_names=None
        )
        return DirectExecutorAdapter(executor, max_det=max_det)

    # 2) Source .pt path: locate (or build) the derived artifact.
    artifact_path = _artifact_path_for(resolved, runtime)

    if _artifact_is_fresh(artifact_path, resolved):
        logger.info("Reusing cached %s OBB artifact: %s", runtime, artifact_path.name)
    else:
        if not auto_export:
            raise ArtifactExportError(
                f"compute_runtime={compute_runtime!r} requested but no fresh "
                f"{_artifact_suffix(runtime)} artifact exists for {resolved.name} "
                f"and auto_export=False. Provide a prebuilt "
                f"{_artifact_suffix(runtime)} (point model_path at it) or enable "
                f"auto_export (CUDA box) — refusing to silently fall back to "
                f"PyTorch (H4)."
            )
        imgsz = _resolve_imgsz(resolved)
        _export_artifact(
            pt_path=resolved,
            artifact_path=artifact_path,
            runtime=runtime,
            imgsz=imgsz,
            batch_size=_DEFAULT_BATCH_SIZE,
        )
        _write_fresh_marker(artifact_path, resolved)
        logger.info("Exported %s OBB artifact: %s", runtime, artifact_path)

    imgsz = _resolve_imgsz(resolved)
    class_names = _model_class_names(resolved)
    executor = _create_direct_executor(
        runtime=runtime,
        artifact_path=artifact_path,
        imgsz=imgsz,
        class_names=class_names,
    )
    return DirectExecutorAdapter(executor, max_det=max_det)
