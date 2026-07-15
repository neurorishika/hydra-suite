"""OBB runtime-artifact loading: TensorRT/CoreML auto-export + direct executor.

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
(runtime_to_compute_runtime) only emits {cpu, mps, cuda, tensorrt, coreml}.

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
    """Export a ``.pt`` model to a ``.engine``/``.mlpackage`` artifact.

    Ported from legacy ``_try_load_tensorrt_model`` export blocks. Runs
    ultralytics ``model.export(...)`` (CUDA-only for TRT) and copies the
    exported file to ``artifact_path``. ``runtime`` is the direct runtime name
    (``"tensorrt"`` or ``"coreml"``).

    This function only runs on a machine with ultralytics (and, for TRT, a CUDA
    device); the tests inject a fake in its place.
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
    task: str = "obb",
) -> Any:
    """Create a direct ONNX/TRT executor (square-letterbox preprocessing).

    Delegates to the ported ``core/detectors/_direct_obb_runtime`` factory. The
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
    from hydra_suite.core.detectors._direct_obb_runtime import (
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
    """Map a compute-runtime to the direct-executor runtime name."""
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


def _write_fresh_marker(artifact_path: Path, source_pt: Path, imgsz: int) -> None:
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
    _meta_path(artifact_path).write_text(
        json.dumps({"source_mtime_ns": source_mtime_ns, "imgsz": int(imgsz)}),
        encoding="utf-8",
    )


def _artifact_is_fresh(artifact_path: Path, source_pt: Path, imgsz: int) -> bool:
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
    # Older markers (pre-imgsz tracking) omit "imgsz" -- treat as stale so they
    # get rebuilt once and gain the field, rather than silently trusting them.
    return int(data.get("imgsz", -1)) == int(imgsz)


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
        onnx_* runtimes are not supported for OBB — the production pipeline
        (runtime_to_compute_runtime) never emits them.
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
            model_path, auto_export=auto_export, imgsz_override=imgsz_override
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


def _load_coreml_executor(
    model_path: str, *, auto_export: bool, imgsz_override: int | None = None
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
        return _load_torch_model(str(resolved))

    artifact_path = _artifact_path_for(resolved, "coreml")
    imgsz = (
        int(imgsz_override)
        if imgsz_override and imgsz_override > 0
        else _resolve_imgsz(resolved)
    )

    if _artifact_is_fresh(artifact_path, resolved, imgsz):
        logger.info("Reusing cached CoreML artifact: %s", artifact_path.name)
    else:
        if not auto_export:
            raise ArtifactExportError(
                "compute_runtime='coreml' requested but no fresh .mlpackage exists "
                f"for {resolved.name} and auto_export=False. "
                "Provide a prebuilt .mlpackage or enable auto_export."
            )
        _export_artifact(
            pt_path=resolved,
            artifact_path=artifact_path,
            runtime="coreml",
            imgsz=imgsz,
            batch_size=_DEFAULT_BATCH_SIZE,
        )
        _write_fresh_marker(artifact_path, resolved, imgsz)
        logger.info("Exported CoreML artifact: %s", artifact_path)

    return _load_torch_model(str(artifact_path))


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
        imgsz = _DEFAULT_IMGSZ
        executor = _create_direct_executor(
            runtime=runtime,
            artifact_path=resolved,
            imgsz=imgsz,
            class_names=None,
            task=task,
        )
        return DirectExecutorAdapter(executor, max_det=max_det)

    # 2) Source .pt path: locate (or build) the derived artifact.
    artifact_path = _artifact_path_for(resolved, runtime, batch_size=batch_size)
    imgsz = (
        int(imgsz_override)
        if imgsz_override and imgsz_override > 0
        else _resolve_imgsz(resolved)
    )

    if _artifact_is_fresh(artifact_path, resolved, imgsz):
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
        _export_artifact(
            pt_path=resolved,
            artifact_path=artifact_path,
            runtime=runtime,
            imgsz=imgsz,
            batch_size=batch_size,
        )
        _write_fresh_marker(artifact_path, resolved, imgsz)
        logger.info("Exported %s OBB artifact: %s", runtime, artifact_path)

    class_names = _model_class_names(resolved)
    executor = _create_direct_executor(
        runtime=runtime,
        artifact_path=artifact_path,
        imgsz=imgsz,
        class_names=class_names,
        task=task,
    )
    return DirectExecutorAdapter(executor, max_det=max_det)
