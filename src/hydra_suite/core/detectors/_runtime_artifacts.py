"""ONNX and TensorRT runtime artifact management for YOLO detectors.

Provided as a mixin class so that ``YOLOOBBDetector`` can inherit these methods
cleanly. All methods depend only on ``self.params``, ``self.device``, and each
other — no YOLO model state is accessed.
"""

import hashlib
import json
import logging
import shutil
from contextlib import contextmanager
from pathlib import Path

logger = logging.getLogger(__name__)


class RuntimeArtifactMixin:
    """Mixin supplying ONNX/TensorRT artifact caching and export helpers."""

    _SESSION_ONNX_CPU_FALLBACK_ARTIFACTS: set[str] = set()
    _SESSION_TENSORRT_DISABLED_CONTEXTS: dict[str, str] = {}

    def _direct_obb_execution_mode(self) -> str:
        mode = str(self.params.get("YOLO_OBB_EXECUTION_MODE", "auto") or "auto")
        mode = mode.strip().lower()
        if mode in {"auto", "direct", "wrapper"}:
            return mode
        return "auto"

    def _clear_direct_obb_executor(self) -> None:
        self._direct_obb_executor = None

    def _maybe_enable_direct_obb_executor(
        self,
        runtime: str,
        artifact_path: Path,
        imgsz: int,
        *,
        class_names: dict[int, str] | None = None,
        class_count: int | None = None,
    ) -> None:
        self._direct_obb_executor = None
        if self._direct_obb_execution_mode() == "wrapper":
            return
        if not str(getattr(self, "device", "")).startswith("cuda"):
            return
        try:
            from hydra_suite.core.inference.direct_executors import (
                create_direct_obb_executor,
            )

            self._direct_obb_executor = create_direct_obb_executor(
                runtime=runtime,
                artifact_path=str(artifact_path),
                imgsz=int(imgsz),
                class_names=class_names,
                class_count=class_count,
            )
            logger.info(
                "Enabled direct %s OBB executor for %s",
                runtime,
                artifact_path.name,
            )
        except Exception as exc:
            self._direct_obb_executor = None
            logger.warning(
                "Direct %s OBB executor unavailable for %s: %s",
                runtime,
                artifact_path,
                exc,
            )

    def _maybe_enable_direct_cuda_obb_executor(
        self,
        pt_model,
        imgsz: int,
        *,
        class_names: dict[int, str] | None = None,
        class_count: int | None = None,
    ) -> None:
        """Enable a ``DirectPyTorchCUDAOBBExecutor`` for the pure CUDA runtime.

        The Ultralytics predictor uses ``LetterBox(auto=True)`` for PyTorch
        ``.pt`` models, which produces rectangular inputs for non-square frames
        (e.g. 576×1024 for a 1920×1080 feed instead of the 1024×1024 expected
        by TRT/ONNX artifacts).  This discrepancy causes different FPN
        activations and inconsistent detection scores across runtimes.

        Using a direct executor forces ``auto=False`` preprocessing so the model
        always sees a square ``imgsz×imgsz`` padded input — identical to what
        the TRT and ONNX direct executors send.
        """
        self._direct_obb_executor = None
        if self._direct_obb_execution_mode() == "wrapper":
            return
        if not str(getattr(self, "device", "")).startswith("cuda"):
            return
        try:
            from hydra_suite.core.inference.direct_executors import (
                create_direct_obb_executor,
            )

            self._direct_obb_executor = create_direct_obb_executor(
                runtime="cuda",
                artifact_path="",
                imgsz=int(imgsz),
                class_names=class_names,
                class_count=class_count,
                pt_model=pt_model,
            )
            logger.info(
                "Enabled direct PyTorch CUDA OBB executor (imgsz=%d); "
                "auto=False square letterboxing matches TRT/ONNX paths.",
                int(imgsz),
            )
        except Exception as exc:
            self._direct_obb_executor = None
            logger.warning("Direct PyTorch CUDA OBB executor unavailable: %s", exc)

    def _maybe_enable_direct_detect_executor(
        self,
        runtime: str,
        artifact_path: Path,
        imgsz: int,
        *,
        class_names: dict[int, str] | None = None,
        class_count: int | None = None,
    ) -> None:
        """Enable a direct ONNX/TRT executor for sequential stage-1 detect.

        Mirrors :meth:`_maybe_enable_direct_obb_executor` but creates a
        :class:`DirectONNXDetectExecutor` / :class:`DirectTensorRTDetectExecutor`
        so the detect step can run entirely on the GPU without the Ultralytics
        wrapper overhead.
        """
        self._direct_detect_executor = None
        if self._direct_obb_execution_mode() == "wrapper":
            return
        if not str(getattr(self, "device", "")).startswith("cuda"):
            return
        try:
            from hydra_suite.core.inference.direct_executors import (
                create_direct_detect_executor,
            )

            self._direct_detect_executor = create_direct_detect_executor(
                runtime=runtime,
                artifact_path=str(artifact_path),
                imgsz=int(imgsz),
                class_names=class_names,
                class_count=class_count,
            )
            logger.info(
                "Enabled direct %s detect executor for %s",
                runtime,
                artifact_path.name,
            )
        except Exception as exc:
            self._direct_detect_executor = None
            logger.warning(
                "Direct %s detect executor unavailable for %s: %s",
                runtime,
                artifact_path,
                exc,
            )

    def _is_realtime_workflow(self) -> bool:
        """Return whether the active tracking workflow is realtime."""
        raw_value = self.params.get(
            "TRACKING_REALTIME_MODE",
            self.params.get("TRACKING_WORKFLOW_MODE", "non_realtime"),
        )
        if isinstance(raw_value, str):
            return raw_value.strip().lower() == "realtime"
        return bool(raw_value)

    @staticmethod
    def _is_detection_runtime_task(task: str) -> bool:
        """Return True for detector tasks that benefit from realtime batch-1 artifacts."""
        return str(task or "").strip().lower() in {"obb", "detect"}

    def _realtime_detection_task_requires_batch1(self, task: str) -> bool:
        """Return whether realtime mode should clamp this task's artifact batch to 1."""
        if not self._is_realtime_workflow():
            return False
        task_name = str(task or "").strip().lower()
        if task_name == "detect":
            return True
        if task_name != "obb":
            return False
        obb_mode = str(self.params.get("YOLO_OBB_MODE", "direct")).strip().lower()
        return obb_mode != "sequential"

    def _should_export_raw_yolo_head(self, task: str) -> bool:
        """Whether deployment exports should bypass Ultralytics end-to-end postprocess."""
        task_name = str(task or "").strip().lower()
        if task_name not in {"obb", "detect"}:
            return False
        return bool(self.params.get("YOLO_EXPORT_RAW_HEAD", True))

    def _aux_export_profile_suffix(self, task: str) -> str:
        """Return a versioned artifact suffix for auxiliary task exports."""
        return "_rawheadv1" if self._should_export_raw_yolo_head(task) else ""

    def _runtime_build_batch_size_override(
        self,
        task: str,
        requested_batch_size: int | None = None,
    ) -> int | None:
        """Resolve optional task-specific build batch overrides for static runtimes."""
        task_name = str(task or "").strip().lower()
        override_keys: list[str] = []
        if task_name == "obb":
            obb_mode = str(self.params.get("YOLO_OBB_MODE", "direct")).strip().lower()
            if obb_mode == "sequential":
                override_keys.append("YOLO_SEQ_STAGE2_RUNTIME_BUILD_BATCH_SIZE")
            override_keys.append("YOLO_OBB_RUNTIME_BUILD_BATCH_SIZE")
        elif task_name == "detect":
            override_keys.append("YOLO_DETECT_RUNTIME_BUILD_BATCH_SIZE")
        elif task_name == "classify":
            override_keys.append("YOLO_CLASSIFY_RUNTIME_BUILD_BATCH_SIZE")

        for key in override_keys:
            raw_value = self.params.get(key, None)
            if raw_value in (None, "", 0, "0"):
                continue
            try:
                return max(1, int(raw_value))
            except (TypeError, ValueError):
                continue
        return requested_batch_size

    @contextmanager
    def _yolo_runtime_export_profile(self, base_model, task: str):
        """Temporarily disable export-time top-k postprocess for deployment artifacts."""
        head = None
        original_end2end = None
        if self._should_export_raw_yolo_head(task):
            try:
                head = base_model.model.model[-1]
                original_end2end = getattr(head, "end2end", None)
                if original_end2end is True:
                    head.end2end = False
                    logger.info(
                        "Exporting YOLO %s runtime artifact with raw head outputs (end2end=False).",
                        task,
                    )
            except Exception:
                head = None
                original_end2end = None
        try:
            yield
        finally:
            if head is not None and original_end2end is not None:
                try:
                    head.end2end = original_end2end
                except Exception:
                    pass

    # ------------------------------------------------------------------
    # ONNX resolution
    # ------------------------------------------------------------------

    def _resolve_onnx_imgsz(self, model_path: Path | None = None) -> int:
        """Resolve ONNX export/inference image size.

        Priority:
        1) Explicit `YOLO_ONNX_IMGSZ`
        2) Explicit `YOLO_IMGSZ`
        3) Model metadata from source .pt (`model.overrides['imgsz']` / `model.args['imgsz']`)
        4) Fallback 640
        """
        raw = self.params.get("YOLO_ONNX_IMGSZ", None)
        if raw is None and "YOLO_IMGSZ" in self.params:
            raw = self.params.get("YOLO_IMGSZ")

        imgsz = None
        if raw is not None:
            try:
                imgsz = int(raw)
            except Exception:
                imgsz = None

        if imgsz is None and model_path is not None and model_path.exists():
            try:
                from ultralytics import YOLO

                model = YOLO(str(model_path), task="obb")
                ov = getattr(model, "overrides", {}) or {}
                arg_imgsz = None
                try:
                    arg_imgsz = ov.get("imgsz")
                except Exception:
                    arg_imgsz = None
                if arg_imgsz is None:
                    margs = getattr(getattr(model, "model", None), "args", {}) or {}
                    if isinstance(margs, dict):
                        arg_imgsz = margs.get("imgsz")
                if arg_imgsz is not None:
                    imgsz = int(arg_imgsz)
            except Exception:
                imgsz = None

        if imgsz is None:
            imgsz = 640
        # Keep this aligned with practical YOLO defaults and export constraints.
        imgsz = max(64, min(4096, int(imgsz)))
        return imgsz

    def _resolve_onnx_build_batch_size(
        self,
        requested_batch_size: int | None = None,
        task: str = "obb",
    ) -> int:
        """Resolve the fixed ONNX export batch size.

        Realtime frame-level detector paths should use batch-1 artifacts to avoid
        padding single-frame inference into a larger static export profile.
        Sequential stage-2 crop OBB paths keep the configured batch size so crop
        grouping can still amortize runtime overhead.
        """
        default_batch = requested_batch_size
        if default_batch is None:
            default_batch = self.params.get("TENSORRT_MAX_BATCH_SIZE", 1)
        try:
            default_batch = max(1, int(default_batch or 1))
        except (TypeError, ValueError):
            default_batch = 1

        if self._realtime_detection_task_requires_batch1(task):
            return 1
        return default_batch

    # ------------------------------------------------------------------
    # TensorRT resolution helpers
    # ------------------------------------------------------------------

    def _resolve_tensorrt_build_batch_size(
        self,
        requested_batch_size: int | None = None,
        task: str = "obb",
    ) -> int:
        """Resolve the fixed TensorRT engine batch size."""
        default_batch = requested_batch_size
        if default_batch is None:
            default_batch = self.params.get("TENSORRT_MAX_BATCH_SIZE", 16)
        try:
            default_batch = max(1, int(default_batch or 1))
        except (TypeError, ValueError):
            default_batch = 1

        if self._realtime_detection_task_requires_batch1(task):
            return 1

        raw_override = self.params.get("TENSORRT_BUILD_BATCH_SIZE", None)
        if raw_override in (None, ""):
            from ._utils import _advanced_config_value

            raw_override = _advanced_config_value(
                self.params, "tensorrt_build_batch_size", None
            )
        if raw_override in (None, "", 0, "0"):
            return default_batch
        try:
            return max(1, int(raw_override))
        except (TypeError, ValueError):
            return default_batch

    def _resolve_tensorrt_workspace_gb(self) -> float:
        """Resolve TensorRT builder workspace limit in GB."""
        raw_value = self.params.get("TENSORRT_BUILD_WORKSPACE_GB", None)
        if raw_value in (None, ""):
            from ._utils import _advanced_config_value

            raw_value = _advanced_config_value(
                self.params, "tensorrt_build_workspace_gb", 4.0
            )
        try:
            return max(0.5, float(raw_value))
        except (TypeError, ValueError):
            return 4.0

    def _get_tensorrt_build_context(self) -> dict[str, str]:
        """Summarize device/runtime details for TensorRT build logging."""
        from hydra_suite.utils.gpu_utils import get_device_info

        gpu_name = str(self.device)
        cuda_version = "unknown"
        try:
            info = get_device_info()
        except Exception:
            info = {}
        if isinstance(info, dict):
            gpu_name = str(
                info.get("torch_cuda_device_name")
                or info.get("cuda_device_name")
                or gpu_name
            )
        try:
            import torch

            cuda_version = str(
                getattr(getattr(torch, "version", None), "cuda", None) or "unknown"
            )
        except Exception:
            cuda_version = "unknown"
        return {"gpu_name": gpu_name, "cuda_version": cuda_version}

    # ------------------------------------------------------------------
    # Artifact signature / metadata
    # ------------------------------------------------------------------

    def _artifact_signature(
        self, runtime: str, batch_size: int = 1, onnx_imgsz: int | None = None
    ) -> str:
        engine_model_id = self.params.get("ENGINE_MODEL_ID")
        inference_model_id = self.params.get("INFERENCE_MODEL_ID")
        if str(runtime) == "tensorrt" and engine_model_id:
            token = str(engine_model_id)
        elif inference_model_id:
            token = str(inference_model_id)
        else:
            token = str(
                self.params.get(
                    "YOLO_MODEL_PATH", getattr(self, "active_obb_model_path", "")
                )
            )
        runtime_profile = str(runtime)
        if str(runtime) == "onnx":
            # Keep ONNX export profile explicit in cache signature so profile changes
            # always trigger a rebuild of potentially incompatible artifacts.
            resolved_imgsz = int(onnx_imgsz or self._resolve_onnx_imgsz())
            raw_head_tag = (
                "_rawheadv1" if self._should_export_raw_yolo_head("obb") else ""
            )
            runtime_profile = (
                f"onnx_v4_static_imgsz{resolved_imgsz}_opset17_nosimplify{raw_head_tag}"
            )
        elif str(runtime) == "tensorrt":
            raw_head_tag = (
                "_rawheadv1" if self._should_export_raw_yolo_head("obb") else ""
            )
            runtime_profile = f"tensorrt_v3{raw_head_tag}"
        return hashlib.sha1(
            f"{token}|runtime={runtime_profile}|batch={int(batch_size)}".encode("utf-8")
        ).hexdigest()[:16]

    def _artifact_meta_path(self, artifact_path: Path) -> Path:
        return artifact_path.with_suffix(f"{artifact_path.suffix}.runtime_meta.json")

    @classmethod
    def _onnx_artifact_cache_key(cls, artifact_path: Path | str | None) -> str:
        if not artifact_path:
            return ""
        try:
            return str(Path(artifact_path).expanduser().resolve())
        except Exception:
            return str(artifact_path)

    @classmethod
    def _mark_onnx_artifact_for_cpu_fallback(
        cls, artifact_path: Path | str | None
    ) -> None:
        cache_key = cls._onnx_artifact_cache_key(artifact_path)
        if cache_key:
            cls._SESSION_ONNX_CPU_FALLBACK_ARTIFACTS.add(cache_key)

    @classmethod
    def _should_force_onnx_cpu_fallback(cls, artifact_path: Path | str | None) -> bool:
        cache_key = cls._onnx_artifact_cache_key(artifact_path)
        return bool(cache_key and cache_key in cls._SESSION_ONNX_CPU_FALLBACK_ARTIFACTS)

    @staticmethod
    def _attach_runtime_artifact_path(model, artifact_path: Path | str | None) -> None:
        if not artifact_path or model is None:
            return
        try:
            model._hydra_runtime_artifact_path = str(
                Path(artifact_path).expanduser().resolve()
            )
        except Exception:
            pass

    def _tensorrt_session_context_key(self) -> str:
        context = self._get_tensorrt_build_context()
        return "|".join(
            [
                str(self.device),
                str(context.get("gpu_name", "unknown")),
                str(context.get("cuda_version", "unknown")),
            ]
        )

    @classmethod
    def _is_fatal_tensorrt_environment_error(cls, error_msg: str) -> bool:
        text = str(error_msg or "").strip().lower()
        if not text:
            return False
        fatal_markers = (
            "cuda initialization failure",
            "createinferbuilder",
            "factory function returned nullptr",
            "pybind11::init()",
            "libnvinfer_builder_resource",
        )
        return any(marker in text for marker in fatal_markers)

    def _get_disabled_tensorrt_reason(self) -> str:
        return self._SESSION_TENSORRT_DISABLED_CONTEXTS.get(
            self._tensorrt_session_context_key(), ""
        )

    def _disable_tensorrt_for_session(self, error_msg: str) -> None:
        self._SESSION_TENSORRT_DISABLED_CONTEXTS[
            self._tensorrt_session_context_key()
        ] = str(error_msg or "TensorRT is unavailable in this session.").strip()

    def _artifact_is_fresh(self, artifact_path: Path, signature: str) -> bool:
        if not artifact_path.exists():
            return False
        data = self._read_artifact_meta(artifact_path)
        if not data:
            return False
        return str(data.get("signature", "")) == str(signature)

    def _read_artifact_meta(self, artifact_path: Path) -> dict:
        meta_path = self._artifact_meta_path(artifact_path)
        if not meta_path.exists():
            return {}
        try:
            data = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception:
            return {}
        return data if isinstance(data, dict) else {}

    def _write_artifact_meta(
        self, artifact_path: Path, signature: str, **extra_meta
    ) -> None:
        meta_path = self._artifact_meta_path(artifact_path)
        payload = {"signature": str(signature)}
        payload.update({str(k): v for k, v in extra_meta.items()})
        meta_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    # ------------------------------------------------------------------
    # ONNX model loading / export
    # ------------------------------------------------------------------

    def _try_load_onnx_model(self, model_path_str):
        """Try to load or export ONNX model for CPU runtime."""
        self.onnx_failure_reason = ""
        self._clear_direct_obb_executor()
        try:
            from ultralytics import YOLO

            self._onnx_predict_device = self.device
            resolved_model = Path(model_path_str).expanduser().resolve()
            requested_batch_size = self._runtime_build_batch_size_override(
                "obb",
                self.params.get("TENSORRT_MAX_BATCH_SIZE", 1),
            )
            onnx_batch_size = self._resolve_onnx_build_batch_size(
                requested_batch_size,
                task="obb",
            )
            if resolved_model.suffix.lower() == ".onnx":
                # User supplied explicit ONNX artifact path.
                onnx_path = resolved_model
                if not onnx_path.exists():
                    raise RuntimeError(f"ONNX model path not found: {onnx_path}")
                meta = self._read_artifact_meta(onnx_path)
                try:
                    meta_imgsz = int(meta.get("imgsz", 0))
                except Exception:
                    meta_imgsz = 0
                try:
                    meta_batch = int(meta.get("batch_size", 0))
                except Exception:
                    meta_batch = 0
                if meta_batch > 0:
                    onnx_batch_size = meta_batch
                else:
                    # Unknown exported batch dimension for user-supplied ONNX.
                    # Keep conservative default to avoid invalid-batch errors.
                    onnx_batch_size = 1
                onnx_imgsz = (
                    meta_imgsz
                    if meta_imgsz > 0
                    else self._resolve_onnx_imgsz(model_path=resolved_model)
                )
                logger.info(f"Loading ONNX model from: {onnx_path}")
                self.model = YOLO(str(onnx_path), task="obb")
                self._attach_runtime_artifact_path(self.model, onnx_path)
                self.use_onnx = True
                self.onnx_model_path = str(onnx_path)
                self.onnx_imgsz = int(onnx_imgsz)
                self.onnx_batch_size = int(onnx_batch_size)
                self._maybe_enable_direct_obb_executor(
                    "onnx",
                    onnx_path,
                    int(onnx_imgsz),
                    class_names=getattr(self.model, "names", None),
                )
                return
            else:
                onnx_path = resolved_model.with_name(
                    f"{resolved_model.stem}_b{onnx_batch_size}.onnx"
                )
            onnx_imgsz = self._resolve_onnx_imgsz(model_path=resolved_model)
            signature = self._artifact_signature(
                runtime="onnx",
                batch_size=int(onnx_batch_size),
                onnx_imgsz=onnx_imgsz,
            )
            force_cpu_fallback = str(
                self.device
            ).strip().lower() == "mps" and self._should_force_onnx_cpu_fallback(
                onnx_path
            )
            if force_cpu_fallback and onnx_path.exists():
                source_mtime_ns = 0
                artifact_mtime_ns = 0
                try:
                    source_mtime_ns = resolved_model.stat().st_mtime_ns
                    artifact_mtime_ns = onnx_path.stat().st_mtime_ns
                except Exception:
                    pass
                if artifact_mtime_ns >= source_mtime_ns:
                    meta = self._read_artifact_meta(onnx_path)
                    try:
                        meta_imgsz = int(meta.get("imgsz", 0))
                    except Exception:
                        meta_imgsz = 0
                    if meta_imgsz > 0:
                        onnx_imgsz = meta_imgsz
                    try:
                        meta_batch = int(meta.get("batch_size", 0))
                    except Exception:
                        meta_batch = 0
                    if meta_batch > 0:
                        onnx_batch_size = meta_batch
                    logger.info(
                        "Loading ONNX model from: %s (CPU ORT fallback after prior CoreML failure)",
                        onnx_path,
                    )
                    self.model = YOLO(str(onnx_path), task="obb")
                    self._attach_runtime_artifact_path(self.model, onnx_path)
                    self.use_onnx = True
                    self.onnx_model_path = str(onnx_path)
                    self.onnx_imgsz = int(onnx_imgsz)
                    self.onnx_batch_size = int(onnx_batch_size)
                    self._onnx_predict_device = "cpu"
                    return

            if self._artifact_is_fresh(onnx_path, signature):
                meta = self._read_artifact_meta(onnx_path)
                try:
                    meta_imgsz = int(meta.get("imgsz", 0))
                except Exception:
                    meta_imgsz = 0
                if meta_imgsz > 0:
                    onnx_imgsz = meta_imgsz
                try:
                    meta_batch = int(meta.get("batch_size", 0))
                except Exception:
                    meta_batch = 0
                if meta_batch > 0:
                    onnx_batch_size = meta_batch
                logger.info(f"Loading ONNX model from: {onnx_path}")
                self.model = YOLO(str(onnx_path), task="obb")
                self._attach_runtime_artifact_path(self.model, onnx_path)
                self.use_onnx = True
                self.onnx_model_path = str(onnx_path)
                self.onnx_imgsz = onnx_imgsz
                self.onnx_batch_size = int(onnx_batch_size)
                self._maybe_enable_direct_obb_executor(
                    "onnx",
                    onnx_path,
                    int(onnx_imgsz),
                    class_names=getattr(self.model, "names", None),
                )
                return

            logger.info("Exporting YOLO OBB model to ONNX runtime artifact...")
            base_model = YOLO(str(resolved_model), task="obb")
            with self._yolo_runtime_export_profile(base_model, task="obb"):
                export_path = base_model.export(
                    format="onnx",
                    imgsz=onnx_imgsz,
                    dynamic=False,
                    simplify=False,
                    nms=False,
                    opset=17,
                    batch=int(onnx_batch_size),
                    verbose=False,
                )
            # Release the PyTorch export model before creating the ORT session.
            # Keeping the PT model in GPU memory while ORT initialises its CUDA
            # execution provider can trigger an "Invalid device id" error because
            # both runtimes compete for the same CUDA device context.
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
                raise RuntimeError(f"ONNX export output missing: {out_path}")
            if out_path != onnx_path:
                shutil.copy2(str(out_path), str(onnx_path))
            self._write_artifact_meta(
                onnx_path,
                signature,
                imgsz=int(onnx_imgsz),
                batch_size=int(onnx_batch_size),
            )
            self.model = YOLO(str(onnx_path), task="obb")
            self._attach_runtime_artifact_path(self.model, onnx_path)
            self.use_onnx = True
            self.onnx_model_path = str(onnx_path)
            self.onnx_imgsz = onnx_imgsz
            self.onnx_batch_size = int(onnx_batch_size)
            self._maybe_enable_direct_obb_executor(
                "onnx",
                onnx_path,
                int(onnx_imgsz),
                class_names=getattr(self.model, "names", None),
            )
            logger.info(f"ONNX model ready: {onnx_path}")
        except Exception as e:
            self.onnx_failure_reason = str(e)
            logger.warning(f"ONNX runtime optimization failed: {e}")
            self.use_onnx = False

    # ------------------------------------------------------------------
    # TensorRT model loading / export
    # ------------------------------------------------------------------

    def _try_load_tensorrt_model(self, model_path_str):
        """Try to load or export TensorRT model for faster inference."""
        self.tensorrt_failure_reason = ""
        self._clear_direct_obb_executor()
        try:
            disabled_reason = self._get_disabled_tensorrt_reason()
            if disabled_reason:
                self.tensorrt_failure_reason = disabled_reason
                logger.warning(
                    "TensorRT is disabled for this session on %s; skipping engine build. Previous failure: %s",
                    self.device,
                    disabled_reason,
                )
                self.use_tensorrt = False
                return

            from ultralytics import YOLO

            requested_batch_size = self._runtime_build_batch_size_override(
                "obb",
                self.params.get("TENSORRT_MAX_BATCH_SIZE", 16),
            )
            build_batch_size = self._resolve_tensorrt_build_batch_size(
                requested_batch_size,
                task="obb",
            )
            build_workspace_gb = self._resolve_tensorrt_workspace_gb()
            build_context = self._get_tensorrt_build_context()

            resolved_model = Path(model_path_str).expanduser().resolve()
            engine_imgsz = 0
            if resolved_model.suffix.lower() in {".engine", ".trt"}:
                engine_path = resolved_model
                meta = self._read_artifact_meta(engine_path)
                try:
                    meta_batch = int(meta.get("batch_size", 0))
                except Exception:
                    meta_batch = 0
                if meta_batch > 0:
                    build_batch_size = meta_batch
                try:
                    engine_imgsz = int(meta.get("imgsz", 0))
                except Exception:
                    engine_imgsz = 0
            else:
                engine_path = resolved_model.with_name(
                    f"{resolved_model.stem}_b{int(build_batch_size)}.engine"
                )
                # Defer image-size resolution: for cache hits, read from stored
                # metadata; for rebuilds, derive from the loaded base model.
                # This avoids loading the .pt checkpoint twice (once here and
                # once again for the TRT export), which caused the test for
                # fatal-builder-failure retry-disabling to observe 2 init calls.
                engine_imgsz = 0
            signature = self._artifact_signature(
                runtime="tensorrt", batch_size=int(build_batch_size)
            )

            logger.info(
                "TensorRT engine cache check: path=%s | signature=%s | build_batch=%d | workspace=%.1f GB | gpu=%s | cuda=%s",
                engine_path,
                signature,
                int(build_batch_size),
                float(build_workspace_gb),
                build_context["gpu_name"],
                build_context["cuda_version"],
            )

            # Check if TensorRT engine already exists and matches current inference signature
            if self._artifact_is_fresh(engine_path, signature):
                logger.info("TensorRT engine cache hit: reusing cached engine.")
                try:
                    # Read imgsz from stored metadata (avoids loading the .pt
                    # source model just to retrieve the image size).
                    if engine_imgsz == 0:
                        _hit_meta = self._read_artifact_meta(engine_path)
                        try:
                            engine_imgsz = int(_hit_meta.get("imgsz", 0))
                        except Exception:
                            engine_imgsz = 0
                        if engine_imgsz == 0:
                            engine_imgsz = self._resolve_onnx_imgsz()
                    self.model = YOLO(str(engine_path), task="obb")
                    self.use_tensorrt = True
                    self.tensorrt_model_path = str(engine_path)
                    self.tensorrt_batch_size = (
                        build_batch_size  # Store batch size for chunking
                    )
                    self.tensorrt_imgsz = int(engine_imgsz or 640)
                    self._maybe_enable_direct_obb_executor(
                        "tensorrt",
                        engine_path,
                        int(self.tensorrt_imgsz),
                        class_names=getattr(self.model, "names", None),
                    )
                    logger.info(
                        "TensorRT engine reused successfully: path=%s | build_batch=%d",
                        engine_path,
                        int(build_batch_size),
                    )
                    return
                except Exception as e:
                    logger.warning(
                        "Failed to load cached TensorRT engine %s: %s",
                        engine_path,
                        e,
                    )
                    engine_path.unlink(missing_ok=True)

            logger.info(
                "TensorRT engine cache miss or stale metadata detected; rebuilding."
            )

            # Export to TensorRT
            logger.info("=" * 60)
            logger.info("BUILDING TENSORRT ENGINE - This is a one-time optimization")
            logger.info("This may take 1-5 minutes. Please wait...")
            logger.info("Engine path: %s", engine_path)
            logger.info("Engine signature: %s", signature)
            logger.info(
                "Build settings: batch=%d | workspace=%.1f GB | gpu=%s | cuda=%s",
                int(build_batch_size),
                float(build_workspace_gb),
                build_context["gpu_name"],
                build_context["cuda_version"],
            )
            logger.info("=" * 60)
            base_model = YOLO(model_path_str)
            base_model.to(self.device)
            # Resolve imgsz from the loaded model (avoids a separate YOLO init
            # in _resolve_onnx_imgsz when model_path was passed there).
            if engine_imgsz == 0:
                try:
                    _ov = getattr(base_model, "overrides", {}) or {}
                    _pt_imgsz = _ov.get("imgsz")
                    if _pt_imgsz:
                        engine_imgsz = int(_pt_imgsz)
                except Exception:
                    pass
                if engine_imgsz == 0:
                    engine_imgsz = self._resolve_onnx_imgsz()

            # Try dynamic batching first, fall back to static if it fails
            logger.info(
                "Building TensorRT engine (batch size: %d, workspace: %.1f GB)...",
                int(build_batch_size),
                float(build_workspace_gb),
            )

            # Export to TensorRT engine format using raw head outputs.
            # _yolo_runtime_export_profile forces head.end2end=False before
            # calling export() so all TRT engines are built with the one2many
            # raw CBC head (matching the CUDA DirectOBB inference path which
            # also disables end2end per _e2e_head_ref in _run_inference).
            # Note: dynamic=False uses fixed batch size which is more compatible
            # but requires batches to exactly match max_batch_size
            _export_kwargs: dict = dict(
                format="engine",
                device=self.device,
                half=True,  # Use FP16 for faster inference
                workspace=float(build_workspace_gb),
                dynamic=False,  # Static shapes (more compatible)
                batch=int(build_batch_size),  # Fixed batch size
                verbose=False,
            )
            with self._yolo_runtime_export_profile(base_model, task="obb"):
                export_path = base_model.export(**_export_kwargs)

            # Move exported engine to cache directory
            if Path(export_path).exists():
                exported_path = Path(export_path).expanduser().resolve()
                if exported_path != engine_path:
                    shutil.copy2(str(exported_path), str(engine_path))
                self._write_artifact_meta(
                    engine_path,
                    signature,
                    batch_size=int(build_batch_size),
                    imgsz=int(engine_imgsz or 640),
                    workspace_gb=float(build_workspace_gb),
                    gpu_name=build_context["gpu_name"],
                    cuda_version=build_context["cuda_version"],
                    end2end=False,  # always raw CBC — _yolo_runtime_export_profile forced it
                )
                logger.info(
                    "TensorRT engine rebuilt and cached: path=%s | signature=%s | end2end=False (raw head)",
                    engine_path,
                    signature,
                )

                # Load the TensorRT model
                self.model = YOLO(str(engine_path), task="obb")
                self.use_tensorrt = True
                self.tensorrt_model_path = str(engine_path)
                self.tensorrt_batch_size = build_batch_size  # Store for batching logic
                self.tensorrt_imgsz = int(engine_imgsz or 640)
                self._maybe_enable_direct_obb_executor(
                    "tensorrt",
                    engine_path,
                    int(self.tensorrt_imgsz),
                    class_names=getattr(self.model, "names", None),
                )
                logger.info("=" * 60)
                logger.info(
                    "TENSORRT OPTIMIZATION COMPLETE (batch=%d, workspace=%.1f GB)",
                    int(build_batch_size),
                    float(build_workspace_gb),
                )
                logger.info("=" * 60)
            else:
                logger.warning("TensorRT export failed - exported file not found")

        except Exception as e:
            error_msg = str(e)
            self.tensorrt_failure_reason = error_msg
            logger.warning(f"TensorRT optimization failed: {error_msg}")

            if self._is_fatal_tensorrt_environment_error(error_msg):
                self._disable_tensorrt_for_session(error_msg)
                logger.warning(
                    "Disabling TensorRT for the rest of this session on %s after a fatal builder initialization error.",
                    self.device,
                )

            # Provide helpful suggestions based on error type
            build_batch_size = self._resolve_tensorrt_build_batch_size(
                self.params.get("TENSORRT_MAX_BATCH_SIZE", 16),
                task="obb",
            )
            if "memory" in error_msg.lower() or "allocate" in error_msg.lower():
                logger.warning("=" * 60)
                logger.warning(
                    f"TensorRT build ran out of GPU memory (build batch = {build_batch_size})."
                )
                logger.warning(
                    "FIX: Reduce 'TensorRT Max Batch Size' in YOLO settings."
                )
                logger.warning(f"Try: 8, 4, or 1 instead of {build_batch_size}")
                logger.warning("=" * 60)
            elif "engine build failed" in error_msg.lower():
                logger.warning("=" * 60)
                logger.warning(
                    f"TensorRT engine build failed (build batch = {build_batch_size})."
                )
                logger.warning(
                    "FIX: Reduce 'TensorRT Max Batch Size' in YOLO settings."
                )
                logger.warning("=" * 60)

            logger.info("Continuing with standard PyTorch inference (still uses GPU)")
            self.use_tensorrt = False

    # ------------------------------------------------------------------
    # Ultralytics logging suppression
    # ------------------------------------------------------------------

    def _configure_ultralytics_logging(self):
        """Reduce per-frame Ultralytics runtime banners unless explicitly requested."""
        if bool(self.params.get("YOLO_VERBOSE_ULTRALYTICS", False)):
            return
        try:
            from ultralytics.utils import LOGGER as ULTRA_LOGGER

            ULTRA_LOGGER.setLevel(logging.WARNING)
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Auxiliary task artifact preparation
    # ------------------------------------------------------------------

    def _prepare_runtime_artifact_for_task(self, model_path_str: str, task: str) -> str:
        """Resolve/export runtime artifact for auxiliary YOLO tasks.

        This keeps sequential stage-1 detect/classify aligned with the selected
        MAT compute runtime. Explicit runtime artifacts (.onnx/.engine/.trt)
        are used as-is. For local .pt checkpoints, ONNX/TensorRT artifacts are
        exported lazily when requested by runtime flags.
        """
        if not model_path_str:
            return model_path_str

        # Built-in model aliases are loaded directly by ultralytics.
        if str(model_path_str).startswith(("yolo26", "yolov8", "yolov11")):
            return model_path_str

        model_path = Path(model_path_str).expanduser().resolve()
        if not model_path.exists() or not model_path.is_file():
            return model_path_str

        suffix = model_path.suffix.lower()
        if suffix in {".onnx", ".engine", ".trt"}:
            return str(model_path)
        if suffix != ".pt":
            return model_path_str

        try:
            from hydra_suite.utils.gpu_utils import (
                ONNXRUNTIME_AVAILABLE,
                TENSORRT_AVAILABLE,
            )

            enable_onnx_runtime = bool(self.params.get("ENABLE_ONNX_RUNTIME", False))
            enable_tensorrt = bool(self.params.get("ENABLE_TENSORRT", False)) and str(
                self.device
            ).startswith("cuda")

            # Make stage artifact names unambiguous when direct/crop/detect models differ.
            task_tag = str(task or "task").strip().lower().replace(" ", "_")

            disabled_tensorrt_reason = self._get_disabled_tensorrt_reason()

            from ultralytics import YOLO

            # Prefer ONNX when requested, matching primary OBB runtime preference.
            if enable_onnx_runtime and ONNXRUNTIME_AVAILABLE:
                requested_batch_size = (
                    self.params.get("TENSORRT_MAX_BATCH_SIZE", 1)
                    if self._is_detection_runtime_task(task)
                    else 1
                )
                requested_batch_size = self._runtime_build_batch_size_override(
                    task,
                    requested_batch_size,
                )
                onnx_batch_size = self._resolve_onnx_build_batch_size(
                    requested_batch_size,
                    task=task,
                )
                onnx_path = model_path.with_name(
                    f"{model_path.stem}_{task_tag}{self._aux_export_profile_suffix(task)}_b{int(onnx_batch_size)}.onnx"
                )
                needs_build = (not onnx_path.exists()) or (
                    onnx_path.stat().st_mtime_ns < model_path.stat().st_mtime_ns
                )
                if needs_build:
                    logger.info("Exporting %s model to ONNX runtime artifact...", task)
                    if task == "detect":
                        seq_detect_imgsz = int(
                            self.params.get("YOLO_SEQ_DETECT_IMGSZ", 0)
                        )
                        onnx_imgsz = (
                            seq_detect_imgsz
                            if seq_detect_imgsz > 0
                            else self._resolve_onnx_imgsz(model_path=model_path)
                        )
                    else:
                        onnx_imgsz = self._resolve_onnx_imgsz(model_path=model_path)
                    base_model = YOLO(str(model_path), task=task)
                    with self._yolo_runtime_export_profile(base_model, task=task):
                        export_path = base_model.export(
                            format="onnx",
                            imgsz=int(onnx_imgsz),
                            dynamic=False,
                            simplify=False,
                            nms=False,
                            opset=17,
                            batch=int(onnx_batch_size),
                            verbose=False,
                        )
                    out_path = Path(export_path).expanduser().resolve()
                    if out_path.exists() and out_path != onnx_path:
                        shutil.copy2(str(out_path), str(onnx_path))
                if onnx_path.exists():
                    return str(onnx_path)

            if enable_tensorrt and TENSORRT_AVAILABLE:
                if disabled_tensorrt_reason:
                    logger.warning(
                        "Skipping TensorRT runtime artifact build for %s model because TensorRT is disabled for this session: %s",
                        task,
                        disabled_tensorrt_reason,
                    )
                    return model_path_str
                requested_batch_size = (
                    self.params.get("TENSORRT_MAX_BATCH_SIZE", 1)
                    if self._is_detection_runtime_task(task)
                    else 1
                )
                requested_batch_size = self._runtime_build_batch_size_override(
                    task,
                    requested_batch_size,
                )
                build_batch_size = self._resolve_tensorrt_build_batch_size(
                    requested_batch_size,
                    task=task,
                )
                engine_path = model_path.with_name(
                    f"{model_path.stem}_{task_tag}{self._aux_export_profile_suffix(task)}_b{int(build_batch_size)}.engine"
                )
                needs_build = (not engine_path.exists()) or (
                    engine_path.stat().st_mtime_ns < model_path.stat().st_mtime_ns
                )
                if needs_build:
                    build_workspace_gb = self._resolve_tensorrt_workspace_gb()
                    logger.info(
                        "Building TensorRT runtime artifact for %s model (workspace=%.1f GB)...",
                        task,
                        float(build_workspace_gb),
                    )
                    base_model = YOLO(str(model_path), task=task)
                    base_model.to(self.device)
                    with self._yolo_runtime_export_profile(base_model, task=task):
                        export_path = base_model.export(
                            format="engine",
                            device=self.device,
                            half=True,
                            workspace=float(build_workspace_gb),
                            dynamic=False,
                            batch=int(build_batch_size),
                            verbose=False,
                        )
                    out_path = Path(export_path).expanduser().resolve()
                    if out_path.exists() and out_path != engine_path:
                        shutil.copy2(str(out_path), str(engine_path))
                if engine_path.exists():
                    return str(engine_path)
        except Exception as exc:
            if self._is_fatal_tensorrt_environment_error(str(exc)):
                self._disable_tensorrt_for_session(str(exc))
            logger.warning(
                "Aux runtime artifact preparation failed for %s model (%s). Using source checkpoint.",
                task,
                exc,
            )

        return model_path_str
