"""Background workers for single-image and bulk pose prediction in PoseKit."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from PySide6.QtCore import QObject, Signal

from hydra_suite.core.inference.api import load_pose_backend
from hydra_suite.integrations.sleap.service import PoseInferenceService

from .utils import _maybe_empty_cuda_cache

logger = logging.getLogger("pose_label")


def _build_pose_backend(
    *,
    backend_family: str,
    model_path: str,
    exported_model_path: str,
    compute_runtime: str,
    min_valid_conf: float,
    batch_size: int,
    conf: float,
    keypoint_names: List[str],
    skeleton_edges: List[Tuple[int, int]],
    out_root: str,
    sleap_env: Optional[str],
    sleap_batch: Optional[int] = None,
    sleap_max_instances: int = 1,
) -> Any:
    """Construct a pose backend via the canonical ``load_pose_backend`` shim.

    Delegates the entire runtime-flavor decision to
    ``core/inference/api.load_pose_backend`` (which routes through
    ``stages/pose.load_pose_model`` — the single source of the pose runtime
    golden rule), instead of duplicating a CUDA/ONNX runtime-flavor ladder
    here. ``load_pose_model`` still honors the SLEAP-flavor debug override
    internally.

    ``compute_runtime`` is the tier-resolved pose runtime flavor string (e.g.
    ``"cpu"``, ``"mps"``, ``"cuda"``, ``"tensorrt_cuda"``, ``"coreml"``) as
    produced by ``MainWindow._pred_runtime_flavor`` (tier -> RuntimeResolver ->
    flavor).
    """
    return load_pose_backend(
        backend_family=backend_family,
        model_path=model_path,
        compute_runtime=compute_runtime,
        keypoint_names=list(keypoint_names),
        skeleton_edges=skeleton_edges,
        confidence_threshold=conf,
        batch_size=max(1, int(batch_size)),
        min_valid_confidence=min_valid_conf,
        out_root=out_root,
        exported_model_path=exported_model_path,
        sleap_env=sleap_env,
        sleap_batch=sleap_batch,
        sleap_max_instances=sleap_max_instances,
    )


class PosePredictWorker(QObject):
    """Background worker for one-image pose prediction."""

    finished = Signal(list)
    failed = Signal(str)
    resolved_exported_model_signal = Signal(str)

    def __init__(
        self,
        model_path: Path,
        image_path: Path,
        out_root: Path,
        keypoint_names: List[str],
        skeleton_edges: Optional[List[Tuple[int, int]]] = None,
        backend: str = "yolo",
        runtime_flavor: str = "auto",
        exported_model_path: Optional[Path] = None,
        device: str = "auto",
        imgsz: int = 640,
        conf: float = 0.25,
        yolo_batch: int = 4,
        sleap_env: Optional[str] = None,
        sleap_device: str = "auto",
        sleap_batch: int = 4,
        sleap_max_instances: int = 1,
        cache_backend: Optional[str] = None,
    ):
        super().__init__()
        self.model_path = Path(model_path)
        self.image_path = Path(image_path)
        self.out_root = Path(out_root)
        self.keypoint_names = list(keypoint_names)
        self.skeleton_edges = list(skeleton_edges or [])
        self.num_kpts = len(self.keypoint_names)
        self.backend = (backend or "yolo").lower()
        self.runtime_flavor = (runtime_flavor or "auto").lower()
        self.exported_model_path = (
            Path(exported_model_path) if exported_model_path else None
        )
        self.device = device
        self.imgsz = int(imgsz)
        self.conf = float(conf)
        self.yolo_batch = int(max(1, yolo_batch))
        self.sleap_env = sleap_env
        self.sleap_device = sleap_device
        self.sleap_batch = int(sleap_batch)
        # Enforce single-instance predictions for PoseKit.
        self.sleap_max_instances = 1
        self.cache_backend = str(cache_backend or self.backend).strip().lower()

    def _resolved_runtime_artifact_path(self, backend_obj: Any) -> str:
        # Only SLEAP flavors that actually export an artifact (onnx/tensorrt)
        # have a resolved path worth surfacing; native SLEAP (Apple GPU/GPU-Fast)
        # and the YOLO backend (always native .pt/.onnx/.engine passed through
        # as-is, never auto-exported here) have nothing new to report.
        rt = str(self.runtime_flavor or "").strip().lower()
        is_apple_like = rt in ("mps", "coreml") or rt.startswith(
            ("onnx_mps", "onnx_coreml")
        )
        if self.backend != "sleap" or is_apple_like:
            return ""
        for attr in ("exported_model_path", "model_path"):
            value = getattr(backend_obj, attr, None)
            if value:
                return str(value)
        return ""

    def run(self) -> None:
        """Run inference and emit either predicted keypoints or an error."""
        try:
            infer = PoseInferenceService(
                self.out_root, self.keypoint_names, self.skeleton_edges
            )
            cached = infer.get_cached_pred(
                self.model_path, self.image_path, backend=self.cache_backend
            )
            if cached is not None:
                self.finished.emit(cached)
                return

            # Preferred path: the shared load_pose_backend shim (single source
            # of the pose runtime golden rule via
            # core/inference/stages/pose.py::load_pose_model), not a hand-rolled
            # runtime-flavor ladder.
            try:
                backend = _build_pose_backend(
                    backend_family=self.backend,
                    model_path=str(self.model_path),
                    exported_model_path=(
                        str(self.exported_model_path)
                        if self.exported_model_path is not None
                        else ""
                    ),
                    compute_runtime=self.runtime_flavor,
                    min_valid_conf=0.0,
                    batch_size=self.yolo_batch,
                    conf=float(self.conf),
                    keypoint_names=self.keypoint_names,
                    skeleton_edges=self.skeleton_edges,
                    out_root=str(self.out_root),
                    sleap_env=self.sleap_env,
                    sleap_batch=int(max(1, self.sleap_batch)),
                    sleap_max_instances=1,
                )
                resolved_path = self._resolved_runtime_artifact_path(backend)
                if resolved_path:
                    self.resolved_exported_model_signal.emit(resolved_path)
                try:
                    # NOTE: no backend.warmup() here -- load_pose_backend
                    # (-> stages/pose.load_pose_model) already warms the
                    # backend it returns. A redundant second warmup() breaks
                    # the SLEAP service backend's _service_started_here
                    # ownership tracking and leaks the service subprocess
                    # past close().
                    img = cv2.imread(str(self.image_path))
                    if img is None:
                        raise RuntimeError(f"Failed to read image: {self.image_path}")
                    out = backend.predict_batch([img])
                    pose = out[0] if out else None
                    if pose is None or pose.keypoints is None:
                        self.finished.emit([])
                    else:
                        arr = np.asarray(pose.keypoints, dtype=np.float32)
                        self.finished.emit(
                            [(float(x), float(y), float(c)) for x, y, c in arr.tolist()]
                        )
                    return
                finally:
                    try:
                        backend.close()
                    except Exception:
                        pass
            except Exception as exc:
                if self.backend == "sleap":
                    raise RuntimeError(
                        "SLEAP shared runtime path failed in PoseKit. "
                        "Legacy fallback is disabled for parity with MAT. "
                        f"Original error: {exc}"
                    ) from exc
                # Fallback to legacy PoseInferenceService path.
                logger.debug(
                    "Shared runtime predict path failed; falling back to legacy path.",
                    exc_info=True,
                )

            preds_map, err = infer.predict(
                self.model_path,
                [self.image_path],
                device=self.device,
                imgsz=self.imgsz,
                conf=self.conf,
                batch=1,
                progress_cb=None,
                cancel_cb=None,
                backend=self.backend,
                sleap_env=self.sleap_env,
                sleap_device=self.sleap_device,
                sleap_batch=self.sleap_batch,
                sleap_max_instances=self.sleap_max_instances,
                sleap_runtime_flavor=self.runtime_flavor,
                sleap_exported_model_path=(
                    str(self.exported_model_path)
                    if self.exported_model_path is not None
                    else None
                ),
            )
            if preds_map is None:
                self.failed.emit(err or "Prediction failed.")
                return
            preds = preds_map.get(str(self.image_path)) or preds_map.get(
                str(self.image_path.resolve())
            )
            if preds is None:
                preds = [(0.0, 0.0, 0.0) for _ in range(len(self.keypoint_names))]
            self.finished.emit(preds)
        except Exception as e:
            _maybe_empty_cuda_cache()
            self.failed.emit(str(e))


class BulkPosePredictWorker(QObject):
    """Background worker for multi-image pose prediction."""

    progress = Signal(int, int)
    finished = Signal(dict)
    failed = Signal(str)
    resolved_exported_model_signal = Signal(str)

    def __init__(
        self,
        model_path: Path,
        image_paths: List[Path],
        out_root: Path,
        keypoint_names: List[str],
        skeleton_edges: Optional[List[Tuple[int, int]]] = None,
        backend: str = "yolo",
        runtime_flavor: str = "auto",
        exported_model_path: Optional[Path] = None,
        device: str = "auto",
        imgsz: int = 640,
        conf: float = 0.25,
        batch: int = 16,
        sleap_env: Optional[str] = None,
        sleap_device: str = "auto",
        sleap_batch: int = 4,
        sleap_max_instances: int = 1,
        cache_backend: Optional[str] = None,
    ):
        super().__init__()
        self.model_path = Path(model_path)
        self.image_paths = list(image_paths)
        self.out_root = Path(out_root)
        self.keypoint_names = list(keypoint_names)
        self.skeleton_edges = list(skeleton_edges or [])
        self.backend = (backend or "yolo").lower()
        self.runtime_flavor = (runtime_flavor or "auto").lower()
        self.exported_model_path = (
            Path(exported_model_path) if exported_model_path else None
        )
        self.device = device
        self.imgsz = int(imgsz)
        self.conf = float(conf)
        self.batch = int(batch)
        self.sleap_env = sleap_env
        self.sleap_device = sleap_device
        self.sleap_batch = int(sleap_batch)
        # Enforce single-instance predictions for PoseKit.
        self.sleap_max_instances = 1
        self.cache_backend = str(cache_backend or self.backend).strip().lower()
        self._cancel = False

    def _resolved_runtime_artifact_path(self, backend_obj: Any) -> str:
        # Only SLEAP flavors that actually export an artifact (onnx/tensorrt)
        # have a resolved path worth surfacing; native SLEAP (Apple GPU/GPU-Fast)
        # and the YOLO backend (always native .pt/.onnx/.engine passed through
        # as-is, never auto-exported here) have nothing new to report.
        rt = str(self.runtime_flavor or "").strip().lower()
        is_apple_like = rt in ("mps", "coreml") or rt.startswith(
            ("onnx_mps", "onnx_coreml")
        )
        if self.backend != "sleap" or is_apple_like:
            return ""
        for attr in ("exported_model_path", "model_path"):
            value = getattr(backend_obj, attr, None)
            if value:
                return str(value)
        return ""

    def cancel(self) -> None:
        """Request cancellation for the running prediction batch."""
        self._cancel = True

    def run(self) -> None:
        """Run batch inference and stream progress updates."""
        try:
            infer = PoseInferenceService(
                self.out_root, self.keypoint_names, self.skeleton_edges
            )

            # Preferred path: the shared load_pose_backend shim (single source
            # of the pose runtime golden rule via
            # core/inference/stages/pose.py::load_pose_model) with chunked image
            # loading, not a hand-rolled runtime-flavor ladder.
            try:
                backend = _build_pose_backend(
                    backend_family=self.backend,
                    model_path=str(self.model_path),
                    exported_model_path=(
                        str(self.exported_model_path)
                        if self.exported_model_path is not None
                        else ""
                    ),
                    compute_runtime=self.runtime_flavor,
                    min_valid_conf=0.0,
                    batch_size=int(max(1, self.batch)),
                    conf=float(self.conf),
                    keypoint_names=self.keypoint_names,
                    skeleton_edges=self.skeleton_edges,
                    out_root=str(self.out_root),
                    sleap_env=self.sleap_env,
                    sleap_batch=int(max(1, self.sleap_batch)),
                    sleap_max_instances=1,
                )
                resolved_path = self._resolved_runtime_artifact_path(backend)
                if resolved_path:
                    self.resolved_exported_model_signal.emit(resolved_path)
                try:
                    # NOTE: no backend.warmup() here -- load_pose_backend
                    # (-> stages/pose.load_pose_model) already warms the
                    # backend it returns. A redundant second warmup() breaks
                    # the SLEAP service backend's _service_started_here
                    # ownership tracking and leaks the service subprocess
                    # past close().
                    preds: Dict[str, List[Tuple[float, float, float]]] = {}
                    total = len(self.image_paths)
                    done = 0
                    chunk_size = (
                        int(max(1, self.sleap_batch))
                        if self.backend == "sleap"
                        else int(max(1, self.batch))
                    )

                    for i in range(0, total, chunk_size):
                        if self._cancel:
                            self.failed.emit("Canceled.")
                            return
                        chunk_paths = self.image_paths[i : i + chunk_size]
                        images = []
                        valid_paths = []
                        for p in chunk_paths:
                            img = cv2.imread(str(p))
                            if img is None:
                                preds[str(p)] = []
                                continue
                            images.append(img)
                            valid_paths.append(p)
                        if images:
                            out = backend.predict_batch(images)
                            for j, p in enumerate(valid_paths):
                                pose = out[j] if j < len(out) else None
                                if pose is None or pose.keypoints is None:
                                    preds[str(p)] = []
                                    continue
                                arr = np.asarray(pose.keypoints, dtype=np.float32)
                                preds[str(p)] = [
                                    (float(x), float(y), float(c))
                                    for x, y, c in arr.tolist()
                                ]
                        done += len(chunk_paths)
                        self.progress.emit(done, total)

                    self.finished.emit(preds)
                    return
                finally:
                    try:
                        backend.close()
                    except Exception:
                        pass
            except Exception as exc:
                if self.backend == "sleap":
                    raise RuntimeError(
                        "SLEAP shared runtime bulk path failed in PoseKit. "
                        "Legacy fallback is disabled for parity with MAT. "
                        f"Original error: {exc}"
                    ) from exc
                logger.debug(
                    "Shared runtime bulk path failed; falling back to legacy path.",
                    exc_info=True,
                )

            preds, err = infer.predict(
                self.model_path,
                self.image_paths,
                device=self.device,
                imgsz=self.imgsz,
                conf=self.conf,
                batch=self.batch,
                progress_cb=lambda d, t: self.progress.emit(d, t),
                cancel_cb=lambda: self._cancel,
                backend=self.backend,
                sleap_env=self.sleap_env,
                sleap_device=self.sleap_device,
                sleap_batch=self.sleap_batch,
                sleap_max_instances=self.sleap_max_instances,
                sleap_runtime_flavor=self.runtime_flavor,
                sleap_exported_model_path=(
                    str(self.exported_model_path)
                    if self.exported_model_path is not None
                    else None
                ),
            )
            if preds is None:
                self.failed.emit(err or "Prediction failed.")
                return
            self.finished.emit(preds)
        except Exception as e:
            _maybe_empty_cuda_cache()
            self.failed.emit(str(e))


class SleapServiceWorker(QObject):
    """Worker that starts and validates the SLEAP backend service."""

    finished = Signal(bool, str, str)

    def __init__(self, env_name: str, out_root: Path) -> None:
        super().__init__()
        self.env_name = env_name
        self.out_root = Path(out_root)

    def run(self) -> None:
        """Start SLEAP service and emit status tuple."""
        try:
            ok, err, log_path = PoseInferenceService.start_sleap_service(
                self.env_name, self.out_root
            )
            self.finished.emit(bool(ok), str(err or ""), str(log_path or ""))
        except Exception as e:
            self.finished.emit(False, str(e), "")
