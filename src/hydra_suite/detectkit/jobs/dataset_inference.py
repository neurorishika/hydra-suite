"""Thin Qt worker coordinating protected, dataset-wide inference."""

from __future__ import annotations

import uuid
from dataclasses import asdict
from pathlib import Path

from PySide6.QtCore import Signal

from hydra_suite.runtime.process_supervisor import WorkloadStillOwnedError
from hydra_suite.runtime.safe_text import bounded_terminal_text
from hydra_suite.widgets.workers import BaseWorker

from ..sidecars.protocol import Operation, SidecarRequest
from ..sidecars.supervisor import ProtectedOperation
from .prediction_cache import (
    DatasetPredictionCache,
    PredictionPathIndex,
    cache_path_for,
    prediction_cache_key,
)


class DatasetInferenceWorker(BaseWorker):
    """Own a protected child; never import or construct a model in the GUI."""

    success = Signal(dict)

    def __init__(
        self,
        *,
        project_dir: str | Path,
        source_path: str | Path,
        model_path: str,
        device_preference: str,
        confidence_threshold: float,
        inference_kind: str = "obb_direct",
        secondary_model_path: str | None = None,
        crop_pad_ratio: float = 0.15,
        stage2_image_size: int = 160,
        slice_settings=None,
        imgsz_obb_direct: int = 640,
    ) -> None:
        super().__init__()
        self._cancel_requested = False
        self.recovery_cleanup_error = ""
        settings = {
            "inference_kind": str(inference_kind),
            "device": str(device_preference or "auto"),
            "confidence_threshold": float(confidence_threshold),
            "crop_pad_ratio": float(crop_pad_ratio),
            "stage2_image_size": int(stage2_image_size),
            "slice_settings": (
                slice_settings.to_dict()
                if slice_settings is not None
                else {"enabled": False}
            ),
            "imgsz_obb_direct": int(imgsz_obb_direct),
        }
        model_paths = [model_path]
        if secondary_model_path:
            model_paths.append(secondary_model_path)
        key = prediction_cache_key(source_path, model_paths, settings)
        base_path = cache_path_for(project_dir, key)
        request_id = uuid.uuid4().hex
        cache_path = base_path.with_name(f"{base_path.stem}-{request_id}.npz")
        payload = {
            **settings,
            "source_path": str(Path(source_path).expanduser().resolve()),
            "cache_path": str(cache_path),
            "model_path": str(Path(model_path).expanduser().resolve()),
            "secondary_model_path": (
                str(Path(secondary_model_path).expanduser().resolve())
                if secondary_model_path
                else ""
            ),
            "cache_key": asdict(key),
            "chunk_frames": 8,
            "max_targets": 300,
        }
        self.cache_key = key
        self.cache_path = cache_path
        self._operation = ProtectedOperation(
            SidecarRequest(request_id, Operation.DATASET_INFERENCE, payload),
            device=device_preference,
            input_paths=model_paths,
            cleanup_paths=(cache_path,),
        )

    def cancel(self) -> None:
        if self.containment_recovery_required:
            self.retry_containment_cleanup()
            return
        self._cancel_requested = True
        self._operation.cancel()

    def is_cancelled(self) -> bool:
        return self._cancel_requested or self._operation.cancelled

    @property
    def containment_recovery_required(self) -> bool:
        """Whether the sidecar remains the durable owner of a workload."""

        return isinstance(self.failure_exception, WorkloadStillOwnedError)

    def retry_containment_cleanup(self) -> bool:
        """Retry teardown without dropping an owner whose exit is unproven."""

        error = self.failure_exception
        if not isinstance(error, WorkloadStillOwnedError):
            return True
        try:
            error.sidecar.cancel()
        except WorkloadStillOwnedError as retry_error:
            retry_error.recovery_cleanup = error.recovery_cleanup
            self.failure_exception = retry_error
            return False
        except Exception as retry_error:  # noqa: BLE001 - retain uncertain owner
            error.recovery_error = bounded_terminal_text(
                retry_error, include_exception_type=False
            )
            return False
        if error.recovery_cleanup is not None:
            try:
                error.recovery_cleanup()
            except Exception as cleanup_error:  # noqa: BLE001 - workload is safe
                self.recovery_cleanup_error = bounded_terminal_text(
                    cleanup_error, include_exception_type=False
                )
        self.failure_exception = None
        return True

    def execute(self) -> None:
        outcome = self._operation.run(
            progress=lambda pct, message: (
                self.progress.emit(pct),
                self.status.emit(message),
            )
        )
        if not outcome.success:
            if outcome.canceled:
                self._cancel_requested = True
                self.status.emit("Inference cancelled.")
                return
            raise RuntimeError(outcome.message)
        try:
            cache = DatasetPredictionCache(self.cache_path, self.cache_key)
            path_index = PredictionPathIndex(self.cache_path)
            image_count = int(outcome.payload.get("image_count", -1))
            expected_coverage = ((0, image_count - 1),) if image_count else ()
            if (
                image_count < 0
                or not cache.is_valid()
                or cache.coverage_ranges() != expected_coverage
                or len(path_index) != image_count
            ):
                raise RuntimeError(
                    "inference sidecar returned an invalid prediction cache"
                )
        except BaseException:
            self._operation.discard_artifacts()
            raise
        self.success.emit(
            {
                **outcome.payload,
                "class_counts": {
                    int(key): int(value)
                    for key, value in dict(
                        outcome.payload.get("class_counts", {})
                    ).items()
                },
                "cache": cache,
                "path_index": path_index,
                "failure_kind": outcome.failure_kind,
                "peak_tree_rss_bytes": outcome.peak_tree_rss_bytes,
                "resource_telemetry": outcome.telemetry,
            }
        )
