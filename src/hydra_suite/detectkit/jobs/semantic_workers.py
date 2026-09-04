"""Thin Qt coordinators for protected SAM3 sidecars."""

from __future__ import annotations

import tempfile
import uuid
from dataclasses import fields
from pathlib import Path

from PySide6.QtCore import Signal

from hydra_suite.widgets.workers import BaseWorker

from ..sidecars.protocol import Operation, SidecarRequest
from ..sidecars.supervisor import ProtectedOperation
from .semantic_artifacts import read_frame_preview


def _device(params: dict) -> str:
    return str(params.get("device", "auto") or "auto")


class FramePreviewWorker(BaseWorker):
    result_ready = Signal(object)

    def __init__(self, sources, prompt, variant, params, labeler=None, parent=None):
        super().__init__(parent)
        if labeler is not None:
            raise ValueError("in-process semantic labelers are not accepted")
        self._cancel = False
        self._output_path = Path(tempfile.gettempdir()) / (
            f"hydra-semantic-preview-{uuid.uuid4().hex}.json"
        )
        payload = {
            "sources": [source.to_dict() for source in sources],
            "prompt": str(prompt),
            "variant": str(variant),
            "params": dict(params),
            "device": _device(params),
            "output_path": str(self._output_path),
        }
        self._operation = ProtectedOperation(
            SidecarRequest(uuid.uuid4().hex, Operation.SEMANTIC_PREVIEW, payload),
            device=_device(params),
            cleanup_paths=(self._output_path,),
        )

    def cancel(self) -> None:
        self._cancel = True
        self._operation.cancel()

    @property
    def cancelled(self) -> bool:
        return self._cancel or self._operation.cancelled

    def execute(self) -> None:
        outcome = self._operation.run(
            progress=lambda pct, msg: (
                self.progress.emit(pct),
                self.status.emit(msg),
            )
        )
        if outcome.canceled:
            self._cancel = True
            return
        if not outcome.success:
            raise RuntimeError(outcome.message)
        try:
            result = read_frame_preview(self._output_path)
        finally:
            self._output_path.unlink(missing_ok=True)
        self.result_ready.emit(result)


TilePreviewWorker = FramePreviewWorker


class CalibrationWorker(BaseWorker):
    result_ready = Signal(object)

    def __init__(
        self,
        sources,
        prompt,
        variant,
        params,
        labeler=None,
        parent=None,
        *,
        project_dir: str | Path | None = None,
    ) -> None:
        super().__init__(parent)
        if labeler is not None:
            raise ValueError("in-process semantic labelers are not accepted")
        self._cancel = False
        self.preview_frames = []
        self.sampled_frames: list[str] = []
        project_root = (
            Path(project_dir).expanduser().resolve()
            if project_dir is not None
            else Path(sources[0].path).expanduser().resolve().parent
        )
        payload = {
            "sources": [source.to_dict() for source in sources],
            "prompt": str(prompt),
            "variant": str(variant),
            "params": dict(params),
            "device": _device(params),
            "project_dir": str(project_root),
            "sample_budget": 12,
        }
        self._project_dir = project_root
        self._operation = ProtectedOperation(
            SidecarRequest(uuid.uuid4().hex, Operation.SEMANTIC_CALIBRATION, payload),
            device=_device(params),
        )

    def cancel(self) -> None:
        self._cancel = True
        self._operation.cancel()

    @property
    def cancelled(self) -> bool:
        return self._cancel or self._operation.cancelled

    def execute(self) -> None:
        from hydra_suite.core.inference.semantic.calibration import CalibrationPoint
        from hydra_suite.detectkit.gui.calibration_preview_store import (
            load_calibration_previews,
        )

        outcome = self._operation.run(
            progress=lambda pct, msg: (
                self.progress.emit(pct),
                self.status.emit(msg),
            )
        )
        if outcome.canceled:
            self._cancel = True
            self.result_ready.emit([])
            return
        if not outcome.success:
            raise RuntimeError(outcome.message)
        self.sampled_frames = [
            str(path) for path in outcome.payload.get("sampled_frames", [])
        ]
        artifact = str(outcome.payload.get("preview_artifact", "") or "")
        self.preview_frames = (
            load_calibration_previews(self._project_dir, artifact) if artifact else []
        )
        self.result_ready.emit(
            [CalibrationPoint(**raw) for raw in outcome.payload.get("points", [])]
        )


class SemanticEscalationWorker(BaseWorker):
    result_ready = Signal(object)
    project_mutated = Signal()

    def __init__(self, request, labeler=None, parent=None) -> None:
        super().__init__(parent)
        if labeler is not None:
            raise ValueError("in-process semantic labelers are not accepted")
        self._request = request
        self._cancel = False
        excluded = {
            "project",
            "source_names",
            "source_paths",
            "variant",
            "prompt",
            "class_name",
        }
        params = {
            field.name: getattr(request, field.name)
            for field in fields(request)
            if field.name not in excluded
        }
        payload = {
            "project_dir": str(
                Path(request.project.project_dir).expanduser().resolve()
            ),
            "source_names": list(request.source_names),
            "source_paths": list(request.source_paths),
            "variant": str(request.variant),
            "prompt": str(request.prompt),
            "class_name": str(request.class_name),
            "params": params,
            "device": _device(params),
        }
        self._operation = ProtectedOperation(
            SidecarRequest(uuid.uuid4().hex, Operation.SEMANTIC_ESCALATION, payload),
            device=_device(params),
        )

    def cancel(self) -> None:
        self._cancel = True
        self._operation.cancel()

    @property
    def cancelled(self) -> bool:
        return self._cancel or self._operation.cancelled

    def execute(self) -> None:
        from hydra_suite.detectkit.gui.models import OBBSource

        from .semantic_escalation import SemanticEscalationResult

        outcome = self._operation.run(
            progress=lambda pct, msg: (
                self.progress.emit(pct),
                self.status.emit(msg),
            )
        )
        if outcome.canceled:
            self._cancel = True
            return
        if not outcome.success:
            raise RuntimeError(outcome.message)
        updated = {
            source.path: source
            for source in (
                OBBSource.from_dict(raw) for raw in outcome.payload.get("sources", [])
            )
        }
        for source in self._request.project.sources:
            replacement = updated.get(source.path)
            if replacement is not None:
                source.staged_review = replacement.staged_review
        self.project_mutated.emit()
        raw_result = dict(outcome.payload.get("semantic_result") or {})
        raw_result["skipped"] = [tuple(item) for item in raw_result.get("skipped", [])]
        self.result_ready.emit(SemanticEscalationResult(**raw_result))
