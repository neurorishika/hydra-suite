"""Background validation worker for DetectKit model evaluation."""

from __future__ import annotations

import uuid
from dataclasses import asdict
from pathlib import Path

from PySide6.QtCore import Signal

from hydra_suite.widgets.workers import BaseWorker

from ..evaluation import EvaluationCandidate, EvaluationResult, new_evaluation_id
from ..sidecars.protocol import Operation, SidecarRequest
from ..sidecars.supervisor import ProtectedOperation


class EvaluationWorker(BaseWorker):
    """Evaluate selected training runs sequentially outside the GUI thread."""

    result_ready = Signal(object)
    log_signal = Signal(str)

    def __init__(
        self,
        candidates: list[EvaluationCandidate],
        *,
        output_root: str | Path,
        device: str,
        batch: int,
    ) -> None:
        super().__init__()
        self._candidates = list(candidates)
        self._output_root = Path(output_root)
        self._device = str(device)
        self._batch = max(1, int(batch))
        self._cancel_requested = False
        self._operation: ProtectedOperation | None = None

    def cancel(self) -> None:
        self._cancel_requested = True
        if self._operation is not None:
            self._operation.cancel()

    def is_cancelled(self) -> bool:
        return bool(self._cancel_requested)

    def execute(self) -> None:
        total = len(self._candidates)
        for index, candidate in enumerate(self._candidates, start=1):
            if self.is_cancelled():
                break
            evaluation_id = new_evaluation_id(candidate.run_id)
            self.status.emit(
                f"Evaluating {index}/{total}: {candidate.run_id} ({candidate.role})"
            )
            self.log_signal.emit(
                f"Validation started for {candidate.run_id} on {candidate.dataset_yaml}"
            )
            operation = ProtectedOperation(
                SidecarRequest(
                    uuid.uuid4().hex,
                    Operation.EVALUATION,
                    {
                        "candidate": asdict(candidate),
                        "output_root": str(self._output_root.expanduser().resolve()),
                        "device": self._device,
                        "batch": self._batch,
                        "evaluation_id": evaluation_id,
                    },
                ),
                device=self._device,
                input_paths=(candidate.model_path, candidate.dataset_yaml),
                cleanup_paths=(self._output_root / evaluation_id,),
            )
            self._operation = operation
            outcome = operation.run(
                progress=lambda pct, message, index=index: (
                    self.progress.emit(
                        int(((index - 1) + pct / 100) * 100 / max(1, total))
                    ),
                    self.status.emit(message),
                ),
                log=self.log_signal.emit,
            )
            self._operation = None
            if outcome.canceled:
                self._cancel_requested = True
                break
            if not outcome.success:
                result = EvaluationResult.failed(
                    candidate, evaluation_id, outcome.message
                )
                self.log_signal.emit(
                    f"Validation failed for {candidate.run_id} "
                    f"[{outcome.failure_kind}]: {outcome.message}"
                )
            else:
                raw_result = outcome.payload.get("evaluation_result")
                if not isinstance(raw_result, dict):
                    raise RuntimeError(
                        "evaluation sidecar returned no bounded metrics record"
                    )
                result = EvaluationResult(**raw_result)
                self.log_signal.emit(
                    f"Validation complete for {candidate.run_id}: "
                    f"mAP50={result.map50:.3f}, "
                    f"mAP50-95={result.map50_95:.3f}"
                )
            self.result_ready.emit(result)
            self.progress.emit(int(index * 100 / max(1, total)))

        if self.is_cancelled():
            self.status.emit("Evaluation cancelled between runs.")
