"""Background validation worker for DetectKit model evaluation."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Signal

from hydra_suite.widgets.workers import BaseWorker

from ..evaluation import (
    EvaluationCandidate,
    EvaluationResult,
    evaluate_candidate,
    new_evaluation_id,
)


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

    def cancel(self) -> None:
        self._cancel_requested = True

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
                f"Validation started for {candidate.run_id} on "
                f"{candidate.dataset_yaml}"
            )
            try:
                result = evaluate_candidate(
                    candidate,
                    output_root=self._output_root,
                    device=self._device,
                    batch=self._batch,
                    evaluation_id=evaluation_id,
                )
            except Exception as exc:  # noqa: BLE001 - preserve comparison batch
                result = EvaluationResult.failed(candidate, evaluation_id, str(exc))
                self.log_signal.emit(f"Validation failed for {candidate.run_id}: {exc}")
            else:
                self.log_signal.emit(
                    f"Validation complete for {candidate.run_id}: "
                    f"mAP50={result.map50:.3f}, "
                    f"mAP50-95={result.map50_95:.3f}"
                )
            self.result_ready.emit(result)
            self.progress.emit(int(index * 100 / max(1, total)))

        if self.is_cancelled():
            self.status.emit("Evaluation cancelled between runs.")
