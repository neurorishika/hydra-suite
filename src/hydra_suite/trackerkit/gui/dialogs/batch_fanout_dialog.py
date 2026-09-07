"""Per-job progress table for a parallel batch run.

Every slot here is driven by a queued signal from ``BatchFanoutWorker``, so
this class only ever runs on the GUI thread. It is deliberately non-modal: the
children are separate processes and the user may keep working while they run.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QDialogButtonBox,
    QHeaderView,
    QLabel,
    QProgressBar,
    QTableWidget,
    QTableWidgetItem,
)

from hydra_suite.trackerkit.batch_plan import BatchJobSpec
from hydra_suite.widgets.dialogs import BaseDialog

_COLS = ("#", "Video", "GPU", "Status", "Progress", "Last message")
_COL_GPU = 2
_COL_STATUS = 3
_COL_PROGRESS = 4
_COL_MESSAGE = 5

# Statuses that mean "this job never reached a terminal state of its own",
# i.e. the scheduler never emitted job_finished for it.
_UNRESOLVED = {"queued", "running"}


class BatchFanoutDialog(BaseDialog):
    """Live job table for :func:`run_batch_fanout`."""

    cancel_requested = Signal()

    def __init__(self, specs: Sequence[BatchJobSpec], parent=None) -> None:
        super().__init__(
            "Parallel Batch Tracking", parent, buttons=QDialogButtonBox.Cancel
        )
        self.setModal(False)
        self.resize(900, 420)
        self._rows: dict[int, int] = {}
        self._log_paths: dict[int, str] = {}
        self._finished = False
        self._header = QLabel(f"{len(specs)} videos queued.")
        self.add_content(self._header)
        self._table = QTableWidget(len(specs), len(_COLS))
        self._table.setHorizontalHeaderLabels(_COLS)
        self._table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        self._table.horizontalHeader().setSectionResizeMode(
            _COL_MESSAGE, QHeaderView.Stretch
        )
        self._table.setEditTriggers(QTableWidget.NoEditTriggers)
        for row, spec in enumerate(specs):
            self._rows[int(spec.index)] = row
            self._table.setItem(row, 0, QTableWidgetItem(str(spec.index)))
            self._table.setItem(row, 1, QTableWidgetItem(Path(spec.video_path).name))
            self._table.setItem(row, _COL_GPU, QTableWidgetItem("-"))
            self._table.setItem(row, _COL_STATUS, QTableWidgetItem("queued"))
            bar = QProgressBar()
            bar.setRange(0, 100)
            self._table.setCellWidget(row, _COL_PROGRESS, bar)
            self._table.setItem(row, _COL_MESSAGE, QTableWidgetItem(""))
        self.add_content(self._table)
        # The button box's Cancel must stop the RUN, not close the window and
        # abandon live children; BaseDialog wired it straight to reject().
        self._buttons.rejected.disconnect(self.reject)
        self._buttons.rejected.connect(self._on_cancel_clicked)

    # --- cancellation ------------------------------------------------------
    def reject(self) -> None:
        """Route every close path (button box, window X, Esc) through cancel.

        ``QDialog.reject`` is what the title-bar close button and Escape call,
        so without this override closing the window mid-run would hide the job
        table while the child processes kept going, with no way back to them.
        """
        if self._finished:
            super().reject()
            return
        self._header.setText("Cancelling… waiting for children to stop.")
        self.cancel_requested.emit()

    def _on_cancel_clicked(self) -> None:
        self.reject()

    # --- table plumbing ----------------------------------------------------
    def _set(self, index: int, col: int, text: str) -> None:
        row = self._rows.get(int(index))
        if row is None:
            return
        item = self._table.item(row, col)
        if item is None:
            item = QTableWidgetItem()
            self._table.setItem(row, col, item)
        item.setText(text)

    def _status(self, index: int) -> str:
        row = self._rows.get(int(index))
        if row is None:
            return ""
        item = self._table.item(row, _COL_STATUS)
        return item.text() if item is not None else ""

    def _show_close_button(self) -> None:
        self._finished = True
        self._buttons.clear()
        self._buttons.addButton(QDialogButtonBox.Close)

    # --- worker slots ------------------------------------------------------
    def on_job_started(
        self, index: int, video: str, log_path: str, gpu: str = "-"
    ) -> None:
        self._log_paths[int(index)] = log_path
        self._set(index, _COL_GPU, gpu or "-")
        self._set(index, _COL_STATUS, "running")

    def on_job_progress(self, index: int, percent: int, message: str) -> None:
        row = self._rows.get(int(index))
        if row is None:
            return
        bar = self._table.cellWidget(row, _COL_PROGRESS)
        if bar is not None:
            bar.setValue(max(0, min(100, int(percent))))
        self._set(index, _COL_MESSAGE, message)

    def on_job_finished(self, index: int, ok: bool, text: str) -> None:
        if ok:
            status = "OK"
        elif str(text).strip().lower() == "cancelled":
            status = "cancelled"
        else:
            status = "FAILED"
        self._set(index, _COL_STATUS, status)
        self._set(index, _COL_MESSAGE, text)

    def on_fanout_finished(self, result) -> None:
        # A job that never launched (halt-on-failure, or a cancel that drained
        # the queue) gets no job_finished event, so its row would sit at
        # "queued" forever. The result carries a verdict for EVERY spec.
        for job in result.jobs:
            index = int(job.spec.index)
            if self._status(index).lower() not in _UNRESOLVED:
                continue
            self._set(
                index, _COL_STATUS, "cancelled" if result.cancelled else "not started"
            )
            self._set(index, _COL_MESSAGE, str(job.error or ""))
        ok = sum(1 for job in result.jobs if job.success)
        suffix = " (cancelled)" if result.cancelled else ""
        self._header.setText(
            f"Done: {ok}/{len(result.jobs)} videos succeeded{suffix}. "
            "Logs are in each video's <stem>_logs/ folder."
        )
        self._show_close_button()

    def on_fanout_error(self, message: str) -> None:
        """The worker itself raised: nothing more is coming, so let it close."""
        for index in self._rows:
            if self._status(index).lower() in _UNRESOLVED:
                self._set(index, _COL_STATUS, "aborted")
        self._header.setText(f"Parallel batch failed: {message}")
        self._show_close_button()
