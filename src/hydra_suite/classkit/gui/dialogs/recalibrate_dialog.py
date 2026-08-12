"""RecalibrateDialog — pick a model + val set and refit calibration in place.

Thin Qt wiring around ``training.calibration_fit.recalibrate_artifact`` (the
Qt-free, tested substance). Mirrors ClassKit's ``QRunnable`` + ``TaskSignals``
background-task pattern.
"""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QObject, QRunnable, QThreadPool, Signal
from PySide6.QtWidgets import (
    QDialogButtonBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from hydra_suite.widgets.dialogs import BaseDialog

from ..project import classkit_export_dir


class _RecalibrateSignals(QObject):
    """Signals for the background recalibration task."""

    finished = Signal()
    success = Signal(object)  # CalibrationResult
    error = Signal(str)


class _RecalibrateWorker(QRunnable):
    """Runs ``recalibrate_artifact`` off the UI thread."""

    def __init__(self, model_path: str, val_dir: str) -> None:
        super().__init__()
        self.setAutoDelete(True)
        self.model_path = model_path
        self.val_dir = val_dir
        self.signals = _RecalibrateSignals()

    def run(self) -> None:
        try:
            from hydra_suite.training.calibration_fit import recalibrate_artifact

            result = recalibrate_artifact(self.model_path, self.val_dir)
            self.signals.success.emit(result)
        except Exception as exc:  # noqa: BLE001 - surfaced to the dialog log
            self.signals.error.emit(str(exc))
        finally:
            self.signals.finished.emit()


class RecalibrateDialog(BaseDialog):
    """Pick a classifier artifact + a validation source, refit calibration,
    and report the resulting ECE.

    ``project_path``, if given, is used to default the validation source to
    ``classkit_export_dir(project_path)/val`` when it exists.
    """

    def __init__(
        self, project_path: Path | None = None, parent: QWidget | None = None
    ) -> None:
        super().__init__(
            "Recalibrate Model",
            parent=parent,
            buttons=QDialogButtonBox.Close,
        )
        self.setMinimumWidth(560)
        self._project_path = Path(project_path) if project_path else None
        self._worker: _RecalibrateWorker | None = None

        content = QWidget()
        layout = QVBoxLayout(content)
        layout.setSpacing(10)

        layout.addWidget(
            QLabel(
                "Refit temperature-scaling calibration for a trained classifier "
                "against a labeled validation set. Model weights are unchanged; "
                "only the stored calibration temperature/ECE are updated."
            )
        )

        model_row = QHBoxLayout()
        model_row.addWidget(QLabel("Model:"))
        self.model_edit = QLineEdit()
        self.model_edit.setReadOnly(True)
        model_row.addWidget(self.model_edit, 1)
        model_browse = QPushButton("Browse...")
        model_browse.clicked.connect(self._browse_model)
        model_row.addWidget(model_browse)
        layout.addLayout(model_row)

        val_row = QHBoxLayout()
        val_row.addWidget(QLabel("Val set:"))
        self.val_edit = QLineEdit()
        self.val_edit.setReadOnly(True)
        val_row.addWidget(self.val_edit, 1)
        val_browse = QPushButton("Browse...")
        val_browse.clicked.connect(self._browse_val_dir)
        val_row.addWidget(val_browse)
        layout.addLayout(val_row)

        default_val = self._default_val_dir()
        if default_val is not None:
            self.val_edit.setText(str(default_val))

        self.recalibrate_btn = QPushButton("Recalibrate")
        self.recalibrate_btn.clicked.connect(self._start_recalibration)
        layout.addWidget(self.recalibrate_btn)

        self.log_view = QPlainTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setMinimumHeight(140)
        layout.addWidget(self.log_view)

        self.add_content(content)

    def _default_val_dir(self) -> Path | None:
        if self._project_path is None:
            return None
        candidate = classkit_export_dir(self._project_path) / "val"
        return candidate if candidate.is_dir() else None

    def append_log(self, msg: str) -> None:
        """Append a line to the dialog's log view."""
        self.log_view.appendPlainText(msg)

    def _browse_model(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select classifier checkpoint",
            str(self._project_path) if self._project_path else "",
            "Classifier checkpoints (*.pth *.pt);;All files (*)",
        )
        if path:
            self.model_edit.setText(path)

    def _browse_val_dir(self) -> None:
        directory = QFileDialog.getExistingDirectory(
            self,
            "Select labeled validation set (ImageFolder-style: <val>/<class>/*.jpg)",
            str(self._project_path) if self._project_path else "",
        )
        if directory:
            self.val_edit.setText(directory)

    def _start_recalibration(self) -> None:
        model_path = self.model_edit.text().strip()
        val_dir = self.val_edit.text().strip()
        if not model_path:
            QMessageBox.warning(self, "Recalibrate", "Select a model checkpoint first.")
            return
        if not val_dir:
            QMessageBox.warning(
                self, "Recalibrate", "Select a labeled validation set first."
            )
            return

        self.recalibrate_btn.setEnabled(False)
        self.append_log(f"Recalibrating {model_path} against {val_dir}...")

        worker = _RecalibrateWorker(model_path, val_dir)
        self._worker = worker
        worker.signals.success.connect(self._on_success)
        worker.signals.error.connect(self._on_error)
        worker.signals.finished.connect(self._on_finished)
        QThreadPool.globalInstance().start(worker)

    def _on_success(self, result) -> None:
        temps = ", ".join(f"{t:.3f}" for t in result.temperatures)
        ece_before = ", ".join(f"{e:.4f}" for e in result.ece_before)
        ece_after = ", ".join(f"{e:.4f}" for e in result.ece_after)
        self.append_log(f"Calibrated. Temperature(s): {temps}")
        self.append_log(f"ECE: {ece_before} -> {ece_after}")

    def _on_error(self, message: str) -> None:
        self.append_log(f"Recalibration failed: {message}")
        QMessageBox.critical(self, "Recalibrate", f"Recalibration failed:\n{message}")

    def _on_finished(self) -> None:
        self.recalibrate_btn.setEnabled(True)
        self._worker = None
