"""GUI panel for configuring a SAM3 LoRA finetuning run.

A separate panel rather than inline widgets in ``training_dialog.py`` --
that dialog is already ~2700 lines, well past the project's 500-line
guidance for a single class.

The panel never imports Meta's ``sam3`` package itself; it only calls
``probe_sam3_training_availability`` (which uses ``importlib.util.find_spec``)
to decide whether to disable the whole widget with a reason. A missing
training dependency must never raise at click time.
"""

from __future__ import annotations

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPlainTextEdit,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from hydra_suite.training.contracts import Sam3LoraParams
from hydra_suite.training.sam3_lora.availability import (
    Sam3TrainingAvailability,
    probe_sam3_training_availability,
)
from hydra_suite.training.sam3_lora.env import DEFAULT_SAM3_ENV
from hydra_suite.widgets.workers import BaseWorker

_GEOMETRY_MODES = ("auto_object", "auto_model", "custom")
_PRECISIONS = ("bf16",)

# Kept short: the probe spawns a `conda run` subprocess, and the panel must
# never block the GUI thread for the probe's full default timeout on every
# construction/show. Users pointed at a genuinely slow/hanging env can still
# hit "Check" and wait -- the button has no shortened timeout.
_AUTO_PROBE_TIMEOUT_S = 5.0


class _AvailabilityProbeWorker(BaseWorker):
    """Runs `probe_sam3_training_availability` off the GUI thread.

    The probe spawns a `conda run` subprocess and can take up to its
    `timeout` to return; running it on `showEvent` synchronously froze the
    GUI thread for that whole window on a panel's first show. This worker
    exists solely so the automatic on-show probe never blocks the GUI --
    the explicit "Check" button stays synchronous (see `check_availability`).
    """

    result: Signal = Signal(object)  # Sam3TrainingAvailability

    def __init__(self, env_name: str, timeout: float, parent=None) -> None:
        super().__init__(parent)
        self._env_name = env_name
        self._timeout = timeout

    def execute(self) -> None:
        availability = probe_sam3_training_availability(
            env=self._env_name, timeout=self._timeout
        )
        self.result.emit((availability, self._env_name))


class Sam3TrainingPanel(QWidget):
    """Owns the SAM3 LoRA hyperparameter widgets and the label-quality ack."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._unavailable_reason = ""
        self._probed_once = False
        self._is_destroyed = False
        self._probe_worker: _AvailabilityProbeWorker | None = None
        self.destroyed.connect(self._mark_destroyed)
        self._build_ui()

    # -- Qt lifecycle ------------------------------------------------------

    # Bounded: this only needs to outlast the `conda run` probe's own
    # `_AUTO_PROBE_TIMEOUT_S`, not hang forever if the subprocess wedges.
    _CLOSE_WAIT_MS = 6000

    def showEvent(self, event) -> None:  # noqa: N802 - Qt override
        super().showEvent(event)
        if not self._probed_once:
            self._start_async_probe()

    def hideEvent(self, event) -> None:  # noqa: N802 - Qt override
        self._stop_probe_worker()
        super().hideEvent(event)

    def closeEvent(self, event) -> None:  # noqa: N802 - Qt override
        self._stop_probe_worker()
        super().closeEvent(event)

    def _stop_probe_worker(self) -> None:
        """Block a running probe `QThread` to completion (bounded) before the
        panel can be hidden/closed/destroyed.

        `QThread`s parented to a widget are NOT auto-joined on destruction --
        a still-running thread destroyed under it raises "QThread: Destroyed
        while thread is still running" and SIGABRTs the whole app. This repo
        has hit exactly that failure mode before; `_is_destroyed` alone only
        guards the result *slot*, not the thread's lifetime.
        """
        worker = self._probe_worker
        if worker is None:
            return
        if worker.isRunning():
            worker.quit()
            worker.wait(self._CLOSE_WAIT_MS)

    def _mark_destroyed(self, *_args) -> None:
        self._is_destroyed = True

    # -- Availability probing -----------------------------------------------

    def _start_async_probe(self) -> None:
        """Kick off the on-show probe on a background thread.

        Unlike `check_availability` (used by the explicit "Check" button,
        which may stay synchronous), the automatic on-show probe must never
        block the GUI thread -- the probe spawns a `conda run` subprocess
        and can take up to `_AUTO_PROBE_TIMEOUT_S` to return.
        """
        self._probed_once = True
        env_name = self.env_edit.text().strip() or DEFAULT_SAM3_ENV
        self.env_status_label.setText(f"Checking {env_name!r}...")
        worker = _AvailabilityProbeWorker(env_name, _AUTO_PROBE_TIMEOUT_S, self)
        worker.result.connect(self._on_async_probe_result)
        worker.error.connect(self._on_async_probe_error)
        worker.finished.connect(worker.deleteLater)
        self._probe_worker = worker
        worker.start()

    def _on_async_probe_error(self, message: str) -> None:
        # Previously unconnected: an exception inside the probe (e.g. the
        # `conda run` subprocess call itself raising) left the label stuck
        # on "Checking '<env>'..." forever, with no way to tell the probe
        # had failed rather than still being in flight.
        if self._is_destroyed:
            return
        self.env_status_label.setText(f"Probe failed: {message}")

    def _on_async_probe_result(self, payload: tuple) -> None:
        # The worker thread may finish after the panel (or its owning
        # dialog) has already been closed/destroyed; a queued signal
        # delivered after that point must be a no-op, not a crash on a
        # dead C++ widget.
        if self._is_destroyed:
            return
        availability, env_name = payload
        self._apply_availability(availability, env_name)

    def check_availability(self) -> Sam3TrainingAvailability:
        """Probe the sidecar env named in `self.env_edit` and reflect the result.

        Spawns a `conda run` subprocess synchronously -- this is the
        explicit "Check" button's path, so a deliberate click may block
        briefly with a short timeout. The automatic on-show probe never
        calls this; it uses `_start_async_probe` instead so first show never
        blocks the GUI thread.
        """
        self._probed_once = True
        env_name = self.env_edit.text().strip() or DEFAULT_SAM3_ENV
        self.env_status_label.setText(f"Checking {env_name!r}...")
        availability: Sam3TrainingAvailability = probe_sam3_training_availability(
            env=env_name, timeout=_AUTO_PROBE_TIMEOUT_S
        )
        self._apply_availability(availability, env_name)
        return availability

    def _apply_availability(
        self, availability: Sam3TrainingAvailability, env_name: str
    ) -> None:
        if availability.usable:
            self._unavailable_reason = ""
            self.env_status_label.setText(f"{env_name!r} is usable.")
            self._body.setEnabled(True)
        else:
            self._unavailable_reason = availability.reason
            self.env_status_label.setText(
                f"{env_name!r} is unusable: {availability.reason}"
            )
            self._body.setEnabled(False)

    # -- UI construction -------------------------------------------------

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)

        self.env_group = QGroupBox("Sidecar environment")
        env_layout = QHBoxLayout(self.env_group)
        env_layout.addWidget(QLabel("Conda env"))
        self.env_edit = QLineEdit(DEFAULT_SAM3_ENV)
        env_layout.addWidget(self.env_edit)
        self.btn_check_env = QPushButton("Check")
        self.btn_check_env.clicked.connect(self.check_availability)
        env_layout.addWidget(self.btn_check_env)
        layout.addWidget(self.env_group)

        self.env_status_label = QLabel("Not checked yet.")
        self.env_status_label.setWordWrap(True)
        layout.addWidget(self.env_status_label)

        # Everything below is disabled with a reason when the sidecar env is
        # unusable; the env row above stays interactive so the user can fix
        # the env name and re-check without recreating the panel.
        self._body = QWidget()
        layout.addWidget(self._body)
        body_layout = QVBoxLayout(self._body)
        body_layout.setContentsMargins(0, 0, 0, 0)
        layout = body_layout

        host_notice = QLabel(
            "SAM3 LoRA finetuning requires a CUDA host with a large GPU "
            "(~32 GB). This role cannot run on this machine if 'sam3' or "
            "its checkpoint is unavailable; the reason is shown when disabled."
        )
        host_notice.setWordWrap(True)
        layout.addWidget(host_notice)

        prompt_group = QGroupBox("Concept")
        prompt_form = QFormLayout(prompt_group)
        self.prompt_edit = QLineEdit()
        prompt_form.addRow("Prompt", self.prompt_edit)
        self.negative_prompts_edit = QPlainTextEdit()
        self.negative_prompts_edit.setPlaceholderText("One negative prompt per line")
        self.negative_prompts_edit.setMaximumHeight(60)
        prompt_form.addRow("Negative prompts", self.negative_prompts_edit)
        self.num_negatives_spin = QSpinBox()
        self.num_negatives_spin.setRange(0, 100)
        prompt_form.addRow("Num negatives", self.num_negatives_spin)
        layout.addWidget(prompt_group)

        lora_group = QGroupBox("LoRA")
        lora_form = QFormLayout(lora_group)
        self.rank_spin = QSpinBox()
        self.rank_spin.setRange(1, 512)
        lora_form.addRow("Rank", self.rank_spin)
        self.alpha_spin = QSpinBox()
        self.alpha_spin.setRange(1, 1024)
        lora_form.addRow("Alpha", self.alpha_spin)
        self.dropout_spin = QDoubleSpinBox()
        self.dropout_spin.setRange(0.0, 1.0)
        self.dropout_spin.setSingleStep(0.01)
        lora_form.addRow("Dropout", self.dropout_spin)
        layout.addWidget(lora_group)

        opt_group = QGroupBox("Optimisation")
        opt_form = QFormLayout(opt_group)
        self.lr_spin = QDoubleSpinBox()
        self.lr_spin.setDecimals(8)
        self.lr_spin.setRange(0.0, 1.0)
        self.lr_spin.setSingleStep(1e-5)
        opt_form.addRow("Learning rate", self.lr_spin)
        self.epochs_spin = QSpinBox()
        self.epochs_spin.setRange(1, 1000)
        opt_form.addRow("Epochs", self.epochs_spin)
        self.batch_spin = QSpinBox()
        self.batch_spin.setRange(1, 256)
        opt_form.addRow("Batch", self.batch_spin)
        self.grad_accum_spin = QSpinBox()
        self.grad_accum_spin.setRange(1, 256)
        opt_form.addRow("Grad accum", self.grad_accum_spin)
        self.precision_combo = QComboBox()
        self.precision_combo.addItems(_PRECISIONS)
        opt_form.addRow("Mixed precision", self.precision_combo)
        layout.addWidget(opt_group)

        safety_group = QGroupBox("Resource safety")
        safety_form = QFormLayout(safety_group)
        self.host_reserve_gb_spin = QDoubleSpinBox()
        self.host_reserve_gb_spin.setRange(0.0, 1024.0)
        self.host_reserve_gb_spin.setDecimals(1)
        safety_form.addRow("Host reserve (GiB)", self.host_reserve_gb_spin)
        self.host_reserve_fraction_spin = QDoubleSpinBox()
        self.host_reserve_fraction_spin.setRange(0.0, 1.0)
        self.host_reserve_fraction_spin.setDecimals(2)
        self.host_reserve_fraction_spin.setSingleStep(0.05)
        safety_form.addRow("Host reserve fraction", self.host_reserve_fraction_spin)
        self.cuda_safety_fraction_spin = QDoubleSpinBox()
        self.cuda_safety_fraction_spin.setRange(0.01, 1.0)
        self.cuda_safety_fraction_spin.setDecimals(2)
        self.cuda_safety_fraction_spin.setSingleStep(0.05)
        safety_form.addRow("CUDA usable fraction", self.cuda_safety_fraction_spin)
        self.host_limit_headroom_spin = QDoubleSpinBox()
        self.host_limit_headroom_spin.setRange(1.0, 4.0)
        self.host_limit_headroom_spin.setDecimals(2)
        self.host_limit_headroom_spin.setSingleStep(0.05)
        safety_form.addRow("Hard-limit headroom", self.host_limit_headroom_spin)
        self.watchdog_poll_spin = QDoubleSpinBox()
        self.watchdog_poll_spin.setRange(0.1, 60.0)
        self.watchdog_poll_spin.setDecimals(1)
        safety_form.addRow("Watchdog interval (s)", self.watchdog_poll_spin)
        layout.addWidget(safety_group)

        adapt_group = QGroupBox("Adapted submodules")
        adapt_form = QFormLayout(adapt_group)
        self.chk_adapt_vision_encoder = QCheckBox("Vision encoder")
        self.chk_adapt_text_encoder = QCheckBox("Text encoder")
        self.chk_adapt_geometry_encoder = QCheckBox("Geometry encoder")
        self.chk_adapt_detr_encoder = QCheckBox("DETR encoder")
        self.chk_adapt_detr_decoder = QCheckBox("DETR decoder")
        self.chk_adapt_mask_decoder = QCheckBox("Mask decoder")
        for chk in (
            self.chk_adapt_vision_encoder,
            self.chk_adapt_text_encoder,
            self.chk_adapt_geometry_encoder,
            self.chk_adapt_detr_encoder,
            self.chk_adapt_detr_decoder,
            self.chk_adapt_mask_decoder,
        ):
            adapt_form.addRow(chk)
        layout.addWidget(adapt_group)

        tiling_group = QGroupBox("Tiling")
        tiling_form = QFormLayout(tiling_group)
        self.geometry_mode_combo = QComboBox()
        self.geometry_mode_combo.addItems(_GEOMETRY_MODES)
        tiling_form.addRow("Geometry mode", self.geometry_mode_combo)
        self.object_tile_fraction_spin = QDoubleSpinBox()
        self.object_tile_fraction_spin.setDecimals(4)
        self.object_tile_fraction_spin.setRange(0.0, 1.0)
        self.object_tile_fraction_spin.setSingleStep(0.001)
        tiling_form.addRow("Object tile fraction", self.object_tile_fraction_spin)
        self.slice_width_spin = QSpinBox()
        self.slice_width_spin.setRange(0, 100000)
        tiling_form.addRow("Slice width (custom)", self.slice_width_spin)
        self.slice_height_spin = QSpinBox()
        self.slice_height_spin.setRange(0, 100000)
        tiling_form.addRow("Slice height (custom)", self.slice_height_spin)
        self.tile_overlap_spin = QDoubleSpinBox()
        self.tile_overlap_spin.setDecimals(3)
        self.tile_overlap_spin.setRange(0.0, 1.0)
        self.tile_overlap_spin.setSingleStep(0.01)
        tiling_form.addRow("Tile overlap", self.tile_overlap_spin)
        self.chk_keep_empty_tiles = QCheckBox("Keep empty tiles")
        tiling_form.addRow(self.chk_keep_empty_tiles)
        layout.addWidget(tiling_group)

        ack_group = QGroupBox("Label quality")
        ack_layout = QVBoxLayout(ack_group)
        ack_label = QLabel(
            "Training runs on ALL of this source's labels, including any "
            "SAM3 escalation output you previously accepted. Provenance does "
            "not survive a review, so bad labels teach SAM3 bad behaviour."
        )
        ack_label.setWordWrap(True)
        ack_layout.addWidget(ack_label)
        self.chk_ack = QCheckBox(
            "I have verified these labels are correct; SAM3 will learn any "
            "systematic error in them."
        )
        self.chk_ack.setChecked(False)
        ack_layout.addWidget(self.chk_ack)
        layout.addWidget(ack_group)

        self.set_params(Sam3LoraParams())

    # -- Public interface --------------------------------------------------

    def params(self) -> Sam3LoraParams:
        negative_prompts = [
            line.strip()
            for line in self.negative_prompts_edit.toPlainText().splitlines()
            if line.strip()
        ]
        return Sam3LoraParams(
            prompt=self.prompt_edit.text(),
            negative_prompts=negative_prompts,
            num_negatives=self.num_negatives_spin.value(),
            rank=self.rank_spin.value(),
            alpha=self.alpha_spin.value(),
            dropout=self.dropout_spin.value(),
            lr=self.lr_spin.value(),
            epochs=self.epochs_spin.value(),
            batch=self.batch_spin.value(),
            grad_accum=self.grad_accum_spin.value(),
            mixed_precision=self.precision_combo.currentText(),
            host_reserve_gb=self.host_reserve_gb_spin.value(),
            host_reserve_fraction=self.host_reserve_fraction_spin.value(),
            cuda_safety_fraction=self.cuda_safety_fraction_spin.value(),
            host_limit_headroom_fraction=self.host_limit_headroom_spin.value(),
            watchdog_poll_seconds=self.watchdog_poll_spin.value(),
            adapt_vision_encoder=self.chk_adapt_vision_encoder.isChecked(),
            adapt_text_encoder=self.chk_adapt_text_encoder.isChecked(),
            adapt_geometry_encoder=self.chk_adapt_geometry_encoder.isChecked(),
            adapt_detr_encoder=self.chk_adapt_detr_encoder.isChecked(),
            adapt_detr_decoder=self.chk_adapt_detr_decoder.isChecked(),
            adapt_mask_decoder=self.chk_adapt_mask_decoder.isChecked(),
            geometry_mode=self.geometry_mode_combo.currentText(),
            object_tile_fraction=self.object_tile_fraction_spin.value(),
            slice_width=self.slice_width_spin.value(),
            slice_height=self.slice_height_spin.value(),
            tile_overlap=self.tile_overlap_spin.value(),
            keep_empty_tiles=self.chk_keep_empty_tiles.isChecked(),
            label_quality_acknowledged=self.chk_ack.isChecked(),
            env_name=self.env_edit.text().strip(),
        )

    def set_params(self, p: Sam3LoraParams) -> None:
        self.prompt_edit.setText(p.prompt)
        self.negative_prompts_edit.setPlainText("\n".join(p.negative_prompts))
        self.num_negatives_spin.setValue(p.num_negatives)
        self.rank_spin.setValue(p.rank)
        self.alpha_spin.setValue(p.alpha)
        self.dropout_spin.setValue(p.dropout)
        self.lr_spin.setValue(p.lr)
        self.epochs_spin.setValue(p.epochs)
        self.batch_spin.setValue(p.batch)
        self.grad_accum_spin.setValue(p.grad_accum)
        idx = self.precision_combo.findText(p.mixed_precision)
        if idx >= 0:
            self.precision_combo.setCurrentIndex(idx)
        else:
            self.precision_combo.setCurrentIndex(0)
        self.host_reserve_gb_spin.setValue(p.host_reserve_gb)
        self.host_reserve_fraction_spin.setValue(p.host_reserve_fraction)
        self.cuda_safety_fraction_spin.setValue(p.cuda_safety_fraction)
        self.host_limit_headroom_spin.setValue(p.host_limit_headroom_fraction)
        self.watchdog_poll_spin.setValue(p.watchdog_poll_seconds)
        self.chk_adapt_vision_encoder.setChecked(p.adapt_vision_encoder)
        self.chk_adapt_text_encoder.setChecked(p.adapt_text_encoder)
        self.chk_adapt_geometry_encoder.setChecked(p.adapt_geometry_encoder)
        self.chk_adapt_detr_encoder.setChecked(p.adapt_detr_encoder)
        self.chk_adapt_detr_decoder.setChecked(p.adapt_detr_decoder)
        self.chk_adapt_mask_decoder.setChecked(p.adapt_mask_decoder)
        idx = self.geometry_mode_combo.findText(p.geometry_mode)
        if idx >= 0:
            self.geometry_mode_combo.setCurrentIndex(idx)
        self.object_tile_fraction_spin.setValue(p.object_tile_fraction)
        self.slice_width_spin.setValue(p.slice_width)
        self.slice_height_spin.setValue(p.slice_height)
        self.tile_overlap_spin.setValue(p.tile_overlap)
        self.chk_keep_empty_tiles.setChecked(p.keep_empty_tiles)
        self.chk_ack.setChecked(p.label_quality_acknowledged)
        self.env_edit.setText(p.env_name or DEFAULT_SAM3_ENV)

    def acknowledged(self) -> bool:
        return self.chk_ack.isChecked()

    def unavailable_reason(self) -> str:
        return self._unavailable_reason
