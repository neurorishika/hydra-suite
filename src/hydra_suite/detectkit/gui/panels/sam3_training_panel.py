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

from PySide6.QtCore import QMimeData, Qt, Signal
from PySide6.QtGui import QInputMethodEvent, QKeyEvent, QKeySequence
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

from hydra_suite.training.contracts import (
    SAM3_MAX_CONFIGURED_PROMPT_BYTES,
    SAM3_MAX_NEGATIVE_PROMPT_COUNT,
    SAM3_MAX_NEGATIVE_QUERIES_PER_TILE,
    SAM3_MAX_PROMPT_CODEPOINTS,
    SAM3_MAX_PROMPT_UTF8_BYTES,
    Sam3LoraParams,
)
from hydra_suite.training.sam3_lora.availability import (
    Sam3TrainingAvailability,
    probe_sam3_training_availability,
)
from hydra_suite.training.sam3_lora.env import DEFAULT_SAM3_ENV  # noqa: F401
from hydra_suite.training.sam3_lora.env import resolve_sam3_env

# DEFAULT_SAM3_ENV is re-exported (not otherwise referenced in this module)
# because tests assert against it as `sam3_training_panel.DEFAULT_SAM3_ENV`.
from hydra_suite.widgets.workers import BaseWorker

from .slice_settings_widget import SliceSettingsGroup

_PRECISIONS = ("bf16",)

# Kept short: the probe spawns a `conda run` subprocess, and the panel must
# never block the GUI thread for the probe's full default timeout on every
# construction/show. Users pointed at a genuinely slow/hanging env can still
# hit "Check" and wait -- the button has no shortened timeout.
_AUTO_PROBE_TIMEOUT_S = 5.0


class _BoundedPromptListEdit(QPlainTextEdit):
    """Plain-text editor that caps content before it enters the Qt document."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setAcceptDrops(False)

    @staticmethod
    def _bounded_text(value: object) -> str:
        if type(value) is not str:
            return ""
        text = value[
            : SAM3_MAX_CONFIGURED_PROMPT_BYTES + SAM3_MAX_NEGATIVE_PROMPT_COUNT
        ]
        if not text:
            return ""
        output: list[str] = []
        total_bytes = 0
        line_bytes = 0
        line_codepoints = 0
        line_count = 1
        previous_was_cr = False
        for original_char in text:
            if original_char == "\n" and previous_was_cr:
                previous_was_cr = False
                continue
            if original_char in ("\r", "\n"):
                previous_was_cr = original_char == "\r"
                if line_count >= SAM3_MAX_NEGATIVE_PROMPT_COUNT:
                    break
                if total_bytes + 1 > SAM3_MAX_CONFIGURED_PROMPT_BYTES:
                    break
                output.append("\n")
                total_bytes += 1
                line_bytes = 0
                line_codepoints = 0
                line_count += 1
                continue
            previous_was_cr = False
            codepoint = ord(original_char)
            if codepoint < 0x20 or codepoint == 0x7F:
                continue
            if 0xD800 <= codepoint <= 0xDFFF:
                char = "\N{REPLACEMENT CHARACTER}"
                char_bytes = 3
            else:
                char = original_char
                char_bytes = (
                    1
                    + (codepoint >= 0x80)
                    + (codepoint >= 0x800)
                    + (codepoint >= 0x10000)
                )
            if (
                line_codepoints >= SAM3_MAX_PROMPT_CODEPOINTS
                or line_bytes + char_bytes > SAM3_MAX_PROMPT_UTF8_BYTES
            ):
                continue
            if total_bytes + char_bytes > SAM3_MAX_CONFIGURED_PROMPT_BYTES:
                break
            output.append(char)
            total_bytes += char_bytes
            line_bytes += char_bytes
            line_codepoints += 1
        return "".join(output)

    def setPlainText(self, text: str) -> None:  # noqa: N802 - Qt override
        super().setPlainText(self._bounded_text(text))

    def _replace_selection(self, text: str) -> None:
        cursor = self.textCursor()
        current = self.toPlainText()
        start, end = cursor.selectionStart(), cursor.selectionEnd()
        # Truncate the incoming fragment before concatenation. The current
        # document is already bounded, so the temporary candidate is bounded.
        incoming = str(text)[:SAM3_MAX_CONFIGURED_PROMPT_BYTES]
        candidate = current[:start] + incoming + current[end:]
        bounded = self._bounded_text(candidate)
        super().setPlainText(bounded)
        cursor = self.textCursor()
        cursor.setPosition(min(start + len(incoming), len(bounded)))
        self.setTextCursor(cursor)

    def insertPlainText(self, text: str) -> None:  # noqa: N802 - Qt override
        self._replace_selection(text)

    def insertFromMimeData(self, source: QMimeData) -> None:  # noqa: N802
        self._replace_selection(source.text())

    def keyPressEvent(self, event: QKeyEvent) -> None:  # noqa: N802
        if event.matches(QKeySequence.StandardKey.Paste):
            self.paste()
            event.accept()
            return
        modifiers = event.modifiers()
        if event.text() and not modifiers & (
            Qt.KeyboardModifier.ControlModifier
            | Qt.KeyboardModifier.AltModifier
            | Qt.KeyboardModifier.MetaModifier
        ):
            self._replace_selection(event.text())
            event.accept()
            return
        super().keyPressEvent(event)

    def inputMethodEvent(self, event: QInputMethodEvent) -> None:  # noqa: N802
        if event.commitString():
            self._replace_selection(event.commitString())
            event.accept()
            return
        super().inputMethodEvent(event)


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
        env_name = self.env_edit.text().strip() or resolve_sam3_env()
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
        env_name = self.env_edit.text().strip() or resolve_sam3_env()
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
        self.env_edit = QLineEdit(resolve_sam3_env())
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
        self.prompt_edit.setMaxLength(SAM3_MAX_PROMPT_CODEPOINTS)
        prompt_form.addRow("Prompt", self.prompt_edit)
        self.negative_prompts_edit = _BoundedPromptListEdit()
        self.negative_prompts_edit.setPlaceholderText("One negative prompt per line")
        self.negative_prompts_edit.setMaximumHeight(60)
        prompt_form.addRow("Negative prompts", self.negative_prompts_edit)
        self.num_negatives_spin = QSpinBox()
        self.num_negatives_spin.setRange(0, SAM3_MAX_NEGATIVE_QUERIES_PER_TILE)
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
        self.auto_batch_checkbox = QCheckBox("Auto (measure)")
        self.auto_batch_checkbox.setToolTip(
            "Measure this workload on this card, then pick a batch that is "
            "safe under BOTH the measurement and a conservative estimate. "
            "This is not an optimal-batch search: it is conservative by "
            "design, it often chooses 1, and it can refuse the run if "
            "nothing fits. The measurement can only raise the estimate, "
            "never lower it."
        )
        self.auto_batch_checkbox.toggled.connect(self._on_auto_batch_toggled)
        batch_row = QHBoxLayout()
        batch_row.addWidget(self.batch_spin)
        batch_row.addWidget(self.auto_batch_checkbox)
        opt_form.addRow("Batch", batch_row)
        self.grad_accum_spin = QSpinBox()
        self.grad_accum_spin.setRange(1, 256)
        opt_form.addRow("Grad accum", self.grad_accum_spin)
        self.precision_combo = QComboBox()
        self.precision_combo.addItems(_PRECISIONS)
        opt_form.addRow("Mixed precision", self.precision_combo)
        # Early stopping lives HERE, in the Optimisation group of the shared
        # training dialog's SAM3 tab, next to Epochs -- because it is a
        # modifier on Epochs ("train up to N, but stop sooner if val_loss
        # stops improving") and is read with it. It is deliberately NOT wired
        # to the dialog's existing project-level `patience` spin: that one is
        # the Ultralytics/YOLO knob, stored on the project, ranged 1-500 with
        # no "off", and it drives a different training path. Sharing one
        # widget between two contracts is exactly the GUI/CLI divergence the
        # parity guard exists to catch.
        self.patience_spin = QSpinBox()
        self.patience_spin.setRange(0, 500)
        self.patience_spin.setToolTip(
            "Stop training after this many consecutive VALIDATED epochs with "
            "no improvement in val_loss_mean. 0 disables early stopping "
            "(the default; a run then trains all its epochs, exactly as "
            "before this control existed). 3 is a reasonable starting point. "
            "Counted in validations, so with HYDRA_SAM3_VAL_EVERY > 1 each "
            "unit is that many epochs. Costs nothing: val_loss_mean is "
            "already computed every epoch."
        )
        opt_form.addRow("Early stop patience (0 = off)", self.patience_spin)
        self.min_delta_spin = QDoubleSpinBox()
        self.min_delta_spin.setRange(0.0, 1.0)
        self.min_delta_spin.setDecimals(4)
        self.min_delta_spin.setSingleStep(0.001)
        self.min_delta_spin.setToolTip(
            "An epoch counts as an improvement only if it beats the best "
            "val_loss_mean so far by MORE than this. The between-seed sd of "
            "val_loss_mean measured 0.0224 across three seeds, so the 0.005 "
            "default sits well inside noise and will not fire spuriously. "
            "Ignored when patience is 0."
        )
        opt_form.addRow("Early stop min delta", self.min_delta_spin)
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
        # Headless-only scope; see the params builder for why it has no widget.
        self._adapt_scoring_head = False
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

        # The SAHI scale-set UI is SHARED with the YOLO training dialog rather
        # than duplicated: `target_size_fraction` and `object_tile_fraction`
        # are the same quantity, so a scale set ports across as the identity
        # map on fractions. The widget's `sam3` backend hides what SAM3 has no
        # contract field for and emits no 640-anchored pixel list.
        self.slice_group = SliceSettingsGroup(backend="sam3")
        layout.addWidget(self.slice_group)

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

    def _on_auto_batch_toggled(self, checked: bool) -> None:
        self.batch_spin.setEnabled(not checked)

    def params(self) -> Sam3LoraParams:
        negative_prompts = [
            line.strip()
            for line in self.negative_prompts_edit.toPlainText().splitlines()
            if line.strip()
        ]
        prompt = "".join(
            char
            for char in self.prompt_edit.text()
            if ord(char) >= 0x20 and ord(char) != 0x7F
        )
        return Sam3LoraParams(
            prompt=prompt,
            negative_prompts=negative_prompts,
            num_negatives=self.num_negatives_spin.value(),
            rank=self.rank_spin.value(),
            alpha=self.alpha_spin.value(),
            dropout=self.dropout_spin.value(),
            lr=self.lr_spin.value(),
            epochs=self.epochs_spin.value(),
            patience=self.patience_spin.value(),
            min_delta=self.min_delta_spin.value(),
            batch=(
                -1 if self.auto_batch_checkbox.isChecked() else self.batch_spin.value()
            ),
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
            # Deliberately NOT a checkbox: the scoring-head scope is an
            # unvalidated experiment (default off, pending its paired retrain)
            # and has no measured sizing coefficient yet, so exposing it in the
            # GUI would invite an unbudgeted run. It is still carried through
            # the round-trip so loading a spec that enables it -- e.g. one
            # written by hand for that retrain -- is not silently reset here.
            adapt_scoring_head=self._adapt_scoring_head,
            **self.slice_group.to_sam3_tiling(),
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
        self.patience_spin.setValue(p.patience)
        self.min_delta_spin.setValue(p.min_delta)
        if p.batch == -1:
            self.auto_batch_checkbox.setChecked(True)
        else:
            self.batch_spin.setValue(p.batch)
            self.auto_batch_checkbox.setChecked(False)
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
        self._adapt_scoring_head = bool(p.adapt_scoring_head)
        self.slice_group.load_sam3_tiling(
            geometry_mode=p.geometry_mode,
            object_tile_fraction=p.object_tile_fraction,
            object_tile_fractions=p.object_tile_fractions,
            full_frame_mix=p.full_frame_mix,
            slice_width=p.slice_width,
            slice_height=p.slice_height,
            tile_overlap=p.tile_overlap,
            keep_empty_tiles=p.keep_empty_tiles,
            min_area_ratio=p.min_area_ratio,
        )
        self.chk_ack.setChecked(p.label_quality_acknowledged)
        self.env_edit.setText(p.env_name or resolve_sam3_env())

    def acknowledged(self) -> bool:
        return self.chk_ack.isChecked()

    def unavailable_reason(self) -> str:
        return self._unavailable_reason
