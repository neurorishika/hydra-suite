"""CalibrationDialog — one-click inference-performance calibration."""

from __future__ import annotations

from PySide6.QtWidgets import (
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from hydra_suite.core.inference.autotune.models import (
    MAXIMUM_CALIBRATION_BUDGET_SECONDS,
    MINIMUM_CALIBRATION_BUDGET_SECONDS,
)
from hydra_suite.trackerkit.gui.workers.calibration_worker import CalibrationWorker
from hydra_suite.widgets.dialogs import BaseDialog

# Statuses that mean a validated profile now exists for this key -- mirrors
# calibrate_cli._SUCCESS_STATUSES, the CLI's own success set.
_SUCCESS_STATUSES = frozenset({"calibrated", "cache_hit", "cache_hit_after_wait"})


def describe_calibration_outcome(payload: dict) -> str:
    """Render one honest sentence for a calibration/lookup outcome.

    The single mapping used by BOTH the dialog's live label and the setup
    panel's ``lbl_inference_autotune_status``, so the two never disagree.
    Covers the three real outcomes the label must be able to state: a
    matching validated profile was found (with its effective vector), no
    profile matches this configuration, and calibration completed without
    finding an improvement. That last one is a CORRECT result -- on MPS,
    cross-frame batching has measured up to 1.58x SLOWER -- so it is
    reported as a kept-your-settings success, never as a failure.
    """
    status = str(payload.get("status", "unknown")).strip().lower()
    reason = str(payload.get("reason", "")).strip()
    profile = str(payload.get("profile_id") or "").strip()
    profile_text = f" Profile {profile}." if profile else ""

    values = _format_effective_vector(payload.get("effective"))
    cache_mode = _format_cache_mode(payload)

    if status == "cancelled":
        return "Calibration cancelled. Your configured settings are unchanged."
    if status in _SUCCESS_STATUSES and reason == "kept_current_settings":
        return (
            "Calibration complete: no configuration beat your current "
            "settings, so they were kept. That is a correct outcome on some "
            "systems (on MPS, cross-frame batching has measured up to 1.58x "
            "slower)." + profile_text + cache_mode
        )
    if status in _SUCCESS_STATUSES:
        return (
            f"Validated profile in use ({reason}).{profile_text}{values}{cache_mode}"
        ).strip()
    # "unavailable" is the status coordinator.resolve emits when no profile
    # can be used for this key (coordinator.py:135, :152). Verified against
    # the coordinator's status vocabulary -- not guessed.
    if status == "unavailable":
        return (
            "No validated profile matches this video, model, and settings. "
            f"Your configured inference values are used unchanged ({reason})."
            f"{values}{cache_mode}"
        )
    return (
        f"{status.replace('_', ' ').capitalize()} — {reason}."
        f"{profile_text}{values}{cache_mode}"
    ).strip()


def _format_cache_mode(payload: dict) -> str:
    """Name the detection-cache mode this profile is keyed for.

    ``RESULT_CACHE_STAGE_MASK`` and ``cached_fields`` are both part of the
    pipeline fingerprint / search space, so a profile measured WITHOUT a
    detection cache is not the profile a cache-reusing run looks up. The
    density bridge (``store.py``) re-keys only ``workload``, not the cache
    mask, so it cannot close that gap -- and synthesising a masked twin
    record would be dishonest, since the winning vector was never validated
    under the masked (smaller) search space.

    So the label SAYS which mode it covers. A user who then gets a miss on
    run 2 has an explicable result and a clear action (calibrate again),
    rather than a mysterious one.
    """
    cached = payload.get("cached_detections")
    if cached is None:
        return ""
    if cached:
        return (
            " This profile covers runs that REUSE the detection cache "
            "(cached detections on)."
        )
    return (
        " This profile covers runs with NO detection cache. With "
        "'Use cached detections' enabled, calibrate again after this "
        "video's first tracking run."
    )


def _format_effective_vector(effective: object) -> str:
    """Render ``effective`` as a compact ``" Effective: k=v, ..."`` suffix.

    Returns "" if ``effective`` is not a populated dict. Shared by every
    branch of :func:`describe_calibration_outcome` so the same values are
    formatted identically regardless of outcome status.
    """
    if not isinstance(effective, dict):
        return ""
    compact = ", ".join(
        f"{name}={value}"
        for name, value in effective.items()
        if value not in (None, {}, [], ())
    )
    return f" Effective: {compact}." if compact else ""


class CalibrationDialog(BaseDialog):
    """Measure and persist a validated inference profile for one video/config.

    Holds a bounded budget spinbox, a live status label fed by the worker's
    ``status`` signal, and a Cancel action that reaches the running search's
    ``should_cancel`` through ``CalibrationWorker.cancel()``. A cancelled
    calibration never calls ``session.calibrate``'s persistence path to
    completion, so it leaves the project's configured settings untouched --
    this dialog never writes to ``config`` itself.
    """

    def __init__(
        self,
        *,
        params: dict,
        config,
        video_path: str,
        frame_width: int,
        frame_height: int,
        start_frame: int,
        end_frame: int,
        realtime: bool,
        cache_dir=None,
        use_cached_detections: bool = False,
        default_budget_seconds: float,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(
            title="Calibrate inference performance",
            parent=parent,
            buttons=QDialogButtonBox.Close,
        )
        self._params = params
        self._config = config
        self._video_path = video_path
        self._frame_width = frame_width
        self._frame_height = frame_height
        self._start_frame = start_frame
        self._end_frame = end_frame
        self._realtime = realtime
        self._cache_dir = cache_dir
        self._use_cached_detections = use_cached_detections
        self._worker: CalibrationWorker | None = None
        self.result_payload: dict | None = None

        content = QWidget()
        layout = QVBoxLayout(content)

        form = QFormLayout()
        self.spin_budget = QDoubleSpinBox()
        self.spin_budget.setRange(
            MINIMUM_CALIBRATION_BUDGET_SECONDS, MAXIMUM_CALIBRATION_BUDGET_SECONDS
        )
        self.spin_budget.setSingleStep(5.0)
        self.spin_budget.setSuffix(" s")
        self.spin_budget.setDecimals(0)
        self.spin_budget.setKeyboardTracking(False)
        self.spin_budget.setValue(float(default_budget_seconds))
        self.spin_budget.setToolTip(
            "Bounded calibration time budget (5-7200s). A one-time cost per "
            "new system/model/workload combination; the validated profile "
            "is reused after that."
        )
        form.addRow("Calibration budget", self.spin_budget)
        layout.addLayout(form)

        self.lbl_status = QLabel(
            "Measure the fastest inference settings for this video."
        )
        self.lbl_status.setWordWrap(True)
        layout.addWidget(self.lbl_status)

        button_row = QHBoxLayout()
        self.btn_start = QPushButton("Start Calibration")
        self.btn_start.clicked.connect(self._start)
        self.btn_cancel = QPushButton("Cancel Calibration")
        self.btn_cancel.setEnabled(False)
        self.btn_cancel.clicked.connect(self._cancel)
        button_row.addWidget(self.btn_start)
        button_row.addWidget(self.btn_cancel)
        layout.addLayout(button_row)

        self.add_content(content)

    def _start(self) -> None:
        self.btn_start.setEnabled(False)
        self.spin_budget.setEnabled(False)
        self.btn_cancel.setEnabled(True)
        self.lbl_status.setText("Calibrating…")

        self._worker = CalibrationWorker(
            self._params,
            self._config,
            video_path=self._video_path,
            budget_seconds=self.spin_budget.value(),
            frame_width=self._frame_width,
            frame_height=self._frame_height,
            start_frame=self._start_frame,
            end_frame=self._end_frame,
            realtime=self._realtime,
            cache_dir=self._cache_dir,
            use_cached_detections=self._use_cached_detections,
            parent=self,
        )
        self._worker.status.connect(self.lbl_status.setText)
        self._worker.completed.connect(self._on_completed)
        self._worker.error.connect(self._on_error)
        self._worker.start()

    def is_calibration_running(self) -> bool:
        """True while the search thread is still alive.

        ``reject()`` gives a cancelled worker a bounded 5s to unwind and then
        closes regardless, so the worker can outlive this dialog. The
        orchestrator consults this to keep refusing to start a track while a
        measurement is still executing -- on MPS/CPU the contention probe
        cannot detect the overlap itself.
        """
        return self._worker is not None and self._worker.isRunning()

    def _cancel(self) -> None:
        if self._worker is not None:
            self._worker.cancel()
        self.btn_cancel.setEnabled(False)
        self.lbl_status.setText("Cancelling…")

    def _on_completed(self, payload: dict) -> None:
        self.result_payload = payload
        self.lbl_status.setText(describe_calibration_outcome(payload))
        self._reset_controls()

    def _on_error(self, message: str) -> None:
        self.lbl_status.setText(f"Calibration failed: {message}")
        self._reset_controls()

    def _reset_controls(self) -> None:
        self.btn_start.setEnabled(True)
        self.spin_budget.setEnabled(True)
        self.btn_cancel.setEnabled(False)

    def reject(self) -> None:
        """Cancel any running calibration before closing."""
        if self._worker is not None and self._worker.isRunning():
            self._worker.cancel()
            self._worker.wait(5000)
        super().reject()
