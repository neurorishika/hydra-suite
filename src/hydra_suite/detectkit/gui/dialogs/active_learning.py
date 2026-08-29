"""Modal dialog for running an active-learning round in DetectKit."""

from __future__ import annotations

from typing import Callable

from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QProgressBar,
    QPushButton,
    QRadioButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from hydra_suite.data.al.acquisition import PRESETS
from hydra_suite.data.al.candidate_pool import CandidatePoolConfig
from hydra_suite.data.al.escalation import achievable_levels
from hydra_suite.detectkit.gui.models import DetectKitProject
from hydra_suite.utils.geometry_levels import GeometryLevel
from hydra_suite.widgets.dialogs import BaseDialog

# Ultralytics model task -> the geometry level that task natively produces.
# segment -> polygon masks, obb -> oriented boxes, detect -> axis-aligned boxes
# only. This is DetectKit's own copy: the equivalent mapping lives in
# TrackerKit's dataset panel, but detectkit and trackerkit are sibling app
# packages and neither may import the other.
_TASK_TO_LEVEL = {
    "segment": GeometryLevel.POLYGON,
    "obb": GeometryLevel.OBB,
    "detect": GeometryLevel.AABB,
}


def _format_level_status(native_level: GeometryLevel) -> str:
    """Human-readable summary of which label levels the active model can produce."""
    labels = " + ".join(lvl.label for lvl in achievable_levels(native_level))
    if native_level is GeometryLevel.POLYGON:
        return f"Will export: {labels}"
    if native_level is GeometryLevel.OBB:
        return f"Will export: {labels} — polygon labels require a segmentation model"
    return (
        f"Will export: {labels} — oriented and polygon labels require an OBB or "
        "segmentation model"
    )


class ActiveLearningDialog(BaseDialog):
    """Active-learning frame-selection dialog (Input / Acquisition / Run)."""

    def __init__(
        self,
        project: DetectKitProject,
        parent: QWidget | None = None,
    ):
        super().__init__(
            "Active Learning",
            parent=parent,
            buttons=QDialogButtonBox.StandardButton.Close,
        )
        self._project = project
        self._run_handler: Callable[[], None] | None = None
        self._running = False
        self._native_level = GeometryLevel.OBB
        self.resize(560, 610)
        self._build_content()
        # Keep checkbox/enabled state self-consistent from construction
        # onward -- `_build_levels_group` starts every checkbox checked and
        # enabled, so a caller that builds a request before ever calling
        # `set_model_task` must still see the default-level gate applied, not
        # an ungated "everything checked" state. "obb" is the literal task
        # this dialog defaults to (matches `_native_level`'s GeometryLevel.OBB
        # default above), not derived from it, so this stays correct even if
        # that default level ever changes independently.
        self.set_model_task("obb")
        self._sync_run_enabled()

    # ------------------------------------------------------------------
    def _build_content(self) -> None:
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)
        layout.addWidget(self._build_input_group())
        layout.addWidget(self._build_acquisition_group())
        layout.addWidget(self._build_levels_group())
        layout.addWidget(self._build_run_group())
        self.add_content(container)

    def _build_input_group(self) -> QGroupBox:
        self.input_group = QGroupBox("Input")
        form = QFormLayout(self.input_group)

        self.rb_video = QRadioButton("Video")
        self.rb_folder = QRadioButton("Image folder")
        self.rb_project = QRadioButton("Existing project source (unlabeled)")
        self.rb_video.setChecked(True)
        for rb in (self.rb_video, self.rb_folder, self.rb_project):
            rb.toggled.connect(self._sync_run_enabled)
        rb_row = QHBoxLayout()
        rb_row.addWidget(self.rb_video)
        rb_row.addWidget(self.rb_folder)
        rb_row.addWidget(self.rb_project)
        form.addRow("Source kind", _wrap(rb_row))

        self.input_path_edit = QLineEdit()
        self.input_path_edit.textChanged.connect(self._sync_run_enabled)
        self.browse_button = QPushButton("Browse...")
        self.browse_button.clicked.connect(self._browse)
        path_row = QHBoxLayout()
        path_row.addWidget(self.input_path_edit)
        path_row.addWidget(self.browse_button)
        form.addRow("Path", _wrap(path_row))

        return self.input_group

    def _build_acquisition_group(self) -> QGroupBox:
        self.acquisition_group = QGroupBox("Acquisition")
        form = QFormLayout(self.acquisition_group)

        self.preset_combo = QComboBox()
        for name in PRESETS:
            if name != "tracker_default":
                self.preset_combo.addItem(name)
        form.addRow("Preset", self.preset_combo)

        self.expected_count_spin = QSpinBox()
        self.expected_count_spin.setRange(0, 1000)
        self.expected_count_spin.setValue(0)
        form.addRow("Expected count per frame (0 = unknown)", self.expected_count_spin)

        self.budget_spin = QSpinBox()
        self.budget_spin.setRange(1, 1000)
        self.budget_spin.setValue(50)
        form.addRow("Budget (top-K)", self.budget_spin)

        defaults = CandidatePoolConfig()
        self.dedup_window_spin = QSpinBox()
        self.dedup_window_spin.setRange(0, 10_000)
        self.dedup_window_spin.setValue(int(defaults.dedup_window or 0))
        self.dedup_window_spin.setToolTip(
            "Compare each frame only with this many recently kept frames. "
            "Set 0 for global deduplication across the entire source."
        )
        form.addRow("Dedup history (kept frames; 0 = global)", self.dedup_window_spin)

        self.motion_threshold_spin = QDoubleSpinBox()
        self.motion_threshold_spin.setRange(0.0, 255.0)
        self.motion_threshold_spin.setDecimals(2)
        self.motion_threshold_spin.setSingleStep(0.5)
        self.motion_threshold_spin.setValue(defaults.motion_threshold)
        self.motion_threshold_spin.setToolTip(
            "Skip full duplicate scoring when the mean grayscale pixel change "
            "is below this value. Set 0 to disable the motion prefilter."
        )
        form.addRow("Motion prefilter threshold (0 = off)", self.motion_threshold_spin)

        return self.acquisition_group

    def _build_levels_group(self) -> QGroupBox:
        self.levels_group = QGroupBox("Export levels")
        form = QFormLayout(self.levels_group)

        self.lbl_export_level_status = QLabel(_format_level_status(self._native_level))
        self.lbl_export_level_status.setWordWrap(True)
        form.addRow("Label levels", self.lbl_export_level_status)

        self.chk_level_polygon = QCheckBox("polygon (segmentation masks)")
        self.chk_level_obb = QCheckBox("obb (oriented boxes)")
        self.chk_level_aabb = QCheckBox("aabb (axis-aligned boxes)")
        for chk in (self.chk_level_polygon, self.chk_level_obb, self.chk_level_aabb):
            chk.setChecked(True)
            chk.setToolTip(
                "Each enabled level is written as its own DetectKit source. "
                "Images are hardlinked, so extra levels cost almost no disk."
            )
        levels_row = QVBoxLayout()
        levels_row.addWidget(self.chk_level_polygon)
        levels_row.addWidget(self.chk_level_obb)
        levels_row.addWidget(self.chk_level_aabb)
        form.addRow("Export as", _wrap(levels_row))

        return self.levels_group

    def _build_run_group(self) -> QGroupBox:
        box = QGroupBox("Run")
        v = QVBoxLayout(box)

        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        v.addWidget(self.progress)

        self.status_label = QLabel("Idle.")
        self.status_label.setWordWrap(True)
        v.addWidget(self.status_label)

        self.run_button = QPushButton("Run")
        self.run_button.clicked.connect(self._on_run)
        btn_row = QHBoxLayout()
        btn_row.addStretch(1)
        btn_row.addWidget(self.run_button)
        v.addLayout(btn_row)
        return box

    # ------------------------------------------------------------------
    def _browse(self) -> None:
        if self.rb_video.isChecked():
            path, _ = QFileDialog.getOpenFileName(
                self,
                "Select video",
                "",
                "Video files (*.mp4 *.mov *.avi)",
            )
        elif self.rb_folder.isChecked():
            path = QFileDialog.getExistingDirectory(self, "Select image folder")
        else:
            path = ""
        if path:
            self.input_path_edit.setText(path)

    def _sync_run_enabled(self, *_):
        if self._running:
            self.run_button.setEnabled(False)
            self.status_label.setText("Active learning is running. Inputs are locked.")
            return
        path_ok = self.rb_project.isChecked() or bool(
            self.input_path_edit.text().strip()
        )
        model_ok = bool(self._project.active_model_path)
        self.run_button.setEnabled(path_ok and model_ok)
        if not model_ok:
            self.status_label.setText(
                "Set an active model in DetectKit before running AL."
            )
        elif not path_ok:
            self.status_label.setText("Pick an input source.")
        else:
            self.status_label.setText("Ready.")

    def _on_run(self) -> None:
        if self._run_handler is not None:
            self._run_handler()

    def set_run_handler(self, handler: Callable[[], None]) -> None:
        """Main window wires this to construct + start the AL worker."""
        self._run_handler = handler

    def set_running(self, running: bool) -> None:
        """Lock editable controls while an AL round is active."""
        self._running = bool(running)
        self.input_group.setEnabled(not self._running)
        self.acquisition_group.setEnabled(not self._running)
        self.levels_group.setEnabled(not self._running)
        self._sync_run_enabled()

    def set_model_task(self, task: str) -> None:
        """Gate level checkboxes to what the active model's task can produce.

        Never claim a geometry level the model did not produce: segment ->
        polygon, obb -> obb, detect -> aabb. Unrecognized tasks leave the
        current gate untouched. Checked state is fully re-derived from the
        new gate on every call (available levels default to checked,
        unavailable ones are force-unchecked) rather than only touching
        boxes that become newly unavailable -- this is what makes the gate
        self-consistent whether the dialog's model task changes once (at
        open) or is called eagerly by `__init__` before any caller-driven
        change, without depending on the checkbox's prior state.
        """
        native = _TASK_TO_LEVEL.get(str(task).strip().lower())
        if native is None:
            return
        self._native_level = native
        allowed = set(achievable_levels(native))
        for level, chk in (
            (GeometryLevel.POLYGON, self.chk_level_polygon),
            (GeometryLevel.OBB, self.chk_level_obb),
            (GeometryLevel.AABB, self.chk_level_aabb),
        ):
            available = level in allowed
            chk.setEnabled(available)
            chk.setChecked(available)
        self.lbl_export_level_status.setText(_format_level_status(native))

    def _checked_levels(self) -> list[GeometryLevel]:
        """Checked levels, highest first."""
        checked = [
            level
            for level, chk in (
                (GeometryLevel.POLYGON, self.chk_level_polygon),
                (GeometryLevel.OBB, self.chk_level_obb),
                (GeometryLevel.AABB, self.chk_level_aabb),
            )
            if chk.isChecked()
        ]
        return sorted(checked, reverse=True)

    def build_request(self, detector=None):
        """Construct an `ALRequest` from the dialog's current field values."""
        from hydra_suite.detectkit.jobs.al_worker import ALRequest

        levels = self._checked_levels() or [self._native_level]
        return ALRequest(
            input_kind=(
                "video"
                if self.rb_video.isChecked()
                else "folder" if self.rb_folder.isChecked() else "project"
            ),
            input_path=self.input_path_edit.text(),
            project=self._project,
            budget=self.budget_spin.value(),
            preset=self.preset_combo.currentText(),
            expected_count=self.expected_count_spin.value(),
            detector=detector,
            candidate_pool=CandidatePoolConfig(
                dedup_window=self.dedup_window_spin.value() or None,
                motion_threshold=self.motion_threshold_spin.value(),
            ),
            export_level=levels[0].label,
            export_levels=[lvl.label for lvl in levels],
            native_level=self._native_level.label,
        )


def _wrap(layout) -> QWidget:
    w = QWidget()
    w.setLayout(layout)
    return w
