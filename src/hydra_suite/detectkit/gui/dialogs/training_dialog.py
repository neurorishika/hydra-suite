"""TrainingDialog — full training configuration and run control for DetectKit."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QImageReader, QPixmap
from PySide6.QtWidgets import (
    QAbstractButton,
    QAbstractSpinBox,
    QCheckBox,
    QComboBox,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFileDialog,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QTabWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from hydra_suite.training.contracts import SourceDataset, SplitConfig, TrainingRole
from hydra_suite.training.geometry_levels import GeometryLevel
from hydra_suite.widgets.dialogs import BaseDialog
from hydra_suite.widgets.workers import BaseWorker

from ...jobs.training import DatasetPreparationCancelled as _DatasetPreparationCancelled
from ...jobs.training import DatasetPreparationRequest as _DatasetPreparationRequest
from ...jobs.training import DatasetPreparationResult as _DatasetPreparationResult
from ...jobs.training import RoleTrainingEntry
from ...jobs.training import prepare_role_datasets as _prepare_role_datasets
from ...jobs.training import run_role_entries
from ..models import SliceTrainingSettings
from ..panels.sam3_training_panel import Sam3TrainingPanel

if TYPE_CHECKING:
    from ..models import DetectKitProject

logger = logging.getLogger(__name__)


def merged_level_and_blocker(sources):
    """Return (min geometry level across sources, the source that set it)."""
    if not sources:
        return GeometryLevel.POLYGON, None
    blocker = min(
        sources, key=lambda s: GeometryLevel.from_str(getattr(s, "level", "obb"))
    )
    return GeometryLevel.from_str(getattr(blocker, "level", "obb")), blocker


_SELECTION_ROLE_MAP: dict[tuple[str, str], tuple[str, ...]] = {
    ("direct", "obb"): ("obb_direct",),
    ("direct", "detect"): ("detect_direct",),
    ("direct", "segment"): ("segment_direct",),
    ("sequential", "obb"): ("seq_detect", "seq_crop_obb"),
    ("sequential", "detect"): ("seq_detect",),
    ("sequential", "segment"): ("seq_detect", "seq_crop_segment"),
    ("semantic", "segment"): ("semantic_sam3",),
}

_SELECTION_DESCRIPTIONS: dict[tuple[str, str], str] = {
    ("direct", "obb"): "Train one full-image oriented bounding-box model.",
    ("direct", "detect"): "Train one full-image axis-aligned detector.",
    ("direct", "segment"): "Train one full-image instance-segmentation model.",
    (
        "sequential",
        "obb",
    ): "Train a full-image detector followed by a crop-focused OBB model.",
    (
        "sequential",
        "detect",
    ): "Train the first-stage full-image detector for a sequential pipeline.",
    (
        "sequential",
        "segment",
    ): "Train a full-image detector followed by a crop-focused segmentation model.",
    ("semantic", "segment"): (
        "Finetune SAM3 on this source's polygons (CUDA host, ~32 GB)."
    ),
}


# ---------------------------------------------------------------------------
# Worker
# ---------------------------------------------------------------------------


class _DatasetPreparationWorker(BaseWorker):
    """Build role datasets in a background thread before training starts."""

    log_signal = Signal(str)
    result_ready = Signal(object)

    def __init__(self, orchestrator, request: _DatasetPreparationRequest) -> None:
        super().__init__()
        self._orchestrator = orchestrator
        self._request = request
        self._cancel = False

    def cancel(self) -> None:
        self._cancel = True

    def is_cancelled(self) -> bool:
        return bool(self._cancel)

    def execute(self) -> None:
        try:
            result = _prepare_role_datasets(
                self._orchestrator,
                self._request,
                log=self.log_signal.emit,
                status=self.status.emit,
                should_cancel=self.is_cancelled,
            )
        except _DatasetPreparationCancelled:
            self.status.emit("Dataset preparation cancelled.")
            return
        if not self.is_cancelled():
            self.result_ready.emit(result)


class _TrainingWorker(BaseWorker):
    """Run selected role trainings sequentially in a background thread."""

    log_signal = Signal(str)
    role_started = Signal(str)
    role_finished = Signal(str, bool, str)
    progress_signal = Signal(str, int, int)
    done_signal = Signal(list)

    def __init__(self, orchestrator, role_entries) -> None:
        super().__init__()
        self.orchestrator = orchestrator
        self.role_entries = role_entries
        self._cancel = False

    def cancel(self) -> None:
        """Request cancellation; the running role loop checks this flag before each role."""
        self._cancel = True

    def _should_cancel(self) -> bool:
        return bool(self._cancel)

    def execute(self) -> None:
        results = run_role_entries(
            self.orchestrator,
            self.role_entries,
            log=self.log_signal.emit,
            progress=self.progress_signal.emit,
            should_cancel=self._should_cancel,
            role_started=self.role_started.emit,
            role_finished=self.role_finished.emit,
        )
        self.done_signal.emit(results)


# ---------------------------------------------------------------------------
# Dialog
# ---------------------------------------------------------------------------


class TrainingDialog(BaseDialog):
    """Full training configuration and run control."""

    training_completed = Signal(list)

    def __init__(self, project: "DetectKitProject", parent=None) -> None:
        super().__init__(
            "Train Model",
            parent=parent,
            buttons=QDialogButtonBox.StandardButton.Close,
        )
        self._project = project
        self._dataset_worker = None
        self._pending_dataset_result: _DatasetPreparationResult | None = None
        self._dataset_preparation_error = ""
        self._worker = None
        self._last_training_results: list[dict] = []
        self._role_logs: dict[str, list[str]] = {}
        self._current_role = ""
        self._dataset_fit_cache_key: tuple | None = None
        self._dataset_fit_cache_text = ""
        self._dataset_fit_dirty = True
        self._training_running = False
        self.role_dataset_dirs: dict[str, str] = {}

        try:
            from hydra_suite.paths import get_training_workspace_dir
            from hydra_suite.training import TrainingOrchestrator

            self._workspace_default = get_training_workspace_dir("YOLO")
            self._orchestrator = TrainingOrchestrator(self._workspace_default)
        except ImportError:
            self._workspace_default = Path("./training_workspace")
            self._orchestrator = None

        self.resize(1080, 960)
        self.setMinimumSize(960, 820)
        self._build_content()
        self._load_from_project()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_content(self) -> None:
        self._apply_training_dialog_styles()
        self._build_role_state()

        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(12)

        layout.addWidget(self._build_header())

        self.training_tabs = QTabWidget()
        self.training_tabs.addTab(self._build_overview_tab(), "Overview")
        self.training_tabs.addTab(self._build_training_tab(), "Advanced")
        self.sam3_panel = Sam3TrainingPanel()
        self._sam3_tab_index = self.training_tabs.addTab(self.sam3_panel, "SAM3")
        self.training_tabs.setTabVisible(
            self._sam3_tab_index, self._selected_mode() == "semantic"
        )
        layout.addWidget(self.training_tabs, 1)

        layout.addWidget(self._build_run_group(), 0)

        self.add_content(container)
        self._connect_summary_signals()
        self._apply_training_tooltips()

    def _apply_training_tooltips(self) -> None:
        """Describe every interactive control in the training dialog.

        Specific tooltips explain the common decisions. A type-specific fallback
        keeps specialised controls (including the SAM3 panel) discoverable as
        that tab evolves without silently introducing unlabelled interaction.
        """
        tips = {
            self.mode_combo: "Choose a direct, sequential, or semantic training recipe.",
            self.task_combo: "Choose the output geometry the selected recipe should learn.",
            self.chk_role_obb_direct: "Train a direct oriented-bounding-box detector.",
            self.chk_role_detect_direct: "Train a direct axis-aligned object detector.",
            self.chk_role_segment_direct: "Train a direct instance-segmentation model.",
            self.chk_role_seq_detect: "Train the full-frame first stage of a sequential recipe.",
            self.chk_role_seq_crop_obb: "Train the crop-focused OBB second stage.",
            self.chk_role_seq_crop_segment: "Train the crop-focused segmentation second stage.",
            self.chk_semantic_sam3: "Fine-tune the semantic SAM3 training role.",
            self.spin_train: "Fraction of the source data assigned to training.",
            self.spin_val: "Fraction of the source data assigned to validation.",
            self.spin_seed: "Random seed used for deterministic splitting and sampling.",
            self.chk_dedup: "Remove source images with identical content before training.",
            self.spin_crop_pad: "Extra context added around sequential second-stage crops, as a fraction of object size.",
            self.spin_crop_min_px: "Smallest allowed sequential second-stage crop side in pixels.",
            self.chk_crop_square: "Make sequential second-stage crops square before model resizing.",
            self.combo_device: "Choose the hardware used for dataset preparation and model training.",
            self.spin_epochs: "Maximum number of complete passes through the training dataset.",
            self.spin_batch: "Number of images processed together in each optimization step.",
            self.chk_auto_batch: "Let Ultralytics choose the largest safe batch size for the selected hardware.",
            self.spin_lr0: "Initial optimizer learning rate. Lower values make training updates more conservative.",
            self.spin_patience: "Stop training after this many epochs without validation improvement.",
            self.spin_workers: "Number of background data-loading worker processes. Zero loads in the main process.",
            self.chk_cache: "Cache decoded training images to speed later epochs at the cost of memory or disk space.",
            self.spin_imgsz_obb_direct: "Square model input size for direct OBB training.",
            self.spin_imgsz_detect_direct: "Square model input size for direct detection training.",
            self.spin_imgsz_segment_direct: "Square model input size for direct segmentation training.",
            self.spin_imgsz_seq_detect: "Square model input size for the sequential first-stage detector.",
            self.spin_imgsz_seq_crop_obb: "Square model input size for sequential crop-focused OBB training.",
            self.spin_imgsz_seq_crop_segment: "Square model input size for sequential crop-focused segmentation training.",
            self.combo_model_obb_direct: "Base checkpoint used to initialize direct OBB training.",
            self.combo_model_detect_direct: "Base checkpoint used to initialize direct detection training.",
            self.combo_model_segment_direct: "Base checkpoint used to initialize direct segmentation training.",
            self.combo_model_seq_detect: "Base checkpoint used to initialize sequential first-stage detection training.",
            self.combo_model_seq_crop_obb: "Base checkpoint used to initialize sequential crop-focused OBB training.",
            self.combo_model_seq_crop_segment: "Base checkpoint used to initialize sequential crop-focused segmentation training.",
            self.aug_group: "Enable or disable the image augmentations below for this training run.",
            self.aug_fliplr: "Probability of a left-right image flip during augmentation.",
            self.aug_flipud: "Probability of an up-down image flip during augmentation.",
            self.aug_degrees: "Maximum random rotation angle used during augmentation.",
            self.aug_mosaic: "Probability of mosaic augmentation, which combines multiple images into one training example.",
            self.aug_mixup: "Probability of mixup augmentation, which blends two training images.",
            self.aug_hsv_h: "Maximum hue adjustment used during colour augmentation.",
            self.aug_hsv_s: "Maximum saturation adjustment used during colour augmentation.",
            self.aug_hsv_v: "Maximum brightness adjustment used during colour augmentation.",
            self.btn_refresh_dataset_fit: "Reinspect the configured sources and update the overview cards.",
            self.btn_start: "Build the selected datasets and begin training.",
            self.btn_cancel: "Request a safe stop after the current training or dataset boundary.",
            self.btn_resume: "Resume the most recent compatible interrupted training run.",
            self.btn_save_config: "Save the current training controls as a reusable preset.",
            self.btn_load_config: "Load a previously saved training preset into this dialog.",
        }
        for control, tip in tips.items():
            control.setToolTip(tip)

        for control in self.findChildren(QWidget):
            if control.toolTip():
                continue
            if isinstance(control, QAbstractSpinBox):
                tip = "Enter a numeric value for this training setting."
            elif isinstance(control, QComboBox):
                tip = "Choose an option for this training setting."
            elif isinstance(control, QLineEdit):
                tip = "Enter a value for this training setting."
            elif isinstance(control, QAbstractButton):
                tip = "Activate or toggle this training option."
            else:
                continue
            control.setToolTip(tip)

    def _apply_training_dialog_styles(self) -> None:
        self.setStyleSheet(self.styleSheet() + """
QFrame#detectkitTrainingHero {
    border: 1px solid #3e3e42;
    border-radius: 12px;
    background-color: #20252d;
}
QLabel#detectkitTrainingEyebrow {
    color: #9cdcfe;
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 0.08em;
}
QLabel#detectkitTrainingTitle {
    color: #ffffff;
    font-size: 20px;
    font-weight: 700;
}
QLabel#detectkitTrainingBody,
QLabel#detectkitTrainingNote,
QLabel#detectkitTrainingSummaryBody,
QLabel#detectkitRunStatus {
    color: #d6d6d6;
}
QLabel#detectkitTrainingChip {
    background-color: #252526;
    border: 1px solid #3e3e42;
    border-radius: 999px;
    color: #d6d6d6;
    padding: 4px 10px;
    font-size: 11px;
    font-weight: 600;
}
QFrame#detectkitTrainingSummaryCard,
QFrame#detectkitTrainingRoleCard {
    border: 1px solid #3e3e42;
    border-radius: 10px;
    background-color: #202124;
}
QLabel#detectkitTrainingSummaryTitle,
QLabel#detectkitTrainingRoleTitle {
    color: #ffffff;
    font-size: 13px;
    font-weight: 700;
}
QLabel#detectkitTrainingRoleBody {
    color: #cfcfcf;
}
QTabWidget::pane {
    border: 1px solid #3e3e42;
    border-radius: 10px;
    background-color: #1e1e1e;
    top: -1px;
}
QTabBar::tab {
    background-color: #252526;
    color: #cfcfcf;
    border: 1px solid #3e3e42;
    border-bottom: none;
    padding: 8px 14px;
    min-width: 120px;
    border-top-left-radius: 8px;
    border-top-right-radius: 8px;
    margin-right: 4px;
}
QTabBar::tab:selected {
    background-color: #1e1e1e;
    color: #ffffff;
}
""")

    def _wrap_scroll_page(self, widget: QWidget) -> QScrollArea:
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(scroll.Shape.NoFrame)
        scroll.setWidget(widget)
        return scroll

    def _build_header(self) -> QFrame:
        frame = QFrame()
        frame.setObjectName("detectkitTrainingHero")
        layout = QVBoxLayout(frame)
        layout.setContentsMargins(18, 18, 18, 18)
        layout.setSpacing(10)

        eyebrow = QLabel("DETECTKIT")
        eyebrow.setObjectName("detectkitTrainingEyebrow")
        layout.addWidget(eyebrow)

        title = QLabel("Clear staged training for DetectKit models")
        title.setObjectName("detectkitTrainingTitle")
        title.setWordWrap(True)
        layout.addWidget(title)

        body = QLabel(
            "Choose a training style and model target, verify the plan, then run with a live view"
            " of progress and outputs. Advanced settings stay available without"
            " crowding the first screen."
        )
        body.setObjectName("detectkitTrainingBody")
        body.setWordWrap(True)
        layout.addWidget(body)

        chip_row = QHBoxLayout()
        chip_row.setSpacing(8)
        chip_row.addWidget(self._build_workflow_chip("1. Choose plan"))
        chip_row.addWidget(self._build_workflow_chip("2. Train and review"))
        chip_row.addStretch(1)
        layout.addLayout(chip_row)
        return frame

    def _build_workflow_chip(self, text: str) -> QLabel:
        chip = QLabel(text)
        chip.setObjectName("detectkitTrainingChip")
        return chip

    def _build_overview_tab(self) -> QWidget:
        page = QWidget()
        layout = QGridLayout(page)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(12)

        selection = self._build_training_selection_group()
        selection.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Maximum
        )
        layout.addWidget(selection, 0, 0)
        layout.addWidget(self._build_summary_card(), 0, 1)
        layout.addWidget(self._build_dataset_fit_card(), 1, 0, 1, 2)
        layout.addWidget(self._build_source_preview_group(), 2, 0, 1, 2)
        layout.setColumnStretch(0, 3)
        layout.setColumnStretch(1, 2)
        layout.setRowStretch(2, 1)
        return self._wrap_scroll_page(page)

    def _build_training_tab(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(12)

        layout.addWidget(self._build_config_group())
        layout.addWidget(self._build_hyperparams_group())

        lower_row = QHBoxLayout()
        lower_row.setSpacing(12)
        base_models = self._build_base_models_group()
        augmentation = self._build_augmentation_group()
        for group in (base_models, augmentation):
            group.setSizePolicy(
                QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Maximum
            )
        lower_row.addWidget(base_models, 1, Qt.AlignmentFlag.AlignTop)
        lower_row.addWidget(augmentation, 1, Qt.AlignmentFlag.AlignTop)
        layout.addLayout(lower_row)

        layout.addWidget(self._build_slice_group())

        layout.addStretch(1)
        return self._wrap_scroll_page(page)

    def _build_slice_group(self) -> QGroupBox:
        from ..panels.slice_settings_widget import SliceSettingsGroup

        self.slice_group = SliceSettingsGroup()
        return self.slice_group

    def _build_summary_card(self) -> QFrame:
        frame = QFrame()
        frame.setObjectName("detectkitTrainingSummaryCard")
        layout = QVBoxLayout(frame)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(10)

        title = QLabel("Training Plan")
        title.setObjectName("detectkitTrainingSummaryTitle")
        layout.addWidget(title)

        self.plan_summary = QLabel("")
        self.plan_summary.setObjectName("detectkitTrainingSummaryBody")
        self.plan_summary.setWordWrap(True)
        self.plan_summary.setTextFormat(self.plan_summary.textFormat())
        layout.addWidget(self.plan_summary)

        note = self._build_section_note(
            "The overview reflects the current selections, publish policy, and dataset readiness."
        )
        layout.addWidget(note)
        layout.addStretch(1)
        return frame

    def _build_dataset_fit_card(self) -> QFrame:
        frame = QFrame()
        frame.setObjectName("detectkitTrainingSummaryCard")
        layout = QVBoxLayout(frame)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(10)

        title = QLabel("Dataset Fit")
        title.setObjectName("detectkitTrainingSummaryTitle")
        layout.addWidget(title)

        self.dataset_fit_status = QLabel(
            "Checks whether the current sources and image sizes are a good fit for the selected recipe."
        )
        self.dataset_fit_status.setObjectName("detectkitTrainingSummaryBody")
        self.dataset_fit_status.setWordWrap(True)
        layout.addWidget(self.dataset_fit_status)

        self.btn_refresh_dataset_fit = QPushButton("Refresh Overview Data")
        self.btn_refresh_dataset_fit.clicked.connect(self._refresh_overview_data_cards)
        layout.addWidget(self.btn_refresh_dataset_fit)

        self.dataset_fit_view = QTextEdit()
        self.dataset_fit_view.setReadOnly(True)
        self.dataset_fit_view.setMinimumHeight(180)
        self.dataset_fit_view.setPlaceholderText(
            "Dataset fit guidance will appear here."
        )
        layout.addWidget(self.dataset_fit_view)
        return frame

    def _build_source_preview_group(self) -> QGroupBox:
        self.source_preview_group = QGroupBox("Source Samples")
        layout = QVBoxLayout(self.source_preview_group)
        layout.setSpacing(10)

        self.source_preview_note = self._build_section_note(
            "Representative frames from the configured DetectKit sources. Use this to sanity-check what the training dialog is about to build from."
        )
        layout.addWidget(self.source_preview_note)

        self.source_preview_status = QLabel(
            "Previewing the first labeled samples discovered in each source dataset."
        )
        self.source_preview_status.setObjectName("detectkitTrainingSummaryBody")
        self.source_preview_status.setWordWrap(True)
        layout.addWidget(self.source_preview_status)

        self.source_preview_scroll = QScrollArea()
        self.source_preview_scroll.setWidgetResizable(True)
        self.source_preview_scroll.setFrameShape(QFrame.Shape.NoFrame)
        self.source_preview_scroll.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAsNeeded
        )
        self.source_preview_scroll.setVerticalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        self.source_preview_scroll.setMinimumHeight(220)

        self.source_preview_container = QWidget()
        self.source_preview_cards_layout = QHBoxLayout(self.source_preview_container)
        self.source_preview_cards_layout.setContentsMargins(0, 0, 0, 0)
        self.source_preview_cards_layout.setSpacing(8)
        self.source_preview_scroll.setWidget(self.source_preview_container)
        layout.addWidget(self.source_preview_scroll)
        return self.source_preview_group

    def _build_training_selection_group(self) -> QGroupBox:
        gb = QGroupBox("Training Selection")
        layout = QGridLayout(gb)
        layout.setContentsMargins(16, 18, 16, 14)
        layout.setHorizontalSpacing(10)
        layout.setVerticalSpacing(8)

        note = self._build_section_note(
            "Choose one style and one target. Train another model in a separate run."
        )
        layout.addWidget(note, 0, 0, 1, 4)

        self.mode_combo = QComboBox()
        self.mode_combo.addItem("Direct", "direct")
        self.mode_combo.addItem("Sequential", "sequential")
        self.mode_combo.addItem("Semantic", "semantic")
        self.mode_combo.setMinimumWidth(150)
        layout.addWidget(QLabel("Style"), 1, 0)
        layout.addWidget(self.mode_combo, 1, 1)

        self.task_combo = QComboBox()
        self.task_combo.addItem("OBB", "obb")
        self.task_combo.addItem("Detect", "detect")
        self.task_combo.addItem("Segment", "segment")
        self.task_combo.setMinimumWidth(150)
        layout.addWidget(QLabel("Target"), 1, 2)
        layout.addWidget(self.task_combo, 1, 3)

        self.selection_description = QLabel("")
        self.selection_description.setObjectName("detectkitTrainingRoleBody")
        self.selection_description.setWordWrap(True)
        layout.addWidget(self.selection_description, 2, 0, 1, 4)
        layout.setColumnStretch(1, 1)
        layout.setColumnStretch(3, 1)
        return gb

    @staticmethod
    def _build_section_note(text: str) -> QLabel:
        label = QLabel(text)
        label.setObjectName("detectkitTrainingNote")
        label.setWordWrap(True)
        return label

    def _build_role_card(
        self,
        checkbox: QCheckBox,
        title: str,
        description: str,
    ) -> QFrame:
        frame = QFrame()
        frame.setObjectName("detectkitTrainingRoleCard")
        layout = QVBoxLayout(frame)
        layout.setContentsMargins(14, 14, 14, 14)
        layout.setSpacing(8)

        checkbox.setText(title)
        layout.addWidget(checkbox)

        label = QLabel(description)
        label.setObjectName("detectkitTrainingRoleBody")
        label.setWordWrap(True)
        layout.addWidget(label)
        layout.addStretch(1)
        return frame

    def _connect_summary_signals(self) -> None:
        self.mode_combo.currentIndexChanged.connect(self._on_training_selection_changed)
        self.task_combo.currentIndexChanged.connect(self._on_training_selection_changed)
        self.spin_crop_pad.valueChanged.connect(self._mark_dataset_fit_dirty)
        self.spin_crop_min_px.valueChanged.connect(self._mark_dataset_fit_dirty)
        self.chk_crop_square.toggled.connect(self._mark_dataset_fit_dirty)
        self.spin_imgsz_obb_direct.valueChanged.connect(self._mark_dataset_fit_dirty)
        self.spin_imgsz_detect_direct.valueChanged.connect(self._mark_dataset_fit_dirty)
        self.spin_imgsz_segment_direct.valueChanged.connect(
            self._mark_dataset_fit_dirty
        )
        self.spin_imgsz_seq_crop_obb.valueChanged.connect(self._mark_dataset_fit_dirty)
        self.spin_imgsz_seq_crop_segment.valueChanged.connect(
            self._mark_dataset_fit_dirty
        )
        for spinner in (
            self.spin_imgsz_obb_direct,
            self.spin_imgsz_detect_direct,
            self.spin_imgsz_segment_direct,
        ):
            spinner.valueChanged.connect(self._sync_slice_model_input_size)

        for spinner in (
            self.spin_train,
            self.spin_val,
            self.spin_seed,
        ):
            spinner.valueChanged.connect(self._refresh_summary)

        self.combo_device.currentTextChanged.connect(self._refresh_summary)

    def _selected_role_keys(self) -> list[str]:
        selected = []
        if self.chk_role_obb_direct.isChecked():
            selected.append("obb_direct")
        if self.chk_role_detect_direct.isChecked():
            selected.append("detect_direct")
        if self.chk_role_segment_direct.isChecked():
            selected.append("segment_direct")
        if self.chk_role_seq_detect.isChecked():
            selected.append("seq_detect")
        if self.chk_role_seq_crop_obb.isChecked():
            selected.append("seq_crop_obb")
        if self.chk_role_seq_crop_segment.isChecked():
            selected.append("seq_crop_segment")
        if self.chk_semantic_sam3.isChecked():
            selected.append("semantic_sam3")
        return selected

    @staticmethod
    def _role_display_name(role: str) -> str:
        return {
            "obb_direct": "OBB direct",
            "seq_detect": "Sequence detect",
            "seq_crop_obb": "Sequence crop OBB",
        }.get(role, role.replace("_", " ").title())

    @staticmethod
    def _preview_values(values: list[str], limit: int = 3) -> str:
        if not values:
            return "none"
        if len(values) <= limit:
            return ", ".join(values)
        return ", ".join(values[:limit]) + f" +{len(values) - limit} more"

    def _refresh_summary(self, *_args) -> None:
        if not hasattr(self, "plan_summary"):
            return

        selected_roles = self._selected_role_keys()
        role_labels = [self._role_display_name(role) for role in selected_roles]
        class_names = self._class_names()
        device = self.combo_device.currentText().strip() or "cpu"
        mode_label = self.mode_combo.currentText().strip() or "Direct"
        task_label = self.task_combo.currentText().strip() or "OBB"
        source_fit = self._source_fit_summary()
        summary = (
            f"<b>Plan:</b> {mode_label} {task_label}<br>"
            f"<b>Stages:</b> {self._preview_values(role_labels)}<br>"
            f"<b>Classes:</b> {len(class_names)} ({self._preview_values(class_names)})<br>"
            f"<b>Sources:</b> {source_fit}<br>"
            f"<b>Split:</b> {int(round(self.spin_train.value() * 100.0))}% train / "
            f"{int(round(self.spin_val.value() * 100.0))}% val"
            f" &nbsp;&bull;&nbsp; <b>Seed:</b> {self.spin_seed.value()}<br>"
            f"<b>Device:</b> {device}"
        )
        self.plan_summary.setText(summary)

    def _set_run_status(self, message: str) -> None:
        if hasattr(self, "run_status_label"):
            self.run_status_label.setText(message)

    def _set_training_running(self, running: bool) -> None:
        """Lock the configuration UI while a training run is active."""
        self._training_running = bool(running)
        if hasattr(self, "training_tabs"):
            self.training_tabs.setEnabled(not running)
        for btn_attr in (
            "btn_start",
            "btn_resume",
            "btn_save_config",
            "btn_load_config",
        ):
            btn = getattr(self, btn_attr, None)
            if btn is not None:
                btn.setEnabled(not running)
        if hasattr(self, "btn_cancel"):
            self.btn_cancel.setEnabled(running)
        # Re-evaluate "Resume" availability when a run completes.
        if not running:
            self._update_resume_enabled()

    def _update_resume_enabled(self) -> None:
        if not hasattr(self, "btn_resume"):
            return
        has_resume = any(
            r.get("_run_dir")
            and Path(r["_run_dir"]).joinpath("weights", "last.pt").exists()
            for r in (self._last_training_results or [])
        )
        self.btn_resume.setEnabled(has_resume)

    # --- 1. Internal role state ---

    def _build_role_state(self) -> None:
        """Keep legacy role fields in sync without exposing stage checkboxes."""
        self.chk_role_obb_direct = QCheckBox("obb_direct")
        self.chk_role_detect_direct = QCheckBox("detect_direct")
        self.chk_role_segment_direct = QCheckBox("segment_direct")
        self.chk_role_seq_detect = QCheckBox("seq_detect")
        self.chk_role_seq_crop_obb = QCheckBox("seq_crop_obb")
        self.chk_role_seq_crop_segment = QCheckBox("seq_crop_segment")
        self.chk_semantic_sam3 = QCheckBox("semantic_sam3")

    # --- 2. Config ---

    def _build_config_group(self) -> QGroupBox:
        gb = QGroupBox("Dataset And Runtime")
        grid = QGridLayout(gb)
        grid.setContentsMargins(16, 18, 16, 14)
        grid.setHorizontalSpacing(12)
        grid.setVerticalSpacing(8)
        grid.addWidget(
            self._build_section_note(
                "Project classes are managed by DetectKit. Configure split, runtime, and sequential crop derivation here."
            ),
            0,
            0,
            1,
            4,
        )

        from ..models import normalize_class_names

        class_names_text = ", ".join(normalize_class_names(self._project.class_names))
        self.class_names_label = QLabel(class_names_text)
        self.class_names_label.setObjectName("detectkitTrainingNote")
        self.class_names_label.setWordWrap(True)
        self.class_names_label.setToolTip(
            "Class names come from the project. Edit them by changing project metadata."
        )
        grid.addWidget(QLabel("Project classes"), 1, 0)
        grid.addWidget(self.class_names_label, 1, 1)

        # Split
        self.spin_train = QDoubleSpinBox()
        self.spin_train.setRange(0.05, 0.95)
        self.spin_train.setSingleStep(0.05)
        self.spin_train.setValue(0.8)
        self.spin_val = QDoubleSpinBox()
        self.spin_val.setRange(0.05, 0.95)
        self.spin_val.setSingleStep(0.05)
        self.spin_val.setValue(0.2)
        h_split = QHBoxLayout()
        h_split.setContentsMargins(0, 0, 0, 0)
        h_split.setSpacing(6)
        h_split.addWidget(QLabel("train"))
        h_split.addWidget(self.spin_train)
        h_split.addWidget(QLabel("val"))
        h_split.addWidget(self.spin_val)
        split_widget = QWidget()
        split_widget.setLayout(h_split)
        grid.addWidget(QLabel("Dataset split"), 2, 0)
        grid.addWidget(split_widget, 2, 1)

        self.spin_seed = QSpinBox()
        self.spin_seed.setRange(0, 999999)
        self.spin_seed.setValue(42)
        grid.addWidget(QLabel("Random seed"), 1, 2)
        grid.addWidget(self.spin_seed, 1, 3)

        self.chk_dedup = QCheckBox("Deduplicate source images by content hash")
        self.chk_dedup.setChecked(True)
        grid.addWidget(self.chk_dedup, 2, 2, 1, 2)

        # Crop derivation
        self.spin_crop_pad = QDoubleSpinBox()
        self.spin_crop_pad.setRange(0.0, 1.0)
        self.spin_crop_pad.setSingleStep(0.01)
        self.spin_crop_pad.setValue(0.15)
        self.spin_crop_min_px = QSpinBox()
        self.spin_crop_min_px.setRange(8, 2048)
        self.spin_crop_min_px.setValue(64)
        self.chk_crop_square = QCheckBox("Square crop")
        self.chk_crop_square.setChecked(True)
        h_crop = QHBoxLayout()
        h_crop.setContentsMargins(0, 0, 0, 0)
        h_crop.setSpacing(6)
        h_crop.addWidget(QLabel("pad"))
        h_crop.addWidget(self.spin_crop_pad)
        h_crop.addWidget(QLabel("min px"))
        h_crop.addWidget(self.spin_crop_min_px)
        h_crop.addWidget(self.chk_crop_square)
        self.crop_settings_label = QLabel("Sequence crop settings")
        self.crop_settings_widget = QWidget()
        self.crop_settings_widget.setLayout(h_crop)
        grid.addWidget(self.crop_settings_label, 3, 0)
        grid.addWidget(self.crop_settings_widget, 3, 1, 1, 3)

        # Device — PyTorch training devices only.
        self.combo_device = QComboBox()
        self.combo_device.setEditable(False)
        self.combo_device.setToolTip(
            "PyTorch training device. ONNX/TensorRT exports happen later from history."
        )
        self.combo_device.addItems(self._build_device_options())
        grid.addWidget(QLabel("Compute device"), 4, 0)
        grid.addWidget(self.combo_device, 4, 1)
        grid.setColumnStretch(1, 2)
        grid.setColumnStretch(3, 1)

        return gb

    # --- 3. Hyperparameters ---

    def _build_hyperparams_group(self) -> QGroupBox:
        gb = QGroupBox("Training Hyperparameters")
        g = QGridLayout(gb)
        g.setContentsMargins(16, 18, 16, 14)
        g.setHorizontalSpacing(12)
        g.setVerticalSpacing(8)

        g.addWidget(
            self._build_section_note(
                "These values apply to every selected role unless a role-specific image size is set below."
            ),
            0,
            0,
            1,
            6,
        )

        # Row 0: epochs, batch + auto, lr0
        self.spin_epochs = QSpinBox()
        self.spin_epochs.setRange(1, 1000)
        self.spin_epochs.setValue(100)
        self.spin_epochs.setMaximumWidth(160)
        g.addWidget(QLabel("epochs"), 1, 0)
        g.addWidget(self.spin_epochs, 1, 1)

        self.spin_batch = QSpinBox()
        self.spin_batch.setRange(1, 256)
        self.spin_batch.setValue(16)
        self.spin_batch.setMaximumWidth(120)
        self.chk_auto_batch = QCheckBox("Auto")
        self.chk_auto_batch.setToolTip(
            "Let Ultralytics auto-detect optimal batch size (batch=-1)."
        )
        self.chk_auto_batch.toggled.connect(
            lambda checked: self.spin_batch.setEnabled(not checked)
        )
        batch_layout = QHBoxLayout()
        batch_layout.setContentsMargins(0, 0, 0, 0)
        batch_layout.setSpacing(6)
        batch_layout.addWidget(self.spin_batch)
        batch_layout.addWidget(self.chk_auto_batch)
        batch_widget = QWidget()
        batch_widget.setLayout(batch_layout)
        g.addWidget(QLabel("batch"), 1, 2)
        g.addWidget(batch_widget, 1, 3)

        self.spin_lr0 = QDoubleSpinBox()
        self.spin_lr0.setRange(1e-5, 1.0)
        self.spin_lr0.setDecimals(5)
        self.spin_lr0.setValue(0.01)
        self.spin_lr0.setMaximumWidth(160)
        g.addWidget(QLabel("lr0"), 1, 4)
        g.addWidget(self.spin_lr0, 1, 5)

        # Row 1: patience, workers, cache
        self.spin_patience = QSpinBox()
        self.spin_patience.setRange(1, 500)
        self.spin_patience.setValue(30)
        self.spin_patience.setMaximumWidth(160)
        g.addWidget(QLabel("patience"), 2, 0)
        g.addWidget(self.spin_patience, 2, 1)

        self.spin_workers = QSpinBox()
        self.spin_workers.setRange(0, 32)
        self.spin_workers.setValue(8)
        self.spin_workers.setMaximumWidth(160)
        g.addWidget(QLabel("workers"), 2, 2)
        g.addWidget(self.spin_workers, 2, 3)

        self.chk_cache = QCheckBox("Cache")
        g.addWidget(self.chk_cache, 2, 4, 1, 2)

        # Row 2: per-role imgsz
        self.spin_imgsz_obb_direct = QSpinBox()
        self.spin_imgsz_obb_direct.setRange(64, 2048)
        self.spin_imgsz_obb_direct.setValue(640)
        self.spin_imgsz_obb_direct.setMaximumWidth(160)
        self.label_imgsz_obb_direct = QLabel("imgsz (obb_direct)")
        g.addWidget(self.label_imgsz_obb_direct, 3, 0)
        g.addWidget(self.spin_imgsz_obb_direct, 3, 1)

        self.spin_imgsz_seq_detect = QSpinBox()
        self.spin_imgsz_seq_detect.setRange(64, 2048)
        self.spin_imgsz_seq_detect.setValue(640)
        self.spin_imgsz_seq_detect.setMaximumWidth(160)
        self.label_imgsz_seq_detect = QLabel("imgsz (seq_detect)")
        g.addWidget(self.label_imgsz_seq_detect, 3, 2)
        g.addWidget(self.spin_imgsz_seq_detect, 3, 3)

        self.spin_imgsz_seq_crop_obb = QSpinBox()
        self.spin_imgsz_seq_crop_obb.setRange(64, 2048)
        self.spin_imgsz_seq_crop_obb.setValue(160)
        self.spin_imgsz_seq_crop_obb.setMaximumWidth(160)
        self.spin_imgsz_seq_crop_obb.setToolTip(
            "Must match YOLO_SEQ_STAGE2_IMGSZ used during inference (default 160)."
        )
        self.label_imgsz_seq_crop_obb = QLabel("imgsz (seq_crop_obb)")
        g.addWidget(self.label_imgsz_seq_crop_obb, 3, 4)
        g.addWidget(self.spin_imgsz_seq_crop_obb, 3, 5)

        self.spin_imgsz_detect_direct = QSpinBox()
        self.spin_imgsz_detect_direct.setRange(64, 2048)
        self.spin_imgsz_detect_direct.setValue(640)
        self.spin_imgsz_detect_direct.setMaximumWidth(160)
        self.label_imgsz_detect_direct = QLabel("imgsz (detect_direct)")
        g.addWidget(self.label_imgsz_detect_direct, 4, 0)
        g.addWidget(self.spin_imgsz_detect_direct, 4, 1)

        self.spin_imgsz_segment_direct = QSpinBox()
        self.spin_imgsz_segment_direct.setRange(64, 2048)
        self.spin_imgsz_segment_direct.setValue(640)
        self.spin_imgsz_segment_direct.setMaximumWidth(160)
        self.label_imgsz_segment_direct = QLabel("imgsz (segment_direct)")
        g.addWidget(self.label_imgsz_segment_direct, 4, 2)
        g.addWidget(self.spin_imgsz_segment_direct, 4, 3)

        self.spin_imgsz_seq_crop_segment = QSpinBox()
        self.spin_imgsz_seq_crop_segment.setRange(64, 2048)
        self.spin_imgsz_seq_crop_segment.setValue(160)
        self.spin_imgsz_seq_crop_segment.setMaximumWidth(160)
        self.spin_imgsz_seq_crop_segment.setToolTip(
            "Must match YOLO_SEQ_STAGE2_IMGSZ used during inference (default 160)."
        )
        self.label_imgsz_seq_crop_segment = QLabel("imgsz (seq_crop_segment)")
        g.addWidget(self.label_imgsz_seq_crop_segment, 4, 4)
        g.addWidget(self.spin_imgsz_seq_crop_segment, 4, 5)

        for col in (1, 3, 5):
            g.setColumnStretch(col, 1)
        return gb

    # --- 4. Base Models ---

    @staticmethod
    def _yolo_detect_options() -> list[str]:
        sizes = ("n", "s", "m", "l", "x")
        families = ("yolov8", "yolo11", "yolo12", "yolo26")
        return [f"{family}{size}.pt" for family in families for size in sizes]

    @staticmethod
    def _yolo_obb_options() -> list[str]:
        sizes = ("n", "s", "m", "l", "x")
        families = ("yolov8", "yolo11", "yolo12", "yolo26")
        return [f"{family}{size}-obb.pt" for family in families for size in sizes]

    def _build_base_models_group(self) -> QGroupBox:
        gb = QGroupBox("Base Checkpoints")
        grid = QGridLayout(gb)
        grid.setContentsMargins(16, 18, 16, 14)
        grid.setHorizontalSpacing(12)
        grid.setVerticalSpacing(8)
        grid.addWidget(
            self._build_section_note(
                "Choose the starting weights for the active stages. You can type a custom path."
            ),
            0,
            0,
            1,
            4,
        )

        obb_options = self._yolo_obb_options()
        detect_options = self._yolo_detect_options()
        seg_options = ["yolo26s-seg.pt", "yolo11s-seg.pt", "yolo11m-seg.pt"]
        default_obb = "yolo26s-obb.pt"
        default_detect = "yolo26s.pt"
        default_seg = "yolo26s-seg.pt"

        self.combo_model_obb_direct = QComboBox()
        self.combo_model_obb_direct.setEditable(True)
        self.combo_model_obb_direct.addItems(obb_options)
        self.combo_model_obb_direct.setCurrentText(default_obb)
        self.combo_model_obb_direct.setMinimumWidth(190)
        self.label_model_obb_direct = QLabel("Direct OBB")

        self.combo_model_detect_direct = QComboBox()
        self.combo_model_detect_direct.setEditable(True)
        self.combo_model_detect_direct.addItems(detect_options)
        self.combo_model_detect_direct.setCurrentText(default_detect)
        self.combo_model_detect_direct.setMinimumWidth(190)
        self.label_model_detect_direct = QLabel("Direct detect")

        self.combo_model_segment_direct = QComboBox()
        self.combo_model_segment_direct.setEditable(True)
        self.combo_model_segment_direct.addItems(seg_options)
        self.combo_model_segment_direct.setCurrentText(default_seg)
        self.combo_model_segment_direct.setMinimumWidth(190)
        self.label_model_segment_direct = QLabel("Direct segment")

        self.combo_model_seq_detect = QComboBox()
        self.combo_model_seq_detect.setEditable(True)
        self.combo_model_seq_detect.addItems(detect_options)
        self.combo_model_seq_detect.setCurrentText(default_detect)
        self.combo_model_seq_detect.setMinimumWidth(190)
        self.label_model_seq_detect = QLabel("Sequential detector")

        self.combo_model_seq_crop_obb = QComboBox()
        self.combo_model_seq_crop_obb.setEditable(True)
        self.combo_model_seq_crop_obb.addItems(obb_options)
        self.combo_model_seq_crop_obb.setCurrentText(default_obb)
        self.combo_model_seq_crop_obb.setMinimumWidth(190)
        self.label_model_seq_crop_obb = QLabel("Sequential crop OBB")

        self.combo_model_seq_crop_segment = QComboBox()
        self.combo_model_seq_crop_segment.setEditable(True)
        self.combo_model_seq_crop_segment.addItems(seg_options)
        self.combo_model_seq_crop_segment.setCurrentText(default_seg)
        self.combo_model_seq_crop_segment.setMinimumWidth(190)
        self.label_model_seq_crop_segment = QLabel("Sequential crop segment")

        model_rows = (
            (
                self.label_model_obb_direct,
                self.combo_model_obb_direct,
                self.label_model_detect_direct,
                self.combo_model_detect_direct,
            ),
            (
                self.label_model_segment_direct,
                self.combo_model_segment_direct,
                self.label_model_seq_detect,
                self.combo_model_seq_detect,
            ),
            (
                self.label_model_seq_crop_obb,
                self.combo_model_seq_crop_obb,
                self.label_model_seq_crop_segment,
                self.combo_model_seq_crop_segment,
            ),
        )
        for row, (left_label, left_combo, right_label, right_combo) in enumerate(
            model_rows, start=1
        ):
            grid.addWidget(left_label, row, 0)
            grid.addWidget(left_combo, row, 1)
            grid.addWidget(right_label, row, 2)
            grid.addWidget(right_combo, row, 3)

        grid.setColumnStretch(1, 1)
        grid.setColumnStretch(3, 1)

        return gb

    # --- 5. Augmentation ---

    def _build_augmentation_group(self) -> QGroupBox:
        self.aug_group = QGroupBox("Augmentation")
        self.aug_group.setCheckable(True)
        self.aug_group.setChecked(True)
        v = QVBoxLayout(self.aug_group)
        v.setContentsMargins(16, 18, 16, 14)
        v.setSpacing(8)

        note = QLabel(
            "These are passed directly to Ultralytics. "
            "Set fliplr=0 for asymmetric animals."
        )
        note.setObjectName("detectkitTrainingNote")
        note.setWordWrap(True)
        v.addWidget(note)

        grid = QGridLayout()
        grid.setHorizontalSpacing(12)
        grid.setVerticalSpacing(8)

        def _spin(default: float, maximum: float = 1.0) -> QDoubleSpinBox:
            sb = QDoubleSpinBox()
            sb.setRange(0.0, maximum)
            sb.setDecimals(3)
            sb.setSingleStep(0.05)
            sb.setValue(default)
            sb.setMaximumWidth(140)
            return sb

        self.aug_fliplr = _spin(0.5)
        self.aug_flipud = _spin(0.0)
        self.aug_degrees = _spin(0.0, 360.0)
        self.aug_mosaic = _spin(1.0)
        self.aug_mixup = _spin(0.0)
        self.aug_hsv_h = _spin(0.015)
        self.aug_hsv_s = _spin(0.7)
        self.aug_hsv_v = _spin(0.4)
        aug_rows = (
            ("fliplr", self.aug_fliplr, "flipud", self.aug_flipud),
            ("degrees", self.aug_degrees, "mosaic", self.aug_mosaic),
            ("mixup", self.aug_mixup, "hsv h", self.aug_hsv_h),
            ("hsv s", self.aug_hsv_s, "hsv v", self.aug_hsv_v),
        )
        for row, (left_name, left_spin, right_name, right_spin) in enumerate(aug_rows):
            grid.addWidget(QLabel(left_name), row, 0)
            grid.addWidget(left_spin, row, 1)
            grid.addWidget(QLabel(right_name), row, 2)
            grid.addWidget(right_spin, row, 3)
        grid.setColumnStretch(1, 1)
        grid.setColumnStretch(3, 1)
        v.addLayout(grid)
        return self.aug_group

    # --- 6. Run Controls ---

    def _build_run_group(self) -> QGroupBox:
        gb = QGroupBox("Run Session")
        v = QVBoxLayout(gb)
        v.setSpacing(10)

        self.run_status_label = QLabel(
            "Ready to start training for the selected roles."
        )
        self.run_status_label.setObjectName("detectkitRunStatus")
        self.run_status_label.setWordWrap(True)
        v.addWidget(self.run_status_label)

        row1 = QHBoxLayout()
        self.btn_start = QPushButton("Start Training")
        self.btn_cancel = QPushButton("Stop Run")
        self.btn_cancel.setEnabled(False)
        self.btn_resume = QPushButton("Resume Last Run")
        self.btn_resume.setEnabled(False)
        self.btn_resume.setToolTip(
            "Resume training from last.pt checkpoint of the most recent run."
        )
        self.btn_save_config = QPushButton("Save Preset")
        self.btn_load_config = QPushButton("Load Preset")
        row1.addWidget(self.btn_start)
        row1.addWidget(self.btn_cancel)
        row1.addWidget(self.btn_resume)
        row1.addWidget(self.btn_save_config)
        row1.addWidget(self.btn_load_config)
        v.addLayout(row1)

        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        self.progress.setFormat("Idle")
        v.addWidget(self.progress)

        monitor_row = QHBoxLayout()
        monitor_row.setSpacing(10)

        loss_box = QGroupBox("Loss Curve")
        loss_layout = QVBoxLayout(loss_box)
        loss_layout.setContentsMargins(8, 14, 8, 8)
        loss_layout.setSpacing(6)
        loss_widget = self._build_loss_plot()
        loss_widget.setMinimumHeight(220)
        loss_layout.addWidget(loss_widget, 1)
        monitor_row.addWidget(loss_box, 1)

        log_box = QGroupBox("Session Log")
        log_layout = QVBoxLayout(log_box)
        log_layout.setContentsMargins(8, 14, 8, 8)
        log_layout.setSpacing(6)
        log_layout.addWidget(self._build_log(), 1)
        monitor_row.addWidget(log_box, 1)

        v.addLayout(monitor_row, 1)

        self.btn_start.clicked.connect(self._start_training)
        self.btn_cancel.clicked.connect(self._cancel_training)
        self.btn_resume.clicked.connect(self._resume_training)
        self.btn_save_config.clicked.connect(self._save_training_config)
        self.btn_load_config.clicked.connect(self._load_training_config)

        return gb

    # --- 8. Loss Plot ---

    def _build_loss_plot(self) -> QWidget:
        try:
            from hydra_suite.trackerkit.gui.widgets.loss_plot_widget import (
                LossPlotWidget,
            )

            self.loss_plot = LossPlotWidget()
            self.loss_plot.setMinimumHeight(220)
            self.loss_plot.setMinimumWidth(360)
            self.loss_plot.setSizePolicy(
                QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
            )
            return self.loss_plot
        except ImportError:
            self.loss_plot = None
            placeholder = QLabel("Loss plot not available (trackerkit not installed).")
            placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
            placeholder.setStyleSheet(
                "color: #cfcfcf; font-style: italic; background-color: #1e1e1e;"
                " border: 1px solid #3e3e42; border-radius: 6px; padding: 12px;"
            )
            placeholder.setMinimumHeight(220)
            return placeholder

    # --- 9. Log ---

    def _build_log(self) -> QTextEdit:
        self.log_view = QTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setPlaceholderText("Training log output appears here.")
        self.log_view.setMinimumHeight(150)
        return self.log_view

    # ------------------------------------------------------------------
    # Project round-trip
    # ------------------------------------------------------------------

    def _load_from_project(self) -> None:
        proj = self._project

        self._set_combo_data(self.mode_combo, getattr(proj, "training_mode", "direct"))
        self._set_combo_data(self.task_combo, getattr(proj, "training_task", "obb"))
        self._apply_selection_roles()

        self.spin_train.setValue(proj.split_train)
        self.spin_val.setValue(proj.split_val)
        self.spin_seed.setValue(proj.seed)
        self.chk_dedup.setChecked(proj.dedup)

        self.spin_crop_pad.setValue(proj.crop_pad_ratio)
        self.spin_crop_min_px.setValue(proj.min_crop_size_px)
        self.chk_crop_square.setChecked(proj.enforce_square)

        self._set_device_combo(proj.device or "")

        self.spin_epochs.setValue(proj.epochs)
        self.spin_batch.setValue(proj.batch)
        self.chk_auto_batch.setChecked(proj.auto_batch)
        self.spin_lr0.setValue(proj.lr0)
        self.spin_patience.setValue(proj.patience)
        self.spin_workers.setValue(proj.workers)
        self.chk_cache.setChecked(proj.cache)

        self.spin_imgsz_obb_direct.setValue(proj.imgsz_obb_direct)
        self.spin_imgsz_detect_direct.setValue(proj.imgsz_detect_direct)
        self.spin_imgsz_segment_direct.setValue(proj.imgsz_segment_direct)
        self.spin_imgsz_seq_detect.setValue(proj.imgsz_seq_detect)
        self.spin_imgsz_seq_crop_obb.setValue(proj.imgsz_seq_crop_obb)
        self.spin_imgsz_seq_crop_segment.setValue(proj.imgsz_seq_crop_segment)

        self.combo_model_obb_direct.setCurrentText(proj.model_obb_direct)
        self.combo_model_detect_direct.setCurrentText(proj.model_detect_direct)
        self.combo_model_segment_direct.setCurrentText(proj.model_segment_direct)
        self.combo_model_seq_detect.setCurrentText(proj.model_seq_detect)
        self.combo_model_seq_crop_obb.setCurrentText(proj.model_seq_crop_obb)
        self.combo_model_seq_crop_segment.setCurrentText(proj.model_seq_crop_segment)

        self.aug_group.setChecked(proj.aug_enabled)
        self.aug_fliplr.setValue(proj.aug_fliplr)
        self.aug_flipud.setValue(proj.aug_flipud)
        self.aug_degrees.setValue(proj.aug_degrees)
        self.aug_mosaic.setValue(proj.aug_mosaic)
        self.aug_mixup.setValue(proj.aug_mixup)
        self.aug_hsv_h.setValue(proj.aug_hsv_h)
        self.aug_hsv_s.setValue(proj.aug_hsv_s)
        self.aug_hsv_v.setValue(proj.aug_hsv_v)

        self.slice_group.load_from(proj.slice_settings)

        self._apply_persistent_state()
        self._refresh_role_gating()
        self._on_training_selection_changed()
        self._set_run_status("Ready to start training for the selected roles.")
        self._refresh_summary()
        self._refresh_overview_data_cards()

    def _set_device_combo(self, device: str) -> None:
        """Select *device* in combo_device, falling back to the first available option."""
        wanted = (device or "").strip()
        idx = self.combo_device.findText(wanted, Qt.MatchFlag.MatchFixedString)
        if idx < 0:
            idx = 0
        if self.combo_device.count() > 0:
            self.combo_device.setCurrentIndex(idx)

    def _write_to_project(self) -> None:
        proj = self._project

        proj.training_mode = self._selected_mode()
        proj.training_task = self._selected_task()
        proj.role_obb_direct = self.chk_role_obb_direct.isChecked()
        proj.role_detect_direct = self.chk_role_detect_direct.isChecked()
        proj.role_segment_direct = self.chk_role_segment_direct.isChecked()
        proj.role_seq_detect = self.chk_role_seq_detect.isChecked()
        proj.role_seq_crop_obb = self.chk_role_seq_crop_obb.isChecked()
        proj.role_seq_crop_segment = self.chk_role_seq_crop_segment.isChecked()

        proj.split_train = self.spin_train.value()
        proj.split_val = self.spin_val.value()
        proj.seed = self.spin_seed.value()
        proj.dedup = self.chk_dedup.isChecked()

        proj.crop_pad_ratio = self.spin_crop_pad.value()
        proj.min_crop_size_px = self.spin_crop_min_px.value()
        proj.enforce_square = self.chk_crop_square.isChecked()

        proj.device = self.combo_device.currentText().strip() or "cpu"

        proj.epochs = self.spin_epochs.value()
        proj.batch = self.spin_batch.value()
        proj.auto_batch = self.chk_auto_batch.isChecked()
        proj.lr0 = self.spin_lr0.value()
        proj.patience = self.spin_patience.value()
        proj.workers = self.spin_workers.value()
        proj.cache = self.chk_cache.isChecked()

        proj.imgsz_obb_direct = self.spin_imgsz_obb_direct.value()
        proj.imgsz_detect_direct = self.spin_imgsz_detect_direct.value()
        proj.imgsz_segment_direct = self.spin_imgsz_segment_direct.value()
        proj.imgsz_seq_detect = self.spin_imgsz_seq_detect.value()
        proj.imgsz_seq_crop_obb = self.spin_imgsz_seq_crop_obb.value()
        proj.imgsz_seq_crop_segment = self.spin_imgsz_seq_crop_segment.value()

        proj.model_obb_direct = self.combo_model_obb_direct.currentText()
        proj.model_detect_direct = self.combo_model_detect_direct.currentText()
        proj.model_segment_direct = self.combo_model_segment_direct.currentText()
        proj.model_seq_detect = self.combo_model_seq_detect.currentText()
        proj.model_seq_crop_obb = self.combo_model_seq_crop_obb.currentText()
        proj.model_seq_crop_segment = self.combo_model_seq_crop_segment.currentText()

        proj.aug_enabled = self.aug_group.isChecked()
        proj.aug_fliplr = self.aug_fliplr.value()
        proj.aug_flipud = self.aug_flipud.value()
        proj.aug_degrees = self.aug_degrees.value()
        proj.aug_mosaic = self.aug_mosaic.value()
        proj.aug_mixup = self.aug_mixup.value()
        proj.aug_hsv_h = self.aug_hsv_h.value()
        proj.aug_hsv_s = self.aug_hsv_s.value()
        proj.aug_hsv_v = self.aug_hsv_v.value()

        proj.slice_settings = self.slice_group.to_settings()

        self._save_persistent_state()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _class_names(self) -> list[str]:
        from ..models import normalize_class_names

        return normalize_class_names(self._project.class_names)

    def _selected_mode(self) -> str:
        return str(self.mode_combo.currentData() or "direct")

    @staticmethod
    def _set_combo_data(combo: QComboBox, value: str) -> None:
        index = combo.findData(str(value))
        combo.setCurrentIndex(index if index >= 0 else 0)

    def _selected_task(self) -> str:
        return str(self.task_combo.currentData() or "obb")

    def _selected_plan_key(self) -> tuple[str, str]:
        return self._selected_mode(), self._selected_task()

    def _plan_role_keys(self) -> tuple[str, ...]:
        return _SELECTION_ROLE_MAP[self._selected_plan_key()]

    def _selected_required_level(self) -> GeometryLevel:
        from hydra_suite.training.dataset_builders import role_min_level

        return max(role_min_level(role) for role in self._selected_roles())

    def _apply_selection_roles(self) -> None:
        selected = set(self._plan_role_keys())
        for role, checkbox in self._role_checkboxes().items():
            checkbox.blockSignals(True)
            checkbox.setChecked(role in selected)
            checkbox.blockSignals(False)

    def _role_checkboxes(self) -> dict[str, QCheckBox]:
        return {
            "obb_direct": self.chk_role_obb_direct,
            "detect_direct": self.chk_role_detect_direct,
            "segment_direct": self.chk_role_segment_direct,
            "seq_detect": self.chk_role_seq_detect,
            "seq_crop_obb": self.chk_role_seq_crop_obb,
            "seq_crop_segment": self.chk_role_seq_crop_segment,
            "semantic_sam3": self.chk_semantic_sam3,
        }

    def _update_advanced_role_controls(self) -> None:
        selected_roles = set(self._selected_role_keys())
        show_direct = "obb_direct" in selected_roles
        show_detect_direct = "detect_direct" in selected_roles
        show_segment_direct = "segment_direct" in selected_roles
        show_seq_detect = "seq_detect" in selected_roles
        show_seq_crop_obb = "seq_crop_obb" in selected_roles
        show_seq_crop_segment = "seq_crop_segment" in selected_roles
        show_sequence_settings = bool(
            selected_roles & {"seq_detect", "seq_crop_obb", "seq_crop_segment"}
        )

        for label, field, visible in (
            (self.label_imgsz_obb_direct, self.spin_imgsz_obb_direct, show_direct),
            (
                self.label_imgsz_detect_direct,
                self.spin_imgsz_detect_direct,
                show_detect_direct,
            ),
            (
                self.label_imgsz_segment_direct,
                self.spin_imgsz_segment_direct,
                show_segment_direct,
            ),
            (self.label_imgsz_seq_detect, self.spin_imgsz_seq_detect, show_seq_detect),
            (
                self.label_imgsz_seq_crop_obb,
                self.spin_imgsz_seq_crop_obb,
                show_seq_crop_obb,
            ),
            (
                self.label_imgsz_seq_crop_segment,
                self.spin_imgsz_seq_crop_segment,
                show_seq_crop_segment,
            ),
            (self.label_model_obb_direct, self.combo_model_obb_direct, show_direct),
            (
                self.label_model_detect_direct,
                self.combo_model_detect_direct,
                show_detect_direct,
            ),
            (
                self.label_model_segment_direct,
                self.combo_model_segment_direct,
                show_segment_direct,
            ),
            (
                self.label_model_seq_detect,
                self.combo_model_seq_detect,
                show_seq_detect,
            ),
            (
                self.label_model_seq_crop_obb,
                self.combo_model_seq_crop_obb,
                show_seq_crop_obb,
            ),
            (
                self.label_model_seq_crop_segment,
                self.combo_model_seq_crop_segment,
                show_seq_crop_segment,
            ),
        ):
            label.setVisible(visible)
            field.setVisible(visible)

        self.crop_settings_label.setVisible(show_sequence_settings)
        self.crop_settings_widget.setVisible(show_sequence_settings)

    def _refresh_overview_data_cards(self) -> None:
        self._refresh_dataset_fit()
        self._refresh_source_preview()

    def _refresh_role_gating(self) -> None:
        """Reflect which targets are supported by at least one source level."""
        sources = list(self._project.sources) if self._project else []
        if not sources:
            return
        highest = max(
            (GeometryLevel.from_str(getattr(src, "level", "obb")) for src in sources),
            default=GeometryLevel.AABB,
        )
        task_requirements = {
            "obb": GeometryLevel.OBB,
            "detect": GeometryLevel.AABB,
            "segment": GeometryLevel.POLYGON,
        }
        model = self.task_combo.model()
        for index in range(self.task_combo.count()):
            task = str(self.task_combo.itemData(index))
            model.item(index).setEnabled(highest >= task_requirements[task])

    def _on_training_selection_changed(self, *_args) -> None:
        # "semantic" mode only ever pairs with the "segment" task -- force it
        # so the plan key is always a valid _SELECTION_ROLE_MAP entry and the
        # unguarded subscript below can never KeyError.
        if self._selected_mode() == "semantic" and self._selected_task() != "segment":
            self.task_combo.blockSignals(True)
            self._set_combo_data(self.task_combo, "segment")
            self.task_combo.blockSignals(False)

        self._apply_selection_roles()
        self.selection_description.setText(
            _SELECTION_DESCRIPTIONS[self._selected_plan_key()]
        )
        self._update_advanced_role_controls()
        self._sync_slice_model_input_size()
        if hasattr(self, "sam3_panel"):
            self.training_tabs.setTabVisible(
                self._sam3_tab_index, self._selected_mode() == "semantic"
            )
        self._refresh_summary()
        self._mark_dataset_fit_dirty()

    def _sync_slice_model_input_size(self, *_args) -> None:
        """Show SAHI's relative object scales against the active direct model."""
        if not hasattr(self, "slice_group"):
            return
        direct_roles = {
            "obb_direct": self.spin_imgsz_obb_direct,
            "detect_direct": self.spin_imgsz_detect_direct,
            "segment_direct": self.spin_imgsz_segment_direct,
        }
        for role in self._selected_role_keys():
            if role in direct_roles:
                self.slice_group.set_model_input_size(direct_roles[role].value())
                return
        self.slice_group.set_model_input_size(640)

    def _source_fit_summary(self) -> str:
        sources = list(self._project.sources)
        required = self._selected_required_level()
        usable = sum(
            GeometryLevel.from_str(getattr(src, "level", "obb")) >= required
            for src in sources
        )
        return f"{len(sources)} configured; {usable} usable for {required.label}"

    def _dataset_fit_key(self) -> tuple:
        source_paths = tuple(
            (str(src.path).strip(), str(getattr(src, "level", "obb")))
            for src in self._project.sources
            if str(src.path).strip()
        )
        return (
            source_paths,
            round(float(self.spin_crop_pad.value()), 4),
            int(self.spin_crop_min_px.value()),
            bool(self.chk_crop_square.isChecked()),
            int(self.spin_imgsz_obb_direct.value()),
            int(self.spin_imgsz_detect_direct.value()),
            int(self.spin_imgsz_segment_direct.value()),
            int(self.spin_imgsz_seq_crop_obb.value()),
            int(self.spin_imgsz_seq_crop_segment.value()),
            tuple(self._selected_role_keys()),
        )

    def _mark_dataset_fit_dirty(self, *_args) -> None:
        self._dataset_fit_dirty = True
        if hasattr(self, "dataset_fit_status"):
            self.dataset_fit_status.setText(
                "Dataset fit summary needs refresh after the latest settings change."
            )

    def _refresh_dataset_fit(self) -> None:
        if not hasattr(self, "dataset_fit_view"):
            return

        cache_key = self._dataset_fit_key()
        if not self._dataset_fit_dirty and cache_key == self._dataset_fit_cache_key:
            self.dataset_fit_view.setPlainText(self._dataset_fit_cache_text)
            return

        try:
            from hydra_suite.training.contracts import SourceDataset
            from hydra_suite.training.dataset_builders import (
                resolve_source_path_for_target,
            )
            from hydra_suite.training.dataset_inspector import (
                DatasetInspection,
                analyze_obb_sizes,
                format_size_analysis,
                inspect_obb_or_detect_dataset,
            )
        except ImportError:
            self.dataset_fit_status.setText(
                "Dataset analysis is unavailable because the training inspector could not be imported."
            )
            self.dataset_fit_view.setPlainText("")
            return

        required_level = self._selected_required_level()
        source_paths = [
            str(
                resolve_source_path_for_target(
                    SourceDataset(
                        path=str(src.path).strip(),
                        name=src.name,
                        level=getattr(src, "level", "obb"),
                    ),
                    required_level,
                )
            )
            for src in self._project.sources
            if str(src.path).strip()
            and GeometryLevel.from_str(getattr(src, "level", "obb")) >= required_level
        ]
        if not source_paths:
            self.dataset_fit_status.setText("No source datasets configured yet.")
            self.dataset_fit_view.setPlainText(
                f"Add one or more {required_level.label}-compatible sources to see guidance for this plan."
            )
            self._dataset_fit_cache_key = cache_key
            self._dataset_fit_cache_text = self.dataset_fit_view.toPlainText()
            self._dataset_fit_dirty = False
            return

        merged = DatasetInspection(root_dir="overview")
        valid_items = 0
        for source_path in source_paths:
            if not Path(source_path).exists():
                continue
            try:
                inspection = inspect_obb_or_detect_dataset(source_path)
            except Exception as exc:
                logger.warning(
                    "Failed to inspect DetectKit source %s: %s", source_path, exc
                )
                continue
            for split_name, items in inspection.splits.items():
                merged.splits.setdefault(split_name, []).extend(items)
                valid_items += len(items)
            merged.class_names.update(inspection.class_names)

        if valid_items <= 0:
            self.dataset_fit_status.setText(
                "No valid dataset items were found in the configured sources."
            )
            self.dataset_fit_view.setPlainText(
                "DetectKit could not discover any image and label pairs in the current sources."
            )
            self._dataset_fit_cache_key = cache_key
            self._dataset_fit_cache_text = self.dataset_fit_view.toPlainText()
            self._dataset_fit_dirty = False
            return

        try:
            stats = analyze_obb_sizes(
                merged,
                pad_ratio=self.spin_crop_pad.value(),
                min_crop_size_px=self.spin_crop_min_px.value(),
                enforce_square=self.chk_crop_square.isChecked(),
            )
        except Exception as exc:
            self.dataset_fit_status.setText(f"Dataset analysis failed: {exc}")
            self.dataset_fit_view.setPlainText("")
            return

        selected_roles = set(self._selected_role_keys())
        lines: list[str] = []
        all_warnings: list[str] = []
        if selected_roles & {"seq_detect", "seq_crop_obb"}:
            report_seq, warnings_seq = format_size_analysis(
                stats,
                training_imgsz=self.spin_imgsz_seq_crop_obb.value(),
                pipeline_mode="crop",
            )
            lines += [
                "=== Sequential Pipeline ===",
                f"(stage-2 imgsz = {self.spin_imgsz_seq_crop_obb.value()})",
                "",
                report_seq,
            ]
            if warnings_seq:
                lines += ["", "Warnings:"] + [
                    f"- {warning}" for warning in warnings_seq
                ]
                all_warnings.extend(warnings_seq)

        if "obb_direct" in selected_roles:
            report_direct, warnings_direct = format_size_analysis(
                stats,
                training_imgsz=self.spin_imgsz_obb_direct.value(),
                pipeline_mode="full_image",
            )
            if lines:
                lines += [""]
            lines += [
                "=== Direct OBB ===",
                f"(imgsz = {self.spin_imgsz_obb_direct.value()})",
                "",
                report_direct,
            ]
            if warnings_direct:
                lines += ["", "Warnings:"] + [
                    f"- {warning}" for warning in warnings_direct
                ]
                all_warnings.extend(warnings_direct)

        if "detect_direct" in selected_roles:
            report_detect_direct, warnings_detect_direct = format_size_analysis(
                stats,
                training_imgsz=self.spin_imgsz_detect_direct.value(),
                pipeline_mode="full_image",
            )
            if lines:
                lines += [""]
            lines += [
                "=== Direct Detect ===",
                f"(imgsz = {self.spin_imgsz_detect_direct.value()})",
                "",
                report_detect_direct,
            ]
            if warnings_detect_direct:
                lines += ["", "Warnings:"] + [
                    f"- {warning}" for warning in warnings_detect_direct
                ]
                all_warnings.extend(warnings_detect_direct)

        if "segment_direct" in selected_roles:
            report_segment_direct, warnings_segment_direct = format_size_analysis(
                stats,
                training_imgsz=self.spin_imgsz_segment_direct.value(),
                pipeline_mode="full_image",
            )
            if lines:
                lines += [""]
            lines += [
                "=== Direct Segment ===",
                f"(imgsz = {self.spin_imgsz_segment_direct.value()})",
                "",
                report_segment_direct,
            ]
            if warnings_segment_direct:
                lines += ["", "Warnings:"] + [
                    f"- {warning}" for warning in warnings_segment_direct
                ]
                all_warnings.extend(warnings_segment_direct)

        if "seq_crop_segment" in selected_roles:
            report_seq_crop_segment, warnings_seq_crop_segment = format_size_analysis(
                stats,
                training_imgsz=self.spin_imgsz_seq_crop_segment.value(),
                pipeline_mode="crop",
            )
            if lines:
                lines += [""]
            lines += [
                "=== Sequential Crop-Segment ===",
                f"(stage-2 imgsz = {self.spin_imgsz_seq_crop_segment.value()})",
                "",
                report_seq_crop_segment,
            ]
            if warnings_seq_crop_segment:
                lines += ["", "Warnings:"] + [
                    f"- {warning}" for warning in warnings_seq_crop_segment
                ]
                all_warnings.extend(warnings_seq_crop_segment)

        if not lines:
            lines.append("No stages are selected for analysis.")

        text = "\n".join(lines)
        self.dataset_fit_view.setPlainText(text)
        if all_warnings:
            self.dataset_fit_status.setText(
                f"Analysis ready. {len(all_warnings)} warning(s) need attention for the current recipe."
            )
        else:
            self.dataset_fit_status.setText(
                f"Analysis ready for {valid_items} discovered labeled items across {len(source_paths)} source(s)."
            )

        self._dataset_fit_cache_key = cache_key
        self._dataset_fit_cache_text = text
        self._dataset_fit_dirty = False

    def _source_preview_records(
        self, max_items: int = 6
    ) -> list[dict[str, str | Path]]:
        try:
            from hydra_suite.training.contracts import SourceDataset
            from hydra_suite.training.dataset_builders import (
                resolve_source_path_for_target,
            )
            from hydra_suite.training.dataset_inspector import (
                inspect_obb_or_detect_dataset,
            )
        except ImportError:
            return []

        buckets: dict[str, list[dict[str, str | Path]]] = {}
        frame_size_counts: dict[tuple[int, int], int] = {}
        seen_frame_paths: set[Path] = set()
        required_level = self._selected_required_level()
        for src in self._project.sources:
            if GeometryLevel.from_str(getattr(src, "level", "obb")) < required_level:
                continue
            source_path = str(
                resolve_source_path_for_target(
                    SourceDataset(
                        path=str(src.path).strip(),
                        name=src.name,
                        level=getattr(src, "level", "obb"),
                    ),
                    required_level,
                )
            )
            if not source_path or not Path(source_path).exists():
                continue
            try:
                inspection = inspect_obb_or_detect_dataset(source_path)
            except Exception as exc:
                logger.warning(
                    "Failed to inspect DetectKit source samples %s: %s",
                    source_path,
                    exc,
                )
                continue

            source_name = src.name.strip() or Path(source_path).name
            records: list[dict[str, str | Path]] = []
            for split_name in ("train", "val", "test", "all"):
                for item in inspection.splits.get(split_name, []):
                    image_path = Path(item.image_path)
                    if not image_path.exists():
                        continue
                    resolved_path = image_path.resolve()
                    if resolved_path not in seen_frame_paths:
                        seen_frame_paths.add(resolved_path)
                        image_size = QImageReader(str(image_path)).size()
                        if image_size.isValid():
                            frame_wh = (image_size.width(), image_size.height())
                            frame_size_counts[frame_wh] = (
                                frame_size_counts.get(frame_wh, 0) + 1
                            )
                    records.append(
                        {
                            "path": image_path,
                            "source": source_name,
                            "split": split_name if split_name != "all" else "dataset",
                        }
                    )
            if records:
                buckets[source_name] = records

        self._source_preview_frame_options = [
            (width, height, count)
            for (width, height), count in frame_size_counts.items()
        ]

        selected: list[dict[str, str | Path]] = []
        while len(selected) < max_items:
            added = False
            for source_name in sorted(buckets):
                bucket = buckets[source_name]
                if not bucket:
                    continue
                selected.append(bucket.pop(0))
                added = True
                if len(selected) >= max_items:
                    break
            if not added:
                break
        return selected[:max_items]

    @staticmethod
    def _source_preview_pixmap(path: Path) -> QPixmap:
        return QPixmap(str(path))

    def _build_source_preview_card(self, record: dict[str, str | Path]) -> QWidget:
        card = QFrame()
        card.setFrameShape(QFrame.Shape.StyledPanel)
        card.setFixedWidth(168)
        card.setStyleSheet(
            "QFrame { background:#1e1e1e; border:1px solid #3e3e42; border-radius:6px; }"
        )

        layout = QVBoxLayout(card)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        image_path = Path(str(record["path"]))
        image_label = QLabel()
        image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        image_label.setFixedSize(150, 112)
        image_label.setStyleSheet(
            "background:#111111; border:1px solid #3e3e42; border-radius:4px; color:#cfcfcf;"
        )
        pixmap = self._source_preview_pixmap(image_path)
        if pixmap.isNull():
            image_label.setText("Preview\nunavailable")
        else:
            image_label.setPixmap(
                pixmap.scaled(
                    146,
                    108,
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation,
                )
            )
        layout.addWidget(image_label)

        caption = QLabel(
            f"{record['source']}\n{str(record['split']).title()} split\n{image_path.name}"
        )
        caption.setTextFormat(Qt.TextFormat.PlainText)
        caption.setWordWrap(True)
        caption.setStyleSheet("color:#ffffff; font-size:11px;")
        caption.setToolTip(f"Source: {record['source']}\nPath: {image_path}")
        layout.addWidget(caption)
        return card

    def _refresh_source_preview(self) -> None:
        if not hasattr(self, "source_preview_cards_layout"):
            return

        while self.source_preview_cards_layout.count():
            item = self.source_preview_cards_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()

        records = self._source_preview_records()
        if not self._project.sources:
            self.slice_group.set_preview_frame_options([])
            self.source_preview_status.setText(
                "No sources configured yet. Add one or more DetectKit datasets to preview sample frames here."
            )
            empty_label = QLabel(
                "No source datasets are connected to this project yet."
            )
            empty_label.setWordWrap(True)
            empty_label.setStyleSheet("color:#cfcfcf; font-size:11px;")
            self.source_preview_cards_layout.addWidget(empty_label)
            return

        if not records:
            self.slice_group.set_preview_frame_options([])
            self.source_preview_status.setText(
                "Source datasets are configured, but DetectKit could not discover previewable image-label pairs yet."
            )
            empty_label = QLabel(
                "Preview unavailable. Check that each source has a valid images/labels layout or dataset.yaml."
            )
            empty_label.setWordWrap(True)
            empty_label.setStyleSheet("color:#cfcfcf; font-size:11px;")
            self.source_preview_cards_layout.addWidget(empty_label)
            return

        self.slice_group.set_preview_frame_options(
            getattr(self, "_source_preview_frame_options", [])
        )

        self.source_preview_status.setText(
            f"Showing {len(records)} representative labeled sample(s) from {len(self._project.sources)} source dataset(s)."
        )
        for record in records:
            self.source_preview_cards_layout.addWidget(
                self._build_source_preview_card(record)
            )
        self.source_preview_cards_layout.addStretch()

    def _build_device_options(self) -> list[str]:
        """Return PyTorch-only training device choices (no ONNX/TRT)."""
        try:
            from hydra_suite.utils.gpu_utils import get_device_info

            info = get_device_info()
        except ImportError:
            info = {}
        options: list[str] = []
        if info.get("torch_cuda_available"):
            count = int(info.get("torch_cuda_device_count", 0) or 0)
            if count > 1:
                options.append("cuda")
            for i in range(count):
                options.append(f"cuda:{i}")
        if info.get("mps_available"):
            options.append("mps")
        options.append("cpu")
        return options

    def _imgsz_for_role(self, role) -> int:
        try:
            from hydra_suite.training import TrainingRole

            if role == TrainingRole.OBB_DIRECT:
                return self.spin_imgsz_obb_direct.value()
            if role == TrainingRole.DETECT_DIRECT:
                return self.spin_imgsz_detect_direct.value()
            if role == TrainingRole.SEGMENT_DIRECT:
                return self.spin_imgsz_segment_direct.value()
            if role == TrainingRole.SEQ_DETECT:
                return self.spin_imgsz_seq_detect.value()
            if role == TrainingRole.SEQ_CROP_OBB:
                return self.spin_imgsz_seq_crop_obb.value()
            if role == TrainingRole.SEQ_CROP_SEGMENT:
                return self.spin_imgsz_seq_crop_segment.value()
        except ImportError:
            pass
        return 640

    def _base_model_for_role(self, role) -> str:
        try:
            from hydra_suite.training import TrainingRole

            if role == TrainingRole.OBB_DIRECT:
                return self.combo_model_obb_direct.currentText().strip()
            if role == TrainingRole.DETECT_DIRECT:
                return self.combo_model_detect_direct.currentText().strip()
            if role == TrainingRole.SEGMENT_DIRECT:
                return self.combo_model_segment_direct.currentText().strip()
            if role == TrainingRole.SEQ_DETECT:
                return self.combo_model_seq_detect.currentText().strip()
            if role == TrainingRole.SEQ_CROP_OBB:
                return self.combo_model_seq_crop_obb.currentText().strip()
            if role == TrainingRole.SEQ_CROP_SEGMENT:
                return self.combo_model_seq_crop_segment.currentText().strip()
        except ImportError:
            pass
        return ""

    def _selected_roles(self) -> list:
        try:
            from hydra_suite.training import TrainingRole
        except ImportError:
            return []
        roles = []
        if self.chk_role_obb_direct.isChecked():
            roles.append(TrainingRole.OBB_DIRECT)
        if self.chk_role_detect_direct.isChecked():
            roles.append(TrainingRole.DETECT_DIRECT)
        if self.chk_role_segment_direct.isChecked():
            roles.append(TrainingRole.SEGMENT_DIRECT)
        if self.chk_role_seq_detect.isChecked():
            roles.append(TrainingRole.SEQ_DETECT)
        if self.chk_role_seq_crop_obb.isChecked():
            roles.append(TrainingRole.SEQ_CROP_OBB)
        if self.chk_role_seq_crop_segment.isChecked():
            roles.append(TrainingRole.SEQ_CROP_SEGMENT)
        if self.chk_semantic_sam3.isChecked():
            roles.append(TrainingRole.SEMANTIC_SAM3)
        return roles

    @staticmethod
    def _sam3_spec_for(source_path, params, derived_dir, seed):
        """One raw source, no merge. Concept training is per-source."""
        from hydra_suite.training import (
            SourceDataset,
            TrainingHyperParams,
            TrainingRole,
            TrainingRunSpec,
        )

        if not params.label_quality_acknowledged:
            raise ValueError(
                "You must acknowledge the label-quality warning before "
                "training: SAM3 learns any systematic error in these labels."
            )
        return TrainingRunSpec(
            role=TrainingRole.SEMANTIC_SAM3,
            source_datasets=[SourceDataset(path=str(source_path), level="polygon")],
            derived_dataset_dir=str(derived_dir),
            base_model="sam3",
            hyperparams=TrainingHyperParams(epochs=params.epochs),
            seed=seed,
            sam3_params=params,
        )

    def _collect_sources(self) -> list:
        try:
            from hydra_suite.training import SourceDataset
        except ImportError:
            return []
        sources = []
        for src in self._project.sources:
            p = src.path.strip()
            if p:
                sources.append(
                    SourceDataset(
                        path=p,
                        source_type="yolo_obb",
                        name=src.name or Path(p).name,
                        level=getattr(src, "level", "obb"),
                    )
                )
        return sources

    @staticmethod
    def _infer_size_token(model_path: str) -> str:
        name = Path(str(model_path or "")).name.lower()
        for token in (
            "26n",
            "26s",
            "26m",
            "26l",
            "26x",
            "11n",
            "11s",
            "11m",
            "11l",
            "11x",
        ):
            if token in name:
                return token
        return "unknown"

    def _publish_meta_for_role(self, role, base_model: str) -> dict:
        training_params: dict = {"imgsz": self._imgsz_for_role(role)}
        try:
            from hydra_suite.training import TrainingRole

            if role in (TrainingRole.SEQ_CROP_OBB, TrainingRole.SEQ_CROP_SEGMENT):
                training_params["crop_pad_ratio"] = self.spin_crop_pad.value()
                training_params["min_crop_size_px"] = self.spin_crop_min_px.value()
                training_params["enforce_square"] = self.chk_crop_square.isChecked()
        except ImportError:
            pass
        return {
            "size": self._infer_size_token(base_model),
            "species": (self._project.species or "").strip() or "species",
            "model_info": f"train_{role.value}",
            "training_params": training_params,
        }

    def _append_log(self, text: str) -> None:
        log_text = str(text)
        self.log_view.append(log_text)
        if self._current_role:
            self._role_logs.setdefault(self._current_role, []).append(log_text)
        if self.loss_plot is not None:
            self.loss_plot.ingest_log_line(log_text)

    def _get_orchestrator(self):
        if self._orchestrator is None:
            try:
                from hydra_suite.training import TrainingOrchestrator

                self._orchestrator = TrainingOrchestrator(self._workspace_default)
            except ImportError:
                return None
        return self._orchestrator

    # ------------------------------------------------------------------
    # Dataset building
    # ------------------------------------------------------------------

    def _dataset_preparation_request(
        self,
        sources: list[SourceDataset],
        roles: list[TrainingRole],
    ) -> _DatasetPreparationRequest:
        slice_settings = SliceTrainingSettings.from_dict(
            self.slice_group.to_settings().to_dict()
        )
        sam3_params = (
            self.sam3_panel.params() if TrainingRole.SEMANTIC_SAM3 in roles else None
        )
        return _DatasetPreparationRequest(
            sources=tuple(sources),
            roles=tuple(roles),
            class_names=tuple(self._class_names()),
            split=SplitConfig(
                train=self.spin_train.value(),
                val=self.spin_val.value(),
                test=0.0,
            ),
            seed=self.spin_seed.value(),
            dedup=self.chk_dedup.isChecked(),
            crop_pad_ratio=self.spin_crop_pad.value(),
            min_crop_size_px=self.spin_crop_min_px.value(),
            enforce_square=self.chk_crop_square.isChecked(),
            imgsz_by_role=tuple(
                (role.value, self._imgsz_for_role(role)) for role in roles
            ),
            slice_settings=slice_settings,
            sam3_params=sam3_params,
        )

    def _launch_dataset_preparation(
        self,
        orchestrator,
        request: _DatasetPreparationRequest,
    ) -> None:
        self.role_dataset_dirs = {}
        self._pending_dataset_result = None
        self._dataset_preparation_error = ""
        worker = _DatasetPreparationWorker(orchestrator, request)
        worker.log_signal.connect(self._append_log)
        worker.status.connect(self._set_run_status)
        worker.result_ready.connect(self._on_dataset_prepared)
        worker.error.connect(self._on_dataset_preparation_error)
        worker.finished.connect(self._on_dataset_worker_finished)
        self._dataset_worker = worker

        self._set_training_running(True)
        self.progress.setRange(0, 0)
        self.progress.setFormat("Preparing datasets…")
        self._set_run_status(
            "Preparing role datasets in the background. You can stop at the next "
            "safe dataset boundary."
        )
        worker.start()

    def _on_dataset_prepared(self, result: _DatasetPreparationResult) -> None:
        self._pending_dataset_result = result

    def _on_dataset_preparation_error(self, message: str) -> None:
        self._dataset_preparation_error = str(message)
        self._append_log(f"Dataset preparation failed:\n{message}")
        self._set_run_status(
            "Dataset preparation failed. See the session log for details."
        )

    def _on_dataset_worker_finished(self) -> None:
        worker = self._dataset_worker
        result = self._pending_dataset_result
        error = self._dataset_preparation_error
        cancelled = bool(worker is not None and worker.is_cancelled())
        self._dataset_worker = None
        self._pending_dataset_result = None
        self.progress.setRange(0, 100)

        if cancelled:
            result = None

        if result is None:
            self._set_training_running(False)
            self.progress.setValue(0)
            self.progress.setFormat("Cancelled" if cancelled else "Failed")
            if cancelled:
                self._append_log("Dataset preparation cancelled.")
                self._set_run_status("Dataset preparation cancelled.")
            elif error:
                QMessageBox.critical(
                    self,
                    "Dataset Preparation Failed",
                    "Dataset preparation failed. See the session log for details.",
                )
            return

        self.role_dataset_dirs = dict(result.role_dataset_dirs)
        measured_ref = float(result.measured_reference_body_px)
        if measured_ref > 0.0:
            from hydra_suite.detectkit.gui.models import populate_measured_reference

            if populate_measured_reference(self._project.slice_settings, measured_ref):
                try:
                    from hydra_suite.detectkit.gui.project import save_project

                    save_project(self._project)
                except Exception as exc:
                    self._append_log(
                        "WARNING: Could not persist the measured reference body "
                        f"size: {exc}"
                    )
                else:
                    self._append_log(
                        f"Updated automatic reference body size: {measured_ref:.1f}px "
                        "(measured from labels)"
                    )
                # Refresh the informational label and live tile schematic in
                # this still-open dialog with the fresh label measurement.
                self.slice_group.load_from(self._project.slice_settings)
                self._sync_slice_model_input_size()

        self._set_run_status(
            f"Prepared datasets for {len(self.role_dataset_dirs)} selected role(s)."
        )
        self._refresh_summary()
        self._start_training_worker(list(result.roles))

    # ------------------------------------------------------------------
    # Training execution
    # ------------------------------------------------------------------

    def _start_training(self) -> None:
        dataset_busy = self._dataset_worker is not None and (
            self._dataset_worker.isRunning()
        )
        training_busy = self._worker is not None and self._worker.isRunning()
        if dataset_busy or training_busy:
            QMessageBox.warning(
                self,
                "Busy",
                "Dataset preparation or training is already running.",
            )
            return

        roles = self._selected_roles()
        if not roles:
            QMessageBox.warning(self, "No Roles", "Select at least one training role.")
            return

        sources = self._collect_sources()
        if not sources:
            QMessageBox.warning(
                self,
                "No Sources",
                "Add at least one labeled DetectKit source dataset.",
            )
            return

        orchestrator = self._get_orchestrator()
        if orchestrator is None:
            self._append_log("Training dependencies not available.")
            return

        if TrainingRole.SEMANTIC_SAM3 in roles and len(sources) != 1:
            QMessageBox.warning(
                self,
                "Multiple Sources Not Supported",
                "SAM3 concept training supports exactly one labeled source "
                "dataset at a time. Remove the extra sources (or run this role "
                "separately per source) before building/training this role.",
            )
            return

        self._write_to_project()
        self._last_training_results = []
        request = self._dataset_preparation_request(sources, roles)
        self._launch_dataset_preparation(orchestrator, request)

    def _start_training_worker(self, roles: list[TrainingRole]) -> None:
        """Build run specs on the GUI thread, then launch the training worker."""

        def abort_start() -> None:
            self._set_training_running(False)
            self.progress.setRange(0, 100)
            self.progress.setValue(0)
            self.progress.setFormat("Not started")

        orchestrator = self._get_orchestrator()
        if orchestrator is None:
            self._append_log("Training dependencies not available.")
            abort_start()
            return

        source_obb = self._collect_sources()
        role_configs = []
        for role in roles:
            ds = self.role_dataset_dirs.get(role.value, "")
            if not ds:
                QMessageBox.warning(
                    self,
                    "Missing Dataset",
                    f"No dataset prepared for role: {role.value}",
                )
                abort_start()
                return
            base_model = (
                "sam3"
                if role is TrainingRole.SEMANTIC_SAM3
                else self._base_model_for_role(role)
            )
            if not base_model:
                QMessageBox.warning(
                    self,
                    "Base Model",
                    f"Set base model for role: {role.value}",
                )
                abort_start()
                return
            from hydra_suite.detectkit.config.training import RoleTrainingConfig

            role_configs.append(
                RoleTrainingConfig(
                    role=role,
                    base_model=base_model,
                    imgsz=self._imgsz_for_role(role),
                )
            )

        try:
            from hydra_suite.detectkit.config.training import (
                DetectTrainingPlan,
                SliceTrainingConfig,
            )
            from hydra_suite.training import (
                AugmentationProfile,
                PublishPolicy,
                TrainingHyperParams,
            )

            aug_args: dict[str, float] = {}
            if self.aug_group.isChecked():
                aug_args = {
                    "fliplr": self.aug_fliplr.value(),
                    "flipud": self.aug_flipud.value(),
                    "degrees": self.aug_degrees.value(),
                    "mosaic": self.aug_mosaic.value(),
                    "mixup": self.aug_mixup.value(),
                    "hsv_h": self.aug_hsv_h.value(),
                    "hsv_s": self.aug_hsv_s.value(),
                    "hsv_v": self.aug_hsv_v.value(),
                }
            batch_val = (
                -1 if self.chk_auto_batch.isChecked() else self.spin_batch.value()
            )
            sam3_params = (
                self.sam3_panel.params()
                if TrainingRole.SEMANTIC_SAM3 in roles
                else None
            )
            plan = DetectTrainingPlan(
                workspace_root=Path(self._workspace_default),
                sources=tuple(source_obb),
                class_names=tuple(self._class_names()),
                roles=tuple(role_configs),
                split=SplitConfig(
                    train=self.spin_train.value(),
                    val=self.spin_val.value(),
                    test=0.0,
                ),
                seed=self.spin_seed.value(),
                dedup=self.chk_dedup.isChecked(),
                crop_pad_ratio=self.spin_crop_pad.value(),
                min_crop_size_px=self.spin_crop_min_px.value(),
                enforce_square=self.chk_crop_square.isChecked(),
                slice_settings=SliceTrainingConfig.from_dict(
                    self.slice_group.to_settings().to_dict()
                ),
                hyperparams=TrainingHyperParams(
                    epochs=self.spin_epochs.value(),
                    batch=batch_val,
                    lr0=self.spin_lr0.value(),
                    patience=self.spin_patience.value(),
                    workers=self.spin_workers.value(),
                    cache=self.chk_cache.isChecked(),
                ),
                device=self.combo_device.currentText().strip() or "cpu",
                augmentation_profile=AugmentationProfile(
                    enabled=self.aug_group.isChecked(),
                    args=aug_args,
                ),
                # Preserve the GUI's existing behavior: successful artifacts
                # are imported so History can export or select them later.
                publish_policy=PublishPolicy(auto_import=True, auto_select=False),
                species=(self._project.species or "").strip() or "species",
                model_tag=(self._project.model_tag or "").strip() or "train",
                sam3_params=sam3_params,
            )
            plan.validate()
            role_entries = plan.role_entries(self.role_dataset_dirs)
        except (ImportError, ValueError) as exc:
            QMessageBox.warning(self, "Training Configuration", str(exc))
            abort_start()
            return

        self._worker = _TrainingWorker(orchestrator, role_entries)
        self._worker.log_signal.connect(self._append_log)
        self._worker.role_started.connect(self._on_role_started)
        self._worker.role_finished.connect(self._on_role_finished)
        self._worker.progress_signal.connect(self._on_role_progress)
        self._worker.done_signal.connect(self._on_done)
        self._worker.finished.connect(self._on_worker_finished)

        self._set_training_running(True)
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        self.progress.setFormat("Starting…")
        self._role_logs = {}
        self._current_role = ""
        if self.loss_plot is not None:
            self.loss_plot.clear()
        self._set_run_status(
            f"Training started for {len(role_entries)} role(s). Watch the loss curve and log below for live output."
        )
        self._refresh_summary()
        self._worker.start()

    def _cancel_training(self) -> None:
        if self._dataset_worker is not None:
            self._dataset_worker.cancel()
            self._append_log(
                "Dataset preparation cancellation requested; waiting for the "
                "current file operation to finish…"
            )
            self._set_run_status(
                "Dataset preparation cancellation requested. It will stop at "
                "the next safe dataset boundary."
            )
            return
        if self._worker:
            self._worker.cancel()
        self._append_log("Cancellation requested…")
        self._set_run_status(
            "Cancellation requested. The current role will stop when it reaches a safe checkpoint."
        )

    def _on_role_started(self, role: str) -> None:
        self._current_role = role
        self._role_logs.setdefault(role, [])
        self._append_log(f"=== START {role} ===")
        self._set_run_status(f"Running {self._role_display_name(role)}.")

    def _on_role_finished(self, role: str, ok: bool, message: str) -> None:
        self._append_log(f"=== {'OK' if ok else 'FAIL'} {role}: {message} ===")

    def _on_role_progress(self, role: str, cur: int, total: int) -> None:
        total = max(1, int(total))
        cur = max(0, min(total, int(cur)))
        pct = int((cur / total) * 100.0)
        self.progress.setValue(pct)
        self.progress.setFormat(f"{role}: {cur}/{total} ({pct}%)")
        self._set_run_status(
            f"{self._role_display_name(role)} in progress: {cur}/{total} steps complete."
        )

    def _on_done(self, results: list) -> None:
        for result in results:
            role = str(result.get("role", "")).strip()
            result["training_log"] = "\n".join(self._role_logs.get(role, []))

        try:
            from ..project import record_training_results

            results = record_training_results(self._project, results)
        except Exception as exc:
            logger.warning(
                "Could not persist DetectKit training history", exc_info=True
            )
            self._append_log(
                f"WARNING: Could not persist project training history: {exc}"
            )

        self._last_training_results = results
        for r in results:
            artifact = r.get("artifact_path", "")
            if artifact:
                wdir = Path(artifact).parent
                r["_run_dir"] = str(wdir.parent) if wdir.name == "weights" else ""
            else:
                r["_run_dir"] = ""

        self._update_resume_enabled()

        succeeded = [r for r in results if r.get("success")]
        failed = [r for r in results if not r.get("success")]

        self._append_log(
            f"Session complete: {len(succeeded)} success, {len(failed)} failed"
        )
        self.training_completed.emit(results)
        self._set_run_status(
            f"Training session finished with {len(succeeded)} success and {len(failed)} failure(s)."
        )
        self._refresh_summary()

        if failed:
            QMessageBox.warning(
                self,
                "Training Completed with Failures",
                f"Succeeded: {len(succeeded)}\nFailed: {len(failed)}\nSee logs for details.",
            )
        else:
            QMessageBox.information(
                self,
                "Training Completed",
                f"All {len(succeeded)} selected roles completed successfully.",
            )

    def _on_worker_finished(self) -> None:
        self._current_role = ""
        self._set_training_running(False)
        self.progress.setFormat("Done")
        self.progress.setValue(100)
        if not self._last_training_results:
            self._set_run_status("Training session finished.")

    # ------------------------------------------------------------------
    # Resume
    # ------------------------------------------------------------------

    def _resume_training(self) -> None:
        last_pt = None
        resume_result = None
        for r in reversed(self._last_training_results):
            run_dir = r.get("_run_dir", "")
            if run_dir:
                candidate = Path(run_dir) / "weights" / "last.pt"
                if candidate.exists():
                    last_pt = candidate
                    resume_result = r
                    break

        if last_pt is None:
            QMessageBox.warning(
                self,
                "No Checkpoint Found",
                "Could not find a last.pt checkpoint from the previous run.",
            )
            return

        role_str = str(resume_result.get("role", ""))
        try:
            from hydra_suite.training import (
                TrainingHyperParams,
                TrainingRole,
                TrainingRunSpec,
            )

            role = TrainingRole(role_str)
        except (ImportError, ValueError) as exc:
            QMessageBox.warning(self, "Resume Failed", f"Cannot resume: {exc}")
            return

        batch_val = (
            -1 if self.chk_auto_batch.isChecked() else int(self.spin_batch.value())
        )
        spec = TrainingRunSpec(
            role=role,
            source_datasets=[],
            derived_dataset_dir=resume_result.get("derived_dataset_dir")
            or resume_result.get("_run_dir", ""),
            base_model=str(last_pt),
            hyperparams=TrainingHyperParams(
                epochs=int(self.spin_epochs.value()),
                imgsz=self._imgsz_for_role(role),
                batch=batch_val,
                lr0=float(self.spin_lr0.value()),
                patience=int(self.spin_patience.value()),
                workers=int(self.spin_workers.value()),
            ),
            resume_from=str(last_pt),
        )

        class_names = self._class_names()
        entry = RoleTrainingEntry(
            role=role,
            spec=spec,
            publish_metadata={
                "class_names": class_names,
                "resumed_from": str(last_pt),
            },
        )

        orchestrator = self._get_orchestrator()
        if orchestrator is None:
            self._append_log("Training dependencies not available.")
            return

        self._append_log(f"Resuming training from {last_pt}")
        self._set_training_running(True)
        self.progress.setValue(0)
        self.progress.setFormat("Resuming…")
        self._role_logs = {}
        self._current_role = ""
        self._set_run_status(
            f"Resuming {self._role_display_name(role_str)} from the latest checkpoint."
        )

        self._worker = _TrainingWorker(orchestrator, [entry])
        self._worker.log_signal.connect(self._append_log)
        self._worker.role_started.connect(self._on_role_started)
        self._worker.role_finished.connect(self._on_role_finished)
        self._worker.progress_signal.connect(self._on_role_progress)
        self._worker.done_signal.connect(self._on_done)
        self._worker.finished.connect(self._on_worker_finished)
        self._worker.start()

    # ------------------------------------------------------------------
    # Save / Load Config
    # ------------------------------------------------------------------

    def _collect_training_state(self) -> dict:
        """Collect dialog widget values into a serialisable dict."""
        return {
            "roles": {
                "obb_direct": self.chk_role_obb_direct.isChecked(),
                "detect_direct": self.chk_role_detect_direct.isChecked(),
                "segment_direct": self.chk_role_segment_direct.isChecked(),
                "seq_detect": self.chk_role_seq_detect.isChecked(),
                "seq_crop_obb": self.chk_role_seq_crop_obb.isChecked(),
                "seq_crop_segment": self.chk_role_seq_crop_segment.isChecked(),
            },
            "training_mode": self._selected_mode(),
            "training_task": self._selected_task(),
            "split_train": self.spin_train.value(),
            "split_val": self.spin_val.value(),
            "seed": self.spin_seed.value(),
            "dedup": self.chk_dedup.isChecked(),
            "crop_pad_ratio": self.spin_crop_pad.value(),
            "min_crop_size_px": self.spin_crop_min_px.value(),
            "enforce_square": self.chk_crop_square.isChecked(),
            "device": self.combo_device.currentText().strip(),
            "epochs": self.spin_epochs.value(),
            "batch": self.spin_batch.value(),
            "auto_batch": self.chk_auto_batch.isChecked(),
            "lr0": self.spin_lr0.value(),
            "patience": self.spin_patience.value(),
            "workers": self.spin_workers.value(),
            "cache": self.chk_cache.isChecked(),
            "imgsz_obb_direct": self.spin_imgsz_obb_direct.value(),
            "imgsz_detect_direct": self.spin_imgsz_detect_direct.value(),
            "imgsz_segment_direct": self.spin_imgsz_segment_direct.value(),
            "imgsz_seq_detect": self.spin_imgsz_seq_detect.value(),
            "imgsz_seq_crop_obb": self.spin_imgsz_seq_crop_obb.value(),
            "imgsz_seq_crop_segment": self.spin_imgsz_seq_crop_segment.value(),
            "model_obb_direct": self.combo_model_obb_direct.currentText(),
            "model_detect_direct": self.combo_model_detect_direct.currentText(),
            "model_segment_direct": self.combo_model_segment_direct.currentText(),
            "model_seq_detect": self.combo_model_seq_detect.currentText(),
            "model_seq_crop_obb": self.combo_model_seq_crop_obb.currentText(),
            "model_seq_crop_segment": self.combo_model_seq_crop_segment.currentText(),
            "aug_enabled": self.aug_group.isChecked(),
            "aug_fliplr": self.aug_fliplr.value(),
            "aug_flipud": self.aug_flipud.value(),
            "aug_degrees": self.aug_degrees.value(),
            "aug_mosaic": self.aug_mosaic.value(),
            "aug_mixup": self.aug_mixup.value(),
            "aug_hsv_h": self.aug_hsv_h.value(),
            "aug_hsv_s": self.aug_hsv_s.value(),
            "aug_hsv_v": self.aug_hsv_v.value(),
        }

    def _apply_training_state(self, data: dict) -> None:
        """Apply a previously saved state dict to the dialog widgets."""
        if "training_mode" in data or "training_task" in data:
            self._set_combo_data(
                self.mode_combo, str(data.get("training_mode", "direct"))
            )
            self._set_combo_data(self.task_combo, str(data.get("training_task", "obb")))

        for attr, widget in [
            ("split_train", self.spin_train),
            ("split_val", self.spin_val),
            ("seed", self.spin_seed),
            ("crop_pad_ratio", self.spin_crop_pad),
            ("min_crop_size_px", self.spin_crop_min_px),
            ("epochs", self.spin_epochs),
            ("batch", self.spin_batch),
            ("lr0", self.spin_lr0),
            ("patience", self.spin_patience),
            ("workers", self.spin_workers),
            ("imgsz_obb_direct", self.spin_imgsz_obb_direct),
            ("imgsz_detect_direct", self.spin_imgsz_detect_direct),
            ("imgsz_segment_direct", self.spin_imgsz_segment_direct),
            ("imgsz_seq_detect", self.spin_imgsz_seq_detect),
            ("imgsz_seq_crop_obb", self.spin_imgsz_seq_crop_obb),
            ("imgsz_seq_crop_segment", self.spin_imgsz_seq_crop_segment),
            ("aug_fliplr", self.aug_fliplr),
            ("aug_flipud", self.aug_flipud),
            ("aug_degrees", self.aug_degrees),
            ("aug_mosaic", self.aug_mosaic),
            ("aug_mixup", self.aug_mixup),
            ("aug_hsv_h", self.aug_hsv_h),
            ("aug_hsv_s", self.aug_hsv_s),
            ("aug_hsv_v", self.aug_hsv_v),
        ]:
            if attr in data:
                widget.setValue(data[attr])

        for attr, widget in [
            ("dedup", self.chk_dedup),
            ("enforce_square", self.chk_crop_square),
            ("auto_batch", self.chk_auto_batch),
            ("cache", self.chk_cache),
            ("aug_enabled", self.aug_group),
        ]:
            if attr in data:
                widget.setChecked(bool(data[attr]))

        for attr, widget in [
            ("model_obb_direct", self.combo_model_obb_direct),
            ("model_detect_direct", self.combo_model_detect_direct),
            ("model_segment_direct", self.combo_model_segment_direct),
            ("model_seq_detect", self.combo_model_seq_detect),
            ("model_seq_crop_obb", self.combo_model_seq_crop_obb),
            ("model_seq_crop_segment", self.combo_model_seq_crop_segment),
        ]:
            if attr in data:
                widget.setCurrentText(str(data[attr]))

        if "device" in data:
            self._set_device_combo(str(data["device"]))

        self._refresh_role_gating()
        self._on_training_selection_changed()

    def _save_training_config(self) -> None:
        import json

        path, _ = QFileDialog.getSaveFileName(
            self, "Save Training Config", "", "JSON (*.json)"
        )
        if not path:
            return
        data = self._collect_training_state()
        try:
            Path(path).write_text(json.dumps(data, indent=2), encoding="utf-8")
            QMessageBox.information(self, "Saved", f"Preset saved to:\n{path}")
        except Exception as exc:
            QMessageBox.critical(self, "Save Failed", str(exc))

    def _load_training_config(self) -> None:
        import json

        path, _ = QFileDialog.getOpenFileName(
            self, "Load Training Config", "", "JSON (*.json)"
        )
        if not path:
            return
        try:
            data = json.loads(Path(path).read_text(encoding="utf-8"))
        except Exception as exc:
            QMessageBox.critical(self, "Load Failed", str(exc))
            return
        self._apply_training_state(data)
        self._set_run_status(f"Loaded training preset from {path}")
        self._refresh_summary()
        QMessageBox.information(self, "Loaded", f"Preset loaded from:\n{path}")

    # ------------------------------------------------------------------
    # Persistent state
    # ------------------------------------------------------------------

    @staticmethod
    def _persistent_state_key() -> str:
        return "training_dialog_state"

    def _apply_persistent_state(self) -> None:
        """Restore the last training-dialog state from the UI settings file."""
        try:
            from ..utils import load_ui_settings
        except ImportError:
            return
        try:
            settings = load_ui_settings() or {}
        except Exception:
            return
        state = settings.get(self._persistent_state_key())
        if isinstance(state, dict):
            try:
                self._apply_training_state(state)
            except Exception:
                logger.warning("Failed to restore training-dialog state", exc_info=True)

    def _save_persistent_state(self) -> None:
        """Persist the current training-dialog state to the UI settings file."""
        try:
            from ..utils import load_ui_settings, save_ui_settings
        except ImportError:
            return
        try:
            settings = load_ui_settings() or {}
            settings[self._persistent_state_key()] = self._collect_training_state()
            save_ui_settings(settings)
        except Exception:
            logger.warning("Failed to persist training-dialog state", exc_info=True)

    def closeEvent(self, event) -> None:  # noqa: N802
        """Persist UI state when the dialog closes; refuse close while training."""
        if self._training_running:
            QMessageBox.information(
                self,
                "Training in progress",
                "Stop the running training session before closing this dialog.",
            )
            event.ignore()
            return
        try:
            self._save_persistent_state()
        except Exception:
            logger.warning(
                "Failed to persist training-dialog state on close", exc_info=True
            )
        super().closeEvent(event)
