"""TrainingDialog — full training configuration and run control for DetectKit."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QTabWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from hydra_suite.widgets.dialogs import BaseDialog
from hydra_suite.widgets.workers import BaseWorker

if TYPE_CHECKING:
    from ..models import DetectKitProject

logger = logging.getLogger(__name__)


_RECIPE_ROLE_MAP: dict[str, tuple[bool, bool, bool]] = {
    "direct_obb": (True, False, False),
    "sequential": (False, True, True),
    "all_stages": (True, True, True),
}

_RECIPE_DESCRIPTIONS: dict[str, str] = {
    "direct_obb": (
        "Train one full-image OBB model. Use this when objects are large enough "
        "to stay readable after the direct-image resize."
    ),
    "sequential": (
        "Train the two-stage sequence pipeline only. This is the better default "
        "for very small objects where direct OBB would shrink them too far."
    ),
    "all_stages": (
        "Train direct OBB and the full sequential pipeline together so you can "
        "compare them and keep both checkpoints available."
    ),
    "custom": (
        "Manual role selection is enabled. Use this only when the preset recipes "
        "do not match the training stages you want."
    ),
}


# ---------------------------------------------------------------------------
# Worker
# ---------------------------------------------------------------------------


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
        results = []
        parent_run = ""
        for entry in self.role_entries:
            if self._cancel:
                break
            role = entry["role"]
            spec = entry["spec"]
            publish_meta = entry["publish_meta"]
            self.role_started.emit(role.value)

            def _log(msg: str, _role=role):
                self.log_signal.emit(f"[{_role.value}] {msg}")

            def _prog(cur: int, total: int, _role=role):
                self.progress_signal.emit(_role.value, int(cur), int(total))

            try:
                result = self.orchestrator.run_role_training(
                    spec,
                    parent_run_id=parent_run,
                    publish_metadata=publish_meta,
                    log_cb=_log,
                    progress_cb=_prog,
                    should_cancel=self._should_cancel,
                )
            except Exception as exc:
                result = {
                    "run_id": "",
                    "success": False,
                    "error": str(exc),
                    "published_registry_key": "",
                    "published_model_path": "",
                }

            result["role"] = role.value
            results.append(result)
            ok = bool(result.get("success", False))
            msg = (
                f"run_id={result.get('run_id', '')}"
                if ok
                else (
                    result.get("error") or f"exit={result.get('exit_code', 'unknown')}"
                )
            )
            self.role_finished.emit(role.value, ok, msg)
            if result.get("run_id"):
                parent_run = str(result["run_id"])

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
        self._worker = None
        self._last_training_results: list[dict] = []
        self._role_logs: dict[str, list[str]] = {}
        self._current_role = ""
        self._dataset_fit_cache_key: tuple | None = None
        self._dataset_fit_cache_text = ""
        self._dataset_fit_dirty = True
        self.role_dataset_dirs: dict[str, str] = {}

        try:
            from hydra_suite.paths import get_training_workspace_dir
            from hydra_suite.training import TrainingOrchestrator

            self._workspace_default = get_training_workspace_dir("YOLO")
            self._orchestrator = TrainingOrchestrator(self._workspace_default)
        except ImportError:
            self._workspace_default = Path("./training_workspace")
            self._orchestrator = None

        self.resize(960, 860)
        self.setMinimumSize(900, 760)
        self._build_content()
        self._load_from_project()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_content(self) -> None:
        self._apply_training_dialog_styles()

        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(12)

        layout.addWidget(self._build_header())

        self.training_tabs = QTabWidget()
        self.training_tabs.addTab(self._build_overview_tab(), "Overview")
        self.training_tabs.addTab(self._build_training_tab(), "Advanced")
        self.training_tabs.addTab(self._build_monitoring_tab(), "Monitoring")
        layout.addWidget(self.training_tabs, 1)

        self.add_content(container)
        self._connect_summary_signals()

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
            "Pick the roles you want, verify the training plan, then run with a"
            " live view of progress and outputs. Advanced settings stay available"
            " without overwhelming the main workflow."
        )
        body.setObjectName("detectkitTrainingBody")
        body.setWordWrap(True)
        layout.addWidget(body)

        chip_row = QHBoxLayout()
        chip_row.setSpacing(8)
        chip_row.addWidget(self._build_workflow_chip("1. Pick recipe"))
        chip_row.addWidget(self._build_workflow_chip("2. Prepare datasets"))
        chip_row.addWidget(self._build_workflow_chip("3. Train and review"))
        chip_row.addStretch(1)
        layout.addLayout(chip_row)
        return frame

    def _build_workflow_chip(self, text: str) -> QLabel:
        chip = QLabel(text)
        chip.setObjectName("detectkitTrainingChip")
        return chip

    def _build_overview_tab(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(12)

        top_row = QHBoxLayout()
        top_row.setSpacing(12)
        left_column = QVBoxLayout()
        left_column.setSpacing(12)
        left_column.addWidget(self._build_recipe_group())
        left_column.addWidget(self._build_roles_group())
        top_row.addLayout(left_column, 2)

        right_column = QVBoxLayout()
        right_column.setSpacing(12)
        right_column.addWidget(self._build_summary_card())
        right_column.addWidget(self._build_dataset_fit_card())
        top_row.addLayout(right_column, 1)
        layout.addLayout(top_row)

        middle_row = QHBoxLayout()
        middle_row.setSpacing(12)
        middle_row.addWidget(self._build_config_group(), 2)
        middle_row.addWidget(self._build_publish_group(), 1)
        layout.addLayout(middle_row)

        layout.addWidget(self._build_run_group())
        layout.addStretch(1)
        return self._wrap_scroll_page(page)

    def _build_training_tab(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(12)

        layout.addWidget(self._build_hyperparams_group())

        lower_row = QHBoxLayout()
        lower_row.setSpacing(12)
        lower_row.addWidget(self._build_base_models_group(), 1)
        lower_row.addWidget(self._build_augmentation_group(), 1)
        layout.addLayout(lower_row)

        layout.addStretch(1)
        return self._wrap_scroll_page(page)

    def _build_monitoring_tab(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(12)

        note = self._build_section_note(
            "Review live metrics and logs here while a run is active. Quick test"
            " becomes available after at least one successful role finishes."
        )
        layout.addWidget(note)
        layout.addWidget(
            self._build_monitoring_group(
                "Loss Curve",
                "Training loss appears here when the loss plot widget is available.",
                self._build_loss_plot(),
            )
        )
        layout.addWidget(
            self._build_monitoring_group(
                "Session Log",
                "Per-role progress, warnings, and orchestration messages are collected here.",
                self._build_log(),
            )
        )
        layout.addStretch(1)
        return self._wrap_scroll_page(page)

    def _build_monitoring_group(
        self,
        title: str,
        description: str,
        content: QWidget,
    ) -> QGroupBox:
        gb = QGroupBox(title)
        layout = QVBoxLayout(gb)
        layout.setSpacing(8)
        layout.addWidget(self._build_section_note(description))
        layout.addWidget(content)
        return gb

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

        self.btn_refresh_dataset_fit = QPushButton("Refresh Dataset Fit")
        self.btn_refresh_dataset_fit.clicked.connect(self._refresh_dataset_fit)
        layout.addWidget(self.btn_refresh_dataset_fit)

        self.dataset_fit_view = QTextEdit()
        self.dataset_fit_view.setReadOnly(True)
        self.dataset_fit_view.setMinimumHeight(220)
        self.dataset_fit_view.setPlaceholderText(
            "Dataset fit guidance will appear here."
        )
        layout.addWidget(self.dataset_fit_view)
        return frame

    def _build_recipe_group(self) -> QGroupBox:
        gb = QGroupBox("Training Recipe")
        layout = QVBoxLayout(gb)
        layout.setSpacing(10)

        layout.addWidget(
            self._build_section_note(
                "Start with a recipe instead of individual stages. You can override the stage selection only when needed."
            )
        )

        self.recipe_combo = QComboBox()
        self.recipe_combo.addItem("Direct OBB", "direct_obb")
        self.recipe_combo.addItem("Sequential", "sequential")
        self.recipe_combo.addItem("Direct + Sequential", "all_stages")
        self.recipe_combo.addItem("Custom", "custom")
        layout.addWidget(self.recipe_combo)

        self.recipe_description = QLabel("")
        self.recipe_description.setObjectName("detectkitTrainingRoleBody")
        self.recipe_description.setWordWrap(True)
        layout.addWidget(self.recipe_description)
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
        self.recipe_combo.currentIndexChanged.connect(self._on_recipe_changed)
        self.spin_crop_pad.valueChanged.connect(self._mark_dataset_fit_dirty)
        self.spin_crop_min_px.valueChanged.connect(self._mark_dataset_fit_dirty)
        self.chk_crop_square.toggled.connect(self._mark_dataset_fit_dirty)
        self.spin_imgsz_obb_direct.valueChanged.connect(self._mark_dataset_fit_dirty)
        self.spin_imgsz_seq_crop_obb.valueChanged.connect(self._mark_dataset_fit_dirty)

        for checkbox in (
            self.chk_role_obb_direct,
            self.chk_role_seq_detect,
            self.chk_role_seq_crop_obb,
            self.chk_customize_roles,
            self.chk_auto_import,
            self.chk_auto_select,
        ):
            checkbox.toggled.connect(self._refresh_summary)

        for spinner in (
            self.spin_train,
            self.spin_val,
            self.spin_seed,
        ):
            spinner.valueChanged.connect(self._refresh_summary)

        self.class_names_edit.textChanged.connect(self._refresh_summary)
        self.line_species.textChanged.connect(self._refresh_summary)
        self.line_model_tag.textChanged.connect(self._refresh_summary)
        self.combo_device.currentTextChanged.connect(self._refresh_summary)
        self.combo_device.editTextChanged.connect(self._refresh_summary)

        self.chk_role_obb_direct.toggled.connect(self._on_role_selection_changed)
        self.chk_role_seq_detect.toggled.connect(self._on_role_selection_changed)
        self.chk_role_seq_crop_obb.toggled.connect(self._on_role_selection_changed)
        self.chk_customize_roles.toggled.connect(self._on_customize_roles_toggled)

    def _selected_role_keys(self) -> list[str]:
        selected = []
        if self.chk_role_obb_direct.isChecked():
            selected.append("obb_direct")
        if self.chk_role_seq_detect.isChecked():
            selected.append("seq_detect")
        if self.chk_role_seq_crop_obb.isChecked():
            selected.append("seq_crop_obb")
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
        prepared_count = sum(
            1 for role in selected_roles if self.role_dataset_dirs.get(role, "")
        )
        class_names = self._class_names()
        device = self.combo_device.currentText().strip() or "auto"
        tag = self.line_model_tag.text().strip() or "train"
        species = self.line_species.text().strip() or "unspecified"
        recipe_key = self._selected_recipe_key()
        recipe_label = self.recipe_combo.currentText().strip() or "Custom"
        publish_mode = (
            "auto-import on" if self.chk_auto_import.isChecked() else "artifacts only"
        )
        select_mode = (
            "auto-select on"
            if self.chk_auto_select.isChecked()
            else "selection unchanged"
        )
        summary = (
            f"<b>Recipe:</b> {recipe_label}"
            f" &nbsp;&bull;&nbsp; <b>Overrides:</b> {'manual' if self.chk_customize_roles.isChecked() or recipe_key == 'custom' else 'guided'}<br>"
            f"<b>Stages:</b> {self._preview_values(role_labels)}<br>"
            f"<b>Classes:</b> {len(class_names)} ({self._preview_values(class_names)})<br>"
            f"<b>Sources:</b> {len(self._project.sources)} OBB source(s)"
            f" &nbsp;&bull;&nbsp; <b>Datasets ready:</b> {prepared_count}/{len(selected_roles)}<br>"
            f"<b>Split:</b> {int(round(self.spin_train.value() * 100.0))}% train / "
            f"{int(round(self.spin_val.value() * 100.0))}% val"
            f" &nbsp;&bull;&nbsp; <b>Seed:</b> {self.spin_seed.value()}<br>"
            f"<b>Device:</b> {device}"
            f" &nbsp;&bull;&nbsp; <b>Publish:</b> {species} / {tag}<br>"
            f"<b>Import policy:</b> {publish_mode}"
            f" &nbsp;&bull;&nbsp; <b>Selection:</b> {select_mode}"
        )
        self.plan_summary.setText(summary)

    def _set_run_status(self, message: str) -> None:
        if hasattr(self, "run_status_label"):
            self.run_status_label.setText(message)

    # --- 1. Roles ---

    def _build_roles_group(self) -> QGroupBox:
        gb = QGroupBox("Stage Selection")
        v = QVBoxLayout(gb)
        v.setSpacing(10)

        v.addWidget(
            self._build_section_note(
                "Most users should keep stage selection guided by the training recipe. Only switch to manual selection if you need a non-standard run."
            )
        )

        self.chk_customize_roles = QCheckBox("Customize stage selection")
        self.chk_customize_roles.setChecked(False)
        v.addWidget(self.chk_customize_roles)

        self.recipe_roles_hint = QLabel("")
        self.recipe_roles_hint.setObjectName("detectkitTrainingRoleBody")
        self.recipe_roles_hint.setWordWrap(True)
        v.addWidget(self.recipe_roles_hint)

        self.role_cards_widget = QWidget()
        h = QHBoxLayout(self.role_cards_widget)
        h.setSpacing(10)
        self.chk_role_obb_direct = QCheckBox("obb_direct")
        self.chk_role_seq_detect = QCheckBox("seq_detect")
        self.chk_role_seq_crop_obb = QCheckBox("seq_crop_obb")
        self.chk_role_obb_direct.setChecked(True)
        self.chk_role_seq_detect.setChecked(True)
        self.chk_role_seq_crop_obb.setChecked(True)
        h.addWidget(
            self._build_role_card(
                self.chk_role_obb_direct,
                "OBB direct",
                "Train the main oriented bounding-box model directly on the merged project dataset.",
            ),
            1,
        )
        h.addWidget(
            self._build_role_card(
                self.chk_role_seq_detect,
                "Sequence detect",
                "Train the sequence detector derived from the OBB role outputs for staged inference.",
            ),
            1,
        )
        h.addWidget(
            self._build_role_card(
                self.chk_role_seq_crop_obb,
                "Sequence crop OBB",
                "Train the crop-focused OBB model used in the second sequence stage.",
            ),
            1,
        )
        v.addWidget(self.role_cards_widget)

        note = self._build_section_note(
            "Run order stays deterministic: OBB direct -> Sequence detect -> Sequence crop OBB."
        )
        v.addWidget(note)
        return gb

    # --- 2. Config ---

    def _build_config_group(self) -> QGroupBox:
        gb = QGroupBox("Dataset And Workspace")
        form = QFormLayout(gb)
        form.setSpacing(10)
        form.addRow(
            "",
            self._build_section_note(
                "Configure dataset merge behavior, class labels, and where training artifacts should be written."
            ),
        )

        self.class_names_edit = QPlainTextEdit()
        self.class_names_edit.setPlaceholderText("ant\nbee")
        self.class_names_edit.setFixedHeight(84)
        form.addRow("Class names", self.class_names_edit)

        self.line_workspace = QLineEdit(str(self._workspace_default))
        self.btn_workspace = QPushButton("Browse…")
        h_ws = QHBoxLayout()
        h_ws.addWidget(self.line_workspace, 1)
        h_ws.addWidget(self.btn_workspace)
        form.addRow("Workspace", h_ws)
        self.btn_workspace.clicked.connect(self._choose_workspace)

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
        h_split.addWidget(QLabel("train"))
        h_split.addWidget(self.spin_train)
        h_split.addWidget(QLabel("val"))
        h_split.addWidget(self.spin_val)
        form.addRow("Dataset split", h_split)

        self.spin_seed = QSpinBox()
        self.spin_seed.setRange(0, 999999)
        self.spin_seed.setValue(42)
        form.addRow("Random seed", self.spin_seed)

        self.chk_dedup = QCheckBox("Deduplicate source images by content hash")
        self.chk_dedup.setChecked(True)
        form.addRow("", self.chk_dedup)

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
        h_crop.addWidget(QLabel("pad"))
        h_crop.addWidget(self.spin_crop_pad)
        h_crop.addWidget(QLabel("min px"))
        h_crop.addWidget(self.spin_crop_min_px)
        h_crop.addWidget(self.chk_crop_square)
        form.addRow("Sequence crop settings", h_crop)

        # Device
        self.combo_device = QComboBox()
        self.combo_device.setEditable(True)
        self.combo_device.setToolTip(
            "Select compute device. For multi-GPU, type comma-separated list."
        )
        self.combo_device.addItems(self._build_device_options())
        form.addRow("Compute device", self.combo_device)

        return gb

    # --- 3. Hyperparameters ---

    def _build_hyperparams_group(self) -> QGroupBox:
        gb = QGroupBox("Training Hyperparameters")
        g = QGridLayout(gb)
        g.setHorizontalSpacing(12)
        g.setVerticalSpacing(10)

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
        g.addWidget(QLabel("epochs"), 1, 0)
        g.addWidget(self.spin_epochs, 1, 1)

        self.spin_batch = QSpinBox()
        self.spin_batch.setRange(1, 256)
        self.spin_batch.setValue(16)
        self.chk_auto_batch = QCheckBox("Auto")
        self.chk_auto_batch.setToolTip(
            "Let Ultralytics auto-detect optimal batch size (batch=-1)."
        )
        self.chk_auto_batch.toggled.connect(
            lambda checked: self.spin_batch.setEnabled(not checked)
        )
        batch_layout = QHBoxLayout()
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
        g.addWidget(QLabel("lr0"), 1, 4)
        g.addWidget(self.spin_lr0, 1, 5)

        # Row 1: patience, workers, cache
        self.spin_patience = QSpinBox()
        self.spin_patience.setRange(1, 500)
        self.spin_patience.setValue(30)
        g.addWidget(QLabel("patience"), 2, 0)
        g.addWidget(self.spin_patience, 2, 1)

        self.spin_workers = QSpinBox()
        self.spin_workers.setRange(0, 32)
        self.spin_workers.setValue(8)
        g.addWidget(QLabel("workers"), 2, 2)
        g.addWidget(self.spin_workers, 2, 3)

        self.chk_cache = QCheckBox("Cache")
        g.addWidget(self.chk_cache, 2, 4, 1, 2)

        # Row 2: per-role imgsz
        self.spin_imgsz_obb_direct = QSpinBox()
        self.spin_imgsz_obb_direct.setRange(64, 2048)
        self.spin_imgsz_obb_direct.setValue(640)
        g.addWidget(QLabel("imgsz (obb_direct)"), 3, 0)
        g.addWidget(self.spin_imgsz_obb_direct, 3, 1)

        self.spin_imgsz_seq_detect = QSpinBox()
        self.spin_imgsz_seq_detect.setRange(64, 2048)
        self.spin_imgsz_seq_detect.setValue(640)
        g.addWidget(QLabel("imgsz (seq_detect)"), 3, 2)
        g.addWidget(self.spin_imgsz_seq_detect, 3, 3)

        self.spin_imgsz_seq_crop_obb = QSpinBox()
        self.spin_imgsz_seq_crop_obb.setRange(64, 2048)
        self.spin_imgsz_seq_crop_obb.setValue(160)
        self.spin_imgsz_seq_crop_obb.setToolTip(
            "Must match YOLO_SEQ_STAGE2_IMGSZ used during inference (default 160)."
        )
        g.addWidget(QLabel("imgsz (seq_crop_obb)"), 3, 4)
        g.addWidget(self.spin_imgsz_seq_crop_obb, 3, 5)

        return gb

    # --- 4. Base Models ---

    def _build_base_models_group(self) -> QGroupBox:
        gb = QGroupBox("Base Checkpoints")
        form = QFormLayout(gb)
        form.setSpacing(10)
        form.addRow(
            "",
            self._build_section_note(
                "Each role can start from a different YOLO checkpoint. Editable fields let you point to custom weights."
            ),
        )

        self.combo_model_obb_direct = QComboBox()
        self.combo_model_obb_direct.setEditable(True)
        self.combo_model_obb_direct.addItems(
            [
                "yolo26n-obb.pt",
                "yolo26s-obb.pt",
                "yolo26m-obb.pt",
                "yolo26l-obb.pt",
                "yolo26x-obb.pt",
            ]
        )
        self.combo_model_obb_direct.setCurrentText("yolo26s-obb.pt")
        form.addRow("obb_direct", self.combo_model_obb_direct)

        self.combo_model_seq_detect = QComboBox()
        self.combo_model_seq_detect.setEditable(True)
        self.combo_model_seq_detect.addItems(
            ["yolo26n.pt", "yolo26s.pt", "yolo26m.pt", "yolo26l.pt", "yolo26x.pt"]
        )
        self.combo_model_seq_detect.setCurrentText("yolo26s.pt")
        form.addRow("seq_detect", self.combo_model_seq_detect)

        self.combo_model_seq_crop_obb = QComboBox()
        self.combo_model_seq_crop_obb.setEditable(True)
        self.combo_model_seq_crop_obb.addItems(
            ["yolo26n-obb.pt", "yolo26s-obb.pt", "yolo26m-obb.pt"]
        )
        self.combo_model_seq_crop_obb.setCurrentText("yolo26s-obb.pt")
        form.addRow("seq_crop_obb", self.combo_model_seq_crop_obb)

        return gb

    # --- 5. Augmentation ---

    def _build_augmentation_group(self) -> QGroupBox:
        self.aug_group = QGroupBox("Augmentation")
        self.aug_group.setCheckable(True)
        self.aug_group.setChecked(True)
        v = QVBoxLayout(self.aug_group)
        v.setSpacing(8)

        note = QLabel(
            "These are passed directly to Ultralytics. "
            "Set fliplr=0 for asymmetric animals."
        )
        note.setObjectName("detectkitTrainingNote")
        note.setWordWrap(True)
        v.addWidget(note)

        form = QFormLayout()

        def _spin(default: float, maximum: float = 1.0) -> QDoubleSpinBox:
            sb = QDoubleSpinBox()
            sb.setRange(0.0, maximum)
            sb.setDecimals(3)
            sb.setSingleStep(0.05)
            sb.setValue(default)
            return sb

        self.aug_fliplr = _spin(0.5)
        form.addRow("fliplr", self.aug_fliplr)
        self.aug_flipud = _spin(0.0)
        form.addRow("flipud", self.aug_flipud)
        self.aug_degrees = _spin(0.0, 360.0)
        form.addRow("degrees", self.aug_degrees)
        self.aug_mosaic = _spin(1.0)
        form.addRow("mosaic", self.aug_mosaic)
        self.aug_mixup = _spin(0.0)
        form.addRow("mixup", self.aug_mixup)
        self.aug_hsv_h = _spin(0.015)
        form.addRow("hsv_h", self.aug_hsv_h)
        self.aug_hsv_s = _spin(0.7)
        form.addRow("hsv_s", self.aug_hsv_s)
        self.aug_hsv_v = _spin(0.4)
        form.addRow("hsv_v", self.aug_hsv_v)

        v.addLayout(form)
        return self.aug_group

    # --- 6. Publish ---

    def _build_publish_group(self) -> QGroupBox:
        gb = QGroupBox("Publish And Naming")
        form = QFormLayout(gb)
        form.setSpacing(10)
        form.addRow(
            "",
            self._build_section_note(
                "Define how successful runs are named and whether they should be imported back into the project automatically."
            ),
        )

        self.line_species = QLineEdit("")
        form.addRow("Species", self.line_species)

        self.line_model_tag = QLineEdit("train")
        form.addRow("Model tag", self.line_model_tag)

        self.chk_auto_import = QCheckBox(
            "Auto-import successful models into repository"
        )
        self.chk_auto_import.setChecked(True)
        form.addRow("", self.chk_auto_import)

        self.chk_auto_select = QCheckBox("Auto-select newly imported models in main UI")
        self.chk_auto_select.setChecked(False)
        form.addRow("", self.chk_auto_select)

        return gb

    # --- 7. Run Controls ---

    def _build_run_group(self) -> QGroupBox:
        gb = QGroupBox("Run Session")
        v = QVBoxLayout(gb)
        v.setSpacing(10)

        self.run_status_label = QLabel(
            "Ready to prepare datasets for the selected roles."
        )
        self.run_status_label.setObjectName("detectkitRunStatus")
        self.run_status_label.setWordWrap(True)
        v.addWidget(self.run_status_label)

        row1 = QHBoxLayout()
        self.btn_build = QPushButton("Prepare Datasets")
        self.btn_start = QPushButton("Start Training")
        self.btn_cancel = QPushButton("Stop Run")
        self.btn_cancel.setEnabled(False)
        self.btn_resume = QPushButton("Resume Last Run")
        self.btn_resume.setEnabled(False)
        self.btn_resume.setToolTip(
            "Resume training from last.pt checkpoint of the most recent run."
        )
        self.btn_detach = QPushButton("Run Detached")
        self.btn_detach.setToolTip("Launch training as a background process.")
        row1.addWidget(self.btn_build)
        row1.addWidget(self.btn_start)
        row1.addWidget(self.btn_cancel)
        row1.addWidget(self.btn_resume)
        row1.addWidget(self.btn_detach)
        v.addLayout(row1)

        row2 = QHBoxLayout()
        self.btn_quick_test = QPushButton("Open Quick Test")
        self.btn_quick_test.setEnabled(False)
        self.btn_save_config = QPushButton("Save Preset")
        self.btn_load_config = QPushButton("Load Preset")
        row2.addWidget(self.btn_quick_test)
        row2.addWidget(self.btn_save_config)
        row2.addWidget(self.btn_load_config)
        v.addLayout(row2)

        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        self.progress.setFormat("Idle")
        v.addWidget(self.progress)

        self.btn_build.clicked.connect(self._build_role_datasets)
        self.btn_start.clicked.connect(self._start_training)
        self.btn_cancel.clicked.connect(self._cancel_training)
        self.btn_resume.clicked.connect(self._resume_training)
        self.btn_detach.clicked.connect(self._start_detached)
        self.btn_quick_test.clicked.connect(self._quick_test)
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
            self.loss_plot.setMinimumHeight(180)
            return self.loss_plot
        except ImportError:
            self.loss_plot = None
            placeholder = QLabel("Loss plot not available (trackerkit not installed).")
            placeholder.setStyleSheet("color: gray; font-style: italic;")
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

        from ..models import normalize_class_names

        self.class_names_edit.setPlainText(
            "\n".join(normalize_class_names(proj.class_names))
        )

        self.chk_role_obb_direct.setChecked(proj.role_obb_direct)
        self.chk_role_seq_detect.setChecked(proj.role_seq_detect)
        self.chk_role_seq_crop_obb.setChecked(proj.role_seq_crop_obb)

        self.spin_train.setValue(proj.split_train)
        self.spin_val.setValue(proj.split_val)
        self.spin_seed.setValue(proj.seed)
        self.chk_dedup.setChecked(proj.dedup)

        self.spin_crop_pad.setValue(proj.crop_pad_ratio)
        self.spin_crop_min_px.setValue(proj.min_crop_size_px)
        self.chk_crop_square.setChecked(proj.enforce_square)

        self.combo_device.setCurrentText(proj.device or "auto")

        self.spin_epochs.setValue(proj.epochs)
        self.spin_batch.setValue(proj.batch)
        self.chk_auto_batch.setChecked(proj.auto_batch)
        self.spin_lr0.setValue(proj.lr0)
        self.spin_patience.setValue(proj.patience)
        self.spin_workers.setValue(proj.workers)
        self.chk_cache.setChecked(proj.cache)

        self.spin_imgsz_obb_direct.setValue(proj.imgsz_obb_direct)
        self.spin_imgsz_seq_detect.setValue(proj.imgsz_seq_detect)
        self.spin_imgsz_seq_crop_obb.setValue(proj.imgsz_seq_crop_obb)

        self.combo_model_obb_direct.setCurrentText(proj.model_obb_direct)
        self.combo_model_seq_detect.setCurrentText(proj.model_seq_detect)
        self.combo_model_seq_crop_obb.setCurrentText(proj.model_seq_crop_obb)

        self.aug_group.setChecked(proj.aug_enabled)
        self.aug_fliplr.setValue(proj.aug_fliplr)
        self.aug_flipud.setValue(proj.aug_flipud)
        self.aug_degrees.setValue(proj.aug_degrees)
        self.aug_mosaic.setValue(proj.aug_mosaic)
        self.aug_mixup.setValue(proj.aug_mixup)
        self.aug_hsv_h.setValue(proj.aug_hsv_h)
        self.aug_hsv_s.setValue(proj.aug_hsv_s)
        self.aug_hsv_v.setValue(proj.aug_hsv_v)

        self.line_species.setText(proj.species or "")
        self.line_model_tag.setText(proj.model_tag or "train")
        self.chk_auto_import.setChecked(proj.auto_import)
        self.chk_auto_select.setChecked(proj.auto_select)
        self._sync_recipe_from_roles()
        self._set_run_status("Ready to prepare datasets for the selected roles.")
        self._refresh_summary()
        self._refresh_dataset_fit()

    def _write_to_project(self) -> None:
        proj = self._project

        from ..models import normalize_class_names

        proj.class_names = normalize_class_names(
            self.class_names_edit.toPlainText().splitlines()
        )

        proj.role_obb_direct = self.chk_role_obb_direct.isChecked()
        proj.role_seq_detect = self.chk_role_seq_detect.isChecked()
        proj.role_seq_crop_obb = self.chk_role_seq_crop_obb.isChecked()

        proj.split_train = self.spin_train.value()
        proj.split_val = self.spin_val.value()
        proj.seed = self.spin_seed.value()
        proj.dedup = self.chk_dedup.isChecked()

        proj.crop_pad_ratio = self.spin_crop_pad.value()
        proj.min_crop_size_px = self.spin_crop_min_px.value()
        proj.enforce_square = self.chk_crop_square.isChecked()

        proj.device = self.combo_device.currentText().strip() or "auto"

        proj.epochs = self.spin_epochs.value()
        proj.batch = self.spin_batch.value()
        proj.auto_batch = self.chk_auto_batch.isChecked()
        proj.lr0 = self.spin_lr0.value()
        proj.patience = self.spin_patience.value()
        proj.workers = self.spin_workers.value()
        proj.cache = self.chk_cache.isChecked()

        proj.imgsz_obb_direct = self.spin_imgsz_obb_direct.value()
        proj.imgsz_seq_detect = self.spin_imgsz_seq_detect.value()
        proj.imgsz_seq_crop_obb = self.spin_imgsz_seq_crop_obb.value()

        proj.model_obb_direct = self.combo_model_obb_direct.currentText()
        proj.model_seq_detect = self.combo_model_seq_detect.currentText()
        proj.model_seq_crop_obb = self.combo_model_seq_crop_obb.currentText()

        proj.aug_enabled = self.aug_group.isChecked()
        proj.aug_fliplr = self.aug_fliplr.value()
        proj.aug_flipud = self.aug_flipud.value()
        proj.aug_degrees = self.aug_degrees.value()
        proj.aug_mosaic = self.aug_mosaic.value()
        proj.aug_mixup = self.aug_mixup.value()
        proj.aug_hsv_h = self.aug_hsv_h.value()
        proj.aug_hsv_s = self.aug_hsv_s.value()
        proj.aug_hsv_v = self.aug_hsv_v.value()

        proj.species = self.line_species.text().strip()
        proj.model_tag = self.line_model_tag.text().strip() or "train"
        proj.auto_import = self.chk_auto_import.isChecked()
        proj.auto_select = self.chk_auto_select.isChecked()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _class_names(self) -> list[str]:
        from ..models import normalize_class_names

        return normalize_class_names(self.class_names_edit.toPlainText().splitlines())

    def _selected_recipe_key(self) -> str:
        return (
            str(self.recipe_combo.currentData() or "all_stages").strip() or "all_stages"
        )

    def _recipe_for_roles(self) -> str:
        signature = (
            bool(self.chk_role_obb_direct.isChecked()),
            bool(self.chk_role_seq_detect.isChecked()),
            bool(self.chk_role_seq_crop_obb.isChecked()),
        )
        for recipe_key, recipe_signature in _RECIPE_ROLE_MAP.items():
            if recipe_signature == signature:
                return recipe_key
        return "custom"

    def _set_recipe_combo(self, recipe_key: str) -> None:
        idx = self.recipe_combo.findData(recipe_key)
        if idx < 0:
            idx = self.recipe_combo.findData("custom")
        if idx >= 0:
            self.recipe_combo.setCurrentIndex(idx)

    def _apply_recipe_roles(self, recipe_key: str) -> None:
        signature = _RECIPE_ROLE_MAP.get(recipe_key)
        if signature is None:
            return
        checkboxes = (
            self.chk_role_obb_direct,
            self.chk_role_seq_detect,
            self.chk_role_seq_crop_obb,
        )
        for checkbox, checked in zip(checkboxes, signature):
            checkbox.blockSignals(True)
            checkbox.setChecked(bool(checked))
            checkbox.blockSignals(False)

    def _update_recipe_description(self) -> None:
        recipe_key = self._selected_recipe_key()
        self.recipe_description.setText(_RECIPE_DESCRIPTIONS.get(recipe_key, ""))

    def _update_role_selection_visibility(self) -> None:
        guided = not self.chk_customize_roles.isChecked()
        self.role_cards_widget.setVisible(not guided)
        selected_labels = [
            self._role_display_name(role) for role in self._selected_role_keys()
        ]
        if guided:
            self.recipe_roles_hint.setText(
                "Guided stage selection: "
                + self._preview_values(selected_labels)
                + ". Switch on manual selection only if you need a different stage mix."
            )
            self.recipe_roles_hint.setVisible(True)
        else:
            self.recipe_roles_hint.setText(
                "Manual stage selection is enabled. DetectKit will train exactly the stages checked below."
            )
            self.recipe_roles_hint.setVisible(True)

    def _sync_recipe_from_roles(self) -> None:
        recipe_key = self._recipe_for_roles()
        self.recipe_combo.blockSignals(True)
        self._set_recipe_combo(recipe_key)
        self.recipe_combo.blockSignals(False)

        self.chk_customize_roles.blockSignals(True)
        self.chk_customize_roles.setChecked(recipe_key == "custom")
        self.chk_customize_roles.blockSignals(False)

        self._update_recipe_description()
        self._update_role_selection_visibility()

    def _on_role_selection_changed(self, *_args) -> None:
        if self.chk_customize_roles.isChecked():
            recipe_key = self._recipe_for_roles()
            self.recipe_combo.blockSignals(True)
            self._set_recipe_combo(recipe_key)
            self.recipe_combo.blockSignals(False)
            self._update_recipe_description()
        self._mark_dataset_fit_dirty()

    def _on_recipe_changed(self, *_args) -> None:
        recipe_key = self._selected_recipe_key()
        if recipe_key == "custom":
            if not self.chk_customize_roles.isChecked():
                self.chk_customize_roles.blockSignals(True)
                self.chk_customize_roles.setChecked(True)
                self.chk_customize_roles.blockSignals(False)
        elif self.chk_customize_roles.isChecked():
            self.chk_customize_roles.blockSignals(True)
            self.chk_customize_roles.setChecked(False)
            self.chk_customize_roles.blockSignals(False)
        self._update_recipe_description()
        if not self.chk_customize_roles.isChecked() and recipe_key != "custom":
            self._apply_recipe_roles(recipe_key)
        self._update_role_selection_visibility()
        self._refresh_summary()
        self._mark_dataset_fit_dirty()

    def _on_customize_roles_toggled(self, checked: bool) -> None:
        if not checked:
            recipe_key = self._selected_recipe_key()
            if recipe_key == "custom":
                recipe_key = self._recipe_for_roles()
                if recipe_key == "custom":
                    recipe_key = "all_stages"
                self.recipe_combo.blockSignals(True)
                self._set_recipe_combo(recipe_key)
                self.recipe_combo.blockSignals(False)
                self._update_recipe_description()
            self._apply_recipe_roles(recipe_key)
        else:
            if self._selected_recipe_key() != "custom":
                self.recipe_combo.blockSignals(True)
                self._set_recipe_combo("custom")
                self.recipe_combo.blockSignals(False)
                self._update_recipe_description()
        self._update_role_selection_visibility()
        self._refresh_summary()
        self._mark_dataset_fit_dirty()

    def _dataset_fit_key(self) -> tuple:
        source_paths = tuple(
            str(src.path).strip()
            for src in self._project.sources
            if str(src.path).strip()
        )
        return (
            source_paths,
            round(float(self.spin_crop_pad.value()), 4),
            int(self.spin_crop_min_px.value()),
            bool(self.chk_crop_square.isChecked()),
            int(self.spin_imgsz_obb_direct.value()),
            int(self.spin_imgsz_seq_crop_obb.value()),
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

        source_paths = [
            str(src.path).strip()
            for src in self._project.sources
            if str(src.path).strip()
        ]
        if not source_paths:
            self.dataset_fit_status.setText("No source datasets configured yet.")
            self.dataset_fit_view.setPlainText(
                "Add one or more DetectKit OBB sources to see size and recipe guidance here."
            )
            self._dataset_fit_cache_key = cache_key
            self._dataset_fit_cache_text = self.dataset_fit_view.toPlainText()
            self._dataset_fit_dirty = False
            return

        merged = DatasetInspection(root_dir="overview")
        valid_items = 0
        for source_path in source_paths:
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

    def _build_device_options(self) -> list[str]:
        try:
            from hydra_suite.utils.gpu_utils import get_device_info

            info = get_device_info()
        except ImportError:
            info = {}
        options = ["auto", "cpu"]
        if info.get("torch_cuda_available"):
            options.append("cuda")
            count = int(info.get("torch_cuda_device_count", 0) or 0)
            for i in range(count):
                options.append(f"cuda:{i}")
        if info.get("mps_available"):
            options.append("mps")
        if info.get("rocm_available"):
            options.append("rocm")
        return options

    def _imgsz_for_role(self, role) -> int:
        try:
            from hydra_suite.training import TrainingRole

            if role == TrainingRole.OBB_DIRECT:
                return self.spin_imgsz_obb_direct.value()
            if role == TrainingRole.SEQ_DETECT:
                return self.spin_imgsz_seq_detect.value()
            if role == TrainingRole.SEQ_CROP_OBB:
                return self.spin_imgsz_seq_crop_obb.value()
        except ImportError:
            pass
        return 640

    def _base_model_for_role(self, role) -> str:
        try:
            from hydra_suite.training import TrainingRole

            if role == TrainingRole.OBB_DIRECT:
                return self.combo_model_obb_direct.currentText().strip()
            if role == TrainingRole.SEQ_DETECT:
                return self.combo_model_seq_detect.currentText().strip()
            if role == TrainingRole.SEQ_CROP_OBB:
                return self.combo_model_seq_crop_obb.currentText().strip()
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
        if self.chk_role_seq_detect.isChecked():
            roles.append(TrainingRole.SEQ_DETECT)
        if self.chk_role_seq_crop_obb.isChecked():
            roles.append(TrainingRole.SEQ_CROP_OBB)
        return roles

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
                    SourceDataset(path=p, source_type="yolo_obb", name=Path(p).name)
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
        species = self.line_species.text().strip() or "species"
        tag = self.line_model_tag.text().strip() or "train"
        training_params: dict = {"imgsz": self._imgsz_for_role(role)}
        try:
            from hydra_suite.training import TrainingRole

            if role == TrainingRole.SEQ_CROP_OBB:
                training_params["crop_pad_ratio"] = self.spin_crop_pad.value()
                training_params["min_crop_size_px"] = self.spin_crop_min_px.value()
                training_params["enforce_square"] = self.chk_crop_square.isChecked()
        except ImportError:
            pass
        return {
            "size": self._infer_size_token(base_model),
            "species": species,
            "model_info": f"{tag}_{role.value}",
            "training_params": training_params,
        }

    def _append_log(self, text: str) -> None:
        log_text = str(text)
        self.log_view.append(log_text)
        if self._current_role:
            self._role_logs.setdefault(self._current_role, []).append(log_text)
        if self.loss_plot is not None:
            self.loss_plot.ingest_log_line(log_text)

    def _choose_workspace(self) -> None:
        d = QFileDialog.getExistingDirectory(self, "Select Workspace Root")
        if d:
            self.line_workspace.setText(d)
            try:
                from hydra_suite.training import TrainingOrchestrator

                self._orchestrator = TrainingOrchestrator(d)
            except ImportError:
                pass
            self._set_run_status(f"Workspace updated to {d}")
            self._refresh_summary()

    def _get_orchestrator(self):
        if self._orchestrator is None:
            try:
                from hydra_suite.training import TrainingOrchestrator

                ws_text = self.line_workspace.text().strip()
                ws = Path(ws_text) if ws_text else self._workspace_default
                self._orchestrator = TrainingOrchestrator(ws)
            except ImportError:
                return None
        return self._orchestrator

    # ------------------------------------------------------------------
    # Dataset building
    # ------------------------------------------------------------------

    def _build_role_datasets(self) -> bool:
        orchestrator = self._get_orchestrator()
        if orchestrator is None:
            QMessageBox.critical(
                self, "Not Available", "Training dependencies not available."
            )
            return False

        obb_sources = self._collect_sources()
        if not obb_sources:
            QMessageBox.warning(
                self, "No OBB Sources", "Add at least one OBB source dataset."
            )
            return False

        roles = self._selected_roles()
        if not roles:
            QMessageBox.warning(self, "No Roles", "Select at least one training role.")
            return False

        try:
            from hydra_suite.training import SplitConfig

            split = SplitConfig(
                train=self.spin_train.value(),
                val=self.spin_val.value(),
                test=0.0,
            )
            merged = orchestrator.build_merged_obb_dataset(
                obb_sources,
                class_names=self._class_names(),
                split_cfg=split,
                seed=self.spin_seed.value(),
                dedup=self.chk_dedup.isChecked(),
            )
            self.role_dataset_dirs = {}
            self._append_log(f"Merged dataset: {merged.dataset_dir}")

            for role in roles:
                build = orchestrator.build_role_dataset(
                    role,
                    merged.dataset_dir,
                    class_names=self._class_names(),
                    crop_pad_ratio=self.spin_crop_pad.value(),
                    min_crop_size_px=self.spin_crop_min_px.value(),
                    enforce_square=self.chk_crop_square.isChecked(),
                )
                self.role_dataset_dirs[role.value] = build.dataset_dir
                self._append_log(
                    f"Prepared [{role.value}] dataset: {build.dataset_dir}"
                )
        except Exception as exc:
            self._set_run_status(f"Dataset preparation failed: {exc}")
            QMessageBox.critical(self, "Build Failed", str(exc))
            return False

        self._set_run_status(
            f"Prepared datasets for {len(self.role_dataset_dirs)} selected role(s)."
        )
        self._refresh_summary()
        QMessageBox.information(
            self, "Datasets Ready", "Role datasets built successfully."
        )
        return True

    # ------------------------------------------------------------------
    # Training execution
    # ------------------------------------------------------------------

    def _start_training(self) -> None:
        if self._worker is not None and self._worker.isRunning():
            QMessageBox.warning(self, "Busy", "Training is already running.")
            return

        roles = self._selected_roles()
        if not roles:
            QMessageBox.warning(self, "No Roles", "Select at least one training role.")
            return

        if not self.role_dataset_dirs:
            if not self._build_role_datasets():
                return

        if not self.chk_auto_import.isChecked() and self.chk_auto_select.isChecked():
            QMessageBox.warning(
                self,
                "Invalid Publish Settings",
                "Auto-select requires auto-import to be enabled.",
            )
            return

        orchestrator = self._get_orchestrator()
        if orchestrator is None:
            self._append_log("Training dependencies not available.")
            return

        self._write_to_project()

        try:
            from hydra_suite.training import (
                AugmentationProfile,
                PublishPolicy,
                TrainingHyperParams,
                TrainingRunSpec,
            )
        except ImportError as exc:
            self._append_log(f"Training dependencies not available: {exc}")
            return

        source_obb = self._collect_sources()
        role_entries = []
        for role in roles:
            ds = self.role_dataset_dirs.get(role.value, "")
            if not ds:
                QMessageBox.warning(
                    self,
                    "Missing Dataset",
                    f"No dataset prepared for role: {role.value}",
                )
                return

            base_model = self._base_model_for_role(role)
            if not base_model:
                QMessageBox.warning(
                    self,
                    "Base Model",
                    f"Set base model for role: {role.value}",
                )
                return

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
            spec = TrainingRunSpec(
                role=role,
                source_datasets=source_obb,
                derived_dataset_dir=ds,
                base_model=base_model,
                hyperparams=TrainingHyperParams(
                    epochs=self.spin_epochs.value(),
                    imgsz=self._imgsz_for_role(role),
                    batch=batch_val,
                    lr0=self.spin_lr0.value(),
                    patience=self.spin_patience.value(),
                    workers=self.spin_workers.value(),
                    cache=self.chk_cache.isChecked(),
                ),
                device=self.combo_device.currentText().strip() or "auto",
                seed=self.spin_seed.value(),
                augmentation_profile=AugmentationProfile(
                    enabled=self.aug_group.isChecked(),
                    args=aug_args,
                ),
                publish_policy=PublishPolicy(
                    auto_import=self.chk_auto_import.isChecked(),
                    auto_select=self.chk_auto_select.isChecked(),
                ),
            )
            role_entries.append(
                {
                    "role": role,
                    "spec": spec,
                    "publish_meta": self._publish_meta_for_role(role, base_model),
                }
            )

        self._worker = _TrainingWorker(orchestrator, role_entries)
        self._worker.log_signal.connect(self._append_log)
        self._worker.role_started.connect(self._on_role_started)
        self._worker.role_finished.connect(self._on_role_finished)
        self._worker.progress_signal.connect(self._on_role_progress)
        self._worker.done_signal.connect(self._on_done)
        self._worker.finished.connect(self._on_worker_finished)

        self.btn_start.setEnabled(False)
        self.btn_cancel.setEnabled(True)
        self.progress.setValue(0)
        self.progress.setFormat("Starting…")
        self._role_logs = {}
        self._current_role = ""
        if self.loss_plot is not None:
            self.loss_plot.clear()
        self._set_run_status(
            f"Training started for {len(role_entries)} role(s). Review Monitoring for live output."
        )
        self._refresh_summary()
        self._worker.start()

    def _cancel_training(self) -> None:
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

        self.btn_resume.setEnabled(
            any(
                r.get("_run_dir")
                and Path(r["_run_dir"]).joinpath("weights", "last.pt").exists()
                for r in results
            )
        )

        succeeded = [r for r in results if r.get("success")]
        failed = [r for r in results if not r.get("success")]
        self.btn_quick_test.setEnabled(bool(self._quick_test_enabled_results()))

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
        self.btn_start.setEnabled(True)
        self.btn_cancel.setEnabled(False)
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
            derived_dataset_dir=resume_result.get("_run_dir", ""),
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
        entry = {
            "role": role,
            "spec": spec,
            "publish_meta": {"class_names": class_names, "resumed_from": str(last_pt)},
        }

        orchestrator = self._get_orchestrator()
        if orchestrator is None:
            self._append_log("Training dependencies not available.")
            return

        self._append_log(f"Resuming training from {last_pt}")
        self.btn_start.setEnabled(False)
        self.btn_resume.setEnabled(False)
        self.btn_cancel.setEnabled(True)
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
    # Detached training
    # ------------------------------------------------------------------

    def _start_detached(self) -> None:
        import subprocess as _subprocess

        roles = self._selected_roles()
        if not roles:
            QMessageBox.warning(self, "No Roles", "Select at least one training role.")
            return
        if not self.role_dataset_dirs:
            if not self._build_role_datasets():
                return

        try:
            from hydra_suite.training import TrainingHyperParams, TrainingRunSpec
            from hydra_suite.training.runner import build_ultralytics_command
        except ImportError as exc:
            QMessageBox.critical(self, "Not Available", f"Training dependencies: {exc}")
            return

        launched = []
        ws_text = self.line_workspace.text().strip()
        ws = Path(ws_text) if ws_text else self._workspace_default

        for role in roles:
            ds = self.role_dataset_dirs.get(role.value, "")
            if not ds:
                continue
            base_model = self._base_model_for_role(role)
            if not base_model:
                continue

            batch_val = (
                -1 if self.chk_auto_batch.isChecked() else self.spin_batch.value()
            )
            spec = TrainingRunSpec(
                role=role,
                source_datasets=self._collect_sources(),
                derived_dataset_dir=ds,
                base_model=base_model,
                hyperparams=TrainingHyperParams(
                    epochs=self.spin_epochs.value(),
                    imgsz=self._imgsz_for_role(role),
                    batch=batch_val,
                    lr0=self.spin_lr0.value(),
                    patience=self.spin_patience.value(),
                    workers=self.spin_workers.value(),
                    cache=self.chk_cache.isChecked(),
                ),
                device=self.combo_device.currentText().strip() or "auto",
                seed=self.spin_seed.value(),
            )

            run_dir = ws / "runs" / ("detached_" + role.value)
            run_dir.mkdir(parents=True, exist_ok=True)
            cmd = build_ultralytics_command(spec, str(run_dir))

            log_file = run_dir / "detached_output.log"
            with open(log_file, "w") as log_fh:
                proc = _subprocess.Popen(
                    cmd,
                    stdout=log_fh,
                    stderr=_subprocess.STDOUT,
                    start_new_session=True,
                )
            launched.append((role.value, proc.pid, str(log_file)))
            self._append_log(
                f"Detached {role.value} training: PID={proc.pid}, log={log_file}"
            )

        if launched:
            self._set_run_status(
                f"Detached training launched for {len(launched)} role(s)."
            )
            msg = "\n".join(
                f"* {r}: PID {pid}\n  Log: {log}" for r, pid, log in launched
            )
            QMessageBox.information(
                self,
                "Detached Training Started",
                f"Training launched in background:\n\n{msg}\n\n"
                "You can close this dialog. Check Run History for results.",
            )

    # ------------------------------------------------------------------
    # Quick Test
    # ------------------------------------------------------------------

    def _quick_test_enabled_results(self) -> list[dict]:
        """Return successful training results that Quick Test can exercise."""
        supported_roles = {"obb_direct", "seq_crop_obb"}
        return [
            result
            for result in (self._last_training_results or [])
            if result.get("success")
            and str(result.get("role") or "obb_direct").strip().lower()
            in supported_roles
        ]

    def _quick_test_detect_model_path(self, role: str) -> str:
        """Resolve the paired seq-detect model path needed for crop-OBB quick tests."""
        if str(role or "").strip().lower() != "seq_crop_obb":
            return ""

        for result in self._last_training_results or []:
            result_role = str(result.get("role") or "").strip().lower()
            if result_role != "seq_detect" or not result.get("success"):
                continue
            candidate = (
                result.get("project_model_path")
                or result.get("published_model_path")
                or result.get("artifact_path", "")
            )
            if candidate and Path(candidate).exists():
                return str(candidate)

        try:
            from ..project import detectkit_latest_model_path_for_role
        except ImportError:
            return ""

        return detectkit_latest_model_path_for_role(self._project, "seq_detect")

    def _quick_test(self) -> None:
        succeeded = self._quick_test_enabled_results()
        if not succeeded:
            QMessageBox.warning(
                self,
                "No Model Available",
                "No quick-testable trained model found. Quick Test currently supports OBB direct models and sequence crop OBB models with a paired detect checkpoint.",
            )
            return

        result = succeeded[0]
        model_path = (
            result.get("project_model_path")
            or result.get("published_model_path")
            or result.get("artifact_path", "")
        )
        if not model_path or not Path(model_path).exists():
            QMessageBox.warning(
                self, "Model Not Found", f"Model file not found: {model_path}"
            )
            return

        try:
            from hydra_suite.trackerkit.gui.dialogs.model_test_dialog import (
                ModelTestDialog,
            )
        except ImportError:
            QMessageBox.information(
                self, "Not Available", "Model test dialog is not available."
            )
            return

        role = str(result.get("role", "obb_direct") or "obb_direct").strip().lower()
        dataset_dir = self.role_dataset_dirs.get(role, "")
        if not dataset_dir:
            QMessageBox.warning(
                self, "No Dataset", f"No dataset directory found for role '{role}'."
            )
            return

        detect_model_path = self._quick_test_detect_model_path(role)
        if role == "seq_crop_obb" and not detect_model_path:
            QMessageBox.warning(
                self,
                "Missing Detect Model",
                "Quick Test for sequence crop OBB needs a paired sequence-detect checkpoint.",
            )
            return

        imgsz = {
            "obb_direct": int(self.spin_imgsz_obb_direct.value()),
            "seq_detect": int(self.spin_imgsz_seq_detect.value()),
            "seq_crop_obb": int(self.spin_imgsz_seq_crop_obb.value()),
        }.get(role, int(self.spin_imgsz_obb_direct.value()))

        dlg = ModelTestDialog(
            model_path=model_path,
            role=role,
            dataset_dir=dataset_dir,
            device=self.combo_device.currentText() or "cpu",
            imgsz=imgsz,
            crop_pad_ratio=float(self.spin_crop_pad.value()),
            min_crop_size_px=int(self.spin_crop_min_px.value()),
            enforce_square=self.chk_crop_square.isChecked(),
            detect_model_path=detect_model_path,
            parent=self,
        )
        dlg.open()

    # ------------------------------------------------------------------
    # Save / Load Config
    # ------------------------------------------------------------------

    def _save_training_config(self) -> None:
        import json

        path, _ = QFileDialog.getSaveFileName(
            self, "Save Training Config", "", "JSON (*.json)"
        )
        if not path:
            return
        self._write_to_project()
        proj = self._project
        data = {
            "roles": {
                "obb_direct": proj.role_obb_direct,
                "seq_detect": proj.role_seq_detect,
                "seq_crop_obb": proj.role_seq_crop_obb,
            },
            "split_train": proj.split_train,
            "split_val": proj.split_val,
            "seed": proj.seed,
            "dedup": proj.dedup,
            "crop_pad_ratio": proj.crop_pad_ratio,
            "min_crop_size_px": proj.min_crop_size_px,
            "enforce_square": proj.enforce_square,
            "epochs": proj.epochs,
            "batch": proj.batch,
            "auto_batch": proj.auto_batch,
            "lr0": proj.lr0,
            "patience": proj.patience,
            "workers": proj.workers,
            "cache": proj.cache,
            "imgsz_obb_direct": proj.imgsz_obb_direct,
            "imgsz_seq_detect": proj.imgsz_seq_detect,
            "imgsz_seq_crop_obb": proj.imgsz_seq_crop_obb,
            "model_obb_direct": proj.model_obb_direct,
            "model_seq_detect": proj.model_seq_detect,
            "model_seq_crop_obb": proj.model_seq_crop_obb,
            "aug_enabled": proj.aug_enabled,
            "aug_fliplr": proj.aug_fliplr,
            "aug_flipud": proj.aug_flipud,
            "aug_degrees": proj.aug_degrees,
            "aug_mosaic": proj.aug_mosaic,
            "aug_mixup": proj.aug_mixup,
            "aug_hsv_h": proj.aug_hsv_h,
            "aug_hsv_s": proj.aug_hsv_s,
            "aug_hsv_v": proj.aug_hsv_v,
            "device": proj.device,
            "species": proj.species,
            "model_tag": proj.model_tag,
            "auto_import": proj.auto_import,
            "auto_select": proj.auto_select,
        }
        try:
            Path(path).write_text(json.dumps(data, indent=2), encoding="utf-8")
            QMessageBox.information(self, "Saved", f"Config saved to:\n{path}")
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

        roles = data.get("roles", {})
        if "obb_direct" in roles:
            self.chk_role_obb_direct.setChecked(bool(roles["obb_direct"]))
        if "seq_detect" in roles:
            self.chk_role_seq_detect.setChecked(bool(roles["seq_detect"]))
        if "seq_crop_obb" in roles:
            self.chk_role_seq_crop_obb.setChecked(bool(roles["seq_crop_obb"]))

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
            ("imgsz_seq_detect", self.spin_imgsz_seq_detect),
            ("imgsz_seq_crop_obb", self.spin_imgsz_seq_crop_obb),
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
            ("auto_import", self.chk_auto_import),
            ("auto_select", self.chk_auto_select),
        ]:
            if attr in data:
                widget.setChecked(bool(data[attr]))

        for attr, widget in [
            ("model_obb_direct", self.combo_model_obb_direct),
            ("model_seq_detect", self.combo_model_seq_detect),
            ("model_seq_crop_obb", self.combo_model_seq_crop_obb),
            ("device", self.combo_device),
        ]:
            if attr in data:
                widget.setCurrentText(str(data[attr]))

        if "species" in data:
            self.line_species.setText(str(data["species"]))
        if "model_tag" in data:
            self.line_model_tag.setText(str(data["model_tag"]))

        self._set_run_status(f"Loaded training preset from {path}")
        self._refresh_summary()
        QMessageBox.information(self, "Loaded", f"Config loaded from:\n{path}")
