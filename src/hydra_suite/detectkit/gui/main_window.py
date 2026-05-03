"""DetectKit main window — thin coordinator with VS Code-style toolbar."""

from __future__ import annotations

import logging
import re
import subprocess
import sys
from pathlib import Path
from typing import Optional

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QAction
from PySide6.QtWidgets import (
    QFileDialog,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QMainWindow,
    QMenu,
    QMessageBox,
    QProgressDialog,
    QSplitter,
    QStackedWidget,
    QStatusBar,
    QToolBar,
    QVBoxLayout,
    QWidget,
)

from hydra_suite.data.project_bundle import (
    export_project_bundle_archive,
    import_project_bundle_archive,
    load_project_bundle_archive_manifest,
)
from hydra_suite.detectkit.config.schemas import DetectKitConfig
from hydra_suite.utils.file_dialogs import HydraFileDialog as QFileDialog  # noqa: F811
from hydra_suite.widgets.busy import BusyTaskError, run_blocking_with_busy_dialog
from hydra_suite.widgets.workers import BaseWorker

from .canvas import OBBCanvas
from .models import DetectKitProject
from .panels.dataset_panel import DatasetPanel
from .panels.tools_panel import ToolsPanel
from .prediction_preview import load_torch_model, predict_preview_detections_for_image
from .project import (
    create_project,
    default_project_parent_dir,
    detectkit_model_path_is_previewable,
    detectkit_project_is_portable,
    detectkit_project_linked_reference_counts,
    detectkit_project_preview_model_paths,
    make_detectkit_project_portable,
    open_project,
    project_exists,
    save_project,
)
from .utils import (
    find_label_for_image,
    list_images_in_source,
    parse_obb_label,
    source_class_id_map,
)

logger = logging.getLogger(__name__)


class _DetectKitDatasetInferenceWorker(BaseWorker):
    """Run PyTorch OBB inference across every image in the active source."""

    success = Signal(dict)

    def __init__(
        self,
        image_paths: list[str],
        model_path: str,
        device_preference: str,
        confidence_threshold: float,
    ) -> None:
        super().__init__()
        self._image_paths = list(image_paths)
        self._model_path = str(model_path)
        self._device_preference = str(device_preference or "auto")
        self._confidence_threshold = float(confidence_threshold)
        self._cancel = False

    def cancel(self) -> None:
        self._cancel = True

    def execute(self) -> None:
        total = len(self._image_paths)
        if total == 0:
            self.success.emit(
                {
                    "per_image": {},
                    "image_count": 0,
                    "detection_count": 0,
                    "class_counts": {},
                    "mean_confidence": 0.0,
                }
            )
            return

        self.status.emit("Loading model…")
        model, device = load_torch_model(self._model_path, self._device_preference)

        per_image: dict[str, list[dict[str, object]]] = {}
        class_counts: dict[int, int] = {}
        confidence_sum = 0.0
        detection_count = 0
        confidence_threshold = self._confidence_threshold

        for index, image_path in enumerate(self._image_paths, start=1):
            if self._cancel:
                self.status.emit("Inference cancelled.")
                return
            self.status.emit(
                f"Running inference on image {index}/{total}: {Path(image_path).name}"
            )
            try:
                detections = predict_preview_detections_for_image(
                    model,
                    image_path,
                    device=device,
                    confidence_threshold=confidence_threshold,
                )
            except Exception:
                logger.warning(
                    "Dataset inference failed on %s", image_path, exc_info=True
                )
                detections = []
            per_image[image_path] = detections
            for det in detections:
                detection_count += 1
                class_id = int(det.get("class_id", 0))
                class_counts[class_id] = class_counts.get(class_id, 0) + 1
                confidence_sum += float(det.get("confidence", 0.0))
            self.progress.emit(int(index / max(1, total) * 100))

        mean_confidence = confidence_sum / detection_count if detection_count else 0.0
        self.success.emit(
            {
                "per_image": per_image,
                "image_count": total,
                "detection_count": detection_count,
                "class_counts": class_counts,
                "mean_confidence": mean_confidence,
            }
        )


class _DetectKitPortableWorker(BaseWorker):
    """Background worker that localizes linked DetectKit sources and artifacts."""

    success = Signal(dict)

    def __init__(self, project_dir: Path):
        super().__init__()
        self._project_dir = Path(project_dir)

    def execute(self) -> None:
        self.status.emit(
            "Copying linked sources and project artifacts into the bundle..."
        )
        project = open_project(self._project_dir)
        if project is None:
            raise RuntimeError(
                f"Could not reopen DetectKit project: {self._project_dir}"
            )
        before_counts = detectkit_project_linked_reference_counts(project)
        after_counts = make_detectkit_project_portable(project)
        self.progress.emit(100)
        self.success.emit({"before": before_counts, "after": after_counts})


_DATASET_PANEL_MIN_WIDTH = 360
_DATASET_PANEL_MAX_WIDTH = 420
_CANVAS_MIN_WIDTH = 480
_TOOLS_PANEL_WIDTH = 280
_WORKSPACE_MIN_HEIGHT = 760
_WORKSPACE_MIN_WIDTH = 1320

_DARK_STYLESHEET = """
QMainWindow {
    background-color: #1e1e1e;
}
QWidget {
    background-color: #1e1e1e;
    color: #e0e0e0;
    font-family: "SF Pro Text", "Helvetica Neue", "Segoe UI", Roboto, Arial, sans-serif;
    font-size: 11px;
}
QWidget[detectkitRole="panelShell"] {
    background-color: #252526;
    border: 1px solid #3e3e42;
    border-radius: 8px;
}
QWidget[detectkitRole="canvasShell"] {
    background-color: #202224;
    border: 1px solid #3e3e42;
    border-radius: 8px;
}
QWidget[detectkitRole="sectionCard"] {
    background-color: #252526;
    border: 1px solid #3e3e42;
    border-radius: 6px;
}
QLabel[detectkitRole="sectionTitle"] {
    color: #9cdcfe;
    font-size: 12px;
    font-weight: 600;
    letter-spacing: 0.4px;
}
QLabel[detectkitRole="sectionHint"] {
    color: #9f9f9f;
    font-size: 11px;
}
QLabel[detectkitRole="compactInfo"] {
    color: #cfcfcf;
    background-color: #252526;
    border: 1px solid #3e3e42;
    border-radius: 4px;
    padding: 6px 8px;
}
QMenuBar {
    background-color: #252526;
    color: #cccccc;
    border-bottom: 1px solid #3e3e42;
    padding: 4px;
}
QMenuBar::item {
    padding: 6px 12px;
    background-color: transparent;
}
QMenuBar::item:selected {
    background-color: #2a2d2e;
}
QMenu {
    background-color: #252526;
    color: #cccccc;
    border: 1px solid #3e3e42;
}
QMenu::item {
    padding: 8px 24px;
}
QMenu::item:selected {
    background-color: #094771;
}
QPushButton {
    background-color: #0e639c;
    color: #ffffff;
    border: none;
    border-radius: 4px;
    padding: 6px 14px;
    font-weight: 500;
}
QPushButton:hover {
    background-color: #1177bb;
}
QPushButton:pressed {
    background-color: #0d5a8f;
}
QPushButton:disabled {
    background-color: #3e3e42;
    color: #888888;
}
QPushButton[detectkitVariant="secondary"] {
    background-color: #3e3e42;
    color: #e0e0e0;
}
QPushButton[detectkitVariant="secondary"]:hover {
    background-color: #555558;
}
QPushButton[detectkitVariant="quiet"] {
    background-color: transparent;
    border: 1px solid #3e3e42;
    color: #cccccc;
}
QPushButton[detectkitVariant="quiet"]:hover {
    background-color: #2a2d2e;
    border-color: #0e639c;
}
QGroupBox {
    background-color: #252526;
    border: 1px solid #3e3e42;
    border-radius: 6px;
    margin-top: 10px;
    padding: 8px;
    font-weight: 600;
    color: #cccccc;
}
QGroupBox::title {
    subcontrol-origin: margin;
    subcontrol-position: top left;
    left: 8px;
    padding: 1px 6px;
    background-color: #1e1e1e;
    color: #9cdcfe;
    border-radius: 3px;
}
QListWidget,
QTextEdit,
QPlainTextEdit {
    background-color: #252526;
    alternate-background-color: #2d2d30;
    border: 1px solid #3e3e42;
    border-radius: 4px;
    padding: 4px;
    color: #e0e0e0;
}
QListWidget::item {
    padding: 8px 10px;
    border-radius: 4px;
    margin: 1px 0px;
}
QListWidget::item:selected {
    background-color: #094771;
    color: #ffffff;
}
QListWidget::item:hover:!selected {
    background-color: #2a2d2e;
}
QComboBox,
QLineEdit,
QSpinBox,
QDoubleSpinBox {
    background-color: #3c3c3c;
    border: 1px solid #3e3e42;
    color: #e0e0e0;
    padding: 4px 8px;
    border-radius: 4px;
    min-height: 22px;
}
QComboBox:hover,
QLineEdit:hover,
QSpinBox:hover,
QDoubleSpinBox:hover,
QTextEdit:hover,
QPlainTextEdit:hover,
QListWidget:hover {
    border-color: #0e639c;
}
QComboBox:focus,
QLineEdit:focus,
QSpinBox:focus,
QDoubleSpinBox:focus,
QTextEdit:focus,
QPlainTextEdit:focus,
QListWidget:focus {
    border-color: #007acc;
}
QComboBox::drop-down {
    subcontrol-origin: padding;
    subcontrol-position: top right;
    width: 20px;
    border-left: 1px solid #3e3e42;
    background-color: #4a4a4a;
    border-top-right-radius: 4px;
    border-bottom-right-radius: 4px;
}
QComboBox QAbstractItemView {
    background-color: #252526;
    border: 1px solid #3e3e42;
    selection-background-color: #094771;
    selection-color: #ffffff;
    color: #e0e0e0;
}
QCheckBox {
    color: #d6d6d6;
    spacing: 8px;
}
QCheckBox::indicator {
    width: 14px;
    height: 14px;
    border: 1px solid #3e3e42;
    background-color: #3c3c3c;
    border-radius: 3px;
}
QCheckBox::indicator:checked {
    background-color: #0e639c;
    border-color: #007acc;
}
QProgressBar {
    border: 1px solid #3e3e42;
    border-radius: 4px;
    text-align: center;
    background-color: #1f1f1f;
    color: #ffffff;
    min-height: 18px;
}
QProgressBar::chunk {
    background-color: #0e639c;
    border-radius: 3px;
}
QSlider::groove:horizontal {
    border: 1px solid #3e3e42;
    height: 4px;
    background: #2a2d2e;
    border-radius: 2px;
}
QSlider::handle:horizontal {
    background: #0e639c;
    border: 1px solid #0e639c;
    width: 14px;
    margin: -6px 0;
    border-radius: 7px;
}
QSplitter::handle {
    background-color: #3e3e42;
    width: 6px;
}
QToolBar {
    background-color: #252526;
    border-bottom: 1px solid #3e3e42;
    spacing: 8px;
    padding: 6px;
}
QToolBar QToolButton {
    background-color: transparent;
    border: none;
    border-radius: 4px;
    padding: 8px 12px;
    color: #cccccc;
}
QToolBar QToolButton:hover {
    background-color: #2a2d2e;
}
QToolBar QToolButton:pressed {
    background-color: #094771;
}
QStatusBar {
    background-color: #007acc;
    color: #ffffff;
    border-top: 1px solid #0098ff;
    font-weight: 500;
}
QScrollArea {
    border: none;
    background: transparent;
}
QScrollBar:vertical {
    background-color: #252526;
    width: 10px;
    border-radius: 5px;
}
QScrollBar::handle:vertical {
    background-color: #5a5a5a;
    border-radius: 5px;
    min-height: 24px;
}
QScrollBar::handle:vertical:hover {
    background-color: #007acc;
}
QScrollBar::add-line:vertical,
QScrollBar::sub-line:vertical,
QScrollBar::add-line:horizontal,
QScrollBar::sub-line:horizontal {
    width: 0px;
    height: 0px;
}
"""


class DetectKitMainWindow(QMainWindow):
    """DetectKit main application window — thin coordinator."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("DetectKit")
        self.setStyleSheet(_DARK_STYLESHEET)
        self.setMinimumSize(_WORKSPACE_MIN_WIDTH, _WORKSPACE_MIN_HEIGHT)

        self.config = DetectKitConfig()
        self._project: Optional[DetectKitProject] = None
        self._current_source_path = ""
        self._current_image_path = ""
        self._last_prediction_request: tuple[str, str, float] | None = None
        self._dataset_predictions: dict[str, list[dict[str, object]]] = {}
        self._dataset_prediction_signature: tuple[str, str, float] | None = None
        self._inference_worker: Optional[_DetectKitDatasetInferenceWorker] = None
        self._inference_progress_dialog: Optional[QProgressDialog] = None
        self._portable_worker = None
        self._portable_progress_dialog = None

        # Build workspace panels first (toolbar actions need them)
        self._dataset_panel = DatasetPanel()
        self._canvas = OBBCanvas()
        self._tools_panel = ToolsPanel()

        # Toolbar (hidden until project loaded)
        self._toolbar = self._build_toolbar()
        self.addToolBar(self._toolbar)
        self._toolbar.setVisible(False)

        # Central stacked widget: welcome (0) vs workspace (1)
        self._stack = QStackedWidget()
        self.setCentralWidget(self._stack)

        self._build_welcome_page()
        self._build_workspace_page()
        self._build_menu_bar()

        self.setStatusBar(QStatusBar())
        self.statusBar().showMessage("Ready")

        self._stack.setCurrentIndex(0)
        self.menuBar().hide()

        # Connect panel signals
        self._dataset_panel.manage_sources_requested.connect(self._open_source_manager)
        self._dataset_panel.train_requested.connect(self._open_training_dialog)
        self._dataset_panel.history_requested.connect(self._open_history_dialog)
        self._tools_panel.overlay_settings_changed.connect(self._on_overlay_changed)
        self._tools_panel.run_inference_requested.connect(self._run_inference_overlay)

    # ------------------------------------------------------------------
    # Toolbar
    # ------------------------------------------------------------------

    def _build_toolbar(self) -> QToolBar:
        tb = QToolBar("Main Toolbar")
        tb.setMovable(False)
        tb.setObjectName("detectkitToolbar")

        act_new = QAction("New", self)
        act_new.triggered.connect(self.new_project)
        tb.addAction(act_new)

        act_open = QAction("Open", self)
        act_open.triggered.connect(self.open_project_dialog)
        tb.addAction(act_open)

        act_save = QAction("Save", self)
        act_save.triggered.connect(self._save_current_project)
        tb.addAction(act_save)

        act_make_portable = QAction("Make Portable", self)
        act_make_portable.triggered.connect(self.make_project_portable)
        tb.addAction(act_make_portable)

        act_export_zip = QAction("Export Zip", self)
        act_export_zip.triggered.connect(self.export_project_zip)
        tb.addAction(act_export_zip)

        act_open_folder = QAction("Open Folder", self)
        act_open_folder.setStatusTip("Reveal project folder in Finder / file manager")
        act_open_folder.triggered.connect(self.open_project_folder)
        tb.addAction(act_open_folder)

        tb.addSeparator()

        act_sources = QAction("Sources", self)
        act_sources.triggered.connect(self._open_source_manager)
        tb.addAction(act_sources)

        tb.addSeparator()

        act_prev = QAction("Prev", self)
        act_prev.triggered.connect(self._dataset_panel.navigate_prev)
        tb.addAction(act_prev)

        act_next = QAction("Next", self)
        act_next.triggered.connect(self._dataset_panel.navigate_next)
        tb.addAction(act_next)

        tb.addSeparator()

        act_train = QAction("Train", self)
        act_train.triggered.connect(self._open_training_dialog)
        tb.addAction(act_train)

        act_run_inference = QAction("Run Inference", self)
        act_run_inference.triggered.connect(self._run_inference_overlay)
        tb.addAction(act_run_inference)

        act_history = QAction("History", self)
        act_history.triggered.connect(self._open_history_dialog)
        tb.addAction(act_history)

        al_action = QAction("Active Learning", self)
        al_action.triggered.connect(self._open_active_learning_dialog)
        tb.addAction(al_action)

        tb.addSeparator()

        act_export = QAction("Export", self)
        act_export.triggered.connect(self._export_stub)
        tb.addAction(act_export)

        return tb

    # ------------------------------------------------------------------
    # Welcome page
    # ------------------------------------------------------------------

    def _build_welcome_page(self) -> None:
        from hydra_suite.widgets import (
            ButtonDef,
            RecentItemsStore,
            WelcomeConfig,
            WelcomePage,
        )

        store = RecentItemsStore("detectkit")
        self._recents_store = store

        config = WelcomeConfig(
            logo_svg="detectkit.svg",
            tagline="OBB Detection Model Training & Dataset Curation",
            buttons=[
                ButtonDef(label="New Project", callback=self.new_project),
                ButtonDef(label="Open Project", callback=self.open_project_dialog),
                ButtonDef(
                    label="Open Project Zip",
                    callback=self.open_project_zip_dialog,
                ),
            ],
            recents_label="Recent Projects",
            recents_store=store,
            on_recent_clicked=self._open_recent_project,
        )
        self._welcome_page = WelcomePage(config)
        self._stack.addWidget(self._welcome_page)  # index 0

    def _open_recent_project(self, path: str) -> None:
        project_dir = Path(path)
        if project_dir.exists():
            proj = open_project(project_dir)
            if proj is not None:
                self._load_project(proj)
            else:
                QMessageBox.warning(
                    self, "Open Failed", f"Could not open project at:\n{path}"
                )
                self._remove_from_recents(path)
        else:
            QMessageBox.warning(self, "Not Found", f"Project not found:\n{path}")
            self._remove_from_recents(path)

    def _remove_from_recents(self, path: str) -> None:
        if hasattr(self, "_recents_store"):
            self._recents_store.remove(path)
            if hasattr(self, "_welcome_page"):
                self._welcome_page.refresh_recents()

    # ------------------------------------------------------------------
    # Workspace page
    # ------------------------------------------------------------------

    def _build_workspace_page(self) -> None:
        page = QWidget()
        page.setObjectName("detectkitWorkspace")
        layout = QHBoxLayout(page)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(12)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setHandleWidth(4)
        splitter.setChildrenCollapsible(False)
        self.splitter = splitter

        self._dataset_panel.setProperty("detectkitRole", "panelShell")
        self._dataset_panel.setMinimumWidth(_DATASET_PANEL_MIN_WIDTH)
        self._dataset_panel.setMaximumWidth(_DATASET_PANEL_MAX_WIDTH)
        splitter.addWidget(self._dataset_panel)  # index 0

        canvas_shell = QWidget()
        canvas_shell.setProperty("detectkitRole", "canvasShell")
        canvas_layout = QVBoxLayout(canvas_shell)
        canvas_layout.setContentsMargins(10, 10, 10, 10)
        canvas_layout.setSpacing(8)

        canvas_title = QLabel("Annotation Preview")
        canvas_title.setProperty("detectkitRole", "sectionTitle")
        canvas_layout.addWidget(canvas_title)

        canvas_hint = QLabel(
            "Review imported labels, compare model overlays, and inspect oriented boxes before training."
        )
        canvas_hint.setWordWrap(True)
        canvas_hint.setProperty("detectkitRole", "sectionHint")
        canvas_layout.addWidget(canvas_hint)

        self._canvas.setMinimumWidth(_CANVAS_MIN_WIDTH)
        canvas_layout.addWidget(self._canvas, 1)
        canvas_shell.setMinimumWidth(_CANVAS_MIN_WIDTH)
        splitter.addWidget(canvas_shell)  # index 1
        self._right_tabs = canvas_shell

        self._tools_panel.setProperty("detectkitRole", "panelShell")
        self._tools_panel.setFixedWidth(_TOOLS_PANEL_WIDTH)
        splitter.addWidget(self._tools_panel)  # index 2

        splitter.setCollapsible(0, False)
        splitter.setCollapsible(1, False)
        splitter.setCollapsible(2, False)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setStretchFactor(2, 0)
        splitter.setSizes(
            [_DATASET_PANEL_MIN_WIDTH, _CANVAS_MIN_WIDTH + 220, _TOOLS_PANEL_WIDTH]
        )

        layout.addWidget(splitter, 1)

        self._stack.addWidget(page)  # index 1

    # ------------------------------------------------------------------
    # Menu bar
    # ------------------------------------------------------------------

    def _build_menu_bar(self) -> None:
        menu_bar = self.menuBar()
        file_menu = menu_bar.addMenu("&File")

        act_new = QAction("New Project...", self)
        act_new.triggered.connect(self.new_project)
        file_menu.addAction(act_new)

        act_open = QAction("Open Project...", self)
        act_open.triggered.connect(self.open_project_dialog)
        file_menu.addAction(act_open)

        act_open_zip = QAction("Open Project Zip...", self)
        act_open_zip.triggered.connect(self.open_project_zip_dialog)
        file_menu.addAction(act_open_zip)

        self._recent_menu = QMenu("Recent Projects", self)
        file_menu.addMenu(self._recent_menu)
        self._refresh_recent_menu()

        file_menu.addSeparator()

        act_save = QAction("Save Project", self)
        act_save.triggered.connect(self._save_current_project)
        file_menu.addAction(act_save)

        act_make_portable = QAction("Make Project Portable", self)
        act_make_portable.triggered.connect(self.make_project_portable)
        file_menu.addAction(act_make_portable)

        act_export_zip = QAction("Export Project Zip...", self)
        act_export_zip.triggered.connect(self.export_project_zip)
        file_menu.addAction(act_export_zip)

        act_open_folder = QAction("Open Project Folder", self)
        act_open_folder.triggered.connect(self.open_project_folder)
        file_menu.addAction(act_open_folder)

        file_menu.addSeparator()

        act_quit = QAction("Quit", self)
        act_quit.triggered.connect(self.close)
        file_menu.addAction(act_quit)

    def _refresh_recent_menu(self) -> None:
        self._recent_menu.clear()
        if hasattr(self, "_recents_store"):
            for p in self._recents_store.load():
                action = self._recent_menu.addAction(p)
                action.setData(p)
                action.triggered.connect(self._on_recent_menu_action)

    def _on_recent_menu_action(self) -> None:
        action = self.sender()
        if action is None:
            return
        path_str = action.data()
        if path_str:
            proj = open_project(Path(path_str))
            if proj is not None:
                self._load_project(proj)
            else:
                QMessageBox.warning(
                    self, "Open Failed", f"Could not open project at:\n{path_str}"
                )

    # ------------------------------------------------------------------
    # Project lifecycle
    # ------------------------------------------------------------------

    def new_project(self) -> None:
        from .dialogs import NewProjectDialog

        dialog = NewProjectDialog(self)
        result = dialog.exec()
        if result != dialog.DialogCode.Accepted:
            return

        project_info = dialog.get_project_info()
        proj_dir = Path(project_info["path"]).expanduser()

        if project_exists(proj_dir):
            ans = QMessageBox.question(
                self,
                "Project Exists",
                f"A project already exists in:\n{proj_dir}\n\nOpen it instead?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            )
            if ans == QMessageBox.StandardButton.Yes:
                proj = open_project(proj_dir)
                if proj is not None:
                    self._load_project(proj)
            return

        proj = create_project(
            proj_dir,
            project_info["class_name"],
            class_names=list(project_info.get("class_names", [])),
        )
        self._load_project(proj)

    @staticmethod
    def _next_available_project_dir(parent_dir: Path, base_name: str) -> Path:
        cleaned = re.sub(r"[\\/]+", "_", str(base_name or "").strip())
        cleaned = cleaned.strip() or "DetectKit Project"
        candidate = parent_dir / cleaned
        counter = 1
        while candidate.exists():
            candidate = parent_dir / f"{cleaned}_{counter}"
            counter += 1
        return candidate

    @staticmethod
    def _sanitize_project_folder_name(name: str, *, fallback: str) -> str:
        cleaned = re.sub(r"[\\/]+", "_", str(name or "").strip())
        return cleaned.strip() or fallback

    def _choose_project_zip_destination(
        self,
        archive_path: str | Path,
        suggested_name: str,
    ) -> Path | None:
        parent_dir = QFileDialog.getExistingDirectory(
            self,
            "Choose DetectKit Project Extraction Folder",
            str(default_project_parent_dir()),
            QFileDialog.ShowDirsOnly,
        )
        if not parent_dir:
            return None

        folder_name, accepted = QInputDialog.getText(
            self,
            "Project Folder Name",
            "Extract project into folder:",
            text=self._sanitize_project_folder_name(
                suggested_name,
                fallback=Path(archive_path).stem,
            ),
        )
        if not accepted:
            return None

        cleaned_name = self._sanitize_project_folder_name(
            folder_name,
            fallback=Path(archive_path).stem,
        )
        destination_dir = Path(parent_dir) / cleaned_name
        if destination_dir.exists() and any(destination_dir.iterdir()):
            QMessageBox.warning(
                self,
                "Open Project Zip",
                f"Destination folder is not empty:\n{destination_dir}",
            )
            return None
        return destination_dir

    def open_project_dialog(self) -> None:
        directory = QFileDialog.getExistingDirectory(
            self, "Open DetectKit Project", str(default_project_parent_dir())
        )
        if not directory:
            return
        proj = open_project(Path(directory))
        if proj is not None:
            self._load_project(proj)
        else:
            QMessageBox.warning(
                self, "Open Failed", f"No DetectKit project found in:\n{directory}"
            )

    def open_project_zip_dialog(self) -> None:
        archive_path, _ = QFileDialog.getOpenFileName(
            self,
            "Open DetectKit Project Zip",
            str(default_project_parent_dir()),
            "Zip Files (*.zip)",
        )
        if not archive_path:
            return

        try:
            manifest = load_project_bundle_archive_manifest(archive_path, strict=True)
            destination_dir = self._choose_project_zip_destination(
                archive_path,
                manifest.display_name or Path(archive_path).stem,
            )
            if destination_dir is None:
                return
        except Exception as exc:
            QMessageBox.warning(self, "Open Project Zip", str(exc))
            return

        def _extract(set_status, _set_progress):
            set_status("Extracting project archive…")
            return import_project_bundle_archive(
                archive_path,
                destination_dir,
                expected_kit="detectkit",
            )

        try:
            imported_dir = run_blocking_with_busy_dialog(
                self,
                _extract,
                title="Open Project Zip",
                message="Extracting project archive…",
            )
        except BusyTaskError as exc:
            QMessageBox.warning(self, "Open Project Zip", str(exc))
            return

        proj = open_project(imported_dir)
        if proj is not None:
            self._load_project(proj)
        else:
            QMessageBox.warning(
                self,
                "Open Project Zip",
                f"Imported project could not be opened:\n{imported_dir}",
            )

    def _load_project(self, proj: DetectKitProject) -> None:
        """Activate proj: wire panels, show toolbar, switch to workspace."""
        self._project = proj
        self._current_source_path = ""
        self._current_image_path = ""
        self._last_prediction_request = None

        linked_counts = detectkit_project_linked_reference_counts(proj)
        portability_status = (
            "Portable" if detectkit_project_is_portable(proj) else "Linked"
        )

        preview_paths = detectkit_project_preview_model_paths(proj)
        if preview_paths and not detectkit_model_path_is_previewable(
            proj, proj.active_model_path
        ):
            proj.active_model_path = preview_paths[0]

        self._dataset_panel.set_project(proj, self)
        self._dataset_panel.set_portability_status(portability_status, linked_counts)
        self._tools_panel.set_project(proj)
        self._tools_panel.set_portability_status(portability_status, linked_counts)
        self._tools_panel.refresh_model_selector(preview_paths)

        self._toolbar.setVisible(True)
        self._stack.setCurrentIndex(1)
        self.menuBar().show()

        if hasattr(self, "_recents_store"):
            self._recents_store.add(str(proj.project_dir))
            if hasattr(self, "_welcome_page"):
                self._welcome_page.refresh_recents()
        self._refresh_recent_menu()

        self.statusBar().showMessage(f"Loaded project: {proj.project_dir}", 5000)

    def _save_current_project(self) -> None:
        if self._project is None:
            return
        self._dataset_panel.collect_state(self._project)
        save_project(self._project)
        self.statusBar().showMessage("Project saved.", 3000)

    def make_project_portable(self, *, interactive: bool = True) -> bool:
        if self._project is None:
            if interactive:
                QMessageBox.information(
                    self,
                    "Make Project Portable",
                    "Open a DetectKit project before making it portable.",
                )
            return False

        before_counts = detectkit_project_linked_reference_counts(self._project)
        if not any(before_counts.values()):
            if interactive:
                QMessageBox.information(
                    self,
                    "Make Project Portable",
                    "This DetectKit project is already portable.",
                )
            self.statusBar().showMessage("Project already portable.", 3000)
            return True

        if interactive:
            self._save_current_project()
            progress = QProgressDialog(
                "Copying linked sources and project artifacts into the bundle...",
                None,
                0,
                0,
                self,
            )
            progress.setWindowTitle("Make Project Portable")
            progress.setCancelButton(None)
            progress.setMinimumDuration(0)
            progress.setWindowModality(Qt.ApplicationModal)
            progress.setAttribute(Qt.WA_DeleteOnClose, True)
            progress.show()

            worker = _DetectKitPortableWorker(self._project.project_dir)
            worker.status.connect(progress.setLabelText)
            worker.error.connect(
                lambda msg: QMessageBox.warning(self, "Make Project Portable", msg)
            )

            def _finish_portable_run() -> None:
                progress.close()
                self._portable_worker = None
                self._portable_progress_dialog = None

            def _handle_portable_success(result: dict) -> None:
                reloaded_project = open_project(self._project.project_dir)
                if reloaded_project is not None:
                    self._load_project(reloaded_project)
                before = result.get("before", {}) if isinstance(result, dict) else {}
                after = result.get("after", {}) if isinstance(result, dict) else {}
                if any(int(value or 0) for value in after.values()):
                    QMessageBox.warning(
                        self,
                        "Make Project Portable",
                        "Some linked sources or artifact references remain outside the project bundle.",
                    )
                    return
                copied_sources = max(0, int(before.get("sources", 0)))
                copied_artifacts = max(0, int(before.get("artifacts", 0)))
                summary = f"Localized {copied_sources:,} source(s) and {copied_artifacts:,} artifact reference(s) into the project bundle."
                self.statusBar().showMessage(summary, 5000)
                QMessageBox.information(self, "Make Project Portable", summary)

            worker.success.connect(_handle_portable_success)
            worker.finished.connect(_finish_portable_run)
            self._portable_worker = worker
            self._portable_progress_dialog = progress
            worker.start()
            return True

        try:
            self._save_current_project()
        except Exception as exc:
            QMessageBox.warning(self, "Make Project Portable", str(exc))
            return False

        project = self._project

        def _materialize(set_status, _set_progress):
            set_status("Copying linked sources and artifacts into the bundle…")
            return make_detectkit_project_portable(project)

        try:
            after_counts = run_blocking_with_busy_dialog(
                self,
                _materialize,
                title="Make Project Portable",
                message="Copying linked sources and artifacts into the bundle…",
            )
        except BusyTaskError as exc:
            QMessageBox.warning(self, "Make Project Portable", str(exc))
            return False

        reloaded_project = open_project(self._project.project_dir)
        if reloaded_project is not None:
            self._load_project(reloaded_project)
        else:
            self._load_project(self._project)

        if any(after_counts.values()):
            QMessageBox.warning(
                self,
                "Make Project Portable",
                "Some linked sources or artifact references remain outside the project bundle.",
            )
            return False

        copied_sources = max(0, int(before_counts.get("sources", 0)))
        copied_artifacts = max(0, int(before_counts.get("artifacts", 0)))
        summary = f"Localized {copied_sources:,} source(s) and {copied_artifacts:,} artifact reference(s) into the project bundle."
        self.statusBar().showMessage(summary, 5000)
        if interactive:
            QMessageBox.information(self, "Make Project Portable", summary)
        return True

    def export_project_zip(self) -> None:
        if self._project is None:
            QMessageBox.information(
                self,
                "Export Project Zip",
                "Open a DetectKit project before exporting a portable zip.",
            )
            return

        default_archive = (
            self._project.project_dir.parent / f"{self._project.project_dir.name}.zip"
        )
        archive_path, _ = QFileDialog.getSaveFileName(
            self,
            "Export DetectKit Project Zip",
            str(default_archive),
            "Zip Files (*.zip)",
        )
        if not archive_path:
            return

        try:
            if not self.make_project_portable(interactive=False):
                return
            self._save_current_project()
        except Exception as exc:
            QMessageBox.warning(self, "Export Project Zip", str(exc))
            return

        project_dir = self._project.project_dir

        def _archive(set_status, _set_progress):
            set_status("Writing zip archive…")
            return export_project_bundle_archive(project_dir, archive_path)

        try:
            written_path = run_blocking_with_busy_dialog(
                self,
                _archive,
                title="Export Project Zip",
                message="Writing zip archive…",
            )
        except BusyTaskError as exc:
            QMessageBox.warning(self, "Export Project Zip", str(exc))
            return

        self.statusBar().showMessage(f"Exported project zip: {written_path}", 5000)

    # ------------------------------------------------------------------
    # Dialog launchers
    # ------------------------------------------------------------------

    def _open_source_manager(self) -> None:
        if self._project is None:
            return
        from .dialogs.source_manager import SourceManagerDialog

        dlg = SourceManagerDialog(self._project, parent=self)
        dlg.exec()
        self._dataset_panel.refresh_sources(self._project)
        self._tools_panel.refresh_overview()

    def _open_training_dialog(self) -> None:
        if self._project is None:
            return
        from .dialogs.training_dialog import TrainingDialog

        dlg = TrainingDialog(self._project, parent=self)
        dlg.training_completed.connect(self._on_training_completed)
        dlg.exec()

    def _open_history_dialog(self) -> None:
        if self._project is None:
            return
        from .dialogs.history_dialog import HistoryDialog

        dlg = HistoryDialog(self._project, parent=self)
        result = dlg.exec()
        if result == dlg.DialogCode.Accepted:
            self._tools_panel.refresh_model_selector(
                detectkit_project_preview_model_paths(self._project)
            )
            self._refresh_prediction_overlay(force=True)

    def _open_active_learning_dialog(self) -> None:
        if self._project is None:
            return
        from .dialogs.active_learning import ActiveLearningDialog

        dlg = ActiveLearningDialog(project=self._project, parent=self)
        dlg.set_run_handler(lambda: self._start_al_round(dlg))
        dlg.open()

    def _start_al_round(self, dlg) -> None:
        from hydra_suite.detectkit.jobs.al_worker import ALRequest, ALWorker

        try:
            detector_fn = self._load_active_detector_fn()
            request = ALRequest(
                input_kind=(
                    "video"
                    if dlg.rb_video.isChecked()
                    else "folder" if dlg.rb_folder.isChecked() else "project"
                ),
                input_path=dlg.input_path_edit.text(),
                project=self._project,
                budget=dlg.budget_spin.value(),
                preset=dlg.preset_combo.currentText(),
                expected_count=dlg.expected_count_spin.value(),
                detector_fn=detector_fn,
            )
        except NotImplementedError as exc:
            dlg.status_label.setText(f"Error: {exc}")
            return
        except Exception as exc:
            dlg.status_label.setText(f"Error: {exc}")
            return

        worker = ALWorker(request)
        worker.progress.connect(dlg.progress.setValue)
        worker.status.connect(dlg.status_label.setText)
        worker.result_ready.connect(
            lambda path, n, _ids: dlg.status_label.setText(
                f"Imported {n} frames -> {path}"
            )
        )
        worker.error.connect(lambda msg: dlg.status_label.setText(f"Error: {msg}"))
        worker.start()
        self._al_worker = worker

    def _cancel_al_round(self) -> None:
        worker = getattr(self, "_al_worker", None)
        if worker is not None:
            worker.requestInterruption()

    def _load_active_detector_fn(self):
        """Return a detector_fn(frame, conf, iou) -> list[(cx,cy,w,h,theta,conf)].

        Loads the project's active model via the same torch loader used by
        `_run_inference_overlay` and adapts the result to the OBB-tuple format
        required by `hydra_suite.data.al`.
        """
        if self._project is None:
            raise RuntimeError("No project loaded.")
        model_path = str(self._project.active_model_path or "").strip()
        if not model_path:
            raise RuntimeError(
                "No active model selected. Set one via Run History or after a training run."
            )
        if not detectkit_model_path_is_previewable(self._project, model_path):
            raise RuntimeError(
                "Selected model does not support direct inference. "
                "Train or load a YOLO OBB model and try again."
            )

        from .prediction_preview import load_torch_model, predict_obb_for_frame

        model, device = load_torch_model(model_path, self._project.device or "auto")

        def _detector_fn(frame, conf, iou):
            return predict_obb_for_frame(
                model, frame, device=device, conf=conf, iou=iou
            )

        return _detector_fn

    def _on_training_completed(self, results: list) -> None:
        if self._project is None:
            return
        self._tools_panel.refresh_model_selector(
            detectkit_project_preview_model_paths(self._project)
        )
        self._refresh_prediction_overlay(force=True)
        self._save_current_project()

    def _export_stub(self) -> None:
        self._open_history_dialog()

    def _on_overlay_changed(self) -> None:
        settings = self._tools_panel.get_overlay_settings()
        if self._project is not None:
            self._project.active_model_path = settings.active_model_path
        self._canvas.set_overlay_visibility(settings.show_gt, settings.show_pred)
        self._canvas.set_class_filter(settings.visible_class_ids)

        signature = self._dataset_signature(settings)

        if signature is None:
            self._canvas.clear_pred_detections()
            self._last_prediction_request = None
            return

        if self._project is not None and not detectkit_model_path_is_previewable(
            self._project,
            signature[1],
        ):
            self._canvas.clear_pred_detections()
            self._last_prediction_request = None
            self.statusBar().showMessage(
                "Selected model does not support direct preview overlays.",
                4000,
            )
            return

        if signature == self._dataset_prediction_signature:
            self._refresh_prediction_overlay(force=True)
        else:
            self._canvas.clear_pred_detections()
            self._last_prediction_request = None
            self.statusBar().showMessage(
                "Inference settings changed. Click Run Inference to refresh overlay predictions.",
                4000,
            )

    def _dataset_signature(self, settings) -> tuple[str, str, float] | None:
        if self._project is None or not self._current_source_path:
            return None
        model_path = str(settings.active_model_path or "").strip()
        if not model_path:
            return None
        return (
            self._current_source_path,
            model_path,
            round(float(settings.confidence_threshold), 4),
        )

    def _run_inference_overlay(self) -> None:
        """Run dataset-wide PyTorch inference for the active source."""
        if self._project is None or not self._current_source_path:
            QMessageBox.information(
                self,
                "Run Inference",
                "Open a project and select a source before running inference.",
            )
            return
        if self._inference_worker is not None:
            QMessageBox.information(
                self,
                "Run Inference",
                "An inference run is already in progress.",
            )
            return

        settings = self._tools_panel.get_overlay_settings()
        signature = self._dataset_signature(settings)
        if signature is None:
            QMessageBox.information(
                self,
                "Run Inference",
                "Select a model with a populated source before running inference.",
            )
            return

        model_path = signature[1]
        if not detectkit_model_path_is_previewable(self._project, model_path):
            QMessageBox.information(
                self,
                "Run Inference",
                "Selected model does not support direct preview inference.",
            )
            return

        image_paths = [str(p) for p in list_images_in_source(self._current_source_path)]
        if not image_paths:
            QMessageBox.information(
                self,
                "Run Inference",
                "No images found in the active source.",
            )
            return

        progress = QProgressDialog(
            f"Running inference on {len(image_paths)} image(s)…",
            "Cancel",
            0,
            100,
            self,
        )
        progress.setWindowTitle("Run Inference")
        progress.setMinimumDuration(0)
        progress.setWindowModality(Qt.ApplicationModal)
        progress.setAttribute(Qt.WA_DeleteOnClose, True)
        progress.setValue(0)

        worker = _DetectKitDatasetInferenceWorker(
            image_paths,
            model_path,
            self._project.device or "auto",
            settings.confidence_threshold,
        )
        worker.progress.connect(progress.setValue)
        worker.status.connect(progress.setLabelText)
        progress.canceled.connect(worker.cancel)
        worker.error.connect(
            lambda msg: QMessageBox.warning(self, "Run Inference", msg)
        )

        def _finish() -> None:
            progress.close()
            self._inference_worker = None
            self._inference_progress_dialog = None

        def _handle_success(result: dict) -> None:
            self._dataset_predictions = dict(result.get("per_image", {}))
            self._dataset_prediction_signature = signature
            self._tools_panel.update_inference_stats(
                result, class_names=self._project.class_names
            )
            self._refresh_prediction_overlay(force=True)
            self.statusBar().showMessage(
                f"Inference complete: {result.get('detection_count', 0):,} "
                f"detection(s) across {result.get('image_count', 0):,} image(s).",
                5000,
            )

        worker.success.connect(_handle_success)
        worker.finished.connect(_finish)
        self._inference_worker = worker
        self._inference_progress_dialog = progress
        progress.show()
        worker.start()

    def _refresh_prediction_overlay(self, *, force: bool = False) -> None:
        if self._project is None or not self._current_image_path:
            self._canvas.clear_pred_detections()
            self._last_prediction_request = None
            return

        settings = self._tools_panel.get_overlay_settings()
        signature = self._dataset_signature(settings)

        if (
            signature is None
            or signature != self._dataset_prediction_signature
            or self._current_image_path not in self._dataset_predictions
        ):
            self._canvas.clear_pred_detections()
            self._last_prediction_request = None
            return

        detections = self._dataset_predictions.get(self._current_image_path, [])
        self._canvas.set_pred_detections(
            detections,
            class_names=self._project.class_names,
        )
        self._last_prediction_request = signature

    # ------------------------------------------------------------------
    # Image display
    # ------------------------------------------------------------------

    def open_project_folder(self) -> None:
        """Reveal the project folder in the system file manager."""
        if self._project is None:
            self.statusBar().showMessage("No project loaded.", 2000)
            return
        folder = Path(self._project.project_dir)
        try:
            if sys.platform == "darwin":
                subprocess.Popen(["open", str(folder)])
            elif sys.platform.startswith("linux"):
                subprocess.Popen(["xdg-open", str(folder)])
            else:
                subprocess.Popen(["explorer", str(folder)])
        except Exception as exc:
            QMessageBox.warning(self, "Open Folder", f"Could not open folder:\n{exc}")

    def on_images_deleted(self, deleted_paths: list[str]) -> None:
        """Drop cached predictions for *deleted_paths* and clear canvas if needed."""
        deleted = set(deleted_paths or [])
        if not deleted:
            return
        for path in deleted:
            self._dataset_predictions.pop(path, None)
        if self._current_image_path in deleted:
            self._current_image_path = ""
            self._last_prediction_request = None
            self._canvas.clear_all()

    def show_image(self, source_path: str, image_path: str) -> None:
        """Load an image and overlay GT labels."""
        new_source = str(source_path or "")
        if new_source != self._current_source_path:
            self._dataset_predictions = {}
            self._dataset_prediction_signature = None
        self._current_source_path = new_source
        self._current_image_path = str(image_path or "")
        self._last_prediction_request = None
        self._canvas.clear_gt_detections()
        self._canvas.clear_pred_detections()
        ok = self._canvas.load_image(image_path)
        if not ok:
            return

        label_path = find_label_for_image(Path(image_path), source_path)
        if label_path is not None:
            import cv2

            img = cv2.imread(image_path)
            if img is not None:
                h, w = img.shape[:2]
                class_names = self._project.class_names if self._project else ["object"]
                class_id_map = None
                if self._project is not None:
                    try:
                        class_id_map = source_class_id_map(
                            source_path, self._project.class_names
                        )
                    except Exception:
                        class_id_map = {}
                        logger.warning(
                            "Skipping incompatible source labels for preview: %s",
                            source_path,
                            exc_info=True,
                        )
                dets = parse_obb_label(label_path, w, h, class_id_map=class_id_map)
                self._canvas.set_gt_detections(dets, class_names=class_names)

        # If we already have predictions for this image from a previous Run Inference, restore them.
        self._refresh_prediction_overlay(force=True)
        if (
            self._last_prediction_request is None
            and self._project is not None
            and str(self._project.active_model_path or "").strip()
        ):
            self.statusBar().showMessage(
                "Image loaded. Click Run Inference to refresh overlay predictions.",
                3000,
            )
        self._canvas.fit_in_view()

    # ------------------------------------------------------------------
    # Public accessors
    # ------------------------------------------------------------------

    def project(self) -> Optional[DetectKitProject]:
        return self._project

    def canvas(self) -> OBBCanvas:
        return self._canvas

    # ------------------------------------------------------------------
    # Events
    # ------------------------------------------------------------------

    def closeEvent(self, event) -> None:  # noqa: N802
        self._save_current_project()
        super().closeEvent(event)


# Backward-compat alias
MainWindow = DetectKitMainWindow
