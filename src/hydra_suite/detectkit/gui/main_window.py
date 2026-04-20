"""DetectKit main window — thin coordinator with VS Code-style toolbar."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from PySide6.QtCore import Qt
from PySide6.QtGui import QAction
from PySide6.QtWidgets import (
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMenu,
    QMessageBox,
    QSplitter,
    QStackedWidget,
    QStatusBar,
    QToolBar,
    QVBoxLayout,
    QWidget,
)

from hydra_suite.detectkit.config.schemas import DetectKitConfig
from hydra_suite.utils.file_dialogs import HydraFileDialog as QFileDialog  # noqa: F811

from .canvas import OBBCanvas
from .models import DetectKitProject
from .panels.dataset_panel import DatasetPanel
from .panels.tools_panel import ToolsPanel
from .prediction_preview import predict_preview_detections
from .project import (
    create_project,
    default_project_parent_dir,
    detectkit_model_path_is_previewable,
    detectkit_project_preview_model_paths,
    open_project,
    project_exists,
    save_project,
)
from .utils import find_label_for_image, parse_obb_label, source_class_id_map

logger = logging.getLogger(__name__)

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

        # Connect ToolsPanel signals
        self._dataset_panel.manage_sources_requested.connect(self._open_source_manager)
        self._tools_panel.overlay_settings_changed.connect(self._on_overlay_changed)
        self._tools_panel.prev_requested.connect(self._dataset_panel.navigate_prev)
        self._tools_panel.next_requested.connect(self._dataset_panel.navigate_next)
        self._tools_panel.train_requested.connect(self._open_training_dialog)
        self._tools_panel.evaluate_requested.connect(self._open_evaluation_dialog)
        self._tools_panel.history_requested.connect(self._open_history_dialog)

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

        act_evaluate = QAction("Evaluate", self)
        act_evaluate.triggered.connect(self._open_evaluation_dialog)
        tb.addAction(act_evaluate)

        act_history = QAction("History", self)
        act_history.triggered.connect(self._open_history_dialog)
        tb.addAction(act_history)

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

        self._recent_menu = QMenu("Recent Projects", self)
        file_menu.addMenu(self._recent_menu)
        self._refresh_recent_menu()

        file_menu.addSeparator()

        act_save = QAction("Save Project", self)
        act_save.triggered.connect(self._save_current_project)
        file_menu.addAction(act_save)

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

    def _load_project(self, proj: DetectKitProject) -> None:
        """Activate proj: wire panels, show toolbar, switch to workspace."""
        self._project = proj
        self._current_source_path = ""
        self._current_image_path = ""
        self._last_prediction_request = None

        preview_paths = detectkit_project_preview_model_paths(proj)
        if preview_paths and not detectkit_model_path_is_previewable(
            proj, proj.active_model_path
        ):
            proj.active_model_path = preview_paths[0]

        self._dataset_panel.set_project(proj, self)
        self._tools_panel.set_project(proj)
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

    def _open_evaluation_dialog(self) -> None:
        if self._project is None:
            return
        from .dialogs.evaluation_dialog import EvaluationDialog

        dlg = EvaluationDialog(self._project, parent=self)
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
        self._refresh_prediction_overlay()

    def _refresh_prediction_overlay(self, *, force: bool = False) -> None:
        if self._project is None or not self._current_image_path:
            self._canvas.clear_pred_detections()
            self._last_prediction_request = None
            return

        settings = self._tools_panel.get_overlay_settings()
        model_path = str(settings.active_model_path or "").strip()
        request = (
            self._current_image_path,
            model_path,
            round(float(settings.confidence_threshold), 4),
        )

        if not model_path:
            self._canvas.clear_pred_detections()
            self._last_prediction_request = None
            return

        if not detectkit_model_path_is_previewable(self._project, model_path):
            self._canvas.clear_pred_detections()
            self._last_prediction_request = None
            self.statusBar().showMessage(
                "Selected model does not support direct preview overlays.",
                4000,
            )
            return

        if not force and request == self._last_prediction_request:
            return

        try:
            detections = predict_preview_detections(
                self._current_image_path,
                model_path,
                device_preference=self._project.device or "auto",
                confidence_threshold=settings.confidence_threshold,
            )
        except Exception as exc:
            logger.warning("DetectKit preview inference failed", exc_info=True)
            self._canvas.clear_pred_detections()
            self._last_prediction_request = None
            self.statusBar().showMessage(
                f"Prediction preview failed: {exc}",
                5000,
            )
            return

        self._canvas.set_pred_detections(
            detections,
            class_names=self._project.class_names,
        )
        self._last_prediction_request = request

    # ------------------------------------------------------------------
    # Image display
    # ------------------------------------------------------------------

    def show_image(self, source_path: str, image_path: str) -> None:
        """Load an image and overlay GT labels."""
        self._current_source_path = str(source_path or "")
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

        self._refresh_prediction_overlay(force=True)
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
