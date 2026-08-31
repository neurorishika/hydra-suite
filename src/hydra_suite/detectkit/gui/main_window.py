"""DetectKit main window — thin coordinator with VS Code-style toolbar."""

from __future__ import annotations

import logging
import re
import subprocess
import sys
from pathlib import Path
from threading import Event
from typing import Optional

from PySide6.QtCore import Qt, QTimer, Signal
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
    QPushButton,
    QSplitter,
    QStackedWidget,
    QStatusBar,
    QToolBar,
    QVBoxLayout,
    QWidget,
)

from hydra_suite.data.al.merge import MergeMode
from hydra_suite.data.project_bundle import (
    export_project_bundle_archive,
    import_project_bundle_archive,
    load_project_bundle_archive_manifest,
)
from hydra_suite.detectkit.config.schemas import DetectKitConfig
from hydra_suite.utils.file_dialogs import HydraFileDialog as QFileDialog  # noqa: F811
from hydra_suite.widgets.busy import BusyTaskError, run_blocking_with_busy_dialog
from hydra_suite.widgets.workers import BaseWorker

from ..jobs.staged_review import (
    accept_all,
    accept_frame,
    finish_review,
    is_complete,
    read_decisions,
    reject_all,
    reject_frame,
    revert_review,
    review_key_for_image,
    review_progress,
    staged_frames,
)
from . import escalation_actions
from .canvas import OBBCanvas
from .models import (
    INFERENCE_CONFIDENCE_FLOOR,
    DetectKitProject,
    InferenceRunSettings,
    SliceTrainingSettings,
)
from .overlays import PROVIDERS, FrameContext
from .panels.dataset_panel import DatasetPanel
from .panels.review_bar import ReviewBar
from .panels.tools_panel import ToolsPanel
from .prediction_preview import (
    dicts_from_obb_result,
    load_torch_model,
    predict_obb_for_frame_sequential,
    predict_preview_detections_for_image,
    predict_sliced_obb_result,
    preview_object_tile_fraction,
)
from .project import (
    create_project,
    default_project_parent_dir,
    detectkit_model_path_is_previewable,
    detectkit_project_is_portable,
    detectkit_project_linked_reference_counts,
    detectkit_project_preview_model_paths,
    detectkit_resolve_inference_models,
    make_detectkit_project_portable,
    open_project,
    project_exists,
    save_project,
)
from .utils import list_images_in_source

logger = logging.getLogger(__name__)


def _filter_detections_by_confidence(
    detections: list[dict[str, object]], confidence_threshold: float
) -> list[dict[str, object]]:
    """Return cached detections visible at the current display threshold."""
    threshold = float(confidence_threshold)
    return [
        detection
        for detection in detections
        if float(detection.get("confidence", 0.0)) >= threshold
    ]


class _DetectKitDatasetInferenceWorker(BaseWorker):
    """Run PyTorch OBB inference across every image in the active source."""

    success = Signal(dict)

    def __init__(
        self,
        image_paths: list[str],
        model_path: str,
        device_preference: str,
        confidence_threshold: float,
        inference_kind: str = "obb_direct",
        secondary_model_path: "str | None" = None,
        crop_pad_ratio: float = 0.15,
        stage2_image_size: int = 160,
        slice_settings: "SliceTrainingSettings | None" = None,
        imgsz_obb_direct: int = 640,
    ) -> None:
        super().__init__()
        self._image_paths = list(image_paths)
        self._model_path = str(model_path)
        self._device_preference = str(device_preference or "auto")
        self._confidence_threshold = float(confidence_threshold)
        self._inference_kind = str(inference_kind or "obb_direct")
        self._secondary_model_path = (
            str(secondary_model_path).strip() if secondary_model_path else None
        )
        self._crop_pad_ratio = float(crop_pad_ratio)
        self._stage2_image_size = int(stage2_image_size)
        self._slice_settings = slice_settings
        self._imgsz_obb_direct = int(imgsz_obb_direct)
        self._cancel_event = Event()

    def cancel(self) -> None:
        """Request cooperative cancellation at the next safe inference boundary."""
        self._cancel_event.set()

    def is_cancelled(self) -> bool:
        """Return whether cancellation has been requested for this run."""
        return self._cancel_event.is_set()

    def _stop_if_cancelled(self) -> bool:
        if not self.is_cancelled():
            return False
        self.status.emit("Inference cancelled.")
        return True

    def execute(self) -> None:
        if self._stop_if_cancelled():
            return
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

        per_image: dict[str, list[dict[str, object]]] = {}
        class_counts: dict[int, int] = {}
        confidence_sum = 0.0
        detection_count = 0
        confidence_threshold = self._confidence_threshold

        if self._secondary_model_path and self._inference_kind == "sequential_segment":
            # The shared runner already knows how to execute a detect ->
            # crop-segment pipeline and preserves the native polygons that its
            # stage-2 segmentation model produces.
            from hydra_suite.core.inference.runner import InferenceRunner
            from hydra_suite.data.al.inference_adapter import build_obb_config_for_al

            self.status.emit("Loading sequential segment models…")
            config = build_obb_config_for_al(
                self._inference_kind,
                self._model_path,
                self._secondary_model_path,
                crop_pad_ratio=self._crop_pad_ratio,
                confidence_threshold=confidence_threshold,
                iou_threshold=0.7,
                max_targets=300,
                stage2_image_size=self._stage2_image_size,
            )
            runner = InferenceRunner(config)
            if self._stop_if_cancelled():
                return

            import cv2

            for index, image_path in enumerate(self._image_paths, start=1):
                if self._stop_if_cancelled():
                    return
                self.status.emit(
                    f"Running inference on image {index}/{total}: {Path(image_path).name}"
                )
                try:
                    frame = cv2.imread(str(image_path))
                    if frame is None:
                        raise RuntimeError(f"Could not read image: {image_path}")
                    obb = runner.detect_batch_raw([frame], [index - 1])[0]
                    detections = dicts_from_obb_result(obb)
                except Exception:
                    logger.warning(
                        "Sequential segment dataset inference failed on %s",
                        image_path,
                        exc_info=True,
                    )
                    detections = []
                if self._stop_if_cancelled():
                    return
                per_image[image_path] = detections
                for det in detections:
                    detection_count += 1
                    class_id = int(det.get("class_id", 0))
                    class_counts[class_id] = class_counts.get(class_id, 0) + 1
                    confidence_sum += float(det.get("confidence", 0.0))
                self.progress.emit(int(index / max(1, total) * 100))
        elif self._secondary_model_path:
            self.status.emit("Loading sequential models…")
            detect_model, detect_device = load_torch_model(
                self._model_path, self._device_preference
            )
            obb_model, obb_device = load_torch_model(
                self._secondary_model_path, self._device_preference
            )
            if self._stop_if_cancelled():
                return

            import cv2

            for index, image_path in enumerate(self._image_paths, start=1):
                if self._stop_if_cancelled():
                    return
                self.status.emit(
                    f"Running inference on image {index}/{total}: {Path(image_path).name}"
                )
                try:
                    frame = cv2.imread(str(image_path))
                    if frame is None:
                        raise RuntimeError(f"Could not read image: {image_path}")
                    tuples = predict_obb_for_frame_sequential(
                        detect_model,
                        obb_model,
                        frame,
                        detect_device=detect_device,
                        obb_device=obb_device,
                        conf=confidence_threshold,
                        iou=0.7,
                        should_stop=self.is_cancelled,
                    )
                    import numpy as np

                    detections: list[dict[str, object]] = []
                    for cx, cy, w, h, theta, conf in tuples:
                        cos_t = float(np.cos(theta))
                        sin_t = float(np.sin(theta))
                        local = np.array(
                            [
                                [-w / 2, -h / 2],
                                [w / 2, -h / 2],
                                [w / 2, h / 2],
                                [-w / 2, h / 2],
                            ],
                            dtype=np.float32,
                        )
                        rot = np.array(
                            [[cos_t, -sin_t], [sin_t, cos_t]], dtype=np.float32
                        )
                        corners = local @ rot.T + np.array([cx, cy], dtype=np.float32)
                        detections.append(
                            {
                                "class_id": 0,
                                "polygon_px": [
                                    (float(p[0]), float(p[1])) for p in corners
                                ],
                                "confidence": float(conf),
                            }
                        )
                except Exception:
                    logger.warning(
                        "Sequential dataset inference failed on %s",
                        image_path,
                        exc_info=True,
                    )
                    detections = []
                if self._stop_if_cancelled():
                    return
                per_image[image_path] = detections
                for det in detections:
                    detection_count += 1
                    class_id = int(det.get("class_id", 0))
                    class_counts[class_id] = class_counts.get(class_id, 0) + 1
                    confidence_sum += float(det.get("confidence", 0.0))
                self.progress.emit(int(index / max(1, total) * 100))
        else:
            self.status.emit("Loading model…")
            task = {
                "detect_direct": "detect",
                "segment_direct": "segment",
            }.get(self._inference_kind, "obb")
            model, device = load_torch_model(
                self._model_path, self._device_preference, task=task
            )
            if self._stop_if_cancelled():
                return

            slice_settings = self._slice_settings
            sliced = bool(slice_settings is not None and slice_settings.enabled)
            for index, image_path in enumerate(self._image_paths, start=1):
                if self._stop_if_cancelled():
                    return
                self.status.emit(
                    f"Running inference on image {index}/{total}: {Path(image_path).name}"
                )
                try:
                    if sliced:
                        import cv2

                        frame = cv2.imread(str(image_path))
                        if frame is None:
                            raise RuntimeError(f"Could not read image: {image_path}")
                        obb = predict_sliced_obb_result(
                            model,
                            frame,
                            geometry_mode=slice_settings.geometry_mode,
                            imgsz=self._imgsz_obb_direct,
                            # 0.0 => tile_size_for_mode degrades auto_object to imgsz tiling (honest; no fabricated scale)
                            reference_body_px=slice_settings.reference_body_px,
                            object_tile_fraction=preview_object_tile_fraction(
                                slice_settings.target_sizes,
                                slice_settings.object_tile_fraction,
                                self._imgsz_obb_direct,
                            ),
                            slice_width=slice_settings.slice_width,
                            slice_height=slice_settings.slice_height,
                            overlap=slice_settings.overlap,
                            merge_threshold=slice_settings.merge_threshold,
                            confidence_threshold=confidence_threshold,
                            task=task,
                            should_stop=self.is_cancelled,
                        )
                        detections = (
                            dicts_from_obb_result(obb) if obb is not None else []
                        )
                    else:
                        detections = predict_preview_detections_for_image(
                            model,
                            image_path,
                            device=device,
                            confidence_threshold=confidence_threshold,
                            task=task,
                            should_stop=self.is_cancelled,
                        )
                except Exception:
                    logger.warning(
                        "Dataset inference failed on %s", image_path, exc_info=True
                    )
                    detections = []
                if self._stop_if_cancelled():
                    return
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
_TOOLS_PANEL_PREFERRED_WIDTH = 380
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
QPushButton[detectkitVariant="danger"] {
    background-color: #c0392b;
    color: white;
}
QPushButton[detectkitVariant="danger"]:hover {
    background-color: #a93226;
}
QPushButton[detectkitVariant="danger"]:pressed {
    background-color: #922b21;
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
        self._last_prediction_request: tuple[object, ...] | None = None
        self._dataset_predictions: dict[str, list[dict[str, object]]] = {}
        self._dataset_prediction_signature: tuple[object, ...] | None = None
        self._inference_settings_override: InferenceRunSettings | None = None
        self._inference_worker: Optional[_DetectKitDatasetInferenceWorker] = None
        self._inference_progress_dialog: Optional[QProgressDialog] = None
        self._portable_worker = None
        self._portable_progress_dialog = None
        self._escalation_worker = None
        self._escalation_progress_dialog: Optional[QProgressDialog] = None
        self._last_escalation_result: object | None = None
        self._last_escalation_error: str | None = None

        # Build workspace panels first (toolbar actions need them)
        self._dataset_panel = DatasetPanel()
        self._canvas = OBBCanvas()
        self._review_bar = ReviewBar()
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
        self._tools_panel.inference_settings_requested.connect(
            self._open_inference_settings_dialog
        )
        self._tools_panel.escalate_geometry_requested.connect(
            lambda: escalation_actions.on_escalate_geometry(self)
        )
        self._tools_panel.semantic_escalation_requested.connect(
            lambda: escalation_actions.on_semantic_escalation(self)
        )
        self._tools_panel.mark_reviewed_requested.connect(self._on_mark_reviewed)
        self._tools_panel.review_escalations_requested.connect(
            self._on_go_to_staged_review
        )

        self._review_bar.accept_overwrite_requested.connect(
            lambda: self._on_review_accept(MergeMode.OVERWRITE)
        )
        self._review_bar.accept_add_new_requested.connect(
            lambda: self._on_review_accept(MergeMode.ADD_NEW)
        )
        self._review_bar.reject_requested.connect(self._on_review_reject)
        self._review_bar.accept_all_requested.connect(
            lambda: self._on_review_bulk(accept=True)
        )
        self._review_bar.reject_all_requested.connect(
            lambda: self._on_review_bulk(accept=False)
        )
        self._review_bar.next_undecided_requested.connect(
            self._on_review_next_undecided
        )
        self._review_bar.revert_requested.connect(self._on_review_revert)
        self._review_bar.rethreshold_requested.connect(self._on_review_rethreshold)

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

        act_inference_settings = QAction("Inference Settings", self)
        act_inference_settings.triggered.connect(self._open_inference_settings_dialog)
        tb.addAction(act_inference_settings)

        act_history = QAction("History", self)
        act_history.triggered.connect(self._open_history_dialog)
        tb.addAction(act_history)

        self._al_action = QAction("Active Learning", self)
        self._al_action.triggered.connect(self._open_active_learning_dialog)
        self._al_action.setEnabled(False)
        tb.addAction(self._al_action)

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
                self._queue_source_manager_if_empty(proj)
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
        canvas_layout.addWidget(self._review_bar)
        canvas_layout.addWidget(self._canvas, 1)
        canvas_shell.setMinimumWidth(_CANVAS_MIN_WIDTH)
        splitter.addWidget(canvas_shell)  # index 1
        self._right_tabs = canvas_shell

        self._tools_panel.setProperty("detectkitRole", "panelShell")
        splitter.addWidget(self._tools_panel)  # index 2

        splitter.setCollapsible(0, False)
        splitter.setCollapsible(1, False)
        splitter.setCollapsible(2, False)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setStretchFactor(2, 0)
        splitter.setSizes(
            [
                _DATASET_PANEL_MIN_WIDTH,
                _CANVAS_MIN_WIDTH + 220,
                _TOOLS_PANEL_PREFERRED_WIDTH,
            ]
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
                self._queue_source_manager_if_empty(proj)
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
                    self._queue_source_manager_if_empty(proj)
            return

        proj = create_project(
            proj_dir,
            project_info["class_name"],
            class_names=list(project_info.get("class_names", [])),
        )
        self._load_project(proj)
        self._queue_source_manager_if_empty(proj)

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
            self._queue_source_manager_if_empty(proj)
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
            self._queue_source_manager_if_empty(proj)
        else:
            QMessageBox.warning(
                self,
                "Open Project Zip",
                f"Imported project could not be opened:\n{imported_dir}",
            )

    def _sync_al_action_enabled(self) -> None:
        """Enable the AL toolbar action only when a project with an active model is loaded."""
        self._al_action.setEnabled(
            bool(self._project and self._project.active_model_path)
        )

    def _load_project(self, proj: DetectKitProject) -> None:
        """Activate proj: wire panels, show toolbar, switch to workspace."""
        self._project = proj
        self._current_source_path = ""
        self._current_image_path = ""
        self._last_prediction_request = None
        self._inference_settings_override = None

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

        # Sync tools panel display with resolved pair info.
        if proj.active_model_path:
            try:
                kind, primary, secondary = detectkit_resolve_inference_models(
                    proj, proj.active_model_path
                )
                self._tools_panel.set_active_model_path(
                    primary,
                    secondary if kind in {"sequential", "sequential_segment"} else None,
                )
            except RuntimeError as exc:
                # Sequential pair incomplete — show with suffix, no secondary.
                self._tools_panel.set_active_model_path(
                    f"{proj.active_model_path} (missing OBB head)"
                )
                logger.warning("Sequential pair resolution failed: %s", exc)

        self._sync_al_action_enabled()

        self._toolbar.setVisible(True)
        self._stack.setCurrentIndex(1)
        self.menuBar().show()

        if hasattr(self, "_recents_store"):
            self._recents_store.add(str(proj.project_dir))
            if hasattr(self, "_welcome_page"):
                self._welcome_page.refresh_recents()
        self._refresh_recent_menu()

        self.statusBar().showMessage(f"Loaded project: {proj.project_dir}", 5000)

    def _queue_source_manager_if_empty(self, proj: DetectKitProject) -> None:
        """Schedule Source Manager when a user-facing load opens an empty project."""
        if proj.sources:
            return
        QTimer.singleShot(100, self._open_source_manager)

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
        # Sync resolved pair display and AL gate regardless of dialog outcome.
        if self._project.active_model_path:
            try:
                kind, primary, secondary = detectkit_resolve_inference_models(
                    self._project, self._project.active_model_path
                )
                self._tools_panel.set_active_model_path(
                    primary,
                    secondary if kind in {"sequential", "sequential_segment"} else None,
                )
            except RuntimeError as exc:
                self._tools_panel.set_active_model_path(
                    f"{self._project.active_model_path} (missing OBB head)"
                )
                logger.warning("Sequential pair resolution failed: %s", exc)
        self._sync_al_action_enabled()

    def _open_active_learning_dialog(self) -> None:
        if self._project is None:
            return
        from .dialogs.active_learning import ActiveLearningDialog

        dlg = ActiveLearningDialog(project=self._project, parent=self)
        model_path = str(self._project.active_model_path or "").strip()
        try:
            kind, _primary, _secondary = detectkit_resolve_inference_models(
                self._project, model_path
            )
        except RuntimeError:
            kind = "obb_direct"
        task = (
            "segment"
            if kind in {"segment_direct", "sequential_segment"}
            else "detect" if kind == "detect_direct" else "obb"
        )
        dlg.set_model_task(task)
        dlg.set_run_handler(lambda: self._start_al_round(dlg))
        dlg.finished.connect(lambda *_: self._cancel_al_round())
        dlg.open()

    def _start_al_round(self, dlg) -> None:
        from hydra_suite.detectkit.jobs.al_worker import ALWorker

        try:
            request = dlg.build_request(detector=self._resolve_active_detector_spec())
        except NotImplementedError as exc:
            dlg.status_label.setText(f"Error: {exc}")
            return
        except Exception as exc:
            dlg.status_label.setText(f"Error: {exc}")
            return

        worker = ALWorker(request)
        dlg.set_running(True)
        worker.progress.connect(dlg.progress.setValue)
        worker.status.connect(dlg.status_label.setText)
        worker.result_ready.connect(
            lambda path, n, _ids: dlg.status_label.setText(
                f"Imported {n} frames -> {path}"
            )
        )
        worker.error.connect(lambda msg: dlg.status_label.setText(f"Error: {msg}"))
        worker.finished.connect(lambda: dlg.set_running(False))
        worker.start()
        self._al_worker = worker

    def _cancel_al_round(self) -> None:
        worker = getattr(self, "_al_worker", None)
        if worker is not None:
            worker.requestInterruption()

    def _resolve_active_detector_spec(self):
        """Return the `ALDetectorSpec` describing the project's active model.

        Replaces the old `_load_active_detector_fn`, which eagerly loaded torch
        models and closed over them in a per-frame `detector_fn(frame, conf,
        iou)`. The AL round now builds its own `InferenceRunner` from this
        declarative spec and runs one batched, cached detection pass, so the
        GUI's job is reduced to resolving *which* checkpoints to use -- the
        same `detectkit_resolve_inference_models` call as before, minus the
        model loading.
        """
        from hydra_suite.detectkit.jobs.al_worker import ALDetectorSpec

        if self._project is None:
            raise RuntimeError("No project loaded.")
        model_path = str(self._project.active_model_path or "").strip()
        if not model_path:
            raise RuntimeError(
                "No active model selected. Pick one from History (double-click)."
            )
        if not detectkit_model_path_is_previewable(self._project, model_path):
            raise RuntimeError(
                "Selected model is incomplete for inference. Select a direct model "
                "or train the matching sequential counterpart."
            )

        from .prediction_preview import _resolve_compute_runtime

        kind, primary, secondary = detectkit_resolve_inference_models(
            self._project, model_path
        )
        crop_pad_ratio = float(getattr(self._project, "crop_pad_ratio", None) or 0.15)
        stage2_image_size = (
            self._project.imgsz_seq_crop_segment
            if kind == "sequential_segment"
            else self._project.imgsz_seq_crop_obb
        )
        # The project's device preference resolved to a cpu/mps/cuda torch
        # runtime for the old closure; Runtime Gen-2's equivalent knob is the
        # tier. "gpu" (not "gpu_fast") keeps AL on the same native torch
        # backend the closure used -- gpu_fast would trigger a TensorRT/CoreML
        # export, which AL scoring never did before.
        runtime_tier = (
            "cpu"
            if _resolve_compute_runtime(self._project.device or "auto") == "cpu"
            else "gpu"
        )

        if kind in {"sequential", "sequential_segment"} and secondary is not None:
            return ALDetectorSpec(
                kind=kind,
                model_path=primary,
                secondary_model_path=secondary,
                crop_pad_ratio=crop_pad_ratio,
                stage2_image_size=stage2_image_size,
                runtime_tier=runtime_tier,
            )

        return ALDetectorSpec(
            kind=kind,
            model_path=primary,
            secondary_model_path=None,
            crop_pad_ratio=crop_pad_ratio,
            runtime_tier=runtime_tier,
        )

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
            self._sync_al_action_enabled()
        self._canvas.set_layer_visible("gt", settings.show_gt)
        self._canvas.set_layer_visible("pred", settings.show_pred)
        self._canvas.set_layer_visible("escalation", settings.show_escalation)
        self._canvas.set_class_filter(settings.visible_class_ids)
        self._canvas.set_derived_levels_visible(settings.show_derived_levels)

        signature = self._dataset_signature(settings)

        if signature is None:
            self._canvas.remove_layer("pred")
            self._last_prediction_request = None
            return

        if self._project is not None and not detectkit_model_path_is_previewable(
            self._project,
            signature[1],
        ):
            self._canvas.remove_layer("pred")
            self._last_prediction_request = None
            self.statusBar().showMessage(
                "Selected model does not support direct preview overlays.",
                4000,
            )
            return

        if signature == self._dataset_prediction_signature:
            self._refresh_prediction_overlay(force=True)
        else:
            self._canvas.remove_layer("pred")
            self._last_prediction_request = None
            self.statusBar().showMessage(
                "Inference settings changed. Click Run Inference to refresh overlay predictions.",
                4000,
            )

    def _effective_inference_settings(self, overlay_settings) -> InferenceRunSettings:
        """Return the active runtime override, or a project-default snapshot."""
        if self._project is None:
            raise RuntimeError("No DetectKit project is open.")
        if self._inference_settings_override is not None:
            # The existing confidence slider remains a quick adjustment even
            # after the more detailed dialog has supplied runtime geometry.
            return InferenceRunSettings(
                device=self._inference_settings_override.device,
                confidence_threshold=overlay_settings.confidence_threshold,
                slice_settings=self._inference_settings_override.slice_settings,
            )
        return InferenceRunSettings.from_project(
            self._project, overlay_settings.confidence_threshold
        )

    def _open_inference_settings_dialog(self) -> None:
        """Let users tune the next dataset inference run without mutating training."""
        if self._project is None:
            QMessageBox.information(
                self,
                "Inference Settings",
                "Open a project before changing inference settings.",
            )
            return

        from .dialogs.inference_settings import InferenceSettingsDialog

        overlay_settings = self._tools_panel.get_overlay_settings()
        defaults = InferenceRunSettings.from_project(
            self._project, overlay_settings.confidence_threshold
        )
        current = self._effective_inference_settings(overlay_settings)
        dialog = InferenceSettingsDialog(current, defaults, parent=self)
        if not dialog.exec():
            return

        self._inference_settings_override = dialog.settings()
        self._tools_panel.set_confidence_threshold(
            self._inference_settings_override.confidence_threshold
        )
        self._on_overlay_changed()
        self.statusBar().showMessage(
            "Inference settings applied for this window. Run Inference to refresh predictions.",
            5000,
        )

    def _dataset_signature(self, settings) -> tuple[object, ...] | None:
        if self._project is None or not self._current_source_path:
            return None
        model_path = str(settings.active_model_path or "").strip()
        if not model_path:
            return None
        return (
            self._current_source_path,
            model_path,
        ) + self._effective_inference_settings(settings).cache_key()

    def _visible_inference_stats(self, confidence_threshold: float) -> dict:
        """Summarize cached candidates after applying the live display filter."""
        per_image = {
            image_path: _filter_detections_by_confidence(
                detections, confidence_threshold
            )
            for image_path, detections in self._dataset_predictions.items()
        }
        detections = [
            detection
            for image_detections in per_image.values()
            for detection in image_detections
        ]
        class_counts: dict[int, int] = {}
        for detection in detections:
            class_id = int(detection.get("class_id", 0))
            class_counts[class_id] = class_counts.get(class_id, 0) + 1
        count = len(detections)
        return {
            "per_image": per_image,
            "image_count": len(per_image),
            "detection_count": count,
            "class_counts": class_counts,
            "mean_confidence": (
                sum(float(detection.get("confidence", 0.0)) for detection in detections)
                / count
                if count
                else 0.0
            ),
        }

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

        try:
            kind, primary, secondary = detectkit_resolve_inference_models(
                self._project, model_path
            )
        except RuntimeError as exc:
            QMessageBox.information(self, "Run Inference", str(exc))
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

        inference_settings = self._effective_inference_settings(settings)

        worker = _DetectKitDatasetInferenceWorker(
            image_paths,
            primary,
            inference_settings.device,
            INFERENCE_CONFIDENCE_FLOOR,
            inference_kind=kind,
            secondary_model_path=(
                secondary if kind in {"sequential", "sequential_segment"} else None
            ),
            crop_pad_ratio=self._project.crop_pad_ratio,
            stage2_image_size=(
                self._project.imgsz_seq_crop_segment
                if kind == "sequential_segment"
                else self._project.imgsz_seq_crop_obb
            ),
            slice_settings=inference_settings.slice_settings,
            imgsz_obb_direct=self._project.imgsz_obb_direct,
        )
        worker.progress.connect(progress.setValue)
        worker.status.connect(progress.setLabelText)
        worker.error.connect(
            lambda msg: QMessageBox.warning(self, "Run Inference", msg)
        )

        def _request_cancel() -> None:
            worker.cancel()
            progress.setLabelText(
                "Cancelling inference… waiting for the current model call to stop."
            )
            progress.setCancelButtonText("Cancelling…")
            cancel_button = progress.findChild(QPushButton)
            if cancel_button is not None:
                cancel_button.setEnabled(False)
            progress.show()
            self.statusBar().showMessage("Cancelling inference…")

        progress.canceled.connect(_request_cancel)

        def _finish() -> None:
            progress.close()
            self._inference_worker = None
            self._inference_progress_dialog = None
            if worker.is_cancelled():
                self.statusBar().showMessage("Inference cancelled.", 5000)

        def _handle_success(result: dict) -> None:
            self._dataset_predictions = dict(result.get("per_image", {}))
            self._dataset_prediction_signature = signature
            self._tools_panel.update_inference_stats(
                self._visible_inference_stats(settings.confidence_threshold),
                class_names=self._project.class_names,
            )
            self._refresh_prediction_overlay(force=True)
            self.statusBar().showMessage(
                f"Inference complete: {result.get('detection_count', 0):,} candidate(s) "
                f"retained at ≥{INFERENCE_CONFIDENCE_FLOOR:.2f}.",
                5000,
            )

        worker.success.connect(_handle_success)
        worker.finished.connect(_finish)
        self._inference_worker = worker
        self._inference_progress_dialog = progress
        progress.show()
        worker.start()

    # ------------------------------------------------------------------
    # SAM2 escalation
    # ------------------------------------------------------------------

    def _on_mark_reviewed(self) -> None:
        """Mark a selected unreviewed derived source as reviewed and persist."""
        if self._project is None:
            QMessageBox.information(self, "Mark reviewed", "Open a project first.")
            return

        unreviewed = [
            s for s in self._project.sources if not getattr(s, "reviewed", True)
        ]
        if not unreviewed:
            QMessageBox.information(
                self,
                "Mark reviewed",
                "There are no unreviewed sources to mark.",
            )
            return

        names = [s.name for s in unreviewed]
        choice, ok = QInputDialog.getItem(
            self,
            "Mark reviewed",
            "Select a source to mark reviewed:",
            names,
            0,
            False,
        )
        if not ok or not choice:
            return

        for src in unreviewed:
            if src.name == choice:
                src.reviewed = True
                break

        self._save_current_project()
        self._dataset_panel.refresh_sources(self._project)
        self._tools_panel.refresh_overview()
        self.statusBar().showMessage(f"Marked '{choice}' as reviewed.", 4000)

    def _refresh_prediction_overlay(self, *, force: bool = False) -> None:
        self._refresh_overlays(keys=("pred",))
        if self._project is not None:
            settings = self._tools_panel.get_overlay_settings()
            self._tools_panel.update_inference_stats(
                self._visible_inference_stats(settings.confidence_threshold),
                class_names=self._project.class_names,
            )

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

    def _refresh_escalation_overlay(self) -> None:
        self._refresh_overlays(keys=("escalation",))

    def _frame_context(self) -> "FrameContext | None":
        """Everything the providers need about the frame on screen.

        The size comes from the pixmap load_image just built. Decoding the
        file again per provider cost ~100 ms per keypress on 4512^2 frames.
        """
        if not self._current_source_path or not self._current_image_path:
            return None
        size = self._canvas.image_size()
        if size is None:
            return None
        settings = self._tools_panel.get_overlay_settings()
        predictions: list[dict] = []
        signature = self._dataset_signature(settings)
        if (
            self._project is not None
            and signature is not None
            and signature == self._dataset_prediction_signature
            and self._current_image_path in self._dataset_predictions
        ):
            predictions = _filter_detections_by_confidence(
                self._dataset_predictions.get(self._current_image_path, []),
                settings.confidence_threshold,
            )
        return FrameContext(
            project=self._project,
            source_path=self._current_source_path,
            image_path=self._current_image_path,
            size=size,
            predictions=predictions,
        )

    def _refresh_overlays(self, keys: "tuple[str, ...] | None" = None) -> None:
        """Ask each provider for its layer and set or remove it.

        A provider returning None means "this layer does not apply to this
        frame", which removes it. There is no path where a stale layer can
        survive a frame change -- the bug that left the previous frame's
        staged masks floating over the new pixmap.

        PROVIDERS is iterated in draw order, which the z values also encode.
        """
        ctx = self._frame_context()
        for provider in PROVIDERS:
            if keys is not None and provider.key not in keys:
                continue
            layer = provider.build(ctx) if ctx is not None else None
            if layer is None:
                self._canvas.remove_layer(provider.key)
            else:
                self._canvas.set_layer(layer)
            if provider.key == "pred":
                self._last_prediction_request = (
                    None
                    if layer is None
                    else self._dataset_signature(
                        self._tools_panel.get_overlay_settings()
                    )
                )

    def show_image(self, source_path: str, image_path: str) -> None:
        """Load an image and overlay GT labels, predictions and staged masks."""
        new_source = str(source_path or "")
        if new_source != self._current_source_path:
            self._dataset_predictions = {}
            self._dataset_prediction_signature = None
        self._current_source_path = new_source
        self._current_image_path = str(image_path or "")
        self._last_prediction_request = None
        # Every layer is removed BEFORE the load can bail: otherwise
        # navigating to an unreadable frame left the previous frame's
        # overlays floating over the previous frame's pixmap.
        for provider in PROVIDERS:
            self._canvas.remove_layer(provider.key)
        if not self._canvas.load_image(image_path):
            return

        self._refresh_overlays()
        self._sync_review_bar()

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
    # Frame-granular review
    # ------------------------------------------------------------------

    def _current_source_obj(self):
        if self._project is None or not self._current_source_path:
            return None
        return next(
            (
                s
                for s in self._project.sources
                if str(s.path) == self._current_source_path
            ),
            None,
        )

    def _current_staged_root(self) -> "str | None":
        source = self._current_source_obj()
        review = getattr(source, "staged_review", None) if source else None
        if review is None or not str(review.staged_path).strip():
            return None
        return str(review.staged_path)

    def _current_staged_rel(self) -> "str | None":
        """The current frame's key into the review. One definition, reused."""
        source = self._current_source_obj()
        if source is None or not self._current_image_path:
            return None
        return review_key_for_image(source.path, self._current_image_path)

    def _sync_review_bar(self) -> None:
        """Show or hide the bar for the source on screen, and refresh its counter."""
        source = self._current_source_obj()
        review = getattr(source, "staged_review", None) if source else None
        if review is None:
            self._review_bar.clear_review_state()
            return
        decided, total = review_progress(review.staged_path)
        detail = (
            f"prompt '{review.prompt}'" if review.prompt else review.producer_variant
        )
        self._review_bar.set_review_state(
            review.producer,
            detail,
            decided,
            total,
            can_rethreshold=review.producer == "sam3",
        )

    def _after_review_change(self) -> None:
        """Everything a decision must trigger, in one place.

        Both layers, directly: the ground truth changed (that is the point of
        reviewing on the frame) and the staged proposal is now decided, so it
        must stop drawing. Neither refresh happens incidentally.
        """
        self._refresh_overlays(keys=("gt", "escalation"))
        self._sync_review_bar()
        self._save_current_project()

    def _on_review_accept(self, mode) -> None:
        source = self._current_source_obj()
        rel = self._current_staged_rel()
        if source is None or rel is None:
            return
        try:
            accept_frame(source, rel, mode=mode)
        except Exception as exc:
            QMessageBox.warning(self, "Review", str(exc))
            return
        self._after_review_change()
        self._offer_finish_if_complete(source)

    def _on_review_reject(self) -> None:
        source = self._current_source_obj()
        rel = self._current_staged_rel()
        if source is None or rel is None:
            return
        try:
            reject_frame(source, rel)
        except Exception as exc:
            QMessageBox.warning(self, "Review", str(exc))
            return
        self._after_review_change()
        self._offer_finish_if_complete(source)

    def _on_review_bulk(self, *, accept: bool) -> None:
        source = self._current_source_obj()
        if source is None:
            return
        verb = "Accept" if accept else "Reject"
        mode = MergeMode.ADD_NEW  # unused on the reject path; bound so it always exists
        if accept:
            choice = QMessageBox.question(
                self,
                "Accept All",
                "Replace each undecided frame's labels with the staged ones, "
                "or add only the non-overlapping staged instances?\n\n"
                "Yes = Replace, No = Add New.",
                QMessageBox.StandardButton.Yes
                | QMessageBox.StandardButton.No
                | QMessageBox.StandardButton.Cancel,
            )
            if choice == QMessageBox.StandardButton.Cancel:
                return
            mode = (
                MergeMode.OVERWRITE
                if choice == QMessageBox.StandardButton.Yes
                else MergeMode.ADD_NEW
            )
        try:
            count = accept_all(source, mode=mode) if accept else reject_all(source)
        except Exception as exc:
            QMessageBox.warning(self, verb + " All", str(exc))
            return
        self._after_review_change()
        self.statusBar().showMessage(f"{verb}ed {count} frame(s).", 4000)
        self._offer_finish_if_complete(source)

    def _on_review_next_undecided(self) -> None:
        staged_root = self._current_staged_root()
        if staged_root is None:
            return
        decided = read_decisions(staged_root)
        for rel in staged_frames(staged_root):
            if rel not in decided:
                self._dataset_panel.select_image_by_relative_label(rel)
                return
        self.statusBar().showMessage("Every staged frame has been decided.", 4000)

    def _on_review_revert(self) -> None:
        source = self._current_source_obj()
        staged_root = self._current_staged_root()
        if source is None or staged_root is None:
            return
        if (
            QMessageBox.question(
                self,
                "Revert Review",
                "Restore this source's labels, geometry level and class list to "
                "their state before this review started? Every decision made so "
                "far is cleared.",
            )
            != QMessageBox.StandardButton.Yes
        ):
            return
        try:
            revert_review(source, staged_root)
        except Exception as exc:
            QMessageBox.warning(self, "Revert Review", str(exc))
            return
        self._after_review_change()

    def _on_review_rethreshold(self) -> None:
        """The one irreplaceable feature of the retired dialog, per-review."""
        from PySide6.QtWidgets import QInputDialog

        from ..jobs.semantic_escalation import rethreshold_floor_for, rethreshold_staged

        source = self._current_source_obj()
        review = getattr(source, "staged_review", None) if source else None
        if review is None or review.producer != "sam3":
            return
        # Re-thresholding rewrites the staged labels underneath any decision
        # already recorded against them, so those decisions would describe
        # geometry that no longer exists. Refuse rather than silently
        # invalidate them; reverting first is the documented way through.
        if read_decisions(review.staged_path):
            QMessageBox.information(
                self,
                "Re-threshold",
                "Frames in this review have already been decided, and "
                "re-thresholding would rewrite the staged labels those "
                "decisions refer to. Revert the review first if you want to "
                "re-threshold it.",
            )
            return
        current = float(review.params.get("confidence", 0.35))
        minimum = rethreshold_floor_for([source])
        value, ok = QInputDialog.getDouble(
            self,
            "Re-threshold",
            f"New confidence (cache floor {minimum:.2f}):",
            max(current, minimum),
            minimum,
            0.99,
            2,
        )
        if not ok:
            return
        try:
            kept = rethreshold_staged(
                source,
                confidence=value,
                merge_iou=float(review.params.get("merge_iou", 0.5)),
            )
        except Exception as exc:
            QMessageBox.warning(self, "Re-threshold", str(exc))
            return
        self._after_review_change()
        self.statusBar().showMessage(
            f"{source.name}: {kept} instance(s) at confidence {value:.2f}.", 5000
        )

    def _offer_finish_if_complete(self, source) -> None:
        if not is_complete(source):
            return
        if (
            QMessageBox.question(
                self,
                "Review Complete",
                "Every staged frame has been decided. Close this review?\n\n"
                "Closing deletes the staged proposals AND the snapshot, so "
                '"Revert Review" is no longer available afterwards.',
            )
            != QMessageBox.StandardButton.Yes
        ):
            return
        finish_review(source, self._project.project_dir if self._project else None)
        self._save_current_project()
        self._dataset_panel.refresh_sources(self._project)
        self._tools_panel.refresh_overview()
        self._refresh_overlays(keys=("gt", "escalation"))
        self._sync_review_bar()

    def _on_go_to_staged_review(self) -> None:
        """Jump to a source with a staged review; the review bar drives it.

        Kept as a menu entry because a staged review is otherwise only
        discoverable by browsing to the right source.
        """
        if self._project is None:
            QMessageBox.information(self, "Staged Reviews", "Open a project first.")
            return
        pending = [s for s in self._project.sources if s.staged_review is not None]
        if not pending:
            QMessageBox.information(
                self, "Staged Reviews", "There are no staged reviews."
            )
            return
        self._dataset_panel.select_source_by_path(str(pending[0].path))

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
