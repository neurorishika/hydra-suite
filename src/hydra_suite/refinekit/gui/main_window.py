"""Main window for RefineKit.

Provides the review interface for correcting identity issues
detected in HYDRA Suite tracking trajectories.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import List, Optional

import pandas as pd
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QKeyEvent
from PySide6.QtWidgets import (
    QCheckBox,
    QFileDialog,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSplitter,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from hydra_suite.core.tracking.confidence.confidence_density import load_regions
from hydra_suite.refinekit.config.schemas import RefineKitConfig
from hydra_suite.refinekit.core.correction_writer import CorrectionWriter
from hydra_suite.refinekit.core.event_scorer import EventScorer
from hydra_suite.refinekit.core.event_types import EventType, SuspicionEvent
from hydra_suite.refinekit.gui.widgets.kinematics_viewer import (
    DEFAULT_SERIES_ENABLED,
    SERIES_SPECS,
    KinematicsViewerWidget,
    build_kinematics_cache,
)
from hydra_suite.refinekit.gui.widgets.suspicion_queue import SuspicionQueueWidget
from hydra_suite.refinekit.gui.widgets.timeline_panel import TimelinePanelWidget
from hydra_suite.refinekit.gui.widgets.video_player import VideoPlayerWidget
from hydra_suite.utils.file_dialogs import HydraFileDialog as QFileDialog  # noqa: F811
from hydra_suite.widgets.workers import BaseWorker

logger = logging.getLogger(__name__)

_VIDEO_FILTER = "Video files (*.mp4 *.avi *.mov *.mkv *.wmv);;All files (*)"
_MAX_MANUAL_REGION = 300  # max frames for a user-selected manual review region
_MAX_ACTIVE_KINEMATICS_SERIES = 4


# ---------------------------------------------------------------------------
# Background scorer worker
# ---------------------------------------------------------------------------


class _ScorerWorker(BaseWorker):
    """Run :meth:`EventScorer.score_all` off the GUI thread."""

    events_ready = Signal(list)

    def __init__(self, scorer, df, parent=None) -> None:
        super().__init__(parent)
        self._scorer = scorer
        self._df = df

    def execute(self):
        """Run the scorer's full pass over the trajectory DataFrame and emit the resulting event list."""
        try:
            events = self._scorer.score_all(self._df)
        except Exception:
            logger.exception("Scorer worker failed")
            events = []
        self.events_ready.emit(events)


class _KinematicsWorker(BaseWorker):
    """Precompute per-track kinematics in the background."""

    progress_changed = Signal(int, int)
    cache_ready = Signal(object, object)

    def __init__(self, df: pd.DataFrame, parent=None) -> None:
        super().__init__(parent)
        self._df = df.copy()

    def execute(self):
        cache, frame_range = build_kinematics_cache(
            self._df,
            progress_callback=lambda done, total: self.progress_changed.emit(
                done, total
            ),
            should_cancel=self.isInterruptionRequested,
        )
        if self.isInterruptionRequested():
            return
        self.cache_ready.emit(cache, frame_range)


class MainWindow(QMainWindow):
    """RefineKit main window."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("RefineKit")

        self.config = RefineKitConfig()
        self._sessions: List[str] = []
        self._session_idx: int = -1
        self._video_path: Optional[str] = None
        self._writer: Optional[CorrectionWriter] = None
        self._df: Optional[pd.DataFrame] = None
        self._scorer: Optional[EventScorer] = None
        self._scorer_worker: Optional[_ScorerWorker] = None
        self._kinematics_worker: Optional[_KinematicsWorker] = None
        self._kinematics_toggles: dict[str, QCheckBox] = {}
        # (frame_start, frame_end, track_ids) tuples for deprioritisation
        self._reviewed_regions: List[tuple] = []

        self._build_ui()
        self.apply_stylesheet()
        self.statusBar().showMessage("RefineKit — ready", 4000)

    # ------------------------------------------------------------------
    # Stylesheet
    # ------------------------------------------------------------------

    def apply_stylesheet(self) -> None:
        """Apply the MAT dark theme to the entire window (matches MAT / PoseKit / ClassKit)."""
        self.setStyleSheet("""
            QMainWindow {
                background-color: #1e1e1e;
            }
            QWidget {
                background-color: #1e1e1e;
                color: #e0e0e0;
                font-family: "SF Pro Text", "Helvetica Neue", "Segoe UI", Roboto, Arial, sans-serif;
                font-size: 11px;
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
                padding: 2px 8px;
                background-color: #1e1e1e;
                color: #9cdcfe;
                border-radius: 3px;
            }
            QListWidget {
                background-color: #252526;
                alternate-background-color: #2d2d30;
                border: 1px solid #3e3e42;
                border-radius: 4px;
                padding: 4px;
                outline: none;
            }
            QListWidget::item {
                padding: 6px 10px;
                border-radius: 3px;
                margin: 1px 0px;
            }
            QListWidget::item:selected {
                background-color: #094771;
                color: #ffffff;
            }
            QListWidget::item:hover:!selected {
                background-color: #2a2d2e;
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
                color: #777777;
            }
            QComboBox {
                background-color: #3c3c3c;
                color: #e0e0e0;
                border: 1px solid #3e3e42;
                border-radius: 4px;
                padding: 4px 8px;
                min-height: 22px;
            }
            QComboBox:hover { border-color: #0e639c; }
            QComboBox:focus { border-color: #007acc; }
            QComboBox::drop-down { border: none; width: 20px; }
            QComboBox QAbstractItemView {
                background-color: #252526;
                border: 1px solid #3e3e42;
                selection-background-color: #094771;
                selection-color: #ffffff;
                outline: none;
            }
            QLineEdit {
                background-color: #3c3c3c;
                color: #e0e0e0;
                border: 1px solid #3e3e42;
                border-radius: 4px;
                padding: 4px 8px;
                min-height: 22px;
            }
            QLineEdit:hover { border-color: #0e639c; }
            QLineEdit:focus { border-color: #007acc; }
            QSpinBox, QDoubleSpinBox {
                background-color: #3c3c3c;
                color: #e0e0e0;
                border: 1px solid #3e3e42;
                border-radius: 4px;
                padding: 4px 4px 4px 8px;
                min-height: 22px;
            }
            QSpinBox:hover, QDoubleSpinBox:hover { border-color: #0e639c; }
            QSpinBox:focus, QDoubleSpinBox:focus { border-color: #007acc; }
            QCheckBox { color: #cccccc; spacing: 8px; }
            QCheckBox::indicator {
                width: 14px; height: 14px;
                border: 1px solid #3e3e42;
                border-radius: 3px;
                background-color: #3c3c3c;
            }
            QCheckBox::indicator:checked {
                background-color: #0e639c;
                border-color: #007acc;
            }
            QRadioButton { color: #cccccc; spacing: 8px; }
            QRadioButton::indicator {
                width: 14px; height: 14px;
                border: 1px solid #3e3e42;
                border-radius: 7px;
                background-color: #3c3c3c;
            }
            QRadioButton::indicator:checked {
                background-color: #007acc;
                border-color: #007acc;
            }
            QLabel {
                color: #cccccc;
                background-color: transparent;
            }
            QToolBar {
                background-color: #252526;
                border-bottom: 1px solid #3e3e42;
                spacing: 6px;
                padding: 4px 6px;
            }
            QToolButton {
                background-color: transparent;
                border: none;
                border-radius: 4px;
                padding: 6px 10px;
                color: #cccccc;
            }
            QToolButton:hover { background-color: #2a2d2e; }
            QToolButton:pressed { background-color: #094771; }
            QTabWidget::pane {
                border: 1px solid #3e3e42;
                border-radius: 0px;
                background-color: #1e1e1e;
            }
            QTabBar::tab {
                background-color: #252526;
                color: #cccccc;
                border: 1px solid #3e3e42;
                border-bottom: none;
                padding: 6px 16px;
                min-width: 80px;
            }
            QTabBar::tab:selected {
                background-color: #1e1e1e;
                color: #ffffff;
                border-top: 2px solid #007acc;
            }
            QTabBar::tab:hover:!selected { background-color: #2a2d2e; }
            QStatusBar {
                background-color: #007acc;
                color: #ffffff;
                border-top: 1px solid #0098ff;
                font-weight: 500;
                font-size: 12px;
            }
            QStatusBar QLabel {
                background-color: transparent;
                color: #ffffff;
                padding: 0px 4px;
            }
            QMenuBar {
                background-color: #252526;
                color: #cccccc;
                border-bottom: 1px solid #3e3e42;
                padding: 2px;
            }
            QMenuBar::item { padding: 5px 10px; background-color: transparent; border-radius: 3px; }
            QMenuBar::item:selected { background-color: #2a2d2e; }
            QMenuBar::item:pressed { background-color: #094771; }
            QMenu {
                background-color: #252526;
                color: #cccccc;
                border: 1px solid #3e3e42;
                border-radius: 4px;
                padding: 4px;
            }
            QMenu::item { padding: 6px 20px 6px 12px; border-radius: 3px; }
            QMenu::item:selected { background-color: #094771; color: #ffffff; }
            QMenu::separator { height: 1px; background-color: #3e3e42; margin: 4px 8px; }
            QSplitter::handle { background-color: #3e3e42; }
            QSplitter::handle:hover { background-color: #007acc; }
            QScrollArea { border: none; background-color: transparent; }
            QScrollBar:vertical {
                background-color: #252526;
                width: 10px;
                border-radius: 5px;
                margin: 0px;
            }
            QScrollBar::handle:vertical {
                background-color: #5a5a5a;
                border-radius: 5px;
                min-height: 24px;
            }
            QScrollBar::handle:vertical:hover { background-color: #007acc; }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0px; }
            QScrollBar:horizontal {
                background-color: #252526;
                height: 10px;
                border-radius: 5px;
                margin: 0px;
            }
            QScrollBar::handle:horizontal {
                background-color: #5a5a5a;
                border-radius: 5px;
                min-width: 24px;
            }
            QScrollBar::handle:horizontal:hover { background-color: #007acc; }
            QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal { width: 0px; }
            QProgressBar {
                border: 1px solid #3e3e42;
                border-radius: 4px;
                text-align: center;
                background-color: #252526;
                color: #cccccc;
                font-size: 11px;
            }
            QProgressBar::chunk { background-color: #0e639c; border-radius: 3px; }
            QSlider::groove:horizontal {
                height: 4px;
                background-color: #3e3e42;
                border-radius: 2px;
            }
            QSlider::handle:horizontal {
                background-color: #007acc;
                border: none;
                width: 12px;
                height: 12px;
                margin: -4px 0;
                border-radius: 6px;
            }
            QSlider::handle:horizontal:hover { background-color: #1177bb; }
            QSlider::sub-page:horizontal { background-color: #007acc; border-radius: 2px; }
            QFrame[frameShape="4"], QFrame[frameShape="5"] { color: #3e3e42; }
        """)

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)

        # --- Top bar: session navigator ---
        nav_bar = QHBoxLayout()

        self._btn_prev = QPushButton("\u25c0")
        self._btn_prev.setFixedWidth(30)
        self._btn_prev.setToolTip("Previous session")
        self._btn_prev.clicked.connect(self._prev_session)
        nav_bar.addWidget(self._btn_prev)

        self._session_label = QLabel("No session loaded")
        self._session_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        nav_bar.addWidget(self._session_label, stretch=1)

        self._btn_next = QPushButton("\u25b6")
        self._btn_next.setFixedWidth(30)
        self._btn_next.setToolTip("Next session")
        self._btn_next.clicked.connect(self._next_session)
        nav_bar.addWidget(self._btn_next)

        nav_bar.addSpacing(8)

        self._btn_load_video = QPushButton("Load Video\u2026")
        self._btn_load_video.setToolTip(
            "Open a single video file for review (*.mp4, *.avi \u2026)"
        )
        self._btn_load_video.clicked.connect(self._load_single_video)
        nav_bar.addWidget(self._btn_load_video)

        self._btn_load_list = QPushButton("Load Video List\u2026")
        self._btn_load_list.setToolTip("Open a .txt file with one video path per line")
        self._btn_load_list.clicked.connect(self._load_video_list)
        nav_bar.addWidget(self._btn_load_list)

        root.addLayout(nav_bar)

        # --- Main layout: suspicion queue | video + timeline ---
        hsplitter = QSplitter(Qt.Orientation.Horizontal)

        # Left: suspicion queue + kinematics controls
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(6)

        self._queue = SuspicionQueueWidget()
        self._queue.event_selected.connect(self._on_event_selected)
        self._queue.rescore_all_requested.connect(self._on_rescore_all)
        self._queue.merge_wizard_requested.connect(lambda: self._run_merge_wizard())
        left_layout.addWidget(self._queue, stretch=1)
        left_layout.addWidget(self._build_kinematics_controls(), stretch=0)
        left_panel.setMinimumWidth(260)
        left_panel.setMaximumWidth(400)
        hsplitter.addWidget(left_panel)

        # Right: video + kinematics + timeline in vertical splitter
        self._review_splitter = QSplitter(Qt.Orientation.Vertical)
        self._player = VideoPlayerWidget()
        self._player.frame_changed.connect(self._on_player_frame_changed)
        self._player.frame_axis_margins_changed.connect(
            self._on_player_frame_axis_changed
        )
        self._review_splitter.addWidget(self._player)

        self._kinematics = KinematicsViewerWidget()
        self._review_splitter.addWidget(self._kinematics)

        self._timeline = TimelinePanelWidget()
        self._timeline.split_requested.connect(self._on_manual_split)
        self._timeline.region_edit_requested.connect(self._on_manual_region_edit)
        self._timeline.track_move_requested.connect(self._on_manual_track_reassign)
        self._timeline.track_selected.connect(self._on_timeline_track_selected)
        self._review_splitter.addWidget(self._timeline)

        self._review_splitter.setStretchFactor(0, 4)
        self._review_splitter.setStretchFactor(1, 1)
        self._review_splitter.setStretchFactor(2, 2)
        self._review_splitter.setSizes([640, 0, 180])
        hsplitter.addWidget(self._review_splitter)

        hsplitter.setStretchFactor(0, 0)
        hsplitter.setStretchFactor(1, 1)

        # Stacked widget: page 0 = welcome splash, page 1 = main working view
        self._content_stack = QStackedWidget()
        self._content_stack.addWidget(self._make_welcome_page())  # index 0
        self._content_stack.addWidget(hsplitter)  # index 1
        root.addWidget(self._content_stack, stretch=1)
        self._update_nav_state()

    def _build_kinematics_controls(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(6, 0, 6, 6)
        layout.setSpacing(4)

        title = QLabel("Kinematics")
        title.setStyleSheet("font-weight: 600; color: #9cdcfe;")
        layout.addWidget(title)

        subtitle = QLabel(
            "Select a timeline track to open the kinematics viewer. Toggle up to 4 traces to overlay."
        )
        subtitle.setWordWrap(True)
        subtitle.setStyleSheet("color: #777777; font-size: 10px;")
        layout.addWidget(subtitle)

        grid = QGridLayout()
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setHorizontalSpacing(8)
        grid.setVerticalSpacing(2)
        for index, (key, label, _color, _default) in enumerate(SERIES_SPECS):
            checkbox = QCheckBox(label)
            checkbox.setChecked(DEFAULT_SERIES_ENABLED[key])
            checkbox.toggled.connect(self._on_kinematics_series_toggled)
            grid.addWidget(checkbox, index // 2, index % 2)
            self._kinematics_toggles[key] = checkbox
        layout.addLayout(grid)
        self._sync_kinematics_toggle_state()

        self._kinematics_progress = QProgressBar()
        self._kinematics_progress.setRange(0, 0)
        self._kinematics_progress.setFormat("Preparing kinematics…")
        self._kinematics_progress.setFixedHeight(16)
        self._kinematics_progress.setVisible(False)
        layout.addWidget(self._kinematics_progress)
        return panel

    def _load_review_dataframe(self, df: pd.DataFrame) -> None:
        self._df = df
        self._player.load_trajectories(df)
        self._timeline.load_trajectories(df)
        self._kinematics.set_data(df)
        self._start_kinematics_precompute(df)

        if df.empty:
            self._kinematics.set_active_track(None)
            self._set_kinematics_collapsed(True)
            return

        current_frame = getattr(
            self._player, "_current_frame", int(df["FrameID"].min())
        )
        self._kinematics.set_current_frame(current_frame)
        self._timeline.set_current_frame(current_frame)
        valid_tracks = {
            int(track_id) for track_id in df["TrajectoryID"].dropna().tolist()
        }
        if self._kinematics.active_track_id not in valid_tracks:
            self._kinematics.set_active_track(None)
            self._set_kinematics_collapsed(True)

    def _start_kinematics_precompute(self, df: pd.DataFrame) -> None:
        self._stop_kinematics_precompute()
        if df.empty:
            self._kinematics_progress.setVisible(False)
            self._kinematics.set_loading(False)
            self._kinematics.set_precomputed_data({}, (0, 0))
            return

        self._kinematics.set_loading(True)
        self._kinematics_progress.setRange(0, 0)
        self._kinematics_progress.setFormat("Preparing kinematics…")
        self._kinematics_progress.setVisible(True)
        self.statusBar().showMessage("Preparing kinematics…", 3000)

        worker = _KinematicsWorker(df, self)
        worker.progress_changed.connect(self._on_kinematics_progress)
        worker.cache_ready.connect(self._on_kinematics_ready)
        worker.error.connect(self._on_kinematics_error)
        worker.finished.connect(self._on_kinematics_worker_finished)
        self._kinematics_worker = worker
        worker.start()

    def _stop_kinematics_precompute(self) -> None:
        worker = self._kinematics_worker
        if worker is None:
            return
        if worker.isRunning():
            worker.requestInterruption()

    def _on_kinematics_progress(self, done: int, total: int) -> None:
        if self.sender() is not self._kinematics_worker:
            return
        self._kinematics_progress.setRange(0, max(total, 1))
        self._kinematics_progress.setValue(done)
        self._kinematics_progress.setFormat("Preparing kinematics… %v / %m")
        self._kinematics_progress.setVisible(True)

    def _on_kinematics_ready(self, cache: object, frame_range: object) -> None:
        if self.sender() is not self._kinematics_worker:
            return
        self._kinematics.set_precomputed_data(cache, frame_range)
        self._kinematics_progress.setVisible(False)
        self.statusBar().showMessage(
            f"Kinematics ready — {len(cache)} track{'s' if len(cache) != 1 else ''} prepared",
            4000,
        )

    def _on_kinematics_error(self, message: str) -> None:
        if self.sender() is not self._kinematics_worker:
            return
        self._kinematics_progress.setVisible(False)
        self._kinematics.set_loading(False)
        self.statusBar().showMessage(f"Kinematics precompute failed: {message}", 5000)

    def _on_kinematics_worker_finished(self) -> None:
        if self.sender() is not self._kinematics_worker:
            return
        self._kinematics_worker = None

    def _set_kinematics_collapsed(self, collapsed: bool) -> None:
        sizes = self._review_splitter.sizes()
        if len(sizes) != 3:
            return
        total = max(sum(sizes), 1)
        if collapsed:
            bottom = max(sizes[2], 160)
            top = max(total - bottom, 1)
            self._review_splitter.setSizes([top, 0, bottom])
            return
        if sizes[1] >= 80:
            return
        middle = min(max(total // 4, 140), 220)
        bottom = max(min(sizes[2], total - middle - 1), 140)
        top = max(total - middle - bottom, 1)
        self._review_splitter.setSizes([top, middle, bottom])

    def _on_kinematics_series_toggled(self, checked: bool) -> None:
        checkbox = self.sender()
        if (
            checked
            and isinstance(checkbox, QCheckBox)
            and self._checked_kinematics_count() > _MAX_ACTIVE_KINEMATICS_SERIES
        ):
            checkbox.blockSignals(True)
            checkbox.setChecked(False)
            checkbox.blockSignals(False)
            self.statusBar().showMessage(
                f"Show at most {_MAX_ACTIVE_KINEMATICS_SERIES} kinematics traces at once",
                3000,
            )
        self._sync_kinematics_toggle_state()
        self._kinematics.set_enabled_series(
            {
                key: checkbox.isChecked()
                for key, checkbox in self._kinematics_toggles.items()
            }
        )

    def _checked_kinematics_count(self) -> int:
        return sum(
            1 for checkbox in self._kinematics_toggles.values() if checkbox.isChecked()
        )

    def _sync_kinematics_toggle_state(self) -> None:
        at_limit = self._checked_kinematics_count() >= _MAX_ACTIVE_KINEMATICS_SERIES
        for checkbox in self._kinematics_toggles.values():
            checkbox.setEnabled(checkbox.isChecked() or not at_limit)

    def _on_timeline_track_selected(self, track_id: object) -> None:
        if track_id is None:
            self._kinematics.set_active_track(None)
            self._set_kinematics_collapsed(True)
            return
        self._kinematics.set_active_track(int(track_id))
        self._set_kinematics_collapsed(False)

    def _on_player_frame_changed(self, frame: int) -> None:
        self._kinematics.set_current_frame(frame)
        self._timeline.set_current_frame(frame)

    def _on_player_frame_axis_changed(
        self, left_margin: int, right_margin: int
    ) -> None:
        self._kinematics.set_frame_axis_margins(left_margin, right_margin)
        self._timeline.set_frame_axis_margins(left_margin, right_margin)

    def _make_welcome_page(self) -> QWidget:
        """Logo/welcome screen shown before any session is loaded."""
        from hydra_suite.widgets import (
            ButtonDef,
            RecentItemsStore,
            WelcomeConfig,
            WelcomePage,
        )

        store = RecentItemsStore("refinekit")
        self._recents_store = store

        config = WelcomeConfig(
            logo_svg="refinekit.svg",
            tagline="Review  \u00b7  Correct  \u00b7  Verify",
            buttons=[
                ButtonDef(
                    label="Load Video\u2026",
                    callback=self._load_single_video,
                    tooltip="Open a single video file for review",
                ),
                ButtonDef(
                    label="Load Video List\u2026",
                    callback=self._load_video_list,
                    tooltip="Open a .txt file listing one video path per line",
                ),
                ButtonDef(label="Quit", callback=self.close),
            ],
            recents_label="Recent Videos",
            recents_store=store,
            on_recent_clicked=self._open_recent_video,
        )
        self._welcome_page = WelcomePage(config)
        return self._welcome_page

    def _open_recent_video(self, path: str):
        """Open a video from the recent items list."""
        from pathlib import Path

        video_path = Path(path)
        if video_path.exists():
            self._sessions = [str(video_path)]
            self._session_idx = 0
            self._open_current_session()
        else:
            from PySide6.QtWidgets import QMessageBox

            QMessageBox.warning(self, "Not Found", f"Video not found:\n{path}")
            if hasattr(self, "_recents_store"):
                self._recents_store.remove(path)
                if hasattr(self, "_welcome_page"):
                    self._welcome_page.refresh_recents()

    # ------------------------------------------------------------------
    # Session management
    # ------------------------------------------------------------------

    def _load_single_video(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Select video", "", _VIDEO_FILTER)
        if path:
            self._sessions = [path]
            self._session_idx = 0
            self._open_current_session()

    def _load_video_list(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Select video list", "", "Text files (*.txt);;All files (*)"
        )
        if not path:
            return

        with open(path, "r", encoding="utf-8") as fh:
            lines = [ln.strip() for ln in fh if ln.strip() and not ln.startswith("#")]

        if not lines:
            QMessageBox.warning(self, "Empty list", "No video paths found.")
            return

        self._sessions = lines
        self._session_idx = 0
        self._open_current_session()

    def _prev_session(self) -> None:
        if self._session_idx > 0:
            self._session_idx -= 1
            self._open_current_session()

    def _next_session(self) -> None:
        if self._session_idx < len(self._sessions) - 1:
            self._session_idx += 1
            self._open_current_session()

    def _update_nav_state(self) -> None:
        has = len(self._sessions) > 0
        self._btn_prev.setEnabled(has and self._session_idx > 0)
        self._btn_next.setEnabled(has and self._session_idx < len(self._sessions) - 1)
        # Avoid duplicated load controls: splash handles initial loading.
        show_inline_load = has and self._content_stack.currentIndex() == 1
        self._btn_load_video.setVisible(show_inline_load)
        self._btn_load_list.setVisible(show_inline_load)
        if has:
            name = Path(self._sessions[self._session_idx]).stem
            self._session_label.setText(
                f"{name} ({self._session_idx + 1}/{len(self._sessions)})"
            )
        else:
            self._session_label.setText("No session loaded")

    def _open_current_session(self) -> None:
        """Open the video, discover CSV, create writer, load data, score."""
        if self._writer is not None:
            self._writer.close()
            self._writer = None

        self._reviewed_regions.clear()
        self._scorer = None

        video_path = self._sessions[self._session_idx]
        self._video_path = video_path

        csv_path = self._discover_csv(video_path)
        if csv_path is None:
            QMessageBox.warning(
                self,
                "CSV not found",
                f"Could not find a tracking CSV for:\n{video_path}",
            )
            self._update_nav_state()
            return

        self._writer = CorrectionWriter(csv_path)
        self._writer.open()

        self._player.load_video(video_path)
        self._load_review_dataframe(self._writer.df)
        self._content_stack.setCurrentIndex(1)  # reveal main view before modal flows

        # --- Merge wizard: offer automatic fragment stitching ---
        self._maybe_run_merge_wizard()

        self._run_scorer()
        if hasattr(self, "_recents_store"):
            self._recents_store.add(video_path)
        self._update_nav_state()
        logger.info(
            "Opened session %d: %s  CSV=%s",
            self._session_idx,
            video_path,
            csv_path,
        )

    @staticmethod
    def _discover_csv(video_path: str) -> Optional[Path]:
        """Find matching CSV for a video by trying common suffixes."""
        vp = Path(video_path)
        stem = vp.stem
        parent = vp.parent
        for suffix in (
            "tracking_final_with_individual.csv",
            "tracking_final_with_pose.csv",
            "tracking_final.csv",
        ):
            candidate = parent / f"{stem}_{suffix}"
            if candidate.exists():
                return candidate

        return None

    # ------------------------------------------------------------------
    # Merge wizard
    # ------------------------------------------------------------------

    def _maybe_run_merge_wizard(self) -> None:
        """Check for merge candidates and offer the wizard if any exist."""
        if self._df is None or self._video_path is None:
            return

        from hydra_suite.refinekit.core.merge_candidates import (
            build_candidates_for_target_count,
            build_swap_candidates,
            extract_segments,
        )

        last_frame = int(self._df["FrameID"].max())
        segments = extract_segments(self._df, last_frame)
        max_animals = self._load_tracking_max_animals()
        candidates, merge_tuning = build_candidates_for_target_count(
            segments,
            max_animals=max_animals,
        )
        swap_candidates = build_swap_candidates(self._df, segments)

        total = sum(len(v) for v in candidates.values()) + sum(
            len(v) for v in swap_candidates.values()
        )
        if total == 0:
            return

        n_merge = sum(len(v) for v in candidates.values())
        n_swap = sum(len(v) for v in swap_candidates.values())
        n_sources = len(set(candidates.keys()) | set(swap_candidates.keys()))
        answer = QMessageBox.question(
            self,
            "Fragment Merge Wizard",
            f"Found {n_merge} merge + {n_swap} swap candidate(s) across "
            f"{n_sources} fragmented track(s).\n\n"
            f"Run the merge wizard now?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.Yes,
        )
        if answer != QMessageBox.StandardButton.Yes:
            return

        self._run_merge_wizard(
            segments,
            candidates,
            swap_candidates,
            merge_tuning=merge_tuning,
            max_animals=max_animals,
        )

    def _run_merge_wizard(
        self,
        segments=None,
        candidates=None,
        swap_candidates=None,
        merge_tuning=None,
        max_animals=None,
    ) -> None:
        """Open the MergeWizardDialog.

        If *segments* / *candidates* are ``None`` they are recomputed from
        the current DataFrame.
        """
        if self._df is None or self._video_path is None or self._writer is None:
            return

        from hydra_suite.refinekit.core.merge_candidates import (
            build_candidates_for_target_count,
            build_swap_candidates,
            extract_segments,
        )
        from hydra_suite.refinekit.gui.dialogs.merge_wizard import MergeWizardDialog

        if segments is None or candidates is None:
            last_frame = int(self._df["FrameID"].max())
            segments = extract_segments(self._df, last_frame)
            max_animals = self._load_tracking_max_animals()
            candidates, merge_tuning = build_candidates_for_target_count(
                segments,
                max_animals=max_animals,
            )
        if swap_candidates is None:
            swap_candidates = build_swap_candidates(self._df, segments)

        total = sum(len(v) for v in candidates.values()) + sum(
            len(v) for v in swap_candidates.values()
        )
        if total == 0:
            self.statusBar().showMessage("No merge/swap candidates found", 3000)
            return

        dlg = MergeWizardDialog(
            video_path=self._video_path,
            df=self._df,
            segments=segments,
            candidates=candidates,
            writer=self._writer,
            parent=self,
            swap_candidates=swap_candidates,
            merge_tuning=merge_tuning,
            max_animals=max_animals,
        )
        dlg.exec()

        n = dlg.merges_applied
        flagged = dlg._model.flagged_events

        if n > 0:
            # Refresh everything from the writer's updated DataFrame
            self._load_review_dataframe(self._writer.df)
            # Store flagged events so they survive the upcoming rescore
            self._pending_flagged = flagged
            # Re-run scorer since track structure changed
            self._run_scorer()
            self.statusBar().showMessage(
                f"Merge wizard: {n} merge{'s' if n != 1 else ''} applied — rescoring\u2026",
                5000,
            )
        else:
            self.statusBar().showMessage("Merge wizard: no merges applied", 3000)
            # No scorer run — inject flagged events directly
            if flagged:
                self._queue.add_events(flagged)
                self._queue.show_rescore_button(True)
                self.statusBar().showMessage(
                    f"Merge wizard: {len(flagged)} pair(s) flagged for detailed editing",
                    4000,
                )

    def _load_tracking_max_animals(self) -> Optional[int]:
        """Load MAX_TARGETS from the sibling tracking config, if present."""
        if self._video_path is None:
            return None

        video_path = Path(self._video_path).expanduser()
        cfg_path = video_path.parent / f"{video_path.stem}_config.json"
        if not cfg_path.is_file():
            return None

        try:
            with open(cfg_path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
        except (json.JSONDecodeError, OSError, TypeError, ValueError):
            return None

        value = data.get("MAX_TARGETS")
        try:
            return max(1, int(value))
        except (TypeError, ValueError):
            return None

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    def _run_scorer(self) -> None:
        """Load regions (if available) and run the swap scorer in the background."""
        if self._df is None or self._video_path is None:
            return

        regions = []
        vp = Path(self._video_path)
        regions_path = vp.parent / f"{vp.stem}_density_regions.json"
        if regions_path.exists():
            try:
                regions = load_regions(regions_path)
            except Exception:
                logger.warning("Failed to load density regions from %s", regions_path)

        self._scorer = EventScorer(regions=regions)
        # Re-register previously reviewed regions so they stay deprioritised
        for rr in self._reviewed_regions:
            self._scorer.add_reviewed_region(*rr)

        # Disconnect any stale worker so its result is silently discarded
        if self._scorer_worker is not None and self._scorer_worker.isRunning():
            try:
                self._scorer_worker.events_ready.disconnect()
            except RuntimeError:
                pass

        self._queue.show_scoring_progress()
        self._scorer_worker = _ScorerWorker(self._scorer, self._df, self)
        self._scorer_worker.events_ready.connect(self._on_scorer_finished)
        self._scorer_worker.start()

    def _on_scorer_finished(self, events: list) -> None:
        """Slot called from the scorer worker thread when scoring is done."""
        self._queue.hide_scoring_progress()
        self._queue.populate(events)
        self._queue.show_rescore_button(False)
        self._queue.show_merge_wizard_button(True)

        # Re-inject flagged events from the merge wizard (they survived
        # the async scorer because we deferred adding them).
        pending = getattr(self, "_pending_flagged", [])
        if pending:
            self._queue.add_events(pending)
            self._pending_flagged = []

        cnt = len(events) + len(pending)
        self.statusBar().showMessage(
            f"Scoring complete — {cnt} suspicious event{'s' if cnt != 1 else ''} found",
            5000,
        )

    def _on_rescore_all(self) -> None:
        """Full rescore triggered by the user via the Rescore All button."""
        self._queue.show_rescore_button(False)
        self._run_scorer()
        self.statusBar().showMessage("Running full rescore\u2026", 3000)

    # ------------------------------------------------------------------
    # Event handling
    # ------------------------------------------------------------------

    def _on_event_selected(self, event: SuspicionEvent) -> None:
        """Seek video, highlight tracks, open the track editor."""
        self._player.seek_to(event.frame_peak)
        self._player.highlight_tracks(event.involved_tracks)
        self._timeline.highlight_event(event)

        self._show_track_editor(event)

    def _show_track_editor(self, event: SuspicionEvent) -> None:
        """Open the timeline-based TrackEditorDialog.

        - **Close / Cancel**: no changes, no rescoring.
        - **Apply**: execute edit ops, refresh trajectories, run a fast
          localized rescore around the affected tracks and frame range.

        A full rescore can be triggered at any time via the *Rescore All*
        button in the suspicion queue.
        """
        if self._video_path is None or self._df is None:
            return

        from hydra_suite.refinekit.gui.dialogs.track_editor_dialog import (
            TrackEditorDialog,
        )

        dlg = TrackEditorDialog(
            video_path=self._video_path,
            df=self._df,
            event=event,
            parent=self,
        )
        dlg.exec()

        if not dlg.applied or not dlg.edit_ops or self._writer is None:
            # User closed without applying — do nothing.
            return

        # --- Apply ---
        self._writer.apply_edit_ops(dlg.edit_ops)
        self._load_review_dataframe(self._writer.df)

        # Record reviewed region for deprioritisation
        rr = (dlg.reviewed_range[0], dlg.reviewed_range[1], dlg.reviewed_tracks)
        self._reviewed_regions.append(rr)
        if self._scorer is not None:
            self._scorer.add_reviewed_region(*rr)

        # Localized rescore: remove stale events, add fresh ones
        affected_tracks = list(event.involved_tracks)
        context = 50
        rescore_range = (
            max(0, event.frame_range[0] - context),
            event.frame_range[1] + context,
        )
        self._queue.remove_events_for_tracks(affected_tracks, rescore_range)

        if self._scorer is not None and self._df is not None:
            new_events = self._scorer.score_local(
                self._df,
                affected_tracks,
                event.frame_range,
                context_frames=context,
            )
            if new_events:
                self._queue.add_events(new_events)

        # Show the Rescore All button so the user can do a full pass later
        self._queue.show_rescore_button(True)
        self._queue.mark_resolved(event)

        self.statusBar().showMessage(
            f"Applied {len(dlg.edit_ops)} edit(s) — local rescore done",
            4000,
        )

    def _on_manual_split(self, track_id: int, frame: int) -> None:
        """Handle a manual split request from the timeline."""
        logger.info("Manual split requested: track %d at frame %d", track_id, frame)
        self.statusBar().showMessage(
            f"Click ‘Review region’ on the timeline (right-drag) "
            f"to edit T{track_id} around frame {frame}",
            4000,
        )

    def _on_manual_region_edit(self, frame_start: int, frame_end: int) -> None:
        """Open the track editor for a user-selected frame range.

        Shows the BboxSelectorDialog on the midpoint frame so the user can
        optionally draw a region of interest.  Creates a synthetic
        SuspicionEvent of type MANUAL and delegates to _show_track_editor.
        """
        if self._video_path is None or self._df is None:
            return

        # Cap the duration
        if frame_end - frame_start > _MAX_MANUAL_REGION:
            frame_end = frame_start + _MAX_MANUAL_REGION
            self.statusBar().showMessage(
                f"Region capped to {_MAX_MANUAL_REGION} frames", 3000
            )

        from PySide6.QtWidgets import QDialog as _QDialog

        from hydra_suite.refinekit.gui.dialogs.bbox_selector import BboxSelectorDialog

        mid_frame = (frame_start + frame_end) // 2
        bbox_dlg = BboxSelectorDialog(self._video_path, mid_frame, parent=self)
        if bbox_dlg.exec() != _QDialog.DialogCode.Accepted:
            return

        bbox = bbox_dlg.bbox
        region_df = self._df[self._df["FrameID"].between(frame_start, frame_end)]

        if bbox is not None:
            x1, y1, x2, y2 = bbox
            in_region = region_df[
                region_df["X"].between(x1, x2) & region_df["Y"].between(y1, y2)
            ]
        else:
            in_region = region_df

        involved = sorted(int(t) for t in in_region["TrajectoryID"].dropna().unique())
        if not involved:
            QMessageBox.information(
                self,
                "No tracks",
                "No tracks found in the selected region.\n"
                "Try a larger area or a wider frame range.",
            )
            return

        event = SuspicionEvent(
            event_type=EventType.MANUAL,
            involved_tracks=involved,
            frame_peak=mid_frame,
            frame_range=(frame_start, frame_end),
            score=1.0,
        )
        self._show_track_editor(event)

    def _on_manual_track_reassign(self, source_id: int, target_id: int) -> None:
        """Move an entire trajectory to another lane when no frames overlap."""
        if self._df is None or self._writer is None or source_id == target_id:
            return

        source_rows = self._df[self._df["TrajectoryID"] == source_id]
        target_rows = self._df[self._df["TrajectoryID"] == target_id]
        if source_rows.empty or target_rows.empty:
            return

        source_frames = {int(frame) for frame in source_rows["FrameID"].tolist()}
        target_frames = {int(frame) for frame in target_rows["FrameID"].tolist()}
        overlap_count = len(source_frames & target_frames)
        if overlap_count > 0:
            self.statusBar().showMessage(
                f"Cannot move T{source_id} to T{target_id}: {overlap_count} overlapping frame(s)",
                5000,
            )
            return

        from hydra_suite.refinekit.core.track_editor_model import EditOp, OpKind

        frame_start = int(source_rows["FrameID"].min())
        frame_end = int(source_rows["FrameID"].max())
        self._writer.apply_edit_ops(
            [
                EditOp(
                    kind=OpKind.REASSIGN,
                    track_id=source_id,
                    frame_start=frame_start,
                    frame_end=frame_end,
                    new_track_id=target_id,
                )
            ]
        )
        self._load_review_dataframe(self._writer.df)

        affected_tracks = [source_id, target_id]
        rescore_range = (
            frame_start,
            max(frame_end, int(target_rows["FrameID"].max())),
        )
        self._queue.remove_events_for_tracks(affected_tracks, rescore_range)
        if self._scorer is not None:
            new_events = self._scorer.score_local(
                self._df,
                affected_tracks,
                rescore_range,
                context_frames=50,
            )
            if new_events:
                self._queue.add_events(new_events)
            self._queue.show_rescore_button(True)

        self.statusBar().showMessage(
            f"Moved T{source_id} to T{target_id}",
            4000,
        )

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def keyPressEvent(self, event: QKeyEvent) -> None:  # noqa: N802
        """Handle Ctrl+O to open a video file and Ctrl+Q/W to quit the application."""
        key = event.key()
        mod = event.modifiers()
        ctrl = Qt.KeyboardModifier.ControlModifier
        if key == Qt.Key.Key_O and mod & ctrl:
            self._load_single_video()
            return
        if key in (Qt.Key.Key_Q, Qt.Key.Key_W) and mod & ctrl:
            self.close()
            return
        super().keyPressEvent(event)

    def closeEvent(self, event) -> None:  # noqa: N802
        """Disconnect the scorer worker and flush the correction writer before closing."""
        # Let any running scorer finish silently rather than blocking shutdown
        if self._scorer_worker is not None and self._scorer_worker.isRunning():
            try:
                self._scorer_worker.events_ready.disconnect()
            except RuntimeError:
                pass
        self._stop_kinematics_precompute()
        if self._writer is not None:
            self._writer.close()
            self._writer = None
        super().closeEvent(event)
