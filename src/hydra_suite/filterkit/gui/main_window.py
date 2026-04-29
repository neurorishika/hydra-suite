"""FilterKit main window."""

import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict

import cv2
from PySide6.QtCore import QRectF, QSize, Qt, QUrl, Signal
from PySide6.QtGui import QColor, QDesktopServices, QPainter, QPixmap
from PySide6.QtSvg import QSvgRenderer
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QFileDialog,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from hydra_suite.filterkit.config.schemas import FilterKitConfig
from hydra_suite.filterkit.core import FilterKitCore
from hydra_suite.utils.file_dialogs import HydraFileDialog as QFileDialog  # noqa: F811
from hydra_suite.widgets.workers import BaseWorker


class FilterWorker(BaseWorker):
    status = Signal(str)
    progress = Signal(int, str)
    finished = Signal(object)
    error = Signal(str)

    def __init__(self, dataset_path, config) -> None:
        super().__init__()
        self.dataset_path = dataset_path
        self.config = config
        self.core = FilterKitCore()
        self._is_running = True
        self.temp_dir = None

    def _sanitize_items_for_qt(self, items):
        sanitized = []
        for item in items:
            clean = {
                "path": str(item.get("path", "")),
                "filename": str(item.get("filename", "")),
                "det_id": int(item.get("det_id", 0)),
                "frame_idx": int(item.get("frame_idx", 0)),
                "det_idx": int(item.get("det_idx", 0)),
            }
            sanitized.append(clean)
        return sanitized

    def _collect_removed_examples(
        self, before_items, after_items, reason, max_keep=None
    ):
        after_paths = {x.get("path") for x in after_items}
        removed = []
        for item in before_items:
            path = item.get("path")
            if path not in after_paths:
                removed.append({"path": path, "reason": reason})
                if max_keep is not None and len(removed) >= max_keep:
                    break
        return removed

    def _emit_stage_progress(
        self,
        stage_start: int,
        stage_end: int,
        current: int,
        total: int,
        label: str,
    ):
        if total <= 0:
            pct = stage_start
        else:
            pct = stage_start + int((stage_end - stage_start) * (current / total))
        pct = max(0, min(100, pct))
        self.progress.emit(pct, f"{label}: {current:,}/{total:,}")

    def _quality_filter(self, dataset):
        min_blur = self.config.get("quality_min_blur", 30)
        min_contrast = self.config.get("quality_min_contrast", 20)

        kept = []
        total = len(dataset)
        for idx, item in enumerate(dataset, start=1):
            img = cv2.imread(item["path"], cv2.IMREAD_GRAYSCALE)
            if img is None:
                if idx == 1 or idx % 250 == 0 or idx == total:
                    self._emit_stage_progress(25, 40, idx, total, "Quality filter")
                continue
            blur_score = float(cv2.Laplacian(img, cv2.CV_64F).var())
            contrast_score = float(img.std())
            if blur_score >= min_blur and contrast_score >= min_contrast:
                kept.append(item)
            if idx == 1 or idx % 250 == 0 or idx == total:
                self._emit_stage_progress(25, 40, idx, total, "Quality filter")
        return kept

    def _should_abort(self):
        if self._is_running:
            return False
        self.stop()
        return True

    def _initialize_run_state(self):
        self.progress.emit(0, "Step 1/7: Loading dataset metadata")
        self.status.emit("Loading dataset...")
        _, dataset = self.core.load_images_from_root(self.dataset_path)
        loaded_count = len(dataset)
        self.progress.emit(10, f"Step 1/7 complete: loaded {loaded_count:,} images")
        stats = {
            "loaded": loaded_count,
            "after_quality": loaded_count,
            "after_temporal": loaded_count,
            "after_dedup": loaded_count,
            "after_diversity": loaded_count,
        }
        return dataset, stats, [], []

    def _extend_removed_examples(self, before, after, reason, removed_examples):
        removed_examples.extend(self._collect_removed_examples(before, after, reason))

    def _run_quality_stage(self, dataset, stats, removed_examples):
        if not self.config.get("quality_enabled"):
            stats["after_quality"] = len(dataset)
            self.progress.emit(40, "Step 3/7 skipped: quality filter disabled")
            return dataset

        self.progress.emit(25, "Step 3/7: Running quality filter")
        self.status.emit("Applying quality filter (blur + contrast)...")
        before = list(dataset)
        dataset = self._quality_filter(dataset)
        self._extend_removed_examples(before, dataset, "quality", removed_examples)
        stats["after_quality"] = len(dataset)
        self.status.emit(f"Remaining after quality: {len(dataset)}")
        self.progress.emit(
            40,
            f"Step 3/7 complete: quality {len(before):,} → {len(dataset):,}",
        )
        return dataset

    def _run_temporal_stage(self, dataset, stats, removed_examples):
        if not self.config.get("temporal_enabled"):
            self.progress.emit(55, "Step 4/7 skipped: temporal filter disabled")
            stats["after_temporal"] = len(dataset)
            return dataset

        interval = self.config.get("temporal_interval", 1)
        if interval <= 1:
            self.progress.emit(55, "Step 4/7 skipped: interval is 1")
            stats["after_temporal"] = len(dataset)
            return dataset

        self.progress.emit(40, "Step 4/7: Applying temporal subsampling")
        self.status.emit(f"Applying temporal subsampling (1/{interval})...")
        before = list(dataset)
        dataset = self.core.temporal_subsample(dataset, interval)
        self._extend_removed_examples(before, dataset, "temporal", removed_examples)
        self.status.emit(f"Remaining after temporal: {len(dataset)}")
        self.progress.emit(
            55,
            f"Step 4/7 complete: temporal {len(before):,} → {len(dataset):,}",
        )
        stats["after_temporal"] = len(dataset)
        return dataset

    def _run_dedup_stage(self, dataset, stats, removed_examples):
        if not self.config.get("dedup_enabled"):
            self.progress.emit(85, "Step 5/7 skipped: deduplication disabled")
            stats["after_dedup"] = len(dataset)
            return dataset, []

        method = self.config.get("dedup_method", "phash")
        raw_threshold = self.config.get("dedup_threshold", 2)
        threshold = (
            float(raw_threshold) / 100.0
            if method == "histogram"
            else float(raw_threshold)
        )
        preserve_color = bool(self.config.get("preserve_color_diversity", False))
        raw_color_threshold = self.config.get("color_threshold", 8)
        color_threshold = float(raw_color_threshold) / 100.0 if preserve_color else None

        self.progress.emit(55, f"Step 5/7: Deduplicating with {method}")
        self.status.emit(
            (
                f"Removing duplicates (method={method}, threshold={raw_threshold}, "
                f"preserve_color={'on' if preserve_color else 'off'})..."
            )
        )
        before = list(dataset)
        dataset, duplicate_clusters = self.core.deduplicate_by_hash(
            dataset,
            threshold=threshold,
            method=method,
            progress_callback=lambda cur, tot: self._emit_stage_progress(
                55, 85, cur, tot, "Deduplication"
            ),
            return_groups=True,
            color_threshold=color_threshold,
        )
        self._extend_removed_examples(before, dataset, "duplicate", removed_examples)
        self.status.emit(f"Remaining after deduplication: {len(dataset)}")
        self.progress.emit(
            85,
            f"Step 5/7 complete: dedup {len(before):,} → {len(dataset):,}",
        )
        stats["after_dedup"] = len(dataset)
        return dataset, duplicate_clusters

    def _run_diversity_stage(self, dataset, stats, removed_examples):
        if not self.config.get("diversity_enabled"):
            self.progress.emit(95, "Step 6/7 skipped: diversity sampling disabled")
            stats["after_diversity"] = len(dataset)
            return dataset

        target = self.config.get("diversity_target", 1000)
        if len(dataset) <= target:
            self.progress.emit(
                95,
                f"Step 6/7 skipped: current set ({len(dataset):,}) <= target ({target:,})",
            )
            stats["after_diversity"] = len(dataset)
            return dataset

        self.progress.emit(85, "Step 6/7: Running diversity sampling")
        self.status.emit(f"Applying diversity sampling (target={target})...")
        before = list(dataset)
        dataset = self.core.diversity_sample(dataset, target)
        self._extend_removed_examples(before, dataset, "diversity", removed_examples)
        self.status.emit(f"Remaining after diversity: {len(dataset)}")
        self.progress.emit(
            95,
            f"Step 6/7 complete: diversity {len(before):,} → {len(dataset):,}",
        )
        stats["after_diversity"] = len(dataset)
        return dataset

    def _finalize_result(self, dataset, stats, removed_examples, duplicate_clusters):
        for item in dataset:
            item.pop("dhash", None)
            item.pop("dedup_signature", None)
            item.pop("color_signature", None)
            item.pop("features", None)

        selected_sanitized = self._sanitize_items_for_qt(dataset)
        self.progress.emit(100, "Step 7/7 complete: Finalized and ready to review")
        return {
            "selected_dataset": selected_sanitized,
            "stats": stats,
            "removed_examples": removed_examples,
            "duplicate_clusters": duplicate_clusters,
            "config_used": dict(self.config),
        }

    def execute(self):
        try:
            dataset, stats, removed_examples, duplicate_clusters = (
                self._initialize_run_state()
            )
            if self._should_abort():
                return

            dataset = self._run_quality_stage(dataset, stats, removed_examples)
            if self._should_abort():
                return

            dataset = self._run_temporal_stage(dataset, stats, removed_examples)
            if self._should_abort():
                return

            dataset, duplicate_clusters = self._run_dedup_stage(
                dataset,
                stats,
                removed_examples,
            )
            if self._should_abort():
                return

            dataset = self._run_diversity_stage(dataset, stats, removed_examples)
            if self._should_abort():
                return

            self.finished.emit(
                self._finalize_result(
                    dataset,
                    stats,
                    removed_examples,
                    duplicate_clusters,
                )
            )

        except Exception as exc:
            self.error.emit(str(exc))
        finally:
            self.stop()

    def stop(self):
        self._is_running = False
        if self.temp_dir:
            shutil.rmtree(self.temp_dir, ignore_errors=True)
            self.temp_dir = None


class PreviewListWidget(QListWidget):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setIconSize(QSize(72, 72))
        self.setViewMode(QListWidget.IconMode)
        self.setResizeMode(QListWidget.Adjust)
        self.setSpacing(6)
        self.itemDoubleClicked.connect(self.open_image_externally)

    def open_image_externally(self, item):
        path = item.data(Qt.UserRole)
        if path and Path(path).exists():
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(path)))


class NoWheelSpinBox(QSpinBox):
    def wheelEvent(self, event) -> None:
        event.ignore()


class DuplicateClusterExplorer(QDialog):
    def __init__(self, clusters, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Duplicate Cluster Explorer")
        self.resize(1000, 700)

        self.clusters = [cluster for cluster in clusters if cluster.get("paths")]
        self.cluster_index = 0
        self.page_index = 0
        self.items_per_page = 120

        layout = QVBoxLayout(self)

        self.lbl_cluster = QLabel("No clusters available")
        self.lbl_cluster.setStyleSheet("font-weight: 700;")
        layout.addWidget(self.lbl_cluster)

        cluster_nav = QHBoxLayout()
        self.btn_prev_cluster = QPushButton("← Prev Cluster")
        self.btn_prev_cluster.clicked.connect(self.previous_cluster)
        cluster_nav.addWidget(self.btn_prev_cluster)

        self.lbl_cluster_page = QLabel("Cluster 1 of 1")
        self.lbl_cluster_page.setAlignment(Qt.AlignCenter)
        cluster_nav.addWidget(self.lbl_cluster_page)

        self.btn_next_cluster = QPushButton("Next Cluster →")
        self.btn_next_cluster.clicked.connect(self.next_cluster)
        cluster_nav.addWidget(self.btn_next_cluster)
        layout.addLayout(cluster_nav)

        self.list_images = PreviewListWidget()
        layout.addWidget(self.list_images)

        image_nav = QHBoxLayout()
        self.btn_prev_page = QPushButton("← Prev Page")
        self.btn_prev_page.clicked.connect(self.previous_page)
        image_nav.addWidget(self.btn_prev_page)

        self.lbl_image_page = QLabel("Page 1 of 1")
        self.lbl_image_page.setAlignment(Qt.AlignCenter)
        image_nav.addWidget(self.lbl_image_page)

        self.btn_next_page = QPushButton("Next Page →")
        self.btn_next_page.clicked.connect(self.next_page)
        image_nav.addWidget(self.btn_next_page)
        layout.addLayout(image_nav)

        self._refresh()

    def _current_cluster(self):
        if not self.clusters:
            return None
        return self.clusters[self.cluster_index]

    def _refresh(self):
        cluster = self._current_cluster()
        self.list_images.clear()

        if cluster is None:
            self.lbl_cluster.setText("No duplicate clusters found.")
            self.lbl_cluster_page.setText("Cluster 0 of 0")
            self.lbl_image_page.setText("Page 0 of 0")
            self.btn_prev_cluster.setEnabled(False)
            self.btn_next_cluster.setEnabled(False)
            self.btn_prev_page.setEnabled(False)
            self.btn_next_page.setEnabled(False)
            return

        paths = cluster.get("paths", [])
        count = len(paths)
        total_image_pages = max(
            1, (count + self.items_per_page - 1) // self.items_per_page
        )
        self.page_index = min(self.page_index, total_image_pages - 1)

        self.lbl_cluster.setText(
            f"Hash {cluster.get('hash', '')[:16]}... | {cluster.get('count', count)} images"
        )
        self.lbl_cluster_page.setText(
            f"Cluster {self.cluster_index + 1} of {len(self.clusters)}"
        )
        self.btn_prev_cluster.setEnabled(self.cluster_index > 0)
        self.btn_next_cluster.setEnabled(self.cluster_index < len(self.clusters) - 1)

        start = self.page_index * self.items_per_page
        end = min(start + self.items_per_page, count)
        self.lbl_image_page.setText(
            f"Page {self.page_index + 1} of {total_image_pages}"
        )
        self.btn_prev_page.setEnabled(self.page_index > 0)
        self.btn_next_page.setEnabled(self.page_index < total_image_pages - 1)

        for path in paths[start:end]:
            pixmap = QPixmap(path)
            if pixmap.isNull():
                continue
            icon = pixmap.scaled(72, 72, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            item = QListWidgetItem(Path(path).name)
            item.setIcon(icon)
            item.setData(Qt.UserRole, path)
            self.list_images.addItem(item)

    def previous_cluster(self):
        if self.cluster_index > 0:
            self.cluster_index -= 1
            self.page_index = 0
            self._refresh()

    def next_cluster(self):
        if self.cluster_index < len(self.clusters) - 1:
            self.cluster_index += 1
            self.page_index = 0
            self._refresh()

    def previous_page(self):
        if self.page_index > 0:
            self.page_index -= 1
            self._refresh()

    def next_page(self):
        cluster = self._current_cluster()
        if cluster is None:
            return
        count = len(cluster.get("paths", []))
        total_image_pages = max(
            1, (count + self.items_per_page - 1) // self.items_per_page
        )
        if self.page_index < total_image_pages - 1:
            self.page_index += 1
            self._refresh()


class FilterKitWindow(QMainWindow):
    TRANSACTION_FILE = ".filterkit_last_transaction.json"

    def __init__(self) -> None:
        super().__init__()
        self.config = FilterKitConfig()
        self.apply_stylesheet()
        self.setWindowTitle("FilterKit")
        self.resize(1400, 950)
        self.setMinimumSize(1300, 900)

        self.dataset_path = None
        self.dataset_root = None
        self._source_kind = "images"
        self.loaded_count = 0
        self.filtered_dataset = []
        self.removed_examples = []
        self.pipeline_stats = {}
        self.duplicate_clusters = []

        self.current_page = 0
        self.images_per_page = 200
        self.removed_page = 0
        self.removed_per_page = 150

        self._set_window_icon()
        self.init_ui()
        self._update_rollback_availability()

    def apply_stylesheet(self):
        """Apply VSCode dark theme (consistent with MAT, PoseKit, ClassKit)."""
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
            QPushButton {
                background-color: #0e639c;
                color: #ffffff;
                border: none;
                border-radius: 4px;
                padding: 6px 14px;
                font-weight: 500;
                min-height: 22px;
            }
            QPushButton:hover { background-color: #1177bb; }
            QPushButton:pressed { background-color: #0d5a8f; }
            QPushButton:disabled { background-color: #3e3e42; color: #777777; }
            QCheckBox { color: #cccccc; spacing: 8px; }
            QCheckBox::indicator {
                width: 14px;
                height: 14px;
                border: 1px solid #3e3e42;
                border-radius: 3px;
                background-color: #3c3c3c;
            }
            QCheckBox::indicator:checked { background-color: #0e639c; border-color: #007acc; }
            QCheckBox::indicator:hover { border-color: #007acc; }
            QLabel { color: #cccccc; background-color: transparent; }
            QSpinBox, QDoubleSpinBox, QComboBox {
                background-color: #3c3c3c;
                color: #e0e0e0;
                border: 1px solid #3e3e42;
                border-radius: 4px;
                padding: 4px 8px;
                min-height: 22px;
            }
            QSpinBox:hover, QDoubleSpinBox:hover, QComboBox:hover { border-color: #0e639c; }
            QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus { border-color: #007acc; }
            QComboBox::drop-down { border: none; width: 20px; }
            QComboBox QAbstractItemView {
                background-color: #252526;
                border: 1px solid #3e3e42;
                selection-background-color: #094771;
                selection-color: #ffffff;
                outline: none;
            }
            QScrollArea { border: none; background-color: transparent; }
            QScrollBar:vertical {
                background-color: #252526; width: 10px; border-radius: 5px; margin: 0px;
            }
            QScrollBar::handle:vertical {
                background-color: #5a5a5a; border-radius: 5px; min-height: 24px;
            }
            QScrollBar::handle:vertical:hover { background-color: #007acc; }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0px; }
            QScrollBar:horizontal {
                background-color: #252526; height: 10px; border-radius: 5px; margin: 0px;
            }
            QScrollBar::handle:horizontal {
                background-color: #5a5a5a; border-radius: 5px; min-width: 24px;
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
            QStatusBar {
                background-color: #007acc;
                color: #ffffff;
                border-top: 1px solid #0098ff;
                font-weight: 500;
                font-size: 11px;
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
        """)

    def _set_window_icon(self):
        try:
            from hydra_suite.paths import get_brand_qicon

            icon = get_brand_qicon("filterkit.svg")
            if icon and not icon.isNull():
                self.setWindowIcon(icon)
        except Exception:
            pass

    def init_ui(self):
        self._content_stack = QStackedWidget()
        self.setCentralWidget(self._content_stack)
        self._content_stack.addWidget(self._make_welcome_page())  # index 0

        central = QWidget()
        self._content_stack.addWidget(central)  # index 1
        main_layout = QHBoxLayout(central)

        left_content = QWidget()
        left_layout = QVBoxLayout(left_content)

        left_scroll = QScrollArea()
        left_scroll.setWidgetResizable(True)
        left_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        left_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        left_scroll.setWidget(left_content)
        left_scroll.setFixedWidth(540)

        right = QWidget()
        right_layout = QVBoxLayout(right)
        right.setStyleSheet("background-color: #000000;")

        self._build_load_group(left_layout)
        self._build_strategy_group(left_layout)
        self._build_run_group(left_layout)
        self._build_status_group(left_layout)
        left_layout.addStretch()

        self._build_preview_group(right_layout)

        main_layout.addWidget(left_scroll)
        main_layout.addWidget(right)

        # Startup mode begins on splash until a valid dataset is loaded.
        self._content_stack.setCurrentIndex(0)

    def _make_welcome_page(self):
        """Logo/welcome screen shown before a dataset is loaded."""
        from hydra_suite.widgets import (
            ButtonDef,
            RecentItemsStore,
            WelcomeConfig,
            WelcomePage,
        )

        store = RecentItemsStore("filterkit")
        self._recents_store = store

        config = WelcomeConfig(
            logo_svg="filterkit.svg",
            tagline="Dataset Subsampling Workspace",
            buttons=[
                ButtonDef(
                    label="Load Dataset Folder\u2026", callback=self.load_dataset_dialog
                ),
                ButtonDef(label="Quit", callback=self.close),
            ],
            recents_label="Recent Datasets",
            recents_store=store,
            on_recent_clicked=self._open_recent_dataset,
        )
        self._welcome_page = WelcomePage(config)
        return self._welcome_page

    def _open_recent_dataset(self, path: str):
        """Open a dataset folder from the recent items list."""
        from pathlib import Path

        dataset_path = Path(path)
        if dataset_path.exists():
            self.load_dataset_root(dataset_path)
        else:
            from PySide6.QtWidgets import QMessageBox

            QMessageBox.warning(self, "Not Found", f"Dataset not found:\n{path}")
            if hasattr(self, "_recents_store"):
                self._recents_store.remove(path)
                if hasattr(self, "_welcome_page"):
                    self._welcome_page.refresh_recents()

    def _build_load_group(self, parent_layout):
        group = QGroupBox("1) Load Dataset")
        layout = QVBoxLayout(group)

        self.lbl_path = QLabel("No folder selected")
        self.lbl_path.setWordWrap(True)
        layout.addWidget(self.lbl_path)

        btn_load = QPushButton("Load Dataset Folder")
        btn_load.clicked.connect(self.load_dataset_dialog)
        layout.addWidget(btn_load)

        self.lbl_load_help = QLabel(
            "Pick your dataset root folder. If it contains an images/ folder, FilterKit uses it automatically."
        )
        self.lbl_load_help.setWordWrap(True)
        self.lbl_load_help.setStyleSheet("color: #6a6a6a;")
        layout.addWidget(self.lbl_load_help)

        parent_layout.addWidget(group)

    def _build_strategy_group(self, parent_layout):
        group = QGroupBox("2) Choose Sampling Strategy")
        layout = QVBoxLayout(group)

        preset_grid = QGridLayout()
        self.btn_preset_fast = QPushButton("Fast")
        self.btn_preset_balanced = QPushButton("Balanced")
        self.btn_preset_diverse = QPushButton("Max Diversity")
        self.btn_preset_mvp = QPushButton("Labeling MVP")
        self.btn_recommend = QPushButton("Recommend")

        self.btn_preset_fast.clicked.connect(lambda: self.apply_preset("fast"))
        self.btn_preset_balanced.clicked.connect(lambda: self.apply_preset("balanced"))
        self.btn_preset_diverse.clicked.connect(lambda: self.apply_preset("diverse"))
        self.btn_preset_mvp.clicked.connect(lambda: self.apply_preset("mvp"))
        self.btn_recommend.clicked.connect(self.recommend_settings)

        for button in [
            self.btn_preset_fast,
            self.btn_preset_balanced,
            self.btn_preset_diverse,
            self.btn_preset_mvp,
            self.btn_recommend,
        ]:
            button.setMinimumWidth(120)

        preset_grid.addWidget(self.btn_preset_fast, 0, 0)
        preset_grid.addWidget(self.btn_preset_balanced, 0, 1)
        preset_grid.addWidget(self.btn_preset_diverse, 0, 2)
        preset_grid.addWidget(self.btn_preset_mvp, 1, 0, 1, 2)
        preset_grid.addWidget(self.btn_recommend, 1, 2)
        layout.addLayout(preset_grid)

        self.lbl_preset_help = QLabel(
            "Presets set practical defaults. Use Recommend for dataset-size-aware values."
        )
        self.lbl_preset_help.setWordWrap(True)
        self.lbl_preset_help.setStyleSheet("color: #6a6a6a;")
        layout.addWidget(self.lbl_preset_help)

        self.chk_temporal = QCheckBox("Keep every Nth frame")
        self.spin_temporal = NoWheelSpinBox()
        self.spin_temporal.setRange(1, 1000)
        self.spin_temporal.setValue(10)
        self.spin_temporal.setSuffix(" frames")
        layout.addWidget(self.chk_temporal)
        layout.addWidget(QLabel("Frame interval:"))
        layout.addWidget(self.spin_temporal)
        self._add_help(
            layout,
            "Reduces frame-to-frame redundancy. Higher interval keeps fewer images and speeds labeling.",
        )

        self.chk_dedup = QCheckBox("Remove near-identical images")
        self.spin_dedup = NoWheelSpinBox()
        self.spin_dedup.setRange(0, 16)
        self.spin_dedup.setValue(2)
        self.spin_dedup.setSuffix(" bits")
        layout.addWidget(self.chk_dedup)
        self.lbl_dedup_threshold = QLabel("Similarity threshold (Hamming bits):")
        layout.addWidget(self.lbl_dedup_threshold)
        layout.addWidget(self.spin_dedup)
        self._add_help(
            layout,
            "pHash is the default and usually the best strict dedup baseline. Lower thresholds are stricter.",
        )
        self.chk_show_advanced_dedup = QCheckBox("Show advanced deduplication settings")
        self.chk_show_advanced_dedup.setChecked(False)
        layout.addWidget(self.chk_show_advanced_dedup)

        self.advanced_dedup_widget = QWidget()
        advanced_dedup_layout = QVBoxLayout(self.advanced_dedup_widget)
        advanced_dedup_layout.setContentsMargins(16, 0, 0, 0)

        self.cmb_dedup_method = QComboBox()
        self.cmb_dedup_method.addItem("pHash (recommended)", "phash")
        self.cmb_dedup_method.addItem("dHash", "dhash")
        self.cmb_dedup_method.addItem("aHash", "ahash")
        self.cmb_dedup_method.addItem("Histogram (grayscale)", "histogram")
        advanced_dedup_layout.addWidget(QLabel("Dedup method:"))
        advanced_dedup_layout.addWidget(self.cmb_dedup_method)
        self._add_help(
            advanced_dedup_layout,
            "Method guide: pHash = best default for strict visual dedup; dHash = fast edge-based; aHash = brightness-average based; Histogram = tone-distribution similarity.",
        )

        self.chk_preserve_color = QCheckBox(
            "Preserve color diversity (for color-tagged animals)"
        )
        self.spin_color_threshold = NoWheelSpinBox()
        self.spin_color_threshold.setRange(1, 100)
        self.spin_color_threshold.setValue(8)
        self.spin_color_threshold.setSuffix(" /100")
        advanced_dedup_layout.addWidget(self.chk_preserve_color)
        advanced_dedup_layout.addWidget(
            QLabel("Color similarity threshold (lower = stricter):")
        )
        advanced_dedup_layout.addWidget(self.spin_color_threshold)
        self._add_help(
            advanced_dedup_layout,
            "If enabled, two crops are merged only when both structure and color are similar. This helps preserve distinct color tags.",
        )

        self.advanced_dedup_widget.setVisible(False)
        layout.addWidget(self.advanced_dedup_widget)

        self.chk_diversity = QCheckBox("Keep a diverse representative subset")
        self.spin_diversity = NoWheelSpinBox()
        self.spin_diversity.setRange(10, 100000)
        self.spin_diversity.setValue(1000)
        self.spin_diversity.setSingleStep(100)
        layout.addWidget(self.chk_diversity)
        layout.addWidget(QLabel("Target selected image count:"))
        layout.addWidget(self.spin_diversity)
        self._add_help(
            layout,
            "Uses clustering to preserve visual variety. Set to your expected labeling budget.",
        )

        self.chk_quality = QCheckBox("Drop low-quality crops")
        self.spin_quality_blur = NoWheelSpinBox()
        self.spin_quality_blur.setRange(1, 500)
        self.spin_quality_blur.setValue(30)
        self.spin_quality_blur.setSuffix(" blur")
        self.spin_quality_contrast = NoWheelSpinBox()
        self.spin_quality_contrast.setRange(1, 255)
        self.spin_quality_contrast.setValue(20)
        self.spin_quality_contrast.setSuffix(" contrast")
        layout.addWidget(self.chk_quality)
        layout.addWidget(QLabel("Minimum sharpness (Laplacian variance):"))
        layout.addWidget(self.spin_quality_blur)
        layout.addWidget(QLabel("Minimum contrast (pixel std-dev):"))
        layout.addWidget(self.spin_quality_contrast)
        self._add_help(
            layout,
            "Filters blurry or low-contrast crops before sampling. Useful on noisy extraction pipelines.",
        )

        self.lbl_estimate = QLabel("Estimated kept images: n/a")
        self.lbl_estimate.setStyleSheet("font-weight: 600;")
        layout.addWidget(self.lbl_estimate)

        parent_layout.addWidget(group)

        self.chk_temporal.stateChanged.connect(self.update_live_estimate)
        self.chk_dedup.stateChanged.connect(self.update_live_estimate)
        self.chk_show_advanced_dedup.toggled.connect(
            self.advanced_dedup_widget.setVisible
        )
        self.cmb_dedup_method.currentIndexChanged.connect(self._on_dedup_method_changed)
        self.chk_preserve_color.stateChanged.connect(self.update_live_estimate)
        self.chk_diversity.stateChanged.connect(self.update_live_estimate)
        self.chk_quality.stateChanged.connect(self.update_live_estimate)
        self.spin_temporal.valueChanged.connect(self.update_live_estimate)
        self.spin_dedup.valueChanged.connect(self.update_live_estimate)
        self.spin_color_threshold.valueChanged.connect(self.update_live_estimate)
        self.spin_diversity.valueChanged.connect(self.update_live_estimate)
        self.spin_quality_blur.valueChanged.connect(self.update_live_estimate)
        self.spin_quality_contrast.valueChanged.connect(self.update_live_estimate)
        self._on_dedup_method_changed()

    def _on_dedup_method_changed(self):
        method = self.cmb_dedup_method.currentData()
        if method == "histogram":
            self.spin_dedup.setRange(1, 100)
            if self.spin_dedup.value() > 100:
                self.spin_dedup.setValue(8)
            self.spin_dedup.setSuffix(" /100")
            self.lbl_dedup_threshold.setText(
                "Similarity threshold (Bhattacharyya distance):"
            )
        else:
            self.spin_dedup.setRange(0, 16)
            if self.spin_dedup.value() > 16:
                self.spin_dedup.setValue(2)
            self.spin_dedup.setSuffix(" bits")
            self.lbl_dedup_threshold.setText("Similarity threshold (Hamming bits):")
        self.update_live_estimate()

    def _build_run_group(self, parent_layout):
        group = QGroupBox("3) Run & Review")
        layout = QVBoxLayout(group)

        self.btn_run = QPushButton("Run Filter")
        self.btn_run.setStyleSheet("font-weight: 700; padding: 8px;")
        self.btn_run.clicked.connect(self.run_sieve)
        layout.addWidget(self.btn_run)

        self.btn_cancel = QPushButton("\u25a0  Stop")
        self.btn_cancel.setEnabled(False)
        self.btn_cancel.setStyleSheet(
            "background-color: #c0392b; color: white; font-weight: 700; padding: 8px;"
        )
        self.btn_cancel.clicked.connect(self._cancel_sieve)
        layout.addWidget(self.btn_cancel)

        self.btn_show_clusters = QPushButton("View Duplicate Clusters")
        self.btn_show_clusters.setEnabled(False)
        self.btn_show_clusters.clicked.connect(self.show_duplicate_clusters)
        layout.addWidget(self.btn_show_clusters)

        self.chk_compare_mode = QCheckBox("Compare mode: show removed examples")
        self.chk_compare_mode.stateChanged.connect(self.toggle_compare_mode)
        layout.addWidget(self.chk_compare_mode)

        self.lbl_summary = QLabel(
            "Run the sieve to see before/after counts and removal reasons."
        )
        self.lbl_summary.setWordWrap(True)
        self.lbl_summary.setStyleSheet(
            "color: #e0e0e0; background-color: #0d3354; padding: 8px; border-radius: 6px;"
        )
        layout.addWidget(self.lbl_summary)

        parent_layout.addWidget(group)

    def _build_status_group(self, parent_layout):
        group = QGroupBox("4) Process Dataset")
        layout = QVBoxLayout(group)

        self.btn_process = QPushButton("Process Dataset")
        self.btn_process.setEnabled(False)
        self.btn_process.clicked.connect(self.process_dataset)
        layout.addWidget(self.btn_process)

        self.btn_rollback = QPushButton("Rollback Last Process")
        self.btn_rollback.clicked.connect(self.rollback_last_process)
        self.btn_rollback.setEnabled(False)
        layout.addWidget(self.btn_rollback)

        self.lbl_process_help = QLabel(
            "Process renames images/ → all_images/ and creates a new images/ with selected files. Rollback restores the previous state."
        )
        self.lbl_process_help.setWordWrap(True)
        self.lbl_process_help.setStyleSheet("color: #6a6a6a;")
        layout.addWidget(self.lbl_process_help)

        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        self.progress.setFormat("%p%")
        self.lbl_progress_details = QLabel("Progress details will appear here.")
        self.lbl_progress_details.setWordWrap(True)
        self.lbl_progress_details.setStyleSheet("color: #6a6a6a;")
        self.lbl_status = QLabel("Ready")
        layout.addWidget(self.progress)
        layout.addWidget(self.lbl_progress_details)
        layout.addWidget(self.lbl_status)

        parent_layout.addWidget(group)

    def _build_preview_group(self, right_layout):
        self.lbl_preview_info = QLabel("No dataset loaded.")
        self.lbl_preview_info.setStyleSheet("color: #e0e0e0;")
        right_layout.addWidget(self.lbl_preview_info)

        pagination_row = QHBoxLayout()
        self.btn_prev_page = QPushButton("← Previous")
        self.btn_prev_page.clicked.connect(self.previous_page)
        self.btn_prev_page.setEnabled(False)
        pagination_row.addWidget(self.btn_prev_page)

        self.lbl_page_info = QLabel("Page 1 of 1")
        self.lbl_page_info.setAlignment(Qt.AlignCenter)
        self.lbl_page_info.setStyleSheet("color: #e0e0e0;")
        pagination_row.addWidget(self.lbl_page_info)

        self.btn_next_page = QPushButton("Next →")
        self.btn_next_page.clicked.connect(self.next_page)
        self.btn_next_page.setEnabled(False)
        pagination_row.addWidget(self.btn_next_page)

        pagination_container = QWidget()
        pagination_container.setLayout(pagination_row)
        self.pagination_widget = pagination_container
        self.pagination_widget.hide()
        right_layout.addWidget(self.pagination_widget)

        self.logo_label = QLabel()
        self.logo_label.setAlignment(Qt.AlignCenter)
        self._show_logo()
        right_layout.addWidget(self.logo_label, 1)

        self.list_preview = PreviewListWidget()
        right_layout.addWidget(self.list_preview, 2)

        self.lbl_removed = QLabel("Removed examples")
        self.lbl_removed.setStyleSheet("color: #e0e0e0;")
        self.lbl_removed.hide()
        right_layout.addWidget(self.lbl_removed)

        self.list_removed = PreviewListWidget()
        self.list_removed.hide()
        right_layout.addWidget(self.list_removed)

        self.removed_pagination = QWidget()
        removed_pagination_row = QHBoxLayout(self.removed_pagination)
        removed_pagination_row.setContentsMargins(0, 0, 0, 0)

        self.btn_removed_prev = QPushButton("← Prev Removed")
        self.btn_removed_prev.clicked.connect(self.previous_removed_page)
        removed_pagination_row.addWidget(self.btn_removed_prev)

        self.lbl_removed_page = QLabel("Removed Page 1 of 1")
        self.lbl_removed_page.setAlignment(Qt.AlignCenter)
        self.lbl_removed_page.setStyleSheet("color: #e0e0e0;")
        removed_pagination_row.addWidget(self.lbl_removed_page)

        self.btn_removed_next = QPushButton("Next Removed →")
        self.btn_removed_next.clicked.connect(self.next_removed_page)
        removed_pagination_row.addWidget(self.btn_removed_next)

        self.removed_pagination.hide()
        right_layout.addWidget(self.removed_pagination)

    def _add_help(self, layout, text):
        label = QLabel(text)
        label.setWordWrap(True)
        label.setStyleSheet("color: #6a6a6a;")
        layout.addWidget(label)

    def _show_logo(self):
        try:
            from PySide6.QtCore import QByteArray

            from hydra_suite.paths import get_brand_icon_bytes

            logo_data = get_brand_icon_bytes("filterkit.svg")

            canvas_w = max(500, self.logo_label.width() or 800)
            canvas_h = max(320, self.logo_label.height() or 520)
            canvas = QPixmap(canvas_w, canvas_h)
            canvas.fill(QColor(0, 0, 0))

            renderer = (
                QSvgRenderer(QByteArray(logo_data)) if logo_data else QSvgRenderer()
            )
            if renderer.isValid():
                view_box = renderer.viewBoxF()
                if view_box.isEmpty():
                    default_size = renderer.defaultSize()
                    view_box = QRectF(
                        0,
                        0,
                        max(1, default_size.width()),
                        max(1, default_size.height()),
                    )

                max_w = int(canvas_w * 0.72)
                max_h = int(canvas_h * 0.72)
                scale = min(max_w / view_box.width(), max_h / view_box.height())
                draw_w = int(view_box.width() * scale)
                draw_h = int(view_box.height() * scale)
                x = (canvas_w - draw_w) // 2
                y = (canvas_h - draw_h) // 2

                painter = QPainter(canvas)
                painter.setRenderHint(QPainter.Antialiasing, True)
                painter.setRenderHint(QPainter.SmoothPixmapTransform, True)
                renderer.render(painter, QRectF(x, y, draw_w, draw_h))
                painter.end()
                self.logo_label.setPixmap(canvas)
                self.logo_label.show()
                return
        except Exception:
            pass

        self.logo_label.setText("FilterKit")
        self.logo_label.setStyleSheet(
            "color: white; font-size: 28px; font-weight: 700;"
        )
        self.logo_label.show()

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        if hasattr(self, "logo_label") and self.logo_label.isVisible():
            self._show_logo()

    def _update_rollback_availability(self):
        if not self.dataset_path:
            self.btn_rollback.setEnabled(False)
            return

        dataset_root = Path(self.dataset_path)
        if getattr(self, "_source_kind", "images") == "images":
            images_path = dataset_root / "images"
            all_images_path = dataset_root / "all_images"
            self.btn_rollback.setEnabled(
                images_path.exists() and all_images_path.exists()
            )
        else:
            tx_path = self._transaction_path(dataset_root)
            if not tx_path.exists():
                self.btn_rollback.setEnabled(False)
                return
            try:
                tx = json.loads(tx_path.read_text(encoding="utf-8"))
                output_path = tx.get("output_path", "")
                self.btn_rollback.setEnabled(
                    bool(output_path) and Path(output_path).exists()
                )
            except Exception:
                self.btn_rollback.setEnabled(False)

    def load_dataset_dialog(self):
        from hydra_suite.paths import get_projects_dir

        folder = QFileDialog.getExistingDirectory(
            self, "Select Dataset Folder", str(get_projects_dir())
        )
        if not folder:
            return

        self.load_dataset_root(Path(folder))

    def load_dataset_root(self, root: Path, show_errors: bool = True) -> bool:
        root = Path(root).resolve()
        if root.name == "images" and root.is_dir():
            root = root.parent

        source_kind, items = FilterKitCore().load_images_from_root(root)
        if not items:
            if show_errors:
                QMessageBox.warning(
                    self,
                    "Invalid Dataset Folder",
                    "No supported images found in the selected folder.\n\n"
                    "Supported formats:\n"
                    "  • Flat image folder or images/ subdirectory\n"
                    "  • COCO JSON dataset\n"
                    "  • YOLO OBB / detect dataset\n"
                    "  • Train/val class-folder dataset",
                )
            return False

        kind_label = {
            "images": "flat / images/",
            "coco": "COCO JSON",
            "yolo_obb": "YOLO OBB/detect",
            "class_folders": "class-folder",
        }.get(source_kind, source_kind)

        self.dataset_path = str(root)
        self.dataset_root = root
        self._source_kind = source_kind
        self.lbl_path.setText(f"{root.name}  —  {kind_label}")

        self.loaded_count = len(items)
        self.lbl_status.setText(
            f"Dataset loaded. Found {self.loaded_count} image(s). Configure strategy and run sieve."
        )
        self.lbl_preview_info.setText(f"Loaded: {self.loaded_count} images")
        self.progress.setValue(0)
        self.progress.setFormat("%p%")
        self.lbl_progress_details.setText("Waiting to start...")
        self.update_live_estimate()
        self._update_rollback_availability()
        if hasattr(self, "_recents_store"):
            self._recents_store.add(str(root))
        if hasattr(self, "_content_stack"):
            self._content_stack.setCurrentIndex(1)
        return True

    def apply_preset(self, name):
        presets = {
            "fast": {
                "temporal": (True, 15),
                "dedup": (True, 0),
                "diversity": (False, 1000),
                "quality": (False, 30, 20),
            },
            "balanced": {
                "temporal": (True, 8),
                "dedup": (True, 1),
                "diversity": (True, 1500),
                "quality": (False, 30, 20),
            },
            "diverse": {
                "temporal": (True, 5),
                "dedup": (True, 1),
                "diversity": (True, 3000),
                "quality": (False, 30, 20),
            },
            "mvp": {
                "temporal": (True, 10),
                "dedup": (True, 0),
                "diversity": (True, 1000),
                "quality": (True, 30, 20),
            },
        }
        cfg = presets[name]
        self.chk_temporal.setChecked(cfg["temporal"][0])
        self.spin_temporal.setValue(cfg["temporal"][1])
        self.chk_dedup.setChecked(cfg["dedup"][0])
        self.cmb_dedup_method.setCurrentIndex(0)
        self._on_dedup_method_changed()
        self.spin_dedup.setValue(cfg["dedup"][1])
        self.chk_preserve_color.setChecked(False)
        self.spin_color_threshold.setValue(8)
        self.chk_diversity.setChecked(cfg["diversity"][0])
        self.spin_diversity.setValue(cfg["diversity"][1])
        self.chk_quality.setChecked(cfg["quality"][0])
        self.spin_quality_blur.setValue(cfg["quality"][1])
        self.spin_quality_contrast.setValue(cfg["quality"][2])
        self.update_live_estimate()
        self.lbl_status.setText(f"Applied preset: {name}.")

    def recommend_settings(self):
        if self.loaded_count <= 0:
            QMessageBox.information(
                self, "Recommendation", "Load a dataset first to get recommendations."
            )
            return

        n = self.loaded_count
        if n > 120000:
            interval = 15
            diversity_target = 2500
        elif n > 50000:
            interval = 10
            diversity_target = 2000
        elif n > 20000:
            interval = 6
            diversity_target = 1500
        else:
            interval = 3
            diversity_target = max(500, min(1200, n // 5))

        self.chk_temporal.setChecked(True)
        self.spin_temporal.setValue(interval)
        self.chk_dedup.setChecked(True)
        self.cmb_dedup_method.setCurrentIndex(0)
        self._on_dedup_method_changed()
        self.spin_dedup.setValue(2)
        self.chk_preserve_color.setChecked(True)
        self.spin_color_threshold.setValue(8)
        self.chk_diversity.setChecked(True)
        self.spin_diversity.setValue(diversity_target)
        self.chk_quality.setChecked(False)

        self.update_live_estimate()
        self.lbl_status.setText(
            f"Recommended settings applied for {n} images. Review and run sieve."
        )

    def _estimate_keep_count(self):
        if self.loaded_count <= 0:
            return None

        est = float(self.loaded_count)

        if self.chk_quality.isChecked():
            est *= 0.9

        if self.chk_temporal.isChecked() and self.spin_temporal.value() > 1:
            est /= float(self.spin_temporal.value())

        if self.chk_dedup.isChecked():
            # Rough heuristic: stronger threshold removes more
            threshold = self.spin_dedup.value()
            method = self.cmb_dedup_method.currentData()
            if method == "histogram":
                est *= max(0.45, 0.95 - 0.003 * threshold)
            else:
                est *= max(0.45, 0.96 - 0.03 * threshold)

            if self.chk_preserve_color.isChecked():
                est *= 1.12

        if self.chk_diversity.isChecked():
            est = min(est, float(self.spin_diversity.value()))

        return max(1, int(est))

    def update_live_estimate(self):
        est = self._estimate_keep_count()
        if est is None:
            self.lbl_estimate.setText("Estimated kept images: n/a")
        else:
            self.lbl_estimate.setText(
                f"Estimated kept images: ~{est:,} / {self.loaded_count:,}"
            )

    def run_sieve(self):
        if not self.dataset_path:
            QMessageBox.warning(self, "Error", "Please load a dataset folder first.")
            return

        config = {
            "temporal_enabled": self.chk_temporal.isChecked(),
            "temporal_interval": self.spin_temporal.value(),
            "dedup_enabled": self.chk_dedup.isChecked(),
            "dedup_method": self.cmb_dedup_method.currentData(),
            "dedup_threshold": self.spin_dedup.value(),
            "preserve_color_diversity": self.chk_preserve_color.isChecked(),
            "color_threshold": self.spin_color_threshold.value(),
            "diversity_enabled": self.chk_diversity.isChecked(),
            "diversity_target": self.spin_diversity.value(),
            "quality_enabled": self.chk_quality.isChecked(),
            "quality_min_blur": self.spin_quality_blur.value(),
            "quality_min_contrast": self.spin_quality_contrast.value(),
        }

        self.btn_process.setEnabled(False)
        self.btn_show_clusters.setEnabled(False)
        self.filtered_dataset = []
        self.removed_examples = []
        self.duplicate_clusters = []
        self.list_preview.clear()
        self.list_removed.clear()
        self.logo_label.hide()

        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        self.progress.setFormat("%p%")
        self.lbl_progress_details.setText("Initializing pipeline...")
        self.lbl_status.setText("Running sieve pipeline...")

        self.btn_run.setEnabled(False)
        self.btn_cancel.setEnabled(True)

        self.worker = FilterWorker(self.dataset_path, config)
        self.worker.status.connect(self.lbl_status.setText)
        self.worker.progress.connect(self.on_sieve_progress)
        self.worker.finished.connect(self.on_sieve_finished)
        self.worker.error.connect(self.on_sieve_error)
        self.worker.start()

    def on_sieve_progress(self, percent, details):
        self.progress.setValue(max(0, min(100, int(percent))))
        self.progress.setFormat(f"{int(percent)}%")
        self.lbl_progress_details.setText(details)

    def on_sieve_finished(self, payload):
        self.btn_run.setEnabled(True)
        self.btn_cancel.setEnabled(False)

        self.progress.setRange(0, 100)
        self.progress.setValue(100)

        self.filtered_dataset = payload.get("selected_dataset", [])
        self.pipeline_stats = payload.get("stats", {})
        self.removed_examples = payload.get("removed_examples", [])
        self.duplicate_clusters = payload.get("duplicate_clusters", [])

        self.current_page = 0
        self.removed_page = 0
        self.pagination_widget.show()
        self.btn_process.setEnabled(bool(self.filtered_dataset))
        self.btn_show_clusters.setEnabled(bool(self.duplicate_clusters))

        selected_count = len(self.filtered_dataset)
        self.lbl_status.setText(f"Finished. Selected {selected_count} images.")
        self.lbl_progress_details.setText(
            f"Completed: selected {selected_count:,} images. Explore results below."
        )
        self.lbl_preview_info.setText(
            f"Selected: {selected_count} image(s). Double-click to open in native viewer."
        )

        self._update_summary_card()
        self.load_previews()
        self.load_removed_examples()

    def on_sieve_error(self, msg):
        self.btn_run.setEnabled(True)
        self.btn_cancel.setEnabled(False)

        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        self.progress.setFormat("Error")
        self.lbl_progress_details.setText("Pipeline failed before completion.")
        self.lbl_status.setText("Error occurred.")
        QMessageBox.critical(self, "Error", msg)

    def _cancel_sieve(self):
        if self.worker is not None and self.worker.isRunning():
            self.worker.stop()
        self.btn_run.setEnabled(True)
        self.btn_cancel.setEnabled(False)
        self.progress.setValue(0)
        self.progress.setFormat("%p%")
        self.lbl_status.setText("Cancelled.")
        self.lbl_progress_details.setText("Filter run cancelled by user.")

    def _update_summary_card(self):
        s = self.pipeline_stats
        if not s:
            self.lbl_summary.setText("No run summary available.")
            return

        loaded = s.get("loaded", 0)
        final_count = s.get("after_diversity", 0)
        removed = max(0, loaded - final_count)

        self.lbl_summary.setText(
            "\n".join(
                [
                    "Run Summary:",
                    f"• Loaded: {loaded:,}",
                    f"• After quality: {s.get('after_quality', loaded):,}",
                    f"• After temporal: {s.get('after_temporal', loaded):,}",
                    f"• After dedup: {s.get('after_dedup', loaded):,}",
                    f"• After diversity: {final_count:,}",
                    f"• Total removed: {removed:,}",
                ]
            )
        )

    def load_previews(self):
        self.list_preview.clear()
        total_images = len(self.filtered_dataset)
        if total_images == 0:
            self.lbl_page_info.setText("Page 1 of 1")
            self.btn_prev_page.setEnabled(False)
            self.btn_next_page.setEnabled(False)
            return

        total_pages = (total_images + self.images_per_page - 1) // self.images_per_page
        start_idx = self.current_page * self.images_per_page
        end_idx = min(start_idx + self.images_per_page, total_images)

        self.lbl_page_info.setText(f"Page {self.current_page + 1} of {total_pages}")
        self.btn_prev_page.setEnabled(self.current_page > 0)
        self.btn_next_page.setEnabled(self.current_page < total_pages - 1)

        for item in self.filtered_dataset[start_idx:end_idx]:
            path = item["path"]
            pixmap = QPixmap(path)
            if pixmap.isNull():
                continue
            icon = pixmap.scaled(72, 72, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            list_item = QListWidgetItem(Path(path).name)
            list_item.setIcon(icon)
            list_item.setData(Qt.UserRole, path)
            self.list_preview.addItem(list_item)

    def load_removed_examples(self):
        self.list_removed.clear()
        if not self.removed_examples:
            self.lbl_removed_page.setText("Removed Page 1 of 1")
            self.btn_removed_prev.setEnabled(False)
            self.btn_removed_next.setEnabled(False)
            return

        total_removed = len(self.removed_examples)
        total_removed_pages = max(
            1, (total_removed + self.removed_per_page - 1) // self.removed_per_page
        )
        self.removed_page = min(self.removed_page, total_removed_pages - 1)

        start = self.removed_page * self.removed_per_page
        end = min(start + self.removed_per_page, total_removed)

        self.lbl_removed_page.setText(
            f"Removed Page {self.removed_page + 1} of {total_removed_pages}"
        )
        self.btn_removed_prev.setEnabled(self.removed_page > 0)
        self.btn_removed_next.setEnabled(self.removed_page < total_removed_pages - 1)

        for item in self.removed_examples[start:end]:
            path = item.get("path")
            reason = item.get("reason", "removed")
            if not path:
                continue
            pixmap = QPixmap(path)
            if pixmap.isNull():
                continue
            icon = pixmap.scaled(72, 72, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            label = f"[{reason}] {Path(path).name}"
            list_item = QListWidgetItem(label)
            list_item.setIcon(icon)
            list_item.setData(Qt.UserRole, path)
            self.list_removed.addItem(list_item)

    def toggle_compare_mode(self):
        visible = self.chk_compare_mode.isChecked()
        self.lbl_removed.setVisible(visible)
        self.list_removed.setVisible(visible)
        self.removed_pagination.setVisible(visible)
        if visible:
            self.load_removed_examples()

    def show_duplicate_clusters(self):
        if not self.duplicate_clusters:
            QMessageBox.information(
                self,
                "Duplicate Clusters",
                "No duplicate clusters available (or dedup was disabled).",
            )
            return

        explorer = DuplicateClusterExplorer(self.duplicate_clusters, self)
        explorer.exec()

    def previous_removed_page(self):
        if self.removed_page > 0:
            self.removed_page -= 1
            self.load_removed_examples()

    def next_removed_page(self):
        total_pages = max(
            1,
            (len(self.removed_examples) + self.removed_per_page - 1)
            // self.removed_per_page,
        )
        if self.removed_page < total_pages - 1:
            self.removed_page += 1
            self.load_removed_examples()

    def previous_page(self):
        if self.current_page > 0:
            self.current_page -= 1
            self.load_previews()

    def next_page(self):
        total_pages = (
            len(self.filtered_dataset) + self.images_per_page - 1
        ) // self.images_per_page
        if self.current_page < total_pages - 1:
            self.current_page += 1
            self.load_previews()

    def _build_process_summary_text(self):
        stats = self.pipeline_stats or {}
        selected = len(self.filtered_dataset)
        loaded = stats.get("loaded", self.loaded_count)
        source_kind = getattr(self, "_source_kind", "images")

        if source_kind == "images":
            return (
                "This will process your dataset with backup support:\n\n"
                "1) Rename images/ → all_images/ (backup)\n"
                "2) Create new images/\n"
                f"3) Copy selected images ({selected:,}) into new images/\n"
                "4) Write transaction log for rollback\n\n"
                f"Loaded: {loaded:,}  |  Selected: {selected:,}\n"
                "Continue?"
            )
        return (
            "This will create a filtered subset of your dataset:\n\n"
            "1) Create filterkit_output/images/ in the dataset root\n"
            f"2) Copy selected images ({selected:,}) into filterkit_output/images/\n"
            "3) Write transaction log (rollback will delete filterkit_output/)\n\n"
            "The original dataset is NOT modified.\n\n"
            f"Loaded: {loaded:,}  |  Selected: {selected:,}\n"
            "Continue?"
        )

    def _transaction_path(self, root_path):
        return Path(root_path) / self.TRANSACTION_FILE

    def process_dataset(self):
        if not self.filtered_dataset or not self.dataset_path:
            return

        reply = QMessageBox.question(
            self,
            "Process Dataset",
            self._build_process_summary_text(),
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if reply != QMessageBox.Yes:
            return

        dataset_root = Path(self.dataset_path)
        source_kind = getattr(self, "_source_kind", "images")

        try:
            self.lbl_status.setText("Processing dataset...")
            QApplication.processEvents()

            if source_kind == "images":
                images_path = dataset_root / "images"
                all_images_path = dataset_root / "all_images"

                if all_images_path.exists():
                    QMessageBox.warning(
                        self,
                        "Cannot Process",
                        "all_images/ already exists. Rename or remove it before processing.",
                    )
                    self.lbl_status.setText("Processing cancelled.")
                    return

                images_path.rename(all_images_path)
                images_path.mkdir(exist_ok=True)

                copied = 0
                for item in self.filtered_dataset:
                    src = all_images_path / Path(item["path"]).name
                    dst = images_path / src.name
                    if src.exists():
                        shutil.copy2(src, dst)
                        copied += 1

                transaction = {
                    "created_at": datetime.now().isoformat(timespec="seconds"),
                    "dataset_root": str(dataset_root),
                    "source_kind": "images",
                    "images_path": str(images_path),
                    "all_images_path": str(all_images_path),
                    "copied_count": copied,
                    "selected_count": len(self.filtered_dataset),
                }
                with open(
                    self._transaction_path(dataset_root), "w", encoding="utf-8"
                ) as f:
                    json.dump(transaction, f, indent=2)

                QMessageBox.information(
                    self,
                    "Success",
                    (
                        "Dataset processed successfully.\n\n"
                        f"Backup: {all_images_path.name}/\n"
                        f"Active: images/ ({copied:,} image(s))\n"
                        "Rollback is available via 'Rollback Last Process'."
                    ),
                )
                self.lbl_status.setText(
                    f"Processed dataset: {copied} images in images/."
                )
            else:
                output_root = dataset_root / "filterkit_output"
                output_images = output_root / "images"

                if output_root.exists():
                    QMessageBox.warning(
                        self,
                        "Cannot Process",
                        "filterkit_output/ already exists. Remove it before processing.",
                    )
                    self.lbl_status.setText("Processing cancelled.")
                    return

                output_images.mkdir(parents=True, exist_ok=True)

                seen_names: Dict[str, int] = {}
                copied = 0
                for item in self.filtered_dataset:
                    src = Path(item["path"])
                    if not src.exists():
                        continue
                    name = src.name
                    if name in seen_names:
                        seen_names[name] += 1
                        name = f"{src.stem}_{seen_names[name]}{src.suffix}"
                    else:
                        seen_names[name] = 0
                    shutil.copy2(src, output_images / name)
                    copied += 1

                transaction = {
                    "created_at": datetime.now().isoformat(timespec="seconds"),
                    "dataset_root": str(dataset_root),
                    "source_kind": source_kind,
                    "output_path": str(output_root),
                    "output_images_path": str(output_images),
                    "copied_count": copied,
                    "selected_count": len(self.filtered_dataset),
                }
                with open(
                    self._transaction_path(dataset_root), "w", encoding="utf-8"
                ) as f:
                    json.dump(transaction, f, indent=2)

                QMessageBox.information(
                    self,
                    "Success",
                    (
                        "Dataset processed successfully.\n\n"
                        f"Output: filterkit_output/images/ ({copied:,} image(s))\n"
                        "The original dataset is unchanged.\n"
                        "Rollback is available via 'Rollback Last Process'."
                    ),
                )
                self.lbl_status.setText(
                    f"Processed dataset: {copied} images in filterkit_output/images/."
                )

            self._update_rollback_availability()

        except Exception as exc:
            QMessageBox.critical(self, "Processing Error", str(exc))
            self.lbl_status.setText("Processing failed.")

    def rollback_last_process(self):
        if not self.dataset_path:
            QMessageBox.information(
                self,
                "Rollback",
                "Load a dataset first so FilterKit can locate transaction files.",
            )
            return

        dataset_root = Path(self.dataset_path)
        source_kind = getattr(self, "_source_kind", "images")
        transaction_file = self._transaction_path(dataset_root)

        if source_kind == "images":
            images_folder = dataset_root / "images"
            all_images_folder = dataset_root / "all_images"
            if not images_folder.exists() or not all_images_folder.exists():
                QMessageBox.information(
                    self,
                    "Rollback",
                    "Rollback is only available when both images/ and all_images/ exist in the selected dataset root.",
                )
                self._update_rollback_availability()
                return

            if not transaction_file.exists():
                QMessageBox.information(
                    self,
                    "Rollback",
                    "No rollback transaction found in this dataset root.",
                )
                return

            try:
                with open(transaction_file, "r", encoding="utf-8") as f:
                    tx = json.load(f)

                images_path = Path(tx["images_path"])
                all_images_path = Path(tx["all_images_path"])
                if not images_path.exists() or not all_images_path.exists():
                    QMessageBox.warning(
                        self,
                        "Rollback Failed",
                        "Expected images/ and all_images/ were not found for rollback.",
                    )
                    return

                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                backup_current = (
                    images_path.parent / f"images_sieved_backup_{timestamp}"
                )
                images_path.rename(backup_current)
                all_images_path.rename(images_path)

                QMessageBox.information(
                    self,
                    "Rollback Complete",
                    (
                        "Rollback successful.\n\n"
                        "Restored original images folder.\n"
                        f"Current reduced set moved to: {backup_current.name}/"
                    ),
                )
                self.lbl_status.setText("Rollback complete: original images restored.")
                self._update_rollback_availability()

            except Exception as exc:
                QMessageBox.critical(self, "Rollback Error", str(exc))
        else:
            if not transaction_file.exists():
                QMessageBox.information(
                    self,
                    "Rollback",
                    "No rollback transaction found in this dataset root.",
                )
                return

            try:
                tx = json.loads(transaction_file.read_text(encoding="utf-8"))
                output_path = Path(tx.get("output_path", ""))
                if not output_path.exists():
                    QMessageBox.information(
                        self,
                        "Rollback",
                        "FilterKit output folder not found. Nothing to roll back.",
                    )
                    self._update_rollback_availability()
                    return

                shutil.rmtree(output_path)
                transaction_file.unlink()

                QMessageBox.information(
                    self,
                    "Rollback Complete",
                    f"Removed FilterKit output folder: {output_path.name}/\n"
                    "The original dataset is unchanged.",
                )
                self.lbl_status.setText("Rollback complete: FilterKit output removed.")
                self._update_rollback_availability()

            except Exception as exc:
                QMessageBox.critical(self, "Rollback Error", str(exc))
