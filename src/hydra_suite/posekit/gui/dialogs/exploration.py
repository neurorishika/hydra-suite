#!/usr/bin/env python3
"""
Embedding Explorer: a UMAP map of frame embeddings with hover previews.

Smart Select lives in :mod:`.smart_select` (re-exported here for old imports).
"""

import logging
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np
from PySide6.QtCore import QObject, QSize, Qt, QThread, QTimer, Signal
from PySide6.QtGui import QBrush, QColor, QPainter, QPen, QPixmap
from PySide6.QtWidgets import (
    QDialog,
    QDoubleSpinBox,
    QFrame,
    QGraphicsEllipseItem,
    QGraphicsItem,
    QGraphicsScene,
    QGraphicsView,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from .utils import keep_thread_until_finished

logger = logging.getLogger("pose_label.dialogs.exploration")


# ──────────────────────────────────────────────────────────────────────────────
# Lightweight Qt-native scatter plot view (adapted from ClassKit ExplorerView)
# ──────────────────────────────────────────────────────────────────────────────


class EmbeddingExplorerView(QGraphicsView):
    """Scatter-plot widget for UMAP embeddings with hover + click signals."""

    point_hovered = Signal(int)
    point_clicked = Signal(int)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setRenderHint(QPainter.Antialiasing, True)
        self.setDragMode(QGraphicsView.ScrollHandDrag)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.AnchorUnderMouse)
        self.setMouseTracking(True)
        self.setBackgroundBrush(QBrush(QColor(30, 30, 30)))
        self.setStyleSheet("QGraphicsView { border: none; }")

        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)

        self._coords: Optional[np.ndarray] = None
        self._cluster_ids: Optional[np.ndarray] = None
        self._labeled_mask: List[bool] = []
        self._points: List[QGraphicsEllipseItem] = []
        self._point_centers: List[tuple] = []
        self._selected_set: set = set()  # multi-selection
        self._last_hover_idx: Optional[int] = None
        self._has_fitted = False
        self._zoom_redraw_limit = 5000

    def _radius_scale(self) -> float:
        view_scale = self.transform().m11()
        if view_scale <= 0:
            view_scale = 1.0
        zoom_gain = min(0.45, max(-0.35, (view_scale - 1.0) * 0.10))
        return 1.0 - zoom_gain

    def _point_color(self, i: int) -> QColor:
        import hashlib

        if i in self._selected_set:
            return QColor(255, 245, 120)
        if self._cluster_ids is not None and i < len(self._cluster_ids):
            h = int(
                hashlib.md5(str(int(self._cluster_ids[i])).encode()).hexdigest(), 16
            )
            r = max(60, min(215, (h & 0xFF0000) >> 16))
            g = max(60, min(215, (h & 0x00FF00) >> 8))
            b = max(60, min(215, h & 0x0000FF))
            return QColor(r, g, b)
        return QColor(100, 149, 237)

    def _point_radius(self, i: int, rs: float) -> float:
        if i in self._selected_set:
            return 8.0 * rs
        if i < len(self._labeled_mask) and self._labeled_mask[i]:
            return 5.0 * rs
        return 3.5 * rs

    def _point_pen(self, i: int, rs: float) -> QPen:
        if i in self._selected_set:
            return QPen(QColor(255, 60, 60), max(2.0, 2.5 * rs))
        if i < len(self._labeled_mask) and self._labeled_mask[i]:
            return QPen(QColor(210, 210, 210), max(0.5, 0.7 * rs))
        return QPen(Qt.NoPen)

    def _point_z(self, i: int) -> float:
        if i in self._selected_set:
            return 5.0
        if i < len(self._labeled_mask) and self._labeled_mask[i]:
            return 2.0
        return 1.0

    def set_data(
        self,
        coords: np.ndarray,
        cluster_ids: Optional[np.ndarray] = None,
        labeled_mask: Optional[List[bool]] = None,
        selected_set: Optional[set] = None,
        preserve_view: bool = True,
    ):
        """Populate scatter plot from (N, 2) UMAP coords."""
        self._coords = coords
        self._cluster_ids = cluster_ids
        self._labeled_mask = labeled_mask or []
        self._selected_set = set(selected_set) if selected_set else set()

        self._scene.clear()
        self._points = []
        self._point_centers = []
        self._last_hover_idx = None

        if coords is None or len(coords) == 0:
            return

        mn = coords.min(axis=0)
        mx = coords.max(axis=0)
        span = mx - mn
        span[span == 0] = 1.0
        norm = (coords - mn) / span * 1000.0

        rs = self._radius_scale()
        for i, (x, y) in enumerate(norm):
            r = self._point_radius(i, rs)
            item = QGraphicsEllipseItem(x - r, y - r, r * 2, r * 2)
            item.setBrush(QBrush(self._point_color(i)))
            item.setPen(self._point_pen(i, rs))
            item.setZValue(self._point_z(i))
            item.setFlag(QGraphicsItem.ItemIsSelectable)
            item.setData(0, i)
            self._scene.addItem(item)
            self._points.append(item)
            self._point_centers.append((x, y))

        if not preserve_view or not self._has_fitted:
            self.fitInView(self._scene.itemsBoundingRect(), Qt.KeepAspectRatio)
            self._has_fitted = True

    def toggle_selected(self, idx: int):
        """Fast-path: toggle a single point in/out of the selection set."""
        if idx in self._selected_set:
            self._selected_set.discard(idx)
        else:
            self._selected_set.add(idx)
        if not self._points or self._coords is None:
            return
        if idx < 0 or idx >= len(self._points):
            return
        rs = self._radius_scale()
        item = self._points[idx]
        cx, cy = self._point_centers[idx]
        r = self._point_radius(idx, rs)
        item.setRect(cx - r, cy - r, r * 2, r * 2)
        item.setBrush(QBrush(self._point_color(idx)))
        item.setPen(self._point_pen(idx, rs))
        item.setZValue(self._point_z(idx))

    def mouseMoveEvent(self, event) -> None:
        """Emit ``point_hovered`` with the index of the scatter point currently under the cursor."""
        super().mouseMoveEvent(event)
        item = self.itemAt(event.pos())
        if isinstance(item, QGraphicsEllipseItem):
            idx = item.data(0)
            if idx != self._last_hover_idx:
                self._last_hover_idx = idx
                self.point_hovered.emit(idx)
        else:
            self._last_hover_idx = None

    def mousePressEvent(self, event) -> None:
        """Emit ``point_clicked`` with the index of the scatter point clicked by the left button."""
        super().mousePressEvent(event)
        if event.button() == Qt.LeftButton:
            item = self.itemAt(event.pos())
            if isinstance(item, QGraphicsEllipseItem):
                self.point_clicked.emit(item.data(0))

    def wheelEvent(self, event) -> None:
        """Zoom the scatter view in or out, redrawing points when the dataset is small enough."""
        delta = event.angleDelta().y()
        if not delta:
            return
        factor = 1.15 if delta > 0 else 1.0 / 1.15
        self.scale(factor, factor)
        if self._coords is not None and len(self._coords) <= self._zoom_redraw_limit:
            self.set_data(
                self._coords,
                cluster_ids=self._cluster_ids,
                labeled_mask=self._labeled_mask,
                selected_set=self._selected_set,
                preserve_view=True,
            )


# ──────────────────────────────────────────────────────────────────────────────
# Embedding Explorer Dialog (Qt-native, replaces Bokeh version)
# ──────────────────────────────────────────────────────────────────────────────


class EmbeddingExplorerDialog(QDialog):
    """Interactive UMAP visualization with Qt scatter plot + image hover preview."""

    def __init__(
        self,
        embeddings: np.ndarray,
        image_paths: List[Path],
        cluster_ids: Optional[np.ndarray] = None,
        is_labeled_fn=None,
        parent=None,
        labeled_mask: Optional[Sequence[bool]] = None,
        initial_selection: Optional[Sequence[int]] = None,
        accept_text: str = "Add to Labeling Set",
    ):
        super().__init__(parent)
        self.setWindowTitle("Embedding Explorer (UMAP)")
        self.setMinimumSize(QSize(1200, 760))

        self.embeddings: Optional[np.ndarray] = embeddings
        self.image_paths = image_paths
        self.cluster_ids = cluster_ids
        self.is_labeled_fn = is_labeled_fn or (lambda p: False)
        # Labeled flags are fixed while the explorer is open; compute them once
        # instead of re-reading label files on every redraw and hover.
        if labeled_mask is not None:
            self._labeled_mask = [bool(v) for v in labeled_mask]
        else:
            self._labeled_mask = [bool(self.is_labeled_fn(p)) for p in image_paths]
        self._accept_text = accept_text

        self.umap_projection: Optional[np.ndarray] = None
        self.selected_indices: List[int] = list(
            dict.fromkeys(int(i) for i in (initial_selection or []))
        )
        self._umap_thread = None
        self._umap_worker = None
        self._stale_umap: list = []
        self._closed = False
        self._pending_preview_idx: Optional[int] = None

        self._init_ui()

    def _init_ui(self):
        self.setStyleSheet("""
            QDialog { background-color: #1e1e1e; }
            QLabel { color: #cccccc; }
            QPushButton {
                background-color: #0e639c; color: #ffffff;
                border: none; border-radius: 4px; padding: 5px 12px;
            }
            QPushButton:hover { background-color: #1177bb; }
            QPushButton:disabled { background-color: #3e3e42; color: #888888; }
            QSpinBox, QDoubleSpinBox {
                background-color: #252526; color: #e0e0e0;
                border: 1px solid #3e3e42; border-radius: 3px; padding: 3px 6px;
            }
        """)

        outer = QVBoxLayout(self)
        outer.setSpacing(6)

        # ── Controls row ─────────────────────────────────────────────────────
        ctrl = QHBoxLayout()
        ctrl.addWidget(QLabel("Neighbors:"))
        self.neighbors_spin = QSpinBox()
        self.neighbors_spin.setRange(5, 100)
        self.neighbors_spin.setValue(15)
        ctrl.addWidget(self.neighbors_spin)

        ctrl.addWidget(QLabel("Min dist:"))
        self.min_dist_spin = QDoubleSpinBox()
        self.min_dist_spin.setRange(0.0, 1.0)
        self.min_dist_spin.setSingleStep(0.05)
        self.min_dist_spin.setValue(0.1)
        ctrl.addWidget(self.min_dist_spin)

        self.btn_compute_umap = QPushButton("Compute UMAP")
        self.btn_compute_umap.clicked.connect(self._compute_umap)
        ctrl.addWidget(self.btn_compute_umap)

        self.btn_cancel_umap = QPushButton("Cancel")
        self.btn_cancel_umap.setEnabled(False)
        self.btn_cancel_umap.clicked.connect(self._cancel_umap)
        ctrl.addWidget(self.btn_cancel_umap)

        ctrl.addStretch()
        outer.addLayout(ctrl)

        self.umap_status = QLabel("Adjust parameters and click 'Compute UMAP'.")
        self.umap_status.setStyleSheet("color: #9e9e9e; padding: 2px;")
        outer.addWidget(self.umap_status)

        # ── Main splitter: scatter (left) | preview (right) ──────────────────
        splitter = QSplitter(Qt.Horizontal)

        self.explorer_view = EmbeddingExplorerView()
        self.explorer_view.setMinimumWidth(500)
        self.explorer_view.point_hovered.connect(self._on_hover)
        self.explorer_view.point_clicked.connect(self._on_click)
        splitter.addWidget(self.explorer_view)

        # Right panel
        right = QWidget()
        right.setMinimumWidth(320)
        right_layout = QVBoxLayout(right)
        right_layout.setSpacing(6)

        right_layout.addWidget(QLabel("<b style='color:#ffffff;'>Image Preview</b>"))

        self.preview_label = QLabel()
        self.preview_label.setAlignment(Qt.AlignCenter)
        self.preview_label.setMinimumSize(300, 300)
        self.preview_label.setStyleSheet(
            "background-color: #252526; border: 1px solid #3e3e42; border-radius: 4px;"
        )
        self.preview_label.setText(
            "<span style='color:#555555;'>Hover a point to preview</span>"
        )
        right_layout.addWidget(self.preview_label, 1)

        self.preview_info = QLabel()
        self.preview_info.setWordWrap(True)
        self.preview_info.setTextFormat(Qt.RichText)
        self.preview_info.setStyleSheet("color: #cccccc; padding: 4px;")
        self.preview_info.setText(
            "<span style='color:#9e9e9e;'>Hover to preview  ·  Click to select / deselect</span>"
        )
        right_layout.addWidget(self.preview_info)

        sep = QFrame()
        sep.setFrameShape(QFrame.HLine)
        sep.setStyleSheet("color: #3e3e42;")
        right_layout.addWidget(sep)

        self.selection_summary = QLabel("No points selected.")
        self.selection_summary.setStyleSheet(
            "color: #4ec9b0; padding: 4px; font-weight: bold;"
        )
        right_layout.addWidget(self.selection_summary)

        btn_row = QHBoxLayout()
        self.btn_clear_sel = QPushButton("Clear Selection")
        self.btn_clear_sel.clicked.connect(self._clear_selection)
        self.btn_add_sel = QPushButton(self._accept_text)
        self.btn_add_sel.clicked.connect(self.accept)
        btn_close = QPushButton("Close")
        btn_close.clicked.connect(self.reject)
        btn_row.addWidget(self.btn_clear_sel)
        btn_row.addWidget(self.btn_add_sel)
        btn_row.addWidget(btn_close)
        right_layout.addLayout(btn_row)

        splitter.addWidget(right)
        splitter.setSizes([720, 380])
        outer.addWidget(splitter, 1)

        # Debounce timer for preview
        self._preview_timer = QTimer(self)
        self._preview_timer.setInterval(60)
        self._preview_timer.setSingleShot(True)
        self._preview_timer.timeout.connect(self._flush_preview)
        self._update_selection_summary()

    # ── UMAP ─────────────────────────────────────────────────────────────────

    def _compute_umap(self):
        if self.embeddings is None or self._closed:
            return
        if self._stale_umap:
            # A canceled fit is still running (UMAP can't be interrupted):
            # don't stack another one on top of it.
            self.umap_status.setText(
                "Waiting for the canceled UMAP run to finish before starting again."
            )
            return
        self.btn_compute_umap.setEnabled(False)
        self.btn_cancel_umap.setEnabled(True)
        self.umap_status.setText("Computing UMAP projection...")
        self._umap_thread = QThread(self)
        self._umap_worker = UMAPWorker(
            embeddings=self.embeddings,
            n_neighbors=self.neighbors_spin.value(),
            min_dist=self.min_dist_spin.value(),
            random_state=42,
        )
        self._umap_worker.moveToThread(self._umap_thread)
        self._umap_thread.started.connect(self._umap_worker.run)
        self._umap_worker.progress.connect(self.umap_status.setText)
        self._umap_worker.finished.connect(self._on_umap_finished)
        self._umap_worker.failed.connect(self._on_umap_failed)
        self._umap_worker.finished.connect(self._umap_thread.quit)
        self._umap_worker.failed.connect(self._umap_thread.quit)
        self._umap_thread.start()

    def _cancel_umap(self):
        worker, thread = self._umap_worker, self._umap_thread
        if worker is not None:
            # UMAP can't be interrupted mid-fit; the worker drops its result
            # and the thread ends when the fit returns.
            worker.cancel()
            for sig in (worker.finished, worker.failed, worker.progress):
                try:
                    sig.disconnect()
                except (RuntimeError, TypeError):
                    pass
        if thread is not None:
            thread.quit()
            if not thread.isFinished():
                entry = (thread, worker)
                self._stale_umap.append(entry)
                thread.finished.connect(lambda e=entry: self._on_stale_umap_done(e))
        self._umap_thread = None
        self._umap_worker = None
        self.btn_compute_umap.setEnabled(not self._stale_umap)
        self.btn_cancel_umap.setEnabled(False)
        self.umap_status.setText(
            "UMAP canceled."
            + (" Finishing the running fit first…" if self._stale_umap else "")
        )

    def _on_stale_umap_done(self, entry) -> None:
        if entry in self._stale_umap:
            self._stale_umap.remove(entry)
        if not self._closed and not self._stale_umap and self._umap_worker is None:
            self.btn_compute_umap.setEnabled(True)
            self.umap_status.setText("UMAP canceled.")

    def _end_umap_thread(self) -> None:
        thread = self._umap_thread
        if thread is not None:
            thread.quit()
            if not thread.wait(2000):
                keep_thread_until_finished(thread, self._umap_worker)
        self._umap_thread = None
        self._umap_worker = None

    def _on_umap_finished(self, projection: np.ndarray):
        # Only the live run counts (a queued result can outlive a cancel).
        if self._closed or self.sender() is not self._umap_worker:
            return
        self._end_umap_thread()
        self.umap_projection = projection
        n = projection.shape[0]
        self.umap_status.setText(
            f"UMAP complete  ·  {n:,} points  (hover to preview, click to select)"
        )
        self.btn_compute_umap.setEnabled(True)
        self.btn_cancel_umap.setEnabled(False)
        self.explorer_view.set_data(
            projection,
            cluster_ids=self.cluster_ids,
            labeled_mask=self._labeled_mask,
            selected_set=set(self.selected_indices),
        )

    def _on_umap_failed(self, msg: str):
        if self._closed or self.sender() is not self._umap_worker:
            return
        self._end_umap_thread()
        self.umap_status.setText(f"UMAP failed: {msg}")
        self.btn_compute_umap.setEnabled(True)
        self.btn_cancel_umap.setEnabled(False)
        QMessageBox.critical(self, "UMAP Error", msg)

    # ── Preview ───────────────────────────────────────────────────────────────

    def _on_hover(self, idx: int):
        self._pending_preview_idx = idx
        if not self._preview_timer.isActive():
            self._preview_timer.start()

    def _flush_preview(self):
        idx = self._pending_preview_idx
        if idx is not None:
            self._load_preview(idx)

    def _load_preview(self, idx: int):
        if idx < 0 or idx >= len(self.image_paths):
            return
        path = self.image_paths[idx]

        try:
            pixmap = QPixmap(str(path))

            if not pixmap.isNull():
                preview_w = max(8, self.preview_label.width() - 8)
                preview_h = max(8, self.preview_label.height() - 8)
                pixmap = pixmap.scaled(
                    preview_w,
                    preview_h,
                    Qt.KeepAspectRatio,
                    Qt.SmoothTransformation,
                )
                self.preview_label.setPixmap(pixmap)
            else:
                self.preview_label.setText(
                    "<span style='color:#f44336;'>Could not load image</span>"
                )
        except Exception as exc:
            self.preview_label.setText(
                f"<span style='color:#f44336;'>Error: {exc}</span>"
            )

        cluster_str = "n/a"
        if self.cluster_ids is not None and idx < len(self.cluster_ids):
            cluster_str = str(int(self.cluster_ids[idx]))
        labeled_str = (
            "yes" if idx < len(self._labeled_mask) and self._labeled_mask[idx] else "no"
        )
        selected_str = "yes" if idx in self.selected_indices else "no"
        self.preview_info.setText(
            f"<div style='line-height:1.6;'>"
            f"<b>Index:</b> {idx} &nbsp; <b>Cluster:</b> {cluster_str}<br>"
            f"<b>Labeled:</b> {labeled_str} &nbsp; <b>Selected:</b> {selected_str}<br>"
            f"<span style='color:#9e9e9e; font-size:11px;'>{path.name}</span>"
            f"</div>"
        )

    # ── Selection ─────────────────────────────────────────────────────────────

    def _on_click(self, idx: int):
        if idx in self.selected_indices:
            self.selected_indices.remove(idx)
        else:
            self.selected_indices.append(idx)
        self._update_selection_summary()
        self.explorer_view.toggle_selected(idx)
        self._load_preview(idx)

    def _update_selection_summary(self):
        n = len(self.selected_indices)
        if n == 0:
            self.selection_summary.setText("No points selected.")
        else:
            self.selection_summary.setText(
                f"{n:,} point(s) selected  \u2014  click "
                f"\u2018{self._accept_text}\u2019 to use them."
            )

    def _clear_selection(self):
        self.selected_indices.clear()
        self._update_selection_summary()
        if self.umap_projection is not None:
            self.explorer_view.set_data(
                self.umap_projection,
                cluster_ids=self.cluster_ids,
                labeled_mask=self._labeled_mask,
                selected_set=set(),
            )

    def _shutdown_umap(self) -> None:
        """Cancel UMAP and hand any running fit to the keep-alive registry.

        Nothing waits here: a fit can take minutes, and the registry waits
        for it on application quit (a destroyed running thread aborts Qt).
        """
        self._closed = True
        if self._umap_worker is not None:
            self._cancel_umap()
        for thread, worker in self._stale_umap:
            keep_thread_until_finished(thread, worker)
        self._stale_umap = []

    def done(self, result: int) -> None:  # accept, reject, Esc
        """Stop UMAP work and release the embeddings, however the dialog closes."""
        self._shutdown_umap()
        self.embeddings = None
        super().done(result)

    def closeEvent(self, event) -> None:
        """Title-bar close: same cleanup as ``done``."""
        self._shutdown_umap()
        self.embeddings = None
        super().closeEvent(event)


class UMAPWorker(QObject):
    """Background worker for UMAP projection."""

    progress = Signal(str)
    finished = Signal(np.ndarray)
    failed = Signal(str)

    def __init__(
        self,
        embeddings: np.ndarray,
        n_neighbors: int,
        min_dist: float = 0.1,
        random_state: int = 42,
    ):
        super().__init__()
        self.embeddings = embeddings
        self.n_neighbors = n_neighbors
        self.min_dist = min_dist
        self.random_state = random_state
        self._cancelled = False

    def cancel(self):
        """Request early termination of the UMAP projection computation."""
        self._cancelled = True

    def run(self):
        """Fit a 2-D UMAP projection on the stored embeddings and emit the result array."""
        try:
            if self._cancelled:
                return
            self.progress.emit("Importing UMAP...")
            import umap

            if self._cancelled:
                return
            self.progress.emit(
                f"Running UMAP  ({self.n_neighbors} neighbors, min_dist={self.min_dist})..."
            )
            reducer = umap.UMAP(
                n_components=2,
                n_neighbors=self.n_neighbors,
                min_dist=self.min_dist,
                metric="cosine",
                random_state=self.random_state,
            )
            projection = reducer.fit_transform(self.embeddings)
            if self._cancelled:
                return
            self.finished.emit(projection.astype(np.float32))
        except Exception as exc:
            self.failed.emit(str(exc))


def __getattr__(name):
    # Smart Select moved to its own module; keep the old import path working
    # without an import cycle (smart_select imports this module).
    if name == "SmartSelectDialog":
        from .smart_select import SmartSelectDialog

        return SmartSelectDialog
    raise AttributeError(name)
