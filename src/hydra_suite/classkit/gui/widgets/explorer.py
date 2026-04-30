"""Interactive scatter-plot view for exploring UMAP embedding projections."""

import numpy as np
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QBrush, QColor, QPainter, QPen, QWheelEvent
from PySide6.QtWidgets import (
    QGraphicsEllipseItem,
    QGraphicsItem,
    QGraphicsScene,
    QGraphicsView,
)

from .color_utils import build_category_color_map, color_for_value


class ExplorerView(QGraphicsView):
    """
    Scatter plot visualization of UMAP embeddings.
    Allows selection and exploration.
    """

    point_clicked = Signal(int)  # Emits index of clicked point
    point_hovered = Signal(int)  # Emits index when hovering over a point
    empty_hovered = Signal()  # Emits when hovering empty space after a point hover
    empty_double_clicked = Signal()  # Emits when empty background is double-clicked

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setRenderHint(QPainter.Antialiasing, True)
        self.setDragMode(QGraphicsView.ScrollHandDrag)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.AnchorUnderMouse)
        self.setMouseTracking(True)

        # Dark theme background
        self.setBackgroundBrush(QBrush(QColor(30, 30, 30)))
        self.setStyleSheet("QGraphicsView { border: none; }")

        self.scene = QGraphicsScene(self)
        self.setScene(self.scene)

        self.points = []  # Keep track of items
        self.selected_index = None
        self.candidate_indices = set()
        self.round_labeled_indices = set()
        self.labeling_mode = False
        self.interactive_indices = set()
        self._last_hover_idx = None
        self._has_fitted_view = False
        self._coords = None
        self._labels = None
        self._confidences = None
        self._base_colors = []
        self._base_radii = []
        self._point_centers = []
        self._category_order = None
        self._category_colors = None
        self._point_tooltips = None
        self._zoom_redraw_limit = 4000
        self.uncertainty_outline_threshold = 0.6
        self.prediction_mode = False
        self.marker_size_multiplier = 1.0

        # Interaction state
        self._zoom_factor = 1.0
        self._min_zoom = 0.1
        self._max_zoom = 50.0

    def clear_data(self) -> None:
        """Remove all plotted items and reset explorer state."""
        self.scene.clear()
        self.points = []
        self.selected_index = None
        self.candidate_indices = set()
        self.round_labeled_indices = set()
        self.interactive_indices = set()
        self._last_hover_idx = None
        self._has_fitted_view = False
        self._coords = None
        self._labels = None
        self._confidences = None
        self._base_colors = []
        self._base_radii = []
        self._point_centers = []
        self._category_order = None
        self._category_colors = None
        self._point_tooltips = None
        self.prediction_mode = False

    def _radius_scale(self) -> float:
        view_scale = self.transform().m11()
        if view_scale <= 0:
            view_scale = 1.0
        zoom_gain = min(0.45, max(-0.35, (view_scale - 1.0) * 0.10))
        return 1.0 - zoom_gain

    @staticmethod
    def _labels_for_color_map(labels) -> list:
        """Return a list suitable for color-map building without ndarray truth checks."""
        if labels is None:
            return []
        return list(labels)

    def set_uncertainty_outline_threshold(self, threshold: float) -> None:
        """Configure the confidence threshold used for white uncertainty outlines."""
        try:
            value = float(threshold)
        except Exception:
            value = 0.6
        # 0 disables uncertainty outlines.
        self.uncertainty_outline_threshold = max(0.0, min(1.0, value))

    def set_marker_size_multiplier(self, multiplier: float) -> None:
        """Configure a global multiplier for explorer marker sizes."""
        try:
            value = float(multiplier)
        except Exception:
            value = 1.0
        self.marker_size_multiplier = max(0.4, min(4.0, value))

    def _set_view_state(
        self,
        labels,
        confidences,
        candidate_indices,
        round_labeled_indices,
        selected_index,
        labeling_mode,
        prediction_mode,
        category_order,
        category_colors,
        point_tooltips,
        point_count: int,
    ) -> None:
        """Apply the current interaction state shared by update and rebuild flows."""
        self._labels = labels
        self._confidences = confidences
        self.selected_index = selected_index
        self.candidate_indices = set(candidate_indices or [])
        self.round_labeled_indices = set(round_labeled_indices or [])
        self.labeling_mode = labeling_mode
        self.prediction_mode = prediction_mode
        self._category_order = (
            list(category_order) if category_order is not None else None
        )
        self._category_colors = (
            dict(category_colors) if category_colors is not None else None
        )
        self._point_tooltips = (
            list(point_tooltips) if point_tooltips is not None else None
        )
        if self.labeling_mode:
            if self._has_active_labeling_batch():
                self.interactive_indices = set(self.candidate_indices) | set(
                    self.round_labeled_indices
                )
                if self.selected_index is not None:
                    self.interactive_indices.add(self.selected_index)
            else:
                self.interactive_indices = set(range(point_count))
        else:
            self.interactive_indices = set(range(point_count))

    def _has_active_labeling_batch(self) -> bool:
        """Return True when labeling mode should restrict emphasis to a sampled batch."""
        return bool(self.candidate_indices or self.round_labeled_indices)

    @staticmethod
    def _is_missing_label_value(value) -> bool:
        """Return True when a point should be treated as unlabeled for styling."""
        if value is None:
            return True
        if isinstance(value, str):
            return value.strip() == ""
        if isinstance(value, float):
            return np.isnan(value)
        return False

    def _point_is_unlabeled(self, index: int, labels) -> bool:
        """Return True when the point has no explicit label in the current label view."""
        if labels is None or index < 0 or index >= len(labels):
            return False
        return self._is_missing_label_value(labels[index])

    def _compute_base_style(
        self, index: int, labels, category_colors, radius_scale: float
    ):
        """Return the base (color, radius) for a point before selection styling."""
        marker_scale = float(self.marker_size_multiplier)
        color = QColor(100, 100, 255)
        if labels is not None and len(labels) > index:
            color = color_for_value(labels[index], category_colors, default=color)

        if self._point_is_unlabeled(index, labels):
            color = QColor(95, 95, 95)

        if (
            self.labeling_mode
            and self._has_active_labeling_batch()
            and index not in self.candidate_indices
            and index not in self.round_labeled_indices
            and self._point_is_unlabeled(index, labels)
        ):
            color = QColor(90, 90, 90)

        base_radius = 3.0 * radius_scale * marker_scale
        if index in self.candidate_indices:
            base_radius = 6.0 * radius_scale * marker_scale
        if index in self.round_labeled_indices:
            base_radius = 5.0 * radius_scale * marker_scale
        return color, base_radius

    def _record_base_style(
        self,
        index: int,
        labels,
        category_colors,
        radius_scale: float,
    ) -> tuple[QColor, float]:
        """Compute and cache the base style for a point."""
        color, base_radius = self._compute_base_style(
            index, labels, category_colors, radius_scale
        )
        self._base_colors.append(color)
        self._base_radii.append(base_radius)
        return color, base_radius

    @staticmethod
    def _normalize_coords(coords: np.ndarray) -> np.ndarray:
        """Normalize coordinates into the explorer view box."""
        min_vals = coords.min(axis=0)
        max_vals = coords.max(axis=0)
        span = max_vals - min_vals
        span[span == 0] = 1
        return (coords - min_vals) / span * 1000.0

    def _update_existing_point_item(
        self,
        index: int,
        item: QGraphicsEllipseItem,
        labels,
        category_colors,
        radius_scale: float,
    ) -> None:
        """Refresh one already-created point item in place."""
        color, base_radius = self._record_base_style(
            index, labels, category_colors, radius_scale
        )
        if index < len(self._point_centers):
            cx, cy = self._point_centers[index]
            item.setRect(
                cx - base_radius,
                cy - base_radius,
                base_radius * 2,
                base_radius * 2,
            )
        item.setBrush(QBrush(color))
        if self._point_tooltips is not None and index < len(self._point_tooltips):
            item.setToolTip(self._point_tooltips[index] or "")
        else:
            item.setToolTip("")
        self._apply_item_style(index, item, radius_scale)

    def _create_point_item(
        self,
        index: int,
        x: float,
        y: float,
        labels,
        category_colors,
        radius_scale: float,
    ) -> QGraphicsEllipseItem:
        """Create, cache, and style a new point item."""
        color, base_radius = self._record_base_style(
            index, labels, category_colors, radius_scale
        )
        item = QGraphicsEllipseItem(
            x - base_radius,
            y - base_radius,
            base_radius * 2,
            base_radius * 2,
        )
        item.setBrush(QBrush(color))
        item.setPen(QPen(Qt.NoPen))
        item.setData(0, index)
        item.setAcceptHoverEvents(True)
        item.setAcceptedMouseButtons(Qt.LeftButton)
        item.setFlag(QGraphicsItem.ItemIgnoresTransformations, False)
        if self._point_tooltips is not None and index < len(self._point_tooltips):
            item.setToolTip(self._point_tooltips[index] or "")
        self.scene.addItem(item)
        self.points.append(item)
        self._point_centers.append((x, y))
        self._apply_item_style(index, item, radius_scale)
        return item

    def _apply_item_style(
        self, idx: int, item: QGraphicsEllipseItem, radius_scale: float
    ):
        """Apply pen/z-order style for a point based on selection and candidate state."""
        base_color = (
            self._base_colors[idx]
            if idx < len(self._base_colors)
            else QColor(100, 100, 255)
        )
        base_radius = (
            self._base_radii[idx]
            if idx < len(self._base_radii)
            else (3.0 * radius_scale)
        )
        center = item.rect().center()

        if self.selected_index is not None and idx == self.selected_index:
            selected_radius = max(
                base_radius * 1.55,
                10.5 * radius_scale * float(self.marker_size_multiplier),
            )
            item.setRect(
                center.x() - selected_radius,
                center.y() - selected_radius,
                selected_radius * 2,
                selected_radius * 2,
            )
            item.setBrush(QBrush(QColor(255, 245, 120)))
            item.setPen(QPen(QColor(255, 40, 40), max(2.2, 2.8 * radius_scale)))
            item.setZValue(8)
        elif idx in self.candidate_indices:
            item.setRect(
                center.x() - base_radius,
                center.y() - base_radius,
                base_radius * 2,
                base_radius * 2,
            )
            item.setBrush(QBrush(base_color))
            item.setPen(QPen(QColor(255, 220, 120), max(0.8, 1.0 * radius_scale)))
            item.setZValue(3)
        elif idx in self.round_labeled_indices:
            item.setRect(
                center.x() - base_radius,
                center.y() - base_radius,
                base_radius * 2,
                base_radius * 2,
            )
            item.setBrush(QBrush(base_color))
            item.setPen(QPen(QColor(200, 200, 200), max(0.6, 0.8 * radius_scale)))
            item.setZValue(2)
        else:
            item.setRect(
                center.x() - base_radius,
                center.y() - base_radius,
                base_radius * 2,
                base_radius * 2,
            )
            item.setBrush(QBrush(base_color))

            if self._point_is_unlabeled(idx, self._labels):
                item.setPen(QPen(Qt.NoPen))
                item.setZValue(0)
                return

            # Highlight low confidence (uncertain) points
            confidence = (
                self._confidences[idx]
                if self._confidences is not None and idx < len(self._confidences)
                else None
            )
            threshold = float(self.uncertainty_outline_threshold)
            if (
                self.prediction_mode
                and confidence is not None
                and threshold > 0.0
                and float(confidence) < threshold
            ):
                item.setPen(QPen(QColor(255, 255, 255), max(1.0, 1.5 * radius_scale)))
                item.setZValue(4)
            else:
                item.setPen(QPen(Qt.NoPen))
                item.setZValue(1)

    def set_selected_index(self, selected_index: int | None) -> bool:
        """Update selected point styling without rebuilding the full scene.

        Returns True when fast-path update was applied, False when caller should fall back
        to a full redraw.
        """
        if not self.points:
            self.selected_index = selected_index
            return False

        previous = self.selected_index
        if previous == selected_index:
            return True

        self.selected_index = selected_index
        if self.labeling_mode:
            self.interactive_indices = set(self.candidate_indices) | set(
                self.round_labeled_indices
            )
            if self.selected_index is not None:
                self.interactive_indices.add(self.selected_index)

        radius_scale = self._radius_scale()

        for idx in (previous, selected_index):
            if idx is None:
                continue
            if idx < 0 or idx >= len(self.points):
                continue
            item = self.points[idx]
            self._apply_item_style(idx, item, radius_scale)

        return True

    def update_state(
        self,
        labels: list = None,
        confidences: list = None,
        candidate_indices: list = None,
        round_labeled_indices: list = None,
        selected_index: int = None,
        labeling_mode: bool = False,
        prediction_mode: bool = False,
        category_order: list | None = None,
        category_colors: dict | None = None,
        point_tooltips: list | None = None,
    ) -> bool:
        """Update point styling/labels without rebuilding scene geometry.

        Returns False when caller should fall back to full set_data.
        """
        if not self.points or self._coords is None:
            return False

        point_count = len(self.points)
        if len(self._coords) != point_count:
            return False

        self._set_view_state(
            labels,
            confidences,
            candidate_indices,
            round_labeled_indices,
            selected_index,
            labeling_mode,
            prediction_mode,
            category_order,
            category_colors,
            point_tooltips,
            point_count,
        )

        radius_scale = self._radius_scale()
        self._base_colors = []
        self._base_radii = []
        category_colors = self._category_colors or build_category_color_map(
            self._labels_for_color_map(labels),
            category_order=self._category_order,
        )

        for i, item in enumerate(self.points):
            self._update_existing_point_item(
                i, item, labels, category_colors, radius_scale
            )

        return True

    def set_data(
        self,
        coords: np.ndarray,
        labels: list = None,
        confidences: list = None,
        candidate_indices: list = None,
        round_labeled_indices: list = None,
        selected_index: int = None,
        labeling_mode: bool = False,
        prediction_mode: bool = False,
        category_order: list | None = None,
        category_colors: dict | None = None,
        point_tooltips: list | None = None,
        preserve_view: bool = True,
    ):
        """
        Populate scatter plot.
        coords: (N, 2) normalized or raw UMAP coordinates.
        labels: check for coloring.
        """
        self._set_view_state(
            labels,
            confidences,
            candidate_indices,
            round_labeled_indices,
            selected_index,
            labeling_mode,
            prediction_mode,
            category_order,
            category_colors,
            point_tooltips,
            len(coords),
        )
        self.candidate_indices = set(candidate_indices or [])
        self.round_labeled_indices = set(round_labeled_indices or [])
        self.labeling_mode = labeling_mode
        self.prediction_mode = prediction_mode
        if self.labeling_mode:
            if self._has_active_labeling_batch():
                self.interactive_indices = set(self.candidate_indices) | set(
                    self.round_labeled_indices
                )
            else:
                self.interactive_indices = set(range(len(coords)))
        self._coords = np.asarray(coords)
        norm_coords = self._normalize_coords(self._coords)
        self.scene.clear()
        self.points = []
        self._base_colors = []
        self._base_radii = []
        self._point_centers = []
        category_colors = self._category_colors or build_category_color_map(
            self._labels_for_color_map(labels),
            category_order=self._category_order,
        )
        radius_scale = self._radius_scale()

        for i, (x, y) in enumerate(norm_coords):
            item = self._create_point_item(
                i, x, y, labels, category_colors, radius_scale
            )
            item.setFlag(QGraphicsItem.ItemIsSelectable)
            # Store index in data field or subclass
            item.setData(0, i)

        if not preserve_view or not self._has_fitted_view:
            self.fitInView(self.scene.itemsBoundingRect(), Qt.KeepAspectRatio)
            self._has_fitted_view = True

    def mousePressEvent(self, event) -> None:
        super().mousePressEvent(event)
        if event.button() == Qt.LeftButton:
            if not self.labeling_mode:
                return
            item = self.itemAt(event.pos())
            if isinstance(item, QGraphicsEllipseItem):
                idx = item.data(0)
                if idx in self.interactive_indices:
                    self.point_clicked.emit(idx)

    def mouseMoveEvent(self, event) -> None:
        super().mouseMoveEvent(event)
        item = self.itemAt(event.pos())
        if isinstance(item, QGraphicsEllipseItem):
            idx = item.data(0)
            if idx in self.interactive_indices and idx != self._last_hover_idx:
                self._last_hover_idx = idx
                self.point_hovered.emit(idx)
        else:
            if self._last_hover_idx is not None:
                self.empty_hovered.emit()
            self._last_hover_idx = None

    def mouseDoubleClickEvent(self, event) -> None:
        super().mouseDoubleClickEvent(event)
        if event.button() == Qt.LeftButton:
            item = self.itemAt(event.pos())
            if not isinstance(item, QGraphicsEllipseItem):
                self.empty_double_clicked.emit()

    def wheelEvent(self, event: QWheelEvent) -> None:
        """Zoom logic."""
        delta = event.angleDelta().y()
        if delta == 0:
            return

        zoom_in = delta > 0
        factor = 1.15 if zoom_in else 1 / 1.15

        self.scale(factor, factor)

        # Re-render to adjust size tiers with zoom level while preserving view.
        # For large datasets, avoid full scene rebuild on every wheel event.
        if self._coords is not None and len(self._coords) <= self._zoom_redraw_limit:
            self.set_data(
                self._coords,
                self._labels,
                self._confidences,
                candidate_indices=list(self.candidate_indices),
                round_labeled_indices=list(self.round_labeled_indices),
                selected_index=self.selected_index,
                labeling_mode=self.labeling_mode,
                prediction_mode=self.prediction_mode,
                category_order=self._category_order,
                category_colors=self._category_colors,
                point_tooltips=self._point_tooltips,
                preserve_view=True,
            )
