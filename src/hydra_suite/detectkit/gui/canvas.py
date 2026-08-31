"""Read-only OBB canvas viewer for DetectKit."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

import cv2
import numpy as np
from PySide6.QtCore import QEvent, QPointF, QRectF, Qt
from PySide6.QtGui import (
    QBrush,
    QColor,
    QFont,
    QImage,
    QPainter,
    QPen,
    QPixmap,
    QPolygonF,
)
from PySide6.QtWidgets import (
    QGraphicsPixmapItem,
    QGraphicsScene,
    QGraphicsTextItem,
    QGraphicsView,
)

from hydra_suite.utils.geometry_derivation import (
    axis_aligned_bbox_quad,
    min_area_rect_quad,
)

from .constants import CANVAS_BG_COLOR, DEFAULT_OBB_FONT_SIZE, DEFAULT_OBB_LINE_WIDTH

if TYPE_CHECKING:
    from hydra_suite.training.geometry_levels import GeometryLevel

logger = logging.getLogger(__name__)

# 8-colour palette for class IDs (cycled via modulo)
_PALETTE = [
    QColor(0, 255, 0),  # green
    QColor(255, 80, 80),  # red
    QColor(80, 180, 255),  # blue
    QColor(255, 200, 0),  # yellow
    QColor(200, 80, 255),  # purple
    QColor(0, 220, 200),  # cyan
    QColor(255, 140, 0),  # orange
    QColor(180, 220, 80),  # lime
]

# The staged-escalation layer's single hue, deliberately OUTSIDE _PALETTE:
# a staged SAM3/SAM2 mask is a proposal, not a labelled class, so the
# distinction it must carry is "not ground truth" -- never a class identity.
ESCALATION_COLOUR = QColor(255, 60, 199)  # magenta


@dataclass(frozen=True)
class _LevelStyle:
    pen_style: "Qt.PenStyle"
    brush_style: "Qt.BrushStyle"
    fill_alpha: int  # 0-255; only used when brush_style != NoBrush


def _level_styles():
    """Lazily built so importing GeometryLevel doesn't happen at module load."""
    from hydra_suite.training.geometry_levels import GeometryLevel

    return {
        # POLYGON's pen is DotLine (not SolidLine) so a polygon-native
        # source's filled outline stays visually distinct from AABB's solid
        # outline when both draw for the same detection -- the fill alone
        # (translucent) is the primary differentiator per spec Decision 1,
        # but a same-style outline underneath it would still be confusable.
        GeometryLevel.POLYGON: _LevelStyle(
            Qt.PenStyle.DotLine, Qt.BrushStyle.SolidPattern, 90
        ),
        GeometryLevel.OBB: _LevelStyle(Qt.PenStyle.DashLine, Qt.BrushStyle.NoBrush, 0),
        GeometryLevel.AABB: _LevelStyle(
            Qt.PenStyle.SolidLine, Qt.BrushStyle.NoBrush, 0
        ),
    }


class OBBCanvas(QGraphicsView):
    """Read-only image viewer with oriented-bounding-box overlays."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)

        # Rendering
        self.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.setDragMode(QGraphicsView.DragMode.NoDrag)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setBackgroundBrush(QBrush(QColor(CANVAS_BG_COLOR)))
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

        # Scene
        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)

        # State
        self._pix_item: Optional[QGraphicsPixmapItem] = None
        # GT layer (ground-truth, solid lines)
        self._gt_obb_items: list = []
        self._gt_label_items: list = []
        self._gt_class_ids: list[int] = []
        # Per-geometry-level GT sub-layers (native level down to AABB); the
        # flat _gt_obb_items/_gt_label_items/_gt_class_ids lists below stay
        # as a concatenation across all drawn levels, for _apply_visibility
        # and clear_gt_detections' pre-existing flat-iteration callers.
        self._gt_level_items: dict = {}
        self._gt_level_label_items: dict = {}
        self._gt_level_class_ids: dict = {}
        self._gt_native_level: Optional["GeometryLevel"] = None
        self._show_derived_levels: bool = True
        # Staged-escalation layer (SAM3/SAM2 proposals awaiting review).
        # Its own per-level buckets, mirroring the GT layer's, because a
        # staged mask is drawn at its native level plus every derived level
        # below it -- the OBB/AABB a promotion would actually produce.
        self._esc_obb_items: list = []
        self._esc_label_items: list = []
        self._esc_class_ids: list[int] = []
        self._esc_level_items: dict = {}
        self._esc_level_label_items: dict = {}
        self._esc_level_class_ids: dict = {}
        self._esc_native_level: Optional["GeometryLevel"] = None
        self._show_escalation: bool = True
        # Prediction layer (model output, dashed lines)
        self._pred_obb_items: list = []
        self._pred_label_items: list = []
        self._pred_class_ids: list[int] = []
        # Visibility state
        self._show_gt: bool = True
        self._show_pred: bool = True
        self._visible_class_ids: set[int] = set()
        # Backward-compat aliases (views of GT layer)
        self._obb_items = self._gt_obb_items
        self._label_items = self._gt_label_items
        self._zoom: float = 1.0
        self._min_zoom: float = 0.1
        self._max_zoom: float = 4.0
        self._panning: bool = False
        self._pan_start: Optional[QPointF] = None
        self._fit_mode: bool = True

        for target in (self, self.viewport()):
            target.setAttribute(Qt.WidgetAttribute.WA_AcceptTouchEvents, True)
            target.grabGesture(Qt.PinchGesture)
            target.installEventFilter(self)

    # ------------------------------------------------------------------
    # Image loading
    # ------------------------------------------------------------------

    def image_size(self) -> "tuple[int, int] | None":
        """(h, w) of the loaded image, or None if nothing is loaded.

        Exists so callers that need the frame's dimensions do not decode the
        file again -- load_image already did, and at 4512^2 each decode is
        ~100 ms of a keypress.
        """
        if self._pix_item is None:
            return None
        pixmap = self._pix_item.pixmap()
        return int(pixmap.height()), int(pixmap.width())

    def load_image(self, image_path: str) -> bool:
        """Load an image from *image_path* via OpenCV."""
        bgr = cv2.imread(image_path)
        if bgr is None:
            logger.warning("Failed to read image: %s", image_path)
            return False
        return self.set_image_array(bgr)

    def set_image_array(self, bgr: np.ndarray) -> bool:
        """Display a BGR numpy array on the canvas."""
        try:
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            h, w, _ = rgb.shape
            bytes_per_line = w * 3
            qimg = QImage(rgb.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
            pixmap = QPixmap.fromImage(qimg)

            if self._pix_item is None:
                self._pix_item = QGraphicsPixmapItem(pixmap)
                self._scene.addItem(self._pix_item)
            else:
                self._pix_item.setPixmap(pixmap)

            self._scene.setSceneRect(QRectF(pixmap.rect()))
            self.fit_in_view()
            return True
        except Exception:
            logger.warning("Failed to set image array", exc_info=True)
            return False

    # ------------------------------------------------------------------
    # Detection overlays
    # ------------------------------------------------------------------

    def _build_class_lookup(
        self, class_names: list[str] | dict[int, str] | None
    ) -> dict[int, str]:
        if isinstance(class_names, dict):
            return {int(k): str(v) for k, v in class_names.items()}
        return {idx: str(n) for idx, n in enumerate(class_names or ["object"])}

    def _draw_detections(
        self,
        detections: list[dict],
        obb_items: list,
        label_items: list,
        class_ids: list,
        class_names: list[str] | dict[int, str] | None,
        line_style: "Qt.PenStyle",
        show_confidence: bool = False,
        *,
        brush_style: "Qt.BrushStyle" = Qt.BrushStyle.NoBrush,
        fill_alpha: int = 255,
        show_labels: bool = True,
        colour_override: "QColor | None" = None,
    ) -> None:
        """Render *detections* into the given item lists.

        ``colour_override`` paints every shape in one colour instead of the
        per-class palette. Only the staged-escalation layer uses it: its
        class ids come from a staging dir's classes.txt (the prompt), so
        indexing the project's palette with them would assert a class
        identity the staged labels do not carry.
        """
        font = QFont()
        font.setPixelSize(DEFAULT_OBB_FONT_SIZE)
        lookup = self._build_class_lookup(class_names)

        for det in detections:
            class_id: int = det.get("class_id", 0)
            polygon_px = det.get("polygon_px", [])
            if len(polygon_px) < 3:
                continue
            confidence = det.get("confidence", None)

            colour = (
                colour_override
                if colour_override is not None
                else _PALETTE[class_id % len(_PALETTE)]
            )
            qpoly = QPolygonF()
            for x, y in polygon_px:
                qpoly.append(QPointF(x, y))
            qpoly.append(QPointF(*polygon_px[0]))

            pen = QPen(colour, DEFAULT_OBB_LINE_WIDTH)
            pen.setCosmetic(True)
            pen.setStyle(line_style)

            if brush_style != Qt.BrushStyle.NoBrush:
                fill_colour = QColor(colour)
                fill_colour.setAlpha(fill_alpha)
                brush = QBrush(fill_colour, brush_style)
            else:
                brush = QBrush(Qt.BrushStyle.NoBrush)

            poly_item = self._scene.addPolygon(qpoly, pen, brush)
            obb_items.append(poly_item)
            class_ids.append(class_id)

            if not show_labels:
                label_items.append(None)
                continue

            label_name = lookup.get(class_id, f"class_{class_id}")
            if show_confidence and confidence is not None:
                label_text = f"{label_name} ({confidence:.2f})"
            elif show_confidence:
                # A layer that ASKED for confidence and has none must not
                # fall back to the class id: "(0)" beside a mask reads as a
                # confidence of 0.00. Staged escalation labels carry no
                # confidence (data/al/labels.py writes class id + coords
                # only), so this is the live path for that layer.
                label_text = label_name
            else:
                label_text = f"{label_name} ({class_id})"
            txt_item = QGraphicsTextItem(label_text)
            txt_item.setFont(font)
            txt_item.setDefaultTextColor(colour)
            txt_item.setPos(QPointF(*polygon_px[0]))
            self._scene.addItem(txt_item)
            label_items.append(txt_item)

    def _apply_visibility(self) -> None:
        """Show/hide items based on visibility flags and class filter."""

        def _set_layer(
            obb_items, label_items, class_ids, layer_visible, *, class_filter=True
        ):
            for obb, lbl, cid in zip(obb_items, label_items, class_ids):
                visible = layer_visible and (
                    not class_filter
                    or not self._visible_class_ids
                    or cid in self._visible_class_ids
                )
                obb.setVisible(visible)
                if lbl is not None:
                    lbl.setVisible(visible)

        if self._gt_level_items:
            for level, items in self._gt_level_items.items():
                label_items = self._gt_level_label_items[level]
                class_ids = self._gt_level_class_ids[level]
                level_visible = self._show_gt and (
                    level == self._gt_native_level or self._show_derived_levels
                )
                _set_layer(items, label_items, class_ids, level_visible)
        else:
            _set_layer(
                self._gt_obb_items,
                self._gt_label_items,
                self._gt_class_ids,
                self._show_gt,
            )

        for level, items in self._esc_level_items.items():
            level_visible = self._show_escalation and (
                level == self._esc_native_level or self._show_derived_levels
            )
            _set_layer(
                items,
                self._esc_level_label_items[level],
                self._esc_level_class_ids[level],
                level_visible,
                # Staged class ids index the STAGING dir's classes.txt (the
                # prompt), not the project's class list, so the project
                # class filter cannot meaningfully address them.
                class_filter=False,
            )

        _set_layer(
            self._pred_obb_items,
            self._pred_label_items,
            self._pred_class_ids,
            self._show_pred,
        )

    def set_gt_detections(
        self,
        detections: list[dict],
        class_names: list[str] | dict[int, str] | None = None,
        *,
        append: bool = False,
        fill_alpha: int = 0,
    ) -> None:
        """Draw ground-truth OBB polygons (solid lines)."""
        if not append:
            self.clear_gt_detections()

        if append and self._gt_native_level is not None:
            # A multi-level draw already populated the per-level GT
            # buckets, so _apply_visibility takes the per-level branch
            # (it iterates _gt_level_items, not the flat lists, whenever
            # the former is non-empty). Appending only into the flat lists
            # here would leave these items outside that iteration -- they'd
            # never be visited by show/hide toggles or class filters. Route
            # them into the native level's buckets instead, and mirror the
            # newly drawn items into the flat lists too since those still
            # serve as the flat concatenation other callers (e.g.
            # clear_gt_detections) read.
            native_level = self._gt_native_level
            obb_items = self._gt_level_items.setdefault(native_level, [])
            label_items = self._gt_level_label_items.setdefault(native_level, [])
            class_ids = self._gt_level_class_ids.setdefault(native_level, [])
            start = len(obb_items)
            self._draw_detections(
                detections,
                obb_items,
                label_items,
                class_ids,
                class_names,
                Qt.PenStyle.SolidLine,
                show_confidence=False,
                brush_style=(
                    Qt.BrushStyle.SolidPattern
                    if fill_alpha > 0
                    else Qt.BrushStyle.NoBrush
                ),
                fill_alpha=fill_alpha,
            )
            self._gt_obb_items.extend(obb_items[start:])
            self._gt_label_items.extend(label_items[start:])
            self._gt_class_ids.extend(class_ids[start:])
        else:
            self._draw_detections(
                detections,
                self._gt_obb_items,
                self._gt_label_items,
                self._gt_class_ids,
                class_names,
                Qt.PenStyle.SolidLine,
                show_confidence=False,
                brush_style=(
                    Qt.BrushStyle.SolidPattern
                    if fill_alpha > 0
                    else Qt.BrushStyle.NoBrush
                ),
                fill_alpha=fill_alpha,
            )
        self._apply_visibility()

    @staticmethod
    def _levels_with_shapes(detections: list[dict], native_level):
        """Yield (level, detections) from native down to AABB.

        The derived levels are what a promotion of these shapes would
        actually produce, so both the GT layer and the staged-escalation
        layer derive them the same way rather than each open-coding it.
        """
        from hydra_suite.training.geometry_levels import GeometryLevel

        for level in (GeometryLevel.POLYGON, GeometryLevel.OBB, GeometryLevel.AABB):
            if level > native_level:
                continue
            level_detections = []
            for det in detections:
                polygon_px = det.get("polygon_px", [])
                if level == native_level:
                    shape = polygon_px
                elif level == GeometryLevel.OBB:
                    shape = min_area_rect_quad(polygon_px)
                else:  # GeometryLevel.AABB
                    shape = axis_aligned_bbox_quad(polygon_px)
                if not shape:
                    continue
                level_detections.append({**det, "polygon_px": shape})
            if level_detections:
                yield level, level_detections

    def set_escalation_detections(
        self,
        detections: list[dict],
        class_names: list[str] | dict[int, str] | None = None,
        *,
        native_level,
    ) -> None:
        """Draw staged escalation masks in ESCALATION_COLOUR.

        A staged SAM3/SAM2 result used to be accepted or rejected entirely
        sight-unseen -- the review dialog is a text list and nothing parsed
        the staging dir's labels for display. This is the preview: the
        proposed shape at its native level plus the OBB and AABB a
        promotion would derive from it, all in one non-palette hue so it can
        never be read as ground truth, and labelled with the per-instance
        confidence so it is visible which masks a re-threshold would drop.
        """
        self.clear_escalation_detections()
        styles = _level_styles()
        for level, level_detections in self._levels_with_shapes(
            detections, native_level
        ):
            style = styles[level]
            obb_items: list = []
            label_items: list = []
            class_ids: list = []
            self._draw_detections(
                level_detections,
                obb_items,
                label_items,
                class_ids,
                class_names,
                style.pen_style,
                show_confidence=True,
                brush_style=style.brush_style,
                fill_alpha=style.fill_alpha,
                show_labels=(level == native_level),
                colour_override=ESCALATION_COLOUR,
            )
            self._esc_level_items[level] = obb_items
            self._esc_level_label_items[level] = label_items
            self._esc_level_class_ids[level] = class_ids
            self._esc_obb_items.extend(obb_items)
            self._esc_label_items.extend(label_items)
            self._esc_class_ids.extend(class_ids)
        self._esc_native_level = native_level
        self._apply_visibility()

    def set_escalation_visible(self, visible: bool) -> None:
        """Toggle the staged-escalation layer."""
        self._show_escalation = bool(visible)
        self._apply_visibility()

    def clear_escalation_detections(self) -> None:
        """Remove all staged-escalation items from the scene."""
        for item in self._esc_obb_items:
            if item is not None:
                self._scene.removeItem(item)
        for item in self._esc_label_items:
            if item is not None:
                self._scene.removeItem(item)
        self._esc_obb_items.clear()
        self._esc_label_items.clear()
        self._esc_class_ids.clear()
        self._esc_level_items.clear()
        self._esc_level_label_items.clear()
        self._esc_level_class_ids.clear()
        self._esc_native_level = None

    def set_gt_detections_multi_level(
        self,
        detections: list[dict],
        class_names: list[str] | dict[int, str] | None = None,
        *,
        native_level: "GeometryLevel",
        reviewed: bool = True,
    ) -> None:
        """Draw ground-truth detections at *native_level*, plus every
        derived level below it down to AABB, each in its own per-level
        style. Only the native level's shapes carry a text label."""
        self.clear_gt_detections()
        styles = _level_styles()

        for level, level_detections in self._levels_with_shapes(
            detections, native_level
        ):
            # Unreviewed native shapes get a hatched fill to flag them as
            # not-yet-confirmed, but MUST keep the level's own pen style --
            # hardcoding SolidLine here would collide with AABB's own
            # SolidLine pen and make an unreviewed OBB-native quad visually
            # merge with its derived AABB outline (they'd be indistinguishable).
            style = (
                _LevelStyle(styles[level].pen_style, Qt.BrushStyle.BDiagPattern, 140)
                if (level == native_level and not reviewed)
                else styles[level]
            )
            obb_items: list = []
            label_items: list = []
            class_ids: list = []
            self._draw_detections(
                level_detections,
                obb_items,
                label_items,
                class_ids,
                class_names,
                style.pen_style,
                show_confidence=False,
                brush_style=style.brush_style,
                fill_alpha=style.fill_alpha,
                show_labels=(level == native_level),
            )
            self._gt_level_items[level] = obb_items
            self._gt_level_label_items[level] = label_items
            self._gt_level_class_ids[level] = class_ids
            self._gt_obb_items.extend(obb_items)
            self._gt_label_items.extend(label_items)
            self._gt_class_ids.extend(class_ids)

        self._gt_native_level = native_level
        self._apply_visibility()

    def set_derived_levels_visible(self, visible: bool) -> None:
        """Toggle whether non-native (derived) GT levels are drawn."""
        self._show_derived_levels = visible
        self._apply_visibility()

    def set_pred_detections(
        self,
        detections: list[dict],
        class_names: list[str] | dict[int, str] | None = None,
        *,
        fill_alpha: int = 0,
    ) -> None:
        """Draw model-prediction OBB polygons (dashed lines)."""
        self.clear_pred_detections()
        self._draw_detections(
            detections,
            self._pred_obb_items,
            self._pred_label_items,
            self._pred_class_ids,
            class_names,
            Qt.PenStyle.DashLine,
            show_confidence=True,
            brush_style=(
                Qt.BrushStyle.SolidPattern if fill_alpha > 0 else Qt.BrushStyle.NoBrush
            ),
            fill_alpha=fill_alpha,
        )
        self._apply_visibility()

    def set_overlay_visibility(self, show_gt: bool, show_pred: bool) -> None:
        """Toggle GT and prediction layer visibility."""
        self._show_gt = show_gt
        self._show_pred = show_pred
        self._apply_visibility()

    def set_class_filter(self, visible_class_ids: set[int]) -> None:
        """Show only the given class IDs (empty set = show all)."""
        self._visible_class_ids = set(visible_class_ids)
        self._apply_visibility()

    # Backward-compat aliases
    def set_detections(
        self,
        detections: list[dict],
        class_names: list[str] | dict[int, str] | None = None,
    ) -> None:
        """Backward-compatible alias: set GT detections."""
        self.set_gt_detections(detections, class_names)

    def clear_detections(self) -> None:
        """Backward-compatible alias: clear GT layer."""
        self.clear_gt_detections()

    def clear_gt_detections(self) -> None:
        """Remove all GT polygon and label items from the scene."""
        for item in self._gt_obb_items:
            self._scene.removeItem(item)
        for item in self._gt_label_items:
            if item is not None:
                self._scene.removeItem(item)
        self._gt_obb_items.clear()
        self._gt_label_items.clear()
        self._gt_class_ids.clear()
        self._gt_level_items.clear()
        self._gt_level_label_items.clear()
        self._gt_level_class_ids.clear()
        self._gt_native_level = None

    def clear_pred_detections(self) -> None:
        """Remove all prediction polygon and label items from the scene."""
        for item in self._pred_obb_items:
            if item is not None:
                self._scene.removeItem(item)
        for item in self._pred_label_items:
            if item is not None:
                self._scene.removeItem(item)
        self._pred_obb_items.clear()
        self._pred_label_items.clear()
        self._pred_class_ids.clear()

    def clear_all(self) -> None:
        """Remove everything from the scene."""
        self._scene.clear()
        self._pix_item = None
        self._gt_obb_items.clear()
        self._gt_label_items.clear()
        self._gt_class_ids.clear()
        self._gt_level_items.clear()
        self._gt_level_label_items.clear()
        self._gt_level_class_ids.clear()
        self._gt_native_level = None
        self._pred_obb_items.clear()
        self._pred_label_items.clear()
        self._pred_class_ids.clear()
        self._esc_obb_items.clear()
        self._esc_label_items.clear()
        self._esc_class_ids.clear()
        self._esc_level_items.clear()
        self._esc_level_label_items.clear()
        self._esc_level_class_ids.clear()
        self._esc_native_level = None
        self._zoom = 1.0
        self._fit_mode = True
        self.setCursor(Qt.CursorShape.ArrowCursor)

    # ------------------------------------------------------------------
    # View helpers
    # ------------------------------------------------------------------

    def _current_zoom(self) -> float:
        return float(self.transform().m11())

    def _set_zoom(self, new_zoom: float) -> bool:
        bounded_zoom = max(self._min_zoom, min(self._max_zoom, float(new_zoom)))
        current_zoom = self._current_zoom()
        if current_zoom <= 0:
            return False
        if abs(bounded_zoom - current_zoom) < 1e-6:
            self._zoom = bounded_zoom
            return False
        self.scale(bounded_zoom / current_zoom, bounded_zoom / current_zoom)
        self._zoom = bounded_zoom
        self._fit_mode = False
        return True

    def _step_zoom(self, delta: int) -> bool:
        if delta == 0:
            return False
        factor = 1.15 if delta > 0 else (1.0 / 1.15)
        return self._set_zoom(self._current_zoom() * factor)

    def fit_in_view(self) -> None:
        """Fit the current pixmap in the viewport, keeping aspect ratio."""
        if self._pix_item is not None:
            self.resetTransform()
            self.fitInView(self._pix_item, Qt.AspectRatioMode.KeepAspectRatio)
            self._zoom = max(
                self._min_zoom,
                min(self._max_zoom, self._current_zoom()),
            )
            if abs(self._zoom - self._current_zoom()) > 1e-6:
                self.resetTransform()
                self.scale(self._zoom, self._zoom)
                self.centerOn(self._pix_item)
            self._fit_mode = True
            self.setCursor(Qt.CursorShape.OpenHandCursor)

    def _handle_native_gesture(self, event) -> bool:
        gesture_type = event.gestureType()
        zoom_gesture = getattr(Qt, "ZoomNativeGesture", None)
        begin_gesture = getattr(Qt, "BeginNativeGesture", None)
        end_gesture = getattr(Qt, "EndNativeGesture", None)
        if gesture_type in (begin_gesture, end_gesture):
            event.accept()
            return True
        if gesture_type != zoom_gesture:
            return False

        scale_delta = float(event.value())
        if abs(scale_delta) < 1e-6:
            event.accept()
            return True

        current_zoom = self._current_zoom()
        scaled_zoom = current_zoom * max(0.2, 1.0 + scale_delta)
        if int(round(scaled_zoom * 100)) == int(round(current_zoom * 100)):
            scaled_zoom = current_zoom + (0.01 if scale_delta > 0 else -0.01)
        self._set_zoom(scaled_zoom)
        event.accept()
        return True

    def _handle_pinch_gesture(self, event) -> bool:
        pinch = event.gesture(Qt.PinchGesture)
        if pinch is None:
            return False
        if pinch.state() == Qt.GestureUpdated:
            zoom_delta = int((pinch.scaleFactor() - 1.0) * 60)
            if zoom_delta != 0:
                self._set_zoom(self._current_zoom() + (zoom_delta / 100.0))
        event.accept()
        return True

    # ------------------------------------------------------------------
    # Events
    # ------------------------------------------------------------------

    def eventFilter(self, watched, event):
        if watched in (self, self.viewport()):
            if event.type() == QEvent.NativeGesture and self._handle_native_gesture(
                event
            ):
                return True
            if event.type() == QEvent.Gesture and self._handle_pinch_gesture(event):
                return True
        return super().eventFilter(watched, event)

    def wheelEvent(self, event) -> None:  # noqa: N802
        """Zoom with Ctrl+wheel; otherwise allow normal viewport scrolling."""
        if event.modifiers() == Qt.KeyboardModifier.ControlModifier:
            self._step_zoom(event.angleDelta().y())
            event.accept()
            return
        super().wheelEvent(event)

    def mousePressEvent(self, event) -> None:  # noqa: N802
        """Start panning on left or middle drag."""
        if event.button() in (Qt.MouseButton.LeftButton, Qt.MouseButton.MiddleButton):
            self._panning = True
            self._pan_start = event.position()
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:  # noqa: N802
        """Adjust scrollbars while panning."""
        if self._panning and self._pan_start is not None:
            delta = event.position() - self._pan_start
            self._pan_start = event.position()
            self.horizontalScrollBar().setValue(
                self.horizontalScrollBar().value() - int(delta.x())
            )
            self.verticalScrollBar().setValue(
                self.verticalScrollBar().value() - int(delta.y())
            )
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event) -> None:  # noqa: N802
        """End panning."""
        if self._panning:
            self._panning = False
            self._pan_start = None
            self.setCursor(Qt.CursorShape.OpenHandCursor)
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def mouseDoubleClickEvent(self, event) -> None:  # noqa: N802
        """Fit the image to the viewport on double-click."""
        if event.button() == Qt.MouseButton.LeftButton:
            self.fit_in_view()
            event.accept()
            return
        super().mouseDoubleClickEvent(event)

    def keyPressEvent(self, event) -> None:  # noqa: N802
        """Support keyboard zoom, fit, and panning."""
        key = event.key()
        if key in (Qt.Key_Plus, Qt.Key_Equal):
            self._set_zoom(self._current_zoom() + 0.1)
            event.accept()
            return
        if key == Qt.Key_Minus:
            self._set_zoom(self._current_zoom() - 0.1)
            event.accept()
            return
        if key in (Qt.Key_0, Qt.Key_F):
            self.fit_in_view()
            event.accept()
            return

        pan_step = 48
        if key == Qt.Key_Left:
            self.horizontalScrollBar().setValue(
                self.horizontalScrollBar().value() - pan_step
            )
            event.accept()
            return
        if key == Qt.Key_Right:
            self.horizontalScrollBar().setValue(
                self.horizontalScrollBar().value() + pan_step
            )
            event.accept()
            return
        if key == Qt.Key_Up:
            self.verticalScrollBar().setValue(
                self.verticalScrollBar().value() - pan_step
            )
            event.accept()
            return
        if key == Qt.Key_Down:
            self.verticalScrollBar().setValue(
                self.verticalScrollBar().value() + pan_step
            )
            event.accept()
            return

        super().keyPressEvent(event)

    def resizeEvent(self, event) -> None:  # noqa: N802
        """Keep fit-to-screen active across resizes when fit mode is enabled."""
        super().resizeEvent(event)
        if self._pix_item is not None and self._fit_mode:
            self.fit_in_view()
