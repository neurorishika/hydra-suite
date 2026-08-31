"""Read-only OBB canvas viewer for DetectKit."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

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

from .colors import ESCALATION_COLOUR
from .constants import CANVAS_BG_COLOR, DEFAULT_OBB_FONT_SIZE, DEFAULT_OBB_LINE_WIDTH
from .overlays import ColourPolicy, Emphasis, LabelMode, LayerStyle, OverlayLayer

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


@dataclass(frozen=True)
class _LevelStyle:
    pen_style: "Qt.PenStyle"
    brush_style: "Qt.BrushStyle"
    fill_alpha: int  # 0-255; only used when brush_style != NoBrush


@dataclass
class _LevelItems:
    """The scene items one layer drew at one geometry level."""

    obb_items: list
    label_items: list
    class_ids: list[int]


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


def _styles_for(layer: "OverlayLayer", level, native_level) -> _LevelStyle:
    """The pen/brush a layer uses at one level.

    An explicit ``layer.style`` wins outright (single-level layers: the
    dialogs' filled GT and the dashed prediction layer). Otherwise the
    per-level defaults apply, with Emphasis.UNREVIEWED substituting a
    hatched fill on the NATIVE level only -- and keeping that level's own
    pen style, because hardcoding SolidLine there once made an unreviewed
    OBB-native quad indistinguishable from its derived AABB outline.
    """
    if layer.style is not None:
        return _LevelStyle(
            layer.style.pen_style, layer.style.brush_style, layer.style.fill_alpha
        )
    style = _level_styles()[level]
    if layer.emphasis is Emphasis.UNREVIEWED and level == native_level:
        return _LevelStyle(style.pen_style, Qt.BrushStyle.BDiagPattern, 140)
    return style


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
        self._layers: dict[str, OverlayLayer] = {}
        # (layer_key, level) -> _LevelItems. One flat registry; there is no
        # per-layer branch anywhere below it.
        self._items: dict[tuple, _LevelItems] = {}
        self._layer_visible: dict[str, bool] = {}
        self._show_derived_levels: bool = True
        self._visible_class_ids: set[int] = set()
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

    def set_layer(self, layer: "OverlayLayer") -> None:
        """Add or replace the layer with ``layer.key``.

        Idempotent by key: this removes that key's items before redrawing,
        which is what makes "clear before refresh" structural instead of a
        rule every caller has to remember.
        """
        self.remove_layer(layer.key)
        self._layers[layer.key] = layer

        if layer.derive_levels:
            level_iter = self._levels_with_shapes(layer.detections, layer.native_level)
        else:
            level_iter = [(layer.native_level, layer.detections)]

        for level, level_detections in level_iter:
            style = _styles_for(layer, level, layer.native_level)
            items = _LevelItems([], [], [])
            self._draw_detections(
                level_detections,
                items,
                layer,
                style,
                show_labels=(level == layer.native_level),
            )
            self._items[(layer.key, level)] = items
        self._apply_visibility()

    def remove_layer(self, key: str) -> None:
        """Remove one layer's items. No-op if the key was never drawn."""
        for (layer_key, _level), items in list(self._items.items()):
            if layer_key != key:
                continue
            for item in items.obb_items:
                if item is not None:
                    self._scene.removeItem(item)
            for item in items.label_items:
                if item is not None:
                    self._scene.removeItem(item)
        self._items = {k: v for k, v in self._items.items() if k[0] != key}
        self._layers.pop(key, None)

    def set_layer_visible(self, key: str, visible: bool) -> None:
        """Toggle a layer. Remembered even for a key not yet drawn --
        _on_overlay_changed can fire before the first show_image."""
        self._layer_visible[key] = bool(visible)
        self._apply_visibility()

    def layer_items(self, key: str) -> dict:
        """The per-level item buckets one layer drew. Read-only accessor
        for tests; production code never needs it."""
        return {
            level: items
            for (layer_key, level), items in self._items.items()
            if layer_key == key
        }

    def _draw_detections(
        self,
        detections: list[dict],
        items: "_LevelItems",
        layer: "OverlayLayer",
        style: "_LevelStyle",
        *,
        show_labels: bool = True,
    ) -> None:
        font = QFont()
        font.setPixelSize(DEFAULT_OBB_FONT_SIZE)
        lookup = self._build_class_lookup(layer.class_names)

        for det in detections:
            class_id: int = det.get("class_id", 0)
            polygon_px = det.get("polygon_px", [])
            if len(polygon_px) < 3:
                continue
            confidence = det.get("confidence", None)

            # A FIXED-policy layer paints one hue because its class ids do
            # not address the project's classes: staged escalation ids index
            # the STAGING dir's classes.txt (the prompt), so indexing the
            # palette with them would assert a class identity the staged
            # labels do not carry.
            colour = (
                layer.fixed_colour
                if layer.colour_policy is ColourPolicy.FIXED
                else _PALETTE[class_id % len(_PALETTE)]
            )
            qpoly = QPolygonF()
            for x, y in polygon_px:
                qpoly.append(QPointF(x, y))
            qpoly.append(QPointF(*polygon_px[0]))

            pen = QPen(colour, DEFAULT_OBB_LINE_WIDTH)
            pen.setCosmetic(True)
            pen.setStyle(style.pen_style)

            if style.brush_style != Qt.BrushStyle.NoBrush:
                fill_colour = QColor(colour)
                fill_colour.setAlpha(style.fill_alpha)
                brush = QBrush(fill_colour, style.brush_style)
            else:
                brush = QBrush(Qt.BrushStyle.NoBrush)

            poly_item = self._scene.addPolygon(qpoly, pen, brush)
            poly_item.setZValue(layer.z)
            items.obb_items.append(poly_item)
            items.class_ids.append(class_id)

            if not show_labels:
                items.label_items.append(None)
                continue

            label_name = lookup.get(class_id, f"class_{class_id}")
            if layer.label_mode is LabelMode.NAME_AND_CONFIDENCE:
                # A layer that asked for confidence and has none must NOT
                # fall back to the class id: "(0)" beside a mask reads as a
                # confidence of 0.00. Staged escalation labels carry no
                # confidence -- data/al/labels.py writes class id + coords
                # only -- so this is the live path for that layer.
                label_text = (
                    f"{label_name} ({confidence:.2f})"
                    if confidence is not None
                    else label_name
                )
            else:
                label_text = f"{label_name} ({class_id})"
            txt_item = QGraphicsTextItem(label_text)
            txt_item.setFont(font)
            txt_item.setDefaultTextColor(colour)
            txt_item.setPos(QPointF(*polygon_px[0]))
            txt_item.setZValue(layer.z)
            self._scene.addItem(txt_item)
            items.label_items.append(txt_item)

    def _apply_visibility(self) -> None:
        """One loop over the registry. No per-layer branch."""
        for (layer_key, level), items in self._items.items():
            layer = self._layers.get(layer_key)
            if layer is None:
                continue
            layer_visible = self._layer_visible.get(layer_key, True) and (
                level == layer.native_level or self._show_derived_levels
            )
            for obb, lbl, cid in zip(
                items.obb_items, items.label_items, items.class_ids
            ):
                visible = layer_visible and (
                    not layer.class_filtered
                    or not self._visible_class_ids
                    or cid in self._visible_class_ids
                )
                obb.setVisible(visible)
                if lbl is not None:
                    lbl.setVisible(visible)

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

    def set_derived_levels_visible(self, visible: bool) -> None:
        """Toggle whether non-native (derived) GT levels are drawn."""
        self._show_derived_levels = visible
        self._apply_visibility()

    def set_class_filter(self, visible_class_ids: set[int]) -> None:
        """Show only the given class IDs (empty set = show all)."""
        self._visible_class_ids = set(visible_class_ids)
        self._apply_visibility()

    # ------------------------------------------------------------------
    # TRANSITIONAL ADAPTERS -- deleted in Task 8 of this refactor. They
    # exist only so the six call sites can migrate one at a time instead
    # of in one unreviewable commit.
    # ------------------------------------------------------------------

    def set_gt_detections(self, detections, class_names=None, *, fill_alpha=0) -> None:
        self.set_layer(
            OverlayLayer(
                key="gt",
                detections=detections,
                native_level=self._single_level(),
                class_names=class_names,
                colour_policy=ColourPolicy.PER_CLASS,
                derive_levels=False,
                style=LayerStyle(
                    Qt.PenStyle.SolidLine,
                    (
                        Qt.BrushStyle.SolidPattern
                        if fill_alpha > 0
                        else Qt.BrushStyle.NoBrush
                    ),
                    fill_alpha,
                ),
                label_mode=LabelMode.NAME_AND_CLASS_ID,
                z=0,
            )
        )

    def set_gt_detections_multi_level(
        self, detections, class_names=None, *, native_level, reviewed=True
    ) -> None:
        self.set_layer(
            OverlayLayer(
                key="gt",
                detections=detections,
                native_level=native_level,
                class_names=class_names,
                colour_policy=ColourPolicy.PER_CLASS,
                label_mode=LabelMode.NAME_AND_CLASS_ID,
                emphasis=None if reviewed else Emphasis.UNREVIEWED,
                z=0,
            )
        )

    def set_pred_detections(
        self, detections, class_names=None, *, fill_alpha=0
    ) -> None:
        self.set_layer(
            OverlayLayer(
                key="pred",
                detections=detections,
                native_level=self._single_level(),
                class_names=class_names,
                colour_policy=ColourPolicy.PER_CLASS,
                derive_levels=False,
                style=LayerStyle(
                    Qt.PenStyle.DashLine,
                    (
                        Qt.BrushStyle.SolidPattern
                        if fill_alpha > 0
                        else Qt.BrushStyle.NoBrush
                    ),
                    fill_alpha,
                ),
                label_mode=LabelMode.NAME_AND_CONFIDENCE,
                z=20,
            )
        )

    def set_escalation_detections(
        self, detections, class_names=None, *, native_level
    ) -> None:
        self.set_layer(
            OverlayLayer(
                key="escalation",
                detections=detections,
                native_level=native_level,
                class_names=class_names,
                colour_policy=ColourPolicy.FIXED,
                fixed_colour=ESCALATION_COLOUR,
                class_filtered=False,
                label_mode=LabelMode.NAME_AND_CONFIDENCE,
                z=10,
            )
        )

    def set_escalation_visible(self, visible: bool) -> None:
        self.set_layer_visible("escalation", visible)

    def set_overlay_visibility(self, show_gt: bool, show_pred: bool) -> None:
        self.set_layer_visible("gt", show_gt)
        self.set_layer_visible("pred", show_pred)

    def clear_gt_detections(self) -> None:
        self.remove_layer("gt")

    def clear_pred_detections(self) -> None:
        self.remove_layer("pred")

    def clear_escalation_detections(self) -> None:
        self.remove_layer("escalation")

    def set_detections(self, detections, class_names=None) -> None:
        self.set_gt_detections(detections, class_names)

    def clear_detections(self) -> None:
        self.remove_layer("gt")

    @staticmethod
    def _single_level():
        """The native level a non-deriving layer declares.

        Single-level layers never derive, so the value only has to satisfy
        `level == native_level` -- which is what makes the layer labelled
        and keeps it visible when derived levels are hidden, matching the
        old flat-list branch of _apply_visibility. AABB is the floor of the
        ordering, so it can never be mistaken for a derived level.
        """
        from hydra_suite.utils.geometry_levels import GeometryLevel

        return GeometryLevel.AABB

    def clear_all(self) -> None:
        """Remove everything from the scene."""
        self._scene.clear()
        self._pix_item = None
        self._layers.clear()
        self._items.clear()
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
