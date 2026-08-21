"""ReferenceScalePreviewWidget — crop-and-overlay preview of Reference Scale.

Shows a crop around one detection from the last "Test Detection" run, with the
canonical crop box (from the current Reference Scale spinboxes) overlaid at
that detection's centre and rotation. When Detection Filters are enabled, the
size/aspect-ratio extremes are drawn too, so a user can see what the current
settings actually do to a real animal before running tracking.

Colors are picked from the colorblind-safe Okabe-Ito palette and switch
between a dark and a light variant based on the crop's own mean luminance, so
the overlay stays legible against both bright (typical bg-sub) and dark
arenas. The crop is taken from the RAW (pre-annotation) frame -- no detector
overlay boxes/labels -- and overlay lines are drawn AFTER the crop is scaled
to display size, so line width stays crisp regardless of crop resolution.
"""

from __future__ import annotations

import math
import random

import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QImage, QPainter, QPen, QPixmap
from PySide6.QtWidgets import QLabel, QVBoxLayout, QWidget

from hydra_suite.core.canonicalization.geometry import CanonicalGeometry

_PLACEHOLDER_TEXT = "Run 'Test Detection' to preview the reference scale."
_EMPTY_TEXT = "No detections found — adjust parameters and try again."
_EDGE_TEXT = "Detection too close to the frame edge to preview."

_LUMINANCE_THRESHOLD = 128.0

#: Okabe-Ito colorblind-safe hues, darkened for legibility on bright/light
#: backgrounds (typical bg-sub arenas) vs. brightened for dark backgrounds.
_PALETTE_FOR_LIGHT_BG = {
    "current": (0, 100, 75),  # dark bluish-green
    "size": (176, 95, 0),  # dark orange
    "aspect": (0, 75, 130),  # dark blue
}
_PALETTE_FOR_DARK_BG = {
    "current": (70, 220, 175),  # light bluish-green
    "size": (255, 160, 60),  # light orange
    "aspect": (120, 185, 235),  # light blue
}

_CATEGORY_LABELS = {
    "current": "reference box",
    "size": "size filter range",
    "aspect": "aspect filter range",
}


def select_boxes(
    reference_body_px: float,
    reference_aspect_ratio: float,
    canonical_margin: float,
    size_filter: tuple[float, float] | None = None,
    aspect_filter: tuple[float, float] | None = None,
) -> list[tuple[str, CanonicalGeometry]]:
    """Return ``(category, geometry)`` pairs for the current box + any filter extremes.

    ``category`` is one of ``"current"``, ``"size"``, ``"aspect"`` and drives
    color/style in the renderer. Pure function so the selection logic is
    testable without Qt.
    """
    body = max(1e-3, float(reference_body_px))
    ar = max(1.0, float(reference_aspect_ratio))
    margin = max(1.0, float(canonical_margin))

    boxes = [("current", CanonicalGeometry.from_reference(body, ar, margin))]

    if size_filter is not None:
        min_mult, max_mult = size_filter
        boxes.append(
            (
                "size",
                CanonicalGeometry.from_reference(body * float(min_mult), ar, margin),
            )
        )
        boxes.append(
            (
                "size",
                CanonicalGeometry.from_reference(body * float(max_mult), ar, margin),
            )
        )

    if aspect_filter is not None:
        min_mult, max_mult = aspect_filter
        boxes.append(
            (
                "aspect",
                CanonicalGeometry.from_reference(body, ar * float(min_mult), margin),
            )
        )
        boxes.append(
            (
                "aspect",
                CanonicalGeometry.from_reference(body, ar * float(max_mult), margin),
            )
        )

    return boxes


def select_palette(mean_luminance: float) -> dict[str, tuple[int, int, int]]:
    """Pick the dark- or light-background color variant for a crop's luminance.

    Pure function so the light/dark switch is testable without Qt or images.
    """
    if float(mean_luminance) >= _LUMINANCE_THRESHOLD:
        return _PALETTE_FOR_LIGHT_BG
    return _PALETTE_FOR_DARK_BG


def _mean_luminance(crop_rgb: np.ndarray) -> float:
    weights = np.array([0.299, 0.587, 0.114], dtype=np.float32)
    return float(crop_rgb.reshape(-1, 3).astype(np.float32).dot(weights).mean())


def _to_hex(rgb: tuple[int, int, int]) -> str:
    return "#%02x%02x%02x" % rgb


class ReferenceScalePreviewWidget(QWidget):
    """Crop-and-overlay preview for the Reference Scale section.

    Populated from a "Test Detection" run's per-detection list via
    :meth:`set_detections`; box geometry (current + filter extremes) is pushed
    separately via :meth:`set_box_params` so spinbox edits can re-render
    without picking a new detection. Click resamples a different random
    detection from the same run.
    """

    _IMAGE_SIZE = 350
    _CURRENT_ALPHA = 190  # out of 255; keeps the reference box legible
    _EXTREME_ALPHA = (
        128  # filter-extreme boxes stay in the background, less distracting
    )

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setCursor(Qt.PointingHandCursor)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        self._image_label = QLabel()
        self._image_label.setFixedSize(self._IMAGE_SIZE, self._IMAGE_SIZE)
        self._image_label.setAlignment(Qt.AlignCenter)
        self._image_label.setWordWrap(True)
        self._image_label.setStyleSheet(
            "background-color: #1e1e1e; border: 1px solid #3a3a3a; "
            "border-radius: 4px; color: #7a7a7a; font-size: 10px; padding: 6px;"
        )
        layout.addWidget(self._image_label)

        self._legend_label = QLabel()
        self._legend_label.setAlignment(Qt.AlignCenter)
        self._legend_label.setWordWrap(True)
        self._legend_label.setStyleSheet("font-size: 9px; color: #9a9a9a;")
        self._legend_label.hide()
        layout.addWidget(self._legend_label)

        self._hint_label = QLabel("Click the image to preview another detection")
        self._hint_label.setAlignment(Qt.AlignCenter)
        self._hint_label.setWordWrap(True)
        self._hint_label.setStyleSheet(
            "font-size: 9px; font-style: italic; color: #6a6a6a;"
        )
        self._hint_label.hide()
        layout.addWidget(self._hint_label)

        self.setToolTip(
            "Crop around one detection with the current Reference Scale box\n"
            "(+ filter-extreme boxes when Detection Filters are on) overlaid.\n"
            "Colors adapt to image brightness -- see the legend below.\n"
            "Click to preview a different detection."
        )

        self._frame_rgb: np.ndarray | None = None
        self._resize_factor: float = 1.0
        self._detections: list[tuple[float, float, float, float, float]] = []
        self._current_index: int = -1

        self._reference_body_px: float = 20.0
        self._reference_aspect_ratio: float = 2.0
        self._canonical_margin: float = 1.3
        self._size_filter: tuple[float, float] | None = None
        self._aspect_filter: tuple[float, float] | None = None

        self._render_placeholder(_PLACEHOLDER_TEXT)

    # ------------------------------------------------------------------
    # Test-facing convenience accessors (mirror the old QLabel-based API)
    # ------------------------------------------------------------------

    def text(self) -> str:
        return self._image_label.text()

    def pixmap(self):
        return self._image_label.pixmap()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_detections(self, frame_rgb, resize_factor, detections) -> None:
        """Load a new "Test Detection" result and pick a random detection."""
        self._frame_rgb = frame_rgb
        self._resize_factor = max(1e-6, float(resize_factor))
        self._detections = list(detections or [])
        if not self._detections:
            self._current_index = -1
            self._render_placeholder(_EMPTY_TEXT)
            return
        self._current_index = random.randrange(len(self._detections))
        self._render()

    def set_box_params(
        self,
        reference_body_px: float,
        reference_aspect_ratio: float,
        canonical_margin: float,
        size_filter: tuple[float, float] | None = None,
        aspect_filter: tuple[float, float] | None = None,
    ) -> None:
        """Update the box geometry to draw and re-render the current detection."""
        self._reference_body_px = float(reference_body_px)
        self._reference_aspect_ratio = float(reference_aspect_ratio)
        self._canonical_margin = float(canonical_margin)
        self._size_filter = size_filter
        self._aspect_filter = aspect_filter
        if self._current_index >= 0:
            self._render()

    def clear(self) -> None:
        self._frame_rgb = None
        self._detections = []
        self._current_index = -1
        self._render_placeholder(_PLACEHOLDER_TEXT)

    # ------------------------------------------------------------------
    # Interaction
    # ------------------------------------------------------------------

    def mousePressEvent(self, event) -> None:  # noqa: N802 (Qt override)
        if event.button() == Qt.LeftButton and len(self._detections) > 1:
            choices = [
                i for i in range(len(self._detections)) if i != self._current_index
            ]
            self._current_index = random.choice(choices)
            self._render()
        super().mousePressEvent(event)

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def _render_placeholder(self, text: str) -> None:
        self._image_label.setPixmap(QPixmap())
        self._image_label.setText(text)
        self._legend_label.hide()
        self._hint_label.hide()

    def _render(self) -> None:
        if self._frame_rgb is None or not (
            0 <= self._current_index < len(self._detections)
        ):
            self._render_placeholder(_PLACEHOLDER_TEXT)
            return

        cx, cy, _major, _minor, angle_deg = self._detections[self._current_index]
        boxes = select_boxes(
            self._reference_body_px * self._resize_factor,
            self._reference_aspect_ratio,
            self._canonical_margin,
            self._size_filter,
            self._aspect_filter,
        )

        max_extent = max(math.hypot(g.canvas_w, g.canvas_h) for _cat, g in boxes)
        half = int(max_extent * 0.65) + 12

        h, w = self._frame_rgb.shape[:2]
        x0 = int(np.clip(cx - half, 0, w))
        y0 = int(np.clip(cy - half, 0, h))
        x1 = int(np.clip(cx + half, 0, w))
        y1 = int(np.clip(cy + half, 0, h))
        if x1 - x0 < 8 or y1 - y0 < 8:
            self._render_placeholder(_EDGE_TEXT)
            return

        crop = np.ascontiguousarray(self._frame_rgb[y0:y1, x0:x1])
        crop_h, crop_w = crop.shape[:2]
        palette = select_palette(_mean_luminance(crop))

        qimg = QImage(
            crop.data, crop_w, crop_h, crop_w * 3, QImage.Format_RGB888
        ).copy()
        raw_pixmap = QPixmap.fromImage(qimg)
        display_pixmap = raw_pixmap.scaled(
            self._IMAGE_SIZE,
            self._IMAGE_SIZE,
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation,
        )

        scale_x = display_pixmap.width() / crop_w
        scale_y = display_pixmap.height() / crop_h
        local_cx = (cx - x0) * scale_x
        local_cy = (cy - y0) * scale_y

        painter = QPainter(display_pixmap)
        painter.setRenderHint(QPainter.Antialiasing, True)
        seen_categories: list[str] = []
        for category, geom in boxes:
            if category not in seen_categories:
                seen_categories.append(category)
            color = QColor(*palette[category])
            if category == "current":
                color.setAlpha(self._CURRENT_ALPHA)
                pen = QPen(color)
                pen.setWidthF(3.0)
            else:
                color.setAlpha(self._EXTREME_ALPHA)
                pen = QPen(color)
                pen.setWidthF(2.0)
                pen.setStyle(Qt.CustomDashLine)
                pen.setDashPattern([4, 3])
            painter.setPen(pen)
            painter.save()
            painter.translate(local_cx, local_cy)
            painter.rotate(angle_deg)
            box_w = geom.canvas_w * scale_x
            box_h = geom.canvas_h * scale_y
            painter.drawRect(QRectF(-box_w / 2, -box_h / 2, box_w, box_h))
            painter.restore()

        # Center marker: a two-tone dot is legible against any background.
        painter.setPen(QPen(QColor(20, 20, 20), 1.2))
        painter.setBrush(QColor(255, 255, 255))
        painter.drawEllipse(QPointF(local_cx, local_cy), 3.0, 3.0)
        painter.end()

        self._image_label.setText("")
        self._image_label.setPixmap(display_pixmap)
        self._legend_label.setText(self._legend_html(palette, seen_categories))
        self._legend_label.show()
        self._hint_label.setVisible(len(self._detections) > 1)

    @staticmethod
    def _legend_html(
        palette: dict[str, tuple[int, int, int]], categories: list[str]
    ) -> str:
        parts = []
        for category in categories:
            glyph = "&#9473;&#9473;&#9473;" if category == "current" else "┅┅┅"
            hex_color = _to_hex(palette[category])
            label = _CATEGORY_LABELS.get(category, category)
            parts.append(f'<span style="color:{hex_color};">{glyph}</span> {label}')
        return "&nbsp;&nbsp;".join(parts)
