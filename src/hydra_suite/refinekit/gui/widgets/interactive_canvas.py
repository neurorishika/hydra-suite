"""Shared interactive video-frame canvas for RefineKit.

Provides zoom (Ctrl+wheel, pinch, slider), drag-pan, and double-click-fit
interaction identical to the MAT main window.  All video-viewing surfaces
in afterhours (main player, frame-picker dialog, identity-assignment dialog)
embed this widget so the behaviour is consistent everywhere.

Public API
----------
InteractiveCanvas.set_pixmap(pixmap)
    Display a new QPixmap (already-converted BGR→RGB frame).
InteractiveCanvas.fit()
    Reset zoom and scroll to fit-to-viewport.
InteractiveCanvas.zoom / InteractiveCanvas.set_zoom(factor)
    Get / set the zoom factor (1.0 = native resolution).
"""

from __future__ import annotations

from typing import Optional

from PySide6.QtCore import QEvent, QPoint, Qt, Signal
from PySide6.QtGui import QPixmap, QWheelEvent

try:
    from PySide6.QtGui import QGestureEvent  # not available in all builds
except ImportError:
    QGestureEvent = None  # type: ignore[assignment,misc]

from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QSlider,
    QVBoxLayout,
    QWidget,
)

_ZOOM_MIN = 0.05
_ZOOM_MAX = 5.0
_ZOOM_STEP = 0.10  # per Ctrl+wheel tick
_ZOOM_SLIDER_SCALE = 100  # slider integer = zoom * scale


class _FrameLabel(QLabel):
    """QLabel that accepts touch events so pinch gestures work."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        if QGestureEvent is not None:
            self.setAttribute(Qt.WidgetAttribute.WA_AcceptTouchEvents, True)
            self.grabGesture(Qt.GestureType.PinchGesture)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)


class InteractiveCanvas(QWidget):
    """Zoomable, pannable viewport for displaying QPixmaps.

    Embeds a ``QScrollArea`` so the image can be larger than the visible
    area.  Zoom and pan state are kept here; callers simply push a new
    QPixmap whenever the frame changes.

    Signals
    -------
    zoom_changed(float)
        Emitted after the zoom factor changes.
    """

    zoom_changed = Signal(float)

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)

        self._pixmap: Optional[QPixmap] = None
        self._zoom: float = 1.0
        self._is_panning: bool = False
        self._pan_start: Optional[QPoint] = None
        self._scroll_h0: int = 0
        self._scroll_v0: int = 0
        self._toolbar: Optional[QHBoxLayout] = None

        self.setFocusPolicy(Qt.FocusPolicy.ClickFocus)
        self._build_ui()

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(2)

        # Scroll area
        self._scroll = QScrollArea()
        self._scroll.setWidgetResizable(False)
        self._scroll.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._scroll.setStyleSheet("background-color: #1e1e1e; border: none;")

        self._label = _FrameLabel()
        self._label.setMinimumSize(160, 120)
        self._label.setStyleSheet("background-color: #1e1e1e;")

        # Install event filter so we can intercept wheel / mouse on the label.
        self._label.installEventFilter(self)

        self._scroll.setWidget(self._label)
        root.addWidget(self._scroll, stretch=1)

        # Controls row: zoom slider + fit button
        ctrl = QHBoxLayout()
        ctrl.setContentsMargins(4, 0, 4, 0)
        ctrl.setSpacing(6)

        zoom_lbl = QLabel("Zoom:")
        zoom_lbl.setFixedWidth(40)
        ctrl.addWidget(zoom_lbl)

        self._zoom_slider = QSlider(Qt.Orientation.Horizontal)
        self._zoom_slider.setRange(
            int(_ZOOM_MIN * _ZOOM_SLIDER_SCALE),
            int(_ZOOM_MAX * _ZOOM_SLIDER_SCALE),
        )
        self._zoom_slider.setValue(int(1.0 * _ZOOM_SLIDER_SCALE))
        self._zoom_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self._zoom_slider.setTickInterval(int(0.5 * _ZOOM_SLIDER_SCALE))
        self._zoom_slider.valueChanged.connect(self._on_slider)
        ctrl.addWidget(self._zoom_slider, stretch=1)

        self._zoom_pct = QLabel("100%")
        self._zoom_pct.setFixedWidth(46)
        ctrl.addWidget(self._zoom_pct)

        self._fit_btn = QPushButton("Fit")
        self._fit_btn.setFixedWidth(38)
        self._fit_btn.setToolTip("Fit to viewport  (F · double-click canvas)")
        self._fit_btn.clicked.connect(self.fit)
        ctrl.addWidget(self._fit_btn)

        hint = QLabel("Ctrl+⇕ zoom  •  drag pan  •  dbl-click/F fit")
        hint.setStyleSheet("color:#6a6a6a; font-size:10px; font-style:italic;")
        ctrl.addWidget(hint)

        ctrl.addStretch(1)
        self._toolbar = ctrl

        root.addLayout(ctrl)

    def add_toolbar_widget(self, widget: QWidget) -> None:
        """Append an extra control to the right side of the zoom toolbar."""
        if self._toolbar is None:
            return
        self._toolbar.addWidget(widget)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_pixmap(self, pixmap: QPixmap) -> None:
        """Replace the displayed image; keeps current zoom/pan."""
        self._pixmap = pixmap
        self._apply_zoom()

    def fit(self) -> None:
        """Fit the image to the available viewport."""
        if self._pixmap is None or self._pixmap.isNull():
            return
        vw = self._scroll.viewport().width()
        vh = self._scroll.viewport().height()
        if vw <= 0 or vh <= 0:
            return
        pw = self._pixmap.width()
        ph = self._pixmap.height()
        if pw <= 0 or ph <= 0:
            return
        self.set_zoom(min(vw / pw, vh / ph))

    @property
    def zoom(self) -> float:
        return self._zoom

    def set_zoom(self, factor: float) -> None:
        factor = max(_ZOOM_MIN, min(_ZOOM_MAX, factor))
        if abs(factor - self._zoom) < 1e-5:
            return
        self._zoom = factor
        self._zoom_slider.blockSignals(True)
        self._zoom_slider.setValue(int(factor * _ZOOM_SLIDER_SCALE))
        self._zoom_slider.blockSignals(False)
        self._zoom_pct.setText(f"{int(factor * 100)}%")
        self._apply_zoom()
        self.zoom_changed.emit(self._zoom)

    # ------------------------------------------------------------------
    # Internal rendering
    # ------------------------------------------------------------------

    def _apply_zoom(self) -> None:
        if self._pixmap is None or self._pixmap.isNull():
            self._label.clear()
            return
        w = max(1, int(self._pixmap.width() * self._zoom))
        h = max(1, int(self._pixmap.height() * self._zoom))
        scaled = self._pixmap.scaled(
            w,
            h,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self._label.setPixmap(scaled)
        self._label.resize(scaled.size())

    def _change_zoom(self, delta: float, anchor: Optional[QPoint] = None) -> None:
        """Change zoom by *delta*, keeping *anchor* (viewport coords) stationary."""
        new_zoom = max(_ZOOM_MIN, min(_ZOOM_MAX, self._zoom + delta))
        if anchor is not None and self._pixmap is not None:
            # Keep the image pixel under the cursor fixed as we zoom.
            sb_h = self._scroll.horizontalScrollBar()
            sb_v = self._scroll.verticalScrollBar()
            # Image-space coordinates of the anchor before zoom.
            old_img_x = (sb_h.value() + anchor.x()) / self._zoom
            old_img_y = (sb_v.value() + anchor.y()) / self._zoom
            self.set_zoom(new_zoom)
            sb_h.setValue(int(old_img_x * new_zoom - anchor.x()))
            sb_v.setValue(int(old_img_y * new_zoom - anchor.y()))
        else:
            self.set_zoom(new_zoom)

    # ------------------------------------------------------------------
    # Slider
    # ------------------------------------------------------------------

    def _on_slider(self, val: int) -> None:
        self.set_zoom(val / _ZOOM_SLIDER_SCALE)

    # ------------------------------------------------------------------
    # Event filter (wheel + mouse on _label)
    # ------------------------------------------------------------------

    def eventFilter(self, watched, event: QEvent) -> bool:
        if watched is not self._label:
            return super().eventFilter(watched, event)

        t = event.type()
        handler = {
            QEvent.Type.Wheel: self._on_wheel_event,
            QEvent.Type.MouseButtonPress: self._on_mouse_press,
            QEvent.Type.MouseMove: self._on_mouse_move,
            QEvent.Type.MouseButtonRelease: self._on_mouse_release,
            QEvent.Type.MouseButtonDblClick: self._on_mouse_dblclick,
        }.get(t)

        if handler is not None:
            result = handler(event)
            if result is not None:
                return result

        if t == QEvent.Type.Gesture and QGestureEvent is not None:
            return self._handle_gesture(event)  # type: ignore[arg-type]

        return super().eventFilter(watched, event)

    def _on_wheel_event(self, event):
        return self._handle_wheel(event)  # type: ignore[arg-type]

    def _on_mouse_press(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self._is_panning = True
            self._pan_start = event.globalPosition().toPoint()
            self._scroll_h0 = self._scroll.horizontalScrollBar().value()
            self._scroll_v0 = self._scroll.verticalScrollBar().value()
            self._label.setCursor(Qt.CursorShape.ClosedHandCursor)
            return True
        return None

    def _on_mouse_move(self, event):
        if self._is_panning and self._pan_start is not None:
            δ = event.globalPosition().toPoint() - self._pan_start
            self._scroll.horizontalScrollBar().setValue(self._scroll_h0 - δ.x())
            self._scroll.verticalScrollBar().setValue(self._scroll_v0 - δ.y())
            return True
        self._label.setCursor(Qt.CursorShape.OpenHandCursor)
        return None

    def _on_mouse_release(self, event):
        if self._is_panning:
            self._is_panning = False
            self._pan_start = None
            self._label.setCursor(Qt.CursorShape.OpenHandCursor)
            return True
        return None

    def _on_mouse_dblclick(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.fit()
            return True
        return None

    def _handle_wheel(self, evt: QWheelEvent) -> bool:
        if evt.modifiers() & Qt.KeyboardModifier.ControlModifier:
            δ = _ZOOM_STEP if evt.angleDelta().y() > 0 else -_ZOOM_STEP
            anchor = evt.position().toPoint()
            self._change_zoom(δ, anchor)
            evt.accept()
            return True
        # Plain scroll — let QScrollArea handle it normally.
        return False

    def _handle_gesture(self, evt: QGestureEvent) -> bool:
        pinch = evt.gesture(Qt.GestureType.PinchGesture)
        if pinch and pinch.state() == Qt.GestureState.GestureUpdated:
            scale = pinch.scaleFactor()
            δ = (scale - 1.0) * self._zoom
            center = self._label.mapFromGlobal(pinch.hotSpot().toPoint())
            self._change_zoom(δ, center)
            evt.accept()
            return True
        return False

    # ------------------------------------------------------------------
    # Resize — re-apply zoom so scrollbars stay correct
    # ------------------------------------------------------------------

    def resizeEvent(self, event) -> None:  # noqa: N802
        super().resizeEvent(event)
        self._apply_zoom()

    def keyPressEvent(self, event) -> None:  # noqa: N802
        if event.key() == Qt.Key.Key_F:
            self.fit()
            return
        super().keyPressEvent(event)
