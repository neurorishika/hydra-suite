"""Bounding-box selector dialog for RefineKit.

Shows a single video frame and lets the user drag a rectangle to choose
a spatial region of interest before the track editor opens.

Typical usage::

    dlg = BboxSelectorDialog(video_path, mid_frame, parent=self)
    if dlg.exec() == QDialog.DialogCode.Accepted:
        bbox = dlg.bbox   # (x1, y1, x2, y2) or None → use full frame
"""

from __future__ import annotations

from typing import Optional, Tuple

import cv2
from PySide6.QtCore import QPoint, QRect, Qt
from PySide6.QtGui import QColor, QImage, QMouseEvent, QPainter, QPen, QPixmap
from PySide6.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

# ---------------------------------------------------------------------------
# _BboxCanvas
# ---------------------------------------------------------------------------


class _BboxCanvas(QWidget):
    """Widget that displays a scaled video frame and tracks a rubber-band rect."""

    def __init__(
        self,
        scaled_pixmap: QPixmap,
        orig_size: Tuple[int, int],
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self._pixmap = scaled_pixmap
        self._orig_w, self._orig_h = orig_size
        self._start: Optional[QPoint] = None
        self._end: Optional[QPoint] = None
        self.setCursor(Qt.CursorShape.CrossCursor)
        self.setFixedSize(scaled_pixmap.size())

    # ------------------------------------------------------------------
    # Painting
    # ------------------------------------------------------------------

    def paintEvent(self, event) -> None:  # noqa: N802
        painter = QPainter(self)
        painter.drawPixmap(0, 0, self._pixmap)

        if self._start is not None and self._end is not None:
            rect = QRect(self._start, self._end).normalized()
            painter.fillRect(rect, QColor(0, 170, 255, 40))
            painter.setPen(QPen(QColor(0, 170, 255), 2))
            painter.drawRect(rect)

        painter.end()

    # ------------------------------------------------------------------
    # Mouse
    # ------------------------------------------------------------------

    def mousePressEvent(self, ev: QMouseEvent) -> None:  # noqa: N802
        """Begin a new selection rectangle at the left-button press position."""
        if ev.button() == Qt.MouseButton.LeftButton:
            self._start = ev.position().toPoint()
            self._end = self._start
            self.update()

    def mouseMoveEvent(self, ev: QMouseEvent) -> None:  # noqa: N802
        """Extend the selection rectangle to the current cursor position and repaint."""
        if self._start is not None:
            self._end = ev.position().toPoint()
            self.update()

    def mouseReleaseEvent(self, ev: QMouseEvent) -> None:  # noqa: N802
        """Finalise the selection rectangle at the release position and trigger a repaint."""
        if ev.button() == Qt.MouseButton.LeftButton and self._start is not None:
            self._end = ev.position().toPoint()
            self.update()

    # ------------------------------------------------------------------
    # Result
    # ------------------------------------------------------------------

    def _to_frame(self, pt: QPoint) -> Tuple[int, int]:
        """Map a widget point to original-frame pixel coordinates."""
        w = max(self.width(), 1)
        h = max(self.height(), 1)
        fx = int(pt.x() / w * self._orig_w)
        fy = int(pt.y() / h * self._orig_h)
        return fx, fy

    @property
    def selected_bbox(self) -> Optional[Tuple[int, int, int, int]]:
        """(x1, y1, x2, y2) in original frame pixels, or ``None`` if too small."""
        if self._start is None or self._end is None:
            return None
        rect = QRect(self._start, self._end).normalized()
        if rect.width() < 4 or rect.height() < 4:
            return None
        x1, y1 = self._to_frame(rect.topLeft())
        x2, y2 = self._to_frame(rect.bottomRight())
        return (
            max(0, x1),
            max(0, y1),
            min(self._orig_w, x2),
            min(self._orig_h, y2),
        )


# ---------------------------------------------------------------------------
# BboxSelectorDialog
# ---------------------------------------------------------------------------


class BboxSelectorDialog(QDialog):
    """Show a video frame and let the user draw a region-of-interest rectangle.

    Parameters
    ----------
    video_path:
        Path to the video file.
    frame_idx:
        Index of the frame to display (typically the midpoint of the range).
    parent:
        Parent widget.

    After ``exec()`` returns ``Accepted``, read :attr:`bbox` for the selected
    region in original-frame pixel coordinates.  If the user clicked
    *Include All Tracks* or no rectangle was drawn, ``bbox`` is ``None``
    and the caller should include all tracks in the time range.
    """

    def __init__(
        self,
        video_path: str,
        frame_idx: int,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Select Region of Interest")
        self._bbox: Optional[Tuple[int, int, int, int]] = None
        self._canvas: Optional[_BboxCanvas] = None

        layout = QVBoxLayout(self)

        instr = QLabel(
            "\u2139  Drag a rectangle over the area you want to review.\n"
            "Only tracks visible inside will be loaded into the track editor.\n"
            "Click \u201cInclude All Tracks\u201d to skip the selection."
        )
        instr.setWordWrap(True)
        instr.setStyleSheet("color: #9cdcfe; font-size: 11px; padding: 4px;")
        layout.addWidget(instr)

        # Load video frame
        cap = cv2.VideoCapture(video_path)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        safe_idx = max(0, min(frame_idx, total - 1)) if total > 0 else 0
        cap.set(cv2.CAP_PROP_POS_FRAMES, safe_idx)
        ret, frame = cap.read()
        orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
        orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480
        cap.release()

        if ret and frame is not None:
            orig_h_actual, orig_w_actual = frame.shape[:2]
            orig_w, orig_h = orig_w_actual, orig_h_actual
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            qimg = QImage(
                rgb.data,
                orig_w,
                orig_h,
                3 * orig_w,
                QImage.Format.Format_RGB888,
            )
            pixmap = QPixmap.fromImage(qimg)
            # Scale to fit comfortably in the dialog
            max_w, max_h = 900, 580
            scale = min(max_w / orig_w, max_h / orig_h, 1.0)
            disp_w = int(orig_w * scale)
            disp_h = int(orig_h * scale)
            scaled = pixmap.scaled(
                disp_w,
                disp_h,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
            self._canvas = _BboxCanvas(scaled, (orig_w, orig_h))
            layout.addWidget(self._canvas)
        else:
            layout.addWidget(
                QLabel(
                    "Could not decode video frame.\n"
                    "Click \u201cInclude All Tracks\u201d to proceed without a selection."
                )
            )

        # Buttons
        btn_row = QHBoxLayout()

        ok_btn = QPushButton("Confirm Selection")
        ok_btn.setToolTip("Use the drawn rectangle as the region of interest")
        ok_btn.clicked.connect(self._on_ok)
        btn_row.addWidget(ok_btn)

        skip_btn = QPushButton("Include All Tracks")
        skip_btn.setToolTip(
            "Skip region selection — load every track visible in the time range"
        )
        skip_btn.clicked.connect(self._on_skip)
        btn_row.addWidget(skip_btn)

        btn_row.addStretch()

        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        btn_row.addWidget(cancel_btn)

        layout.addLayout(btn_row)
        self.adjustSize()

    # ------------------------------------------------------------------

    def _on_ok(self) -> None:
        if self._canvas is not None:
            self._bbox = self._canvas.selected_bbox
        self.accept()

    def _on_skip(self) -> None:
        """Accept with no bbox — caller should include all tracks."""
        self._bbox = None
        self.accept()

    def keyPressEvent(self, event) -> None:  # noqa: N802
        """Confirm the current selection when Enter or Return is pressed."""
        if event.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
            self._on_ok()
            return
        super().keyPressEvent(event)

    @property
    def bbox(self) -> Optional[Tuple[int, int, int, int]]:
        """Selected region as ``(x1, y1, x2, y2)`` in original frame pixels.

        ``None`` if the user did not draw a rectangle (or clicked *Include All
        Tracks*).
        """
        return self._bbox
