"""Square image tile shared by the grid dialogs (Grid Balance, Label Review)."""

from __future__ import annotations

from typing import Optional

from PySide6.QtCore import QRectF, QSize, Qt, QTimer
from PySide6.QtGui import QColor, QFont, QImage, QImageReader, QPainter, QPixmap
from PySide6.QtWidgets import QAbstractButton


def load_scaled(path: str, edge: int) -> Optional[QImage]:
    """Read ``path`` downscaled to fit ``edge`` pixels; ``None`` if unreadable."""
    reader = QImageReader(path)
    reader.setAutoTransform(True)
    size = reader.size()
    if size.isValid() and max(size.width(), size.height()) > edge:
        reader.setScaledSize(size.scaled(QSize(edge, edge), Qt.KeepAspectRatio))
    image = reader.read()
    return None if image.isNull() else image


class ImageTile(QAbstractButton):
    """One grid cell: image, key badge, optional caption, selection and flash."""

    def __init__(self, key: str, edge: int, parent=None) -> None:
        super().__init__(parent)
        self.key = key
        self.pixmap: Optional[QPixmap] = None
        self.empty = True
        self.note = ""
        self.caption = ""
        self.caption_color = "#ddd"
        self.tag = ""
        self.selected = False
        self._flash = False
        self.setFixedSize(edge, edge)
        self.setCursor(Qt.PointingHandCursor)
        self.setFocusPolicy(Qt.NoFocus)

    def set_image(self, pixmap: Optional[QPixmap], empty: bool, note: str = "") -> None:
        self.pixmap, self.empty, self.note = pixmap, empty, note
        self.setEnabled(not empty)
        self.update()

    def set_caption(self, text: str, color: str = "#ddd", tag: str = "") -> None:
        self.caption, self.caption_color, self.tag = text, color, tag
        self.update()

    def set_selected(self, selected: bool) -> None:
        self.selected = bool(selected)
        self.update()

    def flash(self) -> None:
        self._flash = True
        self.update()
        QTimer.singleShot(180, self._end_flash)

    def _end_flash(self) -> None:
        self._flash = False
        self.update()

    def mouseDoubleClickEvent(self, event) -> None:
        # The second press of a double-click would act on a tile whose content
        # may just have changed.
        event.accept()

    def paintEvent(self, _event) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.SmoothPixmapTransform, True)
        r = self.rect()
        p.fillRect(r, QColor("#121212"))
        if self.empty or self.pixmap is None:
            p.setPen(QColor("#555" if self.empty else "#666"))
            text = self.note or ("no more\ncandidates" if self.empty else "loading…")
            p.drawText(r, Qt.AlignCenter, text)
        else:
            # Fit in logical pixels so HiDPI pixmaps stay sharp.
            src = self.pixmap.deviceIndependentSize()
            scale = min(r.width() / src.width(), r.height() / src.height())
            w, h = src.width() * scale, src.height() * scale
            target = QRectF((r.width() - w) / 2, (r.height() - h) / 2, w, h)
            p.drawPixmap(target, self.pixmap, QRectF(self.pixmap.rect()))
        if self.selected:
            p.fillRect(r, QColor(229, 57, 53, 70))
        if self._flash:
            p.fillRect(r, QColor(78, 201, 67, 90))
        if self._flash:
            border = "#4ec943"
        elif self.selected:
            border = "#e53935"
        elif self.underMouse() and self.isEnabled():
            border = "#0e639c"
        else:
            border = "#3e3e42"
        p.setPen(QColor(border))
        p.drawRect(r.adjusted(0, 0, -1, -1))
        if self.selected:
            p.drawRect(r.adjusted(1, 1, -2, -2))
            p.drawRect(r.adjusted(2, 2, -3, -3))
        font = QFont(p.font())
        font.setBold(True)
        p.setFont(font)
        if not self.empty and self.key:
            badge = QRectF(4, 4, 20, 20)
            p.fillRect(badge, QColor(0, 0, 0, 170))
            p.setPen(QColor("#ddd"))
            p.drawText(badge, Qt.AlignCenter, self.key)
        if not self.empty and self.tag:
            badge = QRectF(r.width() - 24, 4, 20, 20)
            p.fillRect(badge, QColor(0, 0, 0, 170))
            p.setPen(QColor("#e0a040"))
            p.drawText(badge, Qt.AlignCenter, self.tag)
        if not self.empty and self.selected:
            mark = QRectF(r.width() - 26, r.height() - 46, 22, 22)
            p.fillRect(mark, QColor("#e53935"))
            p.setPen(QColor("#fff"))
            p.drawText(mark, Qt.AlignCenter, "✕")
        if not self.empty and self.caption:
            font.setBold(False)
            font.setPointSizeF(max(7.0, font.pointSizeF() - 1))
            p.setFont(font)
            strip = QRectF(0, r.height() - 20, r.width(), 20)
            p.fillRect(strip, QColor(0, 0, 0, 185))
            p.setPen(QColor(self.caption_color))
            text = p.fontMetrics().elidedText(
                self.caption, Qt.ElideRight, int(strip.width()) - 8
            )
            p.drawText(strip.adjusted(4, 0, -4, 0), Qt.AlignVCenter, text)
        p.end()

    def enterEvent(self, event) -> None:
        self.update()
        super().enterEvent(event)

    def leaveEvent(self, event) -> None:
        self.update()
        super().leaveEvent(event)
