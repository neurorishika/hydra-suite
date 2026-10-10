"""Live schematic of the SAHI tile grid (moved from DetectKit, S4 Task 15).

Shared-layer module: imports only Qt and ``utils``; never an app layer.
"""

from __future__ import annotations

from PySide6.QtCore import QSize, Qt
from PySide6.QtGui import QColor, QFontMetrics, QPainter, QPen
from PySide6.QtWidgets import QSizePolicy, QWidget

from hydra_suite.utils.slice_geometry import plan_tiles, tile_size_for_mode
from hydra_suite.utils.tiling_spec import BACKEND_DEFAULTS, DEFAULT_YOLO_IMGSZ

__all__ = ["TileLayoutPreview", "_TileLayoutPreview", "elide_at_word"]


def elide_at_word(text: str, metrics: QFontMetrics, width: int) -> str:
    """``text`` if it fits in ``width`` px, else whole words + "…" (never mid-word)."""
    if metrics.horizontalAdvance(text) <= width:
        return text
    words = text.split(" ")
    while words:
        words.pop()
        candidate = " ".join(words).rstrip(" ·") + " …" if words else "…"
        if metrics.horizontalAdvance(candidate) <= width:
            return candidate
    return "…"


class _TileLayoutPreview(QWidget):
    """Draw a compact, schematic view of the tile grid on a sample frame."""

    _FALLBACK_FRAME_WH = (1920, 1080)
    _SCALE_COLORS = ("#00a6d6", "#d16dff", "#f2a900", "#65c466", "#ff6b6b")
    # Painted chrome: title band above the frame, three caption lines below.
    _MARGIN, _TOP, _CAPTIONS = 12, 26, 73
    # Below the controls (preview_position="bottom") the preview follows the
    # host's width at the frame's aspect, within these heights.
    MIN_BOTTOM_HEIGHT = 200
    MAX_BOTTOM_HEIGHT = 380
    MIN_BOTTOM_WIDTH = 240
    # Compact hosts (layout="compact"): shorter, two caption lines.
    MAX_COMPACT_HEIGHT = 260
    _COMPACT_CAPTIONS = 56

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._bottom = False
        self._compact = False
        self.setMinimumSize(290, 180)
        self.setToolTip(
            "A live schematic of the tile grid over a representative labelled source "
            "frame. Click to compare each image size in the project distribution. "
            "The automatic body-size estimate is used after the first sliced dataset "
            "build."
        )
        self._mode = "auto_object"
        self._target_fractions = list(
            BACKEND_DEFAULTS["yolo_train"].object_tile_fractions
        )
        self._slice_wh = (0, 0)
        self._overlap = 0.2
        self._model_input_size = DEFAULT_YOLO_IMGSZ
        self._reference_body_px = 0.0
        self._frame_wh = self._FALLBACK_FRAME_WH
        self._uses_fallback_frame = True
        self._frame_options: list[tuple[int, int, int]] = []
        self._frame_index = 0
        # Caption for a single host-supplied frame (set_frame_size); None for
        # a project distribution (set_frame_options).
        self._single_frame_note: str | None = None
        # Wording of the body-size note; hosts whose body size is not
        # measured from labels (inference) replace it via set_body_notes.
        self._measured_note = "uses last label measurement"
        self._unmeasured_note = "illustrative until labels are measured"

    def set_bottom_layout(self, bottom: bool, *, compact: bool = False) -> None:
        """Below the controls: full host width, height from the frame aspect.

        ``compact`` (a ``layout="compact"`` host) caps the height lower and
        paints the captions on two lines instead of three.
        """
        self._bottom = bool(bottom)
        self._compact = bool(bottom and compact)
        if self._bottom:
            policy = QSizePolicy(
                QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred
            )
            policy.setHeightForWidth(True)
            self.setSizePolicy(policy)
            self.setMinimumSize(self.MIN_BOTTOM_WIDTH, self.MIN_BOTTOM_HEIGHT)
        else:
            self.setSizePolicy(
                QSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Preferred)
            )
            self.setMinimumSize(290, 180)
        self.updateGeometry()

    def hasHeightForWidth(self) -> bool:  # noqa: N802 - Qt override
        return self._bottom

    def heightForWidth(self, width: int) -> int:  # noqa: N802 - Qt override
        if not self._bottom:
            return super().heightForWidth(width)
        frame_w, frame_h = self._frame_wh
        drawable = max(1, width - 2 * self._MARGIN)
        height = (
            self._TOP + self._captions_height() + round(drawable * frame_h / frame_w)
        )
        ceiling = self.MAX_COMPACT_HEIGHT if self._compact else self.MAX_BOTTOM_HEIGHT
        return max(self.MIN_BOTTOM_HEIGHT, min(height, ceiling))

    def _captions_height(self) -> int:
        return self._COMPACT_CAPTIONS if self._compact else self._CAPTIONS

    def sizeHint(self) -> QSize:  # noqa: N802 - Qt override
        if not self._bottom:
            return super().sizeHint()
        return QSize(480, self.heightForWidth(480))

    def resizeEvent(self, event) -> None:  # noqa: N802 - Qt override
        # Fallback for host layouts that do not propagate height-for-width
        # (stacked pages in a scroll area): pin the minimum height to the
        # width-derived one. Stable: the height depends only on the width.
        super().resizeEvent(event)
        if self._bottom:
            wanted = self.heightForWidth(self.width())
            if self.minimumHeight() != wanted:
                self.setMinimumHeight(wanted)

    def set_body_notes(self, measured: str, unmeasured: str) -> None:
        """Set the caption shown with a known / unknown reference body size."""
        self._measured_note = measured
        self._unmeasured_note = unmeasured
        self.update()

    @property
    def frame_size(self) -> tuple[int, int]:
        """Representative source-frame dimensions currently shown by the preview."""
        return self._frame_wh

    @property
    def frame_options(self) -> list[tuple[int, int, int]]:
        """Distinct project frame sizes available to cycle through."""
        return list(self._frame_options)

    def set_frame_size(self, frame_wh: tuple[int, int] | None) -> None:
        """Use a project source frame, or the labelled fallback when unavailable."""
        self.set_frame_options(
            [] if frame_wh is None else [(frame_wh[0], frame_wh[1], 1)]
        )
        self._single_frame_note = None if frame_wh is None else "source frame"
        self.update()

    def set_frame_options(self, options: list[tuple[int, int, int]]) -> None:
        """Set the project image-size distribution shown by click-to-cycle preview."""
        self._single_frame_note = None
        current = self._frame_wh
        counts: dict[tuple[int, int], int] = {}
        for width, height, count in options:
            if int(width) <= 0 or int(height) <= 0 or int(count) <= 0:
                continue
            size = (int(width), int(height))
            counts[size] = counts.get(size, 0) + int(count)
        self._frame_options = [
            (width, height, count)
            for (width, height), count in sorted(
                counts.items(), key=lambda item: (-item[1], item[0])
            )
        ]
        if not self._frame_options:
            self._frame_wh = self._FALLBACK_FRAME_WH
            self._uses_fallback_frame = True
        else:
            self._uses_fallback_frame = False
            self._frame_index = next(
                (
                    index
                    for index, (width, height, _count) in enumerate(self._frame_options)
                    if (width, height) == current
                ),
                0,
            )
            width, height, _count = self._frame_options[self._frame_index]
            self._frame_wh = (width, height)
        self._aspect_changed()
        self.update()

    def _aspect_changed(self) -> None:
        if self._bottom:
            self.setMinimumHeight(self.heightForWidth(max(1, self.width())))
            self.updateGeometry()

    def mousePressEvent(self, event) -> None:  # noqa: N802 - Qt override
        if event.button() == Qt.MouseButton.LeftButton and len(self._frame_options) > 1:
            self._frame_index = (self._frame_index + 1) % len(self._frame_options)
            width, height, _count = self._frame_options[self._frame_index]
            self._frame_wh = (width, height)
            self._aspect_changed()
            self.update()
            event.accept()
            return
        super().mousePressEvent(event)

    def set_settings(
        self,
        *,
        mode: str,
        target_fractions: list[float],
        slice_width: int,
        slice_height: int,
        overlap: float,
        model_input_size: int,
        reference_body_px: float,
    ) -> None:
        self._mode = mode
        self._target_fractions = target_fractions
        self._slice_wh = (slice_width, slice_height)
        self._overlap = overlap
        self._model_input_size = max(1, int(model_input_size))
        self._reference_body_px = max(0.0, float(reference_body_px))
        self.update()

    def _tile_specs(self) -> list[tuple[float | None, int, int, str]]:
        """Return every visible SAHI target scale and its resolved tile geometry."""
        reference = self._reference_body_px
        if self._mode == "auto_object" and reference <= 0.0:
            # Labels have not been measured before the first build. This keeps
            # the preview useful without presenting an illustrative value as a
            # real measurement.
            reference = self._frame_wh[1] / 18.0
        fractions = self._target_fractions if self._mode == "auto_object" else [None]
        specs: list[tuple[float | None, int, int, str]] = []
        for index, fraction in enumerate(fractions):
            tile_w, tile_h = tile_size_for_mode(
                geometry_mode=self._mode,
                imgsz=self._model_input_size,
                reference_body_px=reference,
                object_tile_fraction=(
                    float(fraction)
                    if fraction is not None
                    else BACKEND_DEFAULTS["yolo_infer"].object_tile_fractions[0]
                ),
                slice_width=self._slice_wh[0],
                slice_height=self._slice_wh[1],
            )
            if (fraction, tile_w, tile_h) not in [spec[:3] for spec in specs]:
                specs.append(
                    (
                        fraction,
                        tile_w,
                        tile_h,
                        self._SCALE_COLORS[index % len(self._SCALE_COLORS)],
                    )
                )
        return specs or [
            (None, self._model_input_size, self._model_input_size, "#00a6d6")
        ]

    def paintEvent(self, _event) -> None:  # noqa: N802 - Qt override
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.fillRect(self.rect(), QColor("#202020"))

        margin, top = self._MARGIN, self._TOP
        available_w = max(1, self.width() - 2 * margin)
        available_h = max(1, self.height() - top - self._captions_height())
        scale = min(available_w / self._frame_wh[0], available_h / self._frame_wh[1])
        draw_w, draw_h = int(self._frame_wh[0] * scale), int(self._frame_wh[1] * scale)
        x = (self.width() - draw_w) // 2
        y = top + (available_h - draw_h) // 2

        painter.setPen(QPen(QColor("#808080"), 1))
        painter.setBrush(QColor("#111111"))
        painter.drawRect(x, y, draw_w, draw_h)

        specs = self._tile_specs()
        grid_spec = specs[len(specs) // 2]
        _fraction, tile_w, tile_h, color = grid_spec
        try:
            plan = plan_tiles(
                (self._frame_wh[1], self._frame_wh[0]),
                tile_w,
                tile_h,
                self._overlap,
                self._overlap,
            )
            tiles = plan.tiles
        except ValueError:
            tiles = []

        painter.setPen(QPen(QColor(color), 1.2))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        for x0, y0, x1, y1 in tiles:
            painter.drawRect(
                x + round(x0 * scale),
                y + round(y0 * scale),
                max(1, round((x1 - x0) * scale)),
                max(1, round((y1 - y0) * scale)),
            )

        # The median scale supplies the full grid. The other target scales are
        # nested at the origin so their different tile extents remain legible.
        for fraction, other_w, other_h, other_color in specs:
            if (fraction, other_w, other_h, other_color) == grid_spec:
                continue
            pen = QPen(QColor(other_color), 1.4, Qt.PenStyle.DashLine)
            painter.setPen(pen)
            painter.drawRect(
                x,
                y,
                max(1, round(other_w * scale)),
                max(1, round(other_h * scale)),
            )

        title = self.caption_texts(len(tiles))[0]
        metrics = QFontMetrics(painter.font())
        text_w = self.width() - 2 * margin
        painter.setPen(QColor("#f0f0f0"))
        painter.drawText(margin, 17, elide_at_word(title, metrics, text_w))
        painter.setPen(QColor("#c0c0c0"))
        lines = self.caption_lines(len(tiles))
        for offset, line in enumerate(reversed(lines)):
            painter.drawText(
                margin,
                self.height() - 17 * (offset + 1),
                elide_at_word(line, metrics, text_w),
            )

    def caption_lines(self, tile_count: int | None = None) -> list[str]:
        """The caption lines painted below the frame (two when compact)."""
        _title, tile_line, note_line, scales_line = self.caption_texts(tile_count)
        if self._compact:
            # Fold the body note onto the tile line: two lines, not three.
            return [
                " · ".join(line for line in (tile_line, note_line) if line),
                scales_line,
            ]
        return [line for line in (tile_line, note_line, scales_line) if line]

    def _plan_tile_count(self, tile_w: int, tile_h: int) -> int:
        try:
            plan = plan_tiles(
                (self._frame_wh[1], self._frame_wh[0]),
                tile_w,
                tile_h,
                self._overlap,
                self._overlap,
            )
        except ValueError:
            return 0
        return len(plan.tiles)

    def caption_texts(self, tile_count: int | None = None) -> tuple[str, str, str, str]:
        """(title, tile line, body note, scales line) -- the painted captions.

        A frame the host never supplied is labelled an EXAMPLE frame in the
        title, so a 1920 x 1080 grid is never mistaken for the real video.
        """
        specs = self._tile_specs()
        _fraction, tile_w, tile_h, _color = specs[len(specs) // 2]
        if tile_count is None:
            tile_count = self._plan_tile_count(tile_w, tile_h)
        width, height = self._frame_wh
        if self._uses_fallback_frame:
            title = f"Tile layout on an example {width} × {height} frame"
            frame_note = "example frame"
        else:
            title = f"Tile layout on a {width} × {height} image"
            if len(self._frame_options) > 1:
                title += " · click to compare"
            if self._single_frame_note is not None:
                frame_note = self._single_frame_note
            else:
                _w, _h, count = self._frame_options[self._frame_index]
                frame_note = (
                    f"project size {self._frame_index + 1}/"
                    f"{len(self._frame_options)} · {count} frame(s)"
                )
        tile_line = f"{tile_count} tiles · {tile_w} × {tile_h} px · {frame_note}"
        note = (
            self._measured_note
            if self._reference_body_px > 0.0 and self._mode == "auto_object"
            else (self._unmeasured_note if self._mode == "auto_object" else "")
        )
        scale_note = (
            " · ".join(
                f"● {fraction:g} → {tile_px} px"
                for fraction, tile_px, _h, _color in specs
                if fraction is not None
            )
            or f"● {tile_w} × {tile_h} px"
        )
        return title, tile_line, note, f"Scales: {scale_note}"


# The public name; ``_TileLayoutPreview`` stays importable for old callers.
TileLayoutPreview = _TileLayoutPreview
