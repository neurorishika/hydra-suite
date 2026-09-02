"""Focused SAHI sliced-training controls and a live tile-layout preview."""

from __future__ import annotations

from statistics import median

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QPainter, QPen
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QSpinBox,
    QWidget,
)

from hydra_suite.utils.slice_geometry import plan_tiles, tile_size_for_mode

from ..models import SliceTrainingSettings


class _TileLayoutPreview(QWidget):
    """Draw a compact, schematic view of the tile grid on a sample frame."""

    _FRAME_WH = (1920, 1080)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setMinimumSize(290, 180)
        self.setToolTip(
            "A live schematic of the tile grid over a representative 1920 × 1080 "
            "source image. The automatic body-size estimate is used after the first "
            "sliced dataset build."
        )
        self._mode = "auto_object"
        self._target_fractions = [0.3125, 0.46875, 0.625]
        self._slice_wh = (0, 0)
        self._overlap = 0.2
        self._model_input_size = 640
        self._reference_body_px = 0.0

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

    def _tile_size(self) -> tuple[int, int]:
        reference = self._reference_body_px
        if self._mode == "auto_object" and reference <= 0.0:
            # Labels have not been measured before the first build. This keeps
            # the preview useful without presenting an illustrative value as a
            # real measurement.
            reference = self._FRAME_WH[1] / 18.0
        fraction = (
            float(median(self._target_fractions)) if self._target_fractions else 0.15
        )
        return tile_size_for_mode(
            geometry_mode=self._mode,
            imgsz=self._model_input_size,
            reference_body_px=reference,
            object_tile_fraction=fraction,
            slice_width=self._slice_wh[0],
            slice_height=self._slice_wh[1],
        )

    def paintEvent(self, _event) -> None:  # noqa: N802 - Qt override
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.fillRect(self.rect(), QColor("#202020"))

        margin, top = 12, 26
        available_w = max(1, self.width() - 2 * margin)
        available_h = max(1, self.height() - top - 38)
        scale = min(available_w / self._FRAME_WH[0], available_h / self._FRAME_WH[1])
        draw_w, draw_h = int(self._FRAME_WH[0] * scale), int(self._FRAME_WH[1] * scale)
        x = (self.width() - draw_w) // 2
        y = top + (available_h - draw_h) // 2

        painter.setPen(QPen(QColor("#808080"), 1))
        painter.setBrush(QColor("#111111"))
        painter.drawRect(x, y, draw_w, draw_h)

        tile_w, tile_h = self._tile_size()
        try:
            plan = plan_tiles(
                (self._FRAME_WH[1], self._FRAME_WH[0]),
                tile_w,
                tile_h,
                self._overlap,
                self._overlap,
            )
            tiles = plan.tiles
        except ValueError:
            tiles = []

        painter.setPen(QPen(QColor("#00a6d6"), 1.2))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        for x0, y0, x1, y1 in tiles:
            painter.drawRect(
                x + round(x0 * scale),
                y + round(y0 * scale),
                max(1, round((x1 - x0) * scale)),
                max(1, round((y1 - y0) * scale)),
            )

        painter.setPen(QColor("#f0f0f0"))
        painter.drawText(
            margin,
            17,
            f"Tile layout on a {self._FRAME_WH[0]} × {self._FRAME_WH[1]} image",
        )
        note = (
            "uses last label measurement"
            if self._reference_body_px > 0.0 and self._mode == "auto_object"
            else (
                "illustrative until labels are measured"
                if self._mode == "auto_object"
                else ""
            )
        )
        tile_note = f"{len(tiles)} tiles · {tile_w} × {tile_h} px"
        painter.setPen(QColor("#c0c0c0"))
        painter.drawText(margin, self.height() - 17, f"{tile_note} {note}".strip())


class SliceSettingsGroup(QGroupBox):
    """SAHI settings that reveal only controls relevant to the geometry mode."""

    _DEFAULT_MODEL_INPUT_SIZE = 640

    def __init__(self, parent=None) -> None:
        super().__init__("Sliced dataset / inference (SAHI)", parent)
        self._model_input_size = self._DEFAULT_MODEL_INPUT_SIZE
        self._reference_body_px = 0.0
        outer = QHBoxLayout(self)
        outer.setContentsMargins(16, 18, 16, 14)
        outer.setSpacing(18)

        controls = QWidget()
        grid = QGridLayout(controls)
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setHorizontalSpacing(12)
        grid.setVerticalSpacing(8)

        self.chk_enabled = QCheckBox("Enable sliced training + preview")
        self.cmb_mode = QComboBox()
        self.cmb_mode.addItem("Fit labelled objects", "auto_object")
        self.cmb_mode.addItem("Use model input", "auto_model")
        self.cmb_mode.addItem("Custom tile size", "custom")
        self.txt_targets = QLineEdit()
        self.txt_targets.setPlaceholderText("e.g. 0.31, 0.47, 0.62")
        self.txt_targets.setToolTip(
            "Object size as a fraction of the model input. At a 640px input, "
            "0.31 means about 200px. Larger fractions create smaller tiles."
        )
        self.auto_reference_note = QLabel()
        self.auto_reference_note.setWordWrap(True)
        self.auto_reference_note.setStyleSheet("color: #b8d9e6;")

        self.spin_w = QSpinBox()
        self.spin_w.setRange(0, 8192)
        self.spin_w.setSpecialValueText("Model input size")
        self.spin_h = QSpinBox()
        self.spin_h.setRange(0, 8192)
        self.spin_h.setSpecialValueText("Model input size")
        self.spin_overlap = QDoubleSpinBox()
        self.spin_overlap.setRange(0.0, 0.9)
        self.spin_overlap.setSingleStep(0.05)
        self.spin_min_area = QDoubleSpinBox()
        self.spin_min_area.setRange(0.0, 1.0)
        self.spin_min_area.setSingleStep(0.05)
        self.spin_neg = QDoubleSpinBox()
        self.spin_neg.setRange(0.0, 1.0)
        self.spin_neg.setSingleStep(0.05)
        self.chk_full = QCheckBox("Mix full frames")
        self.spin_merge = QDoubleSpinBox()
        self.spin_merge.setRange(0.0, 1.0)
        self.spin_merge.setSingleStep(0.05)

        for control in (
            self.cmb_mode,
            self.txt_targets,
            self.spin_w,
            self.spin_h,
            self.spin_overlap,
            self.spin_min_area,
            self.spin_neg,
            self.spin_merge,
        ):
            control.setMaximumWidth(210)

        self._rows: dict[str, tuple[QLabel, QWidget]] = {}

        def add_row(row: int, key: str, label: str, control: QWidget) -> None:
            label_widget = QLabel(label)
            self._rows[key] = (label_widget, control)
            grid.addWidget(label_widget, row, 0)
            grid.addWidget(control, row, 1)

        grid.addWidget(self.chk_enabled, 0, 0, 1, 2)
        add_row(1, "mode", "Tile strategy", self.cmb_mode)
        add_row(2, "targets", "Object scale in model input", self.txt_targets)
        grid.addWidget(self.auto_reference_note, 3, 0, 1, 2)
        add_row(4, "width", "Tile width", self.spin_w)
        add_row(5, "height", "Tile height", self.spin_h)
        add_row(6, "overlap", "Tile overlap", self.spin_overlap)
        add_row(7, "min_area", "Keep labelled area", self.spin_min_area)
        add_row(8, "negative", "Keep empty tiles", self.spin_neg)
        grid.addWidget(self.chk_full, 9, 0, 1, 2)
        add_row(10, "merge", "Merge threshold", self.spin_merge)

        self.preview = _TileLayoutPreview()
        outer.addWidget(controls, 0)
        outer.addWidget(self.preview, 1)

        self.cmb_mode.currentIndexChanged.connect(self._refresh_mode_controls)
        for signal in (
            self.txt_targets.textChanged,
            self.spin_w.valueChanged,
            self.spin_h.valueChanged,
            self.spin_overlap.valueChanged,
        ):
            signal.connect(self._refresh_preview)
        self._refresh_mode_controls()

    def set_model_input_size(self, imgsz: int) -> None:
        """Set the active model input size used to resolve relative scales."""
        self._model_input_size = max(1, int(imgsz))
        self._refresh_preview()

    @staticmethod
    def _format_fractions(fractions: list[float]) -> str:
        return ", ".join(f"{value:.8g}" for value in fractions)

    def _target_fractions(self) -> list[float]:
        fractions: list[float] = []
        for token in self.txt_targets.text().split(","):
            try:
                value = float(token.strip())
            except ValueError:
                continue
            if 0.0 < value <= 1.0:
                fractions.append(value)
        return fractions or SliceTrainingSettings().target_fractions()

    def _refresh_mode_controls(self) -> None:
        mode = str(self.cmb_mode.currentData() or "auto_object")
        visible = {
            "mode": True,
            "targets": mode == "auto_object",
            "width": mode == "custom",
            "height": mode == "custom",
            "overlap": True,
            "min_area": True,
            "negative": True,
            "merge": True,
        }
        for key, (label, control) in self._rows.items():
            label.setVisible(visible[key])
            control.setVisible(visible[key])
        self.auto_reference_note.setVisible(mode == "auto_object")
        self.chk_full.setVisible(True)
        self._refresh_preview()

    def _refresh_preview(self, *_args) -> None:
        self.preview.set_settings(
            mode=str(self.cmb_mode.currentData() or "auto_object"),
            target_fractions=self._target_fractions(),
            slice_width=self.spin_w.value(),
            slice_height=self.spin_h.value(),
            overlap=self.spin_overlap.value(),
            model_input_size=self._model_input_size,
            reference_body_px=self._reference_body_px,
        )

    def load_from(self, s: SliceTrainingSettings) -> None:
        self.chk_enabled.setChecked(bool(s.enabled))
        index = self.cmb_mode.findData(s.geometry_mode)
        self.cmb_mode.setCurrentIndex(index if index >= 0 else 0)
        self.txt_targets.setText(self._format_fractions(s.target_fractions()))
        self.spin_w.setValue(int(s.slice_width))
        self.spin_h.setValue(int(s.slice_height))
        self.spin_overlap.setValue(float(s.overlap))
        self.spin_min_area.setValue(float(s.min_area_ratio))
        self.spin_neg.setValue(float(s.negative_tile_fraction))
        self.chk_full.setChecked(bool(s.full_frame_mix))
        self.spin_merge.setValue(float(s.merge_threshold))
        self._reference_body_px = float(s.reference_body_px)
        if s.reference_body_px > 0.0:
            self.auto_reference_note.setText(
                f"Reference body is measured automatically from labels. Last build: {s.reference_body_px:.1f}px."
            )
        else:
            self.auto_reference_note.setText(
                "Reference body is measured automatically from all labelled objects when the sliced dataset is built."
            )
        self._refresh_mode_controls()

    def to_settings(self) -> SliceTrainingSettings:
        fractions = self._target_fractions()
        return SliceTrainingSettings(
            enabled=self.chk_enabled.isChecked(),
            geometry_mode=str(self.cmb_mode.currentData() or "auto_object"),
            object_tile_fraction=float(median(fractions)),
            # This is output metadata from the previous build, not a user
            # override. Dataset preparation always remeasures labels.
            reference_body_px=self._reference_body_px,
            slice_width=self.spin_w.value(),
            slice_height=self.spin_h.value(),
            overlap=self.spin_overlap.value(),
            min_area_ratio=self.spin_min_area.value(),
            negative_tile_fraction=self.spin_neg.value(),
            target_size_fractions=fractions,
            target_sizes=[
                fraction * self._DEFAULT_MODEL_INPUT_SIZE for fraction in fractions
            ],
            full_frame_mix=self.chk_full.isChecked(),
            merge_threshold=self.spin_merge.value(),
        )
