"""Shared SAHI sliced-training/preview settings widget for DetectKit."""

from __future__ import annotations

from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QLineEdit,
    QSpinBox,
)

from ..models import SliceTrainingSettings


class SliceSettingsGroup(QGroupBox):
    """Group box binding widgets to a SliceTrainingSettings block."""

    def __init__(self, parent=None) -> None:
        super().__init__("Sliced dataset / inference (SAHI)", parent)
        form = QFormLayout(self)

        self.chk_enabled = QCheckBox("Enable sliced training + preview")
        self.cmb_mode = QComboBox()
        self.cmb_mode.addItems(["auto_object", "auto_model", "custom"])
        self.spin_frac = QDoubleSpinBox()
        self.spin_frac.setRange(0.01, 0.9)
        self.spin_frac.setSingleStep(0.01)
        self.spin_ref = QDoubleSpinBox()
        self.spin_ref.setRange(0.0, 8192.0)
        self.spin_ref.setSingleStep(1.0)
        self.spin_w = QSpinBox()
        self.spin_w.setRange(0, 8192)
        self.spin_h = QSpinBox()
        self.spin_h.setRange(0, 8192)
        self.spin_overlap = QDoubleSpinBox()
        self.spin_overlap.setRange(0.0, 0.9)
        self.spin_overlap.setSingleStep(0.05)
        self.spin_min_area = QDoubleSpinBox()
        self.spin_min_area.setRange(0.0, 1.0)
        self.spin_min_area.setSingleStep(0.05)
        self.spin_neg = QDoubleSpinBox()
        self.spin_neg.setRange(0.0, 1.0)
        self.spin_neg.setSingleStep(0.05)
        self.txt_targets = QLineEdit()  # comma-separated apparent sizes
        self.chk_full = QCheckBox("Mix full frames")
        self.spin_merge = QDoubleSpinBox()
        self.spin_merge.setRange(0.0, 1.0)
        self.spin_merge.setSingleStep(0.05)

        form.addRow(self.chk_enabled)
        form.addRow("Geometry mode", self.cmb_mode)
        form.addRow("Object tile fraction", self.spin_frac)
        form.addRow("Reference body px (0 = auto/imgsz)", self.spin_ref)
        form.addRow("Custom tile W", self.spin_w)
        form.addRow("Custom tile H", self.spin_h)
        form.addRow("Overlap", self.spin_overlap)
        form.addRow("Min area ratio", self.spin_min_area)
        form.addRow("Negative tile fraction", self.spin_neg)
        form.addRow("Target sizes (px, comma)", self.txt_targets)
        form.addRow(self.chk_full)
        form.addRow("Merge threshold", self.spin_merge)

    def load_from(self, s: SliceTrainingSettings) -> None:
        self.chk_enabled.setChecked(bool(s.enabled))
        idx = self.cmb_mode.findText(s.geometry_mode)
        self.cmb_mode.setCurrentIndex(idx if idx >= 0 else 0)
        self.spin_frac.setValue(float(s.object_tile_fraction))
        self.spin_ref.setValue(float(s.reference_body_px))
        self.spin_w.setValue(int(s.slice_width))
        self.spin_h.setValue(int(s.slice_height))
        self.spin_overlap.setValue(float(s.overlap))
        self.spin_min_area.setValue(float(s.min_area_ratio))
        self.spin_neg.setValue(float(s.negative_tile_fraction))
        self.txt_targets.setText(", ".join(f"{v:g}" for v in s.target_sizes))
        self.chk_full.setChecked(bool(s.full_frame_mix))
        self.spin_merge.setValue(float(s.merge_threshold))

    def to_settings(self) -> SliceTrainingSettings:
        targets: list[float] = []
        for tok in self.txt_targets.text().split(","):
            tok = tok.strip()
            if tok:
                try:
                    targets.append(float(tok))
                except ValueError:
                    continue
        return SliceTrainingSettings(
            enabled=self.chk_enabled.isChecked(),
            geometry_mode=self.cmb_mode.currentText(),
            object_tile_fraction=self.spin_frac.value(),
            reference_body_px=self.spin_ref.value(),
            slice_width=self.spin_w.value(),
            slice_height=self.spin_h.value(),
            overlap=self.spin_overlap.value(),
            min_area_ratio=self.spin_min_area.value(),
            negative_tile_fraction=self.spin_neg.value(),
            target_sizes=targets or SliceTrainingSettings().target_sizes,
            full_frame_mix=self.chk_full.isChecked(),
            merge_threshold=self.spin_merge.value(),
        )
