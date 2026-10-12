"""ViTPose training recipe: augmentation, optimisation and validation.

Builds the controls and owns their settings round-trip; the Training Runner
reads the run parameters through :meth:`ViTPoseRecipePanel.params` (keys of
``vitpose.training.config.RunConfig``).  Defaults are the recommended recipe
(mmpose ViTPose / RTMPose-style fine-tuning); *Recommended settings* restores
them.
"""

from __future__ import annotations

import os
from typing import Any, Dict

from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

SPLIT_LABELS = {
    "frame": "Source frame (recommended)",
    "video": "Video",
    "image": "Image (random)",
}

# Recommended recipe.  Keys are RunConfig fields; "auto" means the variant's
# own ViTPose value (layer decay, drop path).
RECOMMENDED: Dict[str, Any] = {
    "rotation": 40.0,
    "rotation_prob": 0.6,
    "brightness": 0.3,
    "contrast": 0.3,
    "saturation": 0.3,
    "hue": 0.02,
    "blur_prob": 0.1,
    "noise_prob": 0.1,
    "occlusion_prob": 0.3,
    "lr": 5e-4,
    "weight_decay": 0.1,
    "layer_decay": None,
    "drop_path": None,
    "warmup_epochs": 3.0,
    "ema": True,
    "amp": True,
    "num_workers": min(8, os.cpu_count() or 1),
    "patience": 40,
    "val_fraction": 0.2,
    "split_by": "frame",
    "seed": 0,
    "ignore_occluded": False,
    "flip_test": True,
}


def _dspin(lo, hi, step, decimals, tip):
    w = QDoubleSpinBox()
    w.setRange(lo, hi)
    w.setSingleStep(step)
    w.setDecimals(decimals)
    w.setToolTip(tip)
    return w


class ViTPoseRecipePanel(QWidget):
    """Augmentation, optimisation and validation controls for ViTPose."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)

        # ── augmentation ────────────────────────────────────────────────
        aug = QGroupBox("ViTPose Augmentation")
        grid = QGridLayout(aug)
        self.rotation = _dspin(
            0, 180, 5, 0, "Rotation range in degrees (drawn from N(0, r/2), clipped)."
        )
        self.rotation_prob = _dspin(
            0, 1, 0.05, 2, "Fraction of samples that are rotated."
        )
        self.brightness = _dspin(0, 0.9, 0.05, 2, "Brightness jitter (+/- fraction).")
        self.contrast = _dspin(0, 0.9, 0.05, 2, "Contrast jitter (+/- fraction).")
        self.saturation = _dspin(0, 0.9, 0.05, 2, "Saturation jitter (+/- fraction).")
        self.hue = _dspin(
            0, 0.5, 0.01, 2, "Hue shift (fraction of the colour circle; keep small)."
        )
        self.blur_prob = _dspin(
            0, 1, 0.05, 2, "Probability of a Gaussian or median blur (motion, focus)."
        )
        self.noise_prob = _dspin(
            0, 1, 0.05, 2, "Probability of Gaussian pixel noise (sensor noise)."
        )
        self.occlusion_prob = _dspin(
            0,
            1,
            0.05,
            2,
            "Probability of pasting one random rectangle over the crop, so the "
            "model learns keypoints hidden behind neighbours or debris.",
        )
        rows = [
            ("rotation (deg)", self.rotation, "rotation prob", self.rotation_prob),
            ("brightness", self.brightness, "contrast", self.contrast),
            ("saturation", self.saturation, "hue", self.hue),
            ("blur prob", self.blur_prob, "noise prob", self.noise_prob),
            ("occlusion prob", self.occlusion_prob, None, None),
        ]
        for r, (la, wa, lb, wb) in enumerate(rows):
            grid.addWidget(QLabel(la), r, 0)
            grid.addWidget(wa, r, 1)
            if lb:
                grid.addWidget(QLabel(lb), r, 2)
                grid.addWidget(wb, r, 3)
        outer.addWidget(aug)

        # ── optimisation ────────────────────────────────────────────────
        opt = QGroupBox("ViTPose Optimisation")
        form = QFormLayout(opt)
        self.lr = _dspin(1e-6, 1e-2, 1e-4, 6, "AdamW base learning rate.")
        self.weight_decay = _dspin(0, 1, 0.01, 3, "AdamW weight decay.")
        self.layer_decay = _dspin(
            0,
            1,
            0.05,
            2,
            "Layer-wise learning-rate decay (lower layers learn slower). "
            "Auto = the variant's ViTPose value.",
        )
        self.layer_decay.setSpecialValueText("Auto")
        self.drop_path = _dspin(
            0,
            0.9,
            0.05,
            2,
            "Stochastic depth rate. Auto = the variant's ViTPose value.",
        )
        self.drop_path.setSpecialValueText("Auto")
        self.warmup_epochs = _dspin(
            0, 50, 0.5, 1, "Linear learning-rate warmup before the cosine decay."
        )
        self.patience = QSpinBox()
        self.patience.setRange(0, 1000)
        self.patience.setSpecialValueText("Off")
        self.patience.setToolTip(
            "Stop after this many epochs without a better validation score."
        )
        self.num_workers = QSpinBox()
        self.num_workers.setRange(0, 64)
        self.num_workers.setToolTip(
            "Data-loading processes (0 = load in the training process; slower)."
        )
        self.ema = QCheckBox("Average weights (EMA)")
        self.ema.setToolTip(
            "Keep an exponential moving average of the weights; it is what gets "
            "validated and saved (usually more accurate)."
        )
        self.amp = QCheckBox("Mixed precision (bf16, CUDA)")
        self.amp.setToolTip("Faster training with bfloat16 on NVIDIA GPUs.")
        form.addRow("Learning rate", self.lr)
        form.addRow("Weight decay", self.weight_decay)
        form.addRow("Layer decay", self.layer_decay)
        form.addRow("Drop path", self.drop_path)
        form.addRow("Warmup epochs", self.warmup_epochs)
        form.addRow("Early stopping patience", self.patience)
        form.addRow("Loader workers", self.num_workers)
        checks = QHBoxLayout()
        checks.addWidget(self.ema)
        checks.addWidget(self.amp)
        checks.addStretch(1)
        form.addRow("", checks)
        outer.addWidget(opt)

        # ── validation ──────────────────────────────────────────────────
        val = QGroupBox("ViTPose Validation")
        vform = QFormLayout(val)
        self.val_fraction = _dspin(0.05, 0.5, 0.05, 2, "Share of crops held out.")
        self.split_by = QComboBox()
        for key, label in SPLIT_LABELS.items():
            self.split_by.addItem(label, key)
        self.split_by.setToolTip(
            "Hold out whole source frames (or videos) so validation crops are not "
            "near-copies of training crops."
        )
        self.seed = QSpinBox()
        self.seed.setRange(0, 999999)
        self.ignore_occluded = QCheckBox("Ignore occluded keypoints in training")
        self.flip_test = QCheckBox("Flip test (average with mirrored image)")
        self.flip_test.setToolTip(
            "Validate on the image and its left-right mirror, averaged. Needs "
            "left-right flips in training."
        )
        vform.addRow("Validation fraction", self.val_fraction)
        vform.addRow("Split by", self.split_by)
        vform.addRow("Split seed", self.seed)
        vform.addRow("", self.ignore_occluded)
        vform.addRow("", self.flip_test)
        self.btn_recommended = QPushButton("Recommended settings")
        self.btn_recommended.clicked.connect(self.reset_recommended)
        vform.addRow("", self.btn_recommended)
        outer.addWidget(val)

        self.reset_recommended()

    # ── values ─────────────────────────────────────────────────────────

    _FLOATS = (
        "rotation",
        "rotation_prob",
        "brightness",
        "contrast",
        "saturation",
        "hue",
        "blur_prob",
        "noise_prob",
        "occlusion_prob",
        "lr",
        "weight_decay",
        "warmup_epochs",
        "val_fraction",
    )
    _INTS = ("patience", "num_workers", "seed")
    _BOOLS = ("ema", "amp", "ignore_occluded", "flip_test")
    _AUTO = ("layer_decay", "drop_path")  # 0 shows "Auto" -> None

    def reset_recommended(self) -> None:
        self.apply_settings(RECOMMENDED)

    def params(self) -> Dict[str, Any]:
        """Run parameters (RunConfig keys)."""
        out: Dict[str, Any] = {}
        for k in self._FLOATS:
            out[k] = float(getattr(self, k).value())
        for k in self._INTS:
            out[k] = int(getattr(self, k).value())
        for k in self._BOOLS:
            out[k] = bool(getattr(self, k).isChecked())
        for k in self._AUTO:
            v = float(getattr(self, k).value())
            out[k] = None if v <= 0 else v
        out["split_by"] = self.split_by.currentData()
        return out

    def settings(self) -> Dict[str, Any]:
        return self.params()

    def apply_settings(self, data: Dict[str, Any]) -> None:
        """Restore saved values; anything missing or malformed keeps its value."""
        if not isinstance(data, dict):
            return
        for k in self._FLOATS + self._INTS + self._AUTO:
            if k not in data:
                continue
            value = data[k]
            widget = getattr(self, k)
            try:
                number = 0.0 if value is None else float(value)
            except (TypeError, ValueError):
                continue
            widget.setValue(int(number) if k in self._INTS else number)
        for k in self._BOOLS:
            if k in data and isinstance(data[k], (bool, int)):
                getattr(self, k).setChecked(bool(data[k]))
        idx = self.split_by.findData(data.get("split_by"))
        if idx >= 0:
            self.split_by.setCurrentIndex(idx)
