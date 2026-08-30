"""Runtime inference settings dialog for DetectKit."""

from __future__ import annotations

from statistics import median

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialogButtonBox,
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

from hydra_suite.widgets.dialogs import BaseDialog

from ..models import (
    INFERENCE_CONFIDENCE_FLOOR,
    InferenceRunSettings,
    SliceTrainingSettings,
)


def _device_options(current: str) -> list[str]:
    """Return the supported torch device choices, retaining a saved preference."""
    options = ["auto"]
    try:
        from hydra_suite.utils.gpu_utils import get_device_info

        info = get_device_info()
    except Exception:
        info = {}

    if info.get("torch_cuda_available"):
        count = int(info.get("torch_cuda_device_count", 0) or 0)
        options.append("cuda")
        options.extend(f"cuda:{index}" for index in range(count))
    if info.get("mps_available"):
        options.append("mps")
    options.append("cpu")

    current = str(current or "auto").strip().lower()
    if current and current not in options:
        options.insert(1, current)
    return options


class InferenceSettingsDialog(BaseDialog):
    """Edit settings applied only to subsequent dataset inference runs."""

    def __init__(
        self,
        settings: InferenceRunSettings,
        defaults: InferenceRunSettings,
        parent=None,
    ) -> None:
        super().__init__(
            "Inference Settings",
            parent=parent,
            buttons=(
                QDialogButtonBox.StandardButton.Apply
                | QDialogButtonBox.StandardButton.Cancel
            ),
        )
        self._defaults = defaults
        self.resize(620, 520)
        self._build_content()
        self.load_from(settings)
        self._buttons.button(QDialogButtonBox.StandardButton.Apply).clicked.connect(
            self.accept
        )

    def _build_content(self) -> None:
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(12)

        note = QLabel(
            "These controls apply only to inference runs in this DetectKit window. "
            "They do not change your project's training settings or metadata. "
            f"Predictions are retained at {INFERENCE_CONFIDENCE_FLOOR:.2f} and "
            "filtered live by the display threshold."
        )
        note.setWordWrap(True)
        layout.addWidget(note)

        compute = QGroupBox("Compute and detection")
        compute_form = QFormLayout(compute)
        self.combo_device = QComboBox()
        self.combo_device.addItems(_device_options(self._defaults.device))
        self.combo_device.setToolTip(
            "Auto chooses CUDA first, then MPS, then CPU on the current machine."
        )
        self.spin_confidence = QDoubleSpinBox()
        self.spin_confidence.setRange(INFERENCE_CONFIDENCE_FLOOR, 1.0)
        self.spin_confidence.setDecimals(2)
        self.spin_confidence.setSingleStep(0.01)
        compute_form.addRow("Compute device", self.combo_device)
        compute_form.addRow("Display confidence threshold", self.spin_confidence)
        layout.addWidget(compute)

        sahi = QGroupBox("Sliced inference (SAHI)")
        sahi_layout = QVBoxLayout(sahi)
        self.chk_sliced = QCheckBox("Enable sliced inference")
        self.chk_sliced.toggled.connect(self._refresh_enabled_state)
        sahi_layout.addWidget(self.chk_sliced)

        hint = QLabel(
            "For direct detect, OBB, and segment models. Sequential models retain "
            "their trained two-stage inference workflow."
        )
        hint.setWordWrap(True)
        sahi_layout.addWidget(hint)

        grid = QGridLayout()
        self.combo_geometry = QComboBox()
        self.combo_geometry.addItems(["auto_object", "auto_model", "custom"])
        self.combo_geometry.currentTextChanged.connect(self._refresh_enabled_state)
        self.spin_target_size = QSpinBox()
        self.spin_target_size.setRange(16, 4096)
        self.spin_target_size.setSingleStep(16)
        self.spin_target_size.setToolTip(
            "Apparent object size at the model input. Larger values use smaller tiles "
            "and can make high-resolution inference much slower."
        )
        self.spin_reference_body = QDoubleSpinBox()
        self.spin_reference_body.setRange(0.0, 16384.0)
        self.spin_reference_body.setDecimals(1)
        self.spin_reference_body.setSingleStep(1.0)
        self.spin_width = QSpinBox()
        self.spin_width.setRange(0, 16384)
        self.spin_height = QSpinBox()
        self.spin_height.setRange(0, 16384)
        self.spin_overlap = QDoubleSpinBox()
        self.spin_overlap.setRange(0.0, 0.9)
        self.spin_overlap.setDecimals(2)
        self.spin_overlap.setSingleStep(0.05)
        self.spin_merge = QDoubleSpinBox()
        self.spin_merge.setRange(0.0, 1.0)
        self.spin_merge.setDecimals(2)
        self.spin_merge.setSingleStep(0.05)

        grid.addWidget(QLabel("Geometry mode"), 0, 0)
        grid.addWidget(self.combo_geometry, 0, 1)
        grid.addWidget(QLabel("Target object size (px)"), 0, 2)
        grid.addWidget(self.spin_target_size, 0, 3)
        grid.addWidget(QLabel("Reference body (px)"), 1, 0)
        grid.addWidget(self.spin_reference_body, 1, 1)
        grid.addWidget(QLabel("Tile width / height"), 1, 2)
        custom_tile = QWidget()
        custom_tile_layout = QHBoxLayout(custom_tile)
        custom_tile_layout.setContentsMargins(0, 0, 0, 0)
        custom_tile_layout.addWidget(self.spin_width)
        custom_tile_layout.addWidget(QLabel("×"))
        custom_tile_layout.addWidget(self.spin_height)
        grid.addWidget(custom_tile, 1, 3)
        grid.addWidget(QLabel("Tile overlap"), 2, 0)
        grid.addWidget(self.spin_overlap, 2, 1)
        grid.addWidget(QLabel("Merge threshold"), 2, 2)
        grid.addWidget(self.spin_merge, 2, 3)
        sahi_layout.addLayout(grid)
        layout.addWidget(sahi)

        self.btn_restore_defaults = QPushButton("Use Project Defaults")
        self.btn_restore_defaults.clicked.connect(
            lambda: self.load_from(self._defaults)
        )
        layout.addWidget(
            self.btn_restore_defaults, alignment=Qt.AlignmentFlag.AlignLeft
        )
        self.add_content(container)

    def load_from(self, settings: InferenceRunSettings) -> None:
        device = str(settings.device or "auto").strip().lower()
        index = self.combo_device.findText(device, Qt.MatchFlag.MatchFixedString)
        if index < 0:
            self.combo_device.addItem(device)
            index = self.combo_device.count() - 1
        self.combo_device.setCurrentIndex(index)
        self.spin_confidence.setValue(float(settings.confidence_threshold))

        sliced = settings.slice_settings
        self.chk_sliced.setChecked(bool(sliced.enabled))
        index = self.combo_geometry.findText(
            sliced.geometry_mode, Qt.MatchFlag.MatchFixedString
        )
        self.combo_geometry.setCurrentIndex(index if index >= 0 else 0)
        target_sizes = [float(value) for value in sliced.target_sizes if value > 0]
        self.spin_target_size.setValue(
            max(16, round(median(target_sizes))) if target_sizes else 96
        )
        self.spin_reference_body.setValue(float(sliced.reference_body_px))
        self.spin_width.setValue(int(sliced.slice_width))
        self.spin_height.setValue(int(sliced.slice_height))
        self.spin_overlap.setValue(float(sliced.overlap))
        self.spin_merge.setValue(float(sliced.merge_threshold))
        self._refresh_enabled_state()

    def settings(self) -> InferenceRunSettings:
        """Return a fresh runtime configuration from the current dialog state."""
        target_size = float(self.spin_target_size.value())
        return InferenceRunSettings(
            device=self.combo_device.currentText().strip() or "auto",
            confidence_threshold=float(self.spin_confidence.value()),
            slice_settings=SliceTrainingSettings(
                enabled=self.chk_sliced.isChecked(),
                geometry_mode=self.combo_geometry.currentText(),
                # ``target_sizes`` controls auto_object preview geometry; keep a
                # single deliberate runtime scale rather than the training mix.
                object_tile_fraction=target_size / 640.0,
                reference_body_px=float(self.spin_reference_body.value()),
                slice_width=int(self.spin_width.value()),
                slice_height=int(self.spin_height.value()),
                overlap=float(self.spin_overlap.value()),
                target_sizes=[target_size],
                merge_threshold=float(self.spin_merge.value()),
            ),
        )

    def _refresh_enabled_state(self) -> None:
        sliced = self.chk_sliced.isChecked()
        geometry = self.combo_geometry.currentText()
        self.combo_geometry.setEnabled(sliced)
        self.spin_target_size.setEnabled(sliced and geometry == "auto_object")
        self.spin_reference_body.setEnabled(sliced and geometry == "auto_object")
        custom = sliced and geometry == "custom"
        self.spin_width.setEnabled(custom)
        self.spin_height.setEnabled(custom)
        self.spin_overlap.setEnabled(sliced)
        self.spin_merge.setEnabled(sliced)
