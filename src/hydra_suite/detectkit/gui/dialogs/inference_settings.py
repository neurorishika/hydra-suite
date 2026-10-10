"""Runtime inference settings dialog for DetectKit."""

from __future__ import annotations

from dataclasses import replace
from statistics import median

from PySide6.QtCore import QSize, Qt
from PySide6.QtWidgets import (
    QComboBox,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from hydra_suite.detectkit.gui.dialogs._base import DetectKitDialog
from hydra_suite.utils.tiling_spec import (
    BACKEND_DEFAULTS,
    FRACTION_MAX,
    FRACTION_MIN,
    GEOMETRY_MODES,
    OVERLAP_MAX,
    TilingSpec,
)
from hydra_suite.widgets.slice_settings import (
    SliceSettingsWidget,
    SliceWidgetCapabilities,
)
from hydra_suite.widgets.slice_settings_parts import SLICE_SIZE_MAX

from ..models import (
    INFERENCE_CONFIDENCE_FLOOR,
    InferenceRunSettings,
    SliceTrainingSettings,
)
from ..panels.slice_settings_adapter import clamp_saved

# The runtime preview's single scale when nothing is configured: TrackerKit's
# inference default from the one defaults table.
_DEFAULT_INFERENCE_FRACTION = BACKEND_DEFAULTS["yolo_infer"].object_tile_fractions[0]


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


class InferenceSettingsDialog(DetectKitDialog):
    """Edit settings applied only to subsequent dataset inference runs."""

    def __init__(
        self,
        settings: InferenceRunSettings,
        defaults: InferenceRunSettings,
        parent=None,
        *,
        model_input_size: int = 640,
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
        # Only used to SHOW the fraction in pixels; nothing is stored in px.
        self._model_input_size = max(1, int(model_input_size))
        self.resize(920, 600)
        self._build_content()
        self.load_from(settings)
        # SAHI off collapses the slice widget to its checkbox: the dialog
        # shrinks with it, and grows back when SAHI (or Advanced) opens.
        self.fit_to_content(QSize(640, 300), follow_content_height=True)
        self.chk_sliced.toggled.connect(lambda _on: self.schedule_fit())
        self.slice_widget.btn_slice_advanced.toggled.connect(
            lambda _on: self.schedule_fit()
        )
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

        hint = QLabel(
            "Sliced inference applies to direct detect, OBB, and segment models. "
            "Sequential models retain their trained two-stage inference workflow."
        )
        hint.setWordWrap(True)
        layout.addWidget(hint)
        # The shared SAHI widget (S4): same labels, ranges and resolution as
        # TrackerKit. Old attribute names stay as aliases of its controls.
        # No merge policy/metric rows: this dialog does not know the model's
        # SAHI profile (preview_tiling reads them from it), so it shows only
        # the merge threshold it stores.
        self.slice_widget = SliceSettingsWidget(
            role="infer_yolo",
            title="Sliced inference (SAHI)",
            capabilities=SliceWidgetCapabilities(advanced_merge=False),
        )
        self.slice_widget.set_model_input_size(self._model_input_size)
        w = self.slice_widget
        self.chk_sliced = w.chk_slice_enabled
        self.combo_geometry = w.combo_slice_geometry
        self.spin_object_fraction = w.spin_slice_object_fraction
        self.lbl_scale_px = w.lbl_slice_scale_px
        self.spin_reference_body = w.spin_slice_body
        self.spin_width = w.spin_slice_tile_w
        self.spin_height = w.spin_slice_tile_h
        self.spin_overlap = w.spin_slice_overlap
        self.spin_merge = w.spin_slice_merge
        layout.addWidget(self.slice_widget)

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
        # Fractions only (F1). A legacy pixel project resolves through
        # target_fractions() (pixels / 640 -- how it was always interpreted).
        fractions = [float(value) for value in sliced.target_fractions() if value > 0]
        fraction = (
            float(median(fractions)) if fractions else _DEFAULT_INFERENCE_FRACTION
        )
        spec = replace(
            TilingSpec.defaults("yolo_infer"),
            enabled=bool(sliced.enabled),
            geometry_mode=(
                sliced.geometry_mode
                if sliced.geometry_mode in GEOMETRY_MODES
                else "auto_object"
            ),
            object_tile_fractions=(
                clamp_saved(
                    "object_tile_fraction", fraction, FRACTION_MIN, FRACTION_MAX
                ),
            ),
            reference_body_px=max(0.0, float(sliced.reference_body_px)),
            slice_width=int(
                clamp_saved("slice_width", sliced.slice_width, 0, SLICE_SIZE_MAX)
            ),
            slice_height=int(
                clamp_saved("slice_height", sliced.slice_height, 0, SLICE_SIZE_MAX)
            ),
            overlap=clamp_saved("overlap", sliced.overlap, 0.0, OVERLAP_MAX),
            merge_threshold=clamp_saved(
                "merge_threshold", sliced.merge_threshold, 0.0, 1.0
            ),
        )
        self.slice_widget.set_spec(
            spec, extras={"merge_threshold": spec.merge_threshold}
        )

    def settings(self) -> InferenceRunSettings:
        """Return a fresh runtime configuration from the current dialog state."""
        fraction = float(self.spin_object_fraction.value())
        return InferenceRunSettings(
            device=self.combo_device.currentText().strip() or "auto",
            confidence_threshold=float(self.spin_confidence.value()),
            slice_settings=SliceTrainingSettings(
                enabled=self.chk_sliced.isChecked(),
                geometry_mode=str(self.combo_geometry.currentData()),
                # One deliberate runtime scale rather than the training mix,
                # stored as a fraction only; ``target_sizes`` keeps its default
                # and is ignored because fractions are present (F1).
                object_tile_fraction=fraction,
                reference_body_px=float(self.spin_reference_body.value()),
                slice_width=int(self.spin_width.value()),
                slice_height=int(self.spin_height.value()),
                overlap=float(self.spin_overlap.value()),
                target_size_fractions=[fraction],
                merge_threshold=float(self.spin_merge.value()),
            ),
        )
