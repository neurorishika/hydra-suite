"""AprilTagAutoLabelDialog — configure and launch AprilTag auto-labeling."""

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSlider,
    QSpinBox,
    QVBoxLayout,
)

from hydra_suite.widgets.dialogs import HYDRA_DIALOG_STYLE

APRILTAG_FAMILIES = [
    "tag36h11",
    "tag25h9",
    "tag16h5",
    "tagCircle21h7",
    "tagCircle49h12",
    "tagCustom48h12",
    "tagStandard41h12",
    "tagStandard52h13",
]


class AprilTagAutoLabelDialog(QDialog):
    """Configure and launch AprilTag auto-labeling for a ClassKit project.

    On accept, call ``get_config()`` and ``get_threshold()`` to retrieve the
    user's settings. The dialog does NOT run the labeling itself -- that is the
    caller's responsibility.
    """

    def __init__(self, image_paths=None, parent=None) -> None:
        super().__init__(parent)
        self._image_paths = image_paths or []
        self.setWindowTitle("Auto-label AprilTags")
        self.setMinimumWidth(480)
        self.setStyleSheet(HYDRA_DIALOG_STYLE)
        self._build_ui()

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setSpacing(12)

        det_group = QGroupBox("AprilTag Detection Parameters")
        det_form = QFormLayout(det_group)
        det_form.setSpacing(8)

        self._family_combo = QComboBox()
        self._family_combo.addItems(APRILTAG_FAMILIES)
        self._family_combo.setCurrentText("tag36h11")
        self._family_combo.currentTextChanged.connect(self._update_scheme_preview)
        det_form.addRow("Tag family:", self._family_combo)

        self._max_tag_id_spin = QSpinBox()
        self._max_tag_id_spin.setRange(0, 999)
        self._max_tag_id_spin.setValue(9)
        self._max_tag_id_spin.valueChanged.connect(self._update_scheme_preview)
        det_form.addRow("Max tag ID:", self._max_tag_id_spin)

        self._max_hamming_spin = QSpinBox()
        self._max_hamming_spin.setRange(0, 3)
        self._max_hamming_spin.setValue(1)
        det_form.addRow("Max hamming:", self._max_hamming_spin)

        self._decimate_spin = QDoubleSpinBox()
        self._decimate_spin.setRange(1.0, 8.0)
        self._decimate_spin.setSingleStep(0.5)
        self._decimate_spin.setValue(2.0)
        det_form.addRow("Decimate:", self._decimate_spin)

        self._blur_spin = QDoubleSpinBox()
        self._blur_spin.setRange(0.0, 5.0)
        self._blur_spin.setSingleStep(0.1)
        self._blur_spin.setValue(0.8)
        det_form.addRow("Blur:", self._blur_spin)

        self._unsharp_amount_spin = QDoubleSpinBox()
        self._unsharp_amount_spin.setRange(0.0, 10.0)
        self._unsharp_amount_spin.setSingleStep(0.1)
        self._unsharp_amount_spin.setValue(1.0)
        det_form.addRow("Unsharp amount:", self._unsharp_amount_spin)

        self._unsharp_sigma_spin = QDoubleSpinBox()
        self._unsharp_sigma_spin.setRange(0.1, 10.0)
        self._unsharp_sigma_spin.setSingleStep(0.1)
        self._unsharp_sigma_spin.setValue(1.0)
        det_form.addRow("Unsharp sigma:", self._unsharp_sigma_spin)

        self._unsharp_kernel_spin = QSpinBox()
        self._unsharp_kernel_spin.setRange(1, 31)
        self._unsharp_kernel_spin.setSingleStep(2)
        self._unsharp_kernel_spin.setValue(5)
        det_form.addRow(
            "Unsharp kernel size (n \u2192 (n,n)):", self._unsharp_kernel_spin
        )

        root.addWidget(det_group)

        label_group = QGroupBox("Labeling Parameters")
        label_form = QFormLayout(label_group)
        label_form.setSpacing(8)

        thresh_row = QHBoxLayout()
        self._thresh_slider = QSlider(Qt.Orientation.Horizontal)
        self._thresh_slider.setRange(0, 100)
        self._thresh_slider.setValue(60)
        self._thresh_label = QLabel("0.60")
        self._thresh_slider.valueChanged.connect(
            lambda v: self._thresh_label.setText(f"{v / 100:.2f}")
        )
        thresh_row.addWidget(self._thresh_slider)
        thresh_row.addWidget(self._thresh_label)
        label_form.addRow("Confidence threshold:", thresh_row)

        self._preview_btn = QPushButton("Preview (10 images)")
        self._preview_btn.setEnabled(bool(self._image_paths))
        self._preview_btn.clicked.connect(self._run_preview)
        label_form.addRow(self._preview_btn)

        self._preview_result_label = QLabel("")
        self._preview_result_label.setWordWrap(True)
        label_form.addRow(self._preview_result_label)

        root.addWidget(label_group)

        scheme_group = QGroupBox("Labeling Scheme Preview")
        scheme_layout = QVBoxLayout(scheme_group)
        self._scheme_preview = QLabel()
        self._scheme_preview.setWordWrap(True)
        scheme_layout.addWidget(self._scheme_preview)
        root.addWidget(scheme_group)

        self._update_scheme_preview()

        btn_box = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        btn_box.accepted.connect(self.accept)
        btn_box.rejected.connect(self.reject)
        root.addWidget(btn_box)

    def _update_scheme_preview(self):
        family = self._family_combo.currentText()
        max_id = self._max_tag_id_spin.value()
        labels = [f"tag_{i}" for i in range(max_id + 1)] + ["no_tag"]
        self._scheme_preview.setText(
            f"<b>Scheme:</b> apriltag_{family}<br>"
            f"<b>Labels ({len(labels)}):</b> "
            + ", ".join(labels[:12])
            + ("\u2026" if len(labels) > 12 else "")
        )

    def _run_preview(self):
        import random

        sample = random.sample(self._image_paths, min(10, len(self._image_paths)))
        from hydra_suite.classkit.core.autolabel.apriltag import autolabel_images

        results = autolabel_images(sample, self.get_config(), self.get_threshold())
        n_tagged = sum(1 for r in results if r.label and r.label != "no_tag")
        n_no_tag = sum(1 for r in results if r.label == "no_tag")
        n_skip = sum(1 for r in results if r.label is None)
        self._preview_result_label.setText(
            f"Preview ({len(results)} images): "
            f"{n_tagged} tagged, {n_no_tag} no_tag, {n_skip} uncertain/skipped"
        )

    def get_config(self):
        """Return an AprilTagConfig built from the dialog's current settings."""
        from hydra_suite.core.individual.classification.apriltag import AprilTagConfig

        n = self._unsharp_kernel_spin.value()
        return AprilTagConfig(
            family=self._family_combo.currentText(),
            max_hamming=self._max_hamming_spin.value(),
            decimate=self._decimate_spin.value(),
            blur=self._blur_spin.value(),
            unsharp_amount=self._unsharp_amount_spin.value(),
            unsharp_sigma=self._unsharp_sigma_spin.value(),
            unsharp_kernel_size=(n, n),
            max_tag_id=self._max_tag_id_spin.value(),
        )

    def get_threshold(self) -> float:
        """Return the confidence threshold (0.0-1.0)."""
        return self._thresh_slider.value() / 100.0
