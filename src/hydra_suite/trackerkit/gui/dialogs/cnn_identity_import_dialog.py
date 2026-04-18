"""Dialog for importing a ClassKit-trained model for CNN identity classification.

Supports both flat and multi-head models. Multi-head imports prompt for a
``scoring_mode`` (atomic tuple compare vs. per-head averaging).
"""

from __future__ import annotations

from typing import Any

from PySide6.QtWidgets import (
    QButtonGroup,
    QDialogButtonBox,
    QFormLayout,
    QLabel,
    QLineEdit,
    QRadioButton,
    QVBoxLayout,
    QWidget,
)

from hydra_suite.widgets import BaseDialog


def describe_cnn_identity_candidate(model_path: str) -> dict[str, Any]:
    """Return a metadata summary suitable for the import dialog preview."""
    from hydra_suite.core.identity.classification.backend import ClassifierBackend

    backend = ClassifierBackend(model_path, compute_runtime="cpu")
    try:
        meta = backend.metadata
        return {
            "arch": meta.arch,
            "input_size": meta.input_size,
            "is_multihead": meta.is_multihead,
            "factor_names": list(meta.factor_names),
            "class_names_per_factor": [list(c) for c in meta.class_names_per_factor],
            "monochrome": meta.monochrome,
            "source_path": meta.source_path,
        }
    finally:
        backend.close()


class CNNIdentityImportDialog(BaseDialog):
    """Import dialog: preview factor structure + gather species / label /
    scoring_mode from the user.
    """

    def __init__(self, summary: dict[str, Any], parent=None) -> None:
        super().__init__(
            "Import CNN Identity Model",
            parent=parent,
            buttons=QDialogButtonBox.StandardButton.Ok
            | QDialogButtonBox.StandardButton.Cancel,
        )

        form_widget = QWidget()
        layout = QFormLayout(form_widget)

        layout.addRow("Architecture:", QLabel(str(summary.get("arch", "—"))))
        isz = summary.get("input_size")
        layout.addRow(
            "Input size:",
            QLabel(f"{isz[0]} × {isz[1]}" if isz else "—"),
        )

        factor_names = list(summary.get("factor_names") or ["flat"])
        cnpf = list(summary.get("class_names_per_factor") or [[]])
        factor_lines = []
        for name, classes in zip(factor_names, cnpf):
            sample = ", ".join(classes[:6])
            if len(classes) > 6:
                sample += f", … ({len(classes)} total)"
            factor_lines.append(f"• {name} ({len(classes)} classes: {sample})")
        layout.addRow("Factors:", QLabel("\n".join(factor_lines)))

        self._species_edit = QLineEdit()
        self._species_edit.setPlaceholderText("e.g. ant")
        layout.addRow("Species:", self._species_edit)

        self._label_edit = QLineEdit()
        self._label_edit.setPlaceholderText("e.g. apriltag, colortag (optional)")
        layout.addRow("Classification label:", self._label_edit)

        self._is_multihead = bool(summary.get("is_multihead", False))
        self._scoring_mode_buttons: QButtonGroup | None = None
        self._atomic_btn: QRadioButton | None = None
        self._per_head_btn: QRadioButton | None = None

        if self._is_multihead:
            mode_box = QWidget()
            mode_layout = QVBoxLayout(mode_box)
            mode_layout.setContentsMargins(0, 0, 0, 0)
            self._atomic_btn = QRadioButton("Atomic tuple (strict match)")
            self._atomic_btn.setToolTip(
                "(B,G) vs (B,B) → mismatch_penalty. "
                "Any 'unknown'/None yields no signal."
            )
            self._atomic_btn.setChecked(True)
            self._per_head_btn = QRadioButton("Per-head average (partial credit)")
            self._per_head_btn.setToolTip(
                "(B,G) vs (B,B) → (−bonus + penalty)/2. "
                "Unknowns skip that head only."
            )
            self._scoring_mode_buttons = QButtonGroup(self)
            self._scoring_mode_buttons.addButton(self._atomic_btn, 0)
            self._scoring_mode_buttons.addButton(self._per_head_btn, 1)
            mode_layout.addWidget(self._atomic_btn)
            mode_layout.addWidget(self._per_head_btn)
            layout.addRow("Scoring mode:", mode_box)

        self.add_content(form_widget)

    def species(self) -> str:
        """Return the entered species name, defaulting to 'unknown' if blank."""
        return self._species_edit.text().strip() or "unknown"

    def classification_label(self) -> str:
        """Return the optional classification tag entered for the imported model."""
        return self._label_edit.text().strip()

    def scoring_mode(self) -> str:
        """Return selected scoring mode; ``"atomic"`` for flat models."""
        if not self._is_multihead or self._scoring_mode_buttons is None:
            return "atomic"
        return (
            "per_head_average"
            if self._scoring_mode_buttons.checkedId() == 1
            else "atomic"
        )
