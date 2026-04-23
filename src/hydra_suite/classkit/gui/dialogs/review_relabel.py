"""Dialog for choosing a replacement label when rejecting a machine review."""

from __future__ import annotations

from typing import Iterable

from PySide6.QtWidgets import QComboBox, QFormLayout, QLabel, QVBoxLayout, QWidget

from hydra_suite.classkit.config.schemas import LabelingScheme
from hydra_suite.widgets.dialogs import BaseDialog


class ReviewRelabelDialog(BaseDialog):
    """Prompt for the human label to apply after rejecting a machine prediction."""

    def __init__(
        self,
        *,
        classes: Iterable[str],
        scheme: LabelingScheme | None = None,
        initial_label: str | None = None,
        parent=None,
    ) -> None:
        super().__init__("Choose Review Label", parent=parent)
        self._scheme = scheme if getattr(scheme, "factors", None) else None
        self._factor_combos: list[QComboBox] = []
        self._flat_combo: QComboBox | None = None

        root = QWidget(self)
        layout = QVBoxLayout(root)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(12)

        intro = QLabel(
            "Rejecting a machine label applies a human-reviewed replacement label."
        )
        intro.setWordWrap(True)
        layout.addWidget(intro)

        form = QFormLayout()
        form.setContentsMargins(0, 0, 0, 0)
        form.setSpacing(10)
        layout.addLayout(form)

        normalized_initial = str(initial_label or "").strip()
        if self._scheme is not None and len(self._scheme.factors) > 1:
            initial_parts: list[str] = []
            if normalized_initial:
                try:
                    initial_parts = self._scheme.decode_label(normalized_initial)
                except Exception:
                    initial_parts = []

            for factor_index, factor in enumerate(self._scheme.factors):
                combo = QComboBox(self)
                combo.addItems([str(label) for label in factor.labels])
                if factor_index < len(initial_parts):
                    initial_part = initial_parts[factor_index]
                    initial_combo_index = combo.findText(initial_part)
                    if initial_combo_index >= 0:
                        combo.setCurrentIndex(initial_combo_index)
                self._factor_combos.append(combo)
                form.addRow(f"{factor.name}", combo)
        else:
            labels = [str(label).strip() for label in classes if str(label).strip()]
            if not labels and self._scheme is not None and self._scheme.factors:
                labels = [
                    str(label).strip()
                    for label in self._scheme.factors[0].labels
                    if str(label).strip()
                ]

            combo = QComboBox(self)
            combo.addItems(labels)
            if normalized_initial:
                initial_combo_index = combo.findText(normalized_initial)
                if initial_combo_index >= 0:
                    combo.setCurrentIndex(initial_combo_index)
            self._flat_combo = combo
            form.addRow("Label", combo)

        self.add_content(root)
        self.setMinimumWidth(360)

    def selected_label(self) -> str:
        """Return the label chosen in the dialog."""
        if self._factor_combos and self._scheme is not None:
            return self._scheme.encode_label(
                [combo.currentText().strip() for combo in self._factor_combos]
            )
        if self._flat_combo is None:
            return ""
        return self._flat_combo.currentText().strip()
