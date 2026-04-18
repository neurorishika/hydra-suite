"""Dialog for importing a ClassKit-trained classifier as a head-tail model.

Validates against the head-tail contract (flat + labels <= {up, down, left,
right, unknown}) before prompting for species/description.
"""

from __future__ import annotations

from typing import Any

from PySide6.QtWidgets import QDialogButtonBox, QFormLayout, QLabel, QLineEdit, QWidget

from hydra_suite.widgets import BaseDialog


def describe_headtail_candidate(model_path: str) -> dict[str, Any]:
    """Return a structured summary of a candidate head-tail checkpoint.

    Returns a dict with keys ``valid``, ``arch``, ``input_size``,
    ``normalized_labels``, ``raw_labels``, ``reason`` (populated only when
    ``valid`` is False).
    """
    from hydra_suite.core.identity.classification.backend import ClassifierBackend
    from hydra_suite.core.identity.classification.errors import (
        ClassifierError,
        HeadTailFormatError,
    )
    from hydra_suite.core.identity.classification.headtail import (
        validate_headtail_labels,
    )

    summary: dict[str, Any] = {
        "valid": False,
        "arch": "\u2014",
        "input_size": None,
        "raw_labels": [],
        "normalized_labels": [],
        "reason": "",
    }
    try:
        backend = ClassifierBackend(model_path, compute_runtime="cpu")
    except ClassifierError as exc:
        summary["reason"] = f"cannot parse metadata: {exc}"
        return summary
    try:
        meta = backend.metadata
        summary["arch"] = meta.arch
        summary["input_size"] = meta.input_size
        summary["raw_labels"] = list(meta.class_names_per_factor[0])
        if meta.is_multihead:
            summary["reason"] = (
                f"head-tail requires a flat classifier; model is multi-head "
                f"with factors {meta.factor_names}"
            )
            return summary
        try:
            summary["normalized_labels"] = validate_headtail_labels(
                summary["raw_labels"]
            )
        except HeadTailFormatError as exc:
            summary["reason"] = str(exc)
            return summary
        summary["valid"] = True
        return summary
    finally:
        backend.close()


class HeadTailImportDialog(BaseDialog):
    """Modal import dialog for head-tail classifier artifacts."""

    def __init__(self, summary: dict[str, Any], parent=None) -> None:
        super().__init__(
            "Import Head-Tail Model",
            parent=parent,
            buttons=QDialogButtonBox.StandardButton.Ok
            | QDialogButtonBox.StandardButton.Cancel,
        )
        form_widget = QWidget()
        layout = QFormLayout(form_widget)

        layout.addRow("Architecture:", QLabel(str(summary.get("arch", "\u2014"))))
        input_size = summary.get("input_size")
        layout.addRow(
            "Input size:",
            QLabel(f"{input_size[0]} x {input_size[1]}" if input_size else "\u2014"),
        )
        normalized = summary.get("normalized_labels", [])
        raw = summary.get("raw_labels", [])
        display_lbl = QLabel(f"{', '.join(normalized)}  (raw: {', '.join(raw)})")
        layout.addRow("Labels:", display_lbl)

        banner = QLabel()
        banner.setWordWrap(True)
        if summary.get("valid"):
            banner.setText("Valid head-tail model")
            banner.setStyleSheet("color: #4ec9b0;")
        else:
            banner.setText(f"Invalid: {summary.get('reason', 'invalid')}")
            banner.setStyleSheet("color: #f48771;")
        layout.addRow(banner)

        self._species_edit = QLineEdit()
        self._species_edit.setPlaceholderText("e.g. ant")
        layout.addRow("Species:", self._species_edit)

        self._desc_edit = QLineEdit()
        self._desc_edit.setPlaceholderText("optional description")
        layout.addRow("Description:", self._desc_edit)

        # Disable Ok button when the model is invalid.
        if not summary.get("valid"):
            ok_btn = self._buttons.button(QDialogButtonBox.StandardButton.Ok)
            if ok_btn is not None:
                ok_btn.setEnabled(False)

        self.add_content(form_widget)

    def species(self) -> str:
        return self._species_edit.text().strip() or "unknown"

    def description(self) -> str:
        return self._desc_edit.text().strip()
