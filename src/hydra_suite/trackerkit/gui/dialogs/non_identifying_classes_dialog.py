"""Mark a classifier's classes as non-identifying.

A non-identifying class (an untagged animal's ``notag``) names an animal
that carries no unique identity. Marked classes are excluded from the
identity catalog entirely, so any number of animals may carry them
simultaneously without competing for one slot.
"""

from __future__ import annotations

from PySide6.QtWidgets import (
    QCheckBox,
    QGroupBox,
    QLabel,
    QLineEdit,
    QVBoxLayout,
    QWidget,
)

from hydra_suite.widgets.dialogs import BaseDialog


class NonIdentifyingClassesDialog(BaseDialog):
    """Per-factor class checkboxes plus a free-text composite field.

    ``BaseDialog.__init__(title, parent=None, ...)`` builds the Ok/Cancel
    button box itself; subclasses insert their UI above it with
    ``add_content(widget)``.
    """

    def __init__(self, parent, factor_names, class_names_per_factor, selected):
        super().__init__("Non-identifying classes", parent)
        self._checks: list[tuple[str, str, QCheckBox]] = []
        selected = list(selected or [])

        container = QWidget()
        layout = QVBoxLayout(container)
        layout.addWidget(
            QLabel(
                "Classes that do not identify an individual (e.g. 'notag').\n"
                "Excluded from the identity catalog: any number of animals may\n"
                "carry them at once, and they are never merged or swapped."
            )
        )

        for idx, classes in enumerate(class_names_per_factor or []):
            if not classes:
                continue
            factor = (
                str(factor_names[idx])
                if idx < len(factor_names or [])
                else f"factor{idx}"
            )
            box = QGroupBox(factor)
            box_layout = QVBoxLayout()
            for cls in classes:
                cls = str(cls)
                chk = QCheckBox(cls)
                chk.setChecked(cls in selected or f"{factor}:{cls}" in selected)
                box_layout.addWidget(chk)
                self._checks.append((factor, cls, chk))
            box.setLayout(box_layout)
            layout.addWidget(box)

        layout.addWidget(
            QLabel("Whole composites (comma-separated, e.g. notag_notag):")
        )
        # Note: this filter can't distinguish a bare single-word composite
        # label (e.g. "notag" as a whole display label, single-factor case)
        # from a bare any-axis class mark (e.g. "notag" meaning "this class
        # in any factor") -- both are unprefixed and underscore-free. Either
        # reading is a valid, harmless pre-population; this is a display-only
        # ambiguity, not a data-loss risk, since `selected` itself is
        # unaffected and is what actually round-trips.
        self._composites = QLineEdit(
            ", ".join(s for s in selected if "_" in s and ":" not in s)
        )
        layout.addWidget(self._composites)

        self.add_content(container)

    def selected_marks(self) -> list[str]:
        """Checked classes as ``factor:class``, plus any composite entries."""
        marks = [f"{f}:{c}" for f, c, chk in self._checks if chk.isChecked()]
        marks.extend(
            part.strip() for part in self._composites.text().split(",") if part.strip()
        )
        return marks
