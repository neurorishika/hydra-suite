"""DetectKit-specific dialog behavior."""

from __future__ import annotations

from PySide6.QtCore import QEvent, QObject, QSize
from PySide6.QtWidgets import QAbstractSpinBox, QApplication

from hydra_suite.widgets.dialogs import BaseDialog


class DetectKitDialog(BaseDialog):
    """Base dialog that protects numeric values from accidental scrolling.

    The application-level filter also covers spin boxes that are created after
    a dialog is initially shown.  It is restricted to this dialog's widget
    tree, so numeric controls in other kits retain their existing behavior.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        app = QApplication.instance()
        if app is not None:
            app.installEventFilter(self)

    def eventFilter(self, watched: QObject, event: QEvent) -> bool:  # noqa: N802
        if event.type() is QEvent.Type.Wheel and self._is_own_spin_box(watched):
            return True
        return super().eventFilter(watched, event)

    def _is_own_spin_box(self, widget: QObject) -> bool:
        """Return whether *widget* belongs to one of this dialog's spin boxes."""
        current: QObject | None = widget
        while current is not None and current is not self:
            if isinstance(current, QAbstractSpinBox):
                return self.isAncestorOf(current)
            current = current.parent()
        return False

    def fit_to_content(self, base_minimum: QSize) -> None:
        """Keep an explicit minimum size from clipping the layout's content.

        An explicit ``setMinimumSize`` disables the layout's own minimum, so
        a dialog whose content grows (e.g. an expanded Advanced section)
        would otherwise squeeze rows on top of one another.
        """
        layout = self.layout()
        if layout is not None:
            layout.invalidate()
            layout.activate()
        minimum = base_minimum.expandedTo(self.minimumSizeHint())
        self.setMinimumSize(minimum)
        self.resize(self.size().expandedTo(minimum))
