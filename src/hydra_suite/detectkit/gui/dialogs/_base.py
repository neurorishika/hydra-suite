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

    def fit_to_content(self, base_minimum: QSize | None = None) -> None:
        """Keep an explicit minimum size from clipping the layout's content.

        An explicit ``setMinimumSize`` disables the layout's own minimum, so
        a dialog whose content grows (e.g. an expanded Advanced section)
        would otherwise squeeze rows on top of one another. The minimum is
        recomputed from the layout on every call, so it shrinks back too;
        the window returns to its preferred size unless the user resized it.
        Re-run on every show. The first call records ``base_minimum`` and the
        current size as the preferred size.
        """
        if base_minimum is not None and not hasattr(self, "_fit_base"):
            self._fit_base = QSize(base_minimum)
            self._fit_preferred = self.size()
            self._fit_last: QSize | None = None
        elif base_minimum is not None:
            self._fit_base = QSize(base_minimum)
        if not hasattr(self, "_fit_base"):
            return
        layout = self.layout()
        if layout is not None:
            layout.invalidate()
            layout.activate()
        minimum = self._fit_base.expandedTo(self.minimumSizeHint())
        user_resized = self._fit_last is not None and self.size() != self._fit_last
        self.setMinimumSize(minimum)
        target = (self.size() if user_resized else self._fit_preferred).expandedTo(
            minimum
        )
        self.resize(target)
        self._fit_last = QSize(target)

    def showEvent(self, event) -> None:  # noqa: N802 - Qt override
        super().showEvent(event)
        if hasattr(self, "_fit_base"):
            self.fit_to_content()
