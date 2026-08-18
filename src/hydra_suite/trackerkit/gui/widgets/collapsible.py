"""CollapsibleGroupBox and AccordionContainer — expandable section widgets."""

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import QSizePolicy, QToolButton, QVBoxLayout, QWidget

from hydra_suite.trackerkit.gui.widgets.tooltip_button import ImmediateTooltipButton


class CollapsibleGroupBox(QWidget):
    """
    A collapsible group box widget that can expand/collapse its content.
    Used for advanced settings that don't need to be visible all the time.
    """

    toggled = Signal(bool)  # Emitted when expanded/collapsed

    def __init__(
        self, title: str, parent=None, initially_expanded: bool = False
    ) -> None:
        super().__init__(parent)
        self._is_expanded = initially_expanded
        self._title = title
        self._accordion_group = None  # Reference to accordion container
        self._help_tooltip = ""
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Maximum)

        # Main layout
        self._main_layout = QVBoxLayout(self)
        self._main_layout.setContentsMargins(0, 0, 0, 0)
        self._main_layout.setSpacing(0)

        # Header button (acts as toggle)
        self._header_button = QToolButton()
        self._header_button.setStyleSheet("""
            QToolButton {
                background-color: #2d2d30;
                border: 1px solid #3e3e42;
                border-radius: 4px;
                padding: 8px 12px;
                font-weight: 600;
                font-size: 12px;
                color: #9cdcfe;
                text-align: left;
            }
            QToolButton:hover {
                background-color: #37373d;
                border-color: #4a4a4a;
            }
            QToolButton:checked {
                background-color: #37373d;
                border-color: #007acc;
            }
        """)
        self._header_button.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        self._header_button.setArrowType(
            Qt.RightArrow if not initially_expanded else Qt.DownArrow
        )
        self._header_button.setText(title)
        self._header_button.setCheckable(True)
        self._header_button.setChecked(initially_expanded)
        self._header_button.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self._header_button.clicked.connect(self._on_header_clicked)

        self._main_layout.addWidget(self._header_button)

        self._help_button = ImmediateTooltipButton(self)
        self._help_button.setText("?")
        self._help_button.setAutoRaise(True)
        self._help_button.setCursor(Qt.PointingHandCursor)
        self._help_button.setFixedSize(16, 16)
        self._help_button.setStyleSheet(
            "QToolButton {"
            " background-color: #1f3b53; color: #8fd3ff;"
            " border: 1px solid #325978; border-radius: 8px;"
            " font-size: 10px; font-weight: 700; padding: 0px;"
            "}"
            "QToolButton:hover { background-color: #255174; border-color: #4fc1ff; }"
        )
        self._help_button.hide()

        # Content container
        self._content_container = QWidget()
        self._content_layout = QVBoxLayout(self._content_container)
        self._content_layout.setContentsMargins(0, 5, 0, 5)
        self._content_container.setVisible(initially_expanded)

        self._main_layout.addWidget(self._content_container)

    def setContentLayout(self: object, layout: object) -> object:
        """Set the content layout for the collapsible section."""
        # Clear existing layout
        while self._content_layout.count():
            item = self._content_layout.takeAt(0)
            if item.widget():
                item.widget().setParent(None)

        # Add new content as a widget
        content_widget = QWidget()
        content_widget.setLayout(layout)
        self._content_layout.addWidget(content_widget)

    def addWidget(self: object, widget: object) -> object:
        """Add a widget to the content area."""
        self._content_layout.addWidget(widget)

    def addLayout(self: object, layout: object) -> object:
        """Add a layout to the content area."""
        self._content_layout.addLayout(layout)

    def setAccordionGroup(self: object, accordion: object) -> object:
        """Set the accordion group this collapsible belongs to."""
        self._accordion_group = accordion

    def _on_header_clicked(self, checked):
        """Handle header button click."""
        if checked:
            # Notify accordion to collapse others
            if self._accordion_group:
                self._accordion_group.collapseAllExcept(self)
        self.setExpanded(checked)

    def setExpanded(self: object, expanded: bool) -> object:
        """Set the expanded state of the collapsible."""
        self._is_expanded = expanded
        self._header_button.setChecked(expanded)
        self._header_button.setArrowType(Qt.DownArrow if expanded else Qt.RightArrow)
        self._content_container.setVisible(expanded)
        self.toggled.emit(expanded)

    def isExpanded(self) -> bool:
        """Check if the collapsible is expanded."""
        return self._is_expanded

    def title(self) -> str:
        """Get the title of the collapsible."""
        return self._title

    def setHelpToolTip(self, text: str) -> None:
        """Attach a compact help button to the collapsible header."""
        self._help_tooltip = text or ""
        self._help_button.setToolTip(self._help_tooltip)
        self._help_button.setVisible(bool(self._help_tooltip))
        self._position_help_button()

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._position_help_button()

    def showEvent(self, event) -> None:
        super().showEvent(event)
        # The header's real width is only known once the widget is laid out.
        # Without this, a collapsible whose help tooltip was set at
        # construction time (while it was still hidden) keeps the button at
        # its default top-left position, floating outside the header row.
        self._position_help_button()

    def _position_help_button(self) -> None:
        # Guard on the tooltip, NOT on ``isVisible()``: while the panel or the
        # enclosing tab is still hidden the button reports invisible, so an
        # isVisible() guard skipped every positioning pass that ran before the
        # first user-visible paint -- which is all of them.
        if not self._help_tooltip:
            return
        metrics = self._header_button.fontMetrics()
        title_width = metrics.horizontalAdvance(self._title)
        x = min(
            self._header_button.width() - self._help_button.width() - 10,
            34 + title_width,
        )
        x = max(30, x)
        y = (
            self._header_button.y()
            + (self._header_button.height() - self._help_button.height()) // 2
        )
        self._help_button.move(x, y)


class AccordionContainer:
    """
    Manages a group of CollapsibleGroupBox widgets to ensure only one is expanded at a time.
    """

    def __init__(self) -> None:
        self._collapsibles = []

    def addCollapsible(self: object, collapsible: CollapsibleGroupBox) -> object:
        """Add a collapsible to this accordion group."""
        collapsible.setAccordionGroup(self)
        self._collapsibles.append(collapsible)

    def collapseAllExcept(self: object, keep_expanded: CollapsibleGroupBox) -> object:
        """Collapse all collapsibles except the specified one."""
        for collapsible in self._collapsibles:
            if collapsible is not keep_expanded and collapsible.isExpanded():
                collapsible.setExpanded(False)
