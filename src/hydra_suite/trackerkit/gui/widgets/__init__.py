"""trackerkit-local widget utilities."""

try:
    from hydra_suite.trackerkit.gui.widgets.collapsible import (
        AccordionContainer,
        CollapsibleGroupBox,
    )
    from hydra_suite.trackerkit.gui.widgets.help_label import CompactHelpLabel
    from hydra_suite.trackerkit.gui.widgets.stacked_page import CurrentPageStackedWidget
    from hydra_suite.trackerkit.gui.widgets.tooltip_button import ImmediateTooltipButton
except (
    ImportError
):  # pragma: no cover - allows lightweight metadata imports without GUI deps
    AccordionContainer = None
    CollapsibleGroupBox = None
    CompactHelpLabel = None
    CurrentPageStackedWidget = None
    ImmediateTooltipButton = None

__all__ = [
    "AccordionContainer",
    "CollapsibleGroupBox",
    "CompactHelpLabel",
    "CurrentPageStackedWidget",
    "ImmediateTooltipButton",
]
