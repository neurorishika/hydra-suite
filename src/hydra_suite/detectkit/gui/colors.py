"""Canvas colours that are not per-class palette entries."""

from __future__ import annotations

from PySide6.QtGui import QColor

# The staged-escalation layer's single hue, deliberately OUTSIDE the class
# palette: a staged SAM3/SAM2 mask is a proposal, not a labelled class, so
# the distinction it must carry is "not ground truth" -- never a class
# identity.
ESCALATION_COLOUR = QColor(255, 60, 199)  # magenta
