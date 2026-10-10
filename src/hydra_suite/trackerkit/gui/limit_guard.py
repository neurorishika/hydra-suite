"""Shared GUI guard for the per-frame detection limit (N > 1024)."""

from __future__ import annotations

import logging

from PySide6.QtWidgets import QMessageBox

from hydra_suite.core.inference.limits import DetectionLimitError

logger = logging.getLogger(__name__)


def params_or_report_limit(mw, context: str, *, quiet: bool = False, getter=None):
    """``mw.get_parameters_dict()``, or ``None`` when N exceeds the limit.

    ``quiet`` only logs (for cleanup paths); otherwise a message box tells the
    user why the action was aborted.
    """
    try:
        return (getter or mw.get_parameters_dict)()
    except DetectionLimitError as exc:
        logger.error("%s blocked: %s", context, exc)
        if not quiet:
            QMessageBox.warning(mw, "Detection limit exceeded", str(exc))
        return None
