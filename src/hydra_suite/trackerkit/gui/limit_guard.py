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
        report_limit_error(mw, context, exc, quiet=quiet)
        return None


def report_limit_error(mw, context: str, exc, *, quiet: bool = False) -> None:
    """Log ``exc`` and (unless ``quiet``) show the detection-limit message."""
    logger.error("%s blocked: %s", context, exc)
    if not quiet:
        QMessageBox.warning(mw, "Detection limit exceeded", str(exc))


def loaded_target_count_or_report(mw, value) -> int | None:
    """A config's ``max_targets`` if within the limit, else report and ``None``.

    ``QSpinBox.setValue`` silently clamps to the spinbox maximum (the limit),
    so a loaded N above the limit must be rejected before it reaches the
    widget -- otherwise the run would proceed at 1024 without a word.
    """
    from hydra_suite.core.inference.limits import require_target_count_within_limit

    try:
        return require_target_count_within_limit(int(value))
    except DetectionLimitError as exc:
        report_limit_error(
            mw,
            "Loading config",
            DetectionLimitError(
                f"{exc} The loaded configuration's number of animals "
                f"({int(value)}) was NOT applied; the current value is kept."
            ),
        )
        return None
