"""Project-wide warning filters for noisy third-party imports."""

from __future__ import annotations

import warnings


def install_warning_filters() -> None:
    """Suppress known non-actionable third-party warnings."""

    warnings.filterwarnings(
        "ignore",
        message=r"\s*\*\*\* Python is at version 3\.10 now\. _PepUnicode_AsString can now be replaced by PyUnicode_AsUTF8! \*\*\*",
        category=UserWarning,
        module=r"shibokensupport\.signature\.parser",
    )
    warnings.filterwarnings(
        "ignore",
        message=r"\s*\*{3,4} Python is at version 3\.10 now\. layout\.py and pyi_generator\.py can now remove old code! \*{3,4}",
        category=UserWarning,
        module=r"shibokensupport\.signature\.parser",
    )
