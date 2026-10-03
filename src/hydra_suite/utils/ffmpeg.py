"""Locate an ``ffmpeg`` executable on every platform.

conda environments ship ``ffmpeg`` on PATH; pip/venv installs (and most
Windows machines) do not. ``imageio-ffmpeg`` -- a core dependency -- bundles a
static binary for Linux, macOS and Windows, so it is the fallback.
"""

from __future__ import annotations

import shutil
from typing import Optional


def ffmpeg_exe() -> Optional[str]:
    """Path to an ffmpeg binary, or None if none can be found."""
    found = shutil.which("ffmpeg")
    if found:
        return found
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:  # noqa: BLE001 - missing package or unsupported platform
        return None
