"""Locate an ``ffmpeg`` executable on every platform.

A system ``ffmpeg`` on PATH wins. Otherwise ``imageio-ffmpeg`` -- a core
dependency -- supplies a static binary for Linux, macOS and Windows. install.py
deliberately does NOT install conda's ffmpeg (its harfbuzz/freetype libraries
break Qt on hosts with an older system freetype).
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
