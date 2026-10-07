"""Filter OS metadata files out of dataset enumeration.

macOS writes an AppleDouble sidecar ``._<name>`` next to every file it copies
onto a volume without native xattr support (SMB, exFAT), so a copied
``images/frame.jpg`` arrives with ``images/._frame.jpg`` -- a 4 KiB binary blob
with an image suffix. Suffix-only discovery treats it as a frame (unreadable)
or as a YOLO label (binary garbage). Every dataset scan must skip hidden names.
"""

from __future__ import annotations

import os
from pathlib import Path


def is_hidden_file(path: str | os.PathLike[str]) -> bool:
    """Return True for dot-files (AppleDouble ``._*``, ``.DS_Store``, ...)."""
    return Path(path).name.startswith(".")
