"""Stage a Hugging Face cache file into the HYDRA models directory.

Shared by the SAM2 and SAM3 checkpoint helpers. Never reads the checkpoint
into memory (a 3.45 GB SAM3 file read whole can OOM a 16 GB laptop), and never
leaves a partially copied file at ``dest`` that a later ``exists()`` would
mistake for a finished download.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path


def stage_hf_file(src: Path, dest: Path) -> Path:
    """Hardlink (same volume) or stream-copy ``src`` to ``dest``; return ``dest``.

    ``hf_hub_download`` returns a path inside the HF cache SNAPSHOT dir, which on
    Linux is a symlink into ``../../blobs/<sha>``. Hardlinking that entry would
    copy the symlink itself -- a relative target that dangles at ``dest`` -- so
    the real blob is resolved first.
    """
    src = Path(src).resolve()
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(src, dest)
    except OSError:
        partial = dest.with_name(dest.name + ".part")
        shutil.copyfile(src, partial)
        os.replace(partial, dest)
    if not dest.exists():  # never hand back a path we cannot actually open
        raise RuntimeError(f"checkpoint staging produced an unusable path at {dest}")
    return dest
