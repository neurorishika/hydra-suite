"""Delete the project-owned copy of a source when the source is removed.

Portable imports materialise a copy under ``artifacts/imported_sources/``;
unregistering the source without deleting that copy leaves it orphaned on
disk forever. Linked sources (anything outside that folder) are the user's
own data and are never touched.
"""

from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import Iterable

logger = logging.getLogger(__name__)

IMPORTED_SOURCES_RELDIR = "artifacts/imported_sources"


def _resolve(path: str | Path) -> Path | None:
    text = str(path or "").strip()
    if not text:
        return None
    try:
        return Path(text).expanduser().resolve()
    except OSError:
        return None


def owned_imported_copy(
    source_path: str | Path,
    project_dir: str | Path | None,
    remaining_source_paths: Iterable[str | Path] = (),
) -> Path | None:
    """Return the imported copy that removing this source orphans, else None.

    The path must be a direct child of the project's imported-sources folder
    (``source.path`` round-trips through the saved project file, so it is
    untrusted) and no remaining source may live in or above it.
    """
    if project_dir is None:
        return None
    imported_root = _resolve(Path(project_dir) / IMPORTED_SOURCES_RELDIR)
    target = _resolve(source_path)
    if imported_root is None or target is None or target.parent != imported_root:
        return None
    if not target.is_dir():
        return None
    for other in remaining_source_paths:
        other_path = _resolve(other)
        if other_path is None:
            continue
        if other_path == target or target in other_path.parents:
            return None
        if other_path in target.parents:
            return None
    return target


def delete_imported_copy(
    source_path: str | Path,
    project_dir: str | Path | None,
    remaining_source_paths: Iterable[str | Path] = (),
) -> bool:
    """Recursively delete the source's imported copy if it is safe to do so."""
    target = owned_imported_copy(source_path, project_dir, remaining_source_paths)
    if target is None:
        return False
    shutil.rmtree(target)
    logger.info("Deleted imported source copy %s", target)
    return True
