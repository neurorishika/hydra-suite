"""Host mount table: a video on a lab share is referenced, never copied."""

from __future__ import annotations

import json
import os
from pathlib import Path

from ...paths import get_shared_roots_path
from .manifest import TrackingJobError, validate_job_relpath


def load_shared_roots(path: Path | None = None) -> dict[str, str]:
    target = Path(path) if path is not None else get_shared_roots_path()
    if not target.is_file():
        return {}
    try:
        with open(target, encoding="utf-8") as handle:
            data = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise TrackingJobError(
            f"cannot read shared-root table {target}: {exc}"
        ) from exc
    if not isinstance(data, dict):
        raise TrackingJobError(f"shared-root table must be a JSON object: {target}")
    return {str(k): str(v) for k, v in data.items()}


def save_shared_roots(table: dict[str, str], path: Path | None = None) -> None:
    from ..project_bundle import write_json_atomic

    target = Path(path) if path is not None else get_shared_roots_path()
    write_json_atomic(target, {str(k): str(v) for k, v in table.items()})


def save_alias(alias: str, root: str, path: Path | None = None) -> dict[str, str]:
    """Persist ONE alias, preserving every other entry (fix B12a).

    Tasks 10 and 11 promote a one-off ``--shared-root ALIAS=PATH`` into the
    host table. Calling ``save_shared_roots({alias: root})`` there would delete
    every other alias this machine already knows, so the read-modify-write
    lives here, once, instead of being re-derived at each call site.
    """
    target = Path(path) if path is not None else get_shared_roots_path()
    table = load_shared_roots(target) if target.exists() else {}
    table[str(alias)] = str(root)
    save_shared_roots(table, target)
    return table


def match_shared_root(abs_path: str, table: dict[str, str]) -> tuple[str, str] | None:
    """Longest matching alias root wins; ``None`` when the file is off-share."""
    real = os.path.realpath(abs_path)
    best: tuple[str, str] | None = None
    best_len = -1
    for alias, root in table.items():
        root_real = os.path.realpath(root)
        try:
            rel = Path(real).relative_to(root_real)
        except ValueError:
            continue  # component-safe: "/Volumes/labour" is not under "/Volumes/lab"
        if len(root_real) > best_len:
            best_len = len(root_real)
            best = (alias, rel.as_posix())
    return best


def resolve_shared(alias: str, relpath: str, table: dict[str, str]) -> Path:
    if alias not in table:
        known = ", ".join(sorted(table)) or "(none configured)"
        raise TrackingJobError(
            f"unknown shared-root alias {alias!r}; known aliases: {known}. "
            f"Add it with 'trackerkit job shared-root add {alias} <path>' or pass "
            f"--shared-root {alias}=<path>.",
            code=3,
        )
    return Path(table[alias]) / validate_job_relpath(relpath)
