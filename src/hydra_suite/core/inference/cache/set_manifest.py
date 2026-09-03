"""One atomic indirection point for a complete multi-stage cache generation."""

from __future__ import annotations

import json
import os
import re
import tempfile
import uuid
from dataclasses import dataclass
from pathlib import Path

CACHE_SET_FILENAME = "cache_set.json"
CACHE_SET_VERSION = 2
MAX_CACHE_SET_BYTES = 64 * 1024
_GENERATION = re.compile(r"^[0-9a-f]{32}$")
_MEMBER = re.compile(r"^(detection|headtail|pose|apriltag|cnn_[A-Za-z0-9_-]+)\.npz$")


@dataclass(frozen=True)
class CacheSetManifest:
    generation_id: str
    revision_id: str
    members: dict[str, str]


def _validate(
    generation_id: object, revision_id: object, members: object
) -> CacheSetManifest | None:
    if any(
        not isinstance(value, str) or _GENERATION.fullmatch(value) is None
        for value in (generation_id, revision_id)
    ):
        return None
    if not isinstance(members, dict) or not members or len(members) > 64:
        return None
    clean: dict[str, str] = {}
    for name, relative in members.items():
        expected = f".cache-generations/{generation_id}/{revision_id}/{name}"
        if (
            not isinstance(name, str)
            or _MEMBER.fullmatch(name) is None
            or not isinstance(relative, str)
            or relative != expected
        ):
            return None
        clean[name] = relative
    return CacheSetManifest(
        generation_id=generation_id, revision_id=revision_id, members=clean
    )


def load_cache_set(cache_dir: Path) -> CacheSetManifest | None:
    path = Path(cache_dir) / CACHE_SET_FILENAME
    try:
        if not path.is_file() or path.stat().st_size > MAX_CACHE_SET_BYTES:
            return None

        with path.open("rb") as stream:
            payload = stream.read(MAX_CACHE_SET_BYTES + 1)
        if len(payload) > MAX_CACHE_SET_BYTES:
            return None

        def reject_duplicates(pairs):
            result = {}
            for key, value in pairs:
                if key in result:
                    raise ValueError("duplicate cache-set key")
                result[key] = value
            return result

        raw = json.loads(payload.decode("utf-8"), object_pairs_hook=reject_duplicates)
        if not isinstance(raw, dict) or set(raw) != {
            "version",
            "generation_id",
            "revision_id",
            "members",
        }:
            return None
        if raw["version"] != CACHE_SET_VERSION:
            return None
        return _validate(raw["generation_id"], raw["revision_id"], raw["members"])
    except (OSError, UnicodeError, ValueError, json.JSONDecodeError):
        return None


def publish_cache_set(
    cache_dir: Path,
    generation_id: str,
    revision_id: str,
    member_names: list[str],
) -> CacheSetManifest:
    members = {
        name: f".cache-generations/{generation_id}/{revision_id}/{name}"
        for name in member_names
    }
    manifest = _validate(generation_id, revision_id, members)
    if manifest is None:
        raise ValueError("invalid cache-set generation or member names")
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / CACHE_SET_FILENAME
    payload = json.dumps(
        {
            "version": CACHE_SET_VERSION,
            "generation_id": manifest.generation_id,
            "revision_id": manifest.revision_id,
            "members": manifest.members,
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=cache_dir)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(tmp, path)
        directory_fd = os.open(cache_dir, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except BaseException:
        try:
            os.unlink(tmp)
        except FileNotFoundError:
            pass
        raise
    return manifest


def publish_compatibility_links(cache_dir: Path, manifest: CacheSetManifest) -> None:
    """Best-effort canonical names for callers that gate on path existence.

    Cache-aware readers always follow ``cache_set.json``. These links preserve
    established ``detection.npz`` paths without making them the commit point.
    """
    cache_dir = Path(cache_dir)
    for name, relative in manifest.members.items():
        destination = cache_dir / name
        temporary = cache_dir / f".{name}.{uuid.uuid4().hex}.link"
        try:
            os.symlink(relative, temporary)
            os.replace(temporary, destination)
        finally:
            try:
                temporary.unlink()
            except FileNotFoundError:
                pass


def resolve_cache_member(path: Path) -> tuple[Path, str | None]:
    """Resolve a canonical member through the active set, or retain legacy path."""
    path = Path(path)
    manifest = load_cache_set(path.parent)
    if manifest is None:
        return path, None
    relative = manifest.members.get(path.name)
    if relative is None:
        return path, None
    return path.parent / relative, manifest.generation_id
