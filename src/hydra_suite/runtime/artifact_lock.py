"""Blocking cross-process lock for first-run artifact builds.

``HeavyJobLease`` (resource_lease.py) is deliberately non-blocking: it exists
to REFUSE a second heavy job. Artifact builds need the opposite -- the second
process must WAIT for the first to finish exporting, then re-check whether the
artifact now exists. This module provides that blocking primitive and nothing
else. It imports no torch/onnx so it is safe on every host.
"""

from __future__ import annotations

import logging
import os
import time
from contextlib import contextmanager
from pathlib import Path
from typing import IO, Any, Iterator, Optional

logger = logging.getLogger(__name__)

fcntl: Any
try:
    import fcntl as _fcntl

    fcntl = _fcntl
except ImportError:  # pragma: no cover - Windows
    fcntl = None

msvcrt: Any
try:
    import msvcrt as _msvcrt

    msvcrt = _msvcrt
except ImportError:  # pragma: no cover - POSIX
    msvcrt = None

_POLL_SECONDS = 0.05


class ArtifactLockTimeout(RuntimeError):
    """Raised when ``timeout_s`` elapses before the lock is acquired."""


def lock_path_for(target: Path | str) -> Path:
    """Return ``<target>.lock`` beside the artifact (never inside it)."""
    target_path = Path(target)
    return target_path.parent / f"{target_path.name}.lock"


def _try_lock_nonblocking(handle: IO[str]) -> bool:
    if fcntl is not None:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            return True
        except BlockingIOError:
            return False
    if msvcrt is not None:  # pragma: no cover - Windows
        try:
            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            return True
        except OSError:
            return False
    raise RuntimeError("this platform has no supported inter-process file lock")


def _unlock(handle: IO[str]) -> None:
    if fcntl is not None:
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
    elif msvcrt is not None:  # pragma: no cover - Windows
        handle.seek(0)
        msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)


def _open_lock_handle(path: Path) -> Optional[IO[str]]:
    """Open ``path`` for locking, or return ``None`` when the FS refuses.

    A read-only model directory (a shared/mounted checkout, a container image
    layer) cannot host a ``.lock`` file at all. That must not be fatal: the
    lock is an optimisation for CONCURRENT builders, and a directory nobody
    can write to is a directory nobody can build into either. One warning,
    then unlocked operation -- which is exactly the single-process behaviour
    that shipped before this lock existed.
    """
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        logger.warning(
            "artifact_build_lock: cannot create lock directory %s (%s); "
            "proceeding WITHOUT a cross-process lock",
            path.parent,
            exc,
        )
        return None
    try:
        return path.open("a+", encoding="utf-8")
    except OSError as exc:
        logger.warning(
            "artifact_build_lock: cannot open lock file %s (%s); "
            "proceeding WITHOUT a cross-process lock",
            path,
            exc,
        )
        return None


@contextmanager
def artifact_build_lock(
    target: Path | str, *, timeout_s: float | None = None
) -> Iterator[None]:
    """Hold an exclusive lock on ``<target>.lock`` for the ``with`` body.

    Blocks (polling) until acquired. ``timeout_s=None`` waits forever; a
    positive value raises :class:`ArtifactLockTimeout` on expiry. The lock
    inode is never deleted, so PID reuse or a crashed holder cannot leave a
    stale lock: the OS releases ``flock`` when the holder dies.

    Callers must use double-checked locking: check the artifact, acquire,
    RE-CHECK, then build. The second process that blocked here will find the
    first process's finished artifact on its re-check and skip the build.

    When the environment cannot lock at all -- a read-only directory, or a
    filesystem whose ``flock`` returns ``ENOLCK``/``EOPNOTSUPP`` (NFS without
    a lock daemon) -- this DEGRADES to unlocked operation with a single
    warning rather than raising. Timeout semantics are unchanged otherwise.
    """
    path = lock_path_for(target)
    handle = _open_lock_handle(path)
    if handle is None:
        yield
        return
    if msvcrt is not None and fcntl is None:  # pragma: no cover - Windows
        handle.seek(0)
        if not handle.read(1):
            handle.write("\0")
            handle.flush()
    deadline = None if timeout_s is None else time.monotonic() + float(timeout_s)
    locked = False
    try:
        while True:
            try:
                acquired = _try_lock_nonblocking(handle)
            except OSError as exc:
                # ENOLCK / EOPNOTSUPP / EACCES: the filesystem has no working
                # advisory lock. Degrade rather than fail the whole run.
                logger.warning(
                    "artifact_build_lock: %s does not support advisory locking "
                    "(%s); proceeding WITHOUT a cross-process lock",
                    path,
                    exc,
                )
                break
            if acquired:
                locked = True
                break
            if deadline is not None and time.monotonic() >= deadline:
                raise ArtifactLockTimeout(
                    f"timed out after {timeout_s}s waiting for {path}"
                )
            time.sleep(_POLL_SECONDS)
        if locked:
            try:
                handle.seek(0)
                handle.truncate()
                handle.write(f"{os.getpid()}\n")
                handle.flush()
            except OSError:
                pass
        yield
    finally:
        try:
            if locked:
                _unlock(handle)
        finally:
            handle.close()
