"""Per-GPU phase semaphore shared by the children of one TrackerKit fan-out.

A fan-out child spends hours on its GPU (the inference batch pass) and then
hours on the CPU alone (tracking, post-processing). With ``--max-jobs`` the
parent runs more children than there are GPU slots, and each child takes a
slot here only for the stretches it actually computes on the GPU, so the CPU
phases of some children overlap the GPU phases of others.

The semaphore is ``K`` lock files per GPU, ``<dir>/<gpu>.slot<i>.lock``. A
phase holds an OS file lock on one of them (``fcntl.flock`` on POSIX,
``msvcrt.locking`` on Windows); the kernel drops the lock when the holder
dies, so a crashed or killed child can never wedge its GPU.

Configured purely by environment, set by the fan-out parent for each child:

- ``HYDRA_GPU_PHASE_LOCK_DIR``: directory holding the lock files;
- ``HYDRA_GPU_PHASE_GPU``: the child's GPU (its UUID);
- ``HYDRA_GPU_PHASE_SLOTS``: ``K``, concurrent GPU phases allowed per GPU.

With any of them absent (a single run, the GUI, a plain fan-out) every entry
point is a no-op: nothing is locked, nothing is freed, torch is never imported.
"""

from __future__ import annotations

import gc
import logging
import os
import re
import sys
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterator, Mapping, Optional

logger = logging.getLogger(__name__)

ENV_LOCK_DIR = "HYDRA_GPU_PHASE_LOCK_DIR"
ENV_GPU = "HYDRA_GPU_PHASE_GPU"
ENV_SLOTS = "HYDRA_GPU_PHASE_SLOTS"
ENV_VARS = (ENV_LOCK_DIR, ENV_GPU, ENV_SLOTS)

#: Seconds between attempts while every slot is taken.
DEFAULT_POLL_S = 1.0
#: Seconds between "still waiting" log lines.
DEFAULT_WAIT_LOG_S = 600.0


@dataclass(frozen=True)
class GpuPhaseConfig:
    lock_dir: Path
    gpu: str
    slots: int

    @property
    def _safe_gpu(self) -> str:
        return re.sub(r"[^A-Za-z0-9_.-]", "_", self.gpu) or "gpu"

    def slot_path(self, slot: int) -> Path:
        return self.lock_dir / f"{self._safe_gpu}.slot{int(slot)}.lock"

    def waiter_path(self, token: str) -> Path:
        return self.lock_dir / f"{self._safe_gpu}.waiter.{token}"

    def waiter_glob(self) -> str:
        return f"{self._safe_gpu}.waiter.*"


def phase_lock_env(lock_dir: Path | str, gpu: str, slots: int) -> dict[str, str]:
    """The child env entries that turn the semaphore on (parent side)."""
    return {
        ENV_LOCK_DIR: str(lock_dir),
        ENV_GPU: str(gpu),
        ENV_SLOTS: str(int(slots)),
    }


def config_from_env(
    env: Optional[Mapping[str, str]] = None,
) -> Optional[GpuPhaseConfig]:
    """The semaphore config from ``env`` (default ``os.environ``), or None.

    Any missing or invalid value disables the semaphore (with a warning for an
    invalid one) rather than failing the run.
    """
    env = os.environ if env is None else env
    raw_dir = str(env.get(ENV_LOCK_DIR) or "").strip()
    gpu = str(env.get(ENV_GPU) or "").strip()
    raw_slots = str(env.get(ENV_SLOTS) or "").strip()
    if not raw_dir or not gpu or not raw_slots:
        return None
    try:
        slots = int(raw_slots)
    except ValueError:
        slots = 0
    if slots < 1:
        logger.warning(
            "GPU phase lock disabled: invalid %s=%r (need an integer >= 1).",
            ENV_SLOTS,
            raw_slots,
        )
        return None
    return GpuPhaseConfig(lock_dir=Path(raw_dir), gpu=gpu, slots=slots)


def gpu_phase_enabled(env: Optional[Mapping[str, str]] = None) -> bool:
    return config_from_env(env) is not None


# --- OS file locks -----------------------------------------------------------


def _try_lock(fd: int) -> bool:
    """Non-blocking exclusive lock on ``fd``; True when taken."""
    if os.name == "nt":  # pragma: no cover - exercised on Windows only
        import msvcrt

        try:
            msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
            return True
        except OSError:
            return False
    import fcntl

    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return True
    except (BlockingIOError, PermissionError):
        return False


def _unlock(fd: int) -> None:
    try:
        if os.name == "nt":  # pragma: no cover - exercised on Windows only
            import msvcrt

            os.lseek(fd, 0, os.SEEK_SET)
            msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)
        else:
            import fcntl

            fcntl.flock(fd, fcntl.LOCK_UN)
    except OSError:
        pass  # closing the fd below drops the lock anyway


def _try_acquire_any(config: GpuPhaseConfig) -> Optional[tuple[int, int]]:
    """Take the first free slot; return ``(slot, fd)`` or None if all are held."""
    config.lock_dir.mkdir(parents=True, exist_ok=True)
    for slot in range(config.slots):
        fd = os.open(str(config.slot_path(slot)), os.O_RDWR | os.O_CREAT, 0o644)
        if _try_lock(fd):
            return slot, fd
        os.close(fd)
    return None


class _WaiterMark:
    """A locked marker file announcing "this process waits for this GPU".

    Only for the wait log (how many children are queued). The slots have no
    queue: whoever polls first after a release wins, so there is no fairness
    or ordering across phases.
    """

    def __init__(self, config: GpuPhaseConfig) -> None:
        self.config = config
        token = f"{os.getpid()}.{threading.get_ident()}"
        self.path = config.waiter_path(token)
        self.fd: Optional[int] = None
        try:
            self.fd = os.open(str(self.path), os.O_RDWR | os.O_CREAT, 0o644)
            if not _try_lock(self.fd):
                os.close(self.fd)
                self.fd = None
        except OSError:
            self.fd = None

    def others_waiting(self) -> int:
        """Live waiters for this GPU besides this one (stale marks are swept)."""
        count = 0
        for path in self.config.lock_dir.glob(self.config.waiter_glob()):
            if path == self.path:
                continue
            try:
                fd = os.open(str(path), os.O_RDWR)
            except OSError:
                continue
            try:
                if _try_lock(fd):
                    # Nobody holds it: its waiter is gone (killed while waiting).
                    _unlock(fd)
                    try:
                        path.unlink()
                    except OSError:
                        pass
                else:
                    count += 1
            finally:
                os.close(fd)
        return count

    def close(self) -> None:
        if self.fd is None:
            return
        try:
            self.path.unlink()
        except OSError:
            pass
        _unlock(self.fd)
        try:
            os.close(self.fd)
        except OSError:
            pass
        self.fd = None


# --- GPU memory --------------------------------------------------------------


def free_gpu_memory() -> None:
    """Return cached allocator memory to the driver, if torch is already loaded.

    Never imports torch: a process that has not touched it has nothing cached.
    """
    gc.collect()
    torch = sys.modules.get("torch")
    if torch is None:
        return
    try:
        cuda = getattr(torch, "cuda", None)
        if cuda is not None and cuda.is_initialized():
            cuda.empty_cache()
    except Exception:  # noqa: BLE001 - freeing memory is best effort
        logger.debug("torch.cuda.empty_cache failed", exc_info=True)


# --- the phase itself ---------------------------------------------------------

# Re-entrancy is per process: a phase opened inside a phase (any thread) joins
# the outer one instead of taking a second slot, which with K=1 would deadlock.
_state_lock = threading.Lock()
_held: dict = {"depth": 0, "fd": None, "slot": None, "label": None}


class GpuPhaseLease:
    """One acquired (or no-op) GPU phase. ``release()`` is idempotent."""

    def __init__(self, label: str, *, active: bool, acquired: bool) -> None:
        self.label = label
        #: True when the semaphore is configured for this process.
        self.active = active
        #: True when this lease counts towards the held phase.
        self.acquired = acquired
        self._released = False

    def release(self) -> None:
        if self._released:
            return
        self._released = True
        if not (self.active and self.acquired):
            return
        with _state_lock:
            _held["depth"] -= 1
            if _held["depth"] > 0:
                return
            fd, slot, label = _held["fd"], _held["slot"], _held["label"]
            _held.update(depth=0, fd=None, slot=None, label=None)
        # Free memory BEFORE unlocking, so the next phase on this GPU gets it.
        free_gpu_memory()
        if fd is not None:
            _unlock(fd)
            try:
                os.close(fd)
            except OSError:
                pass
        logger.info(
            "GPU phase '%s' finished; released GPU slot %s.",
            label,
            None if slot is None else slot + 1,
        )

    def __enter__(self) -> "GpuPhaseLease":
        return self

    def __exit__(self, *exc) -> None:
        self.release()


def acquire_gpu_phase(
    label: str,
    *,
    should_stop: Optional[Callable[[], bool]] = None,
    poll_s: float = DEFAULT_POLL_S,
    env: Optional[Mapping[str, str]] = None,
) -> GpuPhaseLease:
    """Wait for a GPU slot and return the lease (no-op when unconfigured).

    For phases that do not fit one ``with`` block; prefer :func:`gpu_phase`.
    If ``should_stop`` turns true while waiting, the lease comes back with
    ``acquired=False`` and the caller should skip its GPU work.

    There is no queue: when a slot frees, whichever waiting child polls first
    takes it, so phases get no fairness or ordering guarantee. The wait log
    reports how many other children wait for the same GPU.
    """
    config = config_from_env(env)
    if config is None:
        return GpuPhaseLease(label, active=False, acquired=False)
    if _join_held_phase():
        return GpuPhaseLease(label, active=True, acquired=True)
    started = time.monotonic()
    waiter: Optional[_WaiterMark] = None
    next_log = 0.0
    try:
        while True:
            got = _try_acquire_any(config)
            if got is not None:
                slot, fd = got
                with _state_lock:
                    if _held["depth"] > 0:
                        # Another thread of this process took a slot while we
                        # polled: join its phase, never hold a second slot.
                        _held["depth"] += 1
                        joined = True
                    else:
                        _held.update(depth=1, fd=fd, slot=slot, label=label)
                        joined = False
                if joined:
                    _unlock(fd)
                    os.close(fd)
                    return GpuPhaseLease(label, active=True, acquired=True)
                break
            if _join_held_phase():
                return GpuPhaseLease(label, active=True, acquired=True)
            now = time.monotonic()
            if waiter is None:
                waiter = _WaiterMark(config)
            if now >= next_log:
                others = waiter.others_waiting()
                logger.info(
                    "GPU phase '%s': all %d slot(s) on GPU %s are busy; waiting "
                    "(%d other child(ren) also waiting; no queue order)%s...",
                    label,
                    config.slots,
                    config.gpu,
                    others,
                    f", {now - started:.0f}s so far" if next_log else "",
                )
                next_log = now + DEFAULT_WAIT_LOG_S
            if should_stop is not None and should_stop():
                logger.info("GPU phase '%s': stop requested while waiting.", label)
                return GpuPhaseLease(label, active=True, acquired=False)
            time.sleep(max(0.01, float(poll_s)))
    finally:
        if waiter is not None:
            waiter.close()
    waited = waiter is not None
    logger.info(
        "GPU phase '%s': acquired slot %d/%d on GPU %s%s.",
        label,
        slot + 1,
        config.slots,
        config.gpu,
        f" after waiting {time.monotonic() - started:.1f}s" if waited else "",
    )
    return GpuPhaseLease(label, active=True, acquired=True)


def _join_held_phase() -> bool:
    """Join this process's held phase (re-entrancy); True if one is held."""
    with _state_lock:
        if _held["depth"] > 0:
            _held["depth"] += 1
            return True
    return False


@contextmanager
def gpu_phase(
    label: str,
    *,
    should_stop: Optional[Callable[[], bool]] = None,
    poll_s: float = DEFAULT_POLL_S,
) -> Iterator[GpuPhaseLease]:
    """Hold one of this GPU's K phase slots for the ``with`` body.

    No-op without the fan-out env. On exit the slot is freed after returning
    torch's cached CUDA memory to the driver. If a stop arrives while waiting,
    the body still runs with ``lease.acquired`` False -- check
    :func:`phase_skipped` and skip the GPU work.
    """
    lease = acquire_gpu_phase(label, should_stop=should_stop, poll_s=poll_s)
    try:
        yield lease
    finally:
        lease.release()


def phase_skipped(lease: GpuPhaseLease) -> bool:
    """True when a configured phase was abandoned (stop while waiting)."""
    return bool(lease.active and not lease.acquired)
