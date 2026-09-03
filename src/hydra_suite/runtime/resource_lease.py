"""Cross-process leases for memory-intensive jobs.

The operating-system file lock is authoritative.  JSON metadata is diagnostic
only and is validated with both PID and process start time so PID reuse cannot
make a stale owner look live.
"""

from __future__ import annotations

import hashlib
import json
import os
import socket
import time
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import IO, Any, Optional, Sequence

import psutil

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


class ResourceBusyError(RuntimeError):
    """Raised when another live process owns the requested resource."""

    def __init__(self, resource_key: str, owner: "LeaseOwner | None") -> None:
        self.resource_key = resource_key
        self.owner = owner
        detail = "owner metadata unavailable"
        if owner is not None:
            state = "live" if owner_is_live(owner) else "unverified"
            detail = f"PID {owner.pid} ({state}), job {owner.job_name!r}"
        super().__init__(f"Resource {resource_key!r} is already leased by {detail}")


@dataclass(frozen=True)
class LeaseOwner:
    """Diagnostic identity recorded by the holder of an OS-backed lease."""

    resource_key: str
    job_name: str
    lease_id: str
    pid: int
    process_start_time: float
    hostname: str
    acquired_at: float

    @classmethod
    def from_json(cls, value: object) -> Optional["LeaseOwner"]:
        """Parse untrusted persisted metadata, returning ``None`` if invalid."""

        if not isinstance(value, dict):
            return None
        try:
            return cls(
                resource_key=str(value["resource_key"]),
                job_name=str(value["job_name"]),
                lease_id=str(value["lease_id"]),
                pid=int(value["pid"]),
                process_start_time=float(value["process_start_time"]),
                hostname=str(value["hostname"]),
                acquired_at=float(value["acquired_at"]),
            )
        except (KeyError, TypeError, ValueError):
            return None


def canonical_host_memory_key() -> str:
    """Return the one host-wide key for shared physical RAM admission."""
    return f"{socket.gethostname()}:host-memory"


def canonical_resource_key(
    accelerator: str,
    index: int | str = 0,
    *,
    device_uuid: Optional[str] = None,
    device_pci_bus_id: Optional[str] = None,
) -> str:
    """Return a stable key for the physical memory pool a job will consume.

    MPS devices always share the host's one unified-memory pool. CUDA requires
    a physical UUID or PCI identity supplied by the resolved runtime; a logical
    index is never accepted as an ownership boundary.
    """
    kind = str(getattr(accelerator, "value", accelerator)).strip().lower()
    host = socket.gethostname()
    if kind == "mps":
        return f"{host}:mps:unified"
    if kind == "cpu":
        return canonical_host_memory_key()
    if kind != "cuda":
        raise ValueError(f"unsupported accelerator kind: {accelerator!r}")

    stable_uuid = (device_uuid or "").strip().lower()
    stable_pci = (device_pci_bus_id or "").strip().lower()
    if stable_uuid and stable_pci:
        raise ValueError("provide one physical CUDA identity, not both UUID and PCI")
    if not stable_uuid and not stable_pci:
        raise ValueError(
            "a resolver-supplied physical CUDA UUID or PCI identity is required; "
            f"logical index {index!r} is not a safe lease key"
        )
    identity = f"uuid:{stable_uuid}" if stable_uuid else f"pci:{stable_pci}"
    return f"{host}:cuda:{identity}"


def canonical_heavy_job_lease_set(
    job_name: str,
    accelerator: str,
    index: int | str = 0,
    *,
    device_uuid: Optional[str] = None,
    device_pci_bus_id: Optional[str] = None,
    lease_dir: Optional[Path] = None,
) -> "HeavyJobLeaseSet":
    """Build ordered leases for shared host RAM and a physical accelerator."""
    if lease_dir is None:
        from hydra_suite.paths import get_data_dir

        lease_dir = get_data_dir() / "runtime" / "heavy-job-leases"
    kind = str(getattr(accelerator, "value", accelerator)).strip().lower()
    accelerator_key = canonical_resource_key(
        kind,
        index,
        device_uuid=device_uuid,
        device_pci_bus_id=device_pci_bus_id,
    )
    # MPS accelerator allocations are the host allocation pool, so one unified
    # key represents both resources rather than self-deadlocking on two aliases.
    if kind == "cuda":
        keys = [canonical_host_memory_key(), accelerator_key]
    elif kind == "mps":
        keys = [accelerator_key]
    else:
        keys = [canonical_host_memory_key()]
    unique_keys = sorted(set(keys))
    return HeavyJobLeaseSet(
        [HeavyJobLease(key, job_name, Path(lease_dir)) for key in unique_keys]
    )


def owner_is_live(owner: LeaseOwner) -> bool:
    """Validate an owner using PID and creation time, guarding PID reuse."""
    if owner.hostname != socket.gethostname():
        return False
    try:
        process = psutil.Process(owner.pid)
        process_start_time = float(process.create_time())
        return abs(process_start_time - owner.process_start_time) < 0.01
    except (psutil.Error, OSError):
        return False


class HeavyJobLease:
    """Non-blocking exclusive lease held for the lifetime of this object."""

    def __init__(
        self,
        resource_key: str,
        job_name: str,
        lease_dir: Path,
    ) -> None:
        if not resource_key:
            raise ValueError("resource_key must not be empty")
        if not job_name:
            raise ValueError("job_name must not be empty")
        digest = hashlib.sha256(resource_key.encode("utf-8")).hexdigest()[:24]
        self.resource_key = resource_key
        self.job_name = job_name
        self.path = Path(lease_dir) / f"{digest}.lease"
        self._handle: Optional[IO[str]] = None
        self.owner: Optional[LeaseOwner] = None

    def acquire(self) -> "HeavyJobLease":
        """Acquire atomically, overwriting metadata only after owning the lock."""
        if self._handle is not None:
            raise RuntimeError("lease is already acquired")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        handle = self.path.open("a+", encoding="utf-8")
        locked = False
        try:
            try:
                _try_lock(handle)
                locked = True
            except BlockingIOError as exc:
                owner = _read_owner(handle)
                raise ResourceBusyError(self.resource_key, owner) from exc

            process = psutil.Process(os.getpid())
            owner = LeaseOwner(
                resource_key=self.resource_key,
                job_name=self.job_name,
                lease_id=uuid.uuid4().hex,
                pid=os.getpid(),
                process_start_time=process.create_time(),
                hostname=socket.gethostname(),
                acquired_at=time.time(),
            )
            handle.seek(0)
            handle.truncate()
            json.dump(asdict(owner), handle, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        except BaseException:
            try:
                if locked:
                    _unlock(handle)
            finally:
                handle.close()
            raise
        self._handle = handle
        self.owner = owner
        return self

    def release(self) -> None:
        """Release this process's lock without deleting the shared lock inode."""
        handle = self._handle
        if handle is None:
            return
        try:
            _unlock(handle)
        finally:
            handle.close()
            self._handle = None
            self.owner = None

    def fileno(self) -> int:
        """Return the acquired lock descriptor for guardian inheritance."""
        if self._handle is None:
            raise RuntimeError("lease is not acquired")
        return self._handle.fileno()

    def __enter__(self) -> "HeavyJobLease":
        return self.acquire()

    def __exit__(self, *_args: object) -> None:
        self.release()


class HeavyJobLeaseSet:
    """Deadlock-safe ordered leases held and released as one ownership unit."""

    def __init__(self, leases: Sequence[HeavyJobLease]) -> None:
        ordered = tuple(sorted(leases, key=lambda lease: lease.resource_key))
        keys = tuple(lease.resource_key for lease in ordered)
        if len(keys) != len(set(keys)):
            raise ValueError("lease-set resource keys must be unique")
        if not ordered:
            raise ValueError("a heavy-job lease set must not be empty")
        self.leases = ordered
        self.resource_keys = keys
        self._acquired: list[HeavyJobLease] = []

    def acquire(self) -> "HeavyJobLeaseSet":
        """Acquire every resource in canonical order, unwinding on failure."""
        if self._acquired:
            raise RuntimeError("lease set is already acquired")
        try:
            for lease in self.leases:
                lease.acquire()
                self._acquired.append(lease)
        except BaseException:
            self.release()
            raise
        return self

    def release(self) -> None:
        """Release all acquired resources in reverse canonical order."""
        while self._acquired:
            self._acquired.pop().release()

    def filenos(self) -> tuple[int, ...]:
        """Return descriptors that keep every acquired OS lock owned."""
        if len(self._acquired) != len(self.leases):
            raise RuntimeError("lease set is not fully acquired")
        return tuple(lease.fileno() for lease in self._acquired)

    def __enter__(self) -> "HeavyJobLeaseSet":
        return self.acquire()

    def __exit__(self, *_args: object) -> None:
        self.release()


def _read_owner(handle: IO[str]) -> Optional[LeaseOwner]:
    try:
        handle.seek(0)
        return LeaseOwner.from_json(json.load(handle))
    except (OSError, json.JSONDecodeError):
        return None


def _try_lock(handle: IO[str]) -> None:
    if fcntl is not None:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        return
    if msvcrt is not None:  # pragma: no cover - exercised on Windows CI
        try:
            handle.seek(0)
            if not handle.read(1):
                handle.seek(0)
                handle.write("\0")
                handle.flush()
            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            return
        except OSError as exc:
            raise BlockingIOError from exc
    raise RuntimeError("this platform has no supported inter-process file lock")


def _unlock(handle: IO[str]) -> None:
    if fcntl is not None:
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
    elif msvcrt is not None:  # pragma: no cover - exercised on Windows CI
        handle.seek(0)
        msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
