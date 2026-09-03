"""Cross-process leases for memory-intensive jobs.

The operating-system file lock is authoritative.  JSON metadata is diagnostic
only and is validated with both PID and process start time so PID reuse cannot
make a stale owner look live.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import socket
import subprocess
import time
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import IO, Any, Mapping, Optional

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


def resource_key(accelerator: str, index: int | str = 0) -> str:
    """Return a host-local key for one accelerator or unified-memory pool."""
    return f"{socket.gethostname()}:{accelerator}:{index}"


def canonical_resource_key(
    accelerator: str,
    index: int | str = 0,
    *,
    device_uuid: Optional[str] = None,
    environ: Optional[Mapping[str, str]] = None,
) -> str:
    """Return a stable key for the physical memory pool a job will consume.

    MPS devices always share the host's one unified-memory pool. CUDA prefers a
    UUID, including UUID tokens in ``CUDA_VISIBLE_DEVICES`` and a bounded
    ``nvidia-smi`` probe, so two logical indices cannot lease one physical GPU.
    """
    kind = str(getattr(accelerator, "value", accelerator)).strip().lower()
    host = socket.gethostname()
    if kind == "mps":
        return f"{host}:mps:unified"
    if kind == "cpu":
        return f"{host}:cpu:host-memory"
    if kind != "cuda":
        raise ValueError(f"unsupported accelerator kind: {accelerator!r}")

    visible_index = _physical_cuda_token(index, environ=environ)
    stable_uuid = (device_uuid or "").strip().lower()
    if not stable_uuid and visible_index.lower().startswith(("gpu-", "mig-")):
        stable_uuid = visible_index.lower()
    if not stable_uuid:
        stable_uuid = (_probe_cuda_uuid(visible_index) or "").strip().lower()
    identity = f"uuid:{stable_uuid}" if stable_uuid else f"index:{visible_index}"
    return f"{host}:cuda:{identity}"


def canonical_heavy_job_lease(
    job_name: str,
    accelerator: str,
    index: int | str = 0,
    *,
    device_uuid: Optional[str] = None,
    lease_dir: Optional[Path] = None,
    environ: Optional[Mapping[str, str]] = None,
) -> "HeavyJobLease":
    """Build, but do not acquire, the shared lease for one heavy job."""
    if lease_dir is None:
        from hydra_suite.paths import get_data_dir

        lease_dir = get_data_dir() / "runtime" / "heavy-job-leases"
    key = canonical_resource_key(
        accelerator,
        index,
        device_uuid=device_uuid,
        environ=environ,
    )
    return HeavyJobLease(key, job_name, Path(lease_dir))


def _physical_cuda_token(
    index: int | str, *, environ: Optional[Mapping[str, str]]
) -> str:
    values = os.environ if environ is None else environ
    visible = values.get("CUDA_VISIBLE_DEVICES")
    if not visible:
        return str(index)
    tokens = [token.strip() for token in visible.split(",") if token.strip()]
    try:
        logical_index = int(index)
    except (TypeError, ValueError):
        return str(index)
    if 0 <= logical_index < len(tokens):
        return tokens[logical_index]
    return str(index)


def _probe_cuda_uuid(device_token: str) -> Optional[str]:
    executable = shutil.which("nvidia-smi")
    if executable is None:
        return None
    try:
        completed = subprocess.run(
            [
                executable,
                "--id",
                str(device_token),
                "--query-gpu=uuid",
                "--format=csv,noheader",
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=2.0,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if completed.returncode != 0:
        return None
    first_line = completed.stdout.splitlines()[0].strip() if completed.stdout else ""
    return first_line or None


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

    def __enter__(self) -> "HeavyJobLease":
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
