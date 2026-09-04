"""Process-tree memory watchdog, bounded output, and exit classification."""

from __future__ import annotations

import codecs
import os
import platform
import select
import signal
import subprocess
import threading
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Optional

import psutil

from .process_guardian import spawn_parent_guardian
from .resource_lease import (
    HeavyJobLeaseSet,
    canonical_heavy_job_lease_set,
    canonical_resource_keys,
)
from .resource_limits import (
    CgroupEvidence,
    LimitBackend,
    LimitedLaunch,
    cgroup_path_contains_unit,
    probe_systemd_cgroup_evidence,
    signal_systemd_scope,
)

_FORK_EXCLUDED_FDS: set[int] = set()
_FORK_EXCLUDED_LOCK = threading.Lock()


def _close_excluded_fds_in_fork_child() -> None:
    """Prevent unrelated fork children from extending containment ownership."""

    try:
        for descriptor in tuple(_FORK_EXCLUDED_FDS):
            try:
                os.close(descriptor)
            except OSError:
                pass
        _FORK_EXCLUDED_FDS.clear()
    finally:
        _FORK_EXCLUDED_LOCK.release()


def _register_fork_excluded_fds(descriptors: tuple[int, ...]) -> None:
    with _FORK_EXCLUDED_LOCK:
        _FORK_EXCLUDED_FDS.update(descriptors)


def _close_fork_excluded_fd(descriptor: int) -> None:
    """Close and unregister one descriptor atomically against ``fork``."""

    with _FORK_EXCLUDED_LOCK:
        try:
            os.close(descriptor)
        finally:
            _FORK_EXCLUDED_FDS.discard(descriptor)


if os.name == "posix" and hasattr(os, "register_at_fork"):
    os.register_at_fork(
        before=_FORK_EXCLUDED_LOCK.acquire,
        after_in_parent=_FORK_EXCLUDED_LOCK.release,
        after_in_child=_close_excluded_fds_in_fork_child,
    )


class WatchdogTrigger(str, Enum):
    """Reason the parent watchdog intervened in a workload."""

    SOFT_RSS = "host-soft-rss"
    HARD_RSS = "host-hard-rss"
    SYSTEM_RESERVE = "host-system-reserve"
    PROCESS_COUNT = "process-count"
    OBSERVATION_FAILURE = "observation-failure"


@dataclass(frozen=True)
class WatchdogPolicy:
    """Derived process-tree thresholds and bounded polling behavior."""

    soft_tree_rss_bytes: int
    hard_tree_rss_bytes: int
    minimum_system_available_bytes: int
    poll_interval_seconds: float = 0.25
    terminate_grace_seconds: float = 5.0

    def __post_init__(self) -> None:
        if self.soft_tree_rss_bytes <= 0 or self.hard_tree_rss_bytes <= 0:
            raise ValueError("watchdog RSS limits must be positive")
        if self.soft_tree_rss_bytes > self.hard_tree_rss_bytes:
            raise ValueError("soft RSS limit cannot exceed hard RSS limit")
        if self.minimum_system_available_bytes < 0:
            raise ValueError("system available-memory floor must be non-negative")
        if self.poll_interval_seconds <= 0 or self.terminate_grace_seconds < 0:
            raise ValueError("watchdog timing values are invalid")


@dataclass(frozen=True)
class ContainmentPlan:
    """One immutable source for kernel and watchdog memory boundaries."""

    launch: LimitedLaunch
    job_name: str
    minimum_system_available_bytes: int
    expected_resource_keys: tuple[str, ...] = field(init=False)
    poll_interval_seconds: float = 0.25
    terminate_grace_seconds: float = 5.0

    def __post_init__(self) -> None:
        # Constructing the derived policy here validates every timing/floor
        # value before a child can be created.
        self.watchdog_policy
        if not self.job_name.strip():
            raise ValueError("containment job name must not be empty")
        kind = self.launch.accelerator_kind.value
        canonical_keys = canonical_resource_keys(
            kind,
            device_uuid=self.launch.accelerator_device_uuid,
            device_pci_bus_id=self.launch.accelerator_pci_bus_id,
        )
        object.__setattr__(self, "expected_resource_keys", canonical_keys)

    @property
    def watchdog_policy(self) -> WatchdogPolicy:
        """Derive watchdog thresholds from the launch's retained limits."""

        limits = self.launch.limits
        return WatchdogPolicy(
            soft_tree_rss_bytes=limits.soft_host_bytes,
            hard_tree_rss_bytes=limits.hard_host_bytes,
            minimum_system_available_bytes=self.minimum_system_available_bytes,
            poll_interval_seconds=self.poll_interval_seconds,
            terminate_grace_seconds=self.terminate_grace_seconds,
        )


@dataclass(frozen=True)
class WatchdogOutcome:
    """Observed threshold crossing and the resulting termination behavior."""

    trigger: WatchdogTrigger
    observed_tree_rss_bytes: int
    observed_system_available_bytes: int
    hard_kill_sent: bool
    graceful_exit: bool


@dataclass(frozen=True)
class _ProcessIdentity:
    pid: int
    create_time: float

    def probe(self) -> tuple[Optional[psutil.Process], bool]:
        """Return ``(process, gone)`` without calling access denial death."""

        try:
            process = psutil.Process(self.pid)
            if abs(process.create_time() - self.create_time) >= 0.01:
                return None, True
            if process.status() == psutil.STATUS_ZOMBIE:
                return None, True
            return process, False
        except (psutil.NoSuchProcess, psutil.ZombieProcess):
            return None, True
        except (psutil.AccessDenied, OSError):
            return None, False

    def resolve(self) -> Optional[psutil.Process]:
        """Resolve this PID only when its creation time still matches."""

        return self.probe()[0]


class OwnedProcessTree:
    """Signal only a captured process identity or its dedicated process group."""

    def __init__(
        self,
        process: subprocess.Popen[Any],
        *,
        owns_process_group: bool,
        systemd_unit: Optional[str] = None,
        max_tracked_identities: int = 512,
    ):
        if max_tracked_identities < 1:
            raise ValueError("maximum tracked process count must be positive")
        root = psutil.Process(process.pid)
        self.process = process
        self.root = _ProcessIdentity(process.pid, root.create_time())
        self._known_identities: dict[int, _ProcessIdentity] = {self.root.pid: self.root}
        self._overflow_identity: Optional[_ProcessIdentity] = None
        self._state_lock = threading.RLock()
        self._permanent_ownership_uncertain = False
        self.process_group_id: Optional[int] = None
        self.systemd_unit = systemd_unit
        self.max_tracked_identities = max_tracked_identities
        self.identity_overflowed = False
        self.scope_signal_failed = False
        if owns_process_group and os.name == "posix":
            group = os.getpgid(process.pid)
            if group != process.pid:
                raise ValueError(
                    "owned process groups must be created with start_new_session=True"
                )
            self.process_group_id = group

    def identities(self) -> tuple[_ProcessIdentity, ...]:
        """Capture and return every still-live, identity-validated process."""

        with self._state_lock:
            self._prune_dead_identities()
            discovered, overflow, discovery_uncertain = self._discover_identities()
            if discovery_uncertain:
                self._permanent_ownership_uncertain = True
            for identity in discovered:
                self._known_identities.setdefault(identity.pid, identity)
            if overflow is not None:
                self.identity_overflowed = True
                # Retain the first over-cap identity separately until it is
                # proven gone. The registry remains bounded at max + one.
                self._overflow_identity = overflow
            live = tuple(
                identity
                for identity in (
                    *self._known_identities.values(),
                    *((self._overflow_identity,) if self._overflow_identity else ()),
                )
                if not identity.probe()[1]
            )
            if self.identity_overflowed:
                if not self._signal_snapshot(signal.SIGKILL, live):
                    self._permanent_ownership_uncertain = True
            return live

    def _discover_identities(
        self,
    ) -> tuple[tuple[_ProcessIdentity, ...], Optional[_ProcessIdentity], bool]:
        root, root_gone = self.root.probe()
        if root_gone or root is None:
            return (), None, not root_gone
        discovered: list[_ProcessIdentity] = []
        frontier = {root.pid}
        expanded: set[int] = set()
        discovered_pids: set[int] = set()
        while frontier:
            next_frontier: set[int] = set()
            try:
                processes = psutil.process_iter(["pid", "ppid", "create_time"])
                for child in processes:
                    try:
                        parent_pid = int(child.info["ppid"])
                    except (KeyError, TypeError, ValueError):
                        return tuple(discovered), None, True
                    if parent_pid not in frontier:
                        continue
                    try:
                        child_identity = _ProcessIdentity(
                            child.pid, float(child.info["create_time"])
                        )
                    except (KeyError, TypeError, ValueError):
                        return tuple(discovered), None, True
                    if (
                        child_identity.pid not in self._known_identities
                        and child_identity.pid not in discovered_pids
                    ):
                        if len(self._known_identities) + len(discovered) >= (
                            self.max_tracked_identities
                        ):
                            return tuple(discovered), child_identity, False
                        discovered.append(child_identity)
                        discovered_pids.add(child_identity.pid)
                    if child_identity.pid not in expanded:
                        next_frontier.add(child_identity.pid)
            except (psutil.Error, OSError, RuntimeError):
                return tuple(discovered), None, True
            expanded.update(frontier)
            frontier = next_frontier - expanded
        return tuple(discovered), None, False

    def _prune_dead_identities(self) -> None:
        for pid, identity in tuple(self._known_identities.items()):
            _, gone = identity.probe()
            if gone:
                self._known_identities.pop(pid, None)
        if self._overflow_identity is not None:
            _, gone = self._overflow_identity.probe()
            if gone:
                self._overflow_identity = None

    def is_alive(self) -> bool:
        """Return whether any captured member of the owned tree is live."""

        # Reap a completed root before checking captured descendants.  Without
        # this poll a cooperative child can remain visible as a zombie until
        # the grace period expires and be mislabeled as an unresponsive job.
        with self._state_lock:
            self.process.poll()
            if self.identities():
                return True
            if self.process_group_id is not None:
                return any(
                    self._identity_process_group(identity) == self.process_group_id
                    for identity in self._known_identities.values()
                )
            return False

    def rss_bytes(self) -> int:
        """Return current resident bytes across all observable owned members."""

        with self._state_lock:
            total = 0
            for identity in self.identities():
                process = identity.resolve()
                if process is None:
                    self._permanent_ownership_uncertain = True
                    continue
                try:
                    total += process.memory_info().rss
                except (psutil.NoSuchProcess, psutil.ZombieProcess):
                    continue
                except (psutil.AccessDenied, OSError):
                    self._permanent_ownership_uncertain = True
            return total

    def terminate(self) -> bool:
        """Request graceful termination of the owned boundary."""

        return self._signal(signal.SIGTERM)

    def kill(self) -> bool:
        """Forcefully terminate the owned boundary."""

        if os.name == "posix":
            return self._signal(signal.SIGKILL)
        for identity in reversed(self.identities()):  # pragma: no cover - Windows CI
            process = identity.resolve()
            if process is None:
                continue
            try:
                process.kill()
            except (psutil.Error, OSError):
                continue
        return True

    @property
    def ownership_uncertain(self) -> bool:
        """Return whether the authoritative systemd scope rejected a signal."""
        with self._state_lock:
            return self.scope_signal_failed or self._permanent_ownership_uncertain

    def mark_ownership_uncertain(self) -> None:
        """Permanently retain ownership after an untrustworthy observation."""

        with self._state_lock:
            self._permanent_ownership_uncertain = True

    def _signal(self, signum: signal.Signals) -> bool:
        with self._state_lock:
            identities = self.identities()
            return self._signal_snapshot(signum, identities)

    def _signal_snapshot(
        self,
        signum: signal.Signals,
        identities: tuple[_ProcessIdentity, ...],
    ) -> bool:
        group_signalled = False
        if self.systemd_unit is not None:
            # systemd is authoritative for every process that remains in the
            # cgroup, including descendants invisible to psutil.
            scope_signalled = signal_systemd_scope(self.systemd_unit, int(signum))
            self.scope_signal_failed = not scope_signalled
            direct_ok = True
            for identity in reversed(identities):
                membership = self._identity_in_systemd_scope(identity)
                if membership is True:
                    continue
                if membership is None or not self._signal_identity(identity, signum):
                    direct_ok = False
            if not direct_ok:
                self._permanent_ownership_uncertain = True
            return scope_signalled and direct_ok
        if self.process_group_id is not None and any(
            self._identity_process_group(identity) == self.process_group_id
            for identity in identities
        ):
            # Signal a numeric group only while an identity-validated member
            # still proves that the group is ours. This prevents PID/PGID reuse.
            try:
                os.killpg(self.process_group_id, signum)
                group_signalled = True
            except ProcessLookupError:
                pass
            except (PermissionError, OSError):
                self._permanent_ownership_uncertain = True
        # Also signal every captured descendant that escaped the owned group or
        # cgroup. Descendants go first and every PID is creation-time validated.
        for identity in reversed(identities):
            if group_signalled and (
                self._identity_process_group(identity) == self.process_group_id
            ):
                continue
            if not self._signal_identity(identity, signum):
                self._permanent_ownership_uncertain = True
        return not self._permanent_ownership_uncertain

    def _signal_identity(
        self, identity: _ProcessIdentity, signum: signal.Signals
    ) -> bool:
        process, gone = identity.probe()
        if gone:
            return True
        if process is None:
            return False
        try:
            process.send_signal(signum)
            return True
        except (psutil.NoSuchProcess, psutil.ZombieProcess):
            return True
        except (psutil.AccessDenied, OSError):
            return False

    @staticmethod
    def _identity_process_group(identity: _ProcessIdentity) -> Optional[int]:
        process, gone = identity.probe()
        if gone or process is None or os.name != "posix":
            return None
        try:
            process_group = os.getpgid(process.pid)
        except (ProcessLookupError, PermissionError, OSError):
            return None
        validated_again, gone_after_probe = identity.probe()
        if gone_after_probe or validated_again is None:
            return None
        return process_group

    def _identity_in_systemd_scope(self, identity: _ProcessIdentity) -> Optional[bool]:
        if self.systemd_unit is None or platform.system() != "Linux":
            return None
        process, gone = identity.probe()
        if gone:
            # Exiting between the identity snapshot and membership check is a
            # successful teardown outcome, not ambiguous ownership.
            return True
        if process is None:
            return None
        try:
            cgroup_text = Path(f"/proc/{process.pid}/cgroup").read_text(
                encoding="utf-8"
            )
        except FileNotFoundError:
            _, gone_after_read = identity.probe()
            return True if gone_after_read else None
        except (OSError, UnicodeError):
            return None
        return cgroup_path_contains_unit(cgroup_text, self.systemd_unit)


class ProcessTreeWatchdog:
    """Monitor independently of child output and training-step progress."""

    def __init__(
        self,
        tree: OwnedProcessTree,
        policy: WatchdogPolicy,
        *,
        accelerator_probe: Optional[Callable[[], int]] = None,
    ) -> None:
        self.tree = tree
        self.policy = policy
        self.outcome: Optional[WatchdogOutcome] = None
        self.peak_tree_rss_bytes = 0
        self.minimum_system_available_bytes: Optional[int] = None
        self.peak_accelerator_bytes: Optional[int] = None
        self.accelerator_observation_error: Optional[str] = None
        self._accelerator_probe = accelerator_probe
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        """Start the independent memory-monitoring thread exactly once."""

        if self._thread is not None:
            raise RuntimeError("watchdog has already been started")
        self._thread = threading.Thread(
            target=self._run,
            name=f"hydra-memory-watchdog-{self.tree.root.pid}",
            daemon=True,
        )
        self._thread.start()

    def stop(self, timeout: Optional[float] = None) -> Optional[WatchdogOutcome]:
        """Request watchdog shutdown and return any recorded intervention."""

        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout)
        return self.outcome

    def _run(self) -> None:
        try:
            self._monitor_until_stopped()
        except BaseException:  # noqa: B036, BLE001 - watchdog must fail closed
            self.tree.mark_ownership_uncertain()
            try:
                self.tree.kill()
            except BaseException:  # noqa: B036, BLE001 - ownership remains uncertain
                pass
            self.outcome = WatchdogOutcome(
                WatchdogTrigger.OBSERVATION_FAILURE,
                0,
                0,
                hard_kill_sent=True,
                graceful_exit=False,
            )

    def _monitor_until_stopped(self) -> None:
        while not self._stop.is_set():
            tree_alive = self.tree.is_alive()
            if self.tree.identity_overflowed:
                self.tree.kill()
                self.outcome = WatchdogOutcome(
                    WatchdogTrigger.PROCESS_COUNT,
                    self.tree.rss_bytes(),
                    int(psutil.virtual_memory().available),
                    hard_kill_sent=True,
                    graceful_exit=False,
                )
                return
            if not tree_alive:
                return
            rss = self.tree.rss_bytes()
            available = int(psutil.virtual_memory().available)
            self.peak_tree_rss_bytes = max(self.peak_tree_rss_bytes, rss)
            self.minimum_system_available_bytes = min(
                (
                    self.minimum_system_available_bytes
                    if self.minimum_system_available_bytes is not None
                    else available
                ),
                available,
            )
            if available < self.policy.minimum_system_available_bytes:
                self.tree.kill()
                self.outcome = WatchdogOutcome(
                    WatchdogTrigger.SYSTEM_RESERVE,
                    rss,
                    available,
                    hard_kill_sent=True,
                    graceful_exit=False,
                )
                return
            if rss >= self.policy.hard_tree_rss_bytes:
                self.tree.kill()
                self.outcome = WatchdogOutcome(
                    WatchdogTrigger.HARD_RSS,
                    rss,
                    available,
                    hard_kill_sent=True,
                    graceful_exit=False,
                )
                return
            if rss >= self.policy.soft_tree_rss_bytes:
                self._handle_soft_limit(rss, available)
                return
            # CUDA telemetry is deliberately last: nvidia-smi may be slow or
            # wedged, while the host reserve/RSS checks are the survival guard.
            if self._accelerator_probe is not None:
                try:
                    observed = int(self._accelerator_probe())
                    if observed < 0:
                        raise ValueError("accelerator usage cannot be negative")
                    self.peak_accelerator_bytes = max(
                        self.peak_accelerator_bytes or 0, observed
                    )
                except Exception as exc:  # telemetry is not a CUDA hard cap
                    self.accelerator_observation_error = str(exc)
            self._stop.wait(self.policy.poll_interval_seconds)

    def _handle_soft_limit(self, rss: int, available: int) -> None:
        self.tree.terminate()
        deadline = time.monotonic() + self.policy.terminate_grace_seconds
        while time.monotonic() < deadline and not self._stop.is_set():
            if not self.tree.is_alive():
                self.outcome = WatchdogOutcome(
                    WatchdogTrigger.SOFT_RSS,
                    rss,
                    available,
                    hard_kill_sent=False,
                    graceful_exit=True,
                )
                return
            self._stop.wait(min(self.policy.poll_interval_seconds, 0.05))
        if self.tree.is_alive():
            self.tree.kill()
            hard_kill = True
        else:
            hard_kill = False
        self.outcome = WatchdogOutcome(
            WatchdogTrigger.SOFT_RSS,
            rss,
            available,
            hard_kill_sent=hard_kill,
            graceful_exit=not hard_kill,
        )


class BoundedLineBuffer:
    """Thread-safe output channel bounded by both line and character counts."""

    def __init__(self, *, max_lines: int = 512, max_chars: int = 256 * 1024):
        if max_lines < 1 or max_chars < 1:
            raise ValueError("output buffer limits must be positive")
        self.max_lines = max_lines
        self.max_chars = max_chars
        self._lines: deque[tuple[int, str]] = deque()
        self._chars = 0
        self._next_sequence = 0
        self._drained_through = -1
        self._dropped_lines = 0
        self._condition = threading.Condition()
        self._eof = False
        self._error: Optional[BaseException] = None

    @property
    def dropped_lines(self) -> int:
        """Return the count of unread lines evicted by buffer limits."""

        with self._condition:
            return self._dropped_lines

    @property
    def retained_chars(self) -> int:
        """Return the exact character count currently retained."""

        with self._condition:
            return self._chars

    def append(self, line: str) -> None:
        """Append one line while enforcing both configured bounds."""

        with self._condition:
            if len(line) > self.max_chars:
                line = line[-self.max_chars :]
                self._dropped_lines += 1
            sequence = self._next_sequence
            self._next_sequence += 1
            self._lines.append((sequence, line))
            self._chars += len(line)
            while len(self._lines) > self.max_lines or self._chars > self.max_chars:
                dropped_sequence, dropped = self._lines.popleft()
                self._chars -= len(dropped)
                if dropped_sequence > self._drained_through:
                    self._dropped_lines += 1
            self._condition.notify_all()

    def close(self, error: Optional[BaseException] = None) -> None:
        """Mark the stream complete and wake blocked consumers."""

        with self._condition:
            self._error = error
            self._eof = True
            self._condition.notify_all()

    def drain(
        self, timeout: float = 0.0
    ) -> tuple[list[str], bool, Optional[BaseException]]:
        """Return retained lines, whether EOF was seen, and any reader error."""
        with self._condition:
            has_unread = any(
                sequence > self._drained_through for sequence, _ in self._lines
            )
            if not has_unread and not self._eof and timeout > 0:
                self._condition.wait(timeout)
            unread = [
                (sequence, line)
                for sequence, line in self._lines
                if sequence > self._drained_through
            ]
            if unread:
                self._drained_through = unread[-1][0]
            lines = [line for _, line in unread]
            return lines, self._eof, self._error

    def tail(self) -> tuple[str, ...]:
        """Return a stable snapshot of retained output lines."""

        with self._condition:
            return tuple(line for _, line in self._lines)


def pump_stdout(
    stdout: Any,
    output: BoundedLineBuffer,
    *,
    read_chunk_chars: int = 64 * 1024,
) -> None:
    """Incrementally split fixed-size reads into bounded output records."""
    if read_chunk_chars < 1:
        raise ValueError("stdout read chunks must be positive")
    pending = ""

    def retain(chunk: str) -> None:
        nonlocal pending
        pending += chunk
        while pending:
            newline = pending.find("\n")
            if newline >= 0:
                output.append(pending[: newline + 1])
                pending = pending[newline + 1 :]
            elif len(pending) >= read_chunk_chars:
                # A newline-free log record must never grow with producer
                # output. Preserve it as independently bounded fragments.
                output.append(pending[:read_chunk_chars])
                pending = pending[read_chunk_chars:]
            else:
                break

    binary_buffer = getattr(stdout, "buffer", None)
    read_available = getattr(binary_buffer, "read1", None)
    decoder = None
    if callable(read_available):
        # Buffered ``read1`` returns currently available pipe bytes instead of
        # waiting for the requested count like TextIOWrapper.read(size).
        decoder = codecs.getincrementaldecoder(
            getattr(stdout, "encoding", None) or "utf-8"
        )(errors="replace")
    else:
        read_available = stdout.read
    try:
        while True:
            raw_chunk = read_available(read_chunk_chars)
            if not raw_chunk:
                break
            if isinstance(raw_chunk, bytes):
                if decoder is None:
                    decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
                chunk = decoder.decode(raw_chunk, final=False)
            else:
                chunk = str(raw_chunk)
            retain(chunk)
        if decoder is not None:
            retain(decoder.decode(b"", final=True))
        if pending:
            output.append(pending)
    except Exception as exc:  # noqa: BLE001 - transferred to owner thread
        output.close(error=exc)
    else:
        output.close()


class ExitKind(str, Enum):
    """Stable user-facing completion and failure categories."""

    SUCCESS = "success"
    CANCELED = "canceled"
    HOST_ADMISSION_REFUSAL = "host-admission-refusal"
    HOST_SOFT_LIMIT = "host-soft-limit"
    HOST_HARD_LIMIT = "host-hard-limit"
    ACCELERATOR_OOM = "accelerator-oom"
    SIGNALLED = "signalled"
    ORDINARY_FAILURE = "ordinary-failure"


@dataclass(frozen=True)
class ExitEvidence:
    """Bounded evidence used to classify a completed child."""

    returncode: int
    output_tail: str = ""
    requested_cancel: bool = False
    admission_refused: bool = False
    watchdog: Optional[WatchdogOutcome] = None
    cgroup: Optional[CgroupEvidence] = None
    limit_backend: Optional[LimitBackend] = None


@dataclass(frozen=True)
class ClassifiedExit:
    """Exit category and concise diagnostic message."""

    kind: ExitKind
    message: str


@dataclass(frozen=True)
class SupervisedResult:
    """Bounded result returned after complete process-tree teardown."""

    returncode: int
    classified_exit: ClassifiedExit
    output_tail: tuple[str, ...]
    dropped_output_lines: int
    watchdog: Optional[WatchdogOutcome]
    cgroup: Optional[CgroupEvidence]
    output_error: Optional[str]
    peak_accelerator_bytes: Optional[int] = None
    accelerator_observation_error: Optional[str] = None
    peak_tree_rss_bytes: int = 0
    minimum_system_available_bytes: Optional[int] = None


class WorkloadStillOwnedError(RuntimeError):
    """Teardown could not prove exit; the sidecar intentionally retains ownership."""

    def __init__(self, message: str, sidecar: "SupervisedSidecar") -> None:
        self.sidecar = sidecar
        # Higher-level orchestration may attach the durable run identity while
        # preserving this same recovery-bearing exception object.
        self.run_id = ""
        self.registry_update_error = ""
        self.recovery_error = ""
        self.recovery_cleanup: Optional[Callable[[], None]] = None
        super().__init__(message)


class SupervisedSidecar:
    """Own one limited child, its output pump, and its memory watchdog."""

    def __init__(
        self,
        plan: ContainmentPlan,
        *,
        prelaunch_check: Optional[Callable[[], None]] = None,
        accelerator_probe: Optional[Callable[[], int]] = None,
        output_max_lines: int = 512,
        output_max_chars: int = 256 * 1024,
    ) -> None:
        # Validate all configuration and allocate bounded parent structures
        # before acquiring a lease or creating a child.
        if not isinstance(plan, ContainmentPlan):
            raise TypeError("SupervisedSidecar requires an immutable ContainmentPlan")
        watchdog_policy = plan.watchdog_policy
        output = BoundedLineBuffer(
            max_lines=output_max_lines, max_chars=output_max_chars
        )
        self.plan = plan
        self.launch = plan.launch
        self.output = output
        self._leases: HeavyJobLeaseSet = canonical_heavy_job_lease_set(
            plan.job_name,
            self.launch.accelerator_kind,
            device_uuid=self.launch.accelerator_device_uuid,
            device_pci_bus_id=self.launch.accelerator_pci_bus_id,
        )
        self._leases_released = False
        self._parent_liveness_write_fd: Optional[int] = None
        self._guardian_ack_read_fd: Optional[int] = None
        self._guardian_started = False
        self._guardian_teardown_requested = False
        self._guardian_ack_received = False
        self._guardian_process: Optional[subprocess.Popen[bytes]] = None
        self._fork_excluded_fds: tuple[int, ...] = ()
        # Construction after ``Popen`` is deliberately recoverable.  Keep
        # every later component explicit so a WorkloadStillOwnedError can
        # return this object as a usable cleanup owner even when guardian,
        # reader, or watchdog setup did not finish.
        self.process: Optional[subprocess.Popen[Any]] = None
        self.tree: Optional[OwnedProcessTree] = None
        self._reader: Optional[threading.Thread] = None
        self._reader_started = False
        self.watchdog: Optional[ProcessTreeWatchdog] = None

        if self._leases.resource_keys != plan.expected_resource_keys:
            raise RuntimeError("internal canonical lease construction diverged")
        if prelaunch_check is not None and not callable(prelaunch_check):
            raise TypeError("prelaunch_check must be callable")
        if accelerator_probe is not None and not callable(accelerator_probe):
            raise TypeError("accelerator_probe must be callable")
        if os.name != "posix" and plan.expected_resource_keys:
            raise RuntimeError(
                "leased heavy jobs require a parent-death guardian on this platform"
            )

        child_env = dict(self.launch.environment)
        pass_fds: tuple[int, ...] = ()
        guardian_read_fd: Optional[int] = None
        guardian_ack_write_fd: Optional[int] = None
        start_gate_read_fd: Optional[int] = None
        start_gate_write_fd: Optional[int] = None
        if self.launch.backend is not LimitBackend.SYSTEMD_CGROUP:
            child_env["HYDRA_SUPERVISOR_PID"] = str(os.getpid())
        if os.name == "posix":
            guardian_read_fd, parent_write_fd = os.pipe()
            parent_ack_read_fd, guardian_ack_write_fd = os.pipe()
            start_gate_read_fd, start_gate_write_fd = os.pipe()
            os.set_inheritable(start_gate_read_fd, True)
            child_env["HYDRA_START_GATE_FD"] = str(start_gate_read_fd)
            containment_token = uuid.uuid4().hex
            child_env["HYDRA_CONTAINMENT_TOKEN"] = containment_token
            pass_fds = (start_gate_read_fd,)
            self._parent_liveness_write_fd = parent_write_fd
            self._guardian_ack_read_fd = parent_ack_read_fd

        popen_kwargs: dict[str, Any] = {
            "env": child_env,
            "stdout": subprocess.PIPE,
            "stderr": subprocess.STDOUT,
            "text": True,
            "errors": "replace",
            "bufsize": 1,
        }
        if os.name == "posix":
            popen_kwargs["start_new_session"] = True
            if pass_fds:
                popen_kwargs["pass_fds"] = pass_fds
        process: Optional[subprocess.Popen[Any]] = None
        try:
            self._leases.acquire()
            if prelaunch_check is not None:
                prelaunch_check()
            process = subprocess.Popen(self.launch.command, **popen_kwargs)
            self.process = process
            if start_gate_read_fd is not None:
                os.close(start_gate_read_fd)
                start_gate_read_fd = None
            self.tree = OwnedProcessTree(
                self.process,
                owns_process_group=os.name == "posix",
                systemd_unit=self.launch.systemd_unit,
                max_tracked_identities=self.launch.limits.max_processes,
            )
            if guardian_read_fd is not None and guardian_ack_write_fd is not None:
                assert start_gate_write_fd is not None
                self._guardian_process = spawn_parent_guardian(
                    supervisor_pid=os.getpid(),
                    supervisor_create_time=float(
                        psutil.Process(os.getpid()).create_time()
                    ),
                    workload_pid=self.process.pid,
                    process_group_id=self.tree.process_group_id or self.process.pid,
                    liveness_read_fd=guardian_read_fd,
                    acknowledgement_write_fd=guardian_ack_write_fd,
                    containment_token=containment_token,
                    max_identities=self.launch.limits.max_processes,
                    systemd_unit=self.launch.systemd_unit,
                    lease_fds=self._leases.filenos(),
                    environment=child_env,
                )
                guardian_read_fd = None
                guardian_ack_write_fd = None
                self._guardian_started = True
                self._fork_excluded_fds = tuple(
                    descriptor
                    for descriptor in (
                        self._parent_liveness_write_fd,
                        self._guardian_ack_read_fd,
                        *self._leases.filenos(),
                    )
                    if descriptor is not None
                )
                _register_fork_excluded_fds(self._fork_excluded_fds)
                os.write(start_gate_write_fd, b"G")
                os.close(start_gate_write_fd)
                start_gate_write_fd = None
            if self.process.stdout is None:
                raise RuntimeError("supervised child stdout pipe was not created")
            self._reader = threading.Thread(
                target=pump_stdout,
                args=(self.process.stdout, self.output),
                name=f"hydra-output-pump-{self.process.pid}",
                daemon=True,
            )
            self.watchdog = ProcessTreeWatchdog(
                self.tree, watchdog_policy, accelerator_probe=accelerator_probe
            )
            self._reader.start()
            self._reader_started = True
            self.watchdog.start()
        except BaseException:
            for descriptor in (
                guardian_read_fd,
                guardian_ack_write_fd,
                start_gate_read_fd,
                start_gate_write_fd,
            ):
                if descriptor is not None:
                    try:
                        os.close(descriptor)
                    except OSError:
                        pass
            if process is not None:
                self._kill_and_reap_after_setup_failure(process)
            if self._guardian_started and not self._complete_guardian_teardown():
                raise WorkloadStillOwnedError(
                    "guardian could not prove quiescence after setup failure", self
                )
            self._close_unstarted_guardian_fds()
            self._release_leases()
            raise

    def cancel(self, grace_seconds: float = 5.0) -> None:
        """Cancel the complete owned process group, escalating after a grace."""
        if grace_seconds < 0:
            raise ValueError("cancel grace must be non-negative")
        if self.watchdog is not None:
            self.watchdog.stop(timeout=1.0)
        if self.tree is not None and self.process is not None:
            teardown_complete = self._terminate_and_reap(grace_seconds)
        elif self.process is not None:
            try:
                self._kill_and_reap_after_setup_failure(self.process)
            except WorkloadStillOwnedError:
                teardown_complete = False
            else:
                teardown_complete = True
        else:
            teardown_complete = True
        if not teardown_complete:
            raise WorkloadStillOwnedError(
                "child process tree survived SIGKILL; sidecar and lease remain owned",
                self,
            )
        self._finish_local_teardown()

    def wait(
        self,
        timeout: Optional[float] = None,
        *,
        requested_cancel: bool = False,
        post_exit_check: Optional[Callable[[SupervisedResult], None]] = None,
    ) -> SupervisedResult:
        """Wait for completion and return bounded output plus classified evidence."""
        if timeout is not None and timeout < 0:
            raise ValueError("wait timeout must be non-negative")
        if (
            self.process is None
            or self.tree is None
            or self.watchdog is None
            or self._reader is None
        ):
            raise RuntimeError(
                "sidecar construction did not complete; use cancel() to retry cleanup"
            )
        deadline = None if timeout is None else time.monotonic() + timeout
        while self.tree.is_alive():
            if self.process.poll() is not None:
                self.watchdog.stop(timeout=1.0)
                if not self._terminate_and_reap(0.0):
                    raise WorkloadStillOwnedError(
                        "root exited but owned descendants survived teardown",
                        self,
                    )
                break
            if deadline is not None and time.monotonic() >= deadline:
                self.cancel(self.plan.terminate_grace_seconds)
                assert timeout is not None
                raise subprocess.TimeoutExpired(self.launch.command, timeout)
            time.sleep(0.02)
        returncode = self.process.wait(timeout=1.0)
        watchdog_outcome = self.watchdog.stop(timeout=2.0)
        if not self._complete_guardian_teardown():
            raise WorkloadStillOwnedError(
                "guardian could not prove process-tree quiescence", self
            )
        self._reader.join(timeout=2.0)
        cgroup = None
        if self.launch.systemd_unit is not None:
            cgroup = probe_systemd_cgroup_evidence(self.launch.systemd_unit)
        # At this point the owned tree is fully reaped, so the lease may be
        # released even if evidence parsing or result construction fails.
        try:
            tail = self.output.tail()
            _, _, output_error = self.output.drain()
            classified = classify_exit(
                ExitEvidence(
                    returncode=returncode,
                    output_tail="".join(tail),
                    requested_cancel=requested_cancel,
                    watchdog=watchdog_outcome,
                    cgroup=cgroup,
                    limit_backend=self.launch.backend,
                )
            )
            result = SupervisedResult(
                returncode=returncode,
                classified_exit=classified,
                output_tail=tail,
                dropped_output_lines=self.output.dropped_lines,
                watchdog=watchdog_outcome,
                cgroup=cgroup,
                output_error=str(output_error) if output_error is not None else None,
                peak_accelerator_bytes=self.watchdog.peak_accelerator_bytes,
                accelerator_observation_error=(
                    self.watchdog.accelerator_observation_error
                ),
                peak_tree_rss_bytes=self.watchdog.peak_tree_rss_bytes,
                minimum_system_available_bytes=(
                    self.watchdog.minimum_system_available_bytes
                ),
            )
            if post_exit_check is not None:
                post_exit_check(result)
            return result
        finally:
            self._release_leases()

    def _terminate_and_reap(self, grace_seconds: float) -> bool:
        if self.tree is None or self.process is None:
            return False
        self.tree.terminate()
        deadline = time.monotonic() + grace_seconds
        while self.tree.is_alive() and time.monotonic() < deadline:
            time.sleep(0.02)
        if self.tree.is_alive():
            self.tree.kill()
        kill_deadline = time.monotonic() + 2.0
        while self.tree.is_alive() and time.monotonic() < kill_deadline:
            time.sleep(0.02)
        if self.process.poll() is None:
            try:
                self.process.wait(timeout=max(0.1, kill_deadline - time.monotonic()))
            except subprocess.TimeoutExpired:
                return False
        return not self.tree.is_alive() and not self.tree.ownership_uncertain

    def _finish_local_teardown(self) -> None:
        if self._guardian_started:
            if not self._complete_guardian_teardown():
                raise WorkloadStillOwnedError(
                    "guardian could not prove process-tree quiescence", self
                )
        else:
            # Guardian creation itself may have failed.  These descriptors and
            # the leases intentionally stayed with the returned recovery owner
            # until local cleanup could be retried and proved complete.
            self._close_unstarted_guardian_fds()
        if self._reader is not None and self._reader_started:
            self._reader.join(timeout=2.0)
        self._release_leases()

    def _complete_guardian_teardown(self, timeout: float = 5.0) -> bool:
        if not self._guardian_started:
            return False
        descriptor = self._parent_liveness_write_fd
        if not self._guardian_teardown_requested and descriptor is not None:
            self._guardian_teardown_requested = True
            try:
                os.write(descriptor, b"T")
            except OSError:
                pass
            try:
                _close_fork_excluded_fd(descriptor)
            except OSError:
                pass
            self._parent_liveness_write_fd = None
        acknowledgement = self._guardian_ack_read_fd
        if not self._guardian_ack_received:
            if acknowledgement is None:
                return False
            try:
                readable, _, _ = select.select([acknowledgement], [], [], timeout)
            except OSError:
                return False
            if not readable:
                return False
            try:
                self._guardian_ack_received = os.read(acknowledgement, 1) == b"Q"
            except OSError:
                return False
            if not self._guardian_ack_received:
                return False
            _close_fork_excluded_fd(acknowledgement)
            self._guardian_ack_read_fd = None
        if self._guardian_process is None:
            return False
        try:
            self._guardian_process.wait(timeout=2.0)
        except subprocess.TimeoutExpired:
            return False
        return self._guardian_process.returncode == 0

    def _close_unstarted_guardian_fds(self) -> None:
        for attribute in ("_parent_liveness_write_fd", "_guardian_ack_read_fd"):
            descriptor = getattr(self, attribute)
            if descriptor is not None:
                try:
                    os.close(descriptor)
                except OSError:
                    pass
                setattr(self, attribute, None)

    def _release_leases(self) -> None:
        if not self._leases_released:
            with _FORK_EXCLUDED_LOCK:
                self._leases.release()
                _FORK_EXCLUDED_FDS.difference_update(self._fork_excluded_fds)
            self._leases_released = True

    def _kill_and_reap_after_setup_failure(
        self, process: subprocess.Popen[Any]
    ) -> None:
        if process.poll() is None:
            if self.tree is not None:
                self.tree.kill()
            elif os.name == "posix":
                try:
                    group = os.getpgid(process.pid)
                    if group == process.pid:
                        os.killpg(group, signal.SIGKILL)
                    else:
                        process.kill()
                except (ProcessLookupError, PermissionError, OSError):
                    try:
                        process.kill()
                    except (ProcessLookupError, OSError):
                        pass
            else:  # pragma: no cover - Windows CI
                process.kill()
        try:
            process.wait(timeout=2.0)
        except subprocess.TimeoutExpired as exc:
            raise WorkloadStillOwnedError(
                "child survived supervisor-construction cleanup", self
            ) from exc
        if self.tree is not None and (
            self.tree.is_alive() or self.tree.ownership_uncertain
        ):
            raise WorkloadStillOwnedError(
                "descendant survived supervisor-construction cleanup", self
            )


def classify_exit(evidence: ExitEvidence) -> ClassifiedExit:
    """Map process evidence to a stable, user-facing failure category."""
    if evidence.admission_refused:
        return ClassifiedExit(ExitKind.HOST_ADMISSION_REFUSAL, "Host admission refused")
    if evidence.cgroup is not None and evidence.cgroup.oom_killed:
        return ClassifiedExit(
            ExitKind.HOST_HARD_LIMIT,
            "Worker was killed by its cgroup host-memory limit",
        )
    if evidence.watchdog is not None:
        if evidence.watchdog.trigger is WatchdogTrigger.SOFT_RSS:
            suffix = (
                " and exited cleanly"
                if evidence.watchdog.graceful_exit
                else "; it did not exit during the grace period and was killed"
            )
            return ClassifiedExit(
                ExitKind.HOST_SOFT_LIMIT,
                "Worker crossed its soft host-memory limit" + suffix,
            )
        if evidence.watchdog.trigger is WatchdogTrigger.PROCESS_COUNT:
            return ClassifiedExit(
                ExitKind.HOST_HARD_LIMIT,
                "Worker exceeded its bounded process-identity limit",
            )
        if evidence.watchdog.trigger is WatchdogTrigger.OBSERVATION_FAILURE:
            return ClassifiedExit(
                ExitKind.HOST_HARD_LIMIT,
                "Worker was killed because process-tree observation failed",
            )
        return ClassifiedExit(
            ExitKind.HOST_HARD_LIMIT,
            "Worker was killed to preserve the host memory reserve",
        )
    if evidence.requested_cancel:
        return ClassifiedExit(ExitKind.CANCELED, "Canceled by the user")
    # Successful commands may legitimately print historical/handled allocator
    # errors. Exit status is authoritative before heuristic tail scanning.
    if evidence.returncode == 0:
        return ClassifiedExit(ExitKind.SUCCESS, "Worker completed successfully")
    tail = evidence.output_tail.lower()
    if any(
        marker in tail
        for marker in (
            "cuda out of memory",
            "torch.cuda.outofmemoryerror",
            "mps backend out of memory",
            "mps out of memory",
            "mpsallocator out of memory",
        )
    ):
        return ClassifiedExit(
            ExitKind.ACCELERATOR_OOM, "Accelerator allocator reported out of memory"
        )
    if evidence.limit_backend is LimitBackend.RLIMIT_AS and "memoryerror" in tail:
        return ClassifiedExit(
            ExitKind.HOST_HARD_LIMIT,
            "Worker allocation was rejected by its RLIMIT_AS containment limit",
        )
    if evidence.returncode < 0:
        return ClassifiedExit(
            ExitKind.SIGNALLED,
            f"Worker terminated by signal {-evidence.returncode}",
        )
    return ClassifiedExit(
        ExitKind.ORDINARY_FAILURE,
        f"Worker exited with code {evidence.returncode}",
    )
