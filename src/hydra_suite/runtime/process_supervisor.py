"""Process-tree memory watchdog, bounded output, and exit classification."""

from __future__ import annotations

import os
import signal
import subprocess
import threading
import time
from collections import deque
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Optional

import psutil

from .resource_lease import HeavyJobLease
from .resource_limits import (
    CgroupEvidence,
    LimitBackend,
    LimitedLaunch,
    probe_systemd_cgroup_evidence,
    signal_systemd_scope,
)


class WatchdogTrigger(str, Enum):
    """Reason the parent watchdog intervened in a workload."""

    SOFT_RSS = "host-soft-rss"
    HARD_RSS = "host-hard-rss"
    SYSTEM_RESERVE = "host-system-reserve"


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
    minimum_system_available_bytes: int
    poll_interval_seconds: float = 0.25
    terminate_grace_seconds: float = 5.0

    def __post_init__(self) -> None:
        # Constructing the derived policy here validates every timing/floor
        # value before a child can be created.
        self.watchdog_policy

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

    def resolve(self) -> Optional[psutil.Process]:
        """Resolve this PID only when its creation time still matches."""

        try:
            process = psutil.Process(self.pid)
            if abs(process.create_time() - self.create_time) >= 0.01:
                return None
            if process.status() == psutil.STATUS_ZOMBIE:
                return None
            return process
        except (psutil.Error, OSError):
            return None


class OwnedProcessTree:
    """Signal only a captured process identity or its dedicated process group."""

    def __init__(
        self,
        process: subprocess.Popen[Any],
        *,
        owns_process_group: bool,
        systemd_unit: Optional[str] = None,
    ):
        root = psutil.Process(process.pid)
        self.process = process
        self.root = _ProcessIdentity(process.pid, root.create_time())
        self._known_identities: dict[int, _ProcessIdentity] = {self.root.pid: self.root}
        self.process_group_id: Optional[int] = None
        self.systemd_unit = systemd_unit
        if owns_process_group and os.name == "posix":
            group = os.getpgid(process.pid)
            if group != process.pid:
                raise ValueError(
                    "owned process groups must be created with start_new_session=True"
                )
            self.process_group_id = group

    def identities(self) -> tuple[_ProcessIdentity, ...]:
        """Capture and return every still-live, identity-validated process."""

        root = self.root.resolve()
        if root is not None:
            processes = [root]
            try:
                processes.extend(root.children(recursive=True))
            except (psutil.Error, OSError):
                pass
            for process in processes:
                try:
                    self._known_identities[process.pid] = _ProcessIdentity(
                        process.pid, process.create_time()
                    )
                except (psutil.Error, OSError):
                    continue
        return tuple(
            identity
            for identity in self._known_identities.values()
            if identity.resolve() is not None
        )

    def is_alive(self) -> bool:
        """Return whether any captured member of the owned tree is live."""

        # Reap a completed root before checking captured descendants.  Without
        # this poll a cooperative child can remain visible as a zombie until
        # the grace period expires and be mislabeled as an unresponsive job.
        self.process.poll()
        if any(identity.resolve() is not None for identity in self.identities()):
            return True
        if self.process_group_id is not None:
            return any(
                self._identity_process_group(identity) == self.process_group_id
                for identity in self._known_identities.values()
            )
        return False

    def rss_bytes(self) -> int:
        """Return current resident bytes across all observable owned members."""

        total = 0
        for identity in self.identities():
            process = identity.resolve()
            if process is None:
                continue
            try:
                total += process.memory_info().rss
            except (psutil.Error, OSError):
                continue
        return total

    def terminate(self) -> None:
        """Request graceful termination of the owned boundary."""

        self._signal(signal.SIGTERM)

    def kill(self) -> None:
        """Forcefully terminate the owned boundary."""

        if os.name == "posix":
            self._signal(signal.SIGKILL)
            return
        for identity in reversed(self.identities()):  # pragma: no cover - Windows CI
            process = identity.resolve()
            if process is None:
                continue
            try:
                process.kill()
            except (psutil.Error, OSError):
                continue

    def _signal(self, signum: signal.Signals) -> None:
        identities = self.identities()
        group_signalled = False
        if self.systemd_unit is not None:
            # systemd is authoritative for every process that remains in the
            # cgroup, including descendants invisible to psutil. Do not mix
            # direct PID/group signals into this ownership mode.
            signal_systemd_scope(self.systemd_unit, int(signum))
            return
        if self.process_group_id is not None and any(
            self._identity_process_group(identity) == self.process_group_id
            for identity in identities
        ):
            # Signal a numeric group only while an identity-validated member
            # still proves that the group is ours. This prevents PID/PGID reuse.
            try:
                os.killpg(self.process_group_id, signum)
                group_signalled = True
            except (ProcessLookupError, PermissionError, OSError):
                pass
        # Also signal every captured descendant that escaped the owned group or
        # cgroup. Descendants go first and every PID is creation-time validated.
        for identity in reversed(identities):
            if group_signalled and (
                self._identity_process_group(identity) == self.process_group_id
            ):
                continue
            process = identity.resolve()
            if process is None:
                continue
            try:
                process.send_signal(signum)
            except (psutil.Error, OSError):
                continue

    @staticmethod
    def _identity_process_group(identity: _ProcessIdentity) -> Optional[int]:
        process = identity.resolve()
        if process is None or os.name != "posix":
            return None
        try:
            return os.getpgid(process.pid)
        except (ProcessLookupError, PermissionError, OSError):
            return None


class ProcessTreeWatchdog:
    """Monitor independently of child output and training-step progress."""

    def __init__(self, tree: OwnedProcessTree, policy: WatchdogPolicy) -> None:
        self.tree = tree
        self.policy = policy
        self.outcome: Optional[WatchdogOutcome] = None
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
        while not self._stop.is_set() and self.tree.is_alive():
            rss = self.tree.rss_bytes()
            available = int(psutil.virtual_memory().available)
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


def pump_stdout(stdout: Any, output: BoundedLineBuffer) -> None:
    """Read a pipe into a bounded buffer; safe for a daemon reader thread."""
    try:
        for line in stdout:
            output.append(str(line))
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


class WorkloadStillOwnedError(RuntimeError):
    """Teardown could not prove exit; the sidecar intentionally retains ownership."""

    def __init__(self, message: str, sidecar: "SupervisedSidecar") -> None:
        self.sidecar = sidecar
        super().__init__(message)


class SupervisedSidecar:
    """Own one limited child, its output pump, and its memory watchdog."""

    def __init__(
        self,
        plan: ContainmentPlan,
        *,
        lease: Optional[HeavyJobLease] = None,
        prelaunch_check: Optional[Callable[[], None]] = None,
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
        self._lease = lease
        self._lease_released = lease is None
        self._parent_liveness_write_fd: Optional[int] = None
        self.process: subprocess.Popen[Any]
        self.tree: OwnedProcessTree
        self._reader: threading.Thread
        self.watchdog: ProcessTreeWatchdog

        if lease is not None and lease.owner is not None:
            raise ValueError("the supervisor must acquire its lease itself")
        if prelaunch_check is not None and not callable(prelaunch_check):
            raise TypeError("prelaunch_check must be callable")

        child_env = dict(self.launch.environment)
        pass_fds: tuple[int, ...] = ()
        parent_read_fd: Optional[int] = None
        if self.launch.backend is not LimitBackend.SYSTEMD_CGROUP:
            child_env["HYDRA_SUPERVISOR_PID"] = str(os.getpid())
        if self.launch.backend is LimitBackend.WATCHDOG_ONLY and os.name == "posix":
            parent_read_fd, parent_write_fd = os.pipe()
            os.set_inheritable(parent_read_fd, True)
            child_env["HYDRA_PARENT_LIVENESS_FD"] = str(parent_read_fd)
            pass_fds = (parent_read_fd,)
            self._parent_liveness_write_fd = parent_write_fd

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
            if lease is not None:
                lease.acquire()
                self._lease_released = False
            if prelaunch_check is not None:
                prelaunch_check()
            process = subprocess.Popen(self.launch.command, **popen_kwargs)
            self.process = process
            if parent_read_fd is not None:
                os.close(parent_read_fd)
                parent_read_fd = None
            self.tree = OwnedProcessTree(
                self.process,
                owns_process_group=os.name == "posix",
                systemd_unit=self.launch.systemd_unit,
            )
            if self.process.stdout is None:
                raise RuntimeError("supervised child stdout pipe was not created")
            self._reader = threading.Thread(
                target=pump_stdout,
                args=(self.process.stdout, self.output),
                name=f"hydra-output-pump-{self.process.pid}",
                daemon=True,
            )
            self.watchdog = ProcessTreeWatchdog(self.tree, watchdog_policy)
            self._reader.start()
            self.watchdog.start()
        except BaseException:
            if parent_read_fd is not None:
                os.close(parent_read_fd)
            self._close_parent_liveness_pipe()
            if process is not None:
                self._kill_and_reap_after_setup_failure(process)
            self._release_lease()
            raise

    def cancel(self, grace_seconds: float = 5.0) -> None:
        """Cancel the complete owned process group, escalating after a grace."""
        if grace_seconds < 0:
            raise ValueError("cancel grace must be non-negative")
        self.watchdog.stop(timeout=1.0)
        if not self._terminate_and_reap(grace_seconds):
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
    ) -> SupervisedResult:
        """Wait for completion and return bounded output plus classified evidence."""
        if timeout is not None and timeout < 0:
            raise ValueError("wait timeout must be non-negative")
        deadline = None if timeout is None else time.monotonic() + timeout
        while self.tree.is_alive():
            if self.process.poll() is not None:
                self.watchdog.stop(timeout=1.0)
                self._close_parent_liveness_pipe()
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
        self._close_parent_liveness_pipe()
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
            return SupervisedResult(
                returncode=returncode,
                classified_exit=classified,
                output_tail=tail,
                dropped_output_lines=self.output.dropped_lines,
                watchdog=watchdog_outcome,
                cgroup=cgroup,
                output_error=str(output_error) if output_error is not None else None,
            )
        finally:
            self._release_lease()

    def _terminate_and_reap(self, grace_seconds: float) -> bool:
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
        return not self.tree.is_alive()

    def _finish_local_teardown(self) -> None:
        self._close_parent_liveness_pipe()
        self._reader.join(timeout=2.0)
        self._release_lease()

    def _close_parent_liveness_pipe(self) -> None:
        descriptor = self._parent_liveness_write_fd
        if descriptor is None:
            return
        self._parent_liveness_write_fd = None
        try:
            os.close(descriptor)
        except OSError:
            pass

    def _release_lease(self) -> None:
        if not self._lease_released and self._lease is not None:
            self._lease.release()
            self._lease_released = True

    def _kill_and_reap_after_setup_failure(
        self, process: subprocess.Popen[Any]
    ) -> None:
        if process.poll() is None:
            if hasattr(self, "tree"):
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
        if hasattr(self, "tree") and self.tree.is_alive():
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
