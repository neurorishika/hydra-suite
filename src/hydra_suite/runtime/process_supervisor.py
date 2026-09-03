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
from typing import Any, Optional

import psutil

from .resource_limits import (
    CgroupEvidence,
    LimitBackend,
    LimitedLaunch,
    probe_systemd_cgroup_evidence,
)


class WatchdogTrigger(str, Enum):
    SOFT_RSS = "host-soft-rss"
    HARD_RSS = "host-hard-rss"
    SYSTEM_RESERVE = "host-system-reserve"


@dataclass(frozen=True)
class WatchdogPolicy:
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
class WatchdogOutcome:
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

    def __init__(self, process: subprocess.Popen[Any], *, owns_process_group: bool):
        root = psutil.Process(process.pid)
        self.process = process
        self.root = _ProcessIdentity(process.pid, root.create_time())
        self._known_identities: dict[int, _ProcessIdentity] = {self.root.pid: self.root}
        self.process_group_id: Optional[int] = None
        if owns_process_group and os.name == "posix":
            group = os.getpgid(process.pid)
            if group != process.pid:
                raise ValueError(
                    "owned process groups must be created with start_new_session=True"
                )
            self.process_group_id = group

    def identities(self) -> tuple[_ProcessIdentity, ...]:
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
        elif self.process_group_id is not None:
            # A launcher wrapper may exit before its workload.  Recover the
            # reparented members by the dedicated session/process-group ID.
            try:
                for process in psutil.process_iter(["pid", "create_time"]):
                    if os.getpgid(process.pid) == self.process_group_id:
                        self._known_identities[process.pid] = _ProcessIdentity(
                            process.pid, process.create_time()
                        )
            except (psutil.Error, PermissionError, OSError):
                # Sandboxed macOS processes may be denied the global PID list.
                # The dedicated process group remains signalable even when RSS
                # for a reparented descendant cannot be attributed precisely.
                pass
        return tuple(
            identity
            for identity in self._known_identities.values()
            if identity.resolve() is not None
        )

    def is_alive(self) -> bool:
        # Reap a completed root before checking captured descendants.  Without
        # this poll a cooperative child can remain visible as a zombie until
        # the grace period expires and be mislabeled as an unresponsive job.
        self.process.poll()
        if any(identity.resolve() is not None for identity in self.identities()):
            return True
        if self.process_group_id is not None:
            try:
                os.killpg(self.process_group_id, 0)
                return True
            except (ProcessLookupError, PermissionError, OSError):
                return False
        return False

    def rss_bytes(self) -> int:
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
        self._signal(signal.SIGTERM)

    def kill(self) -> None:
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
        if self.process_group_id is not None:
            # Construction proved this was a fresh session whose group ID was
            # the captured root PID.  It remains the owned boundary even if a
            # short-lived launcher wrapper exits before its descendants.
            try:
                os.killpg(self.process_group_id, signum)
                return
            except (ProcessLookupError, PermissionError, OSError):
                pass
        # Descendants first prevents the root from orphaning them before they
        # are captured.  Every PID is revalidated against its creation time.
        for identity in reversed(identities):
            process = identity.resolve()
            if process is None:
                continue
            try:
                process.send_signal(signum)
            except (psutil.Error, OSError):
                continue


class ProcessTreeWatchdog:
    """Monitor independently of child output and training-step progress."""

    def __init__(self, tree: OwnedProcessTree, policy: WatchdogPolicy) -> None:
        self.tree = tree
        self.policy = policy
        self.outcome: Optional[WatchdogOutcome] = None
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        if self._thread is not None:
            raise RuntimeError("watchdog has already been started")
        self._thread = threading.Thread(
            target=self._run,
            name=f"hydra-memory-watchdog-{self.tree.root.pid}",
            daemon=True,
        )
        self._thread.start()

    def stop(self, timeout: Optional[float] = None) -> Optional[WatchdogOutcome]:
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
        with self._condition:
            return self._dropped_lines

    @property
    def retained_chars(self) -> int:
        with self._condition:
            return self._chars

    def append(self, line: str) -> None:
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
    returncode: int
    output_tail: str = ""
    requested_cancel: bool = False
    admission_refused: bool = False
    watchdog: Optional[WatchdogOutcome] = None
    cgroup: Optional[CgroupEvidence] = None
    limit_backend: Optional[LimitBackend] = None


@dataclass(frozen=True)
class ClassifiedExit:
    kind: ExitKind
    message: str


@dataclass(frozen=True)
class SupervisedResult:
    returncode: int
    classified_exit: ClassifiedExit
    output_tail: tuple[str, ...]
    dropped_output_lines: int
    watchdog: Optional[WatchdogOutcome]
    cgroup: Optional[CgroupEvidence]
    output_error: Optional[str]


class SupervisedSidecar:
    """Own one limited child, its output pump, and its memory watchdog."""

    def __init__(
        self,
        launch: LimitedLaunch,
        watchdog_policy: WatchdogPolicy,
        *,
        output_max_lines: int = 512,
        output_max_chars: int = 256 * 1024,
    ) -> None:
        popen_kwargs: dict[str, Any] = {
            "env": dict(launch.environment),
            "stdout": subprocess.PIPE,
            "stderr": subprocess.STDOUT,
            "text": True,
            "errors": "replace",
            "bufsize": 1,
        }
        if os.name == "posix":
            popen_kwargs["start_new_session"] = True
        self.launch = launch
        self.process = subprocess.Popen(launch.command, **popen_kwargs)
        self.tree = OwnedProcessTree(
            self.process, owns_process_group=os.name == "posix"
        )
        self.output = BoundedLineBuffer(
            max_lines=output_max_lines, max_chars=output_max_chars
        )
        assert self.process.stdout is not None
        self._reader = threading.Thread(
            target=pump_stdout,
            args=(self.process.stdout, self.output),
            name=f"hydra-output-pump-{self.process.pid}",
            daemon=True,
        )
        self.watchdog = ProcessTreeWatchdog(self.tree, watchdog_policy)
        self._reader.start()
        self.watchdog.start()

    def cancel(self, grace_seconds: float = 5.0) -> None:
        """Cancel the complete owned process group, escalating after a grace."""
        self.tree.terminate()
        deadline = time.monotonic() + grace_seconds
        while self.tree.is_alive() and time.monotonic() < deadline:
            time.sleep(0.02)
        if self.tree.is_alive():
            self.tree.kill()
        try:
            self.process.wait(timeout=max(1.0, grace_seconds))
        except subprocess.TimeoutExpired:
            self.tree.kill()
            self.process.wait(timeout=2.0)

    def wait(
        self,
        timeout: Optional[float] = None,
        *,
        requested_cancel: bool = False,
    ) -> SupervisedResult:
        """Wait for completion and return bounded output plus classified evidence."""
        deadline = None if timeout is None else time.monotonic() + timeout
        while self.tree.is_alive():
            if deadline is not None and time.monotonic() >= deadline:
                raise subprocess.TimeoutExpired(self.launch.command, timeout)
            time.sleep(0.02)
        returncode = self.process.wait(timeout=1.0)
        watchdog_outcome = self.watchdog.stop(timeout=2.0)
        self._reader.join(timeout=2.0)
        cgroup = None
        if self.launch.systemd_unit is not None:
            try:
                cgroup = probe_systemd_cgroup_evidence(self.launch.systemd_unit)
            except (OSError, subprocess.SubprocessError):
                cgroup = CgroupEvidence(unit=self.launch.systemd_unit)
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
    tail = evidence.output_tail.lower()
    if any(
        marker in tail
        for marker in (
            "cuda out of memory",
            "torch.cuda.outofmemoryerror",
            "mps backend out of memory",
            "mps allocated",
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
    if evidence.returncode == 0:
        return ClassifiedExit(ExitKind.SUCCESS, "Worker completed successfully")
    if evidence.returncode < 0:
        return ClassifiedExit(
            ExitKind.SIGNALLED,
            f"Worker terminated by signal {-evidence.returncode}",
        )
    return ClassifiedExit(
        ExitKind.ORDINARY_FAILURE,
        f"Worker exited with code {evidence.returncode}",
    )
