"""Minimal executable that installs child limits before loading the workload."""

from __future__ import annotations

import argparse
import ctypes
import errno
import os
import platform
import select
import signal
import sys
import time
from collections.abc import Sequence
from typing import Any, Callable

from .resource_limits import apply_child_limits

_SUPERVISOR_PID_ENV = "HYDRA_SUPERVISOR_PID"
_PARENT_LIVENESS_FD_ENV = "HYDRA_PARENT_LIVENESS_FD"
_PARENT_LEASE_FDS_ENV = "HYDRA_PARENT_LEASE_FDS"
_SYSTEMD_UNIT_ENV = "HYDRA_PARENT_SYSTEMD_UNIT"
_MAX_IDENTITIES_ENV = "HYDRA_PARENT_MAX_IDENTITIES"
_PR_SET_PDEATHSIG = 1


def install_linux_parent_death_signal(
    expected_parent_pid: int,
    *,
    system: str | None = None,
    prctl_call: Callable[[int, int], int] | None = None,
    get_parent_pid: Callable[[], int] | None = None,
) -> bool:
    """Arm ``PDEATHSIG`` and close the setup race before workload imports."""
    if (system or platform.system()) != "Linux":
        return False
    if expected_parent_pid <= 0:
        raise ValueError("expected parent PID must be positive")
    if prctl_call is None:
        libc = ctypes.CDLL(None, use_errno=True)
        prctl_call = libc.prctl
    result = prctl_call(_PR_SET_PDEATHSIG, int(signal.SIGKILL))
    if result != 0:
        error_number = ctypes.get_errno() or errno.EINVAL
        raise OSError(error_number, "could not install Linux parent-death signal")
    parent_pid = (get_parent_pid or os.getppid)()
    if int(parent_pid) != int(expected_parent_pid):
        raise RuntimeError("supervisor parent exited during child bootstrap")
    return True


def arm_parent_liveness_proxy(
    read_fd: int,
    *,
    lease_fds: Sequence[int] = (),
    systemd_unit: str | None = None,
    max_identities: int = 512,
) -> int:
    """Fork a lease-holding guardian that tears down the boundary on EOF."""
    if os.name != "posix":
        raise RuntimeError("parent-liveness proxy requires POSIX file descriptors")
    if read_fd < 0:
        raise ValueError("parent-liveness descriptor must be non-negative")
    if any(descriptor < 0 for descriptor in lease_fds):
        raise ValueError("lease descriptors must be non-negative")
    if max_identities < 1:
        raise ValueError("maximum guardian identity count must be positive")
    owned_process_group = os.getpgrp()
    workload_pid = os.getpid()
    ready_read_fd, ready_write_fd = os.pipe()
    guardian_pid = os.fork()
    if guardian_pid:
        os.close(ready_write_fd)
        try:
            ready = os.read(ready_read_fd, 1)
        finally:
            os.close(ready_read_fd)
            os.close(read_fd)
            for descriptor in lease_fds:
                os.close(descriptor)
        if ready != b"R":
            raise RuntimeError("parent-liveness guardian failed to initialize")
        return guardian_pid
    os.close(ready_read_fd)
    try:  # pragma: no cover - behavior is exercised through SupervisedSidecar
        try:
            os.setsid()
        except OSError:
            # Remaining outside the workload group is required so the guardian
            # can kill escapees after signalling that group.
            os._exit(125)
        root_create_time = _read_process_create_time(workload_pid)
        if root_create_time is None:
            os._exit(125)
        captured: dict[int, float] = {workload_pid: root_create_time}
        os.write(ready_write_fd, b"R")
        os.close(ready_write_fd)
        enumeration_succeeded = False
        normal_shutdown = False
        while True:
            captured_now, overflowed = _capture_descendant_identities(
                workload_pid,
                root_create_time,
                captured,
                max_identities=max_identities,
            )
            enumeration_succeeded = captured_now or enumeration_succeeded
            if overflowed:
                if systemd_unit is not None:
                    _guard_systemd_scope(systemd_unit, captured)
                else:
                    _kill_captured_escapees(captured, owned_process_group)
                    try:
                        os.killpg(owned_process_group, signal.SIGKILL)
                    except OSError:
                        pass
                    _kill_captured_identities(captured)
                os._exit(0)
            try:
                readable, _, _ = select.select([read_fd], [], [], 0.1)
            except OSError:
                readable = [read_fd]
            if not readable:
                continue
            try:
                message = os.read(read_fd, 1)
            except OSError:
                message = b""
            normal_shutdown = bool(message)
            break
        if normal_shutdown:
            os._exit(0)
        # Close the sample-to-EOF race: descendants may fork or call setsid
        # while the guardian is blocked in select(). Capture once more after
        # observing supervisor death and immediately before teardown.
        captured_now, _ = _capture_descendant_identities(
            workload_pid,
            root_create_time,
            captured,
            max_identities=max_identities,
        )
        enumeration_succeeded = captured_now or enumeration_succeeded
        if systemd_unit is not None:
            _guard_systemd_scope(systemd_unit, captured)
        else:
            _kill_captured_escapees(captured, owned_process_group)
            try:
                os.killpg(owned_process_group, signal.SIGKILL)
            except OSError:
                pass
            _kill_captured_identities(captured)
            if lease_fds and not enumeration_succeeded:
                # The platform denied descendant enumeration, so an escaped
                # process cannot be disproved. Retain every resource lock.
                while True:
                    time.sleep(60)
    finally:
        try:
            os.close(ready_write_fd)
        except OSError:
            pass
        try:
            os.close(read_fd)
        except OSError:
            pass
    os._exit(0)


def _capture_descendant_identities(
    workload_pid: int,
    workload_create_time: float,
    captured: dict[int, float],
    *,
    max_identities: int,
) -> tuple[bool, bool]:
    import psutil

    for pid, create_time in tuple(captured.items()):
        if _resolve_captured_identity(pid, create_time) is None:
            captured.pop(pid, None)
    try:
        root = psutil.Process(workload_pid)
        if abs(float(root.create_time()) - workload_create_time) >= 0.01:
            return True, False
        processes = [root, *root.children(recursive=True)]
        overflowed = False
        for process in processes:
            if process.pid != os.getpid():
                create_time = float(process.create_time())
                if captured.get(process.pid) == create_time:
                    continue
                if len(captured) >= max_identities:
                    # Do not retain an unbounded registry. Kill every excess
                    # identity immediately in case it already escaped the
                    # workload's process group.
                    overflowed = True
                    try:
                        process.kill()
                    except (psutil.Error, OSError, RuntimeError):
                        # The caller still tears down the authoritative group
                        # or cgroup when a direct identity signal is denied.
                        pass
                    continue
                captured[process.pid] = create_time
        return True, overflowed
    except (psutil.Error, OSError, RuntimeError):
        return False, False


def _read_process_create_time(pid: int) -> float | None:
    import psutil

    try:
        return float(psutil.Process(pid).create_time())
    except (psutil.Error, OSError):
        return None


def _resolve_captured_identity(pid: int, create_time: float) -> Any:
    import psutil

    try:
        process = psutil.Process(pid)
        if abs(float(process.create_time()) - create_time) >= 0.01:
            return None
        if process.status() == psutil.STATUS_ZOMBIE:
            return None
        return process
    except (psutil.Error, OSError):
        return None


def _kill_captured_escapees(
    captured: dict[int, float], owned_process_group: int
) -> None:
    import psutil

    for pid, create_time in tuple(captured.items()):
        process = _resolve_captured_identity(pid, create_time)
        if process is None:
            continue
        try:
            if os.getpgid(pid) != owned_process_group:
                process.kill()
        except (psutil.Error, OSError, RuntimeError):
            continue


def _kill_captured_identities(captured: dict[int, float]) -> None:
    import psutil

    for pid, create_time in tuple(captured.items()):
        process = _resolve_captured_identity(pid, create_time)
        if process is None:
            continue
        try:
            process.kill()
        except (psutil.Error, OSError, RuntimeError):
            continue


def _guard_systemd_scope(unit: str, captured: dict[int, float]) -> None:
    from .resource_limits import signal_systemd_scope

    while True:
        if signal_systemd_scope(unit, int(signal.SIGKILL)):
            return
        _kill_captured_outside_scope(unit, captured)
        # The unit is the authoritative ownership boundary. A failed signal
        # cannot prove it empty, so this guardian intentionally retains its
        # inherited leases and retries until systemd accepts the request.
        time.sleep(0.25)


def _kill_captured_outside_scope(unit: str, captured: dict[int, float]) -> None:
    import psutil

    for pid, create_time in tuple(captured.items()):
        process = _resolve_captured_identity(pid, create_time)
        if process is None:
            continue
        try:
            with open(f"/proc/{pid}/cgroup", encoding="utf-8") as cgroup_file:
                cgroup = cgroup_file.read()
        except OSError:
            continue
        if any(unit in line.partition("/")[2] for line in cgroup.splitlines()):
            continue
        try:
            process.kill()
        except (psutil.Error, OSError, RuntimeError):
            continue


def build_parser() -> argparse.ArgumentParser:
    """Build the minimal bootstrap command-line parser."""

    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument("--address-space-bytes", type=int)
    parser.add_argument("--mps-high-watermark-ratio", type=float)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Apply parent-death and memory guards, then replace this process."""

    args = build_parser().parse_args(argv)
    command = list(args.command)
    if command and command[0] == "--":
        command.pop(0)
    if not command:
        raise SystemExit("a child command is required after --")
    expected_parent = os.environ.pop(_SUPERVISOR_PID_ENV, None)
    if expected_parent is not None:
        install_linux_parent_death_signal(int(expected_parent))
    liveness_fd = os.environ.pop(_PARENT_LIVENESS_FD_ENV, None)
    if liveness_fd is not None:
        lease_fds = tuple(
            int(value)
            for value in os.environ.pop(_PARENT_LEASE_FDS_ENV, "").split(",")
            if value
        )
        systemd_unit = os.environ.pop(_SYSTEMD_UNIT_ENV, None)
        max_identities = int(os.environ.pop(_MAX_IDENTITIES_ENV, "512"))
        arm_parent_liveness_proxy(
            int(liveness_fd),
            lease_fds=lease_fds,
            systemd_unit=systemd_unit,
            max_identities=max_identities,
        )
    apply_child_limits(
        address_space_bytes=args.address_space_bytes,
        mps_high_watermark_ratio=args.mps_high_watermark_ratio,
    )
    os.execvpe(command[0], command, os.environ)
    return 127  # pragma: no cover - exec replaces the process


if __name__ == "__main__":  # pragma: no cover - exercised as a subprocess
    sys.exit(main())
