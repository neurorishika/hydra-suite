"""Minimal executable that installs child limits before loading the workload."""

from __future__ import annotations

import argparse
import ctypes
import errno
import os
import platform
import signal
import sys
from collections.abc import Sequence
from typing import Callable

from .resource_limits import apply_child_limits

_SUPERVISOR_PID_ENV = "HYDRA_SUPERVISOR_PID"
_PARENT_LIVENESS_FD_ENV = "HYDRA_PARENT_LIVENESS_FD"
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


def arm_parent_liveness_proxy(read_fd: int) -> int:
    """Fork a minimal POSIX guardian that kills the child group on parent EOF."""
    if os.name != "posix":
        raise RuntimeError("parent-liveness proxy requires POSIX file descriptors")
    if read_fd < 0:
        raise ValueError("parent-liveness descriptor must be non-negative")
    guardian_pid = os.fork()
    if guardian_pid:
        os.close(read_fd)
        return guardian_pid
    try:  # pragma: no cover - behavior is exercised through SupervisedSidecar
        try:
            while os.read(read_fd, 1):
                pass
        except OSError:
            pass
        try:
            os.killpg(os.getpgrp(), signal.SIGKILL)
        except OSError:
            os._exit(0)
    finally:
        try:
            os.close(read_fd)
        except OSError:
            pass
    os._exit(0)


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
        arm_parent_liveness_proxy(int(liveness_fd))
    apply_child_limits(
        address_space_bytes=args.address_space_bytes,
        mps_high_watermark_ratio=args.mps_high_watermark_ratio,
    )
    os.execvpe(command[0], command, os.environ)
    return 127  # pragma: no cover - exec replaces the process


if __name__ == "__main__":  # pragma: no cover - exercised as a subprocess
    sys.exit(main())
