"""Out-of-scope guardian for supervised high-memory process trees."""

from __future__ import annotations

import os
import select
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from typing import Mapping, Optional, Sequence

import psutil

from .resource_limits import (
    cgroup_path_contains_unit,
    signal_systemd_scope,
    systemd_scope_is_quiescent,
)

_TOKEN_ENV = "HYDRA_CONTAINMENT_TOKEN"


@dataclass(frozen=True)
class GuardedIdentity:
    """PID plus creation time, safe against numeric PID reuse."""

    pid: int
    create_time: float

    def resolve(self) -> tuple[Optional[psutil.Process], bool]:
        """Return ``(process, gone)``; unknown identities are never called gone."""

        try:
            process = psutil.Process(self.pid)
            if abs(float(process.create_time()) - self.create_time) >= 0.01:
                return None, True
            if process.status() == psutil.STATUS_ZOMBIE:
                return None, True
            return process, False
        except (psutil.NoSuchProcess, psutil.ZombieProcess):
            return None, True
        except (psutil.AccessDenied, OSError):
            return None, False


def spawn_parent_guardian(
    *,
    workload_pid: int,
    process_group_id: int,
    liveness_read_fd: int,
    acknowledgement_write_fd: int,
    containment_token: str,
    max_identities: int,
    systemd_unit: Optional[str],
    lease_fds: Sequence[int] = (),
    environment: Optional[Mapping[str, str]] = None,
) -> subprocess.Popen[bytes]:
    """Spawn a guardian outside the workload cgroup and wait for readiness."""

    if os.name != "posix":
        raise RuntimeError("parent guardian requires POSIX process primitives")
    if not containment_token:
        raise ValueError("containment token must not be empty")
    if max_identities < 1:
        raise ValueError("maximum guardian identity count must be positive")
    ready_read_fd, ready_write_fd = os.pipe()
    command = [
        sys.executable,
        "-m",
        "hydra_suite.runtime.process_guardian",
        "--workload-pid",
        str(workload_pid),
        "--process-group-id",
        str(process_group_id),
        "--liveness-fd",
        str(liveness_read_fd),
        "--acknowledgement-fd",
        str(acknowledgement_write_fd),
        "--ready-fd",
        str(ready_write_fd),
        "--containment-token",
        containment_token,
        "--max-identities",
        str(max_identities),
    ]
    if systemd_unit is not None:
        command.extend(["--systemd-unit", systemd_unit])
    try:
        guardian = subprocess.Popen(
            command,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            pass_fds=(
                liveness_read_fd,
                acknowledgement_write_fd,
                ready_write_fd,
                *lease_fds,
            ),
            start_new_session=True,
            env=None if environment is None else dict(environment),
        )
    except BaseException:
        os.close(ready_read_fd)
        os.close(ready_write_fd)
        raise
    os.close(ready_write_fd)
    try:
        readable, _, _ = select.select([ready_read_fd], [], [], 5.0)
        ready = os.read(ready_read_fd, 1) if readable else b""
    finally:
        os.close(ready_read_fd)
        os.close(liveness_read_fd)
        os.close(acknowledgement_write_fd)
    if ready != b"R":
        try:
            guardian.terminate()
            guardian.wait(timeout=2.0)
        except (ProcessLookupError, subprocess.TimeoutExpired):
            guardian.kill()
            guardian.wait(timeout=2.0)
        raise RuntimeError(
            "cannot prove process ownership: launch-scoped identity scan "
            "is unavailable"
        )
    return guardian


def run_guardian(
    *,
    workload_pid: int,
    process_group_id: int,
    liveness_read_fd: int,
    acknowledgement_write_fd: int,
    ready_write_fd: int,
    containment_token: str,
    max_identities: int,
    systemd_unit: Optional[str],
) -> int:
    """Run the guardian protocol in its dedicated out-of-scope process."""

    try:  # pragma: no cover - exercised through subprocess integration tests
        identities: dict[int, GuardedIdentity] = {}
        scan_ok, _ = _scan_token_identities(
            containment_token,
            identities,
            max_identities=max_identities,
        )
        if not scan_ok:
            os.write(ready_write_fd, b"E")
            return 125
        os.write(ready_write_fd, b"R")
        os.close(ready_write_fd)
        while True:
            readable, _, _ = select.select([liveness_read_fd], [], [], 0.05)
            if readable:
                try:
                    request = os.read(liveness_read_fd, 1)
                except OSError:
                    request = b""
                break
            _scan_token_identities(
                containment_token,
                identities,
                max_identities=max_identities,
            )

        _terminate_until_quiescent(
            containment_token=containment_token,
            identities=identities,
            max_identities=max_identities,
            process_group_id=process_group_id,
            systemd_unit=systemd_unit,
        )
        if request == b"T":
            try:
                os.write(acknowledgement_write_fd, b"Q")
            except OSError:
                pass
    finally:
        for descriptor in (
            ready_write_fd,
            liveness_read_fd,
            acknowledgement_write_fd,
        ):
            try:
                os.close(descriptor)
            except OSError:
                pass
    return 0


def _scan_token_identities(
    token: str,
    identities: dict[int, GuardedIdentity],
    *,
    max_identities: int,
) -> tuple[bool, bool]:
    """Capture token-bearing same-user processes without trusting ancestry."""

    _prune_gone_identities(identities)
    overflowed = False
    try:
        processes = psutil.process_iter(["pid", "uids"])
        for process in processes:
            if process.pid == os.getpid():
                continue
            try:
                uids = process.info.get("uids")
                if uids is not None and uids.real != os.getuid():
                    continue
                if process.environ().get(_TOKEN_ENV) != token:
                    continue
                identity = GuardedIdentity(process.pid, float(process.create_time()))
            except (psutil.NoSuchProcess, psutil.ZombieProcess):
                continue
            except (psutil.AccessDenied, OSError):
                # A same-user environment that cannot be inspected means the
                # token absence cannot be proved.
                return False, overflowed
            if identities.get(identity.pid) == identity:
                continue
            if len(identities) >= max_identities:
                overflowed = True
                # Keep the first over-cap identity in a dedicated bounded
                # overflow slot (max + one) until it is proven gone. Returning
                # an incomplete scan prevents any quiescence acknowledgement.
                identities[identity.pid] = identity
                return False, True
            identities[identity.pid] = identity
        return True, overflowed
    except (psutil.Error, OSError, RuntimeError):
        return False, overflowed


def _prune_gone_identities(identities: dict[int, GuardedIdentity]) -> None:
    for pid, identity in tuple(identities.items()):
        _, gone = identity.resolve()
        if gone:
            identities.pop(pid, None)


def _signal_identity(identity: GuardedIdentity, signum: signal.Signals) -> bool:
    process, gone = identity.resolve()
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


def _terminate_until_quiescent(
    *,
    containment_token: str,
    identities: dict[int, GuardedIdentity],
    max_identities: int,
    process_group_id: int,
    systemd_unit: Optional[str],
) -> None:
    """Retain inherited leases until the complete owned boundary is empty."""

    stable_empty_scans = 0
    permanent_ownership_uncertain = False
    while stable_empty_scans < 2:
        scan_ok, _ = _scan_token_identities(
            containment_token,
            identities,
            max_identities=max_identities,
        )
        authoritative_ok = True
        if systemd_unit is not None:
            authoritative_ok = signal_systemd_scope(
                systemd_unit, int(signal.SIGKILL)
            ) and (systemd_scope_is_quiescent(systemd_unit) is True)
        else:
            try:
                os.killpg(process_group_id, signal.SIGKILL)
            except ProcessLookupError:
                pass
            except (PermissionError, OSError):
                authoritative_ok = False

        direct_ok = True
        for identity in tuple(identities.values()):
            if systemd_unit is not None:
                membership = _identity_in_systemd_scope(identity, systemd_unit)
                if membership is True:
                    continue
                if membership is None:
                    direct_ok = False
                    permanent_ownership_uncertain = True
                    continue
            if not _signal_identity(identity, signal.SIGKILL):
                direct_ok = False
                permanent_ownership_uncertain = True
        _prune_gone_identities(identities)
        if (
            scan_ok
            and authoritative_ok
            and direct_ok
            and not identities
            and not permanent_ownership_uncertain
        ):
            stable_empty_scans += 1
        else:
            stable_empty_scans = 0
        time.sleep(0.05)


def _identity_in_systemd_scope(identity: GuardedIdentity, unit: str) -> Optional[bool]:
    process, gone = identity.resolve()
    if gone:
        return True
    if process is None:
        return None
    try:
        with open(f"/proc/{process.pid}/cgroup", encoding="utf-8") as cgroup_file:
            cgroup_text = cgroup_file.read()
    except OSError:
        return None
    return cgroup_path_contains_unit(cgroup_text, unit)


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Parse inherited descriptor identities and run the guardian."""

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--workload-pid", type=int, required=True)
    parser.add_argument("--process-group-id", type=int, required=True)
    parser.add_argument("--liveness-fd", type=int, required=True)
    parser.add_argument("--acknowledgement-fd", type=int, required=True)
    parser.add_argument("--ready-fd", type=int, required=True)
    parser.add_argument("--containment-token", required=True)
    parser.add_argument("--max-identities", type=int, required=True)
    parser.add_argument("--systemd-unit")
    args = parser.parse_args(argv)
    return run_guardian(
        workload_pid=args.workload_pid,
        process_group_id=args.process_group_id,
        liveness_read_fd=args.liveness_fd,
        acknowledgement_write_fd=args.acknowledgement_fd,
        ready_write_fd=args.ready_fd,
        containment_token=args.containment_token,
        max_identities=args.max_identities,
        systemd_unit=args.systemd_unit,
    )


if __name__ == "__main__":  # pragma: no cover - subprocess entry point
    sys.exit(main())
