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

from .resource_limits import cgroup_path_contains_unit, signal_systemd_scope
from .resource_limits import (
    systemd_scope_invocation_id as probe_systemd_scope_invocation_id,
)
from .resource_limits import systemd_scope_is_quiescent, systemd_scope_member_pids

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
    supervisor_pid: int,
    supervisor_create_time: float,
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
    if supervisor_pid < 1 or supervisor_create_time <= 0:
        raise ValueError("supervisor identity must be positive")
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
        "--supervisor-pid",
        str(supervisor_pid),
        "--supervisor-create-time",
        repr(supervisor_create_time),
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
            "cannot prove process ownership: launch boundary is unavailable"
        )
    return guardian


def run_guardian(
    *,
    supervisor_pid: int,
    supervisor_create_time: float,
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
        try:
            identities, external_identities, launch_started_at, scope_invocation_id = (
                _prepare_guardian_tracking(
                    workload_pid=workload_pid,
                    token=containment_token,
                    max_identities=max_identities,
                    systemd_unit=systemd_unit,
                )
            )
        except RuntimeError:
            os.write(ready_write_fd, b"E")
            return 125
        os.write(ready_write_fd, b"R")
        os.close(ready_write_fd)
        guardian_ownership_uncertain = False
        supervisor_identity = GuardedIdentity(supervisor_pid, supervisor_create_time)
        while True:
            _, supervisor_gone = supervisor_identity.resolve()
            if supervisor_gone:
                request = b""
                break
            readable, _, _ = select.select([liveness_read_fd], [], [], 0.05)
            if readable:
                try:
                    request = os.read(liveness_read_fd, 1)
                except OSError:
                    request = b""
                break
            if systemd_unit is None:
                scan_complete, overflowed = _scan_token_identities(
                    containment_token,
                    identities,
                    external_identities=external_identities,
                    launch_started_at=launch_started_at,
                    max_identities=max_identities,
                )
                if not scan_complete or overflowed:
                    guardian_ownership_uncertain = True

        _terminate_until_quiescent(
            containment_token=containment_token,
            identities=identities,
            external_identities=external_identities,
            launch_started_at=launch_started_at,
            max_identities=max_identities,
            process_group_id=process_group_id,
            systemd_unit=systemd_unit,
            expected_scope_invocation_id=scope_invocation_id,
            initial_ownership_uncertain=guardian_ownership_uncertain,
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


def _prepare_guardian_tracking(
    *,
    workload_pid: int,
    token: str,
    max_identities: int,
    systemd_unit: Optional[str],
) -> tuple[
    dict[int, GuardedIdentity], dict[int, GuardedIdentity], float, Optional[str]
]:
    """Establish the ownership proof before releasing the child start gate."""

    if systemd_unit is not None:
        try:
            workload_identity = GuardedIdentity(
                workload_pid, float(psutil.Process(workload_pid).create_time())
            )
        except (psutil.Error, OSError) as exc:
            raise RuntimeError(
                "systemd launcher disappeared before guardian setup"
            ) from exc
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            invocation_id = probe_systemd_scope_invocation_id(systemd_unit)
            if (
                invocation_id
                and _systemd_scope_contains_workload(systemd_unit, workload_identity)
                is True
            ):
                return {}, {}, workload_identity.create_time, invocation_id
            time.sleep(0.05)
        raise RuntimeError("systemd scope did not become observable")

    identities: dict[int, GuardedIdentity] = {}
    external_identities: dict[int, GuardedIdentity] = {}
    try:
        launch_started_at = float(psutil.Process(workload_pid).create_time())
    except (psutil.Error, OSError) as exc:
        raise RuntimeError(
            "workload identity disappeared before guardian setup"
        ) from exc
    if not _baseline_guardian_identities(
        workload_pid=workload_pid,
        token=token,
        identities=identities,
        external_identities=external_identities,
        launch_started_at=launch_started_at,
        max_identities=max_identities,
    ):
        raise RuntimeError("fallback guardian could not establish ownership baseline")
    return identities, external_identities, launch_started_at, None


def _baseline_guardian_identities(
    *,
    workload_pid: int,
    token: str,
    identities: dict[int, GuardedIdentity],
    external_identities: dict[int, GuardedIdentity],
    launch_started_at: Optional[float] = None,
    max_identities: int,
) -> bool:
    """Classify inaccessible pre-existing processes while the child is gated."""

    if launch_started_at is None:
        try:
            launch_started_at = float(psutil.Process(workload_pid).create_time())
        except (psutil.Error, OSError):
            return False
    try:
        for process in psutil.process_iter(["pid", "uids"]):
            if process.pid == os.getpid():
                continue
            try:
                uids = process.info.get("uids")
                if uids is not None and uids.real != os.getuid():
                    continue
                identity = GuardedIdentity(process.pid, float(process.create_time()))
                if (
                    process.pid != workload_pid
                    and identity.create_time < launch_started_at - 0.01
                ):
                    continue
                process_environment = process.environ()
                if abs(float(process.create_time()) - identity.create_time) >= 0.01:
                    continue
            except (psutil.NoSuchProcess, psutil.ZombieProcess):
                continue
            except (psutil.AccessDenied, OSError):
                # The child is still blocked at its start gate and therefore
                # has no descendants. Every inaccessible non-root identity at
                # this instant is proven external to this launch.
                if process.pid == workload_pid:
                    return False
                try:
                    identity = GuardedIdentity(
                        process.pid, float(process.create_time())
                    )
                except (psutil.NoSuchProcess, psutil.ZombieProcess):
                    continue
                except (psutil.AccessDenied, OSError):
                    return False
                if identity.create_time < launch_started_at - 0.01:
                    # It predates the launch, so it cannot be an escaped child.
                    continue
                if len(external_identities) >= max_identities:
                    return False
                external_identities[identity.pid] = identity
                continue
            if process_environment.get(_TOKEN_ENV) != token:
                continue
            if len(identities) >= max_identities:
                return False
            identities[identity.pid] = identity
        return workload_pid in identities
    except (psutil.Error, OSError, RuntimeError):
        return False


def _scan_token_identities(
    token: str,
    identities: dict[int, GuardedIdentity],
    *,
    external_identities: dict[int, GuardedIdentity],
    launch_started_at: float = 0.0,
    max_identities: int,
) -> tuple[bool, bool]:
    """Capture token-bearing same-user processes without trusting ancestry."""

    _prune_gone_identities(identities)
    _prune_gone_identities(external_identities)
    overflowed = False
    try:
        processes = psutil.process_iter(["pid", "uids"])
        for process in processes:
            if process.pid == os.getpid():
                continue
            identity: Optional[GuardedIdentity] = None
            try:
                uids = process.info.get("uids")
                if uids is not None and uids.real != os.getuid():
                    continue
                identity = GuardedIdentity(process.pid, float(process.create_time()))
                if identity.create_time < launch_started_at - 0.01:
                    continue
                process_environment = process.environ()
                if abs(float(process.create_time()) - identity.create_time) >= 0.01:
                    continue
            except (psutil.NoSuchProcess, psutil.ZombieProcess):
                continue
            except (psutil.AccessDenied, OSError):
                if identity is None:
                    try:
                        identity = GuardedIdentity(
                            process.pid, float(process.create_time())
                        )
                    except (psutil.Error, OSError):
                        return False, overflowed
                if identities.get(identity.pid) == identity:
                    return False, overflowed
                if external_identities.get(identity.pid) == identity:
                    continue
                if identity.create_time < launch_started_at - 0.01:
                    continue
                # A new inaccessible identity was not proven external while
                # the child was gated. It may be an escaped descendant.
                return False, overflowed
            if process_environment.get(_TOKEN_ENV) != token:
                continue
            assert identity is not None
            if identities.get(identity.pid) == identity:
                continue
            if len(identities) > max_identities:
                return False, True
            if len(identities) == max_identities:
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
    external_identities: Optional[dict[int, GuardedIdentity]] = None,
    launch_started_at: float = 0.0,
    max_identities: int,
    process_group_id: int,
    systemd_unit: Optional[str],
    expected_scope_invocation_id: Optional[str] = None,
    initial_ownership_uncertain: bool = False,
) -> None:
    """Retain inherited leases until the complete owned boundary is empty."""

    stable_empty_scans = 0
    permanent_ownership_uncertain = initial_ownership_uncertain
    if external_identities is None:
        external_identities = {}
    while stable_empty_scans < 2:
        if systemd_unit is None:
            scan_ok, _ = _scan_token_identities(
                containment_token,
                identities,
                external_identities=external_identities,
                launch_started_at=launch_started_at,
                max_identities=max_identities,
            )
        else:
            scan_ok = True
        authoritative_ok = True
        if systemd_unit is not None:
            quiescent_before_signal = systemd_scope_is_quiescent(systemd_unit)
            if quiescent_before_signal is True:
                authoritative_ok = True
            elif (
                not expected_scope_invocation_id
                or probe_systemd_scope_invocation_id(systemd_unit)
                != expected_scope_invocation_id
            ):
                authoritative_ok = False
                permanent_ownership_uncertain = True
            else:
                signal_systemd_scope(systemd_unit, int(signal.SIGKILL))
                authoritative_ok = systemd_scope_is_quiescent(systemd_unit) is True
        else:
            authoritative_ok = _signal_fallback_boundary(
                tuple(identities.values()), process_group_id, signal.SIGKILL
            )
            if not authoritative_ok:
                permanent_ownership_uncertain = True

        direct_ok = True
        for identity in tuple(identities.values()) if systemd_unit is not None else ():
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
    except FileNotFoundError:
        _, gone_after_read = identity.resolve()
        return True if gone_after_read else None
    except OSError:
        return None
    return cgroup_path_contains_unit(cgroup_text, unit)


def _signal_fallback_boundary(
    identities: tuple[GuardedIdentity, ...],
    process_group_id: int,
    signum: signal.Signals,
) -> bool:
    """Signal a group only while an exact owned identity proves membership."""

    group_members = {
        identity.pid
        for identity in identities
        if _identity_process_group(identity) == process_group_id
    }
    group_signalled = False
    if group_members:
        try:
            os.killpg(process_group_id, signum)
            group_signalled = True
        except ProcessLookupError:
            pass
        except (PermissionError, OSError):
            return False
    direct_ok = True
    for identity in identities:
        if group_signalled and identity.pid in group_members:
            continue
        if not _signal_identity(identity, signum):
            direct_ok = False
    return direct_ok


def _identity_process_group(identity: GuardedIdentity) -> Optional[int]:
    process, gone = identity.resolve()
    if gone or process is None:
        return None
    try:
        process_group = os.getpgid(process.pid)
    except (ProcessLookupError, PermissionError, OSError):
        return None
    validated_again, gone_after_probe = identity.resolve()
    if gone_after_probe or validated_again is None:
        return None
    return process_group


def _systemd_scope_contains_workload(
    unit: str, workload_identity: GuardedIdentity
) -> Optional[bool]:
    """Verify an exact cgroup member descends from the known launcher."""

    member_pids = systemd_scope_member_pids(unit)
    if member_pids is None:
        return None
    if not member_pids:
        return False
    for pid in member_pids:
        remaining = 64
        current_pid = pid
        try:
            while current_pid > 1 and remaining:
                process = psutil.Process(current_pid)
                if current_pid == workload_identity.pid:
                    if (
                        abs(
                            float(process.create_time()) - workload_identity.create_time
                        )
                        < 0.01
                    ):
                        return True
                    break
                current_pid = int(process.ppid())
                remaining -= 1
        except (psutil.NoSuchProcess, psutil.ZombieProcess):
            continue
        except (psutil.AccessDenied, OSError):
            return None
    return False


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Parse inherited descriptor identities and run the guardian."""

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--supervisor-pid", type=int, required=True)
    parser.add_argument("--supervisor-create-time", type=float, required=True)
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
        supervisor_pid=args.supervisor_pid,
        supervisor_create_time=args.supervisor_create_time,
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
