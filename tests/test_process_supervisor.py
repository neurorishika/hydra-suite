from __future__ import annotations

import itertools
import os
import signal
import subprocess
import sys
import threading
import time
import tracemalloc
from pathlib import Path
from types import SimpleNamespace

import psutil
import pytest

import hydra_suite.runtime.process_guardian as guardian_module
import hydra_suite.runtime.process_supervisor as supervisor_module
from hydra_suite.runtime.process_supervisor import (
    BoundedLineBuffer,
    ContainmentPlan,
    ExitEvidence,
    ExitKind,
    OwnedProcessTree,
    ProcessTreeWatchdog,
    SupervisedSidecar,
    WatchdogPolicy,
    WatchdogTrigger,
    WorkloadStillOwnedError,
    classify_exit,
)
from hydra_suite.runtime.resource_lease import (
    HeavyJobLease,
    HeavyJobLeaseSet,
    ResourceBusyError,
    canonical_heavy_job_lease_set,
)
from hydra_suite.runtime.resource_limits import (
    LimitBackend,
    ProcessMemoryLimits,
    build_limited_launch,
)

pytestmark = pytest.mark.skipif(os.name != "posix", reason="process-group tests")


@pytest.fixture(autouse=True)
def _isolate_canonical_lease_directory(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("HYDRA_DATA_DIR", str(tmp_path))


def _child_env() -> dict[str, str]:
    env = dict(os.environ)
    src = str(Path(__file__).resolve().parents[1] / "src")
    env["PYTHONPATH"] = src + os.pathsep + env.get("PYTHONPATH", "")
    return env


def _cpu_lease_set(tmp_path: Path) -> HeavyJobLeaseSet:
    return canonical_heavy_job_lease_set(
        "supervised", "cpu", lease_dir=tmp_path / "runtime" / "heavy-job-leases"
    )


def _require_process_table_scan() -> None:
    try:
        next(iter(psutil.process_iter(["pid"])), None)
    except (psutil.Error, OSError):
        pytest.skip("sandbox denies process-table enumeration")


def _initialize_fake_tree(tree: OwnedProcessTree) -> None:
    tree._state_lock = threading.RLock()
    tree._overflow_identity = None
    tree._permanent_ownership_uncertain = False


def _wait_until_gone(pid: int, timeout: float = 5.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if not psutil.pid_exists(pid):
            return
        try:
            if psutil.Process(pid).status() == psutil.STATUS_ZOMBIE:
                return
        except psutil.Error:
            return
        time.sleep(0.02)
    raise AssertionError(f"PID {pid} survived its owned process-group teardown")


def test_soft_limit_requests_cooperative_exit():
    script = """
import signal, sys, time
def stop(*_):
    print('cooperative-stop', flush=True)
    sys.exit(0)
signal.signal(signal.SIGTERM, stop)
print('ready', flush=True)
while True: time.sleep(0.1)
"""
    process = subprocess.Popen(
        [sys.executable, "-c", script],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        start_new_session=True,
    )
    assert process.stdout is not None
    assert process.stdout.readline().strip() == "ready"
    watchdog = ProcessTreeWatchdog(
        OwnedProcessTree(process, owns_process_group=True),
        WatchdogPolicy(
            soft_tree_rss_bytes=1,
            hard_tree_rss_bytes=1024**4,
            minimum_system_available_bytes=0,
            poll_interval_seconds=0.01,
            terminate_grace_seconds=1.0,
        ),
    )
    watchdog.start()
    process.wait(timeout=5)
    outcome = watchdog.stop(timeout=2)

    assert outcome is not None
    assert outcome.trigger is WatchdogTrigger.SOFT_RSS
    assert outcome.graceful_exit
    assert process.stdout.read().strip() == "cooperative-stop"
    classified = classify_exit(ExitEvidence(process.returncode, watchdog=outcome))
    assert classified.kind is ExitKind.HOST_SOFT_LIMIT


def test_hard_limit_kills_a_silent_complete_process_tree():
    child_script = (
        "import signal,time; signal.signal(signal.SIGTERM, signal.SIG_IGN); "
        "time.sleep(60)"
    )
    parent_script = """
import signal, subprocess, sys, time
signal.signal(signal.SIGTERM, signal.SIG_IGN)
child = subprocess.Popen([sys.executable, '-c', sys.argv[1]])
print(child.pid, flush=True)
time.sleep(60)
"""
    process = subprocess.Popen(
        [sys.executable, "-c", parent_script, child_script],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        start_new_session=True,
    )
    assert process.stdout is not None
    child_pid = int(process.stdout.readline())
    watchdog = ProcessTreeWatchdog(
        OwnedProcessTree(process, owns_process_group=True),
        WatchdogPolicy(
            soft_tree_rss_bytes=1,
            hard_tree_rss_bytes=1,
            minimum_system_available_bytes=0,
            poll_interval_seconds=0.01,
            terminate_grace_seconds=0.05,
        ),
    )
    watchdog.start()
    process.wait(timeout=5)
    outcome = watchdog.stop(timeout=2)

    assert outcome is not None
    assert outcome.trigger is WatchdogTrigger.HARD_RSS
    assert outcome.hard_kill_sent
    assert process.returncode == -signal.SIGKILL
    _wait_until_gone(child_pid)
    assert classify_exit(ExitEvidence(process.returncode, watchdog=outcome)).kind is (
        ExitKind.HOST_HARD_LIMIT
    )


def test_termination_reaches_descendants_that_escape_the_process_group():
    process = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(60)"],
        start_new_session=True,
    )
    escaped = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(60)"],
        start_new_session=True,
    )
    tree = OwnedProcessTree(process, owns_process_group=True)
    # Model an identity already captured by a prior watchdog sample. macOS's
    # sandbox may forbid global child enumeration, so keep this signal-path
    # regression independent of that OS permission.
    tree._known_identities[escaped.pid] = type(tree.root)(
        escaped.pid, psutil.Process(escaped.pid).create_time()
    )

    tree.kill()
    process.wait(timeout=5)
    escaped.wait(timeout=5)

    _wait_until_gone(escaped.pid)


def test_process_group_is_not_signalled_after_all_owned_identities_are_gone(
    monkeypatch,
):
    tree = object.__new__(OwnedProcessTree)
    _initialize_fake_tree(tree)
    tree.root = type("GoneIdentity", (), {"probe": lambda _self: (None, True)})()
    tree.process_group_id = 424242
    tree.systemd_unit = None
    tree._known_identities = {}
    tree.max_tracked_identities = 8
    tree.identity_overflowed = False
    tree.scope_signal_failed = False
    calls = []
    monkeypatch.setattr(os, "killpg", lambda *args: calls.append(args))

    tree.kill()

    assert calls == []


def test_process_group_is_not_signalled_when_identity_changes_during_probe(monkeypatch):
    process = type("Process", (), {"pid": 42})()
    probes = iter(((process, False), (None, True)))

    class ReusedIdentity:
        def probe(self):
            return next(probes)

    identity = ReusedIdentity()
    tree = object.__new__(OwnedProcessTree)
    _initialize_fake_tree(tree)
    tree.process_group_id = 42
    tree.systemd_unit = None
    tree.scope_signal_failed = False
    monkeypatch.setattr(os, "getpgid", lambda _pid: 42)
    monkeypatch.setattr(
        os, "killpg", lambda *_args: pytest.fail("reused process group was signalled")
    )
    monkeypatch.setattr(tree, "_signal_identity", lambda *_args: True)

    tree._signal_snapshot(signal.SIGKILL, (identity,))


def test_systemd_scope_is_authoritative_for_tree_signals(monkeypatch):
    class FailOnDirectSignal:
        pid = 123

        def send_signal(self, _signum):
            pytest.fail("systemd-owned PID was signalled directly")

    class LiveIdentity:
        def probe(self):
            return FailOnDirectSignal(), False

        def resolve(self):
            return FailOnDirectSignal()

    tree = object.__new__(OwnedProcessTree)
    _initialize_fake_tree(tree)
    tree.root = type("GoneIdentity", (), {"probe": lambda _self: (None, True)})()
    tree.process_group_id = 424242
    tree.systemd_unit = "hydra-owned.scope"
    tree._known_identities = {123: LiveIdentity()}
    tree.max_tracked_identities = 8
    tree.identity_overflowed = False
    tree.scope_signal_failed = False
    calls = []

    def record_systemd_signal(unit, signum):
        calls.append((unit, signum))
        return True

    monkeypatch.setattr(
        supervisor_module,
        "signal_systemd_scope",
        record_systemd_signal,
    )
    monkeypatch.setattr(tree, "_identity_in_systemd_scope", lambda _identity: True)
    monkeypatch.setattr(os, "killpg", lambda *_args: pytest.fail("killpg was used"))

    tree.kill()

    assert calls == [("hydra-owned.scope", int(signal.SIGKILL))]


def test_failed_systemd_signal_targets_only_proven_escapees_and_retains_ownership(
    monkeypatch,
):
    direct_signals = []

    class FakeProcess:
        pid = 123

        def send_signal(self, signum):
            direct_signals.append(signum)

    class LiveIdentity:
        def probe(self):
            return FakeProcess(), False

        def resolve(self):
            return FakeProcess()

    tree = object.__new__(OwnedProcessTree)
    _initialize_fake_tree(tree)
    tree.root = type("GoneIdentity", (), {"probe": lambda _self: (None, True)})()
    tree.process_group_id = 424242
    tree.systemd_unit = "hydra-owned.scope"
    tree.scope_signal_failed = False
    tree._known_identities = {123: LiveIdentity()}
    tree.max_tracked_identities = 8
    tree.identity_overflowed = False
    monkeypatch.setattr(tree, "_discover_identities", lambda: ((), None, False))
    monkeypatch.setattr(tree, "_identity_in_systemd_scope", lambda _identity: False)
    monkeypatch.setattr(
        supervisor_module, "signal_systemd_scope", lambda _unit, _signum: False
    )

    assert not tree.kill()

    assert direct_signals == [signal.SIGKILL]
    assert tree.ownership_uncertain


def test_systemd_membership_treats_identity_that_exited_during_signal_as_gone(
    monkeypatch,
):
    class GoneIdentity:
        def probe(self):
            return None, True

    tree = object.__new__(OwnedProcessTree)
    _initialize_fake_tree(tree)
    tree.systemd_unit = "hydra-owned.scope"
    monkeypatch.setattr(supervisor_module.platform, "system", lambda: "Linux")

    assert tree._identity_in_systemd_scope(GoneIdentity()) is True


def test_systemd_membership_rechecks_identity_when_proc_cgroup_disappears(
    monkeypatch,
):
    process = type("Process", (), {"pid": 42})()
    probes = iter(((process, False), (None, True)))

    class ExitingIdentity:
        def probe(self):
            return next(probes)

    tree = object.__new__(OwnedProcessTree)
    _initialize_fake_tree(tree)
    tree.systemd_unit = "hydra-owned.scope"
    monkeypatch.setattr(supervisor_module.platform, "system", lambda: "Linux")
    monkeypatch.setattr(
        supervisor_module.Path,
        "read_text",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(FileNotFoundError()),
    )

    assert tree._identity_in_systemd_scope(ExitingIdentity()) is True


def test_identity_registry_prunes_dead_entries_retains_escapees_and_fails_closed(
    monkeypatch,
):
    killed = []

    class FakeProcess:
        def __init__(self, pid):
            self.pid = pid

        def send_signal(self, signum):
            killed.append((self.pid, signum))

    live_pids = {2, 3, 4}
    identity_type = supervisor_module._ProcessIdentity
    monkeypatch.setattr(
        identity_type,
        "probe",
        lambda identity: (
            (FakeProcess(identity.pid), False)
            if identity.pid in live_pids
            else (None, True)
        ),
    )
    tree = object.__new__(OwnedProcessTree)
    _initialize_fake_tree(tree)
    tree.root = identity_type(1, 1.0)
    tree.process_group_id = None
    tree.systemd_unit = None
    tree.scope_signal_failed = False
    tree._known_identities = {
        1: identity_type(1, 1.0),  # dead same-group root is pruned
        2: identity_type(2, 2.0),  # live captured escapee must remain
    }
    tree.max_tracked_identities = 2
    tree.identity_overflowed = False
    monkeypatch.setattr(
        tree,
        "_discover_identities",
        lambda: ((identity_type(3, 3.0),), identity_type(4, 4.0), False),
    )

    identities = tree.identities()

    assert {identity.pid for identity in identities} == {2, 3, 4}
    assert set(tree._known_identities) == {2, 3}
    assert tree.identity_overflowed
    assert killed == [
        (4, signal.SIGKILL),
        (3, signal.SIGKILL),
        (2, signal.SIGKILL),
    ]


def test_failed_direct_signal_permanently_retains_uncertain_ownership(monkeypatch):
    class DeniedProcess:
        pid = 42

        def send_signal(self, _signum):
            raise psutil.AccessDenied(42)

    identity = supervisor_module._ProcessIdentity(42, 1.0)
    tree = object.__new__(OwnedProcessTree)
    _initialize_fake_tree(tree)
    tree.process_group_id = None
    tree.systemd_unit = None
    tree.scope_signal_failed = False
    tree._known_identities = {42: identity}
    tree.max_tracked_identities = 1
    tree.identity_overflowed = False
    monkeypatch.setattr(
        supervisor_module._ProcessIdentity,
        "probe",
        lambda _identity: (DeniedProcess(), False),
    )
    assert not tree._signal_snapshot(signal.SIGKILL, (identity,))
    assert tree.ownership_uncertain
    monkeypatch.setattr(
        supervisor_module._ProcessIdentity,
        "probe",
        lambda _identity: (None, True),
    )
    assert tree.ownership_uncertain


def test_bounded_discovery_stops_streaming_process_table_at_cap(monkeypatch):
    calls = []

    class FakeProcess:
        def __init__(self, pid, ppid=0):
            self.pid = pid
            self.info = {"pid": pid, "ppid": ppid, "create_time": float(pid)}

    def stream_processes(_attrs):
        for pid in range(2, 100):
            calls.append(pid)
            yield FakeProcess(pid, ppid=1)

    tree = object.__new__(OwnedProcessTree)
    _initialize_fake_tree(tree)
    root_process = FakeProcess(1)
    tree.root = type(
        "RootIdentity", (), {"probe": lambda _self: (root_process, False)}
    )()
    tree._known_identities = {1: supervisor_module._ProcessIdentity(1, 1.0)}
    tree.max_tracked_identities = 2
    monkeypatch.setattr(supervisor_module.psutil, "process_iter", stream_processes)

    discovered, overflow, uncertain = tree._discover_identities()

    assert [identity.pid for identity in discovered] == [2]
    assert overflow is not None and overflow.pid == 3
    assert not uncertain
    assert calls == [2, 3]


def test_owned_tree_registry_operations_are_serialized_across_threads(monkeypatch):
    identity_type = supervisor_module._ProcessIdentity

    class FakeProcess:
        def __init__(self, pid):
            self.pid = pid

        def send_signal(self, _signum):
            return None

    monkeypatch.setattr(
        identity_type,
        "probe",
        lambda identity: (FakeProcess(identity.pid), False),
    )
    tree = object.__new__(OwnedProcessTree)
    _initialize_fake_tree(tree)
    tree.root = identity_type(1, 1.0)
    tree.process_group_id = None
    tree.systemd_unit = None
    tree.scope_signal_failed = False
    tree._known_identities = {1: identity_type(1, 1.0)}
    tree.max_tracked_identities = 64
    tree.identity_overflowed = False
    counter = itertools.cycle(range(2, 20))
    monkeypatch.setattr(
        tree,
        "_discover_identities",
        lambda: ((identity_type(next(counter), 1.0),), None, False),
    )
    errors = []

    def mutate_registry():
        try:
            for _ in range(50):
                tree.identities()
                tree.kill()
        except BaseException as exc:  # noqa: B036, BLE001 - asserted below
            errors.append(exc)

    threads = [threading.Thread(target=mutate_registry) for _ in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=5)

    assert not errors
    assert all(not thread.is_alive() for thread in threads)


def test_watchdog_observation_exception_kills_and_marks_tree_uncertain():
    class BrokenTree:
        root = type("Root", (), {"pid": 123})()
        killed = False
        uncertain = False

        def is_alive(self):
            raise RuntimeError("registry mutation failed")

        def mark_ownership_uncertain(self):
            self.uncertain = True

        def kill(self):
            self.killed = True

    tree = BrokenTree()
    watchdog = ProcessTreeWatchdog(
        tree,
        WatchdogPolicy(1, 2, 0),
    )

    watchdog._run()

    assert tree.killed
    assert tree.uncertain
    assert watchdog.outcome is not None
    assert watchdog.outcome.trigger is WatchdogTrigger.OBSERVATION_FAILURE


def test_guardian_registry_retains_first_overflow_identity_until_proven_gone(
    monkeypatch,
):
    token = "owned-token"

    class FakeProcess:
        def __init__(self, pid, create_time):
            self.pid = pid
            self._create_time = create_time
            self.info = {"uids": type("Uids", (), {"real": os.getuid()})()}

        def create_time(self):
            return self._create_time

        def environ(self):
            return {"HYDRA_CONTAINMENT_TOKEN": token}

    monkeypatch.setattr(
        guardian_module,
        "_prune_gone_identities",
        lambda identities: identities.pop(9, None),
    )
    monkeypatch.setattr(
        guardian_module.psutil,
        "process_iter",
        lambda _attrs: [
            FakeProcess(3, 3.0),
            FakeProcess(4, 4.0),
            FakeProcess(5, 5.0),
        ],
    )
    identities = {
        2: guardian_module.GuardedIdentity(2, 2.0),
        9: guardian_module.GuardedIdentity(9, 9.0),
    }

    complete, overflowed = guardian_module._scan_token_identities(
        token, identities, external_identities={}, max_identities=3
    )

    assert not complete
    assert overflowed
    assert identities == {
        2: guardian_module.GuardedIdentity(2, 2.0),
        3: guardian_module.GuardedIdentity(3, 3.0),
        4: guardian_module.GuardedIdentity(4, 4.0),
        5: guardian_module.GuardedIdentity(5, 5.0),
    }
    monkeypatch.setattr(
        guardian_module.psutil,
        "process_iter",
        lambda _attrs: [FakeProcess(6, 6.0)],
    )

    complete, overflowed = guardian_module._scan_token_identities(
        token, identities, external_identities={}, max_identities=3
    )

    assert not complete
    assert overflowed
    assert set(identities) == {2, 3, 4, 5}


def test_guardian_does_not_prune_or_acknowledge_unverifiable_identity(monkeypatch):
    identity = guardian_module.GuardedIdentity(42, 1.0)
    identities = {42: identity}
    monkeypatch.setattr(
        guardian_module.GuardedIdentity,
        "resolve",
        lambda _identity: (None, False),
    )

    guardian_module._prune_gone_identities(identities)

    assert identities == {42: identity}
    assert not guardian_module._signal_identity(identity, signal.SIGKILL)


def test_guardian_persists_prior_observation_uncertainty_through_teardown(
    monkeypatch,
):
    class StopRetry(RuntimeError):
        pass

    monkeypatch.setattr(
        guardian_module,
        "_scan_token_identities",
        lambda *_args, **_kwargs: (True, False),
    )
    monkeypatch.setattr(
        guardian_module.os,
        "killpg",
        lambda *_args: (_ for _ in ()).throw(ProcessLookupError()),
    )
    monkeypatch.setattr(
        guardian_module.time,
        "sleep",
        lambda _seconds: (_ for _ in ()).throw(StopRetry()),
    )

    with pytest.raises(StopRetry):
        guardian_module._terminate_until_quiescent(
            containment_token="token",
            identities={},
            max_identities=8,
            process_group_id=123,
            systemd_unit=None,
            initial_ownership_uncertain=True,
        )


def test_guardian_never_signals_reused_group_without_live_owned_member(monkeypatch):
    escaped = guardian_module.GuardedIdentity(42, 1.0)
    direct = []
    monkeypatch.setattr(
        guardian_module,
        "_identity_process_group",
        lambda _identity: 999,
    )

    def record_signal(identity, signum):
        direct.append((identity.pid, signum))
        return True

    monkeypatch.setattr(
        guardian_module,
        "_signal_identity",
        record_signal,
    )
    monkeypatch.setattr(
        guardian_module.os,
        "killpg",
        lambda *_args: pytest.fail("reused original process group was signalled"),
    )

    assert guardian_module._signal_fallback_boundary((escaped,), 123, signal.SIGKILL)
    assert direct == [(42, signal.SIGKILL)]


def test_guardian_directly_signals_identity_that_escapes_after_group_proof(monkeypatch):
    escaped = guardian_module.GuardedIdentity(42, 1.0)
    observed_groups = iter((123, 999))
    group_signals = []
    direct_signals = []
    monkeypatch.setattr(
        guardian_module,
        "_identity_process_group",
        lambda _identity: next(observed_groups),
    )
    monkeypatch.setattr(
        guardian_module.os,
        "killpg",
        lambda group, signum: group_signals.append((group, signum)),
    )

    def record_signal(identity, signum):
        direct_signals.append((identity.pid, signum))
        return True

    monkeypatch.setattr(guardian_module, "_signal_identity", record_signal)

    assert guardian_module._signal_fallback_boundary((escaped,), 123, signal.SIGKILL)
    assert group_signals == [(123, signal.SIGKILL)]
    assert direct_signals == [(42, signal.SIGKILL)]


def test_guardian_baselines_inaccessible_unrelated_process_before_gate(monkeypatch):
    token = "owned-token"

    class FakeProcess:
        def __init__(self, pid, *, owned=False, deny_environment=False):
            self.pid = pid
            self._owned = owned
            self._deny_environment = deny_environment
            self.info = {"uids": type("Uids", (), {"real": 501})()}

        def create_time(self):
            return float(self.pid)

        def environ(self):
            if self._deny_environment:
                raise psutil.AccessDenied(self.pid)
            return {"HYDRA_CONTAINMENT_TOKEN": token} if self._owned else {}

        def ppid(self):
            return 1

    root = FakeProcess(10, owned=True)
    unrelated = FakeProcess(20, deny_environment=True)

    class OldInaccessibleProcess(FakeProcess):
        def environ(self):
            pytest.fail("pre-launch process environment was inspected")

    old_unrelated = OldInaccessibleProcess(5)
    monkeypatch.setattr(
        guardian_module.psutil,
        "process_iter",
        lambda _attrs: iter((root, old_unrelated, unrelated)),
    )
    monkeypatch.setattr(guardian_module.os, "getpid", lambda: 999)
    monkeypatch.setattr(guardian_module.os, "getuid", lambda: 501)
    monkeypatch.setattr(
        guardian_module.GuardedIdentity,
        "resolve",
        lambda _identity: (object(), False),
    )
    owned = {}
    external = {}

    assert guardian_module._baseline_guardian_identities(
        workload_pid=10,
        token=token,
        identities=owned,
        external_identities=external,
        launch_started_at=10.0,
        max_identities=8,
    )
    assert set(owned) == {10}
    assert set(external) == {5, 20}

    complete, overflowed = guardian_module._scan_token_identities(
        token,
        owned,
        external_identities=external,
        launch_started_at=10.0,
        max_identities=8,
    )
    assert complete
    assert not overflowed


def test_guardian_baseline_skips_other_users_before_identity_probe(monkeypatch):
    class OtherUserProcess:
        pid = 5
        info = {"uids": type("Uids", (), {"real": 0})()}

        def create_time(self):
            pytest.fail("another user's protected process identity was probed")

    class OwnedRoot:
        pid = 10
        info = {"uids": type("Uids", (), {"real": 501})()}

        def create_time(self):
            return 10.0

        def environ(self):
            return {"HYDRA_CONTAINMENT_TOKEN": "owned-token"}

        def ppid(self):
            return 1

    monkeypatch.setattr(
        guardian_module.psutil,
        "process_iter",
        lambda _attrs: iter((OtherUserProcess(), OwnedRoot())),
    )
    monkeypatch.setattr(guardian_module.os, "getpid", lambda: 999)
    monkeypatch.setattr(guardian_module.os, "getuid", lambda: 501)

    owned: dict[int, guardian_module.GuardedIdentity] = {}
    assert guardian_module._baseline_guardian_identities(
        workload_pid=10,
        token="owned-token",
        identities=owned,
        external_identities={},
        launch_started_at=10.0,
        max_identities=8,
    )
    assert set(owned) == {10}


def test_guardian_fails_closed_for_new_or_owned_inaccessible_identity(monkeypatch):
    class InaccessibleProcess:
        pid = 30
        info = {"uids": type("Uids", (), {"real": 501})()}

        def create_time(self):
            return 30.0

        def environ(self):
            raise psutil.AccessDenied(self.pid)

        def ppid(self):
            return 1

    monkeypatch.setattr(
        guardian_module.psutil,
        "process_iter",
        lambda _attrs: iter((InaccessibleProcess(),)),
    )
    monkeypatch.setattr(guardian_module.os, "getpid", lambda: 999)
    monkeypatch.setattr(guardian_module.os, "getuid", lambda: 501)

    pending = {}
    complete, overflowed = guardian_module._scan_token_identities(
        "owned-token", {}, external_identities={}, max_identities=8
    )
    assert not complete
    assert not overflowed

    complete, overflowed = guardian_module._scan_token_identities(
        "owned-token",
        {},
        external_identities={},
        pending_external_identities=pending,
        max_identities=8,
    )
    assert complete
    assert not overflowed
    assert pending == {30: guardian_module.GuardedIdentity(30, 30.0)}

    owned = {30: guardian_module.GuardedIdentity(30, 30.0)}
    monkeypatch.setattr(
        guardian_module, "_prune_gone_identities", lambda _identities: None
    )
    complete, overflowed = guardian_module._scan_token_identities(
        "owned-token", owned, external_identities={}, max_identities=8
    )
    assert complete
    assert not overflowed
    assert set(owned) == {30}


def test_guardian_accepts_new_inaccessible_identity_with_captured_external_ancestor(
    monkeypatch,
):
    class InaccessibleProcess:
        pid = 30
        info = {"uids": type("Uids", (), {"real": 501})()}

        def create_time(self):
            return 30.0

        def environ(self):
            raise psutil.AccessDenied(self.pid)

        def ppid(self):
            return 20

    external_parent = guardian_module.GuardedIdentity(20, 20.0)
    monkeypatch.setattr(
        guardian_module.psutil,
        "process_iter",
        lambda _attrs: iter((InaccessibleProcess(),)),
    )
    monkeypatch.setattr(guardian_module.os, "getpid", lambda: 999)
    monkeypatch.setattr(guardian_module.os, "getuid", lambda: 501)
    monkeypatch.setattr(
        guardian_module.GuardedIdentity,
        "resolve",
        lambda identity: (
            (object(), False) if identity == external_parent else (None, True)
        ),
    )

    complete, overflowed = guardian_module._scan_token_identities(
        "owned-token",
        {},
        external_identities={20: external_parent},
        launch_started_at=10.0,
        max_identities=8,
    )

    assert complete
    assert not overflowed

    owned = {30: guardian_module.GuardedIdentity(30, 30.0)}
    complete, _ = guardian_module._scan_token_identities(
        "owned-token", owned, external_identities={}, max_identities=8
    )
    assert not complete


def test_guardian_waits_for_pending_external_identity_without_permanent_uncertainty(
    monkeypatch,
):
    pending = {30: guardian_module.GuardedIdentity(30, 30.0)}
    scans = 0

    def scan(*_args, **kwargs):
        nonlocal scans
        scans += 1
        if scans == 2:
            kwargs["pending_external_identities"].clear()
        return True, False

    monkeypatch.setattr(guardian_module, "_scan_token_identities", scan)
    monkeypatch.setattr(
        guardian_module, "_signal_fallback_boundary", lambda *_args: True
    )
    monkeypatch.setattr(guardian_module.time, "sleep", lambda _seconds: None)

    guardian_module._terminate_until_quiescent(
        containment_token="owned-token",
        identities={},
        external_identities={},
        pending_external_identities=pending,
        max_identities=8,
        process_group_id=123,
        systemd_unit=None,
    )

    assert scans == 3
    assert not pending


def test_guardian_does_not_pair_old_token_with_reused_pid_identity(monkeypatch):
    class ReusedProcess:
        pid = 30
        info = {"uids": type("Uids", (), {"real": 501})()}

        def __init__(self):
            self.create_times = iter((30.0, 31.0))

        def create_time(self):
            return next(self.create_times)

        def environ(self):
            return {"HYDRA_CONTAINMENT_TOKEN": "owned-token"}

    monkeypatch.setattr(
        guardian_module.psutil,
        "process_iter",
        lambda _attrs: iter((ReusedProcess(),)),
    )
    monkeypatch.setattr(guardian_module.os, "getpid", lambda: 999)
    monkeypatch.setattr(guardian_module.os, "getuid", lambda: 501)
    identities = {}

    complete, overflowed = guardian_module._scan_token_identities(
        "owned-token", identities, external_identities={}, max_identities=8
    )

    assert complete
    assert not overflowed
    assert identities == {}


def test_systemd_guardian_never_requires_global_environment_scan(monkeypatch):
    monkeypatch.setattr(
        guardian_module,
        "_baseline_guardian_identities",
        lambda **_kwargs: pytest.fail("systemd used global token scanning"),
    )
    monkeypatch.setattr(
        guardian_module,
        "probe_systemd_scope_invocation_id",
        lambda _unit: "invocation-1",
    )
    monkeypatch.setattr(
        guardian_module,
        "_systemd_scope_contains_workload",
        lambda _unit, _identity: True,
    )
    monkeypatch.setattr(
        guardian_module.psutil,
        "Process",
        lambda pid: type("Process", (), {"create_time": lambda _self: float(pid)})(),
    )

    owned, external, launch_started_at, invocation_id = (
        guardian_module._prepare_guardian_tracking(
            workload_pid=10,
            token="owned-token",
            max_identities=8,
            systemd_unit="hydra-owned.scope",
        )
    )

    assert owned == {}
    assert external == {}
    assert launch_started_at == 10.0
    assert invocation_id == "invocation-1"


def test_systemd_scope_identity_requires_member_descended_from_launcher(monkeypatch):
    parents = {20: 10, 10: 1, 30: 1}
    create_times = {10: 10.0, 20: 20.0, 30: 30.0}

    class FakeProcess:
        def __init__(self, pid):
            self.pid = pid

        def create_time(self):
            return create_times[self.pid]

        def ppid(self):
            return parents[self.pid]

        def environ(self):
            pytest.fail("systemd ownership inspected process environment")

    monkeypatch.setattr(guardian_module.psutil, "Process", FakeProcess)
    monkeypatch.setattr(
        guardian_module,
        "systemd_scope_member_pids",
        lambda _unit: (20,),
    )
    workload = guardian_module.GuardedIdentity(10, 10.0)

    assert (
        guardian_module._systemd_scope_contains_workload("owned.scope", workload)
        is True
    )

    monkeypatch.setattr(
        guardian_module,
        "systemd_scope_member_pids",
        lambda _unit: (30,),
    )
    assert (
        guardian_module._systemd_scope_contains_workload("colliding.scope", workload)
        is False
    )


def test_guardian_is_spawned_as_a_separate_session_outside_systemd(monkeypatch):
    captured = {}

    class FakeGuardian:
        returncode = 0

        def terminate(self):
            pytest.fail("ready guardian was terminated")

    def fake_popen(command, **kwargs):
        captured["command"] = command
        captured["kwargs"] = kwargs
        return FakeGuardian()

    liveness_read, liveness_write = os.pipe()
    acknowledgement_read, acknowledgement_write = os.pipe()
    monkeypatch.setattr(guardian_module.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(
        guardian_module.select, "select", lambda reads, *_args: (reads, [], [])
    )
    monkeypatch.setattr(guardian_module.os, "read", lambda _fd, _size: b"R")
    try:
        guardian_module.spawn_parent_guardian(
            supervisor_pid=os.getpid(),
            supervisor_create_time=psutil.Process(os.getpid()).create_time(),
            workload_pid=100,
            process_group_id=100,
            liveness_read_fd=liveness_read,
            acknowledgement_write_fd=acknowledgement_write,
            containment_token="token",
            max_identities=8,
            systemd_unit="hydra-owned.scope",
        )
    finally:
        os.close(liveness_write)
        os.close(acknowledgement_read)

    assert captured["command"][:3] == [
        sys.executable,
        "-m",
        "hydra_suite.runtime.process_guardian",
    ]
    assert "systemd-run" not in captured["command"]
    assert "--supervisor-pid" in captured["command"]
    assert "--supervisor-create-time" in captured["command"]
    assert captured["kwargs"]["start_new_session"] is True


def test_supervisor_releases_only_after_guardian_quiescence_acknowledgement():
    liveness_read, liveness_write = os.pipe()
    acknowledgement_read, acknowledgement_write = os.pipe()
    os.write(acknowledgement_write, b"Q")
    os.close(acknowledgement_write)

    class CompletedGuardian:
        returncode = 0

        def wait(self, timeout):
            assert timeout == 2.0
            return 0

    sidecar = object.__new__(SupervisedSidecar)
    sidecar._guardian_started = True
    sidecar._guardian_teardown_requested = False
    sidecar._guardian_ack_received = False
    sidecar._parent_liveness_write_fd = liveness_write
    sidecar._guardian_ack_read_fd = acknowledgement_read
    sidecar._guardian_process = CompletedGuardian()

    assert sidecar._complete_guardian_teardown(timeout=0.1)
    assert os.read(liveness_read, 1) == b"T"
    os.close(liveness_read)


def test_guardian_acknowledgement_survives_a_reap_timeout_for_retry():
    liveness_read, liveness_write = os.pipe()
    acknowledgement_read, acknowledgement_write = os.pipe()
    os.write(acknowledgement_write, b"Q")
    os.close(acknowledgement_write)

    class EventuallyReapedGuardian:
        returncode = None

        def __init__(self):
            self.waits = 0

        def wait(self, timeout):
            self.waits += 1
            if self.waits == 1:
                raise subprocess.TimeoutExpired(["guardian"], timeout)
            self.returncode = 0
            return 0

    guardian = EventuallyReapedGuardian()
    sidecar = object.__new__(SupervisedSidecar)
    sidecar._guardian_started = True
    sidecar._guardian_teardown_requested = False
    sidecar._guardian_ack_received = False
    sidecar._parent_liveness_write_fd = liveness_write
    sidecar._guardian_ack_read_fd = acknowledgement_read
    sidecar._guardian_process = guardian

    assert not sidecar._complete_guardian_teardown(timeout=0.1)
    assert sidecar._guardian_ack_received
    assert sidecar._guardian_ack_read_fd is None
    assert sidecar._complete_guardian_teardown(timeout=0.1)
    assert guardian.waits == 2
    os.close(liveness_read)


def test_noisy_output_retains_only_a_fixed_tail():
    output = BoundedLineBuffer(max_lines=10, max_chars=100)
    for index in range(10_000):
        output.append(f"line-{index:05d}\n")

    assert len(output.tail()) <= 10
    assert output.retained_chars <= 100
    assert output.dropped_lines >= 9_990
    assert output.tail()[-1] == "line-09999\n"


def test_bounded_output_records_its_retained_high_water_mark():
    output = BoundedLineBuffer(max_lines=2, max_chars=8)
    output.append("1234")
    output.append("5678")
    output.append("90")

    assert output.high_water_chars == 8
    assert output.retained_chars <= 8


def test_newline_free_multi_megabyte_output_is_read_in_bounded_chunks(tmp_path):
    output_path = tmp_path / "newline-free.log"
    with output_path.open("wb") as target:
        block = b"x" * 4096
        for _ in range(2048):
            target.write(block)
    output = BoundedLineBuffer(max_lines=4, max_chars=16 * 1024)

    with output_path.open("r", encoding="utf-8") as source:
        tracemalloc.start()
        supervisor_module.pump_stdout(source, output, read_chunk_chars=4096)
        _, peak_bytes = tracemalloc.get_traced_memory()
        tracemalloc.stop()

    assert output.retained_chars <= 16 * 1024
    assert peak_bytes < 512 * 1024
    assert output.dropped_lines > 0


def test_short_progress_line_is_available_before_live_pipe_reaches_eof():
    process = subprocess.Popen(
        [
            sys.executable,
            "-c",
            "import time; print('ready', flush=True); time.sleep(60)",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    assert process.stdout is not None
    output = BoundedLineBuffer(max_lines=4, max_chars=1024)
    reader = supervisor_module.threading.Thread(
        target=supervisor_module.pump_stdout,
        args=(process.stdout, output),
        daemon=True,
    )
    reader.start()
    try:
        lines, eof, error = output.drain(timeout=1.0)
    finally:
        process.kill()
        process.wait(timeout=5)
        reader.join(timeout=2)

    assert lines == ["ready\n"]
    assert not eof
    assert error is None


def test_drained_lines_are_retained_as_tail_without_being_reported_as_dropped():
    output = BoundedLineBuffer(max_lines=2, max_chars=100)
    output.append("first\n")
    assert output.drain()[0] == ["first\n"]
    output.append("second\n")
    output.append("third\n")

    assert output.drain()[0] == ["second\n", "third\n"]
    assert output.tail() == ("second\n", "third\n")
    assert output.dropped_lines == 0


def test_noisy_child_is_drained_without_an_unbounded_parent_queue(tmp_path):
    _require_process_table_scan()
    launch = build_limited_launch(
        [
            sys.executable,
            "-c",
            "import sys; [print('x' * 100) for _ in range(10000)]; sys.stdout.flush()",
        ],
        ProcessMemoryLimits(soft_host_bytes=1024**3, hard_host_bytes=2 * 1024**3),
        backend=LimitBackend.WATCHDOG_ONLY,
        environment=_child_env(),
    )
    plan = ContainmentPlan(
        launch=launch,
        job_name="noisy-output",
        minimum_system_available_bytes=0,
        poll_interval_seconds=0.01,
    )
    assert plan.watchdog_policy.soft_tree_rss_bytes == launch.limits.soft_host_bytes
    sidecar = SupervisedSidecar(
        plan,
        output_max_lines=8,
        output_max_chars=1024,
    )

    result = sidecar.wait(timeout=10)

    assert result.classified_exit.kind is ExitKind.SUCCESS
    assert len(result.output_tail) <= 8
    assert sum(map(len, result.output_tail)) <= 1024
    assert result.dropped_output_lines >= 9_992


@pytest.mark.parametrize(
    "backend", [LimitBackend.WATCHDOG_ONLY, LimitBackend.RLIMIT_AS]
)
def test_parent_liveness_guardian_is_enabled_for_each_posix_fallback(
    backend, monkeypatch, tmp_path
):
    _require_process_table_scan()
    launch = build_limited_launch(
        [sys.executable, "-c", "print('done')"],
        ProcessMemoryLimits(soft_host_bytes=1024**3, hard_host_bytes=2 * 1024**3),
        backend=backend,
        environment=_child_env(),
    )
    captured_kwargs = []
    real_popen = subprocess.Popen

    def recording_popen(*args, **kwargs):
        captured_kwargs.append(kwargs)
        return real_popen(*args, **kwargs)

    monkeypatch.setattr(subprocess, "Popen", recording_popen)
    sidecar = SupervisedSidecar(
        ContainmentPlan(
            launch=launch,
            job_name="guardian-backend",
            minimum_system_available_bytes=0,
        )
    )
    sidecar.wait(timeout=5)

    assert "HYDRA_START_GATE_FD" in captured_kwargs[0]["env"]
    assert "HYDRA_CONTAINMENT_TOKEN" in captured_kwargs[0]["env"]
    assert captured_kwargs[0]["pass_fds"]


def test_wait_timeout_terminates_reaps_and_releases_owned_lease(tmp_path):
    _require_process_table_scan()
    launch = build_limited_launch(
        [sys.executable, "-c", "import time; time.sleep(60)"],
        ProcessMemoryLimits(soft_host_bytes=1024**3, hard_host_bytes=2 * 1024**3),
        backend=LimitBackend.WATCHDOG_ONLY,
        environment=_child_env(),
    )
    leases = _cpu_lease_set(tmp_path)
    resource_key = leases.resource_keys[0]
    plan = ContainmentPlan(
        launch=launch,
        job_name="supervised",
        minimum_system_available_bytes=0,
        poll_interval_seconds=0.01,
        terminate_grace_seconds=0.05,
    )
    sidecar = SupervisedSidecar(plan)
    lease_dir = leases.leases[0].path.parent
    with pytest.raises(ResourceBusyError):
        HeavyJobLease(resource_key, "competitor", lease_dir).acquire()

    with pytest.raises(subprocess.TimeoutExpired):
        sidecar.wait(timeout=0.05)

    assert sidecar.process.poll() is not None
    with HeavyJobLease(resource_key, "after teardown", lease_dir):
        pass


def test_wait_timeout_returns_explicit_owner_when_exit_cannot_be_proved(
    tmp_path, monkeypatch
):
    _require_process_table_scan()
    launch = build_limited_launch(
        [sys.executable, "-c", "import time; time.sleep(60)"],
        ProcessMemoryLimits(soft_host_bytes=1024**3, hard_host_bytes=2 * 1024**3),
        backend=LimitBackend.WATCHDOG_ONLY,
        environment=_child_env(),
    )
    leases = _cpu_lease_set(tmp_path)
    resource_key = leases.resource_keys[0]
    plan = ContainmentPlan(
        launch=launch,
        job_name="supervised",
        minimum_system_available_bytes=0,
        terminate_grace_seconds=0,
    )
    sidecar = SupervisedSidecar(plan)
    lease_dir = leases.leases[0].path.parent
    real_teardown = sidecar._terminate_and_reap
    monkeypatch.setattr(sidecar, "_terminate_and_reap", lambda _grace: False)
    try:
        with pytest.raises(WorkloadStillOwnedError) as error:
            sidecar.wait(timeout=0)

        assert error.value.sidecar is sidecar
        with pytest.raises(ResourceBusyError):
            HeavyJobLease(resource_key, "competitor", lease_dir).acquire()
    finally:
        monkeypatch.setattr(sidecar, "_terminate_and_reap", real_teardown)
        # A guardian acknowledgement is explicitly nonterminal: a delayed
        # process-table observation must retain ownership for a later retry,
        # not make test cleanup assume the first five-second window is enough.
        for attempt in range(3):
            try:
                sidecar.cancel(grace_seconds=0.05)
            except WorkloadStillOwnedError:
                if attempt == 2:
                    raise
            else:
                break


def test_final_admission_runs_under_supervisor_owned_lease_before_popen(
    tmp_path, monkeypatch
):
    launch = build_limited_launch(
        [sys.executable, "-c", "raise AssertionError('must not start')"],
        ProcessMemoryLimits(soft_host_bytes=1024**3, hard_host_bytes=2 * 1024**3),
        backend=LimitBackend.WATCHDOG_ONLY,
        environment=_child_env(),
    )
    leases = _cpu_lease_set(tmp_path)
    resource_key = leases.resource_keys[0]
    plan = ContainmentPlan(
        launch=launch,
        job_name="supervised",
        minimum_system_available_bytes=0,
    )
    callback_ran = []

    def refuse_while_locked():
        callback_ran.append(True)
        with pytest.raises(ResourceBusyError):
            HeavyJobLease(
                resource_key, "competitor", leases.leases[0].path.parent
            ).acquire()
        raise RuntimeError("final admission refused")

    monkeypatch.setattr(
        subprocess,
        "Popen",
        lambda *_args, **_kwargs: pytest.fail("child started after admission refusal"),
    )

    with pytest.raises(RuntimeError, match="final admission refused"):
        SupervisedSidecar(plan, prelaunch_check=refuse_while_locked)

    assert callback_ran == [True]
    with HeavyJobLease(resource_key, "after refusal", leases.leases[0].path.parent):
        pass


def test_post_exit_validation_runs_before_canonical_lease_release(tmp_path):
    _require_process_table_scan()
    launch = build_limited_launch(
        [sys.executable, "-c", "print('done')"],
        ProcessMemoryLimits(soft_host_bytes=1024**3, hard_host_bytes=2 * 1024**3),
        backend=LimitBackend.WATCHDOG_ONLY,
        environment=_child_env(),
    )
    plan = ContainmentPlan(
        launch=launch, job_name="post-exit-check", minimum_system_available_bytes=0
    )
    lease = _cpu_lease_set(tmp_path)
    resource_key = lease.resource_keys[0]
    lease_dir = lease.leases[0].path.parent
    sidecar = SupervisedSidecar(plan)

    def validate(_result):
        with pytest.raises(ResourceBusyError):
            HeavyJobLease(resource_key, "competitor", lease_dir).acquire()

    sidecar.wait(timeout=5, post_exit_check=validate)

    with HeavyJobLease(resource_key, "after validation", lease_dir):
        pass


def test_watchdog_records_bounded_accelerator_pressure_telemetry():
    process = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(.15)"],
        start_new_session=True,
    )
    observations = iter((3, 9, 4))
    watchdog = ProcessTreeWatchdog(
        OwnedProcessTree(process, owns_process_group=True),
        WatchdogPolicy(
            soft_tree_rss_bytes=1024**3,
            hard_tree_rss_bytes=2 * 1024**3,
            minimum_system_available_bytes=0,
            poll_interval_seconds=0.01,
        ),
        accelerator_probe=lambda: next(observations, 4),
    )
    watchdog.start()
    process.wait(timeout=5)
    watchdog.stop(timeout=2)

    assert watchdog.peak_accelerator_bytes == 9
    assert watchdog.accelerator_observation_error is None
    assert watchdog.peak_tree_rss_bytes > 0
    assert watchdog.minimum_system_available_bytes is not None


def test_watchdog_checks_host_reserve_before_slow_accelerator_probe(monkeypatch):
    killed = []

    class Tree:
        root = SimpleNamespace(pid=42)
        identity_overflowed = False

        def is_alive(self):
            return True

        def rss_bytes(self):
            return 100

        def kill(self):
            killed.append(True)
            return True

    monkeypatch.setattr(
        supervisor_module.psutil,
        "virtual_memory",
        lambda: SimpleNamespace(available=10),
    )
    watchdog = ProcessTreeWatchdog(
        Tree(),
        WatchdogPolicy(
            soft_tree_rss_bytes=1000,
            hard_tree_rss_bytes=2000,
            minimum_system_available_bytes=20,
            poll_interval_seconds=0.1,
        ),
        accelerator_probe=lambda: pytest.fail(
            "slow accelerator telemetry ran before the host safety checks"
        ),
    )

    watchdog._monitor_until_stopped()

    assert killed == [True]
    assert watchdog.outcome is not None
    assert watchdog.outcome.trigger is WatchdogTrigger.SYSTEM_RESERVE


def test_containment_plan_derives_canonical_keys_and_internal_lease_contends(
    tmp_path, monkeypatch
):
    launch = build_limited_launch(
        [sys.executable, "-c", "pass"],
        ProcessMemoryLimits(soft_host_bytes=1024**3, hard_host_bytes=2 * 1024**3),
        backend=LimitBackend.WATCHDOG_ONLY,
        environment=_child_env(),
    )
    plan = ContainmentPlan(
        launch=launch, job_name="canonical-plan", minimum_system_available_bytes=0
    )
    assert plan.expected_resource_keys == _cpu_lease_set(tmp_path).resource_keys

    cuda_launch = build_limited_launch(
        [sys.executable, "-c", "pass"],
        ProcessMemoryLimits(soft_host_bytes=1024**3, hard_host_bytes=2 * 1024**3),
        backend=LimitBackend.WATCHDOG_ONLY,
        environment=_child_env(),
        accelerator_kind="cuda",
        accelerator_device_uuid="GPU-REAL",
    )
    cuda_plan = ContainmentPlan(
        launch=cuda_launch,
        job_name="cuda-plan",
        minimum_system_available_bytes=0,
    )
    assert any("cuda:uuid:gpu-real" in key for key in cuda_plan.expected_resource_keys)

    true_lease = _cpu_lease_set(tmp_path)
    true_lease.acquire()
    monkeypatch.setattr(
        subprocess,
        "Popen",
        lambda *_args, **_kwargs: pytest.fail("fabricated lease started a child"),
    )
    try:
        with pytest.raises(ResourceBusyError):
            SupervisedSidecar(plan)
    finally:
        true_lease.release()


def test_constructor_failure_after_popen_kills_reaps_and_releases_lease(
    tmp_path, monkeypatch
):
    _require_process_table_scan()
    launch = build_limited_launch(
        [sys.executable, "-c", "import time; time.sleep(60)"],
        ProcessMemoryLimits(soft_host_bytes=1024**3, hard_host_bytes=2 * 1024**3),
        backend=LimitBackend.WATCHDOG_ONLY,
        environment=_child_env(),
    )
    leases = _cpu_lease_set(tmp_path)
    resource_key = leases.resource_keys[0]
    plan = ContainmentPlan(
        launch=launch,
        job_name="supervised",
        minimum_system_available_bytes=0,
        poll_interval_seconds=0.01,
    )
    real_popen = subprocess.Popen
    started = []
    captured_kwargs = []

    def recording_popen(*args, **kwargs):
        captured_kwargs.append(kwargs)
        process = real_popen(*args, **kwargs)
        started.append(process)
        return process

    monkeypatch.setattr(subprocess, "Popen", recording_popen)
    monkeypatch.setattr(
        ProcessTreeWatchdog,
        "start",
        lambda _self: (_ for _ in ()).throw(RuntimeError("watchdog start failed")),
    )

    with pytest.raises(RuntimeError, match="watchdog start failed"):
        SupervisedSidecar(plan)

    assert started and started[0].poll() is not None
    assert "HYDRA_PARENT_LEASE_FDS" not in captured_kwargs[0]["env"]
    with HeavyJobLease(
        resource_key, "after setup failure", leases.leases[0].path.parent
    ):
        pass


def test_partial_constructor_owner_can_retry_cleanup_after_guardian_start_failure(
    tmp_path, monkeypatch
):
    launch = build_limited_launch(
        [sys.executable, "-c", "import time; time.sleep(60)"],
        ProcessMemoryLimits(soft_host_bytes=1024**3, hard_host_bytes=2 * 1024**3),
        backend=LimitBackend.WATCHDOG_ONLY,
        environment=_child_env(),
    )
    leases = _cpu_lease_set(tmp_path)
    resource_key = leases.resource_keys[0]
    lease_dir = leases.leases[0].path.parent
    plan = ContainmentPlan(
        launch=launch,
        job_name="supervised",
        minimum_system_available_bytes=0,
        poll_interval_seconds=0.01,
    )
    cleanup_attempts = 0

    def fail_guardian_start(*_args, **_kwargs):
        raise RuntimeError("guardian failed to start")

    def cleanup_is_initially_uncertain(self, process):
        nonlocal cleanup_attempts
        cleanup_attempts += 1
        process.kill()
        process.wait(timeout=2)
        raise WorkloadStillOwnedError("initial cleanup proof failed", self)

    def retry_proves_quiescence(self, _grace_seconds):
        assert self.process is not None and self.process.poll() is not None
        return True

    monkeypatch.setattr(supervisor_module, "spawn_parent_guardian", fail_guardian_start)
    monkeypatch.setattr(
        SupervisedSidecar,
        "_kill_and_reap_after_setup_failure",
        cleanup_is_initially_uncertain,
    )
    monkeypatch.setattr(
        SupervisedSidecar, "_terminate_and_reap", retry_proves_quiescence
    )

    with pytest.raises(WorkloadStillOwnedError) as raised:
        SupervisedSidecar(plan)

    owner = raised.value.sidecar
    assert cleanup_attempts == 1
    assert owner.process is not None and owner.process.poll() is not None
    assert owner.watchdog is None
    assert owner._reader is None
    assert owner._parent_liveness_write_fd is not None
    assert owner._guardian_ack_read_fd is not None
    with pytest.raises(ResourceBusyError):
        HeavyJobLease(resource_key, "competitor", lease_dir).acquire()

    owner.cancel(grace_seconds=0)

    assert owner._parent_liveness_write_fd is None
    assert owner._guardian_ack_read_fd is None
    with HeavyJobLease(resource_key, "after recovery", lease_dir):
        pass


def test_supervisor_process_death_does_not_orphan_watchdog_only_child(tmp_path):
    _require_process_table_scan()
    helper_script = """
import os, sys
os.environ['HYDRA_DATA_DIR'] = sys.argv[1]
from hydra_suite.runtime.process_supervisor import ContainmentPlan, SupervisedSidecar
from hydra_suite.runtime.resource_limits import (
    LimitBackend, ProcessMemoryLimits, build_limited_launch,
)
launch = build_limited_launch(
    [sys.executable, '-c', 'import time; time.sleep(60)'],
    ProcessMemoryLimits(soft_host_bytes=1024**3, hard_host_bytes=2 * 1024**3),
    backend=LimitBackend.WATCHDOG_ONLY,
)
sidecar = SupervisedSidecar(
    ContainmentPlan(
        launch=launch,
        job_name='parent-death',
        minimum_system_available_bytes=0,
    )
)
print(sidecar.process.pid, flush=True)
os._exit(0)
"""
    helper = subprocess.Popen(
        [sys.executable, "-c", helper_script, str(tmp_path)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=_child_env(),
    )
    assert helper.stdout is not None
    child_pid = int(helper.stdout.readline())
    helper.wait(timeout=5)

    _wait_until_gone(child_pid)


def test_postlaunch_fork_cannot_extend_supervisor_liveness_or_lease(tmp_path):
    _require_process_table_scan()
    helper_script = r"""
import os, sys, time
os.environ['HYDRA_DATA_DIR'] = sys.argv[1]
from hydra_suite.runtime.process_supervisor import ContainmentPlan, SupervisedSidecar
from hydra_suite.runtime.resource_limits import (
    LimitBackend, ProcessMemoryLimits, build_limited_launch,
)
launch = build_limited_launch(
    [sys.executable, '-c', 'import time; time.sleep(60)'],
    ProcessMemoryLimits(soft_host_bytes=1024**3, hard_host_bytes=2 * 1024**3),
    backend=LimitBackend.WATCHDOG_ONLY,
)
sidecar = SupervisedSidecar(
    ContainmentPlan(
        launch=launch,
        job_name='fork-holder-parent-death',
        minimum_system_available_bytes=0,
    )
)
holder_pid = os.fork()
if holder_pid == 0:
    time.sleep(60)
    os._exit(0)
print(f'{sidecar.process.pid},{holder_pid}', flush=True)
os._exit(0)
"""
    helper = subprocess.Popen(
        [sys.executable, "-c", helper_script, str(tmp_path)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=_child_env(),
    )
    assert helper.stdout is not None
    report = helper.stdout.readline().strip()
    helper.wait(timeout=5)
    assert helper.returncode == 0, (
        f"helper failed before reporting owned PIDs: {report!r}; "
        f"stderr={helper.stderr.read() if helper.stderr is not None else ''}"
    )
    workload_pid, holder_pid = map(int, report.split(","))

    try:
        assert psutil.pid_exists(holder_pid), "unrelated fork holder exited too early"
        _wait_until_gone(workload_pid)

        leases = _cpu_lease_set(tmp_path)
        deadline = time.monotonic() + 5.0
        while True:
            try:
                leases.acquire()
                break
            except ResourceBusyError:
                if time.monotonic() >= deadline:
                    raise AssertionError(
                        "guardian did not release leases after supervisor identity died"
                    )
                time.sleep(0.02)
        leases.release()
        assert psutil.pid_exists(
            holder_pid
        ), "lease was tested only after the unrelated holder exited"
    finally:
        try:
            os.kill(holder_pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        _wait_until_gone(holder_pid)


def test_parent_death_guardian_captures_setsid_escapee_when_os_allows_enumeration(
    tmp_path,
):
    _require_process_table_scan()
    helper_script = r"""
import os, psutil, sys, time
os.environ['HYDRA_DATA_DIR'] = sys.argv[1]
from hydra_suite.runtime.process_supervisor import ContainmentPlan, SupervisedSidecar
from hydra_suite.runtime.resource_limits import (
    LimitBackend, ProcessMemoryLimits, build_limited_launch,
)
workload = '''
import subprocess, sys, time
escaped = subprocess.Popen(
    [sys.executable, '-c', 'import time; time.sleep(60)'],
    start_new_session=True,
)
print(escaped.pid, flush=True)
time.sleep(60)
'''
launch = build_limited_launch(
    [sys.executable, '-c', workload],
    ProcessMemoryLimits(soft_host_bytes=1024**3, hard_host_bytes=2 * 1024**3),
    backend=LimitBackend.WATCHDOG_ONLY,
)
sidecar = SupervisedSidecar(
    ContainmentPlan(
        launch=launch,
        job_name='setsid-parent-death',
        minimum_system_available_bytes=0,
    )
)
escaped_pid = None
deadline = time.monotonic() + 5
while escaped_pid is None and time.monotonic() < deadline:
    lines, _, _ = sidecar.output.drain(timeout=0.05)
    for line in lines:
        if line.strip().isdigit():
            escaped_pid = int(line.strip())
if escaped_pid is None:
    print('missing-escaped-pid', flush=True)
    sidecar.cancel(0.05)
    raise SystemExit(2)
try:
    psutil.Process(sidecar.process.pid).children(recursive=True)
except (psutil.Error, OSError):
    print('unsupported', flush=True)
    sidecar.cancel(0.05)
    raise SystemExit(0)
print(f'{sidecar.process.pid},{escaped_pid}', flush=True)
os._exit(0)
"""
    helper = subprocess.Popen(
        [sys.executable, "-c", helper_script, str(tmp_path)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=_child_env(),
    )
    assert helper.stdout is not None
    report = helper.stdout.readline().strip()
    helper.wait(timeout=10)
    if report == "unsupported":
        pytest.skip("sandbox denies descendant enumeration needed by this OS test")
    assert helper.returncode == 0, (
        f"helper failed before reporting the escaped PID: {report!r}; "
        f"stderr={helper.stderr.read() if helper.stderr is not None else ''}"
    )
    assert report != "missing-escaped-pid", "child progress line was not delivered"
    root_pid, escaped_pid = map(int, report.split(","))

    _wait_until_gone(root_pid)
    _wait_until_gone(escaped_pid)


def test_immediate_success_cannot_release_an_uncaptured_setsid_escapee(tmp_path):
    _require_process_table_scan()
    workload = """
import subprocess, sys
escaped = subprocess.Popen(
    [sys.executable, '-c', 'import time; time.sleep(60)'],
    start_new_session=True,
)
print(escaped.pid, flush=True)
"""
    for attempt in range(5):
        launch = build_limited_launch(
            [sys.executable, "-c", workload],
            ProcessMemoryLimits(soft_host_bytes=1024**3, hard_host_bytes=2 * 1024**3),
            backend=LimitBackend.WATCHDOG_ONLY,
            environment=_child_env(),
        )
        sidecar = SupervisedSidecar(
            ContainmentPlan(
                launch=launch,
                job_name=f"immediate-{attempt}",
                minimum_system_available_bytes=0,
            )
        )

        result = sidecar.wait(timeout=5)
        pid_lines = [
            line.strip() for line in result.output_tail if line.strip().isdigit()
        ]

        assert result.classified_exit.kind is ExitKind.SUCCESS
        assert pid_lines, "short-lived root did not report its escaped child PID"
        _wait_until_gone(int(pid_lines[-1]))


def test_abrupt_parent_death_captures_immediate_exit_setsid_escapee_repeatedly(
    tmp_path,
):
    _require_process_table_scan()
    helper_script = r"""
import os, sys, time
os.environ['HYDRA_DATA_DIR'] = sys.argv[1]
from hydra_suite.runtime.process_supervisor import ContainmentPlan, SupervisedSidecar
from hydra_suite.runtime.resource_limits import (
    LimitBackend, ProcessMemoryLimits, build_limited_launch,
)
workload = '''
import subprocess, sys
escaped = subprocess.Popen(
    [sys.executable, '-c', 'import time; time.sleep(60)'],
    start_new_session=True,
)
print(escaped.pid, flush=True)
'''
launch = build_limited_launch(
    [sys.executable, '-c', workload],
    ProcessMemoryLimits(soft_host_bytes=1024**3, hard_host_bytes=2 * 1024**3),
    backend=LimitBackend.WATCHDOG_ONLY,
)
sidecar = SupervisedSidecar(
    ContainmentPlan(
        launch=launch,
        job_name='abrupt-immediate',
        minimum_system_available_bytes=0,
    )
)
deadline = time.monotonic() + 5
escaped_pid = None
while escaped_pid is None and time.monotonic() < deadline:
    lines, _, _ = sidecar.output.drain(timeout=0.05)
    for line in lines:
        if line.strip().isdigit():
            escaped_pid = int(line.strip())
if escaped_pid is None:
    sidecar.cancel(0.05)
    raise SystemExit('missing escaped PID')
print(escaped_pid, flush=True)
os._exit(0)
"""
    for attempt in range(5):
        lease_dir = tmp_path / str(attempt)
        lease_dir.mkdir()
        helper = subprocess.Popen(
            [sys.executable, "-c", helper_script, str(lease_dir)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=_child_env(),
        )
        assert helper.stdout is not None
        escaped_pid = int(helper.stdout.readline())
        helper.wait(timeout=5)

        assert helper.returncode == 0
        _wait_until_gone(escaped_pid)


def test_systemd_parent_guardian_requires_scope_quiescence_after_signal(monkeypatch):
    quiescence = iter((False, True, True))
    signals = []

    def signal_scope(unit, signum):
        signals.append((unit, signum))
        return True

    monkeypatch.setattr(guardian_module, "signal_systemd_scope", signal_scope)
    monkeypatch.setattr(
        guardian_module,
        "probe_systemd_scope_invocation_id",
        lambda _unit: "invocation-1",
    )
    monkeypatch.setattr(
        guardian_module,
        "systemd_scope_is_quiescent",
        lambda _unit: next(quiescence),
    )
    monkeypatch.setattr(
        guardian_module,
        "_scan_token_identities",
        lambda *_args, **_kwargs: (True, False),
    )
    monkeypatch.setattr(guardian_module.time, "sleep", lambda _seconds: None)

    guardian_module._terminate_until_quiescent(
        containment_token="token",
        identities={},
        max_identities=8,
        process_group_id=123,
        systemd_unit="owned.scope",
        expected_scope_invocation_id="invocation-1",
    )

    assert signals == [("owned.scope", int(signal.SIGKILL))]


def test_systemd_guardian_accepts_already_unloaded_scope_without_signal(monkeypatch):
    monkeypatch.setattr(
        guardian_module, "systemd_scope_is_quiescent", lambda _unit: True
    )
    monkeypatch.setattr(
        guardian_module,
        "signal_systemd_scope",
        lambda *_args: pytest.fail("an absent scope was signalled"),
    )
    monkeypatch.setattr(guardian_module.time, "sleep", lambda _seconds: None)

    guardian_module._terminate_until_quiescent(
        containment_token="token",
        identities={},
        max_identities=8,
        process_group_id=123,
        systemd_unit="gone.scope",
    )


def test_systemd_guardian_never_signals_reused_unit_invocation(monkeypatch):
    class StopRetry(RuntimeError):
        pass

    monkeypatch.setattr(
        guardian_module, "systemd_scope_is_quiescent", lambda _unit: False
    )
    monkeypatch.setattr(
        guardian_module,
        "probe_systemd_scope_invocation_id",
        lambda _unit: "replacement-invocation",
    )
    monkeypatch.setattr(
        guardian_module,
        "signal_systemd_scope",
        lambda *_args: pytest.fail("replacement unit was signalled"),
    )
    monkeypatch.setattr(
        guardian_module.time,
        "sleep",
        lambda _seconds: (_ for _ in ()).throw(StopRetry()),
    )

    with pytest.raises(StopRetry):
        guardian_module._terminate_until_quiescent(
            containment_token="token",
            identities={},
            max_identities=8,
            process_group_id=123,
            systemd_unit="reused.scope",
            expected_scope_invocation_id="owned-invocation",
        )


def test_success_is_not_reclassified_from_historical_oom_text():
    success = classify_exit(
        ExitEvidence(0, output_tail="previous attempt: CUDA out of memory")
    )
    informational = classify_exit(
        ExitEvidence(1, output_tail="MPS allocated: 4.0 GiB; ordinary failure")
    )

    assert success.kind is ExitKind.SUCCESS
    assert informational.kind is ExitKind.ORDINARY_FAILURE


def test_accelerator_oom_and_ordinary_failure_remain_distinct():
    accelerator = classify_exit(
        ExitEvidence(1, output_tail="torch.cuda.OutOfMemoryError: CUDA out of memory")
    )
    ordinary = classify_exit(ExitEvidence(2, output_tail="bad annotation"))

    assert accelerator.kind is ExitKind.ACCELERATOR_OOM
    assert ordinary.kind is ExitKind.ORDINARY_FAILURE
    assert classify_exit(ExitEvidence(-signal.SIGTERM, requested_cancel=True)).kind is (
        ExitKind.CANCELED
    )
