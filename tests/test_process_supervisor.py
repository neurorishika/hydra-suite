from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
import tracemalloc
from pathlib import Path

import psutil
import pytest

import hydra_suite.runtime.child_bootstrap as bootstrap_module
import hydra_suite.runtime.process_supervisor as supervisor_module
import hydra_suite.runtime.resource_limits as limits_module
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


def _child_env() -> dict[str, str]:
    env = dict(os.environ)
    src = str(Path(__file__).resolve().parents[1] / "src")
    env["PYTHONPATH"] = src + os.pathsep + env.get("PYTHONPATH", "")
    return env


def _cpu_lease_set(tmp_path: Path) -> HeavyJobLeaseSet:
    return canonical_heavy_job_lease_set("supervised", "cpu", lease_dir=tmp_path)


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
    tree.root = type("GoneIdentity", (), {"resolve": lambda _self: None})()
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


def test_systemd_scope_is_authoritative_for_tree_signals(monkeypatch):
    class FailOnDirectSignal:
        pid = 123

        def send_signal(self, _signum):
            pytest.fail("systemd-owned PID was signalled directly")

    class LiveIdentity:
        def resolve(self):
            return FailOnDirectSignal()

    tree = object.__new__(OwnedProcessTree)
    tree.root = type("GoneIdentity", (), {"resolve": lambda _self: None})()
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
        def resolve(self):
            return FakeProcess()

    tree = object.__new__(OwnedProcessTree)
    tree.root = type("GoneIdentity", (), {"resolve": lambda _self: None})()
    tree.process_group_id = 424242
    tree.systemd_unit = "hydra-owned.scope"
    tree.scope_signal_failed = False
    tree._known_identities = {123: LiveIdentity()}
    tree.max_tracked_identities = 8
    tree.identity_overflowed = False
    monkeypatch.setattr(tree, "_discover_identities", lambda: ())
    monkeypatch.setattr(tree, "_identity_in_systemd_scope", lambda _identity: False)
    monkeypatch.setattr(
        supervisor_module, "signal_systemd_scope", lambda _unit, _signum: False
    )

    assert not tree.kill()

    assert direct_signals == [signal.SIGKILL]
    assert tree.ownership_uncertain


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
        "resolve",
        lambda identity: (
            FakeProcess(identity.pid) if identity.pid in live_pids else None
        ),
    )
    tree = object.__new__(OwnedProcessTree)
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
        lambda: (identity_type(3, 3.0), identity_type(4, 4.0)),
    )

    identities = tree.identities()

    assert {identity.pid for identity in identities} == {2, 3}
    assert set(tree._known_identities) == {2, 3}
    assert tree.identity_overflowed
    assert killed == [
        (4, signal.SIGKILL),
        (3, signal.SIGKILL),
        (2, signal.SIGKILL),
    ]


def test_guardian_registry_prunes_dead_and_kills_every_identity_beyond_bound(
    monkeypatch,
):
    killed = []

    class FakeProcess:
        def __init__(self, pid, create_time):
            self.pid = pid
            self._create_time = create_time

        def create_time(self):
            return self._create_time

        def children(self, recursive):
            assert recursive
            return [FakeProcess(3, 3.0), FakeProcess(4, 4.0)]

        def kill(self):
            killed.append(self.pid)

    monkeypatch.setattr(
        bootstrap_module,
        "_resolve_captured_identity",
        lambda pid, _create_time: (
            FakeProcess(pid, float(pid)) if pid in {1, 2} else None
        ),
    )
    monkeypatch.setattr(psutil, "Process", lambda _pid: FakeProcess(1, 1.0))
    captured = {1: 1.0, 2: 2.0, 5: 5.0}

    succeeded, overflowed = bootstrap_module._capture_descendant_identities(
        1, 1.0, captured, max_identities=3
    )

    assert succeeded
    assert overflowed
    assert captured == {1: 1.0, 2: 2.0, 3: 3.0}
    assert killed == [4]


def test_noisy_output_retains_only_a_fixed_tail():
    output = BoundedLineBuffer(max_lines=10, max_chars=100)
    for index in range(10_000):
        output.append(f"line-{index:05d}\n")

    assert len(output.tail()) <= 10
    assert output.retained_chars <= 100
    assert output.dropped_lines >= 9_990
    assert output.tail()[-1] == "line-09999\n"


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
    leases = _cpu_lease_set(tmp_path)
    plan = ContainmentPlan(
        launch=launch,
        minimum_system_available_bytes=0,
        expected_resource_keys=leases.resource_keys,
        poll_interval_seconds=0.01,
    )
    assert plan.watchdog_policy.soft_tree_rss_bytes == launch.limits.soft_host_bytes
    sidecar = SupervisedSidecar(
        plan,
        leases=leases,
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
    leases = _cpu_lease_set(tmp_path)
    sidecar = SupervisedSidecar(
        ContainmentPlan(
            launch=launch,
            minimum_system_available_bytes=0,
            expected_resource_keys=leases.resource_keys,
        ),
        leases=leases,
    )
    sidecar.wait(timeout=5)

    assert "HYDRA_PARENT_LIVENESS_FD" in captured_kwargs[0]["env"]
    assert captured_kwargs[0]["env"]["HYDRA_PARENT_MAX_IDENTITIES"] == "512"
    assert captured_kwargs[0]["pass_fds"]


def test_wait_timeout_terminates_reaps_and_releases_owned_lease(tmp_path):
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
        minimum_system_available_bytes=0,
        poll_interval_seconds=0.01,
        terminate_grace_seconds=0.05,
        expected_resource_keys=leases.resource_keys,
    )
    sidecar = SupervisedSidecar(plan, leases=leases)
    with pytest.raises(ResourceBusyError):
        HeavyJobLease(resource_key, "competitor", tmp_path).acquire()

    with pytest.raises(subprocess.TimeoutExpired):
        sidecar.wait(timeout=0.05)

    assert sidecar.process.poll() is not None
    with HeavyJobLease(resource_key, "after teardown", tmp_path):
        pass


def test_wait_timeout_returns_explicit_owner_when_exit_cannot_be_proved(
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
    plan = ContainmentPlan(
        launch=launch,
        minimum_system_available_bytes=0,
        terminate_grace_seconds=0,
        expected_resource_keys=leases.resource_keys,
    )
    sidecar = SupervisedSidecar(plan, leases=leases)
    real_teardown = sidecar._terminate_and_reap
    monkeypatch.setattr(sidecar, "_terminate_and_reap", lambda _grace: False)
    try:
        with pytest.raises(WorkloadStillOwnedError) as error:
            sidecar.wait(timeout=0)

        assert error.value.sidecar is sidecar
        with pytest.raises(ResourceBusyError):
            HeavyJobLease(resource_key, "competitor", tmp_path).acquire()
    finally:
        monkeypatch.setattr(sidecar, "_terminate_and_reap", real_teardown)
        sidecar.cancel(grace_seconds=0.05)


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
        minimum_system_available_bytes=0,
        expected_resource_keys=leases.resource_keys,
    )
    callback_ran = []

    def refuse_while_locked():
        callback_ran.append(True)
        with pytest.raises(ResourceBusyError):
            HeavyJobLease(resource_key, "competitor", tmp_path).acquire()
        raise RuntimeError("final admission refused")

    monkeypatch.setattr(
        subprocess,
        "Popen",
        lambda *_args, **_kwargs: pytest.fail("child started after admission refusal"),
    )

    with pytest.raises(RuntimeError, match="final admission refused"):
        SupervisedSidecar(plan, leases=leases, prelaunch_check=refuse_while_locked)

    assert callback_ran == [True]
    with HeavyJobLease(resource_key, "after refusal", tmp_path):
        pass


def test_containment_plan_rejects_noncanonical_or_mismatched_resource_keys(
    tmp_path, monkeypatch
):
    launch = build_limited_launch(
        [sys.executable, "-c", "pass"],
        ProcessMemoryLimits(soft_host_bytes=1024**3, hard_host_bytes=2 * 1024**3),
        backend=LimitBackend.WATCHDOG_ONLY,
        environment=_child_env(),
    )
    with pytest.raises(ValueError, match="unique and sorted"):
        ContainmentPlan(
            launch=launch,
            minimum_system_available_bytes=0,
            expected_resource_keys=("z-device", "a-host"),
        )
    with pytest.raises(ValueError, match="explicit resource lease set"):
        ContainmentPlan(
            launch=launch,
            minimum_system_available_bytes=0,
            expected_resource_keys=(),
        )

    cuda_launch = build_limited_launch(
        [sys.executable, "-c", "pass"],
        ProcessMemoryLimits(soft_host_bytes=1024**3, hard_host_bytes=2 * 1024**3),
        backend=LimitBackend.WATCHDOG_ONLY,
        environment=_child_env(),
        accelerator_kind="cuda",
    )
    with pytest.raises(ValueError, match="cuda memory topology"):
        ContainmentPlan(
            launch=cuda_launch,
            minimum_system_available_bytes=0,
            expected_resource_keys=("host:cuda:uuid:gpu-only",),
        )

    leases = _cpu_lease_set(tmp_path)
    plan = ContainmentPlan(
        launch=launch,
        minimum_system_available_bytes=0,
        expected_resource_keys=("different-host:host-memory",),
    )
    monkeypatch.setattr(
        subprocess,
        "Popen",
        lambda *_args, **_kwargs: pytest.fail("mismatched plan started a child"),
    )

    with pytest.raises(ValueError, match="do not match"):
        SupervisedSidecar(plan, leases=leases)


def test_constructor_failure_after_popen_kills_reaps_and_releases_lease(
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
    plan = ContainmentPlan(
        launch=launch,
        minimum_system_available_bytes=0,
        poll_interval_seconds=0.01,
        expected_resource_keys=leases.resource_keys,
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
        SupervisedSidecar(plan, leases=leases)

    assert started and started[0].poll() is not None
    lease_fds = {
        int(value)
        for value in captured_kwargs[0]["env"]["HYDRA_PARENT_LEASE_FDS"].split(",")
    }
    assert lease_fds.issubset(set(captured_kwargs[0]["pass_fds"]))
    with HeavyJobLease(resource_key, "after setup failure", tmp_path):
        pass


def test_supervisor_process_death_does_not_orphan_watchdog_only_child(tmp_path):
    helper_script = """
import os, sys
from hydra_suite.runtime.process_supervisor import ContainmentPlan, SupervisedSidecar
from hydra_suite.runtime.resource_lease import canonical_heavy_job_lease_set
from hydra_suite.runtime.resource_limits import (
    LimitBackend, ProcessMemoryLimits, build_limited_launch,
)
launch = build_limited_launch(
    [sys.executable, '-c', 'import time; time.sleep(60)'],
    ProcessMemoryLimits(soft_host_bytes=1024**3, hard_host_bytes=2 * 1024**3),
    backend=LimitBackend.WATCHDOG_ONLY,
)
leases = canonical_heavy_job_lease_set('test', 'cpu', lease_dir=sys.argv[1])
sidecar = SupervisedSidecar(
    ContainmentPlan(
        launch=launch,
        minimum_system_available_bytes=0,
        expected_resource_keys=leases.resource_keys,
    ),
    leases=leases,
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


def test_parent_death_guardian_captures_setsid_escapee_when_os_allows_enumeration(
    tmp_path,
):
    helper_script = r"""
import os, psutil, sys, time
from hydra_suite.runtime.process_supervisor import ContainmentPlan, SupervisedSidecar
from hydra_suite.runtime.resource_lease import canonical_heavy_job_lease_set
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
leases = canonical_heavy_job_lease_set('test', 'cpu', lease_dir=sys.argv[1])
sidecar = SupervisedSidecar(
    ContainmentPlan(
        launch=launch,
        minimum_system_available_bytes=0,
        expected_resource_keys=leases.resource_keys,
    ),
    leases=leases,
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


def test_systemd_parent_guardian_retries_failed_scope_signal(monkeypatch):
    results = iter((False, True))
    signals = []
    outside_kills = []

    def signal_scope(unit, signum):
        signals.append((unit, signum))
        return next(results)

    monkeypatch.setattr(limits_module, "signal_systemd_scope", signal_scope)
    monkeypatch.setattr(
        bootstrap_module,
        "_kill_captured_outside_scope",
        lambda unit, captured: outside_kills.append((unit, captured.copy())),
    )
    monkeypatch.setattr(bootstrap_module.time, "sleep", lambda _seconds: None)

    bootstrap_module._guard_systemd_scope("owned.scope", {123: 1.0})

    assert signals == [
        ("owned.scope", int(signal.SIGKILL)),
        ("owned.scope", int(signal.SIGKILL)),
    ]
    assert outside_kills == [("owned.scope", {123: 1.0})]


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
