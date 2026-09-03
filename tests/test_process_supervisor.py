from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import psutil
import pytest

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
from hydra_suite.runtime.resource_lease import HeavyJobLease, ResourceBusyError
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


def test_noisy_output_retains_only_a_fixed_tail():
    output = BoundedLineBuffer(max_lines=10, max_chars=100)
    for index in range(10_000):
        output.append(f"line-{index:05d}\n")

    assert len(output.tail()) <= 10
    assert output.retained_chars <= 100
    assert output.dropped_lines >= 9_990
    assert output.tail()[-1] == "line-09999\n"


def test_drained_lines_are_retained_as_tail_without_being_reported_as_dropped():
    output = BoundedLineBuffer(max_lines=2, max_chars=100)
    output.append("first\n")
    assert output.drain()[0] == ["first\n"]
    output.append("second\n")
    output.append("third\n")

    assert output.drain()[0] == ["second\n", "third\n"]
    assert output.tail() == ("second\n", "third\n")
    assert output.dropped_lines == 0


def test_noisy_child_is_drained_without_an_unbounded_parent_queue():
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


def test_wait_timeout_terminates_reaps_and_releases_owned_lease(tmp_path):
    launch = build_limited_launch(
        [sys.executable, "-c", "import time; time.sleep(60)"],
        ProcessMemoryLimits(soft_host_bytes=1024**3, hard_host_bytes=2 * 1024**3),
        backend=LimitBackend.WATCHDOG_ONLY,
        environment=_child_env(),
    )
    plan = ContainmentPlan(
        launch=launch,
        minimum_system_available_bytes=0,
        poll_interval_seconds=0.01,
        terminate_grace_seconds=0.05,
    )
    lease = HeavyJobLease("host:cpu:memory", "supervised", tmp_path)
    sidecar = SupervisedSidecar(plan, lease=lease)
    with pytest.raises(ResourceBusyError):
        HeavyJobLease("host:cpu:memory", "competitor", tmp_path).acquire()

    with pytest.raises(subprocess.TimeoutExpired):
        sidecar.wait(timeout=0.05)

    assert sidecar.process.poll() is not None
    with HeavyJobLease("host:cpu:memory", "after teardown", tmp_path):
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
    plan = ContainmentPlan(
        launch=launch,
        minimum_system_available_bytes=0,
        terminate_grace_seconds=0,
    )
    lease = HeavyJobLease("host:cpu:memory", "supervised", tmp_path)
    sidecar = SupervisedSidecar(plan, lease=lease)
    real_teardown = sidecar._terminate_and_reap
    monkeypatch.setattr(sidecar, "_terminate_and_reap", lambda _grace: False)
    try:
        with pytest.raises(WorkloadStillOwnedError) as error:
            sidecar.wait(timeout=0)

        assert error.value.sidecar is sidecar
        with pytest.raises(ResourceBusyError):
            HeavyJobLease("host:cpu:memory", "competitor", tmp_path).acquire()
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
    plan = ContainmentPlan(launch=launch, minimum_system_available_bytes=0)
    lease = HeavyJobLease("host:cpu:memory", "supervised", tmp_path)
    callback_ran = []

    def refuse_while_locked():
        callback_ran.append(True)
        with pytest.raises(ResourceBusyError):
            HeavyJobLease("host:cpu:memory", "competitor", tmp_path).acquire()
        raise RuntimeError("final admission refused")

    monkeypatch.setattr(
        subprocess,
        "Popen",
        lambda *_args, **_kwargs: pytest.fail("child started after admission refusal"),
    )

    with pytest.raises(RuntimeError, match="final admission refused"):
        SupervisedSidecar(plan, lease=lease, prelaunch_check=refuse_while_locked)

    assert callback_ran == [True]
    with HeavyJobLease("host:cpu:memory", "after refusal", tmp_path):
        pass


def test_constructor_failure_after_popen_kills_reaps_and_releases_lease(
    tmp_path, monkeypatch
):
    launch = build_limited_launch(
        [sys.executable, "-c", "import time; time.sleep(60)"],
        ProcessMemoryLimits(soft_host_bytes=1024**3, hard_host_bytes=2 * 1024**3),
        backend=LimitBackend.WATCHDOG_ONLY,
        environment=_child_env(),
    )
    plan = ContainmentPlan(
        launch=launch,
        minimum_system_available_bytes=0,
        poll_interval_seconds=0.01,
    )
    real_popen = subprocess.Popen
    started = []

    def recording_popen(*args, **kwargs):
        process = real_popen(*args, **kwargs)
        started.append(process)
        return process

    monkeypatch.setattr(subprocess, "Popen", recording_popen)
    monkeypatch.setattr(
        ProcessTreeWatchdog,
        "start",
        lambda _self: (_ for _ in ()).throw(RuntimeError("watchdog start failed")),
    )

    lease = HeavyJobLease("host:cpu:memory", "supervised", tmp_path)
    with pytest.raises(RuntimeError, match="watchdog start failed"):
        SupervisedSidecar(plan, lease=lease)

    assert started and started[0].poll() is not None
    with HeavyJobLease("host:cpu:memory", "after setup failure", tmp_path):
        pass


def test_supervisor_process_death_does_not_orphan_watchdog_only_child():
    helper_script = """
import os, sys
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
    ContainmentPlan(launch=launch, minimum_system_available_bytes=0)
)
print(sidecar.process.pid, flush=True)
os._exit(0)
"""
    helper = subprocess.Popen(
        [sys.executable, "-c", helper_script],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=_child_env(),
    )
    assert helper.stdout is not None
    child_pid = int(helper.stdout.readline())
    helper.wait(timeout=5)

    _wait_until_gone(child_pid)


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
