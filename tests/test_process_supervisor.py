from __future__ import annotations

import os
import signal
import subprocess
import sys
import time

import psutil
import pytest

from hydra_suite.runtime.process_supervisor import (
    BoundedLineBuffer,
    ExitEvidence,
    ExitKind,
    OwnedProcessTree,
    ProcessTreeWatchdog,
    SupervisedSidecar,
    WatchdogPolicy,
    WatchdogTrigger,
    classify_exit,
)
from hydra_suite.runtime.resource_limits import (
    LimitBackend,
    ProcessMemoryLimits,
    build_limited_launch,
)

pytestmark = pytest.mark.skipif(os.name != "posix", reason="process-group tests")


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
    )
    sidecar = SupervisedSidecar(
        launch,
        WatchdogPolicy(
            soft_tree_rss_bytes=1024**3,
            hard_tree_rss_bytes=2 * 1024**3,
            minimum_system_available_bytes=0,
            poll_interval_seconds=0.01,
        ),
        output_max_lines=8,
        output_max_chars=1024,
    )

    result = sidecar.wait(timeout=10)

    assert result.classified_exit.kind is ExitKind.SUCCESS
    assert len(result.output_tail) <= 8
    assert sum(map(len, result.output_tail)) <= 1024
    assert result.dropped_output_lines >= 9_992


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
