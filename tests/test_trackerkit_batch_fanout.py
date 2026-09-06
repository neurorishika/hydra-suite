# tests/test_trackerkit_batch_fanout.py
"""Scheduler tests drive a FAKE child (a tiny python script) so they need no
models, videos, or GPUs. The fake prints the same progress/summary lines the
real child's logging emits."""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import pytest

from hydra_suite.runtime.cuda_devices import CudaDevice
from hydra_suite.trackerkit.batch_fanout import (
    FanoutOptions,
    build_child_env,
    default_child_command,
    parse_progress_line,
    parse_summary_line,
    run_batch_fanout,
)
from hydra_suite.trackerkit.batch_plan import BatchJobSpec

FAKE_CHILD = r"""
import json, os, signal, sys, time
cfg = json.load(open(sys.argv[1]))
mode = cfg.get("mode", "ok")
sleep = float(cfg.get("sleep", 0.05))
if mode == "trap_sigint":
    def _h(*_):
        print("SIGINT received - requesting clean stop", flush=True); sys.exit(130)
    signal.signal(signal.SIGINT, _h)
if mode in ("ignore_signals", "orphan_maker"):
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
if mode == "orphan_maker":
    import subprocess
    kid = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    print("GRANDCHILD=%d" % kid.pid, flush=True)
print("2026-01-01 - x - INFO - [track forward] 10% starting", flush=True)
print("GPU=" + os.environ.get("CUDA_VISIBLE_DEVICES", "<unset>"), flush=True)
print("OMP=" + os.environ.get("OMP_NUM_THREADS", "<unset>"), flush=True)
time.sleep(sleep)
print("2026-01-01 - x - INFO - [post] 90% merging", flush=True)
if mode == "fail":
    print("2026-01-01 - x - ERROR - Tracker CLI failed for v: boom", flush=True)
    sys.exit(3)
if mode in ("trap_sigint", "ignore_signals", "orphan_maker"):
    time.sleep(30)
print("2026-01-01 - x - INFO - Tracker CLI completed: video=%s | rows=5 | avg_fps=9.0" % cfg["name"], flush=True)
"""


def _spec(tmp_path: Path, name: str, **cfg) -> BatchJobSpec:
    video = tmp_path / f"{name}.mp4"
    video.write_bytes(b"\x00")
    return BatchJobSpec(
        index=0,
        video_path=str(video),
        config_path=None,
        config={"name": name, **cfg},
        provenance="own-sidecar",
    )


def _fake_command(spec: BatchJobSpec, config_json: Path) -> list[str]:
    return [sys.executable, "-c", FAKE_CHILD, str(config_json)]


class _Recorder:
    def __init__(self):
        self.started, self.progress, self.logs, self.finished = [], [], [], []

    def job_started(self, spec, gpu, log_path):
        self.started.append((spec.video_path, gpu, log_path))

    def job_progress(self, spec, percent, message):
        self.progress.append((spec.video_path, percent, message))

    def job_log(self, spec, line):
        self.logs.append((spec.video_path, line))

    def job_finished(self, result):
        self.finished.append(result)


def test_parse_progress_and_summary_lines():
    assert parse_progress_line("t - n - INFO - [track forward] 45% detecting") == (
        45,
        "detecting",
    )
    assert parse_progress_line("t - n - INFO - [track backward] 7% x") == (7, "x")
    assert parse_progress_line("t - n - INFO - [post] 100% done") == (100, "done")
    assert parse_progress_line("random text") is None
    assert parse_summary_line(
        "t - n - INFO - Tracker CLI completed: video=a | rows=5"
    ) == ["video=a", "rows=5"]
    assert parse_summary_line("t - n - INFO - other") is None


def test_build_child_env_pins_gpu_and_unbuffered_and_caps_only_when_unset():
    gpu = CudaDevice(index=3, uuid="GPU-1234", name="x")
    env = build_child_env(
        {"PATH": "/bin", "OMP_NUM_THREADS": "7"}, gpu=gpu, threads_per_job=4
    )
    assert env["CUDA_VISIBLE_DEVICES"] == "GPU-1234"
    assert env["PYTHONUNBUFFERED"] == "1"
    assert env["KMP_DUPLICATE_LIB_OK"] == "TRUE"
    assert env["OMP_NUM_THREADS"] == "7"  # parent's value wins
    assert env["NUMBA_NUM_THREADS"] == "4"
    assert env["MKL_NUM_THREADS"] == "4"
    assert env["OPENBLAS_NUM_THREADS"] == "4"
    assert env["PATH"] == "/bin"


def test_build_child_env_without_gpu_or_caps_sets_nothing_extra():
    env = build_child_env({"PATH": "/bin"}, gpu=None, threads_per_job=None)
    assert "CUDA_VISIBLE_DEVICES" not in env
    assert "OMP_NUM_THREADS" not in env
    assert env["PYTHONUNBUFFERED"] == "1"


def test_default_child_command_shape(tmp_path):
    spec = _spec(tmp_path, "a")
    cmd = default_child_command(spec, tmp_path / "cfg.json", log_level="DEBUG")
    assert cmd[:3] == [sys.executable, "-m", "hydra_suite.trackerkit.app"]
    assert "track" in cmd and spec.video_path in cmd and "--config" in cmd
    assert cmd[cmd.index("--log-level") + 1] == "DEBUG"
    for forbidden in ("--gpus", "--jobs", "--threads-per-job"):
        assert forbidden not in cmd


def test_runs_all_jobs_and_reports_success(tmp_path):
    specs = [_spec(tmp_path, n) for n in ("a", "b", "c")]
    rec = _Recorder()
    result = run_batch_fanout(
        specs,
        FanoutOptions(jobs=2, run_dir=tmp_path / "run", child_command=_fake_command),
        events=rec,
    )
    assert result.success and not result.cancelled
    assert [r.returncode for r in result.jobs] == [0, 0, 0]
    assert sorted(r.summary_lines[0] for r in result.jobs) == [
        "video=a",
        "video=b",
        "video=c",
    ]
    assert len(rec.started) == 3 and len(rec.finished) == 3
    assert any(p == 90 for _, p, _ in rec.progress)
    for r in result.jobs:
        assert r.log_path.exists()
        assert "GPU=<unset>" in r.log_path.read_text()
    # per-job effective config was written for provenance
    assert sorted(p.name for p in (tmp_path / "run").glob("job_*_config.json")) == [
        "job_1_config.json",
        "job_2_config.json",
        "job_3_config.json",
    ]


def test_gpu_slots_pin_each_child(tmp_path):
    gpus = [CudaDevice(0, "GPU-aaaa", "x"), CudaDevice(1, "GPU-bbbb", "x")]
    specs = [_spec(tmp_path, n, sleep=0.3) for n in ("a", "b", "c", "d")]
    result = run_batch_fanout(
        specs,
        FanoutOptions(
            gpus=gpus, jobs=99, run_dir=tmp_path / "run", child_command=_fake_command
        ),
    )
    assert result.success
    seen = sorted(
        r.log_path.read_text().split("GPU=")[1].split("\n")[0] for r in result.jobs
    )
    assert seen == [
        "GPU-aaaa",
        "GPU-aaaa",
        "GPU-bbbb",
        "GPU-bbbb",
    ]  # jobs clamped to 2 slots, reused


def test_unspecified_jobs_uses_one_slot_per_gpu(tmp_path):
    """``jobs=None`` means "one slot per GPU", not one slot total."""
    gpus = [CudaDevice(0, "GPU-aaaa", "x"), CudaDevice(1, "GPU-bbbb", "x")]
    specs = [_spec(tmp_path, n, sleep=0.3) for n in ("a", "b")]
    t0 = time.monotonic()
    result = run_batch_fanout(
        specs,
        FanoutOptions(
            gpus=gpus, jobs=None, run_dir=tmp_path / "run", child_command=_fake_command
        ),
    )
    wall = time.monotonic() - t0
    assert result.success
    seen = sorted(
        r.log_path.read_text().split("GPU=")[1].split("\n")[0] for r in result.jobs
    )
    assert seen == ["GPU-aaaa", "GPU-bbbb"]  # both GPUs used, not just the first
    assert wall < 2.0, f"2 GPUs with jobs=None should overlap; took {wall:.1f}s"


def test_concurrency_respects_jobs(tmp_path):
    specs = [_spec(tmp_path, n, sleep=0.6) for n in ("a", "b", "c", "d")]
    t0 = time.monotonic()
    result = run_batch_fanout(
        specs,
        FanoutOptions(jobs=4, run_dir=tmp_path / "run", child_command=_fake_command),
    )
    wall = time.monotonic() - t0
    assert result.success
    assert wall < 2.0, f"4 jobs at jobs=4 should overlap; took {wall:.1f}s"


def test_failure_stops_new_launches_but_finishes_running(tmp_path):
    specs = [
        _spec(tmp_path, "a", mode="fail"),
        _spec(tmp_path, "b", sleep=0.5),
        _spec(tmp_path, "c"),
        _spec(tmp_path, "d"),
    ]
    result = run_batch_fanout(
        specs,
        FanoutOptions(jobs=2, run_dir=tmp_path / "run", child_command=_fake_command),
    )
    assert not result.success
    by_name = {r.spec.config["name"]: r for r in result.jobs}
    assert by_name["a"].returncode == 3 and by_name["a"].error
    assert by_name["b"].success  # already running: finished
    assert (
        by_name["c"].returncode is None and not by_name["c"].success
    )  # never launched
    assert by_name["d"].returncode is None


def test_cancel_sends_sigint_and_child_exits_cleanly(tmp_path):
    specs = [_spec(tmp_path, "a", mode="trap_sigint")]
    stop = {"flag": False}

    def _should_stop():
        return stop["flag"]

    import threading

    threading.Timer(0.6, lambda: stop.__setitem__("flag", True)).start()
    t0 = time.monotonic()
    result = run_batch_fanout(
        specs,
        FanoutOptions(
            jobs=1,
            run_dir=tmp_path / "run",
            child_command=_fake_command,
            sigint_grace_s=5,
        ),
        should_stop=_should_stop,
    )
    assert result.cancelled and not result.success
    assert result.jobs[0].returncode == 130
    assert time.monotonic() - t0 < 4.0
    assert "SIGINT received" in result.jobs[0].log_path.read_text()


@pytest.mark.skipif(os.name == "nt", reason="POSIX signals")
def test_cancel_escalates_to_kill_when_child_ignores_signals(tmp_path):
    specs = [_spec(tmp_path, "a", mode="ignore_signals")]
    stop = {"flag": False}
    import threading

    threading.Timer(0.5, lambda: stop.__setitem__("flag", True)).start()
    t0 = time.monotonic()
    result = run_batch_fanout(
        specs,
        FanoutOptions(
            jobs=1,
            run_dir=tmp_path / "run",
            child_command=_fake_command,
            sigint_grace_s=0.3,
            term_grace_s=0.3,
        ),
        should_stop=lambda: stop["flag"],
    )
    assert result.cancelled
    assert result.jobs[0].returncode not in (0, None)
    assert time.monotonic() - t0 < 5.0


@pytest.mark.skipif(os.name == "nt", reason="POSIX process groups")
def test_cancel_kills_orphaned_grandchildren(tmp_path):
    """The child's SLEAP service is a grandchild: escalation must signal the
    whole session group, not just the child pid, or it is left orphaned."""
    specs = [_spec(tmp_path, "a", mode="orphan_maker")]
    stop = {"flag": False}
    import threading

    threading.Timer(0.5, lambda: stop.__setitem__("flag", True)).start()
    result = run_batch_fanout(
        specs,
        FanoutOptions(
            jobs=1,
            run_dir=tmp_path / "run",
            child_command=_fake_command,
            sigint_grace_s=0.3,
            term_grace_s=0.3,
        ),
        should_stop=lambda: stop["flag"],
    )
    assert result.cancelled
    log = result.jobs[0].log_path.read_text()
    grandchild_pid = int(log.split("GRANDCHILD=")[1].split("\n")[0])

    deadline = time.monotonic() + 3.0
    while time.monotonic() < deadline:
        try:
            os.kill(grandchild_pid, 0)
        except ProcessLookupError:
            return  # reaped: the group signal reached it
        except PermissionError:
            return  # pid recycled to another owner; no longer ours
        time.sleep(0.05)
    # still alive -> orphaned. Clean up so the test does not leak a process.
    try:
        os.kill(grandchild_pid, 9)
    except OSError:
        pass
    raise AssertionError(f"grandchild {grandchild_pid} survived cancellation")


def test_launch_failure_is_contained_and_running_jobs_finish(tmp_path):
    """An exception while launching must not abandon already-running children."""
    specs = [
        _spec(tmp_path, "a", sleep=0.5),
        _spec(tmp_path, "b"),
        _spec(tmp_path, "c"),
        _spec(tmp_path, "d"),
    ]

    def _flaky_command(spec, config_json):
        if spec.config["name"] == "b":
            raise OSError("cannot spawn: simulated fork failure")
        return _fake_command(spec, config_json)

    rec = _Recorder()
    result = run_batch_fanout(
        specs,
        FanoutOptions(jobs=2, run_dir=tmp_path / "run", child_command=_flaky_command),
        events=rec,
    )
    assert not result.success
    by_name = {r.spec.config["name"]: r for r in result.jobs}
    # the child that was already running finished normally
    assert by_name["a"].success and by_name["a"].returncode == 0
    # the failed launch is reported, not raised
    assert by_name["b"].returncode is None and not by_name["b"].success
    assert by_name["b"].error.startswith("launch failed")
    # the failure halts further launches
    assert by_name["c"].error == "not started" and by_name["c"].returncode is None
    assert by_name["d"].error == "not started"
    # every job, including the failed launch, was announced as finished
    assert {r.spec.config["name"] for r in rec.finished} == {"a", "b"}


def test_cancel_reports_already_exited_child_as_success(tmp_path):
    """A child that exited 0 before the stop signal is a success, not cancelled."""
    specs = [_spec(tmp_path, "a", sleep=0.05), _spec(tmp_path, "b", mode="trap_sigint")]
    stop = {"flag": False}
    import threading

    threading.Timer(0.5, lambda: stop.__setitem__("flag", True)).start()
    result = run_batch_fanout(
        specs,
        FanoutOptions(
            jobs=2,
            run_dir=tmp_path / "run",
            child_command=_fake_command,
            # poll slowly so "a" exits during the sleep and the cancel branch --
            # not the reap loop -- is what observes it.
            poll_s=2.0,
            sigint_grace_s=5,
        ),
        should_stop=lambda: stop["flag"],
    )
    assert result.cancelled and not result.success
    by_name = {r.spec.config["name"]: r for r in result.jobs}
    assert by_name["a"].returncode == 0
    assert by_name["a"].success, "a exited cleanly before the signal"
    assert by_name["a"].error is None
    assert by_name["b"].returncode == 130 and not by_name["b"].success


def test_events_protocol_documents_threading_contract():
    from hydra_suite.trackerkit.batch_fanout import FanoutEvents

    doc = FanoutEvents.__doc__ or ""
    assert "thread" in doc.lower()


def test_module_imports_no_qt():
    import ast

    import hydra_suite.trackerkit.batch_fanout as mod
    import hydra_suite.trackerkit.batch_plan as plan_mod

    for module in (mod, plan_mod):
        tree = ast.parse(Path(module.__file__).read_text())
        for node in ast.walk(tree):
            names = []
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module or ""]
            assert not any(
                n.startswith(("PySide6", "PyQt")) for n in names
            ), module.__file__
