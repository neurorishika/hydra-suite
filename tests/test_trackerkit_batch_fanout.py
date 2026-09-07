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
    decide_gpu_slots,
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
if mode in ("orphan_maker", "orphan_then_crash", "orphan_then_exit0"):
    import subprocess
    kid = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    print("GRANDCHILD=%d" % kid.pid, flush=True)
print("PID=%d" % os.getpid(), flush=True)
print("2026-01-01 - x - INFO - [track forward] 10% starting", flush=True)
print("GPU=" + os.environ.get("CUDA_VISIBLE_DEVICES", "<unset>"), flush=True)
print("OMP=" + os.environ.get("OMP_NUM_THREADS", "<unset>"), flush=True)
time.sleep(sleep)
print("2026-01-01 - x - INFO - [post] 90% merging", flush=True)
if mode == "orphan_then_crash":
    os._exit(3)
if mode == "orphan_then_exit0":
    os._exit(0)
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


def _grandchild_pid(log_path: Path) -> int:
    text = log_path.read_text()
    assert "GRANDCHILD=" in text, text
    return int(text.split("GRANDCHILD=")[1].split("\n")[0])


def _assert_pid_reaped(pid: int, what: str, timeout: float = 3.0) -> None:
    """Fail (after cleaning up) unless *pid* disappears within *timeout*."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            os.kill(pid, 0)
        except (ProcessLookupError, PermissionError):
            return
        time.sleep(0.05)
    try:
        os.kill(pid, 9)
    except OSError:
        pass
    raise AssertionError(f"{what}: pid {pid} survived")


@pytest.mark.skipif(os.name == "nt", reason="POSIX process groups")
def test_child_that_exits_on_its_own_does_not_strand_its_group(tmp_path):
    """A child that dies by itself (crash/OOM/traceback) is reaped by poll(),
    never signalled -- so its SLEAP-service grandchild outlives the batch
    unless ``_finish`` takes the whole session group down."""
    specs = [_spec(tmp_path, "a", mode="orphan_then_crash", sleep=0.2)]
    result = run_batch_fanout(
        specs,
        FanoutOptions(
            jobs=1,
            run_dir=tmp_path / "run",
            child_command=_fake_command,
            poll_s=0.05,
        ),
    )
    assert result.jobs[0].returncode == 3 and not result.success
    _assert_pid_reaped(
        _grandchild_pid(result.jobs[0].log_path),
        "grandchild of a child that exited on its own",
    )


@pytest.mark.skipif(os.name == "nt", reason="POSIX process groups")
def test_cancel_reaps_group_of_a_child_that_already_exited(tmp_path):
    """On cancel, a child already in the ``already_exited`` snapshot is skipped
    by ``_stop_children`` (it is not ``_alive()``), so its group must be reaped
    on the terminal transition itself."""
    specs = [
        _spec(tmp_path, "a", mode="orphan_then_exit0", sleep=0.2),
        _spec(tmp_path, "b", sleep=5.0),
    ]
    stop = {"flag": False}
    import threading

    threading.Timer(0.4, lambda: stop.__setitem__("flag", True)).start()
    result = run_batch_fanout(
        specs,
        FanoutOptions(
            jobs=2,
            run_dir=tmp_path / "run",
            child_command=_fake_command,
            # Poll slowly so "a" is observed by the CANCEL branch, not the reap.
            poll_s=1.0,
            sigint_grace_s=0.5,
            term_grace_s=0.5,
        ),
        should_stop=lambda: stop["flag"],
    )
    assert result.cancelled
    by_name = {r.spec.config["name"]: r for r in result.jobs}
    assert by_name["a"].returncode == 0 and by_name["a"].success
    _assert_pid_reaped(
        _grandchild_pid(by_name["a"].log_path),
        "grandchild of an already-exited child at cancel time",
    )


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


@pytest.mark.skipif(os.name == "nt", reason="POSIX signals")
def test_unexpected_raise_tears_down_running_children(tmp_path):
    """A raise anywhere in the scheduling loop must not strand GPU children.

    ``should_stop`` throwing on its second call stands in for any unexpected
    bug (or a KeyboardInterrupt arriving mid-loop): the exception must
    propagate to the caller AND the running child must be dead, because a
    stranded child holds a whole GPU that nothing will ever reclaim.
    """
    specs = [_spec(tmp_path, "a", mode="trap_sigint")]
    recorder = _Recorder()

    def _child_pid():
        for _, line in list(recorder.logs):
            if line.startswith("PID="):
                return int(line.split("=", 1)[1])
        return None

    # Raise only ONCE THE CHILD HAS PRINTED ITS PID. The fake prints it strictly
    # after installing its SIGINT trap, so this makes the test deterministic:
    # raising earlier could deliver SIGINT before the trap exists, killing the
    # child by default handler with no pid ever logged. The wall-clock ceiling
    # keeps a broken child from hanging the test instead of failing it.
    deadline = time.monotonic() + 5.0

    def _should_stop():
        if _child_pid() is not None or time.monotonic() > deadline:
            raise RuntimeError("scheduler blew up")
        return False

    with pytest.raises(RuntimeError, match="scheduler blew up"):
        run_batch_fanout(
            specs,
            FanoutOptions(
                jobs=1,
                run_dir=tmp_path / "run",
                child_command=_fake_command,
                poll_s=0.05,
                sigint_grace_s=2,
                term_grace_s=1,
            ),
            events=recorder,
            should_stop=_should_stop,
        )

    assert recorder.started, "the child must have been launched before the raise"
    pid = _child_pid()
    assert pid is not None, "child never reported its pid"

    deadline = time.monotonic() + 3.0
    gone = False
    while time.monotonic() < deadline:
        try:
            os.kill(pid, 0)
        except (ProcessLookupError, PermissionError):
            gone = True
            break
        time.sleep(0.05)
    assert gone, f"child pid {pid} survived the scheduler raise"


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


def test_temp_run_dir_is_removed_on_success_and_kept_on_failure(tmp_path, monkeypatch):
    """One temp dir of effective configs was leaked per fan-out run."""
    import tempfile

    import hydra_suite.trackerkit.batch_fanout as bf

    created: list[str] = []
    real_mkdtemp = tempfile.mkdtemp

    def _mkdtemp(*args, **kwargs):
        kwargs["dir"] = str(tmp_path)
        path = real_mkdtemp(*args, **kwargs)
        created.append(path)
        return path

    monkeypatch.setattr(bf.tempfile, "mkdtemp", _mkdtemp)

    ok = run_batch_fanout(
        [_spec(tmp_path, "a")], FanoutOptions(jobs=1, child_command=_fake_command)
    )
    assert ok.success
    assert created and not Path(created[-1]).exists()

    bad = run_batch_fanout(
        [_spec(tmp_path, "b", mode="fail")],
        FanoutOptions(jobs=1, child_command=_fake_command),
    )
    assert not bad.success
    kept = Path(created[-1])
    assert kept.is_dir(), "a failed run must keep its effective configs"
    assert (kept / "job_1_config.json").exists()


def test_two_runs_in_the_same_second_do_not_share_a_log_file(tmp_path):
    """The timestamp used to have 1 s resolution, so a second fan-out of the
    same video appended into the first run's log."""
    first = run_batch_fanout(
        [_spec(tmp_path, "a")],
        FanoutOptions(jobs=1, run_dir=tmp_path / "r1", child_command=_fake_command),
    )
    second = run_batch_fanout(
        [_spec(tmp_path, "a")],
        FanoutOptions(jobs=1, run_dir=tmp_path / "r2", child_command=_fake_command),
    )
    a, b = first.jobs[0].log_path, second.jobs[0].log_path
    assert a != b, a
    assert "job1" in a.name and "job1" in b.name
    for log in (a, b):
        assert log.read_text().count("# trackerkit fan-out job") == 1


# --- GPU slot decision (shared by the CLI and the GUI) ----------------------

_GPUS = [CudaDevice(0, "GPU-aaaa", "x"), CudaDevice(1, "GPU-bbbb", "x")]


def test_decide_gpu_slots_resolves_selectors_against_visible_devices():
    assert decide_gpu_slots(["1"], _GPUS, host_has_cuda=True) == [_GPUS[1]]
    assert decide_gpu_slots(["auto"], _GPUS, host_has_cuda=True) == _GPUS


def test_decide_gpu_slots_no_request_is_unpinned():
    assert decide_gpu_slots([], _GPUS, host_has_cuda=True) == []
    assert decide_gpu_slots([], [], host_has_cuda=False) == []


def test_decide_gpu_slots_aborts_when_smi_is_blind_on_a_cuda_host():
    """nvidia-smi missing or timing out on a CUDA host must NOT silently run
    unpinned: every child would fall back to cuda:0 and contend for one GPU."""
    with pytest.raises(ValueError) as exc:
        decide_gpu_slots(["auto"], [], host_has_cuda=True)
    assert "nvidia-smi" in str(exc.value)
    with pytest.raises(ValueError):
        decide_gpu_slots(["0"], [], host_has_cuda=True)


def test_decide_gpu_slots_on_a_non_cuda_host():
    # "auto" is best-effort: nothing to pin to, so run unpinned.
    assert decide_gpu_slots(["auto"], [], host_has_cuda=False) == []
    # An explicit device the host does not have stays an error.
    with pytest.raises(ValueError):
        decide_gpu_slots(["0"], [], host_has_cuda=False)
