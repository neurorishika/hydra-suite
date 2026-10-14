"""Fan-out ``--max-jobs`` (GPU phase semaphore) and ``--keep-going``.

Fake children only (no models, videos or GPUs), as in
``test_trackerkit_batch_fanout.py``.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import pytest

import hydra_suite
from hydra_suite.runtime import gpu_phase_lock as gpl
from hydra_suite.runtime.cuda_devices import CudaDevice
from hydra_suite.trackerkit import cli
from hydra_suite.trackerkit.app import parse_arguments
from hydra_suite.trackerkit.batch_fanout import (
    FanoutOptions,
    phase_lock_slots,
    plan_fanout_cpu,
    resolve_slot_count,
    run_batch_fanout,
    scheduler_slots,
)
from hydra_suite.trackerkit.batch_plan import BatchJobSpec

_SRC = str(Path(hydra_suite.__file__).resolve().parents[1])
_GPUS = [CudaDevice(0, "GPU-aaaa", "x"), CudaDevice(1, "GPU-bbbb", "x")]

# A child that does a "GPU phase" through the real semaphore and logs the
# interval it held it, then a "CPU phase" outside it.
FAKE_CHILD = r"""
import json, os, sys, time
cfg = json.load(open(sys.argv[1]))
sys.path.insert(0, sys.argv[2])
from hydra_suite.runtime.gpu_phase_lock import gpu_phase
print("GPU=" + os.environ.get("CUDA_VISIBLE_DEVICES", "<unset>"), flush=True)
print("SLOTS=" + os.environ.get("HYDRA_GPU_PHASE_SLOTS", "<unset>"), flush=True)
if cfg.get("mode") == "fail":
    print("2026-01-01 - x - ERROR - Tracker CLI failed for v: boom", flush=True)
    sys.exit(3)
with gpu_phase("inference", poll_s=0.02):
    t0 = time.time()
    time.sleep(float(cfg.get("gpu_s", 0.0)))
    t1 = time.time()
with open(cfg["intervals"], "a") as fh:
    fh.write(json.dumps([os.environ.get("CUDA_VISIBLE_DEVICES"), t0, t1]) + "\n")
time.sleep(float(cfg.get("cpu_s", 0.0)))
print("2026-01-01 - x - INFO - Tracker CLI completed: video=%s | rows=1" % cfg["name"], flush=True)
"""


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for var in gpl.ENV_VARS:
        monkeypatch.delenv(var, raising=False)
    monkeypatch.delenv("HYDRA_CPU_BUDGET", raising=False)
    monkeypatch.delenv("HYDRA_CPU_THREADS", raising=False)


def _spec(tmp_path: Path, name: str, **cfg) -> BatchJobSpec:
    video = tmp_path / f"{name}.mp4"
    video.write_bytes(b"\x00")
    cfg.setdefault("intervals", str(tmp_path / "intervals.jsonl"))
    return BatchJobSpec(
        index=0,
        video_path=str(video),
        config_path=None,
        config={"name": name, **cfg},
        provenance="own-sidecar",
    )


def _fake_command(spec: BatchJobSpec, config_json: Path) -> list[str]:
    return [sys.executable, "-c", FAKE_CHILD, str(config_json), _SRC]


def _intervals(tmp_path: Path) -> list[tuple[str, float, float]]:
    path = tmp_path / "intervals.jsonl"
    return [tuple(json.loads(line)) for line in path.read_text().splitlines()]


def _max_overlap(intervals) -> int:
    events = sorted(
        [(t0, 1) for _, t0, _ in intervals] + [(t1, -1) for _, _, t1 in intervals]
    )
    cur = best = 0
    for _, step in events:
        cur += step
        best = max(best, cur)
    return best


def _record_launches(monkeypatch):
    import subprocess

    real_popen = subprocess.Popen
    launches: list[dict[str, str]] = []

    def _popen(command, **kwargs):
        env = kwargs["env"]
        launches.append({k: v for k, v in env.items() if os.environ.get(k) != v})
        return real_popen([sys.executable, "-c", "print('ok')"], **kwargs)

    monkeypatch.setattr(subprocess, "Popen", _popen)
    return launches


# --- slot arithmetic ----------------------------------------------------------


def test_max_jobs_spreads_children_round_robin_over_the_gpus():
    opts = FanoutOptions(gpus=_GPUS, jobs_per_gpu=1, max_jobs=5)
    assert [g.uuid for g in scheduler_slots(opts)] == [
        "GPU-aaaa",
        "GPU-bbbb",
        "GPU-aaaa",
        "GPU-bbbb",
        "GPU-aaaa",
    ]
    assert resolve_slot_count(opts) == 5
    assert phase_lock_slots(opts, scheduler_slots(opts)) == 1


def test_max_jobs_at_or_below_gpu_slots_is_plain_jobs_and_unlocked():
    for m in (1, 2, 3, 4):
        opts = FanoutOptions(gpus=_GPUS, jobs_per_gpu=2, max_jobs=m)
        plain = FanoutOptions(gpus=_GPUS, jobs_per_gpu=2, jobs=m)
        assert scheduler_slots(opts) == scheduler_slots(plain)
        assert phase_lock_slots(opts, scheduler_slots(opts)) is None


def test_without_max_jobs_slots_and_locks_are_unchanged():
    for jobs in (None, 1, 3, 99):
        opts = FanoutOptions(gpus=_GPUS, jobs=jobs, jobs_per_gpu=2)
        assert phase_lock_slots(opts, scheduler_slots(opts)) is None


def test_plan_fanout_cpu_splits_the_budget_over_max_jobs():
    opts, plan = plan_fanout_cpu(
        FanoutOptions(gpus=_GPUS, max_jobs=8),
        n_videos=20,
        cpu_budget=32,
        env={},
        usable=64,
    )
    assert (opts.max_jobs, opts.threads_per_job, plan.clamped) == (8, 4, False)
    assert resolve_slot_count(opts) == 8


def test_plan_fanout_cpu_clamps_max_jobs_to_the_budget_and_warns():
    opts, plan = plan_fanout_cpu(
        FanoutOptions(gpus=_GPUS, max_jobs=12),
        n_videos=20,
        cpu_budget=6,
        env={},
        usable=64,
    )
    assert plan.clamped and plan.warnings
    assert opts.max_jobs == plan.jobs == 6
    assert resolve_slot_count(opts) == 6


# --- the scheduler with real (fake) children ----------------------------------


def test_max_jobs_runs_m_children_and_k_gpu_phases_per_gpu(tmp_path):
    """4 children on 1 GPU at K=1: all run at once, GPU phases never overlap."""
    specs = [_spec(tmp_path, n, gpu_s=0.3, cpu_s=0.6) for n in "abcd"]
    t0 = time.monotonic()
    result = run_batch_fanout(
        specs,
        FanoutOptions(
            gpus=_GPUS[:1],
            jobs_per_gpu=1,
            max_jobs=4,
            threads_per_job=1,
            run_dir=tmp_path / "run",
            child_command=_fake_command,
        ),
    )
    wall = time.monotonic() - t0
    assert result.success, [r.error for r in result.jobs]
    intervals = _intervals(tmp_path)
    assert len(intervals) == 4
    assert _max_overlap(intervals) == 1, "K=1: GPU phases must be serialised"
    # Serial would be 4 x 0.9 = 3.6 s; overlapping CPU phases bring it to
    # about 4 x 0.3 + 0.6 = 1.8 s.
    assert wall < 3.2, f"children should overlap their CPU phases; took {wall:.1f}s"
    for job in result.jobs:
        log = job.log_path.read_text()
        assert "GPU=GPU-aaaa" in log and "SLOTS=1" in log


def test_max_jobs_two_gpus_k1_allows_one_phase_per_gpu(tmp_path):
    specs = [_spec(tmp_path, n, gpu_s=0.3) for n in "abcd"]
    result = run_batch_fanout(
        specs,
        FanoutOptions(
            gpus=_GPUS,
            jobs_per_gpu=1,
            max_jobs=4,
            threads_per_job=1,
            run_dir=tmp_path / "run",
            child_command=_fake_command,
        ),
    )
    assert result.success
    intervals = _intervals(tmp_path)
    for uuid in ("GPU-aaaa", "GPU-bbbb"):
        mine = [iv for iv in intervals if iv[0] == uuid]
        assert len(mine) == 2
        assert _max_overlap(mine) == 1


def test_max_jobs_children_get_the_lock_env(tmp_path, monkeypatch):
    launches = _record_launches(monkeypatch)
    run_dir = tmp_path / "run"
    result = run_batch_fanout(
        [_spec(tmp_path, n) for n in "abc"],
        FanoutOptions(gpus=_GPUS[:1], max_jobs=3, threads_per_job=1, run_dir=run_dir),
    )
    assert result.success and len(launches) == 3
    for delta in launches:
        assert delta[gpl.ENV_GPU] == "GPU-aaaa"
        assert delta[gpl.ENV_SLOTS] == "1"
        assert Path(delta[gpl.ENV_LOCK_DIR]) == run_dir / "gpu_phase_locks"


@pytest.mark.parametrize(
    "extra",
    [{}, {"jobs": 1}, {"jobs_per_gpu": 2}, {"max_jobs": 2}, {"max_jobs": 1}],
)
def test_no_lock_env_unless_children_outnumber_gpu_slots(tmp_path, monkeypatch, extra):
    launches = _record_launches(monkeypatch)
    result = run_batch_fanout(
        [_spec(tmp_path, n) for n in "ab"],
        FanoutOptions(gpus=_GPUS, threads_per_job=1, run_dir=tmp_path / "r", **extra),
    )
    assert result.success and launches
    for delta in launches:
        assert not set(delta) & set(gpl.ENV_VARS)


# --- --keep-going -------------------------------------------------------------


def test_default_still_halts_after_a_failure(tmp_path):
    specs = [_spec(tmp_path, "a", mode="fail")] + [_spec(tmp_path, n) for n in "bc"]
    result = run_batch_fanout(
        specs,
        FanoutOptions(jobs=1, run_dir=tmp_path / "run", child_command=_fake_command),
    )
    by_name = {r.spec.config["name"]: r for r in result.jobs}
    assert by_name["a"].returncode == 3
    assert by_name["b"].returncode is None and by_name["c"].returncode is None


def test_keep_going_continues_the_queue_after_a_failure(tmp_path):
    specs = [_spec(tmp_path, "a", mode="fail")] + [_spec(tmp_path, n) for n in "bc"]
    result = run_batch_fanout(
        specs,
        FanoutOptions(
            jobs=1,
            keep_going=True,
            run_dir=tmp_path / "run",
            child_command=_fake_command,
        ),
    )
    by_name = {r.spec.config["name"]: r for r in result.jobs}
    assert by_name["a"].returncode == 3 and not by_name["a"].success
    assert by_name["b"].success and by_name["c"].success
    assert not result.success  # a failure still fails the batch


def test_keep_going_continues_after_a_launch_failure(tmp_path):
    calls = []

    def _command(spec, config_json):
        calls.append(spec.config["name"])
        if spec.config["name"] == "a":
            raise RuntimeError("cannot launch")
        return _fake_command(spec, config_json)

    specs = [_spec(tmp_path, n) for n in "abc"]
    result = run_batch_fanout(
        specs,
        FanoutOptions(
            jobs=1, keep_going=True, run_dir=tmp_path / "run", child_command=_command
        ),
    )
    by_name = {r.spec.config["name"]: r for r in result.jobs}
    assert "launch failed" in by_name["a"].error
    assert by_name["b"].success and by_name["c"].success


# --- CLI wiring -----------------------------------------------------------------


def _videos(tmp_path, n=2):
    out = []
    for i in range(n):
        p = tmp_path / f"v{i}.mp4"
        p.write_bytes(b"\x00")
        out.append(str(p))
    return out


def test_parse_max_jobs_and_keep_going(tmp_path):
    v = _videos(tmp_path, 1)
    args = parse_arguments(
        ["track", *v, "--gpus", "0", "--max-jobs", "6", "--keep-going"]
    )
    assert args.max_jobs == 6 and args.keep_going is True
    args = parse_arguments(["track", *v])
    assert args.max_jobs is None and args.keep_going is False


@pytest.mark.parametrize(
    "tail",
    [
        ["--max-jobs", "4"],  # no --gpus
        ["--gpus", "0", "--max-jobs", "0"],
        ["--gpus", "0", "--max-jobs", "4", "--jobs", "2"],
    ],
)
def test_parse_rejects_bad_max_jobs(tmp_path, tail):
    with pytest.raises(SystemExit):
        parse_arguments(["track", *_videos(tmp_path, 1), *tail])


def test_cli_passes_max_jobs_and_keep_going_to_the_fanout(tmp_path, monkeypatch):
    captured = {}
    monkeypatch.setattr(
        cli,
        "list_cuda_devices",
        lambda: [CudaDevice(i, f"GPU-{i}", "x") for i in (0, 1)],
    )
    monkeypatch.setattr(
        cli, "_run_fanout", lambda specs, options: captured.setdefault("o", options)
    )
    monkeypatch.setattr("hydra_suite.runtime.cpu_budget.usable_cores", lambda **_: 64)
    cli.run_tracking_cli(
        _videos(tmp_path, 10), gpus="0,1", max_jobs=6, keep_going=True, cpu_budget=24
    )
    opts = captured["o"]
    assert (opts.max_jobs, opts.threads_per_job, opts.keep_going) == (6, 4, True)
    assert resolve_slot_count(opts) == 6


def test_cli_default_options_have_no_max_jobs(tmp_path, monkeypatch):
    captured = {}
    monkeypatch.setattr(
        cli,
        "list_cuda_devices",
        lambda: [CudaDevice(i, f"GPU-{i}", "x") for i in (0, 1)],
    )
    monkeypatch.setattr(
        cli, "_run_fanout", lambda specs, options: captured.setdefault("o", options)
    )
    monkeypatch.setattr("hydra_suite.runtime.cpu_budget.usable_cores", lambda **_: 64)
    cli.run_tracking_cli(_videos(tmp_path, 4), gpus="0,1")
    opts = captured["o"]
    assert opts.max_jobs is None and opts.keep_going is False


def test_cli_rejects_max_jobs_without_gpus(tmp_path):
    with pytest.raises(ValueError, match="--max-jobs needs --gpus"):
        cli.run_tracking_cli(_videos(tmp_path), jobs=2, max_jobs=4)


def test_sequential_keep_going_runs_every_video(tmp_path, monkeypatch):
    ran = []

    def _session(session):
        ran.append(session)
        return {"success": len(ran) != 1, "error": "boom", "lines": []}

    from types import SimpleNamespace

    monkeypatch.setattr(
        cli,
        "load_tracker_cli_session",
        lambda v, **_: SimpleNamespace(video=v, config={}),
    )
    monkeypatch.setattr(cli, "run_headless_tracking_session", _session)
    assert cli.run_tracking_cli(_videos(tmp_path, 3), keep_going=True) == 1
    assert len(ran) == 3
    ran.clear()
    assert cli.run_tracking_cli(_videos(tmp_path, 3)) == 1
    assert len(ran) == 1


def test_job_run_forwards_max_jobs_and_keep_going():
    from hydra_suite.trackerkit import job_cli

    args = parse_arguments(
        ["job", "run", "/tmp/j", "--gpus", "0", "--max-jobs", "6", "--keep-going"]
    )
    flags = job_cli._fanout_flags(args)
    assert flags[flags.index("--max-jobs") + 1] == "6"
    assert "--keep-going" in flags
