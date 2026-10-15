"""``--max-jobs auto``: resolution rule, parsing and CLI wiring.

No models, videos or GPUs: the fan-out runner is stubbed out.
"""

from __future__ import annotations

import logging

import pytest

from hydra_suite.runtime.cuda_devices import CudaDevice
from hydra_suite.trackerkit import cli
from hydra_suite.trackerkit.app import parse_arguments
from hydra_suite.trackerkit.batch_fanout import (
    MAX_JOBS_AUTO_FACTOR,
    MAX_JOBS_AUTO_MIN_THREADS,
    parse_max_jobs,
    phase_lock_slots,
    resolve_auto_max_jobs,
    resolve_slot_count,
    scheduler_slots,
)


def _auto(**kw):
    base = dict(n_gpus=5, jobs_per_gpu=2, budget_cores=96, n_videos=100)
    base.update(kw)
    return resolve_auto_max_jobs(**base)


# --- resolution rule ----------------------------------------------------------


def test_constants_are_the_approved_rule():
    assert (MAX_JOBS_AUTO_FACTOR, MAX_JOBS_AUTO_MIN_THREADS) == (1.5, 4)


def test_auto_is_one_and_a_half_times_the_gpu_slots():
    m, reason = _auto()
    assert m == 15
    assert reason == ("1.5 x 5 GPUs x 2; 96-core budget allows up to 24 at >=4 threads")


def test_auto_rounds_up():
    assert _auto(n_gpus=1, jobs_per_gpu=1)[0] == 2  # ceil(1.5)
    assert _auto(n_gpus=3, jobs_per_gpu=1)[0] == 5  # ceil(4.5)


def test_auto_is_capped_so_each_child_keeps_four_threads():
    m, reason = _auto(budget_cores=48)  # 48 // 4 = 12 < 15
    assert m == 12
    assert "allows up to 12" in reason


def test_auto_never_goes_below_the_gpu_slots():
    m, reason = _auto(budget_cores=20)  # 20 // 4 = 5 < 10 = GPUs x K
    assert m == 10
    assert "floored at GPUs x K = 10" in reason
    assert "GPU phase lock not engaged" in reason


def test_auto_never_exceeds_the_video_count():
    m, reason = _auto(n_videos=12)
    assert m == 12
    assert "capped to 12 video(s)" in reason


def test_auto_with_fewer_videos_than_gpu_slots_does_not_engage_the_lock():
    m, reason = _auto(n_videos=3)
    assert m == 3 and "GPU phase lock not engaged" in reason


def test_auto_with_threads_per_job_caps_at_budget_over_that():
    m, reason = _auto(threads_per_job=8)  # 96 // 8 = 12
    assert m == 12
    assert reason.endswith("allows up to 12 at --threads-per-job 8")
    assert _auto(threads_per_job=2)[0] == 15  # 48 allowed, 1.5x rule binds
    assert _auto(threads_per_job=32)[0] == 10  # 3 allowed, floored at GPUs x K


# --- parsing --------------------------------------------------------------------


def test_parse_max_jobs_values():
    assert parse_max_jobs("auto") == "auto"
    assert parse_max_jobs(" AUTO ") == "auto"
    assert parse_max_jobs("6") == 6
    assert parse_max_jobs(6) == 6
    for bad in ("0", "-1", "abc", "1.5"):
        with pytest.raises(ValueError):
            parse_max_jobs(bad)


def _videos(tmp_path, n=2):
    out = []
    for i in range(n):
        p = tmp_path / f"v{i}.mp4"
        p.write_bytes(b"\x00")
        out.append(str(p))
    return out


def test_parse_arguments_accepts_auto_and_int(tmp_path):
    v = _videos(tmp_path, 1)
    args = parse_arguments(["track", *v, "--gpus", "0", "--max-jobs", "auto"])
    assert args.max_jobs == "auto"
    args = parse_arguments(["track", *v, "--gpus", "0", "--max-jobs", "6"])
    assert args.max_jobs == 6


@pytest.mark.parametrize(
    "tail",
    [
        ["--max-jobs", "auto"],  # no --gpus
        ["--gpus", "0", "--max-jobs", "auto", "--jobs", "2"],
        ["--gpus", "0", "--max-jobs", "abc"],
        ["--gpus", "0", "--max-jobs", "0"],
    ],
)
def test_parse_arguments_rejects_bad_auto(tmp_path, tail):
    with pytest.raises(SystemExit):
        parse_arguments(["track", *_videos(tmp_path, 1), *tail])


@pytest.mark.parametrize(
    "tail",
    [
        ["--max-jobs", "auto"],
        ["--gpus", "0", "--max-jobs", "auto", "--jobs", "2"],
    ],
)
def test_job_run_rejects_bad_auto(tail):
    with pytest.raises(SystemExit):
        parse_arguments(["job", "run", "/tmp/j", *tail])


def test_job_run_forwards_max_jobs_auto():
    from hydra_suite.trackerkit import job_cli

    args = parse_arguments(
        ["job", "run", "/tmp/j", "--gpus", "0", "--max-jobs", "auto"]
    )
    flags = job_cli._fanout_flags(args)
    assert flags[flags.index("--max-jobs") + 1] == "auto"


# --- CLI wiring -----------------------------------------------------------------


def _run_cli(tmp_path, monkeypatch, *, n_gpus, n_videos, cores, **kw):
    captured = {}
    monkeypatch.setattr(
        cli,
        "list_cuda_devices",
        lambda: [CudaDevice(i, f"GPU-{i}", "x") for i in range(n_gpus)],
    )
    monkeypatch.setattr(
        cli, "_run_fanout", lambda specs, options: captured.setdefault("o", options)
    )
    monkeypatch.setattr(
        "hydra_suite.runtime.cpu_budget.usable_cores", lambda **_: cores
    )
    monkeypatch.delenv("HYDRA_CPU_BUDGET", raising=False)
    monkeypatch.delenv("HYDRA_CPU_THREADS", raising=False)
    gpus = ",".join(str(i) for i in range(n_gpus))
    cli.run_tracking_cli(_videos(tmp_path, n_videos), gpus=gpus, **kw)
    return captured["o"]


def test_cli_resolves_auto_and_logs_why(tmp_path, monkeypatch, caplog):
    caplog.set_level(logging.INFO, logger=cli.__name__)
    opts = _run_cli(
        tmp_path,
        monkeypatch,
        n_gpus=5,
        n_videos=40,
        cores=96,
        jobs_per_gpu=2,
        max_jobs="auto",
    )
    assert opts.max_jobs == 15
    assert resolve_slot_count(opts) == 15
    assert opts.threads_per_job == 96 // 15
    assert phase_lock_slots(opts, scheduler_slots(opts)) == 2
    assert (
        "--max-jobs auto -> 15 (1.5 x 5 GPUs x 2; 96-core budget allows up to "
        "24 at >=4 threads)" in caplog.text
    )


def test_cli_auto_uses_the_cpu_budget_flag(tmp_path, monkeypatch):
    opts = _run_cli(
        tmp_path,
        monkeypatch,
        n_gpus=2,
        n_videos=40,
        cores=96,
        jobs_per_gpu=2,
        max_jobs="auto",
        cpu_budget=20,
    )
    assert opts.max_jobs == 5  # 1.5 x 4 = 6, but 20 // 4 = 5
    assert opts.threads_per_job == 4


def test_cli_auto_below_gpu_slots_is_the_default_path(tmp_path, monkeypatch):
    kw = dict(n_gpus=2, n_videos=40, cores=96, jobs_per_gpu=2, cpu_budget=8)
    auto = _run_cli(tmp_path, monkeypatch, max_jobs="auto", **kw)
    default = _run_cli(tmp_path, monkeypatch, **kw)
    assert auto == default
    assert auto.max_jobs is None


def test_cli_integer_max_jobs_is_unchanged(tmp_path, monkeypatch):
    opts = _run_cli(
        tmp_path,
        monkeypatch,
        n_gpus=2,
        n_videos=10,
        cores=64,
        max_jobs=6,
        cpu_budget=24,
    )
    assert (opts.max_jobs, opts.threads_per_job) == (6, 4)


def test_cli_rejects_auto_without_gpus_or_with_jobs(tmp_path):
    with pytest.raises(ValueError, match="--max-jobs needs --gpus"):
        cli.run_tracking_cli(_videos(tmp_path), jobs=2, max_jobs="auto")
    with pytest.raises(ValueError, match="not both"):
        cli.run_tracking_cli(_videos(tmp_path), gpus="0", jobs=2, max_jobs="auto")
    with pytest.raises(ValueError, match="integer >= 1 or 'auto'"):
        cli.run_tracking_cli(_videos(tmp_path), gpus="0", max_jobs="lots")
