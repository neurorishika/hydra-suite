from __future__ import annotations

import pytest

from hydra_suite.runtime.cuda_devices import CudaDevice
from hydra_suite.trackerkit import cli
from hydra_suite.trackerkit.app import parse_arguments
from hydra_suite.trackerkit.batch_fanout import FanoutJobResult, FanoutResult


def _videos(tmp_path, n=2):
    out = []
    for i in range(n):
        p = tmp_path / f"v{i}.mp4"
        p.write_bytes(b"\x00")
        out.append(str(p))
    return out


def test_parse_track_flags(tmp_path):
    v = _videos(tmp_path, 1)
    args = parse_arguments(
        ["track", v[0], "--gpus", "0-1", "--jobs", "3", "--threads-per-job", "8"]
    )
    assert args.gpus == "0-1" and args.jobs == 3 and args.threads_per_job == 8


def test_parse_track_defaults(tmp_path):
    v = _videos(tmp_path, 1)
    args = parse_arguments(["track", v[0]])
    # --jobs defaults to None ("unspecified"), which means one slot per selected
    # GPU and 1 when no GPUs were named.
    assert args.gpus is None and args.jobs is None and args.threads_per_job is None


def test_fanout_gating_rule():
    assert not cli.fanout_requested(None, 1)
    assert cli.fanout_requested("0", 1)
    assert cli.fanout_requested(None, 2)


def test_sequential_path_untouched_when_not_requested(tmp_path, monkeypatch):
    v = _videos(tmp_path, 2)
    calls = []
    monkeypatch.setattr(
        cli, "_run_sequential", lambda specs: calls.append(len(specs)) or 0
    )
    monkeypatch.setattr(
        cli, "_run_fanout", lambda *a, **k: pytest.fail("fan-out must not run")
    )
    assert cli.run_tracking_cli(v) == 0
    assert calls == [2]


def test_fanout_path_used_when_jobs_gt_1(tmp_path, monkeypatch):
    v = _videos(tmp_path, 2)
    seen = {}

    def _fake_fanout(specs, options, *, should_stop, events=None):
        seen["n"] = len(specs)
        seen["jobs"] = options.jobs
        seen["gpus"] = options.gpus
        seen["threads"] = options.threads_per_job
        return FanoutResult(
            jobs=[
                FanoutJobResult(
                    s, None, 0, True, tmp_path / "l", ["video=x"], None, 1.0
                )
                for s in specs
            ],
            cancelled=False,
        )

    monkeypatch.setattr(cli, "run_batch_fanout", _fake_fanout)
    monkeypatch.setattr(
        cli, "_run_sequential", lambda specs: pytest.fail("sequential must not run")
    )
    assert cli.run_tracking_cli(v, jobs=2, threads_per_job=4) == 0
    assert seen == {"n": 2, "jobs": 2, "gpus": [], "threads": 4}


def test_fanout_exit_code_1_when_any_job_fails(tmp_path, monkeypatch, capsys):
    v = _videos(tmp_path, 2)

    def _fake_fanout(specs, options, *, should_stop, events=None):
        jobs = [
            FanoutJobResult(
                specs[0], None, 0, True, tmp_path / "a.log", ["video=a"], None, 1.0
            ),
            FanoutJobResult(
                specs[1], None, 3, False, tmp_path / "b.log", [], "exit code 3", 1.0
            ),
        ]
        return FanoutResult(jobs=jobs, cancelled=False)

    monkeypatch.setattr(cli, "run_batch_fanout", _fake_fanout)
    assert cli.run_tracking_cli(v, jobs=2) == 1
    out = capsys.readouterr().out
    assert "FAIL" in out and "OK" in out and "b.log" in out


def test_gpus_on_host_without_cuda_is_an_error(tmp_path, monkeypatch):
    v = _videos(tmp_path, 1)
    monkeypatch.setattr(
        cli,
        "resolve_gpu_selectors",
        lambda sel: (_ for _ in ()).throw(ValueError("no CUDA")),
    )
    with pytest.raises(ValueError):
        cli.run_tracking_cli(v, gpus="0")


def _ok_result(specs, tmp_path):
    return FanoutResult(
        jobs=[
            FanoutJobResult(s, None, 0, True, tmp_path / "l", ["video=x"], None, 1.0)
            for s in specs
        ],
        cancelled=False,
    )


def _three_gpus():
    return [
        CudaDevice(0, "GPU-aaaa", "x"),
        CudaDevice(1, "GPU-bbbb", "x"),
        CudaDevice(2, "GPU-cccc", "x"),
    ]


def test_gpus_without_jobs_uses_one_slot_per_gpu(tmp_path, monkeypatch):
    v = _videos(tmp_path, 2)
    seen = {}

    def _fake_fanout(specs, options, *, should_stop, events=None):
        seen["jobs"] = options.jobs
        seen["gpus"] = len(options.gpus)
        return _ok_result(specs, tmp_path)

    monkeypatch.setattr(cli, "resolve_gpu_selectors", lambda sel: _three_gpus())
    monkeypatch.setattr(cli, "run_batch_fanout", _fake_fanout)
    assert cli.run_tracking_cli(v, gpus="0-2") == 0
    assert seen == {"jobs": 3, "gpus": 3}


def test_jobs_caps_below_gpu_count(tmp_path, monkeypatch):
    v = _videos(tmp_path, 2)
    seen = {}

    def _fake_fanout(specs, options, *, should_stop, events=None):
        seen["jobs"] = options.jobs
        return _ok_result(specs, tmp_path)

    monkeypatch.setattr(cli, "resolve_gpu_selectors", lambda sel: _three_gpus())
    monkeypatch.setattr(cli, "run_batch_fanout", _fake_fanout)
    assert cli.run_tracking_cli(v, gpus="0-2", jobs=2) == 0
    assert seen == {"jobs": 2}


def test_jobs_above_gpu_count_is_clamped_to_the_gpus(tmp_path, monkeypatch):
    v = _videos(tmp_path, 2)
    seen = {}

    def _fake_fanout(specs, options, *, should_stop, events=None):
        seen["jobs"] = options.jobs
        return _ok_result(specs, tmp_path)

    monkeypatch.setattr(cli, "resolve_gpu_selectors", lambda sel: _three_gpus())
    monkeypatch.setattr(cli, "run_batch_fanout", _fake_fanout)
    assert cli.run_tracking_cli(v, gpus="0-2", jobs=99) == 0
    assert seen == {"jobs": 3}
