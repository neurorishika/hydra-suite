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
    monkeypatch.setattr(cli, "list_cuda_devices", lambda: [])
    monkeypatch.setattr(cli, "host_has_cuda", lambda: False)
    with pytest.raises(ValueError):
        cli.run_tracking_cli(v, gpus="0")


def test_gpus_aborts_when_nvidia_smi_is_blind_on_a_cuda_host(tmp_path, monkeypatch):
    """nvidia-smi missing/timing out on a CUDA host must not run unpinned."""
    v = _videos(tmp_path, 1)
    monkeypatch.setattr(cli, "list_cuda_devices", lambda: [])
    monkeypatch.setattr(cli, "host_has_cuda", lambda: True)
    monkeypatch.setattr(
        cli, "run_batch_fanout", lambda *a, **k: pytest.fail("must not launch")
    )
    with pytest.raises(ValueError, match="nvidia-smi"):
        cli.run_tracking_cli(v, gpus="auto")


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

    monkeypatch.setattr(cli, "list_cuda_devices", _three_gpus)
    monkeypatch.setattr(cli, "run_batch_fanout", _fake_fanout)
    assert cli.run_tracking_cli(v, gpus="0-2") == 0
    assert seen == {"jobs": 3, "gpus": 3}


def test_jobs_caps_below_gpu_count(tmp_path, monkeypatch):
    v = _videos(tmp_path, 2)
    seen = {}

    def _fake_fanout(specs, options, *, should_stop, events=None):
        seen["jobs"] = options.jobs
        return _ok_result(specs, tmp_path)

    monkeypatch.setattr(cli, "list_cuda_devices", _three_gpus)
    monkeypatch.setattr(cli, "run_batch_fanout", _fake_fanout)
    assert cli.run_tracking_cli(v, gpus="0-2", jobs=2) == 0
    assert seen == {"jobs": 2}


def test_jobs_above_gpu_count_is_clamped_to_the_gpus(tmp_path, monkeypatch):
    v = _videos(tmp_path, 2)
    seen = {}

    def _fake_fanout(specs, options, *, should_stop, events=None):
        seen["jobs"] = options.jobs
        return _ok_result(specs, tmp_path)

    monkeypatch.setattr(cli, "list_cuda_devices", _three_gpus)
    monkeypatch.setattr(cli, "run_batch_fanout", _fake_fanout)
    assert cli.run_tracking_cli(v, gpus="0-2", jobs=99) == 0
    assert seen == {"jobs": 3}


def test_fanout_restores_every_signal_handler_it_installed(tmp_path, monkeypatch):
    """The CLI traps SIGINT/SIGTERM/SIGHUP so a scheduler kill or a closed
    terminal tears the GPU children down -- but it must hand the process back
    exactly as it found it."""
    import signal

    v = _videos(tmp_path, 2)
    watched = [signal.SIGINT, signal.SIGTERM]
    if hasattr(signal, "SIGHUP"):
        watched.append(signal.SIGHUP)
    before = {sig: signal.getsignal(sig) for sig in watched}
    during = {}

    def _fake_fanout(specs, options, **kw):
        during.update({sig: signal.getsignal(sig) for sig in watched})
        return _ok_result(specs, tmp_path)

    monkeypatch.setattr(cli, "run_batch_fanout", _fake_fanout)

    assert cli.run_tracking_cli(v, jobs=2) == 0

    # Not vacuous: every watched signal really was trapped for the duration.
    assert during and all(during[sig] is not before[sig] for sig in watched), during
    after = {sig: signal.getsignal(sig) for sig in watched}
    assert after == before, "fan-out leaked a signal handler"


def test_fanout_stop_signals_include_term_and_hup():
    import signal

    signals = cli._fanout_stop_signals()
    assert signal.SIGINT in signals and signal.SIGTERM in signals
    if hasattr(signal, "SIGHUP"):
        assert signal.SIGHUP in signals


def test_restore_skips_a_none_previous_handler(monkeypatch):
    """``signal.getsignal`` returns ``None`` for a handler installed from C.

    ``signal.signal(sig, None)`` raises ``TypeError``, so the restore path must
    skip those signums instead of blowing up in a ``finally``.
    """
    import signal

    from hydra_suite.trackerkit.headless_tracking import (
        install_stop_signal_handlers,
        restore_signal_handlers,
    )

    watched = [signal.SIGINT, signal.SIGTERM]
    real_signal = signal.signal
    saved = {sig: signal.getsignal(sig) for sig in watched}
    try:
        monkeypatch.setattr(signal, "getsignal", lambda _sig: None)
        stop_event, previous, installed = install_stop_signal_handlers(watched)
        assert installed and previous == {sig: None for sig in watched}
        restore_signal_handlers(previous, installed)  # must not raise
        assert not stop_event.is_set()
    finally:
        monkeypatch.undo()
        for sig, handler in saved.items():
            if handler is not None:
                real_signal(sig, handler)
