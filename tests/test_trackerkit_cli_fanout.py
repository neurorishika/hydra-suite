from __future__ import annotations

import pytest

from hydra_suite.runtime.cuda_devices import CudaDevice
from hydra_suite.trackerkit import cli
from hydra_suite.trackerkit.app import parse_arguments
from hydra_suite.trackerkit.batch_fanout import FanoutJobResult, FanoutResult


@pytest.fixture(autouse=True)
def _hermetic_cpu_budget(monkeypatch):
    """No ambient CPU budget leaks in (C7); a fixed, generous core count."""
    monkeypatch.delenv("HYDRA_CPU_BUDGET", raising=False)
    monkeypatch.delenv("HYDRA_CPU_THREADS", raising=False)
    monkeypatch.setattr("hydra_suite.runtime.cpu_budget.usable_cores", lambda **_: 64)


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


# --------------------------------------------------------------------------
# CPU budget (Task 5)
# --------------------------------------------------------------------------


def _stub_plan(monkeypatch):
    monkeypatch.setattr(
        cli, "plan_batch_jobs", lambda vids, **kw: [object()] * len(vids)
    )


def _stub_cap(monkeypatch):
    calls = []
    monkeypatch.setattr(cli, "set_process_cpu_cap", lambda n: calls.append(n))
    monkeypatch.setattr(cli, "bind_thread_cpu_cap", lambda: calls.append("bind"))
    return calls


def test_cli_fanout_options_carry_cpu_plan(monkeypatch, tmp_path):
    captured = {}
    monkeypatch.setattr(
        cli, "_run_fanout", lambda specs, options: captured.setdefault("o", options)
    )
    monkeypatch.setattr("hydra_suite.runtime.cpu_budget.usable_cores", lambda **_: 12)
    _stub_plan(monkeypatch)
    cli.run_tracking_cli(_videos(tmp_path, 20), jobs=16)
    assert (captured["o"].jobs, captured["o"].threads_per_job) == (6, 2)


def test_cli_fanout_uses_cpu_budget_flag(monkeypatch, tmp_path):
    captured = {}
    monkeypatch.setattr(
        cli, "_run_fanout", lambda specs, options: captured.setdefault("o", options)
    )
    _stub_plan(monkeypatch)
    cli.run_tracking_cli(_videos(tmp_path, 4), jobs=2, cpu_budget=8)
    assert (captured["o"].jobs, captured["o"].threads_per_job) == (2, 4)


def test_cli_fanout_parent_is_never_capped(monkeypatch, tmp_path):
    calls = _stub_cap(monkeypatch)
    monkeypatch.setattr(cli, "_run_fanout", lambda specs, options: 0)
    _stub_plan(monkeypatch)
    cli.run_tracking_cli(_videos(tmp_path, 4), jobs=2, cpu_budget=8)
    assert calls == []


def test_cli_fanout_logs_clamp_warning(monkeypatch, tmp_path, caplog):
    monkeypatch.setattr(cli, "_run_fanout", lambda specs, options: 0)
    monkeypatch.setattr("hydra_suite.runtime.cpu_budget.usable_cores", lambda **_: 12)
    _stub_plan(monkeypatch)
    with caplog.at_level("WARNING", logger=cli.logger.name):
        cli.run_tracking_cli(_videos(tmp_path, 20), jobs=16)
    assert any("running 6 at a time" in r.getMessage() for r in caplog.records)


def test_cli_plain_run_uncapped_without_budget(monkeypatch, tmp_path):
    calls = _stub_cap(monkeypatch)
    monkeypatch.setattr(cli, "_run_sequential", lambda specs: 0)
    _stub_plan(monkeypatch)
    cli.run_tracking_cli(_videos(tmp_path, 1))
    assert calls == []


def test_cli_plain_run_capped_by_flag(monkeypatch, tmp_path):
    calls = _stub_cap(monkeypatch)
    monkeypatch.setattr(cli, "_run_sequential", lambda specs: calls.append("run") or 0)
    _stub_plan(monkeypatch)
    cli.run_tracking_cli(_videos(tmp_path, 1), cpu_budget=8)
    # The cap (and this thread's binding) precede the sequential run, which
    # is where every inference/AprilTag config gets built.
    assert calls == [8, "bind", "run"]


def test_cli_plain_run_capped_by_env_budget(monkeypatch, tmp_path):
    calls = _stub_cap(monkeypatch)
    monkeypatch.setattr(cli, "_run_sequential", lambda specs: 0)
    monkeypatch.setenv("HYDRA_CPU_BUDGET", "6")
    _stub_plan(monkeypatch)
    cli.run_tracking_cli(_videos(tmp_path, 1))
    assert calls == [6, "bind"]


def test_cli_plain_run_capped_by_child_env(monkeypatch, tmp_path):
    calls = _stub_cap(monkeypatch)
    monkeypatch.setattr(cli, "_run_sequential", lambda specs: 0)
    monkeypatch.setenv("HYDRA_CPU_THREADS", "3")
    monkeypatch.setenv("HYDRA_CPU_BUDGET", "40")
    _stub_plan(monkeypatch)
    cli.run_tracking_cli(_videos(tmp_path, 1), cpu_budget=8)
    assert calls == [3, "bind"]


def test_cli_plain_run_warns_on_invalid_env_budget(monkeypatch, tmp_path, caplog):
    calls = _stub_cap(monkeypatch)
    monkeypatch.setattr(cli, "_run_sequential", lambda specs: 0)
    monkeypatch.setenv("HYDRA_CPU_BUDGET", "abc")
    _stub_plan(monkeypatch)
    with caplog.at_level("WARNING", logger=cli.logger.name):
        cli.run_tracking_cli(_videos(tmp_path, 1))
    assert calls == []
    assert any("HYDRA_CPU_BUDGET" in r.getMessage() for r in caplog.records)


def test_cli_plain_run_warns_when_budget_exceeds_usable(monkeypatch, tmp_path, caplog):
    calls = _stub_cap(monkeypatch)
    monkeypatch.setattr(cli, "_run_sequential", lambda specs: 0)
    monkeypatch.setattr("hydra_suite.runtime.cpu_budget.usable_cores", lambda **_: 4)
    _stub_plan(monkeypatch)
    with caplog.at_level("WARNING", logger=cli.logger.name):
        cli.run_tracking_cli(_videos(tmp_path, 1), cpu_budget=99)
    assert calls == [4, "bind"]
    assert any("exceeds" in r.getMessage() for r in caplog.records)


def test_fanout_log_line_reports_slot_count(monkeypatch, tmp_path, caplog):
    """Explicit --jobs below the GPU count: the log names the real slot count."""
    from hydra_suite.trackerkit.batch_fanout import FanoutOptions

    monkeypatch.setattr(
        cli, "run_batch_fanout", lambda s, o, **k: _ok_result(s, tmp_path)
    )
    opts = FanoutOptions(gpus=_three_gpus(), jobs=2)
    with caplog.at_level("INFO", logger=cli.logger.name):
        cli._run_fanout([], opts)
    assert any("2 slot(s)" in r.getMessage() for r in caplog.records)


def test_cpu_budget_must_be_positive(tmp_path):
    v = _videos(tmp_path, 1)
    with pytest.raises(SystemExit):
        parse_arguments(["track", v[0], "--cpu-budget", "0"])
    with pytest.raises(SystemExit):
        parse_arguments(["track", v[0], "--cpu-budget", "-2"])
    assert parse_arguments(["track", v[0], "--cpu-budget", "4"]).cpu_budget == 4
    assert parse_arguments(["track", v[0]]).cpu_budget is None


def test_main_forwards_cpu_budget(monkeypatch, tmp_path):
    from hydra_suite.trackerkit import app

    v = _videos(tmp_path, 1)
    seen = {}
    monkeypatch.setattr(app, "check_dependencies", lambda: True)
    monkeypatch.setattr(app, "setup_logging", lambda **kw: None)
    monkeypatch.setattr(
        app, "run_tracking_cli", lambda vids, **kw: seen.update(kw) or 0
    )
    with pytest.raises(SystemExit):
        app.main(["track", v[0], "--cpu-budget", "5"])
    assert seen["cpu_budget"] == 5


def test_fanout_inherited_cpu_threads_is_a_ceiling(monkeypatch, tmp_path, caplog):
    """A shell (or outer fan-out) HYDRA_CPU_THREADS=3 bounds the whole fan-out.

    budget = 3, 2 CPU-only jobs at the 2-thread floor -> 1 job x 3 threads,
    not 2 x 6 from the 12 usable cores.
    """
    captured = {}
    monkeypatch.setattr(
        cli, "_run_fanout", lambda specs, options: captured.setdefault("o", options)
    )
    monkeypatch.setattr("hydra_suite.runtime.cpu_budget.usable_cores", lambda **_: 12)
    monkeypatch.setenv("HYDRA_CPU_THREADS", "3")
    _stub_plan(monkeypatch)
    with caplog.at_level("INFO", logger=cli.logger.name):
        cli.run_tracking_cli(_videos(tmp_path, 4), jobs=2)
    assert (captured["o"].jobs, captured["o"].threads_per_job) == (1, 3)
    assert any("HYDRA_CPU_THREADS" in r.getMessage() for r in caplog.records)


def test_fanout_inherited_cpu_threads_lowers_a_larger_flag(monkeypatch, tmp_path):
    captured = {}
    monkeypatch.setattr(
        cli, "_run_fanout", lambda specs, options: captured.setdefault("o", options)
    )
    monkeypatch.setenv("HYDRA_CPU_THREADS", "4")
    _stub_plan(monkeypatch)
    cli.run_tracking_cli(_videos(tmp_path, 4), jobs=2, cpu_budget=16)
    assert (captured["o"].jobs, captured["o"].threads_per_job) == (2, 2)


def test_fanout_invalid_inherited_cpu_threads_warns(monkeypatch, tmp_path, caplog):
    captured = {}
    monkeypatch.setattr(
        cli, "_run_fanout", lambda specs, options: captured.setdefault("o", options)
    )
    monkeypatch.setenv("HYDRA_CPU_THREADS", "abc")
    _stub_plan(monkeypatch)
    with caplog.at_level("WARNING", logger=cli.logger.name):
        cli.run_tracking_cli(_videos(tmp_path, 4), jobs=2, cpu_budget=8)
    assert (captured["o"].jobs, captured["o"].threads_per_job) == (2, 4)
    assert any(
        r.levelname == "WARNING" and "HYDRA_CPU_THREADS" in r.getMessage()
        for r in caplog.records
    )


def test_plain_run_resolves_budget_once(monkeypatch, tmp_path):
    import hydra_suite.runtime.cpu_budget as cpu_budget_mod

    calls = _stub_cap(monkeypatch)
    monkeypatch.setattr(cli, "_run_sequential", lambda specs: 0)
    n = []
    # The single-run helper lives in runtime.cpu_budget (shared with the GUI).
    real = cpu_budget_mod.resolve_cpu_budget
    monkeypatch.setattr(
        cpu_budget_mod,
        "resolve_cpu_budget",
        lambda *a, **k: n.append(1) or real(*a, **k),
    )

    def _no_second_resolve(*a, **k):
        pytest.fail("budget path must not re-resolve")

    for target in (cli, cpu_budget_mod):
        monkeypatch.setattr(
            target, "resolve_process_cap", _no_second_resolve, raising=False
        )
    _stub_plan(monkeypatch)
    cli.run_tracking_cli(_videos(tmp_path, 1), cpu_budget=8)
    assert calls == [8, "bind"] and n == [1]


def test_job_run_cpu_budget_error_names_the_run_subcommand(capsys):
    with pytest.raises(SystemExit):
        parse_arguments(["job", "run", "/tmp/j", "--cpu-budget", "0"])
    assert "job run" in capsys.readouterr().err
