"""GUI CPU budget: planned fan-out options and the process cap (Task 6).

No test here calls the real ``set_process_cpu_cap``: it mutates process-global
thread state, so it is always monkeypatched on the orchestrator module.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

import hydra_suite.trackerkit.gui.orchestrators.tracking as tracking_mod
from hydra_suite.trackerkit.gui.orchestrators.tracking import TrackingOrchestrator


@pytest.fixture(autouse=True)
def _clean_cpu_env(monkeypatch):
    monkeypatch.delenv("HYDRA_CPU_BUDGET", raising=False)
    monkeypatch.delenv("HYDRA_CPU_THREADS", raising=False)


def _usable(monkeypatch, n):
    monkeypatch.setattr("hydra_suite.runtime.cpu_budget.usable_cores", lambda **_: n)


def _no_cuda(monkeypatch):
    monkeypatch.setattr(
        "hydra_suite.runtime.cuda_devices.list_cuda_devices", lambda **_: []
    )
    monkeypatch.setattr(
        "hydra_suite.trackerkit.batch_fanout.host_has_cuda", lambda: False
    )


@pytest.fixture
def orchestrator():
    config = SimpleNamespace(
        batch_parallel_gpus="", batch_parallel_jobs=0, cpu_budget=0
    )
    main_window = SimpleNamespace(
        config=config, _on_batch_parallel_changed=lambda: None
    )
    return TrackingOrchestrator(main_window, config, SimpleNamespace())


# --- _resolve_fanout_options -------------------------------------------------


def test_resolve_fanout_options_applies_cpu_plan(orchestrator, monkeypatch):
    _usable(monkeypatch, 12)
    _no_cuda(monkeypatch)
    cfg = orchestrator._mw.config
    cfg.batch_parallel_jobs, cfg.cpu_budget = 16, 0
    options = orchestrator._resolve_fanout_options(n_videos=20)
    assert (options.jobs, options.threads_per_job) == (6, 2)
    plan = orchestrator._last_fanout_cpu_plan
    assert plan.clamped and plan.warnings


def test_resolve_fanout_options_uses_gui_budget(orchestrator, monkeypatch):
    _usable(monkeypatch, 12)
    _no_cuda(monkeypatch)
    cfg = orchestrator._mw.config
    cfg.batch_parallel_jobs, cfg.cpu_budget = 2, 8
    options = orchestrator._resolve_fanout_options(n_videos=5)
    assert (options.jobs, options.threads_per_job) == (2, 4)


def test_resolve_fanout_options_honours_hydra_cpu_budget_env(orchestrator, monkeypatch):
    _usable(monkeypatch, 12)
    _no_cuda(monkeypatch)
    monkeypatch.setenv("HYDRA_CPU_BUDGET", "4")
    orchestrator._mw.config.batch_parallel_jobs = 2
    options = orchestrator._resolve_fanout_options(n_videos=5)
    assert (options.jobs, options.threads_per_job) == (2, 2)


def test_inherited_cpu_threads_bounds_the_fanout_budget(orchestrator, monkeypatch):
    """Parity with the CLI (8a2dfae6): a GUI already under HYDRA_CPU_THREADS
    must not hand its children more threads than it has itself."""
    _usable(monkeypatch, 12)
    _no_cuda(monkeypatch)
    monkeypatch.setenv("HYDRA_CPU_THREADS", "4")
    cfg = orchestrator._mw.config
    cfg.batch_parallel_jobs, cfg.cpu_budget = 2, 8
    options = orchestrator._resolve_fanout_options(n_videos=5)
    assert (options.jobs, options.threads_per_job) == (2, 2)


def test_resolve_fanout_options_never_passes_zero_threads(orchestrator, monkeypatch):
    _usable(monkeypatch, 12)
    _no_cuda(monkeypatch)
    captured = {}

    import hydra_suite.trackerkit.batch_fanout as fanout_mod

    real = fanout_mod.plan_fanout_cpu

    def _spy(options, **kwargs):
        captured["tpj"] = options.threads_per_job
        return real(options, **kwargs)

    monkeypatch.setattr(fanout_mod, "plan_fanout_cpu", _spy)
    orchestrator._mw.config.batch_parallel_jobs = 2
    orchestrator._resolve_fanout_options(n_videos=2)
    assert captured["tpj"] is None


# --- _apply_cpu_budget -------------------------------------------------------


def test_apply_cpu_budget_zero_is_noop_and_positive_caps(orchestrator, monkeypatch):
    _usable(monkeypatch, 12)
    calls = []
    monkeypatch.setattr(tracking_mod, "set_process_cpu_cap", calls.append)
    orchestrator._mw.config.cpu_budget = 0
    orchestrator._apply_cpu_budget()
    orchestrator._mw.config.cpu_budget = 4
    orchestrator._apply_cpu_budget()
    orchestrator._mw.config.cpu_budget = 0
    orchestrator._apply_cpu_budget()
    assert calls == [None, 4, None]


def test_apply_cpu_budget_zero_honours_hydra_cpu_budget(orchestrator, monkeypatch):
    _usable(monkeypatch, 12)
    monkeypatch.setenv("HYDRA_CPU_BUDGET", "3")
    calls = []
    monkeypatch.setattr(tracking_mod, "set_process_cpu_cap", calls.append)
    orchestrator._apply_cpu_budget()
    assert calls == [3]


# --- start_batch_fanout lifts the single-run cap before children spawn -------


def test_start_batch_fanout_lifts_single_run_cap_before_start(monkeypatch, tmp_path):
    import hydra_suite.trackerkit.batch_fanout as fanout_mod
    import hydra_suite.trackerkit.batch_plan as plan_mod
    import hydra_suite.trackerkit.gui.dialogs.batch_fanout_dialog as dialog_mod
    import hydra_suite.trackerkit.gui.workers.batch_fanout_worker as worker_mod

    events: list[str] = []
    video = tmp_path / "a.mp4"
    video.write_bytes(b"")

    class _Signal:
        def connect(self, *_a):
            pass

    class _FakeWorker:
        def __init__(self, specs, options, parent=None):
            for name in (
                "job_started",
                "job_progress",
                "job_finished",
                "fanout_finished",
                "error",
                "finished",
            ):
                setattr(self, name, _Signal())

        def cancel(self):
            pass

        def start(self):
            events.append("start")

    class _FakeDialog:
        def __init__(self, specs, parent=None):
            self.cancel_requested = _Signal()
            self.on_job_started = self.on_job_progress = None
            self.on_job_finished = self.on_fanout_finished = None
            self.on_fanout_error = None

        def show(self):
            pass

    monkeypatch.setattr(plan_mod, "plan_batch_jobs", lambda videos, **_: ["s1", "s2"])
    monkeypatch.setattr(dialog_mod, "BatchFanoutDialog", _FakeDialog)
    monkeypatch.setattr(worker_mod, "BatchFanoutWorker", _FakeWorker)
    monkeypatch.setattr(
        tracking_mod, "set_process_cpu_cap", lambda n: events.append(f"cap:{n}")
    )

    label = SimpleNamespace(text="", setVisible=lambda *_: None)
    label.setText = lambda t: setattr(label, "text", t)
    main_window = SimpleNamespace(
        batch_videos=[str(video)],
        batch_fanout_worker=None,
        batch_fanout_dialog=None,
        btn_start=SimpleNamespace(setText=lambda *_: None),
        progress_bar=SimpleNamespace(
            setVisible=lambda *_: None, setRange=lambda *_: None
        ),
        progress_label=label,
        _apply_ui_state=lambda *_: None,
    )
    setup = SimpleNamespace(
        chk_batch_keystone_override=SimpleNamespace(isChecked=lambda: False)
    )
    orch = TrackingOrchestrator(
        main_window, SimpleNamespace(), SimpleNamespace(setup=setup)
    )
    options = fanout_mod.FanoutOptions(gpus=[], jobs=2, threads_per_job=3)
    orch._resolve_fanout_options = lambda n_videos=1: options
    orch._last_fanout_cpu_plan = SimpleNamespace(
        warnings=("Requested 8 parallel job(s) but the CPU budget is 4 core(s).",)
    )

    assert orch.start_batch_fanout() is True
    assert events.index("cap:None") < events.index("start")
    assert "2 slot(s)" in label.text
    assert "CPU budget" in label.text


# --- plan label text ---------------------------------------------------------


def test_describe_cpu_plan_mentions_reduction():
    from hydra_suite.runtime.cpu_budget import CpuBudget, CpuPlan
    from hydra_suite.trackerkit.batch_fanout import describe_cpu_plan

    budget = CpuBudget(cores=8, usable=8, source="auto")
    text = describe_cpu_plan(CpuPlan(4, 2, True), budget, requested_jobs=6)
    assert text.startswith("4 job(s) x 2 thread(s) of 8 core(s)")
    assert "reduced from 6" in text
    assert "reduced" not in describe_cpu_plan(CpuPlan(2, 4, False), budget, 2)


def test_refresh_cpu_plan_label_tolerates_missing_label(monkeypatch):
    orch = TrackingOrchestrator(
        SimpleNamespace(config=SimpleNamespace(cpu_budget=0), batch_videos=[]),
        SimpleNamespace(),
        SimpleNamespace(setup=SimpleNamespace()),
    )
    orch.refresh_cpu_plan_label()  # no label yet: must not raise


def test_refresh_cpu_plan_label_sets_text(monkeypatch):
    _usable(monkeypatch, 8)
    label = SimpleNamespace(text="")
    label.setText = lambda t: setattr(label, "text", t)
    setup = SimpleNamespace(
        lbl_batch_parallel_cpu_plan=label,
        spin_batch_parallel_jobs=SimpleNamespace(value=lambda: 3),
        edit_batch_parallel_gpus=SimpleNamespace(isVisible=lambda: False),
    )
    orch = TrackingOrchestrator(
        SimpleNamespace(
            config=SimpleNamespace(cpu_budget=0), batch_videos=["a", "b", "c"]
        ),
        SimpleNamespace(),
        SimpleNamespace(setup=setup),
    )
    orch.refresh_cpu_plan_label()
    assert label.text.startswith("3 job(s) x 2 thread(s) of 8 core(s)")


def test_parallel_tooltip_no_longer_claims_byte_identity():
    src = Path(
        __import__(
            "hydra_suite.trackerkit.gui.panels.setup_panel", fromlist=["x"]
        ).__file__
    ).read_text()
    assert "byte-identical to" not in src
    assert "equal share of the CPU budget" in src
