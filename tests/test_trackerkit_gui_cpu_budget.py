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


def test_apply_cpu_budget_warns_on_invalid_env(orchestrator, monkeypatch, caplog):
    """Parity with the CLI single run: an invalid HYDRA_CPU_BUDGET is ignored
    loudly, and the run stays uncapped."""
    _usable(monkeypatch, 12)
    monkeypatch.setenv("HYDRA_CPU_BUDGET", "abc")
    calls = []
    monkeypatch.setattr(tracking_mod, "set_process_cpu_cap", calls.append)
    with caplog.at_level("WARNING", logger=tracking_mod.logger.name):
        orchestrator._apply_cpu_budget()
    assert calls == [None]
    assert any("HYDRA_CPU_BUDGET" in r.getMessage() for r in caplog.records)


def test_apply_cpu_budget_warns_when_budget_exceeds_usable(
    orchestrator, monkeypatch, caplog
):
    _usable(monkeypatch, 4)
    calls = []
    monkeypatch.setattr(tracking_mod, "set_process_cpu_cap", calls.append)
    orchestrator._mw.config.cpu_budget = 16
    with caplog.at_level("WARNING", logger=tracking_mod.logger.name):
        orchestrator._apply_cpu_budget()
    assert calls == [4]
    assert any("exceeds" in r.getMessage() for r in caplog.records)


def test_invalid_inherited_cpu_threads_warns_in_fanout(
    orchestrator, monkeypatch, caplog
):
    _usable(monkeypatch, 12)
    _no_cuda(monkeypatch)
    monkeypatch.setenv("HYDRA_CPU_THREADS", "abc")
    cfg = orchestrator._mw.config
    cfg.batch_parallel_jobs, cfg.cpu_budget = 2, 8
    with caplog.at_level("WARNING", logger=tracking_mod.logger.name):
        options = orchestrator._resolve_fanout_options(n_videos=5)
    assert (options.jobs, options.threads_per_job) == (2, 4)
    assert any("HYDRA_CPU_THREADS" in r.getMessage() for r in caplog.records)


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
    seen_n_videos = []
    orch._resolve_fanout_options = lambda n_videos=1: (
        seen_n_videos.append(n_videos) or options
    )
    orch._last_fanout_cpu_plan = SimpleNamespace(
        warnings=("Requested 8 parallel job(s) but the CPU budget is 4 core(s).",)
    )

    assert orch.start_batch_fanout() is True
    assert seen_n_videos == [2]  # len(specs): the plan sizes threads per video
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


def _label_orchestrator(monkeypatch, *, jobs, gpus="", cuda_devices=(), cpu_budget=0):
    monkeypatch.setattr(
        "hydra_suite.trackerkit.batch_fanout.host_has_cuda", lambda: bool(cuda_devices)
    )
    label = SimpleNamespace(text="")
    label.setText = lambda t: setattr(label, "text", t)
    setup = SimpleNamespace(
        lbl_batch_parallel_cpu_plan=label,
        batch_parallel_cuda_devices=list(cuda_devices),
    )
    config = SimpleNamespace(
        cpu_budget=cpu_budget, batch_parallel_jobs=jobs, batch_parallel_gpus=gpus
    )
    orch = TrackingOrchestrator(
        SimpleNamespace(config=config, batch_videos=["a", "b", "c"]),
        config,
        SimpleNamespace(setup=setup),
    )
    return orch, label


def _fake_gpus(n):
    from hydra_suite.runtime.cuda_devices import CudaDevice

    return [CudaDevice(index=i, uuid=f"GPU-{i:04d}", name="Fake") for i in range(n)]


def test_refresh_cpu_plan_label_sets_text(monkeypatch):
    _usable(monkeypatch, 8)
    orch, label = _label_orchestrator(monkeypatch, jobs=3)
    orch.refresh_cpu_plan_label()
    assert label.text.startswith("3 job(s) x 2 thread(s) of 8 core(s)")


def test_plan_label_honours_inherited_cpu_threads_ceiling(monkeypatch):
    _usable(monkeypatch, 16)
    monkeypatch.setenv("HYDRA_CPU_THREADS", "4")
    orch, label = _label_orchestrator(monkeypatch, jobs=2)
    orch.refresh_cpu_plan_label()
    assert label.text.startswith("2 job(s) x 2 thread(s) of 4 core(s)")


def test_plan_label_clamps_jobs_to_the_gpu_count(monkeypatch):
    _usable(monkeypatch, 16)
    orch, label = _label_orchestrator(
        monkeypatch, jobs=8, gpus="auto", cuda_devices=_fake_gpus(2)
    )
    orch.refresh_cpu_plan_label()
    assert label.text.startswith("2 job(s) x 8 thread(s) of 16 core(s)")


def test_plan_label_matches_the_runs_plan(monkeypatch):
    """Label and run share one planner: same jobs and threads per job."""
    _usable(monkeypatch, 12)
    devices = _fake_gpus(3)
    orch, label = _label_orchestrator(
        monkeypatch, jobs=0, gpus="auto", cuda_devices=devices, cpu_budget=9
    )
    monkeypatch.setattr(
        "hydra_suite.runtime.cuda_devices.list_cuda_devices", lambda **_: devices
    )
    orch._mw._on_batch_parallel_changed = lambda: None
    orch.refresh_cpu_plan_label()
    options = orch._resolve_fanout_options(n_videos=3)
    assert label.text.startswith(
        f"{options.jobs} job(s) x {options.threads_per_job} thread(s) of 9 core(s)"
    )


def test_plan_label_shows_an_invalid_gpu_selection(monkeypatch):
    _usable(monkeypatch, 8)
    orch, label = _label_orchestrator(
        monkeypatch, jobs=2, gpus="7", cuda_devices=_fake_gpus(2)
    )
    orch.refresh_cpu_plan_label()  # must not raise on a bad selector
    assert label.text.startswith("GPU selection:")


# --- eager process cap (I1) -------------------------------------------------


class _Running:
    def __init__(self, running=True):
        self._running = running

    def isRunning(self):
        return self._running


def _record_caps(monkeypatch):
    calls = []
    monkeypatch.setattr(tracking_mod, "set_process_cpu_cap", calls.append)
    return calls


def test_apply_cpu_budget_now_applies_when_idle(orchestrator, monkeypatch):
    _usable(monkeypatch, 16)
    calls = _record_caps(monkeypatch)
    orchestrator._mw.config.cpu_budget = 5
    orchestrator.apply_cpu_budget_now()
    orchestrator._mw.config.cpu_budget = 0
    orchestrator.apply_cpu_budget_now()  # Auto + no env -> lift (no-op if never set)
    assert calls == [5, None]


@pytest.mark.parametrize(
    "attr",
    [
        "batch_fanout_worker",
        "tracking_worker",
        "session_worker",
        "_cache_builder_worker",
    ],
)
def test_apply_cpu_budget_now_is_deferred_while_a_run_is_active(
    orchestrator, monkeypatch, attr
):
    _usable(monkeypatch, 16)
    calls = _record_caps(monkeypatch)
    orchestrator._mw.config.cpu_budget = 5
    setattr(orchestrator._mw, attr, _Running(True))
    orchestrator.apply_cpu_budget_now()
    assert calls == []
    setattr(orchestrator._mw, attr, _Running(False))
    orchestrator.apply_cpu_budget_now()
    assert calls == [5]


def test_fanout_thread_finished_reapplies_the_budget(orchestrator, monkeypatch):
    _usable(monkeypatch, 16)
    calls = _record_caps(monkeypatch)
    orchestrator._mw.config.cpu_budget = 6
    orchestrator._mw.batch_fanout_worker = _Running(False)
    monkeypatch.setattr(
        orchestrator,
        "_cleanup_thread_reference",
        lambda attr: setattr(orchestrator._mw, attr, None),
    )
    orchestrator._on_batch_fanout_thread_finished()
    assert orchestrator._mw.batch_fanout_worker is None
    assert calls == [6]


def _real_main_window(monkeypatch):
    pytest.importorskip("PySide6")
    from PySide6.QtWidgets import QApplication

    from hydra_suite.trackerkit.gui.main_window import MainWindow

    QApplication.instance() or QApplication([])
    monkeypatch.setattr(MainWindow, "_save_advanced_config", lambda self: None)
    monkeypatch.setattr(MainWindow, "_load_advanced_config", lambda self: {})
    return MainWindow()


@pytest.mark.parametrize("env_budget, expected", [("3", [3]), (None, [None])])
def test_main_window_applies_the_cpu_budget_at_startup(
    monkeypatch, env_budget, expected
):
    _usable(monkeypatch, 16)
    if env_budget is not None:
        monkeypatch.setenv("HYDRA_CPU_BUDGET", env_budget)
    calls = _record_caps(monkeypatch)
    window = _real_main_window(monkeypatch)
    try:
        assert calls == expected
        calls.clear()
        window._setup_panel.spin_cpu_budget.setValue(2)
        assert calls == [2]
    finally:
        window.close()
        window.deleteLater()


def test_parallel_tooltip_byte_identity_claim_is_qualified():
    src = Path(
        __import__(
            "hydra_suite.trackerkit.gui.panels.setup_panel", fromlist=["x"]
        ).__file__
    ).read_text()
    # Any byte-identity statement must be qualified as measured, not promised.
    for idx in [i for i in range(len(src)) if src.startswith("byte-identical", i)]:
        assert "in our tests" in src[idx : idx + 120]
    assert "round differently under a CPU cap" in src
    assert "equal share of the CPU budget" in src


def test_cli_doc_byte_identity_claims_are_qualified():
    doc = Path(__file__).resolve().parents[1] / "docs/user-guide/trackerkit-cli.md"
    text = " ".join(doc.read_text(encoding="utf-8").split())
    hits = [i for i in range(len(text)) if text.startswith("byte-identical", i)]
    assert hits, "the CLI doc should state the measured fan-out parity"
    # Same rule as the tooltip: byte-identity is reported as measured.
    for idx in hits:
        assert "in our tests" in text[idx : idx + 120], text[idx : idx + 120]
    assert "may in principle" in text


def test_apply_cpu_budget_now_defers_on_any_active_main_window_worker(
    orchestrator, monkeypatch
):
    """MainWindow's own active-worker check (cache builder, export, merge,
    post-process, csv writer ...) also defers a budget change."""
    _usable(monkeypatch, 16)
    calls = _record_caps(monkeypatch)
    orchestrator._mw.config.cpu_budget = 5
    active = [True]
    orchestrator._mw._has_active_tracking_workers = lambda: active[0]
    orchestrator.apply_cpu_budget_now()
    assert calls == []
    active[0] = False
    orchestrator.apply_cpu_budget_now()
    assert calls == [5]
