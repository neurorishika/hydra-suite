"""The worker's FanoutEvents implementation must map callbacks to signals with
bounded payloads. We call the event methods directly and capture emits via
a lightweight signal spy, so no event loop is required."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

# A real QApplication (not a bare QCoreApplication) is required: the dialog
# tests at the bottom of this module build widgets.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytest.importorskip("PySide6")
from PySide6.QtWidgets import QApplication  # noqa: E402

from hydra_suite.runtime.cuda_devices import CudaDevice  # noqa: E402
from hydra_suite.trackerkit.batch_fanout import (  # noqa: E402
    FanoutJobResult,
    FanoutOptions,
)
from hydra_suite.trackerkit.batch_plan import BatchJobSpec  # noqa: E402
from hydra_suite.trackerkit.gui.workers.batch_fanout_worker import (  # noqa: E402
    BatchFanoutWorker,
)


@pytest.fixture(scope="module")
def app():
    existing = QApplication.instance()
    if existing is not None and not isinstance(existing, QApplication):
        pytest.skip("a non-widget QCoreApplication already owns this process")
    return existing or QApplication([])


def _spec(i):
    return BatchJobSpec(
        index=i,
        video_path=f"/tmp/v{i}.mp4",
        config_path=None,
        config={},
        provenance="own-sidecar",
    )


def test_events_map_to_signals(app):
    worker = BatchFanoutWorker([_spec(1)], FanoutOptions())
    got = {}
    worker.job_started.connect(
        lambda i, v, lp, g: got.__setitem__("started", (i, v, lp, g))
    )
    worker.job_progress.connect(lambda i, p, m: got.__setitem__("progress", (i, p, m)))
    worker.job_log.connect(lambda i, line: got.__setitem__("log", (i, line)))
    worker.job_finished.connect(
        lambda i, ok, msg: got.__setitem__("finished", (i, ok, msg))
    )
    gpu = CudaDevice(0, "GPU-abc", "x")
    worker.job_started_cb(_spec(1), gpu, Path("/tmp/l.log"))
    worker.job_progress_cb(_spec(1), 42, "detecting")
    worker.job_log_cb(_spec(1), "a child log line")
    worker.job_finished_cb(
        FanoutJobResult(
            _spec(1), gpu, 0, True, Path("/tmp/l.log"), ["video=v1"], None, 2.0
        )
    )
    assert got["started"] == (1, "/tmp/v1.mp4", "/tmp/l.log", "GPU-abc")
    assert got["progress"] == (1, 42, "detecting")
    assert got["log"] == (1, "a child log line")
    assert got["finished"][0] == 1 and got["finished"][1] is True
    assert "video=v1" in got["finished"][2]


def test_job_started_without_a_gpu_reports_a_dash(app):
    """A CPU/inherited-device slot has no GPU: the table must show "-", not "None"."""
    worker = BatchFanoutWorker([_spec(3)], FanoutOptions())
    got = {}
    worker.job_started.connect(
        lambda i, v, lp, g: got.__setitem__("started", (i, v, lp, g))
    )
    worker.job_started_cb(_spec(3), None, Path("/tmp/l3.log"))
    assert got["started"] == (3, "/tmp/v3.mp4", "/tmp/l3.log", "-")


def test_failed_job_reports_its_error_not_its_summary(app):
    worker = BatchFanoutWorker([_spec(2)], FanoutOptions())
    got = {}
    worker.job_finished.connect(
        lambda i, ok, msg: got.__setitem__("finished", (i, ok, msg))
    )
    worker.job_finished_cb(
        FanoutJobResult(
            _spec(2), None, 1, False, Path("/tmp/l2.log"), [], "exit code 1", 1.0
        )
    )
    assert got["finished"] == (2, False, "exit code 1")


def test_cancel_sets_stop_flag(app):
    worker = BatchFanoutWorker([_spec(1)], FanoutOptions())
    assert worker.should_stop() is False
    worker.cancel()
    assert worker.should_stop() is True


def test_stop_is_an_alias_for_cancel(app):
    """``_request_qthread_stop`` calls ``worker.stop()`` when it exists."""
    worker = BatchFanoutWorker([_spec(1)], FanoutOptions())
    worker.stop()
    assert worker.should_stop() is True


def test_execute_runs_the_scheduler_and_emits_the_result(app, monkeypatch):
    """``execute`` must hand the scheduler its own events adapter + stop flag."""
    import hydra_suite.trackerkit.gui.workers.batch_fanout_worker as mod

    seen = {}

    def _fake_run(specs, options, *, events, should_stop, child_registry=None):
        seen["specs"] = list(specs)
        seen["options"] = options
        seen["should_stop"] = should_stop
        seen["child_registry"] = child_registry
        events.job_started(specs[0], None, Path("/tmp/l.log"))
        events.job_progress(specs[0], 10, "x")
        events.job_log(specs[0], "line")
        events.job_finished(
            FanoutJobResult(specs[0], None, 0, True, Path("/tmp/l.log"), [], None, 1.0)
        )
        return "SENTINEL-RESULT"

    monkeypatch.setattr(mod, "run_batch_fanout", _fake_run)
    worker = BatchFanoutWorker([_spec(1)], FanoutOptions())
    emitted = []
    worker.job_started.connect(lambda *a: emitted.append(("started",) + a))
    worker.job_progress.connect(lambda *a: emitted.append(("progress",) + a))
    worker.job_log.connect(lambda *a: emitted.append(("log",) + a))
    worker.job_finished.connect(lambda *a: emitted.append(("finished",) + a))
    worker.fanout_finished.connect(lambda r: emitted.append(("fanout", r)))

    worker.execute()

    assert seen["specs"][0].index == 1
    assert seen["should_stop"]() is False
    # The scheduler must be handed the registry the escape hatch reads.
    assert seen["child_registry"] is worker._children
    assert worker.result == "SENTINEL-RESULT"
    assert [e[0] for e in emitted] == [
        "started",
        "progress",
        "log",
        "finished",
        "fanout",
    ]
    assert emitted[-1][1] == "SENTINEL-RESULT"


# --- BatchFanoutDialog -------------------------------------------------------
# The dialog is the only consumer of the worker's signals, and two of its
# behaviours are contractual rather than cosmetic: the GPU column is filled
# from job_started (there is no separate producer), and jobs the scheduler
# never emitted job_finished for must still reach a terminal row state.

from hydra_suite.trackerkit.batch_fanout import FanoutResult  # noqa: E402
from hydra_suite.trackerkit.gui.dialogs.batch_fanout_dialog import (  # noqa: E402
    BatchFanoutDialog,
)


def _job(spec, success, error=None):
    return FanoutJobResult(
        spec, None, 0 if success else 1, success, Path("/tmp/l.log"), [], error, 1.0
    )


def _cell(dialog, index, col):
    return dialog._table.item(dialog._rows[index], col).text()


def test_dialog_fills_the_gpu_column_from_job_started(app):
    dialog = BatchFanoutDialog([_spec(1)])
    assert _cell(dialog, 1, 2) == "-"
    assert _cell(dialog, 1, 3) == "queued"
    dialog.on_job_started(1, "/tmp/v1.mp4", "/tmp/l.log", "GPU-abcdef1234")
    assert _cell(dialog, 1, 2) == "GPU-abcdef1234"
    assert _cell(dialog, 1, 3) == "running"


def test_dialog_marks_never_started_jobs_terminal(app):
    specs = [_spec(1), _spec(2), _spec(3)]
    dialog = BatchFanoutDialog(specs)
    dialog.on_job_started(1, specs[0].video_path, "/tmp/l.log", "-")
    dialog.on_job_finished(1, False, "exit code 1")
    result = FanoutResult(
        jobs=[
            _job(specs[0], False, "exit code 1"),
            _job(specs[1], False, "not started"),
            _job(specs[2], False, "not started"),
        ],
        cancelled=False,
    )
    dialog.on_fanout_finished(result)
    assert _cell(dialog, 1, 3) == "FAILED"
    assert _cell(dialog, 2, 3) == "not started"
    assert _cell(dialog, 3, 3) == "not started"
    assert dialog._finished is True


def test_dialog_marks_unresolved_rows_cancelled_on_a_cancelled_run(app):
    specs = [_spec(1), _spec(2)]
    dialog = BatchFanoutDialog(specs)
    dialog.on_job_started(1, specs[0].video_path, "/tmp/l.log", "-")
    result = FanoutResult(
        jobs=[_job(specs[0], False, "cancelled"), _job(specs[1], False, "cancelled")],
        cancelled=True,
    )
    dialog.on_fanout_finished(result)
    assert _cell(dialog, 1, 3) == "cancelled"
    assert _cell(dialog, 2, 3) == "cancelled"


def test_dialog_reject_cancels_the_run_instead_of_closing(app):
    """Esc / the window X must stop the children, not abandon them."""
    dialog = BatchFanoutDialog([_spec(1)])
    fired = []
    dialog.cancel_requested.connect(lambda: fired.append(True))
    dialog.reject()
    assert fired == [True]
    assert (
        dialog.isVisible() is False
    )  # never shown; the point is it did not accept/close
    assert dialog.result() == 0
    # Once finished, the same path closes for real.
    dialog.on_fanout_finished(
        FanoutResult(jobs=[_job(_spec(1), True)], cancelled=False)
    )
    dialog.reject()
    assert fired == [True]


# --- close-on-exit guard -----------------------------------------------------


def test_close_prompt_counts_a_running_fanout_worker():
    """Closing the window mid-fan-out must hit the "Tracking In Progress" prompt.

    Without ``batch_fanout_worker`` in that tuple, ``closeEvent`` skips both the
    prompt and ``stop_tracking()``: the QThread is destroyed while running and
    the child processes -- started with ``start_new_session`` -- are orphaned
    still holding their GPUs. ``_has_active_tracking_workers`` only ever does
    attribute access, so a stub self is a faithful stand-in for MainWindow.
    """
    from types import SimpleNamespace

    from hydra_suite.trackerkit.gui.main_window import MainWindow

    idle = SimpleNamespace(batch_fanout_worker=None, csv_writer_thread=None)
    assert MainWindow._has_active_tracking_workers(idle) is False

    running = SimpleNamespace(
        batch_fanout_worker=SimpleNamespace(isRunning=lambda: True),
        csv_writer_thread=None,
    )
    assert MainWindow._has_active_tracking_workers(running) is True


# --- GPU resolution on the GUI thread ---------------------------------------


def _orchestrator(gpus: str, jobs: int = 0):
    from types import SimpleNamespace

    from hydra_suite.trackerkit.gui.orchestrators.tracking import TrackingOrchestrator

    config = SimpleNamespace(batch_parallel_gpus=gpus, batch_parallel_jobs=jobs)
    main_window = SimpleNamespace(
        config=config, _on_batch_parallel_changed=lambda: None
    )
    return TrackingOrchestrator(main_window, config, SimpleNamespace())


def _patch_devices(monkeypatch, devices, has_cuda):
    import hydra_suite.runtime.cuda_devices as cuda_mod
    import hydra_suite.trackerkit.batch_fanout as fanout_mod

    monkeypatch.setattr(cuda_mod, "list_cuda_devices", lambda **_kw: list(devices))
    monkeypatch.setattr(fanout_mod, "host_has_cuda", lambda: has_cuda)


def _capture_warnings(monkeypatch):
    from types import SimpleNamespace

    import hydra_suite.trackerkit.gui.orchestrators.tracking as tracking_mod

    shown: list[tuple] = []
    monkeypatch.setattr(
        tracking_mod,
        "QMessageBox",
        SimpleNamespace(warning=lambda *args: shown.append(args)),
    )
    return shown


def test_gui_aborts_when_nvidia_smi_is_blind_on_a_cuda_host(monkeypatch):
    """The GUI used to guard the whole resolution with ``if available:`` and
    silently run UNPINNED -- N children all pinning cuda:0 -- whenever
    nvidia-smi timed out or was missing on a CUDA host."""
    _patch_devices(monkeypatch, [], has_cuda=True)
    shown = _capture_warnings(monkeypatch)
    assert _orchestrator("auto")._resolve_fanout_options() is None
    assert shown and "nvidia-smi" in shown[0][2]


def test_gui_runs_unpinned_on_a_host_with_no_cuda_at_all(monkeypatch):
    _patch_devices(monkeypatch, [], has_cuda=False)
    shown = _capture_warnings(monkeypatch)
    options = _orchestrator("auto", jobs=2)._resolve_fanout_options()
    assert options is not None and options.gpus == [] and options.jobs == 2
    assert not shown


def test_gui_pins_the_selected_devices(monkeypatch):
    devices = [CudaDevice(0, "GPU-aaaa", "x"), CudaDevice(1, "GPU-bbbb", "x")]
    _patch_devices(monkeypatch, devices, has_cuda=True)
    _capture_warnings(monkeypatch)
    options = _orchestrator("1")._resolve_fanout_options()
    assert options is not None and options.gpus == [devices[1]]
    assert options.jobs is None  # unspecified: one slot per selected GPU


# --- Window-close budget ----------------------------------------------------


def test_fanout_stop_timeout_scales_with_jobs_and_is_capped():
    """25 s flat was blown by three stubborn children: `_stop_children` spends
    sigint_grace + term_grace globally and then up to 5 s wait + 5 s reader join
    PER CHILD."""
    from hydra_suite.trackerkit.gui.orchestrators.tracking import fanout_stop_timeout_ms

    options = FanoutOptions(sigint_grace_s=10.0, term_grace_s=5.0)
    assert fanout_stop_timeout_ms(1, options) == 25_000
    assert fanout_stop_timeout_ms(4, options) == 55_000
    assert fanout_stop_timeout_ms(0, options) == 25_000  # at least one job
    assert fanout_stop_timeout_ms(64, options) == 120_000  # capped
    assert fanout_stop_timeout_ms(2, None) == 35_000  # defaults when unknown


def test_close_is_refused_while_the_fanout_is_still_stopping(monkeypatch):
    """After the stop budget expires, closeEvent used to fall through to
    super().closeEvent() -- destroying the QThread and orphaning children that
    still hold GPUs."""
    from types import SimpleNamespace

    import hydra_suite.trackerkit.gui.main_window as mw_mod
    from hydra_suite.trackerkit.gui.main_window import MainWindow

    warnings: list = []
    monkeypatch.setattr(
        mw_mod,
        "QMessageBox",
        SimpleNamespace(
            question=lambda *a, **k: 1,
            warning=lambda *a: warnings.append(a),
            Yes=1,
            No=2,
        ),
    )
    monkeypatch.setattr(
        mw_mod,
        "QApplication",
        SimpleNamespace(
            setOverrideCursor=lambda *a: None, restoreOverrideCursor=lambda *a: None
        ),
    )

    events: list[str] = []
    stub = SimpleNamespace(
        _has_active_tracking_workers=lambda: True,
        _tracking_orch=SimpleNamespace(
            stop_tracking=lambda: events.append("stop_tracking")
        ),
        batch_fanout_worker=SimpleNamespace(isRunning=lambda: True),
        _save_ui_settings=lambda: events.append("saved"),
        _status_log_tail=None,
    )
    event = SimpleNamespace(
        ignore=lambda: events.append("ignore"), accept=lambda: events.append("accept")
    )

    MainWindow.closeEvent(stub, event)

    assert events == ["stop_tracking", "ignore"], events
    assert warnings, "the user was not told why the window stayed open"
    # The refusal MUST arm the escalation, or the second attempt warns again and
    # the window is unclosable forever -- which is the whole reason the escape
    # hatch exists.
    assert stub._fanout_close_warned is True


def test_kill_children_now_sigkills_every_live_child_group(app, tmp_path):
    """The escape hatch must reach the children the scheduler owns.

    ``run_batch_fanout`` keeps its ``running`` list on the scheduler thread, so
    the GUI thread has no handle on the survivors after the stop budget expires.
    The registry is that handle; without it "close anyway" would orphan
    processes still holding their GPUs.
    """
    import signal
    import subprocess
    import sys
    import time

    worker = BatchFanoutWorker([_spec(1)], FanoutOptions())

    # A child in its OWN session (exactly how _launch starts one) that itself
    # spawns a grandchild -- the case a pid-only kill would leave behind.
    script = (
        "import subprocess, sys, time\n"
        "kid = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(120)'])\n"
        "print(kid.pid, flush=True)\n"
        "time.sleep(120)\n"
    )
    proc = subprocess.Popen(
        [sys.executable, "-c", script],
        stdout=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    try:
        grandchild = int(proc.stdout.readline().strip())
        worker._children.add(proc.pid)
        assert worker._children.pids() == [proc.pid]

        killed = worker.kill_children_now()

        assert killed == [proc.pid]
        assert proc.wait(timeout=10) is not None
        # The grandchild went with the group, not just the leader.
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            try:
                os.kill(grandchild, 0)
            except ProcessLookupError:
                break
            time.sleep(0.05)
        else:  # pragma: no cover - only on a regression
            os.kill(grandchild, signal.SIGKILL)
            raise AssertionError("the grandchild survived the group kill")
    finally:
        try:
            proc.kill()
        except Exception:
            pass


def test_registry_forgets_a_child_once_it_is_reaped(tmp_path):
    """A pid must leave the registry when its child does: signalling a recycled
    pid would hit an unrelated process group."""
    from hydra_suite.trackerkit.batch_fanout import LiveChildRegistry, run_batch_fanout

    registry = LiveChildRegistry()
    video = tmp_path / "v.mp4"
    video.write_bytes(b"\x00")
    spec = BatchJobSpec(
        index=1,
        video_path=str(video),
        config_path=None,
        config={},
        provenance="own-sidecar",
    )
    seen: list[list[int]] = []

    class _Events:
        def job_started(self, spec, gpu, log_path):
            seen.append(registry.pids())

        def job_progress(self, spec, percent, message):
            pass

        def job_log(self, spec, line):
            pass

        def job_finished(self, result):
            pass

    result = run_batch_fanout(
        [spec],
        FanoutOptions(
            jobs=1,
            run_dir=str(tmp_path / "run"),
            child_command=lambda s, c: ["true"],
        ),
        events=_Events(),
        child_registry=registry,
    )

    assert result.success, result.jobs[0].error
    assert seen and seen[0], "the child was never registered while it ran"
    assert registry.pids() == [], "a reaped child stayed in the registry"


def test_second_close_offers_to_kill_the_children_and_then_proceeds(monkeypatch):
    """A window that can never be closed is its own failure: after the first
    refusal, a second attempt must offer the hard way out -- and must offer it
    BEFORE paying another full stop_tracking() budget, not after: the kill
    prompt must appear, and cancel()+kill_children_now() must run, before
    stop_tracking() is ever called."""
    from types import SimpleNamespace

    import hydra_suite.trackerkit.gui.main_window as mw_mod
    from hydra_suite.trackerkit.gui.main_window import MainWindow

    answers = iter([1])  # "close anyway?" -- the only question asked this time
    asked: list = []
    monkeypatch.setattr(
        mw_mod,
        "QMessageBox",
        SimpleNamespace(
            question=lambda *a, **k: (asked.append(a[1]), next(answers))[1],
            warning=lambda *a: None,
            Yes=1,
            No=2,
        ),
    )
    monkeypatch.setattr(
        mw_mod,
        "QApplication",
        SimpleNamespace(
            setOverrideCursor=lambda *a: None, restoreOverrideCursor=lambda *a: None
        ),
    )

    events: list[str] = []
    worker = SimpleNamespace(
        isRunning=lambda: True,
        cancel=lambda: events.append("cancel"),
        kill_children_now=lambda: (events.append("kill"), [4242])[1],
        wait=lambda ms: events.append(f"wait({ms})"),
    )
    stub = SimpleNamespace(
        _has_active_tracking_workers=lambda: True,
        _tracking_orch=SimpleNamespace(
            stop_tracking=lambda: events.append("stop_tracking")
        ),
        batch_fanout_worker=worker,
        _save_ui_settings=lambda: events.append("saved"),
        _status_log_tail=None,
        # The first close attempt already happened.
        _fanout_close_warned=True,
    )
    event = SimpleNamespace(
        ignore=lambda: events.append("ignore"), accept=lambda: events.append("accept")
    )

    # The stub is not a real QWidget, so falling through to QWidget.closeEvent
    # raises -- which is itself the proof that the close PROCEEDED rather than
    # being refused. Everything asserted below happened before that point.
    with pytest.raises(TypeError, match="SimpleNamespace"):
        MainWindow.closeEvent(stub, event)

    assert "Children are still stopping" in asked
    assert "cancel" in events and "kill" in events and "stop_tracking" in events
    # The kill happens first -- the whole point of N3 is that the user never
    # pays a second full stop budget just to reach the escape hatch.
    assert events.index("cancel") < events.index("kill") < events.index("stop_tracking")
    assert any(e.startswith("wait(") for e in events)
    assert events.index("kill") < events.index(
        next(e for e in events if e.startswith("wait("))
    )
    assert "saved" in events, "the close did not proceed after the kill"
    assert "ignore" not in events


def test_second_close_answered_no_keeps_the_window_open(monkeypatch):
    """Declining the force-kill must leave everything exactly as it was."""
    from types import SimpleNamespace

    import hydra_suite.trackerkit.gui.main_window as mw_mod
    from hydra_suite.trackerkit.gui.main_window import MainWindow

    answers = iter([2])  # NO to "close anyway" -- the only question asked
    monkeypatch.setattr(
        mw_mod,
        "QMessageBox",
        SimpleNamespace(
            question=lambda *a, **k: next(answers),
            warning=lambda *a: None,
            Yes=1,
            No=2,
        ),
    )
    monkeypatch.setattr(
        mw_mod,
        "QApplication",
        SimpleNamespace(
            setOverrideCursor=lambda *a: None, restoreOverrideCursor=lambda *a: None
        ),
    )

    events: list[str] = []
    worker = SimpleNamespace(
        isRunning=lambda: True,
        cancel=lambda: events.append("cancel"),
        kill_children_now=lambda: events.append("kill") or [],
        wait=lambda ms: None,
    )
    stub = SimpleNamespace(
        _has_active_tracking_workers=lambda: True,
        _tracking_orch=SimpleNamespace(stop_tracking=lambda: None),
        batch_fanout_worker=worker,
        _save_ui_settings=lambda: events.append("saved"),
        _status_log_tail=None,
        _fanout_close_warned=True,
    )
    event = SimpleNamespace(
        ignore=lambda: events.append("ignore"), accept=lambda: events.append("accept")
    )

    MainWindow.closeEvent(stub, event)

    assert events == ["ignore"], events


def test_starting_a_new_batch_resets_the_close_warned_flag(app, monkeypatch, tmp_path):
    """A fresh batch must not inherit the escalation from an unrelated prior
    run: if a previous fan-out left ``_fanout_close_warned`` armed (e.g. it
    got force-killed on close), starting a NEW batch must clear it -- or a
    single innocuous close click on the new, healthy run would jump straight
    to the "children are still stopping, kill them?" prompt with nothing
    actually stuck."""
    from types import SimpleNamespace

    import hydra_suite.trackerkit.batch_plan as batch_plan_mod
    import hydra_suite.trackerkit.gui.dialogs.batch_fanout_dialog as dialog_mod
    import hydra_suite.trackerkit.gui.workers.batch_fanout_worker as worker_mod
    from hydra_suite.trackerkit.gui.orchestrators.tracking import TrackingOrchestrator

    video = tmp_path / "v.mp4"
    video.write_bytes(b"\x00")

    class _FakeSignal:
        def connect(self, *_a, **_kw):
            pass

    class _FakeWorker:
        def __init__(self, *_a, **_kw):
            self.job_started = _FakeSignal()
            self.job_progress = _FakeSignal()
            self.job_finished = _FakeSignal()
            self.fanout_finished = _FakeSignal()
            self.error = _FakeSignal()
            self.finished = _FakeSignal()

        def cancel(self):
            pass

        def isRunning(self):
            return False

        def start(self):
            pass

    class _FakeDialog:
        def __init__(self, *_a, **_kw):
            self.cancel_requested = _FakeSignal()

        def show(self):
            pass

        def __getattr__(self, _name):
            # Any on_job_started/on_job_progress/... slot connect target:
            # a no-op is fine, nothing here ever fires the signals for real.
            return lambda *_a, **_kw: None

    monkeypatch.setattr(worker_mod, "BatchFanoutWorker", _FakeWorker)
    monkeypatch.setattr(dialog_mod, "BatchFanoutDialog", _FakeDialog)
    monkeypatch.setattr(
        batch_plan_mod,
        "plan_batch_jobs",
        lambda videos, keystone_override=False: [_spec(1)],
    )
    _patch_devices(monkeypatch, [], has_cuda=False)

    setup = SimpleNamespace(
        g_batch=SimpleNamespace(isChecked=lambda: True),
        chk_batch_parallel=SimpleNamespace(isChecked=lambda: True),
        chk_batch_keystone_override=SimpleNamespace(isChecked=lambda: False),
    )
    config = SimpleNamespace(batch_parallel_gpus="auto", batch_parallel_jobs=0)
    main_window = SimpleNamespace(
        config=config,
        batch_videos=[str(video)],
        batch_fanout_worker=None,
        batch_fanout_dialog=None,
        # The prior (unrelated) run left the escalation armed.
        _fanout_close_warned=True,
        _on_batch_parallel_changed=lambda: None,
        btn_start=SimpleNamespace(setText=lambda *_a: None),
        progress_bar=SimpleNamespace(
            setVisible=lambda *_a: None, setRange=lambda *_a: None
        ),
        progress_label=SimpleNamespace(
            setVisible=lambda *_a: None, setText=lambda *_a: None
        ),
        _apply_ui_state=lambda *_a: None,
    )
    orch = TrackingOrchestrator(main_window, config, SimpleNamespace(setup=setup))

    assert orch.start_batch_fanout() is True
    assert main_window._fanout_close_warned is False
