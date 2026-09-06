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

    def _fake_run(specs, options, *, events, should_stop):
        seen["specs"] = list(specs)
        seen["options"] = options
        seen["should_stop"] = should_stop
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
