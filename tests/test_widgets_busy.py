"""Tests for the shared busy-indicator helper."""

from __future__ import annotations

import sys

import pytest

pytest.importorskip("PySide6")


@pytest.fixture()
def qapp():
    from PySide6.QtWidgets import QApplication

    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    return app


def _spin(app, predicate, timeout_ms: int = 5000) -> None:
    """Pump events until *predicate* returns True or *timeout_ms* elapses."""
    from PySide6.QtCore import QElapsedTimer

    timer = QElapsedTimer()
    timer.start()
    while not predicate():
        app.processEvents()
        if timer.hasExpired(timeout_ms):
            raise AssertionError("Timed out waiting for busy task to complete")


class TestRunWithBusyDialog:
    def test_runs_callable_in_thread_and_emits_success(self, qapp):
        from PySide6.QtWidgets import QWidget

        from hydra_suite.widgets.busy import run_with_busy_dialog

        parent = QWidget()
        results: list = []
        finished: list[bool] = []

        def _job(set_status, set_progress):
            set_status("Working")
            return {"answer": 42}

        task = run_with_busy_dialog(
            parent,
            _job,
            title="Test",
            message="Initial",
            on_success=results.append,
            on_finished=lambda: finished.append(True),
        )

        _spin(qapp, lambda: bool(finished))

        assert results == [{"answer": 42}]
        assert finished == [True]
        assert not task.dialog.isVisible()

    def test_relays_status_messages_to_dialog(self, qapp):
        from PySide6.QtWidgets import QWidget

        from hydra_suite.widgets.busy import run_with_busy_dialog

        parent = QWidget()
        finished: list[bool] = []

        def _job(set_status, set_progress):
            set_status("Step 1 of 2")

        task = run_with_busy_dialog(
            parent,
            _job,
            title="Test",
            message="Initial",
            on_finished=lambda: finished.append(True),
        )

        _spin(qapp, lambda: task.dialog.labelText() == "Step 1 of 2")
        _spin(qapp, lambda: bool(finished))

    def test_emits_error_on_exception_and_closes_dialog(self, qapp):
        from PySide6.QtWidgets import QWidget

        from hydra_suite.widgets.busy import run_with_busy_dialog

        parent = QWidget()
        errors: list[str] = []
        finished: list[bool] = []

        def _job(set_status, set_progress):
            raise RuntimeError("boom")

        task = run_with_busy_dialog(
            parent,
            _job,
            title="Test",
            message="Initial",
            on_error=errors.append,
            on_finished=lambda: finished.append(True),
        )

        _spin(qapp, lambda: bool(finished))

        assert errors == ["boom"]
        assert not task.dialog.isVisible()

    def test_determinate_mode_relays_progress_values(self, qapp):
        from PySide6.QtWidgets import QWidget

        from hydra_suite.widgets.busy import run_with_busy_dialog

        parent = QWidget()
        finished: list[bool] = []
        progress_values: list[int] = []

        def _job(set_status, set_progress):
            set_progress(25)
            set_progress(75)

        task = run_with_busy_dialog(
            parent,
            _job,
            title="Test",
            message="Initial",
            determinate=True,
            on_finished=lambda: finished.append(True),
        )
        # Capture progress values relayed through the worker's signal so we
        # can assert on them after the dialog has been closed.
        task.worker.progress.connect(progress_values.append)

        _spin(qapp, lambda: bool(finished))

        assert 75 in progress_values


class TestRunBlockingWithBusyDialog:
    def test_returns_callable_result(self, qapp):
        from PySide6.QtWidgets import QWidget

        from hydra_suite.widgets.busy import run_blocking_with_busy_dialog

        parent = QWidget()

        def _job(set_status, set_progress):
            set_status("Working")
            return [1, 2, 3]

        result = run_blocking_with_busy_dialog(
            parent, _job, title="Test", message="Initial"
        )

        assert result == [1, 2, 3]

    def test_raises_busy_task_error_on_exception(self, qapp):
        from PySide6.QtWidgets import QWidget

        from hydra_suite.widgets.busy import (
            BusyTaskError,
            run_blocking_with_busy_dialog,
        )

        parent = QWidget()

        def _job(set_status, set_progress):
            raise ValueError("nope")

        with pytest.raises(BusyTaskError, match="nope"):
            run_blocking_with_busy_dialog(parent, _job, title="Test", message="Initial")
