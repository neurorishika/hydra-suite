# tests/test_base_worker.py
import sys

import pytest


@pytest.fixture()
def qapp():
    """Provide a QApplication for worker tests."""
    from PySide6.QtWidgets import QApplication

    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    return app


def test_base_worker_execute_called(qapp):
    """execute() is called when worker is started."""
    from PySide6.QtCore import QCoreApplication

    from hydra_suite.widgets.workers import BaseWorker

    class _EchoWorker(BaseWorker):
        def execute(self):
            self.status.emit("hello")
            self.progress.emit(100)

    received = []
    worker = _EchoWorker()
    worker.status.connect(received.append)
    worker.start()
    worker.wait(3000)
    QCoreApplication.processEvents()

    assert received == ["hello"]


def test_base_worker_finished_always_fires(qapp):
    """finished signal fires even when execute raises."""
    from PySide6.QtCore import QCoreApplication

    from hydra_suite.widgets.workers import BaseWorker

    class _CrashWorker(BaseWorker):
        def execute(self):
            raise RuntimeError("boom")

    finished_calls = []
    errors = []
    worker = _CrashWorker()
    worker.finished.connect(lambda: finished_calls.append(1))
    worker.error.connect(errors.append)
    worker.start()
    worker.wait(3000)
    QCoreApplication.processEvents()

    assert len(finished_calls) == 1
    assert "boom" in errors[0]


def test_base_worker_error_emitted_on_exception(qapp):
    """error signal carries the exception message."""
    from PySide6.QtCore import QCoreApplication

    from hydra_suite.widgets.workers import BaseWorker

    class _BadWorker(BaseWorker):
        def execute(self):
            raise ValueError("bad value")

    errors = []
    worker = _BadWorker()
    worker.error.connect(errors.append)
    worker.start()
    worker.wait(3000)
    QCoreApplication.processEvents()

    assert len(errors) == 1
    assert "bad value" in errors[0]
    assert worker.failure_exception is not None
    assert isinstance(worker.failure_exception, ValueError)


def test_base_worker_retains_exact_recovery_bearing_exception(qapp):
    from hydra_suite.widgets.workers import BaseWorker

    owned = RuntimeError("exact recovery object")

    class _OwnedWorker(BaseWorker):
        def execute(self):
            raise owned

    worker = _OwnedWorker()
    worker.run()

    assert worker.failure_exception is owned


def test_base_worker_bounds_terminal_error_signal_but_retains_exact_exception(qapp):
    from hydra_suite.widgets.workers import (
        MAX_WORKER_TERMINAL_MESSAGE_BYTES,
        BaseWorker,
    )

    owned = RuntimeError("x" * (MAX_WORKER_TERMINAL_MESSAGE_BYTES * 8))

    class _NoisyFailureWorker(BaseWorker):
        def execute(self):
            raise owned

    worker = _NoisyFailureWorker()
    errors = []
    worker.error.connect(errors.append)
    worker.run()

    assert worker.failure_exception is owned
    assert len(errors) == 1
    assert len(errors[0].encode("utf-8")) <= MAX_WORKER_TERMINAL_MESSAGE_BYTES
    assert errors[0].endswith("[message truncated]")


def test_base_worker_never_calls_arbitrary_exception_str(qapp):
    from hydra_suite.widgets.workers import BaseWorker

    class _ExplosiveError(RuntimeError):
        def __str__(self):
            pytest.fail("terminal signal formatting must not call exception __str__")

    owned = _ExplosiveError("safe diagnostic")

    class _ExplosiveWorker(BaseWorker):
        def execute(self):
            raise owned

    worker = _ExplosiveWorker()
    errors = []
    worker.error.connect(errors.append)
    worker.run()

    assert worker.failure_exception is owned
    assert errors == ["_ExplosiveError: safe diagnostic"]


def test_base_worker_handles_exception_with_explosive_args_property(qapp):
    from hydra_suite.widgets.workers import BaseWorker

    class _ExplosiveArgsError(RuntimeError):
        @property
        def args(self):
            raise RuntimeError("args unavailable")

    owned = _ExplosiveArgsError()

    class _ExplosiveWorker(BaseWorker):
        def execute(self):
            raise owned

    worker = _ExplosiveWorker()
    errors = []
    worker.error.connect(errors.append)
    worker.run()

    assert worker.failure_exception is owned
    assert errors == ["_ExplosiveArgsError"]


def test_bounded_worker_message_always_returns_valid_utf8(qapp):
    from hydra_suite.widgets.workers import bounded_worker_message

    message = bounded_worker_message("before\ud800after")

    assert "\ud800" not in message
    assert message.encode("utf-8")


def test_base_worker_no_error_on_success(qapp):
    """error signal is not emitted when execute succeeds, but finished is."""
    from PySide6.QtCore import QCoreApplication

    from hydra_suite.widgets.workers import BaseWorker

    class _OkWorker(BaseWorker):
        def execute(self):
            pass

    errors = []
    finished_calls = []
    worker = _OkWorker()
    worker.error.connect(errors.append)
    worker.finished.connect(lambda: finished_calls.append(1))
    worker.start()
    worker.wait(3000)
    QCoreApplication.processEvents()

    assert errors == []
    assert len(finished_calls) == 1


def test_base_worker_subclass_extra_signals(qapp):
    """Subclasses can add extra signals beyond the base set."""
    from PySide6.QtCore import QCoreApplication, Signal

    from hydra_suite.widgets.workers import BaseWorker

    class _ResultWorker(BaseWorker):
        result = Signal(int)

        def execute(self):
            self.result.emit(42)

    results = []
    worker = _ResultWorker()
    worker.result.connect(results.append)
    worker.start()
    worker.wait(3000)
    QCoreApplication.processEvents()

    assert results == [42]
