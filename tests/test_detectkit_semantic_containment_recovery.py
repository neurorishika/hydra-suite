"""A retained-ownership failure must not strand this process's heavy-job leases.

The semantic workers retry containment teardown once themselves. If that
cannot prove quiescence, the GUI must retain the worker and offer another
cleanup attempt. Dropping the owner would leave the lease flock held by the
GUI process and block every later protected job.
"""

from __future__ import annotations

import os

import pytest

from hydra_suite.runtime.process_supervisor import WorkloadStillOwnedError


class _Sidecar:
    def __init__(self, *, fail_cleanup: bool) -> None:
        self.fail_cleanup = fail_cleanup
        self.cancel_calls = 0

    def cancel(self, *_args, **_kwargs):
        self.cancel_calls += 1
        if self.fail_cleanup:
            raise WorkloadStillOwnedError("still owned", self)


class _Operation:
    def __init__(self, sidecar: _Sidecar) -> None:
        self.sidecar = sidecar
        self.cancelled = False
        self.recovered = False

    def cancel(self):
        self.cancelled = True

    def run(self, **_kwargs):
        error = WorkloadStillOwnedError(
            "guardian could not prove quiescence", self.sidecar
        )
        error.recovery_cleanup = lambda: setattr(self, "recovered", True)
        raise error


def _preview_worker(monkeypatch, operation):
    from hydra_suite.detectkit.jobs import semantic_workers

    class _Source:
        path = "source"

        def to_dict(self):
            return {"path": self.path}

    monkeypatch.setattr(
        semantic_workers, "ProtectedOperation", lambda *_a, **_k: operation
    )
    return semantic_workers.FramePreviewWorker(
        [_Source()], "ant", "sam3", {"device": "cpu"}
    )


def test_preview_releases_ownership_when_teardown_can_be_proven(monkeypatch):
    operation = _Operation(_Sidecar(fail_cleanup=False))
    worker = _preview_worker(monkeypatch, operation)

    worker.run()

    assert operation.sidecar.cancel_calls == 1
    assert operation.recovered
    # Recovery succeeded, so nothing is retained and the next job can admit.
    assert not worker.containment_recovery_required
    assert isinstance(worker.failure_exception, RuntimeError)


def test_preview_retains_ownership_and_cancel_retries_cleanup(monkeypatch):
    sidecar = _Sidecar(fail_cleanup=True)
    operation = _Operation(sidecar)
    worker = _preview_worker(monkeypatch, operation)

    worker.run()

    assert worker.containment_recovery_required
    assert not operation.recovered

    sidecar.fail_cleanup = False
    worker.cancel()

    assert not worker.containment_recovery_required
    assert operation.recovered
    # Cancel routed to recovery; it must not have been forwarded as a plain
    # cancel of an operation whose ownership was still unproven.
    assert not operation.cancelled


def test_self_owned_lease_error_names_the_current_process(tmp_path):
    from hydra_suite.runtime.resource_lease import HeavyJobLease, ResourceBusyError

    first = HeavyJobLease("test:host-memory", "first job", tmp_path).acquire()
    try:
        with pytest.raises(ResourceBusyError) as excinfo:
            HeavyJobLease("test:host-memory", "second job", tmp_path).acquire()
    finally:
        first.release()
    message = str(excinfo.value)
    assert f"PID {os.getpid()}" in message
    assert "same application process" in message


def test_semantic_gui_keeps_recovery_owner_until_cleanup_is_proven(monkeypatch):
    from hydra_suite.detectkit.gui.escalation_actions import (
        retain_semantic_containment_recovery,
        retry_semantic_containment_cleanup,
    )

    class FakeProgress:
        def __init__(self):
            self.visible = False
            self.closed = False
            self.label = ""
            self.button = ""

        def setLabelText(self, value):
            self.label = value

        def setCancelButtonText(self, value):
            self.button = value

        def show(self):
            self.visible = True

        def close(self):
            self.closed = True
            self.visible = False

        def findChild(self, _cls):
            return None

    class FakeWorker:
        recovery_cleanup_error = ""

        def __init__(self):
            self.can_cleanup = False

        def retry_containment_cleanup(self):
            return self.can_cleanup

    class FakeStatusBar:
        def showMessage(self, *_args):
            pass

    class FakeWindow:
        def __init__(self, worker, progress):
            self._escalation_worker = worker
            self._escalation_progress_dialog = progress
            self._last_escalation_error = "guardian could not prove quiescence"
            self._last_escalation_result = None

        def statusBar(self):
            return FakeStatusBar()

    messages = []
    monkeypatch.setattr(
        "hydra_suite.detectkit.gui.escalation_actions.QMessageBox.information",
        lambda *args: messages.append(args[2]),
    )
    worker = FakeWorker()
    progress = FakeProgress()
    window = FakeWindow(worker, progress)

    retain_semantic_containment_recovery(window, progress)
    assert progress.visible
    assert progress.button == "Retry cleanup"
    assert window._escalation_worker is worker

    assert not retry_semantic_containment_cleanup(window, worker, progress)
    assert progress.visible
    assert window._escalation_worker is worker

    worker.can_cleanup = True
    assert retry_semantic_containment_cleanup(window, worker, progress)
    assert progress.closed
    assert window._escalation_worker is None
    assert window._escalation_progress_dialog is None
    assert window._last_escalation_error is None
    assert messages and "run it again" in messages[0].lower()
