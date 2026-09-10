"""A retained-ownership failure must not strand this process's heavy-job leases.

The semantic workers have no interactive "retry cleanup" affordance, so a
``WorkloadStillOwnedError`` that escapes them leaves the lease flock held on a
descriptor owned by the GUI process for the rest of its life.  Every later
protected job then refuses admission with ``ResourceBusyError ... leased by
PID <self> (live)``.
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
