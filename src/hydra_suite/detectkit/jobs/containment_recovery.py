"""Shared retained-ownership recovery for workers driving a protected sidecar.

When a supervised sidecar cannot prove its process tree exited it raises
``WorkloadStillOwnedError`` and DELIBERATELY keeps the heavy-job leases held --
the workload may still be consuming the memory those leases admit.  The lease
locks live on file descriptors owned by THIS process, so a worker that lets the
exception escape without ever retrying teardown strands them for the lifetime of
the application: every later job in the same process then fails admission with
``ResourceBusyError ... leased by PID <self> (live)``.

Every worker that owns a ``ProtectedOperation`` must therefore either retry the
teardown itself or expose the retry to its GUI owner.  This mixin is the one
implementation of that contract.
"""

from __future__ import annotations

from typing import Any, Callable

from hydra_suite.runtime.process_supervisor import WorkloadStillOwnedError
from hydra_suite.runtime.safe_text import bounded_terminal_text

RETAINED_OWNERSHIP_NOTE = (
    "DetectKit could not yet prove that its protected process tree exited, so "
    "resource ownership is retained for safety. Wait for the model process to "
    "stop, then try again."
)


class ContainmentRecoveryMixin:
    """Retry a retained sidecar teardown without dropping an unproven owner."""

    recovery_cleanup_error: str
    failure_exception: Exception | None

    @property
    def containment_recovery_required(self) -> bool:
        """Whether a sidecar remains the durable owner of a workload."""

        return isinstance(self.failure_exception, WorkloadStillOwnedError)

    def retry_containment_cleanup(self) -> bool:
        """Retry teardown, preserving ownership when the retry cannot prove it."""

        error = self.failure_exception
        if not isinstance(error, WorkloadStillOwnedError):
            return True
        try:
            error.sidecar.cancel()
        except WorkloadStillOwnedError as retry_error:
            retry_error.recovery_cleanup = error.recovery_cleanup
            self.failure_exception = retry_error
            return False
        except Exception as retry_error:  # noqa: BLE001 - retain uncertain owner
            error.recovery_error = bounded_terminal_text(
                retry_error, include_exception_type=False
            )
            return False
        if error.recovery_cleanup is not None:
            try:
                error.recovery_cleanup()
            except Exception as cleanup_error:  # noqa: BLE001 - workload is safe
                self.recovery_cleanup_error = bounded_terminal_text(
                    cleanup_error, include_exception_type=False
                )
        self.failure_exception = None
        return True

    def run_protected(
        self,
        operation: Any,
        *,
        progress: Callable[[int, str], None] | None = None,
        log: Callable[[str], None] | None = None,
    ) -> Any:
        """Run a protected operation, releasing leases when teardown can be proven.

        A worker with no interactive "retry cleanup" affordance must not simply
        propagate ``WorkloadStillOwnedError``: doing so strands this process's
        leases.  Retry the teardown once here; only if THAT cannot prove
        quiescence is the ownership genuinely retained and re-raised.
        """

        try:
            extra = {} if log is None else {"log": log}
            return operation.run(progress=progress, **extra)
        except WorkloadStillOwnedError as error:
            self.failure_exception = error
            if self.retry_containment_cleanup():
                raise RuntimeError(
                    "The protected model process was stopped and its resources "
                    "were released, but this run did not complete. Run it "
                    f"again.\n\nDetails: {bounded_terminal_text(error)}"
                ) from error
            pending = self.failure_exception
            raise pending if isinstance(pending, BaseException) else error
