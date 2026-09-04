"""Headless command-line interface for DetectKit model training."""

from __future__ import annotations

import argparse
import json
import signal
import sys
import threading
import uuid
from contextlib import contextmanager
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterator, TextIO

try:
    import fcntl
except ImportError:  # pragma: no cover - Windows does not provide fcntl.
    fcntl = None

from hydra_suite.data.project_bundle import write_json_atomic
from hydra_suite.detectkit.config.training import TrainingPlanError, load_training_plan
from hydra_suite.detectkit.jobs.training import (
    DatasetPreparationCancelled,
    preflight_sources,
    prepare_role_datasets,
    run_role_entries,
)
from hydra_suite.runtime.process_supervisor import WorkloadStillOwnedError
from hydra_suite.training import PublishPolicy, TrainingOrchestrator
from hydra_suite.training.registry import finalize_run_record


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="detectkit train",
        description="Run DetectKit dataset preparation and training without a GUI.",
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to a DetectKit JSON training plan.",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and print the resolved plan without creating files.",
    )
    mode.add_argument(
        "--prepare-only",
        action="store_true",
        help="Validate and build role datasets without starting training.",
    )
    parser.add_argument(
        "--no-publish",
        action="store_true",
        help="Keep artifacts in the run workspace without importing them into the local model registry.",
    )
    parser.add_argument(
        "--resume",
        metavar="LAST_PT",
        help="Resume a single-role Ultralytics plan from a last.pt checkpoint.",
    )
    return parser


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _print_validation_errors(report) -> None:
    for issue in report.issues:
        path = f" ({issue.path})" if issue.path else ""
        print(
            f"{issue.severity.upper()}: {issue.message}{path}",
            file=sys.stderr,
        )


def _write_session_file(session_dir: Path, name: str, payload: object) -> Path:
    path = session_dir / name
    write_json_atomic(path, _json_safe(payload))
    return path


@contextmanager
def _workspace_session(workspace: Path) -> Iterator[Path]:
    """Create an isolated session directory while exclusively owning a workspace."""

    workspace.mkdir(parents=True, exist_ok=True)
    lock_path = workspace / ".detectkit-training.lock"
    lock_handle: TextIO = lock_path.open("a+", encoding="utf-8")
    try:
        if fcntl is not None:
            try:
                fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise TrainingPlanError(
                    f"Training workspace is already in use: {workspace}"
                ) from exc
        session_id = (
            datetime.now(UTC).strftime("%Y%m%dT%H%M%S.%fZ") + f"-{uuid.uuid4().hex[:8]}"
        )
        session_dir = workspace / "sessions" / session_id
        session_dir.mkdir(parents=True, exist_ok=False)
        yield session_dir
    finally:
        if fcntl is not None:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)
        lock_handle.close()


def _install_cancel_handlers(cancel_event: threading.Event):
    previous: dict[int, Any] = {}

    def request_cancel(signum, _frame) -> None:
        if cancel_event.is_set():
            return
        cancel_event.set()
        print(
            f"Cancellation requested by signal {signum}; stopping at a safe boundary…",
            file=sys.stderr,
            flush=True,
        )

    for signum in (signal.SIGINT, signal.SIGTERM):
        previous[signum] = signal.getsignal(signum)
        signal.signal(signum, request_cancel)
    return previous


def _restore_signal_handlers(previous: dict[int, Any]) -> None:
    for signum, handler in previous.items():
        signal.signal(signum, handler)


def _apply_resume(entries, checkpoint: str | None, config_dir: Path):
    if not checkpoint:
        return entries
    if len(entries) != 1:
        raise TrainingPlanError("--resume requires a plan containing exactly one role")
    entry = entries[0]
    if entry.role.value == "semantic_sam3":
        raise TrainingPlanError("--resume is not supported for SAM3 training")
    checkpoint_path = Path(checkpoint).expanduser()
    if not checkpoint_path.is_absolute():
        checkpoint_path = config_dir / checkpoint_path
    checkpoint_path = checkpoint_path.resolve()
    if not checkpoint_path.is_file():
        raise TrainingPlanError(f"Resume checkpoint not found: {checkpoint_path}")
    resumed_spec = replace(
        entry.spec,
        base_model=str(checkpoint_path),
        resume_from=str(checkpoint_path),
    )
    return [replace(entry, spec=resumed_spec)]


def _validate_resume_request(
    plan, checkpoint: str | None, config_dir: Path
) -> str | None:
    """Reject invalid resume requests before any workspace or dataset work begins."""

    if not checkpoint:
        return None
    if len(plan.roles) != 1:
        raise TrainingPlanError("--resume requires a plan containing exactly one role")
    if plan.roles[0].role.value == "semantic_sam3":
        raise TrainingPlanError("--resume is not supported for SAM3 training")
    checkpoint_path = Path(checkpoint).expanduser()
    if not checkpoint_path.is_absolute():
        checkpoint_path = config_dir / checkpoint_path
    checkpoint_path = checkpoint_path.resolve()
    if not checkpoint_path.is_file():
        raise TrainingPlanError(f"Resume checkpoint not found: {checkpoint_path}")
    return str(checkpoint_path)


def run(args: argparse.Namespace) -> int:
    config_path = Path(args.config).expanduser().resolve()
    plan = load_training_plan(config_path)
    if args.no_publish:
        plan = replace(
            plan,
            publish_policy=PublishPolicy(auto_import=False, auto_select=False),
        )
    resume_checkpoint = _validate_resume_request(plan, args.resume, config_path.parent)

    if args.dry_run:
        print(json.dumps(plan.to_dict(), indent=2))
        return 0

    workspace = plan.workspace_root
    with _workspace_session(workspace) as session_dir:
        print(f"Training session: {session_dir}", flush=True)
        _write_session_file(session_dir, "resolved_training_plan.json", plan.to_dict())

        cancel_event = threading.Event()
        previous_handlers = _install_cancel_handlers(cancel_event)
        try:
            orchestrator = TrainingOrchestrator(workspace)
            try:
                report = preflight_sources(plan.sources)
            except (OSError, RuntimeError, ValueError) as exc:
                raise TrainingPlanError(f"Source preflight failed: {exc}") from exc
            _write_session_file(session_dir, "preflight.json", report.to_dict())
            if not report.valid:
                _print_validation_errors(report)
                _write_session_file(
                    session_dir,
                    "training_result.json",
                    {
                        "success": False,
                        "canceled": False,
                        "stage": "preflight",
                        "results": [],
                    },
                )
                return 2

            prepared = prepare_role_datasets(
                orchestrator,
                plan.preparation_request(),
                log=lambda message: print(message, flush=True),
                status=lambda message: print(message, flush=True),
                should_cancel=cancel_event.is_set,
            )
            preparation_summary = {
                "role_dataset_dirs": prepared.role_dataset_dirs,
                "roles": [role.value for role in prepared.roles],
                "measured_reference_body_px": prepared.measured_reference_body_px,
            }
            _write_session_file(
                session_dir, "prepared_datasets.json", preparation_summary
            )
            if args.prepare_only:
                print(json.dumps(preparation_summary, indent=2))
                return 0

            entries = plan.role_entries(prepared.role_dataset_dirs)
            entries = _apply_resume(entries, resume_checkpoint, config_path.parent)
            results = run_role_entries(
                orchestrator,
                entries,
                log=lambda message: print(message, flush=True),
                progress=lambda role, current, total: print(
                    f"[{role}] progress {current}/{total}", flush=True
                ),
                should_cancel=cancel_event.is_set,
            )
            summary = {
                "success": bool(results)
                and all(bool(result.get("success")) for result in results),
                "canceled": cancel_event.is_set(),
                "results": results,
            }
            summary_path = _write_session_file(
                session_dir, "training_result.json", summary
            )
            print(f"Training summary: {summary_path}")
            if cancel_event.is_set():
                return 130
            return 0 if summary["success"] else 1
        except DatasetPreparationCancelled:
            _write_session_file(
                session_dir,
                "training_result.json",
                {
                    "success": False,
                    "canceled": True,
                    "stage": "dataset_preparation",
                    "results": [],
                },
            )
            print("Dataset preparation canceled.", file=sys.stderr)
            return 130
        except Exception as exc:
            _write_session_file(
                session_dir,
                "training_result.json",
                {
                    "success": False,
                    "canceled": cancel_event.is_set(),
                    "error": str(exc),
                    "results": [],
                },
            )
            raise
        finally:
            _restore_signal_handlers(previous_handlers)


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return run(args)
    except TrainingPlanError as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("Training interrupted.", file=sys.stderr)
        return 130
    except WorkloadStillOwnedError as owned_error:
        # A CLI invocation has no long-lived GUI owner to retain the recovery
        # handle. Try synchronous teardown once; if ownership remains
        # uncertain, preserve the exact exception and sidecar for an in-process
        # caller instead of flattening it to an exit code.
        try:
            owned_error.sidecar.cancel()
        except WorkloadStillOwnedError as retry_error:
            retry_error.run_id = owned_error.run_id
            retry_error.registry_update_error = owned_error.registry_update_error
            retry_error.recovery_error = owned_error.recovery_error
            retry_error.recovery_cleanup = owned_error.recovery_cleanup
            raise
        except Exception as retry_error:  # noqa: BLE001 - retain exact owner
            owned_error.recovery_error = str(retry_error)
            raise owned_error
        if owned_error.recovery_cleanup is not None:
            try:
                owned_error.recovery_cleanup()
            except Exception as cleanup_error:  # noqa: BLE001 - workload is safe
                owned_error.recovery_error = str(cleanup_error)
                print(
                    "Training containment recovery succeeded, but temporary "
                    f"artifact cleanup failed: {cleanup_error}",
                    file=sys.stderr,
                )
        if owned_error.run_id:
            try:
                finalize_run_record(
                    owned_error.run_id,
                    status="failed",
                    error_message=(
                        "Containment recovery completed after process ownership "
                        "was temporarily uncertain."
                    ),
                    failure_details={
                        "failure_kind": "workload-still-owned",
                        "containment": {"ownership": "recovered"},
                    },
                )
            except Exception as registry_error:  # noqa: BLE001 - workload is safe
                owned_error.registry_update_error = str(registry_error)
                print(
                    "Training containment recovery succeeded, but the run "
                    f"registry could not be finalized: {registry_error}",
                    file=sys.stderr,
                )
                return 1
        print(
            "Training failed, but containment recovery completed safely.",
            file=sys.stderr,
        )
        return 1
    except Exception as exc:
        print(f"Training failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
