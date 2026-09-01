"""SAM3 LoRA training launcher.

Qt-free. This module never imports ``sam3`` or ``torch`` -- it only launches
and streams a subprocess that does. The actual training loop lives in
``cli.py``, which runs inside the dedicated ``hydra-sam3`` sidecar conda env
(see ``docs/superpowers/specs/2026-09-01-sam3-training-sidecar-env-design.md``),
because ``sam3`` pins ``numpy<2`` and cannot coexist with the numpy 2.x
runtimes in ``hydra-mps``/``hydra-cuda``.

``train_sam3_lora``'s signature and return contract are unchanged from the
in-process version: callers (the training dispatch, the publish path) do not
need to know training now happens in a child process.

THE CRITICAL RULE, carried across the process boundary: a child that exits 0
without having written ``adapters.pt`` must still produce `success: False`.
Zero-initialised LoRA `lora_B` makes an untrained adapter a mathematical
no-op, so treating a clean exit code as success would publish a checkpoint
identical to stock SAM3 -- exactly the failure mode `cli.py`'s in-process
predecessor guarded against with its zero-datapoint refusal. Never infer
success from the exit code alone.
"""

from __future__ import annotations

import json
import os
import platform
import queue
import signal
import subprocess
import threading
from pathlib import Path
from typing import Any, Callable, Optional

from hydra_suite.utils.conda_utils import popen_conda

from .env import resolve_sam3_env, sam3_env_command, sam3_env_environ
from .preflight import preflight
from .protocol import dispatch_record, parse_record

# How long to wait for a graceful `terminate()` before escalating to `kill()`.
TERMINATE_GRACE_S = 5.0

# How often the main loop polls the stdout queue (and, on every poll, checks
# `should_cancel()`). The child emits progress roughly once per epoch, which
# can be minutes apart; sampling `should_cancel()` only when a line arrives
# left Cancel unresponsive (or entirely inert against a silent/hung child)
# for that whole window. Polling on a timer instead of on output arrival
# decouples cancellation latency from the child's output cadence.
CANCEL_POLL_INTERVAL_S = 0.2

# How many trailing plain-text lines to keep for the error message on a
# non-zero exit (the child's own stderr is merged into this stream).
STDERR_TAIL_LINES = 20

# `conda run` execs a shell wrapper around the real python grandchild that
# does the training; signalling only the wrapper (`process.terminate()`)
# can leave that grandchild alive holding tens of GB of VRAM. On POSIX we
# launch the child in its own session (`start_new_session=True`) and signal
# the whole process group instead. `conda_utils` exists precisely because
# Windows differs here -- there `start_new_session`/process groups work
# differently, so Windows keeps the plain `process.terminate()`/`.kill()`
# path, same as before this fix.
_IS_POSIX = platform.system() != "Windows"


def train_sam3_lora(
    spec: Any,
    run_dir: str,
    *,
    log_cb: Optional[Callable[[str], None]] = None,
    progress_cb: Optional[Callable[[int, int], None]] = None,
    should_cancel: Optional[Callable[[], bool]] = None,
) -> dict:
    """Finetune SAM3 with LoRA adapters against `spec`, in the sidecar env.

    Returns a dict with keys `success`, `artifact_path`, `metrics_path`,
    `canceled` (and `error_message` on refusal/failure).
    """
    log_cb = log_cb or (lambda msg: None)
    progress_cb = progress_cb or (lambda epoch, total: None)
    should_cancel = should_cancel or (lambda: False)

    def _refuse(message: str) -> dict:
        return {
            "success": False,
            "error_message": message,
            "artifact_path": None,
            "metrics_path": None,
            "canceled": False,
        }

    refusals = preflight(spec)
    if refusals:
        return _refuse("; ".join(refusals))

    run_dir_path = Path(run_dir).expanduser().resolve()
    run_dir_path.mkdir(parents=True, exist_ok=True)

    spec_path = run_dir_path / "spec.json"
    spec_path.write_text(json.dumps(spec.to_dict(), indent=2), encoding="utf-8")

    sam3_params = getattr(spec, "sam3_params", None)
    configured_env = getattr(sam3_params, "env_name", None) if sam3_params else None
    env_name = resolve_sam3_env(configured_env)

    command = sam3_env_command(
        env_name,
        [
            "hydra_suite.training.sam3_lora.cli",
            "--spec",
            str(spec_path),
            "--run-dir",
            str(run_dir_path),
        ],
    )
    child_environ = {**os.environ, **sam3_env_environ()}

    # A stale adapters.pt from a reused run_dir must never be mistaken for
    # this run's output: if the child exits 0 without writing a new one, the
    # existence check below has to see it genuinely absent, not a leftover.
    artifact_path = run_dir_path / "adapters.pt"
    artifact_path.unlink(missing_ok=True)

    popen_kwargs: dict[str, Any] = {}
    if _IS_POSIX:
        popen_kwargs["start_new_session"] = True

    try:
        process = popen_conda(
            command,
            env=child_environ,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            errors="replace",
            bufsize=1,
            **popen_kwargs,
        )
    except FileNotFoundError as exc:
        return _refuse(
            f"conda was not found on PATH while launching the {env_name!r} "
            f"sidecar env: {exc}"
        )

    canceled = False
    tail_lines: list[str] = []
    stream_error: Optional[Exception] = None

    assert process.stdout is not None
    line_queue: "queue.Queue[tuple[str, Any]]" = queue.Queue()
    reader_thread = threading.Thread(
        target=_pump_stdout,
        args=(process.stdout, line_queue),
        daemon=True,
    )
    reader_thread.start()

    try:
        while True:
            if should_cancel():
                canceled = True
                break
            try:
                kind, payload = line_queue.get(timeout=CANCEL_POLL_INTERVAL_S)
            except queue.Empty:
                # No output since the last poll -- but we still re-check
                # `should_cancel()` on this timer regardless of whether the
                # child has said anything, which is the whole point.
                continue
            if kind == "eof":
                break
            if kind == "error":
                stream_error = payload
                break
            raw_line: str = payload
            line = raw_line.rstrip("\n")
            if not line:
                continue
            record = parse_record(line)
            if record is not None:
                dispatch_record(record, log_cb, progress_cb)
                continue
            log_cb(line)
            tail_lines.append(line)
            if len(tail_lines) > STDERR_TAIL_LINES:
                tail_lines.pop(0)
    except Exception as exc:  # noqa: BLE001 -- anything here (log_cb or
        # progress_cb raising) must still reap the child below, never
        # propagate past a live orphaned process. Re-raised after the
        # child is reaped.
        stream_error = exc
    finally:
        # Reap the child on EVERY exit path -- cancellation, a clean finish,
        # or an exception above -- so a raising caller can never orphan a
        # live, multi-hour training process. This must happen BEFORE joining
        # the reader thread below: the thread's blocking read only returns
        # once the child's stdout pipe closes, which happens when the child
        # exits (or is killed here), not before.
        if canceled or stream_error is not None:
            _terminate_then_kill(process)
        else:
            try:
                process.wait(timeout=TERMINATE_GRACE_S)
            except subprocess.TimeoutExpired:
                _terminate_then_kill(process)
        # The reader thread is daemonic and exits on its own once the pipe
        # closes, so this join is best-effort cleanup, not a correctness
        # requirement (the process is already reaped above either way).
        reader_thread.join(timeout=TERMINATE_GRACE_S)

    if stream_error is not None:
        raise stream_error

    if canceled:
        return {
            "success": False,
            "canceled": True,
            "artifact_path": None,
            "metrics_path": None,
        }

    if process.returncode != 0:
        tail = "\n".join(tail_lines) if tail_lines else "(no output)"
        return _refuse(
            f"SAM3 training subprocess (env {env_name!r}) exited with code "
            f"{process.returncode}. Child output tail:\n{tail}"
        )

    if not artifact_path.exists() or artifact_path.stat().st_size == 0:
        return _refuse(
            "SAM3 training subprocess exited successfully but did not write "
            f"a non-empty {artifact_path.name}; refusing to report success "
            "for a run that trained nothing."
        )

    metrics_candidate = run_dir_path / "val_stats.json"
    metrics_path = metrics_candidate if metrics_candidate.exists() else None

    return {
        "success": True,
        "artifact_path": str(artifact_path),
        "metrics_path": str(metrics_path) if metrics_path else None,
        "canceled": False,
    }


def _pump_stdout(stdout: Any, line_queue: "queue.Queue[tuple[str, Any]]") -> None:
    """Read `stdout` line-by-line on a background thread, feeding `line_queue`.

    Runs entirely off the main loop so `should_cancel()` can be polled on a
    timer (`CANCEL_POLL_INTERVAL_S`) instead of once per line -- a silent or
    slow-to-produce-output child would otherwise never (or only very late)
    get its cancellation checked.
    """
    try:
        for raw_line in stdout:
            line_queue.put(("line", raw_line))
    except Exception as exc:  # noqa: BLE001 -- surfaced to the main thread
        # via the queue rather than raised here, where nothing could react.
        line_queue.put(("error", exc))
    finally:
        line_queue.put(("eof", None))


def _terminate_then_kill(process: "subprocess.Popen[str]") -> None:
    """Escalate terminate -> grace period -> kill, signalling the whole
    process group on POSIX so `conda run`'s python grandchild is reached
    too (see `_IS_POSIX` above)."""
    _send_signal(process, terminate=True)
    try:
        process.wait(timeout=TERMINATE_GRACE_S)
    except subprocess.TimeoutExpired:
        _send_signal(process, terminate=False)
        process.wait()


def _send_signal(process: "subprocess.Popen[str]", *, terminate: bool) -> None:
    if _IS_POSIX:
        sig = signal.SIGTERM if terminate else signal.SIGKILL
        try:
            os.killpg(os.getpgid(process.pid), sig)
            return
        except (ProcessLookupError, PermissionError, OSError):
            pass  # process group already gone, or this process is not a
            # session leader (e.g. a test double) -- fall back below.
    if terminate:
        process.terminate()
    else:
        process.kill()
