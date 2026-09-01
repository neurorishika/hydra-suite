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
import subprocess
from pathlib import Path
from typing import Any, Callable, Optional

from hydra_suite.utils.conda_utils import popen_conda

from .env import resolve_sam3_env, sam3_env_command, sam3_env_environ
from .preflight import preflight
from .protocol import dispatch_record, parse_record

# How long to wait for a graceful `terminate()` before escalating to `kill()`.
TERMINATE_GRACE_S = 5.0

# How many trailing plain-text lines to keep for the error message on a
# non-zero exit (the child's own stderr is merged into this stream).
STDERR_TAIL_LINES = 20


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

    try:
        process = popen_conda(
            command,
            env=child_environ,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
    except FileNotFoundError as exc:
        return _refuse(
            f"conda was not found on PATH while launching the {env_name!r} "
            f"sidecar env: {exc}"
        )

    canceled = False
    tail_lines: list[str] = []

    try:
        assert process.stdout is not None
        for raw_line in process.stdout:
            if should_cancel():
                canceled = True
                break
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
    finally:
        # Terminate in `finally` so a raising parent (or an exception from a
        # callback above) can never orphan the child.
        if canceled:
            process.terminate()
            try:
                process.wait(timeout=TERMINATE_GRACE_S)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        else:
            process.wait()

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

    artifact_path = run_dir_path / "adapters.pt"
    if not artifact_path.exists():
        return _refuse(
            "SAM3 training subprocess exited successfully but did not write "
            f"{artifact_path.name}; refusing to report success for a run "
            "that trained nothing."
        )

    metrics_candidate = run_dir_path / "val_stats.json"
    metrics_path = metrics_candidate if metrics_candidate.exists() else None

    return {
        "success": True,
        "artifact_path": str(artifact_path),
        "metrics_path": str(metrics_path) if metrics_path else None,
        "canceled": False,
    }
