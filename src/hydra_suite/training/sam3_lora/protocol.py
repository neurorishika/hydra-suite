"""Wire protocol for progress/log records streamed from the SAM3 sidecar
child process back to the launcher.

The child (``cli.py``, running inside the ``hydra-sam3`` conda env) writes
plain text to stdout for ordinary logging, and single-line JSON records
prefixed with ``PROGRESS_PREFIX`` for anything the launcher (``train.py``,
running in the parent ``hydra-mps``/``hydra-cuda`` env) should forward to
``progress_cb``/``log_cb`` structurally. The sentinel means a partial write
of some other line can never be mistaken for a progress record, and a
record is always exactly one line.

Pure stdlib only: importable from both the parent process (no ``sam3``,
no heavy deps) and the sidecar child.
"""

from __future__ import annotations

import json
from typing import Any, Callable, Optional

PROGRESS_PREFIX = "@@HYDRA_SAM3_PROGRESS@@"


def emit_log(message: str) -> None:
    """Print a log record to stdout, for the launcher to forward to `log_cb`."""
    print(PROGRESS_PREFIX + json.dumps({"type": "log", "message": message}), flush=True)


def emit_progress(epoch: int, total: int) -> None:
    """Print a progress record to stdout, for the launcher to forward to `progress_cb`."""
    print(
        PROGRESS_PREFIX
        + json.dumps({"type": "progress", "epoch": epoch, "total": total}),
        flush=True,
    )


def parse_record(line: str) -> Optional[dict[str, Any]]:
    """Parse one child stdout line as a progress/log record.

    Returns ``None`` if the line is not one -- either it lacks the sentinel
    prefix, or the payload after it is not valid JSON, or it is JSON but not
    an object. All three are the caller's cue to treat the *entire* line as
    plain log text rather than raising: a child dependency printing an
    unrelated warning must never crash the launcher.
    """
    if not line.startswith(PROGRESS_PREFIX):
        return None
    payload = line[len(PROGRESS_PREFIX) :]
    try:
        record = json.loads(payload)
    except json.JSONDecodeError:
        return None
    return record if isinstance(record, dict) else None


def dispatch_record(
    record: dict[str, Any],
    log_cb: Callable[[str], None],
    progress_cb: Callable[[int, int], None],
) -> None:
    """Forward a parsed record to the appropriate callback.

    Unrecognised `type` values or malformed payloads are ignored, not
    raised -- a forward-incompatible child (newer than the launcher) should
    degrade silently, not crash the run.
    """
    kind = record.get("type")
    if kind == "log":
        message = record.get("message")
        if isinstance(message, str):
            log_cb(message)
    elif kind == "progress":
        epoch = record.get("epoch")
        total = record.get("total")
        if isinstance(epoch, int) and isinstance(total, int):
            progress_cb(epoch, total)
