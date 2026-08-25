"""``HYDRA_PROFILE=1`` — a process-level span recorder.

Replaces the retired ``HYDRA_RT_PROFILE`` machinery in
``core/inference/runner.py``. Two reasons it is not just an alias:

* ``core/inference`` is also driven by DetectKit and PoseKit, which build no
  ``TrackingProfiler``. Without this, retiring the env var would blind them.
* Debug Mode is not observation-only — it changes intermediate cleanup and CSV
  outputs — so "turn on Debug and re-run" profiles a DIFFERENT run than the
  one that was slow. This is the User-mode route to profile the real run.

Precedence: a ``TrackingProfiler`` recorder (``PRIORITY_SESSION``) wins while
armed and the process recorder resumes outside it. Spans go to exactly one
recorder, never both.
"""

from __future__ import annotations

import atexit
import json
import logging
import os
import threading
from pathlib import Path

from .profiling import _ACTIVE, PRIORITY_PROCESS, SpanRecorder
from .profiling_report import render_tree_lines

logger = logging.getLogger(__name__)

_recorder: SpanRecorder | None = None
_lock = threading.Lock()
_video_log_dir: Path | None = None


def enabled() -> bool:
    return bool(os.environ.get("HYDRA_PROFILE") or os.environ.get("HYDRA_RT_PROFILE"))


def set_log_dir(path) -> None:
    """Tell the dump where a session's ``<video>_logs/`` directory is."""
    global _video_log_dir
    _video_log_dir = Path(path) if path else None


def dump_path() -> Path:
    if _video_log_dir is not None:
        return _video_log_dir / f"span_profile_{os.getpid()}.json"
    from hydra_suite.paths import get_data_dir

    return Path(get_data_dir()) / "profiles" / f"span_profile_{os.getpid()}.json"


def maybe_arm_process_recorder() -> SpanRecorder | None:
    """Arm (once) when the env var is set. Returns the recorder, or None.

    Armed for the process lifetime — no ``reset()`` — so any thread that
    inherits or binds this context records into it.
    """
    global _recorder
    if not enabled():
        return None
    with _lock:
        if _recorder is None:
            _recorder = SpanRecorder(priority=PRIORITY_PROCESS)
            atexit.register(dump)
    # Arm on EVERY call, not just at creation: contextvars do not cross
    # threads, so a runner constructed on the GUI thread and driven from a
    # worker thread would otherwise record nothing. `is None` preserves the
    # spec's precedence — a TrackingProfiler armed here keeps the context.
    if _ACTIVE.get() is None:
        _ACTIVE.set(_recorder)
    return _recorder


def dump() -> None:
    """Write the tree to JSON and log it. Never raises."""
    if _recorder is None:
        return
    snap = _recorder.snapshot()
    try:
        path = dump_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"spans": snap}, indent=2))
        logger.info("Span profile written to %s", path)
    except Exception:  # noqa: BLE001
        logger.warning("Failed to write span profile", exc_info=True)
    for line in render_tree_lines(snap, main_thread=snap["thread"]):
        logger.info("%s", line)


def reset_for_test() -> None:
    """Test hook only."""
    global _recorder, _video_log_dir
    _recorder = None
    _video_log_dir = None
    _ACTIVE.set(None)
