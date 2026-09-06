"""Offscreen end-to-end smoke for the TrackerKit parallel-batch fan-out.

This is the automatable stand-in for "click Start Full Tracking with Batch +
Parallel ticked": it builds a real ``MainWindow`` under
``QT_QPA_PLATFORM=offscreen``, drives the same Setup-panel widgets a user
would, then goes through the REAL click path (``toggle_tracking(True)`` ->
``start_full`` -> ``start_tracking`` -> ``start_batch_fanout``) with the
confirmation dialog auto-answered, and spins the event loop until the run
reports back. Two legs are exercised:

1. **run to completion** -- both fixture clips must reach OK and the window
   must return to idle with the worker reference released;
2. **cancel** -- ``stop_tracking()`` a few seconds in must land a cancelled
   result, kill both children, and restore the same idle UI;
3. **close mid-run** -- ``window.close()`` must hit the "Tracking In Progress"
   prompt, stop the fan-out, and leave no orphaned child processes.

It is NOT a pytest module (no ``test_`` prefix, lives outside the collected
tree) because it launches real child tracking processes for ~1 minute.

Usage::

    conda activate hydra-mps
    PYTHONPATH=$PWD/src python tests/manual/gui_fanout_smoke.py
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import time
from pathlib import Path

# Must precede any PySide6 import, and must be exported so the children (which
# only need it for KMP) inherit a sane environment.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

REPO = Path(__file__).resolve().parents[2]
SRC = REPO / "src"
# The children are `python -m hydra_suite.trackerkit.app`, so they must import
# THIS worktree's src, not an editable install of main.
os.environ["PYTHONPATH"] = (
    f"{SRC}{os.pathsep}{os.environ['PYTHONPATH']}"
    if os.environ.get("PYTHONPATH")
    else str(SRC)
)
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

# The equivalence fixture CLIPS are checked out in the main repo only; a
# worktree's clips/ directory is gitignored and empty. Derive the main repo
# from the worktree layout (<main>/.worktrees/<name>) so this keeps working
# after a merge to main, where REPO already IS the main repo.
MAIN_REPO = REPO.parents[1] if REPO.parent.name == ".worktrees" else REPO
CLIPS = MAIN_REPO / "tools" / "equivalence" / "fixtures" / "clips"
CONFIGS = REPO / "tools" / "equivalence" / "fixtures" / "configs"
VIDEOS = ("fly_obb", "worm_bgsub")
END_FRAME = 99
RUN_TIMEOUT_S = 180.0
CANCEL_AFTER_S = 3.0


def _log(message: str) -> None:
    print(f"[smoke] {message}", flush=True)


# --------------------------------------------------------------------------
# Fixture staging
# --------------------------------------------------------------------------


def _runner_module():
    """Import ``tools/equivalence/runner.py`` for its DISABLE + runtime map."""
    import importlib.util

    path = REPO / "tools" / "equivalence" / "runner.py"
    spec = importlib.util.spec_from_file_location("equiv_runner", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def stage_scratch(tag: str) -> tuple[Path, list[str]]:
    """Symlink the clips and write a per-video sidecar into a fresh dir.

    Each video gets its OWN ``<stem>_config.json`` so the batch planner treats
    them independently -- a single ``--config`` would make one of them the
    implicit keystone for both (Task 5's finding).
    """
    runner = _runner_module()
    scratch = Path(f"/tmp/gui_fanout_smoke_{tag}")
    if scratch.exists():
        shutil.rmtree(scratch)
    scratch.mkdir(parents=True)
    paths: list[str] = []
    for stem in VIDEOS:
        clip = CLIPS / f"{stem}.mp4"
        if not clip.is_file():
            raise SystemExit(f"missing fixture clip: {clip}")
        link = scratch / f"{stem}.mp4"
        link.symlink_to(clip)
        cfg = json.loads((CONFIGS / f"{stem}.json").read_text())
        cfg.update(runner.DISABLE)
        # The fixture configs predate runtime_tier; without this the loader
        # raises the loud migration error.
        cfg.update(runner.runtime_overrides("gpu"))
        cfg["start_frame"] = 0
        cfg["end_frame"] = END_FRAME
        (scratch / f"{stem}_config.json").write_text(json.dumps(cfg, indent=2))
        paths.append(str(link))
    _log(f"staged {scratch} with {len(paths)} videos (frames 0-{END_FRAME})")
    return scratch, paths


# --------------------------------------------------------------------------
# GUI driving
# --------------------------------------------------------------------------


def silence_message_boxes() -> list[str]:
    """Modal ``exec()`` never returns offscreen -- record and auto-answer."""
    from PySide6.QtWidgets import QMessageBox

    seen: list[str] = []

    def _make(name, answer):
        def _shim(parent, title, text, *args, **kwargs):
            seen.append(f"{name}: {title} -- {str(text)[:160]}")
            _log(f"QMessageBox.{name}({title!r}) suppressed")
            return answer

        return staticmethod(_shim)

    QMessageBox.information = _make("information", QMessageBox.Ok)
    QMessageBox.warning = _make("warning", QMessageBox.Ok)
    QMessageBox.critical = _make("critical", QMessageBox.Ok)
    QMessageBox.question = _make("question", QMessageBox.Yes)
    return seen


def build_window(videos: list[str]):
    from hydra_suite.trackerkit.gui.main_window import MainWindow

    window = MainWindow()
    setup = window._setup_panel
    _log(f"loading keystone {Path(videos[0]).name}")
    window._setup_video_file(videos[0])
    # Ticking Batch re-seeds batch_videos from the loaded video, so the list
    # must be assigned AFTER the toggle, not before.
    setup.g_batch.setChecked(True)
    window.batch_videos = list(videos)
    window._sync_batch_list_ui()
    setup.chk_batch_parallel.setChecked(True)
    setup.spin_batch_parallel_jobs.setValue(2)
    window._on_batch_parallel_changed()
    assert window.config.batch_parallel is True
    assert window.config.batch_parallel_jobs == 2
    _log(
        f"batch={setup.g_batch.isChecked()} parallel={setup.chk_batch_parallel.isChecked()} "
        f"jobs={window.config.batch_parallel_jobs} videos={len(window.batch_videos)}"
    )
    return window


def run_leg(app, videos: list[str], *, cancel_after: float | None):
    from PySide6.QtCore import QTimer

    window = build_window(videos)
    orch = window._tracking_orch
    captured: dict = {}
    assert hasattr(window, "batch_fanout_worker"), "MainWindow is missing the attribute"
    # The REAL click path: btn_start -> toggle_tracking -> start_full ->
    # start_tracking -> start_batch_fanout. The confirmation QMessageBox is
    # answered by the shim installed in silence_message_boxes().
    window.btn_start.setChecked(True)
    window.toggle_tracking(True)
    worker = window.batch_fanout_worker
    assert worker is not None, "worker reference was not stored"
    worker.fanout_finished.connect(lambda r: captured.__setitem__("result", r))
    _log(
        f"btn_start text while running: {window.btn_start.text()!r} "
        f"progress_hidden={window.progress_bar.isHidden()}"
    )
    assert window.btn_start.text() == "Stop Tracking"
    # The window is never shown, so isVisible() is False no matter what and
    # would make the post-run check vacuous. isHidden() tracks the explicit
    # setVisible() calls, so this pair can actually fail.
    assert not window.progress_bar.isHidden(), "progress bar was not shown for the run"

    if cancel_after is not None:
        QTimer.singleShot(int(cancel_after * 1000), orch.stop_tracking)

    deadline = time.monotonic() + RUN_TIMEOUT_S
    while "result" not in captured and time.monotonic() < deadline:
        app.processEvents()
        time.sleep(0.05)
    if "result" not in captured:
        raise SystemExit(f"fanout_finished never fired within {RUN_TIMEOUT_S}s")
    # Drain the queued QThread.finished so the reference cleanup can run.
    drain = time.monotonic() + 15.0
    while window.batch_fanout_worker is not None and time.monotonic() < drain:
        app.processEvents()
        time.sleep(0.05)

    result = captured["result"]
    ok = sum(1 for job in result.jobs if job.success)
    _log(
        f"result: {ok}/{len(result.jobs)} succeeded, cancelled={result.cancelled}, "
        f"errors={[j.error for j in result.jobs]}"
    )
    _log(
        f"UI after: btn_start={window.btn_start.text()!r} "
        f"progress_hidden={window.progress_bar.isHidden()} "
        f"batch_index={window.current_batch_index} "
        f"worker_ref={window.batch_fanout_worker!r}"
    )
    dialog = window.batch_fanout_dialog
    rows = [
        dialog._table.item(row, 3).text() for row in range(dialog._table.rowCount())
    ]
    _log(f"dialog status column: {rows}")
    return window, result, rows


def child_pids() -> list[str]:
    """PIDs of live fan-out children (`python -m hydra_suite.trackerkit.app ...`).

    This process is `python tests/manual/gui_fanout_smoke.py`, so it can never
    match the pattern itself -- every hit is a child.
    """
    import subprocess

    out = subprocess.run(
        ["pgrep", "-f", "hydra_suite.trackerkit.app"],
        capture_output=True,
        text=True,
        check=False,
    ).stdout
    return [line for line in out.split() if line.strip()]


def close_leg(app) -> None:
    """Closing the window mid-run must prompt, stop, and orphan no children.

    MainWindow.closeEvent only prompts when _has_active_tracking_workers()
    says something is running. A fan-out leaves tracking_worker None, so if
    batch_fanout_worker is missing from that tuple the close sails straight
    through: the QThread is destroyed while running and the children -- which
    are in their OWN session group -- keep holding their GPUs forever.
    """
    _, videos = stage_scratch("close")
    window = build_window(videos)
    window.btn_start.setChecked(True)
    window.toggle_tracking(True)
    worker = window.batch_fanout_worker
    assert worker is not None, "worker reference was not stored"

    deadline = time.monotonic() + 30.0
    while not child_pids() and time.monotonic() < deadline:
        app.processEvents()
        time.sleep(0.05)
    launched = child_pids()
    _log(f"children launched before close: {launched}")
    check(bool(launched), "at least one child process was running before the close")
    check(
        window._has_active_tracking_workers(),
        "_has_active_tracking_workers() sees the running fan-out",
    )

    # The "Tracking In Progress" prompt is answered Yes by the shim.
    _log("calling window.close() mid-run…")
    window.close()

    drain = time.monotonic() + 30.0
    while worker.isRunning() and time.monotonic() < drain:
        app.processEvents()
        time.sleep(0.05)
    check(not worker.isRunning(), "the fan-out worker thread was joined by the close")

    settle = time.monotonic() + 15.0
    while child_pids() and time.monotonic() < settle:
        app.processEvents()
        time.sleep(0.1)
    remaining = child_pids()
    _log(f"children remaining after close: {remaining}")
    check(not remaining, f"no orphaned child processes after close (got {remaining})")


def check(condition: bool, message: str) -> None:
    _log(("PASS  " if condition else "FAIL  ") + message)
    if not condition:
        raise SystemExit(f"assertion failed: {message}")


def main() -> int:
    from PySide6.QtWidgets import QApplication

    boxes = silence_message_boxes()
    app = QApplication.instance() or QApplication([])

    _log("=" * 70)
    _log("LEG 1: run to completion")
    _log("=" * 70)
    _, videos = stage_scratch("run")
    window, result, rows = run_leg(app, videos, cancel_after=None)
    check(not result.cancelled, "run leg is not cancelled")
    check(
        sum(1 for job in result.jobs if job.success) == 2,
        "2/2 videos succeeded",
    )
    check(rows == ["OK", "OK"], f"dialog rows all OK (got {rows})")
    check(window.btn_start.text() == "Start Full Tracking", "btn_start restored")
    check(window.batch_fanout_worker is None, "worker reference cleared")
    check(window.current_batch_index == -1, "batch index reset")
    check(window.progress_bar.isHidden(), "progress bar hidden")
    window.close()

    _log("")
    _log("=" * 70)
    _log(f"LEG 2: cancel after {CANCEL_AFTER_S}s via stop_tracking()")
    _log("=" * 70)
    _, videos2 = stage_scratch("cancel")
    window2, result2, rows2 = run_leg(app, videos2, cancel_after=CANCEL_AFTER_S)
    check(result2.cancelled, "cancel leg reports cancelled")
    check(
        all(not job.success for job in result2.jobs),
        "no job is reported successful after a cancel",
    )
    check(
        all(row in {"cancelled", "not started"} for row in rows2),
        f"dialog rows terminal after cancel (got {rows2})",
    )
    check(window2.btn_start.text() == "Start Full Tracking", "btn_start restored")
    check(window2.batch_fanout_worker is None, "worker reference cleared")
    check(window2.current_batch_index == -1, "batch index reset")
    check(window2.progress_bar.isHidden(), "progress bar hidden")
    window2.close()

    _log("")
    _log("=" * 70)
    _log("LEG 3: close the main window mid-run")
    _log("=" * 70)
    close_leg(app)

    _log("")
    _log(f"suppressed message boxes ({len(boxes)}):")
    for line in boxes:
        _log("  " + line)
    _log("ALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
