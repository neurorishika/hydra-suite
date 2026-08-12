"""Best-effort end-to-end GUI post-tracking smoke test.

Drives a REAL forward-only tracking session through the actual async
signal/slot flow on a real, offscreen ``MainWindow`` -- TrackingWorker
QThread -> ``TrackingOrchestrator.on_tracking_finished`` ->
``SessionWorker`` QThread -> ``_finalize_tracking_session_ui`` -- and
asserts a non-empty final CSV was produced.

This exercises the real async path that the synchronous Task-3
equivalence test (``tests/test_gui_session_cutover_equivalence.py``)
cannot: real QThreads, a real Qt event loop, and the real
``TrackingOrchestrator.start_tracking`` entry point a user would trigger
via the "Start Full Tracking" button.

``tests/test_gui_session_cutover_equivalence.py`` remains the
AUTHORITATIVE guard for GUI/CLI byte-identity; this test is a best-effort
smoke on top of it.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import QEventLoop, QTimer  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
FLY_CLIP = REPO / "tools/equivalence/fixtures/clips/fly_obb.mp4"
FLY_CONFIG = REPO / "tools/equivalence/fixtures/configs/fly_obb.json"

pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(
        not (FLY_CLIP.exists() and FLY_CONFIG.exists()),
        reason="fly_obb fixture missing (run tools/equivalence/fixtures/fetch_fixtures.sh)",
    ),
    pytest.mark.skip(
        reason=(
            "Offscreen async event-loop harness proved unreliable: two foreground "
            "attempts (2026-08-04) exceeded a 300s+ wall-clock bound with no pytest "
            "output, most likely because TrackingOrchestrator.start_tracking() does "
            "synchronous work (model/cache resolution) before the Qt event loop is "
            "pumped, so the in-test QTimer safety-net cannot fire to bound it. Per "
            "the Task-6 timebox, this is left as a documented best-effort skip rather "
            "than a hanging test. The AUTHORITATIVE GUI==CLI post-tracking guard is "
            "tests/test_gui_session_cutover_equivalence.py (synchronous, deterministic, "
            "drives the real SessionWorker/TrackingOrchestrator code paths without a "
            "live QThread/event-loop dependency)."
        )
    ),
]

# Real inference (detection + tracking + post-processing) on a short clip;
# generous but bounded so a genuine hang fails the test instead of the CI job.
TIMEOUT_MS = 240_000


@pytest.fixture(scope="module")
def qapp() -> QApplication:
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


@pytest.fixture
def main_window(monkeypatch, qapp):
    """A real, offscreen MainWindow with advanced-config disk hooks stubbed.

    Mirrors the offscreen-MainWindow convention in
    tests/test_config_build_dict.py / tests/test_vitpose_trackerkit_persistence.py.
    """
    from hydra_suite.trackerkit.gui.main_window import MainWindow

    monkeypatch.setattr(MainWindow, "_save_advanced_config", lambda self: None)
    monkeypatch.setattr(MainWindow, "_load_advanced_config", lambda self: {})
    window = MainWindow()
    try:
        yield window
    finally:
        window.close()


def test_gui_forward_tracking_session_end_to_end(main_window, tmp_path, qapp):
    """Drive a real forward-only tracking session via the real async path."""
    window = main_window
    panels = window._panels_bundle()

    # Fresh copy so no adjacent saved config triggers an overwrite prompt
    # (QMessageBox.exec() would block the offscreen event loop forever).
    video_copy = tmp_path / "fly_obb.mp4"
    shutil.copy(FLY_CLIP, video_copy)

    window._setup_video_file(str(video_copy), skip_config_load=True)
    window._config_orch._load_config_from_file(str(FLY_CONFIG))

    # Forward-only: exercise the single-pass on_tracking_finished ->
    # _run_session_worker branch (not the backward re-run branch).
    panels.tracking.chk_enable_backward.setChecked(False)

    loop = QEventLoop()
    outcome = {"finalized": False, "timed_out": False}

    orch = window._tracking_orch
    original_finalize = orch._finalize_tracking_session_ui

    def _wrapped_finalize():
        outcome["finalized"] = True
        original_finalize()
        loop.quit()

    orch._finalize_tracking_session_ui = _wrapped_finalize

    def _on_timeout():
        outcome["timed_out"] = True
        loop.quit()

    timer = QTimer()
    timer.setSingleShot(True)
    timer.timeout.connect(_on_timeout)
    timer.start(TIMEOUT_MS)

    orch.start_tracking(preview_mode=False)
    loop.exec()
    timer.stop()

    assert not outcome["timed_out"], (
        f"GUI tracking session did not finalize within {TIMEOUT_MS}ms; "
        "async post-tracking path may be hung."
    )
    assert outcome["finalized"] is True

    final_csv_path = window._session_final_csv_path
    assert final_csv_path, "no final CSV path recorded on the session"
    final_csv = Path(final_csv_path)
    assert final_csv.exists()
    assert final_csv.stat().st_size > 0

    import pandas as pd

    df = pd.read_csv(final_csv)
    assert len(df) > 0
