"""core/tracking is Qt-free; TrackingEngineCore is a plain, callback-driven engine."""

import ast
from pathlib import Path


def test_core_tracking_imports_no_qt():
    import hydra_suite.core.tracking as pkg

    pkg_dir = Path(pkg.__file__).parent
    offenders = []
    for py in pkg_dir.rglob("*.py"):
        tree = ast.parse(py.read_text(), filename=str(py))
        for node in ast.walk(tree):
            mod = None
            if isinstance(node, ast.ImportFrom):
                mod = node.module
            elif isinstance(node, ast.Import):
                mod = ",".join(a.name for a in node.names)
            if mod and any(q in mod for q in ("PySide6", "QtCore")):
                offenders.append(f"{py.relative_to(pkg_dir)}:{node.lineno}")
    assert not offenders, "core/tracking must not import Qt: " + "; ".join(offenders)


def test_engine_core_is_plain_and_callback_driven():
    from hydra_suite.core.tracking.worker import TrackingEngineCore

    seen = []
    core = TrackingEngineCore(
        "dummy.mp4", on_progress=lambda pct, msg: seen.append((pct, msg))
    )
    # Not a QThread — instantiable with no Qt event loop.
    assert not hasattr(type(core), "start")
    # Guarded emit helper routes to the injected callback.
    core._emit_progress(42, "hi")
    assert seen == [(42, "hi")]

    # stop() sets the plain flag and stops an active prefetcher.
    class _FakePref:
        stopped = False

        def stop(self):
            self.stopped = True

    core.frame_prefetcher = _FakePref()
    core.stop()
    assert core._stop_requested is True
    assert core.frame_prefetcher.stopped is True


def test_engine_core_param_lock_roundtrip():
    from hydra_suite.core.tracking.worker import TrackingEngineCore

    core = TrackingEngineCore("dummy.mp4")
    core.set_parameters({"A": 1})
    core.update_parameters({"A": 2, "B": 3})
    assert core.get_current_params() == {"A": 2, "B": 3}


def test_wrapper_delegates_and_exposes_signals(qtbot=None):
    import os

    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from hydra_suite.trackerkit.gui.workers.tracking_worker import TrackingWorker

    w = TrackingWorker("dummy.mp4", preview_mode=True)
    # QThread + all 6 signals present.
    assert hasattr(w, "start") and hasattr(w, "wait") and hasattr(w, "isRunning")
    for sig in (
        "frame_signal",
        "finished_signal",
        "progress_signal",
        "stats_signal",
        "warning_signal",
        "pose_exported_model_resolved_signal",
    ):
        assert hasattr(w, sig)
    # Delegation reaches the core.
    w.set_parameters({"X": 1})
    assert w.get_current_params() == {"X": 1}
    w.stop()
    assert w._stop_requested is True
