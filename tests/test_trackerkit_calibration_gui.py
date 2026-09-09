"""Setup panel: an apply checkbox and a Calibrate button, nothing else."""

import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")

from PySide6.QtWidgets import QApplication  # noqa: E402


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def qtbot(qapp):
    # No pytest-qt plugin installed in this repo; the widgets under test
    # don't need event-loop pumping for boolean toggles.
    return None


@pytest.fixture
def setup_panel(qapp):
    from hydra_suite.trackerkit.gui.main_window import MainWindow

    window = MainWindow()
    try:
        yield window._setup_panel
    finally:
        window.close()


def test_old_calibration_widgets_are_gone(qtbot, setup_panel):
    assert not hasattr(setup_panel, "combo_inference_autotune")
    assert not hasattr(setup_panel, "spin_inference_autotune_budget")
    assert not hasattr(setup_panel, "btn_continue_inference_settings")


def test_apply_checkbox_and_calibrate_button_exist(qtbot, setup_panel):
    assert setup_panel.chk_apply_tuned_inference is not None
    assert setup_panel.btn_calibrate_inference is not None


def test_checkbox_writes_the_config_field(qtbot, setup_panel):
    setup_panel.chk_apply_tuned_inference.setChecked(True)
    assert setup_panel.config.apply_tuned_inference is True


def test_calibration_worker_subclasses_baseworker():
    from hydra_suite.trackerkit.gui.workers.calibration_worker import CalibrationWorker
    from hydra_suite.widgets.workers import BaseWorker

    assert issubclass(CalibrationWorker, BaseWorker)


def test_calibration_dialog_subclasses_basedialog():
    from hydra_suite.trackerkit.gui.dialogs.calibration import CalibrationDialog
    from hydra_suite.widgets.dialogs import BaseDialog

    assert issubclass(CalibrationDialog, BaseDialog)


def test_calibration_dialog_actually_constructs(qapp):
    """issubclass proves nothing: a bad BaseDialog call crashes on first click."""
    from hydra_suite.trackerkit.gui.dialogs.calibration import CalibrationDialog

    dialog = CalibrationDialog(
        params={},
        config=object(),
        video_path="/tmp/v.mp4",
        frame_width=64,
        frame_height=48,
        start_frame=0,
        end_frame=10,
        realtime=False,
        cache_dir=None,
        use_cached_detections=False,
        default_budget_seconds=60.0,
    )
    try:
        assert dialog.spin_budget.value() == 60.0
        assert dialog.lbl_status is not None
        assert dialog.btn_start.isEnabled()
        assert not dialog.btn_cancel.isEnabled()
        assert dialog.is_calibration_running() is False
        assert dialog.result_payload is None
    finally:
        dialog.deleteLater()


def _build_autotune_context_kwargs(module) -> set[str]:
    """Keyword names the module passes to ``build_autotune_context``."""
    import ast
    import inspect

    tree = ast.parse(inspect.getsource(module))
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", "")
        if name == "build_autotune_context":
            return {kw.arg for kw in node.keywords if kw.arg is not None}
    raise AssertionError(f"no build_autotune_context call found in {module!r}")


def test_gui_worker_and_cli_build_the_same_context_kwargs():
    """A third derivation path is exactly how this branch grew two bugs.

    The GUI worker and ``run_calibrate_cli`` must hand
    ``build_autotune_context`` the same set of keywords, or the profile the
    GUI persists lands under a key the production run never looks up.

    Scope, stated honestly: this compares keyword NAMES only. It catches an
    argument DROPPED on one path, and nothing else. It would NOT have caught
    the realtime bug (both paths passed a ``realtime=`` keyword; one passed a
    hardcoded ``False``), and it cannot catch two paths passing different
    VALUES under the same name. The value-level guards are the separate
    behavioural tests below.
    """
    from hydra_suite.trackerkit import calibrate_cli
    from hydra_suite.trackerkit.gui.workers import calibration_worker

    assert _build_autotune_context_kwargs(
        calibration_worker
    ) == _build_autotune_context_kwargs(calibrate_cli) | {"should_cancel"}


def test_worker_forwards_realtime_rather_than_hardcoding_false(monkeypatch):
    """``realtime`` is workload identity; hardcoding False mis-keys the profile."""
    from hydra_suite.core.inference.autotune import session
    from hydra_suite.trackerkit.gui.workers.calibration_worker import CalibrationWorker

    captured = {}

    class _Ctx:
        # The worker reports whether the run it calibrated would replay
        # every stage from cache, so the stub context needs that field.
        class run_context:  # noqa: N801 - stand-in namespace
            execution_mode = "batch"

    def fake_build(config, params, **kwargs):
        captured.update(kwargs)
        return _Ctx()

    class _Overlay:
        status = "calibrated"
        reason = "kept_current_settings"
        profile_id = "abc"
        effective = type("E", (), {"to_dict": lambda self: {}})()

    monkeypatch.setattr(session, "build_autotune_context", fake_build)
    monkeypatch.setattr(
        session, "calibrate", lambda ctx, budget_seconds: (None, _Overlay(), None)
    )

    worker = CalibrationWorker(
        {},
        object(),
        video_path="/tmp/v.mp4",
        budget_seconds=10.0,
        frame_width=64,
        frame_height=48,
        start_frame=0,
        end_frame=10,
        realtime=True,
        cache_dir=None,
        use_cached_detections=False,
    )
    worker.execute()

    assert captured["realtime"] is True
    assert captured["cache_read_only_replay"] is False


def test_cancel_reaches_the_searchs_should_cancel(monkeypatch):
    """Cancel must be observable through the should_cancel callable."""
    from hydra_suite.core.inference.autotune import session
    from hydra_suite.trackerkit.gui.workers.calibration_worker import CalibrationWorker

    captured = {}

    monkeypatch.setattr(
        session,
        "build_autotune_context",
        lambda config, params, **kwargs: captured.update(kwargs) or object(),
    )
    monkeypatch.setattr(
        session, "calibrate", lambda ctx, budget_seconds: (None, None, None)
    )

    worker = CalibrationWorker(
        {},
        object(),
        video_path="/tmp/v.mp4",
        budget_seconds=10.0,
        frame_width=64,
        frame_height=48,
        start_frame=0,
        end_frame=10,
        realtime=False,
    )
    # overlay is None -> the worker must raise rather than AttributeError.
    with pytest.raises(RuntimeError):
        worker.execute()

    should_cancel = captured["should_cancel"]
    assert should_cancel() is False
    worker.cancel()
    assert should_cancel() is True


def test_no_improvement_is_reported_as_success_not_failure():
    from hydra_suite.trackerkit.gui.dialogs.calibration import (
        describe_calibration_outcome,
    )

    text = describe_calibration_outcome(
        {"status": "calibrated", "reason": "kept_current_settings", "profile_id": "a"}
    )
    assert "no configuration beat your current settings" in text
    assert "fail" not in text.lower()

    assert "No validated profile matches" in describe_calibration_outcome(
        {"status": "unavailable", "reason": "no profile"}
    )
    assert "unchanged" in describe_calibration_outcome({"status": "cancelled"})


def test_effective_vector_is_shown_for_unavailable_and_deferred_statuses():
    """InferenceRuntimeOverlay.baseline() always populates ``effective`` with
    the real baseline vector, and both of these statuses are reachable during
    a real tracking run (coordinator.resolve resolves runs with intent
    "lookup"). The user must still see what settings are in force even when
    no validated profile applies.
    """
    from hydra_suite.trackerkit.gui.dialogs.calibration import (
        describe_calibration_outcome,
    )

    effective = {"detection_batch_size": 4, "pipeline_depth": 2}

    unavailable_text = describe_calibration_outcome(
        {"status": "unavailable", "reason": "no profile", "effective": effective}
    )
    assert "No validated profile matches" in unavailable_text
    assert "detection_batch_size=4" in unavailable_text
    assert "pipeline_depth=2" in unavailable_text

    deferred_text = describe_calibration_outcome(
        {
            "status": "deferred_due_to_prior_failure",
            "reason": "prior attempt failed",
            "effective": effective,
        }
    )
    assert "detection_batch_size=4" in deferred_text
    assert "pipeline_depth=2" in deferred_text


def test_status_label_explains_a_replaying_run_and_invents_nothing_otherwise():
    """A run that replays every stage performs no inference, so it neither
    needs nor can have a profile (``execution_mode`` is in the key and
    ``cache_replay`` freezes every field). The label must say that and give
    the one action that changes it.

    The note it replaces was keyed on ``cached_fields``, an unconditional
    empty set since ``01948492``: it therefore ALWAYS claimed "covers runs
    with NO detection cache" and always advised calibrating again after the
    first tracking run -- the exact action that produces the refusal.
    """
    from hydra_suite.trackerkit.gui.dialogs.calibration import (
        describe_calibration_outcome,
    )

    replaying = describe_calibration_outcome(
        {
            "status": "not_tunable",
            "reason": "all inference stages are satisfied by reusable caches",
            "cache_replay": True,
        }
    )
    assert "nothing to measure" in replaying
    assert "Use cached detections" in replaying
    assert "contention" not in replaying

    measured = describe_calibration_outcome(
        {
            "status": "calibrated",
            "reason": "measured",
            "profile_id": "abc",
            "effective": {"detection_batch_size": 4},
            "cache_replay": False,
        }
    )
    assert "detection_batch_size=4" in measured
    assert "detection cache" not in measured

    # A payload with no cache-mode information must not invent one.
    silent = describe_calibration_outcome(
        {"status": "calibrated", "reason": "measured", "profile_id": "abc"}
    )
    assert "detection cache" not in silent


def test_gui_worker_reports_the_cache_mode_it_calibrated_in(monkeypatch, tmp_path):
    """The label can only be honest if the worker actually reports the mode
    from the context it calibrated with -- not a hardcoded guess."""
    from hydra_suite.core.inference.autotune import session
    from hydra_suite.trackerkit.gui.workers.calibration_worker import CalibrationWorker

    class _Ctx:
        class run_context:  # noqa: N801 - stand-in namespace
            execution_mode = "cache_replay"

    class _Overlay:
        status = "calibrated"
        reason = "measured"
        profile_id = "abc"

        class effective:
            @staticmethod
            def to_dict():
                return {"detection_batch_size": 4}

    monkeypatch.setattr(session, "build_autotune_context", lambda *a, **k: _Ctx())
    monkeypatch.setattr(session, "calibrate", lambda *a, **k: (None, _Overlay(), None))

    worker = CalibrationWorker(
        {},
        object(),
        video_path=str(tmp_path / "v.mp4"),
        budget_seconds=60.0,
        frame_width=16,
        frame_height=12,
        start_frame=0,
        end_frame=1,
        realtime=False,
        use_cached_detections=True,
    )
    payloads = []
    worker.completed.connect(payloads.append)
    worker.execute()

    assert payloads and payloads[0]["cache_replay"] is True
