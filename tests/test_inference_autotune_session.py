"""Session-level entry points for autotune: shared context, lookup, calibrate."""

from hydra_suite.core.inference.autotune import session


def test_module_exposes_the_three_entry_points():
    assert callable(session.build_autotune_context)
    assert callable(session.lookup)
    assert callable(session.calibrate)


def test_session_module_is_qt_free():
    """core/ must never import Qt. Guard it at the module source level."""
    from pathlib import Path

    src = Path(session.__file__).read_text(encoding="utf-8")
    assert "PySide6" not in src
    assert "QtCore" not in src


def test_build_context_is_pure_and_does_not_mutate_caller_params():
    """The ephemeral params dict must be a copy: worker.py:147 does dict(params)
    and then mutates. A caller's dict must never gain autotune-only keys."""
    from pathlib import Path

    src = Path(session.__file__).read_text(encoding="utf-8")
    assert "ephemeral_params = dict(params)" in src


def _fake_ctx():
    """A minimal context standing in for build_autotune_context's output.

    Only the attributes ``calibrate`` reads are populated; the resolve call
    it makes is stubbed, so nothing here needs a real video or device.
    """
    from types import SimpleNamespace

    from hydra_suite.core.inference.config import build_inference_config_from_params

    config = build_inference_config_from_params({})
    run_context = SimpleNamespace(
        video_path="video.mp4",
        start_frame=0,
        end_frame=10,
        cached_fields=frozenset(),
    )
    probe = SimpleNamespace(observation=object())
    return SimpleNamespace(
        config=config,
        run_context=run_context,
        params={},
        backend="torch",
        probe=probe,
        device_identity=("uuid", "model", "cc", 0),
        artifact_batch_size=1,
    )


def test_calibrate_asks_the_coordinator_to_measure(monkeypatch):
    """``calibrate`` must flip the policy to mode=calibrate and carry the budget.

    ``build_inference_config_from_params`` always builds ``mode="lookup"``,
    and ``coordinator.resolve`` returns "unavailable / no validated profile"
    for a lookup request WITHOUT measuring. When nothing set the mode, both
    the GUI Calibrate button and ``trackerkit calibrate`` were silently
    inert. The coordinator's search deadline reads ``policy.budget_seconds``,
    so the caller's budget must land there too.
    """
    from hydra_suite.core.inference.autotune import integration

    seen = {}

    def _fake_resolve(config, context, **kwargs):
        seen["config"] = config
        overlay = SimpleNamespaceOverlay()
        return config, overlay, SimpleNamespaceResult()

    class SimpleNamespaceOverlay:
        status = "calibrated"
        reason = "measured"
        profile_id = None

    class SimpleNamespaceResult:
        profile = None

    monkeypatch.setattr(integration, "resolve_tracking_inference_config", _fake_resolve)

    session.calibrate(_fake_ctx(), budget_seconds=123.0)

    policy = seen["config"].inference_autotune
    assert policy.mode == "calibrate"
    assert policy.budget_seconds == 123.0
