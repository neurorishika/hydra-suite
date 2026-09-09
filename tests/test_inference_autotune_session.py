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


def test_a_detection_cache_alone_does_not_declare_the_run_untunable(
    monkeypatch, tmp_path
):
    """Density evidence is not proof that a run will replay every stage.

    ``sample_detection_workload`` reads detection.npz with no config, video
    signature or ROI mask, so it cannot check the cache KEY and never sees
    the headtail/cnn/pose siblings. Treating "reuse ticked and a detection
    cache exists" as full replay froze every tuning field AND changed the
    profile key, so such a video became permanently untunable -- while a run
    whose caches are not actually reusable still does full fresh inference at
    the baseline. Only the worker's own predicate
    (``cache_set_is_fully_reusable``) may make that call.
    """
    from hydra_suite.core.inference.autotune.integration import (
        build_tracking_autotune_request,
    )
    from tests.autotune_helpers import make_calibration_context

    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    ctx = make_calibration_context(
        monkeypatch,
        tmp_path,
        cache_dir=cache_dir,
        measured_counts=(2, 3, 2),
        use_cached_detections=True,
    )

    assert ctx.run_context.execution_mode == "batch"

    request = build_tracking_autotune_request(
        ctx.config,
        ctx.run_context,
        observation=ctx.probe.observation,
        backend=ctx.backend,
        device_identity=ctx.device_identity,
    )
    assert request.eligible
    # The measured density must still reach the key -- only the replay
    # classification changed.
    assert not request.key.workload.density_is_estimated


def test_a_replaying_run_reports_not_tunable_rather_than_contention(
    monkeypatch, tmp_path
):
    """ "Deferred due to contention" promises a retry will work. A cache
    replay is not a temporary condition, and pairing that headline with
    "all inference stages are satisfied by reusable caches" contradicted
    itself in one sentence."""
    from hydra_suite.core.inference.autotune.integration import (
        build_tracking_autotune_request,
    )
    from tests.autotune_helpers import make_calibration_context

    ctx = make_calibration_context(monkeypatch, tmp_path, cache_dir=None)
    replay_context = _replace_execution_mode(ctx.run_context, "cache_replay")

    request = build_tracking_autotune_request(
        ctx.config,
        replay_context,
        observation=ctx.probe.observation,
        backend=ctx.backend,
        device_identity=ctx.device_identity,
    )
    assert not request.eligible
    assert request.ineligible_status == "not_tunable"


def _replace_execution_mode(run_context, mode: str):
    from dataclasses import replace

    return replace(run_context, execution_mode=mode)
