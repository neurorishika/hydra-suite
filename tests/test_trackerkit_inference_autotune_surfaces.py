"""TrackerKit-facing contracts for the inference throughput autotuner."""

from __future__ import annotations

from hydra_suite.core.inference.autotune.coordinator import ResolveResult
from hydra_suite.core.inference.autotune.models import (
    InferenceRuntimeOverlay,
    InferenceTuningSettings,
)
from hydra_suite.core.tracking.worker import _inference_autotune_stats
from hydra_suite.trackerkit.app import parse_arguments
from hydra_suite.trackerkit.cli_config import (
    TrackerCliVideoProbe,
    apply_inference_autotune_override,
    load_tracker_cli_session,
)
from hydra_suite.trackerkit.config.schemas import TrackerConfig
from hydra_suite.trackerkit.gui.orchestrators.tracking import TrackingOrchestrator


def test_tracker_config_round_trips_inference_autotune_policy() -> None:
    config = TrackerConfig(
        apply_tuned_inference=True,
        inference_autotune_manual_fields=["pose_batch_size", "pipeline_depth"],
    )

    restored = TrackerConfig.from_dict(config.to_dict())

    assert restored.apply_tuned_inference is True
    assert restored.inference_autotune_manual_fields == [
        "pose_batch_size",
        "pipeline_depth",
    ]


def test_legacy_tracker_config_keeps_throughput_tuner_disabled() -> None:
    assert TrackerConfig.from_dict({}).apply_tuned_inference is False
    assert TrackerConfig.from_dict({}).inference_autotune_manual_fields == []


def test_engine_params_carry_inference_autotune_policy_without_mutating_batches(
    tmp_path,
) -> None:
    session = load_tracker_cli_session(
        str(tmp_path / "subject.mp4"),
        config_data={
            "detection_batch_size": 8,
            "pipeline_depth": 3,
            "apply_tuned_inference": True,
            "inference_autotune_manual_fields": ["pose_batch_size"],
        },
        video_probe=TrackerCliVideoProbe(fps=20.0, total_frames=50, width=5, height=5),
    )

    assert session.params["YOLO_BATCH_SIZE"] == 8
    assert session.params["PIPELINE_DEPTH"] == 3
    assert session.params["APPLY_TUNED_INFERENCE"] is True
    assert session.params["INFERENCE_AUTOTUNE_MANUAL_FIELDS"] == ["pose_batch_size"]
    assert session.params["INFERENCE_AUTOTUNE_BUDGET_SECONDS"] == 4500.0
    assert session.params["INFERENCE_AUTOTUNE_PROJECT_CONFIG"] == {
        "detection_batch_size": 8,
        "pipeline_depth": 3,
        "apply_tuned_inference": True,
        "inference_autotune_manual_fields": ["pose_batch_size"],
    }
    assert session.params["INFERENCE_AUTOTUNE_PROJECT_CONFIG"] is not session.config


def test_cli_autotune_override_has_explicit_precedence_and_preserves_manuals() -> None:
    """``apply`` speaks the same boolean vocabulary as the config field it
    sets (``TrackerConfig.apply_tuned_inference`` / ``--apply-tuned-inference``
    -- the old off/record/automatic mode vocabulary is retired)."""
    original = {
        "apply_tuned_inference": False,
        "inference_autotune_manual_fields": ["pose_batch_size"],
    }

    overridden = apply_inference_autotune_override(
        original,
        apply=True,
        manual_fields=["pipeline_depth", "pose_batch_size"],
    )

    assert original["apply_tuned_inference"] is False
    assert overridden["apply_tuned_inference"] is True
    assert overridden["inference_autotune_manual_fields"] == [
        "pipeline_depth",
        "pose_batch_size",
    ]


def test_cli_autotune_flags_support_apply_and_per_run_bypass() -> None:
    applied = parse_arguments(["track", "video.mp4", "--apply-tuned-inference"])
    bypass = parse_arguments(["track", "video.mp4", "--no-apply-tuned-inference"])
    unset = parse_arguments(["track", "video.mp4"])

    assert applied.apply_tuned_inference is True
    assert bypass.apply_tuned_inference is False
    assert unset.apply_tuned_inference is None


def test_cli_autotune_manual_field_can_be_repeated() -> None:
    args = parse_arguments(
        [
            "track",
            "video.mp4",
            "--apply-tuned-inference",
            "--inference-autotune-manual",
            "pose_batch_size",
            "--inference-autotune-manual",
            "pipeline_depth",
        ]
    )

    assert args.inference_autotune_manual == ["pose_batch_size", "pipeline_depth"]


def test_gui_status_displays_effective_runtime_overlay() -> None:
    captured = []

    class Setup:
        def set_inference_autotune_status(self, text):
            captured.append(text)

    class MainWindow:
        _stop_all_requested = False

    orchestrator = object.__new__(TrackingOrchestrator)
    orchestrator._mw = MainWindow()
    orchestrator._panels = type("Panels", (), {"setup": Setup()})()

    orchestrator.on_stats_update(
        {
            "inference_autotune": {
                "status": "cache_hit",
                "profile_id": "abc123",
                "reason": "validated profile reused",
                "effective": {"detection_batch_size": 4, "pipeline_depth": 2},
            }
        }
    )

    assert "Cache hit" in captured[0]
    assert "abc123" in captured[0]
    assert "detection_batch_size=4" in captured[0]


def test_gui_no_continue_escape_hatch_remains() -> None:
    """The interim "Continue with current settings" escape hatch existed
    only to interrupt calibration a tracking run should never have been
    doing. A run never calibrates now (only the explicit Calibrate…
    dialog does), so the method is gone."""
    assert not hasattr(TrackingOrchestrator, "continue_with_current_inference_settings")


def test_run_summary_includes_fingerprint_even_without_promoted_profile() -> None:
    runtime = InferenceTuningSettings(detection_batch_size=2, pipeline_depth=1)
    overlay = InferenceRuntimeOverlay.baseline(
        runtime,
        status="fallback",
        reason="budget expired",
    )

    summary = _inference_autotune_stats(
        overlay,
        ResolveResult(overlay, key_digest="a" * 64),
    )

    assert summary["fingerprint_digest"] == "a" * 64
    assert summary["requested"] == runtime.to_dict()
    assert summary["status"] == "fallback"


def test_calibration_budget_above_600s_survives_the_live_params_path(tmp_path) -> None:
    """A budget the project asked for must reach the search, not be clamped.

    A live calibration configured for 3000 s expired at 600.6 s because
    ``InferenceAutotunePolicy`` -- not the sidecar validator -- was the clamp
    that actually binds on this path, and it clamped SILENTLY: the run was
    indistinguishable from one that had only asked for 600 s.
    """
    from hydra_suite.core.inference.config import build_inference_config_from_params

    session = load_tracker_cli_session(
        str(tmp_path / "subject.mp4"),
        config_data={
            "apply_tuned_inference": True,
            "inference_autotune_budget_seconds": 3000.0,
        },
        video_probe=TrackerCliVideoProbe(fps=20.0, total_frames=50, width=5, height=5),
    )

    assert session.params["INFERENCE_AUTOTUNE_BUDGET_SECONDS"] == 3000.0
    config = build_inference_config_from_params(session.params)
    assert config.inference_autotune.budget_seconds == 3000.0
