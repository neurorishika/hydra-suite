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
        inference_autotune_mode="automatic",
        inference_autotune_manual_fields=["pose_batch_size", "pipeline_depth"],
    )

    restored = TrackerConfig.from_dict(config.to_dict())

    assert restored.inference_autotune_mode == "automatic"
    assert restored.inference_autotune_manual_fields == [
        "pose_batch_size",
        "pipeline_depth",
    ]


def test_legacy_tracker_config_keeps_throughput_tuner_disabled() -> None:
    assert TrackerConfig.from_dict({}).inference_autotune_mode == "off"
    assert TrackerConfig.from_dict({}).inference_autotune_manual_fields == []


def test_engine_params_carry_inference_autotune_policy_without_mutating_batches(
    tmp_path,
) -> None:
    session = load_tracker_cli_session(
        str(tmp_path / "subject.mp4"),
        config_data={
            "detection_batch_size": 8,
            "pipeline_depth": 3,
            "inference_autotune_mode": "automatic",
            "inference_autotune_manual_fields": ["pose_batch_size"],
        },
        video_probe=TrackerCliVideoProbe(fps=20.0, total_frames=50, width=5, height=5),
    )

    assert session.params["YOLO_BATCH_SIZE"] == 8
    assert session.params["PIPELINE_DEPTH"] == 3
    assert session.params["INFERENCE_AUTOTUNE_MODE"] == "automatic"
    assert session.params["INFERENCE_AUTOTUNE_MANUAL_FIELDS"] == ["pose_batch_size"]
    assert session.params["INFERENCE_AUTOTUNE_BUDGET_SECONDS"] == 600.0
    assert session.params["INFERENCE_AUTOTUNE_PROJECT_CONFIG"] == {
        "detection_batch_size": 8,
        "pipeline_depth": 3,
        "inference_autotune_mode": "automatic",
        "inference_autotune_manual_fields": ["pose_batch_size"],
    }
    assert session.params["INFERENCE_AUTOTUNE_PROJECT_CONFIG"] is not session.config


def test_cli_autotune_override_has_explicit_precedence_and_preserves_manuals() -> None:
    original = {
        "inference_autotune_mode": "record",
        "inference_autotune_manual_fields": ["pose_batch_size"],
    }

    overridden = apply_inference_autotune_override(
        original,
        mode="automatic",
        manual_fields=["pipeline_depth", "pose_batch_size"],
    )

    assert original["inference_autotune_mode"] == "record"
    assert overridden["inference_autotune_mode"] == "automatic"
    assert overridden["inference_autotune_manual_fields"] == [
        "pipeline_depth",
        "pose_batch_size",
    ]


def test_cli_autotune_flags_support_record_and_per_run_bypass() -> None:
    record = parse_arguments(["track", "video.mp4", "--inference-autotune", "record"])
    bypass = parse_arguments(["track", "video.mp4", "--no-inference-autotune"])

    assert record.inference_autotune == "record"
    assert record.no_inference_autotune is False
    assert bypass.inference_autotune is None
    assert bypass.no_inference_autotune is True


def test_cli_autotune_manual_field_can_be_repeated() -> None:
    args = parse_arguments(
        [
            "track",
            "video.mp4",
            "--inference-autotune",
            "automatic",
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
        def set_inference_autotune_calibration_active(self, _active):
            pass

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


def test_gui_continue_action_cancels_only_calibration() -> None:
    calls = []

    class Worker:
        def cancel_inference_autotune(self):
            calls.append("cancel")

    class Setup:
        def set_inference_autotune_calibration_active(self, active):
            calls.append(("active", active))

        def set_inference_autotune_status(self, text):
            calls.append(("status", text))

    class MainWindow:
        tracking_worker = Worker()

    orchestrator = object.__new__(TrackingOrchestrator)
    orchestrator._mw = MainWindow()
    orchestrator._panels = type("Panels", (), {"setup": Setup()})()

    orchestrator.continue_with_current_inference_settings()

    assert calls == [
        "cancel",
        ("active", False),
        ("status", "Continuing with configured inference settings…"),
    ]


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
