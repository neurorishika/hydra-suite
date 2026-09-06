"""TrackerKit-facing contracts for the inference throughput autotuner."""

from __future__ import annotations

from hydra_suite.trackerkit.app import parse_arguments
from hydra_suite.trackerkit.cli_config import (
    TrackerCliVideoProbe,
    apply_inference_autotune_override,
    load_tracker_cli_session,
)
from hydra_suite.trackerkit.config.schemas import TrackerConfig


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
    assert session.params["INFERENCE_AUTOTUNE_BUDGET_SECONDS"] == 120.0
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
