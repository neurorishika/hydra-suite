from __future__ import annotations

import json
from pathlib import Path

from hydra_suite.core.inference.autotune.integration import (
    TrackingRunContext,
    build_tracking_autotune_request,
    memory_profile_identity,
    record_profile_memory_evidence,
    sample_detection_workload,
)
from hydra_suite.core.inference.autotune.models import (
    CandidateEvidence,
    EquivalenceVerdict,
    InferenceTuningProfile,
    InferenceTuningSettings,
    ProfileState,
)
from hydra_suite.core.inference.config import (
    CNNConfig,
    HeadTailConfig,
    InferenceAutotunePolicy,
    InferenceConfig,
    OBBConfig,
    OBBDirectConfig,
    PoseConfig,
    PoseYOLOConfig,
    SliceConfig,
)
from hydra_suite.core.inference.runner import _load_obb_for_config
from hydra_suite.runtime.memory_profiles import MemoryProfileStore
from hydra_suite.runtime.resource_budget import AcceleratorKind, ResourceObservation


def _model(path: Path, payload: bytes) -> str:
    path.write_bytes(payload)
    return str(path)


def _config(tmp_path: Path) -> InferenceConfig:
    detector = _model(tmp_path / "detector.pt", b"detector")
    headtail = _model(tmp_path / "headtail.pt", b"headtail")
    pose = _model(tmp_path / "pose.pt", b"pose")
    identity = _model(tmp_path / "identity.pt", b"identity")
    return InferenceConfig(
        obb=OBBConfig(
            mode="direct",
            direct=OBBDirectConfig(
                detector,
                slice=SliceConfig(
                    enabled=True,
                    geometry_mode="custom",
                    slice_width=512,
                    slice_height=384,
                    tile_batch_size=2,
                ),
            ),
            target_classes=[0],
            max_detections=25,
        ),
        headtail=HeadTailConfig(headtail, batch_size=8),
        cnn_phases=[CNNConfig("color", identity, batch_size=8)],
        pose=PoseConfig(backend="yolo", yolo=PoseYOLOConfig(pose, batch_size=8)),
        detection_batch_size=2,
        pipeline_depth=2,
        runtime_tier="cpu",
    )


def _observation() -> ResourceObservation:
    return ResourceObservation(
        total_host_bytes=64 * 1024**3,
        available_host_bytes=48 * 1024**3,
        accelerator_kind=AcceleratorKind.CPU,
    )


def test_tracking_request_fingerprints_every_model_and_path_free_geometry(tmp_path):
    config = _config(tmp_path)
    context = TrackingRunContext(
        video_path=tmp_path / "private-name.mp4",
        params={
            "INFERENCE_AUTOTUNE_MODE": "record",
            "MAX_TARGETS": 25,
            "RESIZE_FACTOR": 0.5,
            "INFERENCE_AUTOTUNE_DETECTION_COUNTS": [12, 25, 17],
            "INFERENCE_AUTOTUNE_CROP_COUNTS": [12, 25, 17],
        },
        frame_width=1200,
        frame_height=900,
        channels=3,
        execution_mode="batch",
    )

    request = build_tracking_autotune_request(
        config,
        context,
        observation=_observation(),
        backend="torch",
        device_identity=("cpu", "CPU", "none", 0),
    )

    assert [item.role for item in request.key.models] == [
        "detector.direct",
        "headtail",
        "identity.color",
        "pose.yolo",
    ]
    assert request.key.frame.width == 1200
    assert request.key.frame.resize_factor == 0.5
    assert request.key.slice.tile_width == 512
    assert request.key.slice.tile_height == 384
    assert request.key.workload.detections_p95_bucket == 32
    assert request.key.workload.canonical_crop_geometries
    assert "private-name.mp4" not in str(request.key.to_dict())


def test_tensorrt_profile_identity_is_part_of_every_compiled_detector_key(tmp_path):
    config = _config(tmp_path)
    context = TrackingRunContext(
        video_path=tmp_path / "video.mp4",
        params={
            "MAX_TARGETS": 25,
            "INFERENCE_AUTOTUNE_TENSORRT_PROFILE_BATCH_SIZE": 16,
        },
        frame_width=1200,
        frame_height=900,
    )

    request = build_tracking_autotune_request(
        config,
        context,
        observation=_observation(),
        backend="tensorrt",
        device_identity=("gpu", "GPU", "8.9", 48 * 1024**3),
    )

    by_role = {item.role: item for item in request.key.models}
    assert by_role["detector.direct"].tensorrt_profile_id != "none"
    assert by_role["headtail"].tensorrt_profile_id == "none"


def test_streaming_drops_inert_detector_batch_and_depth_coordinates(tmp_path):
    config = _config(tmp_path)
    context = TrackingRunContext(
        video_path=tmp_path / "video.mp4",
        params={"INFERENCE_AUTOTUNE_MODE": "record", "MAX_TARGETS": 25},
        frame_width=1200,
        frame_height=900,
        execution_mode="streaming",
    )
    request = build_tracking_autotune_request(
        config,
        context,
        observation=_observation(),
        backend="torch",
        device_identity=("cpu", "CPU", "none", 0),
    )

    assert "detection_batch_size" in request.planner.context.cached_fields
    assert "pipeline_depth" in request.planner.context.cached_fields
    assert request.planner.values_for("detection_batch_size", request.baseline) == ()
    assert request.planner.values_for("pipeline_depth", request.baseline) == ()


def test_realtime_is_ineligible_and_never_searches(tmp_path):
    config = _config(tmp_path)
    context = TrackingRunContext(
        video_path=tmp_path / "video.mp4",
        params={"INFERENCE_AUTOTUNE_MODE": "automatic", "MAX_TARGETS": 25},
        frame_width=1200,
        frame_height=900,
        execution_mode="realtime",
    )
    request = build_tracking_autotune_request(
        config,
        context,
        observation=_observation(),
        backend="torch",
        device_identity=("cpu", "CPU", "none", 0),
    )

    assert not request.eligible
    assert request.eligibility_reason == "realtime inference is not tunable"
    assert request.planner.admit(request.baseline).settings.detection_batch_size == 1


def test_automatic_backend_without_evidence_falls_back_but_record_mode_can_measure(
    tmp_path,
):
    config = _config(tmp_path)
    automatic = TrackingRunContext(
        video_path=tmp_path / "video.mp4",
        params={"INFERENCE_AUTOTUNE_MODE": "automatic", "MAX_TARGETS": 25},
        frame_width=100,
        frame_height=100,
        execution_mode="batch",
    )
    record = TrackingRunContext(
        video_path=automatic.video_path,
        params={**automatic.params, "INFERENCE_AUTOTUNE_MODE": "record"},
        frame_width=100,
        frame_height=100,
        execution_mode="batch",
    )

    config.inference_autotune = InferenceAutotunePolicy(mode="automatic")

    auto_request = build_tracking_autotune_request(
        config,
        automatic,
        observation=_observation(),
        backend="torch",
        device_identity=("cpu", "CPU", "none", 0),
    )
    config.inference_autotune = InferenceAutotunePolicy(mode="record")
    record_request = build_tracking_autotune_request(
        config,
        record,
        observation=_observation(),
        backend="torch",
        device_identity=("cpu", "CPU", "none", 0),
    )

    assert not auto_request.eligible
    assert "validated only for CUDA" in (auto_request.eligibility_reason or "")
    assert record_request.eligible


def test_core_inference_policy_roundtrip_and_legacy_default(tmp_path):
    config = _config(tmp_path)
    config.inference_autotune = InferenceAutotunePolicy(
        mode="automatic",
        manual_fields=("pipeline_depth", "pose_batch_size"),
        budget_seconds=90,
    )
    path = tmp_path / "inference.json"
    config.to_json(str(path))

    restored = InferenceConfig.from_json(str(path))
    assert restored.inference_autotune == config.inference_autotune

    raw = json.loads(path.read_text(encoding="utf-8"))
    raw.pop("inference_autotune")
    path.write_text(json.dumps(raw), encoding="utf-8")
    legacy = InferenceConfig.from_json(str(path))
    assert legacy.inference_autotune.mode == "off"


def test_existing_detection_cache_supplies_zero_inclusive_density(
    monkeypatch, tmp_path
):
    class Reader:
        def is_valid(self):
            return True

        def iter_arrays(self):
            yield {
                "written_frames": [0, 1, 2, 3],
                "frame_indices": [0, 0, 2, 3, 3, 3],
            }

    monkeypatch.setattr(
        "hydra_suite.core.inference.cache.open_detection_cache_reader",
        lambda _path: Reader(),
    )

    assert sample_detection_workload(tmp_path, start_frame=1, end_frame=3) == (0, 1, 3)


def test_calibration_uses_one_wide_runtime_profile_without_changing_candidate_batch(
    monkeypatch, tmp_path
):
    config = _config(tmp_path)
    config.detection_batch_size = 2
    config.runtime_artifact_batch_size = 16
    captured = {}

    def load(_obb, _runtime, *, batch_size, stage1_batch_size=None):
        captured["batch_size"] = batch_size
        captured["stage1_batch_size"] = stage1_batch_size
        return object()

    monkeypatch.setattr("hydra_suite.core.inference.stages.obb.load_obb_models", load)

    _load_obb_for_config(config, object())

    assert config.detection_batch_size == 2
    assert captured == {"batch_size": 16, "stage1_batch_size": None}


def test_successful_memory_evidence_is_reused_by_exact_key_admission(tmp_path):
    config = _config(tmp_path)
    context = TrackingRunContext(
        video_path=tmp_path / "video.mp4",
        params={"MAX_TARGETS": 25},
        frame_width=1200,
        frame_height=900,
    )
    initial = build_tracking_autotune_request(
        config,
        context,
        observation=_observation(),
        backend="torch",
        device_identity=("cpu", "CPU", "none", 0),
        memory_records=(),
    )
    selected = InferenceTuningSettings.from_config(config)
    evidence = CandidateEvidence(
        selected,
        (100.0,) * 5,
        stage_seconds_samples=(0.5,) * 5,
        measured_frames=128,
        host_peak_bytes=1024,
        accelerator_peak_bytes=512,
        warmup_calls=3,
        warmup_frames=8,
        equivalence=EquivalenceVerdict(True),
        phase="final_validation",
    )
    profile = InferenceTuningProfile(
        initial.key.digest[:24],
        initial.key,
        selected,
        selected,
        selected,
        selected,
        (evidence,),
        ProfileState.VALIDATED,
        "kept_current_settings",
    )
    memory_store = MemoryProfileStore(tmp_path / "memory.json")
    record_profile_memory_evidence(profile, store=memory_store)

    records = memory_store.load()
    rebuilt = build_tracking_autotune_request(
        config,
        context,
        observation=_observation(),
        backend="torch",
        device_identity=("cpu", "CPU", "none", 0),
        memory_records=records,
    )

    detector_records = dict(rebuilt.planner.context.measured_records)[
        "detection_batch_size"
    ]
    assert detector_records
    assert detector_records[0].identity == memory_profile_identity(
        rebuilt.key, "detection_batch_size"
    )
