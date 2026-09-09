from __future__ import annotations

import json
import os
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from hydra_suite.core.inference.autotune.candidates import (
    AdmissionContext,
    CandidatePlanner,
    MemoryCostModel,
)
from hydra_suite.core.inference.autotune.coordinator import (
    AutotuneCoordinator,
    AutotuneRequest,
)
from hydra_suite.core.inference.autotune.equivalence import (
    CalibrationOutputs,
    compare_outputs,
)
from hydra_suite.core.inference.autotune.fingerprint import (
    AcceleratorFingerprint,
    DetectorFingerprint,
    FrameFingerprint,
    ModelArtifactFingerprint,
    PipelineFingerprint,
    SchemaFingerprint,
    SliceFingerprint,
    SoftwareFingerprint,
    SystemFingerprint,
    TuningProfileKey,
    WorkloadFingerprint,
    count_bucket,
    default_software_fingerprint,
    model_content_digest,
)
from hydra_suite.core.inference.autotune.measure import (
    MeasurementProtocol,
    deterministic_block_order,
    paired_gain_interval,
)
from hydra_suite.core.inference.autotune.models import (
    CandidateEvidence,
    EquivalenceVerdict,
    InferenceRuntimeOverlay,
    InferenceTuningProfile,
    InferenceTuningSettings,
    ProfileState,
)
from hydra_suite.core.inference.autotune.store import InferenceTuningProfileStore
from hydra_suite.core.inference.config import (
    BgSubConfig,
    CNNConfig,
    HeadTailConfig,
    InferenceConfig,
    OBBConfig,
    OBBDirectConfig,
    PoseConfig,
    PoseYOLOConfig,
    SliceConfig,
)
from hydra_suite.runtime.memory_profiles import (
    MemoryMeasurement,
    PressureSettings,
    ProfileIdentity,
)
from hydra_suite.runtime.resource_budget import (
    AcceleratorKind,
    ResourceObservation,
    ResourcePolicy,
)


def _key() -> TuningProfileKey:
    return TuningProfileKey(
        schema=SchemaFingerprint(),
        system=SystemFingerprint("host", "linux-x86", "cpu", 16, 64 * 1024**3),
        accelerator=AcceleratorFingerprint("GPU-1", "RTX", "8.9", 48 * 1024**3),
        software=SoftwareFingerprint(
            "1",
            "abc",
            "cpython-3.13",
            "torch",
            "fp16",
            "1",
            "13",
            "9",
            "10",
            "2",
            "8",
            "absent",
        ),
        models=(ModelArtifactFingerprint("detector", "a" * 64, 640, 640),),
        frame=FrameFingerprint(1200, 1200, 3, "bgr8", 1.0, "ffmpeg"),
        detector=DetectorFingerprint("yolo_obb", "obb", (0,), 0.25, 0.7, 25, "direct"),
        slice=SliceFingerprint(False, 0, 0, 0.2, 0.2, False, "none", "20x2"),
        pipeline=PipelineFingerprint(("detector",), "forward", "batch", ()),
        workload=WorkloadFingerprint(25, 32, 32, 32, 32, ("256x128",)),
    )


def _settings(**changes) -> InferenceTuningSettings:
    values = {
        "detection_batch_size": 4,
        "slice_tile_batch_size": 2,
        "pose_batch_size": 4,
        "headtail_batch_size": 4,
        "identity_batch_sizes": (("color", 4),),
        "pipeline_depth": 2,
    }
    values.update(changes)
    return InferenceTuningSettings(**values)


def _observation(free=32 * 1024**3) -> ResourceObservation:
    return ResourceObservation(
        total_host_bytes=64 * 1024**3,
        available_host_bytes=48 * 1024**3,
        accelerator_kind=AcceleratorKind.CUDA,
        accelerator_name="GPU",
        total_accelerator_bytes=48 * 1024**3,
        available_accelerator_bytes=free,
    )


def _profile(key=None, selected=None) -> InferenceTuningProfile:
    key = key or _key()
    baseline = _settings(detection_batch_size=1)
    selected = selected or _settings(detection_batch_size=4)
    evidence = CandidateEvidence(
        selected,
        (100.0,) * 5,
        stage_seconds_samples=(0.5,) * 5,
        warmup_calls=3,
        warmup_frames=8,
        accelerator_peak_bytes=2 * 1024**3,
        equivalence=EquivalenceVerdict(True),
    )
    return InferenceTuningProfile(
        profile_id=key.digest[:24],
        key=key,
        baseline=baseline,
        requested=baseline,
        admitted=selected,
        selected=selected,
        candidates=(evidence,),
        state=ProfileState.VALIDATED,
        selection_reason="validated_throughput_gain",
        created_at_unix_ns=1,
        last_validation_unix_ns=1,
    )


@pytest.mark.parametrize(
    "field,replacement",
    [
        ("schema", SchemaFingerprint(search_policy_version="other")),
        ("system", SystemFingerprint("other", "linux-x86", "cpu", 16, 64 * 1024**3)),
        ("accelerator", AcceleratorFingerprint("GPU-2", "RTX", "8.9", 48 * 1024**3)),
        (
            "software",
            SoftwareFingerprint(
                "1",
                "abc",
                "cpython-3.13",
                "torch",
                "fp32",
                "1",
                "13",
                "9",
                "10",
                "2",
                "8",
                "absent",
            ),
        ),
        ("models", (ModelArtifactFingerprint("detector", "b" * 64, 640, 640),)),
        ("frame", FrameFingerprint(1201, 1200, 3, "bgr8", 1.0, "ffmpeg")),
        (
            "detector",
            DetectorFingerprint("yolo_obb", "obb", (0,), 0.3, 0.7, 25, "direct"),
        ),
        ("slice", SliceFingerprint(True, 640, 640, 0.2, 0.2, False, "roi", "20x2")),
        ("pipeline", PipelineFingerprint(("detector", "pose"), "forward", "batch", ())),
        ("workload", WorkloadFingerprint(25, 32, 64, 32, 64, ("256x128",))),
    ],
)
def test_exact_key_invalidates_when_any_category_changes(field, replacement):
    key = _key()
    changed = replace(key, **{field: replacement})
    assert changed.digest != key.digest
    assert TuningProfileKey.from_dict(key.to_dict()) == key


def test_density_buckets_and_crop_geometry_are_stable_and_explicit():
    assert [count_bucket(value) for value in (0, 1, 2, 3, 4, 5, 25)] == [
        0,
        1,
        2,
        4,
        4,
        8,
        32,
    ]
    workload = WorkloadFingerprint.from_counts(25, [1, 4, 25], [2, 8, 25], ["256x128"])
    assert workload.detections_p95_bucket == 32
    assert workload.crops_p95_bucket == 32
    assert workload.canonical_crop_geometries == ("256x128",)


def test_software_fingerprint_never_uses_an_unknown_code_revision():
    fingerprint = default_software_fingerprint(backend="torch", precision="fp32")

    assert fingerprint.hydra_commit != "unknown"
    assert fingerprint.hydra_commit


def test_model_digest_detects_same_size_rewrite_with_restored_mtime(tmp_path):
    model = tmp_path / "model.bin"
    model.write_bytes(b"model-a")
    original = model.stat()
    first = model_content_digest(model)

    model.write_bytes(b"model-b")
    os.utime(model, ns=(original.st_atime_ns, original.st_mtime_ns))

    assert model_content_digest(model) != first


def test_overlay_settings_apply_without_mutating_requested_config():
    config = InferenceConfig(
        bgsub=BgSubConfig(),
        headtail=HeadTailConfig("head.pt", batch_size=25),
        cnn_phases=[CNNConfig("color", "color.pt", batch_size=25)],
        pose=PoseConfig(backend="yolo", yolo=PoseYOLOConfig("pose.pt", batch_size=25)),
        detection_batch_size=1,
        pipeline_depth=2,
    )
    selected = _settings()
    effective = selected.apply(config)
    assert config.detection_batch_size == 1
    assert config.headtail.batch_size == 25
    assert config.cnn_phases[0].batch_size == 25
    assert config.pose.yolo.batch_size == 25
    assert effective.detection_batch_size == 4
    assert effective.headtail.batch_size == 4
    assert effective.cnn_phases[0].batch_size == 4
    assert effective.pose.yolo.batch_size == 4


def _obb_config() -> InferenceConfig:
    return InferenceConfig(
        obb=OBBConfig(
            direct=OBBDirectConfig(
                model_path="obb.pt",
                slice=SliceConfig(enabled=True, tile_batch_size=16),
            )
        ),
        detection_batch_size=1,
        pipeline_depth=2,
    )


@pytest.mark.parametrize(
    "status,reason",
    [
        ("recorded", "validated profile already recorded"),
        ("kept_current_settings", "no manual field changed"),
        ("fallback", "calibration failed"),
        ("disabled", "automatic inference tuning is disabled"),
    ],
)
def test_a_no_op_overlay_leaves_the_configured_tile_batch_alone(status, reason):
    """record/kept_current/fallback/disabled overlays reuse the configured
    baseline verbatim, so the explicit Tiles / call value survives."""
    settings = _settings()
    overlay = InferenceRuntimeOverlay.baseline(settings, status=status, reason=reason)
    config = _obb_config()
    effective = overlay.apply(config)
    assert effective.obb.direct.slice.tile_batch_size == settings.slice_tile_batch_size


@pytest.mark.parametrize("status", ["calibrated", "cache_hit", "cache_hit_after_wait"])
def test_an_active_override_overlay_supplies_its_tile_batch_size(status):
    """A real tuner override installs its coordinated slice_tile_batch_size."""
    requested = _settings()
    effective_settings = requested.with_value("slice_tile_batch_size", 4)
    overlay = InferenceRuntimeOverlay(
        requested=requested,
        admitted=effective_settings,
        effective=effective_settings,
        field_sources=tuple((f, "tuned") for f in requested.field_names()),
        status=status,
        reason="tuned",
    )
    config = _obb_config()
    effective = overlay.apply(config)
    assert effective.obb.direct.slice.tile_batch_size == 4


def _always_failing_planner() -> CandidatePlanner:
    """A ``CandidatePlanner`` whose ``admit()`` rejects every settings vector,
    including the baseline -- ``down_admit`` therefore always falls back with
    ``admitted=False``."""
    observation = ResourceObservation(
        total_host_bytes=1,
        available_host_bytes=1,
        accelerator_kind=AcceleratorKind.CUDA,
        accelerator_name="gpu",
        total_accelerator_bytes=1,
        available_accelerator_bytes=1,
    )
    return CandidatePlanner(
        AdmissionContext(
            observation,
            frame_bytes=1,
            crop_count_p95=8,
            hard_maxima=(("detection_batch_size", 4),),
            # A host cost that dwarfs any possible usable_host budget, so
            # every candidate (including the baseline) fails admission.
            cost=MemoryCostModel(fixed_host_bytes=10**12),
            policy=ResourcePolicy(
                reserve_host_bytes=0,
                reserve_host_fraction=0,
                accelerator_safety_fraction=1,
            ),
        )
    )


def test_admission_failure_on_a_cache_hit_falls_back_to_the_baseline(
    tmp_path,
):
    """``coordinator._reuse`` keeps ``status="cache_hit"`` even when live
    resource admission fails -- ``effective`` then falls back to the
    requested baseline verbatim (a pure no-op, reason starting with
    "baseline fallback: ..."), so applying it changes nothing.
    """
    key = _key()
    profile = _profile(key)
    store = InferenceTuningProfileStore(tmp_path)
    store.save(profile)

    baseline = _settings(detection_batch_size=1)
    request = AutotuneRequest(
        key,
        baseline,
        _always_failing_planner(),
        mode="lookup",
    )

    result = AutotuneCoordinator(store).resolve(request)
    overlay = result.overlay

    assert overlay.status == "cache_hit"
    assert overlay.reason.startswith("baseline fallback:")
    assert overlay.effective == overlay.requested == baseline

    config = _obb_config()
    effective_config = overlay.apply(config)
    assert (
        effective_config.obb.direct.slice.tile_batch_size
        == baseline.slice_tile_batch_size
    )


def test_candidate_generation_is_bounded_and_crop_values_canonicalize_to_density():
    context = AdmissionContext(
        _observation(),
        frame_bytes=1200 * 1200 * 3,
        crop_count_p95=25,
        hard_maxima=(
            ("detection_batch_size", 16),
            ("pose_batch_size", 64),
            ("pipeline_depth", 4),
        ),
        policy=ResourcePolicy(reserve_host_bytes=0, reserve_host_fraction=0),
    )
    planner = CandidatePlanner(context)
    settings = _settings(pose_batch_size=64)
    assert planner.canonicalize(settings).pose_batch_size == 25
    assert planner.values_for("pose_batch_size", _settings(pose_batch_size=4)) == (
        1,
        2,
        4,
        8,
        16,
        25,
    )
    assert planner.values_for("detection_batch_size", settings) == (1, 2, 4, 8, 16)


def test_4512_frame_batch_is_pruned_before_model_loading():
    planner = CandidatePlanner(
        AdmissionContext(
            _observation(),
            frame_bytes=4512 * 4512 * 3,
            crop_count_p95=25,
            hard_maxima=(("detection_batch_size", 64), ("pipeline_depth", 4)),
            policy=ResourcePolicy(reserve_host_bytes=0, reserve_host_fraction=0),
        )
    )
    assert planner.admit(_settings(detection_batch_size=8, pipeline_depth=1)).admitted
    rejected = planner.admit(_settings(detection_batch_size=16, pipeline_depth=1))
    assert not rejected.admitted
    assert rejected.reason == "frame_buffer_budget"


def test_measured_envelope_never_extrapolates_above_largest_success():
    identity = ProfileIdentity("detect", "sha256:a", "torch", "GPU-1", "fp16", "obb")
    record = MemoryMeasurement(
        identity,
        PressureSettings(640, 640, batch_size=4),
        AcceleratorKind.CUDA,
        host_peak_bytes=1,
        accelerator_allocated_peak_bytes=100,
        accelerator_reserved_peak_bytes=200,
    )
    context = AdmissionContext(
        _observation(free=1_000),
        frame_bytes=1,
        crop_count_p95=1,
        hard_maxima=(("detection_batch_size", 16),),
        cost=MemoryCostModel(),
        policy=ResourcePolicy(
            reserve_host_bytes=0,
            reserve_host_fraction=0,
            accelerator_safety_fraction=0.8,
        ),
        measured_records=(("detection_batch_size", (record,)),),
    )
    planner = CandidatePlanner(context)
    assert planner.admit(_settings(detection_batch_size=4)).admitted
    assert planner.admit(_settings(detection_batch_size=8)).reason.startswith(
        "above_largest"
    )


def test_profile_store_roundtrip_corruption_schema_and_singleflight(tmp_path):
    store = InferenceTuningProfileStore(tmp_path)
    profile = _profile()
    with store.claim(profile.key) as first:
        assert first.acquired
        with store.claim(profile.key) as second:
            assert not second.acquired
        store.save(profile)
    assert store.load(profile.key) == profile

    path = tmp_path / f"{profile.key.digest}.json"
    raw = json.loads(path.read_text())
    raw["schema_version"] += 1
    path.write_text(json.dumps(raw))
    assert store.load(profile.key) is None
    path.write_text("{broken")
    assert store.load(profile.key) is None


def test_production_observation_rejects_noncanonical_profile_ids(tmp_path):
    store = InferenceTuningProfileStore(tmp_path)
    profile = _profile()
    store.save(profile)

    assert store.observe_production_throughput("*", 100.0) is None
    assert (
        store.observe_production_throughput(profile.profile_id.upper(), 100.0) is None
    )


def test_profile_atomic_failure_keeps_previous_record(tmp_path, monkeypatch):
    store = InferenceTuningProfileStore(tmp_path)
    original = _profile()
    store.save(original)

    def fail_replace(*_args):
        raise OSError("simulated promotion failure")

    monkeypatch.setattr(
        "hydra_suite.core.inference.autotune.store.os.replace", fail_replace
    )
    with pytest.raises(OSError, match="promotion"):
        store.save(replace(original, selection_reason="new"))
    assert store.load(original.key) == original


def test_three_comparable_production_regressions_mark_profile_provisional(tmp_path):
    store = InferenceTuningProfileStore(tmp_path)
    original = _profile()
    original = replace(
        original,
        candidates=(replace(original.candidates[0], phase="final_validation"),),
    )
    store.save(original)

    assert (
        store.observe_production_throughput(original.profile_id, 80.0)
        is ProfileState.VALIDATED
    )
    assert (
        store.observe_production_throughput(original.profile_id, 82.0)
        is ProfileState.VALIDATED
    )
    assert (
        store.observe_production_throughput(original.profile_id, 84.0)
        is ProfileState.PROVISIONAL
    )

    updated = store.load(original.key)
    assert updated is not None
    assert updated.state is ProfileState.PROVISIONAL
    assert updated.selected == original.selected
    assert updated.observed_production_throughput == (80.0, 82.0, 84.0)
    assert "regressed" in (updated.invalidation_reason or "")


def test_production_density_bucket_change_immediately_marks_profile_provisional(
    tmp_path,
):
    store = InferenceTuningProfileStore(tmp_path)
    original = _profile()
    store.save(original)

    state = store.observe_production_throughput(
        original.profile_id,
        100.0,
        detection_counts=(1, 2, 3),
        crop_counts=(1, 2, 3),
    )

    assert state is ProfileState.PROVISIONAL
    updated = store.load(original.key)
    assert updated is not None
    assert updated.selected == original.selected
    assert updated.invalidation_reason == "production workload density bucket changed"


def _outputs(identity="A", *, x=1.0, rows=1):
    frame = pd.DataFrame(
        {
            "FrameID": range(rows),
            "DetectionID": range(rows),
            "X": [x] * rows,
            "Y": [2.0] * rows,
            "Theta": [0.0] * rows,
            "UniqueIdentityKey": [identity] * rows,
            "PoseNoseX": [3.0] * rows,
        }
    )
    return CalibrationOutputs(frame.copy(), frame.copy())


def test_correctness_gate_rejects_empty_count_nan_and_categorical_changes():
    baseline = _outputs()
    assert compare_outputs(baseline, _outputs()).passed
    assert not compare_outputs(baseline, _outputs(identity="B")).passed
    assert not compare_outputs(baseline, _outputs(rows=2)).passed
    changed_nan = _outputs()
    changed_nan.forward.loc[0, "PoseNoseX"] = np.nan
    assert not compare_outputs(baseline, changed_nan).passed

    complete = _outputs()
    missing_pose = CalibrationOutputs(
        complete.forward.drop(columns=["PoseNoseX"]), complete.final
    )
    assert not compare_outputs(baseline, missing_pose).passed
    empty = CalibrationOutputs(baseline.forward.iloc[:0], baseline.final.iloc[:0])
    assert not compare_outputs(empty, empty).passed


def test_categorical_and_nan_gates_follow_positional_matches_when_ids_change():
    baseline = _outputs(identity="A")
    changed = _outputs(identity="B")
    changed.forward.loc[0, "DetectionID"] = 99
    changed.final.loc[0, "DetectionID"] = 99

    verdict = compare_outputs(baseline, changed)

    assert not verdict.passed
    assert verdict.unmatched_rows == 0
    assert verdict.categorical_mismatches == 2

    changed_nan = _outputs(identity="A")
    changed_nan.forward.loc[0, "DetectionID"] = 99
    changed_nan.final.loc[0, "DetectionID"] = 99
    changed_nan.forward.loc[0, "PoseNoseX"] = np.nan
    assert not compare_outputs(baseline, changed_nan).passed


def test_identity_confidence_is_reported_only_not_an_exact_categorical_field():
    """``IdentityRealtimeConfidence`` is reported-only, and never a string.

    Two claims, both narrow. It must not be swept into the exact-string
    categorical set (where 0.75 and 0.750000 would differ for formatting
    reasons alone). And it carries no decision: the decision it feeds is
    ``IdentityRealtimeCommitted``, which IS compared exactly -- so a drifting
    confidence, on its own, is not evidence that anything changed. It is
    enumerated in ``_REPORTED_ONLY_COLUMNS``; see that block for why an
    enumerated exemption was preferred to a hand-picked tolerance.
    """

    baseline = _outputs()
    baseline.forward["IdentityRealtimeConfidence"] = [0.75]
    baseline.final["IdentityRealtimeConfidence"] = [0.75]
    candidate = CalibrationOutputs(baseline.forward.copy(), baseline.final.copy())
    candidate.forward["IdentityRealtimeConfidence"] = [0.75001]
    candidate.final["IdentityRealtimeConfidence"] = [0.75001]

    verdict = compare_outputs(baseline, candidate)
    assert verdict.passed
    assert verdict.categorical_mismatches == 0

    # But its decision shadow is exact: flip that and the gate rejects.
    committed = CalibrationOutputs(baseline.forward.copy(), baseline.final.copy())
    committed.forward["IdentityRealtimeCommitted"] = ["A"]
    committed.final["IdentityRealtimeCommitted"] = ["A"]
    baseline.forward["IdentityRealtimeCommitted"] = ["B"]
    baseline.final["IdentityRealtimeCommitted"] = ["B"]
    assert not compare_outputs(baseline, committed).passed


def test_randomized_blocks_and_paired_confidence_are_deterministic():
    order = deterministic_block_order((1, 2, 4), 5, seed=7)
    assert order == deterministic_block_order((1, 2, 4), 5, seed=7)
    assert all(set(block) == {1, 2, 4} for block in order)
    low, high = paired_gain_interval([100] * 5, [110] * 5, seed=7)
    assert low == pytest.approx(0.1)
    assert high == pytest.approx(0.1)
    assert MeasurementProtocol().minimum_blocks == 5
