from __future__ import annotations

import multiprocessing
import time
from pathlib import Path

import pandas as pd

from hydra_suite.core.inference.autotune.candidates import (
    AdmissionContext,
    CandidatePlanner,
    MemoryCostModel,
)
from hydra_suite.core.inference.autotune.coordinator import (
    AutotuneCoordinator,
    AutotuneRequest,
)
from hydra_suite.core.inference.autotune.equivalence import CalibrationOutputs
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
)
from hydra_suite.core.inference.autotune.models import (
    CandidateEvidence,
    EquivalenceVerdict,
    InferenceTuningProfile,
    InferenceTuningSettings,
    ProfileState,
)
from hydra_suite.core.inference.autotune.search import (
    CoordinateSearch,
    TrialObservation,
)
from hydra_suite.core.inference.autotune.store import InferenceTuningProfileStore
from hydra_suite.runtime.resource_budget import (
    AcceleratorKind,
    ResourceObservation,
    ResourcePolicy,
)


def _key():
    return TuningProfileKey(
        SchemaFingerprint(),
        SystemFingerprint("host", "linux", "cpu", 8, 32 * 1024**3),
        AcceleratorFingerprint("gpu", "gpu", "8.9", 1_000),
        SoftwareFingerprint(
            "1", "c", "py", "torch", "fp16", "d", "c", "u", "t", "p", "y", "s"
        ),
        (ModelArtifactFingerprint("detector", "a" * 64),),
        FrameFingerprint(10, 10, 3, "bgr8", 1.0, "ffmpeg"),
        DetectorFingerprint("obb", "obb", (), 0.2, 0.7, 8, "direct"),
        SliceFingerprint(False, 0, 0, 0.2, 0.2, False, "none", "none"),
        PipelineFingerprint(("detector", "pose"), "forward", "batch", ()),
        WorkloadFingerprint(8, 8, 8, 8, 8, ("10x10",)),
    )


def _settings(det=1, pose=1, identity=1):
    return InferenceTuningSettings(
        detection_batch_size=det,
        pose_batch_size=pose,
        identity_batch_sizes=(("animal", identity),),
        pipeline_depth=2,
    )


def _planner(*, free=10_000, cost=None, cached_fields=frozenset()):
    observation = ResourceObservation(
        total_host_bytes=100_000,
        available_host_bytes=100_000,
        accelerator_kind=AcceleratorKind.CUDA,
        accelerator_name="gpu",
        total_accelerator_bytes=10_000,
        available_accelerator_bytes=free,
    )
    return CandidatePlanner(
        AdmissionContext(
            observation,
            frame_bytes=1,
            crop_count_p95=8,
            hard_maxima=(
                ("detection_batch_size", 4),
                ("pose_batch_size", 4),
                ("identity_batch_size:animal", 8),
                ("pipeline_depth", 2),
            ),
            cost=cost or MemoryCostModel(),
            policy=ResourcePolicy(
                reserve_host_bytes=0,
                reserve_host_fraction=0,
                accelerator_safety_fraction=1,
            ),
            cached_fields=cached_fields,
        )
    )


def _outputs(identity="A"):
    data = pd.DataFrame(
        {
            "FrameID": [0, 1],
            "DetectionID": [0, 10_000],
            "X": [1.0, 1.0],
            "Y": [2.0, 2.0],
            "Theta": [0.0, 0.0],
            "UniqueIdentityKey": [identity, identity],
        }
    )
    return CalibrationOutputs(data.copy(), data.copy())


class ConflictingExecutor:
    """Detector's stage-local winner regresses in the complete pipeline."""

    def __init__(self):
        self.calls = []

    def run(self, settings, *, phase, field_name, block_index, should_cancel):
        self.calls.append((phase, field_name, block_index, settings))
        if phase == "stage" and field_name == "detection_batch_size":
            throughput = {1: 100.0, 2: 160.0, 4: 200.0}[settings.detection_batch_size]
        elif phase == "full" and field_name == "detection_batch_size":
            throughput = {1: 100.0, 2: 95.0, 4: 90.0}[settings.detection_batch_size]
        elif phase == "stage" and field_name == "pose_batch_size":
            throughput = {1: 100.0, 2: 106.0, 4: 112.0}[settings.pose_batch_size]
        elif phase == "full" and field_name == "pose_batch_size":
            throughput = {1: 100.0, 2: 103.0, 4: 110.0}[settings.pose_batch_size]
        elif phase == "stage" and field_name == "identity_batch_size:animal":
            throughput = 150.0 if settings.value_for(field_name) > 1 else 100.0
        elif phase == "full" and field_name == "identity_batch_size:animal":
            throughput = 150.0 if settings.value_for(field_name) > 1 else 100.0
        else:
            throughput = 110.0 if settings.pose_batch_size == 4 else 100.0
        identity = (
            "B"
            if field_name == "identity_batch_size:animal"
            and settings.value_for(field_name) > 1
            else "A"
        )
        return TrialObservation(
            settings,
            throughput,
            0.5,
            _outputs(identity),
            accelerator_peak_bytes=sum(
                settings.value_for(name) or 0 for name in settings.field_names()
            ),
        )


def test_coordinate_search_requires_full_pipeline_and_exact_identity():
    baseline = _settings()
    executor = ConflictingExecutor()
    result = CoordinateSearch(_planner(), executor).run(
        baseline,
        stage_shares={
            "detection_batch_size": 0.7,
            "pose_batch_size": 0.2,
            "identity_batch_size:animal": 0.1,
        },
    )
    assert result.completed
    assert result.selected.detection_batch_size == 1
    assert result.selected.pose_batch_size == 4
    assert result.selected.value_for("identity_batch_size:animal") == 1
    assert any(reason == "gain_or_confidence_gate" for _, reason in result.rejected)
    assert any("categorical" in reason for _, reason in result.rejected)
    assert {block for phase, _, block, _ in executor.calls if phase == "stage"} >= set(
        range(5)
    )


class FinalRegressionExecutor(ConflictingExecutor):
    def run(self, settings, *, phase, field_name, block_index, should_cancel):
        if phase == "final_validation" and settings.pose_batch_size > 1:
            return TrialObservation(
                settings,
                90.0,
                0.5,
                _outputs(),
                warmup_calls=3,
                warmup_frames=8,
            )
        return super().run(
            settings,
            phase=phase,
            field_name=field_name,
            block_index=block_index,
            should_cancel=should_cancel,
        )


def test_fresh_final_vector_must_repeat_the_full_pipeline_performance_gain():
    result = CoordinateSearch(_planner(), FinalRegressionExecutor()).run(
        _settings(),
        stage_shares={"pose_batch_size": 1.0},
    )

    assert not result.completed
    assert result.selected == _settings()
    assert result.reason == "final_performance_gate_failed"


class ExplodingExecutor:
    def run(self, *_args, **_kwargs):
        raise AssertionError("a validated cache hit must not launch trials")


class FixedExecutor:
    def run(self, settings, *, phase, field_name, block_index, should_cancel):
        return TrialObservation(
            settings,
            100.0,
            0.5,
            _outputs(),
            warmup_calls=3,
            warmup_frames=8,
        )


class CountingExecutor:
    def __init__(self, counter_path: str):
        self.counter_path = counter_path

    def run(self, settings, *, phase, field_name, block_index, should_cancel):
        with Path(self.counter_path).open("a", encoding="utf-8") as stream:
            stream.write("trial\n")
        time.sleep(0.02)
        return TrialObservation(
            settings,
            100.0,
            0.5,
            _outputs(),
            warmup_calls=3,
            warmup_frames=8,
            measured_frames=26,
        )


def _resolve_in_fresh_process(root: str, counter: str, barrier, queue) -> None:
    barrier.wait()
    result = AutotuneCoordinator(
        InferenceTuningProfileStore(root),
        trial_executor=CountingExecutor(counter),
    ).resolve(
        AutotuneRequest(
            _key(),
            _settings(),
            _planner(cached_fields=frozenset(_settings().field_names())),
            mode="automatic",
            singleflight_wait_seconds=10.0,
        )
    )
    queue.put(result.overlay.status)


def test_cache_hit_down_admits_only_validated_settings_and_does_not_mutate_store(
    tmp_path,
):
    key = _key()
    baseline = _settings(det=1)
    selected = _settings(det=4)
    smaller = _settings(det=2)
    evidence = tuple(
        CandidateEvidence(
            settings,
            (100.0,) * 5,
            stage_seconds_samples=(0.5,) * 5,
            accelerator_peak_bytes=settings.detection_batch_size * 100,
            warmup_calls=3,
            warmup_frames=8,
            equivalence=EquivalenceVerdict(True),
        )
        for settings in (selected, smaller)
    )
    profile = InferenceTuningProfile(
        key.digest[:24],
        key,
        baseline,
        baseline,
        selected,
        selected,
        evidence,
        ProfileState.VALIDATED,
        "winner",
    )
    store = InferenceTuningProfileStore(tmp_path)
    store.save(profile)
    planner = _planner(
        free=250,
        cost=MemoryCostModel(detector_frame_accelerator_bytes=100),
    )
    result = AutotuneCoordinator(store, trial_executor=ExplodingExecutor()).resolve(
        AutotuneRequest(key, baseline, planner, mode="automatic")
    )
    assert result.overlay.status == "cache_hit"
    assert result.overlay.effective.detection_batch_size == 2
    assert store.load(key).selected.detection_batch_size == 4


def test_manual_field_precedence_over_cached_profile(tmp_path):
    key = _key()
    baseline = _settings(det=1, pose=1)
    selected = _settings(det=4, pose=4)
    evidence = CandidateEvidence(
        selected,
        (120.0,) * 5,
        stage_seconds_samples=(0.5,) * 5,
        warmup_calls=3,
        warmup_frames=8,
        equivalence=EquivalenceVerdict(True),
    )
    profile = InferenceTuningProfile(
        key.digest[:24],
        key,
        baseline,
        baseline,
        selected,
        selected,
        (evidence,),
        ProfileState.VALIDATED,
        "winner",
    )
    store = InferenceTuningProfileStore(tmp_path)
    store.save(profile)
    result = AutotuneCoordinator(store, trial_executor=ExplodingExecutor()).resolve(
        AutotuneRequest(
            key,
            baseline,
            _planner(),
            mode="automatic",
            manual_fields=frozenset({"pose_batch_size"}),
        )
    )
    assert result.overlay.effective.detection_batch_size == 4
    assert result.overlay.effective.pose_batch_size == 1
    assert dict(result.overlay.field_sources)["pose_batch_size"] == "manual"


def test_permanently_ineligible_runtime_never_reuses_validated_cache(tmp_path):
    key = _key()
    baseline = _settings(det=1)
    selected = _settings(det=4)
    profile = InferenceTuningProfile(
        key.digest[:24],
        key,
        baseline,
        baseline,
        selected,
        selected,
        (
            CandidateEvidence(
                selected,
                (120.0,) * 5,
                stage_seconds_samples=(0.5,) * 5,
                warmup_calls=3,
                warmup_frames=8,
                equivalence=EquivalenceVerdict(True),
            ),
        ),
        ProfileState.VALIDATED,
        "winner",
    )
    store = InferenceTuningProfileStore(tmp_path)
    store.save(profile)

    result = AutotuneCoordinator(store, trial_executor=FixedExecutor()).resolve(
        AutotuneRequest(
            key,
            baseline,
            _planner(),
            mode="automatic",
            eligible=False,
            allow_cached_reuse=False,
            eligibility_reason="automatic tuning is CUDA-only",
        )
    )

    assert result.profile is None
    assert result.overlay.status == "deferred_due_to_contention"
    assert result.overlay.effective == baseline
    assert result.overlay.profile_id is None


def test_record_only_persists_but_does_not_apply(tmp_path):
    executor = ConflictingExecutor()
    store = InferenceTuningProfileStore(tmp_path)
    key = _key()
    baseline = _settings()
    result = AutotuneCoordinator(store, trial_executor=executor).resolve(
        AutotuneRequest(
            key,
            baseline,
            _planner(),
            mode="record",
            stage_shares=(("pose_batch_size", 1.0),),
        )
    )
    assert result.overlay.status == "recorded"
    assert result.overlay.effective == baseline
    assert store.load(key).state is ProfileState.VALIDATED


def test_concurrent_fresh_processes_create_one_profile_and_one_trial_set(tmp_path):
    context = multiprocessing.get_context("spawn")
    barrier = context.Barrier(2)
    queue = context.Queue()
    counter = tmp_path / "trial-count.txt"
    args = (str(tmp_path / "profiles"), str(counter), barrier, queue)
    processes = [
        context.Process(target=_resolve_in_fresh_process, args=args) for _ in range(2)
    ]
    for process in processes:
        process.start()
    for process in processes:
        process.join(timeout=60)

    assert all(process.exitcode == 0 for process in processes)
    statuses = {queue.get(timeout=2), queue.get(timeout=2)}
    assert "calibrated" in statuses
    assert statuses <= {"calibrated", "cache_hit", "cache_hit_after_wait"}
    assert len(counter.read_text(encoding="utf-8").splitlines()) == 10
    records = list((tmp_path / "profiles").glob("*.json"))
    assert len(records) == 1
