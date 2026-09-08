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
from hydra_suite.core.inference.autotune.measure import MeasurementProtocol
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

from .autotune_helpers import record_mode_request_with_validated_cache


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


def _planner(*, free=10_000, cost=None, cached_fields=frozenset(), det_max=4):
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
                ("detection_batch_size", det_max),
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
            phase="full",
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


def test_down_admission_excludes_stage_only_evidence_but_admits_full_evidence(
    tmp_path,
):
    """S4: `search.py` states plainly "stage screens never authorize a
    winner" -- down-admission must honor that too. A stage-phase candidate
    that would otherwise be the largest fitting setting must be ignored;
    a full-phase candidate that fits must still be admitted.
    """
    key = _key()
    baseline = _settings(det=1)
    selected = _settings(det=4)
    stage_only = _settings(det=3)  # would fit (300 <= 350) but never ran full
    full_evidence = _settings(det=2)  # fits (200 <= 350) and ran full
    evidence = (
        CandidateEvidence(
            selected,
            (100.0,) * 5,
            stage_seconds_samples=(0.5,) * 5,
            accelerator_peak_bytes=400,
            warmup_calls=3,
            warmup_frames=8,
            equivalence=EquivalenceVerdict(True),
            phase="full",
        ),
        CandidateEvidence(
            stage_only,
            (100.0,) * 5,
            stage_seconds_samples=(0.5,) * 5,
            accelerator_peak_bytes=300,
            warmup_calls=3,
            warmup_frames=8,
            equivalence=EquivalenceVerdict(True),
            phase="stage",
        ),
        CandidateEvidence(
            full_evidence,
            (100.0,) * 5,
            stage_seconds_samples=(0.5,) * 5,
            accelerator_peak_bytes=200,
            warmup_calls=3,
            warmup_frames=8,
            equivalence=EquivalenceVerdict(True),
            phase="full",
        ),
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
        free=350,
        cost=MemoryCostModel(detector_frame_accelerator_bytes=100),
    )
    result = AutotuneCoordinator(store, trial_executor=ExplodingExecutor()).resolve(
        AutotuneRequest(key, baseline, planner, mode="automatic")
    )
    # A bug that includes stage-only evidence would admit det=3 here (300 <=
    # 350, the largest of ALL successful evidence). The fix must skip it and
    # admit the largest FULL-phase evidence that fits: det=2.
    assert result.overlay.effective.detection_batch_size == 2


def test_manual_field_precedence_over_cached_profile(tmp_path):
    # S3: the coordinator no longer splices a manual field's baseline value
    # into a cached ``selected`` at reuse time -- the profile key now folds
    # in the baseline plus which fields were manually pinned, so a cache hit
    # can only happen against a profile whose search already held
    # ``pose_batch_size`` fixed at the same value (1). Splicing a different
    # cached value in here would build a joint vector nobody measured.
    key = _key()
    baseline = _settings(det=1, pose=1)
    selected = _settings(det=4, pose=1)
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
    stored = store.load(key)
    assert stored.state is ProfileState.VALIDATED
    assert dict(stored.calibration_summary)["candidate_count"] == len(stored.candidates)
    assert dict(stored.calibration_summary)["detections_p95_bucket"] == 8


def test_record_mode_never_applies_even_on_a_cache_hit(tmp_path):
    """Run 2 with a VALIDATED profile in the cache must still keep baseline."""
    store, request = record_mode_request_with_validated_cache(tmp_path)
    result = AutotuneCoordinator(store, trial_executor=ExplodingExecutor()).resolve(
        request
    )
    assert result.overlay.status == "recorded"
    assert result.overlay.effective == request.baseline


def test_search_status_reports_incumbent_field_and_budget() -> None:
    statuses = []

    CoordinateSearch(_planner(), FixedExecutor()).run(
        _settings(),
        stage_shares={"pose_batch_size": 1.0},
        status_callback=statuses.append,
    )

    assert statuses[0].startswith("Optimizing inference — baseline")
    assert any("field pose_batch_size" in status for status in statuses)
    assert all("/120s" in status for status in statuses)


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
    # 5 baseline blocks + 1 same-window determinism duplicate + 5 final
    # validation blocks. Exactly ONE calibration ran: the second process
    # waited on the lock and reused the record.
    assert len(counter.read_text(encoding="utf-8").splitlines()) == 11
    records = list((tmp_path / "profiles").glob("*.json"))
    assert len(records) == 1


class WarmupStarvedExecutor:
    """Detector batches at or above 16 cannot reach the warmup-call minimum."""

    def run(self, settings, *, phase, field_name, block_index, should_cancel):
        starved = settings.detection_batch_size >= 16
        return TrialObservation(
            settings,
            100.0 + settings.detection_batch_size,
            0.5,
            _outputs(),
            warmup_calls=2 if starved else 3,
            warmup_frames=8,
        )


def test_large_batch_candidates_are_rejected_with_a_reason_not_dropped():
    """A candidate that cannot satisfy the warmup minimum appears in rejected.

    Before this guard, ``_measure`` dropped an incomplete candidate with a
    bare ``continue``: it produced no evidence and no rejection record, so
    the real detector search space silently collapsed to the small batches.
    """

    result = CoordinateSearch(_planner(det_max=16), WarmupStarvedExecutor()).run(
        _settings(),
        stage_shares={"detection_batch_size": 1.0},
    )

    starved = [
        (label, reason)
        for label, reason in result.rejected
        if label.startswith("detection_batch_size=16,")
    ]
    assert starved, f"batch-16 candidate vanished; rejected={result.rejected}"
    assert any("measurement_incomplete" in reason for _label, reason in starved)


def test_an_admitting_nothing_search_says_why_rather_than_going_quiet():
    """If no candidate can complete, the baseline gate reports the reason.

    This is the guard against a mis-set ``MeasurementProtocol.maximum_frames``:
    when nothing satisfies the protocol the search must fail loudly with a
    recorded reason, not return "kept_current_settings" as if it had measured.
    """

    class NeverCompletes(WarmupStarvedExecutor):
        def run(self, settings, *, phase, field_name, block_index, should_cancel):
            observation = super().run(
                settings,
                phase=phase,
                field_name=field_name,
                block_index=block_index,
                should_cancel=should_cancel,
            )
            return TrialObservation(
                observation.settings,
                observation.throughput,
                0.5,
                observation.outputs,
                warmup_calls=1,
                warmup_frames=8,
            )

    result = CoordinateSearch(_planner(), NeverCompletes()).run(_settings())

    assert not result.completed
    assert result.reason == "baseline_measurement_incomplete"
    assert any("measurement_incomplete" in reason for _label, reason in result.rejected)


def test_a_per_trial_timeout_is_recorded_as_a_rejection_with_its_class():
    """An executor-side failure must be named, not silently dropped.

    ``ContainedTrialExecutor._run_once`` turns a per-trial timeout (and an
    OOM, and a crashed child) into a ``TrialObservation`` carrying a
    ``failure_class``. Those never reach ``measurement_complete`` -- they are
    filtered out as unsuccessful first -- so the block-count exit is the one
    that has to report them.
    """

    class TimingOutExecutor:
        def run(self, settings, *, phase, field_name, block_index, should_cancel):
            if settings.detection_batch_size >= 4:
                return TrialObservation(
                    settings, 0.0, 0.0, None, failure_class="timeout"
                )
            return TrialObservation(
                settings,
                100.0 + settings.detection_batch_size,
                0.5,
                _outputs(),
                warmup_calls=3,
                warmup_frames=8,
            )

    result = CoordinateSearch(_planner(), TimingOutExecutor()).run(
        _settings(),
        stage_shares={"detection_batch_size": 1.0},
    )

    timed_out = [
        (label, reason)
        for label, reason in result.rejected
        if label.startswith("detection_batch_size=4,")
    ]
    assert timed_out, f"timed-out candidate vanished; rejected={result.rejected}"
    assert any("failures=timeout" in reason for _label, reason in timed_out)


# ---------------------------------------------------------------------------
# Block-window pairing (Task 10d)
#
# `sidecar_child._block_window` stripes each measurement block across the clip,
# so block i and block j cover DIFFERENT frames. Any comparison that pairs two
# different block indices is therefore comparing different video segments, not
# the same work run twice. These tests pin the pairing contract: the
# determinism duplicate reuses its partner's block index, and every candidate
# block is compared against the baseline block with the SAME index.
# ---------------------------------------------------------------------------


def _striped_outputs(block_index, identity="A", *, x=1.0):
    """Outputs whose frames depend on the block, exactly as striping produces."""

    base = int(block_index) * 100
    data = pd.DataFrame(
        {
            "FrameID": [base, base + 1],
            "DetectionID": [0, 10_000],
            "X": [x, x],
            "Y": [2.0, 2.0],
            "Theta": [0.0, 0.0],
            "UniqueIdentityKey": [identity, identity],
        }
    )
    return CalibrationOutputs(data.copy(), data.copy())


class StripedWindowExecutor:
    """Deterministic pipeline whose output window is a function of the block."""

    def __init__(self):
        self.calls = []

    def _throughput(self, settings, phase, field_name):
        if field_name == "detection_batch_size" or phase in {
            "final_validation",
            "baseline",
        }:
            return 100.0 * settings.detection_batch_size
        return 100.0

    def run(self, settings, *, phase, field_name, block_index, should_cancel):
        self.calls.append((phase, field_name, block_index, settings))
        return TrialObservation(
            settings,
            self._throughput(settings, phase, field_name),
            0.5,
            _striped_outputs(block_index),
            warmup_calls=3,
            warmup_frames=8,
            accelerator_peak_bytes=settings.detection_batch_size,
        )


def test_determinism_floor_never_compares_two_different_block_windows():
    """A striped, perfectly deterministic pipeline must clear the floor."""

    executor = StripedWindowExecutor()
    result = CoordinateSearch(_planner(), executor).run(
        _settings(),
        stage_shares={"detection_batch_size": 1.0},
    )

    assert result.reason != "baseline_nondeterministic_beyond_contract"
    assert result.completed
    # And the search actually got as far as evaluating candidates.
    assert any(phase == "stage" for phase, _f, _b, _s in executor.calls)


def test_determinism_duplicate_reuses_its_partner_block_index_and_phase():
    executor = StripedWindowExecutor()
    CoordinateSearch(_planner(), executor).run(
        _settings(),
        stage_shares={"detection_batch_size": 1.0},
    )

    baseline_blocks = [
        block for phase, _f, block, _s in executor.calls if phase == "baseline"
    ]
    # Five striped blocks plus one duplicate of the reference block.
    assert len(baseline_blocks) == 6
    assert sorted(baseline_blocks) == [0, 0, 1, 2, 3, 4]


class BlockThreeDivergenceExecutor(StripedWindowExecutor):
    """A candidate that is wrong only in block 3 must still be rejected."""

    def run(self, settings, *, phase, field_name, block_index, should_cancel):
        observation = super().run(
            settings,
            phase=phase,
            field_name=field_name,
            block_index=block_index,
            should_cancel=should_cancel,
        )
        if settings.detection_batch_size > 1 and block_index == 3:
            return TrialObservation(
                observation.settings,
                observation.throughput,
                observation.stage_seconds,
                _striped_outputs(block_index, identity="B", x=900.0),
                warmup_calls=3,
                warmup_frames=8,
            )
        return observation


def test_every_candidate_block_is_compared_not_only_block_zero():
    result = CoordinateSearch(_planner(), BlockThreeDivergenceExecutor()).run(
        _settings(),
        stage_shares={"detection_batch_size": 1.0},
    )

    assert result.selected.detection_batch_size == 1
    assert result.rejected


class SlowScreenExecutor(ConflictingExecutor):
    """Every detection mutation screens slower than the incumbent."""

    def run(self, settings, *, phase, field_name, block_index, should_cancel):
        if field_name == "detection_batch_size":
            throughput = {1: 100.0, 2: 70.0, 4: 60.0}[settings.detection_batch_size]
            self.calls.append((phase, field_name, block_index, settings))
            return TrialObservation(settings, throughput, 0.5, _outputs())
        return super().run(
            settings,
            phase=phase,
            field_name=field_name,
            block_index=block_index,
            should_cancel=should_cancel,
        )


def test_screened_losers_never_reach_full_pipeline_confirmation():
    baseline = _settings()
    executor = SlowScreenExecutor()
    result = CoordinateSearch(_planner(), executor).run(
        baseline,
        stage_shares={
            "detection_batch_size": 0.7,
            "pose_batch_size": 0.2,
            "identity_batch_size:animal": 0.1,
        },
    )
    detection_full = [
        call
        for call in executor.calls
        if call[0] == "full" and call[1] == "detection_batch_size"
    ]
    assert detection_full == []
    reasons = [reason for _, reason in result.rejected]
    slower = [
        reason for reason in reasons if "screened_slower_than_incumbent" in reason
    ]
    assert len(slower) >= 2
    assert any("70.00" in reason and "100.00" in reason for reason in slower)
    assert any("60.00" in reason and "100.00" in reason for reason in slower)
    # A genuinely faster field still earns its full-pipeline confirmation.
    assert any(
        call[0] == "full" and call[1] == "pose_batch_size" for call in executor.calls
    )
    assert result.completed
    assert result.selected.detection_batch_size == 1


def test_budget_exhaustion_names_the_fields_it_did_and_did_not_search(caplog):
    """A stop must say what it measured; silence is what cost two rounds."""
    baseline = _settings()
    clock = {"t": 0.0}

    def monotonic():
        clock["t"] += 9.0
        return clock["t"]

    with caplog.at_level("WARNING"):
        result = CoordinateSearch(
            _planner(),
            ConflictingExecutor(),
            protocol=MeasurementProtocol(budget_seconds=400.0),
            monotonic=monotonic,
        ).run(
            baseline,
            stage_shares={
                "detection_batch_size": 0.7,
                "pose_batch_size": 0.2,
                "identity_batch_size:animal": 0.1,
            },
        )

    assert not result.completed
    assert result.reason == "budget_expired"
    assert result.searched_fields
    assert result.unsearched_fields
    assert not set(result.searched_fields) & set(result.unsearched_fields)
    assert any(
        "stopped early (budget_expired)" in record.getMessage()
        for record in caplog.records
    )


def test_clamped_budget_is_reported_not_swallowed(caplog):
    from hydra_suite.core.inference.config import _clamped_float

    with caplog.at_level("WARNING"):
        value = _clamped_float(
            9_999_999.0, 600.0, 5.0, 7200.0, name="INFERENCE_AUTOTUNE_BUDGET_SECONDS"
        )
    assert value == 600.0
    joined = " ".join(record.getMessage() for record in caplog.records)
    assert "INFERENCE_AUTOTUNE_BUDGET_SECONDS" in joined
    assert "9999999" in joined.replace(",", "") or "1e+07" in joined
    assert "600" in joined
