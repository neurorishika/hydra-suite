"""Negative-caching tests for the inference autotune coordinator/store (S5).

A project that cannot complete calibration (budget exhausted, timeout,
baseline measurement never came back) must not burn the full tuning budget
on *every* run forever -- the coordinator has to remember the failure and
skip straight to a fallback overlay until a retry TTL elapses. See
`docs/superpowers/specs/2026-09-06-trackerkit-inference-autotuner-design.md`
around the `ProfileState` enumeration for the amended state description.
"""

from __future__ import annotations

import json
from dataclasses import replace

from hydra_suite.core.inference.autotune.coordinator import (
    AutotuneCoordinator,
    AutotuneRequest,
)
from hydra_suite.core.inference.autotune.fingerprint import WorkloadFingerprint
from hydra_suite.core.inference.autotune.models import (
    CandidateEvidence,
    EquivalenceVerdict,
    InferenceTuningProfile,
    ProfileState,
)
from hydra_suite.core.inference.autotune.search import TrialObservation
from hydra_suite.core.inference.autotune.store import InferenceTuningProfileStore

from .autotune_helpers import _key, _planner
from .autotune_helpers import _profile as _validated_profile
from .autotune_helpers import _settings, _store_with_validated_profile


class _NeverCompletesExecutor:
    """Always returns a single observation -- never reaches two full blocks.

    Mirrors ``tests/test_inference_autotune_search.py::NeverCompletes`` --
    the baseline measurement never completes, so ``CoordinateSearch`` returns
    ``reason="baseline_measurement_incomplete"`` and ``completed=False``.
    """

    def run(self, settings, *, phase, field_name, block_index, should_cancel):
        return TrialObservation(
            settings, 100.0, 0.5, None, warmup_calls=1, warmup_frames=8
        )


class _NeverCalledExecutor:
    def run(self, *_args, **_kwargs):
        raise AssertionError("a negative-cache hit must not launch a fresh trial")


def test_budget_expiry_is_cached_so_the_next_run_does_not_retune(tmp_path):
    store = InferenceTuningProfileStore(tmp_path)
    key = _key()
    baseline = _settings()
    request = AutotuneRequest(
        key,
        baseline,
        _planner(),
        mode="automatic",
        budget_seconds=1.0,
    )

    first = AutotuneCoordinator(
        store, trial_executor=_NeverCompletesExecutor()
    ).resolve(request)
    assert first.overlay.status in {"fallback", "cancelled"}

    record = store.load(key)
    assert record is not None
    assert record.state is ProfileState.INCOMPLETE
    assert record.invalidation_reason

    second = AutotuneCoordinator(store, trial_executor=_NeverCalledExecutor()).resolve(
        request
    )
    assert second.overlay.status == "deferred_due_to_prior_failure"
    assert second.overlay.reason == record.invalidation_reason


def test_record_mode_also_gets_the_negative_cache(tmp_path):
    """S5: negative caching applies to every mode, including record -- a
    record-only run must not re-burn the full budget on a project that
    cannot calibrate either.
    """
    store = InferenceTuningProfileStore(tmp_path)
    key = _key()
    baseline = _settings()
    request = AutotuneRequest(
        key,
        baseline,
        _planner(),
        mode="record",
        budget_seconds=1.0,
    )

    AutotuneCoordinator(store, trial_executor=_NeverCompletesExecutor()).resolve(
        request
    )
    second = AutotuneCoordinator(store, trial_executor=_NeverCalledExecutor()).resolve(
        request
    )
    assert second.overlay.status == "deferred_due_to_prior_failure"


def test_contention_detected_flag_is_defense_in_depth_not_a_live_path(tmp_path):
    """``AutotuneRequest.contention_detected`` guards ``_save_incomplete``
    directly, but through the real builder (``build_tracking_autotune_request``
    in integration.py) a True value here ALWAYS also forces
    ``eligible=False``, and ``resolve()`` returns at the
    ``not request.eligible`` check before the search loop -- and therefore
    before ``_save_incomplete`` -- ever runs. This test constructs the
    combination directly (``contention_detected=True, eligible=True``),
    which production wiring cannot produce, to pin the guard as an intended
    invariant/defense-in-depth for a future caller, NOT as evidence that
    contention is re-sampled mid-calibration today (it is not).
    """
    store = InferenceTuningProfileStore(tmp_path)
    key = _key()
    baseline = _settings()
    request = AutotuneRequest(
        key,
        baseline,
        _planner(),
        mode="automatic",
        budget_seconds=1.0,
        contention_detected=True,
    )

    AutotuneCoordinator(store, trial_executor=_NeverCompletesExecutor()).resolve(
        request
    )

    assert store.load(key) is None


def _profile(key, baseline, *, state) -> InferenceTuningProfile:
    """A minimal, otherwise-valid profile in the given state."""

    return InferenceTuningProfile(
        key.digest[:24],
        key,
        baseline,
        baseline,
        baseline,
        baseline,
        (
            CandidateEvidence(
                baseline,
                (100.0,) * 5,
                stage_seconds_samples=(0.5,) * 5,
                warmup_calls=3,
                warmup_frames=8,
                equivalence=EquivalenceVerdict(True),
                phase="full",
            ),
        ),
        state,
        "prior evidence",
        invalidation_reason=(
            "prior demotion" if state is ProfileState.PROVISIONAL else None
        ),
    )


def test_save_incomplete_writes_when_no_record_exists(tmp_path):
    store = InferenceTuningProfileStore(tmp_path)
    key = _key()
    baseline = _settings()
    request = AutotuneRequest(
        key, baseline, _planner(), mode="automatic", budget_seconds=1.0
    )

    AutotuneCoordinator(store, trial_executor=_NeverCompletesExecutor()).resolve(
        request
    )

    record = store.load(key)
    assert record is not None
    assert record.state is ProfileState.INCOMPLETE


def test_save_incomplete_refreshes_an_existing_incomplete_record(tmp_path):
    store = InferenceTuningProfileStore(tmp_path)
    key = _key()
    baseline = _settings()
    store.save(_profile(key, baseline, state=ProfileState.INCOMPLETE))
    first_timestamp = store.load(key).last_validation_unix_ns
    request = AutotuneRequest(
        key, baseline, _planner(), mode="automatic", budget_seconds=1.0
    )

    AutotuneCoordinator(store, trial_executor=_NeverCompletesExecutor()).resolve(
        request
    )

    record = store.load(key)
    assert record.state is ProfileState.INCOMPLETE
    assert record.last_validation_unix_ns >= first_timestamp


def test_save_incomplete_never_overwrites_a_provisional_record(tmp_path):
    """IMPORTANT 1: a PROVISIONAL record (e.g. demoted by a density-bucket
    mismatch) may still be the best available knowledge even though this
    particular attempt timed out. A timeout must not destroy it.
    """
    store = InferenceTuningProfileStore(tmp_path)
    key = _key()
    baseline = _settings()
    provisional = _profile(key, baseline, state=ProfileState.PROVISIONAL)
    store.save(provisional)
    request = AutotuneRequest(
        key, baseline, _planner(), mode="automatic", budget_seconds=1.0
    )

    result = AutotuneCoordinator(
        store, trial_executor=_NeverCompletesExecutor()
    ).resolve(request)

    record = store.load(key)
    assert record.state is ProfileState.PROVISIONAL
    assert record.selection_reason == provisional.selection_reason
    assert record.invalidation_reason == provisional.invalidation_reason
    # The run itself must still honestly report that it did not calibrate --
    # not silently behave as though a validated profile was reused.
    assert result.overlay.status in {"fallback", "cancelled"}


def test_save_incomplete_never_overwrites_a_validated_record(tmp_path):
    """IMPORTANT 1: a working VALIDATED profile must survive an unrelated
    timeout on a later calibration attempt at the same exact key.
    """
    store = InferenceTuningProfileStore(tmp_path)
    key = _key()
    baseline = _settings()
    validated = _profile(key, baseline, state=ProfileState.VALIDATED)
    store.save(validated)
    # allow_cached_reuse=False forces past the VALIDATED cache-hit branch so
    # the request actually reaches the search loop and times out, to prove
    # the *write* side never clobbers the record -- not just that a normal
    # cache hit would have skipped calibration anyway.
    request = AutotuneRequest(
        key,
        baseline,
        _planner(),
        mode="automatic",
        budget_seconds=1.0,
        allow_cached_reuse=False,
    )

    result = AutotuneCoordinator(
        store, trial_executor=_NeverCompletesExecutor()
    ).resolve(request)

    record = store.load(key)
    assert record.state is ProfileState.VALIDATED
    assert record.selected == validated.selected
    assert result.overlay.status in {"fallback", "cancelled"}


def test_profile_written_under_a_previous_schema_version_is_a_miss(tmp_path):
    """Carried from Task 5: a profile admitted by a superseded correctness
    gate (e.g. the pre-remediation ID-blind, mean-angle equivalence check)
    must be treated as absent evidence, never silently migrated back to
    life -- ``load`` and ``observe_production_throughput`` both gate on the
    current ``TUNING_SCHEMA_VERSION``.
    """
    store = InferenceTuningProfileStore(tmp_path)
    profile = _validated_profile()
    store.save(profile)
    path = store._record_path(profile.key)
    raw = json.loads(path.read_text())
    raw["schema_version"] = raw["schema_version"] - 1
    path.write_text(json.dumps(raw))

    assert store.load(profile.key) is None
    assert store.observe_production_throughput(profile.profile_id, 100.0) is None


def test_run_one_profile_survives_its_own_production_density_sample(tmp_path):
    """S2: the first run's own density sample must re-key the profile, not
    demote it -- the key's density was never measured (run 1 has no
    detection cache yet), so it cannot have "changed".
    """
    store, key = _store_with_validated_profile(tmp_path, keyed_on_max_targets=True)
    detection_counts = (7,) * 9 + (9,)

    state = store.observe_production_throughput(
        key.digest[:24],
        1.0,
        detection_counts=detection_counts,
        crop_counts=detection_counts,
    )

    assert state is ProfileState.VALIDATED
    corrected_workload = WorkloadFingerprint.from_counts(
        key.workload.configured_target_count,
        detection_counts,
        detection_counts,
        key.workload.canonical_crop_geometries,
        density_is_estimated=False,
    )
    reloaded = store.load(replace(key, workload=corrected_workload))
    assert reloaded is not None
    assert reloaded.state is ProfileState.VALIDATED
    assert reloaded.selected == store.load(key).selected


def test_run_one_reload_under_original_key_still_available_for_the_next_cold_video(
    tmp_path,
):
    """The re-keyed save must not destroy the run-1 fallback entry -- the
    very next brand-new video also has no cache yet and needs it.
    """
    store, key = _store_with_validated_profile(tmp_path, keyed_on_max_targets=True)
    detection_counts = (7,) * 9 + (9,)

    store.observe_production_throughput(
        key.digest[:24],
        1.0,
        detection_counts=detection_counts,
        crop_counts=detection_counts,
    )

    assert store.load(key) is not None
    assert store.load(key).state is ProfileState.VALIDATED


def test_a_measured_density_change_still_demotes_to_provisional(tmp_path):
    """S2 must not weaken the genuine-drift path: a key whose density WAS
    measured (density_is_estimated=False) still demotes on a real mismatch.
    """
    store, key = _store_with_validated_profile(tmp_path, keyed_on_max_targets=False)

    state = store.observe_production_throughput(
        key.digest[:24],
        1.0,
        detection_counts=(1, 2, 3),
        crop_counts=(1, 2, 3),
    )

    assert state is ProfileState.PROVISIONAL
    record = store.load(key)
    assert record is not None
    assert record.state is ProfileState.PROVISIONAL


def test_orphaned_locks_older_than_the_retention_window_are_pruned(tmp_path):
    """``hydra_code_identity`` mints a new key (and lock file) on every code
    edit; locks/ has no record cap, so it must self-prune stale orphans
    (no matching record) once they're old enough that no in-flight claim
    could still hold them.
    """
    import os
    import time

    store = InferenceTuningProfileStore(tmp_path)
    locks_dir = tmp_path / "locks"
    locks_dir.mkdir(parents=True)

    stale_orphan = locks_dir / ("a" * 24 + ".lock")
    stale_orphan.write_text("{}", encoding="utf-8")
    old_time = time.time() - 8 * 24 * 60 * 60
    os.utime(stale_orphan, (old_time, old_time))

    fresh_orphan = locks_dir / ("b" * 24 + ".lock")
    fresh_orphan.write_text("{}", encoding="utf-8")

    # A lock with a matching record must survive regardless of age.
    key = _key()
    profile = _validated_profile(key)
    store.save(profile)
    matching_lock = locks_dir / f"{key.digest}.lock"
    matching_lock.write_text("{}", encoding="utf-8")
    os.utime(matching_lock, (old_time, old_time))

    # Trigger the prune path via a second save (prune runs before each write).
    store.save(
        replace(profile, last_validation_unix_ns=profile.last_validation_unix_ns + 1)
    )

    assert not stale_orphan.exists()
    assert fresh_orphan.exists()
    assert matching_lock.exists()
