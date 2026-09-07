"""Negative-caching tests for the inference autotune coordinator/store (S5).

A project that cannot complete calibration (budget exhausted, timeout,
baseline measurement never came back) must not burn the full tuning budget
on *every* run forever -- the coordinator has to remember the failure and
skip straight to a fallback overlay until a retry TTL elapses. See
`docs/superpowers/specs/2026-09-06-trackerkit-inference-autotuner-design.md`
around the `ProfileState` enumeration for the amended state description.
"""

from __future__ import annotations

from hydra_suite.core.inference.autotune.coordinator import (
    AutotuneCoordinator,
    AutotuneRequest,
)
from hydra_suite.core.inference.autotune.models import (
    CandidateEvidence,
    EquivalenceVerdict,
    InferenceTuningProfile,
    ProfileState,
)
from hydra_suite.core.inference.autotune.search import TrialObservation
from hydra_suite.core.inference.autotune.store import InferenceTuningProfileStore

from .autotune_helpers import _key, _planner, _settings


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
