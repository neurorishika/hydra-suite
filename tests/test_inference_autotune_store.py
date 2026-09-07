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
from hydra_suite.core.inference.autotune.models import ProfileState
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


def test_contention_detected_prevents_the_negative_cache_lockout(tmp_path):
    """A transient GPU neighbour must not buy a 24-hour lockout: if the
    request carries ``contention_detected=True``, an incomplete calibration
    must not be persisted as a negative-cache record at all.
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
