"""Intent replaces the off/record/automatic mode vocabulary."""

import pytest

from hydra_suite.core.inference.autotune.models import ProfileState
from tests.autotune_helpers import (
    make_coordinator,
    make_incomplete_profile,
    make_request,
)


def test_lookup_without_executor_reports_unavailable_on_a_miss():
    coordinator, store = make_coordinator(trial_executor=None)
    result = coordinator.resolve(make_request(mode="lookup"))
    assert result.overlay.status == "unavailable"
    assert store.saves == []
    assert store.claims == []


def test_lookup_serves_a_hit_even_when_ineligible():
    """Eligibility gates measuring only. A backward pass is ineligible and must
    still receive the forward pass's vector."""
    coordinator, _ = make_coordinator(
        trial_executor=None, cached_state=ProfileState.VALIDATED
    )
    result = coordinator.resolve(make_request(mode="lookup", eligible=False))
    assert result.overlay.status == "cache_hit"


def test_explicit_calibrate_bypasses_the_24h_negative_cache():
    """A human clicked the button. The negative cache exists to stop a RUN
    silently re-paying a failed budget, which does not apply here."""
    coordinator, _ = make_coordinator(cached_profile=make_incomplete_profile())
    result = coordinator.resolve(make_request(mode="calibrate"))
    assert result.overlay.status != "deferred_due_to_prior_failure"


def test_lookup_still_honours_the_negative_cache():
    coordinator, _ = make_coordinator(cached_profile=make_incomplete_profile())
    result = coordinator.resolve(make_request(mode="lookup"))
    assert result.overlay.status == "deferred_due_to_prior_failure"


def test_record_mode_is_gone():
    with pytest.raises((ValueError, KeyError)):
        make_request(mode="record")
