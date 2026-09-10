"""The containment cap must leave the pre-launch re-check room to succeed.

`prelaunch_check` re-reads available host memory and refuses if the immutable
cap would expose the protected reserve.  When the cap equals the admitted
usable capacity, that test is exactly tight: any downward drift in available
memory between admission and launch -- routine on an idle machine -- refuses
the run.
"""

from __future__ import annotations

from hydra_suite.detectkit.sidecars.supervisor import (
    HOST_CAP_HEADROOM_FRACTION,
    _containment_limits,
)
from hydra_suite.runtime.resource_budget import (
    AcceleratorKind,
    GiB,
    PhaseEstimate,
    ResourceObservation,
    ResourcePolicy,
    ResourceRequest,
    WorkLimits,
    evaluate_resource_request,
)


def _budget(available_gib: float, total_gib: float = 128.0):
    observation = ResourceObservation(
        total_host_bytes=int(total_gib * GiB),
        available_host_bytes=int(available_gib * GiB),
    )
    request = ResourceRequest(
        job_name="DetectKit semantic-preview",
        phases=(PhaseEstimate("model-operation", host_peak_bytes=7 * GiB),),
        limits=WorkLimits(batch_size=1, workers=0, prefetch_batches=0),
    )
    policy = ResourcePolicy()
    return evaluate_resource_request(request, observation, policy), observation, policy


def test_cap_leaves_headroom_for_a_fresh_availability_reading():
    budget, observation, policy = _budget(available_gib=70.5)
    _soft, hard, _ratio = _containment_limits(budget, observation, AcceleratorKind.MPS)

    assert budget.admitted
    assert hard < budget.usable_host_bytes

    reserve = max(
        policy.reserve_host_bytes,
        int(observation.total_host_bytes * policy.reserve_host_fraction),
    )

    # The pre-launch predicate, evaluated against availability that drifted
    # down since admission by less than the headroom.
    drift = int(budget.usable_host_bytes * HOST_CAP_HEADROOM_FRACTION * 0.5)
    live_available = observation.available_host_bytes - drift
    assert hard <= max(0, live_available - reserve)

    # A genuine collapse is still refused.
    collapsed = observation.available_host_bytes - 30 * GiB
    assert hard > max(0, collapsed - reserve)


def test_cap_still_admits_the_operation_estimate():
    budget, observation, _policy = _budget(available_gib=70.5)
    _soft, hard, _ratio = _containment_limits(budget, observation, AcceleratorKind.MPS)
    assert hard > budget.host_peak_bytes
