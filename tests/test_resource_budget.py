from __future__ import annotations

import pytest

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


def _observation(kind=AcceleratorKind.CPU):
    kwargs = {}
    if kind is AcceleratorKind.CUDA:
        kwargs.update(
            total_accelerator_bytes=24 * GiB,
            available_accelerator_bytes=20 * GiB,
        )
    return ResourceObservation(
        total_host_bytes=64 * GiB,
        available_host_bytes=32 * GiB,
        accelerator_kind=kind,
        **kwargs,
    )


def test_host_reserve_uses_larger_absolute_or_fractional_floor():
    request = ResourceRequest(
        "test job", (PhaseEstimate("train", host_peak_bytes=21 * GiB),)
    )

    budget = evaluate_resource_request(
        request,
        _observation(),
        ResourcePolicy(reserve_host_bytes=4 * GiB, reserve_host_fraction=0.20),
    )

    assert budget.reserved_host_bytes == int(64 * GiB * 0.20)
    assert budget.usable_host_bytes == 32 * GiB - budget.reserved_host_bytes
    assert budget.admitted is False
    assert "reserving" in budget.refusals[0]


def test_phases_are_alternatives_and_dominant_allocations_are_reported():
    request = ResourceRequest(
        "training",
        (
            PhaseEstimate(
                "train",
                host_steady_bytes=5 * GiB,
                host_peak_bytes=8 * GiB,
                dominant_allocations=(("images", 2 * GiB), ("model", 6 * GiB)),
            ),
            PhaseEstimate(
                "publish", host_steady_bytes=7 * GiB, host_peak_bytes=12 * GiB
            ),
        ),
        limits=WorkLimits(batch_size=2, workers=1, prefetch_batches=1),
    )

    budget = evaluate_resource_request(request, _observation())

    assert budget.admitted
    assert budget.host_steady_bytes == 7 * GiB
    assert budget.host_peak_bytes == 12 * GiB
    assert budget.dominant_phase == "publish"
    assert budget.limits.batch_size == 2


def test_cuda_requires_a_measured_device_pool_and_applies_safety_fraction():
    request = ResourceRequest(
        "cuda training",
        (PhaseEstimate("train", accelerator_peak_bytes=18 * GiB),),
    )
    budget = evaluate_resource_request(
        request,
        _observation(AcceleratorKind.CUDA),
        ResourcePolicy(accelerator_safety_fraction=0.85),
    )

    assert budget.usable_accelerator_bytes == 17 * GiB
    assert not budget.admitted
    assert "accelerator peak" in budget.refusals[0]

    unknown = ResourceObservation(
        total_host_bytes=64 * GiB,
        available_host_bytes=32 * GiB,
        accelerator_kind=AcceleratorKind.CUDA,
    )
    assert not evaluate_resource_request(request, unknown).admitted


def test_mps_counts_accelerator_estimate_once_in_unified_host_pool():
    request = ResourceRequest(
        "mps training",
        (
            PhaseEstimate(
                "train",
                host_steady_bytes=4 * GiB,
                host_peak_bytes=6 * GiB,
                accelerator_steady_bytes=5 * GiB,
                accelerator_peak_bytes=10 * GiB,
            ),
        ),
    )
    budget = evaluate_resource_request(
        request,
        _observation(AcceleratorKind.MPS),
        ResourcePolicy(reserve_host_bytes=8 * GiB, reserve_host_fraction=0.0),
    )

    assert budget.host_steady_bytes == 9 * GiB
    assert budget.host_peak_bytes == 16 * GiB
    assert budget.usable_accelerator_bytes is None
    assert budget.admitted


def test_mps_rejects_a_second_fake_device_pool():
    with pytest.raises(ValueError, match="unified"):
        ResourceObservation(
            total_host_bytes=64 * GiB,
            available_host_bytes=32 * GiB,
            accelerator_kind=AcceleratorKind.MPS,
            total_accelerator_bytes=64 * GiB,
            available_accelerator_bytes=32 * GiB,
        )
