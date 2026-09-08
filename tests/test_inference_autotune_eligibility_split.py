"""`eligible` gates measuring; `allow_cached_reuse` gates applying. They differ."""

import pytest

from hydra_suite.core.inference.autotune.integration import (
    build_tracking_autotune_request,
)
from hydra_suite.runtime.resource_budget import AcceleratorKind
from tests.autotune_helpers import make_request_inputs


@pytest.mark.parametrize(
    "kind", [AcceleratorKind.MPS, AcceleratorKind.CPU, AcceleratorKind.CUDA]
)
def test_apply_is_permitted_on_every_accelerator(kind):
    """A validated profile must be applicable on cpu/mps/cuda alike. The gain and
    equivalence gates -- not the device name -- establish that it is safe."""
    request = build_tracking_autotune_request(**make_request_inputs(kind=kind))
    assert request.allow_cached_reuse is True


def test_realtime_may_apply_but_may_not_calibrate():
    request = build_tracking_autotune_request(
        **make_request_inputs(execution_mode="realtime")
    )
    assert request.eligible is False
    assert request.allow_cached_reuse is True


def test_cache_replay_may_apply_but_may_not_calibrate():
    """A backward pass IS a cache_replay. It must apply the forward vector or the
    detection cache key will not match -- the measured courtship abort."""
    request = build_tracking_autotune_request(
        **make_request_inputs(execution_mode="cache_replay")
    )
    assert request.eligible is False
    assert request.allow_cached_reuse is True


def test_failed_baseline_admission_blocks_both():
    """If the baseline vector does not fit in memory, applying it is unsafe too."""
    request = build_tracking_autotune_request(
        **make_request_inputs(available_accelerator_bytes=1)
    )
    assert request.eligible is False
    assert request.allow_cached_reuse is False
