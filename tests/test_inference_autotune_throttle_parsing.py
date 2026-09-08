"""Throttle-reason parsing, pinned against BOTH nvidia-smi representations.

Regression cover for the CUDA gate finding: `device.py` used to string-match a
small set of "inactive" words. Driver 580.x answers
`clocks_throttle_reasons.active` with a hex bitmask, so `0x0000000000000000`
(nothing active) and `0x0000000000000001` (GpuIdle -- the *opposite* of
throttling) both fell through as "throttled". Measured on two idle RTX 4090s:
`probe_runtime_resources("cuda").thermal_throttled` was True on a completely
idle GPU, and `automatic` tuning deferred forever with
"accelerator is thermally throttled".

The point of these tests is that neither representation may be pinned alone,
and that an unparseable sensor reading must fail toward NOT throttled: an
unreadable sensor silently disabling the whole feature is exactly the bug.
"""

from __future__ import annotations

import pytest

from hydra_suite.core.inference.autotune.device import (
    probe_runtime_resources,
    throttle_reason_is_active,
)
from hydra_suite.runtime.resource_budget import AcceleratorKind


@pytest.mark.parametrize(
    "raw",
    [
        # --- hex bitmask form (driver 580.x) ---
        "0x0000000000000000",  # nothing active
        "0x0000000000000001",  # GpuIdle -- idle, not throttled
        "0x0000000000000002",  # ApplicationsClocksSetting -- deliberate policy
        "0x0000000000000100",  # DisplayClockSetting
        "0x0000000000000102",  # both benign bits together
        "0x1",
        "0",
        # --- decimal form ---
        "3",  # GpuIdle | ApplicationsClocksSetting
        # --- worded form (older drivers / per-reason queries) ---
        "Not Active",
        "not active",
        "",
        "N/A",
    ],
)
def test_benign_throttle_readings_are_not_throttling(raw):
    assert throttle_reason_is_active(raw) is False


@pytest.mark.parametrize(
    "raw",
    [
        "0x0000000000000004",  # SwPowerCap
        "0x0000000000000008",  # HwSlowdown
        "0x0000000000000020",  # SwThermalSlowdown
        "0x0000000000000040",  # HwThermalSlowdown
        "0x0000000000000080",  # HwPowerBrakeSlowdown
        "0x0000000000000021",  # SwThermalSlowdown alongside GpuIdle
        "Active",
        "HW Thermal Slowdown",
        "SW Power Cap",
    ],
)
def test_real_throttle_readings_are_throttling(raw):
    assert throttle_reason_is_active(raw) is True


def test_unparseable_reading_fails_toward_not_throttled(caplog):
    """An unreadable sensor must not permanently disable calibration."""
    with caplog.at_level("WARNING"):
        assert throttle_reason_is_active("banana surprise") is False
    assert any("Unrecognized" in record.message for record in caplog.records)


def _row(reason: str) -> str:
    return f"GPU-a, RTX 4090, 8.9, 24564, 22544, 0, 34, {reason}, 580.76.05, bus\n"


@pytest.mark.parametrize(
    "reason, expected",
    [
        ("0x0000000000000000", False),  # courtship, idle
        ("0x0000000000000001", False),  # firebrat, idle (GpuIdle bit)
        ("Not Active", False),  # older driver, idle
        ("0x0000000000000040", True),  # genuinely thermally throttled
        ("HW Thermal Slowdown", True),  # older driver, throttled
    ],
)
def test_probe_reports_idle_cuda_hosts_as_not_throttled(reason, expected):
    """End-to-end through the probe, on both boxes' real nvidia-smi output."""
    result = probe_runtime_resources(
        AcceleratorKind.CUDA,
        query=lambda _command: _row(reason),
        sample_count=2,
        sample_interval_seconds=0,
        host_probe=lambda: (128 * 1024**3, 100 * 1024**3),
    )
    assert result.thermal_throttled is expected
    # The idle rows must also stay free of a contention verdict, or the
    # eligibility gate would defer for the other reason instead.
    if not expected:
        assert not result.contention_detected
