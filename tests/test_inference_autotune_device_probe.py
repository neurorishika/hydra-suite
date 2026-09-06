from __future__ import annotations

from hydra_suite.core.inference.autotune.device import probe_runtime_resources
from hydra_suite.runtime.resource_budget import AcceleratorKind


def test_cuda_probe_samples_multiple_times_and_detects_contention():
    calls = []

    def query(_command):
        calls.append(1)
        return (
            "GPU-abc, RTX 6000 Ada, 8.9, 49140, 42000, 37, 61, Active, "
            "575.57, 00000000:01:00.0\n"
        )

    result = probe_runtime_resources(
        AcceleratorKind.CUDA,
        query=query,
        sample_count=3,
        sample_interval_seconds=0,
        host_probe=lambda: (64 * 1024**3, 48 * 1024**3),
    )

    assert len(calls) == 3
    assert result.device_uuid == "GPU-abc"
    assert result.compute_capability == "8.9"
    assert result.observation.available_accelerator_bytes == 42000 * 1024**2
    assert result.contention_detected
    assert result.thermal_throttled
    assert result.pci_bus_id == "00000000:01:00.0"


def test_cuda_probe_uses_most_conservative_free_memory_sample():
    rows = iter(
        [
            "GPU-a, A, 8.0, 10000, 9500, 0, 40, Not Active, 1, bus\n",
            "GPU-a, A, 8.0, 10000, 9000, 0, 41, Not Active, 1, bus\n",
            "GPU-a, A, 8.0, 10000, 9300, 0, 42, Not Active, 1, bus\n",
        ]
    )
    result = probe_runtime_resources(
        "cuda",
        query=lambda _command: next(rows),
        sample_count=3,
        sample_interval_seconds=0,
        host_probe=lambda: (10_000, 9_000),
    )
    assert result.observation.available_accelerator_bytes == 9000 * 1024**2
    assert not result.contention_detected
    assert result.temperature_range_c == (40.0, 42.0)


def test_cpu_and_mps_never_claim_a_separate_memory_pool():
    for kind in (AcceleratorKind.CPU, AcceleratorKind.MPS):
        result = probe_runtime_resources(
            kind,
            sample_count=3,
            sample_interval_seconds=0,
            host_probe=lambda: (100, 80),
        )
        assert result.observation.accelerator_kind is kind
        assert result.observation.available_accelerator_bytes is None
        assert result.device_uuid == (
            "mps-unified" if kind is AcceleratorKind.MPS else "cpu"
        )
