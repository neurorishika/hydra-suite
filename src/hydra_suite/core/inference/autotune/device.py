"""Lightweight sampled resource and physical-device probing.

The probe runs before model loading.  CUDA telemetry comes from ``nvidia-smi``
and is intentionally sampled several times so a single idle-looking instant
cannot authorize disruptive calibration.
"""

from __future__ import annotations

import os
import platform
import subprocess
import time
from dataclasses import dataclass
from typing import Callable

from hydra_suite.runtime.resource_budget import AcceleratorKind, ResourceObservation

MiB = 1024**2


@dataclass(frozen=True, slots=True)
class RuntimeResourceProbe:
    observation: ResourceObservation
    device_uuid: str
    device_model: str
    compute_capability: str
    pci_bus_id: str | None
    driver_version: str
    contention_detected: bool
    thermal_throttled: bool
    utilization_samples: tuple[float, ...] = ()
    free_memory_samples_bytes: tuple[int, ...] = ()
    temperature_range_c: tuple[float, float] | None = None


def _host_memory() -> tuple[int, int]:
    import psutil

    memory = psutil.virtual_memory()
    return int(memory.total), int(memory.available)


def _query_nvidia(command: tuple[str, ...]) -> str:
    return subprocess.check_output(
        command,
        text=True,
        timeout=2.0,
        stderr=subprocess.DEVNULL,
    )


def _kind(value: AcceleratorKind | str) -> AcceleratorKind:
    return value if isinstance(value, AcceleratorKind) else AcceleratorKind(value)


def cuda_used_memory_bytes(device_uuid: str) -> int:
    """Sample current physical-device memory for the sidecar watchdog."""

    raw = _query_nvidia(
        (
            "nvidia-smi",
            "--query-gpu=memory.used",
            "--format=csv,noheader,nounits",
            f"--id={device_uuid}",
        )
    )
    line = next((item.strip() for item in raw.splitlines() if item.strip()), "")
    return int(float(line) * MiB)


def probe_runtime_resources(
    accelerator_kind: AcceleratorKind | str,
    *,
    query: Callable[[tuple[str, ...]], str] = _query_nvidia,
    sample_count: int = 3,
    sample_interval_seconds: float = 0.1,
    host_probe: Callable[[], tuple[int, int]] = _host_memory,
) -> RuntimeResourceProbe:
    """Return conservative current resources and stable accelerator identity."""

    kind = _kind(accelerator_kind)
    if not 2 <= int(sample_count) <= 8:
        raise ValueError("resource probe sample_count must be between two and eight")
    if not 0 <= sample_interval_seconds <= 2:
        raise ValueError("resource probe interval must be between zero and two seconds")
    total_host, available_host = host_probe()
    if kind is not AcceleratorKind.CUDA:
        name = (
            f"Apple Metal ({platform.machine()})"
            if kind is AcceleratorKind.MPS
            else platform.processor() or "CPU"
        )
        observation = ResourceObservation(
            total_host_bytes=int(total_host),
            available_host_bytes=int(available_host),
            accelerator_kind=kind,
            accelerator_name=name,
        )
        return RuntimeResourceProbe(
            observation=observation,
            device_uuid="mps-unified" if kind is AcceleratorKind.MPS else "cpu",
            device_model=name,
            compute_capability="unified" if kind is AcceleratorKind.MPS else "none",
            pci_bus_id=None,
            driver_version=platform.release(),
            contention_detected=False,
            thermal_throttled=False,
        )

    # nvidia-smi always enumerates by PCI bus order; CUDA itself defaults to
    # FASTEST_FIRST. Without CUDA_DEVICE_ORDER=PCI_BUS_ID, an ordinal selector
    # (here or in whatever later initializes a CUDA context) can address a
    # different physical GPU than nvidia-smi's same ordinal. This probe runs
    # before any model/CUDA-context load (module docstring), so setting the
    # order here -- and only if unset, respecting an explicit override --
    # makes every ordinal used by this process consistent between nvidia-smi
    # and CUDA for the remainder of the process's life (S7).
    os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",", 1)[0].strip()
    device_selector = visible if visible and visible != "-1" else "0"
    command = (
        "nvidia-smi",
        "--query-gpu=uuid,name,compute_cap,memory.total,memory.free,"
        "utilization.gpu,temperature.gpu,clocks_throttle_reasons.active,"
        "driver_version,pci.bus_id",
        "--format=csv,noheader,nounits",
        f"--id={device_selector}",
    )
    samples = []
    for index in range(int(sample_count)):
        raw = query(command)
        line = next((item for item in raw.splitlines() if item.strip()), "")
        fields = tuple(part.strip() for part in line.split(","))
        if len(fields) != 10:
            raise RuntimeError("nvidia-smi returned an incomplete device sample")
        samples.append(fields)
        if index + 1 < sample_count and sample_interval_seconds:
            time.sleep(sample_interval_seconds)
    identity = samples[0][:4] + samples[0][8:]
    if any(sample[:4] + sample[8:] != identity for sample in samples[1:]):
        raise RuntimeError("physical CUDA device identity changed during admission")
    uuid, name, capability, total_mib = samples[0][:4]
    free = tuple(int(float(sample[4]) * MiB) for sample in samples)
    utilization = tuple(float(sample[5]) for sample in samples)
    temperature = tuple(float(sample[6]) for sample in samples)
    throttle = tuple(sample[7].lower() for sample in samples)
    total_bytes = int(float(total_mib) * MiB)
    # Before candidate models load, sustained utilization or substantial used
    # memory is evidence of another accelerator workload.  Both tests are
    # conservative because deferring tuning is always safe.
    used_peak = max(total_bytes - value for value in free)
    contention = max(utilization) >= 10.0 or used_peak >= max(
        1024**3, total_bytes // 10
    )
    inactive_tokens = {"not active", "0", "false", "no", "none"}
    throttled = any(value not in inactive_tokens for value in throttle)
    observation = ResourceObservation(
        total_host_bytes=int(total_host),
        available_host_bytes=int(available_host),
        accelerator_kind=kind,
        accelerator_name=name,
        total_accelerator_bytes=total_bytes,
        available_accelerator_bytes=min(free),
    )
    return RuntimeResourceProbe(
        observation=observation,
        device_uuid=uuid,
        device_model=name,
        compute_capability=capability,
        pci_bus_id=samples[0][9],
        driver_version=samples[0][8],
        contention_detected=contention,
        thermal_throttled=throttled,
        utilization_samples=utilization,
        free_memory_samples_bytes=free,
        temperature_range_c=(min(temperature), max(temperature)),
    )
