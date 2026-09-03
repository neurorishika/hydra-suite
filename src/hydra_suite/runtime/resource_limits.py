"""Operating-system containment adapters for high-memory child processes.

Linux cgroup v2/systemd enforcement is preferred because it constrains resident
memory for a complete process tree.  ``RLIMIT_AS`` is a POSIX fallback: it
limits virtual address space, can conflict with CUDA's large virtual mappings,
and does not constrain discrete GPU VRAM.  A parent watchdog remains required
for diagnostics, graceful cancellation, and system-reserve enforcement.
"""

from __future__ import annotations

import json
import os
import platform
import shutil
import subprocess
import sys
import uuid
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from types import MappingProxyType
from typing import Mapping, Optional, Sequence

from .resource_budget import AcceleratorKind


class LimitBackend(str, Enum):
    """Host-memory enforcement mechanism applied to a protected launch."""

    SYSTEMD_CGROUP = "systemd-cgroup-v2"
    RLIMIT_AS = "rlimit-as"
    WATCHDOG_ONLY = "watchdog-only"


@dataclass(frozen=True)
class ProcessMemoryLimits:
    """Soft/hard host limits plus an optional MPS allocator guard."""

    soft_host_bytes: int
    hard_host_bytes: int
    mps_high_watermark_ratio: Optional[float] = None
    max_processes: int = 512

    def __post_init__(self) -> None:
        if self.soft_host_bytes <= 0 or self.hard_host_bytes <= 0:
            raise ValueError("host memory limits must be positive")
        if self.soft_host_bytes > self.hard_host_bytes:
            raise ValueError("soft host limit cannot exceed hard host limit")
        if self.max_processes < 1:
            raise ValueError("maximum process count must be positive")
        if self.mps_high_watermark_ratio is not None and not (
            0.0 < self.mps_high_watermark_ratio <= 2.0
        ):
            raise ValueError("MPS high-watermark ratio must be in (0, 2]")


@dataclass(frozen=True)
class LimitedLaunch:
    """Immutable child command, environment, and authoritative memory limits."""

    command: tuple[str, ...]
    environment: Mapping[str, str]
    backend: LimitBackend
    limits: ProcessMemoryLimits
    accelerator_kind: AcceleratorKind
    systemd_unit: Optional[str] = None
    limitations: tuple[str, ...] = ()


@dataclass(frozen=True)
class CgroupEvidence:
    """Bounded post-exit observations from a transient systemd scope."""

    unit: str
    available: bool = True
    result: Optional[str] = None
    oom_killed: bool = False
    memory_peak_bytes: Optional[int] = None
    raw_properties: Mapping[str, str] | None = None
    error: Optional[str] = None


def systemd_user_scope_available(
    *,
    system: Optional[str] = None,
    environ: Optional[Mapping[str, str]] = None,
) -> bool:
    """Conservatively detect whether a user cgroup-v2 scope may be launched.

    This is intentionally side-effect free.  A launch can still fail if the
    user manager rejects delegation; callers must report that launch failure
    rather than silently running the workload without its requested cap.
    """
    system = system or platform.system()
    environ = os.environ if environ is None else environ
    return bool(
        system == "Linux"
        and Path("/sys/fs/cgroup/cgroup.controllers").exists()
        and shutil.which("systemd-run")
        and (environ.get("DBUS_SESSION_BUS_ADDRESS") or environ.get("XDG_RUNTIME_DIR"))
    )


def select_limit_backend(
    *,
    system: Optional[str] = None,
    systemd_available: Optional[bool] = None,
) -> LimitBackend:
    """Choose the strongest supported host-memory boundary for this platform."""

    system = system or platform.system()
    if system == "Linux" and (
        systemd_user_scope_available(system=system)
        if systemd_available is None
        else systemd_available
    ):
        return LimitBackend.SYSTEMD_CGROUP
    # Darwin exposes RLIMIT_AS constants but rejects setrlimit(RLIMIT_AS, ...)
    # with EINVAL.  Claiming it as a hard boundary there would be dangerous.
    if system == "Linux":
        return LimitBackend.RLIMIT_AS
    return LimitBackend.WATCHDOG_ONLY


def build_limited_launch(
    command: Sequence[str],
    limits: ProcessMemoryLimits,
    *,
    backend: Optional[LimitBackend] = None,
    environment: Optional[Mapping[str, str]] = None,
    python_executable: Optional[str] = None,
    systemd_unit: Optional[str] = None,
    accelerator_kind: AcceleratorKind | str = AcceleratorKind.CPU,
) -> LimitedLaunch:
    """Wrap ``command`` so limits are applied before accelerator imports."""
    if not command:
        raise ValueError("child command must not be empty")
    accelerator_kind = AcceleratorKind(accelerator_kind)
    if (
        accelerator_kind is AcceleratorKind.MPS
        and limits.mps_high_watermark_ratio is None
    ):
        raise ValueError("MPS jobs require an explicit allocator high-watermark ratio")
    selected = backend or select_limit_backend()
    child_env = dict(os.environ if environment is None else environment)
    bootstrap = [
        python_executable or sys.executable,
        "-m",
        "hydra_suite.runtime.child_bootstrap",
    ]
    if selected is LimitBackend.RLIMIT_AS:
        bootstrap.extend(["--address-space-bytes", str(limits.hard_host_bytes)])
    if limits.mps_high_watermark_ratio is not None:
        bootstrap.extend(
            [
                "--mps-high-watermark-ratio",
                str(limits.mps_high_watermark_ratio),
            ]
        )
    bootstrap.extend(["--", *map(str, command)])

    limitations: list[str] = []
    unit = None
    if selected is LimitBackend.SYSTEMD_CGROUP:
        unit = systemd_unit or f"hydra-job-{uuid.uuid4().hex}.scope"
        wrapped = [
            "systemd-run",
            "--user",
            "--scope",
            "--quiet",
            "--unit",
            unit,
            f"--property=MemoryHigh={limits.soft_host_bytes}",
            f"--property=MemoryMax={limits.hard_host_bytes}",
            "--property=MemorySwapMax=0",
            f"--property=TasksMax={limits.max_processes}",
            "--",
            *bootstrap,
        ]
    else:
        wrapped = bootstrap
        if selected is LimitBackend.RLIMIT_AS:
            limitations.append(
                "RLIMIT_AS constrains virtual address space, may conflict with CUDA "
                "reservations, and does not cap discrete GPU VRAM"
            )
        else:
            limitations.append(
                "No kernel memory controller is available; only the parent RSS and "
                "system-pressure watchdog enforces host limits"
            )
    return LimitedLaunch(
        command=tuple(wrapped),
        environment=MappingProxyType(child_env),
        backend=selected,
        limits=limits,
        accelerator_kind=accelerator_kind,
        systemd_unit=unit,
        limitations=tuple(limitations),
    )


def apply_child_limits(
    *,
    address_space_bytes: Optional[int],
    mps_high_watermark_ratio: Optional[float],
) -> None:
    """Apply limits in a minimal child before replacing it with the workload."""
    if address_space_bytes is not None:
        if address_space_bytes <= 0:
            raise ValueError("address-space limit must be positive")
        if os.name != "posix":
            raise RuntimeError("RLIMIT_AS is unavailable on this platform")
        import resource

        current_soft, current_hard = resource.getrlimit(resource.RLIMIT_AS)
        requested = address_space_bytes
        if current_hard != resource.RLIM_INFINITY:
            requested = min(requested, current_hard)
        try:
            resource.setrlimit(resource.RLIMIT_AS, (requested, requested))
        except (OSError, ValueError) as exc:
            raise RuntimeError(
                "RLIMIT_AS was selected but this operating system does not enforce it"
            ) from exc
    if mps_high_watermark_ratio is not None:
        if not 0.0 < mps_high_watermark_ratio <= 2.0:
            raise ValueError("MPS high-watermark ratio must be in (0, 2]")
        os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = str(mps_high_watermark_ratio)


def probe_systemd_cgroup_evidence(
    unit: str, *, timeout_seconds: float = 3.0
) -> CgroupEvidence:
    """Read transient-unit evidence used to distinguish cgroup OOM kills."""
    if timeout_seconds <= 0:
        raise ValueError("systemd evidence timeout must be positive")
    try:
        completed = subprocess.run(
            [
                "systemctl",
                "--user",
                "show",
                unit,
                "--property=Result",
                "--property=ExecMainStatus",
                "--property=MemoryPeak",
                "--property=OOMKilled",
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
        )
    except subprocess.TimeoutExpired:
        return CgroupEvidence(
            unit=unit,
            available=False,
            error=f"systemd evidence query timed out after {timeout_seconds:g}s",
        )
    except OSError as exc:
        return CgroupEvidence(unit=unit, available=False, error=str(exc))
    if completed.returncode != 0:
        detail = completed.stderr.strip() or f"systemctl exited {completed.returncode}"
        return CgroupEvidence(unit=unit, available=False, error=detail)
    properties: dict[str, str] = {}
    for line in completed.stdout.splitlines():
        key, separator, value = line.partition("=")
        if separator:
            properties[key] = value
    result = properties.get("Result")
    oom_killed = properties.get("OOMKilled", "").lower() in {"yes", "true", "1"}
    oom_killed = oom_killed or result in {"oom-kill", "oom-killed"}
    memory_peak = _optional_int(properties.get("MemoryPeak"))
    return CgroupEvidence(
        unit=unit,
        result=result,
        oom_killed=oom_killed,
        memory_peak_bytes=memory_peak,
        raw_properties=properties,
    )


def signal_systemd_scope(
    unit: str, signum: int, *, timeout_seconds: float = 3.0
) -> bool:
    """Ask systemd to signal every process still owned by one transient scope."""
    if timeout_seconds <= 0:
        raise ValueError("systemd signal timeout must be positive")
    try:
        completed = subprocess.run(
            [
                "systemctl",
                "--user",
                "kill",
                f"--signal={int(signum)}",
                "--kill-whom=all",
                unit,
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    return completed.returncode == 0


def serialize_limit_diagnostic(launch: LimitedLaunch) -> str:
    """Return stable JSON suitable for a run manifest or support report."""
    return json.dumps(
        {
            "backend": launch.backend.value,
            "systemd_unit": launch.systemd_unit,
            "limitations": list(launch.limitations),
            "soft_host_bytes": launch.limits.soft_host_bytes,
            "hard_host_bytes": launch.limits.hard_host_bytes,
            "mps_high_watermark_ratio": launch.limits.mps_high_watermark_ratio,
            "max_processes": launch.limits.max_processes,
            "accelerator_kind": launch.accelerator_kind.value,
        },
        sort_keys=True,
    )


def _optional_int(value: Optional[str]) -> Optional[int]:
    if value is None:
        return None
    try:
        return int(value)
    except ValueError:
        return None
