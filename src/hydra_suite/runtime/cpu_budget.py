"""CPU budget authority: usable cores, budget resolution, fan-out planning.

Single source of truth for how many CPU threads TrackerKit may use. Qt-free;
imports nothing from app layers. See
docs/superpowers/specs/2026-10-12-cpu-oversubscription-design.md.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping, Optional

GPU_BOUND_MIN_THREADS = 1
CPU_BOUND_MIN_THREADS = 2
ENV_CPU_THREADS = "HYDRA_CPU_THREADS"
ENV_CPU_BUDGET = "HYDRA_CPU_BUDGET"
THREAD_ENV_VARS = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMBA_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "OPENCV_FFMPEG_THREADS",
)

_UNSET = object()


def _cgroup_quota_cores(cgroup_root: str, proc_cgroup: str) -> Optional[int]:
    """Smallest finite cgroup v2 cpu.max quota on this process's cgroup path."""
    try:
        rel = None
        for line in Path(proc_cgroup).read_text().splitlines():
            if line.startswith("0::"):
                rel = line[3:].strip()
                break
        if rel is None:
            return None
        root = Path(cgroup_root)
        node = root / rel.lstrip("/")
        best: Optional[int] = None
        while True:
            try:
                parts = (node / "cpu.max").read_text().split()
                if len(parts) == 2 and parts[0] != "max":
                    cores = max(1, math.ceil(int(parts[0]) / int(parts[1])))
                    best = cores if best is None else min(best, cores)
            except (OSError, ValueError, ZeroDivisionError):
                pass
            if node == root or root not in node.parents:
                break
            node = node.parent
        return best
    except OSError:
        return None


def usable_cores(
    *,
    affinity=_UNSET,
    cpu_count=_UNSET,
    cgroup_root: str = "/sys/fs/cgroup",
    proc_cgroup: str = "/proc/self/cgroup",
) -> int:
    """Cores this process may actually use (affinity, then cgroup v2 quota)."""
    if affinity is _UNSET:
        getter = getattr(os, "sched_getaffinity", None)
        affinity = getter(0) if getter is not None else None
    if cpu_count is _UNSET:
        cpu_count = os.cpu_count()
    n = len(affinity) if affinity else (cpu_count or 1)
    quota = _cgroup_quota_cores(cgroup_root, proc_cgroup)
    if quota is not None:
        n = min(n, quota)
    return max(1, int(n))


@dataclass(frozen=True)
class CpuBudget:
    cores: int
    usable: int
    source: str  # "flag" | "env" | "auto"
    warnings: tuple[str, ...] = field(default=())


def _positive_int(value) -> Optional[int]:
    try:
        n = int(str(value).strip())
    except (TypeError, ValueError):
        return None
    return n if n >= 1 else None


def resolve_cpu_budget(
    requested: Optional[int],
    *,
    env: Optional[Mapping[str, str]] = None,
    usable: Optional[int] = None,
) -> CpuBudget:
    """Total cores hydra may use: flag > HYDRA_CPU_BUDGET > usable cores."""
    env = os.environ if env is None else env
    usable = usable_cores() if usable is None else max(1, int(usable))
    warnings: list[str] = []
    cores, source = usable, "auto"
    if requested:
        cores, source = int(requested), "flag"
    else:
        raw = env.get(ENV_CPU_BUDGET)
        if raw is not None and raw.strip() != "":
            parsed = _positive_int(raw)
            if parsed is None:
                warnings.append(
                    f"Ignoring invalid {ENV_CPU_BUDGET}={raw!r} (need an integer >= 1)."
                )
            else:
                cores, source = parsed, "env"
    if cores > usable:
        warnings.append(
            f"CPU budget {cores} exceeds the {usable} usable core(s); using {usable}."
        )
        cores = usable
    return CpuBudget(
        cores=cores, usable=usable, source=source, warnings=tuple(warnings)
    )


@dataclass(frozen=True)
class CpuPlan:
    jobs: int
    threads_per_job: int
    clamped: bool
    warnings: tuple[str, ...] = field(default=())


def plan_cpu(
    *,
    jobs: int,
    n_videos: int,
    budget: CpuBudget,
    gpu_bound: bool,
    threads_per_job: Optional[int] = None,
) -> CpuPlan:
    """Split ``budget`` across ``jobs`` concurrent children (pure)."""
    jobs = max(1, int(jobs))
    warnings = list(budget.warnings)
    if threads_per_job is not None:
        tpj = max(1, int(threads_per_job))
        if jobs * tpj > budget.cores:
            warnings.append(
                f"{jobs} job(s) x {tpj} thread(s) = {jobs * tpj} exceeds the CPU "
                f"budget of {budget.cores} core(s) (explicit --threads-per-job)."
            )
        return CpuPlan(jobs, tpj, False, tuple(warnings))
    floor = GPU_BOUND_MIN_THREADS if gpu_bound else CPU_BOUND_MIN_THREADS
    max_jobs = max(1, budget.cores // floor)
    effective = min(jobs, max_jobs)
    clamped = effective < jobs
    if clamped:
        warnings.append(
            f"Requested {jobs} parallel job(s) but the CPU budget is {budget.cores} "
            f"core(s) ({budget.usable} usable); running {effective} at a time."
        )
    active = max(1, min(effective, int(n_videos) if n_videos else effective))
    tpj = max(1, budget.cores // active)
    return CpuPlan(effective, tpj, clamped, tuple(warnings))


def resolve_process_cap(
    cli_budget: Optional[int],
    *,
    env: Optional[Mapping[str, str]] = None,
    usable: Optional[int] = None,
) -> Optional[int]:
    """The thread cap for THIS process, or None (uncapped, byte-identical).

    HYDRA_CPU_THREADS (fan-out child) > --cpu-budget > HYDRA_CPU_BUDGET.
    """
    env = os.environ if env is None else env
    child = _positive_int(env.get(ENV_CPU_THREADS))
    if child is not None:
        return child
    if not cli_budget and _positive_int(env.get(ENV_CPU_BUDGET)) is None:
        return None
    return resolve_cpu_budget(cli_budget, env=env, usable=usable).cores
