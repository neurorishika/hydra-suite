"""CPU budget authority: usable cores, budget resolution, fan-out planning.

Single source of truth for how many CPU threads TrackerKit may use. Qt-free;
imports nothing from app layers. See
docs/superpowers/specs/2026-10-12-cpu-oversubscription-design.md.
"""

from __future__ import annotations

import logging
import math
import os
import sys
import threading
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
    except (OSError, ValueError):
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


def positive_int(value) -> Optional[int]:
    """``value`` as an int >= 1, or None (non-int, blank, zero, negative)."""
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
    flag = None
    if requested:
        flag = positive_int(requested)
        if flag is None:
            warnings.append(
                f"Ignoring invalid CPU budget {requested!r} (need an integer >= 1)."
            )
    if flag is not None:
        cores, source = flag, "flag"
    else:
        raw = env.get(ENV_CPU_BUDGET)
        if raw is not None and raw.strip() != "":
            parsed = positive_int(raw)
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


def inherited_cpu_threads(
    *,
    env: Optional[Mapping[str, str]] = None,
    logger: Optional[logging.Logger] = None,
) -> Optional[int]:
    """A valid ``HYDRA_CPU_THREADS`` from ``env`` (default os.environ), else None.

    A non-blank invalid value is ignored with a warning (never a crash).
    """
    env = os.environ if env is None else env
    raw = env.get(ENV_CPU_THREADS)
    value = positive_int(raw)
    if value is None and raw is not None and raw.strip() != "":
        (logger or logging.getLogger(__name__)).warning(
            "CPU budget: ignoring invalid %s=%r (need an integer >= 1).",
            ENV_CPU_THREADS,
            raw,
        )
    return value


def single_run_cpu_cap(
    cpu_budget: Optional[int],
    *,
    env: Optional[Mapping[str, str]] = None,
    logger: Optional[logging.Logger] = None,
    usable: Optional[int] = None,
) -> Optional[int]:
    """The thread cap for THIS process's in-process (non-fan-out) run, or None.

    The single authority for an in-process cap (CLI track/calibrate, GUI):

    - ``HYDRA_CPU_THREADS`` (handed down by a fan-out parent or exported in the
      shell) is a ceiling: a valid explicit ``cpu_budget`` can only LOWER it
      (``min``); an absent/invalid budget leaves it as is, and
      ``HYDRA_CPU_BUDGET`` never overrides it (the box-wide budget does not
      apply to a child).
    - otherwise ``cpu_budget`` > ``HYDRA_CPU_BUDGET``; with neither (or both
      invalid) -> None, i.e. uncapped (plain-path invariant).

    Warnings (an invalid ``HYDRA_CPU_BUDGET``, a budget above the usable
    cores) are logged, not silently ignored or clamped.
    """
    env = os.environ if env is None else env
    log = logger or logging.getLogger(__name__)
    inherited = inherited_cpu_threads(env=env, logger=log)
    if inherited is not None:
        explicit = positive_int(cpu_budget) if cpu_budget else None
        if explicit is not None and explicit < inherited:
            log.info(
                "CPU budget: %s=%d is a ceiling; using the explicit budget %d.",
                ENV_CPU_THREADS,
                inherited,
                explicit,
            )
            return explicit
        return inherited
    raw_budget = env.get(ENV_CPU_BUDGET)
    if cpu_budget is None and not (raw_budget or "").strip():
        return None  # no budget source: uncapped (plain-path invariant)
    budget = resolve_cpu_budget(cpu_budget, env=env, usable=usable)
    for warning in budget.warnings:
        log.warning("CPU budget: %s", warning)
    # "auto" means every source was absent or invalid: stay uncapped.
    return None if budget.source == "auto" else budget.cores


# --------------------------------------------------------------------------
# Process-level cap (applied) + per-thread binder
# --------------------------------------------------------------------------

_logger = logging.getLogger(__name__)
_cap_lock = threading.Lock()
_cap: Optional[int] = None
_env_originals: Optional[dict] = None  # recorded on the first non-None set
_tp_originals: Optional[list] = None  # [(lib_controller, original threads)]
_torch_original: Optional[int] = None  # sticky once per process (see below)
_thread_state = threading.local()  # per-thread numba count before binding

_NUMBA_ENV = "NUMBA_NUM_THREADS"


def capped(value: int, cap: Optional[int]) -> int:
    """``value`` limited by ``cap`` (unchanged when cap is None)."""
    return value if cap is None else max(1, min(int(value), int(cap)))


def current_cpu_cap() -> Optional[int]:
    """The active process-wide CPU thread cap, or None when uncapped."""
    return _cap


def apriltag_thread_count(params) -> int:
    """AprilTag detector threads: ``APRILTAG_THREADS`` (default 4), capped."""
    return capped(int(params.get("APRILTAG_THREADS", 4)), current_cpu_cap())


def _env_min(original: Optional[str], n: int) -> str:
    parsed = positive_int(original)
    return str(n if parsed is None else min(parsed, n))


def _restore_threadpools() -> None:
    global _tp_originals
    if _tp_originals is None:
        return
    for lib, original in _tp_originals:
        try:
            lib.set_num_threads(original)
        except Exception:
            _logger.debug("threadpool restore failed", exc_info=True)
    _tp_originals = None


def _limit_threadpools(n: int) -> None:
    """Lower each native pool to ``min(its current size, n)``; never raise one.

    Called after ``_restore_threadpools`` so "current" is the pre-cap size and
    a re-cap 2 -> 6 is computed from the originals. OpenMP limits apply to the
    calling thread only (C4).
    """
    global _tp_originals
    try:
        from threadpoolctl import ThreadpoolController

        originals = []
        for lib in ThreadpoolController().lib_controllers:
            original = int(lib.num_threads)
            originals.append((lib, original))
            if original > n:
                lib.set_num_threads(n)
        _tp_originals = originals
    except Exception:  # threadpoolctl missing/unsupported lib: env applies
        _logger.debug("threadpoolctl limit failed", exc_info=True)


def _set_torch_threads(value: int) -> None:
    torch_mod = sys.modules.get("torch")
    if torch_mod is None:
        return
    try:
        torch_mod.set_num_threads(int(value))
    except Exception:
        _logger.debug("torch.set_num_threads failed", exc_info=True)


def set_process_cpu_cap(n: Optional[int]) -> None:
    """Set (or lift with None) the process-wide CPU thread cap. Idempotent.

    Must be called from ONE controlling thread (the GUI orchestrator or the
    CLI main thread): torch's thread count and the OpenMP limits are the
    calling thread's, so concurrent setters on different threads would
    corrupt what gets restored.

    Caps the thread env vars (``min(existing, n)``, never raised), lowers
    native BLAS/OpenMP pools via threadpoolctl (never raises a smaller pool)
    and torch's intra-op pool. A None when no cap was ever set touches
    nothing (plain-path invariant). Each call is computed from the ORIGINAL
    state, so re-capping 2 -> 6 gives 6.

    ``NUMBA_NUM_THREADS`` is only touched while numba is NOT imported: numba
    re-reads its env on every compile and raises once its pool is launched if
    the value changed. In-process numba is capped per thread by
    :func:`bind_thread_cpu_cap`; children inherit the env from their builder.
    """
    global _cap, _env_originals, _torch_original
    with _cap_lock:
        numba_loaded = "numba" in sys.modules
        if n is None:
            if _env_originals is None:
                return
            for var, value in _env_originals.items():
                if var == _NUMBA_ENV and numba_loaded:
                    # numba's config already consumed the capped value;
                    # changing it now would crash the next compile.
                    continue
                if value is None:
                    os.environ.pop(var, None)
                else:
                    os.environ[var] = value
            _restore_threadpools()
            if _torch_original is not None:
                _set_torch_threads(_torch_original)
            _env_originals = None
            _cap = None
            _logger.info("CPU thread cap lifted")
            return
        n = max(1, int(n))
        torch_mod = sys.modules.get("torch")
        if torch_mod is not None and _torch_original is None:
            # Read FIRST, before capping env vars or native pools: torch's
            # first get_num_threads lazily initialises its pool from
            # OMP_NUM_THREADS (resetting OpenMP) and reads the calling
            # thread's OpenMP limit -- either would record the cap itself.
            try:
                _torch_original = int(torch_mod.get_num_threads())
            except Exception:
                _logger.debug("torch.get_num_threads failed", exc_info=True)
        if _env_originals is None:
            _env_originals = {
                v: os.environ.get(v)
                for v in THREAD_ENV_VARS
                if not (v == _NUMBA_ENV and numba_loaded)
            }
        for var, value in _env_originals.items():
            if var == _NUMBA_ENV and numba_loaded:
                continue  # written before numba loaded; now frozen
            os.environ[var] = _env_min(value, n)
        _restore_threadpools()
        _limit_threadpools(n)
        if _torch_original is not None:
            _set_torch_threads(max(1, min(_torch_original, n)))
        _cap = n
    _logger.info("CPU thread cap set to %d", n)


def bind_thread_cpu_cap() -> None:
    """Apply the cap to thread-local pools (numba) on the calling thread.

    numba's thread count is thread-local, so every thread that may run a
    ``parallel=True`` kernel must call this first. A cheap no-op when the
    process is uncapped and this thread was never bound. A thread bound under
    a cap that has since been lifted is returned to its own original count.
    """
    cap = _cap
    original = getattr(_thread_state, "numba_original", None)
    if cap is None and original is None:
        return
    numba_mod = sys.modules.get("numba")
    if numba_mod is None:
        return
    try:
        if cap is None:
            numba_mod.set_num_threads(original)
            _thread_state.numba_original = None
            return
        if original is None:
            original = numba_mod.get_num_threads()
            _thread_state.numba_original = original
        numba_mod.set_num_threads(max(1, min(original, cap)))
    except Exception:
        _logger.debug("numba thread cap failed", exc_info=True)
