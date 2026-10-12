# TrackerKit CPU Oversubscription Protection Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Bound TrackerKit's CPU thread use (CLI + GUI): fan-out always splits an affinity-aware core budget across children; a plain run is capped only when a budget is explicitly given.

**Architecture:** One Qt-free authority `src/hydra_suite/runtime/cpu_budget.py` (usable cores, budget resolution, pure fan-out planner, a process-level cap setter + per-thread numba binder). Lazily-sized pools (ORT, crop-warp, PyAV decode/encode, AprilTag) read `current_cpu_cap()`; the OpenCV FFmpeg decoder is capped by `OPENCV_FFMPEG_THREADS`. Fan-out passes `HYDRA_CPU_THREADS` + thread env vars to children; the CLI, child and GUI call the same functions.

**Tech Stack:** Python 3.11, threadpoolctl 3.6, onnxruntime, PyAV, OpenCV, numba, torch, PySide6 (GUI), pytest.

**Spec:** `docs/superpowers/specs/2026-10-12-cpu-oversubscription-design.md` (rev 2). Read it before starting any task.

## Global Constraints

- Plain path invariant: with no `--cpu-budget`, no `HYDRA_CPU_BUDGET`, no `HYDRA_CPU_THREADS`, and GUI `cpu_budget == 0`, nothing in this plan may change any thread count, env var, session option or output byte. `current_cpu_cap()` is `None` and every consumer takes its pre-existing branch.
- Core/Runtime/Utils must not import app layers (`trackerkit`, etc.). `cpu_budget.py` lives in `runtime/` and imports only stdlib + optional `threadpoolctl`/`torch`/`numba` lazily.
- Thread env vars managed: `OMP_NUM_THREADS`, `MKL_NUM_THREADS`, `OPENBLAS_NUM_THREADS`, `NUMBA_NUM_THREADS`, `VECLIB_MAXIMUM_THREADS`, `NUMEXPR_NUM_THREADS`, `OPENCV_FFMPEG_THREADS`. Always `min(existing, n)`, never `setdefault`.
- Precedence in one process: `HYDRA_CPU_THREADS` > `--cpu-budget` > `HYDRA_CPU_BUDGET`.
- Floors: `GPU_BOUND_MIN_THREADS = 1`, `CPU_BOUND_MIN_THREADS = 2` (module constants).
- Tests run in conda env `hydra-mps`: `conda run -n hydra-mps python -m pytest ...` from the worktree root with `PYTHONPATH=$PWD/src`.
- Commits: as the configured git user, no Co-Authored-By trailer. Run `make format` (or `black` + `isort` on touched files) before each commit.
- Tests that mutate process-global thread state (torch, threadpoolctl, numba, `os.environ`) must run the assertion in a **subprocess** (`subprocess.run([sys.executable, "-c", ...])`) so they don't leak into the rest of the suite.

## Review Focus

- Parent shell already exports `OMP_NUM_THREADS=64` (or `=1`) → child gets `min(64, n)` = n (or stays 1); never raised. Pinned in Task 4.
- `HYDRA_CPU_BUDGET` exported on the box (diptera default) and a fan-out child inherits the env → child must use `HYDRA_CPU_THREADS`, not the whole-box budget. Pinned in Tasks 4 + 5.
- `--cpu-budget` larger than usable cores, `0`, negative, or non-int `HYDRA_CPU_BUDGET="abc"` → clamp/argparse error/warn-and-ignore, never crash or raise the budget. Pinned in Tasks 1 + 5.
- Fewer videos than slots (2 videos, 8 GPUs, budget 32) → 16 threads each, not 4. Pinned in Task 1 + 4.
- GUI user sets a budget, runs, then sets it back to 0 → process returns to its original thread state (restore), and a never-capped GUI run is a strict no-op. Pinned in Task 2 + 6.

---

### Task 1: Budget resolution and the pure planner

**Files:**
- Create: `src/hydra_suite/runtime/cpu_budget.py`
- Modify: `pyproject.toml` (declare `threadpoolctl` in core `dependencies`, next to `scikit-learn`)
- Test: `tests/test_cpu_budget.py`

**Interfaces:**
- Produces:
  - `usable_cores(*, affinity=None, cpu_count=None, cgroup_root="/sys/fs/cgroup", proc_cgroup="/proc/self/cgroup") -> int`
  - `@dataclass(frozen=True) CpuBudget(cores: int, usable: int, source: str, warnings: tuple[str, ...] = ())` — `source ∈ {"flag","env","auto"}`
  - `resolve_cpu_budget(requested: int | None, *, env: Mapping[str,str] | None = None, usable: int | None = None) -> CpuBudget`
  - `@dataclass(frozen=True) CpuPlan(jobs: int, threads_per_job: int, clamped: bool, warnings: tuple[str, ...] = ())`
  - `plan_cpu(*, jobs: int, n_videos: int, budget: CpuBudget, gpu_bound: bool, threads_per_job: int | None = None) -> CpuPlan`
  - `resolve_process_cap(cli_budget: int | None, *, env: Mapping[str,str] | None = None, usable: int | None = None) -> int | None`
  - Constants `GPU_BOUND_MIN_THREADS = 1`, `CPU_BOUND_MIN_THREADS = 2`, `THREAD_ENV_VARS` (tuple of the 7 names in Global Constraints), `ENV_CPU_THREADS = "HYDRA_CPU_THREADS"`, `ENV_CPU_BUDGET = "HYDRA_CPU_BUDGET"`.

- [ ] **Step 1: Write the failing tests** (`tests/test_cpu_budget.py`)

```python
import pytest

from hydra_suite.runtime import cpu_budget as cb


def _write(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def test_usable_cores_prefers_affinity_over_cpu_count(tmp_path):
    assert cb.usable_cores(affinity={0, 1, 2}, cpu_count=64, cgroup_root=str(tmp_path / "none"),
                           proc_cgroup=str(tmp_path / "none")) == 3


def test_usable_cores_falls_back_to_cpu_count(tmp_path):
    assert cb.usable_cores(affinity=None, cpu_count=8, cgroup_root=str(tmp_path / "x"),
                           proc_cgroup=str(tmp_path / "x")) == 8


def test_usable_cores_cgroup_quota_on_ancestor(tmp_path):
    root = tmp_path / "cg"
    proc = tmp_path / "proc_cgroup"
    _write(proc, "0::/user.slice/user-1.slice/session-3.scope\n")
    _write(root / "user.slice" / "cpu.max", "max 100000\n")
    _write(root / "user.slice" / "user-1.slice" / "cpu.max", "250000 100000\n")
    # session scope has no cpu.max file; ancestor quota 2.5 -> ceil = 3
    assert cb.usable_cores(affinity=set(range(16)), cpu_count=16,
                           cgroup_root=str(root), proc_cgroup=str(proc)) == 3


def test_usable_cores_ignores_garbage_cgroup(tmp_path):
    root = tmp_path / "cg"
    proc = tmp_path / "proc_cgroup"
    _write(proc, "0::/a\n")
    _write(root / "a" / "cpu.max", "garbage\n")
    assert cb.usable_cores(affinity=set(range(4)), cpu_count=4,
                           cgroup_root=str(root), proc_cgroup=str(proc)) == 4


def test_usable_cores_never_below_one(tmp_path):
    assert cb.usable_cores(affinity=set(), cpu_count=None, cgroup_root=str(tmp_path),
                           proc_cgroup=str(tmp_path / "x")) == 1


def test_resolve_budget_auto_env_flag_and_clamp():
    assert cb.resolve_cpu_budget(None, env={}, usable=12) == cb.CpuBudget(12, 12, "auto")
    b = cb.resolve_cpu_budget(None, env={"HYDRA_CPU_BUDGET": "6"}, usable=12)
    assert (b.cores, b.source) == (6, "env")
    b = cb.resolve_cpu_budget(4, env={"HYDRA_CPU_BUDGET": "6"}, usable=12)
    assert (b.cores, b.source) == (4, "flag")
    b = cb.resolve_cpu_budget(40, env={}, usable=12)
    assert b.cores == 12 and b.warnings  # can only lower


@pytest.mark.parametrize("bad", ["abc", "0", "-3", ""])
def test_resolve_budget_invalid_env_is_ignored_with_warning(bad):
    b = cb.resolve_cpu_budget(None, env={"HYDRA_CPU_BUDGET": bad}, usable=8)
    assert b.cores == 8 and b.source == "auto"
    assert bool(b.warnings) == (bad != "")


@pytest.mark.parametrize(
    "jobs,n_videos,cores,gpu_bound,tpj,expected",
    [
        (4, 10, 16, False, None, (4, 4, False)),
        (16, 20, 12, False, None, (6, 2, True)),     # CPU floor 2 -> max 6 jobs
        (16, 20, 12, True, None, (12, 1, True)),     # GPU floor 1 -> max 12 jobs
        (8, 2, 32, True, None, (8, 16, False)),      # divide by active videos
        (1, 1, 12, False, None, (1, 12, False)),
        (16, 20, 12, False, 3, (16, 3, False)),      # explicit override: no clamp
    ],
)
def test_plan_cpu_table(jobs, n_videos, cores, gpu_bound, tpj, expected):
    plan = cb.plan_cpu(jobs=jobs, n_videos=n_videos, budget=cb.CpuBudget(cores, cores, "auto"),
                       gpu_bound=gpu_bound, threads_per_job=tpj)
    assert (plan.jobs, plan.threads_per_job, plan.clamped) == expected
    assert bool(plan.warnings) == (plan.clamped or (tpj is not None and jobs * tpj > cores))


def test_resolve_process_cap_precedence():
    env = {"HYDRA_CPU_THREADS": "3", "HYDRA_CPU_BUDGET": "8"}
    assert cb.resolve_process_cap(6, env=env, usable=16) == 3
    assert cb.resolve_process_cap(6, env={"HYDRA_CPU_BUDGET": "8"}, usable=16) == 6
    assert cb.resolve_process_cap(None, env={"HYDRA_CPU_BUDGET": "8"}, usable=16) == 8
    assert cb.resolve_process_cap(None, env={}, usable=16) is None
    assert cb.resolve_process_cap(None, env={"HYDRA_CPU_THREADS": "junk"}, usable=16) is None
```

- [ ] **Step 2: Run to verify failure**

Run: `PYTHONPATH=$PWD/src conda run -n hydra-mps python -m pytest tests/test_cpu_budget.py -q`
Expected: FAIL (`ModuleNotFoundError: hydra_suite.runtime.cpu_budget`).

- [ ] **Step 3: Implement** `src/hydra_suite/runtime/cpu_budget.py` (first half; Task 2 appends the applier)

```python
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
    return CpuBudget(cores=cores, usable=usable, source=source, warnings=tuple(warnings))


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
```

Also add `"threadpoolctl>=3.1"` to `[project] dependencies` in `pyproject.toml` beside `scikit-learn`.

- [ ] **Step 4: Run tests** — same command, expect PASS.
- [ ] **Step 5: Commit** — `git add src/hydra_suite/runtime/cpu_budget.py tests/test_cpu_budget.py pyproject.toml && git commit -m "feat(runtime): CPU budget resolution and fan-out planner"`

---

### Task 2: Process-level cap + per-thread numba binder

**Files:**
- Modify: `src/hydra_suite/runtime/cpu_budget.py` (append)
- Modify: `src/hydra_suite/core/tracking/worker.py` (call `bind_thread_cpu_cap()` first thing in the engine thread's run entry — the method that `trackerkit/gui/workers/tracking_worker.py` and `trackerkit/headless_tracking.py:112` both execute; find it with `grep -n "def run" core/tracking/worker.py`)
- Test: `tests/test_cpu_budget_apply.py`

**Interfaces:**
- Consumes: `THREAD_ENV_VARS` (Task 1).
- Produces: `set_process_cpu_cap(n: int | None) -> None`, `current_cpu_cap() -> int | None`, `bind_thread_cpu_cap() -> None`, `capped(value: int, cap: int | None) -> int` (helper: `value if cap is None else max(1, min(value, cap))`).

- [ ] **Step 1: Write failing tests** (`tests/test_cpu_budget_apply.py`). Every assertion runs in a fresh interpreter.

```python
import json
import os
import subprocess
import sys
import textwrap


def _run(code: str, env_extra=None) -> dict:
    env = dict(os.environ)
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                "NUMBA_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS",
                "OPENCV_FFMPEG_THREADS", "HYDRA_CPU_THREADS", "HYDRA_CPU_BUDGET"):
        env.pop(var, None)
    env["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    env.update(env_extra or {})
    out = subprocess.run([sys.executable, "-c", textwrap.dedent(code)], env=env,
                         capture_output=True, text=True, check=True)
    return json.loads(out.stdout.strip().splitlines()[-1])


def test_never_set_none_is_a_noop():
    r = _run("""
        import json, os, torch
        from hydra_suite.runtime import cpu_budget as cb
        before = (torch.get_num_threads(), os.environ.get("OMP_NUM_THREADS"))
        cb.set_process_cpu_cap(None); cb.bind_thread_cpu_cap()
        after = (torch.get_num_threads(), os.environ.get("OMP_NUM_THREADS"))
        print(json.dumps({"same": before == after, "cap": cb.current_cpu_cap()}))
    """)
    assert r == {"same": True, "cap": None}


def test_cap_sets_env_threadpools_torch_and_restores():
    r = _run("""
        import json, os, torch
        from threadpoolctl import threadpool_info
        from hydra_suite.runtime import cpu_budget as cb
        t0 = torch.get_num_threads()
        cb.set_process_cpu_cap(2)
        capped = {
          "env": {v: os.environ.get(v) for v in cb.THREAD_ENV_VARS},
          "torch": torch.get_num_threads(),
          "pools": max(p["num_threads"] for p in threadpool_info()),
          "cap": cb.current_cpu_cap(),
        }
        cb.set_process_cpu_cap(None)
        restored = {"torch": torch.get_num_threads() == t0,
                    "env": all(os.environ.get(v) == ("64" if v == "OMP_NUM_THREADS" else None)
                               for v in cb.THREAD_ENV_VARS),
                    "cap": cb.current_cpu_cap()}
        print(json.dumps({"capped": capped, "restored": restored}))
    """, env_extra={"OMP_NUM_THREADS": "64"})
    assert set(r["capped"]["env"].values()) == {"2"}
    assert r["capped"]["torch"] <= 2 and r["capped"]["pools"] <= 2 and r["capped"]["cap"] == 2
    assert r["restored"] == {"torch": True, "env": True, "cap": None}


def test_cap_never_raises_existing_lower_env():
    r = _run("""
        import json, os
        from hydra_suite.runtime import cpu_budget as cb
        cb.set_process_cpu_cap(8)
        print(json.dumps({"omp": os.environ["OMP_NUM_THREADS"]}))
    """, env_extra={"OMP_NUM_THREADS": "1"})
    assert r == {"omp": "1"}


def test_recapping_uses_originals_not_previous_cap():
    r = _run("""
        import json, os
        from hydra_suite.runtime import cpu_budget as cb
        cb.set_process_cpu_cap(2); cb.set_process_cpu_cap(6)
        print(json.dumps({"omp": os.environ["OMP_NUM_THREADS"], "cap": cb.current_cpu_cap()}))
    """, env_extra={"OMP_NUM_THREADS": "12"})
    assert r == {"omp": "6", "cap": 6}


def test_bind_is_thread_local_numba():
    r = _run("""
        import json, threading, numba
        from hydra_suite.runtime import cpu_budget as cb
        base = numba.get_num_threads()
        cb.set_process_cpu_cap(2)
        seen = {}
        def work():
            cb.bind_thread_cpu_cap(); seen["bound"] = numba.get_num_threads()
        t = threading.Thread(target=work); t.start(); t.join()
        print(json.dumps({"bound": seen["bound"], "main_untouched": numba.get_num_threads() == base}))
    """)
    assert r["bound"] <= 2 and r["main_untouched"] is True
```

- [ ] **Step 2: Run** `PYTHONPATH=$PWD/src conda run -n hydra-mps python -m pytest tests/test_cpu_budget_apply.py -q` — expect FAIL (`AttributeError: set_process_cpu_cap`).

- [ ] **Step 3: Implement** (append to `cpu_budget.py`)

```python
import logging
import sys
import threading

_logger = logging.getLogger(__name__)
_cap_lock = threading.Lock()
_cap: Optional[int] = None
_originals: Optional[dict] = None  # recorded on the first non-None set
_tp_limiter = None  # threadpoolctl limiter held while capped


def capped(value: int, cap: Optional[int]) -> int:
    """``value`` limited by ``cap`` (unchanged when cap is None)."""
    return value if cap is None else max(1, min(int(value), int(cap)))


def current_cpu_cap() -> Optional[int]:
    return _cap


def _env_min(original: Optional[str], n: int) -> str:
    parsed = _positive_int(original)
    return str(n if parsed is None else min(parsed, n))


def set_process_cpu_cap(n: Optional[int]) -> None:
    """Set (or lift with None) the process-wide CPU thread cap. Idempotent.

    A None when no cap was ever set touches nothing (plain-path invariant).
    Each call is computed from the ORIGINAL state, so re-capping 2 -> 6 gives 6.
    """
    global _cap, _originals, _tp_limiter
    with _cap_lock:
        if n is None:
            if _originals is None:
                return
            for var, value in _originals["env"].items():
                if value is None:
                    os.environ.pop(var, None)
                else:
                    os.environ[var] = value
            if _tp_limiter is not None:
                _tp_limiter.restore_original_limits()
                _tp_limiter = None
            torch_mod = sys.modules.get("torch")
            if torch_mod is not None and _originals["torch"] is not None:
                torch_mod.set_num_threads(_originals["torch"])
            _originals = None
            _cap = None
            return
        n = max(1, int(n))
        if _originals is None:
            torch_mod = sys.modules.get("torch")
            _originals = {
                "env": {v: os.environ.get(v) for v in THREAD_ENV_VARS},
                "torch": torch_mod.get_num_threads() if torch_mod is not None else None,
            }
        for var, value in _originals["env"].items():
            os.environ[var] = _env_min(value, n)
        if _tp_limiter is not None:
            _tp_limiter.restore_original_limits()
            _tp_limiter = None
        try:
            from threadpoolctl import threadpool_limits

            _tp_limiter = threadpool_limits(limits=n)
        except Exception:  # threadpoolctl missing/unsupported lib: env still applies
            _logger.debug("threadpoolctl limit failed", exc_info=True)
        torch_mod = sys.modules.get("torch")
        if torch_mod is not None:
            base = _originals["torch"] or torch_mod.get_num_threads()
            torch_mod.set_num_threads(max(1, min(base, n)))
        _cap = n
    _logger.info("CPU thread cap set to %d", n)


def bind_thread_cpu_cap() -> None:
    """Apply the cap to thread-local pools (numba) on the calling thread."""
    cap = _cap
    if cap is None:
        return
    numba_mod = sys.modules.get("numba")
    if numba_mod is None:
        return
    try:
        numba_mod.set_num_threads(max(1, min(numba_mod.get_num_threads(), cap)))
    except Exception:
        _logger.debug("numba thread cap failed", exc_info=True)
```

Note: torch in `_originals` is recorded at first set; if torch is imported only later, `set_process_cpu_cap` won't have capped it — the env var + threadpoolctl libomp limit still apply. That is accepted (spec: CLI/child set the cap after `trackerkit.app` imports torch).

Then in `core/tracking/worker.py`, at the very top of the engine thread's run entry, add:

```python
from hydra_suite.runtime.cpu_budget import bind_thread_cpu_cap
...
        bind_thread_cpu_cap()
```

- [ ] **Step 4: Run tests** — PASS. Also run `PYTHONPATH=$PWD/src conda run -n hydra-mps python -m pytest tests/ -q -k "tracking_worker or headless" -x` to make sure nothing on the worker path broke.
- [ ] **Step 5: Commit** — `git commit -m "feat(runtime): process-level CPU cap and per-thread numba binder"`

---

### Task 3: Pool consumers read the cap

**Files:**
- Modify: `src/hydra_suite/runtime/onnx_providers.py` (add `session_options_for`)
- Modify ORT sites: `src/hydra_suite/core/inference/direct_executors.py:~497`, `src/hydra_suite/core/individual/classification/backend.py:~893,895,904,938`, `src/hydra_suite/core/individual/pose/runtime/onnx_session.py:~124,128`, `src/hydra_suite/core/individual/pose/backends/yolo.py:~161`
- Modify: `src/hydra_suite/core/inference/stages/crops.py` (`_crop_warp_threads`)
- Modify: `src/hydra_suite/utils/video_decoder.py` (~line 219)
- Modify: `src/hydra_suite/utils/video_encoder.py` (`_open`, `pyav_software` path)
- Modify: `src/hydra_suite/core/inference/config.py:~1405`, `src/hydra_suite/core/individual/classification/apriltag.py:~104`
- Test: `tests/test_cpu_cap_consumers.py`

**Interfaces:**
- Consumes: `current_cpu_cap()`, `capped()` (Task 2).
- Produces: `session_options_for(threads: int | None) -> onnxruntime.SessionOptions | None`.

- [ ] **Step 1: Write failing tests** (`tests/test_cpu_cap_consumers.py`)

```python
import pytest

from hydra_suite.runtime import cpu_budget as cb


@pytest.fixture
def cap(monkeypatch):
    def _set(n):
        monkeypatch.setattr(cb, "_cap", n)
    return _set


def test_session_options_none_when_uncapped():
    from hydra_suite.runtime.onnx_providers import session_options_for
    assert session_options_for(None) is None


def test_session_options_sets_intra_op():
    pytest.importorskip("onnxruntime")
    from hydra_suite.runtime.onnx_providers import session_options_for
    assert session_options_for(3).intra_op_num_threads == 3


def test_crop_warp_threads_respects_cap(cap, monkeypatch):
    from hydra_suite.core.inference.stages import crops
    monkeypatch.delenv("HYDRA_CROP_WARP_THREADS", raising=False)
    cap(None)
    uncapped = crops._crop_warp_threads()
    cap(2)
    assert crops._crop_warp_threads() == min(uncapped, 2)
    monkeypatch.setenv("HYDRA_CROP_WARP_THREADS", "6")
    assert crops._crop_warp_threads() == 2


def test_apriltag_threads_capped(cap):
    from hydra_suite.core.inference.config import apriltag_thread_count  # helper added in Step 3
    cap(None)
    assert apriltag_thread_count({"APRILTAG_THREADS": 4}) == 4
    cap(2)
    assert apriltag_thread_count({"APRILTAG_THREADS": 4}) == 2
    assert apriltag_thread_count({}) == 2


def test_encoder_thread_count_capped(cap, tmp_path):
    pytest.importorskip("av")
    import numpy as np
    from hydra_suite.utils.video_encoder import VideoEncoder
    cap(2)
    enc = VideoEncoder(str(tmp_path / "x.mp4"), width=64, height=64, fps=10, backend="pyav_software")
    enc.write(np.zeros((64, 64, 3), np.uint8))
    assert enc._stream.thread_count == 2
    enc.close()
```

Before writing `test_encoder_thread_count_capped`, open `utils/video_encoder.py` and match the real `VideoEncoder` constructor/`write`/`close` names and how a backend is forced; adjust the test call to match (do not change the encoder's public API).

- [ ] **Step 2: Run** — expect FAIL.

- [ ] **Step 3: Implement**

`runtime/onnx_providers.py`:

```python
def session_options_for(threads: int | None):
    """ORT SessionOptions honouring the CPU cap; None keeps ORT defaults."""
    if threads is None:
        return None
    import onnxruntime as ort

    opts = ort.SessionOptions()
    opts.intra_op_num_threads = max(1, int(threads))
    return opts
```

Each ORT site: add `sess_options=session_options_for(current_cpu_cap())` to the `ort.InferenceSession(...)` call (keyword; leave `providers=` etc. untouched). When the cap is None this passes `sess_options=None`, which is ORT's default — byte-identical.

`crops.py::_crop_warp_threads` — wrap both return paths:

```python
def _crop_warp_threads() -> int:
    from hydra_suite.runtime.cpu_budget import capped, current_cpu_cap

    cap = current_cpu_cap()
    try:
        v = os.environ.get("HYDRA_CROP_WARP_THREADS")
        if v is not None:
            return capped(max(1, int(v)), cap)
    except Exception:
        pass
    return capped(max(1, min(8, os.cpu_count() or 1)), cap)
```

`video_decoder.py` — after `self._stream.thread_type = "AUTO"`:

```python
            cap = current_cpu_cap()
            if cap is not None:
                self._stream.thread_count = cap
```

`video_encoder.py::_open` — after `self._stream.pix_fmt = "yuv420p"`:

```python
                cap = current_cpu_cap()
                if cap is not None and self._backend == "pyav_software":
                    self._stream.thread_count = cap
```

AprilTag — add to `core/inference/config.py`:

```python
def apriltag_thread_count(params) -> int:
    from hydra_suite.runtime.cpu_budget import capped, current_cpu_cap

    return capped(int(params.get("APRILTAG_THREADS", 4)), current_cpu_cap())
```

and replace `threads=int(params.get("APRILTAG_THREADS", 4))` at both sites with `threads=apriltag_thread_count(params)` (import the helper in `apriltag.py` from `hydra_suite.core.inference.config`; if that import is circular, put the helper in `runtime/cpu_budget.py` as `apriltag_thread_count` instead and import from there in both sites).

- [ ] **Step 4: Run tests** — PASS. Then the cv2 decoder check (manual, records evidence for the review):

```bash
PYTHONPATH=$PWD/src conda run -n hydra-mps python - <<'EOF'
import os, threading, cv2, glob
clip = sorted(glob.glob("tools/equivalence/fixtures/clips/fly_obb*.mp4"))[0]
def nthreads():
    import subprocess; return int(subprocess.check_output(["ps","-M",str(os.getpid())]).decode().count("\n"))-1
base = nthreads(); c = cv2.VideoCapture(clip, cv2.CAP_FFMPEG); c.read(); full = nthreads(); c.release()
os.environ["OPENCV_FFMPEG_THREADS"]="2"; c = cv2.VideoCapture(clip, cv2.CAP_FFMPEG); c.read(); cap2 = nthreads(); c.release()
print("base", base, "uncapped capture", full, "capped capture", cap2)
EOF
```

Expected: capped capture thread count far below uncapped (≈ base + 3 vs base + 31).
- [ ] **Step 5: Commit** — `git commit -m "feat: ORT, crop-warp, PyAV decode/encode and AprilTag honour the CPU cap"`

---

### Task 4: Fan-out — child env and the shared slot+CPU planner

**Files:**
- Modify: `src/hydra_suite/trackerkit/batch_fanout.py` (`_THREAD_CAP_VARS` → `THREAD_ENV_VARS`, `build_child_env`, new `resolve_slot_count`, new `plan_fanout_cpu`, `run_batch_fanout` uses `resolve_slot_count`)
- Modify: `tests/test_trackerkit_batch_fanout.py` (flip the "parent's OMP wins" assertion near line 117)
- Test: `tests/test_trackerkit_batch_fanout.py` (add)

**Interfaces:**
- Consumes: `THREAD_ENV_VARS`, `ENV_CPU_THREADS`, `ENV_CPU_BUDGET`, `resolve_cpu_budget`, `plan_cpu`, `CpuPlan`.
- Produces:
  - `resolve_slot_count(options: FanoutOptions) -> int` — exactly `run_batch_fanout`'s current arithmetic.
  - `plan_fanout_cpu(options: FanoutOptions, *, n_videos: int, cpu_budget: int | None, env: Mapping[str,str] | None = None, usable: int | None = None) -> tuple[FanoutOptions, CpuPlan]` — returns a `dataclasses.replace(options, jobs=plan.jobs, threads_per_job=plan.threads_per_job)`; `gpu_bound=bool(options.gpus)`; passes `options.threads_per_job` as the explicit override.

- [ ] **Step 1: Write failing tests** (append)

```python
from hydra_suite.trackerkit.batch_fanout import (
    FanoutOptions, build_child_env, plan_fanout_cpu, resolve_slot_count,
)


def test_child_env_caps_with_min_and_new_vars():
    env = build_child_env({"OMP_NUM_THREADS": "64", "MKL_NUM_THREADS": "1",
                           "HYDRA_CPU_BUDGET": "48"}, gpu=None, threads_per_job=4)
    assert env["OMP_NUM_THREADS"] == "4"          # parent's high value no longer wins
    assert env["MKL_NUM_THREADS"] == "1"          # never raised
    for var in ("VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS", "OPENCV_FFMPEG_THREADS",
                "OPENBLAS_NUM_THREADS", "NUMBA_NUM_THREADS"):
        assert env[var] == "4"
    assert env["HYDRA_CPU_THREADS"] == "4"
    assert "HYDRA_CPU_BUDGET" not in env


def test_child_env_without_threads_is_unchanged_except_existing_rules():
    env = build_child_env({"OMP_NUM_THREADS": "7"}, gpu=None, threads_per_job=None)
    assert env["OMP_NUM_THREADS"] == "7" and "HYDRA_CPU_THREADS" not in env


def _gpus(n):
    from hydra_suite.runtime.cuda_devices import CudaDevice
    return [CudaDevice(index=i, uuid=f"GPU-{i}", name="x") for i in range(n)]


def test_resolve_slot_count_matches_scheduler():
    assert resolve_slot_count(FanoutOptions(gpus=_gpus(4), jobs=None)) == 4
    assert resolve_slot_count(FanoutOptions(gpus=_gpus(4), jobs=9)) == 4
    assert resolve_slot_count(FanoutOptions(gpus=[], jobs=3)) == 3
    assert resolve_slot_count(FanoutOptions(gpus=[], jobs=None)) == 1


def test_plan_fanout_cpu_clamps_cpu_jobs_and_fills_threads():
    opts, plan = plan_fanout_cpu(FanoutOptions(gpus=[], jobs=16), n_videos=20,
                                 cpu_budget=None, env={}, usable=12)
    assert (opts.jobs, opts.threads_per_job, plan.clamped) == (6, 2, True)


def test_plan_fanout_cpu_few_videos_many_gpus():
    opts, _ = plan_fanout_cpu(FanoutOptions(gpus=_gpus(8), jobs=None), n_videos=2,
                              cpu_budget=None, env={}, usable=32)
    assert (opts.jobs, opts.threads_per_job) == (8, 16)


def test_plan_fanout_cpu_explicit_threads_override():
    opts, plan = plan_fanout_cpu(FanoutOptions(gpus=[], jobs=16, threads_per_job=3),
                                 n_videos=20, cpu_budget=None, env={}, usable=12)
    assert (opts.jobs, opts.threads_per_job, plan.clamped) == (16, 3, False)
```

Check `CudaDevice`'s real constructor fields in `runtime/cuda_devices.py` and match them in `_gpus`.

Also update the existing test near line 117 that asserts the parent's `OMP_NUM_THREADS == "7"` wins when `threads_per_job` is set: it must now assert `min(7, threads_per_job)`.

- [ ] **Step 2: Run** `PYTHONPATH=$PWD/src conda run -n hydra-mps python -m pytest tests/test_trackerkit_batch_fanout.py -q` — FAIL.

- [ ] **Step 3: Implement** in `batch_fanout.py`:

```python
from hydra_suite.runtime.cpu_budget import (
    ENV_CPU_BUDGET,
    ENV_CPU_THREADS,
    THREAD_ENV_VARS,
    CpuPlan,
    plan_cpu,
    resolve_cpu_budget,
)
```

Delete `_THREAD_CAP_VARS`. `build_child_env`:

```python
    if threads_per_job is not None and int(threads_per_job) > 0:
        n = int(threads_per_job)
        env.pop(ENV_CPU_BUDGET, None)
        env[ENV_CPU_THREADS] = str(n)
        for var in THREAD_ENV_VARS:
            try:
                existing = int(env.get(var, ""))
            except ValueError:
                existing = 0
            env[var] = str(min(existing, n) if existing >= 1 else n)
```

`resolve_slot_count` (and replace the inline arithmetic in `run_batch_fanout` with `n = resolve_slot_count(options)`; keep `slots = list(options.gpus[:n])` / `[None] * n`):

```python
def resolve_slot_count(options: FanoutOptions) -> int:
    """Concurrent child slots the scheduler opens for ``options``."""
    if options.gpus:
        requested = len(options.gpus) if options.jobs is None else int(options.jobs)
        return max(1, min(requested, len(options.gpus)))
    return max(1, int(options.jobs or 1))


def plan_fanout_cpu(
    options: FanoutOptions,
    *,
    n_videos: int,
    cpu_budget: Optional[int],
    env: Optional[Mapping[str, str]] = None,
    usable: Optional[int] = None,
) -> tuple[FanoutOptions, CpuPlan]:
    """Fit the slot count into the CPU budget and size each child's threads."""
    budget = resolve_cpu_budget(cpu_budget, env=env, usable=usable)
    plan = plan_cpu(
        jobs=resolve_slot_count(options),
        n_videos=n_videos,
        budget=budget,
        gpu_bound=bool(options.gpus),
        threads_per_job=options.threads_per_job,
    )
    return replace(options, jobs=plan.jobs, threads_per_job=plan.threads_per_job), plan
```

(Make sure `replace` and `Mapping` are imported.)

- [ ] **Step 4: Run** the batch-fanout test files: `tests/test_trackerkit_batch_fanout.py tests/test_trackerkit_cli_fanout.py tests/test_batch_fanout_windows.py tests/test_trackerkit_batch_fanout_worker.py` — PASS.
- [ ] **Step 5: Commit** — `git commit -m "feat(trackerkit): fan-out children always get a CPU-budget thread cap"`

---

### Task 5: CLI — `--cpu-budget`, planning in `run_tracking_cli`, `job run` forwarding

**Files:**
- Modify: `src/hydra_suite/trackerkit/app.py` (`track` parser: `--cpu-budget`, updated `--threads-per-job` help, validation near `:471-476`, pass `cpu_budget=` into `run_tracking_cli`; `job run` parser: `--cpu-budget`)
- Modify: `src/hydra_suite/trackerkit/cli.py` (`run_tracking_cli(..., cpu_budget: int | None = None)`)
- Modify: `src/hydra_suite/trackerkit/job_cli.py` (`_cmd_run` forwards fan-out flags, local and remote)
- Modify: `docs/user-guide/trackerkit-cli.md` (document `--cpu-budget`, `HYDRA_CPU_BUDGET`, clamp behaviour, `--threads-per-job` override)
- Test: `tests/test_trackerkit_cli_fanout.py` (add), `tests/test_trackerkit_job_cli*.py` (add to the existing job-run test file; find with `grep -ln "_cmd_run\|job run" tests/`)

**Interfaces:**
- Consumes: `plan_fanout_cpu` (Task 4), `resolve_process_cap`, `set_process_cpu_cap`, `bind_thread_cpu_cap` (Tasks 1–2).
- Produces: `run_tracking_cli(..., cpu_budget: int | None = None)`; `job_cli._fanout_flags(args) -> list[str]`.

- [ ] **Step 1: Write failing tests**

In `tests/test_trackerkit_cli_fanout.py` (follow that file's existing pattern for monkeypatching `_run_fanout` / `_run_sequential` and creating dummy video files):

```python
def test_cli_fanout_options_carry_cpu_plan(monkeypatch, tmp_path):
    from hydra_suite.trackerkit import cli
    captured = {}
    monkeypatch.setattr(cli, "_run_fanout", lambda specs, options: captured.setdefault("o", options) and 0)
    monkeypatch.setattr("hydra_suite.runtime.cpu_budget.usable_cores", lambda **_: 12)
    videos = [tmp_path / f"v{i}.mp4" for i in range(20)]
    for v in videos:
        v.write_bytes(b"x")
    monkeypatch.setattr(cli, "plan_batch_jobs", lambda vids, **kw: [object()] * len(vids))
    cli.run_tracking_cli([str(v) for v in videos], jobs=16)
    assert (captured["o"].jobs, captured["o"].threads_per_job) == (6, 2)


def test_cli_plain_run_uncapped_without_budget(monkeypatch, tmp_path):
    from hydra_suite.trackerkit import cli
    calls = []
    monkeypatch.setattr(cli, "set_process_cpu_cap", lambda n: calls.append(n))
    monkeypatch.setattr(cli, "_run_sequential", lambda specs: 0)
    monkeypatch.setattr(cli, "plan_batch_jobs", lambda vids, **kw: [object()])
    monkeypatch.delenv("HYDRA_CPU_BUDGET", raising=False)
    monkeypatch.delenv("HYDRA_CPU_THREADS", raising=False)
    v = tmp_path / "v.mp4"; v.write_bytes(b"x")
    cli.run_tracking_cli([str(v)])
    assert calls == []


def test_cli_plain_run_capped_by_child_env(monkeypatch, tmp_path):
    from hydra_suite.trackerkit import cli
    calls = []
    monkeypatch.setattr(cli, "set_process_cpu_cap", lambda n: calls.append(n))
    monkeypatch.setattr(cli, "_run_sequential", lambda specs: 0)
    monkeypatch.setattr(cli, "plan_batch_jobs", lambda vids, **kw: [object()])
    monkeypatch.setenv("HYDRA_CPU_THREADS", "3")
    monkeypatch.setenv("HYDRA_CPU_BUDGET", "40")
    v = tmp_path / "v.mp4"; v.write_bytes(b"x")
    cli.run_tracking_cli([str(v)], cpu_budget=8)
    assert calls == [3]
```

Adapt the `plan_batch_jobs` stub to whatever the existing tests in that file use (specs need `.index`, `.video_path` etc. if `_run_fanout` is not stubbed).

Argparse test (same file or `tests/test_trackerkit_app*.py` — follow existing style):

```python
import pytest

def test_cpu_budget_must_be_positive(tmp_path):
    from hydra_suite.trackerkit.app import parse_arguments
    v = tmp_path / "v.mp4"; v.write_bytes(b"x")
    with pytest.raises(SystemExit):
        parse_arguments(["track", str(v), "--cpu-budget", "0"])
    assert parse_arguments(["track", str(v), "--cpu-budget", "4"]).cpu_budget == 4
```

Job run test: assert `_cmd_run` with a local job dir (follow the existing job-run test fixture) and `--jobs 2 --threads-per-job 3 --cpu-budget 8 --gpus 0,1` produces a `run_argv` containing `["--gpus", "0,1", "--jobs", "2", "--threads-per-job", "3", "--cpu-budget", "8"]`, and the remote ssh command string contains the same flags.

- [ ] **Step 2: Run** — FAIL.

- [ ] **Step 3: Implement**

`app.py` track parser, after `--threads-per-job`:

```python
    track_parser.add_argument(
        "--cpu-budget",
        type=int,
        default=None,
        metavar="N",
        help=(
            "Total CPU cores hydra may use across all parallel jobs (default: "
            "HYDRA_CPU_BUDGET, else every usable core). Parallel jobs are reduced "
            "to fit and each child gets N // jobs threads. On a single run it caps "
            "that run's thread pools."
        ),
    )
```

Update `--threads-per-job` help to: `"Explicit CPU threads per parallel child. Overrides the automatic N // jobs split and disables the job clamp (may oversubscribe)."`. Add validation next to the `--threads-per-job` check: `if cpu_budget is not None and int(cpu_budget) < 1: track_parser.error("--cpu-budget must be >= 1")`. Pass `cpu_budget=getattr(args, "cpu_budget", None)` into `run_tracking_cli`. Add `job_run.add_argument("--cpu-budget", type=int, default=None)`.

`cli.py`:

```python
from hydra_suite.runtime.cpu_budget import (
    bind_thread_cpu_cap,
    resolve_process_cap,
    set_process_cpu_cap,
)
from hydra_suite.trackerkit.batch_fanout import plan_fanout_cpu
```

In `run_tracking_cli` add `cpu_budget: int | None = None`. Plain branch:

```python
    if not fanout_requested(gpus, jobs):
        cap = resolve_process_cap(cpu_budget)
        if cap is not None:
            set_process_cpu_cap(cap)
            bind_thread_cpu_cap()
        return _run_sequential(specs)
```

Fan-out branch, replacing the `options = FanoutOptions(...)` construction:

```python
    options, cpu_plan = plan_fanout_cpu(
        FanoutOptions(
            gpus=devices,
            jobs=effective_jobs,
            threads_per_job=threads_per_job,
            log_level=log_level,
        ),
        n_videos=len(specs),
        cpu_budget=cpu_budget,
    )
    for warning in cpu_plan.warnings:
        logger.warning("CPU budget: %s", warning)
    logger.info(
        "Fan-out: %d concurrent job(s) x %d CPU thread(s)",
        options.jobs,
        options.threads_per_job,
    )
```

`job_cli.py`:

```python
def _fanout_flags(args) -> list[str]:
    flags: list[str] = []
    for name, flag in (("gpus", "--gpus"), ("jobs", "--jobs"),
                       ("threads_per_job", "--threads-per-job"),
                       ("cpu_budget", "--cpu-budget")):
        value = getattr(args, name, None)
        if value is not None and str(value) != "":
            flags += [flag, str(value)]
    return flags
```

In `_cmd_run`: local branch `run_argv = ["./run.sh", *passthrough, *_fanout_flags(args)]`; remote branch `passthrough = list(passthrough or []) + _fanout_flags(args)`. Verify `run.sh` (generated by the job packer; grep `"$@"` in `trackerkit/job_*.py`) passes extra args through to `track`.

Docs: add a "CPU budget" subsection to `docs/user-guide/trackerkit-cli.md` covering `--cpu-budget`, `HYDRA_CPU_BUDGET` (local only; use `--cpu-budget` with remote `job run`), the clamp warning, `--threads-per-job` as the override, and that the pose SLEAP sidecar inherits the same per-job cap (a pose job may use up to ~2×).

- [ ] **Step 4: Run** the CLI/job/fanout test files — PASS. Run `make docs-build` if mkdocs is installed (skip otherwise; note it in the report).
- [ ] **Step 5: Commit** — `git commit -m "feat(trackerkit): --cpu-budget, CPU-planned fan-out, job run forwards fan-out flags"`

---

### Task 6: GUI — budget control, planned fan-out, process cap before workers

**Files:**
- Modify: `src/hydra_suite/trackerkit/config/schemas.py` (`cpu_budget: int = 0` + to_dict/from_dict)
- Modify: `src/hydra_suite/trackerkit/gui/panels/setup_panel.py` (spinbox beside `combo_runtime_tier` ~`:674`; plan label in `container_batch_parallel` ~`:370-400`)
- Modify: `src/hydra_suite/trackerkit/gui/main_window.py` (`_on_batch_parallel_changed` mirrors `cpu_budget` and refreshes the plan label; new `_on_cpu_budget_changed`)
- Modify: `src/hydra_suite/trackerkit/gui/orchestrators/tracking.py` (`_resolve_fanout_options`, `start_batch_fanout`, new `_apply_cpu_budget`, called before every tracking/post-processing/final-media-export/interpolated-crops/session worker `.start()`)
- Modify: the parallel checkbox tooltip (setup_panel `:355-361`): replace "Tracking CSVs are byte-identical to sequential batch tracking" with the outcome measured in Task 7 (see Task 7 Step 3); until then leave it and add "Each child gets an equal share of the CPU budget."
- Test: `tests/test_trackerkit_config_schema_fanout.py` (add), `tests/test_trackerkit_gui_cpu_budget.py` (new; follow the offscreen-Qt fixture pattern of an existing trackerkit GUI orchestrator test — find one with `grep -ln "_resolve_fanout_options\|start_batch_fanout" tests/`)

**Interfaces:**
- Consumes: `plan_fanout_cpu`, `resolve_slot_count` (Task 4); `set_process_cpu_cap`, `usable_cores`, `resolve_cpu_budget`, `plan_cpu` (Tasks 1–2).
- Produces: `TrackerConfig.cpu_budget: int` (0 = auto); `TrackingOrchestrator._apply_cpu_budget() -> None`.

- [ ] **Step 1: Write failing tests**

Schema:

```python
def test_cpu_budget_round_trips():
    from hydra_suite.trackerkit.config.schemas import TrackerConfig
    cfg = TrackerConfig()
    assert cfg.cpu_budget == 0
    cfg.cpu_budget = 6
    assert TrackerConfig.from_dict(cfg.to_dict()).cpu_budget == 6
    assert TrackerConfig.from_dict({}).cpu_budget == 0
```

Orchestrator (`tests/test_trackerkit_gui_cpu_budget.py`) — reuse the existing fixture that builds a main window/orchestrator offscreen; if none is cheap, construct the orchestrator with a `SimpleNamespace` main window exactly as the existing `_resolve_fanout_options` tests do:

```python
def test_resolve_fanout_options_applies_cpu_plan(orchestrator, monkeypatch):
    monkeypatch.setattr("hydra_suite.runtime.cpu_budget.usable_cores", lambda **_: 12)
    monkeypatch.setattr("hydra_suite.runtime.cuda_devices.list_cuda_devices", lambda: [])
    monkeypatch.setattr("hydra_suite.trackerkit.batch_fanout.host_has_cuda", lambda: False)
    cfg = orchestrator._mw.config
    cfg.batch_parallel_jobs, cfg.batch_parallel_gpus, cfg.cpu_budget = 16, "", 0
    options = orchestrator._resolve_fanout_options(n_videos=20)
    assert (options.jobs, options.threads_per_job) == (6, 2)


def test_apply_cpu_budget_zero_is_noop_and_positive_caps(orchestrator, monkeypatch):
    calls = []
    monkeypatch.setattr("hydra_suite.trackerkit.gui.orchestrators.tracking.set_process_cpu_cap",
                        lambda n: calls.append(n))
    orchestrator._mw.config.cpu_budget = 0
    orchestrator._apply_cpu_budget()
    orchestrator._mw.config.cpu_budget = 4
    orchestrator._apply_cpu_budget()
    assert calls == [None, 4]
```

(`set_process_cpu_cap(None)` is itself a no-op when never capped — Task 2 — so passing None for 0 is correct and restores after a previous cap.)

- [ ] **Step 2: Run** — FAIL.

- [ ] **Step 3: Implement**

`schemas.py`: add under the batch fan-out fields

```python
    # --- CPU budget (session state; never emitted into the engine config) ---
    cpu_budget: int = 0  # 0 = auto (all usable cores for fan-out; uncapped single run)
```

with `"cpu_budget": int(self.cpu_budget)` in `to_dict` and `cpu_budget=int(data.get("cpu_budget", 0) or 0)` in `from_dict`.

`setup_panel.py`: beside the runtime tier combo, add

```python
        from hydra_suite.runtime.cpu_budget import usable_cores

        self.spin_cpu_budget = QSpinBox()
        self.spin_cpu_budget.setRange(0, usable_cores())
        self.spin_cpu_budget.setSpecialValueText("Auto")
        self.spin_cpu_budget.setValue(int(self._main_window.config.cpu_budget or 0))
        self.spin_cpu_budget.setToolTip(
            "CPU cores TrackerKit may use. Auto = all usable cores split across\n"
            "parallel batch jobs; a single run is left uncapped. Set a number to\n"
            "share this machine with other work."
        )
        self.spin_cpu_budget.valueChanged.connect(self._main_window._on_cpu_budget_changed)
```

added to the same layout row as the runtime tier with a `QLabel("CPU cores:")`. In `container_batch_parallel` add `self.lbl_batch_parallel_cpu_plan = QLabel("")` (word-wrapped, small/secondary style used elsewhere in the panel).

`main_window.py`:

```python
    def _on_cpu_budget_changed(self, *_args) -> None:
        self.config.cpu_budget = int(self._setup_panel.spin_cpu_budget.value())
        self._refresh_cpu_plan_label()

    def _refresh_cpu_plan_label(self) -> None:
        from hydra_suite.runtime.cpu_budget import plan_cpu, resolve_cpu_budget

        panel = self._setup_panel
        budget = resolve_cpu_budget(int(self.config.cpu_budget or 0) or None)
        jobs = int(panel.spin_batch_parallel_jobs.value())
        n_videos = max(1, len(getattr(self, "batch_videos", []) or []))
        plan = plan_cpu(jobs=jobs, n_videos=n_videos, budget=budget,
                        gpu_bound=panel.edit_batch_parallel_gpus.isVisible())
        text = f"{plan.jobs} job(s) x {plan.threads_per_job} thread(s) of {budget.cores} core(s)"
        if plan.clamped:
            text += f" — reduced from {jobs} to fit the CPU budget"
        panel.lbl_batch_parallel_cpu_plan.setText(text)
```

and call `self._refresh_cpu_plan_label()` at the end of `_on_batch_parallel_changed`. (`gpu_bound` here is a display approximation; the authoritative plan is computed at start from resolved devices.)

`orchestrators/tracking.py`:

```python
from hydra_suite.runtime.cpu_budget import set_process_cpu_cap
...
    def _apply_cpu_budget(self) -> None:
        """Apply the GUI CPU budget to this process (0 = leave/restore uncapped)."""
        budget = int(getattr(self._mw.config, "cpu_budget", 0) or 0)
        set_process_cpu_cap(budget if budget > 0 else None)
```

Note the spec says the GUI budget is clamped to usable cores — `QSpinBox` max already enforces it.

`_resolve_fanout_options(self, n_videos: int = 1)`: keep the device logic, then replace the final `return FanoutOptions(...)` with

```python
        options, plan = plan_fanout_cpu(
            FanoutOptions(gpus=devices, jobs=jobs, log_level="INFO"),
            n_videos=n_videos,
            cpu_budget=int(cfg.cpu_budget or 0) or None,
        )
        for warning in plan.warnings:
            logger.warning("CPU budget: %s", warning)
        return options
```

(import `plan_fanout_cpu` alongside the existing `FanoutOptions` import). In `start_batch_fanout`: call `self._resolve_fanout_options(n_videos=len(specs))`, and replace the inline slot-mirror arithmetic with `slots = resolve_slot_count(options)` (options.jobs is now always an int). Show `plan.warnings` to the user: if `options.jobs` is less than the pre-plan request, put the warning into the status line text that already reports slots.

Find every worker start site: `grep -n "\.start()" src/hydra_suite/trackerkit/gui/orchestrators/*.py src/hydra_suite/trackerkit/gui/main_window.py` and call `self._apply_cpu_budget()` (or `self._mw._tracking_orchestrator._apply_cpu_budget()` from other orchestrators — use whatever accessor the code already uses to reach the tracking orchestrator) immediately before starting `TrackingWorker`, `PostProcessWorker`, `FinalMediaExportWorker`, `InterpolatedCropsWorker` and `SessionWorker`. Do not add it before preview/optimizer/fan-out workers (fan-out children get their own cap via env).

- [ ] **Step 4: Run** the new tests plus `tests/test_trackerkit_config_schema_fanout.py tests/test_trackerkit_batch_fanout_worker.py` and any GUI test that touches `setup_panel`/`_resolve_fanout_options` (`grep -ln "setup_panel\|_resolve_fanout_options" tests/`) — PASS. Use `QT_QPA_PLATFORM=offscreen`.
- [ ] **Step 5: Commit** — `git commit -m "feat(trackerkit-gui): CPU budget control, planned fan-out, process cap before workers"`

---

### Task 7: End-to-end verification — child probe, plain-path equivalence, fan-out parity

**Files:**
- Test: `tests/test_cpu_budget_child_probe.py`
- Modify (only if Step 3 shows bytes differ): `setup_panel.py` parallel tooltip, `docs/user-guide/trackerkit-cli.md`

- [ ] **Step 1: Child probe test** — spawns a real child with the fan-out env and checks pools:

```python
import json, os, subprocess, sys, textwrap

from hydra_suite.trackerkit.batch_fanout import build_child_env


def test_capped_child_pools_respect_cap():
    env = build_child_env(dict(os.environ), gpu=None, threads_per_job=2)
    env["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    code = textwrap.dedent("""
        import json, numba
        import hydra_suite.trackerkit.app  # same import graph as a real child
        from threadpoolctl import threadpool_info
        from hydra_suite.runtime import cpu_budget as cb
        cb.set_process_cpu_cap(cb.resolve_process_cap(None)); cb.bind_thread_cpu_cap()
        import torch
        print(json.dumps({
            "cap": cb.current_cpu_cap(),
            "pools": max(p["num_threads"] for p in threadpool_info()),
            "numba": numba.get_num_threads(),
            "torch": torch.get_num_threads(),
        }))
    """)
    out = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, check=True)
    r = json.loads(out.stdout.strip().splitlines()[-1])
    print("child probe:", r)  # torch is logged, not asserted (import-order dependent)
    assert r["cap"] == 2 and r["pools"] <= 2 and r["numba"] <= 2
```

Run it; PASS.

- [ ] **Step 2: Plain-path equivalence** — kill stale sleap/hydra processes first (`pgrep -fl "sleap|hydra"`; only those). Then, with conda `hydra-mps` active, run the matrix with baseline = this branch's base commit and current = the worktree:

```bash
git worktree add --detach .worktrees/cpu-base <branch-base-sha>   # = git merge-base HEAD main
REPO=$PWD WT=$PWD MAIN_SRC=$PWD/.worktrees/cpu-base/src WT_SRC=$PWD/src \
  OUT=/tmp/equiv_cpu RUNTIME=mps bash tools/equivalence/run_matrix.sh fly_obb worm_bgsub
```

Expected: EQUIVALENT (byte-identical) for both clips, row counts > 1, perf ratio ≤ 1.25. Remove the base worktree afterwards (`git worktree remove --force .worktrees/cpu-base`).

- [ ] **Step 3: Fan-out vs plain parity measurement** — run `fly_obb` and `worm_bgsub` once plain and once via `trackerkit track <clip> --jobs 2 --cpu-budget 4` (CPU-only fan-out on this box) with the same config, and compare the `_tracking_final.csv` files with `cmp`. Record the result in the commit message. If any differ, change the parallel-checkbox tooltip claim from "byte-identical" to "match sequential tracking up to CPU floating-point rounding" and add the same sentence to the CLI doc.

- [ ] **Step 4: Full suite** — `PYTHONPATH=$PWD/src conda run -n hydra-mps make pytest` (or the repo's batched equivalent). Compare failure SETS with the branch base (memory: chunking shifts make raw counts misleading); no new failures.

- [ ] **Step 5: Commit** — `git commit -m "test: CPU budget child probe; verified plain-path byte-identity"`
