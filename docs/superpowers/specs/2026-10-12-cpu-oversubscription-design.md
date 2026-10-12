# TrackerKit CPU oversubscription protection — design

**Status:** approved in chat 2026-10-12; implementation on `feat/cpu-oversubscription`.

## Problem

TrackerKit never caps its own CPU thread pools. The only cap is the opt-in
`trackerkit track --threads-per-job`, which sets four BLAS env vars by
`setdefault` (so a parent shell's value wins) and reaches none of the pools
that actually run full-width. Concretely (map taken 2026-10-12):

| Pool | Where | Size today |
|---|---|---|
| ONNX Runtime intra-op | 5 `InferenceSession` sites, no `sess_options` | ORT default (~1/physical core) **per session** |
| Crop-warp `ThreadPoolExecutor` | `core/inference/stages/crops.py:_crop_warp_threads` | `HYDRA_CROP_WARP_THREADS` else `min(8, cpu_count)` |
| PyAV decode | `utils/video_decoder.py` (`thread_type="AUTO"`) | FFmpeg's choice (~cores) |
| AprilTag detector | `APRILTAG_THREADS` param | 4 |
| torch intra-op / OpenCV | never set | 1 if `ultralytics` was imported first (it sets `OMP_NUM_THREADS=1` when unset), else torch=cores-ish, cv2=cores |
| BLAS / OpenMP | env only | full width unless env set |
| SLEAP service (sleap-nn torch) | `conda run` sidecar, inherits env | full width (never imports ultralytics) |

Consequences, by the three scenarios that motivate this work:

1. **Multi-GPU fan-out** (`--gpus`): N children each run the pools above at
   full width → N× oversubscription.
2. **Shared box** (diptera, shared with other users' jobs): hydra takes every
   core; there is no way to bound hydra's total share. Raw `cpu_count` also
   ignores CPU affinity / cgroup quotas.
3. **CPU-only `--jobs N`**: N is never checked against the core count.

Also: the GUI fan-out cannot set a thread cap at all, and
`trackerkit job run` parses `--gpus/--jobs/--threads-per-job` and silently
drops them.

## Decisions (user-approved)

- **Budget source:** affinity-aware usable cores + explicit lower cap.
- **Scope:** fan-out is *always* capped; a plain single run is unchanged
  (byte-identical) unless a budget is explicitly set.
- **Over budget:** clamp concurrent jobs to fit, warn; explicit
  `--threads-per-job` is the user override and disables the clamp.
- **Accepted trade-offs:** (a) capping BLAS/torch/ORT threads in fan-out
  children may move their output vs a plain run at the BLAS/reduction
  rounding floor (already true of `--threads-per-job` today; no test asserts
  fan-out == plain bytes); (b) a pose slot's SLEAP sidecar inherits the same
  cap, so a pose job may use up to ~2× threads/job — documented, not halved.

## Architecture

One new runtime module, `src/hydra_suite/runtime/cpu_budget.py`, is the single
authority. It is pure-ish, Qt-free, and imports nothing from app layers.

### `usable_cores() -> int`

Cores this process may actually use, ≥ 1:

1. `len(os.sched_getaffinity(0))` where available (Linux), else
   `os.cpu_count()`.
2. If a cgroup v2 quota is readable (`/sys/fs/cgroup/cpu.max`, not `max`),
   `min(previous, ceil(quota / period))`. Read failures are ignored.

### `resolve_cpu_budget(requested: int | None) -> CpuBudget`

- `requested` comes from `--cpu-budget` / the GUI spinbox; `None`/`0` falls back
  to the env var `HYDRA_CPU_BUDGET` (a persistent per-box default for shared
  machines), then to `usable_cores()`.
- A request above `usable_cores()` is clamped down with a warning (the budget
  can only lower).
- `CpuBudget(cores: int, usable: int, source: "flag"|"env"|"auto", warnings: tuple[str, ...])`.
- Invalid `HYDRA_CPU_BUDGET` (non-int, < 1) → warning, ignored.

### `plan_cpu(*, jobs: int, budget: CpuBudget, gpu_bound: bool, threads_per_job: int | None) -> CpuPlan`

Pure. Returns `CpuPlan(jobs: int, threads_per_job: int, clamped: bool, warnings: tuple[str, ...])`.

- `threads_per_job` given (explicit override): `jobs` unchanged, threads as
  given, no clamp. If `jobs × threads > budget.cores`, add an oversubscription
  warning (the user asked for it).
- Otherwise `floor = 1 if gpu_bound else 2` (a GPU-bound child mostly waits on
  the device; a CPU-only child needs ≥ 2 to be worth a slot). Module constants,
  not user-facing.
- `max_jobs = max(1, budget.cores // floor)`; `jobs' = min(jobs, max_jobs)`;
  `clamped = jobs' < jobs` → warning naming requested jobs, budget, usable
  cores and the effective jobs.
- `threads_per_job = max(1, budget.cores // jobs')`.

### `cpu_thread_limits(n: int | None)` — context manager, the applier

`None` → complete no-op (the plain path stays byte-identical). Otherwise, for
the duration of the block, and restoring on exit:

- `os.environ`: `OMP/MKL/OPENBLAS/NUMBA/VECLIB_MAXIMUM/NUMEXPR` thread vars →
  `min(existing, n)` (unset → `n`), so sidecars spawned inside the block
  (SLEAP service, Ultralytics predict subprocess) inherit the cap.
- `threadpoolctl.threadpool_limits(n)` (BLAS + OpenMP already loaded).
- `torch.set_num_threads(min(torch.get_num_threads(), n))` if torch is
  importable (never raises the count).
- `cv2.setNumThreads(min(cv2.getNumThreads(), n))` if cv2 is importable.
- Publishes `n` as the process CPU cap read by `current_cpu_cap() -> int | None`.

Lazily sized pools read `current_cpu_cap()`:

- **ORT:** new `session_options_for(threads: int | None)` in
  `runtime/onnx_providers.py` — `None` returns `None` (callers keep passing
  no `sess_options`, byte-identical); else `SessionOptions` with
  `intra_op_num_threads=n`, `inter_op_num_threads=1`. All five session sites
  pass `sess_options=session_options_for(current_cpu_cap())`, merged with any
  options the site already builds.
- **Crop-warp:** `_crop_warp_threads()` → `min(existing value, cap)` when a cap
  is set. Pool-vs-serial is documented byte-identical, so this is numerics-safe.
- **PyAV:** when a cap is set, `stream.thread_count = cap` beside
  `thread_type = "AUTO"`.
- **AprilTag:** effective threads `min(APRILTAG_THREADS, cap)` at both
  `APRILTAG_THREADS` read sites.

Sessions/pools created before the block keep their size; that is fine because
the CLI applies the cap before any tracking work, and fan-out children apply
it at startup.

### Fan-out children

- `build_child_env(..., threads_per_job)` now always receives a value from
  `plan_cpu` (CLI and GUI). It sets `HYDRA_CPU_THREADS=<n>` and the thread
  env vars to `min(existing, n)` (fixes the `setdefault` hole: a parent's
  `OMP_NUM_THREADS=64` no longer defeats the cap). Adds
  `VECLIB_MAXIMUM_THREADS` and `NUMEXPR_NUM_THREADS`.
- The child entry (`trackerkit/app.py` `track` command) reads
  `HYDRA_CPU_THREADS` (and `--cpu-budget` for a direct run) and wraps the run
  in `cpu_thread_limits(n)`.
- Known effect: with `OMP_NUM_THREADS` now set in children, ultralytics no
  longer forces it to 1, so child torch may run with up to `n` threads where
  it previously ran 1. Within budget, so not oversubscription; measured in
  verification.

## Surfaces

### CLI (`trackerkit track`)

- New `--cpu-budget N` (≥ 1): total cores hydra may use across all jobs.
- `--threads-per-job` help updated: explicit per-job override; disables the
  job clamp.
- `run_tracking_cli`: after the existing GPU slot rules, `plan_cpu` decides
  final jobs + threads; warnings logged at WARNING. Plain run (no fan-out):
  `n = budget.cores` only if `--cpu-budget` or `HYDRA_CPU_BUDGET` was given,
  else `None`.
- `trackerkit job run`: forward `--gpus`, `--jobs`, `--threads-per-job`,
  `--cpu-budget` into the generated `track` invocation instead of dropping
  them.

### GUI

- `TrackerConfig.cpu_budget: int = 0` (0 = auto) in `trackerkit/config/schemas.py`,
  round-tripped in `to_dict/from_dict`.
- Setup panel: a "CPU budget" spinbox (0 = auto, max `usable_cores()`) next to
  the batch-parallel controls, plus a read-only line showing the plan
  ("4 jobs × 3 threads of 12 cores" / clamp warning), recomputed from
  `plan_cpu` when jobs/GPUs/budget change.
- `_resolve_fanout_options` calls `plan_cpu` and passes `threads_per_job`
  and the clamped jobs into `FanoutOptions`.
- Single in-process GUI run: if `cpu_budget > 0`, the tracking worker runs
  inside `cpu_thread_limits(cpu_budget)`; 0 → unchanged.

## Out of scope

- Linux affinity partitioning (disjoint core sets per child) — can layer on
  this later.
- Live load-aware sizing.
- A cross-process / cross-user CPU lease.
- Training/DetectKit/ClassKit pools.

## Testing

- Unit: `usable_cores` (affinity + cgroup parsing via monkeypatched readers),
  `resolve_cpu_budget` (flag/env/auto/clamp/invalid env), `plan_cpu` (table of
  jobs × budget × gpu_bound × override), `build_child_env` (min-not-setdefault,
  new vars), `cpu_thread_limits` (env/torch/cv2 set and restored; `None` no-op),
  `session_options_for`, crop-warp/PyAV/AprilTag cap reads.
- CLI: `--cpu-budget` validation; fan-out options carry the planned jobs and
  threads; `job run` forwards the flags.
- GUI: schema round-trip; `_resolve_fanout_options` uses `plan_cpu`.
- Child probe: spawn a capped child process and assert observed
  `torch.get_num_threads()`, `threadpool_info()` and ORT intra-op ≤ cap.
- Equivalence: `fly_obb` + `worm_bgsub` matrix (current vs main) to prove
  the plain path is byte-identical.
