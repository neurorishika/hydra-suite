# TrackerKit CPU oversubscription protection — design

**Status:** approved in chat 2026-10-12; rev 3 after two adversarial (Fable) reviews (spec + plan); implementation on `feat/cpu-oversubscription`.

## Problem

TrackerKit never caps its own CPU thread pools. The only cap is the opt-in
`trackerkit track --threads-per-job`, which sets four BLAS env vars by
`setdefault` (so a parent shell's value wins) and reaches none of the pools
that actually run full-width. Concretely (map taken 2026-10-12):

| Pool | Where | Size today |
|---|---|---|
| ONNX Runtime intra-op | 5 `InferenceSession` sites, no `sess_options` | ORT default (~1/physical core) **per session** |
| Crop-warp `ThreadPoolExecutor` | `core/inference/stages/crops.py:_crop_warp_threads` | `HYDRA_CROP_WARP_THREADS` else `min(8, cpu_count)` |
| **cv2.VideoCapture (FFmpeg) decode — the tracking decoder** | `core/tracking/worker.py:869,1738,1989,2031`, `core/inference/sources.py:129`, `core/post/media_export.py:945` | **31 threads** per open capture on `fly_obb` (measured); `cv2.setNumThreads` does not reach it |
| PyAV libx264 encode | `utils/video_encoder.py` `_open()` (`pyav_software`), used during tracking render + final media export | **23 threads** (measured) |
| PyAV decode (media export only) | `utils/video_decoder.py` (`thread_type="AUTO"`) | FFmpeg's choice (~cores) |
| AprilTag detector | `APRILTAG_THREADS` param | 4 |
| numba `parallel=True` bg-sub kernel | `core/background/model.py:47` | numba pool = cores; numba is imported at `trackerkit.app` import, so a later env var is dead; `numba.set_num_threads` is **thread-local** |
| torch intra-op | never set | **import-order dependent**: 1 if `ultralytics` (sets `OMP_NUM_THREADS=1` when unset) is imported before the first CPU torch op, else ~physical cores (12 here). Interop pool = cores |
| OpenCV compute | never set (ultralytics calls `cv2.setNumThreads(0)`) | 1 once ultralytics is imported. On the macOS GCD build `setNumThreads(n>1)` is inert — **not a usable lever** |
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
   `os.cpu_count()` (macOS/Windows; on Apple Silicon this counts E-cores too —
   accepted).
2. cgroup v2 (Linux only): resolve this process's cgroup from the `0::<path>`
   line of `/proc/self/cgroup`, then walk from `/sys/fs/cgroup/<path>` up to the
   mount root reading each `cpu.max`; take `min(previous, ceil(quota/period))`
   over every finite quota. (A bare host's root cgroup has no `cpu.max`; the
   quota can sit on any ancestor.) cgroup v1 (`cpu.cfs_quota_us`) is ignored.
   Read/parse failures are ignored.

### `resolve_cpu_budget(requested: int | None) -> CpuBudget`

- `requested` comes from `--cpu-budget` / the GUI spinbox; `None`/`0` falls back
  to the env var `HYDRA_CPU_BUDGET` (a persistent per-box default for shared
  machines), then to `usable_cores()`.
- A request above `usable_cores()` is clamped down with a warning (the budget
  can only lower).
- `CpuBudget(cores: int, usable: int, source: "flag"|"env"|"auto", warnings: tuple[str, ...])`.
- Invalid `HYDRA_CPU_BUDGET` (non-int, < 1) → warning, ignored.

**Precedence inside a process:** `HYDRA_CPU_THREADS` (set by fan-out for its
child, or exported in the shell) is a ceiling: an explicit `--cpu-budget` / GUI
budget can only lower it (`min`), and `HYDRA_CPU_BUDGET` never overrides it.
Without `HYDRA_CPU_THREADS`: `--cpu-budget` > `HYDRA_CPU_BUDGET`. `build_child_env`
strips `HYDRA_CPU_BUDGET` from the child env so a child never re-resolves the
whole-box budget. `HYDRA_CPU_BUDGET` is local-only; `--cpu-budget` is the
only form forwarded by remote `job run`.

### `plan_cpu(*, jobs: int, n_videos: int, budget: CpuBudget, gpu_bound: bool, threads_per_job: int | None) -> CpuPlan`

Pure. Returns `CpuPlan(jobs: int, threads_per_job: int, clamped: bool, warnings: tuple[str, ...])`.

- `jobs` is the **resolved** slot count after the existing GPU slot rules
  (`cli.py` slot rules / `run_batch_fanout`'s `min(requested, len(gpus))`;
  the GUI's `jobs=None` "one per GPU" is resolved to an int before calling).
  The caller writes `CpuPlan.jobs` back into `FanoutOptions.jobs`.
- Threads are divided by `active = min(jobs', n_videos)` so 2 videos on 8
  slots don't waste the budget.
- `gpu_bound = bool(devices)` (explicit `--gpus` / GUI GPU selection). A
  `--jobs N` run without `--gpus` is treated as CPU-bound (floor 2).

- `threads_per_job` given (explicit override): `jobs` unchanged, threads as
  given, no clamp. If `jobs × threads > budget.cores`, add an oversubscription
  warning (the user asked for it).
- Otherwise `floor = 1 if gpu_bound else 2` (a GPU-bound child mostly waits on
  the device; a CPU-only child needs ≥ 2 to be worth a slot). Module constants,
  not user-facing.
- `max_jobs = max(1, budget.cores // floor)`; `jobs' = min(jobs, max_jobs)`;
  `clamped = jobs' < jobs` → warning naming requested jobs, budget, usable
  cores and the effective jobs.
- `threads_per_job = max(1, budget.cores // min(jobs', n_videos))`.

### Process-level cap: `set_process_cpu_cap(n: int | None)` + `bind_thread_cpu_cap()`

A scoped context manager is **not** used: torch, threadpoolctl and
`os.environ` are process-global, so nested/concurrent enter/exit from
different GUI QThreads would capture already-capped values and un-cap a
still-running worker on exit. Instead there is one process-wide cap, set
idempotently (last setter wins, guarded by a lock):

- `set_process_cpu_cap(None)` when no cap was ever set → complete no-op (the
  plain path stays byte-identical). `set_process_cpu_cap(None)` after a cap →
  restores the originals recorded on the first set.
- `set_process_cpu_cap(n)`:
  - `os.environ`: `OMP/MKL/OPENBLAS/VECLIB_MAXIMUM/NUMEXPR_NUM_THREADS`
    and `OPENCV_FFMPEG_THREADS` → `min(original, n)` (unset → `n`).
    `NUMBA_NUM_THREADS` is written only while numba is not yet imported —
    which never holds at the real entry points (importing `trackerkit.cli`
    loads numba) — so in-process numba is capped per thread by
    `bind_thread_cpu_cap()`, and children get `NUMBA_NUM_THREADS` from
    `build_child_env`. Sidecars
    spawned afterwards (SLEAP service, Ultralytics predict subprocess)
    inherit it, and **`OPENCV_FFMPEG_THREADS` is read by OpenCV at each
    `VideoCapture` open**, so it caps every tracking decoder without touching
    the call sites (decoded frames measured byte-identical).
  - `threadpoolctl.threadpool_limits(n)` held (BLAS; note its OpenMP limit is
    per calling thread, so it does NOT cap torch on worker threads).
  - `torch.set_num_threads(min(original, n))` if torch is already imported
    (never raises the count). This process-global call is the load-bearing
    torch cap. Interop is not touched (it can only be set
    before first use).
  - OpenCV compute threads are not touched (not a lever, see table).
  - Publishes `n` for `current_cpu_cap() -> int | None`.
- `bind_thread_cpu_cap()` — called at the top of every thread that runs
  tracking work (the engine thread, the inference pipeline's
  decoder/detector/producer threads, and `run_bgsub_batch`, which is where
  the numba bgsub kernel actually runs): `numba.set_num_threads(min(numba.get_num_threads(), cap))`
  (numba's setting is thread-local). No-op when no cap.

The cap is **per pool**, not a per-process hard ceiling: a child runs
decoder + ORT + crop-warp + torch + prefetch concurrently, so peak runnable
threads can be a small multiple of `threads_per_job`. That is accepted; the
goal is to remove the N×cores blow-up, not to do exact accounting.

Lazily sized pools read `current_cpu_cap()`:

- **ORT:** new `session_options_for(threads: int | None)` in
  `runtime/onnx_providers.py` — `None` returns `None` (callers keep passing
  no `sess_options`, byte-identical); else `SessionOptions` with
  `intra_op_num_threads=n`. Tracking-relevant sites pass
  `sess_options=session_options_for(current_cpu_cap())`:
  `core/inference/direct_executors.py:497`,
  `core/individual/classification/backend.py:893,895,904,938` (938 is a CPU
  probe session that otherwise holds a full default pool until GC),
  `core/individual/pose/runtime/onnx_session.py:124,128`,
  `core/individual/pose/backends/yolo.py:161`. Out of scope: `trt_probe.py`
  (separate child process; inherits env only), `training/tiny_model.py`.
  (Measured: outputs byte-identical across intra 1/4/default on the OBB
  fixture, CPU EP.)
- **libx264 encoder:** when a cap is set, `stream.thread_count = cap` in
  `VideoEncoder._open` for the `pyav_software` path (before first encode).
  The `cv2.VideoWriter` fallback is not controllable; NVENC/VideoToolbox do
  not carry the CPU cost.
- **Crop-warp:** `_crop_warp_threads()` → `min(existing value, cap)` when a cap
  is set. Pool-vs-serial is documented byte-identical, so this is numerics-safe.
- **PyAV decode (media export):** when a cap is set, `stream.thread_count =
  cap` beside `thread_type = "AUTO"`.
- **AprilTag:** effective threads `min(APRILTAG_THREADS, cap)` at both
  `APRILTAG_THREADS` read sites.

Sessions/pools created before the cap is set keep their size; that is fine
because the CLI and fan-out children set the cap before any tracking work, and
the GUI sets it before starting a tracking/post-processing/export worker.

### Fan-out children

- `build_child_env(..., threads_per_job)` now always receives a value from
  `plan_cpu` (CLI and GUI). It sets `HYDRA_CPU_THREADS=<n>`, strips
  `HYDRA_CPU_BUDGET`, and sets the thread env vars to `min(existing, n)`
  (fixes the `setdefault` hole: a parent's `OMP_NUM_THREADS=64` no longer
  defeats the cap). Adds `VECLIB_MAXIMUM_THREADS`, `NUMEXPR_NUM_THREADS` and
  `OPENCV_FFMPEG_THREADS`. Because these are in the env before the child
  imports anything, numba/libomp/OpenCV-FFmpeg pick them up natively.
- The child entry (`trackerkit/app.py` `track` command) resolves the cap by
  the precedence above and calls `set_process_cpu_cap(n)` +
  `bind_thread_cpu_cap()` before tracking.
- Torch thread count in a child is import-order dependent today (1 or ~cores);
  with the cap it is `≤ n`. The child probe logs observed values rather than
  asserting a specific torch count.

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
- `trackerkit calibrate --cpu-budget N` (≥ 1): same single-run cap
  (`single_run_cpu_cap`), applied before the session/inference config is
  built. `job run --calibrate --cpu-budget N` (and `job calibrate
  --cpu-budget N`, local or remote) forwards it to the calibrate leg. Every
  calibration trial runs in a sidecar child: a capped parent launches it with
  `capped_child_env` (`HYDRA_CPU_THREADS=cap`, `HYDRA_CPU_BUDGET` stripped,
  thread env vars lowered) and the child applies the inherited cap; an
  uncapped parent launches it exactly as before. The
  tuning profile key has no cap term: a profile is measured under whatever
  cap is in force.

### GUI

- `TrackerConfig.cpu_budget: int = 0` (0 = auto) in `trackerkit/config/schemas.py`,
  round-tripped in `to_dict/from_dict`.
- Setup panel: a "CPU budget" spinbox (0 = auto, max `usable_cores()`) next to
  the batch-parallel controls, plus a read-only line showing the plan
  ("4 jobs × 3 threads of 12 cores" / clamp warning), recomputed from
  `plan_cpu` when jobs/GPUs/budget change.
- `_resolve_fanout_options` calls `plan_cpu` and passes `threads_per_job`
  and the clamped jobs into `FanoutOptions`.
- Single in-process GUI run: before starting a `TrackingWorker` or
  `SessionWorker` (post-tracking + exports), the orchestrator calls
  `set_process_cpu_cap(single_run_cpu_cap(cpu_budget or None))` — so 0 falls
  back to `HYDRA_CPU_BUDGET`, else `None` (no-op if never set, restore if
  previously set). The same cap is also applied eagerly at GUI startup, on
  every CPU-budget spinbox change (deferred while a run or fan-out is
  active), and again when a fan-out batch ends, so every heavy GUI path
  (calibration, parameter helpers, previews) runs under it, not only the
  tracking workers. Thread-local binding happens in core (engine + pipeline
  threads), so it covers GUI and CLI alike.
- Before starting a GUI fan-out, the orchestrator calls
  `set_process_cpu_cap(None)`, so a previous single-run cap left in
  `os.environ` cannot shrink the children (`build_child_env` takes `min`).

## Out of scope

- Linux affinity partitioning (disjoint core sets per child) — can layer on
  this later.
- Live load-aware sizing.
- A cross-process / cross-user CPU lease.
- Training/DetectKit/ClassKit pools.

## Testing

- Dependency: declare `threadpoolctl` in `pyproject.toml` (today only
  transitive via scikit-learn; `constraints/base.txt` already pins 3.6.0).
- Existing test flips by design: `tests/test_trackerkit_batch_fanout.py`
  asserts the parent's `OMP_NUM_THREADS` wins; it becomes `min`.
- Unit: `usable_cores` (affinity + cgroup ancestor walk via monkeypatched readers),
  `resolve_cpu_budget` (flag/env/auto/clamp/invalid env), `plan_cpu` (table of
  jobs × budget × gpu_bound × override), `build_child_env` (min-not-setdefault,
  new vars), `set_process_cpu_cap` (env/threadpoolctl/torch set, idempotent, restored by
  `None`; never-set `None` is a no-op), `bind_thread_cpu_cap` (numba,
  thread-local), `session_options_for`, crop-warp/PyAV decode/encoder/AprilTag
  cap reads, `OPENCV_FFMPEG_THREADS` reaching `cv2.VideoCapture` (thread count
  of an opened capture drops).
- CLI: `--cpu-budget` validation; fan-out options carry the planned jobs and
  threads; `job run` forwards the flags.
- GUI: schema round-trip; `_resolve_fanout_options` uses `plan_cpu`.
- Child probe: spawn a capped child process and assert `threadpool_info()`
  and numba ≤ cap, and that the process's OS thread count after opening a
  capture + ORT session is far below the uncapped baseline; log torch.
- Equivalence: `fly_obb` + `worm_bgsub` matrix (current vs main) to prove
  the plain path is byte-identical.
