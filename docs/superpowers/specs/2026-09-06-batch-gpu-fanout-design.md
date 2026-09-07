# TrackerKit batch fan-out across GPUs — design

**Status:** approved design; plan at `docs/superpowers/plans/2026-09-06-batch-gpu-fanout.md`
**Date:** 2026-09-06
**Branch:** `feat/batch-gpu-fanout` (worktree `.worktrees/batch-fanout`)

## 1. Goal

Run a TrackerKit batch of N videos as N independent child processes, one per
GPU, from both the CLI (`trackerkit track`) and the GUI Batch panel, so a
multi-GPU host (diptera: 9× RTX 6000 Ada) processes a queue at full width.

Non-goals: splitting one video across several GPUs; changing anything about
the per-video pipeline. The child runs the exact sequential code path that
exists today, so the output is byte-identical by construction.

## 2. Why this shape

Every CUDA site in the pipeline pins `cuda:0` (direct OBB executors, TensorRT
export, pose, properties cache, SLEAP service). The only working device
selector is `CUDA_VISIBLE_DEVICES`, which the SLEAP service child inherits
because its `Popen` passes no `env=`. Process-per-video with a per-child mask
is therefore the one design that touches no inference code.

## 3. Correctness argument (one sentence)

GUI-parallel ≡ `trackerkit track --video-list <batch.txt> [--keystone-override] --gpus …`,
and each child ≡ `trackerkit track <video> --config <effective.json>`, which is
the current sequential path with the current per-video effective config; so
fan-out output equals sequential output as long as the planner that produces
`<effective.json>` is the same function the sequential loop uses.

## 4. Components

### 4.1 `trackerkit/batch_plan.py` — one planner, two consumers (Qt-free)

Extract the effective-config resolution out of `cli.run_tracking_cli`'s loop:

```python
@dataclass(frozen=True)
class BatchJobSpec:
    index: int                 # 1-based, keystone == 1
    video_path: str
    config: dict[str, Any]     # effective config dict handed to the child / session
    provenance: str            # "own-sidecar" | "explicit" | "keystone-baseline"

def plan_batch_jobs(
    video_paths, *, explicit_config_path=None,
    keystone_override=False, sahi_profile=None,
) -> list[BatchJobSpec]
```

Rules are exactly today's: `build_batch_video_plan` decides which config
source each video uses; video 1's resolved config becomes the keystone
baseline; `--sahi-profile` applies to every video. The GUI passes no in-memory
config: it saves the keystone sidecar first (as it does today), so the
planner sees exactly what `trackerkit track --video-list` sees. The sequential loop in `cli.py` is
rewritten to consume `plan_batch_jobs` and then call
`load_tracker_cli_session(video, config_data=spec.config)`.

Parity test: for representative batches (own sidecars, no sidecars,
explicit config, keystone override, sahi profile), the `config` the planner
yields per video equals what the pre-refactor loop passed to
`load_tracker_cli_session`. The pre-refactor loop is kept in the test as a
frozen reference implementation.

Duplicate rejection: the planner raises `ValueError` when two entries resolve
to the same `os.path.realpath`, or when two entries resolve to the same raw
CSV path via `_default_output_paths` (same stem in two read-only directories
redirected to one writable artifact base).

### 4.2 `runtime/cuda_devices.py` — physical GPU enumeration

```python
@dataclass(frozen=True)
class CudaDevice: index: int; uuid: str; name: str
def list_cuda_devices() -> list[CudaDevice]          # nvidia-smi --query-gpu=index,uuid,name
def resolve_gpu_selectors(selectors: Sequence[str]) -> list[CudaDevice]
```

`resolve_gpu_selectors` accepts ordinals (`0`), ranges (`0-3`), and unique
UUID prefixes (`GPU-8f…`), and the word `auto` (all devices). It resolves
against the *physical* list, never against the parent's own
`CUDA_VISIBLE_DEVICES`, and rejects duplicates and MIG devices. Children are
pinned by UUID: `CUDA_VISIBLE_DEVICES=<uuid>`. When `nvidia-smi` is absent
(MPS/CPU hosts) it returns `[]` and any `--gpus` request is an error.

**Amended 2026-09-07 (fix wave, I2):** that last decision now lives in one pure
`batch_fanout.decide_gpu_slots(selectors, devices, host_has_cuda)` shared by the
CLI and the GUI, and it distinguishes three cases: no devices on a CUDA-capable
host is an error naming `nvidia-smi` (it is missing/masked/timing out, and
running unpinned would put every child on `cuda:0`); on a host with no CUDA at
all an explicitly named device is still an error, but `auto` is best-effort and
runs unpinned. The GUI previously swallowed the whole resolution when
`nvidia-smi` returned nothing.

### 4.3 `trackerkit/batch_fanout.py` — scheduler (Qt-free)

```python
@dataclass
class FanoutOptions:
    gpus: list[CudaDevice]          # may be empty → jobs share current visibility
    jobs: int                       # concurrent slots; default len(gpus) or 1
    threads_per_job: int | None     # None → do not set thread-cap env vars
    log_level: str = "INFO"
    run_dir: Path | None = None     # where effective configs + child logs go

@dataclass
class FanoutJobResult: spec, gpu, returncode, success, log_path, summary_lines, error
@dataclass
class FanoutResult: jobs: list[FanoutJobResult]; cancelled: bool
    @property def success(self) -> bool

class FanoutEvents(Protocol):
    def job_started(self, spec, gpu, log_path): ...
    def job_progress(self, spec, percent, message): ...
    def job_log(self, spec, line): ...
    def job_finished(self, result): ...

def run_batch_fanout(specs, options, *, events, should_stop) -> FanoutResult
```

Behaviour:

- **Child command:** `[sys.executable, "-m", "hydra_suite.trackerkit.app",
  "--log-level", L, "track", video, "--config", <run_dir>/job_<index>_config.json]`.
  The child never sees `--gpus`/`--jobs`, so it cannot recurse.
- **Child env:** inherit `os.environ`, then set `CUDA_VISIBLE_DEVICES=<uuid>`
  when a GPU is assigned, `PYTHONUNBUFFERED=1`, `KMP_DUPLICATE_LIB_OK=TRUE`.
  If `threads_per_job` is set, also set `OMP_NUM_THREADS`, `MKL_NUM_THREADS`,
  `OPENBLAS_NUM_THREADS`, `NUMBA_NUM_THREADS` **only where the parent has not
  already set them**. Everything else (conda on PATH, `HYDRA_DATA_DIR`,
  `HYDRA_CONFIG_DIR`, `HYDRA_SLEAP_*`) propagates untouched.
- **Slots:** a GPU is a slot when `gpus` is non-empty (`jobs` is clamped to
  `len(gpus)`); otherwise there are `jobs` anonymous slots. Jobs are launched
  in batch order as slots free up.
- **Output:** child stdout+stderr is pumped by a reader thread
  (`process_supervisor.BoundedLineBuffer`/`pump_stdout`) into a per-job log
  file `<stem>_logs/<stem>_fanout_<timestamp>.log` (via
  `build_video_log_dir`/`choose_writable_artifact_base_dir`) and to
  `events.job_log`. Lines matching `[track forward] NN%`, `[track backward] NN%`
  or `[post] NN%` are also parsed into `events.job_progress`. The final
  `Tracker CLI completed: …` line supplies `summary_lines`.
- **Failure policy:** running jobs finish; no new jobs launch after a
  failure; exit is non-zero if any job failed. A per-video pass/fail table is
  printed (CLI) or shown (GUI).
- **Cancellation:** `should_stop()` polled every 200 ms. On stop: SIGINT to
  every live child (the child's own SIGINT handler requests a clean engine
  stop), `terminate()` after 10 s, `kill()` after 5 more. Children are started
  with `start_new_session=True` so a terminal Ctrl-C reaches only the parent,
  which forwards deliberately. The CLI installs the same SIGINT→stop-flag
  handler that `headless_tracking` uses.
- **Not using `SupervisedSidecar`:** its leases, admission and RSS watchdog
  are training machinery; the fan-out needs only the line pump.

### 4.4 `runtime/artifact_lock.py` — blocking cross-process build lock

`HeavyJobLease` is non-blocking, so add:

```python
@contextmanager
def artifact_build_lock(target: Path, *, timeout_s: float | None = None) -> Iterator[None]
```

A blocking `flock` (msvcrt fallback) on `<target>.lock` in the target's
parent directory. Callers use double-checked locking: check the artifact,
acquire, re-check, build. Applied at:

| Site | Why |
|---|---|
| caller of `_export_artifact` in `core/inference/runtime_artifacts.py` | ultralytics writes `<stem>.engine` beside the `.pt` before the copy to `<stem>_bN.engine` |
| whole check/rmtree/mkdir/export block of `auto_export_sleap_model` (`pose/backends/sleap.py`) | lock file beside `model_path.parent`, never inside `export_dir` (it gets rmtree'd) |
| callers of `build_trt_engine_from_onnx` (`pose/runtime/tensorrt_engine.py`) | native SLEAP TRT engine build |
| `ort.InferenceSession(...)` when providers include `TensorrtExecutionProvider` (`pose/runtime/onnx_session.py`, `classification/backend.py`) | the ORT TRT-EP engine cache dir is shared per machine |

Locks change no numerics; the standard equivalence matrix still runs because
these files are on hot paths.

### 4.5 CLI surface (`trackerkit/app.py`, `trackerkit/cli.py`)

New `track` flags:

- `--gpus <list>`: `0,1,2`, `0-8`, UUID prefixes, or `auto`.
- `--jobs N`: concurrent jobs (default `len(gpus)`, else 1).
- `--threads-per-job N`: opt-in thread caps for the child (see §6).

Gating rule: fan-out engages iff `--gpus` is given or `--jobs > 1`. Otherwise
`run_tracking_cli` runs the in-process sequential path, byte-for-byte
untouched, so `tools/equivalence/runner.py` (which calls `run_tracking_cli`
directly) stays valid. `run_tracking_cli` gains keyword-only `gpus`, `jobs`,
`threads_per_job` parameters and dispatches to `run_batch_fanout` when the
rule fires. The CLI prints a final table: index, video, GPU, status, wall
time, log path.

### 4.6 Session state (`trackerkit/config/schemas.py`)

`TrackerConfig` gains `batch_parallel: bool = False`, `batch_parallel_jobs:
int = 0` (0 = auto), `batch_parallel_gpus: str = "auto"`. These are session
state like `batch_videos`; they are **never** emitted by
`ConfigOrchestrator.build_config_dict()`, so sidecar configs, cache keys and
child invocations are unchanged.

### 4.7 GUI surface (`trackerkit/gui/…`)

- **SetupPanel Batch group:** a "Parallel" row under the keystone-override
  checkbox: `chk_batch_parallel` ("Run videos in parallel (one process per
  GPU)"), `spin_batch_parallel_jobs` (default = CUDA device count, else 1),
  `edit_batch_parallel_gpus` (prefilled `auto`, hidden on non-CUDA hosts).
  Wired to the schema fields above.
- **`gui/workers/batch_fanout_worker.py`:** `BatchFanoutWorker(BaseWorker)`
  implements `FanoutEvents` by emitting `job_started(int, str, str)`,
  `job_progress(int, int, str)`, `job_log(int, str)`,
  `job_finished(int, bool, str)`, `fanout_finished(object)`; `cancel()` sets
  the stop flag that `should_stop` reads.
- **`gui/dialogs/batch_fanout_dialog.py`:** `BatchFanoutDialog(BaseDialog)`,
  non-modal, one row per job (video, GPU, status, progress bar, last
  message), a Cancel button, and a final pass/fail summary with log paths.
- **`TrackingOrchestrator.start_tracking`:** after the existing batch
  confirmation and `save_config` call, if `g_batch` and `chk_batch_parallel`
  are both checked: call `plan_batch_jobs(batch_videos,
  keystone_override=chk…)` (the keystone sidecar was just saved), resolve GPUs,
  start the worker, open the dialog, and enter the `"tracking"` UI state.
  `stop_tracking` cancels the worker. On finish, restore the UI and show the
  summary. No frame preview and no live FPS during fan-out: children are
  headless.

### 4.8 Docs

New `docs/user-guide/trackerkit-cli.md` (the CLI has no user-facing page
today) covering `trackerkit track`, `--video-list`, keystone rules, and the
fan-out flags with a diptera-style example. Configuration reference gains the
three session fields. `CLAUDE.md` gains one line pointing at the gating rule.

## 5. SLEAP interfaces

- Each child starts its own SLEAP service (`conda run -n <pose_sleap_env>`)
  on a free loopback port with pid+uuid-named temp files; ten children run
  ten services. The service inherits the child's `CUDA_VISIBLE_DEVICES`, so
  sleap-nn sees one GPU as `cuda:0`. No SLEAP code changes.
- The per-child env preflight (`_sleap_env_preflight`) runs once per child;
  acceptable cost.
- `gpu_fast` SLEAP export and TRT-EP session creation are covered by §4.4.
- The child needs `conda` on PATH, exactly as the GUI does today.

## 6. Thread caps are a gate risk

`NUMBA_NUM_THREADS`/`OMP_NUM_THREADS` can change reduction order in `prange`
kernels and threaded BLAS, which can break byte-identity against an uncapped
sequential run. Caps are therefore **opt-in** (`--threads-per-job`, off by
default) until the verification below proves identity with caps engaged; if
identity holds, a follow-up may default them to `cpu_count // jobs`.

**Result (2026-09-06, MPS / `hydra-mps`, Apple Silicon): caps are
identity-safe on this platform.** `tools/equivalence/fanout_gate.sh` was run
twice over `fly_obb`, `worm_bgsub`, `ant_obb_sleap`, `ant_cnn_identity` --
once with `--jobs 2` and once with `--jobs 2 --threads-per-job 4`. Both runs
produced **16/16 byte-identical CSVs** against the same *uncapped* in-process
sequential baseline (`_tracking_forward` / `_backward` / `_final` /
`_final_with_individual` per clip; every row count > 1).

The caps run is **not vacuous**: `build_child_env` applies the cap variables
with `setdefault`, so a parent that already exported them would silently
suppress the cap. The gate prints the inherited thread-cap environment before
running, and on both boxes it printed `(none set)` -- so the children really
did run with `OMP_NUM_THREADS` / `MKL_NUM_THREADS` / `OPENBLAS_NUM_THREADS` /
`NUMEXPR_NUM_THREADS` / `NUMBA_NUM_THREADS` = 4 while the sequential run they
were compared against was uncapped.

Caps nevertheless **stay opt-in**: this is one platform's evidence (the CUDA
box was gated uncapped only), so defaulting them to `cpu_count // jobs`
remains a separate change needing its own gate.

## 7. Verification

1. Unit tests (pytest, `PYTHONPATH=<wt>/src`): planner parity and dedup;
   selector parsing (ordinals, ranges, UUID prefix, `auto`, MIG rejection);
   child env construction (pin, unbuffered, caps only when unset); scheduler
   with a fake child script (progress parsing, slot reuse, failure policy,
   cancellation escalation, log files); artifact lock contention across two
   real processes; schema round-trip; worker callback→signal mapping.
2. **Fan-out vs sequential, MPS, this box:** `fly_obb`, `worm_bgsub`,
   `ant_obb_sleap`, `ant_cnn_identity` fixtures run sequentially and with
   `--jobs 2`; `cmp` both `_forward.csv` and `_tracking_final.csv`; row
   counts > 1; conda active. Two concurrent SLEAP service children is the
   SLEAP interface test.
3. Repeat (2) with `--threads-per-job` engaged to decide §6.
4. **Standard equivalence matrix (MPS)** for the lock touches in
   `runtime_artifacts.py`, `sleap.py`, `onnx_session.py`, `backend.py`.
5. **CUDA on mehek:** `--gpus 0 --jobs 1` vs sequential, and `--jobs 2` on the
   one GPU; standard matrix with `RUNTIME=cuda`.
6. True multi-GPU pinning is unverifiable until diptera has an environment;
   the UUID path is covered by unit tests plus the existing training
   precedent (`ultralytics_supervisor.py`, `sam3_lora/train.py`).

## 8. Out of scope

Sharding one video across GPUs; GPU memory admission; a job queue that
survives the parent; Windows testing of the fan-out (the lock has an msvcrt
fallback, the scheduler uses `start_new_session` only on POSIX).
