# TrackerKit command line

`trackerkit track` runs the same tracking pipeline as the GUI, headless.

## One video

```bash
trackerkit track video.mp4                     # uses video_config.json beside the video
trackerkit track video.mp4 --config my.json    # explicit config
```

To package a batch, its models, and its config for a different compute box
(zero model registration, outputs pulled back beside the originals), see
[Portable tracking jobs](trackerkit-jobs.md) (`trackerkit job pack|push|preflight|run|pull`).

## A batch

```bash
trackerkit track a.mp4 b.mp4 c.mp4             # a.mp4 is the keystone
trackerkit track --video-list batch.txt        # one absolute path per line, keystone first (the GUI's Export List format)
trackerkit track --video-list batch.txt --keystone-override
```

Keystone rules: the first video's config is the baseline. Later videos use
their own `<stem>_config.json` when present, otherwise the baseline.
`--keystone-override` (or an explicit `--config` on a multi-video batch)
forces the baseline onto every video. `--sahi-profile` applies to all (see
[SAHI calibration profiles](detectkit-sahi-calibration.md) and
[SAHI Settings](sahi-settings.md)).

Side-output retargeting: a video's own `<stem>_config.json` is used verbatim,
including whatever `video_output_path` it names. A video that inherits the
keystone's config instead of having its own, or any video in a **multi-video**
batch run with an explicit `--config`, gets its side-output paths (annotated
video, CSV, `file_path`) re-derived beside its own video -- otherwise every
inheriting or `--config`-driven video in the batch would collide on the one
render path the config names. A **single** video run with `--config` is the
one case that is not a batch, so it keeps whatever paths that config says.

Videos run one after another in this process. Output per video depends on
whether backward tracking is enabled in the config:

- Backward tracking **on**: `<stem>_tracking_forward.csv` and
  `<stem>_tracking_backward.csv` (the raw per-pass CSVs) plus the merged
  `<stem>_tracking_final.csv`.
- Backward tracking **off**: `<stem>_tracking.csv` (raw) and
  `<stem>_tracking_forward_processed.csv`.

When identity or pose ran, the final CSV also gets a
`..._with_individual.csv` sibling (e.g. `<stem>_tracking_final_with_individual.csv`)
carrying the identity and `PoseKpt_*` columns.

## Annotated video resolution

```bash
trackerkit track video.mp4 --video-scale 1.0    # full-resolution annotated video
trackerkit track --video-list batch.txt --video-scale 0.25
```

`--video-scale` (0.1-1.0) overrides the config's `video_output_scale` for
every video in the batch, including fanned-out GPU children and
`trackerkit job pack`. Without it the config's value is used, and a config
without the key renders at `0.5`. An out-of-range value -- on the command
line or in a config -- stops the run before tracking starts. It only affects
the final annotated video, not tracking or the CSVs.

## Parallel across GPUs

```bash
trackerkit track --video-list batch.txt --gpus auto          # one child per GPU nvidia-smi reports
trackerkit track --video-list batch.txt --gpus 0-3           # four GPUs
trackerkit track --video-list batch.txt --gpus 0,2,GPU-8f1a  # ordinals or UUID prefixes
trackerkit track --video-list batch.txt --jobs 3             # three children sharing the current device (CPU / Apple Silicon)
```

Each video becomes a child process `trackerkit track <video> --config <effective.json>`
pinned with `CUDA_VISIBLE_DEVICES=<uuid>`, so every child (and the SLEAP
service it starts) sees exactly one GPU. Tracking output (the CSVs) was
byte-identical to the sequential run in our tests, including with a CPU cap
(`--jobs 2 --cpu-budget 4`); threaded numeric kernels may in principle round
differently under a CPU cap (see [CPU budget](#cpu-budget), which also covers
rendered videos). Rules:

- Fan-out engages only when `--gpus` is given or `--jobs > 1`.
- `--gpus auto` is best effort: on a host with no CUDA at all it simply runs
  unpinned. A GPU named explicitly (`--gpus 0`) that `nvidia-smi` does not
  report is an error, as is *any* selection on a CUDA-capable host whose
  `nvidia-smi` reports nothing -- there, running unpinned would put every child
  on `cuda:0`.
- `--jobs` defaults to one slot per selected GPU when `--gpus` is given
  (else 1, which stays sequential and in-process), and is clamped to the
  number of selected GPUs.
- A failed video stops new launches; running videos finish. Exit code is 1
  if any video failed or the run was cancelled; the per-video table still
  shows which videos completed.
- Ctrl-C asks every child to stop cleanly, then terminates stragglers.
- Per-child logs: `<video dir>/<stem>_logs/<stem>_fanout_<timestamp>.log`.
- Every child gets a CPU thread cap (`budget // jobs` threads), so N children
  never oversubscribe the machine. See [CPU budget](#cpu-budget) below;
  `--threads-per-job N` sets the per-child count explicitly.

### On a shared machine

`--gpus auto` means *every GPU `nvidia-smi` reports* — it does **not** skip GPUs
that are busy. On a multi-user box it will happily start children on top of
someone else's job. Check first and name the free GPUs explicitly:

```bash
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
trackerkit track --video-list batch.txt --gpus 1,2,4,6      # the idle ones
```

Two related habits on a shared box:

- Single-process runs (no fan-out) always use `cuda:0`, which may be the busiest
  GPU. Pick a device with
  `export CUDA_VISIBLE_DEVICES=$(nvidia-smi --query-gpu=uuid --format=csv,noheader -i 4)`.
- Give hydra a share of the CPU, not the whole box: `--cpu-budget 32` (or
  `export HYDRA_CPU_BUDGET=32` in your shell profile) splits 32 threads
  across the parallel children, and caps each thread pool of a single run at
  32. The cap is per pool, not a hard core limit; see
  [CPU budget](#cpu-budget).
- Requirements in each child: `conda` on `PATH` for SLEAP pose, and the same
  `HYDRA_DATA_DIR`/`HYDRA_CONFIG_DIR` as the parent (inherited automatically).

The GUI exposes the same feature under **Batch › Run videos in parallel (one process per GPU)**.

### CPU budget

Each tracking process sizes its thread pools (PyTorch, OpenMP/BLAS, Numba,
OpenCV's FFmpeg video decoder, ONNX Runtime, PyAV video decode/encode) from
the machine's core count. N parallel children would each do that, multiplying
the thread count by N. The CPU budget stops this.

```bash
trackerkit track --video-list batch.txt --gpus 0-7 --cpu-budget 32   # 8 jobs x 4 threads
trackerkit track --video-list batch.txt --jobs 16                     # clamped to fit the usable cores
trackerkit track one.mp4 --cpu-budget 8                               # cap a single run
export HYDRA_CPU_BUDGET=32                                             # same as --cpu-budget, for every run
```

- **The budget** is the total number of cores hydra may use across all of its
  parallel jobs. It comes from `--cpu-budget N`, else the `HYDRA_CPU_BUDGET`
  environment variable, else every usable core (the CPU affinity mask and any
  container/cgroup CPU quota are respected). A budget larger than the usable
  cores is clamped down to them with a warning. An invalid `HYDRA_CPU_BUDGET`
  (`abc`, `0`) is ignored with a warning; `--cpu-budget` must be at least 1.
- **Parallel runs** always use the budget, even with no flag: each child gets
  `budget // jobs` threads. If there are fewer videos than slots, the active
  children share the whole budget (2 videos on 8 GPUs with a budget of 32 get 16
  threads each).
- **The job clamp:** when the budget cannot give every requested job its
  minimum (1 thread per GPU job, 2 per CPU-only job), fewer jobs run at a time
  and a warning says so, e.g. `Requested 16 parallel job(s) but the CPU budget
  is 12 core(s) (12 usable); running 6 at a time.`
- **`--threads-per-job N`** is the override: every child gets at most N
  threads and the job clamp is turned off. If `jobs x N` exceeds the budget you
  get a warning, but the run goes ahead (it may oversubscribe).
- **A single run** (no `--gpus`, `--jobs` 1) is capped only when you ask for it,
  with `--cpu-budget` or `HYDRA_CPU_BUDGET`. Without either, nothing changes.
  The same applies to `trackerkit calibrate` (it accepts `--cpu-budget` too),
  and the GUI applies its CPU budget to everything it runs (tracking,
  calibration, parameter helpers, previews) as soon as it starts.
- **The cap is per thread pool, not per process.** A capped process sizes each
  of its pools (video decoder, ONNX Runtime, PyTorch, BLAS/Numba, video
  encoder) to at most N threads, but those pools can run at the same time, so
  a child capped at N may briefly have several times N runnable threads
  (roughly 4-6 x N in a full tracking run). The budget stops the N-fold
  multiplication across parallel children; it is not an exact core limit.
- **`HYDRA_CPU_THREADS`** is the per-process cap a parallel run hands each
  child. If it is already set where you launch a run (exported in the shell,
  or a run nested inside another), it is a ceiling: the run's budget becomes
  the smaller of it and `--cpu-budget` (a single run included), and
  `HYDRA_CPU_BUDGET` does not raise it.
- **Pose with SLEAP:** the SLEAP sidecar process inherits the same per-job cap
  but is a separate process, so a pose job may use up to about twice its share.
- **`HYDRA_CPU_BUDGET` is local only.** It is read from the environment of the
  process that plans the run. `trackerkit job run user@box:...` runs over ssh,
  where your local shell's variables are not set; pass `--cpu-budget N` to
  `job run` instead (it is forwarded to `trackerkit track` on the box, along
  with `--gpus`, `--jobs` and `--threads-per-job`, and to the calibration
  step of `job run --calibrate`).

**What a cap changes in the outputs.** Tracking CSVs do not depend on the
encoder's thread count, so they are unaffected by a cap on video encoding.
Rendered annotated or oriented videos can differ byte-for-byte from an
uncapped run, because libx264's output depends on how many threads encode it;
they look the same. Tracking CSVs were byte-identical with and without a cap
in our tests; threaded numeric kernels (BLAS, OpenMP) may in principle reorder
floating-point reductions under a different thread count.

## Applying a tuned inference profile

```bash
trackerkit track video.mp4 --apply-tuned-inference     # use a validated profile if one exists
trackerkit track video.mp4 --no-apply-tuned-inference  # ignore any stored profile (default)
```

`track` only ever *looks up* a stored profile — it never measures. Produce a
profile with `trackerkit calibrate` (below) or the GUI's Calibrate button
first; see `docs/developer-guide/inference-autotuner.md` for what gets tuned
and how a profile is validated. `--inference-autotune-manual FIELD` (repeatable,
e.g. `--inference-autotune-manual pose_batch_size`) pins one tuning coordinate
at its configured value even when a profile exists for it.

## Calibrating

```bash
trackerkit calibrate --video video.mp4                          # uses video_config.json beside the video
trackerkit calibrate --video video.mp4 --config my.json         # explicit config
trackerkit calibrate --video video.mp4 --budget-seconds 1800    # wall-clock cap on the search
trackerkit calibrate --video video.mp4 --cpu-budget 8           # cap its CPU thread pools, like track
```

`calibrate` measures this box's own inference throughput against the given
video and config, and — if it finds a faster, output-equivalent vector —
persists a validated profile that any later `trackerkit track` run (or the
GUI) can apply via `--apply-tuned-inference`. It is the headless equivalent of
the GUI's Calibrate button, and builds engine params through the exact same
path `track` does, so the profile it writes is found under the same key a
`track` run looks up.

`calibrate` deliberately accepts neither `--gpus` nor `--jobs` — passing
either is a loud `unrecognized arguments` error, not a silent ignore.
Concurrent calibration on one box would measure contention between the
calibration children, not real single-job throughput, and would silently
produce a confidently wrong profile. Calibrate one video/config at a time,
sequentially, on the box the tuned profile is meant for.

A calibration that finds no faster vector still writes a validated profile
recording that outcome — a run's later `--apply-tuned-inference` lookup then
correctly resolves to the baseline instead of re-triggering a search. This is
expected and common on Apple Silicon (MPS): see
`docs/developer-guide/inference-autotuner.md` for why "no improvement" is not
a failure there.

## Example: nine-GPU host

```bash
conda activate hydra-cuda
trackerkit track --video-list /data/session_2026-09/batch.txt --gpus auto
```
