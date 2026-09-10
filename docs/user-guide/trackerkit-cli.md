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
forces the baseline onto every video. `--sahi-profile` applies to all.

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

## Parallel across GPUs

```bash
trackerkit track --video-list batch.txt --gpus auto          # one child per GPU nvidia-smi reports
trackerkit track --video-list batch.txt --gpus 0-3           # four GPUs
trackerkit track --video-list batch.txt --gpus 0,2,GPU-8f1a  # ordinals or UUID prefixes
trackerkit track --video-list batch.txt --jobs 3             # three children sharing the current device (CPU / Apple Silicon)
```

Each video becomes a child process `trackerkit track <video> --config <effective.json>`
pinned with `CUDA_VISIBLE_DEVICES=<uuid>`, so every child (and the SLEAP
service it starts) sees exactly one GPU. Output is identical to the sequential
run. Rules:

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
- `--threads-per-job N` (opt-in) caps OMP/MKL/OpenBLAS/Numba threads per child.
  Leave it off unless the host is oversubscribed; it can change floating-point
  reduction order in threaded kernels.
- Requirements in each child: `conda` on `PATH` for SLEAP pose, and the same
  `HYDRA_DATA_DIR`/`HYDRA_CONFIG_DIR` as the parent (inherited automatically).

The GUI exposes the same feature under **Batch › Run videos in parallel (one process per GPU)**.

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
