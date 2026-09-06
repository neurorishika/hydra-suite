# TrackerKit command line

`trackerkit track` runs the same tracking pipeline as the GUI, headless.

## One video

```bash
trackerkit track video.mp4                     # uses video_config.json beside the video
trackerkit track video.mp4 --config my.json    # explicit config
```

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

Videos run one after another in this process. Output per video:
`<stem>_tracking.csv` (raw) and `<stem>_tracking_forward_processed.csv`.

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

## Example: nine-GPU host

```bash
conda activate hydra-cuda
trackerkit track --video-list /data/session_2026-09/batch.txt --gpus auto
```
