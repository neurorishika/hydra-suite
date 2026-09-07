# Performance Tuning

## Highest-Impact Controls

- `resize_factor`: dominant speed/accuracy tradeoff (bg-sub only; see
  `Scale is a bgsub-only knob` in the tracking docs).
- Visualization/overlay toggles: large runtime impact in GUI mode.
- Detection mode and model size: major compute driver.
- Device selection (`runtime_tier`): critical for throughput.
- Batch sizing: far less impactful than it looks — see
  [Batching: what has actually been measured](#batching-what-has-actually-been-measured).

## Throughput Strategies

- For CPU-only workflows: use background subtraction and conservative visualization.
- For GPU workflows: verify backend, then tune model/batch settings.
- For long videos: enable caching to accelerate iterative backward/post-processing runs.

## Stability Strategies

- Use conservative thresholds first.
- Validate on representative frames before full-scale runs.
- Keep session logs and config snapshots for regression comparisons.

## Batching: what has actually been measured

A 2026-09-06 measurement programme (harness and raw data in
`tools/spikes/crossframe_batching/`, full write-up in
`docs/superpowers/specs/notes/2026-09-06-crossframe-batching-spike.md`)
tested the batching knobs end to end on CUDA (RTX 4090) and Apple Silicon.
The results are mostly negative, and the negatives are worth knowing before
anyone spends time here again.

### Downstream stages are the critical path, not detection

On a clip with head-tail + identity CNN + pose enabled, the downstream crop
consumers are **~95% of the critical path**. Detection runs on the pipeline's
producer thread and is almost entirely overlapped. Tuning the detector for
throughput on such a clip buys little; the classifier and pose stages are
where the time goes.

### Cross-frame crop batching does not help

`Pipeline` hands the batch stage functions one frame at a time. Grouping a
whole window into a single downstream call was measured with an in-pipeline
A/B (same binary, one env flag):

- **CUDA: no effect at all** — every stage within +/-0.5%.
- **Apple Silicon: neutral to worse** — up to 1.58x slower on head-tail.

An isolated microbenchmark of the same stage functions on an idle GPU
predicted a 1.2-1.35x win. It did not survive contact with the pipeline. Do
not reintroduce cross-frame accumulation on the strength of a stage
microbenchmark.

### The per-stage `batch_size` knobs are mostly inert

`HeadTailConfig.batch_size`, `CNNConfig.batch_size` and `PoseYOLOConfig.batch_size`
default to 64, which already exceeds the per-frame detection count in typical
recordings. They chunk a single frame's detections, so on a frame with fewer
detections than the knob they do nothing. Raising them is not a tuning lever.

### `pipeline_depth` is already correct

Measured at constant `detection_batch_size`, with repeats:

- **CUDA: depth 2 is a real ~12% win** over depth 1.
- **Apple Silicon: no significant difference.**

Output is byte-identical across depths on both platforms. The default of 2 is
right; no per-platform override is needed.

### `detection_batch_size` has a hard, resolution-dependent ceiling

The GUI exposes 1-64, but the pipeline's frame-buffer admission check rejects
large windows on high-resolution video **at run time**, aborting the run:

```
Inference pipeline frame buffer is not resource-admissible:
one frame=61074432 bytes, batch=8, live_windows=4,
estimated=1954381824 bytes exceeds the 536870912-byte pipeline budget
```

At `pipeline_depth=2` the pipeline retains `queue_bound + 3` windows, so on a
4K clip (about 61 MB per decoded frame) anything above batch 2 is
inadmissible. At `pipeline_depth=1` only one window is retained, so batch 8
fits. **The control's range is not the admissible range** — pick the batch
with the frame size and depth in mind, or expect a run-time abort. See
[Resource admission](resource-containment.md).

## Measuring performance changes

Two rules, both learned the hard way in the programme above:

- **Benchmark in the pipeline, not in isolation.** A stage measured on an idle
  device can be off by 7x from its in-pipeline cost, because the pipeline runs
  the producer thread, the consumer thread and any service process on the same
  device. Use an env-gated flag so one binary serves both arms.
- **For a concurrent pipeline, wall clock is the only honest metric.**
  Per-stage span totals are not additive across threads: at `pipeline_depth=2`
  a stage's span inflates while it waits on the shared device, which looks
  like a regression but is just overlap. Compare `wall_clock_s`, and always
  include an A-vs-A repeat to establish the noise floor before believing any
  A-vs-B difference.

`HYDRA_PROFILE=1` writes a span tree next to the video
(`<stem>_logs/tracking_profile_forward.json`). See
[Profiling](profiling.md).
