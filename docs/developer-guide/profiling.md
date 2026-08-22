# Profiling a tracking run

## Which instrument

| Symptom | Instrument |
|---|---|
| A stage got slower and you want to know which function | Span profiler (below) |
| A cross-cutting tax smeared over thousands of small calls | `cProfile` — see "What the span profiler cannot find" |
| You need device time, not host time | `HYDRA_PROFILE_GPU=1` |

## Turning it on

**Debug Mode.** The span profiler is on whenever `ENABLE_PROFILING` is —
which the Debug Mode toggle already derives. The tree is written to
`<video>_logs/tracking_profile_{forward,backward,session}.json` under a
`"spans"` key, and logged as a `SPAN TREE` block.

**`HYDRA_PROFILE=1`.** Arms a process-level recorder with no `TrackingProfiler`
required. Use it for two cases:

1. DetectKit / PoseKit, which build no `TrackingProfiler` at all.
2. Profiling a **User-mode** run without changing what it does. Debug Mode is
   not observation-only — it changes intermediate cleanup and CSV outputs — so
   "turn on Debug and re-run" profiles a different run than the one that was
   slow.

The dump lands in `<video>_logs/` when a session supplies one, otherwise
`$HYDRA_DATA_DIR/profiles/span_profile_<pid>.json`.

`HYDRA_RT_PROFILE` is kept as an alias for `HYDRA_PROFILE`.

## Reading the tree

Each node reports `total_s` (inclusive), `self_s` (inclusive minus direct
children), `n_calls`, `units`, `max_s` and `first_call_s`.

- **`self_s` localizes the defect.** High inclusive time with near-zero self
  time exonerates a stage and indicts its child.
- **`ms/unit` answers batch-size questions.** At `detection_batch_size=1` each
  window is one frame; at 25 the same work arrives in 1/25th the calls, so
  per-call overhead falls straight out of comparing the two runs.
- **`max_s` and `first_call_s` catch warmup.** A 5 s TensorRT engine build
  inside `backend_forward` (n=500) inflates the mean by 10 ms/call and is
  otherwise indistinguishable from a uniform 10 ms slowdown.
- **Percentages are of the parent, within a thread.** At depth≥2 summed span
  time legitimately exceeds wall-clock when threads overlap. Nodes on another
  thread are marked `concurrent`. A subtree that is 43% of its thread but 4% of
  the pass is both — do not read the first number as a speedup ceiling. This is
  the distortion behind the refuted SLEAP-batching premise: pose measured 4.6%
  of wall and batching returned ~0 end-to-end gain.

## Device time

The default profiled path does **not** synchronize. Spans are host wall-clock.
`torch.{cuda,mps}.synchronize()` is device-wide, and `pipeline_depth` defaults
to 2, so a sync on the consumer thread would drain the producer's in-flight OBB
kernels and bill OBB's device time to CNN.

`HYDRA_PROFILE_GPU=1` opts into a deep pass that syncs on GPU spans **and**
forces `pipeline_depth=1`, so there is no producer to contaminate. That run is
explicitly not the production schedule. The JSON stamps `"gpu_mode"` so nobody
compares the two.

| Mode | `total_s` means | GPU attribution |
|---|---|---|
| default | host cost under the production schedule | device work smears into whichever span later blocks |
| deep | host cost under a serialized depth=1 schedule | per-span device time, uncontaminated |

Per-span CUDA events would give device time without serializing, and would let
the deep pass keep depth=2. They are CUDA-only (no MPS equivalent) and are a
named future slice, not an oversight.

## What the span profiler cannot find

**Diffuse self-time defects.** A cross-cutting tax executed inside many small
operations — grad-mode toggling, measured at 26.5 s over 20 k calls — smears as
slightly-elevated `self_s` across dozens of spans and never aggregates into one
line. No span layout catches it. Use `cProfile`:

```bash
python -m cProfile -o /tmp/prof.out -m hydra_suite.trackerkit.cli track <args>
python -c "import pstats; pstats.Stats('/tmp/prof.out').sort_stats('tottime').print_stats(30)"
```

## Adding a span

1. Add the name to `src/hydra_suite/utils/profiling_names.py`. Never pass a
   string literal — `tests/utils/test_profiling_registry.py` fails on it, and a
   refactor that moved the function would otherwise silently drop the row.
2. Use `@spanned(NAME)` for a function boundary, `with span(NAME):` for a
   sub-function region.
3. **Wrap loops, never loop bodies.** A span inside a per-detection body runs
   5M times at 50 detections × 100k frames, measuring what the aggregate
   already reports. Per-detection cost comes from `units`.
4. If the code runs on its own thread, wrap the thread target in
   `bind_target(...)`. An unbound thread records nothing and the report shows
   that work costing zero.

The warp `ThreadPoolExecutor` (`stages/crops.py`) is deliberately **not**
bound: the only work in those workers is a per-detection body, and the parent
`apply_fit` span already bounds the pool's cost inclusively because the caller
blocks on it.
