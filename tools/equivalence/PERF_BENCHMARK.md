# CUDA Performance Benchmark — Run Instructions for `mehek`

This document explains how to run `perf_benchmark.py` on the `mehek` CUDA box,
capture the results table, and report them back to the equivalence audit.

---

## What the script measures

`perf_benchmark.py` benchmarks the **new inference pipeline** across a matrix of:

| Axis | Values |
|---|---|
| `pipeline_depth` | 1 (sync), 2 (double-buffer), 4 (deep-prefetch) |
| NVDEC | on (hardware H.264/H.265 decode via PyNvVideoCodec) / off (cv2 CPU) |
| TRT/ONNX | on (TensorRT auto-export) / off (PyTorch CUDA or CPU) |

**There is no legacy baseline in this branch** — the legacy precompute path was
removed. The gate is self-contained:

- **Gate 1 (required):** best accelerated config (NVDEC on + TRT on + depth ≥ 2)
  must be faster than the fully unaccelerated baseline (NVDEC off + TRT off +
  depth = 1) in the same new pipeline. Exit non-zero if the accelerations don't
  help.
- **Gate 2 (optional):** if `--baseline-fps` is supplied, the best config must
  also meet or exceed that known legacy production throughput.

Unavailable combos (NVDEC absent, TRT without a `.pt` file, etc.) are silently
skipped and labelled `SKIPPED` in the table — the script never crashes on them.

---

## Prerequisites on `mehek`

1. **Conda environment** — `hydra-suite-cuda` (CUDA 12 or 13 build):

   ```bash
   conda activate hydra-suite-cuda
   ```

2. **Repo worktree checked out**:

   ```bash
   cd /path/to/multi-animal-tracker
   git checkout inference-pipeline-redesign   # or the merged main branch
   ```

3. **Test clip** — an H.264 or H.265 video (MJPEG/AVI will decode correctly via
   cv2 but NVDEC requires H.264/H.265). Recommended: 500–2000 frames at the
   clip's native resolution (the same clip used for equivalence testing works).

   Place it at, e.g.:

   ```
   /data/hydra-bench/ant_obb.mp4
   ```

4. **InferenceConfig JSON** — a real production config with:
   - `obb.mode = "direct"` and `obb.direct.model_path` pointing to a `.pt` file
     (TRT auto-export needs a `.pt` to export from)
   - `headtail`, `cnn_phases`, `pose` as used in production (OBB + headtail +
     CNN + pose for a full pipeline benchmark)

   The equivalence fixture configs in `tools/equivalence/fixtures/configs/` can
   be used; resolve model paths against the fetched models directory.

   Example path:

   ```
   /data/hydra-bench/ant_config.json
   ```

5. **Models archive fetched** (if using equivalence fixture configs):

   ```bash
   bash tools/equivalence/fixtures/fetch_fixtures.sh
   ```

---

## Running the benchmark

```bash
cd /path/to/multi-animal-tracker

PYTHONPATH=src KMP_DUPLICATE_LIB_OK=TRUE \
  /path/to/miniforge3/envs/hydra-suite-cuda/bin/python \
  tools/equivalence/perf_benchmark.py \
    --video   /data/hydra-bench/ant_obb.mp4 \
    --config  /data/hydra-bench/ant_config.json \
    --depths  1,2,4 \
    --nvdec   on,off \
    --trt     on,off \
    --warmup  1 \
    --repeats 3
```

Optional: add your known legacy production fps as a second gate:

```bash
    --baseline-fps 45.0
```

### Expected output

```
Benchmark: ant_obb.mp4
Config:    /data/hydra-bench/ant_config.json
Combos:    12
Warmup:    1  Repeats: 3

------------------------------------------------------------------------
Config                                 Median fps    Speedup  Status
------------------------------------------------------------------------
depth=1  nvdec=off  trt=off [baseline]       28.4        N/A  OK
depth=2  nvdec=off  trt=off                  31.2       1.10x  OK
depth=4  nvdec=off  trt=off                  32.1       1.13x  OK
depth=1  nvdec=on   trt=off                  38.7       1.36x  OK
depth=2  nvdec=on   trt=off                  43.2       1.52x  OK
...
depth=2  nvdec=on   trt=on  [best]           67.8       2.39x  OK
depth=4  nvdec=on   trt=on                   69.1       2.43x  OK
...
NVDEC-only combos without a .pt → SKIPPED or trt skipped on CPU box
------------------------------------------------------------------------

GATE PASS: best accelerated (depth=2  nvdec=on  trt=on) = 67.8 fps, speedup = 2.39x > 1.0 vs. baseline.
```

Exit code 0 = gate passed; exit code 1 = gate failed (accelerations are not helping).

---

## Notes on NVDEC and TRT requirements

- **NVDEC (hardware decode)** requires:
  - A CUDA GPU on `mehek`
  - `PyNvVideoCodec` + `cupy` installed in the conda env
  - The video must be encoded as H.264 or H.265 (MJPEG/AVI → skipped
    automatically with a WARNING, not a crash)

- **TensorRT auto-export** requires:
  - TensorRT installed (`tensorrt` Python package + CUDA libraries)
  - `obb.direct.model_path` pointing to a `.pt` file (auto-export builds
    `.engine` on first run, then caches it)
  - First-run export may take 30–120 seconds; subsequent runs reuse the artifact

- If either is unavailable, the combo logs `SKIPPED (unavailable)` and the
  gate evaluates only the combos that ran.

---

## Controlling agent cannot reach `mehek` directly

The sandbox that runs this agent has no SSH/network access to `mehek`. Options:

1. **Run manually** on `mehek` using the command above, then paste the output
   table back into this session or append it to `PARITY_AUDIT.md`.

2. **Session `!` prefix** (if your terminal session on `mehek` is whitelisted):

   ```
   ! PYTHONPATH=src KMP_DUPLICATE_LIB_OK=TRUE python tools/equivalence/perf_benchmark.py ...
   ```

---

## Local smoke test (CPU-only, no real models)

Before running on `mehek`, verify the script itself is sound on a local CPU box:

```bash
# 1. --help exits 0
PYTHONPATH=src KMP_DUPLICATE_LIB_OK=TRUE \
  /Users/neurorishika/miniforge3/envs/hydra-mps/bin/python \
  tools/equivalence/perf_benchmark.py --help

# 2. --dry-run prints config matrix without executing
PYTHONPATH=src KMP_DUPLICATE_LIB_OK=TRUE \
  /Users/neurorishika/miniforge3/envs/hydra-mps/bin/python \
  tools/equivalence/perf_benchmark.py \
    --dry-run \
    --depths 1,2,4 \
    --nvdec on,off \
    --trt on,off
```

A depth=1 CPU pass with a real video + config (no CUDA, no TRT):

```bash
PYTHONPATH=src KMP_DUPLICATE_LIB_OK=TRUE \
  /Users/neurorishika/miniforge3/envs/hydra-mps/bin/python \
  tools/equivalence/perf_benchmark.py \
    --video  tools/equivalence/fixtures/clips/fly_obb.mp4 \
    --config tools/equivalence/fixtures/configs/fly_obb.json \
    --depths 1 \
    --nvdec  off \
    --trt    off \
    --warmup 0 \
    --repeats 1
```

This will print a single-row table and exit with code 0 (no accelerated combo to
gate against).

---

## Pasting results back

After a successful run on `mehek`, append the table + gate output to
`tools/equivalence/PARITY_AUDIT.md` under a new heading, e.g.:

```markdown
## CUDA perf benchmark — <date> — mehek

**Video:** ant_obb.mp4 (500 frames, 1920×1080, H.264)
**Config:** ant_config.json (OBB+headtail+CNN+pose, TRT auto-export)

<paste table here>

Gate: PASS (best accelerated 67.8 fps, speedup 2.39×)
```
