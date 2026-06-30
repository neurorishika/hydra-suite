# Inference Acceleration Restoration — Design

**Date:** 2026-06-30
**Status:** Approved for implementation
**Branch:** `feature/inference-pipeline-redesign`
**Builds on:** `2026-04-26-inference-pipeline-redesign.md` (the original redesign spec)
**Audit source:** `tools/equivalence/PARITY_AUDIT.md`

---

## Motivation

The inference-pipeline redesign achieved its *cleanliness* goal — `core/inference/` is a
self-contained module with pure stage functions, a typed config schema, and per-type
caches — but in doing so it **dropped the accelerations that were the entire reason the
pipeline exists at production scale**. The parity audit (`PARITY_AUDIT.md`) confirms the
new pipeline runs individual analysis (head-tail / CNN / pose / AprilTag) **per-frame**,
extracts GPU-native crops **only for pose**, never calls the GPU-native classifier paths,
and bypasses NVDEC decode and TensorRT/ONNX auto-export entirely (findings H3, H4, Pose
GAP 3/5/7, Domain-4 "perf regression").

The worktree's mandate was *clean up inference **while keeping** the accelerations*. This
design restores them — cross-frame batching, GPU-native residency, foreign-region
suppression, NVDEC, and TRT/ONNX auto-export — **without** reintroducing the fragmentation
the redesign removed. The organizing principle is a single reproducibility invariant that
lets concurrency be a pure throughput knob rather than a correctness risk.

## Goals

1. **Cross-frame batching** for direct *and* sequential OBB (already present) and for
   **all** individual-analysis stages (head-tail, CNN, pose, AprilTag).
2. **GPU-native end-to-end residency** on the CUDA path: decode → OBB → crops → individual
   inference stay on-device; one host pull per window for AprilTag + cache write.
3. **Foreign-region suppression** restored (canonical-crop masking + pose-keypoint
   suppression), honoring the existing `suppress_foreign_regions` flag, default **on**.
4. **NVDEC** GPU video decode and **TensorRT/ONNX auto-export + direct executor** restored
   as runtime-selected, swappable components (CUDA-only; graceful no-op elsewhere).
5. **Reproducible behavior by construction**: identical output for a fixed
   `(decode-path, executor, pipeline_depth)`; concurrency never changes results.
6. **Clean, modular, configurable, reliable**: pure batch-native stages, one orchestrator
   with a bounded configurable depth, one stream-sync chokepoint, one cache-writer.

### Non-goals (this effort)

- **Per-stage runtime independence** (mixed CUDA/CPU groups in one run). Deferred; the
  validation rule from the redesign spec stays. SLEAP remains the one documented exception
  because it already runs in its own conda subprocess.
- GPU-memory *scheduling* (model offload / sequential model residency). The VRAM concern is
  addressed structurally by making pipeline depth a bounded knob (default 2), not by an
  offload scheduler. A scheduler is a separate follow-up if depth=2 still OOMs with all
  models resident.

## Target hardware

**CUDA is the bar.** "To par" = match the legacy CUDA throughput (GPU-native head-tail /
CNN / pose, NVDEC, TRT/ONNX, cross-frame batching). MPS/Apple Silicon must stay correct and
reasonably batched; MPS-specific optimizations are applied where clearly worthwhile but are
secondary. NVDEC and TRT/ONNX are CUDA-only and no-op on MPS/CPU.

---

## The reproducibility invariant (the load-bearing idea)

> **Batch membership and per-item math are a pure function of frame index — never of
> wall-clock time or thread scheduling. `pipeline_depth` changes *when* work runs, never
> *what* it computes.**

Consequences:

- depth=1, depth=2, depth=N produce **byte-identical caches** for the same
  `(video, decode-path, executor)`.
- Correctness is tested cheaply at **depth=1 on MPS/CPU**; concurrency is a pure throughput
  optimization validated separately by a **depth-invariance test** and a CUDA benchmark.
- Batches are formed by **fixed frame windows**, never by arrival-time accumulation.
- In-stage chunking (when a window has more crops than `batch_size`) iterates in
  **detection-id order**, which is itself frame-index-derived
  (`frame_idx * DETECTION_ID_STRIDE + slot`).

### What reproducibility does *not* mean

Bit-parity holds **within** a fixed configuration, not across configurations that change the
numerics upstream of the models:

- **NVDEC vs CPU decode** are not bit-identical (different chroma handling). This was true in
  legacy too.
- **TensorRT / ONNX (esp. FP16) vs PyTorch FP32** are not bit-identical.

The equivalence harness therefore always compares **like-for-like** (same decode path, same
executor, depth=1). Cross-config differences are expected and documented, not regressions.

---

## Architecture

```
                          ┌───────────────── Pipeline (configurable depth) ─────────────────┐
 FrameSource  ──window──▶ │  PRODUCER side                 CONSUMER side                     │
 (Cpu | Nvdec)            │  decode + run_obb  ──CropBatch──▶ extract_crops (GPU-native,      │
                          │                                   foreign-masked)                │
                          │                                     │                            │
                          │                                     ├─ run_headtail  (batched)    │
                          │                                     ├─ run_cnn        (batched)    │
                          │                                     ├─ run_pose       (batched)    │
                          │                                     └─ run_apriltag   (CPU)        │
                          │                                     │                            │
                          │                                   scatter → FrameResult[]         │
                          └─────────────────────────────────────┬──────────────────────────┘
                                                                 ▼
                                                          CacheWriter (async, frame-ordered)
```

At `depth=1` the producer and consumer run sequentially per window. At `depth=2` the
producer for window k+1 overlaps the consumer for window k. At `depth>2`, more windows are
in flight. The boundary between "producer" and "consumer" is the single stream-synced
`CropBatch` handoff.

### Component map (target file layout under `core/inference/`)

| Component | Location | Responsibility |
|---|---|---|
| `Pipeline` | `pipeline.py` (new) | Stream windows through stages; own the depth knob, queues, supervisor, shutdown |
| `BatchWindow` | `result.py` or `pipeline.py` | W consecutive frames + their device residency |
| `CropBatch` | `result.py` (new type) | Cross-frame canonical crops (device tensor) + detection-id/frame index map |
| `FrameSource` | `sources.py` (new) | `CpuFrameReader` / `NvdecFrameReader`; yields device-resident frame tensors |
| stage fns | `stages/*.py` (existing, adapted) | Pure, batch-native; no I/O, no device probing, no threading |
| crop extractor | `stages/crops.py` + `core/canonicalization/crop.py` | One `gpu_canonical_crop_batch`; foreign masking wired in |
| stream helper | `runtime.py` (extend `RuntimeContext`) | One CUDA-event record/wait chokepoint for cross-stage GPU handoffs |
| OBB executor | `stages/obb.py` + ported `_runtime_artifacts`/`_direct_obb_runtime` | TRT/ONNX auto-export + direct CUDA executor selection |
| `CacheWriter` | `cache/writer.py` (new) | One consumer; frame-ordered npz writes; sync at depth=1 |
| `InferenceRunner` | `runner.py` (existing) | Thin: build config/runtime/models, construct `Pipeline`, expose `run_batch_pass`/`run_realtime`/`load_frame` |

`runner.py`'s current per-frame `_run_batch` inner loop and intra-frame
`ThreadPoolExecutor(4)` are **removed** — replaced by `Pipeline` + batch-native stages.

---

## Detailed design

### 1. Batch-native pure stages

Each stage operates on a cross-frame batch and a frame-index map, not a single frame.

```python
# CropBatch: the cross-frame unit shared read-only by head-tail / CNN / pose
@dataclass
class CropBatch:
    crops: torch.Tensor          # (N, C, H, W) device-resident (CUDA/MPS) or CPU
    detection_ids: np.ndarray    # (N,) int64 — primary key, frame-index-derived
    frame_index: np.ndarray      # (N,) int — which frame each crop came from
    obb_by_frame: dict[int, OBBResult]   # for foreign corners, scatter-back, headings
    native_sizes: np.ndarray     # (N, 2) — per-crop native h,w to undo padding
```

Stage signatures become batch-in / batch-out (illustrative):

```python
run_obb(frames_window, models, obb_cfg, runtime)        -> list[OBBResult]   # already batched
extract_crops(window, obb_list, cfg, runtime)           -> CropBatch         # GPU-native + foreign mask
run_headtail(crop_batch, model, ht_cfg, runtime)        -> HeadTailResult    # one batched call
run_cnn(crop_batch, model, cnn_cfg, runtime)            -> CNNResult         # one batched call
run_pose(crop_batch, model, pose_cfg, runtime)          -> PoseResult        # predict_batch_cuda / SLEAP shm
run_apriltag(aabb_cpu_crops, obb_list, model, cfg)      -> AprilTagResult    # CPU
scatter(window, ht, cnns, pose, tag)                    -> list[FrameResult] # per-frame reassembly
```

The three stage rules from the redesign spec are preserved: **no I/O, no mode branching, no
device detection** inside stage functions. Heading resolution (pose → head-tail → OBB axis)
stays in `scatter`/result assembly, as today.

Stages chunk internally when `N > batch_size`, iterating in detection-id order. The
backend's existing `predict_batch` / `predict_batch_cuda` already chunk; we feed them the
whole window's crops.

### 2. Batching unit — fixed frame window

`BatchWindow` = `W = InferenceConfig.detection_batch_size` consecutive frames. OBB runs on
the W frames at once. All detections across the window form one `CropBatch`. This makes
batch boundaries a pure function of frame index. Larger W ⇒ bigger crop batches ⇒ better GPU
utilization, at higher per-window memory; it is the primary throughput/memory tradeoff knob
alongside `pipeline_depth`.

### 3. `Pipeline` orchestrator with configurable depth

```python
class Pipeline:
    def __init__(self, stages, runtime, cache_writer, *, depth: int = 2,
                 queue_bound: int | None = None): ...
    def run(self, frame_source, frame_range) -> InferencePassResult: ...
```

- **depth=1** — synchronous; window k completes before k+1 begins. No threads, no queues.
  Minimum VRAM. This is the **parity / debug** mode.
- **depth=2** (default) — double-buffer: a producer thread runs decode+OBB for window k+1
  while the main path runs crops+individual+scatter for window k; `CacheWriter` drains k−1.
  Bounded to ~2 windows in flight. This mirrors the legacy `pose_pipeline.py` double-buffer
  and captures the GIL-releasing overlap (decode + GPU inference) that is the real win.
- **depth>2** — additional windows in flight via bounded queues; opt-in for large GPUs.
  VRAM scales ~linearly with depth.

Queues are **bounded** (default derived from depth) to provide backpressure: a slow consumer
cannot let a fast producer balloon memory.

### 4. GPU-native unified crop extraction + foreign suppression

- **One** `gpu_canonical_crop_batch` call per window (CUDA: single `affine_grid` +
  `grid_sample`; CPU/MPS: batched `cv2.warpAffine` fallback) produces canonical crops shared
  read-only by head-tail, CNN, and pose. Today only pose uses the GPU batch path; head-tail
  and CNN go through per-crop `extract_classifier_crops` (warpAffine) — these are unified
  onto the shared `CropBatch`. The dead `_extract_canonical_gpu_legacy` is removed.
- **Foreign-region suppression** is applied inside the extractor: the full set of OBB corners
  in the window is passed in, and `_apply_foreign_mask_canonical`
  (`core/canonicalization/crop.py:559`) blacks out neighbor polygons (CUDA: rasterized mask
  pass; CPU: `cv2.fillPoly`). Gated on `PoseConfig.suppress_foreign_regions`, **default on**.
  The pose-cache key already includes `suppress_foreign_regions`/`background_color`, so
  toggling correctly invalidates.
- **Pose-keypoint foreign suppression** (`filter_keypoints_by_foreign_obbs`,
  `utils/geometry.py:172`) is applied in `scatter` after back-projecting keypoints to frame
  space, matching legacy `pose_pipeline.py:747`.

### 5. One stream-sync chokepoint

`RuntimeContext` gains a single helper used at every cross-stage GPU-tensor handoff
(producer→consumer `CropBatch`, and any depth>1 thread boundary): record a CUDA event on the
producing stream, wait on it from the consuming stream before first read. On CPU/MPS it is a
no-op. This confines the entire cross-thread GPU-race class to one reviewed function.

### 6. NVDEC + TRT/ONNX as runtime-selected components

- **`FrameSource`** abstraction: `RuntimeContext.use_nvdec` selects `NvdecFrameReader`
  (zero-copy PyNvVideoCodec → CUDA tensor, ported from `detection_phase.py:124-224`) vs
  `CpuFrameReader` (cv2 → numpy/CPU tensor). NVDEC is CUDA-only and falls back to CPU read
  if the decoder/driver is unavailable, with a logged notice. Documented as not bit-identical
  to CPU decode.
- **OBB executor**: `stages/obb.py:_load_yolo` delegates model loading to a ported
  `runtime_artifacts` helper that auto-exports `.onnx` / `.engine` from `.pt` on first load
  and selects the direct CUDA executor (`_direct_obb_runtime`) for ONNX/TRT, restoring H4.
  Selection is driven by `compute_runtime` + an `auto_export` toggle. Square-letterbox
  preprocessing parity (the direct-executor preprocessing) is preserved and noted for
  non-square video validation.

### 7. Async cache writer

A single `CacheWriter` consumer thread receives completed `FrameResult`s and writes the
per-type npz caches **in frame order** (it buffers and emits in order even if windows
complete slightly out of order at depth>2). Disk I/O never stalls compute. At depth=1 it runs
inline (synchronous). On stop or error: flush in-order what is complete, close handles, single
join. Frame-range coverage (`written_frames`, H9) is recorded as today so truncated passes
remain detectable.

---

## Reliability & cancellation

- **depth=1**: concurrency-free; trivial correctness and shutdown.
- **depth≥2**: a supervisor owns the worker thread(s) and the `CacheWriter`. Any stage
  exception → cancel the window stream, join workers with a timeout, flush the cache writer,
  re-raise to `TrackingWorker`/GUI. Bounded queues prevent memory blowup. **Stop is checked
  at window boundaries**, so no frame is ever half-written; the worst case is discarding an
  in-flight window's work, not corrupting a partial frame.
- **SLEAP** stays a subprocess (shared-memory transport); it is *not* threaded into the mesh.
  Its latency hides behind the next window's decode+OBB naturally. YOLO-pose on CUDA uses
  `predict_batch_cuda`.

---

## Configuration changes (minimal, additive)

| Field | Location | Default | Meaning |
|---|---|---|---|
| `pipeline_depth` | `InferenceConfig` | `2` | 1 = synchronous/parity; 2 = double-buffer; >2 = deep pipeline (opt-in) |
| `detection_batch_size` | `InferenceConfig` (exists) | unchanged | frame-window size W; drives cross-frame crop batch size |
| `suppress_foreign_regions` | `PoseConfig` (exists) | `True` | now actually applied (was inert) |
| `background_color` | `PoseConfig` (exists) | unchanged | foreign-mask fill color |
| per-stage `batch_size` | HeadTail/CNN/Pose cfg (exists) | unchanged | internal chunk size within a window |
| `auto_export` | OBB/pose runtime cfg | `True` on CUDA | enable TRT/ONNX auto-export + direct executor |
| NVDEC enable | derived | auto | `RuntimeContext.use_nvdec` = cuda_mode ∧ decoder available |

No public CLI entry points or cache *schema* fields change (foreign suppression and crop
parameters already participate in the relevant cache keys). `CACHE_SCHEMA_VERSION` is bumped
only if a cached result type's on-disk shape changes during implementation.

---

## Verification strategy (three gates)

1. **Depth-invariance test** *(runs on MPS/CPU — no CUDA needed)*: run a fixture clip at
   `pipeline_depth` 1, 2, and 4; assert **byte-identical** per-type caches across all three.
   This is the proof that concurrency is safe and is the primary new regression test.
2. **Equivalence harness at depth=1**: keep the 4 currently-passing clips
   (`emi_obb_identity`, `ant_obb_sleap`, `worm_bgsub`, `fly_obb`) at bit-parity. Wiring
   foreign suppression changes outputs *toward* legacy and is expected to **help** the 2
   currently-failing dense pose clips (`ant_pose_headtail`, `ant_cnn_identity`); re-evaluate
   them after foreign masking lands.
3. **Performance benchmark on `mehek` (CUDA)**: throughput vs the legacy precompute path
   across `pipeline_depth` ∈ {1,2,4} × NVDEC {on,off} × TRT/ONNX {on,off}. A benchmark
   script + run instructions are produced; the user runs it on the CUDA box (sandbox cannot
   reach the tailnet host). "To par" = new pipeline ≥ legacy throughput at the matched
   best config.

Existing inference/stage/runner/cnn unit tests must stay green; tests asserting per-frame
behavior are updated to the batched contract.

---

## Migration / sequencing (for the implementation plan)

Rough order (the writing-plans step expands each into tasks):

1. Introduce `CropBatch` + batch-native stage signatures; unify GPU crop extraction across
   head-tail/CNN/pose (no behavior change yet — depth=1, single window).
2. Wire **foreign-region suppression** (crop mask + keypoint suppression) behind the existing
   flag; re-run equivalence harness (expect dense-clip improvement).
3. Build the `Pipeline` orchestrator with `depth=1` only; move the runner onto it; remove the
   per-frame inner loop and intra-frame ThreadPool. Add the **depth-invariance test**
   harness (trivially passes at single depth).
4. Add the stream-sync chokepoint + `depth=2` double-buffer + async `CacheWriter`; make the
   depth-invariance test assert 1≡2.
5. Add `depth>2`; assert 1≡2≡4.
6. Restore **NVDEC** `FrameSource` (CUDA, with CPU fallback).
7. Restore **TRT/ONNX auto-export + direct executor** in the OBB loader.
8. Produce the CUDA benchmark script; user validates throughput-to-par on `mehek`.

Each step keeps the suite green and the equivalence harness at parity (depth=1) before the
next begins.

---

## Risks & mitigations

| Risk | Mitigation |
|---|---|
| Cross-thread GPU tensor race (depth>1) | Single stream-sync chokepoint; depth=1 default-safe path always available |
| Batch-boundary nondeterminism | Fixed frame-window batching + detection-id-ordered chunking; depth-invariance test enforces it |
| VRAM blowup with all models + deep pipeline | depth default 2 (bounded 2 windows); depth is a knob; bounded queues; W tunable |
| NVDEC/TRT numerics differ from CPU/PyTorch | Documented; harness compares like-for-like; both are opt-in |
| SLEAP subprocess stalls a thread mesh | SLEAP not threaded; runs as consumer-side subprocess, hidden behind next window |
| Foreign masking changes outputs vs current new pipeline | Intended (moves toward legacy); gated by existing flag; validated against legacy, not against current-broken |
| Can't benchmark CUDA from sandbox | User runs provided script on `mehek`; correctness gates (1 & 2) run locally on MPS |
