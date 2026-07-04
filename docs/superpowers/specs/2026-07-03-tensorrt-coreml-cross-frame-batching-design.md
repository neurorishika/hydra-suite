# TensorRT + CoreML Cross-Frame Batching for OBB/Classifiers — Design (Spec 1 of 2)

Date: 2026-07-03 (Phase A/B results and decision added 2026-07-04)
Branch: `feature/inference-pipeline-redesign`
Status: Phase A/B complete, decision made — proceeding to Phase C implementation

## Background

An audit of the redesigned inference pipeline (`core/inference/`) found that
cross-frame batching (multiple frames processed in a single forward pass) works
correctly for most stages — head-tail, CNN (torch/ONNX), pose-YOLO, pose-SLEAP,
direct-OBB and sequential-OBB stage-1 on torch/CUDA/MPS. It does **not** work
for two accelerator paths that are supposed to be the fastest tier:

- **TensorRT OBB** (direct executor and sequential stage-1): every exported
  `.engine` is hardcoded to `_DEFAULT_BATCH_SIZE = 1`
  (`core/inference/runtime_artifacts.py:62`), so `predict()` loops one
  `context.execute_async_v3` call per frame/crop instead of batching the
  window (`_direct_obb_runtime.py:292-353`).
- **CoreML classifier backend** (`backend.py:_forward_coreml`, lines
  871-895) loops `model.predict()` once per crop, always, regardless of
  window size.

Separately, the GUI's batch-size controls for these paths (TensorRT batch
spinner, stage-2 crop batch spinner) write to config keys
(`TENSORRT_BUILD_BATCH_SIZE`, `YOLO_SEQ_STAGE2_RUNTIME_BUILD_BATCH_SIZE`)
that are only read by the **legacy** exporter (`core/detectors/_runtime_artifacts.py`)
— the new pipeline's exporter never reads them. So today these controls are
dead for the pipeline that actually runs. Fixing the UI to be honest about
this is **Spec 2**, deliberately sequenced after this spec, because the right
UI can't be designed until we know what's actually controllable.

## Goal & success criteria

Give OBB detection (direct + sequential stage-1) and stage-2 crop
classification a working cross-frame-batching mechanism on the `gpu_fast`
tier (TensorRT on CUDA, CoreML on Apple Silicon) — or, if none is viable,
determine that conclusively and document it as a permanent constraint.

Success is empirical, not assumed:
1. Passes the existing equivalence/parity gate (`tools/equivalence/compare.py`)
   against legacy CSV output, within existing tolerances.
2. Shows a measurable throughput win over the batch=1/per-crop baseline at
   realistic window sizes (4/8/16), measured on real target hardware.

If a candidate mechanism fails either bar, it is rejected — "TensorRT OBB
stays batch=1 by design" is an acceptable and documented outcome, not a
failure of this spec.

## Scope

**In scope:**
- OBB direct executor, TensorRT path (CUDA)
- OBB sequential stage-1, TensorRT path (CUDA)
- OBB sequential stage-2 crop classification, TensorRT path (CUDA)
- CoreML classifier backend (`backend.py:_forward_coreml`), Apple Silicon

**Out of scope:**
- ONNX CPU/CUDA classifier paths — already batched via TRT-EP dynamic
  shape profiles (`backend.py:718-785`), no change needed.
- Pose/head-tail/CNN torch paths — already batched.
- Any UI/config surface changes (that's Spec 2).
- Legacy `core/detectors/` code paths — not touched by this branch's redesign
  and out of scope to fix from here.

## Phase A — Extend the benchmark harness

Add a new script `tools/equivalence/bench_obb_batching.py`, reusing
`runner.py`/`compare.py` plumbing from the existing equivalence harness, that
exercises a `--obb-batch-strategy` axis with four candidates:

| Strategy | Mechanism |
|---|---|
| `static-b1` | Current baseline: batch=1 engine, per-frame loop |
| `dynamic-shape` | Raw `.engine` exported with an optimization profile (min/opt/max batch); `context.set_input_shape(...)` called per window at inference |
| `onnx-trtep` | Route OBB inference through onnxruntime's TensorRT execution provider with dynamic shape profiles — the same mechanism already proven for classifiers in `backend.py` |
| `static-multi` | Export a small fixed set of engines per batch size (e.g. 1/4/8/16); pick the closest match at runtime, pad/truncate the window |

For each strategy × each OBB fixture clip (`ant_obb_sleap`,
`ant_obb_sequential`, `emi_obb_identity`, `fly_obb`) × window size
{1, 4, 8, 16}, the script records: fps, per-frame latency, parity gate
pass/fail vs. legacy CSV (via `compare.py`), and (for `dynamic-shape` /
`static-multi`) export time and on-disk artifact size.

For CoreML, add a parallel local script (or a mode of the same script) that
first answers a prerequisite question — **does `coremltools`/`MLModel.predict`
even accept a batched `MLMultiArray` input for the exported OBB/classifier
models at all** — before measuring throughput, since this determines whether
CoreML batching is possible in principle.

## Phase B — Run for real, decision gate

- CUDA candidates (`dynamic-shape`, `onnx-trtep`, `static-multi`): run via SSH
  on `mehek` (`rutalab@mehek.taild08eb9.ts.net`), the project's CUDA box.
- CoreML candidate: run locally on this machine (Apple Silicon Mac).

Results are reported back as a table (fps, parity pass/fail, export
cost) per strategy/clip/window-size. The winning mechanism per accelerator is
chosen jointly with the user at this gate — **no candidate is implemented in
production code until this gate passes.** If no TRT candidate beats
`static-b1` by a meaningful margin (rule of thumb: <10% fps improvement is
not worth the added complexity), the conclusion is that TensorRT OBB stays
batch=1, and that becomes a hard constraint fed into Spec 2.

## Phase C — Implementation

See "Phase A/B results" and "Decision" below for what was actually chosen
(two-engine TensorRT switch + CoreML classifier batching fix). Implementation
changes land in:
- `core/inference/runtime_artifacts.py` (export both a static-b1 and a
  dynamic-shape engine per OBB model)
- the new-pipeline OBB executor (route by requested batch size: 1 → static
  engine, ≥2 → dynamic engine + `set_input_shape`, one long-lived context)
- `backend.py:_forward_coreml` (batched `predict()` instead of per-crop loop)

Acceptance gate: the same equivalence + throughput suite from Phase A,
checked into `tools/equivalence/` as a standing regression check (not just a
one-off spike script).

## Phase A/B results (2026-07-04)

Ran on `mehek` (RTX 6000 Ada, TensorRT 10.16.1.11, torch 2.11.0+cu130,
onnxruntime-gpu 1.24.1) for CUDA candidates, and locally (Apple Silicon,
torch MPS, coremltools 9.0) for CoreML. All experiments in scratch
directories; no repo files touched during the spike.

**Raw throughput (synthetic input, `yolo26s-obb`, 1024×1024, FP16), fps /
ms-per-frame:**

| Candidate | b1 | b4 | b6 | b8 | b12 | b16 |
|---|---|---|---|---|---|---|
| static engine (dedicated per size) | 885 (1.13ms) | — | — | 1088 (0.92ms) | — | — |
| dynamic-shape `.engine` | 540 (1.85ms) | 1131 (0.88ms) | **1169 (0.855ms, peak)** | 1146 (0.87ms) | 1059 (0.94ms) | 1016 (0.98ms) |
| ONNX + TensorRT EP | 384 | 629 | — | 631 | — | 582 |

Key findings:
- The dynamic engine's batch=1 case is **41% slower** than a dedicated
  static-b1 engine (540 vs 885 fps) — a real, consistent "dynamic-shape tax."
- Peak dynamic-engine throughput is at **batch 6–8**, not 16 — degrades past
  8.
- Reusing one execution context across varying batch sizes (simulating a
  real pipeline's varying window sizes: 8→1→16→4→1→8→16→1) costs only
  noise-level overhead (≤5%, mostly <0.1ms/frame) vs. a fresh context per
  shape — **safe to keep one long-lived context and call
  `set_input_shape` per window.**
- ONNX+TRT-EP works but is 1.7–2x slower in absolute fps than raw TensorRT
  at every batch size, plus a 154s cold engine-build cost. Not selected.
- CoreML classifier (`TinyClassifier`, custom `export_tiny_to_coreml`,
  already exports with `RangeDim(1,512)` on the batch axis): confirmed
  **batching works**, measured **7.8x per-frame speedup** (1.49ms → 0.19ms
  at batch 1 → 32). This is a pure `_forward_coreml` code-loop bug, not an
  export limitation.
- CoreML OBB: dynamic-batch export via ultralytics **hard-crashes at CoreML
  compile time** (`E5RT`: `TopK k=300 not within range [1,21]`), reproduced
  identically at batch=2 and batch=16. This is an architectural limitation
  of ultralytics' CoreML export (batch and spatial dims share one
  `dynamic=True` flag, and OBB's fixed top-k conflicts with the resulting
  symbolic anchor count) — not fixable without custom MIL-level export
  surgery. Out of scope for this spec.

### Decision (made 2026-07-04)

**TensorRT OBB (CUDA):** keep **two engines** per model — the existing
static batch=1 engine, and one dynamic-shape engine with an optimization
profile covering the configured batch range (min=1, opt≈configured window,
max=configured window). At inference time, route by requested batch size:
- Realtime workflow, or any call with batch size 1 → static-b1 engine
  (avoids the dynamic-shape tax entirely for the case that's actually
  latency-sensitive).
- Batch size ≥ 2 (non-realtime, windowed) → dynamic engine, one
  `context.set_input_shape` + `execute_async_v3` call for the whole window,
  using a single long-lived context per engine.

This is simpler than a `static-multi` engine-per-size set (only 2 engines
to build/manage, not N) while still avoiding the batch=1 regression.

**CoreML classifier:** fix `_forward_coreml` to call `predict()` once per
window (batched `MLMultiArray`) instead of looping per crop. No export
change needed.

**CoreML OBB:** stays batch=1, permanently (architectural CoreML/E5RT
limitation, not a config choice). Spec 2's UI must surface this as an
explicit, accurate notice — e.g. "CoreML detection (OBB) runs one frame at
a time; CoreML classification (identity/head-tail/CNN) batches normally" —
rather than a generic "fixed batch" message that implies uniform behavior
across all CoreML-backed stages.

**Acceptance gate for Phase C:** the two-engine TensorRT switch and the
CoreML classifier batching fix must both pass the existing equivalence
harness (`tools/equivalence/compare.py`) on real video fixtures
(`ant_obb_sleap`, `ant_obb_sequential`, `emi_obb_identity`, `fly_obb` for
TensorRT; any CNN-identity clip for CoreML), not just the synthetic
throughput spike above — synthetic timing proves the mechanism is fast, not
that it's correct. Must also confirm the batching change actually reaches
downstream stages: i.e., that an OBB window is genuinely detected in one
batched call before crops flow into head-tail/CNN/pose (which already
batch), so the cross-frame batching win isn't undone by a hidden serial
step in between.

## Handoff artifact for Spec 2

Regardless of which candidates won or lost, this spec's Phase C concludes
with a **Batching Capability Matrix** (short doc or appendix to this spec),
capturing per accelerator/stage:

- Whether batching is real now (yes/no)
- Valid batch-size range: fixed discrete set, continuous range, or n/a
  (always 1)
- Any measured perf cliff or sweet spot (e.g. "batches >8 show diminishing
  returns on this hardware")
- Which legacy config knobs (`TENSORRT_BUILD_BATCH_SIZE`,
  `YOLO_SEQ_STAGE2_RUNTIME_BUILD_BATCH_SIZE`, etc.) are now genuinely
  load-bearing vs. permanently dead, and what new config surface (if any)
  Phase C introduced in their place.

This matrix is the ground truth Spec 2's UI design will be built on — it
determines whether the UI can offer one universal "batch size" control, needs
per-accelerator controls, or should hide the control entirely for paths where
batching isn't real.

## Risks

- SSH/remote environment on `mehek` may differ from local dev (TensorRT
  version, GPU model) — capture environment info (`nvidia-smi`, TRT version)
  alongside every benchmark run for reproducibility.
- Dynamic-shape TensorRT engines are known to sometimes underperform
  statically-shaped ones; this is exactly why Phase A/B measure empirically
  rather than assuming either raw-TRT dynamic shapes or ONNX+TRT-EP is
  superior.
- CoreML may not support batched crop classification at all for the exported
  model format in use; Phase A's prerequisite check exists specifically to
  fail fast on this before investing in a throughput benchmark.
