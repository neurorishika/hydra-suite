# TensorRT + CoreML Cross-Frame Batching for OBB/Classifiers — Design (Spec 1 of 2)

Date: 2026-07-03
Branch: `feature/inference-pipeline-redesign`
Status: Draft — pending Phase A/B verification results

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

Implement whichever candidate(s) won Phase B:
- CUDA: changes land in `core/inference/runtime_artifacts.py` (export logic)
  and the new-pipeline OBB executor (dynamic `set_input_shape` per window, or
  a swap to an ORT+TRT-EP session, depending on winner).
- CoreML: changes land in `backend.py:_forward_coreml` (batched predict), if
  Phase A determined it's supported.

Acceptance gate: the same equivalence + throughput suite from Phase A,
checked into `tools/equivalence/` as a standing regression check (not just a
one-off spike script).

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
