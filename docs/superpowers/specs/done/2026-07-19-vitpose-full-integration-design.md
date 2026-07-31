# ViTPose Full Integration — Design

**Date:** 2026-07-19
**Status:** Design — pending user review
**Supersedes:** the *Spec 2* and *Spec 3* sections of
`2026-07-16-vitpose-backend-roadmap.md` (consolidated here; see
[Deviations](#deviations-from-the-roadmap)).
**Depends on:** `2026-07-19-runtime-resolver-gen2-consolidation-design.md`
(Gen-2). This spec targets Gen-2's **end state** — `runtime_tier` →
`ResolvedBackend` — and does not touch the legacy `ComputeRuntime` string layer
Gen-2 deletes.

## Goal

Make ViTPose a first-class, selectable pose backend that runs in the live
tracking pipeline on **all four runtimes** — native torch (cpu/mps/cuda),
TensorRT, and CoreML — so a user can **fine-tune a ViTPose model (Spec 4) and
immediately track with it**. The end-to-end loop is the acceptance target:

```
train (Spec 4)  →  best.pt  →  select ViTPose backend in the GUI  →  track a video
```

## What already exists (do not rebuild)

**Spec 1 leaf** (`core/identity/pose/vitpose/`, PoseKit-free):
- `ViTPose(nn.Module).forward(x) -> heatmaps` — device-agnostic, fp32
  (`vitpose.py:23-24`); `build_vitpose(variant, head, num_keypoints)`
  (`vitpose.py:27-32`).
- **On-device UDP decode** `decode_udp_torch(heatmaps, kernel=11)` — float32,
  runs on mps/cuda, closed-form 2×2 Hessian inverse chosen to avoid MPS linalg
  gaps (`decode.py:185-284`). A numpy/cv2 oracle `decode_udp_cv2` also exists
  (`decode.py:137-156`).
- Preprocess (numpy/cv2): `box2cs`, `top_down_affine` → `cv2.warpAffine`,
  `normalize` (`transforms.py:26-112`). Inverse `transform_preds` (numpy,
  `transforms.py:115-156`).
- **All three export recipes** (torch → artifact): `export_onnx`
  (opset 17, dynamic batch, fp32; `export.py:49-126`), `build_tensorrt_engine`
  (fp32 default, dynamic batch, `export.py:129-216`), `export_coreml`
  (`.mlpackage`, static batch=1, fp32; `export.py:219-262`).
- `load_checkpoint(model, path, strict)` for classic/upstream checkpoints
  (`weights.py:15-26`).

**Gen-2 resolver** (`runtime/resolver.py`): `ResolvedBackend(backend ∈
{torch, tensorrt, coreml}, device ∈ {cpu, cuda, mps}, used_fallback)`
(`resolver.py:24-28`); `gpu_fast` resolves to `tensorrt`/cuda or `coreml`/mps
**only when `artifact_available()`**, else degrades to torch with
`used_fallback=True` (`resolver.py:65-74`). Fallback is the resolver's job.

## What is missing (this spec builds it)

The leaf emits artifacts and runs a torch forward, but has **zero runtime
execution** of those artifacts and no composed inference. Concretely absent:

1. **Runners** — no ONNX `InferenceSession`, no TensorRT engine
   deserialize/execute, no CoreML `.mlpackage` predict.
2. **A crop→keypoints driver** composing pre → forward/session → decode →
   inverse into one call.
3. **Batching / warmup / artifact-cache (signature + sidecar) / session
   caching.**
4. **A fine-tuned-checkpoint adapter.** The training payload saves
   `{"model_state", "variant", "num_keypoints", "sched_state", ...}`
   (`training/train.py:131-143`), but leaf `load_checkpoint` looks for a
   `"state_dict"` key (`weights.py:19`) — so a `best.pt` will **not** load
   without an adapter that reads `model_state` and the `variant`/`num_keypoints`
   metadata. The training checkpoint also does **not** currently record the
   `head` type (classic deconv vs simple), which the backend needs to rebuild
   the module.
5. **A pose-backend registry + config + resolver stage** so ViTPose is
   selectable at all (`pose/api.py` factory raises on any family that is not
   `yolo`/`sleap`; `resolver.STAGES` has no `vitpose_pose`).
6. **A parity harness** comparing the runtimes against the torch reference.

## Architecture

Post-Gen-2, backend selection is two orthogonal axes:

- **Runtime axis** (Gen-2 owns): `runtime_tier` → `ResolvedBackend(backend,
  device)`. ViTPose consumes this; it never sees a `ComputeRuntime` string.
- **Family axis** (this spec owns): `yolo | sleap | vitpose` → which module +
  pre/post. This spec adds `vitpose`.

Layering (dependency points downward):

```
posekit / trackerkit GUI ── family picker (yolo|sleap|vitpose), tier picker
        │
create_pose_backend_from_config(registry)          ← Phase C (family axis)
        │
core/identity/pose/backends/vitpose.py             ← Phase B (the backend)
   • native torch path  (backend=torch,  device cpu/mps/cuda)
   • auto_export_vitpose_model  (caching wrapper over leaf export.py)
   • crop→keypoints driver + finetuned-ckpt adapter
        │
core/identity/pose/runtime/  (shared pose runtime)  ← Phase A (extracted from sleap.py)
   • OnnxSessionRunner   • TensorRTEngineRunner   • CoreMLRunner
   • execution_providers_for(resolved)  [Gen-2]
        │
core/identity/pose/vitpose/  (Spec 1 leaf: model, decode, export, transforms)
```

### Runtime map (what serves each ResolvedBackend)

| tier | platform | ResolvedBackend | ViTPose path |
|---|---|---|---|
| cpu | any | torch/cpu | native forward + `decode_udp_torch` on CPU |
| gpu | CUDA | torch/cuda | native forward on cuda |
| gpu | Apple | torch/mps | native forward on mps |
| gpu_fast | CUDA + engine | tensorrt/cuda | `TensorRTEngineRunner` (zero-copy device path) |
| gpu_fast | Apple + mlpackage | coreml/mps | `CoreMLRunner` |
| gpu_fast | artifact missing | torch/* (`used_fallback`) | native forward; background export for next run |

The **native torch path needs no runner** — it is the leaf module on
`resolved.device` plus `decode_udp_torch`, which is why cpu+gpu tiers work the
moment Phase B lands. ORT sessions (`OnnxSessionRunner`, EPs from
`execution_providers_for`) serve as the general/fallback mechanism for the
`tensorrt` backend (TRT-EP) when a native engine is unavailable, mirroring
SLEAP's existing ladder.

## Deviations from the roadmap

Three deliberate scope changes from the roadmap's Spec 2/Spec 3, each argued
from a code audit (2026-07-19):

1. **Pose-only runtime, not a pose+detector runtime.** The roadmap's Spec 2
   proposed one runtime shared by pose *and* the OBB detector. Audit finding:
   the detector runtime (`core/inference/direct_executors.py`) and SLEAP's
   embedded runtime overlap only at TRT engine-lifecycle primitives; precision
   (detector fp16 vs pose fp32), input layout, IO-binding vs numpy boundary,
   stream strategy, batching, and decode all diverge — the `Detect` and `OBB`
   executors share more with each other than with pose. Unifying them is
   abstraction-against-divergence. **The shared unit is a *pose* runtime
   (`core/identity/pose/runtime/`), used by SLEAP + ViTPose.** The one genuine
   cross-world win — the detector hardcodes ONNX providers instead of deriving
   them — is folded into Gen-2's `execution_providers_for`, not this spec.

2. **Drop the "tensor-first Protocol" reshape.** The roadmap's Spec 2 proposed
   inverting `PoseInferenceBackend` so a device-resident tensor path is primary.
   Audit finding: the win is confined to the **TensorRT input side** — YOLO is
   locked to Ultralytics' `predict()`, SLEAP-ONNX crosses a numpy boundary
   anyway, and **every backend returns a numpy `PoseResult`** because the stage
   assembler does OpenCV affine inversion on CPU (`stages/pose.py:288`). The
   existing `predict_batch_cuda` + `hasattr` idiom (also used across the
   classifier subsystem) already captures that one win. **Keep the numpy
   `predict_batch` Protocol; keep `predict_batch_cuda` as the optional
   device-tensor entry** for the TensorRT path. Revisit a Protocol reshape only
   if/when affine inversion is moved on-device (out of scope).

3. **Consolidate Spec 2 + Spec 3 into this one phased spec.** For ViTPose the
   backend owns both runtime execution and family wiring; splitting them was the
   artificial boundary the audit flagged. Phases A/B/C below preserve the
   conceptual layering without two separate documents.

## Components

### Phase A — Shared pose runtime (`core/identity/pose/runtime/`)

Extract the model-agnostic device machinery currently embedded in `sleap.py`
(~700–750 lines, ~43% of the file; no SLEAP coupling — audit-confirmed) into a
new pose-local package, re-keyed onto `ResolvedBackend`:

- `OnnxSessionRunner` — from `_DirectOnnxSession` (`sleap.py:342-364`). ORT
  session; providers from `execution_providers_for(resolved)` (Gen-2). numpy
  in/out. Input-spec introspection (layout/channels/min-batch).
- `TensorRTEngineRunner` — from `_DirectTensorRTEngine` (`sleap.py:487-737`).
  Deserialize engine, execution context, device-resident tensors, `run_cuda`
  zero-copy path, dtype from engine (fp32 for ViTPose).
- `CoreMLRunner` — **new** (no SLEAP precedent). Loads a `.mlpackage` via
  `coremltools`, predicts a batch. Honors the leaf export's **static batch=1**
  (`export.py:225`) by looping per-crop; documented as a known CoreML batching
  limitation.
- Fallback ladder — from `_init_tensorrt_runner`/`_ort_trt_ep_fallback`
  (`sleap.py:809-915`): native TRT engine → ORT-TRT-EP. Now gated by the
  resolver's `used_fallback` rather than in-backend string juggling.
- SLEAP is refactored to consume this package (its runtime classes deleted from
  `sleap.py`), which drags `sleap.py` back toward the ~500-line rule and proves
  the runtime against two consumers before ViTPose depends on it.

**No behavior change for SLEAP** is a hard requirement of Phase A — guarded by
the existing `tools/equivalence/verify_sleap_exported_vs_service.py` parity
harness, pinned *before* the extraction.

### Phase B — The ViTPose backend (`core/identity/pose/backends/vitpose.py`)

Mirrors the 293-line `yolo.py` in spirit (thin), not the 1711-line `sleap.py`.

- **Native path**: build the leaf module (`build_vitpose`), place on
  `resolved.device`, forward → `decode_udp_torch` on-device → `.cpu()` →
  `transform_preds`. Covers cpu + gpu tiers.
- **`auto_export_vitpose_model(...)`** — the caching wrapper the leaf explicitly
  defers to it (`export.py:8-11`). Signature/behavior mirrors
  `auto_export_yolo_model`/`auto_export_sleap_model`: co-locate artifact with the
  source checkpoint, `<artifact>.runtime_meta.json` sidecar written only after a
  successful export, checked before reuse, **with a recipe-version tag** in the
  signature (audit: the pose package's signatures lack one — do not copy that
  gap). Uses the leaf's `export_onnx` / `build_tensorrt_engine` / `export_coreml`.
  Lazy, on first use, on a QThread — never the GUI thread.
- **Crop→keypoints driver**: composes `box2cs` → `top_down_affine` → `normalize`
  → runner/forward → `decode_udp_torch` → `transform_preds`, with per-crop
  center/scale bookkeeping and batch collation (the leaf pre is single-image).
- **`predict_batch(crops: Sequence[np.ndarray]) -> List[PoseResult]`** (the
  Protocol) + **`predict_batch_cuda(...)`** (optional, device tensors, only the
  `TensorRTEngineRunner` benefits zero-copy; others fall back to numpy — matches
  the SLEAP shape).
- **Fine-tuned-checkpoint adapter** (leaf `weights.py`, new function
  `load_finetuned_checkpoint(path) -> (module, meta)`): reads `model_state` +
  `variant` + `num_keypoints` from the training payload's `best.pt`, so the
  backend auto-detects variant/K — the user just points at the file. Requires a
  small Spec-4 addition: **training must also record `head` in the checkpoint
  metadata** (`training/train.py:131-143`); as a fallback the adapter infers
  head from `final_layer`/deconv key shapes in `model_state`.
- **FP32 throughout** (roadmap decision; keypoint precision), consistent with
  every leaf export default.

### Phase C — Pose-family integration (the slim old Spec 3)

Post-Gen-2 the runtime-plumbing sites collapse, so this shrinks to the
family axis:

- **Registry factory**: replace the `if backend_family == …` chain in
  `create_pose_backend_from_config` (`pose/api.py:53,99,159`) with a
  `{family: constructor}` dict; add `vitpose`.
- **Config**: reshape `PoseRuntimeConfig` (`pose/types.py:22`) from flat
  `yolo_*`/`sleap_*` fields to **per-backend sub-dicts**, aligning it with
  `PoseConfig` (`inference/config.py:263`) which already uses them; add a
  `vitpose` sub-config (checkpoint path, variant, num_keypoints — or "auto from
  checkpoint").
- **Resolver stage**: add `"vitpose_pose"` to `resolver.STAGES`
  (`resolver.py:15`).
- **Cache key**: add a `vitpose` branch to `pose_cache_key`
  (`inference/cache/keys.py:167`) and the property-cache identity
  (`properties/cache.py`) — **sequenced after Gen-2 Phase 3** rewrites that
  param-bag boundary, so we widen the rewritten code, not the code Gen-2 deletes.
- **GUI family pickers**: extend the posekit/trackerkit backend selectors and
  their model-path routing to offer ViTPose (training.py already lists it;
  `trackerkit/.../config.py` already has partial `vitpose` branches — this
  completes the asymmetry so the GUI can *run*, not only *train*, ViTPose).

### Phase D — Parity, end-to-end, and the train→track loop

- **`tools/equivalence/verify_vitpose_runtimes.py`** — mirror
  `verify_sleap_exported_vs_service.py`: build ViTPose through the production
  selector at each available tier, feed identical real crops, compare keypoints
  against the torch reference (native = oracle). Threshold: sub-pixel on
  torch/onnx, ≲1px on tensorrt/coreml (fp32).
- **End-to-end tracking test**: a short real video tracked with a ViTPose
  checkpoint, asserting non-degenerate keypoints and no runtime downgrade
  surprises.
- **Train→track smoke**: fine-tune a tiny ViTPose for a few epochs (Spec 4),
  load `best.pt` through the adapter, track — the acceptance loop.

## Data flow (one crop, native path)

```
image + bbox
  → box2cs → top_down_affine (cv2 warp, CPU) → normalize (ImageNet, CHW)
  → to(device) → ViTPose.forward → heatmaps (B,K,64,48) on device
  → decode_udp_torch (on device) → coords (B,K,2)+conf on device
  → .cpu().numpy() → transform_preds (crop→image coords)
  → PoseResult(keypoints, valid_mask)   [numpy — Protocol boundary]
```

TensorRT path swaps the middle (forward) for `TensorRTEngineRunner.run_cuda`
keeping the crop tensor device-resident; CoreML path swaps in `CoreMLRunner`
(per-crop, batch=1). Decode/inverse are unchanged.

## Error handling

- **Missing artifact is not an error** — the resolver returns
  `torch/*` with `used_fallback=True` (`resolver.py:67-73`); the backend serves
  native and triggers a background export for the next run. This is Gen-2's
  model and resolves the roadmap's "silent downgrade" tension: the *resolver*
  decides fallback explicitly (surfaced via `used_fallback` in the UI), the
  backend never silently swaps runtimes.
- **A requested export that fails raises** (`ExportError` from the leaf) rather
  than degrading invisibly — matching the OBB clean-rewrite stance that fixed
  parity bug H4. The failure is logged and surfaced; the next resolve falls back
  to native.
- **Checkpoint load failure** (wrong variant/K, corrupt file) raises
  `CheckpointKeyError` (leaf) with the offending keys — no partial load.

## Testing strategy

- **Phase A**: SLEAP parity pinned before extraction
  (`verify_sleap_exported_vs_service.py`), re-run green after. Focused unit
  tests on each runner (session build, engine deserialize, coreml predict).
- **Phase B**: crop→keypoints driver vs the leaf's own oracle
  (`decode_udp_cv2`) on synthetic heatmaps; `auto_export_vitpose_model`
  signature/staleness tests; finetuned-adapter test on a real training `best.pt`
  fixture.
- **Phase C**: registry dispatch test (all three families construct); config
  round-trip (`from_dict`/`to_dict`) for the sub-dict reshape; resolver stage
  present.
- **Phase D**: the parity harness is the primary numeric gate; e2e + train→track
  smoke as integration gates.
- Pre-PR per the repo: `make commit-prep`, `make lint-moderate`,
  `make docs-check`; `hydra-mps` env (base torch broken); run `black`/`isort`
  directly (`make format` is broken).

## Non-goals

- **No MoE / ViTPose+** (settled: hydra is per-project, cross-species
  generalization is not its job).
- **No FP16** (keypoint precision; fp32 everywhere).
- **No on-device affine inversion** and therefore no tensor-first Protocol
  reshape (Deviation 2).
- **No new runtime tiers** — cpu/gpu/gpu_fast as Gen-2 defines them.
- **No pose+detector runtime unification** (Deviation 1).
- **No entry-point plugin discovery** — the registry is a plain dict (N=3).

## Dependencies & sequencing

1. **Gen-2 Phases 0–3** (backends accept `ResolvedBackend`; pose settings +
   `properties/cache.py` boundary rewritten) are the hard prerequisite for
   Phases A and C. Gen-2 Phases 4–7 (config-field deletion, legacy bulk-delete,
   rename) may proceed in parallel with ViTPose Phases A–D.
2. **Phase A** (pose runtime) before **Phase B** (backend consumes the runners).
3. **Phase B** before **Phase C** (registry needs a constructor to register).
4. **Phase C** before **Phase D** (parity runs through the production selector).

The native torch path (Phase B minus the runners) is independently valuable and
could ship first to unblock train→track on cpu/gpu while TensorRT/CoreML (the
runner-dependent gpu_fast paths) follow — a fallback if Gen-2 slips.

## Open questions

- **CoreML batch>1**: the leaf export pins batch=1 (`export.py:225`). Per-crop
  looping is correct but slower; a dynamic-batch CoreML export is a possible
  later optimization, not in scope.
- **`head` in the training checkpoint**: confirm the small Spec-4 addition
  (record `head`) vs. relying solely on state_dict-shape inference.
- **Recipe-version tag format** for the ViTPose artifact signature — align with
  whatever Gen-2 leaves as the surviving artifact-cache convention.
