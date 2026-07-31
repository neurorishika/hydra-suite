# UI Verification Checklist — Runtime Gen-2 (post retirement)

Manual UI sweep to confirm the Runtime Gen-2 consolidation + string-vocabulary
retirement (merged `8b86e8b` + `4e70d2c`) didn't break any user-facing flow.
Run each item on the **MPS box**; run the GPU-specific ones on **CUDA (mehek)** too.
`[ ]` = do it; the arrow is the expected result. If an item breaks it's almost
certainly in the params/cache plumbing that changed (sections B/C/D), not core
inference.

## A. Runtime selector (the tier UI)
- [ ] `trackerkit` → Setup panel → runtime dropdown shows **tier labels**
  ("CPU", "GPU (Metal)"/"GPU (CUDA)", "GPU-Fast (CoreML)"/"GPU-Fast (TensorRT)"),
  NOT `cpu`/`mps`/`onnx_*`.
- [ ] No-GPU host → dropdown shows **only "CPU"**; MPS/CUDA host → all three tiers.
- [ ] `posekit` → prediction runtime dropdown shows tier labels too.

## B. TrackerKit tracking flow — run once per available tier (cpu, gpu, gpu_fast)
- [ ] **Detection preview** (Detection panel → preview a frame) renders detections
  (verifies the deleted `compute_runtime`/`obb_compute_runtime` preview keys didn't
  break preview).
- [ ] **Full tracking run** on a short clip completes, writes a CSV, and frames/
  progress/stats update live (`RUNTIME_TIER` now drives backend selection).
- [ ] On **gpu_fast/CUDA** confirm TensorRT is actually used (log / GPU util); on
  **gpu_fast/MPS** confirm CoreML — the tier still reaches the accelerated backend.
- [ ] **Head/tail + CNN identity + SLEAP pose** run completes (classifier + SLEAP
  backends now take `ResolvedBackend`; pose flavor derivation).
- [ ] **Stop** mid-run is clean (baseline for the upcoming TrackingWorker Qt split).

## C. Config / preset save & load (loud-error + migration behavior)
- [ ] Save a config, reload it → clean round-trip (writes/reads `runtime_tier`).
- [ ] Load an **old pre-tier preset** (has `compute_runtime`/`pose_runtime_flavor`)
  → clean migration OR the **loud error naming `migrate_runtime_config.py`**. Then
  `python scripts/migrate_runtime_config.py <preset>.json` → gains `runtime_tier`,
  drops legacy keys, loads.
- [ ] Bundled **`ooceraea_biroi`** preset loads (migrated to `runtime_tier: gpu`).

## D. Cache validity (the byte-stable cache keys — most likely silent regression)
- [ ] Run tracking cold, then **re-run the identical config** → detection cache
  **reused** (fast 2nd run, no re-detect). Verifies the detection cache key held.
- [ ] With **CNN identity** on: re-run → classify cache reused (no re-embed).
- [ ] With **pose/individual-properties** on: re-run → properties `.npz` cache reused.
  *If any silently recomputes on the 2nd run, the cache key drifted — flag it.*

## E. PoseKit prediction
- [ ] Model-assisted pose inference on an image (YOLO + SLEAP) on gpu/gpu_fast →
  keypoints produced (`_pred_runtime_flavor` tier-derivation + `predict_pose_for_image`
  `runtime_tier` param).
- [ ] Exported-model **file-picker filter** matches the tier (`.engine`/tensorrt on
  CUDA gpu_fast, `.mlpackage`/coreml on MPS gpu_fast).

## F. Headless CLI
- [ ] Headless tracking on a clip with a config carrying `runtime_tier` → completes,
  correct backend. (A **runtime-less** CLI preset now defaults to `gpu`
  host-dependently — its detection cache re-keys once; expected.)

## G. Smoke (all kits launch)
- [ ] `hydra`, `trackerkit`, `posekit`, `classkit`, `detectkit`, `filterkit`,
  `refinekit` all launch without import errors (the `compute_runtime.py`→
  `onnx_providers.py` rename touched shared imports).

---

**Automated backstop:** the equivalence + benchmark harness (see CLAUDE.md
"Equivalence & Benchmark Verification" + `tools/equivalence/README.md`) proves
byte-identical *tracking output* vs the `legacy/main` baseline on MPS + CUDA — that
covers correctness of the inference path itself; this checklist covers the *UI/GUI*
wiring the harness doesn't touch.
