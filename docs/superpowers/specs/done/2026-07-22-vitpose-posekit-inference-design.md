# ViTPose in PoseKit Inference — Design

**Date:** 2026-07-22
**Status:** Design — approved, pending implementation
**Branch:** `vitpose-full-integration` (worktree)
**Depends on:** `2026-07-19-vitpose-full-integration-design.md` (already implemented
on this branch through Phase D — native backend, accelerated runtimes, parity).
**Completes:** the one deliberately-skipped slice of that plan — **PoseKit
model-assisted inference**. The tracking-pipeline integration is done; PoseKit's
own *Predict* / *Predict Dataset* inference still offers only YOLO and SLEAP.

## Goal

Make ViTPose a third selectable backend in **PoseKit's model-assisted inference**
(the single-frame *Predict Keypoints* and *Predict Dataset* features that run a
trained pose model over unlabeled images to pre-populate annotations), exactly
paralleling how SLEAP and YOLO already work there. The acceptance loop:

```
fine-tune ViTPose (best.pt)  →  select "ViTPose" in PoseKit inference
  →  Predict / Predict Dataset  →  review overlay  →  Apply Predictions
```

## What already exists on this branch (do not rebuild)

The tracking plan landed Tasks 1–9 + parity:

- **`ViTPoseBackend`** (`core/identity/pose/backends/vitpose.py`) — Protocol impl,
  native + TensorRT + CoreML, `predict_batch(crops) -> List[PoseResult]`.
- **Checkpoint adapter** (`vitpose/adapter.py`) — `load_finetuned_checkpoint`
  auto-infers variant/head/num_keypoints from a `best.pt`.
- **`PoseViTPoseConfig`** + `PoseConfig.backend: Literal["yolo","sleap","vitpose"]`
  + `_dict_to_config` pop (`core/inference/config.py:195,204,208,360,368`).
- **`stages/pose.py`** vitpose translation branch + `"vitpose_pose"` resolver
  stage. PoseKit inference routes *through* `stages/pose.load_pose_model`, so the
  core inference is already reachable.
- **Registry factory** (`pose/api.py`) + `PoseRuntimeConfig` vitpose fields.
- **Cache key** vitpose branch; **trackerkit** family pickers (detection panel +
  orchestrators), guarded by a regression test.

## What is missing (this spec builds it)

PoseKit does **not** use `create_pose_backend_from_config`. Its inference routes:

```
GUI (_pred_backend, _pred_runtime_flavor)
  → PosePredictWorker / BulkPosePredictWorker   (posekit/gui/workers.py)
  → _build_pose_backend                          (posekit/gui/workers.py:21)
  → load_pose_backend                            (core/inference/api.py:41)   ← shim
  → load_pose_model (stages/pose.py)             ← already vitpose-aware
```

Three untouched gaps break a `"vitpose"` selection today:

1. **`load_pose_backend` shim** (`core/inference/api.py:83-110`) branches only
   `yolo` vs `else→sleap`. A `"vitpose"` family currently falls into the SLEAP
   branch and misbuilds a `PoseSLEAPConfig`. **Fix:** add a `family == "vitpose"`
   branch building `PoseConfig(backend="vitpose", vitpose=PoseViTPoseConfig(...))`,
   plus an optional `vitpose_batch` parameter.
2. **PoseKit workers** (`posekit/gui/workers.py`) thread only `sleap_batch` /
   yolo params. **Fix:** thread a `vitpose_batch` through `_build_pose_backend`
   and both workers; `model_path` already carries the checkpoint.
3. **PoseKit GUI** (`posekit/gui/main_window.py`) — the combo is
   `["YOLO","SLEAP"]`; `_pred_backend()`, `_update_pred_backend_ui()`, the
   model-picker dispatch, `_pred_runtime_flavor()` stage-select, and settings
   persistence are all two-way. **Fix:** add the third `vitpose` path everywhere,
   mirroring the SLEAP wiring.

## Architecture

Two axes, unchanged: **runtime** (`runtime_tier` → `ResolvedBackend`, owned by
Gen-2 and already flowing through `stages/pose.py`) and **family**
(`yolo|sleap|vitpose`, this spec adds the PoseKit-side `vitpose` path). PoseKit
inference is **runtime-agnostic**: it forwards the tier picker through
`stages/pose.load_pose_model`. So the **native** path (cpu/gpu/mps) works the
moment this lands, and needs no accelerated-specific PoseKit code. The
**accelerated** runtimes (TensorRT/CoreML) already exist in the `ViTPoseBackend`
and are reached through the same shared shim as YOLO/SLEAP — but note a
**pre-existing, backend-agnostic limitation**: PoseKit's workers pass a
runtime-*flavor* string (`"tensorrt_cuda"`/`"coreml"`) into
`load_pose_backend(compute_runtime=…)`, which re-derives a tier via
`migrate_runtime_to_tier`; that mapping does not recognize those two flavor
strings and collapses them to the `cpu` tier. As a result, selecting `gpu_fast`
in PoseKit currently runs ViTPose (and YOLO and SLEAP) on torch-CPU rather than
the native engine. This is **not** introduced by this slice — ViTPose inherits
the exact YOLO/SLEAP behavior — and fixing it (normalize the flavors in
`migrate_runtime_to_tier`, or thread `runtime_tier` through the shim directly) is
a shared follow-up tracked separately. Verified paths: **native cpu/gpu**.

### Components

**A. `load_pose_backend` shim (`core/inference/api.py`)**
Add `vitpose_batch=None` to the signature and a `family == "vitpose"` branch:

```python
elif family == "vitpose":
    vp_bs = vitpose_batch if vitpose_batch is not None else batch_size
    pose_cfg = PoseConfig(
        backend="vitpose",
        skeleton_file=skeleton_file or "",
        vitpose=PoseViTPoseConfig(
            model_path=model_path,
            batch_size=max(1, int(vp_bs)),
            variant="auto",       # adapter infers from checkpoint
            num_keypoints=0,      # 0 = infer from checkpoint
        ),
        min_keypoint_confidence=min_valid_confidence,
    )
```

The existing `else` becomes an explicit `elif family == "sleap"` (or stays the
default) — keep the yolo/sleap branches byte-identical.

**B. PoseKit workers (`posekit/gui/workers.py`)**
- `_build_pose_backend(..., vitpose_batch=None)` → forward to `load_pose_backend`.
- `PosePredictWorker` / `BulkPosePredictWorker`: accept `vitpose_batch`, store it,
  pass it in the `_build_pose_backend(...)` call.
- **Fallback policy:** like SLEAP, ViTPose has **no** legacy `PoseInferenceService`
  path — on backend failure it **re-raises** (mirror the SLEAP `re-raise` at
  `workers.py:201-206` / `407-412`, not the YOLO legacy fallback).

**C. PoseKit GUI (`posekit/gui/main_window.py`)** — mirror SLEAP throughout:
- Combo `["YOLO","SLEAP"]` → `["YOLO","SLEAP","ViTPose"]`.
- `_pred_backend()` (`:4205`) → return `"vitpose"` for a ViTPose selection
  (`txt.startswith("vitpose")`).
- New **`vitpose_pred_widget`** built next to `yolo_pred_widget` (`:459`) — a
  clone of the simple YOLO widget: a checkpoint `.pt` path edit + Browse (`*.pt`)
  + "Use Latest" (`project.latest_vitpose_checkpoint`). Variant/K auto-inferred,
  so no extra inputs.
- `_update_pred_backend_ui()` (`:4312`) → three-way show/hide
  (yolo / sleap / vitpose widgets); `_populate_pred_runtime_options("vitpose")`.
- Model-picker dispatch (`_get_pred_model_or_prompt` / browse / use-latest) →
  vitpose case validating a `.pt` file (clone the YOLO `.pt` helpers).
- `_pred_runtime_flavor()` stage-select (`:4654`) →
  `"vitpose_pose"` when backend is vitpose.
- Worker construction in `predict_current_frame` / `predict_dataset` → pass
  `vitpose_batch=self.spin_pred_batch.value()` (reuse the shared batch spin).
- Settings persistence (`:1392`, `:1465`) → save/restore `vitpose_model_path`
  and the backend selection round-trips through `combo.setCurrentText`.
- `project.latest_vitpose_checkpoint` (`gui/models.py`) → new optional field for
  "Use Latest" parity with YOLO's `latest_pose_weights`.

## Data flow & result handling

Unchanged from SLEAP/YOLO. Worker `predict_batch(...)` → `PoseResult.keypoints` →
`(x,y,conf)` list → merged into the prediction cache (`infer.merge_cache`) →
shown as overlay → applied to `self._ann.kpts` via the existing
`_preds_to_keypoints` on user *Apply*. No new result path.

## Error handling

- **Checkpoint load failure** (wrong file, corrupt) → `CheckpointKeyError` from
  the adapter, surfaced via the worker `failed` signal → error dialog. No partial
  load, no silent SLEAP fallback.
- **Missing accelerated artifact** is not an error — the resolver returns
  `torch/*` with `used_fallback=True` and the backend serves native (Gen-2 model,
  already implemented). PoseKit surfaces the resolved path via the existing
  `resolved_exported_model_signal`.

## Testing strategy

- **Non-GUI (primary gate):** `load_pose_backend(backend_family="vitpose",
  model_path=<fixture best.pt>, compute_runtime="cpu", ...)` returns a
  `ViTPoseBackend` and its `predict_batch` yields the right keypoint count. Guards
  the misrouting bug (assert it is **not** a SLEAP backend).
- **Worker:** `_build_pose_backend(backend_family="vitpose", vitpose_batch=2,
  ...)` threads the param; single + bulk workers emit keypoints on a synthetic
  checkpoint (headless, no Qt event loop — call `run()` directly as the existing
  worker tests do).
- **GUI wiring:** import smoke of `main_window`; assert the combo offers
  `"ViTPose"`, `_pred_backend()` maps a vitpose selection to `"vitpose"`, and
  `_update_pred_backend_ui` shows `vitpose_pred_widget`. (Qt widgets are
  constructed under a `QApplication` fixture as existing posekit GUI tests do, or
  asserted via source-grep where a live widget is impractical.)
- Pre-PR: focused tests, then `black`/`isort` on changed files (`make format`
  is broken on this branch); `make lint-moderate`. Env: `hydra-mps`.

## Coordination with the tracking plan

The tracking plan's **Task 7** ("GUI family pickers") lists
`posekit/gui/main_window.py`. That posekit portion was deliberately skipped and
is **superseded by this spec** — this work owns the entire PoseKit inference
story (GUI + workers + shim). Task 7's **trackerkit** portion (detection panel +
orchestrators) is already implemented and guarded by a test; this spec does not
touch it.

## Non-goals

- No new runtime tiers; no accelerated-specific PoseKit code (the backend already
  handles TensorRT/CoreML; PoseKit stays runtime-agnostic).
- No changes to how predictions are cached, overlaid, or applied.
- No auto-export UI (the backend auto-manages artifacts; the exported-model row
  stays hidden as it already is for YOLO/SLEAP).
- No changes to the SLEAP/YOLO PoseKit paths beyond the shared shim's new branch.
- No changes to trackerkit (already done).
