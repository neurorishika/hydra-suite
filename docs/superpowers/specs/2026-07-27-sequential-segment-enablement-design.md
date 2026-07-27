# Sequential-Segment Enablement — Blocker Fixes + A Follow-ups — Design

**Date:** 2026-07-27
**Status:** Design approved (scope decisions locked); pending spec review → implementation plan
**Scope:** Inference `obb.py`/`config.py` (sequential stage-2), DetectKit training dialog,
X-AnyLabeling mode verification, dataset-fit preview
**Depends on:** Geometry-levels piece A (merged, main `004abda8`). This makes the
already-built `seq_crop_segment` role runnable and clears the three inference blockers A
left for the region-source work.
**Not phase C:** the broad region-source abstraction (`regions → executor →
extract-with-transform → merge`) is deliberately deferred. This is the minimal set of
targeted fixes that makes sequential-mode segmentation work, plus A's two follow-ups.

## 1. Motivation

Piece A defined the `seq_crop_segment` training role (taxonomy, level gating, dataset
builder) but left it **hidden and unrunnable**, because the inference pipeline could not
execute a segment checkpoint as sequential stage-2. Three concrete blockers (verified
against current `obb.py`, main `004abda8`):

- `_extract_obb_from_masks` has **no `offset`/`scale` params** — it cannot remap stage-2
  mask geometry from a crop's coordinate space back into the frame.
- `_run_sequential`'s stage-2 loop **hardcodes `extract_obb_result`** (OBB-only); a
  segment stage-2 checkpoint yields `result.obb is None` on every crop.
- `_assert_direct_task_matches_checkpoint` is only invoked for direct mode, so a
  mismatched stage-2 checkpoint fails **silently** (zero detections, no error).

Two A follow-ups ride along: verifying the X-AnyLabeling `--mode` vocabulary, and covering
the two new direct roles in the training dialog's dataset-fit preview.

## 2. Locked scope decisions

- **Stage-2 tasks: `obb` + `segment` only.** `detect` stage-2 has no training role and
  would need `offset`/`scale` added to `_extract_obb_from_boxes` for no consumer — out of
  scope (YAGNI).
- **Full bundle:** the three blockers + a config-builder param + **unhide
  `seq_crop_segment`** in the GUI (checkbox + per-role pickers) so it is usable
  end-to-end + both A follow-ups.

## 3. Blocker #1 — `offset`/`scale` in `_extract_obb_from_masks`

Add keyword-only `offset: tuple[float, float] = (0.0, 0.0)` and
`scale: tuple[float, float] = (1.0, 1.0)`, applied with the **same scale-then-offset
semantics as `extract_obb_result`** (`obb.py`): the extractor already recovers
`cx, cy, w_arr, h_arr` in the crop's own original (square) pixel space; after that
recovery, apply

```
cx = cx * sx + ox;   w_arr = w_arr * sx
cy = cy * sy + oy;   h_arr = h_arr * sy
```

before `_normalize_obb_geometry` / `_corners_from_xywhr`. When native contours are emitted
(`emit_native_geometry`), apply the identical transform to each `(P,2)` contour
(`pts[:,0] = pts[:,0]*sx + ox; pts[:,1] = pts[:,1]*sy + oy`) so export geometry lands in
frame space too.

**Angle caveat (documented, accepted):** anisotropic scaling (`sx != sy`) after the angle
was recovered in the crop's square space can distort the recovered angle — this is the
**same tradeoff `extract_obb_result` already makes** for OBB stage-2. Sequential crops are
built square (`enforce_square`, default true) so `sx == sy` in practice. No new behavior;
mirror the OBB path exactly.

**Parity:** default `(0,0)`/`(1,1)` → direct-mode segment output byte-identical.

## 4. Blocker #2 — `SequentialConfig.stage2_task` + task-aware dispatch

Add to `SequentialConfig` (`config.py`):

```python
stage2_task: Literal["obb", "segment"] = "obb"
# In sequential mode config.direct is None, so the segment tuning params CANNOT
# be read from DirectConfig — add them to SequentialConfig, defaulting to the
# same values the direct segment path uses:
seg_num_angles: int = 24
seg_crop_size: int = 64
seg_pad_ratio: float = 0.15
seg_mask_threshold: float = 0.5
```

In `_run_sequential`'s stage-2 loop, replace the hardcoded
`extract_obb_result(r, frame_idx, offset=offsets[i+j], scale=scale)` with a dispatch on
`seq.stage2_task`:

- `"obb"` → `extract_obb_result(r, frame_idx, offset=offsets[i+j], scale=scale)` (unchanged).
- `"segment"` → `_extract_obb_from_masks(r, frame_idx, config.raw_detection_cap,
  num_angles=seq.seg_num_angles, crop_size=seq.seg_crop_size,
  pad_ratio=seq.seg_pad_ratio, mask_threshold=seq.seg_mask_threshold,
  offset=offsets[i+j], scale=scale)`.

**Config-builder plumbing:** map a `YOLO_SEQ_STAGE2_TASK` param (default `"obb"`) into
`SequentialConfig.stage2_task` wherever the sequential config is built from the params
dict (`config.py` `from_params`/`build_*`), so a sequential-segment inference config is
constructible.

**Parity:** `stage2_task` defaults `"obb"` → `seq_crop_obb` and every existing sequential
run dispatches exactly as today.

## 5. Blocker #3 — assert the stage-2 checkpoint task

`_assert_direct_task_matches_checkpoint(model, model_task, model_path)` is already generic
(checks `model.task == model_task`; warns for artifacts without `.task`). Rename it
`_assert_task_matches_checkpoint` (drop the misleading "direct") and call it in
`load_obb_models`' **sequential** branch on the stage-2 model:
`_assert_task_matches_checkpoint(models.obb_model, config.sequential.stage2_task,
config.sequential.obb_model_path)`. Optionally also assert the stage-1 detect model is a
`detect` checkpoint. A mismatched stage-2 now fails loudly, matching the direct path.

## 6. Unhide `seq_crop_segment` in the training dialog

Everything below the UI already exists (taxonomy, `_ROLE_MIN_LEVEL`, builder, min-level
gating). This adds:

- A `chk_role_seq_crop_segment` checkbox mirroring the existing role checkboxes at every
  site (creation, layout, toggle→`_on_role_selection_changed`, `_selected_role_keys`,
  `_selected_roles`, load/save, JSON state, `_refresh_role_gating`'s `role_checks`).
- A `role_seq_crop_segment: bool = False` field on `DetectKitProject`.
- Per-role base-model + imgsz pickers mirroring Task 10b: `imgsz_seq_crop_segment: int =
  160`, `model_seq_crop_segment: str = "yolo26s-seg.pt"`, the dialog spinbox/combo,
  load/save/visibility, and `_imgsz_for_role`/`_base_model_for_role` branches so it
  trains end-to-end.

`seq_crop_segment` requires `polygon`-level sources (already in `_ROLE_MIN_LEVEL`), so the
existing gating hides it unless every selected source is polygon-level — correct.

## 7. Follow-up A1 — verify X-AnyLabeling `--mode` vocabulary

Run `xanylabeling convert --help` (env `x-anylabeling-cpu` on this box; `x-anylabeling-cu13`
on mehek) to read the accepted `--mode` values. `xal_mode_for_level` currently maps
`aabb→"rectangle"`, `obb→"obb"`, `polygon→"polygon"` (only `obb` is confirmed). If the CLI
names differ, correct the mapping in `xal_mode_for_level` **only** (single source of
truth) and update its docstring to record the verified vocabulary. If the CLI is
unavailable/uninstallable in the env, document the exact command and leave the mapping.

## 8. Follow-up A2 — dataset-fit preview covers new roles

Extend the training dialog's dataset-fit preview (`_refresh_dataset_fit` /
`dataset_fit_view`) and its cache key (`_dataset_fit_key`) to include `detect_direct`,
`segment_direct`, and `seq_crop_segment` (their imgsz values invalidate the cached
preview, and the preview text reports their size analysis). Cosmetic/read-only; no effect
on model resolution or the launch gate.

## 9. Error handling (loud, per codebase convention)

- Stage-2 checkpoint task mismatch → `ValueError` at load (blocker #3).
- A `segment` stage-2 configured but the stage-2 model exposes no masks → the existing
  `_extract_obb_from_masks` empty-result path already returns `_empty_obb_result`; the
  load-time assert is the primary guard.

## 10. Testing

- **Unit (`obb.py`):** `_extract_obb_from_masks` with non-trivial `offset`/`scale` maps a
  known crop-space rect to the expected frame-space `centroids`/`corners`; default
  `(0,0)`/`(1,1)` is byte-identical to today. Native-contour transform under
  `emit_native_geometry`.
- **Dispatch:** `_run_sequential` with `stage2_task="segment"` routes to the mask
  extractor with per-crop `offset`/`scale`; `stage2_task="obb"` unchanged.
- **Assert:** `_assert_task_matches_checkpoint` raises on a stage-2 task/checkpoint
  mismatch.
- **Config:** `YOLO_SEQ_STAGE2_TASK` round-trips into `SequentialConfig.stage2_task`.
- **GUI:** `DetectKitProject` round-trips `role_seq_crop_segment` + the new imgsz/model
  fields; `_base_model_for_role`/`_imgsz_for_role` resolve `SEQ_CROP_SEGMENT`.
- **Follow-up A1:** the verified `--mode` values are recorded; `xal_mode_for_level`
  matches them.
- **Parity gate (the one that matters):** equivalence harness on `ant_obb_sequential`
  (exercises the sequential path) **and** an OBB clip, on **MPS and CUDA**, proving
  `seq_crop_obb` + direct modes stay **byte-identical** — the additive `stage2_task="obb"`
  default and `(0,0)`/`(1,1)` extractor defaults must not perturb existing tracking.

## 11. Relationship to phase C (and B)

Phase C unifies `_run_direct` + `_run_sequential` + SAHI `run_direct_sliced` onto one
region-source abstraction: `regions → executor → extract-with-transform → merge`. This
spec is deliberately **incremental toward C**, not divergent from it:

- **Prerequisite C needs anyway:** `offset`/`scale` on `_extract_obb_from_masks` (§3) is
  exactly the "extract-with-transform" capability C requires of every extractor.
  `extract_obb_result` already has it; this fills the gap. Survives C untouched.
- **Neutral cleanups:** the generic-assert rename (§5) and `SequentialConfig.stage2_task`
  (§4) express real intent C still needs; they survive C.
- **The one piece C absorbs:** the stage-2 task dispatch added to `_run_sequential` (§4).
  It intentionally **mirrors `_run_direct`'s existing obb/detect/segment branches** — a
  parallel of an established pattern, not a new abstraction. C folds both into the shared
  executor. Rework is minimal and expected; consolidating this parallel is C's purpose.
- **Implementation choice (locked: inline):** write §4's dispatch as **inline branches**
  in `_run_sequential` (minimal, mirrors `_run_direct`), NOT a new shared helper and NOT a
  refactor of `_run_direct` (which would touch the parity-critical direct path + CUDA
  raw-tensor fast path for no present benefit). C introduces the shared executor.
- **Known residual for C (not this spec):** after §3, two of three extractors
  (`extract_obb_result`, `_extract_obb_from_masks`) accept `offset`/`scale`;
  `_extract_obb_from_boxes` (detect) does not, because detect stage-2 has no role here.
  C adds it when the region-source abstraction needs uniform detect regions. Incremental,
  not contradictory.
- **B (SAM2 escalation):** untouched by this spec; unaffected.

**Net:** no change here increases codebase complexity beyond the minimum to run
`seq_crop_segment`, and every change is either a C prerequisite, a neutral cleanup, or a
minimal patterned dispatch C is designed to consolidate.

## 12. Non-goals

- The region-source abstraction / sequential-mode slicing / whole unification (**phase C**).
- `detect` stage-2 (no training role) and `offset`/`scale` on `_extract_obb_from_boxes`.
- A shared stage-2 extraction helper or any refactor of `_run_direct` (**phase C**).
- Changing direct-mode behavior, the CUDA raw-tensor fast path, or the `.npz` cache schema.
