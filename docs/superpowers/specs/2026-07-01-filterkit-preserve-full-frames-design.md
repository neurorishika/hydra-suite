# FilterKit: Preserve Full Frames Mode

## Problem

FilterKit's diversity sampler currently treats each individual-animal crop as an
independent item: it clusters on raw pixel features and keeps one representative
crop per cluster, discarding the rest. This is fine for building a diverse
*classification*-style dataset, but it destroys the ability to reconstruct a
full video frame with all its animals — which is what's needed to later build
multi-animal ("bottom-up") pose training data in PoseKit.

We want an opt-in mode where, instead of keeping only the most diverse
individual crops, FilterKit selects a diverse set of *frames* and keeps every
individual's crop from each selected frame.

## Goals

- Add a `preserve_full_frames` toggle to FilterKit's pipeline.
- When enabled, diversity sampling operates on frames (not individual crops),
  and the final export expands each selected frame to include every
  individual detected in it — even ones that would otherwise be dropped by
  quality/dedup/temporal filtering.
- Keep `metadata.json` untouched so existing per-crop frame/OBB/canonicalization
  data remains available for later reconstruction work (out of scope here).

## Non-goals

- Reconstructing an actual full-frame image or building multi-instance PoseKit
  annotations. That is future PoseKit work (a planned "frame mode" toggle) and
  is explicitly out of scope for this change.
- Changing `metadata.json`'s schema or the crop filename encoding.
- Changing behavior of any dataset type other than what FilterKit already
  supports (MAT identity crop datasets with `frame_idx`/`det_idx`, and
  COCO/YOLO/class-folder datasets, to whatever extent `frame_idx` is
  meaningful for them).

## Design

### Config

Add `preserve_full_frames: bool = False` to the pipeline config dict built in
`filterkit/gui/main_window.py` (the same dict that carries
`diversity_enabled`, `diversity_target`, etc. — not the persisted
`FilterKitConfig` dataclass, which only tracks dataset path / pagination).

### Diversity stage (`core.py`)

When `preserve_full_frames` is `True`, `diversity_sample` clusters on frames
instead of individual crops:

1. Group the stage's input items by `frame_idx`.
2. For each frame, average the existing 32×32 grayscale flattened feature
   vectors of all individuals present in that frame into one 1024-d vector.
3. Compute `avg_individuals_per_frame` from the full loaded dataset (total
   crops ÷ total unique frames) — not just the post-filter subset.
4. Back-solve the number of frames to select:
   `n_frames = max(1, round(diversity_target / avg_individuals_per_frame))`.
5. Run `MiniBatchKMeans` with `n_clusters=n_frames` over the per-frame
   vectors, and pick the frame nearest each centroid (same
   `pairwise_distances_argmin_min` approach as today), deduplicated.

When `preserve_full_frames` is `False`, behavior is unchanged (per-crop
clustering as today).

If diversity is disabled but `preserve_full_frames` is enabled, this stage is
skipped entirely — expansion (below) still applies to whatever
temporal/dedup/quality left behind.

### Finalize / expansion step

After all enabled stages produce a "kept" set of items, if
`preserve_full_frames` is `True`:

1. Collect the unique `frame_idx` values present in the kept set.
2. For each such `frame_idx`, look up every individual crop with that
   `frame_idx` in the **original, unfiltered** loaded dataset (i.e. before
   quality/dedup/temporal filtering was applied).
3. The union of these becomes the final selection.

This restores companions that would otherwise have been dropped by
quality/dedup/temporal filtering — a selected frame always exports with
complete individual coverage. This step is independent of which upstream
stages ran, so it works whether diversity sampling is enabled or not.

### GUI (`main_window.py`)

Add a checkbox near the diversity controls:

> "Preserve all individuals per frame (for full-frame pose reconstruction)"

When checked, update the target field's helper text to note that the number
is now an approximate final-image-count target (actual output ≈ frames ×
individuals/frame, since a back-solved frame count is used). Default is
unchecked; existing presets are unaffected (leave the toggle off in all of
them).

### Export

No changes to `process_dataset`. It already copies whatever ends up in the
finalized selection by filename, and `metadata.json` is left in place with all
original per-crop data intact.

## Testing

New tests in `tests/test_filterkit_core.py`:

- Frame-level feature aggregation produces one vector per unique `frame_idx`.
- Target back-solving picks a sensible `n_frames` given a known
  `avg_individuals_per_frame`.
- Expansion restores companions that were removed by quality/dedup/temporal
  filtering, when their frame was otherwise selected.
- With `preserve_full_frames=False`, behavior is byte-for-byte identical to
  today.

## Open questions / risks

None outstanding — scope has been deliberately kept to FilterKit's selection
and export logic. Downstream reconstruction (full-frame image assembly,
multi-instance PoseKit format) is tracked separately as PoseKit's planned
frame-mode toggle.
