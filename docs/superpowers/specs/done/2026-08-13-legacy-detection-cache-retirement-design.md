# Legacy Detection Cache Retirement + `<stem>_caches/` Consolidation

**Date:** 2026-08-13
**Status:** Design approved; spec under review
**Branch:** `chore/legacy-cache-retirement` (worktree `.worktrees/legacy-cache-retirement`)

## Motivation

The pipeline carries a dead legacy detection-cache system alongside the modern
`InferenceRunner` cache. Concretely:

- The legacy `DetectionCache` class (`data/detection_cache.py`) writes a
  per-frame-keyed single-file `.npz` schema that **nothing in the current
  pipeline produces**. The worker's write path (`worker.py:951`, `3853-3867`,
  `3942-3952`) is dead: the `detection_cache` local is set to `None` and never
  reassigned.
- Six sites still *read* that legacy format. Five degrade gracefully when the
  file is absent; one (`oriented_video.py`) hard-throws. All can be served by
  the modern `DetectionCacheHandle` (corners/angles/shapes/detection_ids) plus
  canonical geometry recomputed from params — verified none actually consume the
  cached `canonical_affine`/`canvas_dims`/`M_inverse`.
- Two per-video folders coexist confusingly: the modern hidden
  `.inference_cache_<stem>/` (where `detection.npz` et al. really live) and the
  visible `<stem>_caches/`. The latter is where `detection_cache_path` *points*
  (an anchor file nothing writes) and where pose/detected-props caches land
  "alongside" it.
- Tracking profile JSON (`tracking_profile_*.json`) is dumped **into a cache
  folder** in one code branch — cache dirs should not hold logs.

Goal: delete the legacy cache entirely, collapse everything onto
`.inference_cache_<stem>/`, retire the visible `<stem>_caches/` folder, and move
profiling output to `<stem>_logs/`.

## Verified facts (investigation)

1. **Legacy format is write-dead.** Every `DetectionCache(...)` construction in
   the tree is `mode="r"`. The modern `_npz_save`
   (`core/inference/cache/store.py`) writes a distinct flat schema keyed by
   `cache_key`; the two formats are mutually unreadable.
2. **Geometry consumers don't need the cache.** `oriented_video.py` is already
   wired to the modern `DetectionCacheHandle` via a compat shim (legacy import
   is only an `ImportError` fallback); `interpolated_crops.py` and
   `dataset_export.py` use the legacy cache only as optional size/quality
   enrichment with graceful fallbacks. Canonical geometry is recomputed via
   `canonical_geometry_from_params` + `canonical_affine`
   (`core/canonicalization/geometry.py`).
3. **The anchor is `detection_cache_path`.** `plan_tracking_cache`
   (`trackerkit/tracking_cache.py`) → `build_detection_cache_path`
   (`utils/video_artifacts.py`) resolves it into `<stem>_caches/`. Pose
   (`build_individual_properties_cache_path`) and detected-props
   (`build_detected_properties_cache_path`) caches are written "alongside" it,
   so they follow it wherever it points.
4. **Optimizer detection == main detection.** The parameter optimizer sweeps
   downstream filter/assignment/Kalman params only. The OBB detection cache key
   excludes confidence/IoU (applied as post-hoc filters), so the optimizer's key
   is byte-identical to the main pass's. The code **already** reuses the main
   `.inference_cache_<stem>/detection.npz` when it covers the optimizer's frame
   window (`config.py:2957-2962`). The separate `_opt_cache` dir is only the
   fallback for an uncovered window and must stay a distinct file, because the
   modern cache writes whole-file (`np.savez` rewrites the entire `.npz`) — a
   narrow-window write into the shared `detection.npz` would clobber a broader
   main cache.
5. **Apriltag/classify legacy builders are dead.** `build_apriltag_cache_path`,
   `find_existing_apriltag_cache_path`, `build_classify_cache_path`,
   `find_existing_classify_cache_path` have zero callers.
6. **`build_video_log_dir` exists** (`<stem>_logs/`) and is already used by
   session logging — ready to host the profile JSON.

## Design

### A. Delete outright (verified dead)

- `data/detection_cache.py` — the entire `DetectionCache` class and module.
- `worker.py` dead write-path: the `detection_cache = None` local (951), the
  `add_frame` top-up loop (3853-3867), the save/close branches (3942-3952), and
  the stale "only used for background subtraction" comment.
- `utils/video_artifacts.py`: `build_apriltag_cache_path`,
  `find_existing_apriltag_cache_path`, `build_classify_cache_path`,
  `find_existing_classify_cache_path`.
- The legacy-file branch of the cache-validity probe (`config.py:2949`); the
  `isdir` branch already handles modern directories.

### B. Repoint the anchor to `.inference_cache_<stem>/`

- Change `build_detection_cache_path` / `plan_tracking_cache` so
  `detection_cache_path` resolves to `.inference_cache_<stem>/detection.npz`
  (the real modern file), matching the worker's `_resolve_cache_dir()`.
- Pose and detected-props cache writers already place files "alongside"
  `detection_cache_path`, so they follow into `.inference_cache_<stem>/`
  automatically — no separate change.
- `find_existing_detection_cache_path` updated to look in
  `.inference_cache_<stem>/` (directory-form modern cache).

### C. Port live legacy readers to the modern handle

All read `DetectionCacheHandle.read_frame()` → `OBBResult`
(corners/angles/shapes/detection_ids), recomputing canonical geometry from
params where needed:

- `oriented_video.py` — drop the legacy `ImportError` fallback and the
  raise-if-missing; rely on the existing modern shim.
- `interpolated_crops.py` — repoint the optional OBB size lookup
  (`_get_detection_size`) to the modern handle; keep the `REFERENCE_BODY_SIZE`
  fallback.
- `dataset_export.py` — repoint the optional detection-quality enrichment to the
  modern handle; keep the CSV-column fallback.
- **RefineKit** `overlay_utils.py` + `merge_wizard.py` — read
  `.inference_cache_<stem>/detection.npz` via the modern handle instead of
  globbing the legacy `<stem>_detection_cache_*.npz`. This is the one genuine
  format port (legacy per-frame schema → modern flat schema).

### D. Optimizer fallback cache

- Relocate the fallback `_opt_cache` from `<stem>_caches/` to
  `.inference_cache_<stem>/opt/` (a subdir cache directory).
- Drop the `r<pct>`/model filename encoding — redundant with the `cache_key`
  already stored inside the InferenceRunner cache dir.
- Reuse of the main `detection.npz` when it covers the window is unchanged (the
  common path).
- Update `iter_detection_cache_candidates` to scan the new location.

### E. Profiling → `<stem>_logs/`

- Route `export_summary` to
  `build_video_log_dir(video_path, create=True) / f"tracking_profile_{dir_tag}.json"`.
- Delete the `detection_cache_path.parent` branch (`worker.py:4004-4007`) that
  currently pollutes the cache folder. Keep the output-video-adjacent branch if
  a `video_output_path` is set, else use `<stem>_logs/`.

### F. "Clear All Caches"

- Update `tracking.py:410` (`_iter_cache_artifact_paths`) to scan
  `.inference_cache_<stem>/` for the live artifacts.
- **Keep** a legacy `<stem>_caches/` glob **for deletion only**, so users can
  still purge stale folders left by prior versions. Nothing creates them anymore.

## End state

- `<stem>_caches/` has zero writers; the modern pipeline uses only
  `.inference_cache_<stem>/` (+ `<stem>_logs/` for logs/profiles).
- The legacy `DetectionCache` class and dead path builders are gone.
- Optimizer reuses the main detection cache when possible; its fallback lives
  under `.inference_cache_<stem>/opt/`.

## Testing / verification

Pipeline + cache reuse/replay are touched, so per repo convention:

- **Unit/integration:** cache reuse (forward→reuse gate), backward-pass replay
  (must still resolve `detection.npz`), rich-export pose/props column attachment
  (relocated props caches), RefineKit overlay load against the modern cache,
  optimizer fallback build + reuse, "Clear All Caches" removes both new and
  legacy folders.
- **Equivalence harness byte-identical**, MPS (this box) + CUDA (mehek):
  forward + backward + reuse across the clip matrix. Cache-path changes must not
  alter tracking output.
- Confirm a fresh run creates no `<stem>_caches/` folder and writes the profile
  under `<stem>_logs/`.

## Risks / caveats

- **RefineKit format port** is the only behavioral change; needs its own test.
- **No auto-migration/auto-delete** of existing `<stem>_caches/` folders — they
  become inert; "Clear All Caches" can still remove them.
- `oriented_video` heading-hint fields are already `None` on the modern path
  today, so no regression from dropping the legacy heading hints.
- Whole-file cache semantics mean the optimizer fallback stays a distinct file;
  do not attempt to merge it into the shared `detection.npz`.

## Out of scope

- Merging the optimizer fallback into the main cache file (unsafe under
  whole-file write semantics).
- Any change to the modern cache schema, keys, or reuse/coverage logic.
- Auto-deletion or migration of pre-existing `<stem>_caches/` folders.
