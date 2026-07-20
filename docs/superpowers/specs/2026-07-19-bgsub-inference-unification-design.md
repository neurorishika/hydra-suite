# Background-Subtraction InferenceRunner Unification — Design

- **Date:** 2026-07-19
- **Status:** Draft — pending user review
- **Author:** Rishika Mohanta (with Claude Code)
- **Related:**
  - `docs/superpowers/plans/done/2026-07-16-bgsub-inference-stage-design.md` (bgsub-as-inference-stage; established the `bgsub` detection source and its sequential/cache-only contract)
  - `docs/superpowers/plans/notes/2026-07-16-cache-builder-decision.md` (notes that `DetectionCacheHandle` is meant to be owned by `InferenceRunner`, not constructed by callers)

## 1. Problem

Background-subtraction (bgsub) detection was promoted to a first-class `InferenceRunner`
detection source and is fully unified in the **production tracking worker's model path**.
Two vestigial pieces of the pre-unification design remain, both pure historical duplication
with no functional justification:

1. **Preview duplication.** The TrackerKit "Test Detection on Preview" bgsub branch
   (`preview_worker.py:_preview_run_bg_subtraction`, ~lines 377–471) re-implements bgsub
   detection by hand — `BackgroundModel` + `BackgroundMeasurer` + raw OpenCV
   `findContours`/`fitEllipse` + manual size/aspect-ratio filtering — instead of calling
   `InferenceRunner(bgsub_cfg).run_realtime(...)` the way the YOLO preview branch already
   does (`_preview_run_yolo_branch`, lines 906–1006). Its own inline comments admit it is
   written "to match the production tracking pipeline in worker.py," i.e. it is a parallel
   copy that must be kept in sync by hand.

2. **Cache asymmetry.** In `core/tracking/worker.py`, the YOLO path lets `InferenceRunner`
   own its detection cache end-to-end (forward writes, backward `cache_only` replay via
   `load_frame`). The bgsub path instead **suppresses** the runner cache (`cache_dir=None`,
   `worker.py:1158-1163`) and hand-rolls an equivalent `DetectionCacheHandle`
   (`worker.py:1103-1115`, manual `write_frame` at ~2335, manual `read_frame` at ~2243).
   An inline comment (`worker.py:1134-1142`) documents this split as intentional-but-vestigial.

### Verified facts (from source trace, 2026-07-19)

- `InferenceRunner._open_caches` already owns a detection cache for **both** `obb` and
  `bgsub` sources — identical `DetectionCacheHandle` class and `.npz` format; only the cache
  **key** branches on source (`runner.py:179-231`, `keys.py`).
- The runner writes the bgsub detection unconditionally in `run_realtime`
  (`runner.py:503-504`) and serves backward random access as a **pure cache read**
  (`load_frame`, `runner.py:862-879`) — safe despite bgsub's cross-frame statefulness,
  because `filter_for_source` is the identity on the bgsub branch.
- The **only reader** of the worker's hand-rolled `bgsub_detection.npz` is the backward pass
  (`worker.py:2243`), consuming exactly `centroids`, `angles`, `sizes`, `shapes`,
  `confidences`, `detection_ids` — all fields the runner's handle already stores.
  No pose-precompute, optimizer, or rerun path reads it (grep-confirmed).
- The runner's `FrameResult` already exposes `fg_mask` and `bg_u8` **specifically for bgsub
  preview overlays** (Task 10b, `result.py:130-137`), in the RESIZE_FACTOR-scaled coordinate
  space — the exact data the preview overlay needs.

**Conclusion:** both refactors are behavior-preserving cleanups. No cache format, contract,
or statefulness constraint blocks either.

## 2. Goals / Non-Goals

**Goals**
- Delete the duplicated bgsub detection logic in the preview worker; route the preview bgsub
  branch through `InferenceRunner` exactly as the YOLO preview branch does.
- Make the production bgsub tracking path use the `InferenceRunner`-owned detection cache
  (forward + backward), removing the hand-rolled `DetectionCacheHandle` and the `cache_dir=None`
  suppression.
- Leave the two paths (bgsub, YOLO) structurally symmetric in both preview and worker.

**Non-Goals**
- No change to bgsub detection *results* (measurements, corners, convergence behavior).
  This is a routing/ownership refactor, not an algorithm change.
- No change to the `BackgroundModel` / `BackgroundMeasurer` algorithms themselves.
- No change to the classical-CV preview *overlays* the user sees (FG/BG thumbnails, footer
  text, ellipse + centroid drawing) beyond sourcing their data from `FrameResult`.
- No introduction of a stage registry / plug-in seam (explicitly out of scope per the
  bgsub-inference-stage design).
- No change to public CLI entry points or inter-kit APIs.

## 3. Design

### Slice 1 — Preview bgsub via InferenceRunner (low risk, done first)

**Where:** `src/hydra_suite/trackerkit/gui/workers/preview_worker.py`.

Replace the body of `_preview_run_bg_subtraction(...)` so it mirrors
`_preview_run_yolo_branch(...)`:

1. Build a bgsub `InferenceConfig` from the preview context snapshot. Add a
   `_preview_build_bgsub_params(context, resize_f, use_detection_filters)` helper (parallel to
   `_preview_build_inference_params` for YOLO) that maps the preview UI snapshot into the
   UPPER_SNAKE param dict consumed by `BgSubConfig.from_params` / the bgsub arm of
   `build_inference_config_from_params`. It must carry the same size / aspect-ratio /
   min-contour filter params the current hand-rolled loop applies, so
   `use_detection_filters=False` yields the unfiltered set and `True` yields the filtered set
   (via the runner's filtering stage → `FrameResult.filtered_indices`).
2. `runner = InferenceRunner(cfg)` with `cache_dir=None` (preview is a single ad-hoc frame;
   no cache). Call `fr = runner.run_realtime(frame_to_process, roi_mask=roi_for_bgsub)` inside
   `try/finally: runner.close()`.
3. Draw detections from `fr.obb` (centroid, corners→ellipse dims) honoring
   `fr.filtered_indices`, exactly as the YOLO branch reads `fr.obb`.
4. Draw the FG/BG corner thumbnails and footer from `fr.fg_mask` / `fr.bg_u8` (already exposed
   for this purpose) instead of the locally computed masks. The thumbnail/footer *drawing*
   stays preview-local; only its data source changes.
5. Delete the now-unused local machinery in this function: `_build_preview_background_model`
   usage, direct `BackgroundModel.update_and_get_background` / `generate_foreground_mask`
   calls, the `BackgroundMeasurer` conservative-split call, the `findContours`/`fitEllipse`
   loop, and `_preview_bg_size_thresholds` if it becomes dead. (Confirm each is not used by any
   other preview branch before deletion.)

**Return contract preserved:** the function still returns
`(detections, detected_dimensions, test_frame)` (or the reduced tuple `_run_preview_detection_job`
actually consumes — align with the YOLO branch's `(detected_dimensions, test_frame)` if the
first element is unused downstream; verify at `_run_preview_detection_job` lines 1020-1027).

**Known intentional behavior delta:** any preview visualization that depended on internal
intermediates the runner does not expose (mirroring the YOLO branch's dropped "stage-1 box"
overlay) is surfaced as a log line, not silently changed.

### Slice 2 — Worker bgsub cache owned by InferenceRunner (production path)

**Where:** `src/hydra_suite/core/tracking/worker.py`.

Mirror the YOLO cache lifecycle for bgsub:

- **Forward pass:** construct the forward `bgsub_runner` with
  `cache_dir=self._resolve_cache_dir()` instead of `cache_dir=None`, **except in preview
  mode**, where it stays `cache_dir=None` to preserve the existing
  `should_build_bgsub_detection_cache` truncation guard (a short preview range must not write a
  fixed-path full-range cache). Remove the manual `DetectionCacheHandle` construction
  (`~1103-1115`), the manual `write_frame` call (`~2335`), and the forward-only manual `close`
  (`~4122-4124`). Detection is written by the runner in `run_realtime`, as for YOLO.
- **Backward pass:** replace the hand-rolled read-only `DetectionCacheHandle` +
  `read_frame(actual_frame_index)` (`~1116-1133`, `~2243`) with a `cache_only=True`
  `bgsub_runner`, gated by `caches_all_valid()` + `detection_cache_covers_range()`, replayed
  via `load_frame(actual_frame_index)` — exactly the YOLO backward pattern (`worker.py:1007-1037`).
  Read `fr.obb` and feed the same `frame_result_to_meas` bridge already used elsewhere, so the
  backward consumer (`centroids/angles/sizes/shapes/confidences/detection_ids`) is unchanged.
- Delete the now-dead inline comment block explaining the suppression, and the
  `bgsub_detection_cache` attribute.

**On-disk migration:** the bgsub cache file moves from `bgsub_detection.npz` to the runner's
`detection.npz` under the same `.inference_cache_<stem>` dir. Because a given video is either
`obb` or `bgsub` (never both — `InferenceConfig` enforces exactly-one detection source), there
is no filename collision. Existing `bgsub_detection.npz` caches are simply ignored and
recomputed once on the next forward run. This is acceptable (caches are derived data). Optional:
best-effort delete of a stale `bgsub_detection.npz` when writing the new cache — decide during
planning; default is to leave it (harmless orphan).

## 4. Testing / Verification

**Determinism is the acceptance bar** — both slices must be byte-for-byte behavior-preserving
on detection results.

- **Slice 2 (worker cache):** Run the production bgsub forward+backward pipeline on a fixture
  video before and after the change; assert the exported trajectories / detection measurements
  are identical (this is the project's established migration-parity method — see the
  migration-verification memory). Confirm backward mode still refuses to run without a valid
  cache, and that `caches_all_valid`/`detection_cache_covers_range` gate correctly. Confirm
  preview mode still does **not** persist a cache.
- **Slice 1 (preview):** Manual/GUI check that "Test Detection on Preview" in bgsub mode
  produces the same detection count, ellipses, centroids, and FG/BG thumbnails as before on a
  sample frame, with filters on and off. Add a focused unit test if a seam is testable without
  Qt (e.g. `_preview_build_bgsub_params` mapping, or a headless
  `InferenceRunner(bgsub_cfg).run_realtime` on a synthetic frame asserting `fr.fg_mask`/
  `fr.obb` are populated).
- Run `make lint-moderate` and `make dead-code` after each slice (dead-code will confirm the
  removed helpers are truly unused).

## 5. Risks & Mitigations

| Risk | Mitigation |
|---|---|
| Preview overlay subtly differs (e.g. filter application order) from hand-rolled loop | Compare detection counts/dims on fixture frames with filters on and off; the runner filtering stage is the same one production uses, so parity is expected. |
| Backward parity regression in worker (Slice 2) | Byte-identical trajectory diff on fixture video, both platforms if feasible, per established parity gate. |
| Stale `bgsub_detection.npz` left on disk confuses users | Filename change is documented; optional best-effort cleanup. Orphan is harmless (never read). |
| Preview mode accidentally starts persisting a full-path cache | Preserve `cache_dir=None` in preview mode explicitly; assert in test. |

## 6. Sequencing

1. **Slice 1 (preview)** first — smallest blast radius, proves the fresh-caller
   `InferenceRunner(bgsub_cfg)` pattern, easy to eyeball in the GUI.
2. **Slice 2 (worker cache)** second — production forward/backward path; gated by the
   byte-identical parity check.

Each slice is independently committable and independently revertible.
