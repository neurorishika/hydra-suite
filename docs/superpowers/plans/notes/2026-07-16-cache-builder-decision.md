# DetectionCacheBuilderWorker decision

**Decision:** KEEP

**Spawned from:** `src/hydra_suite/trackerkit/gui/orchestrators/config.py:3419-3444`
(`ConfigOrchestrator._build_optimizer_detection_cache`), called from
`src/hydra_suite/trackerkit/gui/main_window.py:1019-1027`
(`MainWindow._build_optimizer_detection_cache`), which is reached from the
Parameter Helper's "open optimizer" flow when no valid detection cache exists
yet for the current video/params/frame-range combination (see
`_find_or_plan_optimizer_cache_path` just above it in both files).

**Cache consumed by:**
- `TuningOptimizer` / the scorer, via `DetectionCache(self.detection_cache_path, mode="r")`
  at `src/hydra_suite/core/tracking/optimization/optimizer.py:598`.
- `TrackingPreviewWorker`, via `DetectionCache(self.detection_cache_path, mode="r")`
  at `src/hydra_suite/core/tracking/optimization/optimizer_workers.py:533`, used to render
  preview frames after optimization.

**InferenceRunner equivalent exists:** no — evidence:
- `optimizer_workers.py:40-44` still carries the comment explaining why the legacy
  `DetectionCache` (`mode="w"`/`mode="r"`, `add_frame`, `save`, `close`) is bound directly
  instead of `InferenceRunner`'s `DetectionCacheHandle`.
- Confirmed the constructor/lifecycle divergence directly: `DetectionCacheHandle`
  (`src/hydra_suite/core/inference/cache/store.py:55-66`) is a dataclass built from
  `(path, key)` where `key` is a `CacheKey`, exposes `write_frame(frame_idx, *, result, **_)`
  and `is_valid()`/`covers_frame_range()`, and is only ever constructed inside
  `InferenceRunner` itself (`src/hydra_suite/core/inference/runner.py:195`) — there is no
  public API for a caller to open one standalone the way `DetectionCache(path, mode="w", ...)`
  is opened here.
- The builder (`optimizer_workers.py:407`) calls `cache.add_frame(...)` with 8 positional
  raw-detection fields and `cache.save()`/`cache.close()` — none of which exist on
  `DetectionCacheHandle`.
- All three legacy-API construction sites still present and unchanged:
  `optimizer_workers.py:407` (write), `optimizer_workers.py:533` (read, preview),
  `optimizer.py:598` (read, scorer).
- Separately, `DetectionCacheBuilderWorker.run()` (`optimizer_workers.py:365-367`) still
  directly instantiates `YOLOOBBDetector` and calls `detect_objects_batched` — i.e. it is
  still hard-tied to the legacy YOLO OBB detector class, not just the legacy cache format.
  `create_detector` itself is confirmed gone from `src/`, but this worker never used
  `create_detector` — it bypasses that factory entirely.

**Reasoning:** The bg-sub migration replaced `create_detector`/the detector-factory
indirection, but it did not touch the Parameter Helper's cache-builder path or the
`DetectionCache` vs. `DetectionCacheHandle` divergence. `DetectionCacheHandle` remains an
internal `InferenceRunner` implementation detail with a different constructor (`path, key`
vs. `path, mode, start_frame, end_frame`) and a different write/read contract
(`write_frame`/buffer vs. `add_frame`/`save`/`mode="r"`). Three consumers
(builder, scorer, `TrackingPreviewWorker`) all still depend on the legacy API in its current
form, so replacing the builder alone is not possible without also re-plumbing the scorer and
preview worker through a new public `InferenceRunner`-cache-open API that does not currently
exist. Per the decision rule, uncertainty (or partial convergence) resolves to KEEP, and here
there is no convergence at all — this is a clean KEEP.

**Consequences for this plan:**
- Task 7 deletes utils/batch_optimizer.py: no
- Task 7 deletes tests/test_batch_optimizer.py: no
- Task 7 drops the BatchOptimizer stub in tests/test_tracking_worker_helpers.py:81: no

**Follow-up (not part of this plan):** If a future migration wants to retire the legacy
`DetectionCache` API entirely, it would need to (1) add a public, standalone way to open an
`InferenceRunner`-style cache for read/write outside of a full `InferenceRunner` run, and
(2) migrate `DetectionCacheBuilderWorker`, the optimizer scorer, and `TrackingPreviewWorker`
together in one change, since they share the on-disk format and API. That is out of scope
here.

## End-to-end verification (Task 8) — 2026-07-17

Automated (controller-run, hydra-mps env):
- No functional vestige survives: grep clean for detection_phase / FramePrefetcherBackward /
  g_gpu_accel / the deleted widgets & handler. Remaining hits are SetupPanel's own kept
  _sync_batch_policy_controls (delegates to live sync) and explanatory comments/docstrings.
- The three removed keys appear only in allowlisted files (utils/batch_optimizer.py, its test,
  test_tracking_worker_helpers stub, and comments) — never in a live trackerkit read path.
- All 6 kits import (trackerkit/posekit/classkit/refinekit/detectkit/filterkit).
- Old project config carrying enable_yolo_batching/yolo_batch_size_mode/yolo_manual_batch_size loads
  fine (unknown keys ignored); legacy_detection_runtime_fields still derives.
- Offscreen MainWindow construction: builds cleanly; group box + all its widgets/handler ABSENT
  (g_gpu_accel, chk_enable_yolo_batching, spin_tensorrt_batch, _sync_batch_policy_controls);
  surviving Live Detection Batching present (spin_detection_batch_size, g_live_batching,
  _sync_live_detection_batch_controls). PASS.

Pending (needs a real video + display — user):
- A real forward tracking run to confirm PHASE 1 (InferenceRunner batch pass) + PHASE 2 appear and the
  "Frame prefetching ENABLED/disabled" log line (worker.py) is UNCHANGED vs pre-change for the same
  project. Code analysis + review established use_batched_detection computes identically (only the
  True-defaulting enable_yolo_batching term was dropped), so on default config this is provably a
  no-op; the real run is the final confirmation.
