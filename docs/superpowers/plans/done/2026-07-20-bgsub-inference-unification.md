# Background-Subtraction InferenceRunner Unification — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove the two remaining pieces of pre-unification bgsub duplication — the hand-rolled preview bgsub detection path and the hand-rolled worker bgsub detection cache — so bgsub runs entirely through `InferenceRunner`, symmetric with the YOLO/OBB path.

**Architecture:** Two independent slices. Slice 1 rewrites the TrackerKit preview bgsub branch to call `InferenceRunner(bgsub_cfg).run_realtime(...)` and deletes the duplicated `BackgroundModel`/`BackgroundMeasurer`/OpenCV logic and the preview background cache. Slice 2 makes the production tracking worker let the bgsub `InferenceRunner` own its detection cache (forward writes, backward `cache_only` replay via `load_frame`), deleting the manually-managed `DetectionCacheHandle`.

**Tech Stack:** Python, PySide6 (Qt, preview only), NumPy, OpenCV, pytest. Core inference: `hydra_suite.core.inference` (`InferenceRunner`, `InferenceConfig`, `BgSubConfig`, `FrameResult`).

## Global Constraints

- **Determinism is the acceptance bar for Slice 2.** The production bgsub forward+backward pipeline must produce byte-identical exported trajectories / detection measurements before vs. after. Use the project's established migration-parity method (see the `project_migration_verification` memory).
- **Slice 1 acceptance bar is "matches production bgsub," not "matches old preview."** Routing preview through `InferenceRunner` deliberately picks up lighting-stabilization and the production ROI/size-filter semantics the old preview path had drifted away from. Aspect-ratio filtering (which the old preview loop did but the production `BackgroundMeasurer` does not) is intentionally dropped.
- No change to `BackgroundModel` / `BackgroundMeasurer` algorithms, or to the bgsub `OBBResult` output contract.
- No new stage registry / plug-in seam (out of scope per the bgsub-inference-stage design).
- No public CLI entry points or inter-kit APIs change.
- `detection_method` is `"yolo_obb"` / `"background_subtraction"` (strings) in `worker.py`, and an int (`0` == bgsub) in the preview context. Do not "normalize" these — they are separate call sites.
- bgsub cache dir is `<video>.parent/.inference_cache_<stem>` via `TrackingWorker._resolve_cache_dir()` (worker.py:4259). The runner-owned bgsub cache file is `detection.npz` (same as OBB); the old hand-rolled file was `bgsub_detection.npz`.
- Run `make lint-moderate` after each task that changes `src/`. Run `make dead-code` at the end of each slice to confirm removed helpers are truly unused.
- Commit as the configured git user; do NOT add a `Co-Authored-By: Claude` trailer (see `feedback_git_commit_identity` memory).

---

## Slice 1 — Preview bgsub via InferenceRunner

**File structure for this slice:**
- Modify: `src/hydra_suite/trackerkit/gui/workers/preview_worker.py`
  - Add `_preview_build_bgsub_params(context, use_detection_filters)` (new pure helper).
  - Rewrite `_preview_run_bg_subtraction(...)` to use `InferenceRunner`.
  - Delete now-dead bgsub-preview machinery (background cache + primer + size-threshold helper) once confirmed unused.
- Create: `tests/trackerkit/test_preview_bgsub_params.py` (headless seam test — no Qt, no video).

### Task 1: Add and test `_preview_build_bgsub_params`

**Files:**
- Modify: `src/hydra_suite/trackerkit/gui/workers/preview_worker.py`
- Test: `tests/trackerkit/test_preview_bgsub_params.py`

**Interfaces:**
- Consumes: existing `_build_preview_background_params(context)` (preview_worker.py:124) which returns the UPPER_SNAKE bgsub param dict (`BACKGROUND_PRIME_FRAMES`, `THRESHOLD_VALUE`, `MIN_CONTOUR_AREA`, `MAX_TARGETS`, `MIN_OBJECT_SIZE`, `MAX_OBJECT_SIZE`, `RESIZE_FACTOR`, `ROI_MASK`, `ENABLE_CONSERVATIVE_SPLIT`, …); `hydra_suite.core.inference.config.BgSubConfig` and `InferenceConfig`.
- Produces: `_preview_build_bgsub_params(context: dict, use_detection_filters: bool) -> dict` — an UPPER_SNAKE param dict consumable by `BgSubConfig.from_params(...)`, with `ENABLE_SIZE_FILTERING` toggled by `use_detection_filters` and `RUNTIME_TIER` set.

- [ ] **Step 1: Write the failing test**

Create `tests/trackerkit/test_preview_bgsub_params.py`:

```python
from hydra_suite.core.inference.config import BgSubConfig, InferenceConfig
from hydra_suite.trackerkit.gui.workers.preview_worker import (
    _preview_build_bgsub_params,
)


def _ctx():
    return {
        "fps": 30.0,
        "bg_prime_frames": 30,
        "threshold_value": 25,
        "min_contour": 40,
        "max_targets": 7,
        "resize_factor": 0.5,
        "min_object_size": 0.3,
        "max_object_size": 3.0,
        "reference_body_size": 20.0,
        "runtime_tier": "cpu",
    }


def test_filters_off_disables_size_filtering():
    params = _preview_build_bgsub_params(_ctx(), use_detection_filters=False)
    assert params["ENABLE_SIZE_FILTERING"] is False
    # MIN_CONTOUR_AREA always survives (matches production + old preview).
    assert params["MIN_CONTOUR_AREA"] == 40


def test_filters_on_enables_size_filtering():
    params = _preview_build_bgsub_params(_ctx(), use_detection_filters=True)
    assert params["ENABLE_SIZE_FILTERING"] is True
    assert params["MAX_TARGETS"] == 7


def test_runtime_tier_defaults_to_cpu_when_unknown():
    ctx = _ctx()
    ctx["runtime_tier"] = "bogus"
    params = _preview_build_bgsub_params(ctx, use_detection_filters=True)
    assert params["RUNTIME_TIER"] == "cpu"


def test_builds_a_valid_bgsub_inference_config():
    params = _preview_build_bgsub_params(_ctx(), use_detection_filters=True)
    cfg = InferenceConfig(
        obb=None,
        bgsub=BgSubConfig.from_params(params),
        runtime_tier=params["RUNTIME_TIER"],
    )
    assert cfg.detection_source == "bgsub"
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `python -m pytest tests/trackerkit/test_preview_bgsub_params.py -v`
Expected: FAIL with `ImportError` / `cannot import name '_preview_build_bgsub_params'`.

- [ ] **Step 3: Implement the helper**

In `preview_worker.py`, add directly below `_build_preview_background_params` (after preview_worker.py:156):

```python
def _preview_build_bgsub_params(context: dict, use_detection_filters: bool) -> dict:
    """Assemble an UPPER_SNAKE bg-sub param dict for ``BgSubConfig.from_params``.

    Reuses the existing preview bg-sub param mapping and layers on the two
    knobs the InferenceRunner bg-sub stage reads that the raw mapping omits:

    * ``ENABLE_SIZE_FILTERING`` — the ``BackgroundMeasurer`` only applies the
      MIN/MAX_OBJECT_SIZE window when this is set (measure.py:231). Toggling it
      off is how ``use_detection_filters=False`` yields the unfiltered set;
      MIN_CONTOUR_AREA still applies in both modes (measure.py:210), matching
      the old preview loop and production. Aspect-ratio filtering the old
      preview loop did is intentionally dropped — the production measurer has
      no such filter.
    * ``RUNTIME_TIER`` — the sole runtime knob; bg-sub only uses it to pick the
      grayscale/adjustment device, but the config carries it for parity.
    """
    params = _build_preview_background_params(context)
    params["ENABLE_SIZE_FILTERING"] = bool(use_detection_filters)
    _tier = str(context.get("runtime_tier", "") or "").strip().lower()
    params["RUNTIME_TIER"] = _tier if _tier in {"cpu", "gpu", "gpu_fast"} else "cpu"
    return params
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `python -m pytest tests/trackerkit/test_preview_bgsub_params.py -v`
Expected: PASS (4 passed).

- [ ] **Step 5: Lint**

Run: `make lint-moderate`
Expected: no new findings in `preview_worker.py`.

- [ ] **Step 6: Commit**

```bash
git add src/hydra_suite/trackerkit/gui/workers/preview_worker.py tests/trackerkit/test_preview_bgsub_params.py
git commit -m "feat(trackerkit): add _preview_build_bgsub_params for InferenceRunner bg-sub preview"
```

### Task 2: Route the preview bgsub branch through InferenceRunner

**Files:**
- Modify: `src/hydra_suite/trackerkit/gui/workers/preview_worker.py:377-471` (`_preview_run_bg_subtraction`)

**Interfaces:**
- Consumes: `_preview_build_bgsub_params` (Task 1); `_preview_resize_frame` (preview_worker.py:347); `InferenceRunner` and `InferenceConfig`/`BgSubConfig` (already imported / to import); `FrameResult.obb`, `FrameResult.filtered_indices`, `FrameResult.fg_mask`, `FrameResult.bg_u8`.
- Produces: unchanged public signature `_preview_run_bg_subtraction(frame_bgr, test_frame, context, resize_f, use_detection_filters) -> (detections, detected_dimensions, test_frame)` consumed at preview_worker.py:1020-1023.

- [ ] **Step 1: Add the config imports near the top-of-function imports**

The YOLO branch already imports `InferenceRunner` and `build_inference_config_from_params` at module top (preview_worker.py:14-15). Add the structured-config imports the bg-sub branch needs. At module top, alongside line 14:

```python
from hydra_suite.core.inference.config import (
    BgSubConfig,
    InferenceConfig,
    build_inference_config_from_params,
)
```

(Replace the existing single `build_inference_config_from_params` import line 14 with this grouped import.)

- [ ] **Step 2: Rewrite `_preview_run_bg_subtraction`**

Replace the entire body of `_preview_run_bg_subtraction` (preview_worker.py:377-471) with an InferenceRunner-driven version that mirrors `_preview_run_yolo_branch`. It reads detections from `fr.obb` honoring `fr.filtered_indices`, draws ellipses/centroids, and paints the FG/BG corner thumbnails from `fr.fg_mask` / `fr.bg_u8`:

```python
def _preview_run_bg_subtraction(
    frame_bgr, test_frame, context, resize_f, use_detection_filters
):
    """Run bg-sub preview detection through the shared InferenceRunner stage.

    Behaviour matches PRODUCTION bg-sub (worker.py), not the old hand-rolled
    preview loop: the runner primes the background from the video on each call
    (there is no cross-preview background cache anymore), applies lighting
    stabilization, and filters via BackgroundMeasurer. See the plan's Slice 1
    acceptance note.
    """
    frame_to_process, test_frame = _preview_resize_frame(
        frame_bgr, test_frame, resize_f
    )

    params = _preview_build_bgsub_params(context, use_detection_filters)
    cfg = InferenceConfig(
        obb=None,
        bgsub=BgSubConfig.from_params(params),
        runtime_tier=params["RUNTIME_TIER"],
    )

    roi_for_bgsub = context.get("roi_mask")
    if roi_for_bgsub is not None and resize_f < 1.0:
        roi_for_bgsub = cv2.resize(
            roi_for_bgsub,
            (frame_to_process.shape[1], frame_to_process.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        )

    logger.info("Running bg-sub preview via InferenceRunner.run_realtime")

    runner = InferenceRunner(
        cfg, cache_dir=None, video_path=str(context.get("video_path", "")) or None
    )
    try:
        fr = runner.run_realtime(
            frame_to_process, roi_mask=roi_for_bgsub
        )

        obb = getattr(fr, "obb", None)
        keep = list(getattr(fr, "filtered_indices", []) or [])
        if obb is None:
            keep = []

        detections = []
        detected_dimensions = []
        for i in keep:
            cx, cy = float(obb.centroids[i, 0]), float(obb.centroids[i, 1])
            corners = np.asarray(obb.corners[i], dtype=np.float32)
            major_axis = float(np.linalg.norm(corners[1] - corners[0]))
            minor_axis = float(np.linalg.norm(corners[2] - corners[1]))
            if major_axis < minor_axis:
                major_axis, minor_axis = minor_axis, major_axis
            ang = float(np.degrees(obb.angles[i]))
            area = float(obb.sizes[i])
            detections.append(((cx, cy), (major_axis, minor_axis), ang, area))
            detected_dimensions.append((major_axis, minor_axis))
            cv2.ellipse(
                test_frame,
                ((int(cx), int(cy)), (int(major_axis), int(minor_axis)), ang),
                (0, 255, 0),
                2,
            )
            cv2.circle(test_frame, (int(cx), int(cy)), 3, (0, 0, 255), -1)

        # FG / BG thumbnails come straight from what the stage detected on.
        fg_mask = getattr(fr, "fg_mask", None)
        bg_u8 = getattr(fr, "bg_u8", None)
        if fg_mask is not None:
            small_fg = cv2.resize(fg_mask, (0, 0), fx=0.3, fy=0.3)
            test_frame[0 : small_fg.shape[0], 0 : small_fg.shape[1]] = cv2.cvtColor(
                small_fg, cv2.COLOR_GRAY2BGR
            )
        if bg_u8 is not None:
            small_bg = cv2.resize(bg_u8, (0, 0), fx=0.3, fy=0.3)
            bg_bgr = cv2.cvtColor(small_bg, cv2.COLOR_GRAY2BGR)
            test_frame[0 : bg_bgr.shape[0], -bg_bgr.shape[1] :] = bg_bgr

        prime_frames = params.get("BACKGROUND_PRIME_FRAMES", 0)
        cv2.putText(
            test_frame,
            f"Detections: {len(detections)} (BG from {prime_frames} frames)",
            (10, test_frame.shape[0] - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 255),
            2,
        )
        return detections, detected_dimensions, test_frame
    finally:
        try:
            runner.close()
        except Exception:
            pass
```

- [ ] **Step 3: Manually verify in the GUI**

Launch `trackerkit`, load a video with bg-sub as the detection method, click "Test Detection on Preview" with filters both on and off.
Expected: detections/ellipses/centroids draw; FG thumbnail (top-left) and BG thumbnail (top-right) render; footer shows the count. Output matches production bg-sub tracking on that frame (may differ from the pre-change preview — that is intended, see the acceptance note).

- [ ] **Step 4: Lint**

Run: `make lint-moderate`
Expected: no new findings.

- [ ] **Step 5: Commit**

```bash
git add src/hydra_suite/trackerkit/gui/workers/preview_worker.py
git commit -m "refactor(trackerkit): run preview bg-sub through InferenceRunner (drop duplicated CV path)"
```

### Task 3: Delete the now-dead preview bgsub machinery

**Files:**
- Modify: `src/hydra_suite/trackerkit/gui/workers/preview_worker.py`

**Interfaces:**
- Consumes: nothing new.
- Produces: nothing new. Pure deletion of code no longer referenced after Task 2.

- [ ] **Step 1: Confirm each candidate is unused**

Run (from repo root):

```bash
grep -n "_build_preview_background_model\|_get_cached_preview_background_state\|_store_preview_background_state\|_preview_background_cache_key\|_PREVIEW_BACKGROUND_CACHE\|_build_preview_background_params\|_preview_bg_size_thresholds" src/hydra_suite/trackerkit/gui/workers/preview_worker.py
```

Expected after Task 2: `_build_preview_background_params` still has ONE remaining use (inside `_preview_build_bgsub_params`) — **keep it**. The background-cache trio (`_build_preview_background_model`, `_get_cached_preview_background_state`, `_store_preview_background_state`, `_preview_background_cache_key`, the `_PREVIEW_BACKGROUND_CACHE*` module globals and lock) and `_preview_bg_size_thresholds` should now have NO remaining references outside their own definitions.

- [ ] **Step 2: Delete the dead definitions**

Remove, only if Step 1 confirmed zero external references:
- `_build_preview_background_model` (preview_worker.py:193-222)
- `_preview_bg_size_thresholds` (preview_worker.py:358-375)
- `_get_cached_preview_background_state`, `_store_preview_background_state`, `_preview_background_cache_key` and the `_PREVIEW_BACKGROUND_CACHE`, `_PREVIEW_BACKGROUND_CACHE_LOCK`, `_PREVIEW_BACKGROUND_CACHE_MAX_ENTRIES` module globals (grep their exact line ranges first — they live in the same Region 1 block).
- Any now-unused imports (e.g. a top-level `from ... import BackgroundModel` if one exists; the local `from hydra_suite.core.background.measure import BackgroundMeasurer` was inside the old `_preview_run_bg_subtraction` and is already gone with Task 2).

Keep `_build_preview_background_params` (still used by `_preview_build_bgsub_params`).

- [ ] **Step 3: Run the full preview test + dead-code check**

Run:
```bash
python -m pytest tests/trackerkit/test_preview_bgsub_params.py -v
make dead-code
```
Expected: tests pass; `make dead-code` reports no new findings for the removed symbols (and ideally fewer than before).

- [ ] **Step 4: Lint**

Run: `make lint-moderate`
Expected: no new findings; no unused-import warnings.

- [ ] **Step 5: Commit**

```bash
git add src/hydra_suite/trackerkit/gui/workers/preview_worker.py
git commit -m "refactor(trackerkit): remove dead preview bg-sub background cache + helpers"
```

---

## Slice 2 — Worker bgsub cache owned by InferenceRunner

**File structure for this slice:**
- Modify: `src/hydra_suite/core/tracking/worker.py`
  - Hoist bgsub `InferenceConfig` construction so both forward (live) and backward (`cache_only`) build a runner.
  - Give the forward bgsub runner a real `cache_dir`; best-effort delete a stale `bgsub_detection.npz`.
  - Replace the backward hand-rolled `read_frame` with `cache_only` runner `load_frame`.
  - Delete `bgsub_detection_cache` (`DetectionCacheHandle`), its manual `write_frame`/`close`, and `should_build_bgsub_detection_cache` (fold its preview guard into `cache_dir` selection).

**Interfaces (existing, reused):**
- `TrackingWorker._resolve_cache_dir() -> Path` (worker.py:4259).
- `InferenceRunner(config, *, cache_dir, video_path, cache_only)`; methods `run_realtime(frame, frame_idx, roi_mask=...)`, `load_frame(frame_idx) -> FrameResult`, `caches_all_valid() -> bool`, `detection_cache_covers_range(start, end) -> bool`, `detection_cache_missing_frames(start, end)`, `close()`.
- `frame_result_to_meas(centroids, angles)` (already used at worker.py:2329).
- `InferenceConfig`, `BgSubConfig`, `migrate_runtime_to_tier` (already imported/used at worker.py:1169-1183).

**Testing note for this slice:** The acceptance bar is byte-identical forward+backward parity on a fixture video (Global Constraints). Establish the baseline BEFORE any edit:

```bash
# Baseline: run forward then backward bg-sub tracking on the fixture, archive the CSV export.
# (Use the project's standard tracking-parity harness / the same procedure documented in
#  the project_migration_verification memory.)
```

Re-run the same after Task 6 and diff. There is no unit-test seam for the worker's forward loop; parity IS the test.

### Task 4: Hoist bgsub InferenceConfig construction (no behaviour change)

**Files:**
- Modify: `src/hydra_suite/core/tracking/worker.py:1122-1199`

**Interfaces:**
- Produces: a local `bgsub_inference_config: InferenceConfig` available in BOTH backward and forward bgsub branches (currently built only in the forward `else` at worker.py:1178-1183).

- [ ] **Step 1: Move config construction above the `if self.backward_mode:` split**

In the `else:` (background-subtraction) block starting at worker.py:1122, build `bgsub_inference_config` once, before the `if self.backward_mode:` at worker.py:1142. Cut the tier-derivation + `InferenceConfig(...)` block (currently worker.py:1169-1183) out of the forward `else` and place it right after the block comment at ~1123, e.g.:

```python
        else:
            # ── Background subtraction ───────────────────────────────────────
            # Detection runs through InferenceRunner's bg-sub stage, which owns
            # the detection cache for BOTH forward (live writes) and backward
            # (cache_only replay) — full parity with the yolo_obb path.
            from hydra_suite.core.inference.config import migrate_runtime_to_tier

            _compute_runtime = str(p.get("COMPUTE_RUNTIME", "cpu"))
            _raw_tier = str(p.get("RUNTIME_TIER", "") or "").strip().lower()
            _runtime_tier = (
                _raw_tier
                if _raw_tier in {"cpu", "gpu", "gpu_fast"}
                else migrate_runtime_to_tier({_compute_runtime})
            )
            bgsub_inference_config = InferenceConfig(
                obb=None,
                bgsub=BgSubConfig.from_params(p),
                runtime_tier=_runtime_tier,
                detection_batch_size=int(p.get("DETECTION_BATCH_SIZE", 1) or 1),
            )
            _cache_dir = self._resolve_cache_dir()
```

Leave the existing `if self.backward_mode:` / `else:` bodies in place for now (Tasks 5–6 rewrite them). This step only relocates config construction and computes `_cache_dir` once.

- [ ] **Step 2: Sanity-run import/syntax**

Run: `python -c "import hydra_suite.core.tracking.worker"`
Expected: no error.

- [ ] **Step 3: Lint + commit**

```bash
make lint-moderate
git add src/hydra_suite/core/tracking/worker.py
git commit -m "refactor(tracking): hoist bg-sub InferenceConfig for shared forward/backward use"
```

### Task 5: Forward bgsub uses the runner-owned cache

**Files:**
- Modify: `src/hydra_suite/core/tracking/worker.py` — forward bgsub construction (worker.py:1160-1199 region), forward write (worker.py:2359-2363), close (worker.py:4146-4160).

**Interfaces:**
- Produces: forward `bgsub_runner` built with `cache_dir=_cache_dir` in non-preview mode (else `None`), owning its `detection.npz`. Removes `bgsub_detection_cache` writes/close.

- [ ] **Step 1: Replace the forward `else` body**

Replace the forward-branch body (the current `else:` at worker.py:1160 that logs and — after Task 4 — no longer builds the config) with runner construction that owns the cache. In preview mode keep `cache_dir=None` (preserves the old `should_build_bgsub_detection_cache` truncation guard); otherwise pass `_cache_dir` and best-effort delete the stale legacy file:

```python
            else:
                # Forward pass. The runner owns the detection cache exactly like
                # the yolo_obb path. Preview mode uses cache_dir=None: the cache
                # file is one fixed path per video (not qualified by frame range),
                # so a short preview range must not write/truncate it.
                _fwd_cache_dir = None if self.preview_mode else _cache_dir
                if _fwd_cache_dir is not None:
                    _fwd_cache_dir.mkdir(parents=True, exist_ok=True)
                    # Best-effort remove the pre-unification hand-rolled cache so
                    # it does not linger as a confusing orphan. Never fatal.
                    try:
                        (_fwd_cache_dir / "bgsub_detection.npz").unlink(
                            missing_ok=True
                        )
                    except Exception:
                        logger.debug(
                            "Could not remove stale bgsub_detection.npz (non-fatal)",
                            exc_info=True,
                        )
                bgsub_runner = InferenceRunner(
                    bgsub_inference_config,
                    cache_dir=_fwd_cache_dir,
                    video_path=self.video_path,
                    cache_only=False,
                )
                if _fwd_cache_dir is not None:
                    logger.info(
                        "Forward pass caching bg-sub detections via InferenceRunner "
                        "to %s",
                        _fwd_cache_dir / "detection.npz",
                    )
                else:
                    logger.info(
                        "Preview mode: bg-sub runner has no cache (cache_dir=None)."
                    )
```

- [ ] **Step 2: Delete the manual forward write**

Remove the `if bgsub_detection_cache is not None: bgsub_detection_cache.write_frame(...)` block at worker.py:2359-2363 entirely (the runner writes each frame inside `run_realtime`, as it already does for `_bgsub_result`). Also remove the now-stale explanatory comment immediately above it (worker.py:2354-2358).

- [ ] **Step 3: Delete the manual forward close**

Remove the `if bgsub_detection_cache is not None and not self.backward_mode:` close block at worker.py:4142-4148. Keep the `bgsub_runner.close()` block (worker.py:4156-4160) but update its comment — the runner now owns and flushes its cache:

```python
        # Flush/close the bg-sub runner. On a forward pass with a cache_dir this
        # writes the per-frame detection cache to disk (parity with yolo_obb); on
        # preview (cache_dir=None) and backward (cache_only) it is a resource
        # release / no-op flush.
        if bgsub_runner is not None:
            bgsub_runner.close()
```

- [ ] **Step 4: Remove the `bgsub_detection_cache` declaration**

Delete the `bgsub_detection_cache = None` initializer (worker.py:998) and its comment (worker.py:996-997). (Backward reads are rewired in Task 6; if Task 6 is done after this in the same session, the backward branch at worker.py:2261-2299 still references `bgsub_detection_cache` — do Task 6 in the same commit or immediately after so the module stays importable. To keep steps independently runnable, defer deleting the declaration to Task 6 Step 4.)

**Revised Step 4:** Do NOT delete the declaration here (Task 6 removes it together with its last reader). Leave `bgsub_detection_cache = None` in place for this task.

- [ ] **Step 5: Import/syntax + lint**

Run: `python -c "import hydra_suite.core.tracking.worker"` then `make lint-moderate`
Expected: no error / no new findings.

- [ ] **Step 6: Commit**

```bash
git add src/hydra_suite/core/tracking/worker.py
git commit -m "refactor(tracking): forward bg-sub cache owned by InferenceRunner"
```

### Task 6: Backward bgsub replays via cache_only runner

**Files:**
- Modify: `src/hydra_suite/core/tracking/worker.py` — backward branch (worker.py:1142-1159), backward replay read (worker.py:2261-2299), declaration cleanup (worker.py:996-998).

**Interfaces:**
- Consumes: `bgsub_inference_config` (Task 4), `_cache_dir` (Task 4), `InferenceRunner(..., cache_only=True)`, `load_frame`.
- Produces: backward bgsub replay uses `bgsub_runner.load_frame(...).obb`; `bgsub_detection_cache` fully removed.

- [ ] **Step 1: Replace the backward branch construction**

Replace the `if self.backward_mode:` body (worker.py:1142-1159) with a `cache_only` runner gated exactly like the yolo_obb backward path (worker.py:1036-1068):

```python
            if self.backward_mode:
                _cache_dir.mkdir(parents=True, exist_ok=True)
                bgsub_runner = InferenceRunner(
                    bgsub_inference_config,
                    cache_dir=_cache_dir,
                    video_path=self.video_path,
                    cache_only=True,
                )
                if not bgsub_runner.caches_all_valid() or not (
                    bgsub_runner.detection_cache_covers_range(start_frame, end_frame)
                ):
                    logger.error(
                        "Backward tracking requires a valid forward bg-sub "
                        "detection cache covering frames %d-%d at %s. Please run "
                        "forward tracking over the full range first.",
                        start_frame,
                        end_frame,
                        _cache_dir / "detection.npz",
                    )
                    bgsub_runner.close()
                    cap.release()
                    self.finished_signal.emit(False, [], [])
                    return
                use_cached_detections = True
                logger.info(
                    "Backward pass: replaying cached bg-sub detections via "
                    "InferenceRunner from %s",
                    _cache_dir / "detection.npz",
                )
```

Note `start_frame`/`end_frame` are already in scope (used by the yolo_obb backward gate at worker.py:1046-1047).

- [ ] **Step 2: Rewrite the backward replay read**

Replace the backward replay branch (worker.py:2261-2299) so its guard keys off `bgsub_runner` and it reads a `FrameResult` via `load_frame`:

```python
            elif (
                use_cached_detections
                and detection_method == "background_subtraction"
                and bgsub_runner is not None
            ):
                # Backward pass: replay cached bg-sub detections (no live frame).
                _fr = bgsub_runner.load_frame(actual_frame_index)
                _obb = _fr.obb if _fr is not None else None
                if _obb is not None and _obb.num_detections > 0:
                    meas = frame_result_to_meas(_obb.centroids, _obb.angles)
                    sizes = [float(_obb.sizes[i]) for i in range(_obb.num_detections)]
                    shapes = [
                        (float(_obb.shapes[i, 0]), float(_obb.shapes[i, 1]))
                        for i in range(_obb.num_detections)
                    ]
                    detection_confidences = [
                        float(_obb.confidences[i]) for i in range(_obb.num_detections)
                    ]
                    detection_ids = [
                        int(_obb.detection_ids[i]) for i in range(_obb.num_detections)
                    ]
                else:
                    meas, sizes, shapes, detection_confidences, detection_ids = (
                        [],
                        [],
                        [],
                        [],
                        [],
                    )
                filtered_obb_corners = []
                raw_meas = meas
                raw_sizes = sizes
                raw_shapes = shapes
                raw_confidences = detection_confidences
                raw_obb_corners = filtered_obb_corners
                raw_detection_ids = detection_ids
                raw_heading_hints = []
                raw_heading_confidences = []
                raw_directed_mask = []
                raw_canonical_affines = None
```

- [ ] **Step 3: Delete `should_build_bgsub_detection_cache`**

Remove the helper at worker.py:93-103 and its import/usage if any remain (its only caller was the old forward guard, replaced in Task 5). Grep to confirm:

```bash
grep -rn "should_build_bgsub_detection_cache" src/ tests/
```
Expected: no matches after deletion. If a test references it, delete that test (it covered removed behaviour).

- [ ] **Step 4: Remove the `bgsub_detection_cache` declaration and dead imports**

- Delete `bgsub_detection_cache = None` and its comment (worker.py:996-998).
- Grep for remaining references:
```bash
grep -n "bgsub_detection_cache\b" src/hydra_suite/core/tracking/worker.py
```
Expected: no matches.
- If `DetectionCacheHandle` / `bgsub_detection_cache_key` / `with_video_signature` / `video_signature` are now unused in worker.py, remove those imports. Confirm each:
```bash
grep -n "DetectionCacheHandle\|bgsub_detection_cache_key\|with_video_signature\|video_signature" src/hydra_suite/core/tracking/worker.py
```
Remove only the ones with zero remaining non-import references.

- [ ] **Step 5: Import/syntax + lint + dead-code**

```bash
python -c "import hydra_suite.core.tracking.worker"
make lint-moderate
make dead-code
```
Expected: import OK; no new lint findings; `make dead-code` shows the removed symbols gone (no new findings).

- [ ] **Step 6: Byte-identical parity verification**

Re-run the forward+backward bg-sub tracking on the fixture video (same procedure as the pre-edit baseline in this slice's testing note) and diff the exported CSV / measurements against the archived baseline.
Expected: byte-identical. Also confirm: (a) backward mode still errors cleanly when no forward cache exists; (b) preview mode writes no cache; (c) the on-disk cache is now `detection.npz` and any old `bgsub_detection.npz` was removed on the forward run.

If the diff is NOT identical, STOP and debug before committing — this is the acceptance gate.

- [ ] **Step 7: Commit**

```bash
git add src/hydra_suite/core/tracking/worker.py
git commit -m "refactor(tracking): backward bg-sub replays via InferenceRunner cache_only (drop hand-rolled cache)"
```

---

## Self-Review (completed during authoring)

- **Spec coverage:** Slice 1 (preview dedup) → Tasks 1-3. Slice 2 (worker cache unification, forward + backward) → Tasks 4-6. Orphan `bgsub_detection.npz` removal → Task 5 Step 1. Preview `cache_dir=None` guard preserved → Task 5 Step 1. Determinism gate → Task 6 Step 6. Seam test → Task 1. All spec sections mapped.
- **Placeholder scan:** No TBD/TODO; every code step shows real code; the one parity "baseline" step references the project's existing harness (documented in memory) rather than inventing commands, which is correct for this repo.
- **Type consistency:** `_preview_build_bgsub_params(context, use_detection_filters)` signature identical in Task 1 (def) and Task 2 (call). `bgsub_inference_config` / `_cache_dir` produced in Task 4, consumed in Tasks 5-6. `bgsub_runner` used consistently (live in Task 5, cache_only in Task 6). Backward read switched from `bgsub_detection_cache.read_frame` → `bgsub_runner.load_frame().obb` consistently in Task 6 Steps 1-2.
- **Ordering caveat made explicit:** Task 5's declaration-deletion is deferred to Task 6 (noted inline) so the module stays importable between commits.
