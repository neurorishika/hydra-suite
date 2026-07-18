# Detector Retirement — Plan 2: Consumer Migration (Phase C) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Migrate every remaining production consumer off `core/detectors` (`YOLOOBBDetector`, `DetectionFilter`, `_utils._advanced_config_value`) onto `core/inference` (`InferenceRunner`, `build_*_config`, `api.apply_detection_filter`, a shared pose-backend shim), so that after this plan the only `core/detectors` references left are in tests and `core/inference/direct_executors` internals.

**Architecture:** `InferenceRunner` is the one pipeline. Detection-only consumers use `build_obb_only_config` + `InferenceRunner.detect_batch` (dataset) or `run_realtime` (single-frame preview/model-test). Filtering consumers use `api.apply_detection_filter`. Pose-backend construction (duplicated ×4) collapses onto one shim over `stages/pose.load_pose_model`. The optimizer's detection-cache builder is deleted in favor of `run_batch_pass`.

**Tech Stack:** Python 3, PySide6, ultralytics/torch/numpy/opencv, pytest.

## Global Constraints

- **Depends on Plan 1 (Foundation) being fully landed.** Requires `build_inference_config_from_params`, `build_obb_only_config`, `InferenceRunner.detect_batch`, and the fixed `api.predict_pose_for_image` from Plan 1.
- No behavior change to detection/tracking outputs. Where a consumer produced specific tuple shapes or dimensions, the migrated code must produce the same.
- Do not delete any `core/detectors` file in this plan — only stop *importing* it from production code. Deletion is Plan 4. (`core/detectors` must remain importable so the negative-assertion tests that patch `hydra_suite.core.detectors.YOLOOBBDetector` still work.)
- Run `make format` before each commit; tests via `python -m pytest ... -q`.

---

## File Structure

- `src/hydra_suite/data/dataset_generation.py` — **modify** detection functions to use `InferenceRunner.detect_batch`.
- `src/hydra_suite/core/inference/api.py` — **add** `load_pose_backend(...)` shim.
- `src/hydra_suite/posekit/gui/workers.py`, `trackerkit/gui/workers/preview_worker.py`, `trackerkit/gui/workers/crops_worker.py`, `trackerkit/benchmarking.py` — **modify** to call the pose shim.
- `src/hydra_suite/core/tracking/optimization/optimizer.py` — **modify** filter read to `api.apply_detection_filter`; remove `DetectionFilter` dependency.
- `src/hydra_suite/core/tracking/optimization/optimizer_workers.py` + `trackerkit/gui/orchestrators/config.py` — **replace** `DetectionCacheBuilderWorker` (YOLOOBBDetector) with an `InferenceRunner.run_batch_pass` builder.
- `src/hydra_suite/trackerkit/gui/workers/preview_worker.py` — **modify** the YOLO branch to `InferenceRunner.run_realtime` (keep drawing + bg-sub branch).
- `src/hydra_suite/trackerkit/gui/dialogs/model_test_dialog.py` — **modify** to run inference via the runner/executor without private `stages.obb` imports.
- `src/hydra_suite/detectkit/gui/prediction_preview.py` — **modify** raw-ultralytics path to `load_obb_executor`/runner.
- Tests: one per task under `tests/`.

---

### Task 1: Migrate `dataset_generation.py` dimension extraction to `detect_batch`

Current: `_init_yolo_detector` returns a `YOLOOBBDetector`; `_detect_batch` calls `detect_objects_batched` (5-tuple `(meas, sizes, shapes, confidences, obb_corners)`) with a single-frame fallback `detect_objects` (5-tuple `(meas, sizes, shapes, yolo_results, confidences)`), temporarily overriding conf/iou with dataset thresholds. Replace the detector with an `InferenceRunner` built from an OBB-only config carrying the dataset thresholds; `detect_batch` returns `list[OBBResult]`, from which we reconstruct `(meas, shapes, obb_corners)` for `_measurements_to_detections`.

**Files:**
- Modify: `src/hydra_suite/data/dataset_generation.py` (functions at 413–428, 528–540, 543–563, 566–594, 922–933, call site 825)
- Test: `tests/test_dataset_generation_detect.py`

**Interfaces:**
- Consumes: `build_obb_only_config(model_path, *, compute_runtime, confidence_threshold, iou_threshold, max_targets, mode) -> InferenceConfig` and `InferenceRunner(config).detect_batch(frames, frame_indices=None) -> list[OBBResult]` (Plan 1). `OBBResult` fields: `.centroids (N,2)`, `.angles (N,)`, `.shapes (N,2)`, `.corners (N,4,2)`.
- Produces: `_init_detection_runner(params) -> InferenceRunner | None` (replaces `_init_yolo_detector`); unchanged `_detect_batch(runner, batch_frames, batch_frame_ids, valid_batch_indices, params) -> list[dict]` contract.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_dataset_generation_detect.py
import numpy as np

import hydra_suite.data.dataset_generation as dg
from hydra_suite.core.inference.result import OBBResult


class _FakeRunner:
    def __init__(self, per_frame):
        self._per_frame = per_frame

    def detect_batch(self, frames, frame_indices=None):
        out = []
        for n in self._per_frame[: len(frames)]:
            out.append(
                OBBResult(
                    frame_idx=0,
                    centroids=np.tile([5.0, 6.0], (n, 1)).astype(np.float32),
                    angles=np.zeros(n, np.float32),
                    sizes=np.ones(n, np.float32),
                    shapes=np.tile([100.0, 2.0], (n, 1)).astype(np.float32),
                    confidences=np.ones(n, np.float32),
                    corners=np.zeros((n, 4, 2), np.float32),
                    detection_ids=OBBResult.make_detection_ids(0, n),
                )
            )
        return out


def test_detect_batch_produces_one_dict_per_frame(monkeypatch):
    runner = _FakeRunner(per_frame=[2, 0])
    frames = [np.zeros((8, 8, 3), np.uint8), np.zeros((8, 8, 3), np.uint8)]
    params = {"RESIZE_FACTOR": 1.0}
    results = dg._detect_batch(runner, frames, [0, 1], [0, 1], params)
    assert isinstance(results, list) and len(results) == 2
    # frame 0 has 2 detections, frame 1 has 0
    assert results[1] == {} or len(results[1]) == 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_dataset_generation_detect.py -q`
Expected: FAIL (current `_detect_batch` expects a `detect_objects_batched` detector, not a runner).

- [ ] **Step 3: Replace the detector init**

Replace `_init_yolo_detector` (lines 413–428) with:

```python
def _init_detection_runner(params):
    """Build a detection-only InferenceRunner for dataset dimension extraction.

    Returns None for non-yolo_obb methods (dimension extraction then falls back
    to reference-size approximation, as before).
    """
    detection_method = params.get("DETECTION_METHOD", "background_subtraction")
    if detection_method != "yolo_obb":
        return None
    try:
        from hydra_suite.core.inference.config import build_obb_only_config
        from hydra_suite.core.inference.runner import InferenceRunner

        model_path = str(
            params.get(
                "YOLO_OBB_DIRECT_MODEL_PATH",
                params.get("YOLO_MODEL_PATH", "yolo26s-obb.pt"),
            )
            or "yolo26s-obb.pt"
        )
        cfg = build_obb_only_config(
            model_path,
            compute_runtime=str(params.get("COMPUTE_RUNTIME", "cpu")),
            confidence_threshold=float(params.get("DATASET_YOLO_CONFIDENCE_THRESHOLD", 0.05)),
            iou_threshold=float(params.get("DATASET_YOLO_IOU_THRESHOLD", 0.5)),
            max_targets=max(1, int(params.get("MAX_TARGETS", 8))),
            mode=str(params.get("YOLO_OBB_MODE", "direct")).strip().lower(),
        )
        runner = InferenceRunner(cfg)
        logger.info("Detection runner initialized for dimension extraction")
        return runner
    except Exception as e:
        logger.warning(
            f"Could not initialize detection runner: {e}. Using reference size approximation."
        )
        return None
```

- [ ] **Step 4: Replace `_detect_batch` + its helpers**

Delete `_run_batched_detection` (528–540) and `_run_single_frame_detections` (543–563), and replace `_detect_batch` (566–594) with the runner-based version. The dataset conf/iou thresholds are already baked into the runner's config (Step 3), so the temporary param override is no longer needed:

```python
def _detect_batch(runner, batch_frames, batch_frame_ids, valid_batch_indices, params):
    """Run OBB detection on a batch via InferenceRunner, returning detection dicts."""
    if runner is None or not batch_frames:
        return [{}] * len(batch_frames)
    resize_factor = params.get("RESIZE_FACTOR", 1.0)
    try:
        results = runner.detect_batch(batch_frames, frame_indices=list(batch_frame_ids))
    except Exception as e:
        logger.warning(f"Detection failed: {e}")
        return [{}] * len(batch_frames)
    out = []
    for obb in results:
        meas = np.concatenate([obb.centroids, obb.angles[:, None]], axis=1)
        out.append(
            _measurements_to_detections(meas, obb.shapes, resize_factor, obb.corners)
        )
    return out
```

- [ ] **Step 5: Update `_get_detector_batch_size` and the call site**

Replace `_get_detector_batch_size` (922–933) with a runner-aware version (the batch size now comes from the config's `detection_batch_size`):

```python
def _get_detector_batch_size(runner):
    """Return the batch size to use for detection."""
    if runner is not None and getattr(runner, "config", None) is not None:
        return max(1, int(getattr(runner.config, "detection_batch_size", 1)))
    return 1
```

At the call site (line 825), rename the local + init call: `detector = _init_yolo_detector(params)` → `runner = _init_detection_runner(params)`; update the two downstream uses (`_get_detector_batch_size(detector)` → `_get_detector_batch_size(runner)`, `_detect_batch(detector, ...)` → `_detect_batch(runner, ...)`). After the export loop, close the runner: add `if runner is not None: runner.close()` in the function's cleanup path.

- [ ] **Step 6: Run test + confirm no YOLOOBBDetector import remains**

Run:
```bash
python -m pytest tests/test_dataset_generation_detect.py -q
grep -n "YOLOOBBDetector\|_init_yolo_detector\|detect_objects" src/hydra_suite/data/dataset_generation.py
```
Expected: test PASS; grep prints nothing.

- [ ] **Step 7: Commit**

```bash
make format
git add -A
git commit -m "refactor(data): dataset dimension extraction via InferenceRunner.detect_batch"
```

---

### Task 2: Add a shared pose-backend shim and migrate the ×4 duplicates

The identical runtime-string → backend-family gate + `YoloNativeBackend` / `create_pose_backend_from_config` construction is copy-pasted in `posekit/gui/workers.py:21-109`, `preview_worker.py:1339-1400`, `crops_worker.py:128-235`, and `benchmarking.py:1190-1334`. Centralize it as one shim, then repoint all four.

**Files:**
- Modify: `src/hydra_suite/core/inference/api.py` (add `load_pose_backend`)
- Modify: `posekit/gui/workers.py`, `preview_worker.py`, `crops_worker.py`, `trackerkit/benchmarking.py`
- Test: `tests/test_inference_pose_backend_shim.py`

**Interfaces:**
- Consumes: `stages/pose.load_pose_model(config: PoseConfig, runtime: RuntimeContext) -> PoseModel`.
- Produces: `load_pose_backend(*, backend_family: str, model_path: str, compute_runtime: str, keypoint_names=None, confidence_threshold: float = 1e-4, batch_size: int = 64, min_valid_confidence: float = 0.2) -> object` returning a backend exposing `predict_batch(images) -> list` (the same object the GUI workers use today).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_inference_pose_backend_shim.py
import hydra_suite.core.inference.api as api


def test_load_pose_backend_yolo_dispatch(monkeypatch):
    seen = {}

    class _FakeBackend:
        def predict_batch(self, imgs):
            return []

    def fake_load_pose_model(config, runtime):
        seen["backend"] = config.backend
        seen["yolo_path"] = config.yolo.model_path if config.yolo else None
        return _FakeBackend()

    monkeypatch.setattr(
        "hydra_suite.core.inference.stages.pose.load_pose_model", fake_load_pose_model
    )
    backend = api.load_pose_backend(
        backend_family="yolo", model_path="p.pt", compute_runtime="cpu"
    )
    assert hasattr(backend, "predict_batch")
    assert seen["backend"] == "yolo"
    assert seen["yolo_path"] == "p.pt"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_inference_pose_backend_shim.py -q`
Expected: FAIL — `AttributeError: module ... has no attribute 'load_pose_backend'`.

- [ ] **Step 3: Add the shim to `api.py`**

```python
def load_pose_backend(
    *,
    backend_family: str,
    model_path: str,
    compute_runtime: str,
    keypoint_names=None,
    confidence_threshold: float = 1e-4,
    batch_size: int = 64,
    min_valid_confidence: float = 0.2,
):
    """Build a pose backend (YOLO or SLEAP) via the canonical stages/pose loader.

    Single source of the runtime-string -> backend construction that GUI pose
    workers previously duplicated. Returns a backend exposing predict_batch().
    """
    from .config import PoseConfig, PoseSLEAPConfig, PoseYOLOConfig
    from .runtime import RuntimeContext
    from .stages.pose import load_pose_model

    family = (backend_family or "").strip().lower()
    if family == "yolo":
        pose_cfg = PoseConfig(
            backend="yolo",
            yolo=PoseYOLOConfig(
                model_path=model_path,
                compute_runtime=compute_runtime,
                confidence_threshold=confidence_threshold,
                batch_size=batch_size,
            ),
            min_keypoint_confidence=min_valid_confidence,
        )
    else:
        pose_cfg = PoseConfig(
            backend="sleap",
            sleap=PoseSLEAPConfig(
                model_path=model_path,
                compute_runtime=compute_runtime,
                batch_size=batch_size,
            ),
            min_keypoint_confidence=min_valid_confidence,
        )
    from .config import InferenceConfig, OBBConfig, OBBDirectConfig

    _min_cfg = InferenceConfig(
        obb=OBBConfig(
            mode="direct",
            direct=OBBDirectConfig(model_path="", compute_runtime=compute_runtime),
        ),
        pose=pose_cfg,
    )
    runtime = RuntimeContext.from_config(_min_cfg)
    return load_pose_model(pose_cfg, runtime)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_inference_pose_backend_shim.py -q`
Expected: PASS.

- [ ] **Step 5: Repoint the four duplicate sites**

In each site, replace the inline backend-construction block with a call to `api.load_pose_backend(...)`, threading the site's existing `compute_runtime`, `model_path`, `backend_family`, `keypoint_names`, conf, and batch size. Sites:
- `posekit/gui/workers.py:21-109` — `_build_pose_backend` becomes a thin wrapper delegating to `api.load_pose_backend`; keep its signature so `PosePredictWorker`/`BulkPosePredictWorker` are unchanged.
- `preview_worker.py:1339-1400` — inside `_preview_run_pose_overlay`, replace the inline block (leave the surrounding crop/draw code).
- `crops_worker.py:128-235` — `_init_pose_backend` delegates to the shim.
- `trackerkit/benchmarking.py:1190-1334` — `bench_pose` backend build delegates to the shim.

Keep each site's legacy fallback (`posekit/gui/workers.py:253/454`) and SLEAP re-raise gate untouched.

- [ ] **Step 6: Verify**

Run:
```bash
python -m pytest tests/test_inference_pose_backend_shim.py tests/ -m "not benchmark" -k "pose" -q
grep -rn "HYDRA_SLEAP_FLAVOR" src/hydra_suite/trackerkit src/hydra_suite/posekit
```
Expected: pose tests pass; the `HYDRA_SLEAP_FLAVOR` gate now appears only inside `stages/pose.py` (the shim's loader), not duplicated in GUI workers. (If any GUI site still needs the env override, confirm `load_pose_model` honors it — it does, per the stage's loader.)

- [ ] **Step 7: Commit**

```bash
make format
git add -A
git commit -m "refactor(inference): centralize pose-backend construction in api.load_pose_backend"
```

---

### Task 3: Route `optimizer.py` filtering through `api.apply_detection_filter`

`optimizer.py:_filter_cached_detections` already has a new-cache (`OBBResult` → `apply_detection_filter`) branch and a legacy 12-tuple (`DetectionFilter.filter_raw_detections`) branch. With the cache builder migrated to InferenceRunner (Task 4), caches are all `OBBResult`; drop the legacy branch and the `DetectionFilter` import.

**Files:**
- Modify: `src/hydra_suite/core/tracking/optimization/optimizer.py` (`_filter_cached_detections` 112–200)
- Test: `tests/test_tracking_optimizer_helpers.py` (extend/adjust)

**Interfaces:**
- Consumes: `api.apply_detection_filter(raw: OBBResult, config: OBBConfig) -> OBBResult` (already imported as `_apply_detection_filter`).
- Produces: `_filter_cached_detections(det_filter, cache, f_idx, roi_mask)` unchanged signature, `OBBResult`-only.

- [ ] **Step 1: Write the failing test**

```python
# add to tests/test_tracking_optimizer_helpers.py
import numpy as np

from hydra_suite.core.inference.result import OBBResult
from hydra_suite.core.tracking.optimization.optimizer import _filter_cached_detections


class _ParamsFilter:
    def __init__(self, p):
        self.params = p


class _Cache:
    def __init__(self, obb):
        self._obb = obb

    def get_frame(self, f_idx):
        return self._obb


def test_filter_cached_detections_obb_only_path():
    obb = OBBResult(
        frame_idx=0,
        centroids=np.array([[1.0, 2.0]], np.float32),
        angles=np.array([0.3], np.float32),
        sizes=np.array([50.0], np.float32),
        shapes=np.array([[50.0, 2.0]], np.float32),
        confidences=np.array([0.9], np.float32),
        corners=np.zeros((1, 4, 2), np.float32),
        detection_ids=OBBResult.make_detection_ids(0, 1),
    )
    meas, shapes, confs, ids, hints, directed = _filter_cached_detections(
        _ParamsFilter({"DETECTION_CONFIDENCE": 0.0}), _Cache(obb), 0, None
    )
    assert len(meas) == 1 and len(ids) == 1
```

- [ ] **Step 2: Run test to verify it fails or passes-for-wrong-reason**

Run: `python -m pytest tests/test_tracking_optimizer_helpers.py -k obb_only -q`
Expected: PASS already (the OBBResult branch exists). This test pins the behavior we keep; proceed to remove the dead legacy branch.

- [ ] **Step 3: Remove the legacy 12-tuple branch**

In `_filter_cached_detections`, delete everything after the `if isinstance(frame_data, _OBBResult):` block's `return` (the entire `# Legacy 12-tuple path` section, lines ~150–200), and replace with a hard error so a stale legacy cache fails loudly instead of silently using a removed path:

```python
    raise TypeError(
        "optimizer detection cache must contain OBBResult frames "
        f"(got {type(frame_data).__name__}); rebuild the cache with the "
        "current InferenceRunner-based builder."
    )
```

Remove the now-unused `det_filter.filter_raw_detections` code and any `DetectionFilter` import in this file.

- [ ] **Step 4: Run tests**

Run:
```bash
python -m pytest tests/test_tracking_optimizer_helpers.py -q
grep -n "DetectionFilter\|filter_raw_detections" src/hydra_suite/core/tracking/optimization/optimizer.py
```
Expected: tests pass; grep prints nothing.

- [ ] **Step 5: Commit**

```bash
make format
git add -A
git commit -m "refactor(optimizer): OBBResult-only cache filtering via apply_detection_filter"
```

---

### Task 4: Replace `DetectionCacheBuilderWorker` with an InferenceRunner batch pass

`DetectionCacheBuilderWorker` (optimizer_workers.py:330) builds a YOLO detection cache in the **legacy `DetectionCache`** format via `YOLOOBBDetector.detect_objects_batched(..., return_raw=True)`. Replace it with a worker that runs `InferenceRunner.run_batch_pass` into an InferenceRunner cache dir, and point the optimizer at that cache. Its only instantiation is `orchestrators/config.py:3402`.

> **Verify-first:** This task depends on the InferenceRunner detection-cache read API (`_open_caches` / `DetectionCacheHandle.read_frame` returning `OBBResult`). Before writing code, run the Step-1 spike to confirm the handle's read method name and that the optimizer's `cache.get_frame` can be satisfied by (or adapted to) it. Adjust the concrete calls in Steps 3–4 to the confirmed names.

**Files:**
- Modify: `src/hydra_suite/core/tracking/optimization/optimizer_workers.py` (delete `DetectionCacheBuilderWorker` 330–498; add an InferenceRunner-based builder)
- Modify: `src/hydra_suite/trackerkit/gui/orchestrators/config.py` (`_build_optimizer_detection_cache` 3394–3419)
- Modify: `src/hydra_suite/core/tracking/optimization/optimizer.py` (cache open/read path)
- Test: `tests/test_optimizer_cache_builder.py`

**Interfaces:**
- Consumes: `build_inference_config_from_params(params)`, `InferenceRunner(cfg, cache_dir=..., video_path=...).run_batch_pass(video_path, progress_cb=, start_frame=, end_frame=, should_stop=)`.
- Produces: `DetectionCacheBuildWorker` (new name) emitting `progress_signal(int, str)` / `finished_signal(bool, str)` where the str is the **cache dir**.

- [ ] **Step 1: Spike — confirm the cache read API**

Run:
```bash
grep -n "class DetectionCacheHandle\|def read_frame\|def write_frame\|def _open_caches\|class _CacheSet" src/hydra_suite/core/inference/*.py src/hydra_suite/core/inference/cache/*.py
grep -n "get_frame\|read_frame\|_open_caches\|self.cache =" src/hydra_suite/core/tracking/optimization/optimizer.py
```
Record: the read method name (`read_frame`), and how `optimizer.py` opens `self.cache`. Confirm `read_frame(f_idx)` returns an `OBBResult`. Use these names verbatim in Steps 3–4.

- [ ] **Step 2: Write the failing test**

```python
# tests/test_optimizer_cache_builder.py
"""The optimizer cache builder must drive InferenceRunner.run_batch_pass, not
a legacy YOLOOBBDetector."""
import types

import hydra_suite.core.tracking.optimization.optimizer_workers as ow


def test_cache_build_worker_uses_run_batch_pass(monkeypatch, tmp_path):
    calls = {}

    class _FakeRunner:
        def __init__(self, cfg, cache_dir=None, video_path=None, cache_only=False):
            calls["cache_dir"] = cache_dir

        def run_batch_pass(self, video_path, progress_cb=None, start_frame=0, end_frame=None, should_stop=None):
            calls["ran"] = True

        def close(self):
            calls["closed"] = True

    monkeypatch.setattr(ow, "InferenceRunner", _FakeRunner, raising=False)
    monkeypatch.setattr(
        ow, "build_inference_config_from_params", lambda p: object(), raising=False
    )

    # Fail if a legacy detector is constructed.
    def _boom(*a, **k):
        raise AssertionError("cache builder must not construct YOLOOBBDetector")

    monkeypatch.setattr("hydra_suite.core.detectors.YOLOOBBDetector", _boom, raising=False)

    emitted = []
    worker = ow.DetectionCacheBuildWorker(
        video_path="v.mp4", cache_dir=str(tmp_path), params={}, start_frame=0, end_frame=1
    )
    worker.finished_signal = types.SimpleNamespace(emit=lambda *a: emitted.append(a))
    worker.progress_signal = types.SimpleNamespace(emit=lambda *a: None)
    worker.run()
    assert calls.get("ran") is True
    assert emitted and emitted[-1][0] is True
```

- [ ] **Step 3: Run test to verify it fails**

Run: `python -m pytest tests/test_optimizer_cache_builder.py -q`
Expected: FAIL — `DetectionCacheBuildWorker` does not exist yet.

- [ ] **Step 4: Replace the worker**

Delete `DetectionCacheBuilderWorker` (330–498) and its `DetectionCache` import. Add module-level imports `from hydra_suite.core.inference.config import build_inference_config_from_params` and `from hydra_suite.core.inference.runner import InferenceRunner`, and the new worker:

```python
class DetectionCacheBuildWorker(QThread):
    """Phase-1-only worker: runs InferenceRunner.run_batch_pass over a frame
    range to populate an InferenceRunner detection cache for the Bayesian
    optimizer. No Kalman/CSV/pose stages.
    """

    progress_signal = Signal(int, str)
    finished_signal = Signal(bool, str)  # (success, cache_dir)

    def __init__(self, video_path, cache_dir, params, start_frame, end_frame, parent=None):
        super().__init__(parent)
        self.video_path = video_path
        self.cache_dir = cache_dir
        self.params = params.copy()
        self.start_frame = start_frame
        self.end_frame = end_frame
        self._stop_requested = False

    def stop(self):
        self._stop_requested = True

    def run(self):
        from pathlib import Path

        try:
            cfg = build_inference_config_from_params(self.params)
            runner = InferenceRunner(
                cfg, cache_dir=Path(self.cache_dir), video_path=self.video_path
            )
        except Exception as e:
            logger.error("DetectionCacheBuild: could not build runner: %s", e)
            self.finished_signal.emit(False, "")
            return
        try:
            runner.run_batch_pass(
                Path(self.video_path),
                progress_cb=lambda pct, msg="": self.progress_signal.emit(int(pct), msg),
                start_frame=self.start_frame,
                end_frame=self.end_frame,
                should_stop=lambda: self._stop_requested,
            )
            if self._stop_requested:
                self.finished_signal.emit(False, "")
                return
            self.finished_signal.emit(True, str(self.cache_dir))
        except Exception:
            logger.exception("DetectionCacheBuild error")
            self.finished_signal.emit(False, "")
        finally:
            runner.close()
```

(Adjust the `progress_cb` signature to whatever `run_batch_pass` passes — confirm from the Step-1 spike / runner.py.)

- [ ] **Step 5: Update the GUI driver and the optimizer cache read**

- In `orchestrators/config.py:_build_optimizer_detection_cache` (3394–3419): construct `DetectionCacheBuildWorker(video_path, cache_dir, params, start, end)` with a **cache dir** (not a cache file path); update the `finished_signal` handler `_on_optimizer_cache_built` to receive a dir.
- In `optimizer.py`: open the InferenceRunner detection cache from that dir (per the Step-1 spike) and make `_filter_cached_detections`'s `cache.get_frame(f_idx)` resolve to the handle's `read_frame` returning `OBBResult` (add a thin adapter if the method name differs).

- [ ] **Step 6: Run tests**

Run:
```bash
python -m pytest tests/test_optimizer_cache_builder.py tests/ -m "not benchmark" -k "optimizer" -q
grep -rn "DetectionCacheBuilderWorker\|YOLOOBBDetector" src/hydra_suite/core/tracking/optimization/
```
Expected: tests pass; grep prints nothing (the old worker + detector are gone).

- [ ] **Step 7: Commit**

```bash
make format
git add -A
git commit -m "refactor(optimizer): build detection cache via InferenceRunner.run_batch_pass"
```

---

### Task 5: Migrate the `preview_worker.py` YOLO branch to `run_realtime`

The YOLO branch (`_preview_run_yolo_branch` 1750–1938) is a ~1000-line clone of `run_realtime`, plus a fake `RuntimeContext` (`_preview_runtime_context_for` 574), three `_advanced_config_value` reach-ins, two `DetectionFilter` builds, and a documented divergence (1062). Replace the DETECTION functions with a single `InferenceRunner.run_realtime` call over an OBB+overlays config built from the preview context; keep the DRAWING functions and the bg-sub branch.

**Files:**
- Modify: `src/hydra_suite/trackerkit/gui/workers/preview_worker.py`
- Test: `tests/test_trackerkit_preview_worker.py` (existing negative-assertion tests must still pass; add a run_realtime-usage assertion)

**Interfaces:**
- Consumes: `build_inference_config_from_params(context_params) -> InferenceConfig`; `InferenceRunner(cfg).run_realtime(frame, frame_idx=0, roi_mask=None) -> FrameResult`. `FrameResult` exposes the filtered OBB + pose/cnn/apriltag/headtail outputs the drawing code needs.
- Produces: `_preview_run_yolo_branch(...)` returning the same `(detected_dimensions, test_frame)` it does today; the worker still emits `{"test_frame_rgb", "resize_factor", "detected_dimensions"}`.

- [ ] **Step 1: Write/extend the failing test**

Keep `test_preview_run_yolo_branch_uses_load_obb_executor_not_legacy_detector` (30) and its sequential twin (401) — they already assert no `YOLOOBBDetector`. Add:

```python
def test_preview_yolo_branch_drives_inference_runner(monkeypatch):
    import hydra_suite.trackerkit.gui.workers.preview_worker as pw

    built = {}

    class _FakeFrameResult:
        # minimal fields the drawing code reads; fill per the real FrameResult
        obb = None
        pose = None
        cnn = []
        apriltag = []
        headtail = None

    class _FakeRunner:
        def __init__(self, cfg, **kw):
            built["cfg"] = cfg

        def run_realtime(self, frame, frame_idx=0, roi_mask=None):
            built["ran"] = True
            return _FakeFrameResult()

        def close(self):
            built["closed"] = True

    monkeypatch.setattr(pw, "InferenceRunner", _FakeRunner, raising=False)
    # ... invoke _preview_run_yolo_branch with a minimal context, assert built["ran"]
```

(Fill `_FakeFrameResult` fields against the real `FrameResult` once the drawing code's reads are known.)

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_trackerkit_preview_worker.py -q`
Expected: the new test FAILS (branch still uses executors directly); the two negative-assertion tests still PASS.

- [ ] **Step 3: Rewrite `_preview_run_yolo_branch` to use the runner**

Replace the body of `_preview_run_yolo_branch` (1750–1938) with:
1. `_preview_resize_frame` (keep) → resized frame.
2. Build params from the preview context via `_preview_build_yolo_params` (keep) and pass them to `build_inference_config_from_params` (add import). This replaces `_preview_runtime_context_for`, `_preview_load_obb_executors`, `_preview_load_headtail_model`, and the `DetectionFilter` build.
3. `runner = InferenceRunner(cfg); fr = runner.run_realtime(frame, roi_mask=roi_mask)`.
4. Feed `fr` into the DRAWING functions (kept): `_preview_yolo_sequential_stage1_viz`, `_preview_draw_obb_annotations`, `_preview_draw_yolo_footer`, and the drawing tails of pose/cnn/apriltag overlays. Extract `detected_dimensions` from `fr`'s filtered OBB `shapes`.
5. `runner.close()`.

Delete the now-unused DETECTION functions: `_preview_runtime_context_for` (574), `_preview_load_obb_executors` (637), `_preview_select_headtail_candidate_indices` (676), `_preview_load_headtail_model` (757), `_preview_run_headtail` (797), `_preview_run_direct_raw_detection` (887), `_PreviewSeqCropSpec` (923), `_preview_accumulate_crop_detections` (932), `_preview_sort_merged_detections` (990), `_preview_run_sequential_raw_detection` (1015), `_preview_run_yolo_raw_detection` (1141), `_preview_compute_canonical_crops` (1278), and the detection halves of the pose/cnn/apriltag overlay functions (keep their drawing tails). This removes all three `_advanced_config_value` imports (693/764/834) and both `DetectionFilter` builds (1163/1776).

Leave the bg-sub branch (`_preview_run_bg_subtraction` 391–486) and all DRAW functions untouched.

- [ ] **Step 4: Run tests**

Run:
```bash
python -m pytest tests/test_trackerkit_preview_worker.py -q
grep -n "_advanced_config_value\|DetectionFilter\|_preview_runtime_context_for" src/hydra_suite/trackerkit/gui/workers/preview_worker.py
```
Expected: all preview tests pass (incl. the negative-assertion tests); grep prints nothing.

- [ ] **Step 5: Commit**

```bash
make format
git add -A
git commit -m "refactor(trackerkit): preview YOLO branch runs via InferenceRunner.run_realtime"
```

---

### Task 6: Migrate `model_test_dialog.py` off private `stages.obb` helpers

`_TestWorker.execute` (161–243) loads executors and calls private `stages.obb` helpers (`_extract_obb_result`, `_build_crops`, `_merge_obb_results`, `_resize_crops_for_stage2`) at 248/292. Replace the per-image detection with `InferenceRunner.detect_batch` (single frame) over an OBB-only config, and draw from the returned `OBBResult`.

**Files:**
- Modify: `src/hydra_suite/trackerkit/gui/dialogs/model_test_dialog.py`
- Test: `tests/test_model_test_dialog.py` (existing negative-assertion tests must still pass)

**Interfaces:**
- Consumes: `build_obb_only_config(model_path, ...)`, `InferenceRunner.detect_batch([frame]) -> [OBBResult]`.
- Produces: `_TestWorker.execute` emits the same `image_ready(np.ndarray)` / `finished_all()` signals.

- [ ] **Step 1: Extend the test**

Keep `test_test_worker_execute_uses_load_obb_executor_not_legacy_detector` (197) — it asserts no `YOLOOBBDetector`. Since the migration moves from `load_obb_executor` to `InferenceRunner`, update the test's expectation: fake `InferenceRunner` and assert `detect_batch` is called; keep the `_fail_if_called` YOLOOBBDetector guard.

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_model_test_dialog.py -q`
Expected: the updated test FAILS (execute still uses `load_obb_executor` + private helpers).

- [ ] **Step 3: Rewrite `_TestWorker.execute`**

Replace the executor-loading + private-helper flow with: build `cfg = build_obb_only_config(model_path, compute_runtime=..., mode="sequential" if detect_model_path else "direct")`; construct one `InferenceRunner(cfg)`; per sample image `cv2.imread` → `obb = runner.detect_batch([frame])[0]`; draw `obb.corners` quads with `cv2.polylines` (keep the existing drawing at 231–239); emit `image_ready`. Delete `_run_direct` (246), `_run_detect_only` (262), `_run_sequential` (283), `_SeqCropSpec` (147), and the private `stages.obb` imports (248/292). Close the runner in a `finally`.

- [ ] **Step 4: Run tests**

Run:
```bash
python -m pytest tests/test_model_test_dialog.py -q
grep -n "stages.obb import\|_extract_obb_result\|_build_crops\|load_obb_executor" src/hydra_suite/trackerkit/gui/dialogs/model_test_dialog.py
```
Expected: tests pass; grep prints nothing.

- [ ] **Step 5: Commit**

```bash
make format
git add -A
git commit -m "refactor(trackerkit): model-test dialog runs via InferenceRunner.detect_batch"
```

---

### Task 7: Migrate `detectkit/gui/prediction_preview.py` off raw ultralytics

`_get_torch_model` (37) uses raw `from ultralytics import YOLO`; `_detections_from_result` (54) parses `result.obb.xyxyxyxy/conf/cls`. Replace with `load_obb_executor` (all runtimes) + `detect_batch`/`OBBResult`, so the detectkit preview gains ONNX/TRT/CoreML for free.

**Files:**
- Modify: `src/hydra_suite/detectkit/gui/prediction_preview.py`
- Test: `tests/test_detectkit_prediction_preview.py`

**Interfaces:**
- Consumes: `build_obb_only_config`, `InferenceRunner.detect_batch`.
- Produces: `predict_preview_detections(...) -> list[dict]` (unchanged dict shape `{class_id, polygon_px, confidence}`), `predict_preview_detections_for_image(...)`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_detectkit_prediction_preview.py
import numpy as np

import hydra_suite.detectkit.gui.prediction_preview as pp
from hydra_suite.core.inference.result import OBBResult


def test_prediction_preview_emits_polygon_dicts(monkeypatch):
    obb = OBBResult(
        frame_idx=0,
        centroids=np.array([[5.0, 5.0]], np.float32),
        angles=np.zeros(1, np.float32),
        sizes=np.ones(1, np.float32),
        shapes=np.array([[10.0, 2.0]], np.float32),
        confidences=np.array([0.8], np.float32),
        corners=np.array([[[0, 0], [4, 0], [4, 4], [0, 4]]], np.float32),
        detection_ids=OBBResult.make_detection_ids(0, 1),
    )

    class _FakeRunner:
        def __init__(self, *a, **k):
            pass

        def detect_batch(self, frames, frame_indices=None):
            return [obb]

        def close(self):
            pass

    monkeypatch.setattr(pp, "InferenceRunner", _FakeRunner, raising=False)
    dets = pp.predict_preview_detections_for_image(
        np.zeros((16, 16, 3), np.uint8), model_path="m.pt", device="cpu"
    )
    assert dets and set(dets[0]) >= {"class_id", "polygon_px", "confidence"}
    assert len(dets[0]["polygon_px"]) == 4
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_detectkit_prediction_preview.py -q`
Expected: FAIL (current code uses raw ultralytics; signature/behavior differs).

- [ ] **Step 3: Rewrite the detection path**

Replace `_get_torch_model` + `_detections_from_result` usage with a runner-based flow: build `cfg = build_obb_only_config(model_path, compute_runtime=<from device>)`, `runner = InferenceRunner(cfg)`, `obb = runner.detect_batch([frame])[0]`, and convert each detection to `{class_id, polygon_px: [(x, y) for the 4 corners], confidence}` from `obb.corners`/`obb.confidences` (class ids default to 0 unless carried on the result). Update `predict_preview_detections` (file-path variant) and `predict_preview_detections_for_image` (in-memory variant) accordingly. Keep `_resolve_torch_device` for mapping the device preference to a compute-runtime string.

- [ ] **Step 4: Run tests**

Run:
```bash
python -m pytest tests/test_detectkit_prediction_preview.py -q
grep -n "from ultralytics import YOLO\|result.obb" src/hydra_suite/detectkit/gui/prediction_preview.py
```
Expected: test passes; grep prints nothing.

- [ ] **Step 5: Commit**

```bash
make format
git add -A
git commit -m "refactor(detectkit): prediction preview via InferenceRunner (all runtimes)"
```

---

## Final verification (whole plan)

- [ ] **Step 1: Full suite**

Run: `python -m pytest tests/ -m "not benchmark" -q`
Expected: PASS.

- [ ] **Step 2: Confirm no production `core/detectors` imports remain (except tools/benchmark, handled by Plan 3)**

Run:
```bash
grep -rn "core.detectors\|from ..core.detectors\|YOLOOBBDetector\|DetectionFilter\|_advanced_config_value" \
  src/hydra_suite/data src/hydra_suite/trackerkit src/hydra_suite/detectkit src/hydra_suite/posekit \
  src/hydra_suite/core/tracking/optimization
```
Expected: no matches (comments/docstrings referencing the legacy names are acceptable; imports/constructions are not).

- [ ] **Step 3: Format gate** — `make format-check`.

---

## Self-Review notes

- **Spec coverage (Phase C):** Task 1 = dataset_generation; Task 2 = pose ×4 shim; Task 3 = optimizer filter; Task 4 = DetectionCacheBuilder deletion; Task 5 = preview_worker; Task 6 = model_test_dialog; Task 7 = detectkit. All Phase-C bullets covered.
- **Known verify-first:** Task 4's cache read API (`read_frame`/`_open_caches`) is confirmed at execution via its Step 1 spike; Task 5's `FrameResult` field reads are filled against the real dataclass at execution. These are the two spots where the exact field/method names must be confirmed in-repo before finalizing.
- **Deliberately deferred:** `PosePipeline` (pose_pipeline.py:408) is NOT migrated here — it appears dead (only its own docstring references it) and is verified-and-deleted in Plan 4's dead-code sweep. `core/detectors` files remain on disk (Plan 4 deletes them). `tools/benchmark_models.py`'s `YOLOOBBDetector` uses are Plan 3.
- **Type consistency:** `detect_batch(frames, frame_indices=None) -> list[OBBResult]`, `build_obb_only_config(...) -> InferenceConfig`, `run_realtime(frame, frame_idx=0, roi_mask=None) -> FrameResult`, `apply_detection_filter(raw, config) -> OBBResult`, `load_pose_backend(...) -> backend` used consistently across tasks.
