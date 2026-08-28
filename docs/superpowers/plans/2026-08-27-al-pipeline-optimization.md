# AL Pipeline Optimization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make active-learning frame selection and dataset export fast on long
videos in both DetectKit and TrackerKit, by eliminating redundant/uncached
inference, O(n²) hot loops, and per-frame video-decode overhead.

**Architecture:** Fix TrackerKit's export path to actually use the detection
cache tracking already builds (it currently never does). Extract a raw
(pre-NMS-filter) batched detection path on `InferenceRunner` and a small
shared cache-reuse helper, used by both kits. Restructure DetectKit's AL worker
from a strictly-sequential per-frame loop into three phases — sequential decode
with an inline motion prefilter and windowed dedup, one batched cached
detection pass, then a cheap re-filter pass for NMS-instability scoring — so it
gets real GPU batching and cache reuse instead of 4 uncached forward passes per
candidate frame.

**Tech Stack:** Python, PyTorch/Ultralytics YOLO via `core/inference`, OpenCV
(`cv2.VideoCapture`), pandas, pytest.

**Spec:** `docs/superpowers/specs/2026-08-27-al-pipeline-optimization-design.md`

## Global Constraints

- Do implementation work in a git worktree branched from local HEAD
  (`git worktree add .worktrees/<name> -b <branch> HEAD`), never from origin —
  per this repo's `CLAUDE.md`.
- Commit as the configured git user with plain commit messages — no
  `Co-Authored-By: Claude` trailer.
- Run `make format` (black + isort) before each commit that touches `src/`.
- Dependency direction: `core/inference/` must never import from `detectkit/`,
  `trackerkit/`, or any other app layer. `data/al/` must stay Qt-free.
- None of this plan's changes touch the Kalman filter, assignment, or CSV
  tracking output — no MPS/CUDA byte-identical tracking-equivalence run is
  required for this plan. Fixture-based tests (this plan's own) are the bar.
- Never copy-paste the cache-reuse logic between TrackerKit and DetectKit call
  sites — both route through the one shared helper built in Task 3.

---

### Task 1: TrackerKit — fix the O(n²) frame lookup in AL selection scoring

**Files:**
- Modify: `src/hydra_suite/core/post/dataset_export.py:123-343` (function
  `generate_active_learning_dataset`)
- Test: locate the existing test file first (`ls tests | grep -i dataset_export`
  or `grep -rl generate_active_learning_dataset tests/`); add to it, or create
  `tests/test_core_post_dataset_export.py` if none exists.

**Interfaces:**
- No new public interface — internal-only change to a loop body.

- [ ] **Step 1: Read the current function to confirm the exact loop shape**

Read `src/hydra_suite/core/post/dataset_export.py` lines 123-215 to confirm the
current code around:
```python
for idx, frame_id in enumerate(unique_frames):
    ...
    frame_data = df[df["FrameID"] == frame_id]
```
and note the columns of `df` and what `frame_data` is used for afterward (it
must remain a DataFrame with the same columns/behavior for downstream code).

- [ ] **Step 2: Write a test proving current behavior (characterization test)**

If no existing test covers this function end-to-end, add one that builds a
small synthetic tracking CSV (a `pandas.DataFrame` with `FrameID` plus whatever
other columns the function reads — confirmed in Step 1) covering at least 3
distinct `FrameID` values with multiple rows each, calls
`generate_active_learning_dataset(...)` (or whatever internal function actually
does the per-frame loop, if it's been factored out — confirm from Step 1's
read), and asserts the per-frame row counts/content match expectations. Save
this as `test_frame_lookup_matches_per_frame_rows` in the test file located in
Step's file discovery.

```python
def test_frame_lookup_matches_per_frame_rows(tmp_path):
    df = pd.DataFrame({
        "FrameID": [1, 1, 2, 3, 3, 3],
        "X": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0],
        "Y": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
    })
    grouped = {int(fid): sub for fid, sub in df.groupby("FrameID")}
    assert len(grouped[1]) == 2
    assert len(grouped[2]) == 1
    assert len(grouped[3]) == 3
    assert list(grouped[3]["X"]) == [40.0, 50.0, 60.0]
```

This pins the replacement's expected semantics before touching the real
function — it does not yet call `generate_active_learning_dataset` because that
function needs a real detection cache and video to run end-to-end; the full
integration test for this path is Task 4's responsibility. This step is a pure
characterization of the `groupby` replacement's correctness.

- [ ] **Step 2b: Run it**

Run: `python -m pytest <test file> -k test_frame_lookup_matches_per_frame_rows -v`
Expected: PASS (this test doesn't touch the real function yet, just proves the
groupby semantics match manual per-frame filtering).

- [ ] **Step 3: Apply the fix**

In `generate_active_learning_dataset`, before the `for idx, frame_id in
enumerate(unique_frames):` loop, add:
```python
frames_by_id = {int(fid): sub for fid, sub in df.groupby("FrameID")}
```
(mirroring the exact same pattern already used elsewhere in this codebase at
`src/hydra_suite/data/dataset_generation.py:781`). Replace the loop body's
`frame_data = df[df["FrameID"] == frame_id]` with
`frame_data = frames_by_id.get(int(frame_id))` and add a `if frame_data is
None: continue` (or existing equivalent empty-frame handling — check what the
original line did when no rows matched, and preserve that behavior exactly).

- [ ] **Step 4: Run the full existing test suite for this file**

Run: `python -m pytest <test file> -v`
Expected: all PASS, including any pre-existing tests for
`generate_active_learning_dataset`.

- [ ] **Step 5: Commit**

```bash
git add src/hydra_suite/core/post/dataset_export.py <test file>
git commit -m "perf(trackerkit): replace per-frame O(n^2) DataFrame scan with groupby in AL scoring"
```

---

### Task 2: `InferenceRunner` — extract `detect_batch_raw`

**Files:**
- Modify: `src/hydra_suite/core/inference/runner.py` (`detect_batch`, line 1157)
- Test: `tests/test_inference_runner_batch.py` (confirmed to exist, already
  covers `detect_batch`-style usage)

**Interfaces:**
- Produces: `InferenceRunner.detect_batch_raw(self, frames: list[np.ndarray],
  frame_indices: list[int] | None = None, roi_mask=None) -> list[OBBResult]` —
  returns the raw (pre-`filter_for_source`) `OBBResult` per frame, one call to
  `run_obb(...)` for the whole batch. `OBBResult` from
  `src/hydra_suite/core/inference/result.py:20-53`.
- `detect_batch` keeps its existing public signature and return type
  (`list[OBBResult]`, filtered) — becomes a thin wrapper over
  `detect_batch_raw` + the existing per-frame `filter_for_source` call.

- [ ] **Step 1: Read the current `detect_batch` body in full**

Read `src/hydra_suite/core/inference/runner.py` lines 1150-1200 to get the
exact current body (the call to `run_obb(frames, self._models.obb,
self.config.obb, self.runtime)`, the per-frame `filter_for_source(self.config,
raw_obb, roi_mask)` call, and how the raw `OBBResult` for each frame is
currently constructed from `run_obb`'s return value) so the extraction is
behavior-preserving to the line.

- [ ] **Step 2: Write the failing test for `detect_batch_raw`**

Add to `tests/test_inference_runner_batch.py`, following its existing fixture
conventions (synthetic frames, a real small OBB model or whatever fixture the
existing tests in that file already use — check the top of the file for the
`InferenceConfig`/`OBBConfig` fixture pattern before writing this):

```python
def test_detect_batch_raw_returns_unfiltered_results(runner_with_obb_model, synthetic_frames):
    raw_results = runner_with_obb_model.detect_batch_raw(synthetic_frames, frame_indices=[0, 1])
    assert len(raw_results) == 2
    assert all(isinstance(r, OBBResult) for r in raw_results)
    # Raw results are not filtered: confidence floor is far below the configured
    # confidence_threshold, so raw results should have >= detections than filtered.
    filtered_results = runner_with_obb_model.detect_batch(synthetic_frames, frame_indices=[0, 1])
    for raw, filtered in zip(raw_results, filtered_results):
        assert raw.num_detections >= filtered.num_detections
```

(Reuse whatever fixture names `test_inference_runner_batch.py` already defines
for a runner backed by a real/synthetic OBB model — read the file first and
substitute the actual fixture names in place of `runner_with_obb_model`/
`synthetic_frames` above.)

- [ ] **Step 3: Run it to confirm it fails**

Run: `python -m pytest tests/test_inference_runner_batch.py -k test_detect_batch_raw_returns_unfiltered_results -v`
Expected: FAIL with `AttributeError: 'InferenceRunner' object has no attribute 'detect_batch_raw'`

- [ ] **Step 4: Implement `detect_batch_raw`, refactor `detect_batch` to use it**

In `runner.py`, extract the raw batched call and per-frame raw `OBBResult`
construction (the part of `detect_batch`'s current body that runs before
`filter_for_source`) into:

```python
def detect_batch_raw(self, frames, frame_indices=None, roi_mask=None):
    """Run OBB detection over a batch of frames, returning UNFILTERED raw
    results. No cache is read or written."""
    if self._models.obb is None:
        raise RuntimeError("detect_batch_raw requires an OBB model")
    # <the exact raw-construction body confirmed in Step 1, moved here unchanged>
    return raw_results
```

Then rewrite `detect_batch` to:
```python
def detect_batch(self, frames, frame_indices=None, roi_mask=None):
    """Run OBB detection over a list of frames, returning filtered results in
    memory. No cache is read or written."""
    raw_results = self.detect_batch_raw(frames, frame_indices, roi_mask)
    filtered = []
    for raw_obb in raw_results:
        filtered_obb, _ = filter_for_source(self.config, raw_obb, roi_mask)
        filtered.append(filtered_obb)
    return filtered
```

Keep both docstrings' "No cache is read or written" line — still true for both.

- [ ] **Step 5: Run the new test**

Run: `python -m pytest tests/test_inference_runner_batch.py -k test_detect_batch_raw_returns_unfiltered_results -v`
Expected: PASS

- [ ] **Step 6: Run the full existing file to confirm no regression**

Run: `python -m pytest tests/test_inference_runner_batch.py -v`
Expected: all PASS, including every pre-existing `detect_batch` test — this
proves the refactor didn't change `detect_batch`'s observable behavior.

- [ ] **Step 7: Commit**

```bash
git add src/hydra_suite/core/inference/runner.py tests/test_inference_runner_batch.py
git commit -m "refactor(inference): extract detect_batch_raw from detect_batch"
```

---

### Task 3: Shared `get_or_compute_raw` cache-reuse helper

**Files:**
- Create: `src/hydra_suite/core/inference/cache/reuse.py`
- Test: `tests/test_inference_cache_reuse.py` (new)

**Interfaces:**
- Consumes: `InferenceRunner.detect_batch_raw` (Task 2); `DetectionCacheHandle`
  and `open_detection_cache_reader` from
  `src/hydra_suite/core/inference/cache/store.py`; `detection_cache_key`/
  `with_video_signature` from `src/hydra_suite/core/inference/cache/keys.py:63,53`.
- Produces: `get_or_compute_raw(runner: InferenceRunner, cache_dir: Path,
  frames: list[np.ndarray], frame_indices: list[int]) -> dict[int, OBBResult]`
  — used by Task 4 (TrackerKit export) and Task 9 (DetectKit AL worker).

- [ ] **Step 1: Confirm the exact cache-handle construction pattern**

Before writing any code, read:
- `src/hydra_suite/core/inference/cache/store.py` in full — the
  `DetectionCacheHandle` dataclass's `__init__`/field list, `is_valid()`,
  `read_frame()`, `write_frame()`, `close()`, and the `open_detection_cache_reader`
  factory function's exact signature and body.
- `src/hydra_suite/core/inference/cache/keys.py` — `detection_cache_key(config:
  OBBConfig, roi_mask=None) -> CacheKey` and `with_video_signature(key, sig) ->
  CacheKey`, confirming exact parameter order and the `CacheKey` type.
- `src/hydra_suite/core/inference/runner.py`'s `InferenceRunner.__init__` and
  wherever it computes and stores a video signature (grep `_video_sig` in
  `runner.py`) — confirm the exact attribute name and how it's derived from
  `video_path`, since a writable cache handle needs this to build a valid key.
- `tests/test_inference_runner_batch.py`'s existing usage of `_open_caches(cfg,
  cache_dir, runner._video_sig)` (or whatever the actual current call looks
  like) to see the working pattern for constructing a cache handle for writing
  outside of `InferenceRunner`'s own internals.

Use whatever the real, confirmed construction pattern is in Step 2 below — do
not guess parameter names not confirmed by this read.

- [ ] **Step 2: Write the failing tests**

Create `tests/test_inference_cache_reuse.py`:

```python
import numpy as np
import pytest

from hydra_suite.core.inference.cache.reuse import get_or_compute_raw


class _FakeRunner:
    def __init__(self):
        self.calls = []

    def detect_batch_raw(self, frames, frame_indices=None, roi_mask=None):
        self.calls.append(list(frame_indices))
        return [_make_raw_result(idx) for idx in frame_indices]


def test_get_or_compute_raw_computes_on_empty_cache(tmp_path):
    runner = _FakeRunner()
    frames = [np.zeros((4, 4, 3), dtype=np.uint8) for _ in range(3)]
    result = get_or_compute_raw(runner, tmp_path, frames, [0, 1, 2])
    assert set(result.keys()) == {0, 1, 2}
    assert runner.calls == [[0, 1, 2]]


def test_get_or_compute_raw_reads_fully_covered_cache_without_recompute(tmp_path):
    runner = _FakeRunner()
    frames = [np.zeros((4, 4, 3), dtype=np.uint8) for _ in range(2)]
    get_or_compute_raw(runner, tmp_path, frames, [0, 1])  # populates cache
    runner.calls.clear()
    result = get_or_compute_raw(runner, tmp_path, frames, [0, 1])
    assert runner.calls == []  # no new compute — fully covered by existing cache
    assert set(result.keys()) == {0, 1}


def test_get_or_compute_raw_recomputes_whole_set_on_partial_miss(tmp_path):
    runner = _FakeRunner()
    frames2 = [np.zeros((4, 4, 3), dtype=np.uint8) for _ in range(2)]
    get_or_compute_raw(runner, tmp_path, frames2, [0, 1])  # populates cache for [0, 1]
    runner.calls.clear()
    frames3 = [np.zeros((4, 4, 3), dtype=np.uint8) for _ in range(3)]
    get_or_compute_raw(runner, tmp_path, frames3, [0, 1, 2])  # 2 is missing
    # Per the no-merge convention: the whole *requested* set is recomputed fresh,
    # not just the miss.
    assert runner.calls == [[0, 1, 2]]
```

(`_make_raw_result(idx)` is a small local helper building a minimal valid
`OBBResult` for the given `frame_idx` with e.g. 1 detection — write it using
`OBBResult`'s confirmed fields from `result.py:20-53`, e.g.
`OBBResult(frame_idx=idx, centroids=np.zeros((1,2)), angles=np.zeros(1),
sizes=np.zeros(1), shapes=np.zeros((1,2)), confidences=np.array([0.9]),
corners=np.zeros((1,4,2)), detection_ids=OBBResult.make_detection_ids(idx, 1))`
— confirm exact required fields from Step 1's read of `result.py` and adjust.)

- [ ] **Step 3: Run tests to confirm they fail**

Run: `python -m pytest tests/test_inference_cache_reuse.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'hydra_suite.core.inference.cache.reuse'`

- [ ] **Step 4: Implement `get_or_compute_raw`**

Create `src/hydra_suite/core/inference/cache/reuse.py` using the exact
construction pattern confirmed in Step 1. Shape (fill in the real
`DetectionCacheHandle`/key-construction calls from Step 1 in place of the
sketch below — do not leave the sketch's placeholders in the final file):

```python
"""Shared cache-reuse helper: read a fully-covered on-disk detection cache, or
recompute the whole requested frame set fresh in one batched pass and persist
it as a new complete cache session. No incremental merge — matches this
codebase's existing all-or-nothing cache convention (see
core/tracking/worker.py's backward-pass cache handling)."""

from pathlib import Path

from hydra_suite.core.inference.cache.store import (
    DetectionCacheHandle,
    open_detection_cache_reader,
)
from hydra_suite.core.inference.cache.keys import (
    detection_cache_key,
    with_video_signature,
)


def get_or_compute_raw(runner, cache_dir: Path, frames, frame_indices) -> dict:
    cache_path = Path(cache_dir) / "detection.npz"
    if cache_path.exists():
        reader = open_detection_cache_reader(cache_path)
        if reader.is_valid() and all(
            reader.read_frame(i) is not None for i in frame_indices
        ):
            return {i: reader.read_frame(i) for i in frame_indices}

    raw_results = runner.detect_batch_raw(frames, frame_indices=list(frame_indices))
    key = with_video_signature(
        detection_cache_key(runner.config.obb), getattr(runner, "_video_sig", None)
    )
    handle = DetectionCacheHandle(path=cache_path, key=key, read_only=False)
    for idx, result in zip(frame_indices, raw_results):
        handle.write_frame(idx, result=result)
    handle.close()
    return dict(zip(frame_indices, raw_results))
```

Adjust every call in this body to match the exact signatures confirmed in
Step 1 (parameter names/order for `DetectionCacheHandle(...)`,
`detection_cache_key(...)`, `with_video_signature(...)`, and how
`cache_path`/the on-disk filename is actually named elsewhere in this codebase
— confirm against `build_detection_cache_path` in
`src/hydra_suite/utils/video_artifacts.py` rather than hardcoding
`"detection.npz"` if that function is the established way to derive this path).

- [ ] **Step 5: Run tests to confirm they pass**

Run: `python -m pytest tests/test_inference_cache_reuse.py -v`
Expected: all 3 PASS

- [ ] **Step 6: Commit**

```bash
git add src/hydra_suite/core/inference/cache/reuse.py tests/test_inference_cache_reuse.py
git commit -m "feat(inference): add shared get_or_compute_raw cache-reuse helper"
```

---

### Task 4: TrackerKit export — use the existing detection cache

**Files:**
- Modify: `src/hydra_suite/data/dataset_generation.py`
  (`_init_detection_runner` line 322, `_detect_records_for_frames` line 562,
  `export_dataset` line 712)
- Modify: `docs/superpowers/specs/done/2026-08-17-al-escalated-multi-format-export-design.md`
  (correct finding #15)
- Test: locate via `ls tests | grep -i dataset_generation` or `grep -rl
  _init_detection_runner tests/`; add to it, or create
  `tests/test_dataset_generation_cache_reuse.py` if none exists.

**Interfaces:**
- Consumes: `get_or_compute_raw` from Task 3
  (`hydra_suite.core.inference.cache.reuse`), `build_inference_cache_dir` from
  `src/hydra_suite/utils/video_artifacts.py`.
- Modifies: `_init_detection_runner(params, video_path)` — adds a required
  `video_path` parameter (was `_init_detection_runner(params)`).

- [ ] **Step 1: Read the current three functions in full**

Read `src/hydra_suite/data/dataset_generation.py` lines 322-416 (`_init_detection_runner`),
562-630 (`_detect_records_for_frames`), and 700-830 (`export_dataset`, to see
exactly how `video_path` is already in scope there and how it currently calls
the other two) to confirm exact current signatures before changing them.

- [ ] **Step 2: Write the failing test for cache-hit behavior**

Add a test that: builds a tiny synthetic video + a pre-populated detection
cache via `get_or_compute_raw` directly (reusing Task 3's test helpers/fixture
pattern) for a known set of frames, then calls `_detect_records_for_frames`
with a `runner` constructed via `_init_detection_runner(params, video_path)`
pointed at that same video, and asserts the underlying fake/mocked
`detect_batch_raw` is never called (patch `InferenceRunner.detect_batch_raw` on
the runner instance with a `unittest.mock.Mock` that raises if called, since a
full cache hit should never reach it):

```python
def test_detect_records_for_frames_uses_existing_cache(tmp_path, synthetic_video, monkeypatch):
    video_path = synthetic_video  # fixture: writes a small real video file
    cache_dir = build_inference_cache_dir(video_path, artifact_base_dir=tmp_path)
    cache_dir.mkdir(parents=True, exist_ok=True)
    params = {...}  # minimal valid params dict for _init_detection_runner, confirmed from Step 1
    runner = _init_detection_runner(params, video_path)
    # Pre-populate the cache for frames [0, 1] via the real path once.
    frames = [_read_frame(video_path, i) for i in (0, 1)]
    get_or_compute_raw(runner, cache_dir, frames, [0, 1])

    def _fail_if_called(*a, **k):
        raise AssertionError("detect_batch_raw should not be called on a full cache hit")
    monkeypatch.setattr(runner, "detect_batch_raw", _fail_if_called)

    records = _detect_records_for_frames(runner, {0, 1}, params, native_level="obb")
    assert len(records) >= 1  # exact assertion shape confirmed from Step 1's read
```

(Fill in `params`'s real minimal required keys and `native_level`'s real
expected values from Step 1's read of `_detect_records_for_frames`'s body and
any existing tests' fixtures for this module.)

- [ ] **Step 3: Run it to confirm it fails**

Run: `python -m pytest <test file> -k test_detect_records_for_frames_uses_existing_cache -v`
Expected: FAIL — either a `TypeError` (extra `video_path` arg not yet accepted)
or the assertion failing because `detect_batch_raw` IS called (cache bypassed).

- [ ] **Step 4: Implement the fix**

In `_init_detection_runner`, add a `video_path` parameter and pass
`cache_dir=build_inference_cache_dir(video_path)` into the `InferenceRunner(...)`
construction on both branches (YOLO-OBB and, where applicable now or once the
separately-flagged bg-sub bug is fixed, background-subtraction). In
`_detect_records_for_frames`, replace the direct `runner.detect_batch(images,
frame_indices=list(valid_chunk))` call with:
```python
from hydra_suite.core.inference.cache.reuse import get_or_compute_raw
...
results_by_idx = get_or_compute_raw(runner, runner.cache_dir, images, list(valid_chunk))
records = [results_by_idx[idx] for idx in valid_chunk]
```
(confirm `runner.cache_dir`'s exact accessor name from Task 3 Step 1's read of
`InferenceRunner.__init__` — it may be `runner._cache_dir` or a public
property; use whichever is real.) Update `export_dataset` to pass `video_path`
through to `_init_detection_runner(params, video_path)`.

- [ ] **Step 5: Run the new test and the full file's existing tests**

Run: `python -m pytest <test file> -v`
Expected: all PASS.

- [ ] **Step 6: Correct the design-doc finding**

In `docs/superpowers/specs/done/2026-08-17-al-escalated-multi-format-export-design.md`,
find finding #15's text (the "double inference is intentional/defensible" note)
and replace it with a short correction: this was fixed in
`docs/superpowers/plans/2026-08-27-al-pipeline-optimization.md` (Task 4) — the
premise (cache lacks low-enough-confidence detections) was false; export now
reuses the existing detection cache. Do not otherwise rewrite the historical
doc's content, per this repo's docs-lifecycle convention.

- [ ] **Step 7: Commit**

```bash
git add src/hydra_suite/data/dataset_generation.py <test file> docs/superpowers/specs/done/2026-08-17-al-escalated-multi-format-export-design.md
git commit -m "perf(trackerkit): reuse existing detection cache for AL export instead of uncached rerun"
```

---

### Task 5: DetectKit — `data/al/inference_adapter.py`

**Files:**
- Create: `src/hydra_suite/data/al/inference_adapter.py`
- Test: `tests/test_al_inference_adapter.py` (new)

**Interfaces:**
- Consumes: `build_obb_only_config` from
  `src/hydra_suite/core/inference/config.py:1107`;
  `detectkit_resolve_inference_models` from
  `src/hydra_suite/detectkit/gui/project.py:660-663`.
- Produces: `build_obb_config_for_al(kind: str, primary_model_path: str,
  secondary_model_path: str | None, *, crop_pad_ratio: float,
  confidence_threshold: float, iou_threshold: float, runtime_tier=None) ->
  InferenceConfig` — used by Task 9.

- [ ] **Step 1: Read the exact real signatures**

Read `src/hydra_suite/core/inference/config.py` around line 1107
(`build_obb_only_config`) for its exact full parameter list/defaults
(confirmed partial signature from research: `model_path, *,
compute_runtime="cpu", runtime_tier=None, confidence_threshold=0.25,
iou_threshold=0.7, max_targets=8, mode="direct", model_task="obb",
emit_native_geometry=False, extra_params=None`). Read
`src/hydra_suite/detectkit/gui/project.py:660-663`
(`detectkit_resolve_inference_models`) for its exact return type/values for
`kind` (`"obb_direct"`, `"sequential"`, `"unknown"`) and what
`primary`/`secondary` model paths mean for each `kind`. Read
`src/hydra_suite/detectkit/gui/main_window.py:1409-1464`
(`_load_active_detector_fn`) to see exactly which extra fields (e.g.
`crop_pad_ratio`) the `sequential` case currently reads and passes to
`predict_obb_for_frame_sequential`, since those need to reach
`build_obb_only_config`'s `extra_params` for the sequential case.

- [ ] **Step 2: Write the failing tests**

```python
from hydra_suite.data.al.inference_adapter import build_obb_config_for_al


def test_build_obb_config_for_al_direct_mode():
    cfg = build_obb_config_for_al(
        "obb_direct", "/path/to/model.pt", None,
        crop_pad_ratio=0.15, confidence_threshold=0.05, iou_threshold=0.5,
    )
    assert cfg.obb.mode == "direct"
    assert cfg.obb.direct.model_path == "/path/to/model.pt"
    assert cfg.obb.confidence_threshold == 0.05
    assert cfg.obb.iou_threshold == 0.5


def test_build_obb_config_for_al_sequential_mode():
    cfg = build_obb_config_for_al(
        "sequential", "/path/to/detect.pt", "/path/to/obb.pt",
        crop_pad_ratio=0.2, confidence_threshold=0.05, iou_threshold=0.5,
    )
    assert cfg.obb.mode == "sequential"
    assert cfg.obb.sequential.detect_model_path == "/path/to/detect.pt"
    assert cfg.obb.sequential.obb_model_path == "/path/to/obb.pt"
    assert cfg.obb.sequential.crop_pad_ratio == 0.2


def test_build_obb_config_for_al_unknown_kind_raises():
    import pytest
    with pytest.raises(ValueError):
        build_obb_config_for_al("unknown", "/path/to/model.pt", None,
                                 crop_pad_ratio=0.15, confidence_threshold=0.05,
                                 iou_threshold=0.5)
```

(Adjust assertions to match `OBBDirectConfig`/`OBBSequentialConfig`'s real
field names confirmed in Step 1 if they differ from the above.)

- [ ] **Step 3: Run to confirm failure**

Run: `python -m pytest tests/test_al_inference_adapter.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 4: Implement**

```python
"""Adapts DetectKit's resolved model info (kind/model paths from
detectkit_resolve_inference_models) into an InferenceConfig for AL scoring,
via the existing build_obb_only_config helper TrackerKit's export path already
uses. Qt-free."""

from hydra_suite.core.inference.config import build_obb_only_config, InferenceConfig


def build_obb_config_for_al(
    kind: str,
    primary_model_path: str,
    secondary_model_path: str | None,
    *,
    crop_pad_ratio: float,
    confidence_threshold: float,
    iou_threshold: float,
    runtime_tier=None,
) -> InferenceConfig:
    if kind == "obb_direct":
        return build_obb_only_config(
            primary_model_path,
            runtime_tier=runtime_tier,
            confidence_threshold=confidence_threshold,
            iou_threshold=iou_threshold,
            mode="direct",
        )
    if kind == "sequential":
        return build_obb_only_config(
            primary_model_path,
            runtime_tier=runtime_tier,
            confidence_threshold=confidence_threshold,
            iou_threshold=iou_threshold,
            mode="sequential",
            extra_params={
                "obb_model_path": secondary_model_path,
                "crop_pad_ratio": crop_pad_ratio,
            },
        )
    raise ValueError(f"Unsupported AL detector kind: {kind!r}")
```

Adjust the `extra_params` keys and the direct-vs-sequential field mapping to
match exactly what `build_obb_only_config`'s real body (read in Step 1) does
with `extra_params` for the sequential case — the sketch above is illustrative;
the implementer must verify against the real function body, not assume these
keys are correct.

- [ ] **Step 5: Run tests to confirm pass**

Run: `python -m pytest tests/test_al_inference_adapter.py -v`
Expected: all PASS

- [ ] **Step 6: Commit**

```bash
git add src/hydra_suite/data/al/inference_adapter.py tests/test_al_inference_adapter.py
git commit -m "feat(detectkit): add AL inference config adapter over build_obb_only_config"
```

---

### Task 6: DetectKit — sequential single-pass video decode

**Files:**
- Modify: `src/hydra_suite/data/al/frame_source.py` (`VideoFrameSource`, lines 32-58)
- Test: `tests/test_al_frame_source.py` (confirmed to exist)

**Interfaces:**
- `VideoFrameSource.__init__(self, video_path, stride=1)` — signature
  unchanged (confirmed already supports `stride`).
- `VideoFrameSource`'s iteration/read contract must not change from the
  consumer's point of view (`candidate_pool.py`'s `for ref in source:` /
  `source.read(ref)` calls in Task 7/9 keep working unmodified).

- [ ] **Step 1: Read the full current file**

Read `src/hydra_suite/data/al/frame_source.py` in full (140 lines, per prior
research) to see the exact current `read(ref)` implementation (reopen +
`CAP_PROP_POS_FRAMES` seek per call) and every other method/attribute on
`VideoFrameSource` and its iteration protocol (`__iter__`, whatever
`FrameRef`/`ref.frame_id` looks like), so the sequential rewrite preserves the
exact same public contract.

- [ ] **Step 2: Write the failing test for sequential-read correctness**

Add a test proving that reading frames via the source in ascending order
returns pixel-identical frames to today's per-frame reopen approach, and that a
single `cv2.VideoCapture` is reused (not reopened) across reads — e.g. by
patching `cv2.VideoCapture` with a counting wrapper:

```python
def test_video_frame_source_reuses_single_capture(monkeypatch, synthetic_video):
    open_count = {"n": 0}
    real_capture = cv2.VideoCapture

    def counting_capture(*args, **kwargs):
        open_count["n"] += 1
        return real_capture(*args, **kwargs)

    monkeypatch.setattr(cv2, "VideoCapture", counting_capture)
    source = VideoFrameSource(str(synthetic_video))
    refs = list(source)[:5]
    frames = [source.read(ref) for ref in refs]
    assert all(f is not None for f in frames)
    assert open_count["n"] <= 2  # one for iteration/probing, one for reads — not one per frame


def test_video_frame_source_sequential_reads_match_baseline(synthetic_video):
    source = VideoFrameSource(str(synthetic_video))
    refs = list(source)[:5]
    frames = [source.read(ref) for ref in refs]
    cap = cv2.VideoCapture(str(synthetic_video))
    for ref, frame in zip(refs, frames):
        cap.set(cv2.CAP_PROP_POS_FRAMES, ref.frame_id)
        ok, expected = cap.read()
        assert ok
        assert np.array_equal(frame, expected)
    cap.release()
```

(Use whatever `synthetic_video`/`FrameRef` fixtures `tests/test_al_frame_source.py`
already defines, confirmed from Step 1 — do not invent a different fixture
name than what's already there.)

- [ ] **Step 3: Run to confirm the reuse test fails**

Run: `python -m pytest tests/test_al_frame_source.py -k test_video_frame_source_reuses_single_capture -v`
Expected: FAIL (current code opens a new capture per `read()` call — `open_count["n"]` will be >= 5).

- [ ] **Step 4: Implement sequential decode**

Rewrite `VideoFrameSource` to hold one lazily-opened `cv2.VideoCapture` as an
instance attribute, tracking the last-read frame index. `read(ref)` becomes:
if `ref.frame_id` is exactly `last_read_index + 1` (or the capture hasn't been
opened yet and `ref.frame_id == 0`), call `cap.read()` directly (no seek); only
`cap.set(CAP_PROP_POS_FRAMES, ...)` when the requested frame isn't the
immediate next one (preserving correctness for any out-of-order caller, while
making the common in-order scan cheap). Add a `close()`/context-manager method
to release the capture, and ensure any existing caller (`candidate_pool.py`,
confirmed in Task 7) is updated to use it as a context manager or call `close()`
explicitly when done, if it doesn't already.

- [ ] **Step 5: Run both new tests and the full existing file**

Run: `python -m pytest tests/test_al_frame_source.py -v`
Expected: all PASS.

- [ ] **Step 6: Commit**

```bash
git add src/hydra_suite/data/al/frame_source.py tests/test_al_frame_source.py
git commit -m "perf(detectkit): reuse a single VideoCapture for sequential AL frame reads"
```

---

### Task 7: DetectKit — frame-difference prefilter + windowed dedup

**Files:**
- Modify: `src/hydra_suite/data/al/candidate_pool.py` (`build_candidate_pool`,
  lines 40-75)
- Test: `tests/test_al_candidate_pool.py` (confirmed to exist)

**Interfaces:**
- `build_candidate_pool(source, pool_config)` — signature unchanged. If
  `pool_config` (its exact type, confirmed from Step 1) doesn't already have
  fields for prefilter sensitivity / dedup window / periodic-sampling floor,
  add them with defaults that approximate current behavior when unset (dedup
  window default large enough to be a no-op difference from global dedup on
  typical fixture-sized videos; prefilter threshold default permissive enough
  not to drop real motion).

- [ ] **Step 1: Read the current function and its config type in full**

Read `src/hydra_suite/data/al/candidate_pool.py` lines 1-75 in full (the
confirmed 40-line `build_candidate_pool` body plus whatever
`ALCandidatePoolConfig`-equivalent type it takes, and the `FilterKitCore`
`compute_signature`/`is_duplicate` calls' exact signatures) before changing
anything.

- [ ] **Step 2: Write the failing tests**

```python
def test_windowed_dedup_only_compares_against_recent_window(synthetic_video_with_repeats):
    # Video: frames 0-9 are all visually distinct; frame 10 is a near-duplicate
    # of frame 0 (far outside any reasonable window) and frame 11 is a
    # near-duplicate of frame 9 (within window).
    source = VideoFrameSource(str(synthetic_video_with_repeats))
    pool_config = ALCandidatePoolConfig(dedup_window=3)  # confirm real type/field name from Step 1
    candidates = build_candidate_pool(source, pool_config)
    ids = {c.frame_id for c in candidates}
    assert 10 in ids  # far-apart near-duplicate of frame 0 is NOT deduped away
    assert 11 not in ids  # near-duplicate within the window IS deduped away


def test_frame_difference_prefilter_skips_static_frames(synthetic_video_static_and_moving):
    # Video: frames 0-19 are static (identical), frames 20-24 have motion.
    source = VideoFrameSource(str(synthetic_video_static_and_moving))
    pool_config = ALCandidatePoolConfig(motion_threshold=5.0, periodic_sample_every=50)
    candidates = build_candidate_pool(source, pool_config)
    ids = {c.frame_id for c in candidates}
    assert any(20 <= i <= 24 for i in ids)  # motion frames survive the prefilter
    assert sum(1 for i in ids if i < 20) <= 1  # static run: at most the periodic-floor sample
```

(Build the two synthetic-video fixtures using the same `cv2.VideoWriter`/
`np.random.default_rng` convention already used in
`tests/test_al_candidate_pool.py`, confirmed from Step 1 — one with a
far-apart repeated frame, one with a long static run followed by motion.)

- [ ] **Step 3: Run to confirm failure**

Run: `python -m pytest tests/test_al_candidate_pool.py -k "test_windowed_dedup_only_compares_against_recent_window or test_frame_difference_prefilter_skips_static_frames" -v`
Expected: FAIL (config fields don't exist yet / behavior not implemented).

- [ ] **Step 4: Implement windowed dedup**

Replace the current `any(fk.is_duplicate(sig, prev, ...) for prev in
kept_signatures)` (comparing against every previously kept frame) with a
bounded-length recent-window structure — e.g. a `collections.deque(maxlen=pool_config.dedup_window)`
of kept signatures — so the comparison is against at most `dedup_window` prior
entries, not the whole history.

- [ ] **Step 5: Implement the frame-difference prefilter**

Inline in the same scan loop (which now reads frames sequentially via Task 6's
`VideoFrameSource`): maintain a rolling reference frame (initialize to the
first frame). For each subsequent frame, compute a cheap grayscale
mean-absolute-difference against the reference
(`np.abs(gray.astype(np.int16) - reference_gray.astype(np.int16)).mean()`).
If the delta is below `pool_config.motion_threshold`, skip full
signature/detector-eligible scoring for this frame (it doesn't enter the
dedup/candidate consideration at all) — UNLESS it's been more than
`pool_config.periodic_sample_every` frames since the last frame that was
allowed through, in which case let it through anyway (the periodic-sampling
floor) and refresh the rolling reference to this frame either way (so slow
lighting drift doesn't lock the threshold against a stale reference).

- [ ] **Step 6: Run the new tests and the full existing file**

Run: `python -m pytest tests/test_al_candidate_pool.py -v`
Expected: all PASS, including every pre-existing test in this file (confirms
default config values keep close-to-current behavior when new fields are
unset/defaulted).

- [ ] **Step 7: Commit**

```bash
git add src/hydra_suite/data/al/candidate_pool.py tests/test_al_candidate_pool.py
git commit -m "perf(detectkit): windowed dedup + frame-difference prefilter in AL candidate scan"
```

---

### Task 8: DetectKit — `score_nms_instability` reads cached raw detections instead of re-invoking the detector

**Files:**
- Modify: `src/hydra_suite/data/al/signals.py` (`score_nms_instability`, lines 203-224)
- Modify: `src/hydra_suite/detectkit/jobs/al_worker.py` (`_frame_signals`, lines 125-160)
- Test: `tests/test_al_signals.py` if it exists (`grep -rl score_nms_instability tests/`), else `tests/test_detectkit_al_worker.py`'s existing coverage plus a new focused test file `tests/test_al_signals.py`.

**Interfaces:**
- Consumes: `filter_with_indices(raw: OBBResult, config: OBBConfig, roi_mask) ->
  (OBBResult, np.ndarray)` from
  `src/hydra_suite/core/inference/stages/filtering.py:269`.
- Modifies: `score_nms_instability`'s signature from `(frame, detector_fn,
  base_conf, base_iou)` to `(raw_obb_result, obb_config, base_conf, base_iou)`
  — no longer takes a frame or a detector closure; operates purely on an
  already-computed raw `OBBResult`.
- Produces: `_frame_signals(frame_id, raw_obb_result, obb_config,
  expected_count, base_conf, base_iou)` — was `(frame, frame_id, detector_fn,
  expected_count, base_conf, base_iou)`; this only swaps the `frame`+
  `detector_fn` pair for `raw_obb_result`+`obb_config` and keeps `frame_id` in
  its original position, since Step 1 must confirm whether anything inside
  `_frame_signals` uses `frame_id` beyond tagging output (e.g. logging) before
  deciding it's safe to drop. Consumed by Task 9's Phase 3 as:
  `_frame_signals(ref.frame_id, raw, obb_config.obb, req.expected_count,
  req.base_conf, req.base_iou)`.

- [ ] **Step 1: Read both functions in full**

Read `src/hydra_suite/data/al/signals.py` lines 195-230 (`score_nms_instability`
and `_set_iou_greedy`) and `src/hydra_suite/detectkit/jobs/al_worker.py` lines
120-165 (`_frame_signals`) in full to confirm exact current call shapes and
what `_frame_signals` currently does with the base `detector_fn` call's return
value versus what `score_nms_instability` separately recomputes.

- [ ] **Step 2: Write the failing test pinning the new contract**

```python
def test_score_nms_instability_uses_raw_result_no_detector_calls():
    raw = _make_raw_result_with_n_detections(5)  # helper building an OBBResult fixture
    config = OBBConfig(mode="direct", direct=OBBDirectConfig(model_path="unused"),
                        confidence_threshold=0.25, iou_threshold=0.5)
    # Should not require or accept a detector_fn/frame at all.
    score = score_nms_instability(raw, config, base_conf=0.25, base_iou=0.5)
    assert 0.0 <= score <= 1.0
```

- [ ] **Step 3: Run to confirm it fails**

Run: `python -m pytest tests/test_al_signals.py -k test_score_nms_instability_uses_raw_result_no_detector_calls -v`
Expected: FAIL — current signature is `(frame, detector_fn, base_conf, base_iou)`,
so this call raises a `TypeError`.

- [ ] **Step 4: Implement**

Rewrite `score_nms_instability(raw_obb_result, obb_config, base_conf, base_iou)`
to build 3 `OBBConfig` variants (base `(base_conf, base_iou)`, and the same two
perturbations currently hardcoded — `(base_conf*0.7, base_iou)` capped at
`>= 0.01`, and `(base_conf, base_iou*1.3)` capped at `<= 0.95`, confirmed from
Step 1's read), call `filter_with_indices(raw_obb_result, variant_config,
roi_mask=None)` for each, and compute the same set-IoU instability metric
(`_set_iou_greedy`, unchanged) over the three filtered detection sets instead
of the three `detector_fn(...)` outputs. Update `_frame_signals` to: call the
new batched-cache path (this is wired in fully by Task 9; for this task,
`_frame_signals` should accept an already-computed `raw_obb_result` parameter
in place of calling `detector_fn` itself, and pass it to both the base-scoring
logic and `score_nms_instability` — eliminating the redundant 4th call
identified in the spec's corrections).

- [ ] **Step 5: Run the new test and existing signals/al_worker tests**

Run: `python -m pytest tests/test_al_signals.py tests/test_detectkit_al_worker.py -v`
Expected: `test_al_signals.py`'s new test PASSes; `test_detectkit_al_worker.py`
will likely FAIL at this point since `al_worker.py`'s call sites into
`_frame_signals` haven't been updated to supply a `raw_obb_result` yet — that
full wiring is Task 9. Confirm the failures here are exactly "caller doesn't
pass the new parameter" and not something else, then proceed to Task 9 to
close the loop. Do not consider this task done until Task 9's own test run
shows `test_detectkit_al_worker.py` passing again.

- [ ] **Step 6: Commit**

```bash
git add src/hydra_suite/data/al/signals.py src/hydra_suite/detectkit/jobs/al_worker.py tests/test_al_signals.py
git commit -m "refactor(detectkit): score NMS instability from a cached raw OBBResult, not repeated detector calls"
```

---

### Task 9: DetectKit — restructure `al_worker.py` into a 3-phase batched pipeline

**Files:**
- Modify: `src/hydra_suite/detectkit/jobs/al_worker.py` (`run_active_learning`,
  lines 186-362)
- Modify: `src/hydra_suite/detectkit/gui/main_window.py`
  (`_load_active_detector_fn`, lines 1409-1464) — only to the extent its
  outputs (`kind`, model paths, `crop_pad_ratio`) are now consumed by the new
  pipeline instead of building a `detector_fn` closure; confirm during
  implementation whether `DetectorFn`/`_load_active_detector_fn` has any other
  caller before deciding whether to keep it for those callers or retire it
  entirely.
- Test: `tests/test_detectkit_al_worker.py` (confirmed to exist, uses a
  `fake_detector(frame, conf, iou)` closure pattern)

**Interfaces:**
- Consumes: `build_candidate_pool` (Task 7's windowed-dedup/prefiltered
  version), `VideoFrameSource` (Task 6's sequential version),
  `build_obb_config_for_al` (Task 5), `get_or_compute_raw` (Task 3),
  `score_nms_instability`/`_frame_signals` (Task 8's raw-result signatures),
  `build_inference_cache_dir` (`src/hydra_suite/utils/video_artifacts.py`).
- `ALRequest` (whatever its real current fields are, confirmed in Step 1)
  gains whatever new fields are needed to carry `kind`/model paths/
  `crop_pad_ratio` in place of a `detector_fn` closure, OR keeps accepting a
  pre-built `InferenceConfig` directly — confirm which is the smaller,
  more consistent change during Step 1's read and pick one; do not carry both.

- [ ] **Step 1: Read `run_active_learning` and `ALRequest` in full**

Read `src/hydra_suite/detectkit/jobs/al_worker.py` lines 1-362 in full,
including `ALRequest`'s dataclass definition (find it via `grep -n "class
ALRequest" src/hydra_suite/detectkit/jobs/al_worker.py`) and every current
field, to plan the minimal set of field changes needed.

- [ ] **Step 2: Write the failing integration test for the new 3-phase flow**

Extend `tests/test_detectkit_al_worker.py` (follow its existing
`ALRequest`/fixture-video pattern) with a test that builds an `ALRequest`
pointing at a small real OBB model fixture (check if one already exists under
`tests/fixtures/` or similar for `test_inference_runner_batch.py` — reuse it)
instead of a `fake_detector` closure, runs `run_active_learning(req)`, and
asserts: (a) it still returns a sensible `result.n_picked` on a synthetic
video with known motion, matching the shape of existing passing tests in this
file; (b) the detection cache directory (`build_inference_cache_dir(video_path)`)
now exists and contains a `detection.npz` after the run.

```python
def test_run_active_learning_populates_detection_cache(tmp_path, small_obb_model_fixture, synthetic_video_with_motion):
    req = ALRequest(
        video_path=str(synthetic_video_with_motion),
        kind="obb_direct",
        model_path=str(small_obb_model_fixture),
        secondary_model_path=None,
        crop_pad_ratio=0.15,
        base_conf=0.25,
        base_iou=0.5,
        expected_count=1,
        candidate_pool=ALCandidatePoolConfig(),  # confirm real default construction
    )
    result = run_active_learning(req)
    assert result.n_picked >= 0
    cache_dir = build_inference_cache_dir(req.video_path)
    assert (cache_dir / "detection.npz").exists()
```

(Adjust `ALRequest`'s constructor args to match whatever real field set Step 1
lands on — this sketch assumes the closure fields are replaced by
`kind`/`model_path`/`secondary_model_path`/`crop_pad_ratio`, per the Interfaces
note above; if Step 1 instead decides to keep a `detector_fn`-shaped seam for
non-AL callers and add a *parallel* `InferenceConfig`-based path, adjust this
test and every subsequent step accordingly, and document that decision inline
in the code via a one-line comment, not a placeholder.)

- [ ] **Step 3: Run to confirm it fails**

Run: `python -m pytest tests/test_detectkit_al_worker.py -k test_run_active_learning_populates_detection_cache -v`
Expected: FAIL (either a `TypeError` on `ALRequest`'s current fields, or no
cache file being written under the old detector-closure path).

- [ ] **Step 4: Implement the 3-phase restructure**

Rewrite `run_active_learning` as:

```python
def run_active_learning(req, progress=None):
    # Phase 1: sequential decode + prefilter + windowed dedup -> candidate list
    source = _build_frame_source(req)  # Task 6's sequential VideoFrameSource
    candidates = build_candidate_pool(source, req.candidate_pool)  # Task 7

    # Phase 2: one batched, cached detection pass over the whole candidate list
    obb_config = build_obb_config_for_al(  # Task 5
        req.kind, req.model_path, req.secondary_model_path,
        crop_pad_ratio=req.crop_pad_ratio,
        confidence_threshold=req.base_conf, iou_threshold=req.base_iou,
    )
    runner = InferenceRunner(obb_config, video_path=req.video_path)
    cache_dir = build_inference_cache_dir(req.video_path)
    frame_indices = [c.frame_id for c in candidates]
    frames = [source.read(c) for c in candidates]
    raw_by_idx = get_or_compute_raw(runner, cache_dir, frames, frame_indices)  # Task 3

    # Phase 3: score each candidate from its cached raw result — no further model calls
    detections_by_id = {}
    scored = []
    for ref in candidates:
        raw = raw_by_idx[ref.frame_id]
        sig, dets = _frame_signals(ref.frame_id, raw, obb_config.obb,
                                    req.expected_count, req.base_conf,
                                    req.base_iou)  # Task 8's new signature
        detections_by_id[ref.frame_id] = dets
        scored.append((ref, sig))
    # <selection/export logic below this point is unchanged from the current
    #  implementation — confirm from Step 1's read and preserve it exactly,
    #  substituting `detections_by_id`/`scored` for whatever the current
    #  equivalent local variables were named.>
```

This is a structural sketch, not a literal diff — the implementer must
transplant the actual current selection/threshold/export logic from
`run_active_learning`'s current body (Step 1) into Phase 3's tail, unchanged,
and only replace the per-frame decode+detect+score loop with the three phases
above.

- [ ] **Step 5: Run the new test and the full existing file**

Run: `python -m pytest tests/test_detectkit_al_worker.py -v`
Expected: all PASS — including the pre-existing `fake_detector`-based tests, if
`ALRequest` still supports that shape for non-model-backed testing (if Step 1's
decision retired the closure entirely, update those pre-existing tests to use
a real or fixture-backed model instead, and note this explicitly rather than
silently deleting coverage).

- [ ] **Step 6: Run Task 8's previously-blocked test**

Run: `python -m pytest tests/test_al_signals.py tests/test_detectkit_al_worker.py -v`
Expected: all PASS now that the caller wiring is complete.

- [ ] **Step 7: Commit**

```bash
git add src/hydra_suite/detectkit/jobs/al_worker.py src/hydra_suite/detectkit/gui/main_window.py tests/test_detectkit_al_worker.py
git commit -m "perf(detectkit): restructure AL scoring into batched decode/detect/score phases via InferenceRunner"
```

---

### Task 10: Equivalence fixture + wall-clock benchmark

**Files:**
- Create: `tests/test_al_detectkit_equivalence.py`
- Create: `tools/equivalence/al_benchmark.py` (or add a subcommand to an
  existing benchmark script if one already exists under `tools/` — check with
  `ls tools/` first)

**Interfaces:**
- No new production interfaces — this task only adds verification tooling.

- [ ] **Step 1: Locate a real fixture model + video**

Check `tools/equivalence/fixtures/` (per this repo's `CLAUDE.md`, fixtures are
fetched via `bash tools/equivalence/fixtures/fetch_fixtures.sh`) for an
existing small OBB model + `fly_obb`/similarly short clip usable for a direct
comparison; if a sequential-mode fixture model exists there too, use it for
the sequential-mode half of this test. If fetching fixtures isn't already done
on this machine, run `bash tools/equivalence/fixtures/fetch_fixtures.sh` first.

- [ ] **Step 2: Write the numeric-equivalence test**

For at least the `obb_direct` mode (and `sequential` if a fixture model is
available): run the OLD path (`predict_obb_for_frame_export`/`_sequential`
from `prediction_preview.py`, called directly, bypassing `al_worker.py`
entirely) and the NEW path (`build_obb_config_for_al` + `InferenceRunner.detect_batch_raw`
+ `filter_with_indices` at the same `(conf, iou)`) over the same handful of
real frames from the fixture video, and assert the resulting detection
centroids/angles/confidences match within a small numeric tolerance (not
necessarily bit-identical, since the code paths differ, but must agree on
detection count and geometry within e.g. 1e-3 relative tolerance — pick a
tolerance and justify it with a comment referencing floating-point
non-associativity across different code paths, not "close enough").

```python
def test_new_detection_path_matches_old_path_on_fixture(fixture_video, fixture_obb_model):
    frame = _read_frame(fixture_video, 0)
    old_dets = predict_obb_for_frame_export(fixture_obb_model, frame, device="cpu", conf=0.05, iou=0.5)
    new_cfg = build_obb_config_for_al("obb_direct", str(fixture_obb_model), None,
                                       crop_pad_ratio=0.15, confidence_threshold=0.05, iou_threshold=0.5)
    runner = InferenceRunner(new_cfg)
    raw = runner.detect_batch_raw([frame], frame_indices=[0])[0]
    filtered, _ = filter_with_indices(raw, new_cfg.obb, roi_mask=None)
    assert filtered.num_detections == len(old_dets)
    # compare centroids/angles within tolerance, sorted by position to avoid order dependence
```

- [ ] **Step 3: Run it**

Run: `python -m pytest tests/test_al_detectkit_equivalence.py -v`
Expected: PASS. If it fails with a genuine numeric divergence beyond
tolerance, treat this as a real finding — stop and report it rather than
loosening the tolerance to force a pass.

- [ ] **Step 4: Write the wall-clock benchmark script**

Create `tools/equivalence/al_benchmark.py`: given a video path and model path,
times `run_active_learning` end-to-end on the OLD `al_worker.py` code (checked
out at the commit before Task 6, via a throwaway git worktree — same pattern
CLAUDE.md's equivalence harness already uses for `legacy/main`) versus the
current tree, on the same fixture, and prints wall-clock for each plus the
speedup ratio. This does not need to be a pytest test — it's a manual
verification tool, run once to confirm the actual goal (this is a performance
effort) and its output pasted into the final PR/commit description, not
committed as an automated gate.

- [ ] **Step 5: Run the benchmark and record results**

Run: `python tools/equivalence/al_benchmark.py --video <fixture> --model <fixture model>`
Record the before/after wall-clock numbers.

- [ ] **Step 6: Commit**

```bash
git add tests/test_al_detectkit_equivalence.py tools/equivalence/al_benchmark.py
git commit -m "test(detectkit): add AL detection-path equivalence check and wall-clock benchmark"
```
