# Legacy Detection Cache Retirement Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Delete the legacy `DetectionCache` system, collapse every per-video artifact onto the modern `.inference_cache_<stem>/` directory, retire the visible `<stem>_caches/` folder, and move tracking-profile JSON to `<stem>_logs/`.

**Architecture:** The modern `InferenceRunner` cache (`core/inference/cache/store.py`, a directory holding `detection.npz` keyed by `cache_key`) is already the live path. This plan repoints the `detection_cache_path` *anchor* — which pose/detected-props caches place themselves alongside — from `<stem>_caches/<stem>_detection_cache_<model>.npz` (a file nothing writes) to `.inference_cache_<stem>/detection.npz` (the real modern file). All remaining legacy *readers* are ported to one shared modern read-only reader, then the legacy class and dead path builders are deleted.

**Tech Stack:** Python, NumPy (`.npz`), pytest. No new dependencies.

**Spec:** `docs/superpowers/specs/2026-08-13-legacy-detection-cache-retirement-design.md`

## Global Constraints

- **Isolation:** all work happens in the existing worktree `.worktrees/legacy-cache-retirement` on branch `chore/legacy-cache-retirement` (already created from HEAD).
- **Commit identity:** commit as the configured git user; **do not** add a `Co-Authored-By: Claude` trailer.
- **Whole-file cache semantics:** the modern cache writes whole-file (`np.savez` rewrites the entire `.npz`, no append). Never merge a narrow-window cache into a shared `detection.npz`.
- **Detection cache key excludes confidence/IoU** — those are re-applied as post-hoc filters; do not add them to any key.
- **No auto-migration:** do not auto-delete or migrate pre-existing `<stem>_caches/` folders; they become inert.
- **Verification:** tests on `hydra-mps` here; equivalence harness must stay byte-identical on MPS (this box) and CUDA (mehek). Kill stale sleap/hydra processes before heavy runs.
- **Test running:** the whole suite hangs on unrelated classkit modal dialogs — run tests per-file / per-directory, never a bare `pytest tests/`.

---

### Task 1: Shared inference-cache-dir helper + repoint the detection-cache anchor

**Files:**
- Modify: `src/hydra_suite/utils/video_artifacts.py` (`build_video_cache_dir` region, `build_detection_cache_path`, `find_existing_detection_cache_path`)
- Modify: `src/hydra_suite/core/tracking/worker.py:4075-4078` (`_resolve_cache_dir`)
- Test: `tests/utils/test_video_artifacts_inference_cache_dir.py`

**Interfaces:**
- Produces: `build_inference_cache_dir(video_path, artifact_base_dir=None, create=False) -> Path` returning `<base>/.inference_cache_<stem>`; `build_detection_cache_path(video_path, model_id, artifact_base_dir=None, create_dir=False) -> Path` now returns `<inference_cache_dir>/detection.npz` (signature unchanged; `model_id` accepted but no longer encoded in the filename, matching modern cache-key semantics).

- [ ] **Step 1: Write the failing test**

```python
# tests/utils/test_video_artifacts_inference_cache_dir.py
from pathlib import Path
from hydra_suite.utils import video_artifacts as va


def test_inference_cache_dir_next_to_video(tmp_path):
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"x")
    assert va.build_inference_cache_dir(video) == tmp_path / ".inference_cache_clip"


def test_detection_cache_path_is_modern_detection_npz(tmp_path):
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"x")
    p = va.build_detection_cache_path(video, "modelXYZ")
    assert p == tmp_path / ".inference_cache_clip" / "detection.npz"


def test_props_caches_land_alongside_detection_cache(tmp_path):
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"x")
    det = va.build_detection_cache_path(video, "m")
    pose = va.build_individual_properties_cache_path(
        video, "pid", 0, 10, detection_cache_path=str(det)
    )
    assert pose.parent == tmp_path / ".inference_cache_clip"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/utils/test_video_artifacts_inference_cache_dir.py -v`
Expected: FAIL (`build_inference_cache_dir` undefined; `build_detection_cache_path` returns the old `<stem>_caches/...` path).

- [ ] **Step 3: Implement**

In `video_artifacts.py`, add `build_inference_cache_dir` (mirrors `worker._resolve_cache_dir` but honors `artifact_base_dir` like the other builders):

```python
def build_inference_cache_dir(
    video_path: str | os.PathLike[str],
    artifact_base_dir: str | os.PathLike[str] | None = None,
    create: bool = False,
) -> Path:
    """Return the modern InferenceRunner cache dir, ``.inference_cache_<stem>/``."""
    base_dir = (
        _normalize_base_dir(artifact_base_dir) or Path(video_path).expanduser().parent
    )
    stem = _video_stem(video_path)
    expected = f".inference_cache_{stem}"
    cache_dir = base_dir if base_dir.name == expected else base_dir / expected
    if create:
        cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir
```

Rewrite `build_detection_cache_path` to return the modern file:

```python
def build_detection_cache_path(
    video_path, model_id, artifact_base_dir=None, create_dir=False,
) -> Path:
    """Return the modern detection cache file, ``.inference_cache_<stem>/detection.npz``.

    ``model_id`` is accepted for signature stability but no longer encoded in the
    filename — the modern cache key inside the directory carries model identity.
    """
    cache_dir = build_inference_cache_dir(
        video_path, artifact_base_dir=artifact_base_dir, create=create_dir,
    )
    return cache_dir / "detection.npz"
```

Rewrite `find_existing_detection_cache_path` to look in the modern dir and drop the legacy flat-file fallback:

```python
def find_existing_detection_cache_path(
    video_path, model_id, artifact_base_dirs=None,
) -> Path | None:
    base_dirs = artifact_base_dirs or candidate_artifact_base_dirs(video_path)
    for base_dir in base_dirs:
        current = build_detection_cache_path(
            video_path, model_id, artifact_base_dir=base_dir,
        )
        if current.exists():
            return current
    return None
```

In `worker.py`, make `_resolve_cache_dir` delegate so the two never drift:

```python
def _resolve_cache_dir(self) -> Path:
    """Return the per-video cache directory for InferenceRunner caches."""
    from hydra_suite.utils.video_artifacts import build_inference_cache_dir
    return build_inference_cache_dir(self.video_path)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/utils/test_video_artifacts_inference_cache_dir.py -v`
Expected: PASS.

- [ ] **Step 5: Regression-check existing video_artifacts tests**

Run: `python -m pytest tests/ -k "video_artifacts or tracking_cache" -v`
Expected: PASS (fix any test asserting the old `<stem>_caches/...detection_cache...` path — update it to the modern path; that is the intended behavior change).

- [ ] **Step 6: Commit**

```bash
git add src/hydra_suite/utils/video_artifacts.py src/hydra_suite/core/tracking/worker.py tests/utils/test_video_artifacts_inference_cache_dir.py
git commit -m "refactor(cache): repoint detection-cache anchor to .inference_cache_<stem>/detection.npz"
```

---

### Task 2: Shared read-only modern detection-cache reader

**Files:**
- Create: `src/hydra_suite/core/inference/cache/reader.py`
- Modify: `src/hydra_suite/core/inference/cache/__init__.py` (export the helper)
- Test: `tests/core/inference/cache/test_detection_reader.py`

**Interfaces:**
- Produces: `open_detection_cache_reader(path) -> DetectionCacheHandle` — opens an existing `detection.npz` with a path-only `CacheKey` (validity = path exists). Callers use `.read_frame(frame_idx) -> OBBResult | None` and `.path`. This is the single extraction of the ad-hoc shim currently inlined in `oriented_video.py:25-56`.

- [ ] **Step 1: Write the failing test**

```python
# tests/core/inference/cache/test_detection_reader.py
import numpy as np
from hydra_suite.core.inference.cache.base import CacheKey
from hydra_suite.core.inference.cache.store import DetectionCacheHandle
from hydra_suite.core.inference.cache.reader import open_detection_cache_reader
from hydra_suite.core.inference.result import OBBResult


def _write_one_frame(path):
    key = CacheKey(schema_version=3, model_path="m", model_mtime=1.0, config_hash="h")
    h = DetectionCacheHandle(path=path, key=key)
    h.write_frame(
        0,
        result=OBBResult(
            frame_idx=0,
            centroids=np.array([[1.0, 2.0]], np.float32),
            angles=np.array([0.5], np.float32),
            sizes=np.array([10.0], np.float32),
            shapes=np.array([[100.0, 2.0]], np.float32),
            confidences=np.array([0.9], np.float32),
            corners=np.zeros((1, 4, 2), np.float32),
            detection_ids=np.array([7], np.int64),
        ),
    )
    h.close()


def test_reader_round_trips_a_frame(tmp_path):
    p = tmp_path / "detection.npz"
    _write_one_frame(p)
    reader = open_detection_cache_reader(p)
    res = reader.read_frame(0)
    assert res is not None
    assert res.detection_ids.tolist() == [7]
    assert reader.read_frame(999) is None  # unwritten frame -> None


def test_reader_missing_file_reads_none(tmp_path):
    reader = open_detection_cache_reader(tmp_path / "detection.npz")
    assert reader.read_frame(0) is None
```

(Confirm the exact `OBBResult` constructor fields by reading `core/inference/result.py` before running — adjust kwargs to match.)

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/core/inference/cache/test_detection_reader.py -v`
Expected: FAIL (`reader` module missing).

- [ ] **Step 3: Implement**

```python
# src/hydra_suite/core/inference/cache/reader.py
"""Read-only opener for a modern detection.npz, independent of run config."""
from __future__ import annotations
from pathlib import Path
from .base import CacheKey
from .store import DetectionCacheHandle


def open_detection_cache_reader(path: str | Path) -> DetectionCacheHandle:
    """Open an existing ``detection.npz`` for reading.

    Uses a path-only key: ``is_valid()``/``read_frame`` gate on file existence and
    the stored ``written_frames`` set, not on a run-config match. Intended for
    consumers (RefineKit overlays, dataset exporters) that only need the geometry
    already on disk, regardless of which run produced it.
    """
    key = CacheKey(schema_version=0, model_path="", model_mtime=0.0, config_hash="")
    return DetectionCacheHandle(path=Path(path), key=key)
```

Read `store.py:DetectionCacheHandle.is_valid`/`_check_key` first: confirm a path-only key still lets `read_frame` return data. `read_frame` guards on `is_valid()` (→ `_check_key`). If `_check_key` compares the stored `cache_key` against the passed key and would reject a real file, add a `require_key: bool = False` param to the handle or a `read_only` flag so existence alone suffices — mirror exactly what the current `oriented_video.py` shim relies on (its shim used an empty key successfully, so replicate that behavior). Keep the reader's semantics identical to that working shim.

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/core/inference/cache/test_detection_reader.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/hydra_suite/core/inference/cache/reader.py src/hydra_suite/core/inference/cache/__init__.py tests/core/inference/cache/test_detection_reader.py
git commit -m "feat(cache): shared read-only modern detection-cache reader"
```

---

### Task 3: Port `oriented_video.py` to the shared reader

**Files:**
- Modify: `src/hydra_suite/core/individual/dataset/oriented_video.py:20-58` (remove inline shim + legacy `ImportError` fallback), `:625-660` (open via shared reader)
- Test: `tests/core/individual/dataset/test_oriented_video_actual_rows.py`

**Interfaces:**
- Consumes: `open_detection_cache_reader` (Task 2). The existing `_add_actual_tasks` already reads an `OBBResult` from `get_frame`/`read_frame`.

- [ ] **Step 1: Write the failing test** — an integration test that builds a tiny modern `detection.npz` (reuse the `_write_one_frame` helper pattern from Task 2), constructs the oriented-video builder with `detection_cache_path` pointing at it plus a one-row "actual" CSV record for frame 0, and asserts `_add_actual_tasks` produces one task with corners/affine (no exception). Read the builder's constructor + `_add_actual_tasks` signature first to wire the fixture. Assert that a **missing** cache file no longer raises `FileNotFoundError` but yields zero actual tasks (graceful).

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/core/individual/dataset/test_oriented_video_actual_rows.py -v`
Expected: FAIL (currently raises `FileNotFoundError`, or the inline shim path is used).

- [ ] **Step 3: Implement** — delete the `try/except` shim block (`oriented_video.py:25-58`) and the `from ....data.detection_cache import DetectionCache` fallback; replace the `DetectionCache(self.detection_cache_path, mode="r")` open (~632) with `open_detection_cache_reader(self.detection_cache_path)`; replace the raise-if-missing (~628-631) with a guard that skips actual-row geometry (leaves those rows to the same param-based `_build_task` path interpolated rows already use) when the file is absent.

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/core/individual/dataset/test_oriented_video_actual_rows.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/hydra_suite/core/individual/dataset/oriented_video.py tests/core/individual/dataset/test_oriented_video_actual_rows.py
git commit -m "refactor(oriented-video): read modern detection cache via shared reader"
```

---

### Task 4: Port RefineKit overlay + merge-wizard readers

**Files:**
- Modify: `src/hydra_suite/refinekit/gui/overlay_utils.py:104-200` (`discover_detection_cache`, `FrameDetections`, `load_frame_detections`)
- Modify: `src/hydra_suite/refinekit/gui/dialogs/merge_wizard.py:444-467, 1195` (`_discover_detection_cache`, `_FrameDetections`)
- Test: `tests/refinekit/test_overlay_modern_cache.py`

**Interfaces:**
- Consumes: `open_detection_cache_reader` (Task 2) and `build_inference_cache_dir` (Task 1).

- [ ] **Step 1: Write the failing test**

```python
# tests/refinekit/test_overlay_modern_cache.py
import numpy as np
from hydra_suite.refinekit.gui.overlay_utils import (
    discover_detection_cache, load_frame_detections,
)
# reuse Task 2's _write_one_frame to populate the modern cache


def test_discover_finds_modern_inference_cache(tmp_path):
    video = tmp_path / "clip.mp4"; video.write_bytes(b"x")
    cache_dir = tmp_path / ".inference_cache_clip"; cache_dir.mkdir()
    _write_one_frame(cache_dir / "detection.npz")   # helper copied into this test
    assert discover_detection_cache(str(video)) == cache_dir / "detection.npz"


def test_load_frame_detections_from_modern_cache(tmp_path):
    video = tmp_path / "clip.mp4"; video.write_bytes(b"x")
    cache_dir = tmp_path / ".inference_cache_clip"; cache_dir.mkdir()
    _write_one_frame(cache_dir / "detection.npz")
    fd = load_frame_detections(str(video))
    assert fd is not None
    got = fd.get(0)
    assert got is not None                 # (meas, semi_axes, obb) tuple
    meas, semi_axes, obb = got
    assert len(meas) == 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/refinekit/test_overlay_modern_cache.py -v`
Expected: FAIL (`discover_detection_cache` still globs `<stem>_caches/`; `FrameDetections.get` unpacks the legacy 12-tuple).

- [ ] **Step 3: Implement**
  - `discover_detection_cache`: return `build_inference_cache_dir(video_path) / "detection.npz"` if it exists, else `None`. Drop the `<stem>_caches/` and flat-file globs.
  - `FrameDetections.__init__`: accept a `DetectionCacheHandle` from `open_detection_cache_reader`.
  - `FrameDetections.get`: replace the 12-tuple unpack with `res = self._cache.read_frame(frame_idx)`; if `res is None or res.num_detections == 0` return `None`. Build `meas_arr` from `res.centroids`, `shapes_arr` from `res.shapes`, `obb_scaled` from `res.corners` — applying the same `inv_resize` scaling as today. (Keep the existing semi-axis math; it consumes `shapes` area/aspect.)
  - `load_frame_detections`: open via `open_detection_cache_reader`; drop `is_compatible()` (the reader gates on existence).
  - Apply the identical changes to `merge_wizard.py`'s `_discover_detection_cache` and `_FrameDetections`.

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/refinekit/test_overlay_modern_cache.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/hydra_suite/refinekit/gui/overlay_utils.py src/hydra_suite/refinekit/gui/dialogs/merge_wizard.py tests/refinekit/test_overlay_modern_cache.py
git commit -m "refactor(refinekit): read modern detection cache for overlays"
```

---

### Task 5: Port `interpolated_crops.py` + `dataset_export.py` optional readers

**Files:**
- Modify: `src/hydra_suite/core/post/interpolated_crops.py:79-105, 1473-1531` (`_get_detection_size`, setup open)
- Modify: `src/hydra_suite/core/post/dataset_export.py:54-81` (optional detection-quality enrichment)
- Test: `tests/core/post/test_interpolated_crops_size_lookup.py`

**Interfaces:**
- Consumes: `open_detection_cache_reader` (Task 2).

- [ ] **Step 1: Write the failing test** — populate a modern `detection.npz` with one frame whose `shapes` encode a known OBB size; call `_get_detection_size(reader, frame_id=0, detection_id=<id>)` and assert it returns that size; call with a missing detection id and assert the `REFERENCE_BODY_SIZE` fallback. Read `_get_detection_size` first to match its exact signature/return.

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/core/post/test_interpolated_crops_size_lookup.py -v`
Expected: FAIL (`_get_detection_size` unpacks the legacy tuple).

- [ ] **Step 3: Implement** — in `interpolated_crops.py` replace the `DetectionCache(detection_cache_path, mode="r")` open (1475) with `open_detection_cache_reader(...)`; rewrite `_get_detection_size` to read `read_frame(frame_id)` → `OBBResult`, select the row by `detection_id`, derive width/height from `shapes`/`corners`; keep the `REFERENCE_BODY_SIZE` fallback on `None`/miss. In `dataset_export.py` replace the `DetectionCache(...)` open (56) with the reader and adapt the field access; keep the CSV-column fallback.

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/core/post/test_interpolated_crops_size_lookup.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/hydra_suite/core/post/interpolated_crops.py src/hydra_suite/core/post/dataset_export.py tests/core/post/test_interpolated_crops_size_lookup.py
git commit -m "refactor(post): size/quality lookups read modern detection cache"
```

---

### Task 6: Remove the legacy validity-probe branch in orchestrators/config

**Files:**
- Modify: `src/hydra_suite/trackerkit/gui/orchestrators/config.py:2920-2955` (the `_is_valid`/probe helper)
- Test: `tests/trackerkit/test_detection_cache_validity_probe.py`

- [ ] **Step 1: Write the failing test** — assert the probe returns valid for a modern `.inference_cache_<stem>/` dir covering the range, and returns not-valid (no exception) for a bare non-existent file path. Read the probe function first for its exact name/signature.

- [ ] **Step 2: Run to verify it fails** → `python -m pytest tests/trackerkit/test_detection_cache_validity_probe.py -v`

- [ ] **Step 3: Implement** — delete the `else: DetectionCache(path, mode="r")` file branch (2949); keep the `os.path.isdir(path)` modern branch; a non-dir/non-existent path returns not-valid.

- [ ] **Step 4: Run to verify it passes** → same command, Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/hydra_suite/trackerkit/gui/orchestrators/config.py tests/trackerkit/test_detection_cache_validity_probe.py
git commit -m "refactor(trackerkit): drop legacy-file branch from cache validity probe"
```

---

### Task 7: Delete the legacy `DetectionCache` class + dead worker write-path

**Files:**
- Delete: `src/hydra_suite/data/detection_cache.py`
- Modify: `src/hydra_suite/data/__init__.py` (remove the `DetectionCache` re-export)
- Modify: `src/hydra_suite/core/tracking/worker.py:951, 3853-3867, 3942-3952` (dead write-path + stale comment)

- [ ] **Step 1: Prove nothing imports it**

Run: `grep -rn "detection_cache import DetectionCache\|from .detection_cache\|data.detection_cache\|DetectionCache(" src/ tests/`
Expected: **zero** hits after Tasks 3–6 (all readers ported). If any remain, port them before continuing — do not proceed.

- [ ] **Step 2: Delete + strip** — `git rm src/hydra_suite/data/detection_cache.py`; remove its `__init__` export; in `worker.py` delete the `detection_cache = None` line (951), the `add_frame` top-up loop (3853-3867), and the save/close block (3942-3952) including the stale "only used for background subtraction" comment. Leave `inference_runner.close()` / `bgsub_runner.close()` intact.

- [ ] **Step 3: Verify imports + worker still load**

Run: `python -c "import hydra_suite.core.tracking.worker, hydra_suite.data"` then `python -m pytest tests/ -k "worker" -v`
Expected: import OK; worker tests PASS.

- [ ] **Step 4: Commit**

```bash
git add -A
git commit -m "chore: delete legacy DetectionCache class and dead worker write-path"
```

---

### Task 8: Delete dead apriltag/classify/legacy path builders

**Files:**
- Modify: `src/hydra_suite/utils/video_artifacts.py` (`build_apriltag_cache_path`, `find_existing_apriltag_cache_path`, `build_classify_cache_path`, `find_existing_classify_cache_path`, `build_legacy_detection_cache_path`)
- Test: `tests/utils/test_video_artifacts_no_dead_builders.py`

- [ ] **Step 1: Prove they are unused**

Run: `grep -rn "build_apriltag_cache_path\|find_existing_apriltag_cache_path\|build_classify_cache_path\|find_existing_classify_cache_path\|build_legacy_detection_cache_path" src/ tests/`
Expected: only definitions (and any test asserting their absence). If a live caller exists, STOP and re-scope.

- [ ] **Step 2: Write the guard test**

```python
# tests/utils/test_video_artifacts_no_dead_builders.py
from hydra_suite.utils import video_artifacts as va


def test_dead_builders_removed():
    for name in (
        "build_apriltag_cache_path", "find_existing_apriltag_cache_path",
        "build_classify_cache_path", "find_existing_classify_cache_path",
        "build_legacy_detection_cache_path",
    ):
        assert not hasattr(va, name), f"{name} should be deleted"
```

- [ ] **Step 3: Run to verify it fails** → `python -m pytest tests/utils/test_video_artifacts_no_dead_builders.py -v`

- [ ] **Step 4: Delete the five functions.**

- [ ] **Step 5: Run to verify it passes** → same command, Expected: PASS. Also `python -c "import hydra_suite.utils.video_artifacts"`.

- [ ] **Step 6: Commit**

```bash
git add src/hydra_suite/utils/video_artifacts.py tests/utils/test_video_artifacts_no_dead_builders.py
git commit -m "chore: remove dead apriltag/classify/legacy cache path builders"
```

---

### Task 9: Relocate the optimizer fallback cache under `.inference_cache_<stem>/opt/`

**Files:**
- Modify: `src/hydra_suite/utils/video_artifacts.py:build_optimizer_detection_cache_path`
- Modify: `src/hydra_suite/trackerkit/gui/orchestrators/config.py:2975 (iter_detection_cache_candidates), :2994-3004 (_build_optimizer_detection_cache)`
- Test: `tests/utils/test_optimizer_cache_path.py`

**Interfaces:**
- Consumes: `build_inference_cache_dir` (Task 1). Main-cache reuse (step 1 of the resolver) is unchanged.

- [ ] **Step 1: Write the failing test**

```python
# tests/utils/test_optimizer_cache_path.py
from hydra_suite.utils import video_artifacts as va


def test_optimizer_cache_under_inference_cache_opt(tmp_path):
    video = tmp_path / "clip.mp4"; video.write_bytes(b"x")
    p = va.build_optimizer_detection_cache_path(video, "modelA", 100)
    assert p == tmp_path / ".inference_cache_clip" / "opt"
    assert "_caches" not in str(p) and "r100" not in str(p)
```

- [ ] **Step 2: Run to verify it fails** → `python -m pytest tests/utils/test_optimizer_cache_path.py -v`

- [ ] **Step 3: Implement** — change `build_optimizer_detection_cache_path` to return `build_inference_cache_dir(video_path, artifact_base_dir=..., create=create_dir) / "opt"` (a cache *directory*, matching how the optimizer worker passes it as `cache_dir`). Keep the `model_name`/`resize_percent` params for signature stability but stop encoding them. Update `iter_detection_cache_candidates` to scan `.inference_cache_<stem>/opt` (and drop `<stem>_caches` scanning). Confirm `_build_optimizer_detection_cache` still passes a directory to `DetectionCacheBuildWorker`.

- [ ] **Step 4: Run to verify it passes** → same command, Expected: PASS. Also `python -m pytest tests/ -k "optimizer" -v`.

- [ ] **Step 5: Commit**

```bash
git add src/hydra_suite/utils/video_artifacts.py src/hydra_suite/trackerkit/gui/orchestrators/config.py tests/utils/test_optimizer_cache_path.py
git commit -m "refactor(optimizer): fallback detection cache lives under .inference_cache_<stem>/opt"
```

---

### Task 10: Route tracking-profile JSON to `<stem>_logs/`

**Files:**
- Modify: `src/hydra_suite/core/tracking/worker.py:3993-4013` (profile path selection)
- Test: `tests/core/tracking/test_profile_output_path.py`

- [ ] **Step 1: Write the failing test** — extract the profile-path decision into a small pure helper if it is currently inline (e.g. `_resolve_profile_path(self, dir_tag)`), then test: with no `video_output_path`, the path is `build_video_log_dir(video_path)/f"tracking_profile_{dir_tag}.json"`; with a `video_output_path`, it stays next to the output video. Read `worker.py:3993-4013` first to preserve the output-video branch exactly.

- [ ] **Step 2: Run to verify it fails** → `python -m pytest tests/core/tracking/test_profile_output_path.py -v`

- [ ] **Step 3: Implement** — replace the `detection_cache_path.parent` branch with `build_video_log_dir(self.video_path, create=True) / f"tracking_profile_{dir_tag}.json"`; keep the output-video-adjacent branch; keep the "skip on stop_requested" guard.

- [ ] **Step 4: Run to verify it passes** → same command, Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/hydra_suite/core/tracking/worker.py tests/core/tracking/test_profile_output_path.py
git commit -m "refactor(profiling): write tracking_profile JSON under <stem>_logs/"
```

---

### Task 11: Update "Clear All Caches" to scan the modern dir (keep legacy glob for deletion)

**Files:**
- Modify: `src/hydra_suite/trackerkit/gui/orchestrators/tracking.py:400-440` (`_iter_cache_artifact_paths`)
- Test: `tests/trackerkit/test_clear_caches_scan.py`

- [ ] **Step 1: Write the failing test** — create a video with both a `.inference_cache_<stem>/` dir and a stale `<stem>_caches/` dir and a `<stem>_logs/` dir; assert the iterator yields all three (modern dir + legacy folder for cleanup + logs). Read `_iter_cache_artifact_paths` first for its exact yield type (paths vs globs).

- [ ] **Step 2: Run to verify it fails** → `python -m pytest tests/trackerkit/test_clear_caches_scan.py -v`

- [ ] **Step 3: Implement** — add `.inference_cache_<stem>/` (via `build_inference_cache_dir`) and `<stem>_logs/` (via `build_video_log_dir`) to the scan; **retain** the `<stem>_caches/` glob strictly for deletion of stale folders.

- [ ] **Step 4: Run to verify it passes** → same command, Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/hydra_suite/trackerkit/gui/orchestrators/tracking.py tests/trackerkit/test_clear_caches_scan.py
git commit -m "refactor(trackerkit): Clear-All-Caches scans .inference_cache_ + keeps legacy cleanup glob"
```

---

### Task 12: Full verification gate

**Files:** none (verification only). Produces a summary comment on the branch.

- [ ] **Step 1: Delta test suite** — run the touched areas per-file/per-dir (never bare `pytest tests/` — it hangs on classkit dialogs):

```bash
python -m pytest tests/utils tests/core/inference/cache tests/core/post tests/core/individual/dataset tests/refinekit tests/trackerkit tests/core/tracking -v
```

Expected: PASS or, for pre-existing failures, unchanged vs a `main` baseline (compare against the known-residuals list). Investigate any *new* failure.

- [ ] **Step 2: Lint/format**

```bash
make commit-prep && make lint-moderate
```

- [ ] **Step 3: Fresh-run artifact check** — run a short forward tracking pass on one fixture clip and confirm on disk: `.inference_cache_<stem>/` contains `detection.npz` + pose/detected-props caches; `<stem>_logs/tracking_profile_forward.json` exists; **no** `<stem>_caches/` folder was created.

- [ ] **Step 4: Equivalence harness (byte-identical) — MPS here.** Per `tools/equivalence/README.md`, baseline = `legacy/main`, current = `HEAD`. Run forward + backward + reuse across the clip matrix and confirm EQUIVALENCE at the DETERMINISM floor for both `_forward.csv` and `_tracking_final.csv` (cache-path changes must not alter tracking output). Verify row counts > 0.

```bash
conda activate hydra-mps
REPO=$PWD WT=$PWD MAIN_SRC=$PWD/.worktrees/equiv-legacy/src WT_SRC=$PWD/.worktrees/legacy-cache-retirement/src \
  OUT=/tmp/equiv_legacy_cache RUNTIME=mps bash tools/equivalence/run_matrix.sh
```

- [ ] **Step 5: Equivalence harness — CUDA on mehek.** Repeat on `mehek` with `RUNTIME=cuda` (see CLAUDE.md CUDA-box recipe).

- [ ] **Step 6: Record results** — note MPS + CUDA equivalence outcomes and the fresh-run artifact check in the branch's final commit message or a short `docs/superpowers/plans/notes/` entry.

---

## Self-Review

**Spec coverage:** A=Task1/9 (anchor+optimizer→`.inference_cache_`), B=Task1 (repoint), C=Tasks 3/4/5 (reader ports) + Task2 (shared reader), Delete-set=Tasks 6/7/8, D=Task9 (optimizer), E=Task10 (profiling), F=Task11 (Clear-All-Caches). Testing/verification=Task12. All spec sections mapped.

**Placeholder scan:** Tasks with fixture-dependent tests (3,4,5,6,10,11) intentionally instruct "read the exact signature first" because the current code must be read to match names — the *transformation* and *assertion intent* are concrete, not deferred. No "TODO/handle edge cases" placeholders.

**Type consistency:** `build_inference_cache_dir`, `build_detection_cache_path`, `open_detection_cache_reader` names are used identically across Tasks 1–11. `OBBResult` field access (centroids/angles/shapes/corners/detection_ids) is consistent across Tasks 2–5.
