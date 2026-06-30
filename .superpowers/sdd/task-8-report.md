# Task 8 Report: Depth-Invariance Test Harness

## Status
DONE — `test_depth1_is_deterministic_across_runs` passes.

## Files Created
- `tests/helpers/tiny_clip.py` — harness
- `tests/test_inference_depth_invariance.py` — test

## How the Tiny Clip + Stubs Work

**Tiny clip**: 6-frame 64×64 MP4 written via `cv2.VideoWriter` with deterministic pixel content
(frame `i` fills uniformly with `(i*40) % 256`). Written to a `NamedTemporaryFile` and deleted after the pass.

**Stub models**: `_load_all_models` is patched to return an `_AllModels` with:
- `obb=OBBModels(mode="direct", direct_model=MagicMock())` — real structure, never called
- `headtail=None`, `cnn=[]`, `pose=None`, `apriltag=None` — all disabled

**Stub `run_obb`**: `hydra_suite.core.inference.pipeline.run_obb` is patched with `_fake_run_obb`, which returns 2-detection `OBBResult`s seeded from `np.random.default_rng(0)` (frame_idx=0 for all; the pipeline re-stamps real frame indices). This produces stable, deterministic OBB geometry.

**Config**: `InferenceConfig(detection_batch_size=2, pipeline_depth=depth)` — 6 frames with batch=2 → 3 windows, exercising multiple pipeline iterations.

## What Caches Get Written

Only `detection.npz` — because headtail/cnn/pose/apriltag are all disabled. The detection cache stores one `OBBResult` per frame (6 frames total), serialised via `np.savez`.

## Stability Key: Video Signature

The video signature (`video_signature()`) uses `st_size:st_mtime_ns` of the file. Since two separate `run_pipeline_to_caches` calls write to different temp files at different times, `mtime_ns` differs → different cache keys → different `.npz` bytes. Fix: pass `video_path=None` to `InferenceRunner.__init__` (which makes `_video_sig=""`, a no-op signature). The actual video path is only passed to `run_batch_pass` for `cv2.VideoCapture`. Both runs then share the empty signature → identical cache keys → byte-identical `.npz` files.

## TDD Evidence

Step 2 — test collected but `ModuleNotFoundError` on helper:
```
ERROR collecting tests/test_inference_depth_invariance.py
E   ModuleNotFoundError: No module named 'tests.helpers.tiny_clip'
```

Step 4 — after implementing helper (before signature fix), assertion failure:
```
AssertionError: assert {'detection.npz': '45283268...'} == {'detection.npz': '4ce31db5...'}
```

Step 4b — after passing `video_path=None` to constructor:
```
1 passed in 3.28s
```

Final:
```
PYTHONPATH=src KMP_DUPLICATE_LIB_OK=TRUE python -m pytest tests/test_inference_depth_invariance.py -xvs
1 passed in 3.33s
```

## Concerns

1. **`_fake_run_obb` always returns `frame_idx=0`** — the pipeline's re-stamp loop overwrites this with the real frame index, so detection_ids in the cache are correct (`frame_idx * 10000 + slot`). This is consistent with the existing `test_run_batch_iterates_frames_and_writes_caches` pattern.
2. **Only `detection.npz` is written** — headtail/cnn/pose/apriltag require real model shapes that are non-trivial to stub at the cache-write level. The depth-invariance property applies equally to any enabled cache type; when Tasks 11/12 add real concurrency, those caches can be added if needed.
3. **Video encoding reproducibility**: tested explicitly — two `cv2.VideoWriter` calls with identical frames to different temp paths produce byte-identical MP4 files (same fourcc, size, mtime-insensitive content). Only the cache key was mtime-sensitive, which is now bypassed.
