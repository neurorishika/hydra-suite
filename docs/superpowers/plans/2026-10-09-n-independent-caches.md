# N-independent inference caches Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the number of animals N (`MAX_TARGETS`) a replay-time knob: no inference cache key or extraction-time decision depends on N, and one LOUD limit `MAX_DETECTIONS_PER_FRAME = 1024` bounds stored detections, per-animal analyses and N.

**Architecture:** Extraction stores every detection with confidence ≥ `EXTRACTION_CONFIDENCE_FLOOR` (0.01), confidence-ranked and bounded only by the 1024 limit. All N logic moves to replay (`filter_with_indices` / `filter_for_source`): a 2N prefix window, the filters, then the final N cut. Per-animal stages (head-tail, pose, CNN, AprilTag) run on the N-free superset (all filter survivors), are stored by raw cache index, and are narrowed to the final N set on read.

**Tech Stack:** Python 3.13, numpy, torch, pytest; repo `hydra_suite.core.inference`, `hydra_suite.trackerkit`.

**Spec:** `docs/superpowers/specs/2026-10-09-n-independent-caches-design.md`

## Global Constraints

- One constant, `MAX_DETECTIONS_PER_FRAME = 1024`, in the new `src/hydra_suite/core/inference/limits.py`. It replaces `MAX_RAW_CANDIDATES_PER_FRAME` (`stages/obb.py`) and `MAX_DOWNSTREAM_CROPS_PER_FRAME` (`stages/filtering.py`). Both old names are deleted, not aliased.
- `EXTRACTION_CONFIDENCE_FLOOR = 0.01` in `limits.py`. It replaces `TRACKER_RAW_OBB_CONFIDENCE_FLOOR` (1e-3) and the 1e-3 default of `OBBDirectConfig.confidence_floor`.
- N > 1024 is an error (`DetectionLimitError`, a `ValueError`) wherever N enters: engine-param build, `filter_with_indices`. No silent clamp anywhere.
- When a frame exceeds the limit, log a `WARNING` naming the frame index and candidate count, and record it in `DetectionLimitStats`. The worker logs a WARNING summary at the end of the run and emits it through `_emit_warning`.
- `CACHE_SCHEMA_VERSION` goes from 5 to 6. No backward compatibility: every existing cache rebuilds once.
- Backward compatibility is NOT required and tracking outputs may change. The equivalence harness is re-baselined (determinism + attribution), not held byte-identical to the old main.
- `OBBConfig.raw_detection_cap` STAYS, as an explicit expert/DetectKit override (DetectKit preview and the AL adapter set it). Tracking never sets it (0), and 0 now means "the 1024 limit", no longer `2 * max_detections`.
- Test command prefix (MPS box): `KMP_DUPLICATE_LIB_OK=TRUE PYTHONPATH=src ~/miniforge3/envs/hydra-mps/bin/python -m pytest ... -p no:cacheprovider`
- Commit as the configured git user with no `Co-Authored-By` trailer. Run `make format` (or black+isort on the touched files) before each commit.
- Work only in `.worktrees/n-free-caches` (branch `feat/n-independent-caches`). Rebase onto main before merging: concurrent SAHI-unify work touches `stages/slicing.py`, `stages/obb.py` and `stages/regions.py`.

## Review Focus

1. **Raising N after a cache was built (e.g. 10 → 200).** The user expects pose, head-tail and CNN values for the extra animals, never NaN/0. Pinned by Task 9 `test_replay_at_larger_n_reuses_superset_without_nan` and Task 8 `test_load_missing_index_raises`.
2. **Ties in confidence at the 2N window boundary or inside NMS.** Expected: the replay set is always a subset of the stored per-animal superset, and repeated replays are identical. Pinned by Task 4 `test_final_set_is_subset_of_superset_with_ties` (property test over random tied confidences).
3. **Changing the confidence threshold (or ROI) without changing N.** Expected: detection is reused and only the per-animal caches recompute. Never stale per-animal values for a different filter set. Pinned by Task 8 `test_downstream_keys_change_with_filters_not_n`.
4. **Multi-arena N (`n_arenas * animals_per_arena`) above 1024**, e.g. 6 arenas × 200. Expected: a loud error at config build naming the limit, not a crash deep in inference. Pinned by Task 1 `test_engine_params_rejects_total_n_above_limit`.
5. **A noisy frame with > 1024 candidates.** Expected: top 1024 by confidence kept, a WARNING with frame index and count, and an end-of-run summary. Pinned by Task 2 `test_rank_and_bound_truncates_and_records` and Task 11 `test_worker_emits_detection_limit_summary`.

---

## File Structure

| File | Responsibility |
|---|---|
| `src/hydra_suite/core/inference/limits.py` (new) | The limit constants, `DetectionLimitError`, `require_target_count_within_limit`, `DetectionLimitStats` |
| `src/hydra_suite/core/inference/downstream_select.py` (new) | Pure helpers: superset → final position mapping, select/concat per stage result, CNN/AprilTag raw-index ⇄ position conversion, `DownstreamCacheError` |
| `src/hydra_suite/core/inference/stages/filtering.py` | Replay: 2N window, consistent ranking, final cap (no clamp), bgsub N rules |
| `src/hydra_suite/core/inference/stages/obb.py` | N-free extraction cap, `rank_and_bound` |
| `src/hydra_suite/core/inference/config.py` | Builder stops setting caps from N, uses the floor |
| `src/hydra_suite/core/inference/cache/keys.py`, `cache/base.py` | N removed from keys; replay-filter hash for downstream keys; schema 6 |
| `src/hydra_suite/core/inference/stages/bgsub.py`, `core/background/measure.py` | bgsub extraction without N gates |
| `src/hydra_suite/core/inference/pipeline.py` | Batch: superset downstream, chunking, raw-index writes, narrowed in-memory results, limit stats |
| `src/hydra_suite/core/inference/runner.py` | Realtime superset + narrowing; replay loaders by raw index; identity sidecar via loaders; downstream keys with filter hash |
| `src/hydra_suite/trackerkit/engine_params.py`, `trackerkit/gui/panels/setup_panel.py` | N validation; spinbox max 1024 |
| `src/hydra_suite/trackerkit/tracking_cache.py`, `trackerkit/gui/orchestrators/tracking.py` | N out of props-cache model ids; delete dead duplicate builder |
| `src/hydra_suite/core/tracking/worker.py` | End-of-run limit summary |
| `docs/user-guide/detection-limits.md` (new) + `mkdocs.yml` nav | User-facing statement of the limit |
| `tools/equivalence/cache_stats.py` (new) | Cost measurement: per-frame stored detections and per-animal rows |

---

### Task 1: Limits module, loud N validation, remove the silent 128 clamp

**Files:**
- Create: `src/hydra_suite/core/inference/limits.py`
- Modify: `src/hydra_suite/core/inference/stages/filtering.py:23-34` (constant + `_effective_max_detections`), `:97`, `:204`, `:357-361`, `:396-404`
- Modify: `src/hydra_suite/trackerkit/engine_params.py:691` (after `max_targets = n_arenas * animals_per_arena`)
- Modify: `src/hydra_suite/trackerkit/gui/panels/setup_panel.py:221`
- Test: `tests/test_detection_limits.py` (new); update `tests/test_inference_stages_filtering.py:9,166-196`, `tests/test_final_cap_keeps_most_confident.py:140-158`

**Interfaces:**
- Produces: `limits.MAX_DETECTIONS_PER_FRAME: int = 1024`, `limits.EXTRACTION_CONFIDENCE_FLOOR: float = 0.01`, `limits.DOWNSTREAM_CHUNK_SIZE: int = 256`, `class DetectionLimitError(ValueError)`, `require_target_count_within_limit(n: int) -> int`, `@dataclass DetectionLimitStats(frames: list[tuple[int, int]])` with `record(frame_idx: int, candidate_count: int) -> None` and `summary() -> str | None`; `filtering._final_cap(config) -> int`.

- [ ] **Step 1: Write the failing tests** (`tests/test_detection_limits.py`)

```python
import numpy as np
import pytest

from hydra_suite.core.inference.config import OBBConfig, OBBDirectConfig
from hydra_suite.core.inference.limits import (
    MAX_DETECTIONS_PER_FRAME,
    DetectionLimitError,
    DetectionLimitStats,
    require_target_count_within_limit,
)
from hydra_suite.core.inference.result import OBBResult
from hydra_suite.core.inference.stages.filtering import filter_with_indices


def _spread(n: int, conf: float = 0.9) -> OBBResult:
    xs = np.arange(n, dtype=np.float32) * 40.0
    centroids = np.stack([xs, np.zeros(n, np.float32)], axis=1)
    corners = np.stack(
        [centroids + d for d in ([-5, -5], [5, -5], [5, 5], [-5, 5])], axis=1
    ).astype(np.float32)
    return OBBResult(
        frame_idx=0,
        centroids=centroids,
        angles=np.zeros(n, np.float32),
        sizes=np.full(n, 100.0, np.float32),
        shapes=np.tile(np.array([[100.0, 1.0]], np.float32), (n, 1)),
        confidences=np.full(n, conf, np.float32),
        corners=corners,
        detection_ids=OBBResult.make_detection_ids(0, n),
    )


def _cfg(max_detections: int) -> OBBConfig:
    return OBBConfig(
        mode="direct",
        direct=OBBDirectConfig(model_path="/m.pt"),
        max_detections=max_detections,
        confidence_threshold=0.1,
        iou_threshold=1.0,
    )


def test_limit_is_1024():
    assert MAX_DETECTIONS_PER_FRAME == 1024


def test_require_target_count_rejects_above_limit():
    assert require_target_count_within_limit(1024) == 1024
    with pytest.raises(DetectionLimitError, match="1024"):
        require_target_count_within_limit(1025)


def test_n_200_keeps_200_detections_regression_for_128_clamp():
    _, idx = filter_with_indices(_spread(300), _cfg(200))
    assert len(idx) == 200


def test_filter_rejects_n_above_limit():
    with pytest.raises(DetectionLimitError):
        filter_with_indices(_spread(3), _cfg(MAX_DETECTIONS_PER_FRAME + 1))


def test_stats_summary_counts_frames():
    stats = DetectionLimitStats()
    assert stats.summary() is None
    stats.record(7, 1500)
    stats.record(9, 2048)
    msg = stats.summary()
    assert "2 frame(s)" in msg and "1024" in msg and "7" in msg
```

Add to `tests/test_engine_params_extraction.py`:

```python
def test_engine_params_rejects_total_n_above_limit(fly_obb_cfg, fly_obb_probe):
    from hydra_suite.core.inference.limits import DetectionLimitError

    cfg = dict(fly_obb_cfg)
    cfg["max_targets"] = 1025
    cfg["animals_per_arena"] = 1025
    rt = RuntimeContext(
        fps=fly_obb_probe.fps,
        total_frames=fly_obb_probe.total_frames,
        frame_width=fly_obb_probe.width,
        frame_height=fly_obb_probe.height,
    )
    with pytest.raises(DetectionLimitError, match="1024"):
        build_engine_params(cfg, runtime=rt)
```

- [ ] **Step 2: Run the tests and confirm they fail**

Run: `... -m pytest tests/test_detection_limits.py tests/test_engine_params_extraction.py -q -p no:cacheprovider`
Expected: FAIL with `ModuleNotFoundError: hydra_suite.core.inference.limits`.

- [ ] **Step 3: Create `limits.py`**

```python
"""The one declared per-frame limit of the inference pipeline.

``MAX_DETECTIONS_PER_FRAME`` bounds (1) detections stored per frame in the
detection cache, (2) per-animal analyses per frame and (3) the number of
animals N itself. It is a LOUD limit: N above it is rejected, and a frame
exceeding it is truncated with a WARNING plus an end-of-run summary.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

MAX_DETECTIONS_PER_FRAME = 1024
# Raw OBB extraction keeps every candidate at or above this confidence. It is
# N-independent and also the lowest confidence threshold replay can use
# without re-running inference.
EXTRACTION_CONFIDENCE_FLOOR = 0.01
# Per-animal crops are materialised and run in chunks of this many detections
# so a 1024-detection frame does not spike memory.
DOWNSTREAM_CHUNK_SIZE = 256


class DetectionLimitError(ValueError):
    """N (number of animals) exceeds MAX_DETECTIONS_PER_FRAME."""


def require_target_count_within_limit(n: int) -> int:
    n = int(n)
    if n > MAX_DETECTIONS_PER_FRAME:
        raise DetectionLimitError(
            f"Number of animals N={n} exceeds the hard limit of "
            f"{MAX_DETECTIONS_PER_FRAME} detections per frame. Reduce the "
            "number of animals (or arenas x animals per arena)."
        )
    return n


@dataclass
class DetectionLimitStats:
    """Run-scoped record of frames that hit MAX_DETECTIONS_PER_FRAME."""

    frames: list[tuple[int, int]] = field(default_factory=list)

    def record(self, frame_idx: int, candidate_count: int) -> None:
        self.frames.append((int(frame_idx), int(candidate_count)))
        logger.warning(
            "Frame %d produced %d detection candidates; the hard limit is %d "
            "per frame -- kept the top %d by confidence, dropped the rest.",
            frame_idx,
            candidate_count,
            MAX_DETECTIONS_PER_FRAME,
            MAX_DETECTIONS_PER_FRAME,
        )

    def summary(self) -> str | None:
        if not self.frames:
            return None
        worst = max(c for _, c in self.frames)
        first = ", ".join(str(f) for f, _ in self.frames[:10])
        more = "" if len(self.frames) <= 10 else f" (+{len(self.frames) - 10} more)"
        return (
            f"{len(self.frames)} frame(s) hit the {MAX_DETECTIONS_PER_FRAME}-"
            f"detection-per-frame limit (worst: {worst} candidates); frames: "
            f"{first}{more}. Detections beyond the limit were dropped."
        )
```

- [ ] **Step 4: Replace the clamp in `filtering.py`**

Delete `MAX_DOWNSTREAM_CROPS_PER_FRAME` and `_effective_max_detections`, and add:

```python
from ..limits import MAX_DETECTIONS_PER_FRAME, require_target_count_within_limit


def _final_cap(config: OBBConfig) -> int:
    """The replay-time final cap N. 0/unset means 'no N cut' (the limit)."""
    requested = int(getattr(config, "max_detections", 0) or 0)
    if requested <= 0:
        return MAX_DETECTIONS_PER_FRAME
    return require_target_count_within_limit(requested)
```

Replace `_effective_max_detections(config)` with `_final_cap(config)` at all three call sites (`filter_detections`, `filter_from_tensors`, `filter_with_indices`). In `filter_with_indices`, change the `else MAX_DOWNSTREAM_CROPS_PER_FRAME` branch to `else MAX_DETECTIONS_PER_FRAME`. In the `filter_for_source` bgsub branch, replace both uses of `MAX_DOWNSTREAM_CROPS_PER_FRAME` with `MAX_DETECTIONS_PER_FRAME` (Task 3 rewrites this branch).

- [ ] **Step 5: Validate N in `engine_params.py`**

Immediately after `max_targets = n_arenas * animals_per_arena`:

```python
    from hydra_suite.core.inference.limits import require_target_count_within_limit

    require_target_count_within_limit(max_targets)
```

- [ ] **Step 6: Raise the GUI spinbox maximum** (`setup_panel.py:221`)

```python
        from hydra_suite.core.inference.limits import MAX_DETECTIONS_PER_FRAME

        self.spin_max_targets.setRange(1, MAX_DETECTIONS_PER_FRAME)
```

- [ ] **Step 7: Update the tests that pinned 128**

In `tests/test_inference_stages_filtering.py`, change the import to `from hydra_suite.core.inference.limits import MAX_DETECTIONS_PER_FRAME`, and in the two tests at lines 166-196 replace `MAX_DOWNSTREAM_CROPS_PER_FRAME` with `MAX_DETECTIONS_PER_FRAME`. The behaviour is the same, only the constant changed. Do the same in `tests/test_final_cap_keeps_most_confident.py:140-158`. Grep for stragglers:
`grep -rn "MAX_DOWNSTREAM_CROPS_PER_FRAME" src tests`. Expected: only `src/hydra_suite/detectkit/jobs/direct_calibration.py:1058`, a comment. Reword it to name `MAX_DETECTIONS_PER_FRAME`.

- [ ] **Step 8: Run the tests**

Run: `... -m pytest tests/test_detection_limits.py tests/test_engine_params_extraction.py tests/test_inference_stages_filtering.py tests/test_final_cap_keeps_most_confident.py -q -p no:cacheprovider`
Expected: PASS.

- [ ] **Step 9: Commit**

```bash
git add src/hydra_suite/core/inference/limits.py src/hydra_suite/core/inference/stages/filtering.py src/hydra_suite/trackerkit/engine_params.py src/hydra_suite/trackerkit/gui/panels/setup_panel.py src/hydra_suite/detectkit/jobs/direct_calibration.py tests/
git commit -m "feat(inference): one loud 1024 per-frame limit; N>128 no longer silently clamped"
```

---

### Task 2: N-free YOLO extraction, confidence floor, `rank_and_bound`, key + schema 6

**Files:**
- Modify: `src/hydra_suite/core/inference/stages/obb.py:31-46` (constant, `effective_raw_detection_cap`), add `rank_and_bound` next to `_apply_raw_detection_cap` (`:1563`)
- Modify: `src/hydra_suite/core/inference/stages/regions.py` (imports of `MAX_RAW_CANDIDATES_PER_FRAME` at `:536`, `:923`; messages at `:560-566`, `:991-997`)
- Modify: `src/hydra_suite/core/inference/config.py:45` (`TRACKER_RAW_OBB_CONFIDENCE_FLOOR`), `:238` (`confidence_floor` default), `:1033-1035` and `:1147`, `:1192`, `:1210` (builder)
- Modify: `src/hydra_suite/core/inference/cache/keys.py:135-206`, `cache/base.py:24`
- Modify: `src/hydra_suite/core/inference/pipeline.py:352-371` (materialize + cache write), constructor `:180-195`
- Modify: `src/hydra_suite/core/inference/runner.py:1146-1151`, `:1555`, `:1025`, `:1668`
- Test: `tests/test_n_free_extraction.py` (new); update `tests/test_inference_config_from_params.py:24-26`, `tests/test_inference_cache_keys.py:129-137`, `tests/test_inference_stages_obb.py:372-385`, `tests/test_region_source.py:785`

**Interfaces:**
- Consumes: `limits.MAX_DETECTIONS_PER_FRAME`, `limits.EXTRACTION_CONFIDENCE_FLOOR`, `limits.DetectionLimitStats` (Task 1).
- Produces: `obb.effective_raw_detection_cap(config) -> int` (explicit `raw_detection_cap` if > 0, clamped to the limit; else the limit); `obb.rank_and_bound(r: OBBResult) -> tuple[OBBResult, int]` (confidence-ranked, ≤ limit rows; returns the pre-bound count); `Pipeline.detection_limit_stats`, `InferenceRunner.detection_limit_stats: DetectionLimitStats`.

- [ ] **Step 1: Write the failing tests** (`tests/test_n_free_extraction.py`)

```python
from types import SimpleNamespace

import numpy as np

from hydra_suite.core.inference.cache.base import CACHE_SCHEMA_VERSION
from hydra_suite.core.inference.cache.keys import detection_cache_key
from hydra_suite.core.inference.config import (
    OBBConfig,
    OBBDirectConfig,
    build_inference_config_from_params,
)
from hydra_suite.core.inference.limits import (
    EXTRACTION_CONFIDENCE_FLOOR,
    MAX_DETECTIONS_PER_FRAME,
    DetectionLimitStats,
)
from hydra_suite.core.inference.result import OBBResult
from hydra_suite.core.inference.stages.obb import (
    effective_raw_detection_cap,
    rank_and_bound,
)


def _params(n):
    return {
        "DETECTION_METHOD": "yolo_obb",
        "YOLO_OBB_MODE": "direct",
        "YOLO_OBB_DIRECT_MODEL_PATH": "some.pt",
        "COMPUTE_RUNTIME": "cpu",
        "MAX_TARGETS": n,
    }


def _obb(confs):
    n = len(confs)
    return OBBResult(
        frame_idx=3,
        centroids=np.zeros((n, 2), np.float32),
        angles=np.zeros(n, np.float32),
        sizes=np.ones(n, np.float32),
        shapes=np.ones((n, 2), np.float32),
        confidences=np.asarray(confs, np.float32),
        corners=np.zeros((n, 4, 2), np.float32),
        detection_ids=OBBResult.make_detection_ids(3, n),
    )


def test_schema_is_v6():
    assert CACHE_SCHEMA_VERSION == 6


def test_extraction_cap_ignores_n():
    for n in (1, 10, 500):
        cfg = build_inference_config_from_params(_params(n))
        assert cfg.obb.raw_detection_cap == 0
        # Extraction collects one past the limit so a real truncation is
        # observable (rank_and_bound then cuts to the limit and records it).
        assert effective_raw_detection_cap(cfg.obb) == MAX_DETECTIONS_PER_FRAME + 1
        assert cfg.obb.max_detections == n


def test_explicit_raw_cap_still_honoured_and_bounded():
    assert effective_raw_detection_cap(SimpleNamespace(raw_detection_cap=7)) == 7
    assert (
        effective_raw_detection_cap(SimpleNamespace(raw_detection_cap=5000))
        == MAX_DETECTIONS_PER_FRAME
    )


def test_builder_uses_extraction_floor():
    cfg = build_inference_config_from_params(_params(8))
    assert cfg.obb.direct.confidence_floor == EXTRACTION_CONFIDENCE_FLOOR


def test_detection_key_independent_of_n():
    k5 = detection_cache_key(build_inference_config_from_params(_params(5)).obb)
    k50 = detection_cache_key(build_inference_config_from_params(_params(50)).obb)
    assert k5 == k50


def test_rank_and_bound_sorts_by_confidence():
    out, count = rank_and_bound(_obb([0.2, 0.9, 0.5]))
    assert count == 3
    assert out.confidences.tolist() == [np.float32(0.9), np.float32(0.5), np.float32(0.2)]


def test_rank_and_bound_truncates_and_records():
    confs = np.linspace(0.02, 0.99, MAX_DETECTIONS_PER_FRAME + 5)
    out, count = rank_and_bound(_obb(confs))
    assert count == MAX_DETECTIONS_PER_FRAME + 5
    assert out.num_detections == MAX_DETECTIONS_PER_FRAME
    assert out.confidences.min() > np.float32(0.02)
    stats = DetectionLimitStats()
    stats.record(3, count)
    assert "1 frame(s)" in stats.summary()
```

- [ ] **Step 2: Run the tests and confirm they fail**

Run: `... -m pytest tests/test_n_free_extraction.py -q -p no:cacheprovider`
Expected: FAIL (`ImportError: rank_and_bound`, schema 5).

- [ ] **Step 3: Make the cap N-free in `obb.py`**

Replace lines 31-46 with:

```python
from ..limits import MAX_DETECTIONS_PER_FRAME


def effective_raw_detection_cap(config: Any) -> int:
    """Per-frame extraction cap. N-independent.

    An explicit positive ``raw_detection_cap`` (DetectKit preview, AL export)
    is honoured up to the limit; 0 -- what tracking always uses -- means the
    limit itself. N (``max_detections``) is deliberately NOT consulted: it is
    applied at replay (``filtering.filter_with_indices``).

    With no explicit cap, extraction collects ``MAX_DETECTIONS_PER_FRAME + 1``
    candidates (model ``max_det`` included): the one extra row is a probe, so
    ``rank_and_bound`` can tell a genuinely truncated frame (> limit) from one
    with exactly the limit, and record it LOUDLY.
    """
    requested = int(getattr(config, "raw_detection_cap", 0) or 0)
    if requested <= 0:
        return MAX_DETECTIONS_PER_FRAME + 1
    return min(requested, MAX_DETECTIONS_PER_FRAME)
```

Add after `_apply_raw_detection_cap`:

```python
def rank_and_bound(r: OBBResult) -> tuple[OBBResult, int]:
    """Confidence-rank a frame and bound it to MAX_DETECTIONS_PER_FRAME.

    Every frame written to the detection cache passes through here, so cached
    frames are always confidence-sorted (descending) with ids in rank order --
    the invariant the replay window (``filtering``) relies on. Returns the
    bounded result and the pre-bound candidate count so the caller can record
    a limit hit.
    """
    count = int(r.num_detections)
    if count == 0:
        return r, 0
    return _apply_raw_detection_cap(r, MAX_DETECTIONS_PER_FRAME), count
```

Replace every `MAX_RAW_CANDIDATES_PER_FRAME` in `obb.py` and `regions.py` with `MAX_DETECTIONS_PER_FRAME` (imported from `..limits`). In the two `regions.py` error messages, keep the wording but name the limit: `f"the hard {MAX_DETECTIONS_PER_FRAME}-detection-per-frame limit; "`.

- [ ] **Step 4: Builder and floor in `config.py`**

```python
from .limits import EXTRACTION_CONFIDENCE_FLOOR

# Kept as the name sequential configs and keys import; one value.
TRACKER_RAW_OBB_CONFIDENCE_FLOOR = EXTRACTION_CONFIDENCE_FLOOR
```

Set `OBBDirectConfig.confidence_floor: float = EXTRACTION_CONFIDENCE_FLOOR`. In `build_inference_config_from_params`, replace the comment block and lines 1033-1035 with:

```python
    # N is a replay-time knob: extraction keeps every candidate >= the
    # extraction floor (bounded by MAX_DETECTIONS_PER_FRAME) and replay
    # applies the 2N window and the final N cut (filtering.filter_with_indices).
    max_dets = require_target_count_within_limit(
        max(1, int(params.get("MAX_TARGETS", 8)))
    )
```

Import `require_target_count_within_limit` from `.limits`. Delete `raw_cap` and remove `raw_detection_cap=raw_cap,` from both `OBBConfig(...)` constructions (`:1147`, `:1210`). Replace `confidence_floor=1e-3` at `:1192` with `confidence_floor=EXTRACTION_CONFIDENCE_FLOOR`. Update the `OBBConfig.raw_detection_cap` comment (`:314-320`) to: "Explicit per-frame extraction cap for non-tracking callers (DetectKit preview, AL). 0 = MAX_DETECTIONS_PER_FRAME. Tracking never sets it; N is applied at replay."

- [ ] **Step 5: Keys and schema**

`cache/base.py:24`: `CACHE_SCHEMA_VERSION = 6`. In `keys.py`, remove `config.max_detections,` from both `_direct_raw_config_hash` and `_sequential_config_hash`, and bump the tags `"direct-raw-v4"` → `"direct-raw-v5"` and `"sequential-raw-v4"` → `"sequential-raw-v5"`. Add the limit after `config.raw_detection_cap,` in both payloads:

```python
        MAX_DETECTIONS_PER_FRAME,
```

(import from `..limits`). Update docstring line "Classes, raw cap, ..." to "Classes, the explicit raw cap, the extraction limit, ...; N is NOT part of the key".

- [ ] **Step 6: Bound + record at every cache write**

`pipeline.py` constructor (next to `clipping_stats`): add `detection_limit_stats: "DetectionLimitStats | None" = None` and `self.detection_limit_stats = detection_limit_stats if detection_limit_stats is not None else DetectionLimitStats()`. At `:352-371`:

```python
                obb_result = (
                    materialize_tensors(raw, effective_raw_detection_cap(cfg.obb))
                    if isinstance(raw, _RawOBBTensors)
                    else raw
                )
                if cfg.detection_source == "obb":
                    obb_result, candidate_count = rank_and_bound(obb_result)
                    if candidate_count > MAX_DETECTIONS_PER_FRAME:
                        self.detection_limit_stats.record(frame_idx, candidate_count)
```

Note: because `effective_raw_detection_cap` returns `MAX_DETECTIONS_PER_FRAME + 1` (the probe row), every upstream bound -- Ultralytics `max_det`, tile reservoirs, merge caps, `materialize_tensors` -- keeps at most limit+1 rows. `rank_and_bound` therefore sees `count == limit + 1` exactly when something was cut, and `count > MAX_DETECTIONS_PER_FRAME` is a precise hit test. Do NOT record frames with exactly 1024 detections. Add a test: a frame with exactly `MAX_DETECTIONS_PER_FRAME` candidates is stored whole and NOT recorded; one with `MAX_DETECTIONS_PER_FRAME + 1` is cut to the limit and recorded.

`runner.py:1025`: `self.detection_limit_stats = DetectionLimitStats()`; pass `detection_limit_stats=self.detection_limit_stats` where `clipping_stats=self.clipping_stats` is passed (`:1668`). In `run_realtime` (`:1146-1151`) and `detect_batch_raw` (`:1555`), change `materialize_tensors(raw, self.config.obb.raw_detection_cap)` to `materialize_tensors(raw, effective_raw_detection_cap(self.config.obb))`, then apply `rank_and_bound` + record, exactly as in the pipeline, before the cache write.

- [ ] **Step 7: Update the tests that pinned 2N / v5**

- `tests/test_inference_config_from_params.py:24-26`: replace the comment and assert with `assert cfg.obb.max_detections == 8` and `assert cfg.obb.raw_detection_cap == 0  # N is replay-only`.
- `tests/test_inference_cache_keys.py:129-137`: rename to `test_cache_schema_version_is_v6_n_independent_bump` and assert `CACHE_SCHEMA_VERSION == 6`. The `raw_detection_cap = 17` assertion at `:256` stays valid.
- `tests/test_inference_stages_obb.py:384-385` and `tests/test_region_source.py:785`: replace `MAX_RAW_CANDIDATES_PER_FRAME` with `MAX_DETECTIONS_PER_FRAME` (import from `hydra_suite.core.inference.limits`).
- Run the touched families and fix any assertion that encoded `raw_detection_cap=0 ⇒ 2*max_detections`. The expected value becomes `MAX_DETECTIONS_PER_FRAME`. Run: `... -m pytest tests/test_region_source.py tests/test_inference_stages_obb.py tests/test_inference_slicing.py tests/test_obb_cuda_tensor_frames.py tests/test_raw_universe_native_polygons.py tests/test_nvdec_gpu_fast_tier.py tests/test_al_inference_adapter.py tests/test_dataset_generation.py tests/test_direct_calibration_sweep.py -q -p no:cacheprovider`. For each failure, confirm the old expectation was `2*max_detections` before changing it. Any other failure is a real regression: stop and report it.

- [ ] **Step 8: Run all touched tests**

Run: `... -m pytest tests/test_n_free_extraction.py tests/test_inference_cache_keys.py tests/test_inference_config_from_params.py tests/test_inference_runner_batch.py tests/test_inference_runner_rt.py -q -p no:cacheprovider`
Expected: PASS.

- [ ] **Step 9: Commit**

```bash
git commit -am "feat(inference): N-free YOLO extraction (floor 0.01, 1024 limit), N out of detection key, schema v6"
```

---

### Task 3: Background subtraction without N at extraction

**Files:**
- Modify: `src/hydra_suite/core/background/measure.py:189-290` (`detect_objects`)
- Modify: `src/hydra_suite/core/inference/stages/bgsub.py:186-193` (`run_bgsub`)
- Modify: `src/hydra_suite/core/inference/stages/filtering.py` (`filter_for_source` bgsub branch)
- Modify: `src/hydra_suite/core/inference/cache/keys.py:256-289` (`_BGSUB_KEY_PARAMS`)
- Test: `tests/test_bgsub_n_free.py` (new); update `tests/test_bgsub_cache_keys.py:158`, `tests/test_inference_stages_filtering.py` and `tests/test_final_cap_keeps_most_confident.py` bgsub tests

**Interfaces:**
- Consumes: `MAX_DETECTIONS_PER_FRAME`, `DetectionLimitStats` (Task 1).
- Produces: `BackgroundMeasurer.detect_objects(fg_mask, frame_count, return_contours=False, *, apply_target_gates: bool = True)`. `run_bgsub` calls it with `apply_target_gates=False` and returns detections sorted by area descending (stable), bounded at the limit. `filter_for_source(config, raw, roi, apply_max_detections=True)` on bgsub applies, when `config.bgsub` is present and `apply_max_detections`: (a) empty result if `raw.num_detections > max_targets * max_contour_multiplier`; (b) top `max_targets` by area (stable).

- [ ] **Step 1: Write the failing tests** (`tests/test_bgsub_n_free.py`)

```python
import numpy as np

from hydra_suite.core.inference.cache.keys import bgsub_detection_cache_key
from hydra_suite.core.inference.config import BgSubConfig, InferenceConfig
from hydra_suite.core.inference.result import OBBResult
from hydra_suite.core.inference.stages.filtering import filter_for_source


def _bg(sizes):
    n = len(sizes)
    return OBBResult(
        frame_idx=0,
        centroids=np.zeros((n, 2), np.float32),
        angles=np.zeros(n, np.float32),
        sizes=np.asarray(sizes, np.float32),
        shapes=np.ones((n, 2), np.float32),
        confidences=np.full(n, np.nan, np.float32),
        corners=np.zeros((n, 4, 2), np.float32),
        detection_ids=OBBResult.make_detection_ids(0, n),
    )


def _cfg(n, mult=20):
    return InferenceConfig(bgsub=BgSubConfig(max_targets=n, max_contour_multiplier=mult))


def test_bgsub_key_ignores_n_and_multiplier():
    a = BgSubConfig(params={"MAX_TARGETS": 5, "MAX_CONTOUR_MULTIPLIER": 20})
    b = BgSubConfig(params={"MAX_TARGETS": 50, "MAX_CONTOUR_MULTIPLIER": 3})
    assert bgsub_detection_cache_key(a) == bgsub_detection_cache_key(b)


def test_bgsub_replay_keeps_top_n_by_area():
    out, idx = filter_for_source(_cfg(2), _bg([10.0, 30.0, 20.0]))
    assert sorted(out.sizes.tolist()) == [20.0, 30.0]
    assert sorted(idx.tolist()) == [1, 2]


def test_bgsub_replay_skips_frame_over_contour_budget():
    out, idx = filter_for_source(_cfg(1, mult=2), _bg([1.0, 2.0, 3.0]))
    assert out.num_detections == 0 and len(idx) == 0


def test_bgsub_superset_ignores_n():
    out, _ = filter_for_source(
        _cfg(1, mult=2), _bg([1.0, 2.0, 3.0]), apply_max_detections=False
    )
    assert out.num_detections == 3
```

Check that `BgSubConfig` accepts a `params` field before writing the first test: `grep -n "params" src/hydra_suite/core/inference/config.py | sed -n 1,20p` around the class at `:380`. If the field name differs, use the real one; the key reads `config.params`.

- [ ] **Step 2: Run the tests and confirm they fail**

Run: `... -m pytest tests/test_bgsub_n_free.py -q -p no:cacheprovider`
Expected: FAIL (key differs; filter ignores N).

- [ ] **Step 3: `detect_objects` gates become optional**

In `measure.py`, add the keyword-only `apply_target_gates: bool = True`. Wrap the contour-budget skip (`if len(cnts) > max_allowed_contours:`) and the top-N truncation (`if len(meas) > N:`) in `if apply_target_gates:`. Default `True` keeps `core/background/optimizer.py:494` (the parameter optimizer) unchanged.

- [ ] **Step 4: `run_bgsub` stores the N-free set**

At `bgsub.py:186-193`, pass `apply_target_gates=False` to both `detect_objects` calls. After `if not meas: return _empty_result(frame_idx)`, rank by area and bound:

```python
    order = np.argsort(-np.asarray(sizes, dtype=np.float64), kind="stable")
    if len(order) > MAX_DETECTIONS_PER_FRAME:
        logger.warning(
            "Frame %d: %d background-subtraction contours exceed the hard limit "
            "of %d per frame; keeping the %d largest.",
            frame_idx, len(order), MAX_DETECTIONS_PER_FRAME, MAX_DETECTIONS_PER_FRAME,
        )
        order = order[:MAX_DETECTIONS_PER_FRAME]
    meas = [meas[i] for i in order]
    sizes = [sizes[i] for i in order]
    shapes = [shapes[i] for i in order]
    confidences = [confidences[i] for i in order]
    if contours is not None:
        contours = [contours[i] for i in order]
```

(Add `import logging` / `logger = logging.getLogger(__name__)` if `bgsub.py` lacks it, and import `MAX_DETECTIONS_PER_FRAME` from `..limits`.)

- [ ] **Step 5: N rules at replay** (`filter_for_source` bgsub branch)

```python
    if config.detection_source == "bgsub":
        bg = getattr(config, "bgsub", None)
        n = raw.num_detections
        order = np.argsort(-np.asarray(raw.sizes, np.float64), kind="stable")
        cap = MAX_DETECTIONS_PER_FRAME
        if apply_max_detections and bg is not None:
            target = require_target_count_within_limit(max(1, int(bg.max_targets)))
            budget = target * int(bg.max_contour_multiplier)
            if n > budget:
                # Contour-budget noise guard, now on the stored (area- and
                # size-filtered) contour count rather than the raw findContours
                # count -- see spec section 2.
                order = order[:0]
            cap = target
        indices = np.ascontiguousarray(np.sort(order[:cap]), dtype=np.int32)
        return _select(raw, indices), indices
```

Indices are re-sorted ascending so downstream raw-index keying stays monotone.

- [ ] **Step 6: Key** — remove `"MAX_TARGETS"` and `"MAX_CONTOUR_MULTIPLIER"` from `_BGSUB_KEY_PARAMS`.

- [ ] **Step 7: Update the pinned tests**

`tests/test_bgsub_cache_keys.py:158`: move `("MAX_TARGETS", 20, 10)` out of the "changes key" parametrization into a new test asserting the key is UNCHANGED for `MAX_TARGETS` and `MAX_CONTOUR_MULTIPLIER`. The two existing bgsub filter tests (`test_inference_stages_filtering.py::test_bgsub_source_is_capped_before_downstream_crop_materialization`, `test_final_cap_keeps_most_confident.py::test_bgsub_still_uses_size_because_confidences_are_nan`) use a config without `.bgsub`. They now exercise the limit path only, and their assertions hold with `MAX_DETECTIONS_PER_FRAME` (Task 1 already renamed the constant). Run `tests/test_bgsub_stage.py`. Its realtime tests that expect N-gating now go through `run_bgsub` with gates off. Any test that calls `run_bgsub` directly and asserts ≤ N must instead assert the filtered output via `filter_for_source`. Change only those assertions.

- [ ] **Step 8: Run**

Run: `... -m pytest tests/test_bgsub_n_free.py tests/test_bgsub_cache_keys.py tests/test_bgsub_stage.py tests/test_bgsub_contours.py tests/test_inference_stages_filtering.py tests/test_final_cap_keeps_most_confident.py -q -p no:cacheprovider`
Expected: PASS.

- [ ] **Step 9: Commit** — `git commit -am "feat(bgsub): store N-free contours; MAX_TARGETS gates move to replay"`

---

### Task 4: Replay window and one consistent ranking

**Files:**
- Modify: `src/hydra_suite/core/inference/stages/filtering.py` (`filter_with_indices`, `_obb_nms`)
- Test: `tests/test_replay_window.py` (new)

**Interfaces:**
- Produces: `filter_with_indices(raw, config, roi_mask=None, *, apply_max_detections=True)`. With `apply_max_detections=True`: window = first `min(2*N, n)` cached rows → filters → NMS → top-N. With `False`: no window, no N cut (the per-animal superset). Rank order everywhere is `(confidence desc, raw index asc)` via `_rank(conf, positions)`.

- [ ] **Step 1: Write the failing tests**

```python
import numpy as np
import pytest

from hydra_suite.core.inference.config import OBBConfig, OBBDirectConfig
from hydra_suite.core.inference.result import OBBResult
from hydra_suite.core.inference.stages.filtering import filter_with_indices


def _ranked(confs, xs=None, half=5.0):
    confs = np.asarray(confs, np.float32)
    order = np.lexsort((np.arange(len(confs)), -confs))  # cache invariant
    confs = confs[order]
    n = len(confs)
    xs = np.arange(n, dtype=np.float32) * 40.0 if xs is None else np.asarray(xs, np.float32)[order]
    c = np.stack([xs, np.zeros(n, np.float32)], 1)
    corners = np.stack([c + d for d in ([-half, -half], [half, -half], [half, half], [-half, half])], 1)
    return OBBResult(
        frame_idx=0, centroids=c, angles=np.zeros(n, np.float32),
        sizes=np.full(n, 4 * half * half, np.float32),
        shapes=np.tile(np.array([[1.0, 1.0]], np.float32), (n, 1)),
        confidences=confs, corners=corners.astype(np.float32),
        detection_ids=OBBResult.make_detection_ids(0, n),
    )


def _cfg(n, conf=0.0, iou=0.5):
    return OBBConfig(mode="direct", direct=OBBDirectConfig(model_path="/m.pt"),
                     max_detections=n, confidence_threshold=conf, iou_threshold=iou)


def test_window_is_prefix_of_2n():
    raw = _ranked([0.9, 0.8, 0.7, 0.6, 0.5, 0.4])
    # conf threshold drops nothing; N=1 -> window = rows 0..1 only
    _, idx = filter_with_indices(raw, _cfg(1, iou=1.0))
    assert idx.tolist() == [0]
    # a later row can never enter, even when earlier rows are filtered
    _, idx = filter_with_indices(raw, _cfg(1, conf=0.85, iou=1.0))
    assert idx.tolist() == [0]
    _, idx = filter_with_indices(raw, _cfg(1, conf=0.95, iou=1.0))
    assert idx.tolist() == []


def test_superset_has_no_window_or_n_cut():
    raw = _ranked([0.9, 0.8, 0.7, 0.6])
    _, idx = filter_with_indices(raw, _cfg(1, iou=1.0), apply_max_detections=False)
    assert idx.tolist() == [0, 1, 2, 3]


@pytest.mark.parametrize("seed", range(25))
def test_final_set_is_subset_of_superset_with_ties(seed):
    rng = np.random.default_rng(seed)
    n = int(rng.integers(5, 60))
    confs = rng.choice([0.3, 0.5, 0.5, 0.7, 0.9], size=n)  # heavy ties
    xs = rng.uniform(0, 300, size=n)  # overlaps -> NMS active
    raw = _ranked(confs, xs, half=8.0)
    _, sup = filter_with_indices(raw, _cfg(1, conf=0.4), apply_max_detections=False)
    for N in (1, 2, 3, 7, 20, 100):
        _, fin = filter_with_indices(raw, _cfg(N, conf=0.4))
        assert set(fin.tolist()) <= set(sup.tolist()), (seed, N)
        again = filter_with_indices(raw, _cfg(N, conf=0.4))[1]
        assert again.tolist() == fin.tolist()
```

- [ ] **Step 2: Run the tests and confirm they fail**

Run: `... -m pytest tests/test_replay_window.py -q -p no:cacheprovider`
Expected: `test_window_is_prefix_of_2n` FAILS (no window today). The subset property may fail on ties.

- [ ] **Step 3: Implement the window and the ranking**

In `filtering.py` add:

```python
def _rank(confidences: np.ndarray, positions: np.ndarray) -> np.ndarray:
    """Order (into the given arrays) by confidence desc, then raw index asc.

    The single replay ranking. Detection caches are stored in exactly this
    order (``obb.rank_and_bound``), so a cache prefix IS a top-k under this
    ranking -- which makes the 2N window, NMS and the final cut agree for
    every N and keeps every replay set inside the per-animal superset.
    """
    return np.lexsort((np.asarray(positions), -np.asarray(confidences)))
```

In `_obb_nms`, replace `order = indices[np.argsort(raw.confidences[indices])[::-1]]` with `order = indices[_rank(raw.confidences[indices], indices)]`.

In `filter_with_indices`, right after `n = raw.num_detections` / the early return:

```python
    final_cap = _final_cap(config) if apply_max_detections else MAX_DETECTIONS_PER_FRAME
    keep = raw.confidences >= config.confidence_threshold
    if apply_max_detections:
        # 2N replay window: the cache is confidence-ranked, so the first
        # min(2N, n) rows are exactly the legacy "raw cap" candidates.
        keep[min(n, 2 * final_cap):] = False
```

(Replace the existing `keep = raw.confidences >= ...` line.) The NMS call stays `_obb_nms(subset, np.arange(len(indices)), ...)`. Positions there are monotone in raw index, so `_rank` gives the same order. Replace the final-cut block with:

```python
    if len(indices) > final_cap:
        order = _rank(raw.confidences[indices], indices)[:final_cap]
        indices = indices[np.sort(order)]
        subset = _select(raw, indices)
```

`np.sort(order)` keeps survivors in raw-index order, the same row order as before the cut.

Apply the same `_rank`-based final cut in `filter_detections` and `filter_from_tensors` (replace `_numpy_descending_indices(...)` there with `_rank(..., indices)` / `_rank(..., local_idx)`). Those direct-API paths get NO window, because their input is not cache-ranked. Remove the now-unused `_numpy_descending_indices` import if nothing else uses it.

- [ ] **Step 4: Run** — `... -m pytest tests/test_replay_window.py tests/test_inference_stages_filtering.py tests/test_final_cap_keeps_most_confident.py tests/test_detection_limits.py -q -p no:cacheprovider`. Expected: PASS. If a `test_final_cap_keeps_most_confident` test pins the old "later position first" tie order, update it to raw-index-ascending and note the change in the commit body.

- [ ] **Step 5: Commit** — `git commit -am "feat(replay): 2N window over the ranked cache; one (conf desc, raw idx asc) ranking"`

---

### Task 5: `downstream_select` helpers (pure)

**Files:**
- Create: `src/hydra_suite/core/inference/downstream_select.py`
- Test: `tests/test_downstream_select.py`

**Interfaces:**
- Produces:
  - `class DownstreamCacheError(RuntimeError)`
  - `positions_in(superset_idx: np.ndarray, final_idx: np.ndarray) -> np.ndarray` (positions of `final_idx` inside `superset_idx`; raises `DownstreamCacheError` on a missing index)
  - `select_headtail(ht: HeadTailResult | None, pos) -> HeadTailResult | None`
  - `select_pose(p: PoseResult | None, pos) -> PoseResult | None` (carries a `heading_overrides` attribute if present)
  - `select_cnn(r: CNNResult, pos) -> CNNResult` (renumbers `det_index` 0..len(pos)-1)
  - `select_apriltag(at: AprilTagResult | None, pos) -> AprilTagResult | None`
  - `cnn_positions_to_raw(r: CNNResult, superset_idx) -> CNNResult`
  - `cnn_raw_to_positions(preds: list[CNNDetectionPrediction], final_idx, label) -> CNNResult`
  - `apriltag_positions_to_raw(at, superset_idx)` / `apriltag_raw_to_positions(at, final_idx)`
  - `split_rows(obb: OBBResult, size: int) -> list[tuple[int, OBBResult]]` (offset, chunk)
  - `concat_headtail(parts)`, `concat_pose(parts)`, `concat_cnn(parts: list[tuple[int, CNNResult]])`, `concat_apriltag(parts: list[tuple[int, AprilTagResult | None]])`

- [ ] **Step 1: Write the failing tests**

```python
import numpy as np
import pytest

from hydra_suite.core.inference.downstream_select import (
    DownstreamCacheError,
    apriltag_positions_to_raw,
    apriltag_raw_to_positions,
    cnn_positions_to_raw,
    cnn_raw_to_positions,
    concat_cnn,
    concat_headtail,
    positions_in,
    select_cnn,
    select_headtail,
    select_pose,
    split_rows,
)
from hydra_suite.core.inference.result import (
    AprilTagResult,
    CNNDetectionPrediction,
    CNNResult,
    HeadTailResult,
    OBBResult,
    PoseResult,
)


def _ht(vals):
    v = np.asarray(vals, np.float32)
    return HeadTailResult(v, v / 10, (v > 0).astype(np.uint8), None)


def _cnn(n, label="id"):
    return CNNResult(label, [CNNDetectionPrediction(i, []) for i in range(n)])


def test_positions_in_and_missing():
    sup = np.array([2, 5, 9, 11])
    assert positions_in(sup, np.array([5, 11])).tolist() == [1, 3]
    with pytest.raises(DownstreamCacheError, match="7"):
        positions_in(sup, np.array([7]))


def test_select_headtail_and_pose():
    ht = select_headtail(_ht([1.0, 2.0, 3.0]), np.array([2, 0]))
    assert ht.heading_hints.tolist() == [3.0, 1.0]
    p = PoseResult(np.arange(6, dtype=np.float32).reshape(3, 1, 2), np.array([1, 0, 1], bool))
    p.heading_overrides = np.array([0.1, 0.2, 0.3], np.float32)
    s = select_pose(p, np.array([1]))
    assert s.keypoints.tolist() == [[[2.0, 3.0]]]
    assert s.heading_overrides.tolist() == [pytest.approx(0.2)]


def test_cnn_raw_roundtrip():
    sup = np.array([4, 8, 15])
    raw = cnn_positions_to_raw(_cnn(3), sup)
    assert [p.det_index for p in raw.predictions] == [4, 8, 15]
    back = cnn_raw_to_positions(raw.predictions, np.array([15, 4]), "id")
    assert [p.det_index for p in back.predictions] == [0, 1]
    with pytest.raises(DownstreamCacheError):
        cnn_raw_to_positions(raw.predictions, np.array([16]), "id")


def test_select_cnn_renumbers():
    s = select_cnn(_cnn(4), np.array([3, 1]))
    assert [p.det_index for p in s.predictions] == [0, 1]


def test_apriltag_roundtrip():
    at = AprilTagResult([7, 9], [0, 2], np.zeros((2, 2), np.float32), np.zeros((2, 4, 2), np.float32))
    raw = apriltag_positions_to_raw(at, np.array([10, 20, 30]))
    assert raw.det_indices == [10, 30]
    back = apriltag_raw_to_positions(raw, np.array([30]))
    assert back.tag_ids == [9] and back.det_indices == [0]


def test_split_and_concat():
    n = 5
    obb = OBBResult(0, np.zeros((n, 2), np.float32), np.zeros(n, np.float32),
                    np.ones(n, np.float32), np.ones((n, 2), np.float32),
                    np.ones(n, np.float32), np.zeros((n, 4, 2), np.float32),
                    OBBResult.make_detection_ids(0, n))
    chunks = split_rows(obb, 2)
    assert [(o, c.num_detections) for o, c in chunks] == [(0, 2), (2, 2), (4, 1)]
    ht = concat_headtail([_ht([1, 2]), _ht([3, 4]), _ht([5])])
    assert ht.heading_hints.tolist() == [1, 2, 3, 4, 5]
    cnn = concat_cnn([(0, _cnn(2)), (2, _cnn(2)), (4, _cnn(1))])
    assert [p.det_index for p in cnn.predictions] == [0, 1, 2, 3, 4]
```

- [ ] **Step 2: Run the tests and confirm they fail** (`ModuleNotFoundError`).

- [ ] **Step 3: Implement `downstream_select.py`**

```python
"""Map per-animal stage results between the N-free superset and the final N set.

Per-animal stages run on the superset (every filter survivor) and are cached
by RAW detection-cache index; consumers see results positionally aligned with
the final filtered OBB. These pure helpers do that translation in one place.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from .result import (
    AprilTagResult,
    CNNDetectionPrediction,
    CNNResult,
    HeadTailResult,
    OBBResult,
    PoseResult,
)


class DownstreamCacheError(RuntimeError):
    """A replayed detection has no per-animal result (incoherent cache)."""


def positions_in(superset_idx: np.ndarray, final_idx: np.ndarray) -> np.ndarray:
    where = {int(v): i for i, v in enumerate(np.asarray(superset_idx).tolist())}
    out = []
    for v in np.asarray(final_idx).tolist():
        if int(v) not in where:
            raise DownstreamCacheError(
                f"detection index {int(v)} has no per-animal result in the "
                "superset; the inference cache set is incoherent -- rebuild it."
            )
        out.append(where[int(v)])
    return np.asarray(out, dtype=np.int64)


def select_headtail(ht: HeadTailResult | None, pos: np.ndarray) -> HeadTailResult | None:
    if ht is None:
        return None
    return HeadTailResult(
        heading_hints=ht.heading_hints[pos],
        heading_confidences=ht.heading_confidences[pos],
        directed_mask=ht.directed_mask[pos],
        canonical_affines=None if ht.canonical_affines is None else ht.canonical_affines[pos],
    )


def select_pose(p: PoseResult | None, pos: np.ndarray) -> PoseResult | None:
    if p is None:
        return None
    out = PoseResult(keypoints=p.keypoints[pos], valid_mask=p.valid_mask[pos])
    overrides = getattr(p, "heading_overrides", None)
    if overrides is not None:
        out.heading_overrides = np.asarray(overrides)[pos]
    return out


def select_cnn(r: CNNResult, pos: np.ndarray) -> CNNResult:
    by_index = {p.det_index: p for p in r.predictions}
    preds = [
        replace(by_index[int(src)], det_index=dst)
        for dst, src in enumerate(np.asarray(pos).tolist())
        if int(src) in by_index
    ]
    return CNNResult(label=r.label, predictions=preds)


def cnn_positions_to_raw(r: CNNResult, superset_idx: np.ndarray) -> CNNResult:
    sup = np.asarray(superset_idx)
    return CNNResult(
        label=r.label,
        predictions=[replace(p, det_index=int(sup[p.det_index])) for p in r.predictions],
    )


def cnn_raw_to_positions(
    preds: list[CNNDetectionPrediction], final_idx: np.ndarray, label: str
) -> CNNResult:
    by_raw = {p.det_index: p for p in preds}
    out = []
    for dst, raw in enumerate(np.asarray(final_idx).tolist()):
        if int(raw) not in by_raw:
            raise DownstreamCacheError(
                f"CNN '{label}' has no prediction for detection index {int(raw)}."
            )
        out.append(replace(by_raw[int(raw)], det_index=dst))
    return CNNResult(label=label, predictions=out)


def _at_subset(at: AprilTagResult, rows: list[int], det_indices: list[int]) -> AprilTagResult:
    return AprilTagResult(
        tag_ids=[at.tag_ids[i] for i in rows],
        det_indices=det_indices,
        centers=at.centers[rows] if rows else np.zeros((0, 2), np.float32),
        corners=at.corners[rows] if rows else np.zeros((0, 4, 2), np.float32),
    )


def select_apriltag(at: AprilTagResult | None, pos: np.ndarray) -> AprilTagResult | None:
    if at is None:
        return None
    new_of = {int(src): dst for dst, src in enumerate(np.asarray(pos).tolist())}
    rows = [i for i, d in enumerate(at.det_indices) if int(d) in new_of]
    return _at_subset(at, rows, [new_of[int(at.det_indices[i])] for i in rows])


def apriltag_positions_to_raw(at: AprilTagResult | None, superset_idx: np.ndarray):
    if at is None:
        return None
    sup = np.asarray(superset_idx)
    rows = list(range(len(at.tag_ids)))
    return _at_subset(at, rows, [int(sup[d]) for d in at.det_indices])


def apriltag_raw_to_positions(at: AprilTagResult | None, final_idx: np.ndarray):
    if at is None:
        return None
    new_of = {int(raw): dst for dst, raw in enumerate(np.asarray(final_idx).tolist())}
    rows = [i for i, d in enumerate(at.det_indices) if int(d) in new_of]
    return _at_subset(at, rows, [new_of[int(at.det_indices[i])] for i in rows])


def split_rows(obb: OBBResult, size: int) -> list[tuple[int, OBBResult]]:
    from .stages.filtering import _select

    n = obb.num_detections
    if n <= size:
        return [(0, obb)]
    return [
        (start, _select(obb, np.arange(start, min(n, start + size))))
        for start in range(0, n, size)
    ]


def concat_headtail(parts: list[HeadTailResult | None]) -> HeadTailResult | None:
    parts = [p for p in parts if p is not None]
    if not parts:
        return None
    affines = [p.canonical_affines for p in parts]
    return HeadTailResult(
        heading_hints=np.concatenate([p.heading_hints for p in parts]),
        heading_confidences=np.concatenate([p.heading_confidences for p in parts]),
        directed_mask=np.concatenate([p.directed_mask for p in parts]),
        canonical_affines=None if any(a is None for a in affines) else np.concatenate(affines),
    )


def concat_pose(parts: list[PoseResult | None]) -> PoseResult | None:
    parts = [p for p in parts if p is not None]
    if not parts:
        return None
    out = PoseResult(
        keypoints=np.concatenate([p.keypoints for p in parts]),
        valid_mask=np.concatenate([p.valid_mask for p in parts]),
    )
    overrides = [getattr(p, "heading_overrides", None) for p in parts]
    if all(o is not None for o in overrides):
        out.heading_overrides = np.concatenate(overrides)
    return out


def concat_cnn(parts: list[tuple[int, CNNResult]]) -> CNNResult:
    label = parts[0][1].label
    preds = [
        replace(p, det_index=offset + p.det_index)
        for offset, r in parts
        for p in r.predictions
    ]
    return CNNResult(label=label, predictions=preds)


def concat_apriltag(parts: list[tuple[int, AprilTagResult | None]]) -> AprilTagResult | None:
    present = [(o, a) for o, a in parts if a is not None]
    if not present:
        return None
    tag_ids, det, centers, corners = [], [], [], []
    for offset, a in present:
        tag_ids += list(a.tag_ids)
        det += [offset + int(d) for d in a.det_indices]
        centers.append(a.centers)
        corners.append(a.corners)
    return AprilTagResult(tag_ids, det, np.concatenate(centers), np.concatenate(corners))
```

`CNNDetectionPrediction` is a dataclass (`result.py:123`), so `dataclasses.replace` works. Confirm it is not `frozen=False`-incompatible. If `@dataclass` lacks fields usable by `replace`, construct it explicitly.

- [ ] **Step 4: Run** — `... -m pytest tests/test_downstream_select.py -q -p no:cacheprovider` → PASS.

- [ ] **Step 5: Commit** — `git add src/hydra_suite/core/inference/downstream_select.py tests/test_downstream_select.py && git commit -m "feat(inference): downstream_select helpers for superset<->final per-animal results"`

---

### Task 6: Batch pipeline computes per-animal results on the superset, in chunks

**Files:**
- Modify: `src/hydra_suite/core/inference/pipeline.py:345-440` (`_process_window` filtered set + loop), `:442-590` (`_process_downstream_frame`)
- Test: `tests/test_pipeline_superset.py` (new); `tests/test_inference_runner_batch.py` must stay green

**Interfaces:**
- Consumes: `filter_for_source(..., apply_max_detections=False)` (Task 4), `downstream_select.*` (Task 5), `DOWNSTREAM_CHUNK_SIZE` (Task 1).
- Produces: the cache contract consumed by Task 8. `write_downstream(frame_idx, det_indices=<superset raw indices>, headtail=<superset>, cnn_results=<det_index = RAW index>, pose=<superset>, apriltag=<det_indices = RAW index>)`. In-memory `FrameResult`s are narrowed to the final N set, positional.

- [ ] **Step 1: Write the failing test** (`tests/test_pipeline_superset.py`)

```python
from unittest.mock import MagicMock, patch

import numpy as np

from hydra_suite.core.inference.config import (
    HeadTailConfig, InferenceConfig, OBBConfig, OBBDirectConfig,
)
from hydra_suite.core.inference.result import HeadTailResult, OBBResult


def _obb(frame_idx, n):
    xs = np.arange(n, dtype=np.float32) * 50.0
    c = np.stack([xs, np.zeros(n, np.float32)], 1)
    corners = np.stack([c + d for d in ([-4, -4], [4, -4], [4, 4], [-4, 4])], 1)
    return OBBResult(frame_idx, c, np.zeros(n, np.float32), np.full(n, 64.0, np.float32),
                     np.ones((n, 2), np.float32),
                     np.linspace(0.9, 0.5, n).astype(np.float32),
                     corners.astype(np.float32), OBBResult.make_detection_ids(frame_idx, n))


def _fake_ht(frames, obbs, model, cfg, runtime, geometry, canonical_batch=None):
    return {o.frame_idx: HeadTailResult(o.centroids[:, 0].copy(),
                                        np.ones(o.num_detections, np.float32),
                                        np.ones(o.num_detections, np.uint8), None)
            for o in obbs}


def test_batch_writes_superset_and_returns_final(tmp_path):
    from hydra_suite.core.inference.runner import InferenceRunner, _CacheSet

    cfg = InferenceConfig(
        obb=OBBConfig(mode="direct", direct=OBBDirectConfig(model_path="/m.pt"),
                      max_detections=2, confidence_threshold=0.0, iou_threshold=1.0),
        headtail=HeadTailConfig(model_path="/ht.pt"),
    )
    writer_calls = []
    with (
        patch("hydra_suite.core.inference.runner._load_all_models") as ml,
        patch("hydra_suite.core.inference.pipeline.run_obb",
              side_effect=lambda frames, *a, **k: [_obb(0, 6)]),
        patch("hydra_suite.core.inference.pipeline.run_headtail_batch", side_effect=_fake_ht),
    ):
        ml.return_value = MagicMock(obb=MagicMock(), headtail=MagicMock(), cnn=[],
                                    pose=None, apriltag=None)
        runner = InferenceRunner(cfg, cache_dir=tmp_path)
        caches = _CacheSet(detection=MagicMock(), headtail=MagicMock())
        caches.headtail.write_frame.side_effect = lambda fi, **kw: writer_calls.append(kw)
        results = runner._run_batch([np.zeros((64, 400, 3), np.uint8)], [0], caches)

    # cache holds the N-free superset (all 6), keyed by raw index
    assert writer_calls[0]["det_indices"].tolist() == [0, 1, 2, 3, 4, 5]
    # the returned in-memory result is the final N=2 set, aligned
    (fr,) = [r for r in results if r.frame_idx == 0]
    assert fr.filtered_indices == [0, 1]
    assert fr.headtail.heading_hints.tolist() == [0.0, 50.0]
```

Before writing the test, check how `_run_batch` returns results and how `caches.headtail` is written. Read `runner.py:1764-1790` and `cache/writer.py` (`CacheWriter`). If `_run_batch` writes through `CacheWriter._write_to_handles` (as at `writer.py:239-305`), the `caches.headtail.write_frame` side effect above sees the kwargs. Adjust attribute names to the real ones.

- [ ] **Step 2: Run the test and confirm it fails.** Expected: `det_indices` == `[0, 1]` (today the final set is what gets cached).

- [ ] **Step 3: Superset in `_process_window`**

Replace the filter block at `:373-394` with:

```python
                superset_obb, superset_idx = filter_for_source(
                    cfg, obb_result, self.stages.roi_mask, apply_max_detections=False
                )
                final_obb, final_idx = filter_for_source(
                    cfg, obb_result, self.stages.roi_mask
                )
                if superset_obb.num_detections == 0:
                    self.cache_writer.write_downstream(
                        frame_idx,
                        det_indices=superset_idx,
                        headtail=None,
                        cnn_results=[],
                        pose=None,
                        apriltag=None,
                    )
                    continue
                filtered_by_frame[frame_idx] = (superset_obb, final_obb)
                det_indices_by_frame[frame_idx] = (superset_idx, final_idx)
                nonempty_frames.append(frame)
                nonempty_obbs.append(superset_obb)
```

Keep the clipping-stats loop on `nonempty_obbs` (now the superset: every crop that is materialised). Change the loop call to pass both pairs:

```python
        for frame, obb in zip(nonempty_frames, nonempty_obbs):
            superset_idx, final_idx = det_indices_by_frame[obb.frame_idx]
            _, final_obb = filtered_by_frame[obb.frame_idx]
            assembled.extend(
                self._process_downstream_frame(
                    frame, obb, superset_idx, final_obb, final_idx, geometry
                )
            )
```

- [ ] **Step 4: Chunked superset processing + narrowing in `_process_downstream_frame`**

Change the signature to `(self, frame, superset_obb, superset_idx, final_obb, final_idx, geometry)`. Move the current body (head-tail / CNN / pose / AprilTag on `obbs=[obb]`) into a new `_run_stages_on_chunk(self, frame, chunk_obb, geometry) -> tuple[HeadTailResult|None, list[CNNResult], PoseResult|None, AprilTagResult|None]`. It returns the per-frame values (`headtail.get(frame_idx)`, `[phase[frame_idx] ...]`, `pose.get(frame_idx)`, `apriltag.get(frame_idx)`) instead of writing caches. Then:

```python
        from .downstream_select import (
            apriltag_positions_to_raw, cnn_positions_to_raw, concat_apriltag,
            concat_cnn, concat_headtail, concat_pose, positions_in,
            select_apriltag, select_cnn, select_headtail, select_pose, split_rows,
        )
        from .limits import DOWNSTREAM_CHUNK_SIZE

        ht_parts, pose_parts, at_parts = [], [], []
        cnn_parts: list[list[tuple[int, CNNResult]]] = [[] for _ in cfg.cnn_phases]
        for offset, chunk in split_rows(superset_obb, DOWNSTREAM_CHUNK_SIZE):
            ht, cnns, pose_r, at = self._run_stages_on_chunk(frame, chunk, geometry)
            ht_parts.append(ht)
            pose_parts.append(pose_r)
            at_parts.append((offset, at))
            for k, r in enumerate(cnns):
                cnn_parts[k].append((offset, r))
        ht_all = concat_headtail(ht_parts)
        pose_all = concat_pose(pose_parts)
        at_all = concat_apriltag(at_parts)
        cnn_all = [concat_cnn(parts) for parts in cnn_parts if parts]

        with span(N.CACHE_WRITE):
            self.cache_writer.write_downstream(
                frame_idx,
                det_indices=superset_idx,
                headtail=ht_all,
                cnn_results=[cnn_positions_to_raw(r, superset_idx) for r in cnn_all],
                pose=pose_all,
                apriltag=apriltag_positions_to_raw(at_all, superset_idx),
            )

        if final_obb.num_detections == 0:
            return []
        pos = positions_in(superset_idx, final_idx)
        with span(N.ASSEMBLE_SCATTER):
            return scatter(
                {frame_idx: final_obb},
                None if ht_all is None else {frame_idx: select_headtail(ht_all, pos)},
                {frame_idx: [select_cnn(r, pos) for r in cnn_all]},
                None if pose_all is None else {frame_idx: select_pose(pose_all, pos)},
                None if self.stages.apriltag_model is None else {frame_idx: select_apriltag(at_all, pos)},
                cfg,
                overrides_headtail=(cfg.pose.overrides_headtail if cfg.pose is not None else True),
            )
```

Inside `_run_stages_on_chunk`, keep the spans (`N.HEADTAIL`, `N.CNN`, `N.POSE`, `N.APRILTAG`) and the shared canonical batch logic unchanged, with `obbs = [chunk_obb]`. `scatter` builds `filtered_indices` from its own input. Confirm it fills `FrameResult.filtered_indices` with `final_idx`. If `scatter` derives them positionally, pass `final_idx` through by setting `fr.filtered_indices = [int(i) for i in final_idx]` on the returned result.

- [ ] **Step 5: Run** — `... -m pytest tests/test_pipeline_superset.py tests/test_inference_runner_batch.py tests/test_inference_pipeline*.py -q -p no:cacheprovider` → PASS. (`ls tests | grep pipeline` to get the real pipeline test file names.)

- [ ] **Step 6: Commit** — `git commit -am "feat(pipeline): per-animal stages run on the N-free superset in 256-row chunks; cache by raw index"`

---

### Task 7: Realtime runner uses the same superset contract

**Files:**
- Modify: `src/hydra_suite/core/inference/runner.py:1173-1400` (`run_realtime` filter, stages, cache writes, identity evidence, frame result)
- Test: `tests/test_realtime_superset.py` (new); `tests/test_inference_runner_rt.py` must stay green

**Interfaces:**
- Consumes: Tasks 4-5.
- Produces: realtime cache writes identical in shape to Task 6 (superset raw indices; CNN/AprilTag raw `det_index`); the returned `FrameResult` and the identity evidence use the final N set.

- [ ] **Step 1: Write the failing test.** Mirror Task 6's test through `run_realtime`. Read `tests/test_inference_runner_rt.py` for how it builds a runner with fake models (patch `hydra_suite.core.inference.runner.run_obb` and `run_headtail`), and how caches are opened (`cache_dir=tmp_path`, then a read-back with `_open_caches(..., read_only=True)`). Assert that the head-tail cache for frame 0 has `det_indices == [0..5]` and that the returned `FrameResult.obb.num_detections == 2`.

- [ ] **Step 2: Run the test and confirm it fails.**

- [ ] **Step 3: Implement.** In `run_realtime`, replace the single `filter_for_source` call with the superset/final pair (as in Task 6). Run `_do_ht/_do_cnn/_do_pose/_do_at` over `split_rows(superset_obb, DOWNSTREAM_CHUNK_SIZE)` chunks and concat. `canonical_crops` and `aabb_crops` must be built per chunk inside the loop (move their construction at `:1265-1279` into the loop body). Write caches with `det_indices=superset_idx`, `cnn_positions_to_raw(...)` and `apriltag_positions_to_raw(...)`. Then compute `pos = positions_in(superset_idx, final_idx)` and build `ht_result/cnn_results/pose_result/at_result` with `select_*` for `_write_identity_evidence_realtime(frame_idx, final_obb, cnn_results, at_result)` and `_build_frame_result(frame_idx, final_obb, final_idx, ...)`. The empty branch (`:1180-1215`) triggers on `superset_obb.num_detections == 0`. When the superset is non-empty but the final set is empty, still write the superset caches and return an empty frame result.

- [ ] **Step 4: Run** — `... -m pytest tests/test_realtime_superset.py tests/test_inference_runner_rt.py tests/test_coreml_runner.py -q -p no:cacheprovider` → PASS.

- [ ] **Step 5: Commit** — `git commit -am "feat(realtime): per-animal stages on the N-free superset; final N set returned"`

---

### Task 8: Replay loaders by raw index, identity sidecar via loaders, filter-keyed downstream caches

**Files:**
- Modify: `src/hydra_suite/core/inference/runner.py:836-935` (loaders), `:728-800` (`write_identity_evidence_sidecar`), `:530-664` (`_open_caches`), `:1835-1860` (`load_frame`), `:1423-1450` (identity evidence key)
- Modify: `src/hydra_suite/core/inference/cache/keys.py` (add `replay_filter_hash`, `with_replay_filters`)
- Test: `tests/test_replay_loaders_raw_index.py` (new); update `tests/test_inference_runner_batch.py::test_load_headtail_aligns_by_det_indices` only if its contract changes (it should not)

**Interfaces:**
- Consumes: `downstream_select` (Task 5).
- Produces: `keys.replay_filter_hash(config: InferenceConfig, roi_mask) -> str` (OBB: `confidence_threshold`, `min/max_object_size`, `min/max_aspect_ratio`, `iou_threshold`, `target_classes`, ROI content; bgsub: `""`, since bgsub has no N-free filters beyond extraction). `keys.with_replay_filters(key: CacheKey, filter_hash: str) -> CacheKey`. Loaders raise `DownstreamCacheError` for a missing index.

- [ ] **Step 1: Write the failing tests**

```python
from unittest.mock import MagicMock

import numpy as np
import pytest

from hydra_suite.core.inference.cache.keys import replay_filter_hash
from hydra_suite.core.inference.config import (
    CNNConfig, HeadTailConfig, InferenceConfig, OBBConfig, OBBDirectConfig,
)
from hydra_suite.core.inference.downstream_select import DownstreamCacheError
from hydra_suite.core.inference.result import CNNDetectionPrediction


def _cfg(n=4, conf=0.25):
    return InferenceConfig(
        obb=OBBConfig(mode="direct", direct=OBBDirectConfig(model_path="/m.pt"),
                      max_detections=n, confidence_threshold=conf),
        headtail=HeadTailConfig(model_path="/ht.pt"),
    )


def test_downstream_keys_change_with_filters_not_n(tmp_path):
    from hydra_suite.core.inference.runner import _open_caches

    k = lambda c: _open_caches(c, tmp_path, read_only=True).headtail.key
    assert k(_cfg(n=4)) == k(_cfg(n=400))
    assert k(_cfg(conf=0.25)) != k(_cfg(conf=0.5))
    assert replay_filter_hash(_cfg(n=4), None) == replay_filter_hash(_cfg(n=9), None)


def test_cnn_loader_maps_raw_to_positions():
    from hydra_suite.core.inference.runner import _load_cnn_for_indices

    cache = MagicMock()
    cache.read_frame.return_value = [CNNDetectionPrediction(i, []) for i in (3, 5, 8)]
    cfg = MagicMock(label="id")
    (res,) = _load_cnn_for_indices([cache], [cfg], 0, np.array([8, 3]))
    assert [p.det_index for p in res.predictions] == [0, 1]


def test_load_missing_index_raises():
    from hydra_suite.core.inference.runner import _load_headtail_for_indices

    cache = MagicMock()
    cache.read_frame.return_value = (
        np.array([0, 1], np.int32), np.zeros(2, np.float32),
        np.zeros(2, np.float32), np.zeros(2, np.uint8),
    )
    with pytest.raises(DownstreamCacheError):
        _load_headtail_for_indices(cache, 0, np.array([2]), None)
```

Check that `_open_caches(...).headtail` exposes `.key` (read `cache/store.py:162-185` `_init_store`). If the attribute is named differently (e.g. `cache_key`), use the real name.

- [ ] **Step 2: Run the tests and confirm they fail.**

- [ ] **Step 3: Keys.** In `keys.py`:

```python
def replay_filter_hash(config, roi_mask) -> str:
    """Hash of the N-free replay filters that decide the per-animal superset.

    Per-animal caches hold results for every detection surviving these
    filters (``filter_for_source(..., apply_max_detections=False)``), so they
    must invalidate when a filter changes -- and must NOT depend on N.
    """
    if getattr(config, "detection_source", "obb") != "obb" or config.obb is None:
        return ""
    o = config.obb
    return _sha("|".join(map(str, (
        "replay-filters-v1", o.confidence_threshold, o.min_object_size,
        o.max_object_size, o.min_aspect_ratio, o.max_aspect_ratio,
        o.iou_threshold, tuple(o.target_classes),
        _param_repr(roi_mask) if roi_mask is not None else "",
    ))))


def with_replay_filters(key: CacheKey, filter_hash: str) -> CacheKey:
    if not filter_hash:
        return key
    return replace(key, config_hash=_sha(f"{key.config_hash}|filters={filter_hash}"))
```

In `_open_caches`, compute `fh = replay_filter_hash(config, roi_mask)` once and wrap the four downstream keys: `key=_k(with_replay_filters(headtail_cache_key(...), fh))` (likewise CNN, pose, AprilTag). Do the same to the identity-evidence key at `:1439` (`with_replay_filters(identity_evidence_cache_key(...), replay_filter_hash(self.config, self._roi_mask))`). The detection key is NOT wrapped.

- [ ] **Step 4: Loaders.**
  - `_load_headtail_for_indices` / `_load_pose_for_indices`: replace the `idx_map.get` fill-with-NaN loop with `pos = positions_in(cached_det_indices, det_indices)`, then index arrays by `pos`. This raises on a missing index.
  - `_load_cnn_for_indices`: `results.append(cnn_raw_to_positions(preds, det_indices, cfg.label))`. Keep the `preds is None` → empty-result branch, plus an `len(det_indices) == 0` → empty result.
  - `_load_apriltag(cache, frame_idx)`: add a `det_indices` parameter and return `apriltag_raw_to_positions(cache.read_frame(frame_idx), det_indices)`. Update `load_frame`'s call.
  - `write_identity_evidence_sidecar`: after `filtered_obb, det_idx = filter_for_source(...)`, build `cnn_reads` via `_load_cnn_for_indices(cnn_caches, config.cnn_phases, frame_idx, det_idx)` (map label → predictions) and `tag_read` via `_load_apriltag(caches.apriltag, frame_idx, det_idx)`. Positional alignment with `filtered_obb.detection_ids` then holds by construction. Update the docstring, whose "aligned by position" paragraph no longer describes how it works.

- [ ] **Step 5: Run** — `... -m pytest tests/test_replay_loaders_raw_index.py tests/test_inference_runner_batch.py tests/identity/ tests/test_identity_evidence_vectorized.py tests/test_tracking_replay_fidelity.py tests/test_tracking_production_replay.py tests/test_inference_replay_vector.py -q -p no:cacheprovider` → PASS. A failure that builds a cache by hand with positional CNN `det_index` is fixed by writing raw indices in that fixture. Write down each such fixture change in the commit body.

- [ ] **Step 6: Commit** — `git commit -am "feat(replay): per-animal caches keyed by replay filters (not N), read by raw index; missing index is an error"`

---

### Task 9: End-to-end N reuse

**Files:**
- Test: `tests/test_n_independent_replay.py` (new)

**Interfaces:**
- Consumes: everything above. No production code unless the test exposes a defect. Any defect goes back to its owning task's file and gets its own commit.

- [ ] **Step 1: Write the test**

```python
from unittest.mock import MagicMock, patch

import numpy as np

from hydra_suite.core.inference.config import (
    HeadTailConfig, InferenceConfig, OBBConfig, OBBDirectConfig,
)
from tests.test_pipeline_superset import _fake_ht, _obb


def _cfg(n):
    return InferenceConfig(
        obb=OBBConfig(mode="direct", direct=OBBDirectConfig(model_path="/m.pt"),
                      max_detections=n, confidence_threshold=0.0, iou_threshold=1.0),
        headtail=HeadTailConfig(model_path="/ht.pt"),
    )


def _build(tmp, n):
    from hydra_suite.core.inference.runner import InferenceRunner, _open_caches

    with (
        patch("hydra_suite.core.inference.runner._load_all_models") as ml,
        patch("hydra_suite.core.inference.pipeline.run_obb",
              side_effect=lambda frames, *a, **k: [_obb(i, 30) for i in range(len(frames))]),
        patch("hydra_suite.core.inference.pipeline.run_headtail_batch", side_effect=_fake_ht),
    ):
        ml.return_value = MagicMock(obb=MagicMock(), headtail=MagicMock(), cnn=[],
                                    pose=None, apriltag=None)
        runner = InferenceRunner(_cfg(n), cache_dir=tmp)
        caches = _open_caches(_cfg(n), tmp)
        runner._run_batch([np.zeros((64, 1600, 3), np.uint8)] * 2, [0, 1], caches)
        caches.close()


def _replay(tmp, n):
    from hydra_suite.core.inference.runner import InferenceRunner

    with patch("hydra_suite.core.inference.runner._load_all_models"):
        r = InferenceRunner(_cfg(n), cache_dir=tmp, cache_only=True)
        assert r.caches_all_valid()
        return [r.load_frame(i) for i in (0, 1)]


def test_replay_at_any_n_equals_fresh_run(tmp_path):
    built = tmp_path / "n10"
    _build(built, 10)
    for n in (5, 10, 20):
        fresh = tmp_path / f"fresh{n}"
        _build(fresh, n)
        got, want = _replay(built, n), _replay(fresh, n)
        for g, w in zip(got, want):
            assert g.filtered_indices == w.filtered_indices
            np.testing.assert_array_equal(g.headtail.heading_hints, w.headtail.heading_hints)


def test_replay_at_larger_n_reuses_superset_without_nan(tmp_path):
    _build(tmp_path, 10)
    with patch("hydra_suite.core.inference.pipeline.run_headtail_batch") as ht:
        frames = _replay(tmp_path, 20)
        ht.assert_not_called()
    assert all(len(f.filtered_indices) == 20 for f in frames)
    assert all(np.isfinite(f.headtail.heading_hints).all() for f in frames)
```

If `_open_caches` or `InferenceRunner` needs a `video_path` for `caches_all_valid` (a video signature), pass none: both builds and replays use the same empty signature. Confirm by reading `caches_all_valid` (`runner.py:1043`). Also confirm `_run_batch` accepts the `_CacheSet` returned by `_open_caches` (it did in `test_inference_runner_batch.py`).

- [ ] **Step 2: Run** — `... -m pytest tests/test_n_independent_replay.py -q -p no:cacheprovider` → PASS. If it fails, fix the defect in the owning module, commit it separately, and re-run.

- [ ] **Step 3: Commit** — `git add tests/test_n_independent_replay.py && git commit -m "test: N-independent replay end-to-end (N=5/10/20 from one cache)"`

---

### Task 10: Props-cache ids without N; delete the dead duplicate builder

**Files:**
- Modify: `src/hydra_suite/trackerkit/tracking_cache.py:140-145`
- Modify: `src/hydra_suite/trackerkit/gui/orchestrators/tracking.py:1646-1675` (delete if unused)
- Test: `tests/test_trackerkit_tracking_cache.py`

- [ ] **Step 1: Failing test** (append to `tests/test_trackerkit_tracking_cache.py`, reusing its existing params fixture shape at `:17`):

```python
def test_tracking_cache_ids_ignore_max_targets():
    from hydra_suite.trackerkit.tracking_cache import get_tracking_cache_model_ids

    base = {"DETECTION_METHOD": "background_subtraction", "RESIZE_FACTOR": 1.0,
            "COMPUTE_RUNTIME": "cpu", "MAX_TARGETS": 4}
    other = dict(base, MAX_TARGETS=400)
    assert get_tracking_cache_model_ids(base) == get_tracking_cache_model_ids(other)
```

Read the function signature first (`grep -n "def get_tracking_cache_model_ids" -A12 src/hydra_suite/trackerkit/tracking_cache.py`) and match its real arguments and return type.

- [ ] **Step 2: Run the test and confirm it fails.**
- [ ] **Step 3:** Remove `"MAX_TARGETS",` from `common_detection_keys`. Run `grep -rn "_build_tracking_cache_model_ids\|def .*cache_model_id" src/hydra_suite/trackerkit/gui/orchestrators/tracking.py` and `grep -rn "<that function name>" src tests`. If it has no caller, delete it. If it does have a caller, remove `"MAX_TARGETS"` there too and do not delete it.
- [ ] **Step 4: Run** — `... -m pytest tests/test_trackerkit_tracking_cache.py tests/test_trackerkit_tracking_orchestrator*.py -q -p no:cacheprovider` → PASS.
- [ ] **Step 5: Commit** — `git commit -am "feat(trackerkit): props-cache ids no longer depend on N"`

---

### Task 11: Surface the limit summary; user docs

**Files:**
- Modify: `src/hydra_suite/core/tracking/worker.py:4836-4846` (next to the clipping summary)
- Create: `docs/user-guide/detection-limits.md`; modify `mkdocs.yml` nav (find the user-guide section)
- Test: `tests/test_worker_detection_limit_summary.py` (new)

- [ ] **Step 1: Failing test** — exercise the summary emitter in isolation. Extract the reporting into `TrackingWorker._report_inference_run_summaries(self, runners)` (a new method), which logs both clipping and limit summaries and calls `self._emit_warning("Detection limit reached", msg)` for the limit summary:

```python
from types import SimpleNamespace

from hydra_suite.core.canonicalization.geometry import ClippingStats
from hydra_suite.core.inference.limits import DetectionLimitStats


def test_worker_emits_detection_limit_summary():
    from hydra_suite.core.tracking.worker import TrackingWorker

    stats = DetectionLimitStats()
    stats.record(12, 3000)
    runner = SimpleNamespace(clipping_stats=ClippingStats(), detection_limit_stats=stats)
    seen = []
    worker = TrackingWorker.__new__(TrackingWorker)
    worker._on_warning = lambda title, msg: seen.append((title, msg))
    worker._report_inference_run_summaries([runner, None])
    assert seen and seen[0][0] == "Detection limit reached" and "1024" in seen[0][1]
```

- [ ] **Step 2: Run the test and confirm it fails.**
- [ ] **Step 3:** Implement `_report_inference_run_summaries`. Move the existing `_clip_msgs` loop into it unchanged, and add:

```python
        for _runner in runners:
            stats = getattr(_runner, "detection_limit_stats", None) if _runner else None
            msg = stats.summary() if stats is not None else None
            if msg:
                logger.warning("Detection limit summary: %s", msg)
                self._emit_warning("Detection limit reached", msg)
```

Call it at the old location with `(inference_runner, bgsub_runner)`.

- [ ] **Step 4: Docs.** `docs/user-guide/detection-limits.md`:

```markdown
# Detection limits

HYDRA tracks at most **1024 animals per frame**. The same number is the
maximum number of detections stored per frame and the maximum number of
animals that get per-animal analysis (pose, head-tail, identity).

- Setting more than 1024 animals (or arenas × animals per arena above 1024)
  is rejected with an error when tracking starts.
- If a frame has more than 1024 candidate detections, the 1024 most
  confident are kept, a warning names the frame, and the end of the run
  reports how many frames hit the limit.

## Changing the number of animals reuses inference

Detections are stored independently of the number of animals, so changing
it re-runs only tracking, never the detector or the per-animal models.
Changing a detection filter (confidence, size, aspect ratio, ROI, NMS IoU)
re-runs only the per-animal models. Detections are stored down to
confidence 0.01; a confidence threshold below that has no effect.
```

Add it to `mkdocs.yml` under the user-guide nav. Run `make docs-build` (strict) and fix any nav/link error.

- [ ] **Step 5: Run** — `... -m pytest tests/test_worker_detection_limit_summary.py tests/test_worker_real_inference_integration.py -q -p no:cacheprovider` → PASS.
- [ ] **Step 6: Commit** — `git commit -am "feat(tracking): loud end-of-run detection-limit summary; user docs for the 1024 limit"`

---

### Task 12: Full suite + cost measurement + memory stress

**Files:**
- Create: `tools/equivalence/cache_stats.py`
- Create: `tests/test_downstream_1024_stress.py` (marked `benchmark`, excluded by default)

- [ ] **Step 1: Full suite.** `make pytest` (or `... -m pytest -q -p no:cacheprovider -x --timeout 600`). Compare the failure SET against the base branch (`git stash` is NOT allowed; run the suite on `main` in a separate worktree from `2f8f9726`, the branch base) to separate pre-existing failures from new ones (memory: "pre-existing is tested against the branch base"; "suite batching chunk-boundary trap"). New failures must be fixed in their owning task's files before proceeding.

- [ ] **Step 2: `cache_stats.py`** — reads one `.inference_cache_<stem>` directory and prints per-frame detection counts (mean, p50, p99, max), the number of frames at the limit, and per-animal rows per frame from `headtail.npz`/`pose.npz` when present:

```python
"""Per-frame stored-detection and per-animal-row statistics for one cache dir.

Usage: python tools/equivalence/cache_stats.py <video_dir>/.inference_cache_<stem>
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np


def _member(cache_dir: Path, name: str) -> Path | None:
    manifest = json.loads((cache_dir / "cache_set.json").read_text())
    rel = manifest.get("members", {}).get(name)
    return (cache_dir / rel) if rel else None


def _per_frame_counts(npz_path: Path, key: str) -> np.ndarray:
    with np.load(npz_path, allow_pickle=False) as data:
        frames = data[key]
    _, counts = np.unique(frames, return_counts=True)
    return counts


def main(cache_dir: str) -> None:
    d = Path(cache_dir)
    for member, key in (("detection.npz", "frame_indices"), ("headtail.npz", "frame_indices"), ("pose.npz", "frame_indices")):
        p = _member(d, member)
        if p is None or not p.exists():
            continue
        c = _per_frame_counts(p, key)
        print(f"{member}: frames={len(c)} mean={c.mean():.1f} p50={np.median(c):.0f} "
              f"p99={np.percentile(c, 99):.0f} max={c.max()} at_limit={(c >= 1024).sum()}")


if __name__ == "__main__":
    main(sys.argv[1])
```

Before relying on it, confirm the per-row frame-index array name in the stored npz: `python -c "import numpy as np; print(np.load('<member path>').files)"` on a fixture cache. Chunked stores may split into several files (`cache/chunked.py`). Adapt `_member`/loading to the real layout and keep the printed fields.

- [ ] **Step 3: Stress test** (`tests/test_downstream_1024_stress.py`, `@pytest.mark.benchmark`). Build one 4512×4512 frame and 1024 non-overlapping OBBs. Run `extract_canonical_crops_batch` over `split_rows(obb, DOWNSTREAM_CHUNK_SIZE)` chunks. Assert every chunk's crop batch has ≤ 256 rows and that `tracemalloc` peak (CPU) stays under 1 GiB. On MPS, also record `torch.mps.driver_allocated_memory()` before and after, and print it. Run: `... -m pytest tests/test_downstream_1024_stress.py -m benchmark -q -s -p no:cacheprovider`. Record the printed peak in the PR description.

- [ ] **Step 4: Commit** — `git add tools/equivalence/cache_stats.py tests/test_downstream_1024_stress.py && git commit -m "test(tools): cache stats + 1024-detection downstream stress"`

---

### Task 13: Equivalence re-baseline on MPS and CUDA

**Files:** none (verification + report).

- [ ] **Step 1: Kill stale sleap/hydra processes** (`pgrep -af "sleap|hydra"`; only own sleap/hydra PIDs).
- [ ] **Step 2: Base worktree** — `git worktree add --detach .worktrees/equiv-base 2f8f9726` (the branch base, i.e. pre-change main).
- [ ] **Step 3: MPS matrix, base vs branch** (conda active):

```bash
conda activate hydra-mps
REPO=$PWD WT=$PWD MAIN_SRC=$PWD/.worktrees/equiv-base/src WT_SRC=$PWD/src \
  OUT=/tmp/equiv_nfree_mps RUNTIME=mps bash tools/equivalence/run_matrix.sh
```

Expected: DETERMINISM (branch ×2) at its floor on every clip. EQUIVALENCE vs base is EXPECTED to differ. For each clip record: row counts, unmatched, p99 position delta, and `cache_stats.py` on both caches. Attribute every difference to one of: (a) floor 0.001→0.01, (b) 2N window/ranking tie order, (c) per-animal superset (should not change tracked positions), (d) the removed 128 clamp. A difference that fits none of these is a bug: stop and investigate. Verify row counts > 1 in every CSV (empty-CSV false pass).

- [ ] **Step 4: Attribution run for (a).** Re-run the branch with `EXTRACTION_CONFIDENCE_FLOOR` temporarily set to 0.001 (local edit, not committed) on `fly_obb worm_bgsub ant_obb_sleap`. If equivalence vs base then collapses to the floor, (a) explains the drift. Revert the edit and confirm with `git diff --quiet`.
- [ ] **Step 5: CUDA matrix** on mehek (`rutalab@mehek.taild08eb9.ts.net`, `hydra-cuda`, `CUDA_MAJOR=13`) or on diptera with an explicitly named idle GPU (`nvidia-smi`, `export CUDA_VISIBLE_DEVICES=<uuid>`, `CUDA_MAJOR=12`, never `--gpus auto`). Push the branch to the box (`git push <box>:hydra-suite HEAD:refs/heads/feat/n-independent-caches`) and use a worktree there. Run the same matrix with `RUNTIME=cuda`. Same acceptance and attribution.
- [ ] **Step 6: Performance.** Report the `new/base` wall ratio per clip. Above `PERF_TOLERANCE` (1.25) on any clip: report the per-animal crop counts from `cache_stats.py` (superset vs final) as the cause, and stop for a decision. Do not tune silently.
- [ ] **Step 7: Write the results** to `docs/superpowers/notes/2026-10-09-n-independent-caches-verification.md` (per-clip table, MPS + CUDA, attribution, perf, stress peak) and commit: `git commit -m "docs(notes): N-independent caches verification (MPS+CUDA)"`.
- [ ] **Step 8: Cleanup** — `git worktree remove --force .worktrees/equiv-base && git worktree prune`.
