# Inference Autotuner — Adversarial-Review Remediation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close the 4 blockers and 9 serious findings from the Fable adversarial review of `codex/trackerkit-inference-autotuner`, prove the fixes with tests that actually execute the sidecar child, and drive the branch to a defensible merge-ready state on both MPS and CUDA.

**Architecture:** The autotuner package is structurally sound where it is unit-testable (profile store locking, atomic writes, artifact lifecycle, fingerprint memoization) and defective everywhere the sidecar child actually runs — because **no existing test ever executes `sidecar_child.py`**. The remediation is therefore anchored on a real end-to-end subprocess test built first (Task 1), which fails on today's code and becomes the regression harness every later task is validated against. Fixes then proceed blocker-first (serialization, record-mode leak, measurement honesty, correctness gate), then the serious findings, then the real equivalence gates.

**Tech Stack:** Python 3.11, PyTorch (MPS/CUDA), pandas, pytest, numpy, conda envs `hydra-mps` (this box) / `hydra-cuda` (mehek), `tools/equivalence/` harness.

**Spec:** `docs/superpowers/specs/2026-09-06-trackerkit-inference-autotuner-design.md`

**Review source:** Fable adversarial review, 2026-09-07 (findings B1–B4, S1–S9, minors). Finding IDs are used throughout.

## Global Constraints

- **Worktree:** all work happens in `/Users/neurorishika/.codex/worktrees/6e86/multi-animal-tracker` on branch `codex/trackerkit-inference-autotuner`. Never touch the primary checkout at `/Users/neurorishika/Projects/Rockefeller/Kronauer/multi-animal-tracker`.
- **Foreign stash:** this worktree contains a pre-existing `stash@{0}` from `feature/trackingworker-qt-split` that belongs to another agent. **Never** run `git stash drop`, `git stash pop`, `git reset --hard`, or `git checkout` of tracked files.
- **PYTHONPATH:** every pytest and every equivalence run launched from this worktree MUST set `PYTHONPATH=/Users/neurorishika/.codex/worktrees/6e86/multi-animal-tracker/src`. Without it Python imports main's editable install and you silently test the wrong tree.
- **Numba/pycache:** before any equivalence run, `find . -name __pycache__ -prune -exec rm -rf {} +`. A stale `@jit(cache=True)` entry from another worktree at the same path has previously faked both an "EQUIVALENT" and a "CSV missing" verdict.
- **Heavy runs:** before any equivalence/calibration run, kill stale `sleap`/`hydra` processes only. Never interfere with any other process.
- **conda:** `conda activate hydra-mps` must be active for any pose/SLEAP clip, or CSVs come out empty and falsely compare EQUIVALENT. Always assert row counts > 1.
- **Correctness-gate policy (user decision, 2026-09-07):** keep the spec's deliberate geometry tolerances (0.5 px position p99, spec:347) but make `TrackID`/`TrajectoryID`/`State` exact, and change the angular test from a mean to a per-row max. Do NOT make the gate byte-exact — the spec (:354) records measured evidence that batch 8 changes a CUDA identity row, so a byte-exact gate would admit nothing.
- **Calibration budget (user decision, 2026-09-07):** raise the default from 120 s to 600 s (the `SidecarTrialSpec.__post_init__` validator already permits 5–600). Keep the full joint search space.
- **CUDA transport (user decision, 2026-09-07):** a **source-only** snapshot is authorized (`git archive`, never `git bundle`). Do NOT transfer branch history. **Use `courtship` and `firebrat`, not `mehek`** — mehek is running unrelated SAM3 LoRA training (PID 3487055) and must not be disturbed.
- **Commit style:** conventional commits, one per task minimum. Commit as the configured git user (no Claude Co-Authored-By trailer).
- **Docs lifecycle:** the spec stays in `docs/superpowers/specs/` until merge; on merge, `git mv` both spec and this plan into the matching `done/` subfolders in the merge commit.

---

## File Structure

| File | Responsibility | Tasks |
|---|---|---|
| `tests/test_inference_autotune_sidecar_child_e2e.py` | **New.** Real subprocess execution of `sidecar_child` against `fly_obb` with an injected ROI. The anchor regression harness. | 1 |
| `src/hydra_suite/core/inference/autotune/sidecar.py` | Request serialization (`_json_value`), block sizing (`_frames_for_block`), trial spec defaults | 2, 4, 6 |
| `src/hydra_suite/core/inference/autotune/sidecar_child.py` | Child entry point: windows, warmup, per-window engine runs, CSV header | 4, 8, minors |
| `src/hydra_suite/core/inference/autotune/coordinator.py` | Resolve order (record vs cache), `_reuse` splicing/down-admission | 3, 6, 7 |
| `src/hydra_suite/core/inference/autotune/integration.py` | Request building, `allow_cached_reuse`, overlay application | 3, 7 |
| `src/hydra_suite/core/inference/autotune/equivalence.py` | Correctness gate policy and column classification | 5 |
| `src/hydra_suite/core/inference/autotune/measure.py` | `measurement_complete` minimums | 4 |
| `src/hydra_suite/core/inference/autotune/search.py` | Candidate loop, rejection bookkeeping | 4, 6 |
| `src/hydra_suite/core/inference/autotune/candidates.py` | `down_admit` candidate filtering | 6 |
| `src/hydra_suite/core/inference/autotune/fingerprint.py` | `TuningProfileKey` fields | 7 |
| `src/hydra_suite/core/inference/autotune/store.py` | Profile states, negative caching, lock pruning | 6, 7, minors |
| `src/hydra_suite/core/inference/autotune/device.py` | CUDA device probe targeting | 8 |
| `src/hydra_suite/core/inference/autotune/models.py` | Overlay `apply()` side effects | minors |
| `src/hydra_suite/core/tracking/worker.py` | Profiler phase boundary, preflight fallback envelope | 4, 6 |
| `tools/equivalence/run_matrix.sh` + new autotune leg | Real MPS/CUDA gates with autotune enabled | 9, 10 |

---

### Task 1: Anchor — a test that actually runs the sidecar child

**Why first:** `sidecar_child.run` has never been executed by any test or by any human. Every fix below is guesswork until this exists. It fails today on B1.

**Files:**
- Create: `tests/test_inference_autotune_sidecar_child_e2e.py`

**Interfaces:**
- Consumes: `hydra_suite.core.inference.autotune.sidecar.write_sidecar_request`, `SidecarTrialSpec`, `hydra_suite.trackerkit.engine_params.build_engine_params`
- Produces: pytest marker `sidecar_e2e`; **`tests/autotune_helpers.py`** — the shared fixture module every later task's tests import.

**Also create `tests/autotune_helpers.py` in this task.** Later tasks reference helpers (`_settings`, `_key`, `_profile_with`, `_evidence`, `_memory_store`, `_request`, `_never_called`, `_child_env`, `_search_with_candidates`, `make_roi_params`) that must have exactly one definition. Build them by reading the existing constructions in `tests/test_inference_autotune_search.py` and `tests/test_inference_autotune_store.py` and lifting them into this module; later tasks import from it rather than redefining. If a helper a later task names does not exist yet, add it here when that task needs it — never re-invent it locally.

- [ ] **Step 1: Register the marker**

In `pyproject.toml` under `[tool.pytest.ini_options] markers`, add:
```
"sidecar_e2e: end-to-end sidecar child subprocess runs (slow, needs fixtures)",
```

- [ ] **Step 2: Write the failing test**

```python
"""End-to-end execution of the autotune sidecar child process.

No other test in the suite executes ``sidecar_child``; this file is the
regression anchor for every finding that only manifests when the child
actually runs.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

FIXTURES = Path(__file__).resolve().parents[1] / "tools/equivalence/fixtures/clips"
CLIP = FIXTURES / "fly_obb.mp4"

pytestmark = pytest.mark.sidecar_e2e


def make_roi_params(video_path: Path, tmp_path: Path) -> dict:
    """Engine params for fly_obb WITH a non-empty ROI.

    A non-empty ``roi_shapes`` makes ``build_engine_params`` emit
    ``ARENA_LABELS`` (a uint16 ndarray) alongside ``ROI_MASK`` -- the exact
    shape that broke serialization (finding B1).
    """
    from hydra_suite.trackerkit.engine_params import build_engine_params

    config = json.loads((FIXTURES.parent / "configs/fly_obb.json").read_text())
    # Cover the WHOLE frame: an ROI smaller than the frame suppresses
    # detections and yields a header-only CSV, failing the test for the
    # wrong reason. Read the real dimensions.
    import cv2

    cap = cv2.VideoCapture(str(video_path))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    config["roi_shapes"] = [
        {"mode": "include", "arena_id": 0,
         "type": "rectangle", "points": [[0, 0], [width, height]]}
    ]
    return build_engine_params(config, video_path=str(video_path))


@pytest.mark.skipif(not CLIP.exists(), reason="equivalence fixtures not fetched")
def test_sidecar_child_completes_on_a_project_with_an_roi(tmp_path):
    from hydra_suite.core.inference.autotune.sidecar import write_sidecar_request

    params = make_roi_params(CLIP, tmp_path)
    request_path = tmp_path / "request.json"
    write_sidecar_request(
        request_path,
        video_path=CLIP,
        params=params,
        settings_overrides={},
        start_frame=0,
        end_frame=31,
        maximum_frames=32,
    )

    env = dict(os.environ)
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[1] / "src")
    proc = subprocess.run(
        [sys.executable, "-m",
         "hydra_suite.core.inference.autotune.sidecar_child",
         str(request_path), str(tmp_path / "result.json")],
        capture_output=True, text=True, timeout=600, env=env,
    )

    assert proc.returncode == 0, f"child failed:\n{proc.stderr[-4000:]}"
    result = json.loads((tmp_path / "result.json").read_text())
    assert result.get("failure_class") is None, result
    assert result["measured_frames"] > 0
    forward = Path(result["forward_csv"])
    assert forward.exists() and len(forward.read_text().splitlines()) > 1
```

- [ ] **Step 3: Adapt the test to the real signatures**

`write_sidecar_request` and the child's `__main__` argument shape may differ from the sketch above. Read `src/hydra_suite/core/inference/autotune/sidecar.py` (`write_sidecar_request`, `SidecarTrialSpec`) and the bottom of `sidecar_child.py` and adjust the call sites so the test invokes the child exactly as `ContainedTrialExecutor` does. Do not change production code in this step.

- [ ] **Step 4: Run it and confirm it fails on B1**

```bash
cd /Users/neurorishika/.codex/worktrees/6e86/multi-animal-tracker
PYTHONPATH=$PWD/src python -m pytest tests/test_inference_autotune_sidecar_child_e2e.py -v -m sidecar_e2e
```
Expected: FAIL with `TypeError: non-ROI arrays are not valid sidecar params: ARENA_LABELS` (raised at `sidecar.py:145`).

- [ ] **Step 5: Commit the failing anchor**

```bash
git add tests/test_inference_autotune_sidecar_child_e2e.py pyproject.toml
git commit -m "test(autotune): add end-to-end sidecar child harness (RED, reproduces B1)"
```

---

### Task 2: B1 — serialize every ndarray param, not just ROI_MASK

**Failure being fixed:** `engine_params.py:1513` emits `ARENA_LABELS` (uint16 ndarray) whenever `roi_shapes` is non-empty. `sidecar.py:143-145` raises `TypeError` for any ndarray whose key is not `ROI_MASK`. The raise is caught by `_run_once` (`sidecar.py:339-347`) → every block reports `failure_class="TypeError"` → `SearchResult(..., "baseline_measurement_incomplete")` → permanent silent fallback on every arena project. Gate fixtures all have `roi_shapes: []`, so no existing gate can see it.

**Files:**
- Modify: `src/hydra_suite/core/inference/autotune/sidecar.py:136-160` (`_json_value`) and its `read_sidecar_request` counterpart
- Test: `tests/test_inference_autotune_sidecar_child_e2e.py` (Task 1), plus a unit round-trip test

- [ ] **Step 1: Write the failing unit test**

Append to `tests/test_inference_autotune_sidecar.py`:
```python
def test_array_params_round_trip_by_key(tmp_path):
    """Every ndarray param survives the request round-trip, not just ROI_MASK."""
    import numpy as np
    from hydra_suite.core.inference.autotune.sidecar import (
        read_sidecar_request, write_sidecar_request,
    )

    labels = np.arange(6, dtype=np.uint16).reshape(2, 3)
    mask = (labels > 0)
    path = tmp_path / "request.json"
    write_sidecar_request(
        path, video_path=tmp_path / "v.mp4",
        params={"ARENA_LABELS": labels, "ROI_MASK": mask, "N_ARENAS": 1},
        settings_overrides={}, start_frame=0, end_frame=7, maximum_frames=8,
    )
    restored = read_sidecar_request(path)["params"]
    np.testing.assert_array_equal(restored["ARENA_LABELS"], labels)
    assert restored["ARENA_LABELS"].dtype == labels.dtype
    np.testing.assert_array_equal(restored["ROI_MASK"], mask)
```

- [ ] **Step 2: Run it, confirm it fails**

```bash
PYTHONPATH=$PWD/src python -m pytest tests/test_inference_autotune_sidecar.py::test_array_params_round_trip_by_key -v
```
Expected: FAIL with `TypeError: non-ROI arrays are not valid sidecar params: ARENA_LABELS`.

- [ ] **Step 3: Implement — one .npy per array key**

Replace the ndarray branch in `_json_value` (`sidecar.py:143-148`). `roi_path` becomes a directory, and each array is saved under a key-derived filename:
```python
    if isinstance(value, np.ndarray):
        if not key:
            raise TypeError("array parameters require a key to be serialized")
        array_dir.mkdir(parents=True, exist_ok=True)
        name = f"{key}.npy"
        np.save(array_dir / name, value, allow_pickle=False)
        return {"__hydra_npy__": name}
```
Thread `array_dir: Path` through `_json_value` in place of `roi_path`, and update `write_sidecar_request` to pass a per-request `arrays/` subdirectory. In the reader, restore any dict carrying `__hydra_npy__` via `np.load(array_dir / name, allow_pickle=False)`. Keep accepting the legacy `__hydra_roi_npy__` key on read so an in-flight request written by an older build still loads.

- [ ] **Step 4: Run both tests**

```bash
PYTHONPATH=$PWD/src python -m pytest tests/test_inference_autotune_sidecar.py -v
PYTHONPATH=$PWD/src python -m pytest tests/test_inference_autotune_sidecar_child_e2e.py -v -m sidecar_e2e
```
Expected: both PASS. The e2e test now exercises the child for real — if it fails for a *different* reason, that is a new genuine finding: record it and fix it here before moving on.

- [ ] **Step 5: Verify the size guard still holds**

`MAX_REQUEST_BYTES` (`sidecar.py:40`) bounds the JSON, not the sidecar `.npy` files. Confirm arrays are excluded from that measurement and that a 4K `ARENA_LABELS` (≈16 MB) does not trip it.

- [ ] **Step 6: Commit**

```bash
git add src/hydra_suite/core/inference/autotune/sidecar.py tests/test_inference_autotune_sidecar.py
git commit -m "fix(autotune): serialize all ndarray sidecar params (B1: ARENA_LABELS killed every ROI project)"
```

---

### Task 3: B2 — record mode must never apply, on any run

**Failure being fixed:** `coordinator.py:88-94` returns `_reuse(...)` for a VALIDATED cached profile *before* the `mode == "record"` check at `:207`. `integration.py:588-616` only clears `allow_cached_reuse` for realtime / cache_replay / non-CUDA-automatic, so record keeps it True. Run 1 records and keeps configured values; run 2 hits the cache and silently applies the tuned vector — including on MPS, while the GUI label reads "Record-only — configured values run." This violates the spec's own rollout stage 1 (spec:494).

**Files:**
- Modify: `src/hydra_suite/core/inference/autotune/coordinator.py:88-94`
- Modify: `src/hydra_suite/core/inference/autotune/integration.py:588-616`
- Test: `tests/test_inference_autotune_search.py`

- [ ] **Step 1: Write the failing test**

```python
def test_record_mode_never_applies_even_on_a_cache_hit():
    """Run 2 with a VALIDATED profile in the cache must still keep baseline."""
    store, request = _record_mode_request_with_validated_cache()  # see Step 2
    result = AutotuneCoordinator(store, trial_executor=_never_called).resolve(request)
    assert result.overlay.status == "recorded"
    assert result.overlay.effective == request.baseline
```

- [ ] **Step 2: Build the fixture helper**

Follow the construction already used by `test_record_only_persists_but_does_not_apply` (`tests/test_inference_autotune_search.py:395-415`), but pre-populate the store with a VALIDATED `InferenceTuningProfile` whose `selected` differs from `baseline` in `detection_batch_size`, and pass a `trial_executor` that raises if called (proving no re-calibration happens either).

- [ ] **Step 3: Run it, confirm it fails**

```bash
PYTHONPATH=$PWD/src python -m pytest tests/test_inference_autotune_search.py::test_record_mode_never_applies_even_on_a_cache_hit -v
```
Expected: FAIL — status is `cache_hit` and `effective` carries the tuned batch size.

- [ ] **Step 4: Fix in both places (defence in depth)**

In `integration.py`, add to the eligibility chain before the `automatic`/CUDA branch:
```python
    if policy.mode == "record":
        allow_cached_reuse = False
```
(leave `eligible` alone — record mode must still calibrate).

In `coordinator.py:88`, guard the reuse return so the mode check cannot be bypassed:
```python
        cached = self.store.load(request.key)
        if (
            request.allow_cached_reuse
            and request.mode != "record"
            and cached is not None
            and cached.state is ProfileState.VALIDATED
        ):
            return self._reuse(request, cached, status="cache_hit")
```
Then, immediately after, short-circuit record mode on a cache hit so run 2 does not needlessly re-calibrate:
```python
        if request.mode == "record" and cached is not None and cached.state is ProfileState.VALIDATED:
            return ResolveResult(
                InferenceRuntimeOverlay.baseline(
                    request.baseline,
                    status="recorded",
                    reason="validated profile already recorded; record-only mode kept configured settings",
                ),
                cached,
            )
```

**Watch the side effect:** the short-circuit returns `cached` as `result.profile`, so `integration.py:679-684` will call `record_profile_memory_evidence` on *every* record-mode cache hit. Confirm it is idempotent, or gate that call to `status == "calibrated"` only.

- [ ] **Step 5: Run the full search test file**

```bash
PYTHONPATH=$PWD/src python -m pytest tests/test_inference_autotune_search.py -v
```
Expected: all PASS, including the pre-existing run-1 record test.

- [ ] **Step 6: Commit**

```bash
git add src/hydra_suite/core/inference/autotune/coordinator.py \
        src/hydra_suite/core/inference/autotune/integration.py \
        tests/test_inference_autotune_search.py
git commit -m "fix(autotune): record mode never applies, including on a cache hit (B2)"
```

---

### Task 4: B3 + S9 — honest measurement (phase boundary, window size, no silent pruning)

**Failure being fixed, three parts:**
1. `worker.py:1189` ends the `initialization` phase *before* every `InferenceRunner(...)` construction (`:1303`, `:1447`, `:1492`), where all model loading happens. `sidecar_child.py:107-127` computes `steady = wall − initialization − cleanup`, so every measured window's "steady" time contains a full model load (plus SLEAP service spawn / TRT engine load).
2. `_frames_for_block(128, i)` → 25/26 frames per block; `_representative_windows(start, end, 26)` → `per_window = max(8, 26//3) = 8` → three 8-frame engine runs per block. Throughput is 24 frames measured against 3 model loads. Batch/depth effects are swamped.
3. `sidecar_child.py:330-344`: `warmup_calls = min(3, ceil(warmup_frames / batch))`, so any `detection_batch_size >= 13` yields `warmup_calls = 2 < 3` → `measurement_complete` False (`measure.py:142-159`) → `search.py:462-463` `continue`, and the candidate vanishes with **no entry in `rejected`**. The real detector search space is silently {1,2,4,8}.
4. **S9 (closed by part 1):** calibration itself runs after `phase_end("initialization")` (worker.py:1189 vs 1252), so production `_steady` (`worker.py:4573-4580`) absorbs up to the full calibration budget plus model load. On short videos that trips the 0.85 regression rule in `observe_production_throughput` (`store.py:243-250`) → the just-validated profile is demoted to PROVISIONAL → re-tune next run → loop. **Verify explicitly** that the moved boundary lands after `_resolve_inference_autotune_before_load` (`:1252-1275`), not merely after the runner constructions.

**User decision applied:** raise the default budget to 600 s and keep the full search space.

**Files:**
- Modify: `src/hydra_suite/core/tracking/worker.py:1189`
- Modify: `src/hydra_suite/core/inference/autotune/sidecar_child.py:86-104, 320-350`
- Modify: `src/hydra_suite/core/inference/autotune/sidecar.py:45-49, 61`
- Modify: `src/hydra_suite/core/inference/autotune/search.py:462`
- Test: `tests/test_inference_autotune_sidecar.py`, `tests/test_inference_autotune_search.py`

- [ ] **Step 1: Move the profiler phase boundary**

Move `profiler.phase_end("initialization")` from `worker.py:1189` to immediately after the last `InferenceRunner(...)` construction in the setup block (after `:1492`), so model loading is billed to `initialization` and excluded from `steady`. Read the surrounding code first — the boundary must still precede the frame loop, and all three runner constructions must fall inside.

- [ ] **Step 2: Verify no consumer asserts the old boundary**

```bash
PYTHONPATH=$PWD/src python -m pytest tests/ -k "profil or span" -v
grep -rn "initialization" docs/developer-guide/performance-tuning.md tools/equivalence/
```
The profile JSON is not part of the CSV equivalence gate, so this cannot affect byte-identity — but any perf-gate consumer that hard-codes the phase split must be updated in this commit.

- [ ] **Step 3: Write the failing window-size test**

```python
def test_measurement_windows_are_large_enough_to_amortize_model_load():
    """One contiguous window per block, at least 128 frames when available."""
    from hydra_suite.core.inference.autotune.sidecar_child import _representative_windows
    windows = _representative_windows(0, 999, 128)
    assert len(windows) == 1
    assert windows[0][1] - windows[0][0] + 1 == 128
```

- [ ] **Step 4: Write the failing no-silent-pruning test**

```python
def test_large_batch_candidates_are_rejected_with_a_reason_not_dropped():
    """A candidate that cannot satisfy the warmup minimum appears in rejected."""
    result = _search_with_candidates(detection_batch_sizes=(1, 16))
    names = {item.settings.detection_batch_size for item in result.rejected}
    assert 16 in names or 16 == result.selected.detection_batch_size
```
Build `_search_with_candidates` from the existing harness in `tests/test_inference_autotune_search.py`; the assertion is that no candidate leaves the loop unrecorded.

- [ ] **Step 5: Run both, confirm they fail**

```bash
PYTHONPATH=$PWD/src python -m pytest tests/test_inference_autotune_sidecar.py::test_measurement_windows_are_large_enough_to_amortize_model_load tests/test_inference_autotune_search.py::test_large_batch_candidates_are_rejected_with_a_reason_not_dropped -v
```
Expected: FAIL — three 8-frame windows, and the 16-batch candidate missing from `rejected`.

- [ ] **Step 6: Implement — one contiguous window per block**

In `sidecar_child.py`, replace `_representative_windows` with a single-window function: each measurement block gets **one** contiguous run of `min(per_block_frames, available)` frames. Representativeness across the clip is already provided by the five interleaved blocks — sample the *block's* start position across the clip instead of splitting each block into three sub-windows:
```python
def _block_window(start: int, end: int, frames: int, block_index: int, blocks: int) -> tuple[int, int]:
    """One contiguous window per block, striped across the clip.

    Splitting a block into sub-windows means paying a model load per
    sub-window, which swamps the batch effect being measured. Spread the
    blocks across the clip instead.
    """
    total = end - start + 1
    span = min(frames, total)
    if blocks <= 1 or total <= span:
        return (start, start + span - 1)
    stride = (total - span) // (blocks - 1)
    offset = start + stride * (block_index % blocks)
    return (offset, offset + span - 1)
```
Update the call site and delete `_representative_windows` (and its now-stale test at `tests/test_inference_autotune_sidecar.py:193-198`).

- [ ] **Step 7: Implement — raise per-block frames and the budget**

**Two different `maximum_frames` fields exist — get both right or the search admits nothing.**

- `SidecarTrialSpec.maximum_frames` (`sidecar.py:57`) is the **per-phase** cap that `_frames_for_block` divides across five blocks. Raise it to `640` so each block gets >=128 frames.
- `MeasurementProtocol.maximum_frames` (`measure.py`) is the **completion** threshold in `measurement_complete`: `measured_frames >= protocol.maximum_frames` OR `stage_seconds >= minimum_stage_seconds`. Leave this at 128 or set it to the per-block value — do **not** set it to 640. If it exceeds what a short fixture clip can supply, every candidate becomes `measurement_incomplete`, the search admits nothing, and Task 4's own no-silent-pruning test still passes (it only asserts the rejection is *recorded*). Add an assertion in the search that at least one candidate completed, or the failure is silent again.

Also raise `SidecarTrialSpec.budget_seconds` to `600.0` and `per_trial_timeout_seconds` to `120.0` (the validator's ceiling; needed for a cold TensorRT build, S5), and update the matching defaults wherever `INFERENCE_AUTOTUNE_BUDGET_SECONDS` is read in `integration.py`/`engine_params.py`.

- [ ] **Step 8: Implement — warmup scales with batch, and nothing is dropped silently**

The child has no `MeasurementProtocol` — it reads a request JSON. Serialize `warmup_calls` into the request in `write_sidecar_request` and read it in the child (preferred), or use the literal `3` with a comment naming `MeasurementProtocol.warmup_calls` as the source of truth. Then in `sidecar_child.py:330-344`, size warmup from that requirement rather than capping it:
```python
    required_calls = int(request.get("warmup_calls", 3))
    warmup_frames = min(
        end - start + 1,
        max(8, required_calls * settings.detection_batch_size),
    )
```
so `warmup_calls = ceil(warmup_frames / batch)` reaches the minimum for every batch the clip can support. If the clip is too short to supply that many frames, the candidate must be **recorded as rejected**, never dropped. In `search.py:462`, replace the bare `continue` with a recorded rejection:
```python
            if not measurement_complete(evidence, self.protocol):
                rejected.append(
                    RejectedCandidate(
                        settings=candidate,
                        reason="measurement_incomplete",
                        detail=(
                            f"warmup_calls={evidence.warmup_calls} "
                            f"warmup_frames={evidence.warmup_frames} "
                            f"blocks={len(evidence.throughput_samples)}"
                        ),
                    )
                )
                continue
```
Match the existing rejection record type in `search.py` — read it and reuse it rather than inventing a new one.

- [ ] **Step 9: Run the tests plus the e2e anchor**

```bash
PYTHONPATH=$PWD/src python -m pytest tests/test_inference_autotune_sidecar.py tests/test_inference_autotune_search.py tests/test_inference_autotune_measure.py -v
PYTHONPATH=$PWD/src python -m pytest tests/test_inference_autotune_sidecar_child_e2e.py -v -m sidecar_e2e
```
Expected: all PASS.

- [ ] **Step 10: Commit**

```bash
git add -A src/hydra_suite/core/inference/autotune src/hydra_suite/core/tracking/worker.py tests/
git commit -m "fix(autotune): honest measurement -- model load excluded, contiguous 128-frame blocks, no silent candidate pruning (B3)"
```

---

### Task 5: B4 — close the correctness-gate holes the spec did not intend

**Failure being fixed:** `equivalence.py:78-84` `_row_key` picks `["FrameID", "DetectionID"]` first, and `TrackID`/`TrajectoryID`/`State` are absent from `_categorical_columns` (`:60-75`) — so a Hungarian ID swap (detection 3 in frame 10 assigned to track 1 instead of track 2) matches on row counts, keys, and XY, and **passes**. Separately, `angle_mean_tolerance` is a *mean*: 1% of rows flipping by π averages to 0.031 rad and passes.

**Scope note:** the geometry tolerances themselves (0.5 px p99) stay — spec:347 chose them deliberately, and spec:354 records measured evidence that batch 8 changes a CUDA identity row, so a byte-exact gate would admit nothing. This task closes the *unintended* holes only.

**Files:**
- Modify: `src/hydra_suite/core/inference/autotune/equivalence.py:31-35, 60-84`
- Test: `tests/test_inference_autotune_equivalence.py`

- [ ] **Step 1: Write the failing ID-swap test**

```python
def test_a_track_identity_swap_fails_the_gate():
    """Same positions, swapped TrackIDs, must NOT be judged equivalent."""
    import pandas as pd
    from hydra_suite.core.inference.autotune.equivalence import (
        CalibrationOutputs, EquivalencePolicy, compare,
    )
    base = pd.DataFrame({
        "FrameID": [10, 10], "DetectionID": [0, 1],
        "TrackID": [1, 2], "State": ["confirmed", "confirmed"],
        "CentroidX": [5.0, 90.0], "CentroidY": [5.0, 90.0],
        "OrientationRad": [0.0, 0.0],
    })
    swapped = base.copy()
    swapped["TrackID"] = [2, 1]
    verdict = compare(
        CalibrationOutputs(base, base), CalibrationOutputs(swapped, swapped),
        EquivalencePolicy(),
    )
    assert not verdict.passed
```

- [ ] **Step 2: Write the failing pi-flip test**

```python
def test_a_single_pi_flip_fails_even_though_the_mean_is_small():
    """One flipped row in 100 averages to 0.031 rad; a mean test lets it pass."""
    import numpy as np, pandas as pd
    from hydra_suite.core.inference.autotune.equivalence import (
        CalibrationOutputs, EquivalencePolicy, compare,
    )
    n = 100
    base = pd.DataFrame({
        "FrameID": range(n), "DetectionID": [0] * n, "TrackID": [1] * n,
        "State": ["confirmed"] * n,
        "CentroidX": np.zeros(n), "CentroidY": np.zeros(n),
        "OrientationRad": np.zeros(n),
    })
    flipped = base.copy()
    flipped.loc[0, "OrientationRad"] = np.pi
    verdict = compare(
        CalibrationOutputs(base, base), CalibrationOutputs(flipped, flipped),
        EquivalencePolicy(),
    )
    assert not verdict.passed
```

- [ ] **Step 3: Run both, confirm they fail**

```bash
PYTHONPATH=$PWD/src python -m pytest tests/test_inference_autotune_equivalence.py -k "swap or pi_flip" -v
```
Expected: both FAIL (verdict passes).

- [ ] **Step 4: Adapt the tests to the real API**

`compare` may be named differently and may take a determinism floor. Read `equivalence.py` and fix the call sites — do not weaken the assertions.

- [ ] **Step 5: Implement — exact identity/state columns**

Add a mandatory-exact set that is checked independently of the token heuristics:
```python
_MANDATORY_EXACT_COLUMNS = ("TrackID", "TrajectoryID", "State", "ArenaID")
```
and in `_categorical_columns`, mark any column whose name is in that tuple as categorical regardless of the token match. These columns are compared with `.equals()` after the row alignment, and any difference sets `passed=False` with a detail naming the column and the first differing `FrameID`.

- [ ] **Step 6: Implement — per-row max angular test**

Replace `angle_mean_tolerance` with `angle_max_tolerance: float = 0.05` in `EquivalencePolicy` and compare the per-row maximum of the wrapped angular difference (wrap to `(-pi, pi]` before taking the absolute value, so 0 vs 2π is not a false failure).

**Critical interaction — read before implementing.** Spec:349 makes the effective limit `max(policy_tolerance, measured_floor)`. Under a *mean*, one π-flip in the A-vs-A baseline produced a floor of ~0.031. Under a per-row *max*, a **single** bistable head/tail π-flip in the A-vs-A run — the repo's documented noise floor on `ant_pose_headtail` (memory `project_migration_verification`) — sets the floor to π, and the angle test then passes literally everything. That would make this task a regression, not a fix.

Compute the floor as the **per-row max over the A-vs-A run, excluding rows whose head/tail categorical column differs**. Those rows are already caught by the exact categorical check, so excluding them loses no coverage while stopping a known-bistable row from disarming the whole geometry gate. Add a test asserting that a floor computed this way on a clip with one π-flip stays below 0.05. Update every construction site of `EquivalencePolicy` and the spec's §Correctness gate wording (the spec says "angular mean"; amend that line to "angular per-row max" and note the reason).

- [ ] **Step 7: Run the tests**

```bash
PYTHONPATH=$PWD/src python -m pytest tests/test_inference_autotune_equivalence.py -v
```
Expected: all PASS.

- [ ] **Step 8: Commit**

```bash
git add src/hydra_suite/core/inference/autotune/equivalence.py \
        tests/test_inference_autotune_equivalence.py \
        docs/superpowers/specs/2026-09-06-trackerkit-inference-autotuner-design.md
git commit -m "fix(autotune): track identity and per-row angle are exact in the correctness gate (B4)"
```

---

### Task 6: S1, S4, S5 — fallback envelope, honest down-admission, negative caching

**Failures being fixed:**
- **S1:** `worker.py:1252-1275` calls `_resolve_inference_autotune_before_load` with no enclosing `try`; only the inner `resolve_tracking_inference_config` is guarded (`integration.py:663-692`). Unguarded: `probe_runtime_resources` (`device.py:122-127` — missing `nvidia-smi` → `FileNotFoundError`; an `[N/A]` field → `ValueError`) and `AutotuneRequest.__post_init__` (`coordinator.py:46-50`, raises for a manual field the project lacks). A `--inference-autotune-manual pose_batch_size` on a project without pose kills the whole tracking run.
- **S4:** `candidates.py:236-256` builds `{selected, baseline, *successful_equivalent}` and admits the largest that fits — but `successful` (`coordinator.py:230-236`) includes evidence with `phase="stage"`. `search.py:76` states "stage screens never authorize a winner"; down-admission violates it.
- **S5:** a `budget_expired` or `timeout` outcome saves nothing (`coordinator.py:156-165`), so a project that cannot finish calibration burns the full budget on **every** run forever.

**Files:**
- Modify: `src/hydra_suite/core/tracking/worker.py:1252-1275`
- Modify: `src/hydra_suite/core/inference/autotune/coordinator.py:230-236`
- Modify: `src/hydra_suite/core/inference/autotune/store.py`
- Test: `tests/test_inference_autotune_search.py`, `tests/test_inference_autotune_store.py`

- [ ] **Step 1: Write the failing S1 test**

```python
def test_preflight_probe_failure_does_not_kill_the_run(monkeypatch):
    """A missing nvidia-smi must degrade to fallback, not raise."""
    import hydra_suite.core.inference.autotune.device as device
    monkeypatch.setattr(
        device, "probe_runtime_resources",
        lambda *a, **k: (_ for _ in ()).throw(FileNotFoundError("nvidia-smi")),
    )
    config, overlay, result = _call_resolve_before_load_with_autotune_on()
    assert overlay.status == "fallback"
```

- [ ] **Step 2: Write the failing S4 test**

```python
def test_down_admission_ignores_stage_only_evidence():
    """A stage screen must never authorize a production setting."""
    profile = _profile_with(
        selected=_settings(detection_batch_size=4),
        candidates=[_evidence(_settings(pose_batch_size=64), phase="stage", passed=True)],
    )
    result = AutotuneCoordinator(_store_with(profile), trial_executor=_never_called).resolve(
        _request(free_vram_too_small_for_selected=True)
    )
    assert result.overlay.effective.pose_batch_size != 64
```

- [ ] **Step 3: Write the failing S5 test**

```python
def test_budget_expiry_is_cached_so_the_next_run_does_not_retune():
    store = _memory_store()
    request = _request(budget_seconds=5, executor=_always_times_out)
    AutotuneCoordinator(store, trial_executor=_always_times_out).resolve(request)
    record = store.load(request.key)
    assert record is not None and record.state is ProfileState.INCOMPLETE
    second = AutotuneCoordinator(store, trial_executor=_never_called).resolve(request)
    assert second.overlay.status in {"fallback", "deferred_due_to_prior_failure"}
```

- [ ] **Step 4: Run all three, confirm they fail**

```bash
PYTHONPATH=$PWD/src python -m pytest tests/test_inference_autotune_search.py tests/test_inference_autotune_store.py -k "preflight or stage_only or budget_expiry" -v
```

- [ ] **Step 5: Implement S1 — wrap the preflight**

In `worker.py`, wrap the entire `_resolve_inference_autotune_before_load` call block in `try/except Exception`, logging with `logger.exception` and leaving `_inference_cfg` untouched plus a `fallback` overlay — mirroring the envelope already inside `integration.py:684-692`. The point is that the *builder* and the *probe*, not just the resolver, are inside the envelope.

- [ ] **Step 6: Implement S4 — filter to full-pipeline evidence**

In `coordinator.py:230-236`, add `and evidence.phase == "full"` to the `successful` comprehension. Then fix the existing test `test_cache_hit_down_admits_only_validated_settings` (`tests/test_inference_autotune_search.py:267-309`), which currently builds evidence with `phase="unknown"` and therefore documents the hole rather than guarding it: change it to assert that `phase="stage"` evidence is excluded and `phase="full"` evidence is included.

- [ ] **Step 7: Implement S5 — negative caching with a TTL**

Add `ProfileState.INCOMPLETE` to `store.py` **and amend spec:222** (which currently enumerates only validated/provisional) to describe the new state. On `budget_expired`/`timeout`/`baseline_measurement_incomplete`, save a record in that state carrying `last_attempt_unix_ns` and the failure reason. **Do not negative-cache when `contention_detected` was true** — a transient GPU neighbour must not buy a 24-hour lockout. In `coordinator.resolve`, if an `INCOMPLETE` record exists and `now - last_attempt < INCOMPLETE_RETRY_SECONDS` (default 24 h), return a baseline overlay with status `deferred_due_to_prior_failure` without calibrating. Persist and surface the reason in telemetry so the user can see why it is not tuning.

- [ ] **Step 8: Run the tests**

```bash
PYTHONPATH=$PWD/src python -m pytest tests/test_inference_autotune_search.py tests/test_inference_autotune_store.py tests/test_inference_autotune_coordinator.py -v
```
Expected: all PASS.

- [ ] **Step 9: Commit**

```bash
git add -A src/hydra_suite/core/inference/autotune src/hydra_suite/core/tracking/worker.py tests/
git commit -m "fix(autotune): preflight fallback envelope, full-phase-only down-admission, negative caching (S1/S4/S5)"
```

---

### Task 7: S2, S3 — fingerprint the baseline; stop self-invalidating on run 1

**Failures being fixed:**
- **S2:** run 1 has no detection cache, so `worker.py:~135-150` falls back to `_counts(None, MAX_TARGETS)` and keys the workload on `bucket(MAX_TARGETS)`. At the end of run 1, `worker.py:4590-4605` samples the now-populated cache and `store.py:224-233` compares real density to the key's → mismatch → PROVISIONAL. Run 2 builds a different key → miss → full re-calibration. Every new video double-tunes.
- **S3:** `TuningProfileKey` (`fingerprint.py:154-165`) has no baseline. Project A (baseline batch 1) tunes to 4; project B (baseline batch 8, same models/host/geometry) hits the cache and is *lowered* to 4 with source `validated_profile` — a comparison never made. `coordinator.py:226-229` then splices B's manual fields into A's `selected`, producing a joint vector no one measured, admitted only by the analytical memory model. This directly contradicts "joint bounded coordinate tuning."

**Files:**
- Modify: `src/hydra_suite/core/inference/autotune/fingerprint.py:154-165`
- Modify: `src/hydra_suite/core/inference/autotune/coordinator.py:226-229`
- Modify: `src/hydra_suite/core/inference/autotune/store.py:224-233`
- Modify: `src/hydra_suite/core/tracking/worker.py:4590-4605`
- Test: `tests/test_inference_autotune_fingerprint.py`, `tests/test_inference_autotune_store.py`

- [ ] **Step 1: Write the failing S3 test**

```python
def test_a_different_baseline_is_a_different_key():
    """A profile tuned from batch 1 must not be reused by a batch-8 project."""
    a = _key(baseline=_settings(detection_batch_size=1))
    b = _key(baseline=_settings(detection_batch_size=8))
    assert a.digest != b.digest
```

- [ ] **Step 2: Write the failing S2 test**

```python
def test_run_one_profile_survives_its_own_production_density_sample():
    """The first run's own density must re-key the profile, not demote it."""
    store, key = _store_with_validated_profile(keyed_on_max_targets=True)
    observe_production_throughput(store, key, observed_density=_density(p50=7, p95=9), throughput=1.0)
    reloaded = store.load(_key_for_density(p50=7, p95=9))
    assert reloaded is not None and reloaded.state is ProfileState.VALIDATED
```

- [ ] **Step 3: Run both, confirm they fail**

```bash
PYTHONPATH=$PWD/src python -m pytest tests/test_inference_autotune_fingerprint.py tests/test_inference_autotune_store.py -k "different_baseline or run_one_profile" -v
```

- [ ] **Step 4: Implement S3 — baseline in the key, and stop splicing**

Add a `baseline_digest: str` field to `TuningProfileKey`, computed from the sorted `(field_name, value)` pairs of the baseline settings, and include it in the key digest. Then delete the manual-field splice loop in `coordinator._reuse` (`:226-229`): with the baseline in the key, a cached profile already matches the requesting project's manual fields, so splicing can only produce an unmeasured vector. If `request.manual_fields` is non-empty, include the manual field *names* in the key too — a project pinning `pose_batch_size` searched a different space than one that did not.

- [ ] **Step 5: Implement S2 — re-key rather than demote**

In the production-observation path, when the only mismatch is the workload/density bucket **and** the key was built from the `MAX_TARGETS` fallback (mark this on the key as `density_is_estimated: bool`), re-save the validated profile under the corrected key instead of demoting it to PROVISIONAL. Demotion stays for a genuine density change on a key whose density was measured.

- [ ] **Step 6: Run the tests**

```bash
PYTHONPATH=$PWD/src python -m pytest tests/test_inference_autotune_fingerprint.py tests/test_inference_autotune_store.py tests/test_inference_autotune_search.py -v
```
Expected: all PASS. Note that adding key fields invalidates every existing on-disk profile — this is correct and the store's version cap handles it, but bump the store's schema version so old records are treated as misses rather than parsed.

- [ ] **Step 7: Commit**

```bash
git add -A src/hydra_suite/core/inference/autotune tests/
git commit -m "fix(autotune): baseline and manual fields are part of the profile key; run-1 density re-keys instead of demoting (S2/S3)"
```

---

### Task 8: S6, S7, S8 — trials must model the pipeline production will actually run

**Failures being fixed:**
- **S6:** `worker.py:~152-155` freezes detector fields and sets `RESULT_CACHE_STAGE_MASK=("detector",)` when detections are cached, but `apply_settings_to_params` (`sidecar.py:122`) unconditionally forces `USE_CACHED_DETECTIONS: False`. Every trial runs the detector while production replays the cache, so pose/identity screens are timed against detector-dominated wall time.
- **S7:** `device.py:111-120` runs `nvidia-smi --id=<first CUDA_VISIBLE_DEVICES entry or 0>`. nvidia-smi enumerates by PCI bus; CUDA uses `FASTEST_FIRST` unless `CUDA_DEVICE_ORDER=PCI_BUS_ID`. The child is not pinned (no `CUDA_VISIBLE_DEVICES` handling in `child_bootstrap.py`/`resource_limits.py`/`process_supervisor.py`). On diptera (9× RTX 6000 Ada) the accelerator UUID, VRAM, contention detection, and the memory watchdog can all describe a different physical GPU than the one tracking uses.
- **S8:** `sidecar_child.py:222-238` passes `detection_cache_path=cache_dir`, but `session.py:214-221` derives `inference_cache_dir=build_inference_cache_dir(video_path)` from the private symlink → `output/.inference_cache_source/`, not `run_root/inference-cache` where the engine wrote. Post-tracking consumes different caches than the ones just produced.

**Files:**
- Modify: `src/hydra_suite/core/inference/autotune/sidecar.py:122`
- Modify: `src/hydra_suite/core/inference/autotune/device.py:111-127`
- Modify: `src/hydra_suite/core/inference/autotune/child_bootstrap.py`
- Modify: `src/hydra_suite/core/inference/autotune/sidecar_child.py:222-238`
- Test: `tests/test_inference_autotune_sidecar.py`, `tests/test_inference_autotune_device.py`

- [ ] **Step 1: Write the failing S6 test**

```python
def test_trials_inherit_the_production_cached_detection_mode():
    """A cache-replaying production run must be screened against a replaying trial."""
    from hydra_suite.core.inference.autotune.sidecar import apply_settings_to_params
    params = apply_settings_to_params(
        {"USE_CACHED_DETECTIONS": True, "RESULT_CACHE_STAGE_MASK": ("detector",)},
        _settings(),
    )
    assert params["USE_CACHED_DETECTIONS"] is True
```

- [ ] **Step 2: Write the failing S7 test**

```python
def test_the_child_is_pinned_to_the_probed_device(monkeypatch):
    """The probed UUID and the child's visible device must be the same GPU."""
    env = _child_env(probe=_probe(accelerator_uuid="GPU-abc123"))
    assert env["CUDA_VISIBLE_DEVICES"] == "GPU-abc123"
    assert env["CUDA_DEVICE_ORDER"] == "PCI_BUS_ID"
```

- [ ] **Step 3: Run both, confirm they fail**

```bash
PYTHONPATH=$PWD/src python -m pytest tests/test_inference_autotune_sidecar.py tests/test_inference_autotune_device.py -k "cached_detection_mode or pinned_to_the_probed" -v
```

- [ ] **Step 4: Implement S6**

In `apply_settings_to_params`, stop forcing `USE_CACHED_DETECTIONS: False`; instead preserve the caller's value and preserve `RESULT_CACHE_STAGE_MASK`. The trial's *result* caches must still be written under the trial's own private root — verify with the e2e test that no trial output lands in the production `.inference_cache_<stem>/` (the spec requires "no candidate outputs contaminate production result caches", spec:470).

- [ ] **Step 5: Implement S7**

First run `grep -rn 'cuda:[0-9]\|set_device(\|torch.device(0' src/hydra_suite/core/inference/autotune/` — memory `project_measured_auto_batch_branch` records five bare-ordinal/UUID device sites in this codebase that failed silently, and pinning is only sufficient if the child does not then index a specific ordinal itself. Then, in `child_bootstrap.py`, set `CUDA_VISIBLE_DEVICES` to the probed accelerator **UUID** (CUDA accepts `GPU-<uuid>` strings, which are ordering-independent) and `CUDA_DEVICE_ORDER=PCI_BUS_ID` in the child environment. In `device.py`, resolve the nvidia-smi target from the same UUID rather than an ordinal. Then fix `test_cuda_probe_targets_the_process_visible_physical_device`, which currently enshrines the ordinal assumption.

- [ ] **Step 6: Implement S8**

Pass the engine's actual inference cache directory into post-tracking. Either set the sidecar's `video_path` so `build_inference_cache_dir` resolves to `run_root/inference-cache`, or thread an explicit override through `TrackingSessionCore`. Prefer the former — fewer moving parts, and it keeps the child's pipeline shape identical to production.

- [ ] **Step 7: Run everything, including the e2e anchor**

```bash
PYTHONPATH=$PWD/src python -m pytest tests/ -k "autotune" -v
PYTHONPATH=$PWD/src python -m pytest tests/test_inference_autotune_sidecar_child_e2e.py -v -m sidecar_e2e
```
Expected: all PASS.

- [ ] **Step 8: Commit**

```bash
git add -A src/hydra_suite/core/inference/autotune tests/
git commit -m "fix(autotune): trials mirror production cache mode, pin the probed GPU, use the engine's cache dir (S6/S7/S8)"
```

---

### Task 9: Minors — dead knobs, no-op side effects, unbounded growth, drift

**Files:**
- Modify: `src/hydra_suite/core/inference/autotune/models.py:165-174`
- Modify: `src/hydra_suite/trackerkit/engine_params.py:1144-1148`
- Modify: `src/hydra_suite/core/inference/autotune/store.py:101`
- Modify: `src/hydra_suite/core/inference/autotune/sidecar_child.py:56-83`
- Modify: `src/hydra_suite/trackerkit/config.py:1239`, `integration.py:500-501, 619`
- Test: `tests/test_inference_autotune_models.py`, `tests/test_inference_autotune_store.py`

- [ ] **Step 1: Write the failing no-op-overlay test**

```python
def test_a_no_op_overlay_does_not_disable_the_tile_batch_autotuner():
    """record/kept_current/cache-miss overlays must not silently kill SAHI tuning."""
    config = {"tile_batch_autotune": True}
    overlay = InferenceRuntimeOverlay.baseline(_settings(), status="recorded", reason="")
    assert overlay.apply(config)["tile_batch_autotune"] is True
```

- [ ] **Step 2: Run it, confirm it fails**

```bash
PYTHONPATH=$PWD/src python -m pytest tests/test_inference_autotune_models.py -k tile_batch -v
```

- [ ] **Step 3: Implement the fixes**

- `models.py:165-174`: set `tile_batch_autotune=False` only when the overlay actually changes a setting (status `calibrated`/`cache_hit`/`validated_down_admission`), never for `recorded`/`kept_current_settings`/`fallback`/`disabled`.
- `engine_params.py:1144-1148`: stop forcing `SLICE_TILE_BATCH_AUTOTUNE=False` for `mode == "automatic"` on platforms where automatic never applies (non-CUDA).
- `store.py:101`: prune `locks/` alongside the 512-record cap — delete lock files with no matching record and an mtime older than 7 days. Every code edit mints a new `hydra_code_identity` and therefore a new lock file.
- `sidecar_child.py:56-83`: delete the hand-copied CSV header and call `headless_tracking.build_tracking_csv_header` directly.
- `sidecar.py:283-286`: OOM-adapted retries are cached and returned out of block order, breaking the paired-block structure. Return retries in their original block position or discard the candidate with a recorded reason.
- Dead knobs — either wire or delete, no middle ground: `INFERENCE_AUTOTUNE_SINGLEFLIGHT_WAIT_SECONDS` (`config.py:1239`, read but never emitted), `INFERENCE_AUTOTUNE_STAGE_SHARES` (`integration.py:619`), `INFERENCE_AUTOTUNE_CUDA_VERSION`/`CUDNN_VERSION` (`integration.py:500-501` — these fall back to pip metadata and read "absent" on a conda CUDA install, which is the common case here, so fix the lookup to use `torch.version.cuda`/`torch.backends.cudnn.version()`).
- The `streaming`/`cache_replay` policy branches are unreachable (`TrackingRunContext.execution_mode` is only ever `batch`/`realtime` from `worker.py:~166`). Delete the dead branches or wire `cache_replay` from `self.cache_read_only_replay`; prefer wiring, since the spec calls for a result-cache-hit gate leg (spec:469).

- [ ] **Step 4: Expose `record` mode and the budget in the GUI**

`record` is currently unreachable from the GUI — a config set to `record` renders the checkbox unchecked, and toggling it destroys the value. Replace the checkbox with a three-state control (`off` / `record` / `automatic`) and add a spinbox for `inference_autotune_budget_seconds` (persisted but widget-less today). Follow the kit's typed-schema pattern in `trackerkit/config/schemas.py`.

- [ ] **Step 5: Refresh the stale characterization golden**

The regen in `7cf7e8c9` added `SLICE_TILE_BATCH_SIZE`/`SLICE_MEMORY_BUDGET_MIB`/`SLICE_TILE_BATCH_AUTOTUNE`, which main already emitted at `engine_params.py:1113-1115` — the golden was stale before this branch. Regenerate it and confirm the diff contains only autotune keys.

- [ ] **Step 6: Run the affected tests**

```bash
PYTHONPATH=$PWD/src python -m pytest tests/ -k "autotune or engine_params or golden" -v
```

- [ ] **Step 7: Commit**

```bash
git add -A
git commit -m "fix(autotune): no-op overlays keep tile autotuning, prune locks, wire or delete dead knobs, expose record mode"
```

---

### Task 10: MPS gates — the real equivalence run with autotune enabled

**Why this matters:** `grep -rn INFERENCE_AUTOTUNE tools/equivalence/` returns nothing today. The harness runs `mode=off`, so a green gate says **nothing** about this feature. The spec (:465-477) requires real gates covering detection-only, sliced detection, pose/head-tail, CNN identity, sequential OBB, and a result-cache hit.

**Files:**
- Modify: `tools/equivalence/run_matrix.sh` (new optional autotune leg)
- Create: `tools/equivalence/configs/fly_obb_roi.json` (fly_obb with a non-empty `roi_shapes`, to give the ROI path gate coverage)

- [ ] **Step 1: Add an ROI fixture config**

Copy `tools/equivalence/configs/fly_obb.json` to `fly_obb_roi.json` and give it a non-empty `roi_shapes` covering most of the frame (so detections are not suppressed). Register the clip in `run_matrix.sh`'s clip list. This closes the coverage hole that let B1 ship.

- [ ] **Step 2: Add the autotune leg — forced settings, NOT a seeded profile**

A committed seed profile can never hit the cache: `hydra_code_identity` hashes all package sources, so the key changes with every edit, and Task 7 adds `baseline_digest` plus manual-field names on top. Seeding would silently miss and the leg would compare defaults against defaults.

The leg isolates the two questions separately:

**(a) The settings question — the headline number.** Run each clip twice with `INFERENCE_AUTOTUNE_MODE=off` both times, but the second run's config forced to the tuned vector directly (`detection_batch_size`, `pipeline_depth`, pose/identity batch sizes). No key, no cache, no autotune code in the path. Compare both `_forward.csv` and `_tracking_final.csv`; assert row counts > 1 on both sides. This answers "do tuned settings perturb output" with zero confounds.

**(b) The plumbing question.** One run in `automatic` mode with a profile seeded **in-process** — built via the same key builder immediately before launch, in the same interpreter, so the key matches by construction. Do not compare CSVs here; assert only that the worker's log line (`worker.py:1279-1285`) reports `status=cache_hit` and an `effective=` that differs from `requested=`.

- [ ] **Step 3: Clear caches and run the standard (non-autotune) matrix first**

```bash
cd /Users/neurorishika/.codex/worktrees/6e86/multi-animal-tracker
conda activate hydra-mps
find . -name __pycache__ -prune -exec rm -rf {} +
pgrep -fl "sleap|hydra" # kill only stale sleap/hydra
git worktree add --detach .worktrees/equiv-base 0d4d4cae
REPO=$PWD WT=$PWD MAIN_SRC=$PWD/.worktrees/equiv-base/src WT_SRC=$PWD/src \
  OUT=/tmp/equiv_autotune RUNTIME=mps bash tools/equivalence/run_matrix.sh
```
The baseline is **local main at the merge base (`0d4d4cae`)**, not the `legacy/main` tag. Legacy-vs-branch conflates ~40 merged features with this one; the no-op claim needs attribution to this branch alone. Run the standard `legacy/main` matrix too if you want the repo's usual drift check, but it does not substitute.

Expected: every clip EQUIVALENT at its determinism floor with autotune off — proving the branch changes nothing when the feature is disabled. Verify `wc -l` > 1 on every CSV before trusting any verdict.

- [ ] **Step 4: Run the autotune leg**

```bash
REPO=$PWD WT=$PWD MAIN_SRC=$PWD/src WT_SRC=$PWD/src AUTOTUNE=1 \
  OUT=/tmp/equiv_autotune_on RUNTIME=mps bash tools/equivalence/run_matrix.sh
```
This is a current-vs-current comparison: `mode=off` vs `mode=automatic` with a forced profile. Record, per clip, whether output is byte-identical or merely within the gate's tolerance. **This is the headline number for the merge decision** — it tells you empirically whether tuned settings perturb MPS output at all.

- [ ] **Step 5: Record the results in the plan**

Write the per-clip verdicts into a `## MPS gate results` section at the bottom of this file, with the date, the commit sha, and the exact deltas. Do not summarize as "passed" — record the numbers.

- [ ] **Step 6: Clean up and commit**

```bash
git worktree remove --force .worktrees/equiv-base && git worktree prune
git add tools/equivalence docs/superpowers/plans/2026-09-07-inference-autotuner-review-remediation.md
git commit -m "test(equivalence): add ROI clip and an autotune-enabled gate leg; record MPS results"
```

---

### Task 11: Quality gates and full-suite delta

- [ ] **Step 1: Format and lint**

```bash
make commit-prep
make lint-moderate
```
Fix anything the branch introduced. The 11 pre-existing violations listed in the implementer's summary (identity_panel.py:1184, the four test files, perf_benchmark.py) are **not** this branch's to fix — confirm each is pre-existing by checking it against the branch base, not by reverting your own files.

- [ ] **Step 2: Full autotune + inference subsystem suite**

```bash
PYTHONPATH=$PWD/src python -m pytest tests/ -k "autotune" -v
PYTHONPATH=$PWD/src python -m pytest tests/ -k "inference" -v
```
Expected: green. Record exact counts.

- [ ] **Step 3: Delta gate against main, batched**

The whole suite hangs in unchanged Qt teardown (memory `project_main_suite_blockers`), so batch per-file and compare **failure sets**, not counts — new test files shift chunk boundaries and fake collection errors (memory `project_test_suite_batching_chunk_boundary_trap`). Run the same batching on main and on the branch, and assert the branch's failure set is a subset of main's.

- [ ] **Step 4: Docs**

```bash
make docs-check
```
The global docs-quality metric fails against a stale 100% baseline repo-wide — that is pre-existing. Confirm the strict mkdocs build passes and that the autotuner has a real `docs/developer-guide/` page (the branch's "docs" commit `7ca072aa` touched only docstrings).

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "chore(autotune): formatting, lint, and docs page"
```

---

### Task 12: CUDA gates on courtship (correctness) and firebrat (timing)

**Box survey, 2026-09-07 — mehek is NOT needed:**

| Box | GPU | Free? | `hydra-cuda`? | Role |
|---|---|---|---|---|
| `mehek` | (in use) | No — SAM3 LoRA PID 3487055, ~19.7 GB | yes | **Do not touch.** |
| `courtship` | RTX 4090, 24 GB | Yes, after cleanup | yes — torch 2.11.0+cu128, `torch.cuda.is_available()` True; repo at `~/hydra-suite`; `sleap-nn` env present | **Primary: correctness gates.** |
| `firebrat` | RTX 4090, 24 GB | Yes — 1.9 GB used, **zero** compute processes | **no conda envs at all** | **Timing box** (needs `make setup-cuda` first). |

`courtship` currently holds **four orphaned SLEAP service processes** (PIDs 2653980, 2654475, 2658963, 2659744) that have been alive 20+ hours and hold ~9.5 GB. These are exactly the stale `sleap`/`hydra` processes CLAUDE.md instructs you to clear before a heavy run — kill these four and nothing else.

**Note on the earlier `courtship` caveat:** memory `project_courtship_box_traps` recorded that courtship's `sleap` env is TF SLEAP 1.4.1 (unusable for pose work). That is still true of the env named `sleap`, but courtship now *also* has a separate `sleap-nn` env, so pose/SLEAP clips are runnable there. Update that memory when this task completes.

**Authorization:** the user authorized a **source-only** snapshot on 2026-09-07. Do not transfer branch history.

- [ ] **Step 1: Clear the stale SLEAP services on courtship**

```bash
ssh rutalab@courtship.taild08eb9.ts.net \
  'for p in 2653980 2654475 2658963 2659744; do
     ppid=$(ps -o ppid= -p $p | tr -d " ");
     echo "pid=$p ppid=$ppid parent=$(ps -o cmd= -p $ppid 2>/dev/null | cut -c1-80)";
   done'
```
A PID whose parent is **alive and is a `trackerkit`/`hydra` process** is somebody's running job — do not kill it. An orphan (parent gone or reparented to init) is the known bug from memory `project_batch_gpu_fanout`. Only after confirming orphan status:
```bash
ssh rutalab@courtship.taild08eb9.ts.net 'kill 2653980 2654475 2658963 2659744; sleep 5; nvidia-smi'
```
Kill **only** these four. Re-check PIDs before killing — they may have changed.

- [ ] **Step 2: Transfer a source-only snapshot**

```bash
cd /Users/neurorishika/.codex/worktrees/6e86/multi-animal-tracker
git archive --format=tar HEAD src tools tests pyproject.toml Makefile | gzip > /tmp/autotune-src.tgz
scp /tmp/autotune-src.tgz rutalab@courtship.taild08eb9.ts.net:/tmp/
```
`git archive` of a single tree carries no history — this is what was authorized. Do NOT use `git bundle`.

- [ ] **Step 3: Fetch fixtures on courtship (once)**

```bash
ssh rutalab@courtship.taild08eb9.ts.net
mkdir -p ~/autotune-gate && tar xzf /tmp/autotune-src.tgz -C ~/autotune-gate
source ~/anaconda3/etc/profile.d/conda.sh && conda activate hydra-cuda
export KMP_DUPLICATE_LIB_OK=TRUE
cd ~/autotune-gate && bash tools/equivalence/fixtures/fetch_fixtures.sh
ls tools/equivalence/fixtures/clips/   # must be non-empty before proceeding
```

- [ ] **Step 4: Standard matrix (autotune off) on courtship**

```bash
cd ~/hydra-suite && git fetch origin --tags
git worktree add --detach .worktrees/equiv-legacy legacy/main
find ~/autotune-gate -name __pycache__ -prune -exec rm -rf {} +
REPO=$PWD WT=~/autotune-gate MAIN_SRC=$PWD/.worktrees/equiv-legacy/src \
  WT_SRC=~/autotune-gate/src OUT=/tmp/equiv_autotune RUNTIME=cuda \
  nohup bash ~/autotune-gate/tools/equivalence/run_matrix.sh > /tmp/equiv_cuda.log 2>&1 &
```
Expected: every clip EQUIVALENT at its determinism floor — proving the branch is a no-op when disabled. **Verify `wc -l` > 1 on every CSV** before trusting a verdict; a bare shell yields empty CSVs that falsely compare EQUIVALENT.

Known pre-existing hazard: memory `project_cuda_main_vs_legacy_yolo_obb_divergence` records that `fly_obb`/`ant_obb_sleap` already diverge on CUDA vs the `legacy/main` tag, independent of this branch. If those clips diverge, re-run with `MAIN_SRC` set to **local main** rather than `legacy/main` to isolate this branch's contribution.

- [ ] **Step 5: Autotune leg on courtship**

```bash
REPO=$PWD WT=~/autotune-gate MAIN_SRC=~/autotune-gate/src WT_SRC=~/autotune-gate/src \
  AUTOTUNE=1 OUT=/tmp/equiv_autotune_on RUNTIME=cuda \
  bash ~/autotune-gate/tools/equivalence/run_matrix.sh
```
Current-vs-current: `mode=off` vs `mode=automatic` with the pre-seeded profile from Task 10. This is the empirical answer to whether tuned settings perturb CUDA output — the question the spec (:354) answered anecdotally with one identity row at batch 8.

- [ ] **Step 6: Live calibration on courtship**

Run one **genuine** `automatic`-mode calibration on the ROI clip. This is the first time `sidecar_child` will ever have executed on CUDA in production shape. Capture the full decision trace: candidates measured, candidates rejected and why, the winner, and the measured gain.

- [ ] **Step 7: Stand up firebrat and re-run timing there**

courtship is a shared box; its numbers are correctness-valid but timing-suspect. firebrat has a genuinely idle 4090 and is the only place a throughput claim can be trusted — this repo has been burned twice by benchmarks taken under the wrong conditions (memory `project_crossframe_batching_spike`).

```bash
ssh rutalab@firebrat.taild08eb9.ts.net
# no conda present -- install mambaforge, then:
cd ~/hydra-suite || git clone <repo> ~/hydra-suite
make setup-cuda && conda activate hydra-cuda && make install-cuda
bash tools/equivalence/fixtures/fetch_fixtures.sh
```
Budget several hours for this; it is environment setup, not debugging. Then repeat Steps 2, 5, and 6 on firebrat and treat **firebrat's** numbers as the performance record, courtship's as correctness only.

- [ ] **Step 8: Check the results against the spec's regression evidence**

Compare the live calibration decisions against spec:477-489: batch 4 should beat batch 1 by ≥10% at 1200 px without selecting 8/16; the 4512 px SAHI search must not prefer batch 2; frame-batch 16/32 must be pruned *before* inference; the pose pipeline must reject the depth-1 composition; the identity pipeline must retain the baseline when alternatives regress. A spec expectation that is not met is a **finding** — record it; do not adjust the expectation.

Caveat to state plainly in the write-up: mehek is an RTX 6000 Ada and the spec's numbers came from it. courtship/firebrat are RTX 4090s. Different VRAM, different SM count — a different winner is not automatically a regression, but the *shape* of the curve should agree.

- [ ] **Step 9: Record the numbers**

Append a `## CUDA gate results` section to this plan: per-clip verdicts, the full calibration decision trace, which box produced each number, and whether each spec expectation held.

- [ ] **Step 10: Clean up both boxes**

```bash
ssh rutalab@courtship.taild08eb9.ts.net 'rm -rf ~/autotune-gate /tmp/autotune-src.tgz; cd ~/hydra-suite && git worktree remove --force .worktrees/equiv-legacy && git worktree prune'
# same on firebrat
```

- [ ] **Step 11: Update the courtship memory**

`project_courtship_box_traps` says no pose/SLEAP work can run on courtship. Amend it: the `sleap` env is still TF 1.4.1, but a separate `sleap-nn` env exists and `hydra-cuda` is functional (torch 2.11.0+cu128).

---

### Task 13: Second adversarial review, then merge

- [ ] **Step 1: Re-run the adversarial reviewer**

Dispatch a fresh Fable adversarial review against the remediated branch, giving it this plan and the original review so it can check each finding is genuinely closed rather than narrowed. Per the repo convention (memory `feedback_adversarial_review_before_merge`), a different model before a big merge finds what the normal chain misses.

- [ ] **Step 2: Triage using the receiving-code-review skill**

Use `superpowers:receiving-code-review`. Verify each finding technically before implementing — do not perform agreement.

- [ ] **Step 3: Merge**

```bash
cd /Users/neurorishika/Projects/Rockefeller/Kronauer/multi-animal-tracker
git merge --no-ff codex/trackerkit-inference-autotuner
```
In the merge commit, `git mv` the spec to `docs/superpowers/specs/done/` and this plan to `docs/superpowers/plans/done/`, and stamp the spec's status header with `Shipped — merged to main (<sha>)`.

- [ ] **Step 4: Post-merge cleanup**

Remove the worktree and delete the local branch **only** after confirming the foreign `stash@{0}` is still intact and belongs to someone else's branch.

---

## Merge-readiness criteria

The branch is merge-ready only when **all** of the following hold. Any unchecked item means it is not.

- [ ] B1–B4 fixed, each with a test that fails on the pre-fix commit.
- [ ] S1–S9 fixed or explicitly deferred with a written reason in this file.
- [ ] `sidecar_child` executes end-to-end in CI-runnable tests on an ROI project.
- [ ] MPS: autotune-off equivalence at the determinism floor on every clip.
- [ ] MPS: autotune-on leg run, per-clip deltas recorded as numbers.
- [ ] CUDA correctness: both legs run on `courtship`, plus one live calibration whose decision trace is recorded.
- [ ] CUDA timing: throughput numbers taken on the idle `firebrat` box, not the shared `courtship`.
- [ ] Spec regression expectations (spec:477-489) checked against the live CUDA calibration.
- [ ] Second adversarial review clean or its findings addressed.
- [ ] Docs page exists; spec and plan moved to `done/` in the merge commit.

**CUDA is no longer blocked.** `courtship` (RTX 4090, `hydra-cuda` working, repo present) can run the correctness gates today once four orphaned SLEAP services are cleared; `firebrat` (idle RTX 4090) is the clean-timing box but needs a `make setup-cuda` first. `mehek` stays untouched. Until Task 12's boxes are checked this branch is **not merge-ready**, regardless of how green MPS looks — `automatic` mode is CUDA-only by policy (`integration.py:606-611`), so CUDA is the *only* platform where this feature affects production at all.

---

## MPS gate results

**Date:** 2026-09-07 · **Branch:** `codex/trackerkit-inference-autotuner` · **Branch HEAD under test:** `095e1863` · **Baseline:** local `main` at the merge base `0d4d4cae` (NOT the `legacy/main` tag — legacy-vs-branch conflates ~40 merged features with this one, and the no-op claim needs attribution to this branch alone) · **Box:** Apple Silicon, `hydra-mps`, `RUNTIME=mps` · **Harness commit:** `3366e4d8`

All numbers below are `tools/equivalence/compare.py` output: `pos |Δ| p99` (px), `θ |Δ| max` (rad), row counts, unmatched. Every CSV was checked for `wc -l > 1` before any verdict was read; no comparison in this section rests on a header-only file. Raw logs: `/tmp/equiv_noop.log`, `/tmp/equiv_noop2.log`, `/tmp/equiv_baseAA.log`, `/tmp/equiv_3a.log`, `/tmp/equiv_3a2.log`.

### 1. The no-op proof — autotune OFF, `0d4d4cae` vs `095e1863`

`EQUIVALENCE legacy vs new_a`, per clip, per CSV kind:

| clip | forward (p99 / θmax / rows / unmatched) | final | final_with_individual |
|---|---|---|---|
| `ant_pose_headtail` | 0.000e+00 / 0.000e+00 / 12500=12500 / 0 | 0 / 0 / 9096=9096 / 0 | 0 / 0 / 9096=9096 (55 cols) / 0 |
| `worm_bgsub` | 0 / 0 / 5000=5000 / 0 | 0 / 0 / 2706=2706 / 0 | 0 / 0 / 2706=2706 / 0 |
| `worm_bgsub_scaled` | 0 / 0 / 5000=5000 / 0 | 0 / 0 / 1002=1002 / 0 | 0 / 0 / 1002=1002 / 0 |
| `ant_cnn_identity` | 0 / 0 / 12225=12225 / 0 | 0 / 0 / 10201=10201 / 0 | 0 / 0 / 10201=10201 (73 cols) / 0 |
| `ant_cnn_identity_relink` | 0 / 0 / 12225=12225 / 0 | 0 / 0 / **10221 vs 10213** / **8** ❌ | same ❌ |
| `fly_obb` | 0 / 0 / 1494=1494 / 0 | 0 / 0 / 1500=1500 / 0 | 0 / 0 / 1500=1500 / 0 |
| `fly_obb_roi` (new) | 0 / 0 / 1494=1494 / 0 | 0 / 0 / 1500=1500 / 0 | 0 / 0 / 1500=1500 / 0 |

Every matched row is bit-for-bit identical: `pos max`, `pos mean`, `pos p99`, `θ max` and `θ mean` are all exactly `0.000e+00` on every clip and every CSV kind. Not "within tolerance" — zero. **Confirmed at the byte level**, not merely numerically: `cmp` on all 21 legacy/new_a CSV pairs reports byte-identical files for 19 of 21; the 2 that differ are `ant_cnn_identity_relink`'s `final` and `final_with_individual`, i.e. exactly the pre-existing nondeterminism below.

**The one red line is not this branch.** `ant_cnn_identity_relink`'s final CSV differs by 8 rows. Re-running that clip with `MAIN_SRC = WT_SRC = 0d4d4cae/src` — i.e. all three runs on the *baseline* tree, branch code absent entirely — reproduces the identical 10221-vs-10213 / 8-unmatched split (`/tmp/equiv_baseAA.log`). In the 3a leg the same clip produced the split in the *opposite direction* (`new_a` 10213, `new_b` 10221) with both runs on the same tree and the same settings. It is pre-existing run-order-dependent nondeterminism in that clip's relink/stitch postprocessing, independent of source tree and of settings. The 10213 rows common to both are byte-identical. **Not attributable to this branch; pre-existing bug, worth its own investigation.**

**Not measured:** `ant_obb_sleap` and `ant_obb_sequential`. Not run to completion in this session (throughput on a contended box). They are **not** covered by the numbers above and must not be read as passing.

### 2. Harness defect found while doing this

`run_matrix.sh`'s `cmp()` calls `note_failure` only for compare.py exit code 2 ("no data"), never for exit code 1 ("real differences"). The no-op matrix printed two `VERDICT: DIFFERENCES ❌` lines and still exited **0** with "all clips produced comparable output." A red verdict does not fail the matrix. Left unchanged here (out of scope, and changing it mid-measurement would have muddied attribution), but it means **the matrix's exit code is not a gate** — someone must read the verdicts.

### 3a. The settings question — the headline number

Both sides `INFERENCE_AUTOTUNE_MODE=off`, same source tree, second/third runs forced to a tuned vector via `runner.py`'s `--*-batch-size` / `--pipeline-depth`. No key, no cache, no autotune code in the path.

**The vector was verified live, not merely echoed into `equiv_config.json`.** Resolving each config through `build_tracking_parameters → build_inference_config_from_params → InferenceTuningSettings.from_config` — the exact object the tuner mutates — shows the change reaching the pipeline:

- `fly_obb`: `{det 1, depth 2}` → `{det 8, depth 4}`
- `ant_pose_headtail`: `{det 1, pose 25, headtail 25, depth 2}` → `{det 2, pose 8, headtail 8, depth 2}`
- `ant_cnn_identity`: `{det 1, pose 25, headtail 25, identity{colortag: 8}, depth 2}` → `{det 2, pose 8, headtail 8, identity{colortag: 32}, depth 2}`

`SETTINGS default vs tuned`, per clip:

| clip | tuned vector | forward p99 / θmax | final p99 / θmax | rows |
|---|---|---|---|---|
| `worm_bgsub` | det 8, depth 4 | 0.000e+00 / 0.000e+00 | 0 / 0 | 5000, 2706 identical |
| `worm_bgsub_scaled` | det 8, depth 4 | 0 / 0 | 0 / 0 | 5000, 1002 identical |
| `fly_obb` | det 8, depth 4 | **6.104e-05 / 3.815e-06** | 0 / **3.815e-06** | 1494, 1500 identical, 0 unmatched |
| `fly_obb_roi` | det 8, depth 4 | **6.104e-05 / 3.815e-06** | 0 / **3.815e-06** | 1494, 1500 identical, 0 unmatched |
| `ant_cnn_identity` | det 2, pose/ht 8, cnn 32, depth 2 | 0 / 0 | 0 / 0 | 12225, 10201 identical; identity columns (73 cols) identical |
| `ant_cnn_identity_relink` | same | 0 / 0 | 0 / 0 | identical |
| `ant_pose_headtail` | det 2, pose/ht 8, depth 2 | 1.221e-04 / **9.360e-02** | 0 / **9.360e-02** | 12500, 9096 identical, 0 unmatched |

**Verdict: tuned settings are NOT universally output-neutral on MPS, and one clip exceeds the tuner's own tolerance.**

- bg-sub and CNN-identity clips: exactly zero. Batch size and pipeline depth are bit-exact there.
- YOLO-OBB (`fly_obb`, `fly_obb_roi`): nonzero but float-noise — `pos p99 = 6.1e-05 px` against a 0.5 px tolerance (~8000× inside), `θ max = 3.8e-06 rad` against 0.05 rad (~13000× inside). Batching YOLO changes results at the last few ULPs. Real, tiny, and comfortably admissible.
- **`ant_pose_headtail`: `θ max = 9.360e-02 rad` (≈5.4°) — ABOVE the autotuner's `EquivalencePolicy.angle_max_tolerance` of 0.05 rad.** Localised precisely: **exactly 1 row of 9096**, at **FrameID 63** — *inside* the 128-frame calibration window, so the tuner's gate would in fact see it and reject the candidate. Attribution is not fully isolated: the vector moved `detection_batch_size` 1→2 *and* `pose_batch_size`/`headtail_batch_size` 25→8 together. `fly_obb`'s pure detection-batch change (1→8) produced only 3.8e-06 rad, so the pose/head-tail batch is the **likely** cause, but a single-variable run was not done. Direct evidence that **pose batch size is probably not output-neutral**; pose batch fields must never be admitted on evidence weaker than the full equivalence gate. Note `compare.py` still printed `EQUIVALENT ✅` for this clip because its criterion is `θ mean`, not `θ max`; only the max exposes it.

**Also found (pre-existing, correct behaviour):** the first 3a attempt used `det 8 / depth 4` on the large-frame ant clips and every tuned run **died** with `ValueError: Inference pipeline frame buffer is not resource-admissible: one frame=61074432 bytes, batch=8, live_windows=6, estimated=2931572736 bytes exceeds the 536870912-byte pipeline budget`. That guard is in `core/inference/pipeline.py` (untouched by this branch). It is not a defect and not tuner-reachable: `CandidatePlanner.admit` uses `retained_windows = pipeline_depth + 2`, which equals the pipeline's `queue_bound + 3`, so the planner rejects exactly the vectors the pipeline rejects. The two formulas agree. The vector was re-chosen to the largest admissible one for those clips (`det 2 / depth 2`) and the table above reports that run.

### 3b. The plumbing question — unsatisfiable on MPS by policy

`tools/equivalence/autotune_plumbing_probe.py`, profile seeded **in process** by wrapping the live `build_tracking_autotune_request`, so the key matches by construction (`profile_seeded: true`, key digest `78c0388e03dd5f2e…`, `selected.detection_batch_size = 4` vs `baseline = 1`). Real tracking run on `fly_obb`, `automatic` mode. The worker's own line:

```
Inference throughput autotuner: status=deferred_due_to_contention profile=None
  requested={'detection_batch_size': 1, ..., 'pipeline_depth': 2}
  admitted  ={'detection_batch_size': 1, ..., 'pipeline_depth': 2}
  effective ={'detection_batch_size': 1, ..., 'pipeline_depth': 2}
  reason=automatic inference tuning is validated only for CUDA
```

`eligible=false`, `allow_cached_reuse=false`. `integration.py` refuses `automatic` on any non-CUDA accelerator *before* the store is consulted, so `AutotuneCoordinator.resolve` can never reach its `cache_hit` branch here. **`status=cache_hit` with `effective != requested` is unreachable on MPS.** It was not forced by patching `accelerator_kind` — that would have tested a fiction. **This assertion belongs to Task 12 (CUDA).**

Consequence worth stating plainly: **on MPS this branch cannot change effective settings in any mode.** The no-op proof in §1 therefore *is* the MPS merge criterion, and §3a measures the pipeline's batch-invariance rather than the tuner's behaviour.

### Carried from Task 5 — the angular determinism floor

Measured with the **autotuner's own** gate (`autotune/equivalence.py`, per-row angular max after head/tail-flip exclusion), not `compare.py`'s `θ mean`, via `tools/equivalence/autotune_determinism_floor.py` over the A-vs-A pair of every clip:

| clip | angular floor (rad) | tolerance | tunable? | pos p99 |
|---|---|---|---|---|
| `ant_pose_headtail` | **0.000e+00** | 0.05 | yes | 0.000e+00 |
| `fly_obb` | **0.000e+00** | 0.05 | yes | 0.000e+00 |
| `fly_obb_roi` | **0.000e+00** | 0.05 | yes | 0.000e+00 |
| `worm_bgsub` | 0.000e+00 | 0.05 | yes | 0.000e+00 |
| `worm_bgsub_scaled` | 0.000e+00 | 0.05 | yes | 0.000e+00 |
| `ant_cnn_identity` | 0.000e+00 | 0.05 | yes | 0.000e+00 |
| `ant_cnn_identity_relink` | 0.000e+00 | 0.05 | yes | 0.000e+00 |

**No clip's own determinism run exceeds 0.05 rad — every clip measured is tunable, and the Task 5 risk did not materialize on MPS.** The gate was not softened. `ant_obb_sequential` was not measured (see §1). Caveat on the tool's other columns: its `unmatched_rows` / `passed` fields count forward-CSV rows with NaN X/Y that never positionally match, and for streaming-pose clips the forward slot is a synthetic empty frame; only `angle_max` and `position_p99` are meaningful here, and those are the numbers the carried requirement asks for.

### CORRECTION + NEW BLOCKER — `record` mode DOES calibrate on MPS, and the tuner can never tune anything

Two claims made earlier in this section were **wrong** and are corrected here rather than edited away.

**Correction 1: `record` mode is eligible on MPS.** The CUDA-only refusal in `integration.py` is scoped to `elif policy.mode == "automatic" and ... not CUDA`. `record` falls through to `planner.admit(baseline)` and gets `eligible=True`; only `allow_cached_reuse` is forced off. `worker.py:226` always passes a real `ContainedTrialExecutor`. So a record-mode run on MPS *does* launch the sidecar calibration child on real video. Verified empirically: a `record` run on `fly_obb_roi` logged `Optimizing inference (bounded calibration)` and `Optimizing inference — baseline; incumbent detection_batch_size=1,pipeline_depth=2; 0.0/600s`. **This is the sidecar path, and it runs on this platform.** "On MPS *nothing* exercises the sidecar" and "Task 12 is the only test of this feature that can exist" were both false.

**Correction 2 (BLOCKER): the tuner's determinism floor fails on byte-identical output, so calibration always aborts.** That same record run ended:

```
status=fallback  effective == requested == baseline
reason=baseline_nondeterministic_beyond_contract
```

`search.py:141` returns that reason when `compare_outputs(baseline_run_1, baseline_run_2, for_determinism_floor=True).passed` is False. Feeding that function **two byte-identical CSV pairs** (verified with `filecmp.cmp(..., shallow=False) == True`) returns:

```
passed=False  unmatched=6  position_p99=0.0  angle_max=0.0
details: "forward: keyed rows are not aligned", "forward: 6 unmatched positional rows"
```

Root cause: `_positional` matches rows by nearest (X, Y) and **cannot match a row whose X/Y is NaN** — a lost or coasting track. `fly_obb`'s forward CSV has 1494 rows of which exactly **3** have NaN X/Y; 3 per side × 2 sides = the 6 unmatched. `_compare_one` requires `unmatched_rows == 0`, so the verdict fails on a perfectly deterministic baseline, and `len(pairs) != len(reference)` additionally trips "keyed rows are not aligned".

**Consequence: any project whose forward output contains even one NaN-position row — i.e. essentially every real tracking run — can never pass the tuner's own determinism floor, so the search aborts before evaluating a single candidate and the feature is a permanent no-op.** This is platform-independent: it is arithmetic in `equivalence.py`, not a device behaviour, so CUDA will hit it identically. It is also why my `autotune_determinism_floor.py` printed `passed=False` for all 7 clips; I initially dismissed that as a tool artifact, and **that dismissal was wrong** — it was the product reporting its own defect. `angle_max` and `position_p99` in that table remain correct.

Fix direction (not implemented here): exclude NaN-position rows from the unmatched count symmetrically, or match them by row key instead of position, in `_positional`/`_compare_one`.

**FIXED by Task 10b.** `_positional` now pairs NaN-position rows by row key (`_nan_position_pairs`) and feeds those pairs to `_mandatory_exact_mismatches`, so they are excluded from the *distance* population but still compared exactly on `TrackID`/`TrajectoryID`/`State`/`ArenaID` and on the categorical columns. A second defect was found in the same place: `_aligned` compared key values with `==`, which is False for the NaN `DetectionID` a lost track carries, so keyed alignment returned `None` on every real output — silently disabling the NaN-pattern and heuristic-categorical checks. Both fixed; keys are now compared NaN-safely. Re-running `autotune_determinism_floor.py` over `/tmp/equiv_autotune/mps` flips **all 7 clips** from `passed=false, unmatched>0` to `passed=true, unmatched_rows=0, nan_pattern_mismatches=0, categorical_mismatches=0`, with `position_p99` and `angle_max_rad` unchanged at `0.0`. See `.superpowers/sdd/.../task-10b-report.md`.

### Carried from Task 8 — S8 still has no coverage, for a different reason

The S8 fix (post-tracking consumes the trial's own cache directory, not production's) is exercised only inside the **sidecar calibration child's trial**. 3a runs with `mode=off` (no autotune code in the path at all) and 3b is refused before calibration starts. The `record` run above *did* reach the sidecar — but it aborted at the baseline determinism floor **before running any trial**, so it never got as far as the code S8 touches. **S8 remains empirically unverified.** It cannot be verified on any platform until the determinism-floor blocker above is fixed, because no trial ever executes. Stated explicitly rather than implied by silence.

### Performance

`PERF_TOLERANCE=1.25`, new/legacy wall-clock, autotune-off matrix: `worm_bgsub` 1.02x ✅, `worm_bgsub_scaled` 1.07x ✅, `ant_cnn_identity_relink` 0.91x ✅, `fly_obb` 0.92x ✅, `fly_obb_roi` 1.00x ✅, `ant_pose_headtail` **1.28x ❌**, `ant_cnn_identity` **1.39x ❌**.

Both red numbers are contended-box artifacts, not regressions. The box carried `load average 7.8–11.8` throughout from non-hydra system processes (`StorageManagement` ~83% CPU, a `Storage` extension ~59%, `WindowServer` ~57%, a long-running `top`), none of which may be killed under this repo's process rules, plus an idle `detectkit` GUI (PID 67394) left untouched. Legacy and new ran under nonstationary load, so the ratio is noise in either direction — and `ant_cnn_identity_relink`, the *same clip and video* as `ant_cnn_identity`, came in at 0.91x in the same matrix. `perfcmp` does not feed `FAILED_CLIPS`, so these did not fail the matrix. Reported as red rather than silently dropped; a clean-box re-measurement is the way to settle them.

### What a reader should conclude

1. With the feature off, this branch is a **bit-exact no-op** on 7 clips (the 8th red line is reproduced on the baseline tree alone). Two OBB/SLEAP clips are unmeasured.
2. The ROI coverage hole that let B1 ship is now closed by a fixture proven to produce detections (row counts identical to the non-ROI clip).
3. Tuned settings are bit-exact on bg-sub and CNN-identity, float-noise on YOLO-OBB, and **exceed the tuner's angular tolerance on the pose/head-tail clip** — pose batch size is the field to distrust.
4. **Nothing here validates the tuner itself, and one run showed the tuner cannot work at all.** `automatic` is CUDA-only by policy, so the *apply* path is dead on MPS. But `record` mode does calibrate here, and doing so surfaced a platform-independent blocker: the tuner's own determinism floor fails on byte-identical output whenever the forward CSV contains a NaN-position row, so calibration aborts before evaluating any candidate. **Fix that before Task 12 runs, or CUDA will measure the same permanent no-op.**

---

### Task 10c (2026-09-07, `28c09f0a`, MPS) — a real `record`-mode calibration STILL aborts, for a NEW reason

Two live `record`-mode runs on `fly_obb` (500 frames, `runtime=mps`, budget 600 s,
`hydra-mps`, log `commit=28c09f0ae9`). **The fixture had to be switched from
`tracking_workflow_mode: realtime` to `offline`** — `integration.py:604` declares
realtime runs ineligible (`"realtime inference is not tunable"`), so the shipped
`fly_obb_roi.json`/`fly_obb.json` as-is would never calibrate.

Both runs ended:

```
Inference throughput autotuner: status=fallback profile=None
  requested={'detection_batch_size': 1, ..., 'pipeline_depth': 2}
  admitted =same   effective=same
  reason=baseline_nondeterministic_beyond_contract
```

**Task 10b's NaN fix was necessary but not sufficient.** A wrapper around
`search.compare_outputs` captured the sidecar's own A-vs-A frames — the pair the
floor actually judges, which nothing before had ever looked at:

```
FLOOR#1 passed=False unmatched=1122 pos_p99=0.0 angle_max=0.0 categorical=288
details=['forward: keyed rows are not aligned',
         'forward: TrackID differs on 96 row(s); first differing FrameID=96',
         'forward: TrajectoryID differs on 96 row(s); first differing FrameID=96',
         'forward: 564 unmatched positional rows',
         'forward: 192 categorical mismatches',
         'final: keyed rows are not aligned',
         'final: TrajectoryID differs on 96 row(s); first differing FrameID=96',
         'final: 558 unmatched positional rows',
         'final: 96 categorical mismatches']
shapes fwd a=(378, 21) b=(378, 21) | final a=(375, 21) b=(375, 21)
```

**Root cause — the "A-vs-A" pair is not A-vs-A. The two sides are different
segments of the video.**

| side | forward FrameID range | final FrameID range | rows | NaN-X rows |
| --- | --- | --- | --- | --- |
| `baseline_outputs[0]` (block 0) | 2 – 127 | 3 – 127 | 378 | 3 |
| `baseline_outputs[1]` (block 1) | 95 – 220 | 96 – 220 | 378 | 3 |

`sidecar_child._block_window` deliberately **stripes** each measurement block
across the clip ("Spread the blocks across the clip instead"): for
`total=500, span=128, blocks=5` the windows are `(0,127) (93,220) (186,313)
(279,406) (372,499)`. `search.run` then compares `baseline_outputs[0]` against
`baseline_outputs[1]` with `for_determinism_floor=True`. Those are frames
0–127 vs frames 93–220 — non-overlapping except for 33 frames, in which track
numbering legitimately differs. `pos_p99 = 0.0` and `angle_max = 0.0` because
nothing geometric is even comparable; the verdict is carried entirely by
`keyed rows are not aligned` + 1122 unmatched rows.

The same defect poisons the **candidate** gate, not just the floor:
`reference = baseline_outputs[0]` (block 0's window) is compared against every
candidate output, four fifths of which come from blocks 1–4's *other* windows.
So even with the floor bypassed, no candidate could pass the correctness gate.

**This is arithmetic, not device behaviour — CUDA will reproduce it exactly.**
Striping is right for throughput representativeness and wrong for output
comparison; the two purposes need decoupling (e.g. a fixed comparison window
run once per candidate, separate from the striped timing blocks).

**Measured facts from the runs**

1. Determinism floor: **still FAILS** (new cause, above).
2. Candidates evaluated: **zero**. The search returns at `search.py:141` before
   the first mutation is proposed. No admitted, no rejected, no reasons.
3. Profile persisted: **none**. `inference_tuning_profiles/` contains only
   `locks/` (2 lock files, no record). `baseline_nondeterministic_beyond_contract`
   is not in S5's negative-cache list, so every run re-pays the calibration cost
   for nothing.
4. Record mode applied nothing — `requested == admitted == effective == baseline`
   (`detection_batch_size=1, pipeline_depth=2`), consistent with Task 3's
   guarantee. But it is *vacuously* consistent: nothing was ever selected.
5. Wall clock: run 1 total 115.7 s for 500 frames (4.32 fps), of which the
   calibration abort took **≈ 75–85 s** (bracketed by polls: still calibrating at
   59 s, at frame 45 of the main pass at 93 s). Diag run: process start →
   floor verdict = **112 s** including imports and model load. Against a
   **600 s** budget — the budget was never the binding constraint.
6. **S8: NOT exercised.** No trial ran past the baseline blocks, and `fly_obb`
   has neither pose nor identity, so the pose/identity cache path has **zero**
   coverage from this run. S8 remains empirically unverified on every platform.

Artifacts: `/tmp/task10c_record.log`, `/tmp/task10c_diag.log`,
`/tmp/task10c_floor/floor1_{a,b}_{forward,final}.csv`,
`/tmp/task10c_floor_diag.py`.

**The window mismatch is the ENTIRE failure.** On the 33-frame overlap
(FrameID 95–127, 99 rows a side), joining the two sides on
`(FrameID, X, Y, Theta)` rounded to 6 dp matches **96 of 99** rows; the 3 that do
not join are exactly the 3 NaN-position rows per side. The only columns that
differ on the joined rows are `TrackID`, `TrajectoryID`, `Index`,
`AssignmentConfidence`, `PositionUncertainty` — all run-history dependent (side A
entered the overlap having seen 95 frames, side B at its own frame 0). Geometry
is bit-identical wherever it is comparable, so the pipeline is deterministic and
decoupling the comparison window from the striped timing blocks should clear the
floor outright.
