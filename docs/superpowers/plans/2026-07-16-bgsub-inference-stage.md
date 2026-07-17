# Background-Subtraction Inference Stage Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make background-subtraction detection a first-class `InferenceRunner` stage so it gets the detection cache, batch precompute, backward-pass replay, and `runtime_tier` handling that the YOLO/OBB path already has.

**Architecture:** bg-sub's dependency on tracking state (`tracking_stabilized`) is replaced by a background-convergence latch internal to `BackgroundModel`, making detection feed-forward and therefore cacheable. The mask→ellipse logic moves from `core/detectors/bg_detector.py` to `core/background/measure.py` (because `core/inference` may not import `core/detectors`), and a thin `core/inference/stages/bgsub.py` adapter speaks the runner's `load_*`/`run_*`/`run_*_batch` protocol.

**Tech Stack:** Python 3, NumPy, OpenCV (`cv2`), CuPy (CUDA), PyTorch (MPS), Numba (CPU JIT), pytest.

**Spec:** `docs/superpowers/specs/2026-07-16-bgsub-inference-stage-design.md`

## CANONICAL CONVERGENCE DEFAULTS (single source of truth)

Earlier drafts of this plan used a whole-frame MEAN delta with
`BACKGROUND_CONVERGENCE_EPSILON = 0.05`. That design was replaced (it was
frame-size dependent and latched early). Any `0.05` still visible in a code
snippet below is STALE — it already caused one real defect, where Task 6 copied
it verbatim and shipped a 500x disagreement against `model.py`.

The CANONICAL values, which `core/background/model.py` actually reads, are:

| param | default | meaning |
|---|---|---|
| `BACKGROUND_CONVERGENCE_EPSILON` | `1e-4` | changed-pixel FRACTION below which a frame counts as settled |
| `BACKGROUND_CONVERGENCE_FRAMES` | `30` | consecutive settled frames required to latch |
| `BACKGROUND_CONVERGENCE_PIXEL_DELTA` | `5.0` | grey levels; the NOISE GATE — must exceed sensor noise |

**Any task adding a typed field or default for these MUST cross-check against
`core/background/model.py::_update_convergence` rather than trusting a snippet
here.**

## Execution Environment (READ FIRST — the plan's test commands depend on this)

Work happens in the worktree `.worktrees/bgsub-inference-stage` on branch
`feature/bgsub-inference-stage`.

**The conda env's editable install points at the MAIN checkout, not this
worktree.** Without an override, `import hydra_suite` resolves to
`<repo>/src/hydra_suite` — so you would edit worktree files while pytest
imported a different tree, and your tests would pass against code you never
changed. Verified: `PYTHONPATH` wins over the editable `.pth`, including under
pytest.

**Every python/pytest invocation in this plan MUST use this form:**

```bash
PYTHONPATH="$PWD/src" /Users/neurorishika/miniforge3/envs/hydra-mps/bin/$PY -m pytest <files> -v --timeout=60
```

Shorthand used below: `PY` = `PYTHONPATH="$PWD/src" /Users/neurorishika/miniforge3/envs/hydra-mps/bin/python`.

Sanity-check the override at any time:

```bash
PYTHONPATH="$PWD/src" /Users/neurorishika/miniforge3/envs/hydra-mps/bin/$PY -c "import hydra_suite; print(hydra_suite.__file__)"
# MUST contain 'bgsub-inference-stage'. If it does not, STOP — you are testing the wrong tree.
```

**DO NOT run `make format`** — it is BROKEN in this environment, for reasons
unrelated to this work: it shells out to the base conda env's `black`
(`~/miniforge3/bin/black`), whose `pathspec` dependency is broken
(`ModuleNotFoundError: No module named 'pathspec.patterns.gitignore'`). The
repo's pre-commit hooks run black/ruff/flake8/isort in their own isolated envs
and DO work — they run automatically on `git commit`, so formatting is handled.
If you want to format manually, use the working copies in the hydra-mps env:
`/Users/neurorishika/miniforge3/envs/hydra-mps/bin/black <files>` and
`/Users/neurorishika/miniforge3/envs/hydra-mps/bin/isort <files>`.

**NEVER run the full suite** (`pytest tests/`). It is messy and contains hangs.
Run only the named files, always with `--timeout=60` (pytest-timeout is
installed) so a hang fails instead of stalling.

**Known pre-existing failure — do NOT try to fix it, it is unrelated:**
`tests/test_bg_parameter_helper.py::test_bg_parameter_helper_slider_scrub_does_not_render_until_release`
(asserts `"3/3"`, gets `"0/0"`). It fails on a clean baseline.

**Baseline (established before any work):** 107 passed, 1 failed (the above)
across `test_background_model.py`, `test_background_model_integration.py`,
`test_detectors_engine.py`, `test_detector_integration.py`,
`test_inference_cache_keys.py`, `test_bg_parameter_helper.py`.

**Existing tests this plan will break — they must be updated by the task that
breaks them, not left red:**

| File | Broken by |
|---|---|
| `tests/test_background_model.py` | Task 4 (`update_and_get_background` loses `tracking_stabilized`) |
| `tests/test_background_model_integration.py` | Task 2 (deterministic priming), Task 4 |
| `tests/test_detectors_engine.py` | Task 5 (`ObjectDetector` → `BackgroundMeasurer`, 5-tuple → 4-tuple) |
| `tests/test_detector_integration.py` | Task 5, Task 13 (`create_detector` deleted) |
| `tests/test_inference_cache_keys.py` | Task 8 (schema bump; may assert a literal version) |
| `tests/test_bg_parameter_helper.py` | Task 13 (`bg_optimizer` moves to `core/background/optimizer.py`) |

Each task's verification step must run its own new tests AND the affected
existing files from this table. Note `pytest.ini` (not `pyproject.toml`, despite
CLAUDE.md) is the live pytest config; it already applies `-m "not benchmark"`.

## Global Constraints

- **Dependency direction:** `core/inference` must NEVER import from `core/detectors`. `core/inference -> core/background` is legal. Core must never import from any app layer (trackerkit, posekit, etc.).
- **Acceptance gate:** `bash tools/equivalence/run_matrix.sh worm_bgsub` passes at default tolerances (`pos_p99 <= 0.5px`, `theta_mean <= theta_atol`, `unmatched == 0`). Do NOT loosen tolerances to make it green.
- **Corner ordering:** TL/TR/BR/BL, matching `_corners_from_xywhr` (`core/inference/stages/obb.py:249`). Wrong ordering put SLEAP ~86 px off historically.
- **Angles:** radians (bg-sub already emits `np.deg2rad(ang)`).
- **Confidences:** `np.nan` for bg-sub. All downstream consumers must be NaN-safe.
- **Formatting:** do NOT run `make format` (BROKEN — see Execution Environment). Pre-commit hooks run black/ruff/flake8/isort automatically on `git commit`.
- **Tests:** named files only, never the full suite — see "Execution Environment" above. `$PY -m pytest tests/test_<name>.py -v --timeout=60`.
- **Legacy policy:** never import from `legacy/` in `src/` or `tests/`.
- **Spec corrections found during planning** (the spec is slightly wrong on both; trust this plan):
  - `BackgroundModel._setup_gpu_acceleration` (`core/background/model.py:197-239`) DOES gate on the `ENABLE_GPU_BACKGROUND` param. It is not config-blind — it simply never consults `runtime_tier`. Task 11 changes only the tier consultation.
  - `InferenceConfig.obb: OBBConfig` (`core/inference/config.py:229`) is a REQUIRED field. Adding bg-sub requires making it optional plus an exactly-one-detection-source validation. Task 9 covers this; the spec did not call it out.

---

## File Structure

| File | Responsibility |
|---|---|
| `src/hydra_suite/core/background/model.py` (modify) | `BackgroundModel`: priming, lightest/adaptive state, convergence latch, GPU tiers |
| `src/hydra_suite/core/background/measure.py` (create) | mask → ellipse measurements → `OBBResult`; moved from `detectors/bg_detector.py` |
| `src/hydra_suite/core/inference/stages/bgsub.py` (create) | thin runner-protocol adapter: `load_bgsub_model` / `run_bgsub` / `run_bgsub_batch` |
| `src/hydra_suite/core/inference/config.py` (modify) | `BgSubConfig`; make `obb` optional; exactly-one validation |
| `src/hydra_suite/core/inference/cache/keys.py` (modify) | fix `_BGSUB_KEY_PARAMS`; bump `CACHE_SCHEMA_VERSION` |
| `src/hydra_suite/core/inference/runner.py` (modify) | `_AllModels.bgsub`; wire load/cache/realtime/pipeline |
| `src/hydra_suite/runtime/resolver.py` (modify) | `"bgsub"` pipeline key + capability rules |
| `src/hydra_suite/core/detectors/bg_detector.py` (delete) | superseded by `core/background/measure.py` |
| `src/hydra_suite/core/detectors/factory.py` (delete) | `create_detector` — last string-keyed factory |
| `tests/test_bgsub_stage.py` (create) | stage, convergence, determinism, corners |
| `tests/test_bgsub_cache_keys.py` (create) | cache-key regression tests |

---

## Task 1: Fix `_BGSUB_KEY_PARAMS` param names

The cache key hashes `SUBTRACTION_THRESHOLD` and `BACKGROUND_PRIME_SECONDS`, which exist nowhere in the codebase. `params.get()` returns `None` for both, so they hash to a constant — changing the bg-sub threshold does not invalidate the cache and stale detections replay silently. The real names are `THRESHOLD_VALUE` (`core/background/model.py:487`) and `BACKGROUND_PRIME_FRAMES` (`core/background/model.py:300`).

**Files:**
- Modify: `src/hydra_suite/core/inference/cache/keys.py:60-80`
- Test: `tests/test_bgsub_cache_keys.py` (create)

**Interfaces:**
- Consumes: nothing (first task)
- Produces: corrected `_BGSUB_KEY_PARAMS`; `bgsub_detection_cache_key(params: dict) -> CacheKey` signature unchanged

- [ ] **Step 1: Write the failing test**

Create `tests/test_bgsub_cache_keys.py`:

```python
"""Regression tests for background-subtraction cache keys."""

from hydra_suite.core.inference.cache.keys import bgsub_detection_cache_key


def _base_params() -> dict:
    return {
        "THRESHOLD_VALUE": 20,
        "DARK_ON_LIGHT_BACKGROUND": True,
        "ENABLE_CONSERVATIVE_SPLIT": False,
        "ENABLE_ADAPTIVE_BACKGROUND": True,
        "BACKGROUND_LEARNING_RATE": 0.001,
        "BACKGROUND_PRIME_FRAMES": 30,
        "ENABLE_SIZE_FILTERING": False,
        "MIN_OBJECT_SIZE": 0,
        "MAX_OBJECT_SIZE": 10000,
        "ENABLE_ASPECT_RATIO_FILTERING": False,
        "BRIGHTNESS": 0,
        "CONTRAST": 1.0,
        "GAMMA": 1.0,
        "ENABLE_LIGHTING_STABILIZATION": False,
        "MORPH_KERNEL_SIZE": 5,
        "DILATION_KERNEL_SIZE": 3,
        "CONSERVATIVE_KERNEL_SIZE": 3,
        "START_FRAME": 0,
        "END_FRAME": 500,
        "RESIZE_FACTOR": 1.0,
    }


def test_threshold_change_invalidates_cache_key():
    """THRESHOLD_VALUE is the most important bg-sub param; it must be keyed."""
    a = bgsub_detection_cache_key(_base_params())
    p = _base_params()
    p["THRESHOLD_VALUE"] = 40
    b = bgsub_detection_cache_key(p)
    assert a.config_hash != b.config_hash


def test_prime_frames_change_invalidates_cache_key():
    a = bgsub_detection_cache_key(_base_params())
    p = _base_params()
    p["BACKGROUND_PRIME_FRAMES"] = 60
    b = bgsub_detection_cache_key(p)
    assert a.config_hash != b.config_hash


def test_identical_params_produce_identical_key():
    assert (
        bgsub_detection_cache_key(_base_params()).config_hash
        == bgsub_detection_cache_key(_base_params()).config_hash
    )


def test_key_params_all_exist_in_codebase_naming():
    """Guard against re-introducing param names nothing else uses."""
    from hydra_suite.core.inference.cache.keys import _BGSUB_KEY_PARAMS

    assert "SUBTRACTION_THRESHOLD" not in _BGSUB_KEY_PARAMS
    assert "BACKGROUND_PRIME_SECONDS" not in _BGSUB_KEY_PARAMS
    assert "THRESHOLD_VALUE" in _BGSUB_KEY_PARAMS
    assert "BACKGROUND_PRIME_FRAMES" in _BGSUB_KEY_PARAMS
```

- [ ] **Step 2: Run test to verify it fails**

Run: `$PY -m pytest tests/test_bgsub_cache_keys.py -v`
Expected: `test_threshold_change_invalidates_cache_key` FAILS (hashes equal — both params resolve to `None`), `test_prime_frames_change_invalidates_cache_key` FAILS, `test_key_params_all_exist_in_codebase_naming` FAILS.

- [ ] **Step 3: Fix the param names**

In `src/hydra_suite/core/inference/cache/keys.py`, in `_BGSUB_KEY_PARAMS`:
- Replace `"SUBTRACTION_THRESHOLD",` with `"THRESHOLD_VALUE",`
- Replace `"BACKGROUND_PRIME_SECONDS",` with `"BACKGROUND_PRIME_FRAMES",`

- [ ] **Step 4: Run tests to verify they pass**

Run: `$PY -m pytest tests/test_bgsub_cache_keys.py -v`
Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add tests/test_bgsub_cache_keys.py src/hydra_suite/core/inference/cache/keys.py
git commit -m "fix(cache): bg-sub cache key hashed two param names that do not exist

_BGSUB_KEY_PARAMS keyed on SUBTRACTION_THRESHOLD and BACKGROUND_PRIME_SECONDS.
Neither name appears anywhere else in the codebase; the real params are
THRESHOLD_VALUE and BACKGROUND_PRIME_FRAMES. params.get() returned None for
both, so they hashed to a constant and changing the threshold never
invalidated the detection cache -- stale detections replayed silently."
```

---

## Task 2: Deterministic evenly-spaced background priming

`prime_background` samples frames via the unseeded module-level `random` (`core/background/model.py:308`), so identical params + identical video produce a different background each run. Caching is unsound by construction. This is known — `tools/equivalence/runner.py:302-308` works around it with `random.seed(0)` — but that workaround is test-only; production stays nondeterministic.

Evenly-spaced sampling is deterministic without a seed AND strictly guarantees the temporal coverage random sampling only achieves on average.

**Files:**
- Modify: `src/hydra_suite/core/background/model.py:308`
- Modify: `src/hydra_suite/core/background/model.py:7` (drop `import random`)
- Modify: `tools/equivalence/runner.py:302-315` (the seed becomes a no-op; the comment must not imply a guarantee it no longer provides)
- Test: `tests/test_bgsub_stage.py` (create)

**Interfaces:**
- Consumes: Task 1's corrected key params (no code dependency)
- Produces: `BackgroundModel.prime_background(cap)` — signature unchanged, now deterministic

- [ ] **Step 1: Write the failing test**

Create `tests/test_bgsub_stage.py`:

```python
"""Tests for the background-subtraction inference stage."""

import cv2
import numpy as np
import pytest

from hydra_suite.core.background.model import BackgroundModel


@pytest.fixture
def synthetic_video(tmp_path):
    """A 60-frame 64x64 video with a moving dark blob on a light background."""
    path = tmp_path / "synthetic.avi"
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"MJPG"), 10, (64, 64), True
    )
    for i in range(60):
        frame = np.full((64, 64, 3), 200, dtype=np.uint8)
        cx = 8 + i
        if cx < 56:
            cv2.circle(frame, (cx, 32), 4, (30, 30, 30), -1)
        writer.write(frame)
    writer.release()
    return str(path)


def _params(**overrides) -> dict:
    p = {
        "BACKGROUND_PRIME_FRAMES": 20,
        "BRIGHTNESS": 0,
        "CONTRAST": 1.0,
        "GAMMA": 1.0,
        "RESIZE_FACTOR": 1.0,
        "THRESHOLD_VALUE": 20,
        "DARK_ON_LIGHT_BACKGROUND": True,
        "ENABLE_ADAPTIVE_BACKGROUND": True,
        "BACKGROUND_LEARNING_RATE": 0.001,
        "MORPH_KERNEL_SIZE": 3,
        "ENABLE_GPU_BACKGROUND": False,
    }
    p.update(overrides)
    return p


def test_priming_is_deterministic(synthetic_video):
    """Same video + same params must produce a byte-identical background.

    This is the property that makes the bg-sub cache key honest.
    """
    backgrounds = []
    for _ in range(2):
        model = BackgroundModel(_params())
        cap = cv2.VideoCapture(synthetic_video)
        model.prime_background(cap)
        cap.release()
        backgrounds.append(model.lightest_background.copy())

    np.testing.assert_array_equal(backgrounds[0], backgrounds[1])


def test_priming_covers_video_temporally(synthetic_video):
    """Evenly-spaced sampling must span the whole video, not cluster."""
    model = BackgroundModel(_params(BACKGROUND_PRIME_FRAMES=10))
    cap = cv2.VideoCapture(synthetic_video)
    model.prime_background(cap)
    cap.release()
    # The blob traverses the frame; a background spanning the video is the
    # light plate everywhere, so its minimum stays near the plate value.
    assert model.lightest_background is not None
    assert float(model.lightest_background.min()) > 150.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `$PY -m pytest tests/test_bgsub_stage.py -v`
Expected: `test_priming_is_deterministic` FAILS — arrays differ because `random.sample` picks different frames each construction.

- [ ] **Step 3: Replace random sampling with evenly-spaced indices**

In `src/hydra_suite/core/background/model.py`, replace line 308:

```python
        idxs = random.sample(range(total), count)
```

with:

```python
        # Evenly-spaced rather than random: deterministic without needing a
        # seed (which makes the detection cache key honest -- see
        # core/inference/cache/keys.py) and strictly guarantees the temporal
        # coverage random sampling only achieves on average.
        idxs = [int(round(i)) for i in np.linspace(0, total - 1, count)]
```

Then remove the now-unused `import random` at line 7.

- [ ] **Step 4: Run tests to verify they pass**

Run: `$PY -m pytest tests/test_bgsub_stage.py -v`
Expected: 2 passed.

- [ ] **Step 5: Update the equivalence harness comment**

The `random.seed(0)` in `tools/equivalence/runner.py` is now a no-op for priming. Leaving its comment intact would assert a guarantee that no longer exists. Replace the comment block at `tools/equivalence/runner.py:302-308` with:

```python
    # Deterministic seeding so the run is reproducible and legacy-vs-new is
    # comparable. NOTE: bgsub background priming no longer consumes the global
    # `random` module -- core/background/model.py now samples evenly-spaced
    # frames, which is deterministic without a seed. These seeds are retained
    # for any other stochastic code paths; they are a no-op for bgsub priming.
```

Leave the `_random.seed(0)` / `_np.random.seed(0)` calls in place — they are harmless and may cover other paths.

- [ ] **Step 6: Run the full test suite**

Run: `$PY -m pytest tests/test_background_model.py tests/test_background_model_integration.py tests/test_detectors_engine.py tests/test_detector_integration.py tests/test_inference_cache_keys.py -q --timeout=60`
Expected: no failures beyond the known-bad listed in "Execution Environment". NEVER run the full suite.

- [ ] **Step 7: Commit**

```bash
git add tests/test_bgsub_stage.py src/hydra_suite/core/background/model.py tools/equivalence/runner.py
git commit -m "fix(background): deterministic evenly-spaced priming

prime_background sampled frames via the unseeded global random module, so
identical params + identical video produced a different background every
run -- making any bg-sub cache key a lie. The equivalence harness already
worked around this with random.seed(0), but that is test-only; production
stayed nondeterministic.

Evenly-spaced sampling is deterministic without a seed and guarantees the
temporal coverage random sampling only achieved on average. Updates the
harness comment, whose seeding is now a no-op for priming."
```

---

## Task 3: `ENABLE_ADAPTIVE_BACKGROUND=False` must not switch

At `core/background/model.py:434`, `if tracking_stabilized:` returns `adaptive_background` regardless of whether adaptive updating is enabled. When `ENABLE_ADAPTIVE_BACKGROUND=False`, that array is frozen at its primed value — so disabling adaptive silently means "switch to a stale snapshot" rather than "don't switch".

**Files:**
- Modify: `src/hydra_suite/core/background/model.py:433-437`
- Test: `tests/test_bgsub_stage.py` (append)

**Interfaces:**
- Consumes: Task 2's deterministic priming
- Produces: `update_and_get_background` honors `ENABLE_ADAPTIVE_BACKGROUND` (signature still has `tracking_stabilized` — removed in Task 4)

- [ ] **Step 1: Write the failing test**

Append to `tests/test_bgsub_stage.py`:

```python
def test_adaptive_disabled_never_switches_to_frozen_snapshot():
    """ENABLE_ADAPTIVE_BACKGROUND=False must mean 'do not switch', not
    'switch to a stale primed snapshot'."""
    model = BackgroundModel(_params(ENABLE_ADAPTIVE_BACKGROUND=False))
    gray_a = np.full((16, 16), 200, dtype=np.uint8)
    model.update_and_get_background(gray_a, None, tracking_stabilized=False)

    gray_b = np.full((16, 16), 240, dtype=np.uint8)
    model.update_and_get_background(gray_b, None, tracking_stabilized=False)

    stabilized_bg = model.update_and_get_background(
        gray_b, None, tracking_stabilized=True
    )
    lightest = cv2.convertScaleAbs(model.lightest_background)
    np.testing.assert_array_equal(stabilized_bg, lightest)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `$PY -m pytest tests/test_bgsub_stage.py::test_adaptive_disabled_never_switches_to_frozen_snapshot -v`
Expected: FAIL — returns the frozen `adaptive_background` (200s) rather than the lightest (240s).

- [ ] **Step 3: Gate the switch on adaptive being enabled**

In `src/hydra_suite/core/background/model.py`, replace lines 433-437:

```python
        if tracking_stabilized:
            return cv2.convertScaleAbs(self.adaptive_background)
        else:
            return cv2.convertScaleAbs(self.lightest_background)
```

with:

```python
        # Only switch when adaptive updating is actually on. Otherwise
        # adaptive_background is frozen at its primed value and switching to
        # it would silently subtract against a stale snapshot.
        adaptive_enabled = p.get("ENABLE_ADAPTIVE_BACKGROUND", True)
        if tracking_stabilized and adaptive_enabled:
            return cv2.convertScaleAbs(self.adaptive_background)
        return cv2.convertScaleAbs(self.lightest_background)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `$PY -m pytest tests/test_bgsub_stage.py -v`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add tests/test_bgsub_stage.py src/hydra_suite/core/background/model.py
git commit -m "fix(background): ENABLE_ADAPTIVE_BACKGROUND=False no longer switches

On stabilization the model returned adaptive_background regardless of
whether adaptive updating was enabled. With it disabled that array is
frozen at its primed value, so disabling adaptive silently meant 'switch
to a stale snapshot' instead of 'do not switch'."
```

---

## Task 4: Background-convergence latch replaces `tracking_stabilized`

This is the core of the design. `tracking_stabilized` is set at `core/tracking/worker.py:3656` from Hungarian assignment cost — a feedback loop from tracking into detection, which a feed-forward cacheable pass cannot have. It is removable because it is a monotonic latch and the background state already evolves independently of it (`model.py:418-431` updates both backgrounds every frame; only the selection reads the flag).

The replacement measures the thing directly: the lightest background has converged when it stops growing.

**Files:**
- Modify: `src/hydra_suite/core/background/model.py` (`__init__`, `update_and_get_background`)
- Modify: `src/hydra_suite/core/tracking/worker.py:942, 2297-2298, 3656-3660`
- Modify: `src/hydra_suite/core/detectors/bg_optimizer.py:467-470`
- Modify: `src/hydra_suite/trackerkit/gui/workers/preview_worker.py:419-423`
- Modify: `src/hydra_suite/trackerkit/cli_config.py` (add the two new params)
- Test: `tests/test_bgsub_stage.py` (append)

**Interfaces:**
- Consumes: Task 3's `adaptive_enabled` gating
- Produces:
  - `BackgroundModel.update_and_get_background(gray, roi_mask) -> Optional[np.ndarray]` — **`tracking_stabilized` parameter REMOVED**
  - `BackgroundModel.stabilized -> bool` (read-only property)
  - New params: `BACKGROUND_CONVERGENCE_EPSILON: float = 1e-4` (changed-pixel FRACTION), `BACKGROUND_CONVERGENCE_FRAMES: int = 30`, `BACKGROUND_CONVERGENCE_PIXEL_DELTA: float = 1.0` (grey levels)

- [ ] **Step 1: Write the failing test**

Append to `tests/test_bgsub_stage.py`:

```python
def test_convergence_latch_sets_when_lightest_stops_growing():
    model = BackgroundModel(
        _params(
            BACKGROUND_CONVERGENCE_EPSILON=1e-4,
            BACKGROUND_CONVERGENCE_FRAMES=3,
        )
    )
    gray = np.full((16, 16), 200, dtype=np.uint8)
    model.update_and_get_background(gray, None)  # first frame primes, returns None
    assert not model.stabilized

    for _ in range(3):
        model.update_and_get_background(gray, None)
    assert model.stabilized


def test_convergence_latch_resets_counter_when_background_grows():
    model = BackgroundModel(
        _params(
            BACKGROUND_CONVERGENCE_EPSILON=1e-4,
            BACKGROUND_CONVERGENCE_FRAMES=3,
        )
    )
    model.update_and_get_background(np.full((16, 16), 200, np.uint8), None)
    model.update_and_get_background(np.full((16, 16), 200, np.uint8), None)
    model.update_and_get_background(np.full((16, 16), 200, np.uint8), None)
    # A brighter frame grows the running max -> counter resets.
    model.update_and_get_background(np.full((16, 16), 250, np.uint8), None)
    assert not model.stabilized


def test_convergence_latch_is_monotonic():
    """Once latched, never un-latches, even if the background grows again."""
    model = BackgroundModel(
        _params(
            BACKGROUND_CONVERGENCE_EPSILON=1e-4,
            BACKGROUND_CONVERGENCE_FRAMES=2,
        )
    )
    gray = np.full((16, 16), 200, dtype=np.uint8)
    for _ in range(4):
        model.update_and_get_background(gray, None)
    assert model.stabilized

    model.update_and_get_background(np.full((16, 16), 255, np.uint8), None)
    assert model.stabilized
```

- [ ] **Step 2: Run test to verify it fails**

Run: `$PY -m pytest tests/test_bgsub_stage.py -k convergence -v`
Expected: FAIL — `BackgroundModel` has no `stabilized` attribute, and `update_and_get_background` still requires `tracking_stabilized`.

- [ ] **Step 3: Add the latch to `BackgroundModel.__init__`**

In `src/hydra_suite/core/background/model.py`, in `__init__` after line 188 (`self.reference_intensity = ...`), add:

```python
        # Background-convergence latch. Replaces the old `tracking_stabilized`
        # flag, which was fed in from the tracker (Hungarian assignment cost)
        # and made detection depend on tracking state -- impossible to cache.
        # The latch is monotonic: once set, never cleared.
        self._stabilized: bool = False
        self._converged_frames: int = 0
```

Add this property immediately after `__init__`:

```python
    @property
    def stabilized(self) -> bool:
        """True once the lightest background has stopped growing.

        Monotonic latch. Selects adaptive (True) over lightest (False) as the
        subtraction background.
        """
        return self._stabilized
```

- [ ] **Step 4: Replace the flag with the latch in `update_and_get_background`**

Replace the whole method body (`model.py:404-437`, as amended by Task 3) with:

```python
    def update_and_get_background(
        self,
        gray: np.ndarray,
        roi_mask: Optional[np.ndarray],
    ) -> Optional[np.ndarray]:
        """Update the background model and return the active subtraction background."""
        p = self.params
        if self.lightest_background is None:
            self.lightest_background = gray.astype(np.float32)
            self.adaptive_background = gray.astype(np.float32)
            return None  # Indicates first frame

        # Update full-frame background (ROI masking happens during detection)
        gray_f32 = gray.astype(np.float32)
        previous_lightest = self.lightest_background
        self.lightest_background = np.maximum(previous_lightest, gray_f32)

        self._update_convergence(previous_lightest)

        if (
            p.get("ENABLE_ADAPTIVE_BACKGROUND", True)
            and self.adaptive_background is not None
        ):
            learning_rate = p.get("BACKGROUND_LEARNING_RATE", 0.001)
            if self.use_gpu:
                self._adaptive_update_gpu(gray_f32, learning_rate)
            else:
                self._adaptive_update_cpu(gray_f32, learning_rate)

        # Only switch when adaptive updating is actually on. Otherwise
        # adaptive_background is frozen at its primed value and switching to
        # it would silently subtract against a stale snapshot.
        adaptive_enabled = p.get("ENABLE_ADAPTIVE_BACKGROUND", True)
        if self._stabilized and adaptive_enabled:
            return cv2.convertScaleAbs(self.adaptive_background)
        return cv2.convertScaleAbs(self.lightest_background)

    def _update_convergence(self, previous_lightest: np.ndarray) -> None:
        """Latch stabilization once the lightest background stops growing.

        This is what the old tracking-derived `tracking_stabilized` flag was
        really proxying for: "has the background been revealed?".
        MIN_TRACKING_COUNTS answered that indirectly via assignment cost; the
        background can answer it about itself, deterministically -- which is
        what makes bg-sub detection cacheable.
        """
        if self._stabilized:
            return  # monotonic: never un-latch

        p = self.params
        epsilon = float(p.get("BACKGROUND_CONVERGENCE_EPSILON", 1e-4) or 1e-4)
        needed = int(p.get("BACKGROUND_CONVERGENCE_FRAMES", 30) or 30)

        delta = float(
            np.mean(np.abs(self.lightest_background - previous_lightest))
        )
        if delta < epsilon:
            self._converged_frames += 1
        else:
            self._converged_frames = 0

        if self._converged_frames >= needed:
            self._stabilized = True
            logger.info(
                "Background converged (delta=%.4f < %.4f for %d frames)",
                delta,
                epsilon,
                needed,
            )
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `$PY -m pytest tests/test_bgsub_stage.py -k convergence -v`
Expected: 3 passed.

Note `test_adaptive_disabled_never_switches_to_frozen_snapshot` from Task 3 now fails to call — update it to drop the `tracking_stabilized` argument:

```python
def test_adaptive_disabled_never_switches_to_frozen_snapshot():
    """ENABLE_ADAPTIVE_BACKGROUND=False must mean 'do not switch', not
    'switch to a stale primed snapshot'."""
    model = BackgroundModel(
        _params(
            ENABLE_ADAPTIVE_BACKGROUND=False,
            BACKGROUND_CONVERGENCE_EPSILON=1e-4,
            BACKGROUND_CONVERGENCE_FRAMES=1,
        )
    )
    gray = np.full((16, 16), 200, dtype=np.uint8)
    model.update_and_get_background(gray, None)
    model.update_and_get_background(gray, None)
    model.update_and_get_background(gray, None)
    assert model.stabilized

    result = model.update_and_get_background(gray, None)
    np.testing.assert_array_equal(result, cv2.convertScaleAbs(model.lightest_background))
```

Run: `$PY -m pytest tests/test_bgsub_stage.py -v`
Expected: 6 passed.

- [ ] **Step 6: Update the three callers**

`src/hydra_suite/core/tracking/worker.py:2297-2298` — drop the third argument:

```python
                bg_u8 = bg_model.update_and_get_background(
                    gray, ROI_mask_current
                )
```

`src/hydra_suite/core/tracking/worker.py:942` — change:

```python
        detection_initialized, tracking_stabilized = False, False
```

to:

```python
        detection_initialized = False
```

`src/hydra_suite/core/tracking/worker.py:3656-3660` — delete the latch block entirely:

```python
                if (
                    tracking_counts >= params["MIN_TRACKING_COUNTS"]
                    and not tracking_stabilized
                ):
                    tracking_stabilized = True
                    logger.info(f"Tracking stabilized (avg cost={avg_cost:.2f})")
```

Keep the `tracking_counts` increment/reset immediately above it — it serves other purposes.

`src/hydra_suite/core/detectors/bg_optimizer.py:467-470` — drop the argument:

```python
    bg_u8 = bg_model.update_and_get_background(
        gray,
        roi_mask,
    )
```

(Verify the exact argument names at that call site before editing; the second positional is the ROI mask.)

`src/hydra_suite/trackerkit/gui/workers/preview_worker.py:419-423` — replace the comment and call:

```python
    # Background selection is now internal to BackgroundModel via its
    # convergence latch; the preview shows whatever the model would use.
    bg_u8 = bg_model.update_and_get_background(gray, roi_mask=None)
```

- [ ] **Step 7: Add the new params to config defaults**

In `src/hydra_suite/trackerkit/cli_config.py`, alongside `"BACKGROUND_PRIME_FRAMES": bg_prime_frames,` (line ~716), add:

```python
        "BACKGROUND_CONVERGENCE_EPSILON": background_convergence_epsilon,
        "BACKGROUND_CONVERGENCE_FRAMES": background_convergence_frames,
```

and define the two locals near the other background defaults, following the surrounding style:

```python
    background_convergence_epsilon = float(
        cfg.get("background_convergence_epsilon", 1e-4)
    )
    background_convergence_frames = int(cfg.get("background_convergence_frames", 30))
```

- [ ] **Step 8: Verify no `tracking_stabilized` references remain**

Run: `grep -rn "tracking_stabilized" src/`
Expected: no output.

- [ ] **Step 9: Run the full test suite**

Run: `$PY -m pytest tests/test_background_model.py tests/test_background_model_integration.py tests/test_detectors_engine.py tests/test_detector_integration.py tests/test_inference_cache_keys.py -q --timeout=60`
Expected: no failures beyond the known-bad listed in "Execution Environment". NEVER run the full suite.

- [ ] **Step 10: Commit**

```bash
git add -A src/ tests/test_bgsub_stage.py
git commit -m "feat(background): replace tracking_stabilized with a convergence latch

bg-sub detection depended on tracking state: update_and_get_background took
a tracking_stabilized flag set from Hungarian assignment cost in worker.py.
A feed-forward cacheable inference pass cannot have that feedback loop, which
is why bg-sub never got an InferenceRunner stage.

The loop is removable because it is a monotonic latch and the background
state already evolved independently of it -- only the one-bit selection read
the flag. MIN_TRACKING_COUNTS was a tracking-quality threshold reused as a
'has the background settled?' proxy; the background answers that directly.

update_and_get_background loses its tracking_stabilized parameter. Adds
BACKGROUND_CONVERGENCE_EPSILON/FRAMES."
```

---

## Task 5: Move mask→ellipse measurement to `core/background/measure.py`

`core/inference` may not import `core/detectors` — a rule `stages/obb.py:76` already broke down and duplicated `_gpu_letterbox_batch` over. So `bg_detector.py`'s logic must move to be reachable from a stage. It joins the model it belongs with.

**Files:**
- Create: `src/hydra_suite/core/background/measure.py`
- Delete: `src/hydra_suite/core/detectors/bg_detector.py`
- Modify: `src/hydra_suite/core/detectors/__init__.py`, `src/hydra_suite/core/__init__.py:11`
- Test: `tests/test_bgsub_stage.py` (append)

**Interfaces:**
- Consumes: nothing from Task 4
- Produces:
  - `BackgroundMeasurer(params: dict)` with `apply_conservative_split(fg_mask, gray=None, background=None) -> np.ndarray` and `detect_objects(fg_mask, frame_count) -> tuple[list, list, list, list]`
  - **Note the changed return arity: 4-tuple `(meas, sizes, shapes, confidences)`.** The legacy 5-tuple's fourth element was `yolo_results=None`, a pure compatibility stub.

- [ ] **Step 1: Create `measure.py` as a move of `bg_detector.py`**

Copy `src/hydra_suite/core/detectors/bg_detector.py` to `src/hydra_suite/core/background/measure.py` verbatim, then apply these changes:

1. Rename the class `ObjectDetector` → `BackgroundMeasurer`.
2. Update the module docstring to `"""Measure objects from a background-subtraction foreground mask."""`.
3. Replace the import `from ._utils import _CONSERVATIVE_SPLIT_MIN_ANIMALS` — `core/background` must not import from `core/detectors`. Inline the constant at the top of `measure.py` (read its value from `src/hydra_suite/core/detectors/_utils.py` first and copy it exactly):

```python
# Copied from core/detectors/_utils.py: core/background must not import from
# core/detectors (and detectors is being retired from the bg-sub path).
_CONSERVATIVE_SPLIT_MIN_ANIMALS = <exact value from _utils.py>
```

4. Change `detect_objects` to drop the `yolo_results` stub. Its signature and the two early-exit returns become:

```python
    def detect_objects(
        self, fg_mask: np.ndarray, frame_count: int
    ) -> tuple[list, list, list, list]:
        """Detect and measure objects from the final foreground mask.

        Returns:
            meas: list of np.array([cx, cy, angle_radians], float32)
            sizes: list of ellipse areas (px^2)
            shapes: list of (ellipse_area, aspect_ratio) tuples
            confidences: list of np.nan -- confidence is not feasible for
                background subtraction (quality is too context-specific).
        """
```

Replace both `return [], [], [], None, []` (lines 178 and 224 of the original) with `return [], [], [], []`, and the final `return meas, sizes, shapes, None, confidences` (line 238) with `return meas, sizes, shapes, confidences`.

Note the original's vacuous `self: object, fg_mask: object, frame_count: object -> object` annotations are replaced with real types above.

- [ ] **Step 2: Write the characterization test**

Append to `tests/test_bgsub_stage.py`:

```python
from hydra_suite.core.background.measure import BackgroundMeasurer


def _measure_params(**overrides) -> dict:
    p = {
        "MAX_TARGETS": 10,
        "MIN_CONTOUR_AREA": 5,
        "MAX_CONTOUR_MULTIPLIER": 20,
        "ENABLE_SIZE_FILTERING": False,
        "MIN_OBJECT_SIZE": 0,
        "MAX_OBJECT_SIZE": float("inf"),
        "THRESHOLD_VALUE": 20,
        "CONSERVATIVE_KERNEL_SIZE": 3,
        "CONSERVATIVE_ERODE_ITER": 1,
        "RESIZE_FACTOR": 1.0,
    }
    p.update(overrides)
    return p


def _mask_with_ellipse() -> np.ndarray:
    mask = np.zeros((64, 64), dtype=np.uint8)
    cv2.ellipse(mask, (32, 32), (12, 6), 30, 0, 360, 255, -1)
    return mask


def test_detect_objects_returns_four_tuple_without_yolo_stub():
    measurer = BackgroundMeasurer(_measure_params())
    result = measurer.detect_objects(_mask_with_ellipse(), 0)
    assert len(result) == 4
    meas, sizes, shapes, confidences = result
    assert len(meas) == 1
    assert len(sizes) == 1
    assert len(shapes) == 1
    assert len(confidences) == 1


def test_detect_objects_confidence_is_nan():
    measurer = BackgroundMeasurer(_measure_params())
    _, _, _, confidences = measurer.detect_objects(_mask_with_ellipse(), 0)
    assert np.isnan(confidences[0])


def test_detect_objects_angle_is_radians():
    measurer = BackgroundMeasurer(_measure_params())
    meas, _, _, _ = measurer.detect_objects(_mask_with_ellipse(), 0)
    assert 0.0 <= float(meas[0][2]) <= np.pi


def test_too_many_contours_returns_empty_four_tuple():
    measurer = BackgroundMeasurer(_measure_params(MAX_TARGETS=1, MAX_CONTOUR_MULTIPLIER=1))
    mask = np.zeros((64, 64), dtype=np.uint8)
    for x in range(4, 60, 8):
        for y in range(4, 60, 8):
            cv2.circle(mask, (x, y), 2, 255, -1)
    assert measurer.detect_objects(mask, 0) == ([], [], [], [])
```

- [ ] **Step 3: Run tests to verify they pass**

Run: `$PY -m pytest tests/test_bgsub_stage.py -v`
Expected: all pass.

- [ ] **Step 4: Delete the old module and fix re-exports**

```bash
git rm src/hydra_suite/core/detectors/bg_detector.py
```

In `src/hydra_suite/core/detectors/__init__.py`, remove `ObjectDetector` from the imports and `__all__`.

In `src/hydra_suite/core/__init__.py:11`, remove `from .detectors import ObjectDetector` and drop `ObjectDetector` from `__all__` if present.

- [ ] **Step 5: Verify no stale importers**

Run: `grep -rn "bg_detector\|ObjectDetector" src/ tests/`
Expected: only `factory.py` (deleted in Task 13) and `preview_worker.py` (migrated in Task 12). Note these two for those tasks; do not fix them here.

- [ ] **Step 6: Commit**

```bash
git add -A src/hydra_suite/core/background/measure.py src/hydra_suite/core/detectors/ src/hydra_suite/core/__init__.py tests/test_bgsub_stage.py
git commit -m "refactor(background): move mask->ellipse measurement out of detectors

core/inference may not import core/detectors -- a rule stages/obb.py already
broke down and duplicated _gpu_letterbox_batch over. Moving the measurement
logic to core/background lets the new bgsub stage reach it via a legal
downward edge, with no duplication.

ObjectDetector -> BackgroundMeasurer. detect_objects drops the yolo_results
stub from its return (5-tuple -> 4-tuple); it was always None, existing only
for symmetry with the YOLO detector. Vacuous 'object' annotations replaced
with real types."
```

---

## Task 6: `BgSubConfig` and ellipse→corners derivation

bg-sub must emit `OBBResult` like every other detection path. The one field it lacks is `corners`, derived from the fitted ellipse as a rotated rect. Ordering is load-bearing.

**Files:**
- Modify: `src/hydra_suite/core/inference/config.py`
- Modify: `src/hydra_suite/core/background/measure.py` (add `corners_from_ellipse`)
- Test: `tests/test_bgsub_stage.py` (append)

**Interfaces:**
- Consumes: Task 5's `BackgroundMeasurer`
- Produces:
  - `BgSubConfig` dataclass with `from_params(params: dict) -> BgSubConfig`
  - `corners_from_ellipse(cx, cy, major, minor, angle_rad) -> np.ndarray` shape `(4, 2)`, order TL/TR/BR/BL

- [ ] **Step 1: Write the failing test**

Append to `tests/test_bgsub_stage.py`:

```python
from hydra_suite.core.background.measure import corners_from_ellipse


def test_corners_from_ellipse_axis_aligned_order_is_tl_tr_br_bl():
    """Order must match _corners_from_xywhr (stages/obb.py:249). Wrong order
    historically put SLEAP ~86 px off."""
    corners = corners_from_ellipse(10.0, 20.0, 8.0, 4.0, 0.0)
    assert corners.shape == (4, 2)
    expected = np.array(
        [[6.0, 18.0], [14.0, 18.0], [14.0, 22.0], [6.0, 22.0]], dtype=np.float32
    )
    np.testing.assert_allclose(corners, expected, atol=1e-4)


def test_corners_from_ellipse_rotated_90_degrees():
    corners = corners_from_ellipse(0.0, 0.0, 8.0, 4.0, np.pi / 2)
    # Major axis now vertical: bounding corners swap extents.
    assert np.isclose(np.abs(corners[:, 0]).max(), 2.0, atol=1e-4)
    assert np.isclose(np.abs(corners[:, 1]).max(), 4.0, atol=1e-4)


def test_corners_from_ellipse_centroid_is_mean_of_corners():
    corners = corners_from_ellipse(5.0, 7.0, 10.0, 3.0, 0.7)
    np.testing.assert_allclose(corners.mean(axis=0), [5.0, 7.0], atol=1e-4)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `$PY -m pytest tests/test_bgsub_stage.py -k corners -v`
Expected: FAIL — `ImportError: cannot import name 'corners_from_ellipse'`.

- [ ] **Step 3: Implement `corners_from_ellipse`**

Before implementing, read `src/hydra_suite/core/inference/stages/obb.py:249` (`_corners_from_xywhr`) and match its corner ordering exactly.

Add to `src/hydra_suite/core/background/measure.py`:

```python
def corners_from_ellipse(
    cx: float, cy: float, major: float, minor: float, angle_rad: float
) -> np.ndarray:
    """Derive rotated-rect corners from a fitted ellipse.

    Returns (4, 2) float32 in TL/TR/BR/BL order, matching
    core/inference/stages/obb.py::_corners_from_xywhr. The ordering is
    load-bearing: canonical crops come out mirrored if it is wrong, which
    historically put SLEAP ~86 px off instead of ~2.7 px.

    *major*/*minor* are full axis lengths (not semi-axes), matching what
    cv2.fitEllipse returns.
    """
    hw = major / 2.0
    hh = minor / 2.0
    # TL, TR, BR, BL in the ellipse's own frame.
    local = np.array(
        [[-hw, -hh], [hw, -hh], [hw, hh], [-hw, hh]], dtype=np.float32
    )
    cos_a = np.cos(angle_rad)
    sin_a = np.sin(angle_rad)
    rot = np.array([[cos_a, -sin_a], [sin_a, cos_a]], dtype=np.float32)
    return (local @ rot.T + np.array([cx, cy], dtype=np.float32)).astype(np.float32)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `$PY -m pytest tests/test_bgsub_stage.py -k corners -v`
Expected: 3 passed.

- [ ] **Step 5: Add `BgSubConfig`**

Params today flow as a plain dict with inline `.get()` defaults across three drifting key lists (`cache/keys.py:58-80`, `trackerkit/tracking_cache.py:228-234`, `trackerkit/cli_config.py:638-720`) — Task 1's defect was a direct symptom. The dict stays at the API boundary; the dataclass is the internal contract.

Add to `src/hydra_suite/core/inference/config.py`, after `OBBConfig` (line ~138):

```python
@dataclass
class BgSubConfig:
    """Background-subtraction detection.

    Unlike OBB there is no model file: the 'model' is the primed
    BackgroundModel, derived from the video itself.
    """

    threshold_value: float = 20.0
    dark_on_light_background: bool = True
    enable_adaptive_background: bool = True
    background_learning_rate: float = 0.001
    background_prime_frames: int = 30
    convergence_epsilon: float = 1e-4
    convergence_frames: int = 30
    convergence_pixel_delta: float = 5.0
    enable_conservative_split: bool = False
    morph_kernel_size: int = 5
    dilation_kernel_size: int = 3
    conservative_kernel_size: int = 3
    max_targets: int = 20
    min_contour_area: float = 5.0
    max_contour_multiplier: int = 20
    enable_size_filtering: bool = False
    min_object_size: float = 0.0
    max_object_size: float = float("inf")
    # The raw param dict, retained for BackgroundModel/BackgroundMeasurer,
    # which still read params by legacy UPPER_SNAKE key.
    params: dict = field(default_factory=dict)

    @staticmethod
    def from_params(params: dict) -> "BgSubConfig":
        return BgSubConfig(
            threshold_value=float(params.get("THRESHOLD_VALUE", 20) or 20),
            dark_on_light_background=bool(
                params.get("DARK_ON_LIGHT_BACKGROUND", True)
            ),
            enable_adaptive_background=bool(
                params.get("ENABLE_ADAPTIVE_BACKGROUND", True)
            ),
            background_learning_rate=float(
                params.get("BACKGROUND_LEARNING_RATE", 0.001) or 0.001
            ),
            background_prime_frames=int(params.get("BACKGROUND_PRIME_FRAMES", 30) or 30),
            convergence_epsilon=float(
                params.get("BACKGROUND_CONVERGENCE_EPSILON", 1e-4) or 1e-4
            ),
            convergence_pixel_delta=float(
                params.get("BACKGROUND_CONVERGENCE_PIXEL_DELTA", 5.0) or 5.0
            ),
            convergence_frames=int(
                params.get("BACKGROUND_CONVERGENCE_FRAMES", 30) or 30
            ),
            enable_conservative_split=bool(
                params.get("ENABLE_CONSERVATIVE_SPLIT", False)
            ),
            morph_kernel_size=int(params.get("MORPH_KERNEL_SIZE", 5) or 5),
            dilation_kernel_size=int(params.get("DILATION_KERNEL_SIZE", 3) or 3),
            conservative_kernel_size=int(params.get("CONSERVATIVE_KERNEL_SIZE", 3) or 3),
            max_targets=int(params.get("MAX_TARGETS", 20) or 20),
            min_contour_area=float(params.get("MIN_CONTOUR_AREA", 5) or 5),
            max_contour_multiplier=int(params.get("MAX_CONTOUR_MULTIPLIER", 20) or 20),
            enable_size_filtering=bool(params.get("ENABLE_SIZE_FILTERING", False)),
            min_object_size=float(params.get("MIN_OBJECT_SIZE", 0) or 0),
            max_object_size=float(params.get("MAX_OBJECT_SIZE", float("inf"))),
            params=dict(params),
        )
```

- [ ] **Step 6: Write and run the config test**

Append to `tests/test_bgsub_stage.py`:

```python
from hydra_suite.core.inference.config import BgSubConfig


def test_bgsub_config_from_params_reads_legacy_keys():
    cfg = BgSubConfig.from_params(
        {"THRESHOLD_VALUE": 42, "BACKGROUND_PRIME_FRAMES": 99}
    )
    assert cfg.threshold_value == 42.0
    assert cfg.background_prime_frames == 99
    assert cfg.convergence_epsilon == 1e-4  # default (see canonical table)


def test_bgsub_config_retains_raw_params():
    cfg = BgSubConfig.from_params({"THRESHOLD_VALUE": 42, "CUSTOM": "x"})
    assert cfg.params["CUSTOM"] == "x"
```

Run: `$PY -m pytest tests/test_bgsub_stage.py -v`
Expected: all pass.

- [ ] **Step 7: Commit**

```bash
git add src/hydra_suite/core/inference/config.py src/hydra_suite/core/background/measure.py tests/test_bgsub_stage.py
git commit -m "feat(inference): add BgSubConfig and ellipse->corners derivation

BgSubConfig replaces scattered dict .get() defaults with a typed contract;
the dict stays at the API boundary since BackgroundModel/BackgroundMeasurer
still read legacy UPPER_SNAKE keys.

corners_from_ellipse lets bg-sub emit OBBResult.corners, the one field it
lacked. Order matches _corners_from_xywhr (TL/TR/BR/BL)."
```

---

## Task 7: `stages/bgsub.py` — the stage adapter

Follows the established `load_*` / `run_*` / `run_*_batch` shape (cf. `stages/cnn.py:19-113`). `BackgroundModel` slots into the `XModel`-wraps-an-opaque-backend role; priming is the "load".

**Files:**
- Create: `src/hydra_suite/core/inference/stages/bgsub.py`
- Test: `tests/test_bgsub_stage.py` (append)

**Interfaces:**
- Consumes: `BgSubConfig`, `corners_from_ellipse`, `BackgroundMeasurer`, `BackgroundModel`
- Produces:
  - `BgSubModel` dataclass with `.bg_model`, `.measurer`, `.close()`
  - `load_bgsub_model(config: BgSubConfig, runtime: RuntimeContext, video_path: str | None = None) -> BgSubModel`
  - `run_bgsub(frame, frame_idx, model, config, runtime, roi_mask=None) -> OBBResult`
  - `run_bgsub_batch(frames, frame_indices, model, config, runtime, roi_mask=None) -> list[OBBResult]`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_bgsub_stage.py`:

```python
from hydra_suite.core.inference.result import DETECTION_ID_STRIDE
from hydra_suite.core.inference.runtime import RuntimeContext
from hydra_suite.core.inference.stages.bgsub import load_bgsub_model, run_bgsub


def _cpu_runtime() -> RuntimeContext:
    from hydra_suite.core.inference.config import InferenceConfig

    return RuntimeContext.from_config(
        InferenceConfig(obb=None, bgsub=BgSubConfig.from_params(_params()), runtime_tier="cpu")
    )


def test_run_bgsub_emits_obbresult_with_corners(synthetic_video):
    cfg = BgSubConfig.from_params(_params(**_measure_params()))
    model = load_bgsub_model(cfg, _cpu_runtime(), video_path=synthetic_video)

    cap = cv2.VideoCapture(synthetic_video)
    cap.read()  # first frame primes the running state
    ok, frame = cap.read()
    cap.release()
    assert ok

    result = run_bgsub(frame, 1, model, cfg, _cpu_runtime())
    assert result.frame_idx == 1
    assert result.corners.ndim == 3
    assert result.corners.shape[1:] == (4, 2)
    assert result.centroids.shape[0] == result.corners.shape[0]
    if result.num_detections:
        assert np.isnan(result.confidences).all()
        assert result.detection_ids[0] == 1 * DETECTION_ID_STRIDE


def test_run_bgsub_is_deterministic(synthetic_video):
    """Same video + params twice -> identical detections. This is what makes
    the bgsub cache key sound."""
    runs = []
    for _ in range(2):
        cfg = BgSubConfig.from_params(_params(**_measure_params()))
        model = load_bgsub_model(cfg, _cpu_runtime(), video_path=synthetic_video)
        cap = cv2.VideoCapture(synthetic_video)
        outs = []
        for i in range(5):
            ok, frame = cap.read()
            if not ok:
                break
            outs.append(run_bgsub(frame, i, model, cfg, _cpu_runtime()).centroids)
        cap.release()
        runs.append(outs)

    assert len(runs[0]) == len(runs[1])
    for a, b in zip(runs[0], runs[1]):
        np.testing.assert_array_equal(a, b)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `$PY -m pytest tests/test_bgsub_stage.py -k run_bgsub -v`
Expected: FAIL — `ModuleNotFoundError: hydra_suite.core.inference.stages.bgsub`.

- [ ] **Step 3: Implement the stage**

Before writing, read `src/hydra_suite/core/inference/stages/cnn.py` in full and match its structure. Read `src/hydra_suite/core/background/model.py:475-510` for `_foreground_mask_cpu` / `generate_foreground_mask` and `src/hydra_suite/utils` for `apply_image_adjustments`.

Create `src/hydra_suite/core/inference/stages/bgsub.py`:

```python
"""Background-subtraction detection stage.

Mirrors the load/run/run_batch shape of the other stages. Unlike OBB there is
no model file: the "model" is a BackgroundModel primed from the video itself.

bg-sub is strictly sequential -- BackgroundModel carries cross-frame state
(running-max lightest background, EMA adaptive background, convergence latch).
Pipeline drives a single in-order consumer, so this is safe, but random access
is NOT: load_frame must be served from cache only.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import cv2
import numpy as np

from hydra_suite.core.background.measure import BackgroundMeasurer, corners_from_ellipse
from hydra_suite.core.background.model import BackgroundModel
from hydra_suite.utils.image import apply_image_adjustments

from ..config import BgSubConfig
from ..result import DETECTION_ID_STRIDE, OBBResult
from ..runtime import RuntimeContext

logger = logging.getLogger(__name__)


@dataclass
class BgSubModel:
    bg_model: BackgroundModel
    measurer: BackgroundMeasurer

    def close(self) -> None:
        # BackgroundModel holds only numpy/CuPy arrays; nothing to release.
        pass


def load_bgsub_model(
    config: BgSubConfig,
    runtime: RuntimeContext,
    video_path: str | None = None,
) -> BgSubModel:
    """Construct and prime the background model.

    Priming is this stage's equivalent of loading model weights.
    """
    bg_model = BackgroundModel(config.params)
    bg_model.configure_runtime(runtime)
    if video_path:
        cap = cv2.VideoCapture(video_path)
        try:
            bg_model.prime_background(cap)
        finally:
            cap.release()
    return BgSubModel(bg_model=bg_model, measurer=BackgroundMeasurer(config.params))


def _empty_result(frame_idx: int) -> OBBResult:
    return OBBResult(
        frame_idx=frame_idx,
        centroids=np.zeros((0, 2), np.float32),
        angles=np.zeros((0,), np.float32),
        sizes=np.zeros((0,), np.float32),
        shapes=np.zeros((0, 2), np.float32),
        confidences=np.zeros((0,), np.float32),
        corners=np.zeros((0, 4, 2), np.float32),
        detection_ids=np.zeros((0,), np.int64),
    )


def _to_gray(frame: np.ndarray, config: BgSubConfig, use_gpu: bool) -> np.ndarray:
    p = config.params
    resize_f = float(p.get("RESIZE_FACTOR", 1.0) or 1.0)
    if resize_f < 1.0:
        frame = cv2.resize(
            frame, (0, 0), fx=resize_f, fy=resize_f, interpolation=cv2.INTER_AREA
        )
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return apply_image_adjustments(
        gray, p["BRIGHTNESS"], p["CONTRAST"], p["GAMMA"], use_gpu
    )


def run_bgsub(
    frame: np.ndarray,
    frame_idx: int,
    model: BgSubModel,
    config: BgSubConfig,
    runtime: RuntimeContext,
    roi_mask: np.ndarray | None = None,
) -> OBBResult:
    """Detect on one frame. Frames MUST arrive in order."""
    gray = _to_gray(frame, config, model.bg_model.use_gpu)

    background = model.bg_model.update_and_get_background(gray, roi_mask)
    if background is None:
        return _empty_result(frame_idx)  # first frame: model has no history yet

    fg_mask = model.bg_model.generate_foreground_mask(gray, background)
    if roi_mask is not None:
        fg_mask = cv2.bitwise_and(fg_mask, roi_mask)

    if config.enable_conservative_split:
        fg_mask = model.measurer.apply_conservative_split(fg_mask, gray, background)

    meas, sizes, shapes, confidences = model.measurer.detect_objects(fg_mask, frame_idx)
    if not meas:
        return _empty_result(frame_idx)

    centroids = np.array([[m[0], m[1]] for m in meas], np.float32)
    angles = np.array([m[2] for m in meas], np.float32)
    sizes_arr = np.array(sizes, np.float32)
    shapes_arr = np.array(shapes, np.float32)

    corners = np.stack(
        [
            corners_from_ellipse(
                float(centroids[i][0]),
                float(centroids[i][1]),
                float(_major_from_shape(shapes_arr[i])),
                float(_minor_from_shape(shapes_arr[i])),
                float(angles[i]),
            )
            for i in range(len(meas))
        ]
    ).astype(np.float32)

    detection_ids = np.array(
        [frame_idx * DETECTION_ID_STRIDE + i for i in range(len(meas))], np.int64
    )

    return OBBResult(
        frame_idx=frame_idx,
        centroids=centroids,
        angles=angles,
        sizes=sizes_arr,
        shapes=shapes_arr,
        confidences=np.array(confidences, np.float32),
        corners=corners,
        detection_ids=detection_ids,
    )


def _major_from_shape(shape: np.ndarray) -> float:
    """Recover the ellipse major axis from (ellipse_area, aspect_ratio).

    detect_objects stores area = pi * (major/2) * (minor/2) and
    aspect = major/minor, so major = sqrt(4 * area * aspect / pi).
    """
    area, aspect = float(shape[0]), float(shape[1])
    if aspect <= 0 or area <= 0:
        return 0.0
    return float(np.sqrt(4.0 * area * aspect / np.pi))


def _minor_from_shape(shape: np.ndarray) -> float:
    area, aspect = float(shape[0]), float(shape[1])
    if aspect <= 0 or area <= 0:
        return 0.0
    return float(np.sqrt(4.0 * area / (np.pi * aspect)))


def run_bgsub_batch(
    frames: list[np.ndarray],
    frame_indices: list[int],
    model: BgSubModel,
    config: BgSubConfig,
    runtime: RuntimeContext,
    roi_mask: np.ndarray | None = None,
) -> list[OBBResult]:
    """Detect a window of frames.

    Sequential by necessity: each frame mutates the background model. The
    "batch" is a window boundary for the cache/pipeline, not vectorisation.
    """
    return [
        run_bgsub(frame, idx, model, config, runtime, roi_mask=roi_mask)
        for frame, idx in zip(frames, frame_indices)
    ]
```

Note: `_major_from_shape`/`_minor_from_shape` recover axes from the stored `(ellipse_area, aspect_ratio)`. If profiling shows this round-trip is lossy or slow, the cleaner fix is to have `detect_objects` return the axes directly — but do NOT change that in this task; it would alter the `shapes` contract that `OBBResult` consumers already rely on.

- [ ] **Step 4: Add `BackgroundModel.configure_runtime` (stub for now)**

Task 11 implements tier-driven selection. For now add to `src/hydra_suite/core/background/model.py`:

```python
    def configure_runtime(self, runtime) -> None:
        """Let the caller's runtime drive GPU selection.

        Task 11 replaces the ENABLE_GPU_BACKGROUND self-selection with this.
        """
        pass
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `$PY -m pytest tests/test_bgsub_stage.py -k run_bgsub -v`
Expected: 2 passed. `InferenceConfig(obb=None, bgsub=...)` already exists (Task 6b), so no xfail is needed — if these tests cannot construct a config, STOP and report rather than marking them xfail.

- [ ] **Step 6: Commit**

```bash
git add src/hydra_suite/core/inference/stages/bgsub.py src/hydra_suite/core/background/model.py tests/test_bgsub_stage.py
git commit -m "feat(inference): add the bgsub detection stage

Follows the established load/run/run_batch stage shape. BackgroundModel is
the 'model'; priming is the load. Emits OBBResult with derived corners and
NaN confidences, so no new result type and no downstream branch.

bg-sub is strictly sequential: run_bgsub_batch is a window boundary for the
cache/pipeline, not vectorisation. Documented that random access must be
served from cache only."
```

---

## Task 8: bg-sub cache key takes `BgSubConfig`; bump schema version

Every bg-sub cache on disk was produced under random priming (Task 2) and keyed by a hash ignoring the threshold (Task 1). Those artifacts are unsound and must be invalidated, not inherited.

**Files:**
- Modify: `src/hydra_suite/core/inference/cache/keys.py`
- Test: `tests/test_bgsub_cache_keys.py` (append)

**Interfaces:**
- Consumes: `BgSubConfig` (Task 6)
- Produces: `bgsub_detection_cache_key(config: BgSubConfig) -> CacheKey` — **signature changed from `params: dict`**

- [ ] **Step 1: Write the failing test**

Append to `tests/test_bgsub_cache_keys.py`:

```python
from hydra_suite.core.inference.config import BgSubConfig


def test_cache_key_accepts_bgsub_config():
    cfg = BgSubConfig.from_params(_base_params())
    key = bgsub_detection_cache_key(cfg)
    assert key.model_path == "background_subtraction"


def test_convergence_params_are_keyed():
    p = _base_params()
    p["BACKGROUND_CONVERGENCE_EPSILON"] = 0.05
    a = bgsub_detection_cache_key(BgSubConfig.from_params(p))
    p["BACKGROUND_CONVERGENCE_EPSILON"] = 0.5
    b = bgsub_detection_cache_key(BgSubConfig.from_params(p))
    assert a.config_hash != b.config_hash


def test_convergence_frames_are_keyed():
    p = _base_params()
    p["BACKGROUND_CONVERGENCE_FRAMES"] = 30
    a = bgsub_detection_cache_key(BgSubConfig.from_params(p))
    p["BACKGROUND_CONVERGENCE_FRAMES"] = 60
    b = bgsub_detection_cache_key(BgSubConfig.from_params(p))
    assert a.config_hash != b.config_hash
```

Update the four Task 1 tests to pass `BgSubConfig.from_params(...)` instead of a raw dict.

- [ ] **Step 2: Run test to verify it fails**

Run: `$PY -m pytest tests/test_bgsub_cache_keys.py -v`
Expected: FAIL — `bgsub_detection_cache_key` calls `.get()` on a `BgSubConfig`.

- [ ] **Step 3: Add the convergence params and change the signature**

In `src/hydra_suite/core/inference/cache/keys.py`, add to `_BGSUB_KEY_PARAMS`:

```python
    "BACKGROUND_CONVERGENCE_EPSILON",
    "BACKGROUND_CONVERGENCE_FRAMES",
    "BACKGROUND_CONVERGENCE_PIXEL_DELTA",
```

All THREE must be keyed. They decide which background is subtracted on every
frame past the latch, so they change detections as directly as the neighbouring
`ENABLE_ADAPTIVE_BACKGROUND` / `BACKGROUND_LEARNING_RATE`, which are keyed. If
they are not keyed, retuning epsilon and rerunning silently serves a stale
cache — the exact bug Task 1 fixed.

Change `bgsub_detection_cache_key` to:

```python
def bgsub_detection_cache_key(config: BgSubConfig) -> CacheKey:
    """Cache key for background-subtraction detections.

    There is no model file, so model_path is a sentinel and the
    detection-affecting parameters are hashed into config_hash. Callers should
    fold in the video signature via ``with_video_signature`` so the cache is
    bound to the source file.

    Soundness depends on deterministic priming (core/background/model.py samples
    evenly-spaced frames, not unseeded random ones) -- without that, identical
    params would legitimately produce different detections and this key would
    be a lie.
    """
    params = config.params
    payload = "|".join(f"{k}={params.get(k)}" for k in _BGSUB_KEY_PARAMS)
    return CacheKey(
        schema_version=CACHE_SCHEMA_VERSION,
        model_path="background_subtraction",
        model_mtime=0.0,
        config_hash=_sha(payload),
    )
```

Add the import at the top of `keys.py`: `from ..config import BgSubConfig` (match the existing import style for `HeadTailConfig` etc.).

- [ ] **Step 4: Bump the cache schema version**

Find `CACHE_SCHEMA_VERSION` (grep: `grep -rn "CACHE_SCHEMA_VERSION" src/`) and increment it by one, with a comment:

```python
# Bumped for bg-sub: every prior bgsub cache was produced under unseeded
# random priming and keyed by a hash that ignored THRESHOLD_VALUE. Those
# artifacts are unsound and must not be inherited.
CACHE_SCHEMA_VERSION = <previous + 1>
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `$PY -m pytest tests/test_bgsub_cache_keys.py -v`
Expected: 7 passed.

Run: `$PY -m pytest tests/test_inference_cache_keys.py -q --timeout=60`
Expected: cache tests that assert a literal schema version may fail — update those literals; do not revert the bump.

- [ ] **Step 6: Commit**

```bash
git add src/hydra_suite/core/inference/cache/keys.py tests/test_bgsub_cache_keys.py
git commit -m "feat(cache): bgsub key takes BgSubConfig; bump schema version

Keys the convergence params and bumps CACHE_SCHEMA_VERSION. Every bgsub
cache on disk predates deterministic priming and the key-name fix, so it is
unsound and must be invalidated rather than inherited."
```

---

## Task 9: Wire bg-sub into `InferenceConfig` and `InferenceRunner`

`InferenceConfig.obb` is a REQUIRED field (`config.py:229`) and `_AllModels.obb` is non-optional (`runner.py:80`). bg-sub is an *alternative* detection source, so both must become optional with an exactly-one validation.

**Files:**
- (config.py changes moved to Task 6b — `InferenceConfig.obb` optional, `bgsub` field, `detection_source`, exactly-one validation are ALREADY DONE. Do not redo them.)
- Modify: `src/hydra_suite/core/inference/runner.py` (`_AllModels`, `_load_all_models`, `_open_caches`, `run_realtime`, `_build_pipeline`, `load_frame`, `close`)
- Test: `tests/test_bgsub_stage.py` (append)

**Interfaces:**
- Consumes: `load_bgsub_model`, `run_bgsub`, `run_bgsub_batch`, `bgsub_detection_cache_key`
- Produces:
  - `InferenceConfig.obb: OBBConfig | None = None`, `InferenceConfig.bgsub: BgSubConfig | None = None`
  - `InferenceConfig.detection_source -> Literal["obb", "bgsub"]`
  - `_AllModels.obb: OBBModels | None`, `_AllModels.bgsub: BgSubModel | None`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_bgsub_stage.py`:

```python
from hydra_suite.core.inference.config import InferenceConfig, InferenceConfigError
from hydra_suite.core.inference.config import OBBConfig


def test_config_requires_exactly_one_detection_source():
    with pytest.raises(InferenceConfigError, match="exactly one"):
        InferenceConfig(obb=None, bgsub=None)

    with pytest.raises(InferenceConfigError, match="exactly one"):
        InferenceConfig(obb=OBBConfig(), bgsub=BgSubConfig.from_params({}))


def test_config_detection_source_reports_bgsub():
    cfg = InferenceConfig(obb=None, bgsub=BgSubConfig.from_params({}))
    assert cfg.detection_source == "bgsub"


def test_config_detection_source_reports_obb():
    cfg = InferenceConfig(obb=OBBConfig())
    assert cfg.detection_source == "obb"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `$PY -m pytest tests/test_bgsub_stage.py -k detection_source -v`
Expected: FAIL — `InferenceConfig` has no `bgsub` field and `obb` is required.

- [ ] **Step 3: Make the detection source a choice**

In `src/hydra_suite/core/inference/config.py`, change `InferenceConfig`:

```python
@dataclass
class InferenceConfig:
    # Exactly one detection source must be set. OBB is the YOLO path; bgsub is
    # background subtraction. They are alternatives, not composable.
    obb: OBBConfig | None = None
    bgsub: BgSubConfig | None = None
    headtail: HeadTailConfig | None = None
```

(the rest of the fields unchanged)

Add to `__post_init__`:

```python
    def __post_init__(self) -> None:
        self._validate_pipeline_depth()
        self._validate_detection_source()

    def _validate_detection_source(self) -> None:
        if (self.obb is None) == (self.bgsub is None):
            raise InferenceConfigError(
                "InferenceConfig requires exactly one detection source: set "
                "either `obb` or `bgsub`, not both and not neither."
            )

    @property
    def detection_source(self) -> Literal["obb", "bgsub"]:
        return "obb" if self.obb is not None else "bgsub"
```

Guard `_collect_all_runtimes` against `obb is None`:

```python
        if self.obb is not None and self.obb.direct:
            runtimes.add(self.obb.direct.compute_runtime)
        if self.obb is not None and self.obb.sequential:
            runtimes.add(self.obb.sequential.detect_compute_runtime)
            runtimes.add(self.obb.sequential.obb_compute_runtime)
```

(bg-sub has no legacy per-stage `compute_runtime`, so it contributes nothing to migration.)

Update `_dict_to_config` and `_config_to_dict` to round-trip `bgsub`, mirroring how `obb` is handled. Read both functions first and follow their existing style.

- [ ] **Step 4: Wire the runner**

In `src/hydra_suite/core/inference/runner.py`:

`_AllModels`:

```python
@dataclass
class _AllModels:
    obb: OBBModels | None
    bgsub: BgSubModel | None
    headtail: HeadTailModel | None
    cnn: list[CNNModel]
    pose: PoseModel | None
    apriltag: AprilTagModel | None
```

`_load_all_models` — replace lines 130-146:

```python
    from .stages.bgsub import load_bgsub_model

    obb = None
    bgsub = None
    if config.detection_source == "obb":
        obb = load_obb_models(
            config.obb, runtime, batch_size=config.detection_batch_size
        )
    else:
        # bg-sub has no model file; the "load" is priming from the video, and
        # the cache key needs no model at all -- so cache_only can skip it
        # entirely, unlike OBB.
        if not cache_only:
            bgsub = load_bgsub_model(config.bgsub, runtime, video_path=video_path)

    if cache_only:
        logger.debug(
            "InferenceRunner cache_only=True: skipping HeadTail/CNN/Pose/AprilTag "
            "model init (backward/replay pass reads from cache only)."
        )
        return _AllModels(
            obb=obb, bgsub=bgsub, headtail=None, cnn=[], pose=None, apriltag=None
        )

    headtail = (
        load_headtail_model(config.headtail, runtime)
        if config.headtail is not None
        else None
    )
    cnn = [load_cnn_model(c, runtime) for c in config.cnn_phases]
    pose = load_pose_model(config.pose, runtime) if config.pose is not None else None
    apriltag = load_apriltag_model(config.apriltag) if config.apriltag.enabled else None
    return _AllModels(
        obb=obb, bgsub=bgsub, headtail=headtail, cnn=cnn, pose=pose, apriltag=apriltag
    )
```

`_load_all_models` needs `video_path` — add it as a keyword parameter and pass `str(video_path) if video_path else None` from `InferenceRunner.__init__` (which already holds it at `runner.py:341`). Store it as `self._video_path` before the `_load_all_models` call.

`_open_caches` — swap the detection key:

```python
    from ..cache.keys import bgsub_detection_cache_key

    detection_key = (
        detection_cache_key(config.obb)
        if config.detection_source == "obb"
        else bgsub_detection_cache_key(config.bgsub)
    )
```

then use `key=_k(detection_key)` in the `DetectionCacheHandle`.

`run_realtime` and `_build_pipeline` — wherever `run_obb` / `run_obb_batch` is called, branch on `config.detection_source`. Read `runner.py:387-600` and `runner.py:602-716` first, then dispatch:

```python
    if self.config.detection_source == "bgsub":
        raw_obb = run_bgsub(
            frame, frame_idx, self._models.bgsub, self.config.bgsub, self.runtime,
            roi_mask=roi_mask,
        )
    else:
        raw_obb = run_obb(...)  # existing call, unchanged
```

bg-sub never returns `_RawOBBTensors` (it is CPU numpy throughout), so any `tensor_on_cuda` / `handoff` handling can be skipped on the bgsub branch.

`close` — guard the OBB close and add bgsub:

```python
        if self._models.obb is not None:
            self._models.obb.close()
        if self._models.bgsub is not None:
            self._models.bgsub.close()
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `$PY -m pytest tests/test_bgsub_stage.py -v`
Expected: all pass.

Run: `$PY -m pytest tests/test_background_model.py tests/test_background_model_integration.py tests/test_detectors_engine.py tests/test_detector_integration.py tests/test_inference_cache_keys.py -q --timeout=60`
Expected: no failures beyond the known-bad listed in "Execution Environment". NEVER run the full suite. Any test constructing `InferenceConfig(obb=...)` positionally still works since `obb` stays first.

- [ ] **Step 7: Verify imports still resolve**

```bash
$PY -c "import hydra_suite"
python -c "from hydra_suite.core.inference import InferenceConfig, InferenceRunner"
```
Expected: no output, exit 0.

- [ ] **Step 8: Commit**

```bash
git add src/hydra_suite/core/inference/ tests/test_bgsub_stage.py
git commit -m "feat(inference): wire bgsub into InferenceConfig and InferenceRunner

obb was a required field and _AllModels.obb non-optional; bg-sub is an
alternative detection source, so both become optional with an exactly-one
validation and a detection_source property.

bgsub needs no model for cache-key validation (unlike OBB, which cache_only
must still load), so cache_only skips it entirely."
```

---

## Task 10: Runtime tier support for bg-sub

`BackgroundModel._setup_gpu_acceleration` (`model.py:197-239`) gates on the `ENABLE_GPU_BACKGROUND` param and then self-selects CUDA > MPS > CPU. It never consults `runtime_tier`, so under the runner `runtime_tier="cpu"` would still run CuPy on the GPU. Invert control: `RuntimeContext` drives selection.

**Files:**
- Modify: `src/hydra_suite/runtime/resolver.py`
- Modify: `src/hydra_suite/core/background/model.py` (`configure_runtime`)
- Modify: `src/hydra_suite/runtime/compute_runtime.py` (`_pipeline_supports_runtime`)
- Test: `tests/test_bgsub_stage.py` (append)

**Interfaces:**
- Consumes: Task 7's `configure_runtime` stub
- Produces: `BackgroundModel.configure_runtime(runtime: RuntimeContext) -> None` — real implementation

Per `docs/developer-guide/runtime-integration.md`, read that checklist before starting.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_bgsub_stage.py`:

```python
def test_cpu_tier_does_not_enable_gpu():
    """runtime_tier='cpu' must win over ENABLE_GPU_BACKGROUND=True.

    Regression: BackgroundModel used to self-select CUDA>MPS>CPU and never
    consulted the tier, so a cpu-tier run could silently use CuPy.
    """
    model = BackgroundModel(_params(ENABLE_GPU_BACKGROUND=True))
    cfg = InferenceConfig(
        obb=None, bgsub=BgSubConfig.from_params(_params()), runtime_tier="cpu"
    )
    model.configure_runtime(RuntimeContext.from_config(cfg))
    assert model.use_gpu is False
    assert model.gpu_type is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `$PY -m pytest tests/test_bgsub_stage.py::test_cpu_tier_does_not_enable_gpu -v`
Expected: FAIL — `configure_runtime` is a no-op stub (Task 7).

- [ ] **Step 3: Add the `"bgsub"` pipeline key to the resolver**

Read `src/hydra_suite/runtime/resolver.py:13-60` (`RuntimeResolver.resolve`) and add a `"bgsub"` branch following the `"obb"` pattern:

| tier | resolved |
|---|---|
| `cpu` | `ResolvedBackend(backend="torch", device="cpu", used_fallback=False)` — CPU means Numba |
| `gpu` | CUDA where available else MPS else CPU |
| `gpu_fast` | same as `gpu`, `used_fallback=True` — bg-sub has no TensorRT/CoreML path and needs none: it is elementwise work, not a network |

Match the exact `ResolvedBackend` construction the `"obb"` branch uses.

- [ ] **Step 4: Implement `configure_runtime`**

Replace the Task 7 stub in `src/hydra_suite/core/background/model.py`:

```python
    def configure_runtime(self, runtime) -> None:
        """Let the caller's RuntimeContext drive GPU selection.

        Previously the model self-selected CUDA > MPS > CPU from
        ENABLE_GPU_BACKGROUND alone and never consulted runtime_tier, so a
        cpu-tier run could silently execute CuPy kernels. The tier now wins.
        """
        self.use_gpu = False
        self.gpu_type = None
        self.gpu_device = None
        self.torch_device = None

        if runtime.device == "cpu":
            logger.info("bg-sub: cpu tier -- using Numba CPU path")
            return

        if runtime.cuda_mode and CUDA_AVAILABLE:
            try:
                self.gpu_device = cp.cuda.Device(self.params.get("GPU_DEVICE_ID", 0))
                self.gpu_type = "cuda"
                self.use_gpu = True
                logger.info("bg-sub: CUDA (CuPy) path")
                return
            except Exception as e:
                logger.warning("bg-sub: CUDA init failed (%s); falling back to CPU", e)
                self.use_gpu = False
                self.gpu_type = None
                return

        if runtime.device == "mps" and MPS_AVAILABLE:
            try:
                self.torch_device = torch.device("mps")
                _ = torch.zeros(1, device=self.torch_device)
                self.gpu_type = "mps"
                self.use_gpu = True
                logger.info("bg-sub: MPS (PyTorch) path")
                return
            except Exception as e:
                logger.warning("bg-sub: MPS init failed (%s); falling back to CPU", e)
                self.use_gpu = False
                self.gpu_type = None
                return

        logger.info("bg-sub: no GPU available for tier -- using Numba CPU path")
```

Read `src/hydra_suite/core/inference/runtime.py:33-90` first to confirm the exact `RuntimeContext` attribute names (`device`, `cuda_mode`, `tensor_on_cuda`) before writing this.

Leave `_setup_gpu_acceleration` in place — legacy non-runner callers (`bg_optimizer.py`) still construct `BackgroundModel` without a runtime and rely on it. Add a note to its docstring:

```python
        """
        Initialize GPU acceleration if available.
        Priority: CUDA (NVIDIA) > MPS (Apple Silicon) > CPU fallback

        NOTE: this is the legacy self-selection path, used by callers that
        construct BackgroundModel without a RuntimeContext. Runner-driven
        callers should use configure_runtime(), which lets runtime_tier win.
        """
```

- [ ] **Step 5: Add the capability rule**

In `src/hydra_suite/runtime/compute_runtime.py`, add `"bgsub"` to `_pipeline_supports_runtime()`. bg-sub supports `cpu`, `mps`, `cuda`; it does NOT support `onnx_cpu`, `onnx_cuda`, `onnx_coreml`, `tensorrt`. Read the existing rules and match their structure.

- [ ] **Step 6: Run tests to verify they pass**

Run: `$PY -m pytest tests/test_bgsub_stage.py -v`
Expected: all pass.

- [ ] **Step 7: Commit**

```bash
git add src/hydra_suite/runtime/ src/hydra_suite/core/background/model.py tests/test_bgsub_stage.py
git commit -m "feat(runtime): bg-sub honors runtime_tier instead of self-selecting

BackgroundModel gated on ENABLE_GPU_BACKGROUND and then self-selected
CUDA>MPS>CPU, never consulting runtime_tier -- so under the runner a
cpu-tier run could silently execute CuPy kernels. configure_runtime inverts
control: the tier wins.

Adds the 'bgsub' pipeline key. gpu_fast resolves to gpu with used_fallback
(bg-sub is elementwise work, not a network -- no TensorRT/CoreML path)."
```

---

## Task 11: Migrate `worker.py` to the runner for bg-sub

**Files:**
- Modify: `src/hydra_suite/core/tracking/worker.py:20, 1160, 2297, 2315`
- Test: manual + equivalence gate (Task 14)

**Interfaces:**
- Consumes: `InferenceRunner` with `bgsub` wired (Task 9)
- Produces: `worker.py` no longer imports `create_detector`

- [ ] **Step 1: Read the current call sites**

```bash
sed -n '1155,1175p' src/hydra_suite/core/tracking/worker.py
sed -n '2290,2325p' src/hydra_suite/core/tracking/worker.py
```

Understand how `bg_model`, `detector`, `fg_mask`, and `detect_objects` interact before editing. `worker.py` is 19k lines and mid-refactor (see CLAUDE.md's Simplification Sprint); make the minimal change.

- [ ] **Step 2: Build a `BgSubConfig` where the detector is constructed**

At `worker.py:1160`, replace `create_detector(p)` with a runner constructed from params. Follow whatever pattern the OBB path already uses in this file to construct `InferenceRunner` (grep: `grep -n "InferenceRunner" src/hydra_suite/core/tracking/worker.py`). Build the config with:

```python
from hydra_suite.core.inference.config import BgSubConfig, InferenceConfig

inference_config = InferenceConfig(
    obb=None,
    bgsub=BgSubConfig.from_params(p),
    runtime_tier=<same tier the OBB path resolves>,
    detection_batch_size=p.get("DETECTION_BATCH_SIZE", 1),
)
```

- [ ] **Step 3: Replace the detect call**

At `worker.py:2315`, replace `detector.detect_objects(fg_mask, actual_frame_index)` with the runner's `FrameResult`. The runner now owns the gray conversion, background update, mask generation, and measurement — so lines 2297-2315's manual `bg_model.update_and_get_background(...)` + `generate_foreground_mask(...)` + `detect_objects(...)` sequence collapses into one `runner.run_realtime(frame, frame_idx, roi_mask=ROI_mask_current)` call.

Unpack `FrameResult.obb` into whatever `meas`/`sizes`/`shapes`/`confidences` locals the downstream Kalman/assignment code expects:

```python
result = runner.run_realtime(frame, actual_frame_index, roi_mask=ROI_mask_current)
obb = result.obb
meas = [
    np.array([obb.centroids[i][0], obb.centroids[i][1], obb.angles[i]], np.float32)
    for i in range(obb.num_detections)
]
sizes = list(obb.sizes)
shapes = [tuple(s) for s in obb.shapes]
confidences = list(obb.confidences)
```

- [ ] **Step 4: Remove the import**

At `worker.py:20`, drop `create_detector` from the `core.detectors` import. If it becomes empty, delete the import line.

- [ ] **Step 5: Verify**

```bash
$PY -c "import hydra_suite.core.tracking.worker"
$PY -m pytest tests/test_background_model.py tests/test_detectors_engine.py tests/test_inference_cache_keys.py -q --timeout=60
```
Expected: import succeeds; no new failures.

- [ ] **Step 6: Commit**

```bash
git add src/hydra_suite/core/tracking/worker.py
git commit -m "refactor(tracking): worker uses InferenceRunner for bg-sub detection

The manual update_and_get_background + generate_foreground_mask +
detect_objects sequence collapses into one run_realtime call; the runner
owns that pipeline now."
```

---

## Task 12: Migrate the remaining call sites

**Files:**
- Modify: `src/hydra_suite/trackerkit/gui/workers/preview_worker.py:394, 691, 762, 832, 1159, 1751`
- Modify: `src/hydra_suite/data/dataset_generation.py:415, 421, 531, 552, 579`
- Modify: `src/hydra_suite/core/tracking/optimization/optimizer_workers.py:367, 372, 441`

**Interfaces:**
- Consumes: `InferenceRunner`, `BgSubConfig`, `BackgroundMeasurer`
- Produces: no `src/` module imports `ObjectDetector` or `create_detector`

- [ ] **Step 1: Migrate `preview_worker.py`**

Read each site first: `grep -n "ObjectDetector\|DetectionFilter\|_advanced_config_value" src/hydra_suite/trackerkit/gui/workers/preview_worker.py`

- `ObjectDetector(...)` → `BackgroundMeasurer(...)` (import from `hydra_suite.core.background.measure`). The preview shows intermediate masks, so it legitimately uses the measurer directly rather than the runner.
- Update the `detect_objects` unpacking from 5-tuple to 4-tuple: `meas, sizes, shapes, _, confidences = ...` becomes `meas, sizes, shapes, confidences = ...`
- `DetectionFilter` (lines 1159, 1751) → `apply_detection_filter` from `hydra_suite.core.inference.api`, matching how `optimizer.py:36` already imports it.
- `_advanced_config_value` is in `core/detectors/_utils.py`, which survives (YOLO still uses it). Leave that import alone.

- [ ] **Step 2: Migrate `dataset_generation.py`**

Replace `create_detector(params)` with an `InferenceRunner` built as in Task 11. Update `detect_objects` unpacking to the 4-tuple. Where `detect_objects_batched` is used on the YOLO path, leave it alone — this task only touches the bg-sub branch.

- [ ] **Step 3: Migrate `optimizer_workers.py`**

It already imports the new filter shim (`optimizer_workers.py:38`). Replace the remaining `create_detector` (line 367) and `detect_objects` (line 372) with `BackgroundMeasurer`; leave `detect_objects_batched` (line 441, YOLO) alone.

- [ ] **Step 4: Verify no legacy references remain**

```bash
grep -rn "create_detector\|ObjectDetector" src/
```
Expected: only `core/detectors/factory.py` and `core/detectors/__init__.py` (deleted in Task 13).

- [ ] **Step 5: Verify every entry point imports**

```bash
$PY -c "import hydra_suite"
for kit in trackerkit posekit classkit refinekit detectkit filterkit; do
  $PY -c "import hydra_suite.$kit" || echo "FAILED: $kit"
done
$PY -m pytest tests/test_background_model.py tests/test_detectors_engine.py tests/test_inference_cache_keys.py -q --timeout=60
```
Expected: no ImportError; no new test failures.

- [ ] **Step 6: Commit**

```bash
git add src/hydra_suite/trackerkit/gui/workers/preview_worker.py src/hydra_suite/data/dataset_generation.py src/hydra_suite/core/tracking/optimization/optimizer_workers.py
git commit -m "refactor: migrate remaining bg-sub call sites off core/detectors

preview_worker/dataset_generation/optimizer_workers now use
BackgroundMeasurer or InferenceRunner. Updates detect_objects unpacking for
the dropped yolo_results stub. DetectionFilter -> apply_detection_filter,
matching what optimizer.py already did."
```

---

## Task 13: Delete `create_detector` and move `bg_optimizer.py`

`create_detector` is the last string-keyed factory in the codebase (`core/detectors/factory.py:8`). With bg-sub gone from `core/detectors`, it dispatches to exactly one branch.

**Files:**
- Delete: `src/hydra_suite/core/detectors/factory.py`
- Move: `src/hydra_suite/core/detectors/bg_optimizer.py` → `src/hydra_suite/core/background/optimizer.py`
- Modify: `src/hydra_suite/core/detectors/__init__.py`
- Modify: `src/hydra_suite/trackerkit/gui/dialogs/bg_parameter_helper.py:45`

**Interfaces:**
- Consumes: Task 12's migrations
- Produces: `core/detectors/` contains only YOLO/OBB code

- [ ] **Step 1: Delete the factory**

```bash
git rm src/hydra_suite/core/detectors/factory.py
```

Remove `create_detector` from `src/hydra_suite/core/detectors/__init__.py`'s imports and `__all__`.

- [ ] **Step 2: Move `bg_optimizer.py`**

```bash
git mv src/hydra_suite/core/detectors/bg_optimizer.py src/hydra_suite/core/background/optimizer.py
```

Fix its imports: it imported `ObjectDetector` from `.bg_detector` — change to `from .measure import BackgroundMeasurer` and rename usages. Update its `detect_objects` unpacking to the 4-tuple.

Update `src/hydra_suite/trackerkit/gui/dialogs/bg_parameter_helper.py:45` to import from `hydra_suite.core.background.optimizer`.

**Scope note:** `bg_optimizer.py` imports `QThread`/`Signal` inside a `core` package — a real dependency-direction violation. It is explicitly OUT OF SCOPE per the spec. Move it as-is; do not restructure.

- [ ] **Step 3: Verify**

```bash
grep -rn "create_detector\|bg_detector\|bg_optimizer" src/ tests/
```
Expected: no output.

```bash
$PY -c "import hydra_suite"
$PY -c "import hydra_suite.trackerkit"
$PY -m pytest tests/test_background_model.py tests/test_detectors_engine.py tests/test_inference_cache_keys.py -q --timeout=60
```
Expected: no ImportError; no new failures.

- [ ] **Step 4: Confirm the endpoint honestly**

```bash
ls src/hydra_suite/core/detectors/
```
Expected: `__init__.py`, `_direct_obb_runtime.py`, `_obb_geometry.py`, `_runtime_artifacts.py`, `_utils.py`, `detection_filter.py`, `yolo_detector.py` — all YOLO/OBB.

`core/detectors/` is NOT retired. `_direct_obb_runtime.py` and `_runtime_artifacts.py` remain live dependencies of the new pipeline (`core/inference/runtime_artifacts.py:217`). Relocating them is a separate project.

- [ ] **Step 5: Commit**

```bash
git add -A src/
git commit -m "refactor(detectors): delete create_detector; move bg_optimizer to background

create_detector was the last string-keyed factory; with bg-sub gone from
core/detectors it dispatched to one branch. bg_optimizer joins the model it
tunes.

core/detectors/ is now purely YOLO/OBB. It is NOT retired: _direct_obb_runtime
and _runtime_artifacts remain live dependencies of core/inference. That
relocation is a separate project."
```

---

## Task 14: Calibrate `eps`/`K` and run the acceptance gate

**Files:**
- Modify: `src/hydra_suite/trackerkit/cli_config.py` (final defaults)
- Modify: `docs/superpowers/specs/2026-07-16-bgsub-inference-stage-design.md` (record findings)

**Interfaces:**
- Consumes: everything above
- Produces: calibrated `BACKGROUND_CONVERGENCE_EPSILON` / `BACKGROUND_CONVERGENCE_FRAMES` defaults

- [ ] **Step 1: Fetch the fixtures**

```bash
bash tools/equivalence/fixtures/fetch_fixtures.sh
```

- [ ] **Step 2: Measure legacy's switchover frame**

Check out the pre-Task-4 commit in a scratch worktree, add a temporary log line where `tracking_stabilized` was set (`worker.py:3656`), and run `worm_bgsub`. Record the frame index `S_legacy`.

```bash
git worktree add /tmp/bgsub-baseline <commit-before-task-4>
```

- [ ] **Step 3: Measure the convergence switchover frame**

On the current branch, run `worm_bgsub` and record `S_convergence` from the `"Background converged"` log line added in Task 4.

- [ ] **Step 4: Tune `eps`/`K` until the two align**

Sweep `BACKGROUND_CONVERGENCE_EPSILON` and `BACKGROUND_CONVERGENCE_FRAMES` in the `worm_bgsub` config, minimizing `|S_convergence - S_legacy|`. Record the values.

- [ ] **Step 5: Run the acceptance gate**

```bash
bash tools/equivalence/run_matrix.sh worm_bgsub
```
Expected: PASS at default tolerances (`pos_p99 <= 0.5px`, `theta_mean <= theta_atol`, `unmatched == 0`).

**If it fails:** diagnose before touching tolerances.
- `unmatched > 0` → a blob split or merged. Check whether the frame sits near `S`.
- `pos_p99` slightly over → check whether it is the prime change (isolate by running with `BACKGROUND_PRIME_FRAMES` high enough that sampling barely matters) or the switchover.
- **If the gate cannot pass no matter how `S` is tuned, STOP.** That is real evidence the change is not as behavior-preserving as the design assumed. Report it; do not loosen `--pos-atol` to make it green.

- [ ] **Step 6: Set the tuned defaults**

Update `background_convergence_epsilon` / `background_convergence_frames` defaults in `src/hydra_suite/trackerkit/cli_config.py` to the calibrated values.

- [ ] **Step 7: Record findings in the spec**

Append a "Calibration Results" section to `docs/superpowers/specs/2026-07-16-bgsub-inference-stage-design.md`: `S_legacy`, `S_convergence`, the tuned `eps`/`K`, and the gate's `pos_p99` / `unmatched`.

Note the coverage caveat: `worm_bgsub` is a single worm clip and the only bg-sub coverage. Worms on a plate may converge differently from an ant colony. If bg-sub runs on ant videos in practice, these defaults may be wrong for them — recommend adding an ant bg-sub clip to the fixtures.

- [ ] **Step 8: Clean up and commit**

```bash
git worktree remove /tmp/bgsub-baseline
git add -A
git commit -m "feat(background): calibrate convergence defaults against worm_bgsub

Tunes BACKGROUND_CONVERGENCE_EPSILON/FRAMES so the convergence-derived
switchover matches the legacy tracking-derived one on worm_bgsub, and
records the gate results in the spec.

Caveat recorded: worm_bgsub is the only bg-sub coverage, so these defaults
are calibrated on worms alone."
```

---

## Self-Review

**Spec coverage:**

| Spec section | Task |
|---|---|
| Defect 1 (cache key names) | 1 |
| Defect 2 (unseeded priming) | 2 |
| Defect 3 (adaptive-disabled switch) | 3 |
| Convergence rule | 4 |
| `measure.py` move | 5 |
| `BgSubConfig` + corners | 6 |
| `stages/bgsub.py` | 7 |
| Cache key + schema bump | 8 |
| Runner wiring | 9 |
| Runtime tiers | 10 |
| Call-site migration | 11, 12 |
| `create_detector` death | 13 |
| Acceptance gate + calibration | 14 |
| Out-of-scope (bg_optimizer Qt, detectors retirement) | noted in 13 |

**Type consistency check:** `BackgroundMeasurer.detect_objects` returns a 4-tuple in Tasks 5, 7, 11, 12, 13. `update_and_get_background(gray, roi_mask)` (2 args) consistent from Task 4 onward. `bgsub_detection_cache_key(config: BgSubConfig)` consistent in Tasks 8, 9. `corners_from_ellipse(cx, cy, major, minor, angle_rad)` consistent in Tasks 6, 7.

**Resolved (was a suspected gap):** Task 7's `_major_from_shape`/`_minor_from_shape` recover ellipse axes from `(area, aspect)` rather than carrying them through, to avoid widening the `shapes` contract mid-plan. The inversion is algebraically exact — `detect_objects` stores `area = pi*(ax1/2)*(ax2/2)` and `aspect = ax1/ax2`, so `ax1 = sqrt(4*area*aspect/pi)` recovers it losslessly up to float rounding. Measured worst-case error over 200k random ellipses: **2.8e-14 px**, against a 0.5px gate — 13 orders of magnitude of headroom. Do NOT chase this if Task 14's gate fails; it is not the cause.

Note `measure.py` CANNOT return `OBBResult` directly: `OBBResult` lives in `core/inference/result.py`, and `core/inference -> core/background` is the established edge, so the reverse would be circular. Assembling `OBBResult` is necessarily the stage's job.
