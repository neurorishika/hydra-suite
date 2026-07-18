# Detector Retirement — Plan 4: Retire `core/detectors` + Verify (Phases E + F + deferred) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Delete `src/hydra_suite/core/detectors/` entirely, migrate its test coverage to the `core/inference` locations, close the three deferred bg-sub follow-ups, and gate the whole retirement on a real frame-by-frame parity run.

**Architecture:** After Plans 1–3, `core/detectors` has no production importers left except `_utils._advanced_config_value` (removed in Plan 2's preview migration) and `direct_executors` (already relocated to `core/inference` in Plan 1). This plan removes the last file-path test dependencies, relocates `_advanced_config_value` to a public home, deletes the package, and handles the Qt-in-core and dead-param cleanups the bg-sub work deferred here.

**Tech Stack:** Python 3, PySide6, pytest, numpy.

## Global Constraints

- **Depends on Plans 1, 2, and 3 all landed.** In particular: Plan 2 removed every production `YOLOOBBDetector`/`DetectionFilter`/`_advanced_config_value` import; Plan 3 deleted `tests/test_benchmark_models.py` (which required `YOLOOBBDetector` importable).
- Retire files with `git rm`; per the repo's `legacy/` policy, superseded code may instead be moved to `legacy/` for one release — but here the parity work is the gate, so prove-dead-then-delete one file at a time.
- The parity gate (Task 7) is mandatory before the final `git rm` of `yolo_detector.py`.
- Run `make format` before each commit.

---

## File Structure

- `src/hydra_suite/core/inference/config.py` — **add** public `advanced_config_value(...)` (relocated from `_utils`).
- `src/hydra_suite/core/detectors/_utils.py` — **delete** after relocation.
- `src/hydra_suite/core/detectors/{_obb_geometry,_runtime_artifacts,detection_filter,yolo_detector}.py` + `__init__.py` — **delete**.
- `tests/test_detectors_engine.py` — **split/relocate** the still-relevant coverage to `core/inference` test files; delete the rest.
- `tests/test_yolo_detector_seq_obb_imgsz.py` — **delete or port** to the inference OBB stage.
- `src/hydra_suite/core/tracking/pose/pose_pipeline.py` — **verify dead → delete** (the `filter_raw_detections` consumer).
- `src/hydra_suite/core/background/optimizer.py` — **split** Qt workers out of `core` (deferred follow-up).
- `resources/configs/*.json`, `trackerkit/cli_config.py`, `trackerkit/gui/orchestrators/config.py` — **deprecate** `MIN_TRACKING_COUNTS` / `min_track_seconds` (deferred follow-up).
- Fresh tests replacing `test_tracking_worker_realtime_live_features.py` (deferred follow-up).

---

### Task 1: Relocate `_advanced_config_value` to a public home; delete `_utils.py`

After Plan 2, the only importer of `_utils._advanced_config_value` was `preview_worker.py`, now migrated. Move the helper to `config.py` as public `advanced_config_value` for any future caller and delete `_utils.py`. (`_normalize_detection_ids` in `_utils.py` has independent local copies in `detected_cache.py`/`detection_cache.py`; it has no external importers — it dies with the file.)

**Files:**
- Modify: `src/hydra_suite/core/inference/config.py`
- Delete: `src/hydra_suite/core/detectors/_utils.py`
- Test: `tests/test_inference_config_from_params.py` (extend)

**Interfaces:**
- Produces: `advanced_config_value(params: dict, key: str, default=None)`.

- [ ] **Step 1: Write the failing test**

```python
def test_advanced_config_value_reads_advanced_config():
    from hydra_suite.core.inference.config import advanced_config_value

    p = {"ADVANCED_CONFIG": {"reference_aspect_ratio": 2.5}}
    assert advanced_config_value(p, "reference_aspect_ratio", 2.0) == 2.5
    assert advanced_config_value(p, "missing", 7) == 7
    assert advanced_config_value({}, "x", 3) == 3
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_inference_config_from_params.py::test_advanced_config_value_reads_advanced_config -q`
Expected: FAIL — `ImportError`.

- [ ] **Step 3: Add the function to `config.py`**

```python
def advanced_config_value(params, key, default=None):
    """Read a power-user override from ADVANCED_CONFIG when present."""
    advanced_config = params.get("ADVANCED_CONFIG", {})
    if isinstance(advanced_config, dict) and key in advanced_config:
        return advanced_config.get(key)
    return default
```

- [ ] **Step 4: Confirm `_utils` has no remaining importers, then delete it**

Run:
```bash
grep -rn "detectors._utils\|detectors import _utils\|_advanced_config_value" src/ | grep -v "tests/"
```
Expected: no production matches. Then:
```bash
git rm src/hydra_suite/core/detectors/_utils.py
```

- [ ] **Step 5: Run test**

Run: `python -m pytest tests/test_inference_config_from_params.py -q`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
make format
git add -A
git commit -m "refactor(inference): relocate advanced_config_value; delete detectors/_utils"
```

---

### Task 2: Verify `PosePipeline` is dead and delete `pose_pipeline.py`

The extraction found `PosePipeline` referenced only by its own docstring; its `run()` is the sole `detector.filter_raw_detections` consumer left. Confirm no live caller, then delete (mirrors the earlier `detection_phase.py` orphan removal).

**Files:**
- Delete (if dead): `src/hydra_suite/core/tracking/pose/pose_pipeline.py`

- [ ] **Step 1: Prove it dead**

Run:
```bash
grep -rn "PosePipeline\|pose_pipeline" src/ tests/ tools/ | grep -v "pose_pipeline.py:"
```
Expected: no instantiation or import outside the module itself. If a live caller exists, STOP — migrate that caller to `InferenceRunner` pose (via `run_realtime`/`predict_pose_for_image`) instead of deleting, and record it.

- [ ] **Step 2: Delete**

```bash
git rm src/hydra_suite/core/tracking/pose/pose_pipeline.py
```
Also delete any dedicated test (`grep -rln PosePipeline tests/`).

- [ ] **Step 3: Verify**

Run: `python -m pytest tests/ -m "not benchmark" -q && python -c "import hydra_suite.core.tracking"`
Expected: PASS; no ImportError.

- [ ] **Step 4: Commit**

```bash
git add -A
git commit -m "chore(tracking): delete orphaned PosePipeline (last filter_raw_detections consumer)"
```

---

### Task 3: Migrate `test_detectors_engine.py` coverage to `core/inference`

`tests/test_detectors_engine.py` (1857 lines, ~45 tests) file-path-loads `_utils`, `_obb_geometry`, `_runtime_artifacts`, `_direct_obb_runtime`, `yolo_detector`, `detection_filter`. The behavior it exercises (artifact path stability/batch-specificity, raw-detection caps, filtering, sequential crop batching, direct executor, head-tail, canonical affines, IoU) now lives in `core/inference` (`runtime_artifacts.py`, `direct_executors.py`, `stages/obb.py`, `stages/filtering.py`, `stages/crops.py`, `stages/headtail.py`). Move the still-relevant assertions onto the inference modules; drop assertions that tested legacy-only internals.

**Files:**
- Create/extend: `tests/test_inference_obb_artifacts.py`, `tests/test_inference_direct_executors.py`, `tests/test_inference_stages_*.py` (as needed)
- Delete: `tests/test_detectors_engine.py`, `tests/test_yolo_detector_seq_obb_imgsz.py`

- [ ] **Step 1: Inventory what each test asserts and its inference home**

Run:
```bash
grep -n "def test_" tests/test_detectors_engine.py
```
For each test, decide: (a) already covered by an existing `tests/test_inference_*` test → drop; (b) still-unique behavior on a live inference module → port; (c) tested a legacy-only path that no longer exists → drop. Write the mapping into the commit message.

- [ ] **Step 2: Port the (b) tests**

For each still-unique test, write an equivalent against the inference module (import from `hydra_suite.core.inference...` directly, no file-path loading). Example — a raw-detection-cap test moves to `tests/test_inference_stages_obb.py` and calls `run_obb`/`materialize_tensors` with `raw_detection_cap` set.

- [ ] **Step 3: Delete the legacy test files**

```bash
git rm tests/test_detectors_engine.py tests/test_yolo_detector_seq_obb_imgsz.py
```
Note: `tests/test_benchmark_models.py` (which imported `_load_engine_module` from this file) was already deleted in Plan 3.

- [ ] **Step 4: Verify coverage didn't silently drop**

Run:
```bash
grep -rn "test_detectors_engine\|_load_engine_module" tests/
python -m pytest tests/ -m "not benchmark" -q
```
Expected: no references to the deleted helper; suite passes.

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "test: migrate detector-engine coverage onto core/inference modules"
```

---

### Task 4: Delete the remaining `core/detectors` files

Now `_obb_geometry.py`, `_runtime_artifacts.py`, `detection_filter.py`, and `yolo_detector.py` have no importers (production migrated in Plan 2; tests migrated in Task 3). The negative-assertion tests patch `hydra_suite.core.detectors.YOLOOBBDetector` with `raising=False`, so they tolerate its absence.

**Files:**
- Delete: `src/hydra_suite/core/detectors/{_obb_geometry,_runtime_artifacts,detection_filter,yolo_detector,__init__}.py`

- [ ] **Step 1: Final importer scan (must be empty)**

Run:
```bash
grep -rn "core.detectors\|from ..core.detectors\|from .detectors\|import detectors" src/ tests/ tools/ \
  | grep -v "core.inference" | grep -vi "#"
```
Expected: no production import lines. Comments/docstrings referencing the old names are fine. If anything live remains, STOP and migrate it (do not force the delete).

- [ ] **Step 2: Delete files one at a time, testing between**

```bash
git rm src/hydra_suite/core/detectors/detection_filter.py
python -c "import hydra_suite" && python -m pytest tests/ -m "not benchmark" -q
git rm src/hydra_suite/core/detectors/_obb_geometry.py
python -c "import hydra_suite" && python -m pytest tests/ -m "not benchmark" -q
git rm src/hydra_suite/core/detectors/_runtime_artifacts.py
python -c "import hydra_suite" && python -m pytest tests/ -m "not benchmark" -q
git rm src/hydra_suite/core/detectors/yolo_detector.py
git rm src/hydra_suite/core/detectors/__init__.py
python -c "import hydra_suite" && python -m pytest tests/ -m "not benchmark" -q
```
Expected: each stage imports cleanly and passes. (The negative-assertion tests in `test_trackerkit_preview_worker.py` / `test_model_test_dialog.py` use `raising=False`, so patching the now-absent name is a no-op.)

- [ ] **Step 3: Confirm the directory is gone**

Run:
```bash
ls src/hydra_suite/core/detectors/ 2>&1 || echo "removed"
```
Expected: `removed` (or only `__pycache__`, which `git` ignores).

- [ ] **Step 4: Commit**

```bash
git add -A
git commit -m "chore(core): delete retired core/detectors package"
```

---

### Task 5: Move the Qt workers out of `core/background/optimizer.py` (deferred follow-up)

`core/background/optimizer.py` defines `BgSubtractionOptimizer(QThread)` (L523) and `BgDetectionPreviewWorker(QThread)` (L846) with PySide6 `Signal`s — a `core`→GUI dependency-direction violation. Its only importer is `trackerkit/gui/dialogs/bg_parameter_helper.py`. Separate the pure optimization logic (stays in `core`) from the Qt wrapper (moves to `trackerkit`).

**Files:**
- Modify: `src/hydra_suite/core/background/optimizer.py` (keep pure Optuna/optimization functions)
- Create: `src/hydra_suite/trackerkit/gui/workers/bg_optimizer_workers.py` (the two `QThread` classes)
- Modify: `src/hydra_suite/trackerkit/gui/dialogs/bg_parameter_helper.py:46,48` (import from the new location)
- Test: `tests/test_bg_optimizer_core_no_qt.py`

**Interfaces:**
- Produces: pure functions in `core/background/optimizer.py` (no `QThread`/`Signal`); `BgSubtractionOptimizer` / `BgDetectionPreviewWorker` in the trackerkit workers module.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_bg_optimizer_core_no_qt.py
import ast
from pathlib import Path

import hydra_suite.core.background.optimizer as opt


def test_core_bg_optimizer_has_no_qt_imports():
    src = Path(opt.__file__).read_text()
    tree = ast.parse(src)
    bad = []
    for node in ast.walk(tree):
        mod = None
        if isinstance(node, ast.ImportFrom):
            mod = node.module or ""
        elif isinstance(node, ast.Import):
            mod = ",".join(a.name for a in node.names)
        if mod and ("PySide6" in mod or "QtCore" in mod):
            bad.append(mod)
    assert not bad, f"core/background/optimizer.py must not import Qt: {bad}"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_bg_optimizer_core_no_qt.py -q`
Expected: FAIL (Qt is imported for the two worker classes).

- [ ] **Step 3: Split**

Move `BgSubtractionOptimizer` and `BgDetectionPreviewWorker` (and their `Signal`s / `QThread` base) into a new `trackerkit/gui/workers/bg_optimizer_workers.py`, having them call the pure functions that remain in `core/background/optimizer.py`. Remove the `PySide6` import from the core module.

- [ ] **Step 4: Repoint the dialog**

In `bg_parameter_helper.py`, change the imports at L46/L48 from `hydra_suite.core.background.optimizer` to `hydra_suite.trackerkit.gui.workers.bg_optimizer_workers`.

- [ ] **Step 5: Verify**

Run:
```bash
python -m pytest tests/test_bg_optimizer_core_no_qt.py tests/ -m "not benchmark" -q -k "bg or optim"
python -c "import hydra_suite.trackerkit.gui.dialogs.bg_parameter_helper"
```
Expected: PASS; no ImportError.

- [ ] **Step 6: Commit**

```bash
make format
git add -A
git commit -m "refactor(core): move bg-optimizer Qt workers out of core into trackerkit"
```

---

### Task 6: Deprecate the orphaned `MIN_TRACKING_COUNTS` / `min_track_seconds` param (deferred follow-up)

The bg-sub work deleted this param's only reader (the `tracking_stabilized` latch). `MIN_TRACKING_COUNTS` is now emitted but never read (only a comment at `core/background/model.py:571`). Because existing user configs carry `min_track_seconds` and the GUI exposes a control, remove it via accept-and-ignore, not a bare delete.

**Files:**
- Modify: `trackerkit/cli_config.py` (L414-418, L695), `trackerkit/gui/orchestrators/config.py` (L718, L1686, L2035-2037, L2221), `resources/configs/default.json:81`, `resources/configs/ooceraea_biroi.json:106`, `tests/test_trackerkit_cli_config.py:98`
- Modify: `trackerkit/gui/panels/*` (the `spin_min_track` control)

- [ ] **Step 1: Confirm it is truly unread**

Run:
```bash
grep -rn "MIN_TRACKING_COUNTS" src/ | grep -v "cli_config.py\|orchestrators/config.py\|model.py:"
```
Expected: no consumer. (Only the two emitters + the model.py comment.)

- [ ] **Step 2: Stop emitting `MIN_TRACKING_COUNTS`**

Remove the emit at `cli_config.py:695` and `orchestrators/config.py:2221`, and the derivations feeding them (`cli_config.py:414-418`, `orchestrators/config.py:2035-2037`). Keep reading `min_track_seconds` from old configs without error (accept-and-ignore): leave the `.get("min_track_seconds", ...)` reads but stop threading the value into params.

- [ ] **Step 3: Remove the GUI control + config keys**

Remove the `spin_min_track` widget and its `orchestrators/config.py` save/load lines (L718, L1686). Delete `min_track_seconds` from `resources/configs/default.json:81` and `ooceraea_biroi.json:106`.

- [ ] **Step 4: Update the CLI-config test**

In `tests/test_trackerkit_cli_config.py`, remove the `assert params["MIN_TRACKING_COUNTS"] == 6` (L98) and add an assertion that a config carrying `min_track_seconds` loads without error and does not emit `MIN_TRACKING_COUNTS`.

- [ ] **Step 5: Verify**

Run: `python -m pytest tests/test_trackerkit_cli_config.py tests/ -m "not benchmark" -q -k "config"`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
make format
git add -A
git commit -m "chore(config): deprecate orphaned MIN_TRACKING_COUNTS/min_track_seconds"
```

---

### Task 7: Re-cover the legacy detector-init path + parity gate (deferred follow-up + Phase F)

The bg-sub work deleted `tests/test_tracking_worker_realtime_live_features.py` (it monkeypatched removed symbols). The behavior it exercised — backward-cached runs skipping runtime detector init; cache-write mode when reuse is disabled — now lives in `core/tracking/ingest/detection_phase.py`... which was itself deleted by the legacy-batching plan. Re-cover the equivalent behavior on the **current** `InferenceRunner`-based worker path, then run the mandatory parity gate.

**Files:**
- Create: `tests/test_worker_inference_init_modes.py`
- (Parity gate is a manual/scripted run, not a committed test.)

**Interfaces:**
- Consumes: `InferenceRunner(cfg, cache_only=...)`, `caches_all_valid()`, `load_frame()`.

- [ ] **Step 1: Write tests for the current init modes**

Assert against the current worker/runner: (a) backward/replay mode constructs `InferenceRunner(..., cache_only=True)` and never loads pose/CNN/AprilTag; (b) forward mode with cache-reuse disabled opens caches for writing. Use fakes for the runner where model loading would otherwise occur (mirror `tests/test_inference_runner_detect_batch.py`'s `__new__` fake pattern).

```python
# tests/test_worker_inference_init_modes.py (skeleton — fill against worker.py)
def test_backward_mode_uses_cache_only_runner(monkeypatch):
    ...  # assert InferenceRunner constructed with cache_only=True in backward_mode
```

- [ ] **Step 2: Run tests**

Run: `python -m pytest tests/test_worker_inference_init_modes.py -q`
Expected: PASS.

- [ ] **Step 3: Parity gate (MANDATORY before declaring the retirement done)**

On a machine with real models + a real test video, run the full pipeline once on `main` at the pre-retirement commit and once at HEAD, and diff trajectory outputs:

```bash
# baseline (pre-retirement tag) and HEAD each produce a CSV for the same video+config
python -c "import numpy as np, pandas as pd, sys; \
a=pd.read_csv(sys.argv[1]).to_numpy(); b=pd.read_csv(sys.argv[2]).to_numpy(); \
d=np.abs(a-b); print('max diff', d.max()); \
assert (d < 1e-4).all(), 'Output mismatch -- retirement changed behavior'" baseline.csv head.csv
```
Expected: `max diff` below `1e-4`; assertion passes. If it fails, trace the diff before considering the retirement complete.

- [ ] **Step 4: Commit**

```bash
git add -A
git commit -m "test: cover InferenceRunner init modes; record parity gate"
```

---

### Task 8: Retire the stale deletion-step brief + close the spec

**Files:**
- Delete/mark: `docs/superpowers/plans/2026-07-16-inference-redesign-deletion-step.md`
- Modify: `docs/superpowers/specs/2026-07-17-legacy-detector-retirement-design.md` (mark Done)

- [ ] **Step 1: Confirm the brief's live remnants are handled**

The brief's only still-live asks were the `api.py` pose helper (done in Plan 1 Task 4) and pose-backend deletions (Plan 2 Task 2). Confirm:
```bash
grep -rn "core/identity/pose/api\|create_pose_backend_from_config" src/ | grep -v "test"
```
Verify no remaining dependence blocks the brief's closure; note any leftover for a follow-up.

- [ ] **Step 2: Retire the brief**

```bash
git mv docs/superpowers/plans/2026-07-16-inference-redesign-deletion-step.md \
       docs/superpowers/plans/done/2026-07-16-inference-redesign-deletion-step.md
```
Add a one-line note at its top: "Superseded by the 2026-07-17 legacy-detector-retirement plans (1–4)."

- [ ] **Step 3: Mark the design spec Done**

In `docs/superpowers/specs/2026-07-17-legacy-detector-retirement-design.md`, change `**Status:**` to `Done` and note the four executing plans.

- [ ] **Step 4: Commit**

```bash
git add -A
git commit -m "docs: retire stale deletion-step brief; close detector-retirement spec"
```

---

## Final verification (whole retirement, all 4 plans)

- [ ] **Step 1: `core/detectors` is gone**

```bash
ls src/hydra_suite/core/detectors/ 2>&1 || echo removed
grep -rn "core.detectors\|core/detectors" src/ tests/ tools/ | grep -vi "#\|docstring"
```
Expected: `removed`; no production references.

- [ ] **Step 2: Full suite + import health** — `python -m pytest tests/ -m "not benchmark" -q` and `python -c "import hydra_suite; import hydra_suite.trackerkit; import hydra_suite.posekit; import hydra_suite.detectkit"`.

- [ ] **Step 3: The parity gate (Task 7 Step 3) passed** — confirm the `< 1e-4` assertion was run and green.

- [ ] **Step 4: Format + lint** — `make format-check && make lint-moderate`.

---

## Self-Review notes

- **Spec coverage (Phases E + F + deferred):** Task 1 = `_utils` relocation; Task 2 = dead `PosePipeline`; Task 3 = test migration; Task 4 = delete the package; Task 5 = Qt-in-core split (deferred follow-up 1); Task 6 = `MIN_TRACKING_COUNTS` deprecation (deferred follow-up 2); Task 7 = re-cover init path + parity gate (deferred follow-up 3 + Phase F); Task 8 = retire the stale brief + close the spec.
- **Ordering within the plan:** Tasks 1–3 remove the last importers/tests; Task 4 deletes files one at a time with tests between; Tasks 5–6 are independent cleanups; Task 7's parity gate is the hard stop; Task 8 is bookkeeping. The whole plan depends on Plans 1–3.
- **Prove-dead-then-delete:** every deletion (Tasks 2, 4) is preceded by an importer scan that must come back empty, matching the repo's careful-deletion posture; the negative-assertion tests survive deletion via `raising=False`.
- **Type consistency:** `advanced_config_value(params, key, default=None)`; deletions reference the exact files listed in the File Structure. The parity gate mirrors the original inference-redesign Task 18 threshold (`< 1e-4`).
