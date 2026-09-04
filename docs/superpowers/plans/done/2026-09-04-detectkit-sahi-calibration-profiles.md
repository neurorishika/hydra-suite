# DetectKit SAHI Calibration Profiles Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let a DetectKit user measure real TrackerKit sliced-inference operating points against their own labelled frames and save several explicitly-chosen, named SAHI profiles alongside one model artifact, which TrackerKit then applies on model selection.

**Architecture:** Calibration reuses the *production* direct-inference path. `run_obb` is split so its pre-merge, frame-space per-region parts can be collected once per fixed tile geometry; confidence and merge alternatives are then replayed offline through the same `merge_per_frame` → `filter_for_source` seam, so no second model call and no DetectKit-local approximation. Every calibration config is built from the shared `SLICE_*` params mapping the GUI and CLI already use, so a measured operating point is expressible as TrackerKit settings by construction. Pure logic lives in `core/inference/`; DetectKit owns the worker, evidence store and UI; TrackerKit only consumes the sidecar.

**Tech Stack:** Python 3.11, NumPy, OpenCV, PySide6 (offscreen in tests), pytest, Ultralytics/ONNX/TensorRT executors behind `core/inference`.

**Spec:** `docs/superpowers/specs/2026-09-01-detectkit-sahi-calibration-profiles-design.md`

**Revision note:** this is revision 2. Revision 1 was rejected by an adversarial review that verified its code against the codebase; every API signature below has now been read from source and is cited with `file:line`. Do not reintroduce `OBBConfig(model_path=...)`, `RuntimeContext.for_tier`, `build_obb_config_for_model`, `BaseWorker.should_stop`, or `tools/equivalence/fixtures/models/` — none of them exist.

---

## Global Constraints

- **Worktree isolation (CLAUDE.md):** work in a worktree branched from local HEAD: `git worktree add .worktrees/sahi-calib -b feat/sahi-calibration-profiles HEAD`. Run tests with `PYTHONPATH=$PWD/src` from inside the worktree.
- **Dependency direction:** `core/`, `runtime/`, `data/`, `training/`, `utils/` must never import from `detectkit/`, `trackerkit/`, `widgets/`, or `integrations/`.
- **No DetectKit-local inference approximation.** Do not build on `detectkit/gui/prediction_preview.predict_sliced_obb_result`.
- **Every calibration config is built from params**, via `build_obb_only_config(..., extra_params={"SLICE_*": ...})` (`core/inference/config.py:1151`), never by hand-constructing `SliceConfig`. This is what makes "the measured point is a TrackerKit setting" true rather than asserted.
- **Production-inference edits require the equivalence gate.** Task 1 touches `core/inference/stages/obb.py`; prove byte-identical on MPS (`hydra-mps`, this box) and CUDA (mehek, `hydra-cuda`).
- **Never write `REFERENCE_BODY_SIZE`.** A profile's `trained_body_px` maps only to `SLICE_TRAINED_BODY_PX` (`config.py:923`).
- **Nothing is saved without an explicit user action**, and nothing is written to disk before the results dialog is accepted.
- **Sidecar writes are atomic** — always via `core.inference.slice_meta.write_slice_meta`.
- **One artifact = one model.** No weight copies, no extra registry entries per profile.
- **Ship labelled `Experimental calibration`** in user-facing copy.
- **`max_detections` is part of the measurement.** It truncates results (`filtering.py:30-34`) and bounds the merge input via `effective_raw_detection_cap = 2 * max_detections` (`obb.py:39-46`). It must be set from the user's expected animal count, recorded in the profile `settings`, and shown in the UI.
- **Task order is OBB-first.** Detect/segment land in Task 17.
- Format before each commit: `make format`; lint gate: `make lint`.

---

## Verified API facts (read from source — do not re-derive, do not guess past these)

| Thing | Truth | Cite |
|---|---|---|
| Detection config root | `InferenceConfig(obb=OBBConfig(...), ...)`; `detection_source` is a property | `config.py:446-517` |
| `OBBConfig` | `mode`, `direct: OBBDirectConfig`, `sequential`, `target_classes`, `max_detections=20`, `raw_detection_cap=0`, `min_object_size`, `confidence_threshold=0.25`, `iou_threshold=0.7` | `config.py:201-222` |
| `OBBDirectConfig` | `model_path`, `confidence_floor=1e-3`, `confidence_threshold=0.25`, `auto_export`, `model_task`, `slice: SliceConfig` | `config.py:127-170` |
| `SliceConfig` | `enabled`, `geometry_mode`, `slice_height`, `slice_width`, `overlap_height_ratio`, `overlap_width_ratio`, `object_tile_fraction`, `reference_body_px`, `merge_policy ∈ {nms,nmm,greedy_nmm}`, `merge_metric ∈ {iou,ios}`, `merge_threshold`, `merge_backend ∈ {cv2,gpu}`, `perform_standard_pred`, `tile_batch_size`, `tile_memory_budget_bytes`; validates in `__post_init__` | `config.py:62-124` |
| Building a config | `build_obb_only_config(model_path, *, compute_runtime="cpu", runtime_tier=None, confidence_threshold=0.25, iou_threshold=0.7, max_targets=8, mode="direct", model_task="obb", emit_native_geometry=False, extra_params=None) -> InferenceConfig` | `config.py:1151-1197` |
| Slice params | `_slice_config_from_params(params, "SLICE_", reference_body_px=...)` reads `SLICE_ENABLED`, `SLICE_GEOMETRY_MODE`, `SLICE_HEIGHT`, `SLICE_WIDTH`, `SLICE_OVERLAP`, `SLICE_OBJECT_TILE_FRACTION`, `SLICE_MERGE_POLICY/METRIC/THRESHOLD/BACKEND`, `SLICE_PERFORM_STANDARD_PRED`, `SLICE_TILE_BATCH_SIZE`, `SLICE_MEMORY_BUDGET_MIB`; body px from `SLICE_TRAINED_BODY_PX` | `config.py:628-680, 923-939` |
| Runtime | `RuntimeContext.from_config(InferenceConfig)` — there is **no** `for_tier` | `runtime.py:126` |
| Model loading | `load_obb_models(config: OBBConfig, runtime, *, batch_size=1) -> OBBModels` — takes `config.obb`, not the `InferenceConfig` | `obb.py:396-398` |
| Merge | `merge_per_frame(parts, merge_policy, plan, config: OBBConfig, runtime)`; `merge_policy` here is the **RegionSource** policy (`"plain"` / `"overlap_band_nms"`, `obb.py:1473`), *not* `SliceConfig.merge_policy` | `obb.py:1476-1523` |
| Filtering | `filter_for_source(config: InferenceConfig, raw: OBBResult, roi_mask) -> (result, indices)`; delegates to `filter_with_indices(raw, config.obb, roi_mask)` which gates on `config.obb.confidence_threshold` (`filtering.py:305`) and truncates to `_effective_max_detections(config.obb)` | `filtering.py:289-366, 30-34` |
| CUDA parts | on `runtime.tensor_on_cuda`, parts/merge output are `_RawOBBTensors`; production materializes with `materialize_tensors(raw, cap)` before filtering | `obb.py:1713`, `runner.py:1393` |
| Empty result | `stages.obb._empty_obb_result(frame_idx)` (also re-exported into `merge.py`) | `obb.py:1440` |
| Tile plan | `plan_slices(frame_hw, slice_cfg, imgsz, roi_mask, ref_object_px=0.0) -> SlicePlan`; `SlicePlan.tiles: list[(x0,y0,x1,y1)]`, `.full_frame`, `.slice_wh`, `.frame_wh`, `.jobs_per_frame` | `slicing.py:232`, `utils/slice_geometry.py:24-34` |
| Model imgsz | `stages.obb._resolve_imgsz(model)` | `obb.py:49` |
| `BaseWorker` | signals `progress(int)`, `status(str)`, `error(str)`; `run()` wraps `execute()`; **no stop flag, no `should_stop`** — a worker that must cancel owns its own flag | `widgets/workers.py:37-57` |
| Labels | `detectkit.gui.utils.parse_obb_label(label_path, img_w, img_h, class_id_map=None) -> [{"class_id": int, "polygon_px": [[x,y],...]}]` | `gui/utils.py:325-373` |
| `LabelRecord` / `GeometryLevel` | `hydra_suite.data.al.escalation:20` / `hydra_suite.utils.geometry_levels:20` | — |
| Equivalence fixtures | `tools/equivalence/fixtures/` has `clips/`, `configs/`, `staging/` and `manifest.json`; **no `models/` dir** — models extract to `get_models_dir()`; the fly OBB checkpoint is named in `fixtures/configs/fly_obb.json` | — |
| Publish sidecar | `model_publish.py:851-868` writes a **fresh** `normalized_slice_meta(slice_geometry)` at the destination and never copies the source `.slice_meta.json` | `model_publish.py:851-868` |
| DetectKit model UI | there is **no** registered-model list page; `detectkit/gui/models.py` is a data module (`OBBSource`, `DetectKitProject`) | `detectkit/gui/models.py:20-322` |

### Design facts established during planning

- `InferenceRunner.detect_batch_raw` (`runner.py:1367`) returns **post-merge** results. The pre-merge parts live inside `run_obb` (`obb.py:486-575`) as `parts_by_frame`. That local is the seam Task 1 extracts.
- Production order is **merge → filter**. Predict-time confidence is `direct.confidence_floor` (fixed `1e-3`, `config.py:130`), independent of the filter threshold, so parts collected once are valid for the whole confidence sweep. `_bound_compact_parts` is a top-k-by-confidence reservoir applied identically in both paths, so it does not interact with the sweep — **provided `max_detections` is identical**, hence the global constraint above.
- The detection cache key (`cache/keys.py:107-140`) folds in geometry, overlap, `object_tile_fraction`, `reference_body_px`, merge policy/metric/threshold/backend and `perform_standard_pred`, and **deliberately excludes confidence**. Task 16 pins that split.
- `merge_backend` is forced to `cv2` on all host (non-native-CUDA) paths (`obb.py:1571-1581`), so sweeping it on MPS produces duplicate rows. Do not sweep it; record the resolved value.
- `core/inference/semantic/calibration.py` is the structural template for the sweep. `detectkit/jobs/semantic_escalation.py:1143` (`labelled_frames_for`) already parses labels correctly — reuse it, do not re-write parsing.
- `detectkit/gui/dialogs/calibration_results_dialog.py` is SAM3-specific. Reuse `dialogs/_overlay_helpers.py` and `gui/canvas.OBBCanvas`, not that dialog.

---

## Shipped foundation (verify, do not rebuild)

Commits `5f44d280`, `209e7a89`, `c1830626` landed part of the spec already.

| Spec area | Shipped in | Tests | Status |
|---|---|---|---|
| v2 sidecar schema, atomic read/write, legacy promotion | `core/inference/slice_meta.py` | `tests/test_slice_meta_read.py` | Done |
| Profile CRUD helpers | `core/inference/slice_meta.py` | thin | Done, **2 spec deviations** → Task 6 |
| `slice_meta_to_panel_values` | `core/inference/slice_meta.py` | `tests/test_slice_meta_read.py`, `tests/test_trackerkit_slice_meta_prefill.py` | Done |
| TrackerKit profile combo + Custom state + apply-on-select + session id | `trackerkit/gui/panels/detection_panel.py:767-2680`, `orchestrators/config.py:441,1660` | `tests/test_trackerkit_slice_meta_prefill.py:82-161` covers apply / `__training__` / `__custom__` | Done. Gaps: **no staleness check, no visible fallback, no model-switch clear** → Tasks 14–15 |
| Slice widgets exist/visibility | `detection_panel.py` | `tests/test_detection_panel_slice_widgets.py` | Done (widgets only — it does **not** test Custom state) |
| One-to-one class-aware scoring with duplicates | `core/inference/direct_calibration.py` | `tests/test_direct_calibration.py` | Done (polygon/OBB) |
| Initial v2 sidecar at publish | `training/model_publish.py:851-868` | `tests/test_model_publish_slice_geometry.py` | Done, but **destroys pre-existing profiles** → Task 13 |
| Candidate grid, sweep, cache, recommendation | — | — | **Gap** → Tasks 2–5 |
| DetectKit job, wizard, results/profile UI, entry points | — | — | **Gap** → Tasks 7–12 |
| Detect/segment evaluation, docs | — | — | **Gap** → Task 17 |

Note: `tests/test_model_publish_slice_geometry.py:227-230` asserts the *old* auto-promotion behaviour and must be updated in Task 6.

---

## File structure

**Create**
- `src/hydra_suite/core/inference/direct_calibration_grid.py` — candidate settings, grid, work estimate, fingerprints.
- `src/hydra_suite/core/inference/direct_calibration_sweep.py` — config construction from params + offline re-scoring.
- `src/hydra_suite/detectkit/jobs/direct_calibration.py` — evidence selection, sweep driver, `BaseWorker`, evidence persistence.
- `src/hydra_suite/detectkit/gui/dialogs/direct_calibration_wizard.py` — wizard + shared launcher.
- `src/hydra_suite/detectkit/gui/dialogs/direct_calibration_results.py` — frontier, overlays, profile management.
- Tests: `tests/test_direct_calibration_grid.py`, `tests/test_direct_calibration_sweep.py`, `tests/test_direct_calibration_parity.py`, `tests/test_slice_profile_mutations.py`, `tests/test_detectkit_direct_calibration_job.py`, `tests/test_detectkit_direct_calibration_ui.py`, `tests/test_trackerkit_profile_session.py`, `tests/test_profile_cache_keys.py`.

**Modify**
- `core/inference/stages/obb.py:486-575` (Task 1), `core/inference/direct_calibration.py` (Tasks 5, 17), `core/inference/slice_meta.py` (Tasks 6, 13, 14).
- `detectkit/gui/calibration_preview_store.py` (Task 9), `detectkit/gui/dialogs/training_dialog.py`, `dialogs/history_dialog.py`, `detectkit/gui/main_window.py` (Task 12).
- `training/model_publish.py:851-868` (Task 13).
- `trackerkit/gui/panels/detection_panel.py`, `trackerkit/gui/orchestrators/config.py` (Tasks 14–15).
- `docs/` (Task 17).

---

## Task 0: Shared test fixture for a real direct-OBB config

**Files:**
- Modify: `tests/conftest.py`
- Test: consumed by Tasks 1 and 3.

**Interfaces:**
- Produces: fixture `direct_obb_fixture` → `(frames: list[np.ndarray], models: OBBModels, inference_config: InferenceConfig, runtime: RuntimeContext)`, where `inference_config.obb.direct.slice.enabled is True`. Skips when no checkpoint is available.
- Produces: helper `build_calibration_config(model_path, *, slice_params: dict, max_targets: int, confidence: float, runtime_tier: str) -> InferenceConfig` is **not** defined here — it lands in Task 3 and the fixture imports it.

- [ ] **Step 1: Locate a real OBB checkpoint on this machine**

```bash
python - <<'PY'
import json, pathlib
from hydra_suite.paths import get_models_dir
cfg = json.loads(pathlib.Path("tools/equivalence/fixtures/configs/fly_obb.json").read_text())
print(json.dumps({k: v for k, v in cfg.items() if "MODEL" in k or "PATH" in k}, indent=2))
print("models dir:", get_models_dir())
print(sorted(p.name for p in pathlib.Path(get_models_dir()).rglob("*.pt"))[:20])
PY
```
Record the resolved path; the fixture must derive it the same way (config JSON + `get_models_dir()`), never a hardcoded `fixtures/models/...` path, which does not exist.

- [ ] **Step 2: Write the fixture**

```python
# tests/conftest.py  (append)
import json
from pathlib import Path

import numpy as np
import pytest

_FLY_OBB_CONFIG = Path("tools/equivalence/fixtures/configs/fly_obb.json")


def _fixture_obb_checkpoint() -> Path | None:
    """Resolve the equivalence fixture's OBB checkpoint, or None when absent.

    The fixture bundle extracts models into ``get_models_dir()``; the clip
    config names the file. There is no ``fixtures/models/`` directory.
    """
    if not _FLY_OBB_CONFIG.exists():
        return None
    from hydra_suite.paths import get_models_dir

    params = json.loads(_FLY_OBB_CONFIG.read_text())
    raw = str(params.get("YOLO_OBB_DIRECT_MODEL_PATH", ""))
    if not raw:
        return None
    candidate = Path(raw)
    if candidate.is_file():
        return candidate
    resolved = Path(get_models_dir()) / raw
    return resolved if resolved.is_file() else None


@pytest.fixture
def direct_obb_fixture():
    """Real sliced direct-OBB config + loaded models, or skip."""
    checkpoint = _fixture_obb_checkpoint()
    if checkpoint is None:
        pytest.skip("equivalence fixture OBB checkpoint not present")
    from hydra_suite.core.inference.direct_calibration_sweep import (
        build_calibration_config,
    )
    from hydra_suite.core.inference.runtime import RuntimeContext
    from hydra_suite.core.inference.stages.obb import load_obb_models

    config = build_calibration_config(
        str(checkpoint),
        slice_params={
            "SLICE_ENABLED": True,
            "SLICE_GEOMETRY_MODE": "auto_object",
            "SLICE_OBJECT_TILE_FRACTION": 0.4,
            "SLICE_OVERLAP": 0.2,
            "SLICE_TRAINED_BODY_PX": 120.0,
        },
        max_targets=64,
        confidence=0.25,
        runtime_tier="cpu",
    )
    runtime = RuntimeContext.from_config(config)
    models = load_obb_models(config.obb, runtime)
    rng = np.random.default_rng(0)
    frames = [rng.integers(0, 255, (480, 640, 3), dtype=np.uint8) for _ in range(2)]
    return frames, models, config, runtime
```

- [ ] **Step 3: Confirm it skips cleanly for now**

Run: `PYTHONPATH=$PWD/src python -m pytest tests/conftest.py -q` (collection only) then proceed — the fixture is exercised in Task 3, after `build_calibration_config` exists. If fixtures are missing locally, fetch them: `bash tools/equivalence/fixtures/fetch_fixtures.sh`.

- [ ] **Step 4: Commit**

```bash
make format
git add tests/conftest.py
git commit -m "test: shared real direct-OBB fixture for SAHI calibration"
```

---

## Task 1: Extract the pre-merge collection seam from `run_obb`

**Files:**
- Modify: `src/hydra_suite/core/inference/stages/obb.py:486-575`
- Test: `tests/test_direct_calibration_parity.py`

**Interfaces:**
- Produces: `collect_obb_parts_by_frame(frames, models, config: OBBConfig, runtime, roi_mask=None, should_stop=None) -> tuple[list[list], RegionSource]` — per-frame pre-merge parts in frame coordinates, plus the `RegionSource` whose `merge_policy` / `merge_plan(fi)` must collapse them.
- Produces: `run_obb` keeps its exact signature and return type.

- [ ] **Step 1: Write the failing parity test**

```python
# tests/test_direct_calibration_parity.py
"""collect + merge must be run_obb, exactly. This is the load-bearing claim."""

import numpy as np

from hydra_suite.core.inference.stages import obb as obb_stage


def test_collect_then_merge_equals_run_obb(direct_obb_fixture):
    frames, models, config, runtime = direct_obb_fixture
    expected = obb_stage.run_obb(frames, models.__class__ and models, config.obb, runtime)
    parts_by_frame, source = obb_stage.collect_obb_parts_by_frame(
        frames, models, config.obb, runtime
    )
    actual = []
    for index, parts in enumerate(parts_by_frame):
        if not parts:
            actual.append(obb_stage._empty_obb_result(index))
            continue
        actual.append(
            obb_stage.merge_per_frame(
                parts, source.merge_policy, source.merge_plan(index), config.obb, runtime
            )
        )
    assert len(expected) == len(actual)
    for want, got in zip(expected, actual):
        if isinstance(want, obb_stage._RawOBBTensors):
            want = obb_stage.materialize_tensors(want, config.obb.raw_detection_cap)
            got = obb_stage.materialize_tensors(got, config.obb.raw_detection_cap)
        assert want.num_detections == got.num_detections
        np.testing.assert_array_equal(want.centroids, got.centroids)
        np.testing.assert_array_equal(want.angles, got.angles)
        np.testing.assert_array_equal(want.confidences, got.confidences)


def test_parts_are_in_frame_coordinates(direct_obb_fixture):
    """Tile-local coordinates would silently mis-score every candidate."""
    frames, models, config, runtime = direct_obb_fixture
    height, width = frames[0].shape[:2]
    parts_by_frame, _source = obb_stage.collect_obb_parts_by_frame(
        frames, models, config.obb, runtime
    )
    for parts in parts_by_frame:
        for part in parts:
            if isinstance(part, obb_stage._RawOBBTensors):
                part = obb_stage.materialize_tensors(part, 0)
            if part.num_detections:
                assert part.centroids[:, 0].max() <= width + 1
                assert part.centroids[:, 1].max() <= height + 1
```

Simplify the first line of the first test to `obb_stage.run_obb(frames, models, config.obb, runtime)` — the `models.__class__ and` is a typo guard; write it plainly.

- [ ] **Step 2: Run it and confirm it fails**

Run: `PYTHONPATH=$PWD/src python -m pytest tests/test_direct_calibration_parity.py -v`
Expected: FAIL — `AttributeError: ... has no attribute 'collect_obb_parts_by_frame'`. A SKIP means the fixture checkpoint is missing — fetch fixtures first; do not proceed on a skip.

- [ ] **Step 3: Extract the function — pure move, no behaviour change**

Cut `run_obb`'s body from `from .regions import select_region_source` through `del chunk, fi, region, result, part` into the new function; `run_obb` keeps only the merge loop.

```python
def collect_obb_parts_by_frame(
    frames: list,
    models: OBBModels,
    config: OBBConfig,
    runtime: RuntimeContext,
    roi_mask: np.ndarray | None = None,
    should_stop: Any = None,
) -> tuple[list[list], Any]:
    """Run every region/tile and return PRE-MERGE, frame-space parts per frame.

    The first half of ``run_obb``, verbatim, and it must stay that way: SAHI
    calibration measures here, then replays merge + filter offline. The
    ``del`` of the loop locals and the ``_bound_compact_parts`` reservoir are
    part of the memory contract, not incidental -- keep both.
    """
    from .regions import select_region_source

    source = select_region_source(config)
    task = source.task(config)
    seg_source = source.seg_source(config)
    parts_by_frame: list[list] = [[] for _ in frames]
    with span(N.MODEL_EXECUTE, gpu=True):
        iterator_kwargs = {"roi_mask": roi_mask}
        if should_stop is not None:
            iterator_kwargs["should_stop"] = should_stop
        chunks = source.iter_region_results(
            frames, models, config, runtime, **iterator_kwargs
        )
        for chunk in chunks:
            with span(N.EXTRACT_RAW):
                for fi, region, result in chunk:
                    part = extract_with_transform(
                        result,
                        fi,
                        task,
                        region.affine,
                        config,
                        runtime,
                        seg_source=seg_source,
                        force_numpy=source.force_numpy,
                    )
                    if not isinstance(part, (OBBResult, _RawOBBTensors)) or (
                        int(part.xywhr.shape[0])
                        if isinstance(part, _RawOBBTensors)
                        else part.num_detections
                    ):
                        parts_by_frame[fi].append(part)
                    parts_by_frame[fi] = _bound_compact_parts(
                        parts_by_frame[fi], fi, effective_raw_detection_cap(config)
                    )
            chunk.clear()
            del chunk, fi, region, result, part
    return parts_by_frame, source
```

`run_obb`'s remaining body:

```python
    parts_by_frame, source = collect_obb_parts_by_frame(
        frames, models, config, runtime, roi_mask=roi_mask, should_stop=should_stop
    )
    out: list[OBBResult | _RawOBBTensors] = []
    with span(N.EXTRACT_RAW):
        for fi, parts in enumerate(parts_by_frame):
            if not parts:
                out.append(_empty_obb_result(fi))
            else:
                out.append(
                    merge_per_frame(
                        parts, source.merge_policy, source.merge_plan(fi), config, runtime
                    )
                )
    return out
```

Note the memory consequence and accept it for calibration only: `run_obb` already retained all frames' parts before this change, so nothing regresses; but a calibration caller must pass **small frame batches** (Task 8 passes one frame at a time) because it holds parts across the whole confidence sweep.

- [ ] **Step 4: Run the parity test**

Run: `PYTHONPATH=$PWD/src python -m pytest tests/test_direct_calibration_parity.py -v`
Expected: PASS (2 tests).

- [ ] **Step 5: Record the pre-edit baseline and re-run neighbours**

```bash
git worktree add --detach .worktrees/pre-seam HEAD
PYTHONPATH=$PWD/.worktrees/pre-seam/src python -m pytest tests/test_slice_geometry_parity.py \
  tests/test_detectkit_sliced_preview.py tests/test_core_qtfree_slice2.py \
  tests/test_core_qtfree_slice3.py -q | tail -3
PYTHONPATH=$PWD/src python -m pytest tests/test_slice_geometry_parity.py \
  tests/test_detectkit_sliced_preview.py tests/test_core_qtfree_slice2.py \
  tests/test_core_qtfree_slice3.py -q | tail -3
```
Expected: identical failure sets.

- [ ] **Step 6: Format and commit**

```bash
make format
git add src/hydra_suite/core/inference/stages/obb.py tests/test_direct_calibration_parity.py
git commit -m "refactor(inference): extract pre-merge part collection from run_obb"
```

- [ ] **Step 7: Kill stale sleap/hydra processes, then run the MPS gate**

```bash
pgrep -fl "sleap|hydra" | grep -v grep    # kill ONLY stale sleap/hydra pids
conda activate hydra-mps
find . -name '__pycache__' -prune -exec rm -rf {} +   # stale numba JIT poisons equivalence
git worktree add --detach .worktrees/equiv-base HEAD~1
REPO=$PWD WT=$PWD MAIN_SRC=$PWD/.worktrees/equiv-base/src WT_SRC=$PWD/src \
  OUT=/tmp/equiv_sahi_seam RUNTIME=mps \
  bash tools/equivalence/run_matrix.sh fly_obb ant_obb_sleap worm_bgsub
wc -l /tmp/equiv_sahi_seam/**/*.csv    # every CSV MUST have > 1 row
```
Expected: EQUIVALENCE at the DETERMINISM floor for every clip.

---

## Task 2: Candidate settings and bounded grid

**Files:**
- Create: `src/hydra_suite/core/inference/direct_calibration_grid.py`
- Test: `tests/test_direct_calibration_grid.py`

**Interfaces:**
- Produces:
  - `@dataclass(frozen=True) CandidateGeometry(enabled: bool, geometry_mode: str, slice_width: int, slice_height: int, overlap: float, object_tile_fraction: float, trained_body_px: float, label: str)` with `slice_params() -> dict` returning the `SLICE_*` params dict (never a `SliceConfig`).
  - `build_candidate_grid(training_geometry: dict, *, custom: tuple[int, int] | None = None, fraction_steps=FRACTION_STEPS, overlaps=OVERLAP_STEPS) -> list[CandidateGeometry]`
  - `@dataclass(frozen=True) GridWorkEstimate(candidate, tiles_per_frame: int, total_tiles: int, failed_reason: str = "")`
  - `estimate_grid_work(candidates, *, frame_hw, imgsz, frames, max_total_tiles=DEFAULT_MAX_TOTAL_TILES) -> list[GridWorkEstimate]`
  - `FRACTION_STEPS = (0.75, 1.0, 1.5)`, `OVERLAP_STEPS = (0.1, 0.2, 0.3)`, `DEFAULT_MAX_TOTAL_TILES = 20000`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_direct_calibration_grid.py
import pytest

from hydra_suite.core.inference.direct_calibration_grid import (
    DEFAULT_MAX_TOTAL_TILES,
    build_candidate_grid,
    estimate_grid_work,
)

TRAINING = {
    "geometry_mode": "auto_object",
    "imgsz": 640,
    "object_tile_fraction": 0.4,
    "overlap": 0.2,
    "reference_body_px": 560.0,
}


def test_grid_always_contains_full_frame_and_training_geometry():
    grid = build_candidate_grid(TRAINING)
    assert any(not c.enabled for c in grid), "full-frame baseline missing"
    training = [
        c for c in grid
        if c.enabled
        and c.object_tile_fraction == pytest.approx(0.4)
        and c.overlap == pytest.approx(0.2)
    ]
    assert len(training) == 1 and training[0].label == "Training geometry"


def test_grid_is_deduplicated_and_every_row_is_labelled():
    grid = build_candidate_grid(TRAINING)
    keys = [
        (c.enabled, c.geometry_mode, c.slice_width, c.slice_height,
         round(c.overlap, 4), round(c.object_tile_fraction, 4))
        for c in grid
    ]
    assert len(keys) == len(set(keys))
    assert all(c.label for c in grid)


def test_slice_params_use_the_real_param_keys():
    """These keys are read verbatim by config._slice_config_from_params."""
    candidate = build_candidate_grid(TRAINING)[1]
    params = candidate.slice_params()
    assert params["SLICE_ENABLED"] is True
    assert params["SLICE_GEOMETRY_MODE"] == "auto_object"
    assert params["SLICE_OVERLAP"] == pytest.approx(candidate.overlap)
    assert params["SLICE_OBJECT_TILE_FRACTION"] == pytest.approx(
        candidate.object_tile_fraction
    )
    assert params["SLICE_TRAINED_BODY_PX"] == pytest.approx(candidate.trained_body_px)
    assert set(params) <= {
        "SLICE_ENABLED", "SLICE_GEOMETRY_MODE", "SLICE_WIDTH", "SLICE_HEIGHT",
        "SLICE_OVERLAP", "SLICE_OBJECT_TILE_FRACTION", "SLICE_TRAINED_BODY_PX",
    }


def test_full_frame_candidate_disables_slicing():
    full = next(c for c in build_candidate_grid(TRAINING) if not c.enabled)
    assert full.slice_params()["SLICE_ENABLED"] is False


def test_custom_geometry_is_appended_when_requested():
    grid = build_candidate_grid(TRAINING, custom=(1024, 768))
    custom = [c for c in grid if c.geometry_mode == "custom"]
    assert custom and custom[0].slice_width == 1024 and custom[0].slice_height == 768
    assert custom[0].slice_params()["SLICE_WIDTH"] == 1024


def test_work_estimate_reports_tiles_and_flags_over_budget():
    grid = build_candidate_grid(TRAINING)
    estimates = estimate_grid_work(grid, frame_hw=(2160, 3840), imgsz=640, frames=80)
    assert len(estimates) == len(grid)
    full = next(e for e in estimates if not e.candidate.enabled)
    assert full.tiles_per_frame == 1
    assert all(e.total_tiles == e.tiles_per_frame * 80 for e in estimates)
    huge = estimate_grid_work(grid, frame_hw=(2160, 3840), imgsz=640, frames=10**6)
    assert any(e.failed_reason for e in huge)


def test_unplannable_candidate_is_flagged_not_dropped():
    grid = build_candidate_grid(TRAINING, custom=(1, 1))
    estimates = estimate_grid_work(grid, frame_hw=(2160, 3840), imgsz=640, frames=10)
    assert len(estimates) == len(grid), "a failed candidate must still get a row"
    custom = next(e for e in estimates if e.candidate.geometry_mode == "custom")
    assert custom.failed_reason
```

- [ ] **Step 2: Run and confirm failure**

Run: `PYTHONPATH=$PWD/src python -m pytest tests/test_direct_calibration_grid.py -v`
Expected: FAIL — `ModuleNotFoundError`.

- [ ] **Step 3: Implement**

```python
"""Bounded, transparent SAHI candidate grid for detector calibration.

Deliberately a stated grid, not an optimizer: the UI prints the exact candidate
list and its tile cost before any model runs. Candidates carry ``SLICE_*``
PARAMS, never a hand-built ``SliceConfig`` -- routing through the shared params
mapping is what makes a measured point expressible as TrackerKit settings.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

FRACTION_STEPS: tuple[float, ...] = (0.75, 1.0, 1.5)
OVERLAP_STEPS: tuple[float, ...] = (0.1, 0.2, 0.3)
DEFAULT_MAX_TOTAL_TILES = 20000


@dataclass(frozen=True)
class CandidateGeometry:
    """One fixed tile plan to measure. ``enabled=False`` is the full-frame baseline."""

    enabled: bool
    geometry_mode: str
    slice_width: int
    slice_height: int
    overlap: float
    object_tile_fraction: float
    trained_body_px: float
    label: str

    def slice_params(self) -> dict:
        """The ``SLICE_*`` params consumed by ``config._slice_config_from_params``."""
        params: dict = {
            "SLICE_ENABLED": bool(self.enabled),
            "SLICE_GEOMETRY_MODE": self.geometry_mode,
            "SLICE_OVERLAP": float(self.overlap),
            "SLICE_OBJECT_TILE_FRACTION": float(self.object_tile_fraction),
            "SLICE_TRAINED_BODY_PX": float(self.trained_body_px),
        }
        if self.geometry_mode == "custom":
            params["SLICE_WIDTH"] = int(self.slice_width)
            params["SLICE_HEIGHT"] = int(self.slice_height)
        return params


@dataclass(frozen=True)
class GridWorkEstimate:
    candidate: CandidateGeometry
    tiles_per_frame: int
    total_tiles: int
    failed_reason: str = ""


def build_candidate_grid(
    training_geometry: dict,
    *,
    custom: tuple[int, int] | None = None,
    fraction_steps: tuple[float, ...] = FRACTION_STEPS,
    overlaps: tuple[float, ...] = OVERLAP_STEPS,
) -> list[CandidateGeometry]:
    """Full frame + training geometry + nearby fractions x overlaps (+ custom)."""
    base_fraction = float(training_geometry.get("object_tile_fraction") or 0.15)
    base_overlap = float(training_geometry.get("overlap") or 0.2)
    mode = str(training_geometry.get("geometry_mode") or "auto_object")
    if mode not in {"auto_model", "auto_object", "custom"}:
        mode = "auto_object"
    body_px = float(training_geometry.get("reference_body_px") or 0.0)
    out = [
        CandidateGeometry(
            enabled=False, geometry_mode=mode, slice_width=0, slice_height=0,
            overlap=base_overlap, object_tile_fraction=base_fraction,
            trained_body_px=body_px, label="Full frame (no SAHI)",
        )
    ]
    seen: set[tuple] = set()

    def _add(fraction: float, overlap: float, label: str) -> None:
        fraction = max(0.01, min(0.9, round(float(fraction), 4)))
        overlap = max(0.0, min(0.9, round(float(overlap), 4)))
        key = (mode, fraction, overlap)
        if key in seen:
            return
        seen.add(key)
        out.append(
            CandidateGeometry(
                enabled=True, geometry_mode=mode, slice_width=0, slice_height=0,
                overlap=overlap, object_tile_fraction=fraction,
                trained_body_px=body_px, label=label,
            )
        )

    _add(base_fraction, base_overlap, "Training geometry")
    for step in fraction_steps:
        for overlap in overlaps:
            _add(base_fraction * float(step), overlap,
                 f"fraction x{step:g}, overlap {overlap:g}")
    if custom is not None:
        out.append(
            CandidateGeometry(
                enabled=True, geometry_mode="custom", slice_width=int(custom[0]),
                slice_height=int(custom[1]), overlap=base_overlap,
                object_tile_fraction=base_fraction, trained_body_px=body_px,
                label=f"Custom {int(custom[0])}x{int(custom[1])}",
            )
        )
    return out


def estimate_grid_work(
    candidates: list[CandidateGeometry],
    *,
    frame_hw: tuple[int, int],
    imgsz: int,
    frames: int,
    max_total_tiles: int = DEFAULT_MAX_TOTAL_TILES,
) -> list[GridWorkEstimate]:
    """Tiles/frame per candidate; over-budget or unplannable candidates are FLAGGED.

    A silently omitted candidate looks to the user like a measured,
    unremarkable one -- so failures get a row with a reason instead.
    """
    from hydra_suite.core.inference.config import _slice_config_from_params
    from hydra_suite.core.inference.stages.slicing import plan_slices

    frames = max(0, int(frames))
    estimates: list[GridWorkEstimate] = []
    running = 0
    for candidate in candidates:
        if not candidate.enabled:
            tiles, reason = 1, ""
        else:
            try:
                params = candidate.slice_params()
                slice_cfg = _slice_config_from_params(
                    params, "SLICE_", reference_body_px=candidate.trained_body_px
                )
                plan = plan_slices(
                    frame_hw, slice_cfg, int(imgsz), None, float(candidate.trained_body_px)
                )
                tiles, reason = int(plan.jobs_per_frame), ""
            except Exception as exc:  # ValueError from MAX_TILES_PER_FRAME, etc.
                tiles, reason = 0, str(exc)
        total = tiles * frames
        running += total
        if not reason and running > max_total_tiles:
            reason = (
                f"exceeds the {max_total_tiles} tile budget "
                "(confirm a broader sweep to include it)"
            )
        estimates.append(GridWorkEstimate(candidate, tiles, total, reason))
    return estimates
```

`jobs_per_frame` (not `len(plan.tiles)`) is the right count: it includes the optional extra full-frame pass (`utils/slice_geometry.py:32-34`).

- [ ] **Step 4: Run the tests**

Run: `PYTHONPATH=$PWD/src python -m pytest tests/test_direct_calibration_grid.py -v`
Expected: PASS (7 tests).

- [ ] **Step 5: Commit**

```bash
make format
git add src/hydra_suite/core/inference/direct_calibration_grid.py tests/test_direct_calibration_grid.py
git commit -m "feat(calibration): bounded SAHI candidate grid over the shared params mapping"
```

---

## Task 3: Config construction and offline re-scoring through the production seam

**Files:**
- Create: `src/hydra_suite/core/inference/direct_calibration_sweep.py`
- Test: `tests/test_direct_calibration_sweep.py`

**Interfaces:**
- Consumes: `collect_obb_parts_by_frame` (Task 1), `CandidateGeometry.slice_params()` (Task 2).
- Produces:
  - `build_calibration_config(model_path: str, *, slice_params: dict, max_targets: int, confidence: float, runtime_tier: str = "cpu", model_task: str = "obb") -> InferenceConfig`
  - `@dataclass(frozen=True) MergeSettings(policy: str, metric: str, threshold: float)` — `policy ∈ {"nms","nmm","greedy_nmm"}`, `metric ∈ {"iou","ios"}`. `merge_backend` is **not** swept (forced to `cv2` on host paths, `obb.py:1571-1581`).
  - `config_for_point(base_model_path, *, slice_params, merge: MergeSettings, confidence: float, max_targets: int, runtime_tier: str, model_task: str) -> InferenceConfig`
  - `rescore_parts(parts, source, inference_config, runtime, *, frame_idx: int) -> OBBResult` — merge with `inference_config.obb`, materialize any `_RawOBBTensors`, then `filter_for_source`.
  - `detections_from_result(result) -> list[CalibrationDetection]`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_direct_calibration_sweep.py
"""Offline rescoring must be bit-identical to a fresh production run."""

import numpy as np
import pytest

from hydra_suite.core.inference.direct_calibration_sweep import (
    MergeSettings,
    build_calibration_config,
    config_for_point,
    detections_from_result,
    rescore_parts,
)
from hydra_suite.core.inference.stages import obb as obb_stage
from hydra_suite.core.inference.stages.filtering import filter_for_source

SLICE_PARAMS = {
    "SLICE_ENABLED": True,
    "SLICE_GEOMETRY_MODE": "auto_object",
    "SLICE_OBJECT_TILE_FRACTION": 0.4,
    "SLICE_OVERLAP": 0.2,
    "SLICE_TRAINED_BODY_PX": 120.0,
}


def test_config_is_built_from_params_and_carries_every_claimed_field(tmp_path):
    model = tmp_path / "m.pt"
    model.write_bytes(b"weights")
    config = build_calibration_config(
        str(model), slice_params=SLICE_PARAMS, max_targets=64,
        confidence=0.35, runtime_tier="cpu",
    )
    slice_cfg = config.obb.direct.slice
    assert slice_cfg.enabled is True
    assert slice_cfg.geometry_mode == "auto_object"
    assert slice_cfg.object_tile_fraction == pytest.approx(0.4)
    assert slice_cfg.overlap_width_ratio == pytest.approx(0.2)
    assert slice_cfg.reference_body_px == pytest.approx(120.0)
    assert config.obb.confidence_threshold == pytest.approx(0.35)
    assert config.obb.max_detections == 64
    assert config.obb.direct.model_path == str(model)


def test_merge_settings_reach_the_slice_config(tmp_path):
    model = tmp_path / "m.pt"
    model.write_bytes(b"weights")
    config = config_for_point(
        str(model), slice_params=SLICE_PARAMS,
        merge=MergeSettings("nmm", "iou", 0.65), confidence=0.35,
        max_targets=64, runtime_tier="cpu", model_task="obb",
    )
    assert config.obb.direct.slice.merge_policy == "nmm"
    assert config.obb.direct.slice.merge_metric == "iou"
    assert config.obb.direct.slice.merge_threshold == pytest.approx(0.65)


def test_offline_rescore_matches_a_fresh_production_run(direct_obb_fixture):
    frames, models, config, runtime = direct_obb_fixture
    model_path = config.obb.direct.model_path
    parts_by_frame, source = obb_stage.collect_obb_parts_by_frame(
        frames, models, config.obb, runtime
    )
    merge = MergeSettings("greedy_nmm", "ios", 0.5)
    for confidence in (0.10, 0.35, 0.60):
        point_config = config_for_point(
            model_path, slice_params=SLICE_PARAMS, merge=merge,
            confidence=confidence, max_targets=config.obb.max_detections,
            runtime_tier="cpu", model_task="obb",
        )
        fresh = []
        for raw in obb_stage.run_obb(frames, models, point_config.obb, runtime):
            if isinstance(raw, obb_stage._RawOBBTensors):
                raw = obb_stage.materialize_tensors(
                    raw, point_config.obb.raw_detection_cap
                )
            fresh.append(filter_for_source(point_config, raw, None)[0])
        cached = [
            rescore_parts(
                parts_by_frame[i], source, point_config, runtime, frame_idx=i
            )
            for i in range(len(frames))
        ]
        for want, got in zip(fresh, cached):
            assert want.num_detections == got.num_detections
            np.testing.assert_array_equal(want.centroids, got.centroids)
            np.testing.assert_array_equal(want.confidences, got.confidences)


def test_max_detections_truncation_is_visible_in_rescoring(direct_obb_fixture):
    """A too-small max_detections silently caps recall -- prove it bites."""
    frames, models, config, runtime = direct_obb_fixture
    model_path = config.obb.direct.model_path
    parts_by_frame, source = obb_stage.collect_obb_parts_by_frame(
        frames, models, config.obb, runtime
    )
    merge = MergeSettings("greedy_nmm", "ios", 0.5)
    counts = []
    for max_targets in (2, 64):
        point_config = config_for_point(
            model_path, slice_params=SLICE_PARAMS, merge=merge, confidence=0.05,
            max_targets=max_targets, runtime_tier="cpu", model_task="obb",
        )
        counts.append(
            rescore_parts(
                parts_by_frame[0], source, point_config, runtime, frame_idx=0
            ).num_detections
        )
    assert counts[0] <= 2
    assert counts[0] <= counts[1]


def test_detections_carry_frame_space_polygons_and_class_ids(direct_obb_fixture):
    frames, models, config, runtime = direct_obb_fixture
    result = obb_stage.run_obb(frames, models, config.obb, runtime)[0]
    if isinstance(result, obb_stage._RawOBBTensors):
        result = obb_stage.materialize_tensors(result, config.obb.raw_detection_cap)
    detections = detections_from_result(result)
    assert len(detections) == result.num_detections
    for detection in detections:
        assert detection.polygon_px.ndim == 2 and detection.polygon_px.shape[1] == 2
        assert detection.polygon_px.shape[0] >= 3
        assert isinstance(detection.class_id, int)
```

- [ ] **Step 2: Run and confirm failure**

Run: `PYTHONPATH=$PWD/src python -m pytest tests/test_direct_calibration_sweep.py -v`
Expected: FAIL — module missing.

- [ ] **Step 3: Implement**

```python
"""Build calibration configs from params and replay cached detections offline.

Two config types are in play and they are NOT interchangeable:
``merge_per_frame`` takes an ``OBBConfig`` (obb.py:1476) while
``filter_for_source`` takes the whole ``InferenceConfig`` (filtering.py:339).
This module always holds the ``InferenceConfig`` and passes ``.obb`` where an
``OBBConfig`` is wanted.

Production order is merge -> filter, and predict-time confidence is the fixed
``direct.confidence_floor`` (1e-3), independent of the filter threshold -- so
parts collected once are valid across the whole confidence sweep.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from hydra_suite.core.inference.config import InferenceConfig, build_obb_only_config
from hydra_suite.core.inference.direct_calibration import CalibrationDetection
from hydra_suite.core.inference.stages.filtering import filter_for_source
from hydra_suite.core.inference.stages.obb import (
    _RawOBBTensors,
    _empty_obb_result,
    materialize_tensors,
    merge_per_frame,
)


@dataclass(frozen=True)
class MergeSettings:
    """Cross-tile merge knobs a profile may claim to set.

    ``merge_backend`` is deliberately absent: it is forced to cv2 on every host
    path (obb.py:1571-1581), so sweeping it would produce duplicate rows on
    MPS. The resolved backend is recorded with the measurement instead.
    """

    policy: str = "greedy_nmm"
    metric: str = "ios"
    threshold: float = 0.5


def build_calibration_config(
    model_path: str,
    *,
    slice_params: dict,
    max_targets: int,
    confidence: float,
    runtime_tier: str = "cpu",
    model_task: str = "obb",
) -> InferenceConfig:
    """One production InferenceConfig, built through the shared params mapping.

    Going through ``build_obb_only_config`` (config.py:1151) rather than
    hand-building dataclasses is what guarantees a measured operating point is
    expressible as TrackerKit settings: both sides read the same SLICE_* keys.
    """
    return build_obb_only_config(
        model_path,
        runtime_tier=runtime_tier,
        confidence_threshold=float(confidence),
        max_targets=int(max_targets),
        model_task=model_task,
        extra_params=dict(slice_params),
    )


def config_for_point(
    model_path: str,
    *,
    slice_params: dict,
    merge: MergeSettings,
    confidence: float,
    max_targets: int,
    runtime_tier: str = "cpu",
    model_task: str = "obb",
) -> InferenceConfig:
    """A config for one measured row: geometry + merge + confidence + cap."""
    params = dict(slice_params)
    params.update(
        {
            "SLICE_MERGE_POLICY": merge.policy,
            "SLICE_MERGE_METRIC": merge.metric,
            "SLICE_MERGE_THRESHOLD": float(merge.threshold),
        }
    )
    return build_calibration_config(
        model_path,
        slice_params=params,
        max_targets=max_targets,
        confidence=confidence,
        runtime_tier=runtime_tier,
        model_task=model_task,
    )


def rescore_parts(parts, source, inference_config: InferenceConfig, runtime, *, frame_idx: int):
    """Merge one frame's cached parts, materialize, then filter -- production order."""
    if not parts:
        return _empty_obb_result(frame_idx)
    merged = merge_per_frame(
        parts,
        source.merge_policy,
        source.merge_plan(frame_idx),
        inference_config.obb,
        runtime,
    )
    if isinstance(merged, _RawOBBTensors):
        merged = materialize_tensors(merged, inference_config.obb.raw_detection_cap)
    filtered, _indices = filter_for_source(inference_config, merged, None)
    return filtered


def detections_from_result(result) -> list[CalibrationDetection]:
    """Frame-space calibration records for one post-merge, filtered result."""
    out: list[CalibrationDetection] = []
    polygons = getattr(result, "polygons", None)
    for index in range(int(result.num_detections)):
        polygon = None
        if polygons is not None and polygons[index] is not None:
            polygon = np.asarray(polygons[index], dtype=np.float32).reshape(-1, 2)
        if polygon is None or polygon.shape[0] < 3:
            polygon = np.asarray(result.corners[index], dtype=np.float32).reshape(-1, 2)
        out.append(
            CalibrationDetection(
                class_id=int(result.class_ids[index]),
                polygon_px=polygon,
                confidence=float(result.confidences[index]),
            )
        )
    return out
```

Note `source.merge_policy` is the RegionSource policy (`"plain"` / `"overlap_band_nms"`); the sweepable `MergeSettings.policy` reaches the merge through `config.obb.direct.slice`, which is exactly how production carries it.

- [ ] **Step 4: Run the tests**

Run: `PYTHONPATH=$PWD/src python -m pytest tests/test_direct_calibration_sweep.py -v`
Expected: PASS (5 tests). If `test_max_detections_truncation_is_visible_in_rescoring` finds fewer than 3 detections on the synthetic frames, replace the random frames in the fixture with the first frames of `tools/equivalence/fixtures/clips/fly_obb.mp4` so real detections exist.

- [ ] **Step 5: Commit**

```bash
make format
git add src/hydra_suite/core/inference/direct_calibration_sweep.py tests/test_direct_calibration_sweep.py
git commit -m "feat(calibration): params-built configs and offline merge/confidence rescoring"
```

---

## Task 4: Fingerprints for candidate caches and evidence sets

**Files:**
- Modify: `src/hydra_suite/core/inference/direct_calibration_grid.py`
- Test: `tests/test_direct_calibration_grid.py` (extend)

**Interfaces:**
- Produces: `candidate_cache_fingerprint(*, checkpoint_path, task, image_paths, candidate, imgsz, executor_key, max_detections, confidence_floor) -> str`; `label_set_fingerprint(frames) -> str`; `checkpoint_fingerprint(path) -> str` (`"sha256:<hex>"`).

- [ ] **Step 1: Write failing tests**

```python
def _checkpoint(tmp_path, payload=b"weights"):
    path = tmp_path / "m.pt"
    path.write_bytes(payload)
    return path


def test_fingerprint_distinguishes_every_geometry(tmp_path):
    from hydra_suite.core.inference.direct_calibration_grid import (
        build_candidate_grid, candidate_cache_fingerprint,
    )

    args = dict(
        checkpoint_path=_checkpoint(tmp_path), task="obb",
        image_paths=[tmp_path / "a.png"], imgsz=640, executor_key="torch:cpu",
        max_detections=64, confidence_floor=1e-3,
    )
    keys = {
        candidate_cache_fingerprint(candidate=c, **args)
        for c in build_candidate_grid(TRAINING)
    }
    assert len(keys) == len(build_candidate_grid(TRAINING))


def test_fingerprint_changes_when_weights_or_cap_change(tmp_path):
    from hydra_suite.core.inference.direct_calibration_grid import (
        build_candidate_grid, candidate_cache_fingerprint,
    )

    checkpoint = _checkpoint(tmp_path)
    candidate = build_candidate_grid(TRAINING)[1]
    args = dict(
        checkpoint_path=checkpoint, task="obb", image_paths=[tmp_path / "a.png"],
        candidate=candidate, imgsz=640, executor_key="torch:cpu",
        max_detections=64, confidence_floor=1e-3,
    )
    before = candidate_cache_fingerprint(**args)
    assert candidate_cache_fingerprint(**{**args, "max_detections": 8}) != before
    checkpoint.write_bytes(b"retrained")
    assert candidate_cache_fingerprint(**args) != before


def test_checkpoint_fingerprint_is_prefixed_and_stable(tmp_path):
    from hydra_suite.core.inference.direct_calibration_grid import checkpoint_fingerprint

    checkpoint = _checkpoint(tmp_path)
    digest = checkpoint_fingerprint(checkpoint)
    assert digest.startswith("sha256:") and len(digest) == 71
    assert digest == checkpoint_fingerprint(checkpoint)
```

- [ ] **Step 2: Run and confirm failure**

Run: `PYTHONPATH=$PWD/src python -m pytest tests/test_direct_calibration_grid.py -k fingerprint -v`
Expected: FAIL — `ImportError`.

- [ ] **Step 3: Implement**

```python
def _file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def checkpoint_fingerprint(path) -> str:
    """``sha256:<hex>`` of the weights, as stamped into a profile measurement."""
    return "sha256:" + _file_digest(Path(path))


def candidate_cache_fingerprint(
    *,
    checkpoint_path,
    task: str,
    image_paths,
    candidate: CandidateGeometry,
    imgsz: int,
    executor_key: str,
    max_detections: int,
    confidence_floor: float,
) -> str:
    """Identity of one measured candidate pass.

    Weights, task, image list, resolved tile plan, executor/imgsz, the raw
    detection cap (max_detections bounds the reservoir and truncates results)
    and the PREDICT-time floor all change which raw predictions exist. The
    filter-time confidence deliberately does NOT -- that is exactly what makes
    the offline sweep sound.
    """
    payload = json.dumps(
        {
            "checkpoint": _file_digest(Path(checkpoint_path)),
            "task": str(task),
            "images": [str(Path(p)) for p in image_paths],
            "candidate": [
                candidate.enabled, candidate.geometry_mode, candidate.slice_width,
                candidate.slice_height, round(candidate.overlap, 6),
                round(candidate.object_tile_fraction, 6),
                round(candidate.trained_body_px, 6),
            ],
            "imgsz": int(imgsz),
            "executor": str(executor_key),
            "max_detections": int(max_detections),
            "confidence_floor": round(float(confidence_floor), 9),
        },
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def label_set_fingerprint(frames) -> str:
    """Stable identity of the evidence set: file names plus label geometry."""
    digest = hashlib.sha256()
    for path, labels in sorted(frames, key=lambda item: str(item[0])):
        digest.update(str(Path(path).name).encode("utf-8"))
        for label in labels:
            points = np.asarray(label.points, dtype=np.float32).reshape(-1, 2)
            digest.update(
                f"{int(label.class_id)}:{np.round(points, 3).tobytes().hex()}".encode()
            )
    return digest.hexdigest()
```

- [ ] **Step 4: Run the tests**

Run: `PYTHONPATH=$PWD/src python -m pytest tests/test_direct_calibration_grid.py -v`
Expected: PASS (10 tests).

- [ ] **Step 5: Commit**

```bash
make format
git add src/hydra_suite/core/inference/direct_calibration_grid.py tests/test_direct_calibration_grid.py
git commit -m "feat(calibration): fingerprint candidate caches, weights and evidence sets"
```

---

## Task 5: Measured points and the printed `Balanced` rule

**Files:**
- Modify: `src/hydra_suite/core/inference/direct_calibration.py`
- Test: `tests/test_direct_calibration.py` (extend)

**Interfaces:**
- Produces: `@dataclass(frozen=True) DirectCalibrationPoint(label, enabled, geometry_mode, tile_width, tile_height, overlap, object_tile_fraction, max_detections, tiles_per_frame, seconds_per_frame, confidence, merge_policy, merge_metric, merge_threshold, merge_backend, score: CalibrationScore, failed_reason: str = "")`
- Produces: `recommend_balanced(points, *, min_matched=MIN_MATCHED_INSTANCES, f1_tolerance=F1_TOLERANCE, min_iou=MIN_LOCALIZATION) -> tuple[DirectCalibrationPoint | None, str]`; constants `MIN_MATCHED_INSTANCES = 60`, `F1_TOLERANCE = 0.01`, `MIN_LOCALIZATION = 0.5`, `RECOMMENDATION_RULE`.

**Note on the rule:** the recommendation is a **strict** reading of the printed sentence. There is no "or fall back to the frontier" escape — if nothing is within tolerance the tolerance filter cannot be empty (the best-F1 point is always within tolerance of itself), so a fallback would only ever fire on a bug.

- [ ] **Step 1: Write failing tests**

```python
# append to tests/test_direct_calibration.py
from hydra_suite.core.inference.direct_calibration import (
    CalibrationScore,
    DirectCalibrationPoint,
    MIN_MATCHED_INSTANCES,
    RECOMMENDATION_RULE,
    recommend_balanced,
)


def _point(label, f1, seconds, *, matched=200, missed=10, extra=10,
           mean_iou=0.8, failed=""):
    score = CalibrationScore(
        frames=20, matched=matched, missed=missed, extra=extra,
        precision=f1, recall=f1, f1=f1, duplicate=0, mean_iou=mean_iou,
    )
    return DirectCalibrationPoint(
        label=label, enabled=True, geometry_mode="auto_object", tile_width=640,
        tile_height=640, overlap=0.2, object_tile_fraction=0.4, max_detections=64,
        tiles_per_frame=9, seconds_per_frame=seconds, confidence=0.35,
        merge_policy="greedy_nmm", merge_metric="ios", merge_threshold=0.5,
        merge_backend="cv2", score=score, failed_reason=failed,
    )


def test_recommendation_prefers_the_cheapest_point_within_f1_tolerance():
    """'fast' is on the frontier (cheapest) and within 0.01 F1 of the best."""
    best, reason = recommend_balanced(
        [
            _point("slow", 0.920, 2.0, missed=8, extra=8),
            _point("fast", 0.915, 0.4, missed=9, extra=9),
            _point("bad", 0.600, 0.1, missed=60, extra=60),
        ]
    )
    assert best.label == "fast"
    assert RECOMMENDATION_RULE in reason


def test_a_dominated_cheap_point_never_wins():
    """'cheap' is worse on misses AND extras AND time is not enough to save it."""
    best, _reason = recommend_balanced(
        [
            _point("good", 0.930, 1.0, missed=5, extra=5),
            _point("cheap", 0.930, 2.0, missed=9, extra=9),
        ]
    )
    assert best.label == "good"


def test_failed_and_undersampled_points_are_never_recommended():
    best, reason = recommend_balanced(
        [
            _point("broken", 0.99, 0.1, failed="tile budget exceeded"),
            _point("thin", 0.99, 0.1, matched=MIN_MATCHED_INSTANCES - 1),
        ]
    )
    assert best is None
    assert "matched instances" in reason


def test_poor_localization_is_excluded_even_at_high_f1():
    best, _reason = recommend_balanced(
        [
            _point("sloppy", 0.99, 0.1, mean_iou=0.2),
            _point("clean", 0.90, 1.0, missed=5, extra=5),
        ]
    )
    assert best.label == "clean"


def test_empty_input_refuses_rather_than_raising():
    best, reason = recommend_balanced([])
    assert best is None and RECOMMENDATION_RULE in reason
```

- [ ] **Step 2: Run and confirm failure**

Run: `PYTHONPATH=$PWD/src python -m pytest tests/test_direct_calibration.py -v`
Expected: FAIL — `ImportError: cannot import name 'DirectCalibrationPoint'`.

- [ ] **Step 3: Implement**

```python
MIN_MATCHED_INSTANCES = 60
F1_TOLERANCE = 0.01
MIN_LOCALIZATION = 0.5
RECOMMENDATION_RULE = (
    "Balanced rule: drop failed and undersampled points, keep the Pareto "
    "frontier of misses, extras and time, then take the fastest point whose F1 "
    "is within 0.01 of the best and whose localization quality is at least 0.5."
)


@dataclass(frozen=True)
class DirectCalibrationPoint:
    """One fully measured SAHI operating point, with its evidence attached."""

    label: str
    enabled: bool
    geometry_mode: str
    tile_width: int
    tile_height: int
    overlap: float
    object_tile_fraction: float
    max_detections: int
    tiles_per_frame: int
    seconds_per_frame: float
    confidence: float
    merge_policy: str
    merge_metric: str
    merge_threshold: float
    merge_backend: str
    score: CalibrationScore
    failed_reason: str = ""


def _pareto(points: Sequence[DirectCalibrationPoint]) -> list[DirectCalibrationPoint]:
    """Keep points not dominated on (misses, extras, seconds) simultaneously."""

    def cost(point: DirectCalibrationPoint) -> tuple[float, float, float]:
        return (
            float(point.score.missed),
            float(point.score.extra),
            float(point.seconds_per_frame),
        )

    keep: list[DirectCalibrationPoint] = []
    for candidate in points:
        this = cost(candidate)
        dominated = any(
            all(o <= t for o, t in zip(cost(other), this))
            and any(o < t for o, t in zip(cost(other), this))
            for other in points
            if other is not candidate
        )
        if not dominated:
            keep.append(candidate)
    return keep


def recommend_balanced(
    points: Sequence[DirectCalibrationPoint],
    *,
    min_matched: int = MIN_MATCHED_INSTANCES,
    f1_tolerance: float = F1_TOLERANCE,
    min_iou: float = MIN_LOCALIZATION,
) -> tuple[DirectCalibrationPoint | None, str]:
    """Explain a suggestion, or refuse. It is never applied automatically.

    The floors are ELIGIBILITY filters, not vetoes on the winner: a
    configuration that finds almost nothing would otherwise post a perfect F1
    on a handful of matches and win.
    """
    eligible = [
        point
        for point in points
        if not point.failed_reason
        and point.score.matched >= min_matched
        and point.score.mean_iou >= min_iou
    ]
    if not eligible:
        return None, (
            f"No point cleared the floors: at least {min_matched} matched "
            f"instances and {min_iou:g} localization quality. Label a few more "
            "frames or widen the sweep. " + RECOMMENDATION_RULE
        )
    best_f1 = max(point.score.f1 for point in eligible)
    frontier = _pareto(eligible)
    near_best = [p for p in frontier if p.score.f1 >= best_f1 - f1_tolerance]
    if not near_best:
        # The best-F1 point is always within tolerance of itself, so an empty
        # set here means it was dominated on every cost axis; fall back to it
        # explicitly rather than to an arbitrary frontier member.
        near_best = [p for p in eligible if p.score.f1 >= best_f1 - f1_tolerance]
    chosen = min(near_best, key=lambda p: p.seconds_per_frame)
    return chosen, (
        f"{chosen.label}: F1 {chosen.score.f1:.3f} (best {best_f1:.3f}), "
        f"{chosen.seconds_per_frame:.2f}s/frame on this machine and data. "
        + RECOMMENDATION_RULE
    )
```

Add `from typing import Sequence` if absent.

- [ ] **Step 4: Run the tests**

Run: `PYTHONPATH=$PWD/src python -m pytest tests/test_direct_calibration.py -v`
Expected: PASS (7 tests).

- [ ] **Step 5: Commit**

```bash
make format
git add src/hydra_suite/core/inference/direct_calibration.py tests/test_direct_calibration.py
git commit -m "feat(calibration): measured points and the explainable Balanced rule"
```

---

## Task 6: Fix the two shipped sidecar deviations from the spec

**Files:**
- Modify: `src/hydra_suite/core/inference/slice_meta.py:75-133`
- Modify: `tests/test_model_publish_slice_geometry.py:227-230` (asserts the old behaviour)
- Test: `tests/test_slice_profile_mutations.py`

Shipped `upsert_slice_profile` auto-designates the first saved profile primary; `remove_slice_profile` silently promotes `profiles[0]`. The spec: "The user marks at most one profile **Primary**" and "Removing the primary profile requires choosing a replacement or explicitly clearing the designation."

**Interfaces:**
- Produces: `upsert_slice_profile(..., primary: bool = False)` no longer auto-promotes; `remove_slice_profile(meta, profile_id, *, new_primary_id: str | None = None)` raises `ValueError` when removing the primary without an explicit decision (`""` clears).

- [ ] **Step 1: Write failing tests**

```python
# tests/test_slice_profile_mutations.py
import pytest

from hydra_suite.core.inference.slice_meta import (
    available_slice_profiles,
    remove_slice_profile,
    upsert_slice_profile,
)

BASE = {"geometry_mode": "auto_object", "imgsz": 640, "overlap": 0.2}
SETTINGS = {"enabled": True, "geometry_mode": "auto_object", "overlap": 0.2}


def test_saving_a_profile_does_not_silently_make_it_primary():
    meta = upsert_slice_profile(BASE, name="Balanced", settings=SETTINGS)
    assert meta["primary_profile_id"] == ""
    assert len(meta["profiles"]) == 1


def test_primary_is_set_only_when_explicitly_requested():
    meta = upsert_slice_profile(BASE, name="Balanced", settings=SETTINGS, primary=True)
    assert meta["primary_profile_id"] == meta["profiles"][0]["id"]


def test_removing_the_primary_requires_an_explicit_decision():
    meta = upsert_slice_profile(BASE, name="Balanced", settings=SETTINGS, primary=True)
    meta = upsert_slice_profile(meta, name="Fast scan", settings=SETTINGS)
    primary_id = meta["primary_profile_id"]
    other = next(p["id"] for p in meta["profiles"] if p["id"] != primary_id)
    with pytest.raises(ValueError, match="replacement"):
        remove_slice_profile(meta, primary_id)
    assert remove_slice_profile(meta, primary_id, new_primary_id="")["primary_profile_id"] == ""
    moved = remove_slice_profile(meta, primary_id, new_primary_id=other)
    assert moved["primary_profile_id"] == other
    assert len(available_slice_profiles(moved)) == 1


def test_unknown_replacement_is_rejected():
    meta = upsert_slice_profile(BASE, name="Balanced", settings=SETTINGS, primary=True)
    with pytest.raises(ValueError, match="Unknown"):
        remove_slice_profile(meta, meta["primary_profile_id"], new_primary_id="nope")


def test_removing_a_non_primary_profile_needs_no_decision():
    meta = upsert_slice_profile(BASE, name="Balanced", settings=SETTINGS, primary=True)
    meta = upsert_slice_profile(meta, name="Fast scan", settings=SETTINGS)
    victim = next(p for p in meta["profiles"] if p["id"] != meta["primary_profile_id"])
    result = remove_slice_profile(meta, victim["id"])
    assert result["primary_profile_id"] == meta["primary_profile_id"]


def test_unknown_future_keys_round_trip_untouched():
    meta = upsert_slice_profile(
        BASE, name="Balanced", settings=dict(SETTINGS, future_knob=7)
    )
    assert available_slice_profiles(meta)[0]["settings"]["future_knob"] == 7
```

- [ ] **Step 2: Run and confirm failure**

Run: `PYTHONPATH=$PWD/src python -m pytest tests/test_slice_profile_mutations.py -v`
Expected: FAIL on the first four tests.

- [ ] **Step 3: Implement**

In `upsert_slice_profile`, replace `if primary or not result["primary_profile_id"]:` with:

```python
    # Primary is a user designation, never a side effect of saving the first
    # profile: an inferred default is exactly what the spec forbids.
    if primary:
        result["primary_profile_id"] = selected_id
```

Replace `remove_slice_profile`:

```python
def remove_slice_profile(
    meta: dict[str, Any],
    profile_id: str,
    *,
    new_primary_id: str | None = None,
) -> dict[str, Any]:
    """Remove one profile, preserving weights, geometry and every other profile.

    Removing the PRIMARY requires an explicit decision: ``new_primary_id=""``
    clears the designation, an id promotes that profile. Silently promoting
    whatever happened to be first would hand the user an operating point they
    never chose.
    """
    result = normalized_slice_meta(meta)
    target = str(profile_id)
    remaining = [p for p in result["profiles"] if p["id"] != target]
    if len(remaining) == len(result["profiles"]):
        return result
    if result["primary_profile_id"] == target:
        if new_primary_id is None:
            raise ValueError(
                "Removing the primary SAHI profile needs a replacement "
                "(pass new_primary_id) or an explicit clear (new_primary_id='')."
            )
        chosen = str(new_primary_id)
        if chosen and chosen not in {p["id"] for p in remaining}:
            raise ValueError(f"Unknown replacement primary profile {chosen!r}.")
        result["primary_profile_id"] = chosen
    result["profiles"] = remaining
    return result
```

- [ ] **Step 4: Update the shipped test that asserts auto-promotion**

Read `tests/test_model_publish_slice_geometry.py:227-230`; change its expectation to `""` and add a comment naming the spec line ("The user marks at most one profile Primary").

- [ ] **Step 5: Run the tests**

Run: `PYTHONPATH=$PWD/src python -m pytest tests/test_slice_profile_mutations.py tests/test_slice_meta_read.py tests/test_trackerkit_slice_meta_prefill.py tests/test_model_publish_slice_geometry.py -v`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
make format
git add src/hydra_suite/core/inference/slice_meta.py tests/test_slice_profile_mutations.py tests/test_model_publish_slice_geometry.py
git commit -m "fix(slice-meta): make primary designation an explicit user decision"
```

---

## Task 7: Evidence selection — val default, recording-aware sampling, exhaustive-label gate

**Files:**
- Create: `src/hydra_suite/detectkit/jobs/direct_calibration.py`
- Test: `tests/test_detectkit_direct_calibration_job.py`

**Interfaces:**
- Produces: `@dataclass(frozen=True) EvidenceSet(frames, split: str, instances: int, size_range, sampled_from: int, fingerprint: str)`; `collect_evidence(*, dataset_yaml: Path | None, sources: list, split: str = "val", budget: int = 80) -> EvidenceSet`; `EXHAUSTIVE_LABEL_WARNING`, `MIN_MATCHED_NOTE`.
- Sampling keeps frames from one recording together: group by parent directory + filename stem prefix, take whole groups until the budget is met, rather than striding across the sorted list (which would scatter neighbours across recordings).

- [ ] **Step 1: Write failing tests**

```python
# tests/test_detectkit_direct_calibration_job.py
from pathlib import Path

import cv2
import numpy as np

from hydra_suite.detectkit.jobs.direct_calibration import (
    EXHAUSTIVE_LABEL_WARNING,
    collect_evidence,
)

LABEL_LINE = "0 0.1 0.1 0.2 0.1 0.2 0.2 0.1 0.2\n"


def _dataset(tmp_path: Path, split: str, names: list[str]) -> Path:
    images = tmp_path / "images" / split
    labels = tmp_path / "labels" / split
    images.mkdir(parents=True, exist_ok=True)
    labels.mkdir(parents=True, exist_ok=True)
    for name in names:
        cv2.imwrite(str(images / f"{name}.png"), np.zeros((200, 300, 3), np.uint8))
        (labels / f"{name}.txt").write_text(LABEL_LINE)
    yaml = tmp_path / "data.yaml"
    yaml.write_text(
        f"path: {tmp_path}\ntrain: images/train\nval: images/val\nnames:\n  0: ant\n"
    )
    return yaml


def test_val_split_is_the_default_evidence(tmp_path):
    _dataset(tmp_path, "train", ["a", "b", "c"])
    yaml = _dataset(tmp_path, "val", ["v0", "v1"])
    evidence = collect_evidence(dataset_yaml=yaml, sources=[])
    assert evidence.split == "val"
    assert len(evidence.frames) == 2 and evidence.instances == 2
    assert evidence.size_range == ((200, 300), (200, 300))


def test_missing_val_split_falls_back_to_train_and_reports_it(tmp_path):
    yaml = _dataset(tmp_path, "train", ["a", "b"])
    evidence = collect_evidence(dataset_yaml=yaml, sources=[], split="val")
    assert evidence.split == "train"


def test_sampling_keeps_one_recording_together(tmp_path):
    yaml = _dataset(
        tmp_path, "val",
        [f"rec1_{i:03d}" for i in range(6)] + [f"rec2_{i:03d}" for i in range(6)],
    )
    evidence = collect_evidence(dataset_yaml=yaml, sources=[], budget=6)
    stems = {Path(p).stem.split("_")[0] for p, _labels in evidence.frames}
    assert stems == {"rec1"}, "budget must consume whole recordings, not stride"
    assert evidence.sampled_from == 12


def test_evidence_fingerprint_changes_with_labels(tmp_path):
    yaml = _dataset(tmp_path, "val", ["v0", "v1"])
    before = collect_evidence(dataset_yaml=yaml, sources=[]).fingerprint
    (tmp_path / "labels" / "val" / "v0.txt").write_text(
        "0 0.3 0.3 0.4 0.3 0.4 0.4 0.3 0.4\n"
    )
    assert collect_evidence(dataset_yaml=yaml, sources=[]).fingerprint != before


def test_exhaustive_label_warning_is_stated():
    assert "exhaustively labelled" in EXHAUSTIVE_LABEL_WARNING
```

- [ ] **Step 2: Run and confirm failure**

Run: `PYTHONPATH=$PWD/src python -m pytest tests/test_detectkit_direct_calibration_job.py -v`
Expected: FAIL — module missing.

- [ ] **Step 3: Implement**

```python
"""DetectKit-side adapter and worker for direct-detector SAHI calibration.

Core owns the grid, the sweep and the scoring; this module supplies labelled
frames, drives the production runner, and persists project-local evidence.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import yaml as _yaml

from hydra_suite.core.inference.direct_calibration_grid import label_set_fingerprint
from hydra_suite.detectkit.jobs.semantic_escalation import (
    stratified_calibration_frames,
)

EXHAUSTIVE_LABEL_WARNING = (
    "Confirm these frames are exhaustively labelled. A real animal missing "
    "from the labels looks like a false positive and biases calibration "
    "toward settings that are too strict."
)
MIN_MATCHED_NOTE = (
    "Too few matched instances for a recommendation. The measurements are "
    "still shown, but label a few more frames before trusting them."
)
_IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


@dataclass(frozen=True)
class EvidenceSet:
    frames: list
    split: str
    instances: int
    size_range: tuple
    sampled_from: int
    fingerprint: str


def _recording_key(image_path: Path) -> tuple[str, str]:
    """Group frames by their recording: parent dir + filename stem prefix.

    Neighbouring frames from one video must stay together -- scattering them
    across recordings makes the evidence set look more diverse than it is.
    """
    stem = image_path.stem
    prefix = stem.rsplit("_", 1)[0] if "_" in stem else stem
    return (str(image_path.parent), prefix)


def _split_frames(dataset_yaml: Path, split: str) -> list:
    from hydra_suite.data.al.escalation import LabelRecord
    from hydra_suite.detectkit.gui.utils import parse_obb_label
    from hydra_suite.utils.geometry_levels import GeometryLevel

    document = _yaml.safe_load(Path(dataset_yaml).read_text(encoding="utf-8")) or {}
    root = Path(document.get("path") or Path(dataset_yaml).parent)
    rel = document.get(split)
    images_dir = (root / rel) if rel else (root / "images" / split)
    if not images_dir.is_dir():
        return []
    labels_dir = Path(str(images_dir).replace("/images/", "/labels/"))
    out = []
    for image_path in sorted(
        p for p in images_dir.rglob("*") if p.suffix.lower() in _IMG_EXTS
    ):
        label_path = labels_dir / (image_path.stem + ".txt")
        if not label_path.exists() or not label_path.read_text().strip():
            continue
        image = cv2.imread(str(image_path))
        if image is None:
            continue
        height, width = image.shape[:2]
        parsed = parse_obb_label(label_path, width, height)
        if not parsed:
            continue
        out.append(
            (
                image_path,
                [
                    LabelRecord(
                        class_id=int(d["class_id"]),
                        confidence=1.0,
                        points=np.asarray(d["polygon_px"], dtype=np.float32).reshape(-1, 2),
                        level=GeometryLevel.POLYGON,
                    )
                    for d in parsed
                ],
            )
        )
    return out


def _bounded_by_recording(frames: list, budget: int) -> list:
    """Take whole recordings until the budget is reached."""
    if not budget or len(frames) <= budget:
        return frames
    grouped: dict[tuple[str, str], list] = {}
    for item in frames:
        grouped.setdefault(_recording_key(Path(item[0])), []).append(item)
    output: list = []
    for _key, group in sorted(grouped.items()):
        if output and len(output) + len(group) > budget:
            break
        output.extend(group)
    return output or frames[:budget]


def collect_evidence(
    *,
    dataset_yaml: Path | None,
    sources: list,
    split: str = "val",
    budget: int = 80,
) -> EvidenceSet:
    """Labelled full-resolution evidence, defaulting to the held-out val split.

    Tuning on frames the model took gradient steps on reports optimistic
    numbers, so ``val`` is the default and any fallback is reported in
    ``EvidenceSet.split`` for the UI to show.
    """
    used_split = split
    frames: list = []
    if dataset_yaml is not None:
        frames = _split_frames(Path(dataset_yaml), split)
        if not frames and split != "train":
            frames = _split_frames(Path(dataset_yaml), "train")
            if frames:
                used_split = "train"
    if not frames and sources:
        frames = stratified_calibration_frames(sources, budget=budget)
        used_split = "sources"
    total = len(frames)
    frames = _bounded_by_recording(frames, budget)
    sizes = []
    for image_path, _labels in frames:
        image = cv2.imread(str(image_path))
        if image is not None:
            sizes.append(tuple(image.shape[:2]))
    size_range = (min(sizes), max(sizes)) if sizes else ((0, 0), (0, 0))
    return EvidenceSet(
        frames=frames,
        split=used_split,
        instances=sum(len(labels) for _p, labels in frames),
        size_range=size_range,
        sampled_from=total,
        fingerprint=label_set_fingerprint(frames),
    )
```

- [ ] **Step 4: Run the tests**

Run: `PYTHONPATH=$PWD/src python -m pytest tests/test_detectkit_direct_calibration_job.py -v`
Expected: PASS (5 tests).

- [ ] **Step 5: Commit**

```bash
make format
git add src/hydra_suite/detectkit/jobs/direct_calibration.py tests/test_detectkit_direct_calibration_job.py
git commit -m "feat(detectkit): recording-aware labelled evidence for SAHI calibration"
```

---

## Task 8: The calibration sweep driver and worker

**Files:**
- Modify: `src/hydra_suite/detectkit/jobs/direct_calibration.py`
- Test: `tests/test_detectkit_direct_calibration_job.py` (extend)

**Interfaces:**
- Consumes: everything from Tasks 1–5, 7.
- Produces:
  - `@dataclass(frozen=True) DirectCalibrationRequest(model_path: Path, task: str, evidence: EvidenceSet, candidates: list, confidences: tuple, merge_settings: tuple, runtime_tier: str, max_targets: int, evidence_dir: Path)`
  - `@dataclass DirectCalibrationOutcome(points: list, previews: list, partial: bool, message: str)`
  - `run_direct_calibration(request, *, progress=None, should_stop=None) -> DirectCalibrationOutcome` (Qt-free)
  - `load_calibration_models(request, candidate) -> tuple[OBBModels, RuntimeContext, InferenceConfig, int]` — the last element is the resolved `imgsz`. **Always a 4-tuple**; tests monkeypatch this exact arity.
  - `class DirectCalibrationWorker(BaseWorker)` with `result_ready = Signal(object)`, `cancel()`, and its **own** `_should_stop` flag (`BaseWorker` has none).

- [ ] **Step 1: Write failing tests**

```python
class _FakeSource:
    """Mimics a RegionSource. merge_policy here is the REGION policy vocabulary."""

    merge_policy = "plain"

    def merge_plan(self, _frame_idx):
        return None


def _fake_models(request, candidate):
    """4-tuple matching load_calibration_models: (models, runtime, config, imgsz)."""
    from hydra_suite.core.inference.direct_calibration_sweep import config_for_point

    config = config_for_point(
        str(request.model_path), slice_params=candidate.slice_params(),
        merge=request.merge_settings[0], confidence=request.confidences[0],
        max_targets=request.max_targets, runtime_tier="cpu", model_task="obb",
    )
    return object(), object(), config, 640


def _request(tmp_path, confidences=(0.35,), merges=None):
    from hydra_suite.core.inference.direct_calibration_grid import build_candidate_grid
    from hydra_suite.core.inference.direct_calibration_sweep import MergeSettings
    from hydra_suite.detectkit.jobs.direct_calibration import (
        DirectCalibrationRequest, EvidenceSet,
    )

    model = tmp_path / "m.pt"
    model.write_bytes(b"weights")
    evidence = EvidenceSet(
        frames=[], split="val", instances=0, size_range=((720, 1280), (720, 1280)),
        sampled_from=0, fingerprint="deadbeef",
    )
    return DirectCalibrationRequest(
        model_path=model, task="obb", evidence=evidence,
        candidates=build_candidate_grid(
            {"geometry_mode": "auto_object", "imgsz": 640,
             "object_tile_fraction": 0.4, "overlap": 0.2}
        )[:2],
        confidences=confidences,
        merge_settings=merges or (MergeSettings("greedy_nmm", "ios", 0.5),),
        runtime_tier="cpu", max_targets=64, evidence_dir=tmp_path / "evidence",
    )


def test_one_inference_pass_per_geometry_regardless_of_sweep_size(monkeypatch, tmp_path):
    from hydra_suite.core.inference.direct_calibration_sweep import MergeSettings
    from hydra_suite.detectkit.jobs import direct_calibration as job

    calls = []
    monkeypatch.setattr(job, "load_calibration_models", _fake_models)
    monkeypatch.setattr(
        job, "collect_obb_parts_by_frame",
        lambda frames, *a, **k: (calls.append(1) or ([[] for _ in frames], _FakeSource())),
    )
    request = _request(
        tmp_path, confidences=(0.1, 0.2, 0.3, 0.4),
        merges=(MergeSettings("greedy_nmm", "ios", 0.5), MergeSettings("nmm", "iou", 0.6)),
    )
    outcome = job.run_direct_calibration(request)
    assert len(calls) == len(request.candidates), "one model pass per geometry"
    assert len(outcome.points) == len(request.candidates) * 4 * 2


def test_cancellation_returns_partial_and_never_claims_completeness(monkeypatch, tmp_path):
    from hydra_suite.detectkit.jobs import direct_calibration as job

    monkeypatch.setattr(job, "load_calibration_models", _fake_models)
    monkeypatch.setattr(
        job, "collect_obb_parts_by_frame",
        lambda frames, *a, **k: ([[] for _ in frames], _FakeSource()),
    )
    outcome = job.run_direct_calibration(_request(tmp_path), should_stop=lambda: True)
    assert outcome.partial is True and outcome.points == []


def test_failed_candidate_becomes_a_failed_row_not_a_silent_omission(monkeypatch, tmp_path):
    from hydra_suite.detectkit.jobs import direct_calibration as job

    monkeypatch.setattr(job, "load_calibration_models", _fake_models)

    def boom(*_a, **_k):
        raise ValueError("tile budget exceeded")

    monkeypatch.setattr(job, "collect_obb_parts_by_frame", boom)
    outcome = job.run_direct_calibration(_request(tmp_path))
    assert len(outcome.points) == len(_request(tmp_path).candidates)
    assert all(point.failed_reason for point in outcome.points)
    assert "tile budget exceeded" in outcome.points[0].failed_reason


def test_points_record_the_detection_cap_they_were_measured_under(monkeypatch, tmp_path):
    from hydra_suite.detectkit.jobs import direct_calibration as job

    monkeypatch.setattr(job, "load_calibration_models", _fake_models)
    monkeypatch.setattr(
        job, "collect_obb_parts_by_frame",
        lambda frames, *a, **k: ([[] for _ in frames], _FakeSource()),
    )
    outcome = job.run_direct_calibration(_request(tmp_path))
    assert all(point.max_detections == 64 for point in outcome.points)
```

- [ ] **Step 2: Run and confirm failure**

Run: `PYTHONPATH=$PWD/src python -m pytest tests/test_detectkit_direct_calibration_job.py -v`
Expected: FAIL — `DirectCalibrationRequest` missing.

- [ ] **Step 3: Implement the sweep**

```python
import time

from hydra_suite.core.inference.direct_calibration import (
    CalibrationDetection,
    CalibrationScore,
    DirectCalibrationPoint,
    score_frames,
)
from hydra_suite.core.inference.direct_calibration_grid import estimate_grid_work
from hydra_suite.core.inference.direct_calibration_sweep import (
    config_for_point,
    detections_from_result,
    rescore_parts,
)
from hydra_suite.core.inference.stages.obb import collect_obb_parts_by_frame

PREVIEW_FRAMES = 8


@dataclass(frozen=True)
class DirectCalibrationRequest:
    model_path: Path
    task: str
    evidence: EvidenceSet
    candidates: list
    confidences: tuple
    merge_settings: tuple
    runtime_tier: str
    max_targets: int
    evidence_dir: Path


@dataclass
class DirectCalibrationOutcome:
    points: list = field(default_factory=list)
    previews: list = field(default_factory=list)
    partial: bool = False
    message: str = ""


def load_calibration_models(request: DirectCalibrationRequest, candidate):
    """Load the production models/runtime/config for one candidate geometry.

    Returns a 4-tuple ``(models, runtime, config, imgsz)``. Tiny and
    monkeypatchable so the sweep's control flow is testable without weights.
    """
    from hydra_suite.core.inference.runtime import RuntimeContext
    from hydra_suite.core.inference.stages.obb import _resolve_imgsz, load_obb_models

    config = config_for_point(
        str(request.model_path),
        slice_params=candidate.slice_params(),
        merge=request.merge_settings[0],
        confidence=request.confidences[0],
        max_targets=request.max_targets,
        runtime_tier=request.runtime_tier,
        model_task=request.task,
    )
    runtime = RuntimeContext.from_config(config)
    models = load_obb_models(config.obb, runtime)
    return models, runtime, config, int(_resolve_imgsz(models.direct_model))


def _label_detections(labels) -> list[CalibrationDetection]:
    return [
        CalibrationDetection(
            class_id=int(label.class_id),
            polygon_px=np.asarray(label.points, dtype=np.float32).reshape(-1, 2),
            confidence=1.0,
        )
        for label in labels
    ]


def _zero_score() -> CalibrationScore:
    return CalibrationScore(
        frames=0, matched=0, missed=0, extra=0, duplicate=0,
        precision=0.0, recall=0.0, f1=0.0, mean_iou=0.0,
    )


def _point_for(candidate, *, request, merge, confidence, tiles, seconds, score,
               merge_backend="cv2", failed_reason=""):
    return DirectCalibrationPoint(
        label=candidate.label,
        enabled=candidate.enabled,
        geometry_mode=candidate.geometry_mode,
        tile_width=candidate.slice_width,
        tile_height=candidate.slice_height,
        overlap=candidate.overlap,
        object_tile_fraction=candidate.object_tile_fraction,
        max_detections=int(request.max_targets),
        tiles_per_frame=int(tiles),
        seconds_per_frame=float(seconds),
        confidence=float(confidence),
        merge_policy=merge.policy,
        merge_metric=merge.metric,
        merge_threshold=float(merge.threshold),
        merge_backend=merge_backend,
        score=score,
        failed_reason=failed_reason,
    )


def run_direct_calibration(request, *, progress=None, should_stop=None):
    """One model pass per geometry; confidence x merge swept offline.

    Frames are processed one at a time: a calibration run holds each frame's
    pre-merge parts for the whole sweep, so batching them would multiply peak
    memory by the frame count.

    Partial work is returned with ``partial=True``. It is inspectable but the
    caller must never let it replace complete calibration or become a profile.
    """
    outcome = DirectCalibrationOutcome()
    frames = request.evidence.frames
    estimates = {
        estimate.candidate.label: estimate
        for estimate in estimate_grid_work(
            request.candidates,
            frame_hw=request.evidence.size_range[1],
            imgsz=0,
            frames=len(frames),
        )
    }
    for index, candidate in enumerate(request.candidates):
        if should_stop is not None and should_stop():
            outcome.partial = True
            outcome.message = "Cancelled; prior complete calibration is untouched."
            return outcome
        if progress is not None:
            progress(index, len(request.candidates), candidate.label)
        try:
            models, runtime, base_config, imgsz = load_calibration_models(
                request, candidate
            )
        except Exception as exc:
            for merge in request.merge_settings:
                for confidence in request.confidences:
                    outcome.points.append(
                        _point_for(candidate, request=request, merge=merge,
                                   confidence=confidence, tiles=0, seconds=0.0,
                                   score=_zero_score(), failed_reason=str(exc))
                    )
            continue
        # Re-estimate with the model's real imgsz: auto_model/custom geometries
        # depend on it, and the pre-run estimate had to guess.
        measured = estimate_grid_work(
            [candidate], frame_hw=request.evidence.size_range[1], imgsz=imgsz,
            frames=len(frames),
        )[0]
        tiles = measured.tiles_per_frame or (
            estimates[candidate.label].tiles_per_frame
        )
        parts_per_frame: list = []
        source = None
        elapsed = 0.0
        failure = ""
        for image_path, _labels in frames:
            if should_stop is not None and should_stop():
                outcome.partial = True
                outcome.message = "Cancelled; prior complete calibration is untouched."
                return outcome
            image = cv2.imread(str(image_path))
            if image is None:
                failure = f"could not read {Path(image_path).name}"
                break
            started = time.perf_counter()
            try:
                parts, source = collect_obb_parts_by_frame(
                    [image], models, base_config.obb, runtime
                )
            except Exception as exc:
                failure = str(exc)
                break
            elapsed += time.perf_counter() - started
            parts_per_frame.append(parts[0])
        if failure or source is None:
            for merge in request.merge_settings:
                for confidence in request.confidences:
                    outcome.points.append(
                        _point_for(candidate, request=request, merge=merge,
                                   confidence=confidence, tiles=tiles, seconds=0.0,
                                   score=_zero_score(),
                                   failed_reason=failure or "no regions produced")
                    )
            continue
        seconds_per_frame = elapsed / max(1, len(parts_per_frame))
        backend = base_config.obb.direct.slice.merge_backend
        for merge in request.merge_settings:
            for confidence in request.confidences:
                point_config = config_for_point(
                    str(request.model_path),
                    slice_params=candidate.slice_params(),
                    merge=merge,
                    confidence=confidence,
                    max_targets=request.max_targets,
                    runtime_tier=request.runtime_tier,
                    model_task=request.task,
                )
                scored = [
                    (
                        detections_from_result(
                            rescore_parts(
                                parts_per_frame[i], source, point_config, runtime,
                                frame_idx=i,
                            )
                        ),
                        _label_detections(frames[i][1]),
                    )
                    for i in range(len(parts_per_frame))
                ]
                outcome.points.append(
                    _point_for(candidate, request=request, merge=merge,
                               confidence=confidence, tiles=tiles,
                               seconds=seconds_per_frame,
                               score=score_frames(scored), merge_backend=backend)
                )
        outcome.previews.append(
            _preview_for(request, candidate, parts_per_frame[:PREVIEW_FRAMES],
                         source, base_config, runtime)
        )
        del parts_per_frame
    return outcome
```

Write `_preview_for(...)` to store, for at most `PREVIEW_FRAMES` frames: the image path, the ground-truth polygons, and the post-merge prediction polygons at the *training-geometry* merge/confidence — no decoded image arrays retained.

- [ ] **Step 4: Implement the worker (Qt half, at the bottom of the module)**

```python
from PySide6.QtCore import Signal

from hydra_suite.widgets.workers import BaseWorker


class DirectCalibrationWorker(BaseWorker):
    """Runs the sweep off the GUI thread.

    ``BaseWorker`` provides progress/status/error and wraps ``execute()``; it
    has NO stop flag (widgets/workers.py:37-57), so cancellation is this
    worker's own responsibility.
    """

    result_ready = Signal(object)

    def __init__(self, request: DirectCalibrationRequest, parent=None) -> None:
        super().__init__(parent)
        self._request = request
        self._should_stop = False

    def cancel(self) -> None:
        self._should_stop = True

    def execute(self) -> None:
        outcome = run_direct_calibration(
            self._request,
            progress=lambda done, total, label: (
                self.progress.emit(int(100 * done / max(1, total))),
                self.status.emit(f"Measuring {label}"),
            ),
            should_stop=lambda: self._should_stop,
        )
        self.result_ready.emit(outcome)
```

- [ ] **Step 5: Run the tests**

Run: `PYTHONPATH=$PWD/src python -m pytest tests/test_detectkit_direct_calibration_job.py -v`
Expected: PASS (9 tests).

- [ ] **Step 6: Commit**

```bash
make format
git add src/hydra_suite/detectkit/jobs/direct_calibration.py tests/test_detectkit_direct_calibration_job.py
git commit -m "feat(detectkit): cancellable SAHI calibration sweep worker"
```

---

## Task 9: Persist calibration evidence in the project artifact directory

**Files:**
- Modify: `src/hydra_suite/detectkit/jobs/direct_calibration.py`
- Modify: `src/hydra_suite/detectkit/gui/calibration_preview_store.py`
- Test: `tests/test_detectkit_direct_calibration_job.py` (extend)

**Interfaces:**
- Produces: `save_direct_calibration(evidence_dir: Path, outcome, request) -> Path`, `load_direct_calibration(evidence_dir: Path) -> DirectCalibrationOutcome | None`.

- [ ] **Step 1: Write failing tests**

```python
def _scored_point(label="Training geometry"):
    from hydra_suite.core.inference.direct_calibration import (
        CalibrationScore, DirectCalibrationPoint,
    )

    return DirectCalibrationPoint(
        label=label, enabled=True, geometry_mode="auto_object", tile_width=640,
        tile_height=640, overlap=0.2, object_tile_fraction=0.4, max_detections=64,
        tiles_per_frame=9, seconds_per_frame=0.42, confidence=0.35,
        merge_policy="greedy_nmm", merge_metric="ios", merge_threshold=0.5,
        merge_backend="cv2",
        score=CalibrationScore(
            frames=20, matched=200, missed=10, extra=10, duplicate=1,
            precision=0.95, recall=0.95, f1=0.95, mean_iou=0.81,
        ),
    )


def test_evidence_round_trips(tmp_path):
    from hydra_suite.detectkit.jobs.direct_calibration import (
        DirectCalibrationOutcome, load_direct_calibration, save_direct_calibration,
    )

    request = _request(tmp_path)
    save_direct_calibration(
        request.evidence_dir, DirectCalibrationOutcome(points=[_scored_point()]), request
    )
    restored = load_direct_calibration(request.evidence_dir)
    assert restored is not None and restored.partial is False
    assert restored.points[0].score.f1 == 0.95
    assert restored.points[0].max_detections == 64


def test_partial_work_never_overwrites_complete_evidence(tmp_path):
    from hydra_suite.detectkit.jobs.direct_calibration import (
        DirectCalibrationOutcome, load_direct_calibration, save_direct_calibration,
    )

    request = _request(tmp_path)
    save_direct_calibration(
        request.evidence_dir, DirectCalibrationOutcome(points=[_scored_point()]), request
    )
    save_direct_calibration(
        request.evidence_dir, DirectCalibrationOutcome(points=[], partial=True), request
    )
    still = load_direct_calibration(request.evidence_dir)
    assert still.partial is False and still.points
```

- [ ] **Step 2: Run and confirm failure**

Run: `PYTHONPATH=$PWD/src python -m pytest tests/test_detectkit_direct_calibration_job.py -k evidence -v`
Expected: FAIL — `ImportError`.

- [ ] **Step 3: Implement**

Serialize points (dataclass → dict with the `score` flattened) and previews (image path relative to the project dir + polygons as nested lists) to `<evidence_dir>/direct_calibration.json` atomically (`.tmp` + `replace`), plus the request provenance (checkpoint fingerprint, label-set fingerprint, split, runtime tier, max_targets).

```python
def save_direct_calibration(evidence_dir, outcome, request) -> Path:
    """Persist the frontier. A partial run NEVER replaces complete evidence."""
    evidence_dir = Path(evidence_dir)
    evidence_dir.mkdir(parents=True, exist_ok=True)
    target = evidence_dir / "direct_calibration.json"
    existing = load_direct_calibration(evidence_dir)
    if outcome.partial and existing is not None and not existing.partial:
        return target
    ...
```

Generalize `calibration_preview_store.py` by parameterizing its frame payload on the candidate key (SAM3 uses `float | None` tile fractions; direct calibration uses the candidate `label` string) instead of duplicating the module. Keep the SAM3 call sites working.

- [ ] **Step 4: Run the tests**

Run: `PYTHONPATH=$PWD/src python -m pytest tests/test_detectkit_direct_calibration_job.py tests/test_semantic_calibration_preview.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
make format
git add src/hydra_suite/detectkit/jobs/direct_calibration.py src/hydra_suite/detectkit/gui/calibration_preview_store.py tests/test_detectkit_direct_calibration_job.py
git commit -m "feat(detectkit): persist SAHI calibration evidence in the project dir"
```

---

## Task 10: Calibration wizard dialog

**Files:**
- Create: `src/hydra_suite/detectkit/gui/dialogs/direct_calibration_wizard.py`
- Test: `tests/test_detectkit_direct_calibration_ui.py`

**Interfaces:**
- Produces: `class DirectCalibrationWizard(BaseDialog)` with attributes `chk_exhaustive`, `lbl_evidence_summary`, `table_candidates`, `spin_max_targets`, `chk_confirm_broad_sweep`, `btn_run`, `candidates` (list), and methods `evidence() -> EvidenceSet`, `request() -> DirectCalibrationRequest`, `set_calibration_enabled(enabled: bool, reason: str = "")`.

- [ ] **Step 1: Write failing tests**

```python
# tests/test_detectkit_direct_calibration_ui.py
import os
from pathlib import Path

import cv2
import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")
from PySide6.QtWidgets import QApplication

_app = QApplication.instance() or QApplication([])

LABEL_LINE = "0 0.1 0.1 0.2 0.1 0.2 0.2 0.1 0.2\n"


def _dataset(tmp_path: Path, split: str, names: list[str]) -> Path:
    images = tmp_path / "images" / split
    labels = tmp_path / "labels" / split
    images.mkdir(parents=True, exist_ok=True)
    labels.mkdir(parents=True, exist_ok=True)
    for name in names:
        cv2.imwrite(str(images / f"{name}.png"), np.zeros((200, 300, 3), np.uint8))
        (labels / f"{name}.txt").write_text(LABEL_LINE)
    yaml = tmp_path / "data.yaml"
    yaml.write_text(
        f"path: {tmp_path}\ntrain: images/train\nval: images/val\nnames:\n  0: ant\n"
    )
    return yaml


@pytest.fixture
def calibration_wizard(tmp_path):
    from hydra_suite.detectkit.gui.dialogs.direct_calibration_wizard import (
        DirectCalibrationWizard,
    )

    yaml = _dataset(tmp_path, "val", ["rec1_000", "rec1_001"])
    model = tmp_path / "m.pt"
    model.write_bytes(b"weights")
    wizard = DirectCalibrationWizard(
        None,
        model_path=model,
        task="obb",
        dataset_yaml=yaml,
        sources=[],
        training_geometry={"geometry_mode": "auto_object", "imgsz": 640,
                           "object_tile_fraction": 0.4, "overlap": 0.2},
        evidence_dir=tmp_path / "evidence",
    )
    yield wizard
    wizard.close()


def test_run_is_blocked_until_exhaustive_labels_are_acknowledged(calibration_wizard):
    assert calibration_wizard.chk_exhaustive.isChecked() is False
    assert calibration_wizard.btn_run.isEnabled() is False
    calibration_wizard.chk_exhaustive.setChecked(True)
    assert calibration_wizard.btn_run.isEnabled() is True


def test_summary_states_frames_instances_sizes_and_split(calibration_wizard):
    text = calibration_wizard.lbl_evidence_summary.text()
    assert "2 frames" in text and "2 instances" in text
    assert "val" in text and "200" in text


def test_candidate_table_lists_every_candidate_with_its_tile_cost(calibration_wizard):
    table = calibration_wizard.table_candidates
    assert table.rowCount() == len(calibration_wizard.candidates)
    assert table.item(0, 0).text() == "Full frame (no SAHI)"
    assert table.item(0, 1).text() == "1"


def test_detection_cap_is_user_visible_and_reaches_the_request(calibration_wizard):
    calibration_wizard.chk_exhaustive.setChecked(True)
    calibration_wizard.spin_max_targets.setValue(120)
    assert calibration_wizard.request().max_targets == 120


def test_experimental_label_is_shown(calibration_wizard):
    assert "Experimental calibration" in calibration_wizard.windowTitle()
```

- [ ] **Step 2: Run and confirm failure**

Run: `PYTHONPATH=$PWD/src python -m pytest tests/test_detectkit_direct_calibration_ui.py -v`
Expected: FAIL — module missing.

- [ ] **Step 3: Implement**

Build on `hydra_suite.widgets.dialogs.BaseDialog` (read its constructor and button-box contract first). Layout: evidence picker (split combo defaulting to `val` + labelled-source checkboxes), `lbl_evidence_summary` (frames / instances / image-size range / sampling cap / split-fallback note), `spin_max_targets` (default: the largest label count seen on any evidence frame, floor 20; tooltip states that a value below the real animal count caps recall), `chk_exhaustive` wired to `btn_run.setEnabled`, `table_candidates` (label, tiles/frame, total tiles, estimated duration, failure reason), an `Add custom geometry…` row, the confidence/merge grid summary, and `RECOMMENDATION_RULE` printed verbatim. `btn_run` stays disabled while any candidate is over budget unless `chk_confirm_broad_sweep` is ticked. Window title: `SAHI calibration — Experimental calibration`.

- [ ] **Step 4: Run the tests**

Run: `PYTHONPATH=$PWD/src python -m pytest tests/test_detectkit_direct_calibration_ui.py -v`
Expected: PASS (5 tests).

- [ ] **Step 5: Commit**

```bash
make format
git add src/hydra_suite/detectkit/gui/dialogs/direct_calibration_wizard.py tests/test_detectkit_direct_calibration_ui.py
git commit -m "feat(detectkit): SAHI calibration wizard with stated candidate cost"
```

---

## Task 11: Results frontier, overlays, and deferred profile saving

**Files:**
- Create: `src/hydra_suite/detectkit/gui/dialogs/direct_calibration_results.py`
- Test: `tests/test_detectkit_direct_calibration_ui.py` (extend)

**Spec constraint that shapes this task:** "Closing the dialog without confirmation saves nothing." Profiles are therefore staged **in memory** and written to the sidecar only in `accept()`, in one atomic `write_slice_meta`.

**Interfaces:**
- Produces: `class DirectCalibrationResultsDialog(BaseDialog)` with `COL_LABEL = 0`, `COL_STATUS` (declare all `COL_*` constants), `table_rows`, `outcome`, `canvas`, `chk_show_gt`, `chk_show_pred`, `btn_prev_frame`, `btn_next_frame`, `combo_primary`, and methods `settings_for_row(row) -> dict`, `measurement_for_row(row) -> dict`, `save_profile(name, note="", primary=False)`, `remove_profile(profile_id, new_primary_id=None)`, `staged_profiles() -> list[dict]`, `staged_meta() -> dict`, `_render_preview()`, `accept()`.

- [ ] **Step 1: Write failing tests**

```python
@pytest.fixture
def results_dialog(tmp_path):
    from hydra_suite.core.inference.direct_calibration import (
        CalibrationScore, DirectCalibrationPoint,
    )
    from hydra_suite.detectkit.gui.dialogs.direct_calibration_results import (
        DirectCalibrationResultsDialog,
    )
    from hydra_suite.detectkit.jobs.direct_calibration import DirectCalibrationOutcome

    def point(label, f1, failed=""):
        return DirectCalibrationPoint(
            label=label, enabled=label != "Full frame (no SAHI)",
            geometry_mode="auto_object", tile_width=640, tile_height=640,
            overlap=0.2, object_tile_fraction=0.4, max_detections=64,
            tiles_per_frame=9, seconds_per_frame=0.4, confidence=0.35,
            merge_policy="greedy_nmm", merge_metric="ios", merge_threshold=0.5,
            merge_backend="cv2", failed_reason=failed,
            score=CalibrationScore(
                frames=20, matched=200, missed=10, extra=10, duplicate=1,
                precision=f1, recall=f1, f1=f1, mean_iou=0.8,
            ),
        )

    model = tmp_path / "m.pt"
    model.write_bytes(b"weights")
    outcome = DirectCalibrationOutcome(
        points=[
            point("Full frame (no SAHI)", 0.70),
            point("Training geometry", 0.92),
            point("fraction x1.5, overlap 0.1", 0.90),
            point("Custom 1x1", 0.0, failed="tile budget exceeded"),
        ]
    )
    dialog = DirectCalibrationResultsDialog(
        None, model_path=model, outcome=outcome,
        training_geometry={"geometry_mode": "auto_object", "imgsz": 640},
        previews=[],
    )
    yield dialog
    dialog.close()


def test_every_measured_row_including_failures_is_listed(results_dialog):
    dialog = results_dialog
    assert dialog.table_rows.rowCount() == len(dialog.outcome.points)
    statuses = [
        dialog.table_rows.item(i, dialog.COL_STATUS).text()
        for i in range(dialog.table_rows.rowCount())
    ]
    assert any("tile budget" in text for text in statuses)
    assert any("Recommended" in text for text in statuses)


def test_changing_rows_never_runs_the_model(results_dialog, monkeypatch):
    import hydra_suite.core.inference.stages.obb as obb_stage

    def explode(*_a, **_k):
        raise AssertionError("selecting a row must not run the model")

    monkeypatch.setattr(obb_stage, "run_obb", explode)
    monkeypatch.setattr(obb_stage, "collect_obb_parts_by_frame", explode)
    results_dialog.table_rows.setCurrentCell(1, 0)
    results_dialog._render_preview()


def test_nothing_touches_the_sidecar_until_accept(results_dialog, tmp_path):
    from hydra_suite.core.inference.slice_meta import sidecar_path

    results_dialog.table_rows.setCurrentCell(1, 0)
    results_dialog.save_profile("Balanced", note="Routine tracking", primary=True)
    assert results_dialog.staged_profiles(), "profile should be staged in memory"
    assert not sidecar_path(tmp_path / "m.pt").exists(), "sidecar written too early"
    results_dialog.accept()
    assert sidecar_path(tmp_path / "m.pt").exists()


def test_rejecting_the_dialog_saves_nothing(results_dialog, tmp_path):
    from hydra_suite.core.inference.slice_meta import sidecar_path

    results_dialog.table_rows.setCurrentCell(1, 0)
    results_dialog.save_profile("Balanced")
    results_dialog.reject()
    assert not sidecar_path(tmp_path / "m.pt").exists()


def test_several_profiles_from_one_run_live_on_one_artifact(results_dialog):
    results_dialog.table_rows.setCurrentCell(1, 0)
    results_dialog.save_profile("Balanced", primary=True)
    results_dialog.table_rows.setCurrentCell(2, 0)
    results_dialog.save_profile("High recall")
    assert [p["name"] for p in results_dialog.staged_profiles()] == [
        "Balanced", "High recall"
    ]
    meta = results_dialog.staged_meta()
    assert meta["primary_profile_id"] == meta["profiles"][0]["id"]


def test_duplicate_profile_names_are_rejected(results_dialog):
    results_dialog.table_rows.setCurrentCell(1, 0)
    results_dialog.save_profile("Balanced")
    with pytest.raises(ValueError):
        results_dialog.save_profile("balanced")


def test_a_failed_row_cannot_become_a_profile(results_dialog):
    failed_row = next(
        i for i, p in enumerate(results_dialog.outcome.points) if p.failed_reason
    )
    results_dialog.table_rows.setCurrentCell(failed_row, 0)
    with pytest.raises(ValueError, match="failed"):
        results_dialog.save_profile("Broken")


def test_settings_payload_is_complete_and_omits_reference_body_size(results_dialog):
    settings = results_dialog.settings_for_row(1)
    assert "REFERENCE_BODY_SIZE" not in settings
    for key in (
        "enabled", "geometry_mode", "slice_width", "slice_height", "overlap",
        "object_tile_fraction", "trained_body_px", "confidence_threshold",
        "merge_policy", "merge_metric", "merge_threshold", "merge_backend",
        "max_detections",
    ):
        assert key in settings


def test_measurement_records_provenance(results_dialog):
    measurement = results_dialog.measurement_for_row(1)
    for key in (
        "created_at", "checkpoint_fingerprint", "task", "frames", "instances",
        "runtime", "seconds_per_frame", "precision", "recall", "f1",
        "localization_quality", "max_detections",
    ):
        assert key in measurement
    assert measurement["checkpoint_fingerprint"].startswith("sha256:")


def test_removing_the_primary_profile_prompts_for_a_replacement(results_dialog):
    dialog = results_dialog
    dialog.table_rows.setCurrentCell(1, 0)
    dialog.save_profile("Balanced", primary=True)
    dialog.table_rows.setCurrentCell(2, 0)
    dialog.save_profile("High recall")
    staged = dialog.staged_profiles()
    with pytest.raises(ValueError, match="replacement"):
        dialog.remove_profile(staged[0]["id"])
    dialog.remove_profile(staged[0]["id"], new_primary_id=staged[1]["id"])
    assert [p["name"] for p in dialog.staged_profiles()] == ["High recall"]
```

- [ ] **Step 2: Run and confirm failure**

Run: `PYTHONPATH=$PWD/src python -m pytest tests/test_detectkit_direct_calibration_ui.py -v`
Expected: FAIL — module missing.

- [ ] **Step 3: Implement**

Columns (declare each as a `COL_*` constant): label, full-frame flag, tile size, overlap, confidence, merge policy/metric/threshold, detection cap, tiles/frame, s/frame, projected duration, matched, missed, extra, duplicate, precision, recall, F1, localization quality, frames+instances, status. Status is `""`, `Recommended`, or the failure reason. Print `RECOMMENDATION_RULE` verbatim under the table.

State: `self._staged_meta` starts as `read_slice_meta(model_path) or {"training_geometry": training_geometry}`; `save_profile` calls `upsert_slice_profile` on it (raising when the selected row has a `failed_reason`), `remove_profile` calls `remove_slice_profile`; `accept()` performs the single `write_slice_meta(self._model_path, self._staged_meta)` and then `super().accept()`. `reject()` does nothing but close.

`settings_for_row` returns exactly the keys the test asserts, with `max_detections` from the point; `measurement_for_row` stamps `checkpoint_fingerprint(self._model_path)`. Overlay rendering uses only stored preview polygons through `OBBCanvas` + `_overlay_helpers`.

- [ ] **Step 4: Run the tests**

Run: `PYTHONPATH=$PWD/src python -m pytest tests/test_detectkit_direct_calibration_ui.py -v`
Expected: PASS (15 tests).

- [ ] **Step 5: Commit**

```bash
make format
git add src/hydra_suite/detectkit/gui/dialogs/direct_calibration_results.py tests/test_detectkit_direct_calibration_ui.py
git commit -m "feat(detectkit): calibration frontier with overlays and deferred profile saving"
```

---

## Task 12: Entry points — Review & Register, Run History, and a menu action

**Files:**
- Modify: `src/hydra_suite/detectkit/gui/dialogs/training_dialog.py`
- Modify: `src/hydra_suite/detectkit/gui/dialogs/history_dialog.py`
- Modify: `src/hydra_suite/detectkit/gui/main_window.py`
- Modify: `src/hydra_suite/detectkit/gui/dialogs/direct_calibration_wizard.py`
- Test: `tests/test_detectkit_direct_calibration_ui.py` (extend)

**Note:** DetectKit has **no registered-model list page** (`detectkit/gui/models.py` is a data module). The "calibrate an already registered model" path is therefore a main-window menu action with a model picker, not a row action.

**Interfaces:**
- Produces: `open_direct_calibration(parent, *, model_path, task, dataset_yaml, sources, training_geometry, evidence_dir) -> list[dict]` in `direct_calibration_wizard.py` — the single launcher all three entry points call; returns saved profiles (empty on cancel).
- Produces: `TrainingDialog.register_with_training_geometry()`, `TrainingDialog.calibrate_then_register()`, `DetectKitMainWindow.calibrate_registered_model(model_path)`.

- [ ] **Step 1: Write failing tests**

```python
def test_register_with_training_geometry_skips_calibration(monkeypatch, training_dialog):
    calls = []
    monkeypatch.setattr(
        "hydra_suite.detectkit.gui.dialogs.training_dialog.open_direct_calibration",
        lambda *a, **k: (calls.append(k), [])[1],
    )
    training_dialog.register_with_training_geometry()
    assert calls == []
    assert len(training_dialog.registered_model_paths) == 1


def test_calibrate_then_register_produces_one_artifact(monkeypatch, training_dialog):
    monkeypatch.setattr(
        "hydra_suite.detectkit.gui.dialogs.training_dialog.open_direct_calibration",
        lambda *a, **k: [{"id": "balanced-1", "name": "Balanced"}],
    )
    training_dialog.calibrate_then_register()
    assert len(training_dialog.registered_model_paths) == 1


def test_calibration_is_disabled_with_a_reason_when_labels_are_missing(training_dialog):
    training_dialog.set_calibration_enabled(False, "no labelled val split")
    assert training_dialog.btn_calibrate.isEnabled() is False
    assert "no labelled val split" in training_dialog.btn_calibrate.toolTip()
```

Define the `training_dialog` fixture in the test file by constructing the real dialog the way existing DetectKit dialog tests do (grep `TrainingDialog(` in `tests/` for the established construction); if no test constructs it, build a minimal project via `DetectKitProject` and pass it.

- [ ] **Step 2: Run and confirm failure**

Run: `PYTHONPATH=$PWD/src python -m pytest tests/test_detectkit_direct_calibration_ui.py -k register -v`
Expected: FAIL — methods missing.

- [ ] **Step 3: Implement**

`training_dialog.py`: the post-training review gets `btn_register` ("Register with training geometry", today's path untouched) and `btn_calibrate` ("Calibrate for TrackerKit…"), the latter opening the wizard/results pair and then registering the same artifact once. `set_calibration_enabled(enabled, reason)` mirrors `SemanticEscalationDialog.set_calibration_enabled` (`semantic_escalation_dialog.py:678`). `history_dialog.py`: the same action for a completed direct-detector run. `main_window.py`: a Tools menu action "Calibrate a model for TrackerKit…" that picks a `.pt` under `get_models_dir()` and calls the launcher.

- [ ] **Step 4: Run the tests**

Run: `PYTHONPATH=$PWD/src python -m pytest tests/test_detectkit_direct_calibration_ui.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
make format
git add src/hydra_suite/detectkit/gui/dialogs/training_dialog.py src/hydra_suite/detectkit/gui/dialogs/history_dialog.py src/hydra_suite/detectkit/gui/main_window.py src/hydra_suite/detectkit/gui/dialogs/direct_calibration_wizard.py tests/test_detectkit_direct_calibration_ui.py
git commit -m "feat(detectkit): calibrate from Review & Register, history and the Tools menu"
```

---

## Task 13: Publish must preserve profiles, and the registry summary must agree

**Files:**
- Modify: `src/hydra_suite/training/model_publish.py:851-868`
- Modify: `src/hydra_suite/core/inference/slice_meta.py`
- Test: `tests/test_model_publish_slice_geometry.py` (extend)

`model_publish.py:851-868` writes a **fresh** `normalized_slice_meta(slice_geometry)` at the destination and never copies the source sidecar, so profiles saved before registration are destroyed.

**Interfaces:**
- Produces: `profile_summary(meta) -> {"count": int, "primary_profile_id": str, "names": [str, ...]}` in `slice_meta.py`.
- Produces: `merge_training_geometry(existing_meta, training_geometry) -> dict` in `slice_meta.py` — replaces `training_geometry`, preserves `profiles` and `primary_profile_id`.
- Produces: `verify_profile_summary(model_path, recorded: dict) -> None` in `model_publish.py`, raising `RuntimeError` on disagreement. It must call `slice_meta.profile_summary` through a **module-level import** so tests can monkeypatch it.

- [ ] **Step 1: Write failing tests**

```python
def test_publishing_preserves_profiles_saved_before_registration(tmp_path):
    """Calibrate-then-register must not destroy the user's profiles."""
    from hydra_suite.core.inference.slice_meta import (
        available_slice_profiles, merge_training_geometry, upsert_slice_profile,
    )

    existing = upsert_slice_profile(
        {"geometry_mode": "auto_object", "imgsz": 640},
        name="Balanced", settings={"enabled": True}, primary=True,
    )
    merged = merge_training_geometry(
        existing, {"geometry_mode": "auto_object", "imgsz": 1024, "overlap": 0.3}
    )
    assert merged["training_geometry"]["imgsz"] == 1024
    assert [p["name"] for p in available_slice_profiles(merged)] == ["Balanced"]
    assert merged["primary_profile_id"] == existing["primary_profile_id"]


def test_registry_records_the_sidecar_profile_summary():
    from hydra_suite.core.inference.slice_meta import profile_summary, upsert_slice_profile

    meta = upsert_slice_profile(
        {"geometry_mode": "auto_object", "imgsz": 640},
        name="Balanced", settings={"enabled": True}, primary=True,
    )
    assert profile_summary(meta) == {
        "count": 1,
        "primary_profile_id": meta["primary_profile_id"],
        "names": ["Balanced"],
    }


def test_disagreement_between_registry_and_sidecar_raises(tmp_path, monkeypatch):
    """A second source of truth must fail loudly, not drift."""
    import hydra_suite.training.model_publish as publish

    monkeypatch.setattr(
        publish, "profile_summary",
        lambda meta: {"count": 99, "primary_profile_id": "x", "names": []},
    )
    with pytest.raises(RuntimeError, match="profile summary"):
        publish.verify_profile_summary(
            tmp_path / "m.pt",
            {"count": 0, "primary_profile_id": "", "names": []},
        )
```

- [ ] **Step 2: Run and confirm failure**

Run: `PYTHONPATH=$PWD/src python -m pytest tests/test_model_publish_slice_geometry.py -v`
Expected: FAIL — `merge_training_geometry` / `profile_summary` / `verify_profile_summary` missing.

- [ ] **Step 3: Implement**

In `slice_meta.py`:

```python
def profile_summary(meta: dict[str, Any]) -> dict[str, Any]:
    """Inventory summary the registry stores; the sidecar stays the source of truth."""
    profiles = available_slice_profiles(meta)
    return {
        "count": len(profiles),
        "primary_profile_id": str(meta.get("primary_profile_id", "") or ""),
        "names": [profile["name"] for profile in profiles],
    }


def merge_training_geometry(
    existing: dict[str, Any] | None, training_geometry: dict[str, Any]
) -> dict[str, Any]:
    """Replace training geometry while preserving user-approved profiles.

    Publishing must never destroy calibration a user did before registering.
    """
    result = normalized_slice_meta(existing or {})
    result["training_geometry"] = dict(training_geometry)
    return result
```

In `model_publish.py`, import `profile_summary`, `merge_training_geometry`, `read_slice_meta`, `sidecar_path` at module level, then replace the write block:

```python
        source_meta = read_slice_meta(src)  # profiles saved before registration
        merged = merge_training_geometry(source_meta, dict(slice_geometry))
        slice_sidecar = dst.with_suffix(dst.suffix + ".slice_meta.json")
        slice_sidecar.write_text(json.dumps(merged, indent=2), encoding="utf-8")
        slice_geom_sidecar_name = slice_sidecar.name
```

and after the metadata dict is built:

```python
    if slice_geometry and role in _DIRECT_DETECTOR_ROLES:
        metadata["slice_geometry"] = dict(slice_geometry)
        metadata["slice_profiles"] = profile_summary(merged)
        if slice_geom_sidecar_name:
            metadata["slice_meta_sidecar"] = slice_geom_sidecar_name
        verify_profile_summary(dst, metadata["slice_profiles"])
```

with

```python
def verify_profile_summary(model_path, recorded: dict) -> None:
    """Fail loudly when the registry and the sidecar disagree about profiles.

    ``profile_summary`` is imported at module level on purpose: a function-local
    import would make this unpatchable and the guard untestable.
    """
    actual = profile_summary(read_slice_meta(model_path) or {})
    if actual != recorded:
        raise RuntimeError(
            f"Registry profile summary {recorded} disagrees with the sidecar "
            f"{actual} for {model_path}."
        )
```

Also add a note in the DetectKit results dialog docstring: a later calibration edits the sidecar without touching the registry, so the registry summary is refreshed on the next publish; document that the sidecar wins on any read.

- [ ] **Step 4: Run the tests**

Run: `PYTHONPATH=$PWD/src python -m pytest tests/test_model_publish_slice_geometry.py tests/test_service_publish_slice_geometry.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
make format
git add src/hydra_suite/training/model_publish.py src/hydra_suite/core/inference/slice_meta.py tests/test_model_publish_slice_geometry.py
git commit -m "fix(publish): preserve calibrated profiles and assert registry agreement"
```

---

## Task 14: TrackerKit — stale-evidence detection and visible fallback

**Files:**
- Modify: `src/hydra_suite/core/inference/slice_meta.py`
- Modify: `src/hydra_suite/trackerkit/gui/panels/detection_panel.py:2660-2700`
- Test: `tests/test_trackerkit_profile_session.py`

**Interfaces:**
- Produces: `profile_evidence_state(profile, *, checkpoint_path) -> tuple[bool, str]` in `slice_meta.py`.
- Produces: `DetectionPanel.slice_profile_status_text() -> str` and `lbl_slice_profile_status`.

- [ ] **Step 1: Write failing tests**

```python
# tests/test_trackerkit_profile_session.py
import hashlib
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from hydra_suite.core.inference.slice_meta import profile_evidence_state


def _profile(digest):
    return {"id": "a", "name": "Balanced", "settings": {},
            "measurement": {"checkpoint_fingerprint": digest}}


def test_replaced_weights_invalidate_profile_evidence(tmp_path):
    checkpoint = tmp_path / "m.pt"
    checkpoint.write_bytes(b"weights")
    digest = "sha256:" + hashlib.sha256(b"weights").hexdigest()
    fresh, reason = profile_evidence_state(_profile(digest), checkpoint_path=checkpoint)
    assert fresh is True and reason == ""
    checkpoint.write_bytes(b"retrained")
    fresh, reason = profile_evidence_state(_profile(digest), checkpoint_path=checkpoint)
    assert fresh is False and "weights changed" in reason


def test_missing_provenance_is_not_fatal(tmp_path):
    checkpoint = tmp_path / "m.pt"
    checkpoint.write_bytes(b"weights")
    fresh, reason = profile_evidence_state(
        {"id": "a", "name": "n", "settings": {}, "measurement": {}},
        checkpoint_path=checkpoint,
    )
    assert fresh is True and reason == ""


def test_unreadable_checkpoint_is_not_fatal(tmp_path):
    fresh, reason = profile_evidence_state(
        _profile("sha256:deadbeef"), checkpoint_path=tmp_path / "absent.pt"
    )
    assert fresh is True and reason == ""


def test_unknown_saved_profile_falls_back_visibly(monkeypatch):
    from tests.test_main_window_config_persistence import _make_main_window

    window = _make_main_window(monkeypatch)
    panel = window._detection_panel
    panel._slice_meta = {
        "schema_version": 2,
        "training_geometry": {"geometry_mode": "auto_object", "imgsz": 640},
        "primary_profile_id": "",
        "profiles": [],
    }
    window.advanced_config["slice_profile_id"] = "gone-1234"
    panel._apply_slice_meta_values("gone-1234")
    status = panel.slice_profile_status_text()
    assert "Training geometry" in status and "no longer" in status
    window.close()
```

- [ ] **Step 2: Run and confirm failure**

Run: `PYTHONPATH=$PWD/src python -m pytest tests/test_trackerkit_profile_session.py -v`
Expected: FAIL — `profile_evidence_state` missing.

- [ ] **Step 3: Implement**

```python
def profile_evidence_state(
    profile: dict[str, Any], *, checkpoint_path
) -> tuple[bool, str]:
    """Is this profile's measured evidence still about THESE weights?

    Missing or unreadable provenance is not fatal -- an imported or legacy
    profile simply has nothing to contradict. A fingerprint that DISAGREES is:
    applying settings measured on other weights silently misdescribes the
    operating point.
    """
    recorded = str((profile.get("measurement") or {}).get("checkpoint_fingerprint", ""))
    if not recorded:
        return True, ""
    try:
        digest = hashlib.sha256(Path(checkpoint_path).read_bytes()).hexdigest()
    except Exception:
        return True, ""
    if recorded.split(":")[-1] == digest:
        return True, ""
    return False, (
        f"'{profile.get('name', 'profile')}' was measured before the model's "
        "weights changed; its numbers no longer describe this checkpoint."
    )
```

Add `import hashlib` and `from pathlib import Path` if absent. In the panel, add `lbl_slice_profile_status` under the combo and set it for three cases: stale evidence (state the reason), unknown saved id (`The saved profile is no longer in this model's sidecar; using Training geometry.`), manual edit (`Custom (based on <name>)`).

- [ ] **Step 4: Run the tests**

Run: `PYTHONPATH=$PWD/src python -m pytest tests/test_trackerkit_profile_session.py tests/test_detection_panel_slice_widgets.py tests/test_trackerkit_slice_meta_prefill.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
make format
git add src/hydra_suite/core/inference/slice_meta.py src/hydra_suite/trackerkit/gui/panels/detection_panel.py tests/test_trackerkit_profile_session.py
git commit -m "feat(trackerkit): flag stale profile evidence and explain fallbacks"
```

---

## Task 15: Session round-trip and model-switch isolation

**Files:**
- Modify: `src/hydra_suite/trackerkit/gui/orchestrators/config.py:441,1660`
- Modify: `src/hydra_suite/trackerkit/gui/panels/detection_panel.py` (`apply_slice_meta_for_model`)
- Test: `tests/test_trackerkit_profile_session.py` (extend)

- [ ] **Step 1: Write failing tests**

```python
def test_session_restores_the_saved_profile_not_a_changed_primary(monkeypatch):
    from hydra_suite.core.inference.slice_meta import upsert_slice_profile
    from tests.test_main_window_config_persistence import _make_main_window

    meta = upsert_slice_profile(
        {"geometry_mode": "auto_object", "imgsz": 640},
        name="Balanced",
        settings={"enabled": True, "geometry_mode": "auto_object", "overlap": 0.2,
                  "object_tile_fraction": 0.4},
        primary=True,
    )
    meta = upsert_slice_profile(
        meta, name="Fast scan",
        settings={"enabled": True, "geometry_mode": "auto_object", "overlap": 0.1,
                  "object_tile_fraction": 0.6},
        primary=True,   # primary later moved to Fast scan
    )
    balanced = next(p for p in meta["profiles"] if p["name"] == "Balanced")

    window = _make_main_window(monkeypatch)
    panel = window._detection_panel
    panel._slice_meta = meta
    window.advanced_config["slice_profile_id"] = balanced["id"]
    panel._apply_slice_meta_values(balanced["id"])
    assert window.advanced_config["slice_object_tile_fraction"] == 0.4
    assert window.advanced_config["slice_profile_id"] == balanced["id"]
    window.close()


def test_switching_models_does_not_carry_profile_settings_over(monkeypatch, tmp_path):
    from tests.test_main_window_config_persistence import _make_main_window

    window = _make_main_window(monkeypatch)
    panel = window._detection_panel
    window.advanced_config["slice_profile_id"] = "stale-1234"
    panel.apply_slice_meta_for_model(str(tmp_path / "other.pt"))  # no sidecar
    assert window.advanced_config.get("slice_profile_id", "") in ("", "__training__")
    window.close()
```

- [ ] **Step 2: Run and confirm failure**

Run: `PYTHONPATH=$PWD/src python -m pytest tests/test_trackerkit_profile_session.py -v`
Expected: FAIL on the second test — the no-sidecar early return leaves `slice_profile_id` set.

- [ ] **Step 3: Implement**

Clear `advanced_config["slice_profile_id"]` on `apply_slice_meta_for_model`'s no-sidecar early return and whenever the model path changes. In `orchestrators/config.py`, persist the effective slice settings alongside the id and resolve on restore in this order: saved id if still valid → saved effective settings if still valid → primary → training geometry, setting the status text at each fallback.

- [ ] **Step 4: Run the tests**

Run: `PYTHONPATH=$PWD/src python -m pytest tests/test_trackerkit_profile_session.py tests/test_main_window_config_persistence.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
make format
git add src/hydra_suite/trackerkit/gui/orchestrators/config.py src/hydra_suite/trackerkit/gui/panels/detection_panel.py tests/test_trackerkit_profile_session.py
git commit -m "feat(trackerkit): round-trip SAHI profile and effective settings in sessions"
```

---

## Task 16: End-to-end — profiles through the real params path into cache keys

**Files:**
- Test: `tests/test_profile_cache_keys.py`

This must exercise the **real** path — panel values → `advanced_config` → `build_engine_params` → `_slice_config_from_params` → `_slice_config_hash` — not a hand-built `SliceConfig`, or a mismatch anywhere on that chain would go unnoticed.

- [ ] **Step 1: Find the real seam**

```bash
grep -n "SLICE_" src/hydra_suite/trackerkit/*/engine_params.py src/hydra_suite/**/engine_params.py 2>/dev/null | head -20
grep -rn "def build_engine_params" -A5 src/hydra_suite | head
```
Record the module path and the `advanced_config` key names it reads (`slice_overlap`, `slice_object_tile_fraction`, `slice_merge_threshold`, …). The test below imports that function by its real name.

- [ ] **Step 2: Write the failing test**

```python
# tests/test_profile_cache_keys.py
"""Profiles must split the detection cache exactly where they differ.

Confidence is DELIBERATELY excluded from the key (cache/keys.py:100 -- it is
re-applied at tracking time over raw detections), so two profiles differing
only in confidence correctly share detections. Geometry, overlap and merge
settings change which raw detections exist and must not.
"""

from hydra_suite.core.inference.cache.keys import _slice_config_hash
from hydra_suite.core.inference.config import _slice_config_from_params
from hydra_suite.core.inference.slice_meta import (
    slice_meta_to_panel_values,
    upsert_slice_profile,
)

TRAINING = {"geometry_mode": "auto_object", "imgsz": 640, "overlap": 0.2}
BASE_SETTINGS = {
    "enabled": True, "geometry_mode": "auto_object", "slice_width": 0,
    "slice_height": 0, "overlap": 0.2, "object_tile_fraction": 0.4,
    "trained_body_px": 560.0, "confidence_threshold": 0.35,
    "merge_policy": "greedy_nmm", "merge_metric": "ios", "merge_threshold": 0.5,
    "merge_backend": "cv2",
}


def _hash_for(**overrides) -> str:
    """Panel values -> SLICE_* params -> SliceConfig -> cache hash (the real path)."""
    meta = upsert_slice_profile(
        TRAINING, name="P", settings=dict(BASE_SETTINGS, **overrides)
    )
    values = slice_meta_to_panel_values(meta, meta["profiles"][0]["id"])
    params = {
        "SLICE_ENABLED": values["enabled"],
        "SLICE_GEOMETRY_MODE": values["geometry_mode"],
        "SLICE_WIDTH": values["slice_width"],
        "SLICE_HEIGHT": values["slice_height"],
        "SLICE_OVERLAP": values["overlap"],
        "SLICE_OBJECT_TILE_FRACTION": values["object_tile_fraction"],
        "SLICE_MERGE_POLICY": values["merge_policy"] or "greedy_nmm",
        "SLICE_MERGE_METRIC": values["merge_metric"] or "ios",
        "SLICE_MERGE_THRESHOLD": values["merge_threshold"] or 0.5,
        "SLICE_MERGE_BACKEND": values["merge_backend"] or "cv2",
    }
    slice_cfg = _slice_config_from_params(
        params, "SLICE_", reference_body_px=values["trained_body_px"]
    )
    return _slice_config_hash(slice_cfg)


def test_geometry_difference_splits_the_detection_cache():
    assert _hash_for() != _hash_for(object_tile_fraction=0.7, overlap=0.1)


def test_merge_difference_splits_the_detection_cache():
    assert _hash_for() != _hash_for(merge_threshold=0.8)
    assert _hash_for() != _hash_for(merge_policy="nmm")


def test_confidence_only_difference_shares_the_cache_by_design():
    assert _hash_for() == _hash_for(confidence_threshold=0.15)


def test_every_profile_field_survives_the_panel_translation():
    meta = upsert_slice_profile(TRAINING, name="P", settings=BASE_SETTINGS)
    values = slice_meta_to_panel_values(meta, meta["profiles"][0]["id"])
    for key in ("merge_policy", "merge_metric", "merge_threshold", "merge_backend",
                "confidence_threshold"):
        assert values[key] is not None, f"{key} dropped in translation"


def test_two_profiles_live_on_one_artifact():
    meta = upsert_slice_profile(TRAINING, name="Balanced", settings=BASE_SETTINGS)
    meta = upsert_slice_profile(
        meta, name="Fast scan", settings=dict(BASE_SETTINGS, object_tile_fraction=0.7)
    )
    assert len(meta["profiles"]) == 2 and meta["schema_version"] == 2
```

- [ ] **Step 3: Run and fix any dropped field**

Run: `PYTHONPATH=$PWD/src python -m pytest tests/test_profile_cache_keys.py -v`
Any failure means `slice_meta_to_panel_values` drops a field the cache key needs — fix `slice_meta.py`, not the test.

- [ ] **Step 4: Replace the params dict with the real builder if one exists**

If Step 1 found a `build_engine_params`-style function that maps `advanced_config` → params, rewrite `_hash_for` to go through it (setting `advanced_config` keys from `values`) so the test covers the production translation rather than a local copy of it. Note in a comment which function it is.

- [ ] **Step 5: Commit**

```bash
make format
git add tests/test_profile_cache_keys.py src/hydra_suite/core/inference/slice_meta.py
git commit -m "test(profiles): pin which profile differences split the detection cache"
```

---

## Task 17: Detect and segment evaluation, docs, and the experimental label

**Files:**
- Modify: `src/hydra_suite/core/inference/direct_calibration.py`
- Modify: `src/hydra_suite/detectkit/jobs/direct_calibration.py` (thread `task`)
- Create: `docs/user-guide/detectkit-sahi-calibration.md`
- Modify: `mkdocs.yml`
- Test: `tests/test_direct_calibration.py` (extend)

**Interfaces:**
- Produces: `match_frame(..., task: str = "obb")` and `score_frames(..., task: str = "obb")`. `obb`/`segment` use polygon IoU; `detect` compares axis-aligned bounding boxes of both sides.

- [ ] **Step 1: Write failing tests**

```python
def test_axis_aligned_matching_counts_crowded_boxes_one_to_one():
    import numpy as np
    from hydra_suite.core.inference.direct_calibration import (
        CalibrationDetection, match_frame,
    )

    def box(x, y, w=10, h=10):
        return np.array([[x, y], [x + w, y], [x + w, y + h], [x, y + h]], np.float32)

    labels = [CalibrationDetection(0, box(0, 0)), CalibrationDetection(0, box(30, 0))]
    predictions = [
        CalibrationDetection(0, box(0, 0)),
        CalibrationDetection(0, box(1, 1)),      # duplicate on label 0
        CalibrationDetection(0, box(100, 100)),  # extra
    ]
    score = match_frame(predictions, labels, task="detect")
    assert score.matched == 1
    assert score.missed == 1
    assert score.extra == 2
    assert score.duplicate == 1


def test_segment_polygons_match_on_mask_overlap():
    import numpy as np
    from hydra_suite.core.inference.direct_calibration import (
        CalibrationDetection, match_frame,
    )

    triangle = np.array([[0, 0], [20, 0], [10, 20]], np.float32)
    score = match_frame(
        [CalibrationDetection(0, triangle)], [CalibrationDetection(0, triangle)],
        task="segment",
    )
    assert score.matched == 1 and score.mean_iou > 0.99


def test_rotated_prediction_is_scored_as_its_aabb_under_detect():
    """A detect model cannot express rotation; scoring must not credit it."""
    import numpy as np
    from hydra_suite.core.inference.direct_calibration import (
        CalibrationDetection, match_frame,
    )

    rotated = np.array([[10, 0], [20, 10], [10, 20], [0, 10]], np.float32)
    aabb = np.array([[0, 0], [20, 0], [20, 20], [0, 20]], np.float32)
    obb_score = match_frame(
        [CalibrationDetection(0, rotated)], [CalibrationDetection(0, aabb)], task="obb"
    )
    detect_score = match_frame(
        [CalibrationDetection(0, rotated)], [CalibrationDetection(0, aabb)],
        task="detect",
    )
    assert detect_score.mean_iou > obb_score.mean_iou
```

- [ ] **Step 2: Run and confirm failure**

Run: `PYTHONPATH=$PWD/src python -m pytest tests/test_direct_calibration.py -k "axis_aligned or segment or rotated" -v`
Expected: FAIL — `match_frame() got an unexpected keyword argument 'task'`.

- [ ] **Step 3: Implement**

Add `task` to `match_frame`/`score_frames`. For `detect`, convert each polygon to its axis-aligned bounding quad before the IoU call:

```python
def _as_task_polygon(polygon: np.ndarray, task: str) -> np.ndarray:
    """detect models cannot express rotation -- score them as AABBs on both sides."""
    if task != "detect":
        return polygon
    x0, y0 = polygon.min(axis=0)
    x1, y1 = polygon.max(axis=0)
    return np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], dtype=np.float32)
```

Document the 0.5 IoU floor in the docstring. Thread `request.task` from the job into `score_frames`.

- [ ] **Step 4: Write the user guide**

`docs/user-guide/detectkit-sahi-calibration.md` covers: what calibration measures and what it never touches (weights, `REFERENCE_BODY_SIZE`); the exhaustive-label requirement; why `val` is the default; the candidate grid and its cost estimate; the detection cap and why a too-low value caps recall; how to read the frontier columns; `RECOMMENDATION_RULE` verbatim; saving/naming/primary/removal; TrackerKit application and `Custom (based on <name>)`; the `Experimental calibration` caveat; and that timings are measurements on that machine and data, not portable guarantees. Add it to `mkdocs.yml` nav.

- [ ] **Step 5: Run tests, lint and docs**

Run: `PYTHONPATH=$PWD/src python -m pytest tests/test_direct_calibration.py -v && make lint && make docs-check`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
make format
git add src/hydra_suite/core/inference/direct_calibration.py src/hydra_suite/detectkit/jobs/direct_calibration.py docs/user-guide/detectkit-sahi-calibration.md mkdocs.yml tests/test_direct_calibration.py
git commit -m "feat(calibration): detect/segment evaluation plus user documentation"
```

---

## Task 18: Full verification gate and merge

- [ ] **Step 1: Run every touched test file**

```bash
PYTHONPATH=$PWD/src python -m pytest \
  tests/test_direct_calibration.py tests/test_direct_calibration_grid.py \
  tests/test_direct_calibration_sweep.py tests/test_direct_calibration_parity.py \
  tests/test_slice_meta_read.py tests/test_slice_profile_mutations.py \
  tests/test_detectkit_direct_calibration_job.py tests/test_detectkit_direct_calibration_ui.py \
  tests/test_trackerkit_profile_session.py tests/test_profile_cache_keys.py \
  tests/test_trackerkit_slice_meta_prefill.py tests/test_detection_panel_slice_widgets.py \
  tests/test_model_publish_slice_geometry.py tests/test_service_publish_slice_geometry.py \
  tests/test_semantic_calibration_preview.py tests/test_slice_geometry_parity.py \
  tests/test_detectkit_sliced_preview.py -q
```
Expected: all PASS. Never run `pytest tests/` whole-suite — classkit modal dialogs hang it; batch per file.

- [ ] **Step 2: Delta gate against main**

Run the same command from a clean `main` worktree and compare **failure sets**, not counts (new files shift chunk boundaries). Expected: no branch-only failure.

- [ ] **Step 3: Kill stale sleap/hydra processes, then the MPS matrix**

```bash
pgrep -fl "sleap|hydra" | grep -v grep    # kill ONLY stale sleap/hydra
conda activate hydra-mps
find . -name '__pycache__' -prune -exec rm -rf {} +
git worktree add --detach .worktrees/equiv-legacy legacy/main
REPO=$PWD WT=$PWD MAIN_SRC=$PWD/.worktrees/equiv-legacy/src WT_SRC=$PWD/src \
  OUT=/tmp/equiv_sahi_calib RUNTIME=mps bash tools/equivalence/run_matrix.sh
wc -l /tmp/equiv_sahi_calib/**/*.csv   # every CSV MUST have > 1 row
```
Expected: EQUIVALENCE at the DETERMINISM floor on every clip (head/tail π-flips are the documented noise floor); PERFORMANCE ratio ≤ 1.25.

- [ ] **Step 4: CUDA matrix on mehek**

```bash
ssh rutalab@mehek.taild08eb9.ts.net
cd ~/hydra-suite && git fetch origin && git checkout <branch-sha>
source ~/mambaforge/etc/profile.d/conda.sh && conda activate hydra-cuda
git worktree add --detach .worktrees/equiv-legacy legacy/main
REPO=$PWD WT=$PWD MAIN_SRC=$PWD/.worktrees/equiv-legacy/src WT_SRC=$PWD/src \
  OUT=/tmp/equiv_gen2 RUNTIME=cuda nohup bash tools/equivalence/run_matrix.sh > /tmp/equiv_cuda.log 2>&1 &
```
Also run `tests/test_direct_calibration_parity.py` and `tests/test_direct_calibration_sweep.py` there: the `_RawOBBTensors` branch of `rescore_parts` only executes on native CUDA and is otherwise untested.

- [ ] **Step 5: Pre-PR checks**

```bash
make commit-prep && make lint-moderate && make docs-check
```

- [ ] **Step 6: Retire the docs**

```bash
git mv docs/superpowers/specs/2026-09-01-detectkit-sahi-calibration-profiles-design.md docs/superpowers/specs/done/
git mv docs/superpowers/plans/2026-09-04-detectkit-sahi-calibration-profiles.md docs/superpowers/plans/done/
# set the spec Status header to: Shipped -- merged to main (<sha>)
git commit -m "docs: retire SAHI calibration spec and plan to done/"
```

- [ ] **Step 7: Merge and clean up**

```bash
git checkout main && git merge --no-ff feat/sahi-calibration-profiles
git worktree remove .worktrees/sahi-calib && git worktree prune
```

---

## Self-review notes

**Spec coverage:** workflow/entry points (12), evidence choice (7, 10), candidate measurement (2, 3, 4, 8), inspect & choose (5, 11), evaluation semantics (5, 17), data model (6, 9, 13), compatibility rules (6, 14), TrackerKit behaviour (14, 15), architecture/seam (0, 1, 3), safeguards (8, 11, 13, 14, 16), tests (each task), rollout labelling (17).

**Deliberate deviations from a literal reading of the spec:**
- The spec's file names `core/inference/direct_calibration.py` and `detectkit/gui/dialogs/direct_calibration_*.py` are partly taken by shipped code, so the grid/sweep modules are split out.
- The spec lists "merge policy/metric where the production merger supports them" — `merge_backend` is excluded from the sweep because it is forced to `cv2` on host paths (`obb.py:1571-1581`).
- The spec's "any registered direct model" entry point is a Tools-menu action, because DetectKit has no registered-model list page.

**Carried risk, stated:** `max_detections` is now an explicit calibration input. If a user sets it below their real animal count, every row reports irreducible misses — the wizard tooltip and the user guide both say so, and the value is recorded in each profile's `settings` and `measurement`.

**Open question for the user:** the `Balanced` floors (`MIN_MATCHED_INSTANCES = 60`, `F1_TOLERANCE = 0.01`, `MIN_LOCALIZATION = 0.5`) are chosen by analogy with SAM3's constants (`semantic/calibration.py:60-66`). Confirm or adjust before Task 5 lands; they are single constants.
