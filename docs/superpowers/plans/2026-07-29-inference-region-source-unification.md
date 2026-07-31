# Inference Region-Source Unification (Phase C) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Unify the OBB stage's three region-generation modes (whole frame / grid tiles / stage-1 proposals) onto one `region_source → executor → extract_with_transform → merge_per_frame` pipeline, collapsing 5× task-dispatch / 3× remap / 3× merge duplication, while keeping tracking output **byte-identical** on MPS + CUDA and preserving every tier's fast path. Ships sequential-mode slicing as a new capability.

**Architecture:** A `Region(image, affine, frame_idx)` value; a `RegionSource` protocol with four implementations (`WholeFrame`, `Grid`, `Stage1Proposals`, `SlicedStage1Proposals`); a single `extract_with_transform` dispatching `task × universe` (numpy `Results` vs raw `_RawOBBTensors`) and applying the region's affine in the matching universe; a `merge_per_frame` with per-source policy. The two extract universes are preserved exactly (raw = gpu-native `tensor_on_cuda`; numpy = cpu/mps/all gpu_fast via `DirectExecutorAdapter`).

**Tech Stack:** Python 3.10+, NumPy, PyTorch/ultralytics, cv2, pytest. `hydra-mps` here; CUDA parity on `mehek` (`hydra-cuda`). Equivalence harness `tools/equivalence/run_matrix.sh`.

## Global Constraints

- **HOT-PATH BYTE-IDENTICAL (overriding):** `direct`, `sliced`, `seq_crop_obb`, `seq_crop_segment` produce byte-identical tracking output vs pre-C main, on **MPS and CUDA**, at every step. This is a hot-path rewrite — no big-bang cutover; one mode rerouted per task, parity-gated before the next.
- **Preserve the two extract universes exactly:** raw `_RawOBBTensors` iff `runtime.tensor_on_cuda` (gpu-native torch/cuda); numpy `Results` for cpu/mps/all gpu_fast. Do NOT change which universe any mode uses (except the §5.2 opt-in, Task 12, gated on the harness).
- **Affine invariant:** a region transform is `(offset, scale)`. `scale != (1,1)` occurs ONLY for sequential crops (numpy). The raw universe applies **translate only** — never add scale to `_translate_raw`/raw extractors.
- **Per-source merge policy:** `Grid` = overlap-band NMS gated by the `tiles_overlap` geometry predicate (NEVER `overlap_*_ratio`); `WholeFrame`/`Stage1Proposals`/`SlicedStage1Proposals` = plain concat + `_apply_raw_detection_cap`, NO merge-time NMS. Adding NMS to sequential/direct would break byte-identity.
- **Do NOT touch:** detection results/numerics, merge *algorithms* (`merge.py`/`merge_gpu.py` kernels), `DirectExecutorAdapter`/`direct_executors.py` internals, NVDEC, the filtering/tracking stages, or the `.npz` cache schema. Consume them unchanged.
- **Locate by SYMBOL, not line number.** Test per-file (base suite has pre-existing failures + a `test_detectkit_main_window.py` segfault).
- **Test invocation:** `conda run -n hydra-mps env PYTHONPATH="$PWD/src" python -m pytest <file> -v`.
- **Commit as the configured git user; NO `Co-Authored-By: Claude` trailer.**

## Parity gate (controller-run between structural tasks — NOT inside implementers)

After each task that reroutes a mode (Tasks 5, 6, 7, 9, 10, 11, 12), the CONTROLLER runs the equivalence harness before dispatching the next task. Implementers run only unit tests + import sanity. Gate command (MPS; baseline = pre-C main `src`, current = feature worktree `src`):

```bash
REPO=$PWD WT=$PWD MAIN_SRC=$PWD/src WT_SRC=<feature-worktree>/src OUT=/tmp/equiv_c RUNTIME=mps \
  bash tools/equivalence/run_matrix.sh <clips>
```
Fast smoke = `fly_obb ant_obb_sequential` + a sliced clip; **full matrix (all 7 clips) at Tasks 9, 10, 12**. CUDA (mehek/`hydra-cuda`) full matrix at Tasks 9, 10, 12. Every clip must be EQUIVALENT (0 unmatched, pos p99≈0, θ≈0) with row counts > 1.

---

## File Structure

**New file**
- `src/hydra_suite/core/inference/stages/regions.py` — `Region`, `Affine`, the `RegionSource` protocol + four implementations, and `select_region_source`. Owns region generation; imports geometry (`plan_slices`, `build_crops`) from existing modules.
- `tests/test_region_source.py` — unit tests for affine, `extract_with_transform`, and each region source.

**Modified**
- `src/hydra_suite/core/inference/stages/obb.py` — `_extract_obb_from_boxes` offset/scale; new `extract_with_transform` + `_translate_raw`; `run_obb` rerouted; `_run_direct`/`_run_sequential` retired at the end.
- `src/hydra_suite/core/inference/stages/slicing.py` / `slicing_cuda.py` — `_extract_tile`/`extract_raw_tile`/`_offset_result`/`_remap_raw`/`_merge_frame_obb_results`/`assemble_raw_frames` retired in favor of the shared seam; `run_direct_sliced` rerouted then thinned.
- `src/hydra_suite/core/inference/config.py` — `OBBSequentialConfig.stage1_slice: SliceConfig` (Task 11) + `YOLO_SEQ_STAGE1_SLICE_*` builder param.

---

## Task 1: Affine model + numpy `_extract_obb_from_boxes` offset/scale

**Files:**
- Create: `src/hydra_suite/core/inference/stages/regions.py`
- Modify: `src/hydra_suite/core/inference/stages/obb.py` (`_extract_obb_from_boxes`)
- Test: `tests/test_region_source.py`

**Interfaces:**
- Produces: `@dataclass(frozen=True) class Affine: offset: tuple[float,float] = (0.0,0.0); scale: tuple[float,float] = (1.0,1.0)` with `IDENTITY` classattr and `is_translate_only` property. `_extract_obb_from_boxes` gains `offset=(0.0,0.0)`, `scale=(1.0,1.0)` keyword params (scale-then-offset on cx/cy/w/h, mirroring `extract_obb_result`).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_region_source.py
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from hydra_suite.core.inference.stages.regions import Affine
from hydra_suite.core.inference.stages import obb as m


def test_affine_identity_and_translate_only():
    assert Affine().offset == (0.0, 0.0) and Affine().scale == (1.0, 1.0)
    assert Affine(offset=(5.0, 6.0)).is_translate_only
    assert not Affine(scale=(2.0, 1.0)).is_translate_only


class _FakeBoxes:
    def __init__(self, xyxy, conf):
        self.xyxy = torch.tensor(xyxy, dtype=torch.float32)
        self.conf = torch.tensor(conf, dtype=torch.float32)


class _FakeDetResult:
    def __init__(self, xyxy, conf):
        self.boxes = _FakeBoxes(xyxy, conf)


def test_extract_boxes_offset_scale_maps_to_frame():
    res = _FakeDetResult([[10.0, 20.0, 30.0, 60.0]], [0.9])  # cx=20,cy=40,w=20,h=40
    out = m._extract_obb_from_boxes(res, 0, 0.0, offset=(100.0, 200.0), scale=(2.0, 2.0))
    assert out.centroids[0][0] == pytest.approx(140.0)  # 20*2+100
    assert out.centroids[0][1] == pytest.approx(280.0)  # 40*2+200


def test_extract_boxes_default_identity_byte_identical():
    res = _FakeDetResult([[10.0, 20.0, 30.0, 60.0]], [0.9])
    a = m._extract_obb_from_boxes(res, 0, 0.0)
    b = m._extract_obb_from_boxes(res, 0, 0.0, offset=(0.0, 0.0), scale=(1.0, 1.0))
    assert np.array_equal(a.centroids, b.centroids) and np.array_equal(a.corners, b.corners)
```

- [ ] **Step 2: Run test → RED**

Run: `conda run -n hydra-mps env PYTHONPATH="$PWD/src" python -m pytest tests/test_region_source.py -v`
Expected: FAIL — no `regions` module / `_extract_obb_from_boxes` has no `offset`.

- [ ] **Step 3: Implement**

Create `regions.py` with the `Affine` dataclass:

```python
"""Region-source abstraction for the OBB stage (phase C)."""
from __future__ import annotations
from dataclasses import dataclass


@dataclass(frozen=True)
class Affine:
    """Maps region-local pixel coords -> frame coords: p_frame = p_region * scale + offset."""
    offset: tuple[float, float] = (0.0, 0.0)
    scale: tuple[float, float] = (1.0, 1.0)

    @property
    def is_translate_only(self) -> bool:
        return self.scale == (1.0, 1.0)


Affine.IDENTITY = Affine()
```

In `obb.py`, add the two keyword params to `_extract_obb_from_boxes` (after `fixed_angle_rad`, before `emit_native_geometry`):

```python
    fixed_angle_rad: float,
    *,
    offset: tuple[float, float] = (0.0, 0.0),
    scale: tuple[float, float] = (1.0, 1.0),
    emit_native_geometry: bool = False,
```

Immediately after `h_arr = xyxy[:, 3] - xyxy[:, 1]` and BEFORE `angle_arr = ...`, apply scale-then-offset (mirrors `extract_obb_result`; identity default keeps it byte-identical):

```python
    ox, oy = offset
    sx, sy = scale
    cx = cx * sx + ox
    cy = cy * sy + oy
    w_arr = w_arr * sx
    h_arr = h_arr * sy
```

- [ ] **Step 4: Run test → GREEN**

Run: `conda run -n hydra-mps env PYTHONPATH="$PWD/src" python -m pytest tests/test_region_source.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/hydra_suite/core/inference/stages/regions.py src/hydra_suite/core/inference/stages/obb.py tests/test_region_source.py
git commit -m "feat(inference): affine model + offset/scale on _extract_obb_from_boxes"
```

---

## Task 2: `extract_with_transform` — unified numpy dispatch

**Files:**
- Modify: `src/hydra_suite/core/inference/stages/obb.py`
- Test: `tests/test_region_source.py`

**Interfaces:**
- Consumes: `Affine` (Task 1); existing `extract_obb_result`/`_extract_obb_from_boxes`/`_extract_obb_from_masks` (all now offset/scale-capable).
- Produces: `extract_with_transform(result, frame_idx, task, affine, config, runtime) -> OBBResult | _RawOBBTensors`. THIS TASK implements only the numpy branch (`not runtime.tensor_on_cuda`); the raw branch raises `NotImplementedError` (filled in Task 3). Reads seg params from `config.direct` when present (for the direct/grid callers); the sequential caller passes them explicitly in a later task via a thin wrapper (documented in Task 9).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_region_source.py  (append)
def test_extract_with_transform_numpy_detect(monkeypatch):
    from hydra_suite.core.inference.stages.regions import Affine

    class _Rt:  # numpy universe
        tensor_on_cuda = False
        device = "cpu"

    class _Cfg:
        class direct:
            fixed_angle_deg = 0.0
            seg_num_angles = 24; seg_crop_size = 64; seg_pad_ratio = 0.15; seg_mask_threshold = 0.5
        raw_detection_cap = 0
        emit_native_geometry = False

    res = _FakeDetResult([[10.0, 20.0, 30.0, 60.0]], [0.9])
    out = m.extract_with_transform(res, 0, "detect", Affine(offset=(100.0, 0.0)), _Cfg, _Rt())
    assert out.centroids[0][0] == pytest.approx(120.0)  # 20 + 100
```

- [ ] **Step 2: Run test → RED**

Expected: FAIL — no `extract_with_transform`.

- [ ] **Step 3: Implement (numpy branch only)**

In `obb.py`:

```python
def extract_with_transform(result, frame_idx, task, affine, config, runtime):
    """Single task x universe extraction seam. Applies `affine` in the matching universe.

    numpy universe (cpu/mps/all gpu_fast): affine applied during extraction (offset+scale).
    raw universe (gpu-native tensor_on_cuda): translate-only on-device (Task 3).
    """
    ox, oy = affine.offset
    sx, sy = affine.scale
    if runtime.tensor_on_cuda:
        raise NotImplementedError("raw universe: Task 3")
    d = config.direct
    if task == "detect":
        import math
        return _extract_obb_from_boxes(
            result, frame_idx, math.radians(d.fixed_angle_deg if d else 0.0),
            offset=(ox, oy), scale=(sx, sy),
            emit_native_geometry=config.emit_native_geometry,
        )
    if task == "segment":
        return _extract_obb_from_masks(
            result, frame_idx, config.raw_detection_cap,
            num_angles=d.seg_num_angles, crop_size=d.seg_crop_size,
            pad_ratio=d.seg_pad_ratio, mask_threshold=d.seg_mask_threshold,
            offset=(ox, oy), scale=(sx, sy),
            emit_native_geometry=config.emit_native_geometry,
        )
    return extract_obb_result(
        result, frame_idx, offset=(ox, oy), scale=(sx, sy),
        emit_native_geometry=config.emit_native_geometry,
    )
```

- [ ] **Step 4: Run test → GREEN**; **Step 5: Commit**

```bash
git commit -am "feat(inference): extract_with_transform numpy branch (unified task dispatch)"
```

---

## Task 3: `extract_with_transform` — raw (on-device) branch + `_translate_raw`

**Files:** `obb.py`; `tests/test_region_source.py`

**Interfaces:**
- Consumes: existing `_extract_raw_tensors`/`_extract_raw_tensors_from_boxes`/`_extract_raw_tensors_from_masks`; `slicing_cuda._remap_raw` logic.
- Produces: `_translate_raw(raw, offset) -> _RawOBBTensors` (generalized `_remap_raw`, translate-only, asserts `scale` unused); the raw branch of `extract_with_transform` (task dispatch + `_translate_raw`).

- [ ] **Step 1: Write the failing test** (raw fake with `.xywhr`/`.corners`; assert translate; assert a non-identity `scale` under `tensor_on_cuda` raises — enforcing the invariant). [Full test code: build a `_RawOBBTensors` via a fake result path or a direct `_translate_raw` unit test on a hand-built `_RawOBBTensors`.]

```python
# tests/test_region_source.py  (append)
def test_translate_raw_offsets_only():
    from hydra_suite.core.inference.stages.obb import _translate_raw, _RawOBBTensors
    raw = _RawOBBTensors(
        frame_idx=0,
        xywhr=torch.tensor([[10.0, 20.0, 5.0, 6.0, 0.3]]),
        corners=torch.zeros((1, 4, 2)),
        conf=torch.tensor([0.9]),
        cls=torch.tensor([0.0]),
    )
    out = _translate_raw(raw, (100.0, 200.0))
    assert out.xywhr[0, 0].item() == pytest.approx(110.0)
    assert out.xywhr[0, 1].item() == pytest.approx(220.0)
    assert out.xywhr[0, 2].item() == pytest.approx(5.0)  # w unchanged


def test_extract_with_transform_raw_rejects_scale():
    from hydra_suite.core.inference.stages.regions import Affine
    class _Rt: tensor_on_cuda = True; device = "cpu"
    class _Cfg:
        class direct: model_task = "obb"
        raw_detection_cap = 0
    with pytest.raises((AssertionError, ValueError)):
        m.extract_with_transform(object(), 0, "obb", Affine(scale=(2.0, 1.0)), _Cfg, _Rt())
```

- [ ] **Step 2: RED** → **Step 3: Implement**

Add `_translate_raw` (generalize `slicing_cuda._remap_raw`, in `obb.py` so both universes live together):

```python
def _translate_raw(raw, offset):
    """Pure on-device translation of a _RawOBBTensors by `offset`. Never scales
    (raw universe is translate-only; scale != 1 implies the numpy universe)."""
    ox, oy = offset
    if raw.xywhr.shape[0] == 0 or (ox == 0.0 and oy == 0.0):
        return raw
    xywhr = raw.xywhr.clone(); xywhr[:, 0] += ox; xywhr[:, 1] += oy
    corners = raw.corners.clone(); corners[..., 0] += ox; corners[..., 1] += oy
    return _RawOBBTensors(frame_idx=raw.frame_idx, xywhr=xywhr, corners=corners,
                          conf=raw.conf, cls=raw.cls)
```

Fill the raw branch of `extract_with_transform` (replace the `raise NotImplementedError`):

```python
    if runtime.tensor_on_cuda:
        assert affine.is_translate_only, "raw universe must be translate-only (invariant)"
        import math
        d = config.direct
        if task == "detect":
            raw = _extract_raw_tensors_from_boxes(result, frame_idx, math.radians(d.fixed_angle_deg), runtime.device)
        elif task == "segment":
            raw = _extract_raw_tensors_from_masks(
                result, frame_idx, runtime.device, config.raw_detection_cap,
                num_angles=d.seg_num_angles, crop_size=d.seg_crop_size,
                pad_ratio=d.seg_pad_ratio, mask_threshold=d.seg_mask_threshold)
        else:
            raw = _extract_raw_tensors(result, frame_idx, runtime.device)
        return _translate_raw(raw, affine.offset)
```

- [ ] **Step 4: GREEN**; **Step 5: Commit**

```bash
git commit -am "feat(inference): extract_with_transform raw branch + _translate_raw"
```

---

## Task 4: `RegionSource` protocol + `WholeFrame` + `Grid` + `Stage1Proposals` (planning only)

**Files:** `regions.py`; `tests/test_region_source.py`

**Interfaces:**
- Produces: `@dataclass class Region: image; affine: Affine; frame_idx: int`; a `RegionSource` protocol with `plan(frames, models, config, runtime) -> list[list[Region]]`, `merge_policy: str` (`"plain"|"overlap_band_nms"`), `device_residency: str` (`"on_device_capable"|"cpu_crop_boundary"`); implementations `WholeFrame`, `Grid`, `Stage1Proposals` that PLAN regions by reusing existing geometry (`WholeFrame`: one region=frame, `Affine.IDENTITY`; `Grid`: `plan_slices` tiles → translate affines, absorbing `slicing._build_tile_jobs`/`plan_slices`; `Stage1Proposals`: run stage-1 predict + `build_crops` → resized-crop regions with offset+scale affines, absorbing `_run_sequential`'s stage-1+crop logic). `select_region_source(config)`. This task builds the PLANNING side only; execution/extraction wiring is Tasks 5-7. Do NOT reroute `run_obb` yet — pure addition with unit tests over region counts/affines/policies.

- [ ] Steps: TDD unit tests asserting, for a synthetic frame + fake models, each source's region count, affine kinds (identity / translate-only / offset+scale), `merge_policy`, `device_residency`. Implement by extracting the geometry from `slicing.py`/`_run_sequential`/`crops.py` into planners (call the existing functions; do not duplicate their bodies). Commit: `feat(inference): RegionSource protocol + WholeFrame/Grid/Stage1Proposals planners`.

---

## Task 5: Route `_run_direct` (WholeFrame) through the unified seam

**Files:** `obb.py`; `tests/test_region_source.py`

Rewrite `_run_direct` to: build WholeFrame regions, run the (existing) predict machinery, and call `extract_with_transform(res, idx, task, Affine.IDENTITY, config, runtime)` + `_apply_raw_detection_cap` — replacing its inline 5-way numpy+raw dispatch (obb.py ~549-633) with the seam. The predict + cuda-tensor-letterbox logic stays. NO change to results.

- [ ] Unit: a fake direct run routes through `extract_with_transform`. **Controller PARITY GATE after this task:** MPS `fly_obb` (direct clip) EQUIVALENT. Commit: `refactor(inference): route _run_direct through extract_with_transform`.

---

## Task 6: Route the sliced path (Grid) through the seam

**Files:** `slicing.py`, `slicing_cuda.py`, `obb.py`; tests

Replace `slicing._extract_tile` + `_offset_result` and `slicing_cuda.extract_raw_tile` + `_remap_raw` with `extract_with_transform(res, tile_idx, task, Region.affine, config, runtime)` (tile affine = translate `(x0,y0)`, scale 1). Keep `plan_slices`/`_build_tile_jobs`/`_predict_tiles`/tile ROI gating and the merge (`_merge_frame_obb_results`/`assemble_raw_frames`) UNCHANGED for now (merge unifies in Task 8).

- [ ] Unit + **Controller PARITY GATE:** MPS a sliced clip (e.g. `fly_obb` with slicing on, or the project's sliced fixture) EQUIVALENT, both host + cuda merge backends exercised. Commit: `refactor(inference): sliced tiles use extract_with_transform`.

---

## Task 7: Route `_run_sequential` (Stage1Proposals) through the seam

**Files:** `obb.py`; tests

Replace `_run_sequential`'s stage-2 inline obb|segment dispatch (obb.py ~684-693) with `extract_with_transform(r, frame_idx, seq.stage2_task, Region.affine, config, runtime)` where the region affine carries the per-crop `offset=offsets[i+j]` and `scale=scale`. Sequential stays numpy universe (`tensor_on_cuda` path not taken — crops are CPU). `merge_obb_results`+cap unchanged.

- [ ] Unit + **Controller PARITY GATE:** MPS `ant_obb_sequential` (obb) + a seq_crop_segment smoke EQUIVALENT. Commit: `refactor(inference): sequential stage-2 uses extract_with_transform`.

---

## Task 8: Unify merge into `merge_per_frame`

**Files:** `obb.py`, `slicing.py`, `slicing_cuda.py`, `regions.py`; tests

Add `merge_per_frame(per_frame_obbs, merge_policy, plan_or_none, config, runtime)` that: concatenates a frame's parts (numpy `merge_obb_results` or raw `_concat_raw` by universe), applies `_apply_raw_detection_cap`, and — iff `merge_policy=="overlap_band_nms"` and `tiles_overlap(plan.tiles)` — runs the band NMS (`merge.merge_obb_detections`/`merge_gpu` via `assemble_raw_frames`'s existing logic). Route Grid through it; WholeFrame/Stage1Proposals use the plain branch (identical to their current `merge_obb_results`+cap). Retire `_merge_frame_obb_results` + fold `assemble_raw_frames`'s merge decision into `merge_per_frame`.

- [ ] Unit (plain vs nms policy; overlap predicate) + **Controller PARITY GATE:** MPS sliced + direct + sequential clips EQUIVALENT. Commit: `refactor(inference): unify per-frame merge into merge_per_frame`.

---

## Task 9: Collapse `run_obb` onto the pipeline; retire dead orchestrators

**Files:** `obb.py`, `slicing.py`, `slicing_cuda.py`; tests

Rewrite `run_obb` to the §4 pipeline: `source = select_region_source(config); regions = source.plan(...); results = execute_regions(...); obbs = [extract_with_transform(...)]; return merge_per_frame(...)`. `_run_direct`/`_run_sequential`/`run_direct_sliced` bodies become thin shims or are deleted; delete the now-unused `_extract_tile`/`_offset_result`/`_remap_raw`/`extract_raw_tile`/`_merge_frame_obb_results`/`assemble_raw_frames`. `grep -rn` confirms no stale references.

- [ ] Unit + **Controller PARITY GATE — FULL MATRIX MPS + CUDA (all 7 clips).** Commit: `refactor(inference): collapse run_obb onto the region-source pipeline; delete dead paths`.

---

## Task 10: (buffer) Full cross-platform parity confirmation + cleanup

**Files:** none (verification) / minor cleanup

- [ ] Controller runs the **full equivalence matrix on MPS and CUDA** on the collapsed pipeline; fix any divergence found (must land back at byte-identical). Format/lint the touched files. Commit any cleanup.

---

## Task 11: Ship sequential-mode slicing (`SlicedStage1Proposals`)

**Files:** `regions.py`, `config.py`; `tests/test_region_source.py`

Add `OBBSequentialConfig.stage1_slice: SliceConfig = field(default_factory=SliceConfig)` (off by default) + `YOLO_SEQ_STAGE1_SLICE_*` in `build_inference_config_from_params`. Add `SlicedStage1Proposals`: Grid-plan the frame for stage-1, run stage-1 per tile, remap+overlap-band-merge stage-1 boxes to frame space, then `build_crops`/stage-2 as `Stage1Proposals`. `select_region_source`: `sequential` + `stage1_slice.enabled` → this source.

- [ ] **Correctness tests (new capability, no parity baseline):** tiled stage-1 recovers a small object a whole-frame stage-1 misses; stage-1 boxes merge to correct frame coords; resulting crop offsets are right. Existing sequential (flag off) still byte-identical (**Controller PARITY GATE:** `ant_obb_sequential` MPS). Commit: `feat(inference): sequential-mode slicing (SlicedStage1Proposals)`.

---

## Task 12: (opt-in, harness-gated) sequential stage-2 raw acceleration

**Files:** `regions.py`/`obb.py`; verification

Add a `Stage1Proposals` capability flag routing stage-2 extraction through the raw universe on the gpu-native tier (the crop images are CPU, but stage-2 `Results` on the gpu tier are cuda tensors → `extract_with_transform` raw branch, translate-only affine since stage-2 crops... NOTE: sequential crops use scale≠1 → this only works if the per-crop affine's scale is folded before the raw translate; if scale≠1 cannot be expressed on the raw path, this opt-in is INFEASIBLE and ships off — document that finding).

- [ ] **Controller GATE:** run the equivalence harness (MPS + CUDA, `ant_obb_sequential` + seq_crop_segment) comparing raw-on vs the numpy baseline. **EQUIVALENT ⇒ ship the flag on (default); NOT equivalent OR infeasible (scale≠1 on raw) ⇒ ship off + document in the spec's §5.2.** Commit accordingly.

> **Note (plan author):** the §4 affine invariant says `scale≠1 ⟹ numpy`. Sequential crops are resized to `stage2_image_size` (scale≠1), so routing them through the raw universe would violate the invariant unless the crop is fed at native size (scale=1). Task 12 must first determine whether stage-2 can run at crop-native size on the gpu tier; if not, the opt-in is correctly infeasible and ships off. This is the honest outcome, not a failure.

---

## Task 13: Format, lint, final verification

- [ ] `make format`-equivalent (black/isort) on touched files; `flake8` clean. Full new-suite (`tests/test_region_source.py` + existing obb/slicing/sequential tests) green. Final full MPS+CUDA equivalence matrix. Commit any formatting.

---

## Self-Review Notes (spec coverage)

- §4 architecture → Tasks 1-9 (affine, extract seam both universes, region sources, executor via existing predict, merge). §5.1 fast-path preservation → universes never changed (Tasks 2/3/5-7). §5.2 sequential raw opt-in → Task 12 (harness-gated, may ship off — honest). §6 per-source merge policy → Task 8. §7 sequential-mode slicing → Task 11. §8 incremental parity-gated order → the task order + controller gates. §9 non-goals honored (no merge-kernel/executor-internal/cache changes). §10 testing → unit per task + controller equivalence gates.
- **Byte-identity** is gated by the controller equivalence run after every reroute (Tasks 5,6,7,8,9,11) with full MPS+CUDA matrices at 9, 10, 12, 13.
- **Task 12 explicitly may conclude "infeasible/ship off"** — the invariant (`scale≠1⟹numpy`) predicts sequential crops (scale≠1) can't ride raw unless run at native size; the task determines this empirically and documents the outcome rather than forcing it.
