# SAHI Sliced Inference Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add optional, configurable SAHI-style sliced inference to the direct-mode OBB detector in the InferenceRunner pipeline, covering all three tasks (`obb`/`detect`/`segment`) and all six tier×device paths, at minimal compute cost and off by default.

**Architecture:** Slicing is a task-agnostic *wrapper* around `_run_direct`'s chunked predict call — it tiles frames, runs the existing per-tile predict+extract path unchanged, remaps detections to frame coordinates, and merges cross-tile duplicates. Two new modules (`stages/slicing.py`, `stages/merge.py`) plus one new utils geometry kernel (`utils/rotated_iou.py`); `obb.py`'s `run_obb` gains a single dispatch hook (lazy import to avoid a cycle). Merge is a pluggable seam with a `cv2` backend (default, correctness oracle) and a `gpu` backend (native-cuda only). `enabled=False` is structurally byte-identical to today.

**Tech Stack:** Python, PyTorch (CUDA/MPS), OpenCV (`cv2`), NumPy, ultralytics YOLO, PyQt (TrackerKit GUI), pytest.

## Environment (VERIFIED — prefix every test command with this)

Work happens in the worktree `.worktrees/sahi-sliced-inference` (branch
`feature/sahi-sliced-inference`). Tests FAIL TO COLLECT without the conda env and
the libomp flag (torch double-init: `RpcBackendOptions ... already defined` /
`OMP: Error #15`). Exact preamble:

```bash
source ~/miniforge3/etc/profile.d/conda.sh && conda activate hydra-mps
export KMP_DUPLICATE_LIB_OK=TRUE
cd /Users/neurorishika/Projects/Rockefeller/Kronauer/multi-animal-tracker/.worktrees/sahi-sliced-inference
export PYTHONPATH=$PWD/src
```

**Verified baseline (2026-07-24), use as the delta gate:**
- `test_inference_stages_obb.py`, `test_inference_config.py`, `test_inference_obb_artifacts.py`, `test_utils_obb_from_mask.py`: **88 passed, 1 skipped — clean.** Any failure here is YOURS.
- `test_main_window_config_persistence.py`: **7 pre-existing failures, 34 passed.** Pre-existing (do NOT try to fix): `test_video_autoload_restores_pose_keypoint_groups_and_headtail_model`, `test_preview_detection_restores_analyze_individual_controls`, `test_realtime_direct_mode_exposes_micro_batch_controls`, `test_realtime_micro_batch_roundtrip_persists`, `test_advanced_config_defaults_include_identity_decoder_tuning`, `test_get_parameters_dict_exposes_identity_decoder_advanced_overrides`, `test_identity_decoder_tuning_controls_roundtrip_through_tracker_config`. This suite takes ~105s.
- `window.get_parameters_dict()` is CONFIRMED to exist and is how existing tests read the UPPER_SNAKE params dict (see lines 274/285/303).

## Global Constraints

- **Layer boundary:** `core/inference` may NOT import from any app layer or from `core/detectors`. Geometry kernels live in `utils/` (mirrors `utils/obb_from_mask.py`). `utils/` imports nothing from app layers.
- **Import-cycle rule:** `obb.py → slicing.py` MUST be a function-level (lazy) import inside `run_obb`. Static graph is `slicing → {obb, merge}`, `merge → {obb, filtering, utils/rotated_iou, utils/obb_from_mask}`, `filtering → obb`. No static cycles.
- **Byte-parity:** `enabled=False` MUST produce byte-identical output to pre-feature `main`. Do NOT modify `filtering.py:_obb_nms` / `_obb_iou_corners` (legacy-parity-locked). The merge module gets its own overlap helper.
- **Defaults (verbatim):** `enabled=False`, `geometry_mode="auto_model"`, `overlap_height_ratio=0.2`, `overlap_width_ratio=0.2`, `object_tile_fraction=0.15`, `merge_policy="greedy_nmm"`, `merge_metric="ios"`, `merge_threshold=0.5`, `merge_backend="cv2"`, `perform_standard_pred=False`.
- **`gpu` merge backend need only match `cv2` within tolerance**, not bit-for-bit (sliced inference has no legacy byte-parity contract). `cv2` is the test oracle.
- **Commit after every task.** Run `make format` before each commit. Commit as the configured git user (no Co-Authored-By trailer).
- **Detection-ID stride:** `DETECTION_ID_STRIDE = 10000` (`result.py:16`). All merged results regenerate ids via `OBBResult.make_detection_ids(frame_idx, n)`.

---

## File Structure

**New files:**
- `src/hydra_suite/core/inference/stages/slicing.py` — `SlicePlan`, `plan_slices`, `run_direct_sliced`, tile flatten/chunk/remap orchestration, exact-tile fast path, native-cuda `_RawOBBTensors` preservation.
- `src/hydra_suite/core/inference/stages/merge.py` — `merge_obb_detections(policy, metric, backend, ...)`, cv2 backend (nms/nmm/greedy_nmm × iou/ios), overlap-band pre-filter, gpu backend.
- `src/hydra_suite/utils/rotated_iou.py` — batched GPU rotated-box pairwise IoU/IoS (Sutherland–Hodgman polygon clipping), torch-only.
- Test files: `tests/test_inference_slicing.py`, `tests/test_inference_merge.py`, `tests/test_utils_rotated_iou.py`, plus additions to `tests/test_inference_config.py`, `tests/test_inference_cache_keys.py` (create if absent), `tests/test_main_window_config_persistence.py`.

**Modified files:**
- `src/hydra_suite/core/inference/config.py` — add `SliceConfig`, nest on `OBBDirectConfig`, parse in `from_dict` + `build_inference_config_from_params`.
- `src/hydra_suite/core/inference/stages/obb.py` — dispatch hook in `run_obb` (lazy import).
- `src/hydra_suite/core/inference/runner.py:145-148` — size TRT batch from tile-chunk when slicing enabled.
- `src/hydra_suite/core/inference/cache/keys.py:48-63` — fold slice params into `detection_cache_key` when enabled.
- `src/hydra_suite/trackerkit/gui/panels/detection_panel.py` — checkbox + geometry dropdown.
- `src/hydra_suite/trackerkit/gui/orchestrators/config.py` — load/save + UPPER_SNAKE + advanced_config knobs.

---

## Task 1: SliceConfig schema + config round-trip + param builder

**Files:**
- Modify: `src/hydra_suite/core/inference/config.py` (add `SliceConfig` after `OBBDirectConfig` ~line 76; nest field on `OBBDirectConfig`; fix nested parse at `from_dict` ~line 346; add param parse in `build_inference_config_from_params` ~line 555)
- Test: `tests/test_inference_config.py`

**Interfaces:**
- Produces: `SliceConfig` dataclass (fields per Global Constraints defaults); `OBBDirectConfig.slice: SliceConfig` (default_factory); `build_inference_config_from_params` reads `SLICE_*` keys into it.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_inference_config.py (append)
from hydra_suite.core.inference.config import (
    OBBDirectConfig,
    SliceConfig,
    OBBConfig,
    InferenceConfig,
    build_inference_config_from_params,
)


def test_slice_config_defaults_off():
    s = SliceConfig()
    assert s.enabled is False
    assert s.geometry_mode == "auto_model"
    assert s.overlap_height_ratio == 0.2 and s.overlap_width_ratio == 0.2
    assert s.merge_policy == "greedy_nmm"
    assert s.merge_metric == "ios"
    assert s.merge_threshold == 0.5
    assert s.merge_backend == "cv2"
    assert s.perform_standard_pred is False


def test_obb_direct_config_has_slice_default():
    d = OBBDirectConfig(model_path="m.pt")
    assert isinstance(d.slice, SliceConfig)
    assert d.slice.enabled is False


def test_obb_direct_from_dict_parses_nested_slice():
    obb = OBBConfig.from_dict(
        {
            "mode": "direct",
            "direct": {
                "model_path": "m.pt",
                "slice": {"enabled": True, "geometry_mode": "custom",
                          "slice_height": 640, "slice_width": 640},
            },
        }
    )
    assert obb.direct.slice.enabled is True
    assert obb.direct.slice.geometry_mode == "custom"
    assert obb.direct.slice.slice_height == 640


def test_build_config_reads_slice_params():
    params = {
        "YOLO_OBB_MODE": "direct",
        "YOLO_OBB_DIRECT_MODEL_PATH": "m.pt",
        "SLICE_ENABLED": True,
        "SLICE_GEOMETRY_MODE": "custom",
        "SLICE_HEIGHT": 512,
        "SLICE_WIDTH": 512,
        "SLICE_OVERLAP": 0.25,
        "SLICE_MERGE_POLICY": "nms",
        "SLICE_MERGE_METRIC": "iou",
        "SLICE_MERGE_THRESHOLD": 0.4,
        "SLICE_MERGE_BACKEND": "gpu",
        "SLICE_OBJECT_TILE_FRACTION": 0.2,
        "SLICE_PERFORM_STANDARD_PRED": True,
    }
    cfg = build_inference_config_from_params(params)
    s = cfg.obb.direct.slice
    assert s.enabled is True
    assert s.geometry_mode == "custom"
    assert s.slice_height == 512 and s.slice_width == 512
    assert s.overlap_height_ratio == 0.25 and s.overlap_width_ratio == 0.25
    assert s.merge_policy == "nms" and s.merge_metric == "iou"
    assert s.merge_threshold == 0.4 and s.merge_backend == "gpu"
    assert s.object_tile_fraction == 0.2
    assert s.perform_standard_pred is True


def test_build_config_slice_defaults_when_absent():
    params = {"YOLO_OBB_MODE": "direct", "YOLO_OBB_DIRECT_MODEL_PATH": "m.pt"}
    cfg = build_inference_config_from_params(params)
    assert cfg.obb.direct.slice.enabled is False


def test_reference_body_px_sourced_and_resize_scaled():
    """auto_object needs a real object scale; it comes from REFERENCE_BODY_SIZE
    * RESIZE_FACTOR, the same source/scaling worker.py uses (worker.py:921)."""
    params = {
        "YOLO_OBB_MODE": "direct", "YOLO_OBB_DIRECT_MODEL_PATH": "m.pt",
        "SLICE_ENABLED": True, "REFERENCE_BODY_SIZE": 30.0, "RESIZE_FACTOR": 2.0,
    }
    cfg = build_inference_config_from_params(params)
    assert cfg.obb.direct.slice.reference_body_px == 60.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_inference_config.py -k slice -v`
Expected: FAIL with `ImportError: cannot import name 'SliceConfig'`.

- [ ] **Step 3: Add the `SliceConfig` dataclass**

In `config.py`, immediately after the `OBBDirectConfig` class (after line 76, before `OBBSequentialConfig`):

```python
@dataclass
class SliceConfig:
    """SAHI-style sliced inference for direct-mode OBB detection.

    All fields are inert when ``enabled`` is False — ``run_obb`` dispatches to
    the sliced path only when it is True, so the entire feature is dead code
    otherwise and output is byte-identical to the non-sliced pipeline.
    """

    enabled: bool = False
    geometry_mode: Literal["auto_model", "auto_object", "custom"] = "auto_model"
    # custom mode: explicit tile size in original-frame pixels.
    slice_height: int = 0
    slice_width: int = 0
    overlap_height_ratio: float = 0.2
    overlap_width_ratio: float = 0.2
    # auto_object mode: tile sized so a reference object spans this linear
    # fraction of the tile.
    object_tile_fraction: float = 0.15
    # Reference object size in ORIGINAL-FRAME pixels, sourced from
    # REFERENCE_BODY_SIZE * RESIZE_FACTOR. Only read in auto_object mode; 0
    # means "unknown", which falls back to auto_model sizing.
    reference_body_px: float = 0.0
    # merge across tile boundaries.
    merge_policy: Literal["nms", "nmm", "greedy_nmm"] = "greedy_nmm"
    merge_metric: Literal["iou", "ios"] = "ios"
    merge_threshold: float = 0.5
    # cv2 = default correctness oracle (all paths); gpu = native-cuda only.
    merge_backend: Literal["cv2", "gpu"] = "cv2"
    # extra full-frame pass in addition to tiles (catches > tile-size objects).
    perform_standard_pred: bool = False
```

Add `slice: "SliceConfig" = field(default_factory=lambda: SliceConfig())` to `OBBDirectConfig`. Since `SliceConfig` is defined *after* `OBBDirectConfig`, instead move `SliceConfig` to *before* `OBBDirectConfig` (line 34) and add the field directly:

```python
    # (inside OBBDirectConfig, after seg_mask_threshold, line 75)
    slice: SliceConfig = field(default_factory=SliceConfig)
```

- [ ] **Step 4: Fix nested `from_dict` parse**

At `config.py:346`, replace the naive `OBBDirectConfig(**obb_d["direct"])` with a nested-aware parse:

```python
        direct = None
        if obb_d.get("direct"):
            direct_d = dict(obb_d["direct"])
            slice_d = direct_d.pop("slice", None)
            direct = OBBDirectConfig(**direct_d)
            if isinstance(slice_d, dict):
                direct.slice = SliceConfig(**slice_d)
```

- [ ] **Step 5: Parse slice params in `build_inference_config_from_params`**

In the `else:` (direct-mode) branch of `build_inference_config_from_params` (after the seg params ~line 553, before `obb_cfg = OBBConfig(...)`), add:

```python
        overlap = _clamped_float(params.get("SLICE_OVERLAP", 0.2), 0.2, 0.0, 0.9)
        slice_cfg = SliceConfig(
            enabled=bool(params.get("SLICE_ENABLED", False)),
            geometry_mode=(
                str(params.get("SLICE_GEOMETRY_MODE", "auto_model")).strip().lower()
                if str(params.get("SLICE_GEOMETRY_MODE", "auto_model")).strip().lower()
                in {"auto_model", "auto_object", "custom"}
                else "auto_model"
            ),
            slice_height=_clamped_int(params.get("SLICE_HEIGHT", 0), 0, 0, 8192),
            slice_width=_clamped_int(params.get("SLICE_WIDTH", 0), 0, 0, 8192),
            overlap_height_ratio=overlap,
            overlap_width_ratio=overlap,
            object_tile_fraction=_clamped_float(
                params.get("SLICE_OBJECT_TILE_FRACTION", 0.15), 0.15, 0.01, 0.9
            ),
            # auto_object needs a real object scale or it silently degrades to
            # auto_model. Same source/scaling worker.py uses (worker.py:921).
            reference_body_px=_clamped_float(
                float(params.get("REFERENCE_BODY_SIZE", 20.0) or 20.0)
                * float(params.get("RESIZE_FACTOR", 1.0) or 1.0),
                0.0, 0.0, 8192.0,
            ),
            merge_policy=(
                str(params.get("SLICE_MERGE_POLICY", "greedy_nmm")).strip().lower()
                if str(params.get("SLICE_MERGE_POLICY", "greedy_nmm")).strip().lower()
                in {"nms", "nmm", "greedy_nmm"}
                else "greedy_nmm"
            ),
            merge_metric=(
                str(params.get("SLICE_MERGE_METRIC", "ios")).strip().lower()
                if str(params.get("SLICE_MERGE_METRIC", "ios")).strip().lower()
                in {"iou", "ios"}
                else "ios"
            ),
            merge_threshold=_clamped_float(
                params.get("SLICE_MERGE_THRESHOLD", 0.5), 0.5, 0.0, 1.0
            ),
            merge_backend=(
                str(params.get("SLICE_MERGE_BACKEND", "cv2")).strip().lower()
                if str(params.get("SLICE_MERGE_BACKEND", "cv2")).strip().lower()
                in {"cv2", "gpu"}
                else "cv2"
            ),
            perform_standard_pred=bool(params.get("SLICE_PERFORM_STANDARD_PRED", False)),
        )
```

Then add `slice=slice_cfg,` to the `OBBDirectConfig(...)` constructor call (line 557-567).

- [ ] **Step 6: Run tests to verify they pass**

Run: `python -m pytest tests/test_inference_config.py -k slice -v`
Expected: PASS (5 tests).

- [ ] **Step 7: Format and commit**

```bash
make format
git add src/hydra_suite/core/inference/config.py tests/test_inference_config.py
git commit -m "feat(inference): add SliceConfig schema + config round-trip for sliced inference"
```

---

## Task 2: Slice geometry planning (`plan_slices`)

**Files:**
- Create: `src/hydra_suite/core/inference/stages/slicing.py`
- Test: `tests/test_inference_slicing.py`

**Interfaces:**
- Consumes: `SliceConfig` (Task 1).
- Produces:
  - `SlicePlan` dataclass: `tiles: list[tuple[int,int,int,int]]` (x0,y0,x1,y1 per tile), `full_frame: bool`, `slice_wh: tuple[int,int]`, `frame_wh: tuple[int,int]`.
  - `def get_slice_bboxes(frame_h, frame_w, slice_h, slice_w, overlap_h, overlap_w) -> list[tuple[int,int,int,int]]`
  - `def plan_slices(frame_hw: tuple[int,int], slice_cfg: SliceConfig, imgsz: int, roi_mask: np.ndarray | None, ref_object_px: float = 0.0) -> SlicePlan`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_inference_slicing.py
import numpy as np
from hydra_suite.core.inference.config import SliceConfig
from hydra_suite.core.inference.stages.slicing import (
    SlicePlan,
    get_slice_bboxes,
    plan_slices,
)


def test_grid_covers_frame_and_flushes_to_edge():
    # 1000x1000 frame, 640 tiles, 0.2 overlap -> step 512.
    boxes = get_slice_bboxes(1000, 1000, 640, 640, 0.2, 0.2)
    # every pixel covered: min corner at 0, max corner at frame edge.
    xs0 = sorted({b[0] for b in boxes})
    assert xs0[0] == 0
    # last tile flush to right edge (no runt): some tile ends exactly at 1000.
    assert max(b[2] for b in boxes) == 1000
    assert max(b[3] for b in boxes) == 1000
    # no tile exceeds the frame.
    assert all(b[2] <= 1000 and b[3] <= 1000 for b in boxes)
    # tile size preserved (last tile shifted back, not shrunk).
    assert all((b[2] - b[0]) == 640 and (b[3] - b[1]) == 640 for b in boxes)


def test_zero_overlap_tiles_are_disjoint_step_equals_size():
    boxes = get_slice_bboxes(1280, 1280, 640, 640, 0.0, 0.0)
    assert len(boxes) == 4
    assert (0, 0, 640, 640) in boxes


def test_auto_model_uses_imgsz():
    plan = plan_slices((2000, 2000), SliceConfig(enabled=True, geometry_mode="auto_model"),
                        imgsz=1024, roi_mask=None)
    assert plan.slice_wh == (1024, 1024)


def test_custom_uses_explicit_size():
    cfg = SliceConfig(enabled=True, geometry_mode="custom", slice_height=512, slice_width=768)
    plan = plan_slices((2000, 2000), cfg, imgsz=1024, roi_mask=None)
    assert plan.slice_wh == (768, 512)  # (w, h)


def test_auto_object_sizes_from_reference():
    # ref object 64px, want it to span 0.1 of the tile -> tile ~640px.
    cfg = SliceConfig(enabled=True, geometry_mode="auto_object", object_tile_fraction=0.1)
    plan = plan_slices((4000, 4000), cfg, imgsz=1024, roi_mask=None, ref_object_px=64.0)
    assert 600 <= plan.slice_wh[0] <= 680


def test_roi_gating_drops_tiles_outside_mask():
    # ROI only in the top-left quadrant.
    mask = np.zeros((1000, 1000), dtype=np.uint8)
    mask[:400, :400] = 1
    cfg = SliceConfig(enabled=True, geometry_mode="custom", slice_height=256, slice_width=256)
    full = plan_slices((1000, 1000), SliceConfig(enabled=True, geometry_mode="custom",
                       slice_height=256, slice_width=256), imgsz=256, roi_mask=None)
    gated = plan_slices((1000, 1000), cfg, imgsz=256, roi_mask=mask)
    assert len(gated.tiles) < len(full.tiles)
    # every kept tile intersects the ROI.
    for x0, y0, x1, y1 in gated.tiles:
        assert mask[y0:y1, x0:x1].any()


def test_perform_standard_pred_flag():
    cfg = SliceConfig(enabled=True, geometry_mode="custom", slice_height=256,
                      slice_width=256, perform_standard_pred=True)
    plan = plan_slices((512, 512), cfg, imgsz=256, roi_mask=None)
    assert plan.full_frame is True
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_inference_slicing.py -v`
Expected: FAIL with `ModuleNotFoundError: ...stages.slicing`.

- [ ] **Step 3: Implement `slicing.py` planning section**

```python
# src/hydra_suite/core/inference/stages/slicing.py
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..config import SliceConfig


@dataclass
class SlicePlan:
    """A memoizable tiling of a fixed-size frame."""

    tiles: list[tuple[int, int, int, int]]  # (x0, y0, x1, y1) per tile
    full_frame: bool  # append one full-frame pass in addition to tiles
    slice_wh: tuple[int, int]  # (w, h) of each tile
    frame_wh: tuple[int, int]  # (w, h) of the source frame


def get_slice_bboxes(
    frame_h: int,
    frame_w: int,
    slice_h: int,
    slice_w: int,
    overlap_h: float,
    overlap_w: float,
) -> list[tuple[int, int, int, int]]:
    """SAHI ``get_slice_bboxes``: fixed-size tiles, last tile flush to the edge.

    Step = slice * (1 - overlap). The final tile in each axis is shifted back so
    its far edge sits exactly on the frame edge (never a shrunken runt tile), so
    two frames of the same size always tile identically.
    """
    slice_w = min(slice_w, frame_w)
    slice_h = min(slice_h, frame_h)
    step_x = max(1, int(slice_w * (1.0 - overlap_w)))
    step_y = max(1, int(slice_h * (1.0 - overlap_h)))

    def _starts(total: int, size: int, step: int) -> list[int]:
        if size >= total:
            return [0]
        starts = list(range(0, total - size + 1, step))
        last = total - size
        if starts[-1] != last:
            starts.append(last)
        return starts

    xs = _starts(frame_w, slice_w, step_x)
    ys = _starts(frame_h, slice_h, step_y)
    return [(x, y, x + slice_w, y + slice_h) for y in ys for x in xs]


def _tile_size(
    slice_cfg: SliceConfig, imgsz: int, ref_object_px: float
) -> tuple[int, int]:
    """Return (w, h) tile size for the configured geometry mode."""
    if slice_cfg.geometry_mode == "custom":
        w = slice_cfg.slice_width if slice_cfg.slice_width > 0 else imgsz
        h = slice_cfg.slice_height if slice_cfg.slice_height > 0 else imgsz
        return int(w), int(h)
    if slice_cfg.geometry_mode == "auto_object" and ref_object_px > 0:
        frac = max(0.01, min(0.9, slice_cfg.object_tile_fraction))
        size = int(round(ref_object_px / frac))
        size = max(64, min(4096, size))
        return size, size
    # auto_model (and auto_object fallback when no ref object is known).
    return int(imgsz), int(imgsz)


def plan_slices(
    frame_hw: tuple[int, int],
    slice_cfg: SliceConfig,
    imgsz: int,
    roi_mask: "np.ndarray | None",
    ref_object_px: float = 0.0,
) -> SlicePlan:
    """Compute the tile plan for one frame size. Cheap; caller memoizes per video."""
    frame_h, frame_w = int(frame_hw[0]), int(frame_hw[1])
    slice_w, slice_h = _tile_size(slice_cfg, imgsz, ref_object_px)
    tiles = get_slice_bboxes(
        frame_h,
        frame_w,
        slice_h,
        slice_w,
        slice_cfg.overlap_height_ratio,
        slice_cfg.overlap_width_ratio,
    )
    if roi_mask is not None:
        h, w = roi_mask.shape[:2]
        kept = []
        for x0, y0, x1, y1 in tiles:
            yy0, yy1 = max(0, y0), min(h, y1)
            xx0, xx1 = max(0, x0), min(w, x1)
            if yy1 > yy0 and xx1 > xx0 and roi_mask[yy0:yy1, xx0:xx1].any():
                kept.append((x0, y0, x1, y1))
        tiles = kept if kept else tiles  # never empty the plan
    return SlicePlan(
        tiles=tiles,
        full_frame=bool(slice_cfg.perform_standard_pred),
        slice_wh=(slice_w, slice_h),
        frame_wh=(frame_w, frame_h),
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_inference_slicing.py -v`
Expected: PASS (8 tests).

- [ ] **Step 5: Format and commit**

```bash
make format
git add src/hydra_suite/core/inference/stages/slicing.py tests/test_inference_slicing.py
git commit -m "feat(inference): slice geometry planning with three modes + ROI gating"
```

---

## Task 3: Merge — cv2 backend (policy × metric) + overlap-band pre-filter

**Files:**
- Create: `src/hydra_suite/core/inference/stages/merge.py`
- Test: `tests/test_inference_merge.py`

**Interfaces:**
- Consumes: `OBBResult` (`result.py`); `_normalize_obb_geometry`, `_corners_from_xywhr`, `_empty_obb_result` from `obb.py`.
- Produces:
  - `def merge_obb_detections(result: OBBResult, *, policy: str, metric: str, threshold: float, backend: str, overlap_bands: np.ndarray | None = None, runtime=None) -> OBBResult` — the public seam.
  - `def _pair_overlap(hull_a, area_a, hull_b, area_b, metric: str) -> float`
  - `def band_membership(corners: np.ndarray, tiles: list[tuple[int,int,int,int]]) -> np.ndarray` — bool (D,), True where a detection's AABB touches ≥2 tiles (overlap band).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_inference_merge.py
import numpy as np
from hydra_suite.core.inference.result import OBBResult
from hydra_suite.core.inference.stages.merge import merge_obb_detections, band_membership


def _obb(cx, cy, w, h, angle=0.0, conf=0.9, cls=0):
    from hydra_suite.core.inference.stages.obb import (
        _normalize_obb_geometry,
        _corners_from_xywhr,
    )
    cx_a = np.array([cx], np.float32); cy_a = np.array([cy], np.float32)
    w_a = np.array([w], np.float32); h_a = np.array([h], np.float32)
    ang, sizes, aspect = _normalize_obb_geometry(w_a, h_a, np.array([angle], np.float32))
    corners = _corners_from_xywhr(cx_a, cy_a, w_a, h_a, ang)
    return OBBResult(
        frame_idx=0,
        centroids=np.stack([cx_a, cy_a], axis=1),
        angles=ang, sizes=sizes,
        shapes=np.stack([sizes, aspect], axis=1),
        confidences=np.array([conf], np.float32),
        corners=corners,
        detection_ids=OBBResult.make_detection_ids(0, 1),
        class_ids=np.array([cls], np.int64),
    )


def _concat(*results):
    from hydra_suite.core.inference.stages.obb import merge_obb_results
    return merge_obb_results(0, list(results))


def test_nms_suppresses_duplicate_keeps_one():
    dup = _concat(_obb(100, 100, 40, 40, conf=0.9), _obb(102, 101, 40, 40, conf=0.5))
    out = merge_obb_detections(dup, policy="nms", metric="iou", threshold=0.5, backend="cv2")
    assert out.num_detections == 1
    assert out.confidences[0] == 0.9  # higher-conf survivor


def test_nmm_unions_truncated_pair_into_one_larger_box():
    # Two half-boxes straddling a boundary: low IoU, high IoS.
    left = _obb(90, 100, 40, 40, conf=0.8)
    right = _obb(120, 100, 40, 40, conf=0.7)
    out = merge_obb_detections(left_right := _concat(left, right),
                               policy="greedy_nmm", metric="ios", threshold=0.5, backend="cv2")
    assert out.num_detections == 1
    # union box wider than either member.
    assert out.sizes[0] > left.sizes[0]
    assert out.confidences[0] == 0.8  # max conf


def test_ios_vs_iou_threshold_behavior():
    a = _obb(100, 100, 60, 20, conf=0.9)   # small box fully inside big one
    b = _obb(100, 100, 60, 60, conf=0.6)
    iou_out = merge_obb_detections(_concat(a, b), policy="nms", metric="iou",
                                   threshold=0.6, backend="cv2")
    ios_out = merge_obb_detections(_concat(a, b), policy="nms", metric="ios",
                                   threshold=0.6, backend="cv2")
    # IoU of nested boxes is low (< 0.6) -> both kept; IoS is 1.0 -> one kept.
    assert iou_out.num_detections == 2
    assert ios_out.num_detections == 1


def test_overlap_zero_returns_input_unchanged():
    r = _concat(_obb(10, 10, 5, 5), _obb(500, 500, 5, 5))
    # merge with threshold 1.0 (no pair can meet it) is a no-op count-wise.
    out = merge_obb_detections(r, policy="greedy_nmm", metric="ios", threshold=1.01, backend="cv2")
    assert out.num_detections == 2


def test_band_membership_flags_only_overlap_region():
    tiles = [(0, 0, 100, 100), (80, 0, 180, 100)]  # overlap band x in [80,100]
    corners = np.array([
        [[10, 10], [20, 10], [20, 20], [10, 20]],   # exclusive to tile 0
        [[85, 40], [95, 40], [95, 50], [85, 50]],   # in the band
    ], dtype=np.float32)
    band = band_membership(corners, tiles)
    assert band.tolist() == [False, True]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_inference_merge.py -v`
Expected: FAIL with `ModuleNotFoundError: ...stages.merge`.

- [ ] **Step 3: Implement `merge.py` cv2 backend**

```python
# src/hydra_suite/core/inference/stages/merge.py
from __future__ import annotations

import cv2
import numpy as np

from ..result import OBBResult
from .obb import (
    _corners_from_xywhr,
    _empty_obb_result,
    _normalize_obb_geometry,
)


def _hull(corners: np.ndarray) -> tuple[np.ndarray, float]:
    p = cv2.convexHull(np.asarray(corners, dtype=np.float32)).reshape(-1, 2)
    return p, float(abs(cv2.contourArea(p)))


def _pair_overlap(
    hull_a: np.ndarray, area_a: float, hull_b: np.ndarray, area_b: float, metric: str
) -> float:
    """IoU or IoS of two convex corner polygons (cv2 intersection area)."""
    if area_a <= 1e-9 or area_b <= 1e-9:
        return 0.0
    try:
        inter, _ = cv2.intersectConvexConvex(hull_a, hull_b)
        inter = float(max(0.0, inter))
    except Exception:
        inter = 0.0
    if metric == "ios":
        denom = min(area_a, area_b)
    else:  # iou
        denom = area_a + area_b - inter
    return float(inter / denom) if denom > 1e-9 else 0.0


def band_membership(
    corners: np.ndarray, tiles: list[tuple[int, int, int, int]]
) -> np.ndarray:
    """(D,) bool: True where a detection's AABB touches >= 2 tiles (overlap band).

    A detection inside a single tile's exclusive region cannot have a cross-tile
    duplicate, so only band members need the O(n^2) merge.
    """
    n = corners.shape[0]
    if n == 0:
        return np.zeros(0, dtype=bool)
    amin = corners.min(axis=1)  # (D, 2)
    amax = corners.max(axis=1)
    counts = np.zeros(n, dtype=np.int32)
    for tx0, ty0, tx1, ty1 in tiles:
        hit = (
            (amax[:, 0] > tx0)
            & (amin[:, 0] < tx1)
            & (amax[:, 1] > ty0)
            & (amin[:, 1] < ty1)
        )
        counts += hit.astype(np.int32)
    return counts >= 2


def _union_obb(members: OBBResult, idxs: list[int], frame_idx: int) -> tuple:
    """Union the member corners into one OBB via cv2.minAreaRect.

    Returns (cx, cy, w, h, angle_rad, conf, cls) renormalized through the shared
    geometry pipeline so the merged box is indistinguishable from a native OBB.
    """
    pts = members.corners[idxs].reshape(-1, 2).astype(np.float32)
    (cx, cy), (w, h), angle_deg = cv2.minAreaRect(pts)
    ang, _, _ = _normalize_obb_geometry(
        np.array([w], np.float32),
        np.array([h], np.float32),
        np.array([np.deg2rad(angle_deg)], np.float32),
    )
    conf = float(members.confidences[idxs].max())
    top = idxs[int(np.argmax(members.confidences[idxs]))]
    cls = int(members.class_ids_or_zeros[top])
    return float(cx), float(cy), float(w), float(h), float(ang[0]), conf, cls


def merge_obb_detections(
    result: OBBResult,
    *,
    policy: str,
    metric: str,
    threshold: float,
    backend: str,
    overlap_bands: "np.ndarray | None" = None,
    runtime=None,
) -> OBBResult:
    """Merge cross-tile duplicate detections. cv2 backend is the oracle.

    ``overlap_bands`` (D,) bool restricts the quadratic stage to band members;
    exclusive-region detections pass through untouched. None => all considered.
    """
    n = result.num_detections
    if n <= 1:
        return result
    if backend == "gpu":
        from .merge_gpu import merge_obb_detections_gpu  # lazy; Task 5

        return merge_obb_detections_gpu(
            result, policy=policy, metric=metric, threshold=threshold, runtime=runtime
        )

    if overlap_bands is None:
        band_idx = np.arange(n)
        passthrough_idx = np.array([], dtype=int)
    else:
        band_idx = np.where(overlap_bands)[0]
        passthrough_idx = np.where(~overlap_bands)[0]
    if band_idx.size <= 1:
        return result

    # confidence-descending order over band members.
    order = band_idx[np.argsort(result.confidences[band_idx])[::-1]]
    hulls: dict[int, tuple[np.ndarray, float]] = {}

    def hull(i: int) -> tuple[np.ndarray, float]:
        c = hulls.get(i)
        if c is None:
            c = _hull(result.corners[i])
            hulls[i] = c
        return c

    consumed = np.zeros(n, dtype=bool)
    merged_rows: list[tuple] = []  # unioned OBBs
    keep_single: list[int] = []  # nms survivors / lone members
    for i in order:
        if consumed[i]:
            continue
        group = [int(i)]
        pi, ai = hull(i)
        for j in order:
            if j == i or consumed[j]:
                continue
            pj, aj = hull(j)
            if _pair_overlap(pi, ai, pj, aj, metric) >= threshold:
                consumed[j] = True
                group.append(int(j))
        consumed[i] = True
        if policy == "nms" or len(group) == 1:
            keep_single.append(int(i))  # highest-conf member of the group
        else:  # nmm / greedy_nmm -> union
            merged_rows.append(_union_obb(result, group, result.frame_idx))

    return _assemble(result, keep_single, passthrough_idx.tolist(), merged_rows)


def _assemble(
    src: OBBResult,
    keep_single: list[int],
    passthrough: list[int],
    merged_rows: list[tuple],
) -> OBBResult:
    """Concatenate nms/lone survivors + passthrough + unioned rows into one OBBResult."""
    keep = sorted(keep_single + passthrough)
    cxs, cys, ws, hs, angs, confs, clss = [], [], [], [], [], [], []
    for i in keep:
        cxs.append(src.centroids[i, 0]); cys.append(src.centroids[i, 1])
        # recover w,h from sizes/shapes is lossy; reuse stored corners' minAreaRect.
        (mcx, mcy), (mw, mh), mdeg = cv2.minAreaRect(
            src.corners[i].astype(np.float32)
        )
        ws.append(mw); hs.append(mh)
        angs.append(src.angles[i]); confs.append(src.confidences[i])
        clss.append(int(src.class_ids_or_zeros[i]))
    for (cx, cy, w, h, ang, conf, cls) in merged_rows:
        cxs.append(cx); cys.append(cy); ws.append(w); hs.append(h)
        angs.append(ang); confs.append(conf); clss.append(cls)
    if not cxs:
        return _empty_obb_result(src.frame_idx)
    cx = np.asarray(cxs, np.float32); cy = np.asarray(cys, np.float32)
    w = np.asarray(ws, np.float32); h = np.asarray(hs, np.float32)
    ang = np.asarray(angs, np.float32)
    ang_fixed, sizes, aspect = _normalize_obb_geometry(w, h, ang)
    corners = _corners_from_xywhr(cx, cy, w, h, ang_fixed)
    m = len(cxs)
    return OBBResult(
        frame_idx=src.frame_idx,
        centroids=np.stack([cx, cy], axis=1),
        angles=ang_fixed, sizes=sizes,
        shapes=np.stack([sizes, aspect], axis=1),
        confidences=np.asarray(confs, np.float32),
        corners=corners,
        detection_ids=OBBResult.make_detection_ids(src.frame_idx, m),
        class_ids=np.asarray(clss, np.int64),
    )
```

Note: `_assemble` re-derives (w,h) for kept singles via `minAreaRect` of their stored corners because `OBBResult` stores `sizes`/`aspect`, not raw (w,h); this reconstruction is exact for rectangles.

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_inference_merge.py -v`
Expected: PASS (5 tests).

- [ ] **Step 5: Format and commit**

```bash
make format
git add src/hydra_suite/core/inference/stages/merge.py tests/test_inference_merge.py
git commit -m "feat(inference): cv2 merge backend (nms/nmm/greedy_nmm x iou/ios) + overlap-band pre-filter"
```

---

## Task 4: GPU rotated-IoU pairwise kernel (`utils/rotated_iou.py`)

**Files:**
- Create: `src/hydra_suite/utils/rotated_iou.py`
- Test: `tests/test_utils_rotated_iou.py`

**Interfaces:**
- Produces: `def pairwise_obb_overlap(corners: torch.Tensor, metric: str = "iou") -> torch.Tensor` — `corners` (N,4,2) on any device; returns (N,N) overlap matrix (metric ∈ {iou, ios}), diagonal = 1.

> **HARD REQUIREMENT — genuinely vectorized, and it must beat cv2 or it does not ship.**
> This kernel exists ONLY to be faster than the cv2 merge on the native-cuda path.
> A Python `for i: for j:` loop over pairs (or over polygon vertices) issuing scalar
> tensor ops is **NOT acceptable**: at ~200 band members that is ~400k kernel
> launches and would be *slower* than cv2, inverting the entire rationale.
>
> Implement the clipping as **batched tensor ops over all N² pairs at once**:
> broadcast subject polygons to `(N, N, K, 2)` and clip against all 4 edges of
> every other box simultaneously (Sutherland–Hodgman is 4 sequential edge passes,
> each fully vectorized across pairs and vertices — a fixed 4-iteration Python
> loop over *edges* is fine; loops over pairs or vertices are not). Convex
> quad∩quad yields ≤ 8 vertices, so pad to a fixed K=8 and mask.
> Shoelace area is then one batched reduction.
>
> **Perf gate (Step 4a below) is part of this task's definition of done.**

- [ ] **Step 1: Write the failing test**

```python
# tests/test_utils_rotated_iou.py
import cv2
import numpy as np
import torch
from hydra_suite.utils.rotated_iou import pairwise_obb_overlap


def _cv2_overlap(ca, cb, metric):
    pa = cv2.convexHull(ca.astype(np.float32)).reshape(-1, 2)
    pb = cv2.convexHull(cb.astype(np.float32)).reshape(-1, 2)
    aa = abs(cv2.contourArea(pa)); ab = abs(cv2.contourArea(pb))
    inter, _ = cv2.intersectConvexConvex(pa, pb)
    inter = max(0.0, inter)
    denom = min(aa, ab) if metric == "ios" else aa + ab - inter
    return inter / denom if denom > 1e-9 else 0.0


def _rect(cx, cy, w, h, deg):
    box = cv2.boxPoints(((cx, cy), (w, h), deg))
    return box.astype(np.float32)


def test_matches_cv2_within_tolerance_axis_aligned():
    corners = np.stack([_rect(100, 100, 40, 40, 0), _rect(120, 100, 40, 40, 0)])
    t = torch.from_numpy(corners)
    for metric in ("iou", "ios"):
        m = pairwise_obb_overlap(t, metric=metric).numpy()
        expected = _cv2_overlap(corners[0], corners[1], metric)
        assert abs(m[0, 1] - expected) < 1e-2
        assert abs(m[1, 0] - expected) < 1e-2


def test_matches_cv2_rotated_random():
    rng = np.random.default_rng(0)
    corners = np.stack([
        _rect(*rng.uniform([80, 80, 30, 30, 0], [140, 140, 60, 60, 90]))
        for _ in range(6)
    ]).astype(np.float32)
    m = pairwise_obb_overlap(torch.from_numpy(corners), metric="iou").numpy()
    for i in range(6):
        for j in range(6):
            if i == j:
                continue
            assert abs(m[i, j] - _cv2_overlap(corners[i], corners[j], "iou")) < 3e-2


def test_diagonal_is_one_and_empty_ok():
    corners = np.stack([_rect(100, 100, 40, 40, 15)])
    m = pairwise_obb_overlap(torch.from_numpy(corners), metric="iou")
    assert abs(float(m[0, 0]) - 1.0) < 1e-3
    empty = pairwise_obb_overlap(torch.zeros((0, 4, 2)), metric="iou")
    assert empty.shape == (0, 0)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_utils_rotated_iou.py -v`
Expected: FAIL with `ModuleNotFoundError: ...utils.rotated_iou`.

- [ ] **Step 3: Implement `rotated_iou.py` — fully vectorized**

Write `src/hydra_suite/utils/rotated_iou.py` exposing
`pairwise_obb_overlap(corners: torch.Tensor, metric: str = "iou") -> torch.Tensor`.
Torch-only (no cv2, no numpy round-trip), device-preserving, runs on CPU/CUDA/MPS.

Algorithm — **every step batched across all N² pairs; no Python loop over pairs
or vertices** (a fixed 4-iteration loop over clip *edges* is the only Python loop
allowed):

1. **Orient**: make every quad counter-clockwise via batched signed area
   (shoelace over `(N,4,2)`); flip the ones with negative area using
   `torch.where`, not a Python branch.
2. **Broadcast**: build subject `(N, N, K, 2)` (K padded to 8) and clip-box
   `(N, N, 4, 2)` so pair `(i, j)` is subject `i` against clip `j`.
3. **Clip**: 4 sequential Sutherland–Hodgman edge passes. Each pass computes,
   for all pairs and all vertices at once: the inside-test cross product,
   the edge-intersection point, and a validity mask. Carry results in fixed-size
   `(N, N, 8, 2)` tensors with a companion `(N, N, 8)` bool mask instead of
   variable-length vertex lists. Guard the intersection denominator with
   `torch.where(|denom| < 1e-9, ...)` — never a Python `if`.
4. **Area**: masked shoelace reduction over the clipped polygons → `(N, N)`
   intersection areas. Zero out entries with < 3 valid vertices.
5. **Metric**: `iou = inter / (a_i + a_j - inter)`; `ios = inter / min(a_i, a_j)`.
   Guard zero denominators with `torch.where`. Set the diagonal to 1.
6. `N == 0` returns a `(0, 0)` tensor.

Numerical note: use the input dtype (float32); an inside-test epsilon of `-1e-6`
matches the tolerance the tests assert against cv2.

- [ ] **Step 4: Run correctness tests**

Run: `python -m pytest tests/test_utils_rotated_iou.py -v`
Expected: PASS (3 tests).

- [ ] **Step 4a: Perf gate — MUST be faster than cv2, or report BLOCKED**

This kernel's only justification is beating cv2 on the native-cuda path. Verify
it, and put the numbers in your report. Write `/tmp/sahi_perf_check.py`:

```python
import time, numpy as np, torch, cv2
from hydra_suite.utils.rotated_iou import pairwise_obb_overlap

rng = np.random.default_rng(0)
N = 200
corners = np.stack([
    cv2.boxPoints(((float(rng.uniform(0, 2000)), float(rng.uniform(0, 2000))),
                   (float(rng.uniform(20, 80)), float(rng.uniform(20, 80))),
                   float(rng.uniform(0, 180)))).astype(np.float32)
    for _ in range(N)
])

def cv2_matrix(c):
    hulls = [cv2.convexHull(x).reshape(-1, 2) for x in c]
    areas = [abs(cv2.contourArea(h)) for h in hulls]
    m = np.zeros((len(c), len(c)), np.float32)
    for i in range(len(c)):
        for j in range(i + 1, len(c)):
            inter, _ = cv2.intersectConvexConvex(hulls[i], hulls[j])
            inter = max(0.0, inter)
            d = areas[i] + areas[j] - inter
            m[i, j] = m[j, i] = inter / d if d > 1e-9 else 0.0
    return m

t = torch.from_numpy(corners)
pairwise_obb_overlap(t)                      # warm up
t0 = time.perf_counter(); pairwise_obb_overlap(t); gpu_s = time.perf_counter() - t0
t0 = time.perf_counter(); cv2_matrix(corners); cv2_s = time.perf_counter() - t0
print(f"N={N}  vectorized={gpu_s*1000:.1f}ms  cv2={cv2_s*1000:.1f}ms  speedup={cv2_s/gpu_s:.2f}x")
assert gpu_s < cv2_s, f"NOT FASTER: {gpu_s*1000:.1f}ms vs cv2 {cv2_s*1000:.1f}ms"
```

Run: `python /tmp/sahi_perf_check.py`
Expected: prints a speedup > 1.0 and the assert passes.

**If it is NOT faster on this CPU/MPS box**, do not silently continue: note the
measured numbers and report `DONE_WITH_CONCERNS`, stating that the kernel is
CPU/MPS-measured only and its real target is CUDA. Do NOT loosen the assert to
make it pass.

- [ ] **Step 5: Format and commit**

```bash
make format
git add src/hydra_suite/utils/rotated_iou.py tests/test_utils_rotated_iou.py
git commit -m "feat(utils): torch-only batched rotated-box pairwise IoU/IoS kernel"
```

---

## Task 5: Merge — gpu backend (grouping + union via obb_from_mask kernel)

**Files:**
- Create: `src/hydra_suite/core/inference/stages/merge_gpu.py`
- Test: `tests/test_inference_merge.py` (append equivalence tests)

**Interfaces:**
- Consumes: `pairwise_obb_overlap` (Task 4); `rotated_rect_from_masks` (`utils/obb_from_mask.py`); `_normalize_obb_geometry`, `_corners_from_xywhr`, `_empty_obb_result` (`obb.py`); `merge_obb_detections` cv2 path (Task 3) as fallback/oracle.
- Produces: `def merge_obb_detections_gpu(result, *, policy, metric, threshold, runtime) -> OBBResult`.

- [ ] **Step 1: Write the failing test (append)**

```python
# tests/test_inference_merge.py (append)
import pytest


@pytest.mark.parametrize("policy", ["nms", "greedy_nmm"])
def test_gpu_backend_matches_cv2_within_tolerance(policy):
    rng = np.random.default_rng(1)
    parts = []
    for _ in range(8):
        parts.append(_obb(*rng.uniform([60, 60, 30, 30], [200, 200, 60, 60]),
                          angle=float(rng.uniform(0, np.pi)),
                          conf=float(rng.uniform(0.3, 0.99))))
    r = _concat(*parts)
    cv2_out = merge_obb_detections(r, policy=policy, metric="ios",
                                   threshold=0.5, backend="cv2")
    gpu_out = merge_obb_detections(r, policy=policy, metric="ios",
                                   threshold=0.5, backend="gpu")
    # same count (grouping decisions agree) within tolerance.
    assert gpu_out.num_detections == cv2_out.num_detections
    # centroids of survivors match within a few px after sorting.
    cc = np.sort(cv2_out.centroids.sum(axis=1))
    gc = np.sort(gpu_out.centroids.sum(axis=1))
    assert np.allclose(cc, gc, atol=3.0)


def test_gpu_backend_single_member_passthrough():
    r = _obb(100, 100, 40, 40, conf=0.9)
    out = merge_obb_detections(r, policy="greedy_nmm", metric="ios",
                               threshold=0.5, backend="gpu")
    assert out.num_detections == 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_inference_merge.py -k gpu -v`
Expected: FAIL with `ModuleNotFoundError: ...stages.merge_gpu`.

- [ ] **Step 3: Implement `merge_gpu.py`**

```python
# src/hydra_suite/core/inference/stages/merge_gpu.py
from __future__ import annotations

import numpy as np
import torch

from ..result import OBBResult
from .obb import _corners_from_xywhr, _empty_obb_result, _normalize_obb_geometry


def _greedy_groups(matrix: np.ndarray, order: np.ndarray, threshold: float) -> list[list[int]]:
    """Greedy grouping on a synced (N,N) overlap matrix (pure index bookkeeping)."""
    n = matrix.shape[0]
    consumed = np.zeros(n, dtype=bool)
    groups: list[list[int]] = []
    for i in order:
        if consumed[i]:
            continue
        group = [int(i)]
        consumed[i] = True
        for j in order:
            if j == i or consumed[j]:
                continue
            if matrix[i, j] >= threshold:
                consumed[j] = True
                group.append(int(j))
        groups.append(group)
    return groups


def merge_obb_detections_gpu(
    result: OBBResult, *, policy: str, metric: str, threshold: float, runtime
) -> OBBResult:
    """GPU-native merge: pairwise matrix on device, greedy grouping on CPU,
    union via the shared angle-search kernel. Matches cv2 within tolerance.
    """
    from hydra_suite.utils.rotated_iou import pairwise_obb_overlap

    n = result.num_detections
    if n <= 1:
        return result
    device = getattr(runtime, "device", "cpu") if runtime is not None else "cpu"
    corners_t = torch.as_tensor(result.corners, dtype=torch.float32, device=device)
    matrix = pairwise_obb_overlap(corners_t, metric=metric).cpu().numpy()
    order = np.argsort(result.confidences)[::-1]
    groups = _greedy_groups(matrix, order, threshold)

    cxs, cys, ws, hs, angs, confs, clss = [], [], [], [], [], [], []
    for group in groups:
        if policy == "nms" or len(group) == 1:
            top = group[int(np.argmax(result.confidences[group]))]
            box = result.corners[top].astype(np.float32)
            (mcx, mcy), (mw, mh) = _min_rect(box)
            cxs.append(mcx); cys.append(mcy); ws.append(mw); hs.append(mh)
            angs.append(result.angles[top]); confs.append(result.confidences[top])
            clss.append(int(result.class_ids_or_zeros[top]))
        else:
            pts = torch.as_tensor(
                result.corners[group].reshape(-1, 2), dtype=torch.float32, device=device
            )
            (ucx, ucy, uw, uh, uang) = _union_via_kernel(pts, device)
            cxs.append(ucx); cys.append(ucy); ws.append(uw); hs.append(uh)
            angs.append(uang)
            confs.append(float(result.confidences[group].max()))
            top = group[int(np.argmax(result.confidences[group]))]
            clss.append(int(result.class_ids_or_zeros[top]))

    if not cxs:
        return _empty_obb_result(result.frame_idx)
    cx = np.asarray(cxs, np.float32); cy = np.asarray(cys, np.float32)
    w = np.asarray(ws, np.float32); h = np.asarray(hs, np.float32)
    ang = np.asarray(angs, np.float32)
    ang_fixed, sizes, aspect = _normalize_obb_geometry(w, h, ang)
    corners = _corners_from_xywhr(cx, cy, w, h, ang_fixed)
    m = len(cxs)
    return OBBResult(
        frame_idx=result.frame_idx,
        centroids=np.stack([cx, cy], axis=1),
        angles=ang_fixed, sizes=sizes,
        shapes=np.stack([sizes, aspect], axis=1),
        confidences=np.asarray(confs, np.float32),
        corners=corners,
        detection_ids=OBBResult.make_detection_ids(result.frame_idx, m),
        class_ids=np.asarray(clss, np.int64),
    )


def _min_rect(box: np.ndarray) -> tuple:
    """Axis metrics of a (4,2) quad: center + side lengths (rectangle-exact)."""
    cx = float(box[:, 0].mean()); cy = float(box[:, 1].mean())
    e0 = float(np.linalg.norm(box[1] - box[0]))
    e1 = float(np.linalg.norm(box[2] - box[1]))
    return (cx, cy), (e0, e1)


def _union_via_kernel(pts: torch.Tensor, device, num_angles: int = 64) -> tuple:
    """Tightest rotated rect over member corner points — exact, no rasterization.

    Same angle-projection idea as ``utils/obb_from_mask.rotated_rect_from_masks``,
    but applied DIRECTLY to the corner point set instead of a rasterized mask, so
    there is no grid-quantization error (rasterizing to a 64px grid would cost
    ~1-3px of accuracy for no benefit — we already have exact points).

    Projects all points onto ``num_angles`` candidate axes at once, takes the
    axis whose bounding extent has minimum area, and reconstructs the rect
    centered on that extent's midpoint. Fully vectorized: one (num_angles, P)
    matmul, no Python loop over angles or points.
    """
    angles = torch.linspace(
        0.0, float(torch.pi), num_angles + 1, device=device, dtype=torch.float32
    )[:num_angles]                                        # (A,)
    cos_a, sin_a = torch.cos(angles), torch.sin(angles)
    # Project every point onto every candidate axis and its perpendicular.
    u = pts[:, 0][None, :] * cos_a[:, None] + pts[:, 1][None, :] * sin_a[:, None]
    v = -pts[:, 0][None, :] * sin_a[:, None] + pts[:, 1][None, :] * cos_a[:, None]
    umin, umax = u.min(dim=1).values, u.max(dim=1).values   # (A,)
    vmin, vmax = v.min(dim=1).values, v.max(dim=1).values
    w_all, h_all = umax - umin, vmax - vmin
    best = int(torch.argmin(w_all * h_all))
    ang = float(angles[best])
    uw, uh = float(w_all[best]), float(h_all[best])
    # Centre in rotated frame -> back to frame coords.
    uc = (umin[best] + umax[best]) * 0.5
    vc = (vmin[best] + vmax[best]) * 0.5
    ucx = float(uc * torch.cos(angles[best]) - vc * torch.sin(angles[best]))
    ucy = float(uc * torch.sin(angles[best]) + vc * torch.cos(angles[best]))
    return ucx, ucy, uw, uh, ang
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_inference_merge.py -k gpu -v`
Expected: PASS. If tolerance fails on the union path, widen `atol` to 4.0 in the test (documented tolerance, not bit-parity) — do NOT loosen the count assertion.

- [ ] **Step 5: Format and commit**

```bash
make format
git add src/hydra_suite/core/inference/stages/merge_gpu.py tests/test_inference_merge.py
git commit -m "feat(inference): gpu merge backend (pairwise matrix + kernel union), cv2-validated"
```

---

## Task 6: `run_direct_sliced` orchestration + dispatch hook (CPU / gpu_fast → OBBResult)

**Files:**
- Modify: `src/hydra_suite/core/inference/stages/slicing.py` (add orchestration)
- Modify: `src/hydra_suite/core/inference/stages/obb.py` (`run_obb` dispatch, lazy import)
- Test: `tests/test_inference_slicing.py` (append)

**Interfaces:**
- Consumes: `plan_slices`/`SlicePlan` (Task 2); `merge_obb_detections`/`band_membership` (Task 3); from `obb.py`: `_resolve_imgsz`, `_extract_obb_from_boxes`, `_extract_obb_from_masks`, `extract_obb_result`, `_apply_raw_detection_cap`, `merge_obb_results`, `_empty_obb_result`, `_RawOBBTensors`.
- Produces: `def run_direct_sliced(frames, model, config, runtime) -> list[OBBResult | _RawOBBTensors]` — same return contract as `_run_direct`.

- [ ] **Step 1: Write the failing test (append)**

```python
# tests/test_inference_slicing.py (append)
import types
import math
import torch
from hydra_suite.core.inference.config import OBBConfig, OBBDirectConfig, SliceConfig
from hydra_suite.core.inference.stages.obb import run_obb, OBBModels
from hydra_suite.core.inference.stages.slicing import run_direct_sliced


class _FakeRuntime:
    device = "cpu"
    tensor_on_cuda = False


class _FakeYOLO:
    """Stub: returns one obb detection at a fixed frame-space point per tile that
    covers (200,200). imgsz reported = 256 so slice_size==imgsz exact path."""
    imgsz = 256
    overrides = {"imgsz": 256}

    def predict(self, source, **kw):
        # `source` is a list of tile images; emit a detection only when the tile
        # is the one containing (200,200) — detect straddle via image content.
        results = []
        for img in source:
            r = types.SimpleNamespace()
            # tile is 256x256; put a detection at local (60,60) always.
            r.obb = _FakeOBB(cx=60, cy=60, w=30, h=30)
            results.append(r)
        return results


class _FakeOBB:
    def __init__(self, cx, cy, w, h):
        import numpy as np
        self._xywhr = np.array([[cx, cy, w, h, 0.0]], np.float32)
        self._conf = np.array([0.9], np.float32)
    def __len__(self): return 1
    @property
    def xywhr(self):
        import torch
        return torch.from_numpy(self._xywhr)
    @property
    def conf(self):
        import torch
        return torch.from_numpy(self._conf)
    @property
    def cls(self):
        import torch
        return torch.zeros(1)


def _direct_cfg(enabled, **slice_kw):
    return OBBConfig(
        mode="direct",
        direct=OBBDirectConfig(
            model_path="m.pt", model_task="obb",
            slice=SliceConfig(enabled=enabled, geometry_mode="auto_model", **slice_kw),
        ),
        confidence_threshold=0.0, raw_detection_cap=0, max_detections=100,
    )


def test_sliced_cpu_obb_remaps_into_frame_space():
    frame = np.zeros((300, 300, 3), np.uint8)
    cfg = _direct_cfg(True, overlap_height_ratio=0.0, overlap_width_ratio=0.0)
    out = run_direct_sliced([frame], _FakeYOLO(), cfg, _FakeRuntime())
    assert len(out) == 1
    res = out[0]
    # detections remapped: each tile contributes one det at tile_x0+60, tile_y0+60.
    assert res.num_detections >= 1
    # at least one detection lands beyond a single tile's local coords (proves offset).
    assert res.centroids[:, 0].max() > 60


def test_enabled_false_dispatch_uses_plain_run_direct(monkeypatch):
    frame = np.zeros((300, 300, 3), np.uint8)
    cfg = _direct_cfg(False)
    models = OBBModels(mode="direct", direct_model=_FakeYOLO())
    called = {"sliced": False}
    import hydra_suite.core.inference.stages.obb as obbmod
    monkeypatch.setattr(
        "hydra_suite.core.inference.stages.slicing.run_direct_sliced",
        lambda *a, **k: called.__setitem__("sliced", True) or [],
    )
    run_obb([frame], models, cfg, _FakeRuntime())
    assert called["sliced"] is False  # disabled -> never dispatched


def test_enabled_true_dispatches_to_sliced(monkeypatch):
    frame = np.zeros((300, 300, 3), np.uint8)
    cfg = _direct_cfg(True)
    models = OBBModels(mode="direct", direct_model=_FakeYOLO())
    marker = object()
    monkeypatch.setattr(
        "hydra_suite.core.inference.stages.slicing.run_direct_sliced",
        lambda *a, **k: [marker],
    )
    out = run_obb([frame], models, cfg, _FakeRuntime())
    assert out == [marker]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_inference_slicing.py -k "sliced or dispatch" -v`
Expected: FAIL with `ImportError: cannot import name 'run_direct_sliced'`.

- [ ] **Step 3: Implement `run_direct_sliced` (CPU/numpy + gpu_fast → OBBResult path)**

Append to `slicing.py`:

```python
from typing import Any


def _extract_tile(result: Any, model_task: str, config, tile_local_idx: int):
    """Run the correct per-task extractor on one tile's ultralytics result.

    Returns an OBBResult in the TILE's local coordinate space (frame_idx is a
    throwaway tile index; re-stamped after remap).
    """
    from .obb import (
        _extract_obb_from_boxes,
        _extract_obb_from_masks,
        extract_obb_result,
    )
    import math as _math

    if model_task == "detect":
        return _extract_obb_from_boxes(
            result, tile_local_idx, _math.radians(config.direct.fixed_angle_deg)
        )
    if model_task == "segment":
        return _extract_obb_from_masks(
            result, tile_local_idx, config.raw_detection_cap,
            num_angles=config.direct.seg_num_angles,
            crop_size=config.direct.seg_crop_size,
            pad_ratio=config.direct.seg_pad_ratio,
            mask_threshold=config.direct.seg_mask_threshold,
        )
    return extract_obb_result(result, tile_local_idx)


def _offset_result(res, x0: int, y0: int, frame_idx: int):
    """Return a copy of ``res`` with all coordinates shifted by (x0, y0)."""
    from .obb import _empty_obb_result
    if res.num_detections == 0:
        return _empty_obb_result(frame_idx)
    centroids = res.centroids.copy()
    centroids[:, 0] += x0
    centroids[:, 1] += y0
    corners = res.corners.copy()
    corners[..., 0] += x0
    corners[..., 1] += y0
    return OBBResult(
        frame_idx=frame_idx,
        centroids=centroids, angles=res.angles, sizes=res.sizes, shapes=res.shapes,
        confidences=res.confidences, corners=corners,
        detection_ids=OBBResult.make_detection_ids(frame_idx, res.num_detections),
        class_ids=res.class_ids_or_zeros,
    )


def run_direct_sliced(frames, model, config, runtime):
    """Sliced-inference wrapper around the direct predict+extract path.

    Same return contract as ``_run_direct``. This module-level function is
    dispatched from ``run_obb`` when ``config.obb.direct.slice.enabled`` is True.
    CPU/MPS/gpu_fast return ``OBBResult`` per frame; the native-cuda tensor path
    is handled in Task 7.
    """
    from .obb import (
        _apply_raw_detection_cap,
        _resolve_imgsz,
        merge_obb_results,
        _empty_obb_result,
        _RawOBBTensors,
    )
    from .merge import band_membership, merge_obb_detections

    slice_cfg = config.direct.slice
    model_task = config.direct.model_task
    imgsz = _resolve_imgsz(model)

    # Native-cuda tensor path: delegated to Task 7 helper.
    if getattr(runtime, "tensor_on_cuda", False):
        from .slicing_cuda import run_direct_sliced_cuda

        return run_direct_sliced_cuda(frames, model, config, runtime, imgsz)

    # Plan is identical for every frame in the window (same size). Memoize on
    # the first frame's shape.
    first = frames[0]
    frame_hw = (int(first.shape[0]), int(first.shape[1]))
    plan = plan_slices(
        frame_hw, slice_cfg, imgsz, None,
        ref_object_px=slice_cfg.reference_body_px,
    )

    # Build the flattened tile job list across all frames.
    jobs = []  # (frame_idx, x0, y0, x1, y1) ; x1==-1 marks a full-frame job
    for fi, frame in enumerate(frames):
        for (x0, y0, x1, y1) in plan.tiles:
            jobs.append((fi, x0, y0, x1, y1))
        if plan.full_frame:
            jobs.append((fi, 0, 0, -1, -1))

    # Crop every tile (numpy views; contiguous copy for the predict call).
    tile_imgs = []
    for (fi, x0, y0, x1, y1) in jobs:
        if x1 < 0:
            tile_imgs.append(frames[fi])
        else:
            tile_imgs.append(np.ascontiguousarray(frames[fi][y0:y1, x0:x1]))

    conf_floor = config.direct.confidence_floor
    results = model.predict(
        tile_imgs, conf=conf_floor, iou=1.0,
        classes=config.target_classes or None, verbose=False, device=runtime.device,
    )

    # Extract + offset-remap, grouped by source frame.
    per_frame: dict[int, list] = {fi: [] for fi in range(len(frames))}
    for (job, res) in zip(jobs, results):
        fi, x0, y0, x1, y1 = job
        local = _extract_tile(res, model_task, config, fi)
        per_frame[fi].append(_offset_result(local, max(0, x0), max(0, y0), fi))

    out = []
    overlap = max(slice_cfg.overlap_height_ratio, slice_cfg.overlap_width_ratio)
    for fi in range(len(frames)):
        concat = merge_obb_results(fi, per_frame[fi])
        if concat.num_detections <= 1 or overlap <= 0.0:
            merged = concat
        else:
            bands = band_membership(concat.corners, plan.tiles)
            merged = merge_obb_detections(
                concat,
                policy=slice_cfg.merge_policy,
                metric=slice_cfg.merge_metric,
                threshold=slice_cfg.merge_threshold,
                backend="cv2",  # non-cuda paths always cv2
                overlap_bands=bands,
                runtime=runtime,
            )
        out.append(_apply_raw_detection_cap(merged, config.raw_detection_cap))
    return out
```

- [ ] **Step 4: Add the dispatch hook in `obb.py`**

In `run_obb` (`obb.py:457`), replace the `if models.mode == "direct":` block:

```python
    if models.mode == "direct":
        slice_cfg = getattr(config.direct, "slice", None) if config.direct else None
        if slice_cfg is not None and slice_cfg.enabled:
            from .slicing import run_direct_sliced  # lazy: avoids import cycle

            return run_direct_sliced(frames, models.direct_model, config, runtime)
        return _run_direct(frames, models.direct_model, config, runtime)
    return _run_sequential(frames, models, config, runtime)
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `python -m pytest tests/test_inference_slicing.py -v`
Expected: PASS. Also run the full stage suite for regressions:
Run: `python -m pytest tests/test_inference_stages_obb.py -q`
Expected: PASS (no regressions — disabled path untouched).

- [ ] **Step 6: Format and commit**

```bash
make format
git add src/hydra_suite/core/inference/stages/slicing.py src/hydra_suite/core/inference/stages/obb.py tests/test_inference_slicing.py
git commit -m "feat(inference): run_direct_sliced orchestration + run_obb dispatch hook"
```

---

## Task 7: Native-cuda `_RawOBBTensors` preservation + minimal sync

**Files:**
- Create: `src/hydra_suite/core/inference/stages/slicing_cuda.py`
- Test: `tests/test_inference_slicing.py` (append; CPU-simulated cuda-tensor path)

**Interfaces:**
- Consumes: `run_direct_sliced` cuda delegation (Task 6); `_gpu_letterbox_batch`, `_extract_raw_tensors*`, `_RawOBBTensors`, `materialize_tensors` (`obb.py`); `merge_obb_detections` (Task 3).
- Produces: `def run_direct_sliced_cuda(frames, model, config, runtime, imgsz) -> list[_RawOBBTensors | OBBResult]`.

**Design note:** `overlap == 0` → concatenate `_RawOBBTensors` on-device (no merge, no sync). `overlap > 0` → materialize only band members for the merge; concatenate merged band with exclusive-region detections. Since `_RawOBBTensors` carries no frame-space offset, remap adds `(x0,y0)` to `xywhr[:, :2]` and `corners` on-device.

- [ ] **Step 1: Write the failing test (append)**

```python
# tests/test_inference_slicing.py (append)
class _FakeCudaRuntime:
    device = "cpu"          # simulate: real cuda uses "cuda", tensors stay torch
    tensor_on_cuda = True


class _FakeYOLOCudaTensors:
    """Predict returns results whose .obb yields torch tensors (device-resident sim)."""
    imgsz = 256
    overrides = {"imgsz": 256}

    def predict(self, source, **kw):
        results = []
        # source is a batched (B,3,imgsz,imgsz) tensor on the cuda path.
        b = source.shape[0] if hasattr(source, "shape") else len(source)
        for _ in range(b):
            r = types.SimpleNamespace()
            r.obb = _FakeOBB(cx=60, cy=60, w=30, h=30)
            results.append(r)
        return results


def test_cuda_overlap_zero_returns_raw_tensors():
    from hydra_suite.core.inference.stages.obb import _RawOBBTensors
    frame = torch.zeros((300, 300, 3), dtype=torch.uint8)  # HWC uint8 (cuda sim)
    cfg = _direct_cfg(True, overlap_height_ratio=0.0, overlap_width_ratio=0.0)
    from hydra_suite.core.inference.stages.slicing import run_direct_sliced
    out = run_direct_sliced([frame], _FakeYOLOCudaTensors(), cfg, _FakeCudaRuntime())
    assert len(out) == 1
    assert isinstance(out[0], _RawOBBTensors)  # zero-overlap preserves on-device


def test_cuda_overlap_positive_materializes_and_merges():
    frame = torch.zeros((300, 300, 3), dtype=torch.uint8)
    cfg = _direct_cfg(True, overlap_height_ratio=0.2, overlap_width_ratio=0.2)
    from hydra_suite.core.inference.stages.slicing import run_direct_sliced
    out = run_direct_sliced([frame], _FakeYOLOCudaTensors(), cfg, _FakeCudaRuntime())
    # merged result is an OBBResult (band members synced) — still one per frame.
    assert len(out) == 1
    assert out[0].num_detections >= 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_inference_slicing.py -k cuda -v`
Expected: FAIL with `ModuleNotFoundError: ...stages.slicing_cuda`.

- [ ] **Step 3: Implement `slicing_cuda.py`**

```python
# src/hydra_suite/core/inference/stages/slicing_cuda.py
from __future__ import annotations

import numpy as np
import torch

from ..result import OBBResult
from .slicing import plan_slices


def _remap_raw(raw, x0: int, y0: int):
    """Return a copy of a _RawOBBTensors shifted by (x0, y0) on-device."""
    from .obb import _RawOBBTensors

    if raw.xywhr.shape[0] == 0:
        return raw
    xywhr = raw.xywhr.clone()
    xywhr[:, 0] += x0
    xywhr[:, 1] += y0
    corners = raw.corners.clone()
    corners[..., 0] += x0
    corners[..., 1] += y0
    return _RawOBBTensors(
        frame_idx=raw.frame_idx, xywhr=xywhr, corners=corners, conf=raw.conf, cls=raw.cls
    )


def _concat_raw(parts, frame_idx: int):
    from .obb import _RawOBBTensors

    non_empty = [p for p in parts if p.xywhr.shape[0] > 0]
    if not non_empty:
        dev = parts[0].xywhr.device if parts else torch.device("cpu")
        return _RawOBBTensors(
            frame_idx=frame_idx,
            xywhr=torch.zeros((0, 5), dtype=torch.float32, device=dev),
            corners=torch.zeros((0, 4, 2), dtype=torch.float32, device=dev),
            conf=torch.zeros(0, dtype=torch.float32, device=dev),
            cls=torch.zeros(0, dtype=torch.float32, device=dev),
        )
    return _RawOBBTensors(
        frame_idx=frame_idx,
        xywhr=torch.cat([p.xywhr for p in non_empty], dim=0),
        corners=torch.cat([p.corners for p in non_empty], dim=0),
        conf=torch.cat([p.conf for p in non_empty], dim=0),
        cls=torch.cat([p.cls if p.cls is not None else
                       torch.zeros(p.xywhr.shape[0], device=p.xywhr.device)
                       for p in non_empty], dim=0),
    )


def run_direct_sliced_cuda(frames, model, config, runtime, imgsz):
    """Native-cuda sliced path: preserve _RawOBBTensors; band-only sync when merging."""
    from .obb import (
        _gpu_letterbox_batch,
        _extract_raw_tensors,
        _extract_raw_tensors_from_boxes,
        _extract_raw_tensors_from_masks,
        materialize_tensors,
        _apply_raw_detection_cap,
    )
    from .merge import band_membership, merge_obb_detections

    slice_cfg = config.direct.slice
    model_task = config.direct.model_task
    overlap = max(slice_cfg.overlap_height_ratio, slice_cfg.overlap_width_ratio)
    frame_hw = (int(frames[0].shape[0]), int(frames[0].shape[1]))
    plan = plan_slices(
        frame_hw, slice_cfg, imgsz, None,
        ref_object_px=slice_cfg.reference_body_px,
    )

    # Tile every frame on-device (zero-copy views), collect tiles + provenance.
    jobs, tiles = [], []
    for fi, frame in enumerate(frames):
        for (x0, y0, x1, y1) in plan.tiles:
            jobs.append((fi, x0, y0))
            tiles.append(frame[y0:y1, x0:x1])
        if plan.full_frame:
            jobs.append((fi, 0, 0))
            tiles.append(frame)

    batched, _ = _gpu_letterbox_batch(tiles, imgsz)  # exact-tile: r=1, no pad
    results = model.predict(batched, conf=config.direct.confidence_floor, iou=1.0,
                            classes=config.target_classes or None,
                            verbose=False, device=runtime.device)

    per_frame = {fi: [] for fi in range(len(frames))}
    for (job, res) in zip(jobs, results):
        fi, x0, y0 = job
        if model_task == "detect":
            import math
            raw = _extract_raw_tensors_from_boxes(
                res, fi, math.radians(config.direct.fixed_angle_deg), runtime.device)
        elif model_task == "segment":
            raw = _extract_raw_tensors_from_masks(
                res, fi, runtime.device, config.raw_detection_cap,
                num_angles=config.direct.seg_num_angles,
                crop_size=config.direct.seg_crop_size,
                pad_ratio=config.direct.seg_pad_ratio,
                mask_threshold=config.direct.seg_mask_threshold)
        else:
            raw = _extract_raw_tensors(res, fi, runtime.device)
        per_frame[fi].append(_remap_raw(raw, x0, y0))

    out = []
    for fi in range(len(frames)):
        concat = _concat_raw(per_frame[fi], fi)
        if overlap <= 0.0 or concat.xywhr.shape[0] <= 1:
            out.append(concat)  # preserve _RawOBBTensors end-to-end
            continue
        # overlap > 0: materialize for the cross-tile merge (band-only sync in
        # the cv2 path via band_membership; gpu backend keeps geometry on device).
        materialized = materialize_tensors(concat, config.raw_detection_cap)
        bands = band_membership(materialized.corners, plan.tiles)
        merged = merge_obb_detections(
            materialized, policy=slice_cfg.merge_policy, metric=slice_cfg.merge_metric,
            threshold=slice_cfg.merge_threshold, backend=slice_cfg.merge_backend,
            overlap_bands=bands, runtime=runtime)
        out.append(_apply_raw_detection_cap(merged, config.raw_detection_cap))
    return out
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_inference_slicing.py -k cuda -v`
Expected: PASS (2 tests).

- [ ] **Step 5: Format and commit**

```bash
make format
git add src/hydra_suite/core/inference/stages/slicing_cuda.py tests/test_inference_slicing.py
git commit -m "feat(inference): native-cuda sliced path preserves _RawOBBTensors + band-only sync"
```

---

## Task 8: Batch-shape threading (tile-chunk size → `load_obb_models`)

**Files:**
- Modify: `src/hydra_suite/core/inference/runner.py:145-148`
- Test: `tests/test_inference_obb_artifacts.py` (append)

**Interfaces:**
- Consumes: `SlicePlan`/`plan_slices` (Task 2); `config.detection_batch_size`.
- Produces: TRT engine batch profile sized to `tiles-per-chunk` when slicing enabled (else window depth, unchanged).

**Design note:** When slicing is on, the model is fed **tile** batches, not frame batches, so the TRT dynamic-batch profile must cover the tile-chunk size or `setInputShape` fails (spec §5c). Compute the **real** tile count — do NOT guess a constant. Both inputs are available at load time: the video's frame size and the resolved `imgsz`. Reuse `plan_slices` + `get_slice_bboxes` (Task 2) to count tiles exactly, then the profile size is `len(plan.tiles) (+1 if perform_standard_pred)`, clamped to a sane ceiling.

A hardcoded hint is wrong: a 4K frame at `imgsz=640` yields ~35 tiles/frame, which would exceed any small guess and reintroduce the exact silent failure §5c warns about.

`_load_obb_for_config` needs the frame size. `runner.py` already receives `video_path` in this area (used by `load_bgsub_model`); read the frame size from it with the same probe the pipeline uses. **If the frame size genuinely cannot be resolved at load time, do not invent a constant — report BLOCKED and say so**, so we can decide between plumbing it through or deferring gpu_fast support.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_inference_obb_artifacts.py (append)
from hydra_suite.core.inference.config import (
    InferenceConfig, OBBConfig, OBBDirectConfig, SliceConfig,
)


def test_load_obb_models_receives_tile_batch_when_sliced(monkeypatch):
    import hydra_suite.core.inference.stages.obb as obbmod
    captured = {}

    def _fake_load(config, runtime, *, batch_size=1):
        captured["batch_size"] = batch_size
        from hydra_suite.core.inference.stages.obb import OBBModels
        return OBBModels(mode="direct", direct_model=object())

    monkeypatch.setattr(obbmod, "load_obb_models", _fake_load)

    import hydra_suite.core.inference.runner as runnermod
    from hydra_suite.core.inference.runner import _load_obb_for_config
    # 2160x3840 frame at imgsz 640, overlap 0.2 -> step 512 ->
    # ceil-ish grid of 8 cols x 5 rows = 40 tiles (well above any small guess).
    monkeypatch.setattr(runnermod, "_probe_frame_hw", lambda p: (2160, 3840))
    monkeypatch.setattr(runnermod, "_probe_model_imgsz", lambda p: 640)

    cfg = InferenceConfig(
        detection_source="obb", detection_batch_size=2,
        obb=OBBConfig(mode="direct", direct=OBBDirectConfig(
            model_path="m.pt",
            slice=SliceConfig(enabled=True, geometry_mode="auto_model"))),
    )

    class _RT: device = "cpu"; tensor_on_cuda = False
    _load_obb_for_config(cfg, _RT(), video_path="v.mp4")
    # Must be the REAL tile count, not the 2-frame window and not a small constant.
    from hydra_suite.core.inference.runner import _sliced_tile_batch
    expected = _sliced_tile_batch(cfg, (2160, 3840), 640)
    assert expected > 16, f"test must exercise a >16-tile case, got {expected}"
    assert captured["batch_size"] == expected


def test_load_obb_models_unchanged_when_slicing_disabled(monkeypatch):
    import hydra_suite.core.inference.stages.obb as obbmod
    captured = {}

    def _fake_load(config, runtime, *, batch_size=1):
        captured["batch_size"] = batch_size
        from hydra_suite.core.inference.stages.obb import OBBModels
        return OBBModels(mode="direct", direct_model=object())

    monkeypatch.setattr(obbmod, "load_obb_models", _fake_load)
    from hydra_suite.core.inference.runner import _load_obb_for_config
    cfg = InferenceConfig(
        detection_source="obb", detection_batch_size=2,
        obb=OBBConfig(mode="direct", direct=OBBDirectConfig(model_path="m.pt")),
    )

    class _RT: device = "cpu"; tensor_on_cuda = False
    _load_obb_for_config(cfg, _RT(), video_path="v.mp4")
    assert captured["batch_size"] == 2  # window depth, exactly as before
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_inference_obb_artifacts.py -k tile_batch -v`
Expected: FAIL with `ImportError: cannot import name '_load_obb_for_config'` (or AttributeError).

- [ ] **Step 3: Extract a helper + thread the tile-batch hint**

In `runner.py`, replace the inline `load_obb_models(...)` call (lines 145-148) with a helper:

```python
# Upper bound on the TRT tile-batch profile. Not a guess at tile count (that is
# computed exactly below) — purely a guard so a pathological frame/imgsz ratio
# cannot request an unbuildable engine.
_MAX_TILE_BATCH = 128


def _sliced_tile_batch(config, frame_hw, imgsz):
    """Exact tiles-per-frame for the configured slice plan (+1 for full-frame pass)."""
    from .stages.slicing import plan_slices

    slice_cfg = config.obb.direct.slice
    plan = plan_slices(
        frame_hw, slice_cfg, imgsz, None,
        ref_object_px=slice_cfg.reference_body_px,
    )
    n = len(plan.tiles) + (1 if plan.full_frame else 0)
    return max(1, min(n, _MAX_TILE_BATCH))


def _load_obb_for_config(config, runtime, video_path=None):
    """Load OBB models, sizing the TRT engine batch from the real tile count.

    With slicing on, the model is fed TILE batches, not frame batches, so the
    engine's dynamic profile must cover tiles-per-chunk (spec 5c) — otherwise
    TensorRT fails setInputShape at runtime.
    """
    from .stages.obb import load_obb_models

    batch_size = config.detection_batch_size
    direct = config.obb.direct if config.obb else None
    slice_cfg = getattr(direct, "slice", None) if direct else None
    if slice_cfg is not None and slice_cfg.enabled:
        frame_hw = _probe_frame_hw(video_path)
        imgsz = _probe_model_imgsz(direct.model_path)
        if frame_hw is not None and imgsz:
            batch_size = max(batch_size, _sliced_tile_batch(config, frame_hw, imgsz))
    return load_obb_models(config.obb, runtime, batch_size=batch_size)
```

Implement `_probe_frame_hw(video_path)` (returns `(h, w)` or `None` — use the
same video-open path the pipeline already uses; `cv2.VideoCapture` reading
`CAP_PROP_FRAME_HEIGHT`/`WIDTH` is acceptable, close it immediately) and
`_probe_model_imgsz(model_path)` (reuse
`runtime_artifacts._resolve_imgsz(Path(model_path))`, which already exists at
`runtime_artifacts.py:88`). If either probe fails, fall back to
`config.detection_batch_size` and log a WARNING naming the consequence
(gpu_fast + slicing may need a manually sized engine) — a silent fallback here
is the failure mode §5c is about.

Then at the original call site:

```python
    if config.detection_source == "obb":
        obb = _load_obb_for_config(config, runtime)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_inference_obb_artifacts.py -k tile_batch -v`
Expected: PASS. Also: `python -m pytest tests/test_inference_obb_artifacts.py -q` → no regressions.

- [ ] **Step 5: Format and commit**

```bash
make format
git add src/hydra_suite/core/inference/runner.py tests/test_inference_obb_artifacts.py
git commit -m "feat(inference): size TRT engine batch from tile-chunk when slicing enabled"
```

---

## Task 9: Cache key folds slice params when enabled

**Files:**
- Modify: `src/hydra_suite/core/inference/cache/keys.py:48-63`
- Test: `tests/test_inference_cache_keys.py` (create if absent)

**Interfaces:**
- Consumes: `SliceConfig` (Task 1); existing `detection_cache_key(config: OBBConfig) -> CacheKey`, `_sha`.
- Produces: `detection_cache_key` folds an enabled `SliceConfig` into `config_hash`; disabled → unchanged (empty contribution).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_inference_cache_keys.py
from hydra_suite.core.inference.config import OBBConfig, OBBDirectConfig, SliceConfig
from hydra_suite.core.inference.cache.keys import detection_cache_key


def _cfg(slice_cfg):
    return OBBConfig(mode="direct",
                     direct=OBBDirectConfig(model_path="m.pt", slice=slice_cfg))


def test_disabled_slice_key_equals_no_slice_baseline():
    # Baseline: default (disabled) slice.
    base = detection_cache_key(_cfg(SliceConfig()))
    # A config whose slice is disabled but has non-default *other* fields must
    # still hash identically (disabled => inert).
    other = detection_cache_key(_cfg(SliceConfig(enabled=False, merge_threshold=0.9,
                                                 slice_height=999)))
    assert base.config_hash == other.config_hash


def test_enabling_slice_changes_key():
    off = detection_cache_key(_cfg(SliceConfig(enabled=False)))
    on = detection_cache_key(_cfg(SliceConfig(enabled=True)))
    assert off.config_hash != on.config_hash


def test_slice_param_change_changes_key_when_enabled():
    a = detection_cache_key(_cfg(SliceConfig(enabled=True, merge_threshold=0.5)))
    b = detection_cache_key(_cfg(SliceConfig(enabled=True, merge_threshold=0.6)))
    assert a.config_hash != b.config_hash
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_inference_cache_keys.py -v`
Expected: FAIL (`test_enabling_slice_changes_key` — key doesn't yet depend on slice).

- [ ] **Step 3: Fold slice params into `detection_cache_key`**

In `keys.py`, modify `detection_cache_key` (lines 48-63) to compute a slice contribution:

```python
def detection_cache_key(config: OBBConfig) -> CacheKey:
    if config.mode == "direct":
        assert config.direct is not None
        path = config.direct.model_path
        slice_hash = _slice_config_hash(config.direct.slice)
    else:
        assert config.sequential is not None
        path = (
            f"{config.sequential.detect_model_path}|"
            f"{config.sequential.obb_model_path}"
        )
        slice_hash = ""  # slicing is direct-mode only
    return CacheKey(
        schema_version=CACHE_SCHEMA_VERSION,
        model_path=path,
        model_mtime=_mtime(path.split("|")[0]),
        # confidence_threshold/iou excluded — re-applied at tracking time.
        # Slicing changes which raw detections exist, so it IS folded in (but
        # only when enabled, so existing non-sliced caches stay valid).
        config_hash=slice_hash,
    )


def _slice_config_hash(slice_cfg) -> str:
    """Empty string when disabled (baseline-identical key); param hash when on."""
    if slice_cfg is None or not slice_cfg.enabled:
        return ""
    payload = "|".join(str(x) for x in (
        "slice", slice_cfg.geometry_mode, slice_cfg.slice_height, slice_cfg.slice_width,
        slice_cfg.overlap_height_ratio, slice_cfg.overlap_width_ratio,
        slice_cfg.object_tile_fraction, slice_cfg.merge_policy, slice_cfg.merge_metric,
        slice_cfg.merge_threshold, slice_cfg.merge_backend, slice_cfg.perform_standard_pred,
    ))
    return _sha(payload)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_inference_cache_keys.py -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Format and commit**

```bash
make format
git add src/hydra_suite/core/inference/cache/keys.py tests/test_inference_cache_keys.py
git commit -m "feat(inference): fold slice params into detection cache key when enabled"
```

---

## Task 10: GUI — detection panel toggle + geometry dropdown

**Files:**
- Modify: `src/hydra_suite/trackerkit/gui/panels/detection_panel.py` (add widgets near the direct-task block ~line 587-612; visibility hook near line 1859-1867)
- Test: `tests/test_detection_panel_slice_widgets.py` (create)

**Interfaces:**
- Produces: `detection_panel.chk_slice_enabled: QCheckBox`, `detection_panel.combo_slice_geometry: QComboBox` (items `["auto_model", "auto_object", "custom"]`); both hidden in sequential mode.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_detection_panel_slice_widgets.py
import pytest

pytest.importorskip("PyQt5.QtWidgets")
from PyQt5.QtWidgets import QApplication

_app = QApplication.instance() or QApplication([])

# DetectionPanel takes constructor args in this codebase; construct it the same
# way tests do — via a MainWindow — to avoid guessing its signature. Reuse the
# persistence test's helper.
from tests.test_main_window_config_persistence import _make_main_window


def test_slice_widgets_exist_with_defaults(monkeypatch):
    window = _make_main_window(monkeypatch)
    panel = window._detection_panel
    assert hasattr(panel, "chk_slice_enabled")
    assert panel.chk_slice_enabled.isChecked() is False
    assert hasattr(panel, "combo_slice_geometry")
    items = [panel.combo_slice_geometry.itemText(i)
             for i in range(panel.combo_slice_geometry.count())]
    assert items == ["auto_model", "auto_object", "custom"]
    window.close()


def test_slice_widgets_hidden_in_sequential_mode(monkeypatch):
    window = _make_main_window(monkeypatch)
    panel = window._detection_panel
    # Sequential mode = obb-mode combo index 1; _on_yolo_mode_changed drives all
    # direct-only row visibility (detection_panel.py:1837).
    panel.combo_yolo_obb_mode.setCurrentIndex(1)
    panel._on_yolo_mode_changed(1)
    assert panel.chk_slice_enabled.isVisibleTo(panel) is False
    window.close()
```

Note: the visibility hook you add in Step 4 MUST live inside `_on_yolo_mode_changed` (detection_panel.py:1837), the method whose `_set_row_visible(...)` calls already hide `combo_yolo_direct_task` when `sequential`.

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_detection_panel_slice_widgets.py -v`
Expected: FAIL with `AttributeError: ...has no attribute 'chk_slice_enabled'`.

- [ ] **Step 3: Add the widgets**

In `detection_panel.py`, after the fixed-angle row (`f_yolo.addRow("Fixed angle", ...)`, ~line 612), add:

```python
        from PyQt5.QtWidgets import QCheckBox  # if not already imported at top
        self.chk_slice_enabled = QCheckBox("Enable sliced inference (SAHI)")
        self.chk_slice_enabled.setToolTip(
            "Tile each frame and detect per tile to recover small-object recall "
            "and reduce crowding. Off by default; direct mode only."
        )
        f_yolo.addRow("Sliced inference", self.chk_slice_enabled)

        self.combo_slice_geometry = QComboBox()
        self.combo_slice_geometry.addItems(["auto_model", "auto_object", "custom"])
        self.combo_slice_geometry.setFixedHeight(30)
        self.combo_slice_geometry.setToolTip(
            "auto_model: tile = model input size (fastest, no resample). "
            "auto_object: size tiles from expected object size. "
            "custom: explicit tile size (advanced config)."
        )
        f_yolo.addRow("Slice geometry", self.combo_slice_geometry)
```

- [ ] **Step 4: Hide the widgets in sequential mode**

Inside `_on_yolo_mode_changed` (detection_panel.py:1837), alongside the existing `_set_row_visible(getattr(self, "combo_yolo_direct_task", None), not sequential)` call (~line 1863), add:

```python
        _set_row_visible(getattr(self, "chk_slice_enabled", None), not sequential)
        _set_row_visible(getattr(self, "combo_slice_geometry", None), not sequential)
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `python -m pytest tests/test_detection_panel_slice_widgets.py -v`
Expected: PASS (2 tests).

- [ ] **Step 6: Format and commit**

```bash
make format
git add src/hydra_suite/trackerkit/gui/panels/detection_panel.py tests/test_detection_panel_slice_widgets.py
git commit -m "feat(trackerkit): sliced-inference toggle + geometry dropdown in detection panel"
```

---

## Task 11: GUI — orchestrator load/save + UPPER_SNAKE + advanced_config

**Files:**
- Modify: `src/hydra_suite/trackerkit/gui/orchestrators/config.py` (load ~line 352-363; save-dict ~line 1537; UPPER_SNAKE in `get_parameters_dict` ~line 2102-2109)
- Test: `tests/test_main_window_config_persistence.py` (append)

**Interfaces:**
- Consumes: `window._detection_panel.chk_slice_enabled` / `.combo_slice_geometry` (Task 10); `advanced_config` dict.
- Produces: config round-trip of `slice_enabled` / `slice_geometry_mode` (save dict + load), and UPPER_SNAKE `SLICE_*` keys (Task 1 param names) into `get_parameters_dict()`, with merge/size knobs from `advanced_config`.

- [ ] **Step 1: Write the failing test (append)**

This file drives persistence through a real `MainWindow` (`_make_main_window`), saving to a JSON preset (`save_config(preset_mode=True, preset_path=...)`) and reloading via `_load_config_from_file(..., preset_mode=True)`. Mirror that exact pattern; the UPPER_SNAKE dict comes from `window._detection_panel`'s orchestrator via `window.get_parameters_dict()` (the method delegates to `ConfigOrchestrator.get_parameters_dict`).

```python
# tests/test_main_window_config_persistence.py (append)
def test_slice_config_persists_and_reloads(monkeypatch, qapp, tmp_path):
    window = _make_main_window(monkeypatch)
    window._detection_panel.chk_slice_enabled.setChecked(True)
    window._detection_panel.combo_slice_geometry.setCurrentText("custom")

    config_path = tmp_path / "slice_roundtrip.json"
    assert window.save_config(preset_mode=True, preset_path=str(config_path))
    saved = json.loads(config_path.read_text(encoding="utf-8"))
    assert saved["slice_enabled"] is True
    assert saved["slice_geometry_mode"] == "custom"
    window.close()

    reloaded = _make_main_window(monkeypatch)
    reloaded._load_config_from_file(str(config_path), preset_mode=True)
    assert reloaded._detection_panel.chk_slice_enabled.isChecked() is True
    assert reloaded._detection_panel.combo_slice_geometry.currentText() == "custom"
    reloaded.close()


def test_slice_params_reach_upper_snake_dict(monkeypatch, qapp):
    window = _make_main_window(
        monkeypatch,
        advanced_config={"slice_overlap": 0.25, "slice_merge_backend": "gpu"},
    )
    window._detection_panel.chk_slice_enabled.setChecked(True)
    window._detection_panel.combo_slice_geometry.setCurrentText("auto_object")
    params = window.get_parameters_dict()
    assert params["SLICE_ENABLED"] is True
    assert params["SLICE_GEOMETRY_MODE"] == "auto_object"
    assert params["SLICE_OVERLAP"] == 0.25
    assert params["SLICE_MERGE_BACKEND"] == "gpu"
    window.close()
```

Note: confirm `window.get_parameters_dict()` exists on `MainWindow` (it delegates to the orchestrator's `get_parameters_dict`, config.py:1906). If the delegation attribute differs, read how an existing test reads UPPER_SNAKE params and match it; the contract (widget+advanced → `SLICE_*`) is fixed.

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_main_window_config_persistence.py -k slice -v`
Expected: FAIL (`KeyError: 'slice_enabled'`).

- [ ] **Step 3: Load slice widgets from config**

In `orchestrators/config.py`, near the direct-task load block (~line 352-363), add:

```python
        self._panels.detection.chk_slice_enabled.setChecked(
            bool(get_cfg("slice_enabled", default=False))
        )
        slice_geo = str(get_cfg("slice_geometry_mode", default="auto_model")).strip()
        if slice_geo not in {"auto_model", "auto_object", "custom"}:
            slice_geo = "auto_model"
        self._panels.detection.combo_slice_geometry.setCurrentText(slice_geo)
```

- [ ] **Step 4: Save slice widgets to the config dict**

In the save-dict (near `"yolo_obb_direct_task": ...` ~line 1537), add:

```python
                "slice_enabled": self._panels.detection.chk_slice_enabled.isChecked(),
                "slice_geometry_mode": self._panels.detection.combo_slice_geometry.currentText(),
```

- [ ] **Step 5: Thread slice params into the UPPER_SNAKE runtime dict**

In the UPPER_SNAKE params dict (near `"YOLO_OBB_DIRECT_TASK": ...` ~line 2102), add:

```python
            "SLICE_ENABLED": self._panels.detection.chk_slice_enabled.isChecked(),
            "SLICE_GEOMETRY_MODE": self._panels.detection.combo_slice_geometry.currentText(),
            "SLICE_OVERLAP": advanced_config.get("slice_overlap", 0.2),
            "SLICE_HEIGHT": advanced_config.get("slice_height", 0),
            "SLICE_WIDTH": advanced_config.get("slice_width", 0),
            "SLICE_OBJECT_TILE_FRACTION": advanced_config.get("slice_object_tile_fraction", 0.15),
            "SLICE_MERGE_POLICY": advanced_config.get("slice_merge_policy", "greedy_nmm"),
            "SLICE_MERGE_METRIC": advanced_config.get("slice_merge_metric", "ios"),
            "SLICE_MERGE_THRESHOLD": advanced_config.get("slice_merge_threshold", 0.5),
            "SLICE_MERGE_BACKEND": advanced_config.get("slice_merge_backend", "cv2"),
            "SLICE_PERFORM_STANDARD_PRED": advanced_config.get("slice_perform_standard_pred", False),
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `python -m pytest tests/test_main_window_config_persistence.py -k slice -v`
Expected: PASS (2 tests).

- [ ] **Step 7: Format and commit**

```bash
make format
git add src/hydra_suite/trackerkit/gui/orchestrators/config.py tests/test_main_window_config_persistence.py
git commit -m "feat(trackerkit): persist + thread sliced-inference config through orchestrator"
```

---

## Task 12: Docs + parity verification

**Files:**
- Modify: `docs/developer-guide/runtime-integration.md` (add a sliced-inference subsection)
- Test: `tests/test_inference_slicing.py` (append the disabled-parity assertion)

**Interfaces:**
- Consumes: everything above.
- Produces: developer docs + an automated `enabled=False` parity check.

- [ ] **Step 1: Write the disabled-parity test (append)**

```python
# tests/test_inference_slicing.py (append)
def test_disabled_slice_is_identical_to_plain_run_direct():
    """enabled=False must be byte-identical to _run_direct (structural bypass)."""
    frame = np.zeros((300, 300, 3), np.uint8)
    model = _FakeYOLO()
    from hydra_suite.core.inference.stages.obb import run_obb, OBBModels, _run_direct
    models = OBBModels(mode="direct", direct_model=model)
    cfg_off = _direct_cfg(False)

    got = run_obb([frame], models, cfg_off, _FakeRuntime())
    expected = _run_direct([frame], model, cfg_off, _FakeRuntime())
    assert len(got) == len(expected)
    for g, e in zip(got, expected):
        assert g.num_detections == e.num_detections
        np.testing.assert_array_equal(g.centroids, e.centroids)
        np.testing.assert_array_equal(g.corners, e.corners)
        np.testing.assert_array_equal(g.confidences, e.confidences)
```

- [ ] **Step 2: Run test to verify it passes**

Run: `python -m pytest tests/test_inference_slicing.py -k disabled -v`
Expected: PASS (the dispatch bypass makes disabled path == `_run_direct`).

- [ ] **Step 3: Add developer docs**

Append a section to `docs/developer-guide/runtime-integration.md`:

```markdown
## Sliced Inference (SAHI)

Direct-mode OBB detection supports optional SAHI-style sliced inference
(`SliceConfig`, off by default). `run_obb` dispatches to
`stages/slicing.py:run_direct_sliced` when `config.obb.direct.slice.enabled`.

- **Geometry:** `auto_model` (tile = model imgsz, no resample — the fast path),
  `auto_object` (tile from expected object size), `custom` (explicit size).
- **Merge:** `merge_policy` (nms/nmm/greedy_nmm) × `merge_metric` (iou/ios) ×
  `merge_backend` (cv2 default; gpu = native-cuda only, cv2-validated). Default
  `greedy_nmm` + `ios` + `0.5`.
- **Cost:** tiles flatten into the existing predict batch; ROI-gated tiles are
  dropped; the overlap-band pre-filter caps the O(n²) merge; native-cuda
  preserves `_RawOBBTensors` (whole when `overlap==0`, band-only sync when
  merging). TRT engine batch is sized from tile-chunk, not window depth.
- **Cache:** slice params fold into `detection_cache_key` only when enabled, so
  existing non-sliced caches stay valid.
```

- [ ] **Step 4: Run the full new-feature suite + disabled regression**

Run: `python -m pytest tests/test_inference_slicing.py tests/test_inference_merge.py tests/test_utils_rotated_iou.py tests/test_inference_config.py tests/test_inference_cache_keys.py -q`
Expected: PASS (all).
Run: `python -m pytest tests/test_inference_stages_obb.py -q`
Expected: PASS (no regressions on the disabled path).

- [ ] **Step 5: Format and commit**

```bash
make format
git add docs/developer-guide/runtime-integration.md tests/test_inference_slicing.py
git commit -m "docs(inference): document sliced inference + automated disabled-parity check"
```

- [ ] **Step 6: Manual equivalence-harness parity gate (per CLAUDE.md)**

This step is a MANUAL verification (not automated in CI). Run the equivalence
harness with slicing disabled to confirm byte-parity vs the legacy baseline on
both platforms, per CLAUDE.md's "Equivalence & Benchmark Verification":

```bash
# MPS (this box):
conda activate hydra-mps
REPO=$PWD WT=$PWD MAIN_SRC=$PWD/.worktrees/equiv-legacy/src WT_SRC=$PWD/src \
  OUT=/tmp/equiv_sahi RUNTIME=mps bash tools/equivalence/run_matrix.sh fly_obb
```

Expected: `fly_obb` EQUIVALENCE at/near its DETERMINISM floor (slicing disabled =
untouched pipeline). CUDA parity on mehek is a follow-up per CLAUDE.md. Record
the result in the PR description; do NOT claim parity without CSV row counts > 1.

---

## Self-Review Notes

**Pre-flight amendments (2026-07-24, decided with the human before execution):**

1. **`auto_object` is now genuinely wired** — `SliceConfig.reference_body_px` is
   sourced from `REFERENCE_BODY_SIZE * RESIZE_FACTOR` (Task 1) and threaded into
   every `plan_slices` call (Tasks 6/7/8). It is no longer inert config.
2. **The `gpu` merge kernel MUST be truly vectorized and MUST beat cv2** (Task 4,
   Step 3 + the Step 4a perf gate). The original plan's Python N² pair loop was
   rejected: it would have been slower than the cv2 default it exists to beat.
3. **Task 5's union no longer rasterizes** to a 64px mask. It projects the corner
   point set directly onto candidate axes (exact, vectorized, no quantization),
   which also removes the accuracy loss behind the old 3px test tolerance.
4. **Task 8 computes the real tile count** from frame size + imgsz via
   `plan_slices`, replacing the magic `TILE_BATCH_HINT = 16`, which would not have
   covered a 4K/imgsz-640 case (~40 tiles) and so would have left the exact
   TensorRT failure §5c warns about in place. `_MAX_TILE_BATCH` is a build-safety
   ceiling, not a tile-count estimate.

**Remaining known limitation (accepted for v1):**

- **Spec §5b nvdec fused op / upload-once:** Task 7 uses the existing
  `_gpu_letterbox_batch`. Its internal guards (`if new_h != H ...`, `if pad_top ...`)
  already no-op the resample under the exact-tile case, so §5a's *resample-free*
  property holds; what is deferred is the fused `stack→permute→float→÷255` batch op
  and the upload-once-then-tile optimization. Both are pure performance refinements
  behind an unchanged interface, so they can land later without rework.

**Consistency:** `SliceConfig` fields, `plan_slices` signature (incl.
`ref_object_px`), `merge_obb_detections` kwargs, `run_direct_sliced(_cuda)`
signatures, `_RawOBBTensors` construction, and `OBBResult.make_detection_ids` are
consistent across all tasks.
