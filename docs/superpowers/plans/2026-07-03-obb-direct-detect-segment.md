# Direct-Mode Detect/Segment-as-OBB Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let `OBBConfig.mode == "direct"` accept a plain YOLO **detect** model (boxes only, fixed angle) or a YOLO **segment** model (instance masks, angle from a GPU-native batched rotating-rectangle search) as the source of `OBBResult`, in addition to the existing native-OBB model — with TensorRT acceleration working for all three from the start, and with no CPU image-processing library (`cv2`) anywhere on the segment hot path. No new `OBBConfig.mode` value is introduced.

**Architecture:** Add a `model_task: Literal["obb","detect","segment"]` field to `OBBDirectConfig`. `stages/obb.py` already threads a `task` string end-to-end into `load_obb_executor`/`_create_direct_executor` for the sequential pipeline's stage-1 detect model — direct mode reuses that same plumbing instead of adding a parallel path. Two new extraction functions (`_extract_obb_from_boxes`, `_extract_obb_from_masks`) convert `Results(boxes=...)` / `Results(masks=...)` into the existing `OBBResult` contract by feeding derived `(cx, cy, w, h, angle)` through the *existing* `_normalize_obb_geometry` / `_corners_from_xywhr` / `_valid_detection_mask` helpers — no new geometry/output contract.

For the segment path, angle/size extraction is done **without `cv2`**: a new pure-tensor kernel, `rotated_rect_from_masks`, crops each detection's mask to an isotropic square tile via `torchvision.ops.roi_align` (a single batched GPU call, no Python loop over detections), then finds the minimum-area rotated rectangle with a batched multi-angle projection search (all detections × all candidate angles in one matmul) refined to sub-grid accuracy with a closed-form 3-point parabolic fit. TensorRT acceleration for `detect` already works today (the sequential stage-1 detect executor is reused verbatim); `segment` needs one new direct executor (`DirectTensorRTSegmentExecutor`) that binds the model's two raw outputs (detections + mask prototypes), decodes masks with ultralytics' `ops.process_mask`, and feeds the result straight into the same `rotated_rect_from_masks` kernel — no ultralytics `Results.masks.xy` / `cv2.findContours` involved anywhere.

**Zero-CPU-sync fast path under the native `cuda` runtime.** The existing `"obb"` direct path already has a deferred-sync mechanism for the NVDEC/native-CUDA case: when `runtime.tensor_on_cuda` is true, `_run_direct` returns `_RawOBBTensors` (raw device tensors, no `.cpu()` call) instead of a materialized `OBBResult`, and the one-time CPU sync is deferred to a later, batched `materialize_tensors()` call that already generically re-derives `_normalize_obb_geometry`/`_corners_from_xywhr`/the finite-value filter from any `(xywhr, conf)` device-tensor pair — it does not care whether that pair came from an OBB head, a detect head, or a segment head. Since both `_extract_obb_from_boxes`'s underlying box math and `rotated_rect_from_masks`'s output are already plain device-tensor arithmetic with no `.cpu()` calls internally, `detect` and `segment` get two small mirror functions (`_extract_raw_tensors_from_boxes`, `_extract_raw_tensors_from_masks`) that package their `(xywhr, conf)` as device tensors exactly like `_extract_raw_tensors` already does for `"obb"`, and plug into the same `runtime.tensor_on_cuda` branch — no new geometry code, no CPU sync until the same shared, batched `materialize_tensors()` call site every detection source already uses. Under TensorRT/CPU/MPS (where `runtime.tensor_on_cuda` is always false and `"obb"` already syncs per frame via `_extract_obb_result`), `detect`/`segment` do the same — that per-frame sync is an existing architecture boundary of this pipeline, not something newly introduced here.

**Tech Stack:** Python, PyTorch, `torchvision.ops.roi_align`, ultralytics YOLO (`ultralytics.utils.nms.non_max_suppression`, `ultralytics.utils.ops.process_mask`/`scale_boxes`, `ultralytics.engine.results.Results` — used only as a lightweight boxes/masks container, its `.xy` polygon/cv2 machinery is never invoked), TensorRT (already vendored via `_direct_obb_runtime.py`), pytest.

## Global Constraints

- Do not add a new `OBBConfig.mode` value. `mode` stays `Literal["direct", "sequential"]`; the new axis lives entirely inside `OBBDirectConfig.model_task`.
- Torch runtimes (`cpu`/`mps`/`cuda`) and CoreML already work for any YOLO task with **zero code changes** to the loaders — `_load_torch_executor`/`_load_coreml_executor` just load/export the checkpoint and let ultralytics infer its own task. Do not add executor classes for those; only wire the new extraction functions and the `task=` passthrough.
- TensorRT is the only runtime that hand-parses raw model output and therefore needs new code for `segment`. `detect` already has a working TensorRT path (`DirectTensorRTDetectExecutor`, reused from the sequential pipeline) — do not duplicate it.
- ONNX (`onnx_*`) is out of scope: `load_obb_executor` already rejects `onnx_*` compute runtimes for OBB (`ArtifactExportError`), so an ONNX segment executor is not needed. Do not add one.
- **No `cv2` on the segment hot path.** `cv2.findContours`/`cv2.minAreaRect` are never called. Angle/size extraction is a pure-tensor batched search (`rotated_rect_from_masks`), runnable on CPU or GPU tensors identically. Under the native `cuda` runtime (`runtime.tensor_on_cuda`), `detect`/`segment` route through `_RawOBBTensors` exactly like `"obb"` already does, so no per-frame CPU sync happens at all — the sync is deferred to the same shared, batched `materialize_tensors()` call site every detection source already uses. Under TensorRT/CPU/MPS, the small final `(N, 5)` geometry array is synced once per frame, matching the existing convention `_extract_obb_result` already follows there. No per-pixel or per-angle work ever leaves the device in either case.
- Angle-preserving coordinate conversion: whenever converting between the mask tensor's own (square) resolution and the true original frame, use the letterbox `gain`/`pad` formula ultralytics itself uses (`ops.scale_boxes`/`scale_coords`: `gain = min(mh/oh, mw/ow)`, symmetric `pad_x`/`pad_y`) — a single uniform scalar gain plus a translation. Do NOT use independent per-axis ratios (`mw/ow`, `mh/oh` separately) to convert `(cx, cy, w, h, angle)` — for a non-square original frame this silently distorts the angle. This is implemented once, in `_letterbox_gain_pad` (Task 3), and reused everywhere a mask-space ↔ original-space conversion is needed.
- Reuse `_normalize_obb_geometry`, `_corners_from_xywhr`, `_valid_detection_mask`, `_apply_raw_detection_cap` from `stages/obb.py` for both new extraction paths. Do not write parallel geometry code.
- Scope is the `core/inference` pipeline plus the `core/tracking/worker.py` params bridge and a minimal `trackerkit` GUI toggle. Do not touch legacy `core/detectors/_obb_geometry.py` / `yolo_detector.py` — those are a separate, older pipeline (see CLAUDE.md `legacy/` policy notes) and are not consumed by `InferenceRunner`.
- Follow existing test conventions: TensorRT-engine-dependent code is *not* unit tested directly (no GPU in CI); factor decode/postprocess math into plain-tensor functions that run on CPU tensors so they ARE unit-testable (this applies doubly here since `rotated_rect_from_masks` and `roi_align` both run correctly on CPU tensors — no GPU is needed to verify the geometry math at all).

---

## File Structure

| File | Change |
|---|---|
| `src/hydra_suite/core/inference/config.py` | Add `model_task` + `fixed_angle_deg` to `OBBDirectConfig` |
| `src/hydra_suite/core/detectors/_obb_from_mask.py` (new) | `_letterbox_gain_pad`, `rotated_rect_from_masks` — pure-tensor, cv2-free, GPU-native |
| `src/hydra_suite/core/inference/stages/obb.py` | New `_extract_obb_from_boxes`, `_extract_obb_from_masks`; dispatch in `load_obb_models`/`_run_direct` |
| `src/hydra_suite/core/inference/runtime_artifacts.py` | `_create_direct_executor` gains a `task == "segment"` branch |
| `src/hydra_suite/core/detectors/_direct_obb_runtime.py` | New `DirectTensorRTSegmentExecutor`, `create_direct_segment_executor` (built on `_obb_from_mask.rotated_rect_from_masks`) |
| `src/hydra_suite/core/tracking/worker.py` | Thread `YOLO_OBB_DIRECT_TASK` / `YOLO_OBB_FIXED_ANGLE_DEG` params into `OBBDirectConfig` |
| `src/hydra_suite/trackerkit/gui/panels/detection_panel.py` | Add a "Direct model task" combo + fixed-angle spinbox, following the existing `combo_yolo_obb_mode` pattern |
| `tests/test_inference_config.py` | Round-trip test for new `OBBDirectConfig` fields |
| `tests/test_obb_from_mask.py` (new) | CPU unit tests for `rotated_rect_from_masks` / `_letterbox_gain_pad` |
| `tests/test_inference_stages_obb.py` | Tests for `_extract_obb_from_boxes` / `_extract_obb_from_masks` |
| `tests/test_inference_obb_artifacts.py` | Selection-logic test: `task="segment"` routes to the segment factory |

---

### Task 1: `OBBDirectConfig` gains `model_task` + `fixed_angle_deg`

**Files:**
- Modify: `src/hydra_suite/core/inference/config.py:83-95`
- Test: `tests/test_inference_config.py`

**Interfaces:**
- Produces: `OBBDirectConfig.model_task: Literal["obb", "detect", "segment"] = "obb"`, `OBBDirectConfig.fixed_angle_deg: float = 0.0` — read by Task 5.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_inference_config.py` (mirror the file's existing round-trip style — check an existing `to_json`/`from_json` test in that file for the exact fixture pattern, then add):

```python
def test_obb_direct_config_model_task_round_trips(tmp_path):
    from hydra_suite.core.inference.config import (
        InferenceConfig,
        OBBConfig,
        OBBDirectConfig,
    )

    cfg = InferenceConfig(
        obb=OBBConfig(
            mode="direct",
            direct=OBBDirectConfig(
                model_path="yolo26s-seg.pt",
                model_task="segment",
                fixed_angle_deg=0.0,
            ),
        )
    )
    path = tmp_path / "cfg.json"
    cfg.to_json(str(path))
    loaded = InferenceConfig.from_json(str(path))

    assert loaded.obb.direct.model_task == "segment"
    assert loaded.obb.direct.fixed_angle_deg == 0.0


def test_obb_direct_config_model_task_defaults_to_obb():
    from hydra_suite.core.inference.config import OBBDirectConfig

    direct = OBBDirectConfig(model_path="yolo26s-obb.pt")
    assert direct.model_task == "obb"
    assert direct.fixed_angle_deg == 0.0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_inference_config.py -k model_task -v`
Expected: FAIL with `TypeError: OBBDirectConfig.__init__() got an unexpected keyword argument 'model_task'`

- [ ] **Step 3: Implement**

In `src/hydra_suite/core/inference/config.py`, edit the `OBBDirectConfig` dataclass:

```python
@dataclass
class OBBDirectConfig:
    model_path: str
    # Deprecated: runtime decisions now use InferenceConfig.runtime_tier.
    # Kept for serialization round-trip; no longer read by stage loaders.
    compute_runtime: ComputeRuntime = "cpu"
    confidence_floor: float = 1e-3
    confidence_threshold: float = 0.25
    # Auto-export the .engine (TensorRT) / .mlpackage (CoreML) artifact from a
    # .pt source on first load for the gpu_fast runtimes. When False and no
    # artifact exists, loading raises a
    # clear error instead of silently running PyTorch (parity finding H4).
    auto_export: bool = True
    # "obb": model_path is a native-OBB YOLO checkpoint (existing behaviour).
    # "detect": model_path is a plain axis-aligned YOLO detect checkpoint;
    # every detection is assigned the fixed angle below instead of a
    # model-predicted angle.
    # "segment": model_path is a YOLO instance-segmentation checkpoint; the
    # angle is derived per-detection from a GPU batched rotated-rectangle
    # search over the predicted mask (see core/detectors/_obb_from_mask.py).
    model_task: Literal["obb", "detect", "segment"] = "obb"
    # Only read when model_task == "detect". Degrees; converted to radians
    # before being folded through the same normalize/corners pipeline as
    # native-OBB angles.
    fixed_angle_deg: float = 0.0
```

`_dict_to_config`'s `OBBDirectConfig(**obb_d["direct"])` (config.py:311) needs no change — new dataclass fields with defaults round-trip automatically via `asdict`/`**kwargs`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_inference_config.py -k model_task -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/hydra_suite/core/inference/config.py tests/test_inference_config.py
git commit -m "feat(inference): add model_task/fixed_angle_deg to OBBDirectConfig"
```

---

### Task 2: `_extract_obb_from_boxes` — detect-as-OBB with fixed angle

**Files:**
- Modify: `src/hydra_suite/core/inference/stages/obb.py` (add functions near `_extract_obb_result`/`_extract_raw_tensors`, i.e. after line 626)
- Test: `tests/test_inference_stages_obb.py`

**Interfaces:**
- Consumes: `_normalize_obb_geometry`, `_corners_from_xywhr`, `_valid_detection_mask`, `_empty_obb_result`, `_RawOBBTensors` (all already defined in `stages/obb.py`).
- Produces:
  - `_extract_obb_from_boxes(result, frame_idx, fixed_angle_rad, offset=(0.0,0.0), scale=(1.0,1.0)) -> OBBResult` — the CPU-materializing path, consumed by Task 5 for TensorRT/CPU/MPS runtimes.
  - `_extract_raw_tensors_from_boxes(result, frame_idx, fixed_angle_rad, device) -> _RawOBBTensors` — the zero-CPU-sync fast path, mirroring `_extract_raw_tensors`'s existing contract exactly (raw, unfiltered device tensors; normalize/corners/finite-filtering is deferred to `materialize_tensors()`). Consumed by Task 5 for the native `cuda` runtime (`runtime.tensor_on_cuda`).

- [ ] **Step 1: Write the failing test**

Add to `tests/test_inference_stages_obb.py` (check the file's existing imports/fixtures first — it already imports from `hydra_suite.core.inference.stages.obb`; follow its existing style for constructing a fake ultralytics `Results`-like object, e.g. `SimpleNamespace(boxes=SimpleNamespace(xyxy=..., conf=...))`):

```python
def test_extract_obb_from_boxes_applies_fixed_angle():
    import numpy as np
    import torch
    from types import SimpleNamespace

    from hydra_suite.core.inference.stages.obb import _extract_obb_from_boxes

    # One box: x1,y1,x2,y2 = 10,20,30,60 -> cx=20,cy=40,w=20,h=40
    result = SimpleNamespace(
        boxes=SimpleNamespace(
            xyxy=torch.tensor([[10.0, 20.0, 30.0, 60.0]]),
            conf=torch.tensor([0.9]),
        )
    )

    out = _extract_obb_from_boxes(result, frame_idx=3, fixed_angle_rad=0.0)

    assert out.num_detections == 1
    assert out.frame_idx == 3
    np.testing.assert_allclose(out.centroids[0], [20.0, 40.0], atol=1e-4)
    # w=20 < h=40, so _normalize_obb_geometry swaps to major=h=40, minor=w=20
    # and adds 90deg to the (here, 0deg) fixed angle.
    np.testing.assert_allclose(out.angles[0], np.pi / 2, atol=1e-4)
    np.testing.assert_allclose(out.sizes[0], 800.0, atol=1e-3)  # 20*40
    np.testing.assert_allclose(out.confidences[0], 0.9, atol=1e-4)


def test_extract_obb_from_boxes_empty_boxes_returns_empty_result():
    from types import SimpleNamespace
    import torch

    from hydra_suite.core.inference.stages.obb import _extract_obb_from_boxes

    result = SimpleNamespace(
        boxes=SimpleNamespace(xyxy=torch.zeros((0, 4)), conf=torch.zeros(0))
    )
    out = _extract_obb_from_boxes(result, frame_idx=0, fixed_angle_rad=0.0)
    assert out.num_detections == 0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_inference_stages_obb.py -k extract_obb_from_boxes -v`
Expected: FAIL with `ImportError: cannot import name '_extract_obb_from_boxes'`

- [ ] **Step 3: Implement**

Add to `src/hydra_suite/core/inference/stages/obb.py`, directly after `_extract_obb_result` (after line 626):

```python
def _extract_obb_from_boxes(
    result: Any,
    frame_idx: int,
    fixed_angle_rad: float,
    offset: tuple[float, float] = (0.0, 0.0),
    scale: tuple[float, float] = (1.0, 1.0),
) -> OBBResult:
    """Build an OBBResult from a plain (axis-aligned) detect model's boxes.

    Every detection is assigned ``fixed_angle_rad`` before being folded through
    the same ``_normalize_obb_geometry`` / ``_corners_from_xywhr`` pipeline used
    for native-OBB output, so downstream consumers (filtering, assignment,
    canonical crops) cannot tell the geometry did not come from an OBB head.
    """
    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        return _empty_obb_result(frame_idx)
    xyxy = boxes.xyxy.cpu().numpy().copy()  # (N, 4): x1,y1,x2,y2
    conf = boxes.conf.cpu().numpy()  # (N,)
    ox, oy = offset
    sx, sy = scale
    cx = (xyxy[:, 0] + xyxy[:, 2]) / 2.0
    cy = (xyxy[:, 1] + xyxy[:, 3]) / 2.0
    w_arr = xyxy[:, 2] - xyxy[:, 0]
    h_arr = xyxy[:, 3] - xyxy[:, 1]
    # Mirrors _extract_obb_result's crop-space rescale-then-offset order.
    cx *= sx
    w_arr *= sx
    cy *= sy
    h_arr *= sy
    cx += ox
    cy += oy
    angle_arr = np.full(cx.shape, float(fixed_angle_rad), dtype=np.float32)
    angles_fixed, sizes, aspect = _normalize_obb_geometry(w_arr, h_arr, angle_arr)
    mask = _valid_detection_mask(cx, cy, w_arr, h_arr, angles_fixed, conf)
    if not mask.all():
        dropped = int(mask.size - int(mask.sum()))
        if dropped > 0:
            logger.warning(
                "Dropping %d invalid detect-as-OBB detections with non-finite "
                "or non-positive geometry.",
                dropped,
            )
        cx, cy, w_arr, h_arr = cx[mask], cy[mask], w_arr[mask], h_arr[mask]
        conf, angles_fixed, sizes, aspect = (
            conf[mask],
            angles_fixed[mask],
            sizes[mask],
            aspect[mask],
        )
    n = int(len(conf))
    corners = _corners_from_xywhr(cx, cy, w_arr, h_arr, angles_fixed)
    return OBBResult(
        frame_idx=frame_idx,
        centroids=np.stack([cx, cy], axis=1).astype(np.float32),
        angles=angles_fixed,
        sizes=sizes,
        shapes=np.stack([sizes, aspect], axis=1).astype(np.float32),
        confidences=conf.astype(np.float32),
        corners=corners.astype(np.float32),
        detection_ids=OBBResult.make_detection_ids(frame_idx, n),
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_inference_stages_obb.py -k extract_obb_from_boxes -v`
Expected: PASS

- [ ] **Step 5: Write the failing test for the zero-CPU-sync fast-path variant**

Add to `tests/test_inference_stages_obb.py`:

```python
def test_extract_raw_tensors_from_boxes_keeps_everything_on_device():
    import torch
    from types import SimpleNamespace

    from hydra_suite.core.inference.stages.obb import _extract_raw_tensors_from_boxes

    result = SimpleNamespace(
        boxes=SimpleNamespace(
            xyxy=torch.tensor([[10.0, 20.0, 30.0, 60.0]]),
            conf=torch.tensor([0.9]),
        )
    )

    raw = _extract_raw_tensors_from_boxes(
        result, frame_idx=3, fixed_angle_rad=0.5, device="cpu"
    )

    assert raw.frame_idx == 3
    assert isinstance(raw.xywhr, torch.Tensor)
    assert raw.xywhr.shape == (1, 5)
    torch.testing.assert_close(raw.xywhr[0, :4], torch.tensor([20.0, 40.0, 20.0, 40.0]))
    torch.testing.assert_close(raw.xywhr[0, 4], torch.tensor(0.5))
    torch.testing.assert_close(raw.conf, torch.tensor([0.9]))


def test_extract_raw_tensors_from_boxes_empty_boxes():
    import torch
    from types import SimpleNamespace

    from hydra_suite.core.inference.stages.obb import _extract_raw_tensors_from_boxes

    result = SimpleNamespace(
        boxes=SimpleNamespace(xyxy=torch.zeros((0, 4)), conf=torch.zeros(0))
    )
    raw = _extract_raw_tensors_from_boxes(result, frame_idx=0, fixed_angle_rad=0.0, device="cpu")
    assert raw.xywhr.shape == (0, 5)
```

- [ ] **Step 6: Run test to verify it fails**

Run: `python -m pytest tests/test_inference_stages_obb.py -k extract_raw_tensors_from_boxes -v`
Expected: FAIL with `ImportError: cannot import name '_extract_raw_tensors_from_boxes'`

- [ ] **Step 7: Implement the fast-path variant**

Add directly after `_extract_obb_from_boxes`:

```python
def _extract_raw_tensors_from_boxes(
    result: Any, frame_idx: int, fixed_angle_rad: float, device: str
) -> _RawOBBTensors:
    """Keep detect-as-OBB tensors on the compute device -- no .cpu() call.

    Mirrors _extract_raw_tensors's contract exactly: raw, unfiltered geometry.
    normalize/corners/finite-value filtering is deferred to
    materialize_tensors(), which already works generically for ANY
    (xywhr, conf) device-tensor pair regardless of detection source.
    """
    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        dev = torch.device(device)
        return _RawOBBTensors(
            frame_idx=frame_idx,
            xywhr=torch.zeros((0, 5), dtype=torch.float32, device=dev),
            corners=torch.zeros((0, 4, 2), dtype=torch.float32, device=dev),
            conf=torch.zeros(0, dtype=torch.float32, device=dev),
        )
    xyxy = boxes.xyxy  # (N, 4), stays on whatever device it already is
    cx = (xyxy[:, 0] + xyxy[:, 2]) / 2.0
    cy = (xyxy[:, 1] + xyxy[:, 3]) / 2.0
    w_arr = xyxy[:, 2] - xyxy[:, 0]
    h_arr = xyxy[:, 3] - xyxy[:, 1]
    angle = torch.full_like(cx, float(fixed_angle_rad))
    xywhr = torch.stack([cx, cy, w_arr, h_arr, angle], dim=1)
    # materialize_tensors() ignores raw.corners and rebuilds corners fresh
    # from xywhr (see its existing implementation) -- this field is a
    # placeholder, exactly like _extract_raw_tensors's own corners field is
    # for the "obb" fast path today.
    corners = torch.zeros(
        (xywhr.shape[0], 4, 2), dtype=torch.float32, device=xywhr.device
    )
    return _RawOBBTensors(frame_idx=frame_idx, xywhr=xywhr, corners=corners, conf=boxes.conf)
```

- [ ] **Step 8: Run tests to verify they pass**

Run: `python -m pytest tests/test_inference_stages_obb.py -k "extract_obb_from_boxes or extract_raw_tensors_from_boxes" -v`
Expected: PASS

- [ ] **Step 9: Commit**

```bash
git add src/hydra_suite/core/inference/stages/obb.py tests/test_inference_stages_obb.py
git commit -m "feat(inference): extract OBB from plain detect boxes with a fixed angle"
```

---

### Task 3: `rotated_rect_from_masks` — GPU-native batched rotated-rectangle search (no `cv2`)

**Files:**
- Create: `src/hydra_suite/core/detectors/_obb_from_mask.py`
- Test: `tests/test_obb_from_mask.py` (new)

**Interfaces:**
- Produces:
  - `_letterbox_gain_pad(mask_shape: tuple[int, int], orig_shape: tuple[int, int]) -> tuple[float, float, float]` → `(gain, pad_x, pad_y)`, mirroring `ultralytics.utils.ops.scale_boxes`'s internal formula.
  - `rotated_rect_from_masks(masks: torch.Tensor, boxes_xyxy: torch.Tensor, *, num_angles: int = 24, crop_size: int = 64, pad_ratio: float = 0.15) -> torch.Tensor` → `(N, 5)` tensor of `(cx, cy, w, h, angle_rad)`, in the SAME coordinate space as the input `masks`/`boxes_xyxy` (caller converts to original-frame space via `_letterbox_gain_pad`). Rows for detections with an empty mask crop are filled with `NaN` (caller filters via the existing `_valid_detection_mask`, which already checks `np.isfinite`).
- Consumed by: Task 4 (`_extract_obb_from_masks`) and Task 7 (`DirectTensorRTSegmentExecutor`).

This is the highest-value task to get right in isolation: everything here is plain tensor math, runs identically on CPU or CUDA tensors, and is fully unit-testable without any GPU, TensorRT, or even ultralytics import.

- [ ] **Step 1: Write the failing test**

Create `tests/test_obb_from_mask.py`:

```python
"""CPU unit tests for the cv2-free, GPU-native mask -> rotated-rect kernel."""

from __future__ import annotations

import math

import numpy as np
import torch

from hydra_suite.core.detectors._obb_from_mask import (
    _letterbox_gain_pad,
    rotated_rect_from_masks,
)


def _rasterize_rotated_rect(
    size: int, cx: float, cy: float, w: float, h: float, angle_deg: float
) -> torch.Tensor:
    """Build a (size, size) binary mask of a rotated rectangle, for ground truth."""
    ys, xs = torch.meshgrid(
        torch.arange(size, dtype=torch.float32),
        torch.arange(size, dtype=torch.float32),
        indexing="ij",
    )
    dx, dy = xs - cx, ys - cy
    theta = math.radians(angle_deg)
    # Rotate the query grid into the rectangle's own axis-aligned frame.
    u = dx * math.cos(theta) + dy * math.sin(theta)
    v = -dx * math.sin(theta) + dy * math.cos(theta)
    return ((u.abs() <= w / 2) & (v.abs() <= h / 2)).float()


def test_letterbox_gain_pad_matches_scale_boxes_formula():
    # Square mask canvas (160x160), non-square original frame (1080x1920) --
    # the exact scenario that breaks a naive per-axis ratio.
    gain, pad_x, pad_y = _letterbox_gain_pad((160, 160), (1080, 1920))
    expected_gain = min(160 / 1080, 160 / 1920)
    assert math.isclose(gain, expected_gain, rel_tol=1e-6)
    assert pad_x >= 0 and pad_y >= 0
    # The wider dimension (1920) should produce zero pad on that axis, all
    # the pad should land on the shorter (1080) axis.
    assert math.isclose(pad_x, 0.0, abs_tol=1e-3)
    assert pad_y > 0


def test_rotated_rect_from_masks_recovers_axis_aligned_rectangle():
    mask = _rasterize_rotated_rect(128, cx=64, cy=64, w=50, h=20, angle_deg=0.0)
    masks = mask.unsqueeze(0)  # (1, 128, 128)
    boxes = torch.tensor([[64 - 25 - 5, 64 - 10 - 5, 64 + 25 + 5, 64 + 10 + 5]])

    rect = rotated_rect_from_masks(masks, boxes, num_angles=24, crop_size=64)

    assert rect.shape == (1, 5)
    cx, cy, w, h, angle = rect[0].tolist()
    assert math.isclose(cx, 64, abs_tol=1.5)
    assert math.isclose(cy, 64, abs_tol=1.5)
    major, minor = max(w, h), min(w, h)
    assert math.isclose(major, 50, abs_tol=3.0)
    assert math.isclose(minor, 20, abs_tol=3.0)
    # Major axis along x -> angle ~ 0 (mod pi).
    assert min(angle % math.pi, math.pi - (angle % math.pi)) < math.radians(5)


def test_rotated_rect_from_masks_recovers_rotated_rectangle():
    mask = _rasterize_rotated_rect(128, cx=64, cy=64, w=50, h=20, angle_deg=35.0)
    masks = mask.unsqueeze(0)
    # Loose axis-aligned bbox covering the rotated rect, padded generously.
    boxes = torch.tensor([[14.0, 14.0, 114.0, 114.0]])

    rect = rotated_rect_from_masks(masks, boxes, num_angles=36, crop_size=96)

    _, _, w, h, angle = rect[0].tolist()
    major, minor = max(w, h), min(w, h)
    assert math.isclose(major, 50, abs_tol=4.0)
    assert math.isclose(minor, 20, abs_tol=4.0)
    expected_rad = math.radians(35.0)
    diff = min(
        abs((angle % math.pi) - expected_rad),
        math.pi - abs((angle % math.pi) - expected_rad),
    )
    assert diff < math.radians(6)


def test_rotated_rect_from_masks_empty_mask_yields_nan_row():
    masks = torch.zeros((1, 64, 64))
    boxes = torch.tensor([[10.0, 10.0, 20.0, 20.0]])
    rect = rotated_rect_from_masks(masks, boxes)
    assert torch.isnan(rect[0]).all()


def test_rotated_rect_from_masks_handles_zero_detections():
    masks = torch.zeros((0, 64, 64))
    boxes = torch.zeros((0, 4))
    rect = rotated_rect_from_masks(masks, boxes)
    assert rect.shape == (0, 5)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_obb_from_mask.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'hydra_suite.core.detectors._obb_from_mask'`

- [ ] **Step 3: Implement**

Create `src/hydra_suite/core/detectors/_obb_from_mask.py`:

```python
"""GPU-native, cv2-free rotated-rectangle extraction from segmentation masks.

Used to treat a YOLO instance-segmentation checkpoint as an OBB source: given
a batch of binary/soft masks and their (matching-space) bounding boxes, find
each detection's minimum-area rotated rectangle without ever calling
``cv2.findContours``/``cv2.minAreaRect`` or leaving the accelerator.

Method
------
1. Crop each detection's mask to an ISOTROPIC (square, non-rotated) tile via
   a single batched ``torchvision.ops.roi_align`` call -- isotropic scale is
   required so that an angle measured in the crop's local coordinate frame
   equals the angle in the source mask's coordinate frame (a non-square
   resample would shear angles).
2. Compute the mask-weighted centroid of the crop.
3. Project the crop's foreground pixels onto ``num_angles`` candidate axes in
   one batched matmul, take the masked (min, max) extent per axis to get a
   width/height/area per candidate angle, and pick the angle minimizing area
   (a coarse, batched analogue of the rotating-calipers objective that
   ``cv2.minAreaRect`` solves exactly for a polygon hull).
4. Refine the winning angle to sub-grid accuracy with a closed-form 3-point
   parabolic fit through the winning bin and its two neighbours (same idea as
   sub-pixel peak refinement in stereo/optical-flow disparity search), then
   recompute width/height once more at the refined angle.

Everything after step 1 operates on a small, fixed-size tensor
(``crop_size x crop_size`` per detection), so this stays cheap even for many
detections per frame and never performs a per-detection Python loop.
"""

from __future__ import annotations

import math

import torch
from torchvision.ops import roi_align


def _letterbox_gain_pad(
    mask_shape: tuple[int, int], orig_shape: tuple[int, int]
) -> tuple[float, float, float]:
    """Return ``(gain, pad_x, pad_y)`` mapping ``orig_shape`` -> ``mask_shape``.

    Mirrors ``ultralytics.utils.ops.scale_boxes``'s own formula exactly: a
    single uniform ``gain`` (``mask`` canvases are always square, matching a
    square YOLO letterboxed input) plus a symmetric pad per axis. Using a
    single scalar gain (rather than independent per-axis ratios) is what
    keeps rotation angles correct when the original frame is not square.

    To go orig -> mask space: ``x_mask = x_orig * gain + pad_x`` (same for y).
    To go mask -> orig space: ``x_orig = (x_mask - pad_x) / gain``.
    """
    mh, mw = mask_shape
    oh, ow = orig_shape
    gain = min(mh / oh, mw / ow)
    pad_x = (mw - round(ow * gain)) / 2.0
    pad_y = (mh - round(oh * gain)) / 2.0
    return float(gain), float(pad_x), float(pad_y)


def rotated_rect_from_masks(
    masks: torch.Tensor,
    boxes_xyxy: torch.Tensor,
    *,
    num_angles: int = 24,
    crop_size: int = 64,
    pad_ratio: float = 0.15,
    mask_threshold: float = 0.5,
) -> torch.Tensor:
    """Find each detection's minimum-area rotated rectangle from its mask.

    Parameters
    ----------
    masks:
        ``(N, H, W)`` tensor (bool/float/uint8), same coordinate space as
        ``boxes_xyxy``. May be on CPU or CUDA.
    boxes_xyxy:
        ``(N, 4)`` axis-aligned boxes in the SAME coordinate space as
        ``masks``, used only to size/center the isotropic crop.
    num_angles:
        Number of coarse candidate angles searched over ``[0, pi)`` before
        parabolic refinement.
    crop_size:
        Side length (pixels) of the isotropic square tile each detection's
        mask is resampled into.
    pad_ratio:
        Fractional padding added around the (square-ified) box before
        cropping, so the crop is not clipped exactly at the mask edge.

    Returns
    -------
    torch.Tensor
        ``(N, 5)``: ``(cx, cy, w, h, angle_rad)``, same coordinate space as
        the inputs, same device/dtype family (float32). A detection whose
        mask has no foreground pixels inside its crop yields an all-``NaN``
        row -- callers should drop these via the existing finite-value guard
        (``_valid_detection_mask`` in ``stages/obb.py`` already does this).
    """
    device = masks.device
    n = masks.shape[0]
    if n == 0:
        return torch.zeros((0, 5), dtype=torch.float32, device=device)

    # --- 1. Isotropic square crop via a single batched roi_align call. ---
    x1, y1, x2, y2 = boxes_xyxy.unbind(-1)
    bw = (x2 - x1).clamp(min=1.0)
    bh = (y2 - y1).clamp(min=1.0)
    bcx = (x1 + x2) / 2.0
    bcy = (y1 + y2) / 2.0
    half = torch.maximum(bw, bh) / 2.0 * (1.0 + pad_ratio)
    side = 2.0 * half  # physical size (pixels) of each detection's crop
    sx1, sy1 = bcx - half, bcy - half
    sx2, sy2 = bcx + half, bcy + half
    batch_idx = torch.arange(n, device=device, dtype=masks.dtype)
    roi_boxes = torch.stack([batch_idx, sx1, sy1, sx2, sy2], dim=1)

    crops = roi_align(
        masks.unsqueeze(1).float(),
        roi_boxes,
        output_size=(crop_size, crop_size),
        aligned=True,
    ).squeeze(1)  # (N, crop_size, crop_size)
    weights = (crops > mask_threshold).float()  # (N, crop_size, crop_size)
    weights_flat = weights.reshape(n, -1)  # (N, P), P = crop_size**2

    # --- 2. Mask-weighted centroid, in LOCAL unit-square coordinates. ---
    lin = torch.linspace(-0.5, 0.5, crop_size, device=device, dtype=torch.float32)
    grid_y, grid_x = torch.meshgrid(lin, lin, indexing="ij")
    grid = torch.stack([grid_x.reshape(-1), grid_y.reshape(-1)], dim=0)  # (2, P)

    total_weight = weights_flat.sum(dim=1).clamp(min=1e-6)  # (N,)
    centroid_local = (weights_flat @ grid.T) / total_weight[:, None]  # (N, 2)
    has_foreground = weights_flat.sum(dim=1) > 0  # (N,)

    coords = grid[None, :, :] - centroid_local.T[:, :, None]  # (N, 2, P)
    coords = coords.transpose(1, 2)  # (N, P, 2)

    # --- 3. Coarse batched angle search. ---
    angles = torch.linspace(
        0.0, math.pi, num_angles + 1, device=device, dtype=torch.float32
    )[:-1]
    cos_a, sin_a = torch.cos(angles), torch.sin(angles)
    # Rotation matrices for every candidate angle: (K, 2, 2).
    rot = torch.stack(
        [torch.stack([cos_a, -sin_a], dim=1), torch.stack([sin_a, cos_a], dim=1)],
        dim=1,
    )
    # (N, P, 2) @ (K, 2, 2)^T broadcast -> (N, K, P, 2)
    proj = torch.einsum("npc,kdc->nkpd", coords, rot)
    u, v = proj[..., 0], proj[..., 1]  # each (N, K, P)

    fg = weights_flat[:, None, :] > 0  # (N, 1, P) broadcast over K
    pos_inf = torch.full_like(u, float("inf"))
    neg_inf = torch.full_like(u, float("-inf"))
    u_max = torch.where(fg, u, neg_inf).amax(dim=-1)
    u_min = torch.where(fg, u, pos_inf).amin(dim=-1)
    v_max = torch.where(fg, v, neg_inf).amax(dim=-1)
    v_min = torch.where(fg, v, pos_inf).amin(dim=-1)
    width_k = (u_max - u_min).clamp(min=0.0)  # (N, K)
    height_k = (v_max - v_min).clamp(min=0.0)
    area_k = width_k * height_k

    k_star = area_k.argmin(dim=1)  # (N,)

    # --- 4. Closed-form 3-point parabolic sub-grid refinement (circular). ---
    k_prev = (k_star - 1) % num_angles
    k_next = (k_star + 1) % num_angles
    area_prev = area_k.gather(1, k_prev[:, None]).squeeze(1)
    area_star = area_k.gather(1, k_star[:, None]).squeeze(1)
    area_next = area_k.gather(1, k_next[:, None]).squeeze(1)
    denom = area_next - 2.0 * area_star + area_prev
    safe = denom.abs() > 1e-6
    offset = torch.where(
        safe, -0.5 * (area_next - area_prev) / denom.clamp(min=1e-6), torch.zeros_like(denom)
    )
    offset = offset.clamp(-1.0, 1.0)
    angle_step = math.pi / num_angles
    theta_star = angles.gather(0, k_star)
    theta_refined = (theta_star + offset * angle_step) % math.pi

    # --- Recompute width/height once more at the refined per-detection angle. ---
    cos_r, sin_r = torch.cos(theta_refined), torch.sin(theta_refined)
    rot_r = torch.stack(
        [torch.stack([cos_r, -sin_r], dim=1), torch.stack([sin_r, cos_r], dim=1)], dim=1
    )  # (N, 2, 2)
    proj_r = torch.bmm(coords, rot_r.transpose(1, 2))  # (N, P, 2)
    u_r, v_r = proj_r[..., 0], proj_r[..., 1]
    fg2 = weights_flat > 0
    u_r_max = torch.where(fg2, u_r, torch.full_like(u_r, float("-inf"))).amax(dim=-1)
    u_r_min = torch.where(fg2, u_r, torch.full_like(u_r, float("inf"))).amin(dim=-1)
    v_r_max = torch.where(fg2, v_r, torch.full_like(v_r, float("-inf"))).amax(dim=-1)
    v_r_min = torch.where(fg2, v_r, torch.full_like(v_r, float("inf"))).amin(dim=-1)
    w_local = (u_r_max - u_r_min).clamp(min=0.0)
    h_local = (v_r_max - v_r_min).clamp(min=0.0)

    # --- Map centroid + size back from local unit-square units to the input
    #     masks'/boxes' physical coordinate space. ---
    cx = bcx + centroid_local[:, 0] * side
    cy = bcy + centroid_local[:, 1] * side
    w = w_local * side
    h = h_local * side

    result = torch.stack([cx, cy, w, h, theta_refined], dim=1)
    nan_row = torch.full((5,), float("nan"), device=device, dtype=torch.float32)
    result = torch.where(has_foreground[:, None], result, nan_row[None, :])
    return result
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_obb_from_mask.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/hydra_suite/core/detectors/_obb_from_mask.py tests/test_obb_from_mask.py
git commit -m "feat(detectors): add GPU-native, cv2-free rotated-rect-from-mask kernel"
```

---

### Task 4: `_extract_obb_from_masks` — segment-as-OBB via `rotated_rect_from_masks`

**Files:**
- Modify: `src/hydra_suite/core/inference/stages/obb.py` (add after `_extract_obb_from_boxes` from Task 2)
- Test: `tests/test_inference_stages_obb.py`

**Interfaces:**
- Consumes: `rotated_rect_from_masks`, `_letterbox_gain_pad` (Task 3, imported from `..detectors._obb_from_mask` — mirrors the existing precedent of `runtime_artifacts.py` already importing from `core.detectors._direct_obb_runtime`), plus the same `_normalize_obb_geometry`/`_corners_from_xywhr`/`_valid_detection_mask`/`_RawOBBTensors` helpers used by Task 2.
- Produces:
  - `_extract_obb_from_masks(result, frame_idx, offset=(0.0,0.0), scale=(1.0,1.0)) -> OBBResult` — the CPU-materializing path, consumed by Task 5 for TensorRT/CPU/MPS runtimes.
  - `_extract_raw_tensors_from_masks(result, frame_idx, device) -> _RawOBBTensors` — the zero-CPU-sync fast path (mirrors `_extract_raw_tensors_from_boxes` from Task 2), consumed by Task 5 for the native `cuda` runtime.

  Both work on any duck-typed `result` exposing `.masks.data` (a `(N, mh, mw)` tensor, mh==mw, in some square letterboxed/proto coordinate space), `.boxes.xyxy` (original-frame-space boxes), `.boxes.conf`, and `.orig_shape` (`(H, W)` of the true original frame) — i.e. both a plain ultralytics segmentation model's `Results` (which already provides exactly these four attributes) and a lightweight hand-built object from the TensorRT segment executor (Task 7).

- [ ] **Step 1: Write the failing test**

Add to `tests/test_inference_stages_obb.py`:

```python
def test_extract_obb_from_masks_computes_rotated_rect():
    import math
    import numpy as np
    import torch
    from types import SimpleNamespace

    from hydra_suite.core.inference.stages.obb import _extract_obb_from_masks

    # A 40x20 axis-aligned rectangle mask at (50, 30) in a 100x60 "mask-space"
    # canvas that is ALSO treated as the original frame (gain=1, no padding)
    # for this test -- Task 3 already covers the gain/pad math independently.
    mh = mw = 100
    ys, xs = torch.meshgrid(
        torch.arange(mh, dtype=torch.float32),
        torch.arange(mw, dtype=torch.float32),
        indexing="ij",
    )
    mask = (
        (xs >= 30) & (xs <= 70) & (ys >= 20) & (ys <= 40)
    ).float().unsqueeze(0)  # (1, 100, 100)

    result = SimpleNamespace(
        masks=SimpleNamespace(data=mask),
        boxes=SimpleNamespace(
            xyxy=torch.tensor([[30.0, 20.0, 70.0, 40.0]]),
            conf=torch.tensor([0.8]),
        ),
        orig_shape=(100, 100),
    )

    out = _extract_obb_from_masks(result, frame_idx=5)

    assert out.num_detections == 1
    assert out.frame_idx == 5
    np.testing.assert_allclose(out.centroids[0], [50.0, 30.0], atol=1.5)
    np.testing.assert_allclose(out.sizes[0], 800.0, atol=60.0)  # ~40*20
    assert out.angles[0] < math.radians(8) or out.angles[0] > math.radians(172)
    np.testing.assert_allclose(out.confidences[0], 0.8, atol=1e-4)


def test_extract_obb_from_masks_no_masks_returns_empty_result():
    from types import SimpleNamespace

    from hydra_suite.core.inference.stages.obb import _extract_obb_from_masks

    result = SimpleNamespace(masks=None, boxes=SimpleNamespace(conf=None), orig_shape=(10, 10))
    out = _extract_obb_from_masks(result, frame_idx=1)
    assert out.num_detections == 0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_inference_stages_obb.py -k extract_obb_from_masks -v`
Expected: FAIL with `ImportError: cannot import name '_extract_obb_from_masks'`

- [ ] **Step 3: Implement**

Add near the top of `src/hydra_suite/core/inference/stages/obb.py`, alongside the existing `..runtime_artifacts` import (line 15):

```python
from ...detectors._obb_from_mask import _letterbox_gain_pad, rotated_rect_from_masks
```

(Confirm the relative-import depth: `stages/obb.py` is `core/inference/stages/obb.py`, so `...detectors` resolves to `core.detectors` — three levels up from `stages` reaches `core`. Adjust to `from hydra_suite.core.detectors._obb_from_mask import ...` (absolute) if that matches this file's existing import style better; check line 12-15's existing relative-vs-absolute convention before choosing.)

Add after `_extract_obb_from_boxes` (from Task 2):

```python
def _extract_obb_from_masks(
    result: Any,
    frame_idx: int,
    offset: tuple[float, float] = (0.0, 0.0),
    scale: tuple[float, float] = (1.0, 1.0),
) -> OBBResult:
    """Build an OBBResult from a segmentation model's predicted masks.

    Angle/size come from ``rotated_rect_from_masks`` (GPU-native, no cv2) run
    on the mask tensor's own square coordinate space; ``_letterbox_gain_pad``
    converts the caller's original-frame boxes into that space beforehand and
    the resulting (cx, cy, w, h) back afterwards -- a single uniform gain plus
    translation, which (unlike independent per-axis ratios) never distorts
    the recovered angle. The result is then folded through the same
    ``_normalize_obb_geometry`` / ``_corners_from_xywhr`` pipeline as every
    other OBB source for output-contract consistency.
    """
    masks = result.masks
    if masks is None or masks.data is None or masks.data.shape[0] == 0:
        return _empty_obb_result(frame_idx)
    mask_tensor = masks.data
    boxes = result.boxes
    conf_all = boxes.conf if boxes is not None else None
    if conf_all is None or len(conf_all) == 0:
        return _empty_obb_result(frame_idx)

    gain, pad_x, pad_y = _letterbox_gain_pad(
        tuple(mask_tensor.shape[-2:]), tuple(result.orig_shape)
    )
    boxes_orig = boxes.xyxy
    pad = torch.tensor(
        [pad_x, pad_y, pad_x, pad_y], device=boxes_orig.device, dtype=boxes_orig.dtype
    )
    boxes_mask_space = boxes_orig * gain + pad

    rect_mask_space = rotated_rect_from_masks(mask_tensor, boxes_mask_space)
    cx_m, cy_m, w_m, h_m, angle_rad = rect_mask_space.unbind(-1)
    cx = ((cx_m - pad_x) / gain).cpu().numpy()
    cy = ((cy_m - pad_y) / gain).cpu().numpy()
    w_arr = (w_m / gain).cpu().numpy()
    h_arr = (h_m / gain).cpu().numpy()
    angle_arr = angle_rad.cpu().numpy()
    conf = conf_all.cpu().numpy()

    ox, oy = offset
    sx, sy = scale
    cx = cx * sx + ox
    cy = cy * sy + oy
    w_arr = w_arr * sx
    h_arr = h_arr * sy

    angles_fixed, sizes, aspect = _normalize_obb_geometry(w_arr, h_arr, angle_arr)
    mask_valid = _valid_detection_mask(cx, cy, w_arr, h_arr, angles_fixed, conf)
    if not mask_valid.all():
        dropped = int(mask_valid.size - int(mask_valid.sum()))
        if dropped > 0:
            logger.warning(
                "Dropping %d invalid segment-as-OBB detections (non-finite "
                "geometry or empty mask crop).",
                dropped,
            )
        cx, cy, w_arr, h_arr = (
            cx[mask_valid],
            cy[mask_valid],
            w_arr[mask_valid],
            h_arr[mask_valid],
        )
        conf, angles_fixed, sizes, aspect = (
            conf[mask_valid],
            angles_fixed[mask_valid],
            sizes[mask_valid],
            aspect[mask_valid],
        )
    n = int(len(conf))
    corners = _corners_from_xywhr(cx, cy, w_arr, h_arr, angles_fixed)
    return OBBResult(
        frame_idx=frame_idx,
        centroids=np.stack([cx, cy], axis=1).astype(np.float32),
        angles=angles_fixed,
        sizes=sizes,
        shapes=np.stack([sizes, aspect], axis=1).astype(np.float32),
        confidences=conf.astype(np.float32),
        corners=corners.astype(np.float32),
        detection_ids=OBBResult.make_detection_ids(frame_idx, n),
    )
```

Note: `_valid_detection_mask`'s `np.isfinite` checks already reject the `NaN` rows `rotated_rect_from_masks` emits for empty-mask detections — no separate empty-crop handling needed here.

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_inference_stages_obb.py -k extract_obb_from_masks -v`
Expected: PASS

- [ ] **Step 5: Write the failing test for the zero-CPU-sync fast-path variant**

Add to `tests/test_inference_stages_obb.py`:

```python
def test_extract_raw_tensors_from_masks_keeps_everything_on_device():
    import torch
    from types import SimpleNamespace

    from hydra_suite.core.inference.stages.obb import _extract_raw_tensors_from_masks

    mh = mw = 100
    ys, xs = torch.meshgrid(
        torch.arange(mh, dtype=torch.float32),
        torch.arange(mw, dtype=torch.float32),
        indexing="ij",
    )
    mask = ((xs >= 30) & (xs <= 70) & (ys >= 20) & (ys <= 40)).float().unsqueeze(0)

    result = SimpleNamespace(
        masks=SimpleNamespace(data=mask),
        boxes=SimpleNamespace(
            xyxy=torch.tensor([[30.0, 20.0, 70.0, 40.0]]),
            conf=torch.tensor([0.8]),
        ),
        orig_shape=(100, 100),
    )

    raw = _extract_raw_tensors_from_masks(result, frame_idx=5, device="cpu")

    assert raw.frame_idx == 5
    assert isinstance(raw.xywhr, torch.Tensor)
    assert raw.xywhr.shape == (1, 5)
    assert raw.xywhr.device.type == "cpu"  # sanity: still a tensor, no numpy conversion
    torch.testing.assert_close(raw.conf, torch.tensor([0.8]))


def test_extract_raw_tensors_from_masks_no_masks_returns_empty():
    import torch
    from types import SimpleNamespace

    from hydra_suite.core.inference.stages.obb import _extract_raw_tensors_from_masks

    result = SimpleNamespace(masks=None, boxes=SimpleNamespace(conf=None), orig_shape=(10, 10))
    raw = _extract_raw_tensors_from_masks(result, frame_idx=1, device="cpu")
    assert raw.xywhr.shape == (0, 5)
```

- [ ] **Step 6: Run test to verify it fails**

Run: `python -m pytest tests/test_inference_stages_obb.py -k extract_raw_tensors_from_masks -v`
Expected: FAIL with `ImportError: cannot import name '_extract_raw_tensors_from_masks'`

- [ ] **Step 7: Implement the fast-path variant**

Add directly after `_extract_obb_from_masks`:

```python
def _extract_raw_tensors_from_masks(
    result: Any, frame_idx: int, device: str
) -> _RawOBBTensors:
    """Keep segment-as-OBB tensors on the compute device -- no .cpu() call.

    Mirrors _extract_raw_tensors_from_boxes's contract: the gain/pad
    conversion is plain tensor arithmetic (not a sync), and
    rotated_rect_from_masks already returns a device tensor with no internal
    .cpu() calls, so this function never leaves the accelerator.
    normalize/corners/finite-value filtering is deferred to
    materialize_tensors().
    """
    masks = result.masks
    if masks is None or masks.data is None or masks.data.shape[0] == 0:
        dev = torch.device(device)
        return _RawOBBTensors(
            frame_idx=frame_idx,
            xywhr=torch.zeros((0, 5), dtype=torch.float32, device=dev),
            corners=torch.zeros((0, 4, 2), dtype=torch.float32, device=dev),
            conf=torch.zeros(0, dtype=torch.float32, device=dev),
        )
    boxes = result.boxes
    conf_all = boxes.conf if boxes is not None else None
    if conf_all is None or len(conf_all) == 0:
        dev = torch.device(device)
        return _RawOBBTensors(
            frame_idx=frame_idx,
            xywhr=torch.zeros((0, 5), dtype=torch.float32, device=dev),
            corners=torch.zeros((0, 4, 2), dtype=torch.float32, device=dev),
            conf=torch.zeros(0, dtype=torch.float32, device=dev),
        )
    gain, pad_x, pad_y = _letterbox_gain_pad(
        tuple(masks.data.shape[-2:]), tuple(result.orig_shape)
    )
    boxes_orig = boxes.xyxy
    pad = torch.tensor(
        [pad_x, pad_y, pad_x, pad_y], device=boxes_orig.device, dtype=boxes_orig.dtype
    )
    boxes_mask_space = boxes_orig * gain + pad
    rect_mask_space = rotated_rect_from_masks(masks.data, boxes_mask_space)
    cx_m, cy_m, w_m, h_m, angle = rect_mask_space.unbind(-1)
    cx, cy = (cx_m - pad_x) / gain, (cy_m - pad_y) / gain
    w_arr, h_arr = w_m / gain, h_m / gain
    xywhr = torch.stack([cx, cy, w_arr, h_arr, angle], dim=1)
    # NaN rows from rotated_rect_from_masks (empty mask crops) are dropped
    # later by materialize_tensors()'s existing isfinite-based valid-mask
    # check -- no special-casing needed here.
    corners = torch.zeros(
        (xywhr.shape[0], 4, 2), dtype=torch.float32, device=xywhr.device
    )
    return _RawOBBTensors(
        frame_idx=frame_idx, xywhr=xywhr, corners=corners, conf=conf_all
    )
```

- [ ] **Step 8: Run tests to verify they pass**

Run: `python -m pytest tests/test_inference_stages_obb.py -k "extract_obb_from_masks or extract_raw_tensors_from_masks" -v`
Expected: PASS

- [ ] **Step 9: Commit**

```bash
git add src/hydra_suite/core/inference/stages/obb.py tests/test_inference_stages_obb.py
git commit -m "feat(inference): extract OBB from segmentation masks via GPU rotated-rect search"
```

---

### Task 5: Wire `model_task` dispatch into `load_obb_models` / `_run_direct`

**Files:**
- Modify: `src/hydra_suite/core/inference/stages/obb.py:284-345` (`load_obb_models`, `run_obb`, `_run_direct`)
- Test: `tests/test_inference_stages_obb.py`

**Interfaces:**
- Consumes: `_extract_obb_from_boxes`/`_extract_raw_tensors_from_boxes` (Task 2), `_extract_obb_from_masks`/`_extract_raw_tensors_from_masks` (Task 4), `OBBDirectConfig.model_task`/`fixed_angle_deg` (Task 1).
- Produces: `run_obb(...)` now returns correct `OBBResult`/`_RawOBBTensors` for `model_task in {"obb","detect","segment"}` under any runtime, with `detect`/`segment` getting the same zero-CPU-sync `_RawOBBTensors` fast path as `"obb"` whenever `runtime.tensor_on_cuda` is true.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_inference_stages_obb.py`. Follow the file's existing pattern for faking `models.direct_model.predict(...)` (check how existing `_run_direct` tests construct a fake model — reuse that fixture style):

```python
def test_run_direct_dispatches_to_detect_extraction(monkeypatch):
    import numpy as np
    import torch
    from types import SimpleNamespace

    from hydra_suite.core.inference.config import OBBConfig, OBBDirectConfig
    from hydra_suite.core.inference.stages.obb import OBBModels, run_obb

    class _FakeDetectModel:
        def predict(self, frames, **kwargs):
            return [
                SimpleNamespace(
                    boxes=SimpleNamespace(
                        xyxy=torch.tensor([[0.0, 0.0, 10.0, 10.0]]),
                        conf=torch.tensor([0.7]),
                    )
                )
                for _ in frames
            ]

    config = OBBConfig(
        mode="direct",
        direct=OBBDirectConfig(
            model_path="fake.pt", model_task="detect", fixed_angle_deg=45.0
        ),
    )
    models = OBBModels(mode="direct", direct_model=_FakeDetectModel())
    runtime = SimpleNamespace(tensor_on_cuda=False, device="cpu")

    results = run_obb([np.zeros((20, 20, 3), dtype=np.uint8)], models, config, runtime)

    assert len(results) == 1
    assert results[0].num_detections == 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_inference_stages_obb.py -k dispatches_to_detect -v`
Expected: FAIL — `_run_direct` calls `_extract_obb_result` (expects `.obb`), so `SimpleNamespace` without an `obb` attribute raises `AttributeError`.

- [ ] **Step 3: Implement**

In `src/hydra_suite/core/inference/stages/obb.py`:

1. `load_obb_models` (line ~295-304), pass `task` through for direct mode:

```python
    if config.mode == "direct":
        assert config.direct is not None
        auto_export = config.direct.auto_export
        m = _load_yolo(
            config.direct.model_path,
            compute_runtime,
            auto_export=auto_export,
            max_det=config.max_detections,
            task=config.direct.model_task,
        )
        return OBBModels(mode="direct", direct_model=m)
```

2. `_run_direct` (line 348-415), replace the final dispatch (after the `model.predict(...)` call that produces `results`) with a `model_task`-aware branch. Replace:

```python
    # Only native PyTorch "cuda" runtime leaves tensors on device.
    # onnx_cuda and tensorrt: predict() returns CPU numpy regardless of GPU use.
    if runtime.tensor_on_cuda:
        return [
            _extract_raw_tensors(r, idx, runtime.device)
            for idx, r in enumerate(results)
        ]
    return [
        _apply_raw_detection_cap(_extract_obb_result(r, idx), config.raw_detection_cap)
        for idx, r in enumerate(results)
    ]
```

with:

```python
    model_task = config.direct.model_task if config.direct else "obb"

    if model_task == "detect":
        fixed_angle_rad = math.radians(
            config.direct.fixed_angle_deg if config.direct else 0.0
        )
        # Zero-CPU-sync fast path under the native cuda runtime, mirroring
        # "obb"'s own tensor_on_cuda branch below -- normalize/corners/
        # finite-filtering is deferred to the shared materialize_tensors().
        if runtime.tensor_on_cuda:
            return [
                _extract_raw_tensors_from_boxes(r, idx, fixed_angle_rad, runtime.device)
                for idx, r in enumerate(results)
            ]
        return [
            _apply_raw_detection_cap(
                _extract_obb_from_boxes(r, idx, fixed_angle_rad),
                config.raw_detection_cap,
            )
            for idx, r in enumerate(results)
        ]

    if model_task == "segment":
        # rotated_rect_from_masks does all the heavy per-pixel/per-angle work
        # on-device with no internal .cpu() calls, so under the native cuda
        # runtime segment gets the exact same zero-CPU-sync _RawOBBTensors
        # fast path as "obb"/"detect" -- the sync is deferred to
        # materialize_tensors(), same as every other detection source.
        if runtime.tensor_on_cuda:
            return [
                _extract_raw_tensors_from_masks(r, idx, runtime.device)
                for idx, r in enumerate(results)
            ]
        return [
            _apply_raw_detection_cap(
                _extract_obb_from_masks(r, idx), config.raw_detection_cap
            )
            for idx, r in enumerate(results)
        ]

    # model_task == "obb": existing native-OBB behaviour, unchanged.
    # Only native PyTorch "cuda" runtime leaves tensors on device.
    # onnx_cuda and tensorrt: predict() returns CPU numpy regardless of GPU use.
    if runtime.tensor_on_cuda:
        return [
            _extract_raw_tensors(r, idx, runtime.device)
            for idx, r in enumerate(results)
        ]
    return [
        _apply_raw_detection_cap(_extract_obb_result(r, idx), config.raw_detection_cap)
        for idx, r in enumerate(results)
    ]
```

3. Add `import math` near the top of the file (alongside the `import cv2` etc. block, line 1-9) if not already present.

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_inference_stages_obb.py -v`
Expected: PASS (full file, to confirm no regression on existing `obb`/`sequential` tests)

- [ ] **Step 5: Write the failing test for the fast-path dispatch under `tensor_on_cuda`**

Add to `tests/test_inference_stages_obb.py`:

```python
def test_run_direct_detect_uses_raw_tensor_fast_path_when_tensor_on_cuda():
    import numpy as np
    import torch
    from types import SimpleNamespace

    from hydra_suite.core.inference.config import OBBConfig, OBBDirectConfig
    from hydra_suite.core.inference.stages.obb import OBBModels, run_obb

    class _FakeDetectModel:
        def predict(self, frames, **kwargs):
            return [
                SimpleNamespace(
                    boxes=SimpleNamespace(
                        xyxy=torch.tensor([[0.0, 0.0, 10.0, 10.0]]),
                        conf=torch.tensor([0.7]),
                    )
                )
                for _ in frames
            ]

    config = OBBConfig(
        mode="direct",
        direct=OBBDirectConfig(model_path="fake.pt", model_task="detect"),
    )
    models = OBBModels(mode="direct", direct_model=_FakeDetectModel())
    runtime = SimpleNamespace(tensor_on_cuda=True, device="cpu")

    results = run_obb([np.zeros((20, 20, 3), dtype=np.uint8)], models, config, runtime)

    # tensor_on_cuda=True must return _RawOBBTensors (a torch-tensor
    # namedtuple), NOT an already-materialized OBBResult.
    assert hasattr(results[0], "xywhr")
    assert not hasattr(results[0], "corners") or isinstance(results[0].xywhr, torch.Tensor)
```

- [ ] **Step 6: Run test to verify it fails**

Run: `python -m pytest tests/test_inference_stages_obb.py -k raw_tensor_fast_path -v`
Expected: FAIL — before this task's Step 3 edit, `_run_direct`'s `tensor_on_cuda` branch is unconditional and only calls `_extract_raw_tensors` (the `"obb"`-only function), which reads `r.obb` and raises `AttributeError` on the fake detect `Results`.

- [ ] **Step 7: Run the full file again to confirm the fast-path dispatch (Step 3's edit) fixes it**

Run: `python -m pytest tests/test_inference_stages_obb.py -v`
Expected: PASS

- [ ] **Step 8: Commit**

```bash
git add src/hydra_suite/core/inference/stages/obb.py tests/test_inference_stages_obb.py
git commit -m "feat(inference): dispatch direct-mode OBB extraction on model_task"
```

---

### Task 6: `create_direct_segment_executor` factory + `_create_direct_executor` dispatch

**Files:**
- Modify: `src/hydra_suite/core/inference/runtime_artifacts.py:214-253` (`_create_direct_executor`)
- Test: `tests/test_inference_obb_artifacts.py`

**Interfaces:**
- Consumes: `create_direct_segment_executor` (defined in Task 7, imported lazily exactly like the existing `create_direct_detect_executor`/`create_direct_obb_executor` imports at `runtime_artifacts.py:238-241`).
- Produces: `_create_direct_executor(task="segment", ...)` routes to the new factory. `load_obb_executor(..., task="segment")` (already generic — no change needed there) now returns a working `DirectExecutorAdapter` for segment models.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_inference_obb_artifacts.py`. First check the file's existing `fake_loader` fixture (around line 51+) to see exactly which hooks it monkeypatches (`_load_torch_model`, `_export_artifact`, `_create_direct_executor`) — reuse that fixture. Add:

```python
def test_load_obb_executor_segment_task_routes_to_create_direct_executor(
    fake_loader, tmp_path
):
    from hydra_suite.core.inference.runtime_artifacts import load_obb_executor

    pt_path = tmp_path / "model.pt"
    pt_path.write_bytes(b"fake")

    load_obb_executor(str(pt_path), "tensorrt", auto_export=True, task="segment")

    # fake_loader records _create_direct_executor calls; assert the last call
    # (or whichever attribute the fixture exposes -- match its existing
    # assertion style used by the neighbouring task="detect" test) was made
    # with task="segment".
    assert fake_loader.create_direct_calls[-1]["task"] == "segment"
```

(If `fake_loader`'s fixture records calls under a different attribute name than `create_direct_calls`, use that name instead — match whatever the existing `test_load_obb_executor_...task="detect"...` test in this file already asserts on, for consistency.)

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_inference_obb_artifacts.py -k segment_task -v`
Expected: FAIL — either the fixture call-recording assertion mismatches (align it first), or it passes trivially once aligned (confirming the *selection* plumbing already forwards `task` verbatim to the monkeypatched hook — in that case proceed to Step 3 for the real dispatch branch, which Task 7's import will make importable).

- [ ] **Step 3: Implement**

In `src/hydra_suite/core/inference/runtime_artifacts.py`, edit `_create_direct_executor` (lines 214-253):

```python
def _create_direct_executor(
    *,
    runtime: str,
    artifact_path: Path,
    imgsz: int,
    class_names: dict[int, str] | None = None,
    task: str = "obb",
) -> Any:
    """Create a direct ONNX/TRT executor (square-letterbox preprocessing).

    ...(existing docstring, extend the task paragraph)...

    ``task="segment"`` returns an executor that decodes the model's raw
    detection + mask-prototype outputs and derives OBB geometry via
    ``core.detectors._obb_from_mask.rotated_rect_from_masks`` -- a GPU-native,
    cv2-free batched rotated-rectangle search -- for treating a YOLO
    segmentation checkpoint as an OBB source.
    """
    from hydra_suite.core.detectors._direct_obb_runtime import (
        create_direct_detect_executor,
        create_direct_obb_executor,
        create_direct_segment_executor,
    )

    if task == "detect":
        factory = create_direct_detect_executor
    elif task == "segment":
        factory = create_direct_segment_executor
    else:
        factory = create_direct_obb_executor
    return factory(
        runtime=runtime,
        artifact_path=str(artifact_path),
        imgsz=int(imgsz),
        class_names=class_names,
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_inference_obb_artifacts.py -v`
Expected: PASS (this will still fail until Task 7 defines `create_direct_segment_executor` — implement Task 7 first if running this test standalone, or land Tasks 6 and 7 in the same review pass.)

- [ ] **Step 5: Commit**

```bash
git add src/hydra_suite/core/inference/runtime_artifacts.py tests/test_inference_obb_artifacts.py
git commit -m "feat(inference): route task=segment to a dedicated direct executor factory"
```

---

### Task 7: `DirectTensorRTSegmentExecutor` — accelerated segment-as-OBB, no `cv2`/`Results.masks.xy`

**Files:**
- Modify: `src/hydra_suite/core/detectors/_direct_obb_runtime.py` (add after `create_direct_detect_executor`, i.e. after line 989)
- Test: `tests/test_direct_obb_runtime_segment.py` (new)

**Interfaces:**
- Consumes: `_BaseDirectOBBExecutor` (existing base class), `ultralytics.utils.nms.non_max_suppression`, `ultralytics.utils.ops.process_mask`/`scale_boxes`, `rotated_rect_from_masks` (Task 3).
- Produces: `_decode_segment_predictions(preds, protos, img_tensor_shape, orig_shape, *, conf_thres, classes, max_det, nc) -> list[SimpleNamespace]` (pure function, CPU-testable) and `create_direct_segment_executor(*, runtime, artifact_path, imgsz, class_names=None, class_count=None)` — consumed by Task 6.

This is the highest-risk task: it is the only piece of new code that talks to a raw two-output TensorRT engine. It is de-risked by putting ALL of the tensor math (NMS + mask decode + rotated-rect search) in one pure function that takes plain tensors and is fully unit-tested; the executor class itself is a thin wrapper that cannot reasonably be unit-tested without a real TensorRT engine (same limitation `DirectTensorRTOBBExecutor` already has today).

**Design note — no `ultralytics.engine.results.Results` needed here.** Earlier drafts of this plan built a full `Results(masks=...)` object and relied on `Results.masks.xy` to convert the mask to polygon points in original-frame space — that property internally calls `cv2.findContours`, which is exactly what we're avoiding. Since `_extract_obb_from_masks` (Task 4) only ever reads `.masks.data`, `.boxes.xyxy`, `.boxes.conf`, and `.orig_shape` off its `result` argument, `_decode_segment_predictions` returns a lightweight duck-typed `SimpleNamespace` exposing exactly those four attributes instead of a real `Results` — cheaper to construct and it makes the "no cv2" property visible directly in this function's return type rather than depending on which `Results` methods happen not to be called.

- [ ] **Step 1: Write the failing test**

Create `tests/test_direct_obb_runtime_segment.py`:

```python
"""CPU unit tests for the segment-as-OBB raw-output decode (Task 7).

Exercises _decode_segment_predictions with synthetic CPU tensors shaped like
a real YOLO-segment raw head output -- no TensorRT/ONNX session, no GPU, and
no ultralytics Results/cv2 machinery is involved.
"""

from __future__ import annotations

import torch

from hydra_suite.core.detectors._direct_obb_runtime import (
    _decode_segment_predictions,
)


def _make_synthetic_prediction(nc: int, nm: int, num_anchors: int) -> torch.Tensor:
    """Build a (1, 4+nc+nm, num_anchors) raw-head tensor with one confident box."""
    pred = torch.zeros((1, 4 + nc + nm, num_anchors), dtype=torch.float32)
    pred[0, 0, 0] = 32.0  # cx
    pred[0, 1, 0] = 32.0  # cy
    pred[0, 2, 0] = 20.0  # w
    pred[0, 3, 0] = 10.0  # h
    pred[0, 4, 0] = 5.0  # class-0 score
    pred[0, 4 + nc : 4 + nc + nm, 0] = 1.0  # mask coefficients
    return pred


def test_decode_segment_predictions_returns_duck_typed_detections():
    nc, nm, mh, mw, imgsz = 1, 4, 16, 16, 64
    preds = _make_synthetic_prediction(nc, nm, num_anchors=8)
    protos = torch.ones((1, nm, mh, mw), dtype=torch.float32)

    results = _decode_segment_predictions(
        preds,
        protos,
        img_tensor_shape=(1, 3, imgsz, imgsz),
        orig_shape=(imgsz, imgsz),
        conf_thres=0.05,
        classes=None,
        max_det=10,
        nc=nc,
    )

    assert len(results) == 1
    r = results[0]
    assert r.orig_shape == (imgsz, imgsz)
    assert r.boxes is not None and len(r.boxes.conf) == 1
    assert r.masks is not None and r.masks.data.shape[0] == 1
    # Uniform-positive prototypes + all-ones coefficients -> a non-degenerate
    # (all-"on") decoded mask.
    assert r.masks.data.sum() > 0


def test_decode_segment_predictions_empty_below_threshold():
    nc, nm, mh, mw, imgsz = 1, 4, 8, 8, 32
    preds = torch.zeros((1, 4 + nc + nm, 4), dtype=torch.float32)
    protos = torch.zeros((1, nm, mh, mw), dtype=torch.float32)

    results = _decode_segment_predictions(
        preds,
        protos,
        img_tensor_shape=(1, 3, imgsz, imgsz),
        orig_shape=(imgsz, imgsz),
        conf_thres=0.5,
        classes=None,
        max_det=10,
        nc=nc,
    )
    assert len(results) == 1
    assert results[0].boxes is None or len(results[0].boxes.conf) == 0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_direct_obb_runtime_segment.py -v`
Expected: FAIL with `ImportError: cannot import name '_decode_segment_predictions'`

- [ ] **Step 3: Implement**

Append to `src/hydra_suite/core/detectors/_direct_obb_runtime.py` (after `create_direct_detect_executor`, line 989):

```python
# ---------------------------------------------------------------------------
# Direct YOLO segment executor (segment-as-OBB direct mode)
# ---------------------------------------------------------------------------
# Segmentation raw heads emit TWO outputs (detections + mask prototypes)
# instead of one, so these executors do NOT subclass DirectONNXOBBExecutor /
# DirectTensorRTOBBExecutor (which assert exactly one output tensor). All
# shared tensor math lives in _decode_segment_predictions, a pure function
# unit-testable on CPU tensors without a real TRT engine. It never builds an
# ultralytics Results.masks and never calls cv2 -- angle/size come from
# core.detectors._obb_from_mask.rotated_rect_from_masks, which is GPU-native
# and stays entirely in tensor ops.


def _decode_segment_predictions(
    preds,
    protos,
    *,
    img_tensor_shape,
    orig_shape,
    conf_thres: float,
    classes,
    max_det: int,
    nc: int,
):
    """Decode a raw YOLO-segment head output into duck-typed detections.

    Parameters
    ----------
    preds:
        Raw detection tensor, shape ``(B, 4+nc+nm, num_anchors)``.
    protos:
        Raw mask-prototype tensor, shape ``(B, nm, mh, mw)``.
    img_tensor_shape:
        The model input tensor's shape, e.g. ``(B, 3, imgsz, imgsz)`` — used
        by ``ops.scale_boxes`` to map letterbox-space boxes back to
        ``orig_shape`` pixel space.
    orig_shape:
        ``(H, W)`` of the true original frame.

    Returns
    -------
    list[types.SimpleNamespace]
        One namespace per input frame, each exposing exactly the four
        attributes ``_extract_obb_from_masks`` (stages/obb.py) reads:
        ``.orig_shape`` (``(H, W)``), ``.boxes.xyxy``/``.boxes.conf``
        (original-frame space), and ``.masks.data`` (the RAW, letterbox-
        space, NOT-upsampled proto-resolution mask tensor — cheaper than
        upsampling since ``rotated_rect_from_masks`` needs only a small
        crop of it and handles the resulting scale via
        ``_letterbox_gain_pad``, so upsampling first would be wasted work).
    """
    import types

    import torch
    from ultralytics.utils import nms, ops

    if not isinstance(preds, torch.Tensor):
        preds = torch.as_tensor(preds)
    if not isinstance(protos, torch.Tensor):
        protos = torch.as_tensor(protos)

    filtered = nms.non_max_suppression(
        preds,
        conf_thres=conf_thres,
        iou_thres=0.5,
        classes=classes,
        max_det=max_det,
        nc=nc,
        rotated=False,
    )

    results = []
    for i, pred in enumerate(filtered):
        if pred is None or len(pred) == 0:
            results.append(
                types.SimpleNamespace(orig_shape=tuple(orig_shape), boxes=None, masks=None)
            )
            continue
        proto_i = protos[i] if protos.shape[0] == len(filtered) else protos[0]
        boxes_letterboxed = pred[:, :4]
        mask_coeffs = pred[:, 6:]
        # upsample=False: keep proto resolution -- rotated_rect_from_masks
        # crops a small tile per detection regardless, so upsampling the
        # full mask to imgsz here would be wasted GPU work.
        masks = ops.process_mask(
            proto_i, mask_coeffs, boxes_letterboxed, img_tensor_shape[2:], upsample=False
        )
        boxes_orig = pred[:, :6].clone()
        boxes_orig[:, :4] = ops.scale_boxes(
            img_tensor_shape[2:], boxes_orig[:, :4], orig_shape
        )
        results.append(
            types.SimpleNamespace(
                orig_shape=tuple(orig_shape),
                boxes=types.SimpleNamespace(
                    xyxy=boxes_orig[:, :4], conf=boxes_orig[:, 4]
                ),
                masks=types.SimpleNamespace(data=masks),
            )
        )
    return results


class DirectTensorRTSegmentExecutor(_BaseDirectOBBExecutor):
    """Direct TensorRT executor for a YOLO *segment* checkpoint.

    Binds the engine's TWO output tensors (detections, mask prototypes)
    instead of the one output every other direct executor in this module
    assumes -- this is why it subclasses ``_BaseDirectOBBExecutor`` directly
    rather than ``DirectTensorRTOBBExecutor`` (whose ``__init__`` hard-asserts
    exactly one output tensor).
    """

    def __init__(
        self,
        artifact_path: str,
        imgsz: int,
        class_names: dict[int, str] | None = None,
        class_count: int | None = None,
    ) -> None:
        super().__init__(artifact_path, imgsz, class_names, class_count)

        import struct

        import tensorrt as trt  # type: ignore[import-not-found]

        with open(self.artifact_path, "rb") as handle:
            meta_len = struct.unpack("<I", handle.read(4))[0]
            meta_json = handle.read(meta_len).decode("utf-8")
            engine_data = handle.read()

        meta = json.loads(meta_json)
        if not self.names:
            self.names = {
                int(key): str(value)
                for key, value in dict(meta.get("names") or {}).items()
            }
            self.nc = max(1, len(self.names) or self.nc)

        logger = trt.Logger(trt.Logger.WARNING)
        runtime = trt.Runtime(logger)
        self.engine = runtime.deserialize_cuda_engine(engine_data)
        if self.engine is None:
            raise RuntimeError("TensorRT failed to deserialize segment engine")
        self.context = self.engine.create_execution_context()

        tensor_names = [
            self.engine.get_tensor_name(i) for i in range(self.engine.num_io_tensors)
        ]
        input_names = [
            n
            for n in tensor_names
            if self.engine.get_tensor_mode(n) == trt.TensorIOMode.INPUT
        ]
        output_names = [
            n
            for n in tensor_names
            if self.engine.get_tensor_mode(n) == trt.TensorIOMode.OUTPUT
        ]
        if len(input_names) != 1 or len(output_names) != 2:
            raise RuntimeError(
                "TensorRT segment engine must expose exactly one input and "
                "two outputs (detections, mask prototypes)"
            )
        self._input_name = input_names[0]
        # Prototypes are the rank-4 output (B, nm, mh, mw); detections are
        # rank-3 (B, 4+nc+nm, num_anchors). Disambiguate by tensor rank
        # rather than assuming export order, since output order is not
        # contractually guaranteed across ultralytics versions.
        shapes = {n: tuple(self.engine.get_tensor_shape(n)) for n in output_names}
        proto_candidates = [n for n, s in shapes.items() if len(s) == 4]
        det_candidates = [n for n, s in shapes.items() if len(s) == 3]
        if len(proto_candidates) != 1 or len(det_candidates) != 1:
            raise RuntimeError(
                f"Could not disambiguate segment engine outputs by rank: {shapes}"
            )
        self._proto_name = proto_candidates[0]
        self._det_name = det_candidates[0]

        batch_dim = self.engine.get_tensor_shape(self._input_name)[0]
        self._model_batch_size = int(batch_dim) if batch_dim > 0 else 1
        self._static_batch: bool = batch_dim > 0
        self._end2end = False  # segment raw heads are always CBC, never end2end

        import torch

        self._cuda_stream = torch.cuda.Stream()
        self._sync_event = torch.cuda.Event()

        try:
            _warmup = torch.zeros(
                (self._model_batch_size, 3, self.imgsz, self.imgsz),
                dtype=torch.float32,
                device="cuda",
            )
            self._run_inference(_warmup)
            self._run_inference(_warmup)
            torch.cuda.synchronize()
            del _warmup
        except Exception:
            pass

    def _run_inference(self, img_tensor):
        import torch

        x = img_tensor.float().contiguous()
        self.context.set_input_shape(self._input_name, tuple(x.shape))
        det_shape = tuple(self.context.get_tensor_shape(self._det_name))
        proto_shape = tuple(self.context.get_tensor_shape(self._proto_name))
        det_out = torch.empty(det_shape, dtype=torch.float32, device=x.device)
        proto_out = torch.empty(proto_shape, dtype=torch.float32, device=x.device)
        self.context.set_tensor_address(self._input_name, x.data_ptr())
        self.context.set_tensor_address(self._det_name, det_out.data_ptr())
        self.context.set_tensor_address(self._proto_name, proto_out.data_ptr())
        self._sync_event.record(torch.cuda.current_stream())
        self._cuda_stream.wait_event(self._sync_event)
        self.context.execute_async_v3(self._cuda_stream.cuda_stream)
        self._cuda_stream.synchronize()
        return det_out, proto_out

    def _postprocess(self, raw_preds, img_tensor, orig_frames, conf_thres, classes, max_det):
        det_out, proto_out = raw_preds
        orig_shape = orig_frames[0].shape[:2]
        return _decode_segment_predictions(
            det_out,
            proto_out,
            img_tensor_shape=tuple(img_tensor.shape),
            orig_shape=orig_shape,
            conf_thres=conf_thres,
            classes=classes,
            max_det=max_det,
            nc=self.nc,
        )


def create_direct_segment_executor(
    *,
    runtime: str,
    artifact_path: str,
    imgsz: int,
    class_names: dict[int, str] | None = None,
    class_count: int | None = None,
):
    """Instantiate a direct executor for the YOLO *segment* task.

    Only ``"tensorrt"`` is supported: ``load_obb_executor`` never requests
    ``onnx_*`` runtimes for OBB (see ``runtime_artifacts.ArtifactExportError``
    for unsupported runtimes), and cpu/mps/cuda/coreml already work through
    the plain ultralytics-model path (no direct executor involved).
    """
    runtime_name = str(runtime or "").strip().lower()
    if runtime_name == "tensorrt":
        return DirectTensorRTSegmentExecutor(
            artifact_path,
            imgsz,
            class_names=class_names,
            class_count=class_count,
        )
    raise ValueError(f"Unsupported direct segment runtime: {runtime}")
```

Note: `_BaseDirectOBBExecutor.predict`/`_predict_chunk` (lines 292-384) call `self._postprocess(raw_preds, ...)` where `raw_preds = self._run_inference(img_tensor)` — for the OBB/detect executors `_run_inference` returns a single tensor, but `DirectTensorRTSegmentExecutor._run_inference` returns a `(det, proto)` tuple. `_predict_chunk` (line 376) passes `raw_preds` straight through to `_postprocess` untouched, so no base-class change is needed.

Since `DirectExecutorAdapter.predict` (in `runtime_artifacts.py`) already forwards the executor's `predict()` output straight through to `stages/obb.py`'s `_run_direct`, and `_run_direct`'s `model_task == "segment"` branch (Task 5) calls `_extract_obb_from_masks(r, idx)` on each returned item — the `SimpleNamespace` objects `_decode_segment_predictions` produces satisfy that contract directly, no adapter change needed.

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_direct_obb_runtime_segment.py -v`
Expected: PASS

- [ ] **Step 5: Run the full inference test suite to confirm no regression**

Run: `python -m pytest tests/test_inference_obb_artifacts.py tests/test_inference_stages_obb.py tests/test_obb_from_mask.py tests/test_direct_obb_runtime_segment.py -v`
Expected: PASS (this also completes Task 6's Step 4, deferred until this task supplied `create_direct_segment_executor`)

- [ ] **Step 6: Commit**

```bash
git add src/hydra_suite/core/detectors/_direct_obb_runtime.py src/hydra_suite/core/inference/runtime_artifacts.py tests/test_direct_obb_runtime_segment.py tests/test_inference_obb_artifacts.py
git commit -m "feat(inference): add TensorRT direct executor for segment-as-OBB (cv2-free)"
```

---

### Task 8: Thread `model_task`/`fixed_angle_deg` through the tracking-worker params bridge

**Files:**
- Modify: `src/hydra_suite/core/tracking/worker.py:4349-4458` (`_build_inference_config_from_params`)
- Test: search the repo for the existing test of `_build_inference_config_from_params` (grep for the method name) and extend it; if no direct unit test exists, add one.

**Interfaces:**
- Consumes: `params.get("YOLO_OBB_DIRECT_TASK", "obb")`, `params.get("YOLO_OBB_FIXED_ANGLE_DEG", 0.0)` — new legacy-style params-dict keys, following the exact convention of every other key in this method (e.g. `YOLO_OBB_MODE`, `MAX_TARGETS`).
- Produces: `OBBDirectConfig(model_task=..., fixed_angle_deg=...)` populated in the `obb_mode != "sequential"` branch.

- [ ] **Step 1: Write the failing test**

First run: `grep -rn "_build_inference_config_from_params" tests/` to find the existing test file and its exact call/fixture pattern (it likely constructs a `TrackingWorker` or calls the method directly via a test hook — match that pattern exactly rather than guessing). Add a test alongside the existing ones for the `direct` branch:

```python
def test_build_inference_config_from_params_threads_model_task_and_angle(...):
    # Use the same worker/params construction as the neighbouring
    # test_build_inference_config_from_params_direct_mode-style test in this
    # file.
    params = {
        "YOLO_OBB_MODE": "direct",
        "YOLO_OBB_DIRECT_MODEL_PATH": "yolo26s-seg.pt",
        "YOLO_OBB_DIRECT_TASK": "segment",
        "YOLO_OBB_FIXED_ANGLE_DEG": 12.5,
    }
    cfg = worker._build_inference_config_from_params(params)
    assert cfg.obb.direct.model_task == "segment"
    assert cfg.obb.direct.fixed_angle_deg == 12.5
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest <the located test file> -k model_task_and_angle -v`
Expected: FAIL with `AssertionError: assert 'obb' == 'segment'` (the field exists from Task 1 with its default, but nothing populates it yet).

- [ ] **Step 3: Implement**

In `src/hydra_suite/core/tracking/worker.py`, edit the `else:` branch (direct mode) inside `_build_inference_config_from_params` (lines 4440-4458):

```python
        else:
            model_task = str(params.get("YOLO_OBB_DIRECT_TASK", "obb")).strip().lower()
            if model_task not in {"obb", "detect", "segment"}:
                model_task = "obb"
            obb_cfg = OBBConfig(
                mode="direct",
                direct=OBBDirectConfig(
                    model_path=direct_model_path,
                    compute_runtime=compute_runtime,
                    confidence_floor=1e-3,
                    confidence_threshold=yolo_conf,
                    model_task=model_task,
                    fixed_angle_deg=float(params.get("YOLO_OBB_FIXED_ANGLE_DEG", 0.0)),
                ),
                target_classes=target_classes,
                confidence_threshold=yolo_conf,
                iou_threshold=yolo_iou,
                min_object_size=min_obj,
                max_object_size=max_obj,
                min_aspect_ratio=min_ar,
                max_aspect_ratio=max_ar,
                max_detections=max_dets,
                raw_detection_cap=raw_cap,
            )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest <the located test file> -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/hydra_suite/core/tracking/worker.py <test file>
git commit -m "feat(trackerkit): thread YOLO_OBB_DIRECT_TASK/FIXED_ANGLE_DEG into OBBDirectConfig"
```

---

### Task 9: Minimal GUI toggle in `detection_panel.py`

**Files:**
- Modify: `src/hydra_suite/trackerkit/gui/panels/detection_panel.py` (near lines 570-618, alongside `combo_yolo_obb_mode`/`combo_yolo_model`)

**Interfaces:**
- Produces: two new widgets, `self.combo_yolo_direct_task` (QComboBox: "OBB (native)" / "Detect (fixed angle)" / "Segment (rotated mask)") and `self.spin_yolo_fixed_angle` (a `QDoubleSpinBox`, degrees, range -180..180, visible only when "Detect" is selected). Their values must be read wherever `combo_yolo_obb_mode`'s value is currently read into the params dict (search this file and `orchestrators/config.py` for every place `"YOLO_OBB_MODE"` or `combo_yolo_obb_mode.currentIndex()` is written, e.g. lines 1285, 1763, 2005, 2037 — add the two new keys, `YOLO_OBB_DIRECT_TASK` and `YOLO_OBB_FIXED_ANGLE_DEG`, at each corresponding save-to-params-dict site) and wherever the panel is populated FROM a loaded config/params dict (the counterpart load path for `combo_yolo_obb_mode`).

Because this panel is a single ~2000+ line file already flagged for a future monolith split (CLAUDE.md Slice 4) and its save/load round-trip touches several call sites, this task is scoped to the two new widgets and their direct wiring; do not restructure the panel.

- [ ] **Step 1: Add the widgets**

After `f_yolo.addRow("YOLO OBB mode", self.combo_yolo_obb_mode)` (line 582) and its warning label (lines 584-590), add:

```python
        self.combo_yolo_direct_task = QComboBox()
        self.combo_yolo_direct_task.addItems(
            ["OBB (native)", "Detect (fixed angle)", "Segment (rotated mask)"]
        )
        self.combo_yolo_direct_task.setFixedHeight(30)
        self.combo_yolo_direct_task.currentIndexChanged.connect(
            self._on_yolo_direct_task_changed
        )
        self.combo_yolo_direct_task.setToolTip(
            "Direct-mode model source: a native OBB checkpoint, a plain "
            "detect checkpoint (fixed angle applied to every detection), or "
            "a segmentation checkpoint (angle derived from a GPU-native "
            "rotated-rectangle search over the predicted mask)."
        )
        f_yolo.addRow("Direct model task", self.combo_yolo_direct_task)

        self.spin_yolo_fixed_angle = QDoubleSpinBox()
        self.spin_yolo_fixed_angle.setRange(-180.0, 180.0)
        self.spin_yolo_fixed_angle.setDecimals(1)
        self.spin_yolo_fixed_angle.setSuffix(" deg")
        self.spin_yolo_fixed_angle.setFixedHeight(30)
        self.spin_yolo_fixed_angle.setToolTip(
            "Fixed OBB angle applied to every detection when Direct model "
            "task is 'Detect (fixed angle)'."
        )
        self.lbl_fixed_angle = f_yolo.addRow(
            "Fixed angle", self.spin_yolo_fixed_angle
        )
```

(`QDoubleSpinBox` must already be imported at the top of this file alongside `QComboBox`/`QCheckBox` — verify and add the import if missing.)

- [ ] **Step 2: Add the visibility-toggle handler**

Add a method next to `_on_yolo_mode_changed` (near line 1988):

```python
    def _on_yolo_direct_task_changed(self, _index: object) -> object:
        is_detect = self.combo_yolo_direct_task.currentIndex() == 1
        self.spin_yolo_fixed_angle.setVisible(is_detect)
        # If the row widget helper used elsewhere in this panel returns a
        # QWidget wrapping the row's label, toggle that too; otherwise the
        # form-layout row created by f_yolo.addRow above needs its own
        # label-widget reference toggled here as well, matching how
        # lbl_obb_mode_warning's row is shown/hidden by _on_yolo_mode_changed.
```

Call `self._on_yolo_direct_task_changed(self.combo_yolo_direct_task.currentIndex())` once at the end of panel init, mirroring the existing call at line 793 (`self._on_yolo_mode_changed(...)`).

- [ ] **Step 3: Wire save/load**

At each of the params-dict save sites already identified for `combo_yolo_obb_mode` (grep `"YOLO_OBB_MODE"` in this file and in `orchestrators/config.py`), add the two companion keys:

```python
"YOLO_OBB_DIRECT_TASK": ["obb", "detect", "segment"][
    self.combo_yolo_direct_task.currentIndex()
],
"YOLO_OBB_FIXED_ANGLE_DEG": self.spin_yolo_fixed_angle.value(),
```

At the corresponding load-from-params site (where `combo_yolo_obb_mode.setCurrentIndex(...)` is populated from a loaded config), add:

```python
self.combo_yolo_direct_task.setCurrentIndex(
    ["obb", "detect", "segment"].index(
        params.get("YOLO_OBB_DIRECT_TASK", "obb")
    )
)
self.spin_yolo_fixed_angle.setValue(
    float(params.get("YOLO_OBB_FIXED_ANGLE_DEG", 0.0))
)
```

- [ ] **Step 4: Manual verification**

Run: `trackerkit` (per CLAUDE.md's launch command), open the detection panel, switch "Direct model task" to "Detect (fixed angle)" and confirm the "Fixed angle" row appears/disappears correctly, then save and reload a project config and confirm the selection round-trips. This is a GUI-only change with no automated test in this task — verify manually since Qt widget wiring is not meaningfully unit-testable without a running `QApplication` fixture (check whether this repo already has one in `tests/` before deciding to skip automated coverage here; if a `qtbot`/`pytest-qt` fixture pattern already exists for this panel, use it instead of manual-only verification).

- [ ] **Step 5: Commit**

```bash
git add src/hydra_suite/trackerkit/gui/panels/detection_panel.py
git commit -m "feat(trackerkit): add direct-mode task selector and fixed-angle control"
```

---

## Self-Review Notes

- **Spec coverage:** detect-as-OBB (fixed angle) → Tasks 2, 5, 8, 9. Segment-as-OBB (GPU-native rotated-rect search, no `cv2`) → Tasks 3, 4, 5, 6, 7, 8, 9. "Direct mode only, no new mode" → enforced via Global Constraints and Task 1's field placement inside `OBBDirectConfig`. "Acceleration enabled and functional to start with" → Task 7 delivers the TensorRT segment executor before any GUI/worker wiring lands, and Task 6 makes it reachable; `detect`'s TensorRT path already exists today (reused, not rebuilt) and CoreML/native-CUDA/CPU/MPS need no new code (Global Constraints), so all runtimes are covered by the time Task 9 exposes the toggle to users. "Keep everything GPU native, avoid the CPU sync" → Task 3's kernel and Task 7's decode path never call `cv2`; under the native `cuda` runtime, Tasks 2 and 4 add `_extract_raw_tensors_from_boxes`/`_extract_raw_tensors_from_masks`, which plug `detect`/`segment` into the *same* zero-CPU-sync `_RawOBBTensors` fast path `"obb"` already has (Task 5) — no per-frame sync at all, deferred to the existing shared, batched `materialize_tensors()` call site. Under TensorRT/CPU/MPS, the one small `(N, 5)`-per-frame sync that remains is identical to what `"obb"` already does there — not a new cost introduced by this work.
- **Deferred/explicitly out of scope:** ONNX segment executor (unreachable — OBB never requests `onnx_*`), porting `_normalize_obb_geometry`/`_corners_from_xywhr` to torch (not needed — `materialize_tensors()` already generically handles any `(xywhr, conf)` device-tensor pair, which is why Tasks 2/4's fast-path variants can stay this small), legacy `core/detectors/_obb_geometry.py`/`yolo_detector.py` pipeline (superseded by `core/inference`, not touched), deep GUI panel refactor (tracked separately under the Slice 4 monolith split in `CLAUDE.md`).
- **Risk concentration:** Task 7 is the only task touching an unverified ultralytics-internal contract (the exact channel layout `preds[:, 6:]` for mask coefficients after `non_max_suppression`, and whether TensorRT's two named outputs can reliably be told apart — mitigated here by disambiguating on tensor rank (3 vs 4) rather than assumed name/order). Before running Task 7 against a real GPU/TensorRT box, additionally: (a) export a real YOLO-segment `.pt` to `.engine` and dump `engine.get_tensor_shape(name)` for both outputs to confirm the rank-based disambiguation holds, and (b) run one real inference through `_decode_segment_predictions` → `rotated_rect_from_masks` on that engine's actual output and visually overlay the decoded rectangle on the input image before trusting it in production. A secondary, lower risk: `_RawOBBTensors.corners` being a placeholder (zeros) for the `detect`/`segment` fast-path variants relies on `materialize_tensors()` continuing to ignore that field — confirmed by reading its current implementation (Task 5), but worth re-checking if `materialize_tensors()` is ever changed to consume `raw.corners` directly.
- **Type/name consistency check:** `_extract_obb_from_masks`/`_extract_raw_tensors_from_masks` (Task 4) and `_decode_segment_predictions` (Task 7) agree on the duck-typed `result` contract (`.orig_shape`, `.boxes.xyxy`, `.boxes.conf`, `.masks.data`) exactly — verified by re-reading both signatures side by side after drafting. `rotated_rect_from_masks`'s output column order `(cx, cy, w, h, angle_rad)` is consumed identically in both Task 4 functions' `rect_mask_space.unbind(-1)` unpack. `_letterbox_gain_pad`'s `(gain, pad_x, pad_y)` return order matches all of its call sites. `_extract_raw_tensors_from_boxes`/`_extract_raw_tensors_from_masks` (Tasks 2, 4) both return `_RawOBBTensors` with the exact same field names/order (`frame_idx, xywhr, corners, conf`) as the pre-existing `_extract_raw_tensors`, and Task 5's dispatch calls all three the same way.
