# Sequential-Segment Enablement Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the already-built `seq_crop_segment` role runnable end-to-end by clearing the three inference blockers A left, plus two A follow-ups — without perturbing existing direct or `seq_crop_obb` tracking.

**Architecture:** Additive changes to the inference sequential path (`obb.py`/`config.py`): `_extract_obb_from_masks` gains `offset`/`scale`; `OBBSequentialConfig` gains `stage2_task` + seg tuning fields; `_run_sequential` dispatches stage-2 extraction by task; a generic checkpoint-task assert covers stage-2. Then the DetectKit training dialog unhides `seq_crop_segment` with per-role pickers, and two follow-ups (X-AnyLabeling mode vocabulary, dataset-fit preview) land.

**Tech Stack:** Python 3.10+, NumPy, PyTorch/ultralytics, PySide6, pytest. Conda env `hydra-mps`; CUDA validation on `mehek` (`hydra-cuda`).

## Global Constraints

- **HOT-PATH BYTE-IDENTICAL:** existing direct modes and `seq_crop_obb` must be unperturbed. New behavior activates ONLY when `stage2_task == "segment"`. Every new parameter defaults to the identity: `offset=(0.0,0.0)`, `scale=(1.0,1.0)`, `stage2_task="obb"`. (Spec §3, §4, §10)
- **Stage-2 tasks: `obb` + `segment` ONLY.** No `detect` stage-2; do NOT add `offset`/`scale` to `_extract_obb_from_boxes`. (Spec §2, §11)
- **Mask `offset`/`scale` semantics mirror `extract_obb_result` exactly:** scale-then-offset, `cx=cx*sx+ox; w*=sx; cy=cy*sy+oy; h*=sy`. (Spec §3)
- **Inline dispatch only** in `_run_sequential`; NO shared extraction helper, NO refactor of `_run_direct` (phase C). (Spec §11)
- **The real config class is `OBBSequentialConfig`** (the spec says "SequentialConfig" as shorthand). Direct-segment defaults live on `OBBDirectConfig` (`seg_num_angles=24`, `seg_crop_size=64`, `seg_pad_ratio=0.15`, `seg_mask_threshold=0.5`).
- **Fail loudly:** a stage-2 checkpoint whose task disagrees with `stage2_task` raises at load. (Spec §5, §9)
- **Commit style:** commit as the configured git user; NO `Co-Authored-By: Claude` trailer.
- **Test invocation:** `conda run -n hydra-mps env PYTHONPATH="$PWD/src" python -m pytest <file> -v`. Run per-file, not the whole suite (base suite has pre-existing collection failures + a native-extension segfault in `test_detectkit_main_window.py`).
- **Locate every edit site by SYMBOL name**, not line number (main moves).

---

## File Structure

**New files**
- `tests/test_sequential_segment.py` — unit tests for the mask extractor transform, the sequential dispatch, the config field, and the assert.

**Modified files**
- `src/hydra_suite/core/inference/stages/obb.py` — `_extract_obb_from_masks` offset/scale; `_run_sequential` task dispatch; assert rename + stage-2 call.
- `src/hydra_suite/core/inference/config.py` — `OBBSequentialConfig.stage2_task` + seg fields; `from_params` threading.
- `src/hydra_suite/detectkit/gui/models.py` — `role_seq_crop_segment` + `imgsz_seq_crop_segment` + `model_seq_crop_segment`.
- `src/hydra_suite/detectkit/gui/dialogs/training_dialog.py` — unhide checkbox + per-role pickers + dispatch branches + dataset-fit preview coverage.
- `src/hydra_suite/detectkit/gui/panels/dataset_panel.py` — `xal_mode_for_level` mapping (only if the verified CLI vocabulary differs).

---

## Task 1: `offset`/`scale` on `_extract_obb_from_masks`

**Files:**
- Modify: `src/hydra_suite/core/inference/stages/obb.py` — `_extract_obb_from_masks`
- Test: `tests/test_sequential_segment.py`

**Interfaces:**
- Consumes: nothing new.
- Produces: `_extract_obb_from_masks(result, frame_idx, raw_detection_cap=0, *, num_angles=24, crop_size=64, pad_ratio=0.15, mask_threshold=0.5, offset: tuple[float,float]=(0.0,0.0), scale: tuple[float,float]=(1.0,1.0), emit_native_geometry=False) -> OBBResult`. Applies scale-then-offset to `cx,cy,w,h` (and native contours) so crop-space geometry lands in frame space. Defaults `(0,0)`/`(1,1)` ⇒ byte-identical.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_sequential_segment.py
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from hydra_suite.core.inference.stages import obb as obb_stage


class _FakeMasks:
    def __init__(self, data, xy):
        self.data = data
        self.xy = xy


class _FakeBoxes:
    def __init__(self, xyxy, conf):
        self.xyxy = torch.tensor(xyxy, dtype=torch.float32)
        self.conf = torch.tensor(conf, dtype=torch.float32)


class _FakeSegResult:
    """A minimal segment result: one square mask centered in an 80x80 crop."""

    def __init__(self):
        m = np.zeros((1, 80, 80), dtype=np.float32)
        m[0, 30:50, 30:50] = 1.0  # 20x20 block centered at (40,40)
        self.masks = _FakeMasks(torch.tensor(m), [np.array([[30, 30], [50, 30], [50, 50], [30, 50]], np.float32)])
        self.boxes = _FakeBoxes([[30, 30, 50, 50]], [0.9])
        self.orig_shape = (80, 80)


def _centroid(res, **kw):
    out = obb_stage._extract_obb_from_masks(res, frame_idx=0, **kw)
    assert out.num_detections == 1
    return out.centroids[0]


def test_mask_extract_identity_default():
    cx, cy = _centroid(_FakeSegResult())
    assert cx == pytest.approx(40.0, abs=1.5) and cy == pytest.approx(40.0, abs=1.5)


def test_mask_extract_offset_scale_maps_to_frame():
    # scale x2 then offset (+100,+200): crop centroid (40,40) -> (40*2+100, 40*2+200)
    cx, cy = _centroid(_FakeSegResult(), offset=(100.0, 200.0), scale=(2.0, 2.0))
    assert cx == pytest.approx(180.0, abs=3.0) and cy == pytest.approx(280.0, abs=3.0)


def test_mask_extract_offset_scale_contours_in_frame_space():
    out = obb_stage._extract_obb_from_masks(
        _FakeSegResult(), frame_idx=0, offset=(100.0, 200.0), scale=(2.0, 2.0),
        emit_native_geometry=True,
    )
    poly = out.polygons[0]
    # contour x in [30,50]*2+100 = [160,200]; y in [30,50]*2+200 = [260,300]
    assert poly[:, 0].min() == pytest.approx(160.0, abs=2.0)
    assert poly[:, 1].max() == pytest.approx(300.0, abs=2.0)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `conda run -n hydra-mps env PYTHONPATH="$PWD/src" python -m pytest tests/test_sequential_segment.py -v`
Expected: FAIL — `_extract_obb_from_masks() got an unexpected keyword argument 'offset'`.

- [ ] **Step 3: Write minimal implementation**

In `_extract_obb_from_masks`, add the two params to the keyword-only block (after `mask_threshold`, before `emit_native_geometry`):

```python
    mask_threshold: float = 0.5,
    offset: tuple[float, float] = (0.0, 0.0),
    scale: tuple[float, float] = (1.0, 1.0),
    emit_native_geometry: bool = False,
```

The extractor already computes `cx, cy, w_arr, h_arr` (the `((cx_m - pad_x) / gain).cpu().numpy()` block). Immediately AFTER those four assignments and BEFORE `_normalize_obb_geometry`, apply scale-then-offset (mirrors `extract_obb_result`):

```python
    # Remap crop-space geometry into frame space (sequential stage-2). Same
    # scale-then-offset semantics as extract_obb_result; identity by default so
    # the direct segment path stays byte-identical.
    ox, oy = offset
    sx, sy = scale
    cx = cx * sx + ox
    cy = cy * sy + oy
    w_arr = w_arr * sx
    h_arr = h_arr * sy
```

In the `emit_native_geometry` block that builds `out.polygons`, transform each native contour `p` (from `masks.xy`, in crop space) by the same map, leaving the `corners[i]` fallback untouched (it is already frame-space, built from the transformed cx/cy):

```python
    if emit_native_geometry:
        assert polygons_native is not None and len(polygons_native) == n
        out.polygons = []
        for i, p in enumerate(polygons_native):
            if p.shape[0] > 0:
                q = p.astype(np.float32).copy()
                q[:, 0] = q[:, 0] * sx + ox
                q[:, 1] = q[:, 1] * sy + oy
                out.polygons.append(q)
            else:
                out.polygons.append(corners[i].astype(np.float32).copy())
```

- [ ] **Step 4: Run test to verify it passes**

Run: `conda run -n hydra-mps env PYTHONPATH="$PWD/src" python -m pytest tests/test_sequential_segment.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/hydra_suite/core/inference/stages/obb.py tests/test_sequential_segment.py
git commit -m "feat(inference): offset/scale remap on _extract_obb_from_masks for sequential stage-2"
```

---

## Task 2: `OBBSequentialConfig.stage2_task` + seg fields + `from_params` threading

**Files:**
- Modify: `src/hydra_suite/core/inference/config.py` — `OBBSequentialConfig` + `from_params` sequential branch
- Test: `tests/test_sequential_segment.py`

**Interfaces:**
- Consumes: nothing new.
- Produces: `OBBSequentialConfig` gains `stage2_task: Literal["obb","segment"] = "obb"`, `seg_num_angles: int = 24`, `seg_crop_size: int = 64`, `seg_pad_ratio: float = 0.15`, `seg_mask_threshold: float = 0.5`. `from_params` maps `YOLO_SEQ_STAGE2_TASK` (default `"obb"`, coerced to `{"obb","segment"}`) and `YOLO_SEQ_SEG_*` params into them.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_sequential_segment.py  (append)
from hydra_suite.core.inference.config import OBBSequentialConfig, OBBConfig


def test_sequential_config_stage2_task_defaults_obb():
    c = OBBSequentialConfig(detect_model_path="d.pt", obb_model_path="s.pt")
    assert c.stage2_task == "obb"
    assert (c.seg_num_angles, c.seg_crop_size, c.seg_pad_ratio, c.seg_mask_threshold) == (24, 64, 0.15, 0.5)


def test_sequential_config_from_params_threads_stage2_task():
    params = {
        "DETECTION_METHOD": "yolo_obb", "YOLO_OBB_MODE": "sequential",
        "YOLO_DETECT_MODEL_PATH": "d.pt", "YOLO_CROP_OBB_MODEL_PATH": "s.pt",
        "YOLO_SEQ_STAGE2_TASK": "segment", "RUNTIME_TIER": "cpu",
    }
    cfg = OBBConfig.from_params(params)
    assert cfg.sequential.stage2_task == "segment"


def test_sequential_config_from_params_coerces_bad_stage2_task():
    params = {
        "DETECTION_METHOD": "yolo_obb", "YOLO_OBB_MODE": "sequential",
        "YOLO_DETECT_MODEL_PATH": "d.pt", "YOLO_CROP_OBB_MODEL_PATH": "s.pt",
        "YOLO_SEQ_STAGE2_TASK": "banana", "RUNTIME_TIER": "cpu",
    }
    assert OBBConfig.from_params(params).sequential.stage2_task == "obb"
```

> **Note:** if `OBBConfig.from_params` is not the exact constructor name/signature, locate the params→config entry point (grep `def from_params` / `YOLO_OBB_MODE` in `config.py`) and adapt the test's call accordingly; keep the assertions.

- [ ] **Step 2: Run test to verify it fails**

Run: `conda run -n hydra-mps env PYTHONPATH="$PWD/src" python -m pytest tests/test_sequential_segment.py -v -k stage2 or config`
Expected: FAIL — `unexpected keyword argument 'stage2_task'` / `AttributeError: stage2_task`.

- [ ] **Step 3: Write minimal implementation**

In `OBBSequentialConfig`, add after `stage2_batch_size`:

```python
    stage2_task: Literal["obb", "segment"] = "obb"
    # Read only when stage2_task == "segment"; forwarded to _extract_obb_from_masks.
    # Defaults match OBBDirectConfig's segment defaults.
    seg_num_angles: int = 24
    seg_crop_size: int = 64
    seg_pad_ratio: float = 0.15
    seg_mask_threshold: float = 0.5
```

In `from_params`' `if obb_mode == "sequential":` branch, inside the `OBBSequentialConfig(...)` construction, add:

```python
                stage2_task=(
                    "segment"
                    if str(params.get("YOLO_SEQ_STAGE2_TASK", "obb")).strip().lower()
                    == "segment"
                    else "obb"
                ),
                seg_num_angles=int(params.get("YOLO_SEQ_SEG_NUM_ANGLES", 24)),
                seg_crop_size=int(params.get("YOLO_SEQ_SEG_CROP_SIZE", 64)),
                seg_pad_ratio=float(params.get("YOLO_SEQ_SEG_PAD_RATIO", 0.15)),
                seg_mask_threshold=float(params.get("YOLO_SEQ_SEG_MASK_THRESHOLD", 0.5)),
```

- [ ] **Step 4: Run test to verify it passes**

Run: `conda run -n hydra-mps env PYTHONPATH="$PWD/src" python -m pytest tests/test_sequential_segment.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/hydra_suite/core/inference/config.py tests/test_sequential_segment.py
git commit -m "feat(inference): OBBSequentialConfig stage2_task + seg tuning fields"
```

---

## Task 3: task-aware stage-2 dispatch in `_run_sequential`

**Files:**
- Modify: `src/hydra_suite/core/inference/stages/obb.py` — `_run_sequential`
- Test: `tests/test_sequential_segment.py`

**Interfaces:**
- Consumes: `_extract_obb_from_masks` offset/scale (Task 1); `OBBSequentialConfig.stage2_task` + seg fields (Task 2).
- Produces: `_run_sequential` routes stage-2 extraction by `seq.stage2_task`: `"obb"` → `extract_obb_result` (unchanged); `"segment"` → `_extract_obb_from_masks(...)` with the same per-crop `offset`/`scale` and the seq seg params.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_sequential_segment.py  (append)
from hydra_suite.core.inference.config import OBBSequentialConfig


class _FakeStage2Model:
    """Returns one _FakeSegResult per crop, ignoring inputs."""

    def __init__(self):
        self.calls = 0

    def predict(self, batch, **kw):
        self.calls += 1
        return [_FakeSegResult() for _ in batch]


def test_run_sequential_segment_dispatch(monkeypatch):
    # Route stage-2 through the mask extractor and confirm frame-space centroids.
    from hydra_suite.core.inference.stages import obb as m

    seq = OBBSequentialConfig(detect_model_path="d.pt", obb_model_path="s.pt", stage2_task="segment")
    called = {"masks": 0}
    real = m._extract_obb_from_masks

    def spy(*a, **k):
        called["masks"] += 1
        return real(*a, **k)

    monkeypatch.setattr(m, "_extract_obb_from_masks", spy)
    # Build one 80x80 crop with a stage-1 box; stub build_crops to return it.
    frame = np.zeros((200, 200, 3), np.uint8)
    monkeypatch.setattr(m, "build_crops", lambda *a, **k: ([np.zeros((80, 80, 3), np.uint8)], [(20.0, 30.0)]))
    monkeypatch.setattr(m, "resize_crops_for_stage2", lambda crops, size: crops)

    class _S1Boxes:
        def __len__(self): return 1
        conf = None
    class _S1: boxes = _S1Boxes()
    class _DetModel:
        def predict(self, *a, **k): return [_S1()]

    from hydra_suite.core.inference.config import OBBConfig
    cfg = OBBConfig(mode="sequential", sequential=seq)

    class _Models:
        detect_model = _DetModel()
        obb_model = _FakeStage2Model()
    class _Rt:
        device = "cpu"; tensor_on_cuda = False

    out = m._run_sequential([frame], _Models(), cfg, _Rt())
    assert called["masks"] >= 1
    assert out[0].num_detections >= 1
```

> **Note:** the fake stage-1/models shapes above are illustrative — adapt the fakes to the ACTUAL `build_crops`/`OBBModels`/`RuntimeContext` interfaces you find (grep their defs). The load-bearing assertion is: with `stage2_task="segment"`, `_extract_obb_from_masks` is what runs (not `extract_obb_result`).

- [ ] **Step 2: Run test to verify it fails**

Run: `conda run -n hydra-mps env PYTHONPATH="$PWD/src" python -m pytest tests/test_sequential_segment.py -v -k run_sequential`
Expected: FAIL — the mask spy is never called (stage-2 still hardcodes `extract_obb_result`).

- [ ] **Step 3: Write minimal implementation**

In `_run_sequential`, replace the single hardcoded stage-2 extraction call:

```python
                sub.append(
                    extract_obb_result(r, frame_idx, offset=offsets[i + j], scale=scale)
                )
```

with a dispatch on `seq.stage2_task`:

```python
                if seq.stage2_task == "segment":
                    sub.append(
                        _extract_obb_from_masks(
                            r,
                            frame_idx,
                            config.raw_detection_cap,
                            num_angles=seq.seg_num_angles,
                            crop_size=seq.seg_crop_size,
                            pad_ratio=seq.seg_pad_ratio,
                            mask_threshold=seq.seg_mask_threshold,
                            offset=offsets[i + j],
                            scale=scale,
                        )
                    )
                else:
                    sub.append(
                        extract_obb_result(
                            r, frame_idx, offset=offsets[i + j], scale=scale
                        )
                    )
```

Leave everything else (`merge_obb_results`, `_apply_raw_detection_cap`) unchanged.

- [ ] **Step 4: Run test to verify it passes**

Run: `conda run -n hydra-mps env PYTHONPATH="$PWD/src" python -m pytest tests/test_sequential_segment.py -v`
Expected: PASS (all Task 1-3 tests).

- [ ] **Step 5: Commit**

```bash
git add src/hydra_suite/core/inference/stages/obb.py tests/test_sequential_segment.py
git commit -m "feat(inference): task-aware stage-2 dispatch (obb|segment) in _run_sequential"
```

---

## Task 4: generic checkpoint-task assert on the stage-2 model

**Files:**
- Modify: `src/hydra_suite/core/inference/stages/obb.py` — rename `_assert_direct_task_matches_checkpoint` → `_assert_task_matches_checkpoint`; call it in `load_obb_models`' sequential branch.
- Test: `tests/test_sequential_segment.py`

**Interfaces:**
- Consumes: `OBBSequentialConfig.stage2_task` (Task 2).
- Produces: `_assert_task_matches_checkpoint(model, model_task, model_path)` (renamed, identical body). Called for the direct model (existing call site) AND the sequential stage-2 model (`models.obb_model`, `config.sequential.stage2_task`, `config.sequential.obb_model_path`).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_sequential_segment.py  (append)
def test_assert_task_matches_checkpoint_raises_on_mismatch():
    from hydra_suite.core.inference.stages import obb as m

    class _Ckpt:
        task = "segment"
    with pytest.raises(ValueError, match="task"):
        m._assert_task_matches_checkpoint(_Ckpt(), "obb", "s.pt")


def test_assert_task_matches_checkpoint_ok_on_match():
    from hydra_suite.core.inference.stages import obb as m

    class _Ckpt:
        task = "segment"
    m._assert_task_matches_checkpoint(_Ckpt(), "segment", "s.pt")  # no raise
```

- [ ] **Step 2: Run test to verify it fails**

Run: `conda run -n hydra-mps env PYTHONPATH="$PWD/src" python -m pytest tests/test_sequential_segment.py -v -k assert_task`
Expected: FAIL — `module 'obb' has no attribute '_assert_task_matches_checkpoint'`.

- [ ] **Step 3: Write minimal implementation**

Rename the function `_assert_direct_task_matches_checkpoint` → `_assert_task_matches_checkpoint` (definition + the existing direct-branch call in `load_obb_models`). Update the docstring first line to drop "Direct-mode" wording (the logic is task-generic). In `load_obb_models`' sequential branch (locate `if config.mode == "sequential":` / where `detect_model` + `obb_model` are loaded), add after the stage-2 model is loaded:

```python
        _assert_task_matches_checkpoint(
            obb_model,  # the loaded stage-2 model (use the actual local var name)
            config.sequential.stage2_task,
            config.sequential.obb_model_path,
        )
```

(Use the actual local variable holding the loaded stage-2 model — grep the sequential branch.)

- [ ] **Step 4: Run test to verify it passes**

Run: `conda run -n hydra-mps env PYTHONPATH="$PWD/src" python -m pytest tests/test_sequential_segment.py -v`
Expected: PASS.

Also grep to confirm no stale references remain: `grep -rn "_assert_direct_task_matches_checkpoint" src tests` returns nothing.

- [ ] **Step 5: Commit**

```bash
git add src/hydra_suite/core/inference/stages/obb.py tests/test_sequential_segment.py
git commit -m "feat(inference): assert stage-2 checkpoint task; generalize the task-match assert"
```

---

## Task 5: `DetectKitProject` fields for `seq_crop_segment`

**Files:**
- Modify: `src/hydra_suite/detectkit/gui/models.py` — `DetectKitProject`
- Test: `tests/test_sequential_segment.py`

**Interfaces:**
- Produces: `role_seq_crop_segment: bool = False`, `imgsz_seq_crop_segment: int = 160`, `model_seq_crop_segment: str = "yolo26s-seg.pt"` on `DetectKitProject`; persist via the existing `fields()`-based `to_dict`/`from_dict`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_sequential_segment.py  (append)
def test_project_roundtrips_seq_crop_segment(tmp_path):
    from hydra_suite.detectkit.gui.models import DetectKitProject
    p = DetectKitProject()
    p.role_seq_crop_segment = True
    p.imgsz_seq_crop_segment = 192
    p.model_seq_crop_segment = "custom-seg.pt"
    dest = tmp_path / "p.json"
    p.save(dest)
    q = DetectKitProject.load(dest)
    assert q.role_seq_crop_segment is True
    assert q.imgsz_seq_crop_segment == 192
    assert q.model_seq_crop_segment == "custom-seg.pt"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `conda run -n hydra-mps env PYTHONPATH="$PWD/src" python -m pytest tests/test_sequential_segment.py -v -k seq_crop_segment`
Expected: FAIL — `AttributeError: role_seq_crop_segment`.

- [ ] **Step 3: Write minimal implementation**

In `DetectKitProject`, next to the existing role block, add `role_seq_crop_segment: bool = False`; next to the imgsz block add `imgsz_seq_crop_segment: int = 160`; next to the model block add `model_seq_crop_segment: str = "yolo26s-seg.pt"`. No `to_dict`/`from_dict` changes (they iterate `fields()`).

- [ ] **Step 4: Run test to verify it passes**

Run: `conda run -n hydra-mps env PYTHONPATH="$PWD/src" python -m pytest tests/test_sequential_segment.py -v -k seq_crop_segment`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/hydra_suite/detectkit/gui/models.py tests/test_sequential_segment.py
git commit -m "feat(detectkit): project fields for the seq_crop_segment role"
```

---

## Task 6: unhide `seq_crop_segment` in the training dialog (checkbox + pickers)

**Files:**
- Modify: `src/hydra_suite/detectkit/gui/dialogs/training_dialog.py`
- Test: import-sanity + models round-trip (Task 5 already covers persistence).

**Interfaces:**
- Consumes: `DetectKitProject.role_seq_crop_segment` + imgsz/model fields (Task 5); `TrainingRole.SEQ_CROP_SEGMENT` (exists from A).
- Produces: a `chk_role_seq_crop_segment` checkbox and per-role imgsz spinbox + model combo, mirroring the existing `seq_crop_obb` role and the Task 10b pickers at EVERY site.

- [ ] **Step 1: Mirror the checkbox at every `chk_role_seq_crop_obb` site**

Grep `grep -n "chk_role_seq_crop_obb\|role_seq_crop_obb\|SEQ_CROP_OBB" src/hydra_suite/detectkit/gui/dialogs/training_dialog.py`. At each, add the `seq_crop_segment` parallel: widget creation (`self.chk_role_seq_crop_segment = QCheckBox("seq_crop_segment")`), layout, `toggled`→`_on_role_selection_changed` connection, `_selected_role_keys`, `_selected_roles` (append `TrainingRole.SEQ_CROP_SEGMENT`), `_load_from_project`/`_write_to_project` (`role_seq_crop_segment`), JSON `_collect_training_state`/`_apply_training_state`, and `_refresh_role_gating`'s `role_checks` dict (so its POLYGON min-level gating auto-applies).

- [ ] **Step 2: Mirror the pickers at every `spin_imgsz_seq_crop_obb` / `combo_model_...` site (Task 10b pattern)**

Add `self.spin_imgsz_seq_crop_segment` (range 64-2048, value 160, label "imgsz (seq_crop_segment)") and `self.combo_model_seq_crop_segment` (editable, items = a seg-options list e.g. `["yolo26s-seg.pt","yolo11s-seg.pt"]`, default `"yolo26s-seg.pt"`, label "seq_crop_segment"), wired into layout, `_load_from_project`/`_write_to_project`, JSON state, the visibility block (`show_seq_crop_segment = "seq_crop_segment" in selected_roles`), and:
- `_imgsz_for_role`: `if role == TrainingRole.SEQ_CROP_SEGMENT: return self.spin_imgsz_seq_crop_segment.value()`
- `_base_model_for_role`: `if role == TrainingRole.SEQ_CROP_SEGMENT: return self.combo_model_seq_crop_segment.currentText().strip()`

- [ ] **Step 3: Verify**

```bash
cd /Users/neurorishika/Projects/Rockefeller/Kronauer/multi-animal-tracker
conda run -n hydra-mps env PYTHONPATH="$PWD/src" python -c "import hydra_suite.detectkit.gui.dialogs.training_dialog; print('import ok')"
conda run -n hydra-mps env PYTHONPATH="$PWD/src" python -m pytest tests/test_sequential_segment.py -q
```
Expected: import ok; tests pass.

- [ ] **Step 4: Commit**

```bash
git add src/hydra_suite/detectkit/gui/dialogs/training_dialog.py
git commit -m "feat(detectkit): unhide seq_crop_segment role with per-role pickers"
```

---

## Task 7: follow-up A1 — verify X-AnyLabeling `--mode` vocabulary

**Files:**
- Possibly modify: `src/hydra_suite/detectkit/gui/panels/dataset_panel.py` — `xal_mode_for_level` (only if the CLI vocabulary differs).

- [ ] **Step 1: Read the real accepted `--mode` values**

```bash
conda run -n x-anylabeling-cpu xanylabeling convert --help 2>&1 | grep -iA3 -- "--mode" || \
conda run -n x-anylabeling-cpu python -c "import x_anylabeling" 2>&1 | head
```
Record the accepted `--mode` values (e.g. does it accept `rectangle`/`polygon`, or `rect`/`poly`, or only `obb`?).

- [ ] **Step 2: Reconcile `xal_mode_for_level`**

If the verified vocabulary matches the current mapping (`aabb→"rectangle"`, `obb→"obb"`, `polygon→"polygon"`), change nothing but update the `xal_mode_for_level` docstring to record "verified against x-anylabeling <version>: modes = …". If it differs, correct the returned strings in `xal_mode_for_level` ONLY, and update the docstring.

- [ ] **Step 3: Verify + commit (only if changed)**

```bash
conda run -n hydra-mps env PYTHONPATH="$PWD/src" python -m pytest tests/test_geometry_levels.py -q -k xal_mode
git add src/hydra_suite/detectkit/gui/panels/dataset_panel.py
git commit -m "fix(detectkit): reconcile xal_mode_for_level with verified x-anylabeling --mode vocabulary"
```
If nothing changed, make NO commit; record the verified vocabulary in the task report instead.

---

## Task 8: follow-up A2 — dataset-fit preview covers the new roles

**Files:**
- Modify: `src/hydra_suite/detectkit/gui/dialogs/training_dialog.py` — `_dataset_fit_key` + `_refresh_dataset_fit` (or `dataset_fit_view`).

**Interfaces:**
- Consumes: the per-role imgsz spinboxes for `detect_direct`, `segment_direct`, `seq_crop_segment`.
- Produces: the dataset-fit cache key includes those three imgsz values (so changing them invalidates the preview), and the preview text reports their size analysis alongside the existing roles.

- [ ] **Step 1: Locate the preview**

`grep -n "_dataset_fit_key\|_refresh_dataset_fit\|dataset_fit" src/hydra_suite/detectkit/gui/dialogs/training_dialog.py`. Read `_dataset_fit_key` and the preview builder to see how the existing roles' imgsz values feed both.

- [ ] **Step 2: Add the three roles**

Include `imgsz_detect_direct`, `imgsz_segment_direct`, `imgsz_seq_crop_segment` (via their spinboxes) in `_dataset_fit_key`, and add their rows to the preview text using the same per-role formatting the existing roles use. Guard on whether each role is selected (mirror how existing roles are conditionally included).

- [ ] **Step 3: Verify + commit**

```bash
conda run -n hydra-mps env PYTHONPATH="$PWD/src" python -c "import hydra_suite.detectkit.gui.dialogs.training_dialog; print('ok')"
git add src/hydra_suite/detectkit/gui/dialogs/training_dialog.py
git commit -m "feat(detectkit): dataset-fit preview covers detect/segment/crop-segment roles"
```

---

## Task 9: format, lint, and cross-platform parity gate

**Files:** none (verification only)

- [ ] **Step 1: Format + lint the touched files**

```bash
cd /Users/neurorishika/Projects/Rockefeller/Kronauer/multi-animal-tracker
conda run -n hydra-mps black --check src/hydra_suite/core/inference/stages/obb.py src/hydra_suite/core/inference/config.py src/hydra_suite/detectkit/gui/models.py src/hydra_suite/detectkit/gui/dialogs/training_dialog.py src/hydra_suite/detectkit/gui/panels/dataset_panel.py tests/test_sequential_segment.py
conda run -n hydra-mps flake8 src/hydra_suite/core/inference/stages/obb.py src/hydra_suite/core/inference/config.py src/hydra_suite/detectkit/gui/models.py src/hydra_suite/detectkit/gui/dialogs/training_dialog.py
```
Fix any new findings in the touched files only; commit if changed (`style(inference): formatter pass on sequential-segment enablement`).

- [ ] **Step 2: Full new-suite run**

```bash
conda run -n hydra-mps env PYTHONPATH="$PWD/src" python -m pytest tests/test_sequential_segment.py -v
```
Expected: all pass.

- [ ] **Step 3: MPS parity gate (byte-identical seq_crop_obb + direct)**

Baseline = pre-feature main; current = the feature src. From the main repo with the feature worktree present:
```bash
source "$(conda info --base)/etc/profile.d/conda.sh" && conda activate hydra-mps
REPO=$PWD WT=$PWD MAIN_SRC=$PWD/src WT_SRC=<feature-worktree>/src OUT=/tmp/equiv_seqseg RUNTIME=mps \
  bash tools/equivalence/run_matrix.sh ant_obb_sequential fly_obb
```
Expected: every clip EQUIVALENT (0 unmatched, 0.0 pos/θ delta) — the additive `stage2_task="obb"` default + `(0,0)`/`(1,1)` extractor defaults must not perturb `seq_crop_obb` or direct tracking. Verify CSV row counts > 1 (conda active).

- [ ] **Step 4: CUDA parity gate on mehek**

Push the branch, then on `mehek` (baseline + feature worktrees, fixtures fetched with conda active):
```bash
REPO=$PWD WT=$PWD MAIN_SRC=<baseline>/src WT_SRC=<feature>/src OUT=/tmp/equiv_seqseg_cuda RUNTIME=cuda \
  bash tools/equivalence/run_matrix.sh ant_obb_sequential fly_obb
```
Expected: every clip EQUIVALENT on CUDA too.

- [ ] **Step 5: Commit any formatting; report parity results**

---

## Self-Review Notes (spec coverage)

- §3 mask offset/scale → Task 1. §4 config `stage2_task`+seg fields+builder → Task 2; dispatch → Task 3. §5 assert → Task 4. §6 unhide GUI → Tasks 5-6. §7 A1 → Task 7. §8 A2 → Task 8. §10 testing + parity → Tasks 1-4 unit + Task 9 parity. §11 relationship/inline-only → honored (no `_run_direct` refactor, no shared helper, no `_extract_obb_from_boxes` change).
- **Byte-identical guard** appears in every core task's defaults and is gated end-to-end by Task 9's MPS+CUDA equivalence on `ant_obb_sequential` + `fly_obb`.
- **Class-name correction:** plan uses the real `OBBSequentialConfig`/`OBBDirectConfig`, not the spec's "SequentialConfig" shorthand.
