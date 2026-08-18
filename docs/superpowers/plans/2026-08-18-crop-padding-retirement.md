# Crop-Padding Retirement Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `canonical_margin` the single crop-framing dial for every model- and dataset-facing crop, replace `individual_crop_padding` with a stage-local `apriltag_crop_padding` (default `0.0` = bare AABB of the inference OBB), and rename the detection-batching GUI control.

**Architecture:** The canonical path (`CanonicalGeometry`) already frames every crop except AprilTag's, so this is a removal, not a rewrite. Work proceeds outside-in: introduce the new AprilTag key first (Task 2) so no stage is ever left without a padding source, unify the two divergent AABB helpers (Task 3), then strip `individual_crop_padding` from core (Task 4) and from config/GUI (Task 5), then gate (Tasks 6-8).

**Tech Stack:** Python 3.11, PyQt5, NumPy, OpenCV, PyTorch, pytest. Conda env `hydra-mps` on this box, `hydra-cuda` on mehek.

**Spec:** `docs/superpowers/specs/2026-08-18-crop-padding-retirement-design.md`

## Global Constraints

- **Isolation:** all work happens in a git worktree branched from local HEAD: `git worktree add .worktrees/crop-padding -b refactor/crop-padding-retirement HEAD`. Never branch from `origin/main` — local `main` is ahead.
- **Worktree tests need `PYTHONPATH`:** run pytest as `PYTHONPATH=$PWD/src python -m pytest ...` from inside the worktree, or the installed editable `src` from the main checkout is imported instead.
- **Commit identity:** commit as the configured git user. Do **not** add a `Co-Authored-By: Claude` trailer.
- **AprilTag default is `0.0`** — bare AABB extent of the inference OBB. Range `-0.5 … 2.0`, step `0.05`. Applies to `default.json` and `ooceraea_biroi.json` alike; no per-lab pin.
- **Canonical margin default stays `1.3`**; `reference_aspect_ratio` default stays `2.0`. Neither is touched.
- **`roi_crop_padding_fraction` is out of scope** — a different knob (ROI bounding box). Never edit it.
- **Byte-identity contract:** tracking CSVs (`_forward.csv`, `_tracking_final.csv`) must be byte-identical to the pre-change baseline for every non-AprilTag clip. AprilTag tag columns are expected to change and are reported, not asserted equal.
- **Before any equivalence run:** kill stale `sleap`/`hydra` processes; never touch other users' processes. `conda activate hydra-mps` (or `hydra-cuda`) must be active or pose/SLEAP clips silently produce empty CSVs that falsely compare EQUIVALENT — always verify `wc -l` > 1 on the CSVs.
- **Formatting gate before each commit:** `make format` then `make lint`.

---

## File Structure

**Modified — core (Qt-free):**
- `src/hydra_suite/core/inference/stages/crops.py` — `extract_aabb_crops`, the one AABB helper after Task 3.
- `src/hydra_suite/core/tracking/pose/pose_pipeline.py` — `_expand_obb_to_aabb` delegates to the crops.py geometry.
- `src/hydra_suite/core/inference/config.py` — `PoseConfig.crop_padding` removed; `AprilTagConfig.crop_padding` re-sourced from `APRILTAG_CROP_PADDING`.
- `src/hydra_suite/core/inference/cache/keys.py` — pose key drops the `crop_padding` term.
- `src/hydra_suite/core/individual/classification/apriltag.py` — `AprilTagConfig.padding_fraction` re-sourced, default `0.0`.
- `src/hydra_suite/core/individual/properties/cache.py` — AprilTag cache payload key renamed.
- `src/hydra_suite/core/canonicalization/crop.py` — `padding_fraction` parameter removed from `extract_and_classify_batch`.
- `src/hydra_suite/core/individual/dataset/generator.py` — padding fields + legacy AABB branch removed; `reference_aspect_ratio <= 0` raises.
- `src/hydra_suite/core/individual/dataset/oriented_video.py` — `padding_fraction` ctor arg removed; mask expansion from `geometry.margin`.
- `src/hydra_suite/core/post/media_export.py`, `src/hydra_suite/core/tracking/session.py` — stop threading `padding_fraction`.
- `src/hydra_suite/core/post/interpolated_crops.py` — reads `APRILTAG_CROP_PADDING`.

**Modified — app layer:**
- `src/hydra_suite/trackerkit/gui/panels/detection_panel.py` — batching rename; AprilTag key in the context dict; `individual_crop_padding` removed.
- `src/hydra_suite/trackerkit/gui/panels/identity_panel.py` — padding row removed; AprilTag `Crop padding` field added.
- `src/hydra_suite/trackerkit/gui/orchestrators/config.py` — load/save keys; legacy warning.
- `src/hydra_suite/trackerkit/gui/workers/preview_worker.py` — param mapping.
- `src/hydra_suite/trackerkit/engine_params.py`, `src/hydra_suite/trackerkit/cli_config.py` — param table.
- `src/hydra_suite/resources/configs/default.json`, `ooceraea_biroi.json`.

**Modified — tests:** `tests/test_cache_ids.py`, `test_canonical_crop.py`, `test_canonical_dataset_provenance.py`, `test_crop_export_lossless.py`, `test_export_clipping_surfaced.py`, `test_inference_cache_keys.py`, `test_media_export.py`, `test_oriented_track_video_export.py`, `test_trackerkit_preview_worker.py`, `tests/core/individual/dataset/test_oriented_video_actual_rows.py`, and the two goldens in `tests/data/get_parameters_dict_golden/`.

**Created:** `tests/test_crop_padding_retirement.py` — the contract tests for this change.

---

### Task 1: Rename the detection-batching GUI control

**Files:**
- Modify: `src/hydra_suite/trackerkit/gui/panels/detection_panel.py:1082-1112`, `:1424-1487`
- Modify: `src/hydra_suite/trackerkit/gui/orchestrators/config.py:602`, `:1704` (stale comments only)

**Interfaces:**
- Consumes: nothing.
- Produces: nothing. GUI strings only; `detection_batch_size` / `InferenceConfig.detection_batch_size` / `YOLO_BATCH_SIZE` are untouched, so no config, cache key, or engine profile moves.

- [ ] **Step 1: Replace the group box and help text**

In `detection_panel.py`, replace the block starting at the `Live Detection Batching` comment:

```python
        # ============================================================
        # Detection Frame Batching (stage-1 detector input batch)
        # ============================================================
        self.g_live_batching = QGroupBox("Detection Frame Batching")
        self._main_window._set_compact_section_widget(self.g_live_batching)
        vl_live_batch = QVBoxLayout(self.g_live_batching)
        vl_live_batch.setSpacing(6)
        vl_live_batch.addWidget(
            self._main_window._create_help_label(
                "How many video frames the DETECTOR (stage 1: YOLO OBB / background "
                "subtraction) processes per GPU call during a tracking run. This is not "
                "the crop batching used by head-tail, identity, pose, and AprilTag -- "
                "those always run once over every crop in the chunk. Higher batches are "
                "faster on TensorRT/CUDA/MPS; some runtimes are locked to 1 (see below)."
            )
        )
```

- [ ] **Step 2: Replace the row label and tooltip**

```python
        self.spin_detection_batch_size.setToolTip(
            "Video frames per detector (stage-1) GPU call during a tracking run.\n"
            "Feeds InferenceConfig.detection_batch_size directly.\n"
            "Does NOT affect stage-2 crop batching (head-tail / identity / pose / AprilTag).\n"
            "Higher = faster on TensorRT/CUDA/MPS, more GPU memory used.\n"
            "Typical values: 4-16 depending on GPU."
        )
```

and the row label:

```python
        _live_batch_row.addWidget(QLabel("Frames per detector call"))
```

- [ ] **Step 3: Re-word the policy notices**

In `_sync_live_detection_batch_controls`, update the three user-visible strings so they name the new label:

```python
                message = "Realtime tracking fixes the detector to one frame per call. Sequential stage-2 crop batching still uses the Stage-2 crop batch setting."
```

```python
                message = "Realtime tracking processes detection one frame at a time; frames per detector call is fixed to 1."
```

```python
            message = (
                "On this platform, gpu_fast detection (OBB) runs on "
                "CoreML, which supports only one frame per call, "
                "regardless of this setting. CoreML classification "
                "(identity/head-tail/CNN) is unaffected and still "
                "batches normally."
            )
```

- [ ] **Step 4: Fix the two stale comments**

In `orchestrators/config.py`, change both `# Live Detection Batching (drives InferenceConfig.detection_batch_size)` comments (lines 602 and 1704) to `# Detection frame batching (drives InferenceConfig.detection_batch_size)`.

- [ ] **Step 5: Verify nothing else references the old strings**

Run: `grep -rn "Live Detection Batching\|Frame batch size" src/ tests/ docs/`
Expected: no hits.

- [ ] **Step 6: Run the GUI persistence tests that touch this widget**

Run: `PYTHONPATH=$PWD/src python -m pytest tests/test_main_window_config_persistence.py -v -k detection_batch`
Expected: PASS (they assert `isEnabled()`, not labels, so they must be unaffected).

- [ ] **Step 7: Commit**

```bash
make format && make lint
git add src/hydra_suite/trackerkit/gui/panels/detection_panel.py src/hydra_suite/trackerkit/gui/orchestrators/config.py
git commit -m "refactor(gui): rename Live Detection Batching to Detection Frame Batching"
```

---

### Task 2: Introduce `apriltag_crop_padding` (default 0.0)

The AprilTag stage stops reading `INDIVIDUAL_CROP_PADDING` and reads its own key. `individual_crop_padding` still exists after this task — it is removed in Tasks 4-5. Doing it in this order means no stage is ever left without a padding source.

**Files:**
- Create: `tests/test_crop_padding_retirement.py`
- Modify: `src/hydra_suite/core/individual/classification/apriltag.py:94-95`, `:117`
- Modify: `src/hydra_suite/core/inference/config.py:1082`
- Modify: `src/hydra_suite/core/individual/properties/cache.py:291`
- Modify: `src/hydra_suite/core/post/interpolated_crops.py:714`
- Modify: `src/hydra_suite/trackerkit/engine_params.py:1191` (next to `APRILTAG_DECIMATE`)
- Modify: `src/hydra_suite/trackerkit/cli_config.py`
- Modify: `src/hydra_suite/trackerkit/gui/panels/identity_panel.py:170-185`
- Modify: `src/hydra_suite/trackerkit/gui/panels/detection_panel.py:1940`
- Modify: `src/hydra_suite/trackerkit/gui/orchestrators/config.py:1215`, `:1919`
- Modify: `src/hydra_suite/trackerkit/gui/workers/preview_worker.py:572`
- Modify: `src/hydra_suite/resources/configs/default.json`, `ooceraea_biroi.json`
- Test: `tests/test_crop_padding_retirement.py`

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces:
  - Config key `apriltag_crop_padding: float` (default `0.0`).
  - Param key `APRILTAG_CROP_PADDING: float`.
  - `AprilTagConfig.padding_fraction` (in `core/individual/classification/apriltag.py`) default `0.0`, sourced from `APRILTAG_CROP_PADDING`.
  - `AprilTagConfig.crop_padding` (in `core/inference/config.py`) default `0.0`, sourced from `APRILTAG_CROP_PADDING`.
  - GUI widget `identity_panel.spin_apriltag_crop_padding` (`QDoubleSpinBox`).

- [ ] **Step 1: Write the failing tests**

Create `tests/test_crop_padding_retirement.py`:

```python
"""Contract tests for the crop-padding retirement (spec 2026-08-18)."""

import numpy as np
import pytest

from hydra_suite.core.inference.config import build_inference_config_from_params
from hydra_suite.core.individual.classification.apriltag import AprilTagConfig


def test_apriltag_config_reads_its_own_key_default_zero():
    cfg = AprilTagConfig.from_params({})
    assert cfg.padding_fraction == 0.0


def test_apriltag_config_ignores_individual_crop_padding():
    cfg = AprilTagConfig.from_params({"INDIVIDUAL_CROP_PADDING": 0.5})
    assert cfg.padding_fraction == 0.0


def test_apriltag_config_honours_apriltag_crop_padding():
    cfg = AprilTagConfig.from_params({"APRILTAG_CROP_PADDING": 0.25})
    assert cfg.padding_fraction == 0.25


def _params(cfg_overrides):
    """Build engine params from a minimal config, the Qt-free way.

    Mirrors tests/test_get_parameters_dict_characterization.py:263-281.
    """
    from hydra_suite.trackerkit import cli_config
    from hydra_suite.trackerkit.engine_params import (
        RuntimeContext,
        build_engine_params,
    )

    cfg = dict(cli_config.load_tracker_cli_config())
    cfg.update(cfg_overrides)
    rt = RuntimeContext(fps=100.0, total_frames=500, frame_width=640, frame_height=480)
    return build_engine_params(cfg, runtime=rt)


def test_engine_params_emit_apriltag_crop_padding():
    assert _params({"apriltag_crop_padding": 0.2})["APRILTAG_CROP_PADDING"] == 0.2
```

Note: `build_engine_params(config, *, runtime, advanced_config=None)` requires a
`RuntimeContext`. Read `tests/test_get_parameters_dict_characterization.py:263-281`
for the exact construction — it loads a real fixture config via
`cli_config.load_tracker_cli_config` and builds a synthetic `RuntimeContext`.
`build_engine_params` wants the **saved TrackerKit JSON config shape**, not the
advanced-config table, so if `load_tracker_cli_config()` needs a path argument,
use a fixture config from `tests/fixtures` the way that test does rather than
hand-rolling a dict.

- [ ] **Step 2: Run to verify failure**

Run: `PYTHONPATH=$PWD/src python -m pytest tests/test_crop_padding_retirement.py -v`
Expected: FAIL — `AprilTagConfig.from_params({})` returns `0.1`, and `APRILTAG_CROP_PADDING` is absent from params.

- [ ] **Step 3: Re-source the two AprilTag config dataclasses**

`core/individual/classification/apriltag.py`, replacing the field and its comment at 94-95:

```python
    # Crop padding as a fraction of the OBB bounding box. 0.0 = the detection's
    # exact axis-aligned extent -- an AprilTag is a rigid printed square, so the
    # crop is deliberately un-rotated, un-scaled, and un-padded by default.
    padding_fraction: float = 0.0
```

and in `from_params`:

```python
            padding_fraction=float(params.get("APRILTAG_CROP_PADDING", 0.0)),
```

`core/inference/config.py`: change `AprilTagConfig.crop_padding` (line 408) to `crop_padding: float = 0.0` and its `from_parameters` source (line 1082) to:

```python
        crop_padding=float(params.get("APRILTAG_CROP_PADDING", 0.0)),
```

- [ ] **Step 4: Re-source the AprilTag cache payload**

`core/individual/properties/cache.py:291`, replacing the `"padding_fraction"` entry:

```python
        "apriltag_crop_padding": float(params.get("APRILTAG_CROP_PADDING", 0.0)),
```

- [ ] **Step 5: Re-source the interpolated-crop AprilTag path**

`core/post/interpolated_crops.py:714`:

```python
    _crop_padding = float(params.get("APRILTAG_CROP_PADDING", 0.0))
```

- [ ] **Step 6: Add the param and config-table entries**

`trackerkit/engine_params.py`, immediately after the `APRILTAG_DECIMATE` line:

```python
        "APRILTAG_CROP_PADDING": float(
            _cfg_get(cfg, "apriltag_crop_padding", default=0.0)
        ),
```

`trackerkit/cli_config.py`, in the same defaults table that holds `apriltag`-family keys, add `"apriltag_crop_padding": 0.0,`.

`resources/configs/default.json` and `resources/configs/ooceraea_biroi.json`: add `"apriltag_crop_padding": 0.0,` adjacent to the existing `"apriltag_decimate"` entry.

- [ ] **Step 7: Add the GUI field**

`trackerkit/gui/panels/identity_panel.py`, after `spin_apriltag_decimate` is built:

```python
        self.spin_apriltag_crop_padding = QDoubleSpinBox()
        self.spin_apriltag_crop_padding.setRange(-0.5, 2.0)
        self.spin_apriltag_crop_padding.setValue(0.0)
        self.spin_apriltag_crop_padding.setSingleStep(0.05)
        self.spin_apriltag_crop_padding.setDecimals(2)
        self.spin_apriltag_crop_padding.setToolTip(
            "Padding around the detection's axis-aligned bounding box, as a\n"
            "fraction of its size, for AprilTag crops only.\n"
            "0.0 = the detection's exact extent (default). Negative tightens.\n"
            "Tag crops are never rotated or rescaled -- an AprilTag is a rigid\n"
            "printed square and any transform degrades decode."
        )
```

and extend the inline field row:

```python
        self.apriltag_row_widget = self._build_inline_fields_row(
            [
                ("Family", self.combo_apriltag_family, 1),
                ("Downsampling", self.spin_apriltag_decimate, 0),
                ("Crop padding", self.spin_apriltag_crop_padding, 0),
            ]
        )
```

- [ ] **Step 8: Wire the GUI through load, save, context, and preview**

`orchestrators/config.py` load path, after the `spin_apriltag_decimate.setValue(...)` block at :1215:

```python
        self._panels.identity.spin_apriltag_crop_padding.setValue(
            float(get_cfg("apriltag_crop_padding", default=0.0))
        )
```

`orchestrators/config.py` save path, in the `cfg.update({...})` at :1919:

```python
                "apriltag_crop_padding": self._panels.identity.spin_apriltag_crop_padding.value(),
```

`detection_panel.py` context dict, after the `"apriltag_decimate"` entry at :1940:

```python
            "apriltag_crop_padding": (
                ip.spin_apriltag_crop_padding.value() if ip is not None else 0.0
            ),
```

`workers/preview_worker.py`, in the AprilTag stage block after `APRILTAG_DECIMATE`:

```python
    params["APRILTAG_CROP_PADDING"] = float(context.get("apriltag_crop_padding", 0.0))
```

- [ ] **Step 9: Run the tests**

Run: `PYTHONPATH=$PWD/src python -m pytest tests/test_crop_padding_retirement.py tests/test_cache_ids.py tests/test_trackerkit_preview_worker.py -v`
Expected: the four new tests PASS. `test_cache_ids.py` will FAIL at `INDIVIDUAL_CROP_PADDING` assertions (lines 22, 61-62) — fix those now by switching the fixture keys to `APRILTAG_CROP_PADDING` (`0.0` base, `0.2` alt), since that test is asserting exactly the AprilTag cache-id sensitivity this task moved.

- [ ] **Step 10: Commit**

```bash
make format && make lint
git add -A
git commit -m "feat(apriltag): stage-local apriltag_crop_padding, default 0.0 (bare OBB extent)"
```

---

### Task 3: Unify the two AABB crop helpers

Two implementations produce the AprilTag crop today and they **disagree by up to one pixel even at padding 0.0**: `extract_aabb_crops` (`core/inference/stages/crops.py:110`, live path) truncates with `int(x1 - pad)` / `int(x2 + pad)`, while `_expand_obb_to_aabb` (`core/tracking/pose/pose_pipeline.py:66`, interpolated-crop path) scales corners about the centroid and uses `floor` / `ceil + 1`. They also mean different things by "padding": a fraction of `max(bw, bh)` versus a scale factor on the corner vectors. With one knob now explicitly owned by the AprilTag stage, two semantics is a latent divergence between live and interpolated tag crops.

**Files:**
- Modify: `src/hydra_suite/core/tracking/pose/pose_pipeline.py:64-82`
- Test: `tests/test_crop_padding_retirement.py`

**Interfaces:**
- Consumes: `apriltag_crop_padding` semantics from Task 2.
- Produces: `_expand_obb_to_aabb(corners, padding_fraction, frame_h, frame_w) -> tuple[int, int, int, int]` — same signature, now returning exactly the bounds `extract_aabb_crops` would use for the same corners and padding.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_crop_padding_retirement.py`:

```python
def _corners(cx, cy, w, h):
    return np.array(
        [[cx - w / 2, cy - h / 2], [cx + w / 2, cy - h / 2],
         [cx + w / 2, cy + h / 2], [cx - w / 2, cy + h / 2]],
        dtype=np.float32,
    )


def test_zero_padding_is_the_exact_obb_extent():
    from hydra_suite.core.tracking.pose.pose_pipeline import _expand_obb_to_aabb

    corners = _corners(100.0, 80.0, 40.0, 20.0)
    x0, y0, x1, y1 = _expand_obb_to_aabb(corners, 0.0, 480, 640)
    assert (x0, y0, x1, y1) == (80, 70, 120, 90)


def test_negative_padding_shrinks_symmetrically():
    from hydra_suite.core.tracking.pose.pose_pipeline import _expand_obb_to_aabb

    corners = _corners(100.0, 80.0, 40.0, 20.0)
    x0, y0, x1, y1 = _expand_obb_to_aabb(corners, -0.25, 480, 640)
    # pad = -0.25 * max(40, 20) = -10 on every side
    assert (x0, y0, x1, y1) == (90, 80, 110, 80)


def test_aabb_helpers_agree_with_the_live_path():
    from hydra_suite.core.inference.stages.crops import extract_aabb_crops
    from hydra_suite.core.tracking.pose.pose_pipeline import _expand_obb_to_aabb

    class _StubOBB:
        num_detections = 1
        corners = np.stack([_corners(100.0, 80.0, 40.0, 20.0)])

    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    for pad in (0.0, 0.1, 0.3):
        live = extract_aabb_crops(frame, _StubOBB(), padding=pad)[0]
        x0, y0, x1, y1 = _expand_obb_to_aabb(
            _StubOBB.corners[0], pad, frame.shape[0], frame.shape[1]
        )
        assert live.shape[:2] == (y1 - y0, x1 - x0), f"mismatch at padding={pad}"
```

- [ ] **Step 2: Run to verify failure**

Run: `PYTHONPATH=$PWD/src python -m pytest tests/test_crop_padding_retirement.py -v -k aabb or extent or shrink`
Expected: FAIL — the centroid-scaling implementation produces different bounds.

- [ ] **Step 3: Rewrite `_expand_obb_to_aabb` on the live-path geometry**

Replace the body in `core/tracking/pose/pose_pipeline.py`:

```python
def _expand_obb_to_aabb(
    corners: np.ndarray,
    padding_fraction: float,
    frame_h: int,
    frame_w: int,
) -> Tuple[int, int, int, int]:
    """Axis-aligned bounding box of ``corners``, padded, as ``(x0, y0, x1, y1)``.

    Deliberately identical to ``core.inference.stages.crops.extract_aabb_crops``:
    the padding is a fraction of the AABB's larger side, applied to all four
    sides. The two call sites (live inference and interpolated crops) feed the
    same AprilTag decoder, so they must cut the same pixels -- previously they
    disagreed by up to one pixel even at ``padding_fraction=0.0``.
    """
    x1 = float(corners[:, 0].min())
    y1 = float(corners[:, 1].min())
    x2 = float(corners[:, 0].max())
    y2 = float(corners[:, 1].max())
    pad = float(padding_fraction) * max(x2 - x1, y2 - y1)
    x0 = max(0, int(x1 - pad))
    y0 = max(0, int(y1 - pad))
    x1i = min(frame_w, int(x2 + pad))
    y1i = min(frame_h, int(y2 + pad))
    return x0, y0, x1i, y1i
```

- [ ] **Step 4: Run to verify pass**

Run: `PYTHONPATH=$PWD/src python -m pytest tests/test_crop_padding_retirement.py tests/test_inference_stages_crops.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
make format && make lint
git add src/hydra_suite/core/tracking/pose/pose_pipeline.py tests/test_crop_padding_retirement.py
git commit -m "fix(apriltag): one AABB crop geometry for live and interpolated paths"
```

---

### Task 4: Remove `padding_fraction` from the core crop / dataset / export APIs

**Files:**
- Modify: `src/hydra_suite/core/canonicalization/crop.py:339`, `:358-360`, `:373-387`
- Modify: `src/hydra_suite/core/inference/config.py:383` (`PoseConfig.crop_padding`), `:984`
- Modify: `src/hydra_suite/core/inference/cache/keys.py:255`
- Modify: `src/hydra_suite/core/individual/dataset/generator.py:79-124`, `:363`, `:445`, `:559`, `:597`, `:663-760`, `:848`
- Modify: `src/hydra_suite/core/individual/dataset/oriented_video.py:166`, `:194-215`, `:985`, `:1329-1340`
- Modify: `src/hydra_suite/core/post/media_export.py:846`, `:904`
- Modify: `src/hydra_suite/core/tracking/session.py:480`
- Test: `tests/test_crop_padding_retirement.py`, plus updates to `tests/test_canonical_crop.py`, `test_inference_cache_keys.py`, `test_media_export.py`, `test_oriented_track_video_export.py`, `test_canonical_dataset_provenance.py`, `test_crop_export_lossless.py`, `test_export_clipping_surfaced.py`, `tests/core/individual/dataset/test_oriented_video_actual_rows.py`

**Interfaces:**
- Consumes: nothing from Tasks 2-3 (AprilTag is already independent).
- Produces:
  - `extract_and_classify_batch(frames, per_frame_corners, canvas_w=None, canvas_h=None, bg_color=(0,0,0), suppress_foreign=True, per_frame_all_corners=None, *, geometry=None)` — **no `padding_fraction` parameter**.
  - `OrientedTrackVideoExporter(...)` — **no `padding_fraction` keyword**.
  - `export_final_media(...)` — **no `padding_fraction` argument**.
  - `IndividualDatasetGenerator.__init__` raises `ValueError` when `ADVANCED_CONFIG.reference_aspect_ratio <= 0`.
  - `PoseConfig` — **no `crop_padding` field**.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_crop_padding_retirement.py`:

```python
def test_extract_and_classify_batch_rejects_padding_fraction():
    from hydra_suite.core.canonicalization.crop import extract_and_classify_batch

    frame = np.zeros((200, 200, 3), dtype=np.uint8)
    with pytest.raises(TypeError):
        extract_and_classify_batch(
            [frame], [[_corners(100.0, 100.0, 40.0, 20.0)]],
            128, 64, padding_fraction=0.1,
        )


def test_dataset_generator_rejects_non_positive_aspect_ratio():
    from hydra_suite.core.individual.dataset.generator import (
        IndividualDatasetGenerator,
    )

    params = {
        "REFERENCE_BODY_SIZE": 20.0,
        "RESIZE_FACTOR": 1.0,
        "ADVANCED_CONFIG": {"reference_aspect_ratio": 0.0, "canonical_margin": 1.3},
    }
    with pytest.raises(ValueError, match="reference_aspect_ratio"):
        IndividualDatasetGenerator(params, output_dir=None, video_name="v")


def test_pose_config_has_no_crop_padding():
    from hydra_suite.core.inference.config import PoseConfig

    assert not hasattr(PoseConfig(), "crop_padding")
```

Note: match `IndividualDatasetGenerator`'s real constructor signature at the call site — read it before writing the test rather than assuming the keyword names above.

- [ ] **Step 2: Run to verify failure**

Run: `PYTHONPATH=$PWD/src python -m pytest tests/test_crop_padding_retirement.py -v -k "padding_fraction or aspect_ratio or pose_config"`
Expected: FAIL on all three.

- [ ] **Step 3: Strip the parameter from `extract_and_classify_batch`**

In `core/canonicalization/crop.py`: delete the `padding_fraction` parameter, its docstring entry, and the whole 373-387 reconciliation block. Replace the effective-geometry construction so the synthesized fallback derives its margin from the canvas instead of a padding argument:

```python
    canvas_w, canvas_h = _resolve_canvas(canvas_w, canvas_h, geometry)

    results: List[List[Optional[CanonicalCropResult]]] = []

    for fi, frame in enumerate(frames):
        ...
        # One code path: a caller that passed bare canvas dimensions gets a
        # geometry synthesised from them. There is no separate padding knob --
        # the canvas IS the framing (spec 2026-08-18).
        effective_geometry = geometry or CanonicalGeometry(
            canvas_wh=(int(canvas_w), int(canvas_h)),
            margin=1.0,
            aspect_ratio=max(1.0, float(canvas_w) / max(1.0, float(canvas_h))),
        )
```

Then update `tests/test_canonical_crop.py:234`, `:349`, `:362`: drop the `padding_fraction=` arguments. The `:362` case (which asserted the disagreement `ValueError`) is replaced by the new `TypeError` test in this task's Step 1 — delete it.

- [ ] **Step 4: Remove `PoseConfig.crop_padding` and its cache-key term**

`core/inference/config.py`: delete the `crop_padding: float = 0.1` field from `PoseConfig` (line 383) and the `crop_padding=float(params.get("INDIVIDUAL_CROP_PADDING", 0.1)),` line in `common_pose_kwargs` (line 984).

`core/inference/cache/keys.py:255`:

```python
    config_hash = _sha(
        f"{config.suppress_foreign_regions}|{canonical_geometry_key(geometry)}"
    )
```

Update `tests/test_inference_cache_keys.py`: drop `crop_padding=padding` from the pose-config factory at :72, and replace `test_pose_key_changes_with_crop_padding` with a test that the pose key changes with `geometry` (the term that now carries framing):

```python
def test_pose_key_changes_with_geometry():
    from hydra_suite.core.canonicalization.geometry import CanonicalGeometry

    g1 = CanonicalGeometry.from_reference(80.0, 2.0, 1.3)
    g2 = CanonicalGeometry.from_reference(80.0, 2.0, 1.6)
    assert _pose_key(geometry=g1) != _pose_key(geometry=g2)
```

Match `_pose_key`'s real helper name/signature in that file.

- [ ] **Step 5: Strip padding from `IndividualDatasetGenerator`**

In `core/individual/dataset/generator.py`:
- Delete `self.padding_fraction = params.get("INDIVIDUAL_CROP_PADDING", 0.1)` (line 80) and `self._canonical_padding` (line 99).
- Replace the `self._canonical_enabled` warning block (lines 100-118) with a hard failure:

```python
        _adv = params.get("ADVANCED_CONFIG", {})
        self._canonical_ref_ar = float(_adv.get("reference_aspect_ratio", 2.0))
        if self._canonical_ref_ar <= 0:
            raise ValueError(
                "ADVANCED_CONFIG.reference_aspect_ratio must be > 0: the crop "
                "dataset is extracted through the project-wide canonical canvas, "
                "the same one inference, head-tail, and the classifiers use. There "
                "is no non-canonical fallback (spec 2026-08-18)."
            )
```

- Delete every `self._canonical_enabled` guard (lines 363, 559) and the legacy branches they select — `_extract_obb_masked_crop` (lines 663-760) and its call sites at 445 and 597 are unreachable once the canonical path is unconditional. Remove the function and the `obb_corners_expanded_local` metadata field that only it produced (lines 431, 513-514, 591, 643, 743).
- Remove `"padding_fraction": self.padding_fraction,` from the metadata `parameters` block (line 848). The `canonical` sub-block stays.

Update `tests/test_canonical_dataset_provenance.py:32` — it asserts `parameters.padding_fraction`; switch the assertion to the `canonical` block (`margin`, `canvas_wh`), which is what now records framing.

- [ ] **Step 6: Strip padding from `OrientedTrackVideoExporter`**

In `core/individual/dataset/oriented_video.py`:
- Delete the `padding_fraction: float = 0.1` keyword (line 166) and `self.padding_fraction = ...` (line 194).
- The geometry-less fallback becomes:

```python
        if geometry is not None:
            self._geometry: CanonicalGeometry = geometry
        else:
            self._geometry = CanonicalGeometry.from_reference(
                reference_body_px=_DEFAULT_REFERENCE_BODY_PX,
                aspect_ratio=_DEFAULT_REFERENCE_ASPECT_RATIO,
                margin=_DEFAULT_CANONICAL_MARGIN,
            )
```

with a new module constant next to the other two:

```python
_DEFAULT_CANONICAL_MARGIN = 1.3
```

Keep the existing loud warning that follows.
- Line 985 becomes:

```python
            expanded_corners=self._expand_corners(
                corners, self._geometry.margin - 1.0
            ),
```

`_expand_corners` itself (1329-1340) is unchanged — it still takes a fraction; only its caller's source changes.

Update the `padding_fraction=` arguments in `tests/test_oriented_track_video_export.py` (lines 245, 376, 474, 582, 678, 790, 816, 848) and `tests/core/individual/dataset/test_oriented_video_actual_rows.py:59` — delete them. The two 0.1 cases at 816/848 were asserting padded-mask behavior; re-point them at a `geometry=` with the margin they want.

- [ ] **Step 7: Stop threading padding through the export entry points**

`core/post/media_export.py`: delete the `padding_fraction,` parameter (line 846) and the `padding_fraction=max(0.0, float(padding_fraction)),` argument to the exporter (line 904).

`core/tracking/session.py`: delete `padding_fraction=float(self.config.get("individual_crop_padding", 0.1)),` (line 480). The `geometry=canonical_geometry_from_params(self.params)` argument directly below it stays and is now the only framing input.

Update `tests/test_media_export.py:267`, `:308` and `tests/test_export_clipping_surfaced.py:92` — drop the `padding_fraction=` arguments; `tests/test_crop_export_lossless.py:19` and `test_export_clipping_surfaced.py:24` — drop `INDIVIDUAL_CROP_PADDING` from the params fixtures.

- [ ] **Step 8: Run the affected test files**

Run:
```bash
PYTHONPATH=$PWD/src python -m pytest \
  tests/test_crop_padding_retirement.py tests/test_canonical_crop.py \
  tests/test_inference_cache_keys.py tests/test_media_export.py \
  tests/test_oriented_track_video_export.py tests/test_canonical_dataset_provenance.py \
  tests/test_crop_export_lossless.py tests/test_export_clipping_surfaced.py \
  tests/core/individual/dataset/test_oriented_video_actual_rows.py -v
```
Expected: PASS. Run these per-file if the batch hangs — `make pytest` on the whole suite is known to hang on classkit modal dialogs.

- [ ] **Step 9: Commit**

```bash
make format && make lint
git add -A
git commit -m "refactor(core): canonical geometry is the only crop framing input"
```

---

### Task 5: Remove `individual_crop_padding` from config, params, and GUI

**Files:**
- Modify: `src/hydra_suite/trackerkit/engine_params.py:1332-1334`
- Modify: `src/hydra_suite/trackerkit/cli_config.py`
- Modify: `src/hydra_suite/trackerkit/gui/panels/identity_panel.py:260-272`
- Modify: `src/hydra_suite/trackerkit/gui/panels/detection_panel.py:1959-1961`
- Modify: `src/hydra_suite/trackerkit/gui/orchestrators/config.py:1225-1233`, `:1392-1394`, `:1978`
- Modify: `src/hydra_suite/trackerkit/gui/workers/preview_worker.py:561-563`
- Modify: `src/hydra_suite/resources/configs/default.json:138`, `ooceraea_biroi.json:217`
- Modify: `tests/data/get_parameters_dict_golden/fly_obb.json`, `ant_cnn_identity.json`
- Modify: `tests/test_trackerkit_preview_worker.py:49`, `:65`, `:511`
- Test: `tests/test_crop_padding_retirement.py`

**Interfaces:**
- Consumes: `apriltag_crop_padding` from Task 2 (the only surviving padding knob).
- Produces: no `individual_crop_padding` config key, no `INDIVIDUAL_CROP_PADDING` param key, no `identity_panel.spin_individual_padding` widget.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_crop_padding_retirement.py`:

```python
def test_engine_params_no_longer_emit_individual_crop_padding():
    assert "INDIVIDUAL_CROP_PADDING" not in _params({"individual_crop_padding": 0.5})


def test_legacy_individual_crop_padding_warns_once(caplog):
    """A config still carrying the retired key loads, warns, and ignores it."""
    with caplog.at_level("WARNING"):
        _params({"individual_crop_padding": 0.5})
    messages = [r.message for r in caplog.records if "individual_crop_padding" in r.message]
    assert len(messages) == 1
    assert "canonical_margin" in messages[0]
    assert "apriltag_crop_padding" in messages[0]
```

- [ ] **Step 2: Run to verify failure**

Run: `PYTHONPATH=$PWD/src python -m pytest tests/test_crop_padding_retirement.py -v -k individual_crop_padding`
Expected: FAIL — the key is still emitted and nothing warns.

- [ ] **Step 3: Remove the param and add the legacy warning**

`trackerkit/engine_params.py`: delete the `"INDIVIDUAL_CROP_PADDING": float(...)` entry (lines 1332-1334). In the same builder, before the params dict is assembled, add:

```python
    if _cfg_get(cfg, "individual_crop_padding", default=None) is not None:
        logger.warning(
            "Config key 'individual_crop_padding' is retired and ignored. Crop "
            "framing for every model- and dataset-facing crop now comes from "
            "ADVANCED_CONFIG.canonical_margin; AprilTag crops have their own "
            "'apriltag_crop_padding' (default 0.0 = the detection's exact "
            "extent). See docs/superpowers/specs/2026-08-18-crop-padding-"
            "retirement-design.md."
        )
```

Use the module's existing `logger`; if `engine_params.py` has none, add `logger = logging.getLogger(__name__)` next to its imports following the pattern in `cli_config.py`.

`trackerkit/cli_config.py`: delete the `individual_crop_padding` default if the table carries one.

- [ ] **Step 4: Delete the stale `cnn_classifier_crop_padding` warning**

`orchestrators/config.py:1225-1233`: delete the whole `_legacy_crop_padding` block. Its message points users at `individual_crop_padding`, which no longer exists; the Task-5 Step-3 warning supersedes it.

- [ ] **Step 5: Remove the GUI widget and its wiring**

`identity_panel.py`: delete `self.spin_individual_padding` (lines 260-269) and the `fl_common.addRow("Crop padding fraction (all phases)", ...)` row (lines 270-272).

`orchestrators/config.py`: delete the load block at :1392-1394 and the `"individual_crop_padding": ...` save entry at :1978.

`detection_panel.py`: delete the `"individual_crop_padding": (...)` context entry at :1959-1961.

`preview_worker.py`: delete the `params["INDIVIDUAL_CROP_PADDING"] = ...` assignment (lines 561-563) and re-word the `# Shared crop geometry (pose + AprilTag).` comment above it to `# Foreign-OBB suppression (shared by every crop stage).`

- [ ] **Step 6: Remove the key from bundled configs and refresh the goldens**

Delete `"individual_crop_padding"` from `resources/configs/default.json:138` and `resources/configs/ooceraea_biroi.json:217`.

The two characterization goldens are `tests/data/get_parameters_dict_golden/fly_obb.json` and `ant_cnn_identity.json`, gated by `tests/test_get_parameters_dict_characterization.py`. That test has **no regeneration flag by design** — a missing or stale golden is a hard failure, so the diff must be made deliberately.

Run: `PYTHONPATH=$PWD/src python -m pytest tests/test_get_parameters_dict_characterization.py -v`
Expected: FAIL, naming `INDIVIDUAL_CROP_PADDING` (present in golden, absent from capture) and `APRILTAG_CROP_PADDING` (absent from golden, present in capture).

Then hand-edit both golden files: delete the `"INDIVIDUAL_CROP_PADDING"` line (`fly_obb.json:142` = 0.1, `ant_cnn_identity.json:176` = 0.5) and add `"APRILTAG_CROP_PADDING": 0.0,` in key order. Re-run.
Expected: PASS. If the failure names **any** key beyond those two, stop — this change is not supposed to move anything else, and that is exactly what this golden exists to catch.

- [ ] **Step 7: Update the preview-worker tests**

`tests/test_trackerkit_preview_worker.py`: at :49 replace `"individual_crop_padding": 0.2` with `"apriltag_crop_padding": 0.2`; at :65 assert `params["APRILTAG_CROP_PADDING"] == 0.2`; at :511 replace `{"individual_crop_padding": 0.0}` with `{"apriltag_crop_padding": 0.0}`.

- [ ] **Step 8: Run the tests**

Run:
```bash
PYTHONPATH=$PWD/src python -m pytest \
  tests/test_crop_padding_retirement.py tests/test_trackerkit_preview_worker.py \
  tests/test_get_parameters_dict_characterization.py tests/test_main_window_config_persistence.py -v
```
Expected: PASS.

- [ ] **Step 9: Commit**

```bash
make format && make lint
git add -A
git commit -m "refactor: retire individual_crop_padding; canonical_margin is the framing dial"
```

---

### Task 6: Grep gate and documentation

**Files:**
- Modify: `docs/` — any user-facing page naming `individual_crop_padding` or the batching control.
- Test: shell grep gate (run manually; not a pytest).

**Interfaces:**
- Consumes: the completed removals from Tasks 1-5.
- Produces: nothing consumed downstream.

- [ ] **Step 1: Run the grep gate**

Run:
```bash
grep -rn "INDIVIDUAL_CROP_PADDING\|individual_crop_padding" src/ tests/ scripts/
```
Expected: exactly one hit — the legacy warning added in Task 5 Step 3 (`engine_params.py`), plus any test asserting that warning. Nothing else.

- [ ] **Step 2: Confirm `roi_crop_padding_fraction` survived untouched**

Run: `git diff main --stat -- src/hydra_suite/advanced_config.json src/hydra_suite/trackerkit/advanced_config.json`
Expected: no changes to either file.

- [ ] **Step 3: Update docs**

Run: `grep -rn "individual_crop_padding\|Crop padding fraction\|Live Detection Batching" docs/ --include="*.md" | grep -v superpowers/specs`
For each hit, replace the description with: crop framing comes from `canonical_margin`; AprilTag crops have `apriltag_crop_padding` (default 0.0); the batching control is "Detection Frame Batching / Frames per detector call" and is stage-1 only.

- [ ] **Step 4: Build the docs**

Run: `make docs-check`
Expected: PASS (strict mkdocs build + terminology check).

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "docs: crop framing is canonical_margin; apriltag_crop_padding for tag crops"
```

---

### Task 7: Full test-suite delta gate

**Files:** none modified (verification only).

**Interfaces:**
- Consumes: everything from Tasks 1-6.
- Produces: a recorded pass/fail delta against the pre-change baseline.

- [ ] **Step 1: Record the baseline**

From the **main checkout** (not the worktree), on the pre-change commit:

```bash
conda activate hydra-mps
python -m pytest tests/ -x --co -q > /tmp/baseline_collect.txt 2>&1 || true
```

The whole suite is known to hang on classkit modal dialogs, so run in per-file batches rather than one invocation, and record which files fail on baseline. Memory `project-main-suite-blockers` and `project-runtime-gen2-core-done` document ~24 pre-existing failures — the gate is the **delta**, not zero failures.

- [ ] **Step 2: Run the same batches in the worktree**

```bash
cd .worktrees/crop-padding
PYTHONPATH=$PWD/src python -m pytest tests/<file>.py -q   # per file
```

- [ ] **Step 3: Diff the failure sets**

Expected: the worktree's failing set is a subset of the baseline's. Any new failure is a regression — fix it before proceeding; do not proceed on "probably pre-existing".

- [ ] **Step 4: Record the result in the plan**

Append the baseline-vs-worktree failure counts and the names of any file whose status changed, as a comment block at the bottom of this plan file, then commit it.

---

### Task 8: Equivalence gate on MPS and CUDA

**Files:** none modified (verification only).

**Interfaces:**
- Consumes: the complete change.
- Produces: the merge decision.

- [ ] **Step 1: Kill stale processes**

Run: `pgrep -fl "sleap|hydra" ` and terminate only stale sleap/hydra processes. Never interfere with anything else.

- [ ] **Step 2: Build the baseline worktree from the pre-change commit**

The baseline for this slice is **pre-change HEAD**, not `legacy/main` — the point is to isolate this slice's effect.

```bash
git worktree add --detach .worktrees/equiv-base <pre-change-sha>
```

- [ ] **Step 3: Run the matrix on MPS**

```bash
conda activate hydra-mps
bash tools/equivalence/fixtures/fetch_fixtures.sh    # once per machine
REPO=$PWD WT=$PWD \
  MAIN_SRC=$PWD/.worktrees/equiv-base/src WT_SRC=$PWD/.worktrees/crop-padding/src \
  OUT=/tmp/equiv_croppad RUNTIME=mps \
  bash tools/equivalence/run_matrix.sh
```

- [ ] **Step 4: Check the non-AprilTag clips**

Expected **byte-identical** (at the determinism floor, modulo the documented bistable head/tail π-flips) on both `_forward.csv` and `_tracking_final.csv`: `fly_obb`, `worm_bgsub`, `ant_pose_headtail`, `ant_obb_sleap`, `ant_obb_sequential`, `ant_cnn_identity`.

Before trusting any EQUIVALENT verdict, run `wc -l` on the produced CSVs and confirm > 1 — an inactive conda env yields empty CSVs that falsely compare equivalent.

- [ ] **Step 5: Diff the AprilTag clip column-wise**

`emi_obb_identity` is expected to differ. Diff it by column:

```bash
python - <<'PY'
import pandas as pd
a = pd.read_csv("/tmp/equiv_croppad/emi_obb_identity/base/..._tracking_final.csv")
b = pd.read_csv("/tmp/equiv_croppad/emi_obb_identity/new_a/..._tracking_final.csv")
assert list(a.columns) == list(b.columns)
for col in a.columns:
    same = a[col].equals(b[col])
    print(f"{'SAME' if same else 'DIFF'}  {col}")
PY
```

Expected: geometry columns (`x`, `y`, `theta`, track/detection ids, frame) **SAME**; differences confined to tag/identity columns. A difference in any geometry column is a regression — stop and investigate.

- [ ] **Step 6: Repeat on CUDA (mehek)**

```bash
ssh rutalab@mehek.taild08eb9.ts.net
cd ~/hydra-suite && git fetch origin && git checkout <branch-sha>
source ~/mambaforge/etc/profile.d/conda.sh && conda activate hydra-cuda
git worktree add --detach .worktrees/equiv-base <pre-change-sha>
REPO=$PWD WT=$PWD MAIN_SRC=$PWD/.worktrees/equiv-base/src WT_SRC=$PWD/src \
  OUT=/tmp/equiv_croppad RUNTIME=cuda nohup bash tools/equivalence/run_matrix.sh > /tmp/equiv_cuda.log 2>&1 &
```

Same acceptance as steps 4-5.

- [ ] **Step 7: Report and clean up**

Report per-clip verdicts (including the AprilTag column diff) honestly — name any clip that was skipped or produced empty CSVs. Then:

```bash
git worktree remove --force .worktrees/equiv-base && git worktree prune
```

- [ ] **Step 8: Finish the branch**

Use the `superpowers:finishing-a-development-branch` skill to decide integration. Default for this repo: merge `--no-ff` into local `main`, delete the branch and worktree, and record the outcome in memory (`project_crop_padding_retirement`).

---

## Self-Review Notes

**Spec coverage:** Part 1 → Task 1. Part 2 config/params/GUI → Task 5; core → Task 4; legacy-key warning → Task 5 Step 3; bundled configs → Tasks 2 Step 6 and 5 Step 6. Part 3 → Task 2. Oriented-video mask change → Task 4 Step 6. Testing section items 1-6 → Tasks 2-6; items 7-9 → Task 8.

**Added beyond the spec:** Task 3 (unifying the two AABB helpers). Found while writing this plan: `extract_aabb_crops` and `_expand_obb_to_aabb` disagree by up to one pixel even at padding 0.0, and mean different things by "padding". With one AprilTag-owned knob, two semantics is a live divergence between the inference and interpolated tag-crop paths. Folding it in here is the right moment; it is called out separately so it can be dropped without disturbing the rest.

**Known unverified call shapes** (read the file before writing, do not assume): the `RuntimeContext` import path and config shape `build_engine_params` wants, per `tests/test_get_parameters_dict_characterization.py:263-281`; `IndividualDatasetGenerator.__init__`'s keyword names; the pose-key helper name in `tests/test_inference_cache_keys.py`.
