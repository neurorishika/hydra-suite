# ViTPose in PoseKit Inference Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make ViTPose a third selectable backend in PoseKit's model-assisted inference (single-frame *Predict Keypoints* and *Predict Dataset*), exactly paralleling the existing YOLO and SLEAP paths.

**Architecture:** PoseKit inference routes GUI selection → `PosePredictWorker`/`BulkPosePredictWorker` → `_build_pose_backend` → `load_pose_backend` (`core/inference/api.py`) → `stages/pose.load_pose_model`. The `stages/pose.py` vitpose branch and `PoseViTPoseConfig` already exist on this branch (tracking plan Tasks 5–6). This plan fills the three untouched links: the `load_pose_backend` shim branch, the worker param threading + fallback policy, and the PoseKit GUI selector/widget. No new runtime code — PoseKit stays runtime-agnostic, so native (cpu/gpu/mps) works immediately and accelerated (TensorRT/CoreML) comes for free from the already-built backend.

**Tech Stack:** PyQt (offscreen-testable), PyTorch (fp32 ViTPose leaf), the existing `ViTPoseBackend` + `PoseViTPoseConfig` on this branch.

## Global Constraints

Every task's requirements implicitly include these:

- **Worktree:** all work happens in `.worktrees/vitpose-full-integration` (branch `vitpose-full-integration`). Run all commands from that directory.
- **Environment:** use the `hydra-mps` conda env (base env torch is broken). Prefix python/pytest with `conda run -n hydra-mps`.
- **Do NOT run `make format`** (broken: black `pathspec` error). Run `conda run -n hydra-mps black <files>` and `conda run -n hydra-mps isort <files>` directly on the task's changed files.
- **Line length 88.** Follow the surrounding code style (`yolo`/`sleap` paths are the size/style model).
- **Commit as the configured git user** — do NOT add a `Co-Authored-By: Claude` trailer.
- **Fallback policy for ViTPose = SLEAP's, not YOLO's:** ViTPose has no legacy `PoseInferenceService` path, so on backend failure it **re-raises** (never silently falls back to the legacy predict path).
- **No user-facing variant/K inputs:** the checkpoint adapter infers variant/head/num_keypoints; `PoseViTPoseConfig(variant="auto", num_keypoints=0)` is always used.
- **No "Use Latest" for ViTPose (YAGNI):** nothing populates a `latest_vitpose_checkpoint` yet, so the ViTPose widget offers **Browse only** (a dead "Use Latest" button would mislead). Do not touch `gui/models.py`.

## Verified existing APIs (consume these verbatim — do not guess)

- **`core/inference/api.py:41`** `load_pose_backend(*, backend_family, model_path, compute_runtime, keypoint_names=None, skeleton_edges=None, skeleton_file="", confidence_threshold=1e-4, batch_size=64, min_valid_confidence=0.2, out_root=".", exported_model_path="", sleap_env="sleap", sleap_batch=None, sleap_max_instances=1)`. Branch at `:83-110` (`if family == "yolo": ... else: <sleap>`). Imports `PoseViTPoseConfig` is NOT yet imported here — add it to the `from .config import (...)` block at `:71-79`.
- **`core/inference/config.py:195`** `PoseViTPoseConfig(model_path: str, variant: str = "auto", num_keypoints: int = 0, batch_size: int = 4)`; `:204` `PoseConfig.backend: Literal["yolo","sleap","vitpose"]`; `:208` `PoseConfig.vitpose: PoseViTPoseConfig | None`.
- **`core/identity/pose/backends/vitpose.py`** `ViTPoseBackend` (Protocol impl; `predict_batch(crops) -> List[PoseResult]`, `output_keypoint_names`, `close()`).
- **`core/identity/pose/vitpose/vitpose.py:27`** `build_vitpose(variant, head, num_keypoints=17) -> ViTPose` (for building test checkpoints).
- **`posekit/gui/workers.py:21`** `_build_pose_backend(*, backend_family, model_path, exported_model_path, compute_runtime, min_valid_conf, batch_size, conf, keypoint_names, skeleton_edges, out_root, sleap_env, sleap_batch=None, sleap_max_instances=1)`. Single worker `PosePredictWorker` (`:68`), bulk `BulkPosePredictWorker` (`:248`). Fallback branch in `run()` at `:201-206` (single) / the parallel site in bulk `run()` — `if self.backend == "sleap": raise ...`.
- **`posekit/gui/main_window.py`** — combo `:402-404`; `_pred_backend` `:4205`; `_update_pred_backend_ui` `:4312`; `_get_pred_model_or_prompt` `:4629`; `_get_pred_model_silent` `:4636`; `_pred_runtime_flavor` stage-select `:4654`; `_browse_pred_weights` `:4735`; `_get_pred_weights_or_prompt` `:4747`; `_get_pred_weights_silent`; `_set_bulk_prediction_locked` widget list `:4291-4316`; YOLO widget build `:459-472`; settings save `:1391-1392`; `_apply_pred_settings` `:1457`; worker construction in `predict_current_frame` (`:4928`) and `predict_dataset` (`:5240`).

## File Structure

**Modified:**
- `src/hydra_suite/core/inference/api.py` — add `vitpose_batch` param + `family == "vitpose"` branch to `load_pose_backend`.
- `src/hydra_suite/posekit/gui/workers.py` — thread `vitpose_batch`; re-raise (no legacy fallback) for vitpose in both workers.
- `src/hydra_suite/posekit/gui/main_window.py` — combo item, ViTPose widget, `_pred_backend`, `_update_pred_backend_ui`, model-picker getters, `_pred_runtime_flavor` stage, bulk-lock list, settings save/restore, worker construction.

**New tests:**
- `tests/test_vitpose_posekit_load_backend.py`
- `tests/test_vitpose_posekit_workers.py`
- `tests/test_vitpose_posekit_gui_wiring.py`

---

## Task 1: `load_pose_backend` ViTPose branch (`core/inference/api.py`)

**Files:**
- Modify: `src/hydra_suite/core/inference/api.py:41-131`
- Test: `tests/test_vitpose_posekit_load_backend.py`

**Interfaces:**
- Consumes: `PoseViTPoseConfig` (`core/inference/config.py:195`), `ViTPoseBackend`, `build_vitpose` (test only).
- Produces: `load_pose_backend(backend_family="vitpose", model_path, compute_runtime, vitpose_batch=None, ...)` returns a `ViTPoseBackend` built from a `PoseConfig(backend="vitpose", vitpose=PoseViTPoseConfig(...))`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_vitpose_posekit_load_backend.py
"""PoseKit's load_pose_backend shim must route a "vitpose" family to a
ViTPoseBackend (not misroute it into the SLEAP else-branch)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from hydra_suite.core.inference.api import load_pose_backend
from hydra_suite.core.identity.pose.backends.vitpose import ViTPoseBackend
from hydra_suite.core.identity.pose.vitpose.vitpose import build_vitpose


def _tiny_ckpt(tmp_path: Path, k: int = 4) -> Path:
    model = build_vitpose("S", "classic", num_keypoints=k)
    p = tmp_path / "best.pt"
    torch.save(
        {"model_state": model.state_dict(), "variant": "S", "num_keypoints": k}, p
    )
    return p


def test_load_pose_backend_builds_vitpose(tmp_path):
    ckpt = _tiny_ckpt(tmp_path, k=4)
    backend = load_pose_backend(
        backend_family="vitpose",
        model_path=str(ckpt),
        compute_runtime="cpu",
        keypoint_names=["a", "b", "c", "d"],
        skeleton_edges=[],
        min_valid_confidence=0.0,
        batch_size=8,
        vitpose_batch=2,
        out_root=str(tmp_path),
    )
    assert isinstance(backend, ViTPoseBackend)
    assert backend.output_keypoint_names == ["a", "b", "c", "d"]
    out = backend.predict_batch([np.zeros((60, 40, 3), np.uint8)])
    assert len(out) == 1
    assert out[0].num_keypoints == 4
    backend.close()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `conda run -n hydra-mps python -m pytest tests/test_vitpose_posekit_load_backend.py -v`
Expected: FAIL — `load_pose_backend` has no `vitpose_batch` kwarg (TypeError) and/or the vitpose family falls into the SLEAP branch and builds a `PoseSLEAPConfig` (not a `ViTPoseBackend`).

- [ ] **Step 3a: Add `PoseViTPoseConfig` to the import block** at `api.py:71-79`

Change the `from .config import (` block to also import `PoseViTPoseConfig`:

```python
    from .config import (
        InferenceConfig,
        OBBConfig,
        OBBDirectConfig,
        PoseConfig,
        PoseSLEAPConfig,
        PoseViTPoseConfig,
        PoseYOLOConfig,
        migrate_runtime_to_tier,
    )
```

- [ ] **Step 3b: Add `vitpose_batch` to the signature** — insert after `sleap_max_instances=1,` at `api.py:56`:

```python
    sleap_max_instances=1,
    vitpose_batch=None,
):
```

- [ ] **Step 3c: Add the `vitpose` branch** — replace the `if family == "yolo": ... else:` at `api.py:84-110` so the three families are explicit. Keep the yolo and sleap bodies byte-identical; insert the vitpose branch between them:

```python
    family = (backend_family or "").strip().lower()
    if family == "yolo":
        pose_cfg = PoseConfig(
            backend="yolo",
            skeleton_file=skeleton_file or "",
            yolo=PoseYOLOConfig(
                model_path=model_path,
                confidence_threshold=confidence_threshold,
                batch_size=batch_size,
            ),
            min_keypoint_confidence=min_valid_confidence,
        )
    elif family == "vitpose":
        vp_bs = vitpose_batch if vitpose_batch is not None else batch_size
        pose_cfg = PoseConfig(
            backend="vitpose",
            skeleton_file=skeleton_file or "",
            vitpose=PoseViTPoseConfig(
                model_path=model_path,
                variant="auto",
                num_keypoints=0,
                batch_size=max(1, int(vp_bs)),
            ),
            min_keypoint_confidence=min_valid_confidence,
        )
    else:
        # SLEAP uses its own batch (sleap_batch) when supplied; fall back to the
        # shared batch_size otherwise. PoseSLEAPConfig has a single batch field
        # which load_pose_model forwards to PoseRuntimeConfig.sleap_batch.
        sleap_bs = sleap_batch if sleap_batch is not None else batch_size
        pose_cfg = PoseConfig(
            backend="sleap",
            skeleton_file=skeleton_file or "",
            sleap=PoseSLEAPConfig(
                model_path=model_path,
                conda_env=sleap_env or "sleap",
                batch_size=max(1, int(sleap_bs)),
                max_instances=int(sleap_max_instances),
            ),
            min_keypoint_confidence=min_valid_confidence,
        )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `conda run -n hydra-mps python -m pytest tests/test_vitpose_posekit_load_backend.py -v`
Expected: PASS (1 test). Output pristine.

- [ ] **Step 5: Format + commit**

```bash
conda run -n hydra-mps black src/hydra_suite/core/inference/api.py tests/test_vitpose_posekit_load_backend.py
conda run -n hydra-mps isort src/hydra_suite/core/inference/api.py tests/test_vitpose_posekit_load_backend.py
git add src/hydra_suite/core/inference/api.py tests/test_vitpose_posekit_load_backend.py
git commit -m "feat(vitpose): route posekit load_pose_backend to ViTPoseBackend"
```

---

## Task 2: PoseKit worker ViTPose plumbing (`posekit/gui/workers.py`)

**Files:**
- Modify: `src/hydra_suite/posekit/gui/workers.py` (`_build_pose_backend` `:21`; `PosePredictWorker` `:68`; `BulkPosePredictWorker` `:248`; both `run()` fallback branches)
- Test: `tests/test_vitpose_posekit_workers.py`

**Interfaces:**
- Consumes: `load_pose_backend` (Task 1, with `vitpose_batch`).
- Produces: `_build_pose_backend(..., vitpose_batch=None)` forwards `vitpose_batch` to `load_pose_backend`; both workers accept a `vitpose_batch` param and pass it; a `"vitpose"` backend re-raises on failure (no legacy fallback).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_vitpose_posekit_workers.py
"""ViTPose worker plumbing: _build_pose_backend threads vitpose_batch, and a
vitpose backend re-raises (no legacy PoseInferenceService fallback)."""
from __future__ import annotations

import importlib

import pytest


def _workers():
    return importlib.import_module("hydra_suite.posekit.gui.workers")


def test_build_pose_backend_threads_vitpose_batch(monkeypatch):
    workers = _workers()
    captured: dict[str, object] = {}

    def _fake(**kwargs):
        captured.update(kwargs)
        return object()

    monkeypatch.setattr(workers, "load_pose_backend", _fake)
    workers._build_pose_backend(
        backend_family="vitpose",
        model_path="/models/vit.pt",
        exported_model_path="",
        compute_runtime="cpu",
        min_valid_conf=0.0,
        batch_size=4,
        conf=0.25,
        keypoint_names=["a", "b"],
        skeleton_edges=[],
        out_root="/tmp/out",
        sleap_env=None,
        vitpose_batch=6,
    )
    assert captured["backend_family"] == "vitpose"
    assert captured["model_path"] == "/models/vit.pt"
    assert captured["vitpose_batch"] == 6


def test_vitpose_worker_reraises_no_legacy_fallback(monkeypatch, tmp_path):
    workers = _workers()

    # Force the shared build path to fail.
    def _boom(**kwargs):
        raise RuntimeError("build failed")

    monkeypatch.setattr(workers, "_build_pose_backend", _boom)

    # Guard: the legacy fallback path must NOT be reached for vitpose.
    called = {"legacy": False}

    class _FakeInfer:
        def __init__(self, *a, **k):
            pass

        def get_cached_pred(self, *a, **k):
            return None

        def predict(self, *a, **k):
            called["legacy"] = True
            return None, "legacy reached"

    monkeypatch.setattr(workers, "PoseInferenceService", _FakeInfer)

    img = tmp_path / "f.png"
    import cv2
    import numpy as np

    cv2.imwrite(str(img), np.zeros((16, 16, 3), np.uint8))

    errors: list[str] = []
    w = workers.PosePredictWorker(
        model_path=tmp_path / "vit.pt",
        image_path=img,
        out_root=tmp_path,
        keypoint_names=["a", "b"],
        skeleton_edges=[],
        backend="vitpose",
        runtime_flavor="cpu",
        vitpose_batch=4,
    )
    w.failed.connect(lambda msg: errors.append(msg))
    w.run()

    assert called["legacy"] is False
    assert errors and "vitpose" in errors[0].lower()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `conda run -n hydra-mps python -m pytest tests/test_vitpose_posekit_workers.py -v`
Expected: FAIL — `_build_pose_backend`/`PosePredictWorker` have no `vitpose_batch` kwarg; and the vitpose failure currently falls through to the legacy path (so `called["legacy"]` is True / no vitpose error emitted).

- [ ] **Step 3a: Add `vitpose_batch` to `_build_pose_backend`** (`workers.py:21-65`). Add the param to the signature and forward it:

```python
def _build_pose_backend(
    *,
    backend_family: str,
    model_path: str,
    exported_model_path: str,
    compute_runtime: str,
    min_valid_conf: float,
    batch_size: int,
    conf: float,
    keypoint_names: List[str],
    skeleton_edges: List[Tuple[int, int]],
    out_root: str,
    sleap_env: Optional[str],
    sleap_batch: Optional[int] = None,
    sleap_max_instances: int = 1,
    vitpose_batch: Optional[int] = None,
) -> Any:
```

and in the `return load_pose_backend(` call add, after `sleap_max_instances=sleap_max_instances,`:

```python
        sleap_max_instances=sleap_max_instances,
        vitpose_batch=vitpose_batch,
    )
```

- [ ] **Step 3b: Add `vitpose_batch` to `PosePredictWorker.__init__`** (`workers.py:75-116`). Add `vitpose_batch: Optional[int] = None,` to the signature (after `sleap_max_instances`), and store it in the body (after `self.sleap_max_instances = 1`):

```python
        self.vitpose_batch = vitpose_batch
```

- [ ] **Step 3c: Pass `vitpose_batch` in the single worker's `_build_pose_backend` call** (`workers.py:153-172`). After `sleap_max_instances=1,` add:

```python
                    sleap_max_instances=1,
                    vitpose_batch=self.vitpose_batch,
                )
```

- [ ] **Step 3d: Re-raise for vitpose in the single worker fallback** (`workers.py:201-206`). Change the guard from `if self.backend == "sleap":` to cover vitpose:

```python
            except Exception as exc:
                if self.backend in ("sleap", "vitpose"):
                    raise RuntimeError(
                        f"{self.backend} shared runtime path failed in PoseKit. "
                        "Legacy fallback is disabled for parity with MAT. "
                        f"Original error: {exc}"
                    ) from exc
                # Fallback to legacy PoseInferenceService path.
                logger.debug(
                    "Shared runtime predict path failed; falling back to legacy path.",
                    exc_info=True,
                )
```

- [ ] **Step 3e: Repeat 3b–3d for `BulkPosePredictWorker`** (`workers.py:248-420`). Add `vitpose_batch: Optional[int] = None,` to its `__init__` signature and `self.vitpose_batch = vitpose_batch`; add `vitpose_batch=self.vitpose_batch,` to its `_build_pose_backend(...)` call; and change its fallback guard (the bulk `run()` parallel of `:407-412`) from `if self.backend == "sleap":` to `if self.backend in ("sleap", "vitpose"):` with the same `{self.backend}`-formatted message.

- [ ] **Step 4: Run test to verify it passes**

Run: `conda run -n hydra-mps python -m pytest tests/test_vitpose_posekit_workers.py -v`
Expected: PASS (2 tests).

Also run the existing worker regression to confirm no behavior drift:
Run: `conda run -n hydra-mps python -m pytest tests/test_posekit_workers_pose_backend.py -v`
Expected: PASS (unchanged).

- [ ] **Step 5: Format + commit**

```bash
conda run -n hydra-mps black src/hydra_suite/posekit/gui/workers.py tests/test_vitpose_posekit_workers.py
conda run -n hydra-mps isort src/hydra_suite/posekit/gui/workers.py tests/test_vitpose_posekit_workers.py
git add src/hydra_suite/posekit/gui/workers.py tests/test_vitpose_posekit_workers.py
git commit -m "feat(vitpose): thread vitpose_batch + no-legacy-fallback in posekit workers"
```

---

## Task 3: PoseKit GUI selector + widget + worker wiring (`posekit/gui/main_window.py`)

**Files:**
- Modify: `src/hydra_suite/posekit/gui/main_window.py`
- Test: `tests/test_vitpose_posekit_gui_wiring.py`

**Interfaces:**
- Consumes: the vitpose worker params (Task 2) and `load_pose_backend` (Task 1).
- Produces: `"ViTPose"` is selectable; `_pred_backend()` returns `"vitpose"`; a `.pt` checkpoint is picked via `pred_vitpose_edit`; `_pred_runtime_flavor()` resolves the `"vitpose_pose"` stage; the workers are constructed with the vitpose checkpoint + `vitpose_batch`.

**Note:** the repo's PoseKit GUI tests do not construct a full `MainWindow`; they test pure methods via `SimpleNamespace` and assert source content. Follow that pattern.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_vitpose_posekit_gui_wiring.py
"""ViTPose GUI wiring: selectable in the combo, mapped by _pred_backend, and
the vitpose_pose resolver stage is used for its runtime flavor."""
from __future__ import annotations

import importlib
from types import SimpleNamespace


def _mw_module():
    import os

    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    return importlib.import_module("hydra_suite.posekit.gui.main_window")


class _Combo:
    def __init__(self, text: str):
        self._t = text

    def currentText(self) -> str:
        return self._t


def test_pred_backend_maps_vitpose():
    mw = _mw_module()
    self_ns = SimpleNamespace(combo_pred_backend=_Combo("ViTPose"))
    assert mw.MainWindow._pred_backend(self_ns) == "vitpose"
    self_ns2 = SimpleNamespace(combo_pred_backend=_Combo("SLEAP"))
    assert mw.MainWindow._pred_backend(self_ns2) == "sleap"
    self_ns3 = SimpleNamespace(combo_pred_backend=_Combo("YOLO"))
    assert mw.MainWindow._pred_backend(self_ns3) == "yolo"


def test_pred_runtime_flavor_uses_vitpose_stage(monkeypatch):
    mw = _mw_module()
    captured = {}

    class _Resolved:
        backend = "torch"
        device = "cpu"

    class _Resolver:
        def __init__(self, tier, platform):
            pass

        def resolve(self, stage):
            captured["stage"] = stage
            return _Resolved()

    monkeypatch.setattr(
        "hydra_suite.runtime.resolver.RuntimeResolver", _Resolver
    )
    self_ns = SimpleNamespace(
        _selected_tier=lambda: "cpu",
        _pred_backend=lambda: "vitpose",
    )
    flavor = mw.MainWindow._pred_runtime_flavor(self_ns)
    assert captured["stage"] == "vitpose_pose"
    assert flavor == "cpu"


def test_combo_and_widget_present_in_source():
    import hydra_suite.posekit.gui.main_window as mw

    with open(mw.__file__, "r", encoding="utf-8") as fh:
        text = fh.read()
    assert '"ViTPose"' in text  # combo item
    assert "vitpose_pred_widget" in text  # dedicated settings widget
    assert "pred_vitpose_edit" in text  # checkpoint line edit
```

- [ ] **Step 2: Run test to verify it fails**

Run: `conda run -n hydra-mps python -m pytest tests/test_vitpose_posekit_gui_wiring.py -v`
Expected: FAIL — `_pred_backend` never returns `"vitpose"`; `_pred_runtime_flavor` resolves `"yolo_pose"`; the source lacks `"ViTPose"`/`vitpose_pred_widget`/`pred_vitpose_edit`.

- [ ] **Step 3a: Add the combo item** (`main_window.py:403`):

```python
        self.combo_pred_backend.addItems(["YOLO", "SLEAP", "ViTPose"])
```

- [ ] **Step 3b: Build the ViTPose settings widget** — insert immediately after the YOLO widget block (after `model_layout.addWidget(self.yolo_pred_widget)` at `main_window.py:472`), mirroring the YOLO widget but Browse-only:

```python
        # ViTPose settings (fine-tuned checkpoint .pt)
        self.vitpose_pred_widget = QWidget()
        vitpose_layout = QVBoxLayout(self.vitpose_pred_widget)
        vitpose_layout.setContentsMargins(0, 0, 0, 0)
        vitpose_layout.addWidget(QLabel("ViTPose checkpoint (.pt)"))
        pred_vitpose_row = QHBoxLayout()
        self.pred_vitpose_edit = QLineEdit("")
        self.pred_vitpose_edit.setPlaceholderText("Select fine-tuned checkpoint (.pt)")
        self.btn_pred_vitpose = QPushButton("Browse…")
        pred_vitpose_row.addWidget(self.pred_vitpose_edit, 1)
        pred_vitpose_row.addWidget(self.btn_pred_vitpose)
        vitpose_layout.addLayout(pred_vitpose_row)
        model_layout.addWidget(self.vitpose_pred_widget)
```

- [ ] **Step 3c: Connect the Browse button** — in the signal-wiring section near `main_window.py:664` (where `combo_pred_backend.currentTextChanged` is connected), add:

```python
        self.btn_pred_vitpose.clicked.connect(self._browse_pred_vitpose)
```

- [ ] **Step 3d: Extend `_pred_backend`** (`main_window.py:4205-4210`):

```python
    def _pred_backend(self) -> str:
        try:
            txt = self.combo_pred_backend.currentText().strip().lower()
            if txt.startswith("sleap"):
                return "sleap"
            if txt.startswith("vitpose"):
                return "vitpose"
            return "yolo"
        except Exception:
            return "yolo"
```

- [ ] **Step 3e: Three-way widget toggle in `_update_pred_backend_ui`** (`main_window.py:4312-4320`). Replace the yolo/sleap visibility lines:

```python
        backend = self._pred_backend()
        is_sleap = backend == "sleap"
        is_vitpose = backend == "vitpose"
        if hasattr(self, "yolo_pred_widget"):
            self.yolo_pred_widget.setVisible(backend == "yolo")
        if hasattr(self, "vitpose_pred_widget"):
            self.vitpose_pred_widget.setVisible(is_vitpose)
        if hasattr(self, "sleap_pred_widget"):
            self.sleap_pred_widget.setVisible(is_sleap)
```

(The rest of the method — the SLEAP-service start/stop/shutdown logic keyed on `is_sleap` — is unchanged; vitpose is "not sleap", so it correctly shuts the SLEAP service down.)

- [ ] **Step 3f: Add the ViTPose model-picker getters + Browse** — add these methods next to `_browse_pred_weights`/`_get_pred_weights_or_prompt` (near `main_window.py:4735`):

```python
    def _browse_pred_vitpose(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select ViTPose checkpoint", "", "*.pt"
        )
        if path:
            self.pred_vitpose_edit.setText(path)

    def _get_pred_vitpose_or_prompt(self) -> Optional[Path]:
        txt = self.pred_vitpose_edit.text().strip()
        if txt:
            p = Path(txt).expanduser().resolve()
            if p.exists() and p.is_file() and p.suffix == ".pt":
                return p
            QMessageBox.warning(
                self, "Invalid checkpoint", "ViTPose checkpoint not found."
            )
            return None
        QMessageBox.warning(
            self, "No checkpoint", "Select a ViTPose checkpoint (.pt)."
        )
        return None

    def _get_pred_vitpose_silent(self) -> Optional[Path]:
        txt = self.pred_vitpose_edit.text().strip()
        if txt:
            p = Path(txt).expanduser().resolve()
            if p.exists() and p.is_file() and p.suffix == ".pt":
                return p
        return None
```

- [ ] **Step 3g: Dispatch the model picker to vitpose** — `_get_pred_model_or_prompt` (`:4629`) and `_get_pred_model_silent` (`:4636`):

```python
    def _get_pred_model_or_prompt(self) -> Optional[Path]:
        backend = self._pred_backend()
        if backend == "sleap":
            return self._get_sleap_model_or_prompt()
        if backend == "vitpose":
            return self._get_pred_vitpose_or_prompt()
        return self._get_pred_weights_or_prompt()

    def _get_pred_model_silent(self) -> Optional[Path]:
        backend = self._pred_backend()
        if backend == "sleap":
            return self._get_sleap_model_silent()
        if backend == "vitpose":
            return self._get_pred_vitpose_silent()
        return self._get_pred_weights_silent()
```

- [ ] **Step 3h: Select the `vitpose_pose` resolver stage** in `_pred_runtime_flavor` (`main_window.py:4654`). Replace the `stage = ...` line:

```python
        _b = self._pred_backend()
        if _b == "sleap":
            stage = "sleap_pose"
        elif _b == "vitpose":
            stage = "vitpose_pose"
        else:
            stage = "yolo_pose"
        resolved = RuntimeResolver(tier, detect_platform()).resolve(stage)
```

- [ ] **Step 3i: Add the ViTPose widgets to the bulk-lock list** (`main_window.py:4291-4316`, the `widget_names` list). After `"btn_pred_weights_latest",` add:

```python
            "pred_vitpose_edit",
            "btn_pred_vitpose",
```

- [ ] **Step 3j: Persist the ViTPose checkpoint** — settings save (`main_window.py:1391`, in the dict) add:

```python
            "pred_vitpose": self.pred_vitpose_edit.text().strip(),
```

and in `_apply_pred_settings` (`main_window.py:1457`) after the `pred_weights` restore:

```python
        if "pred_vitpose" in settings:
            self.pred_vitpose_edit.setText(str(settings["pred_vitpose"]))
```

- [ ] **Step 3k: Pass the vitpose batch to both workers.** In `predict_current_frame` the `PosePredictWorker(...)` call (`main_window.py:4928-4948`) — after `sleap_max_instances=1,` add:

```python
            sleap_max_instances=1,
            vitpose_batch=int(self.spin_pred_batch.value()),
```

In `predict_dataset` the `BulkPosePredictWorker(...)` call (`main_window.py:5240-5258`) — after `sleap_max_instances=1,` add:

```python
                sleap_max_instances=1,
                vitpose_batch=int(pred_batch),
```

- [ ] **Step 4: Run test to verify it passes**

Run: `conda run -n hydra-mps python -m pytest tests/test_vitpose_posekit_gui_wiring.py -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Format + commit**

```bash
conda run -n hydra-mps black src/hydra_suite/posekit/gui/main_window.py tests/test_vitpose_posekit_gui_wiring.py
conda run -n hydra-mps isort src/hydra_suite/posekit/gui/main_window.py tests/test_vitpose_posekit_gui_wiring.py
git add src/hydra_suite/posekit/gui/main_window.py tests/test_vitpose_posekit_gui_wiring.py
git commit -m "feat(vitpose): posekit inference GUI selector + checkpoint widget + worker wiring"
```

---

## Task 4: Full suite gate + manual end-to-end verification

**Files:** none (verification only).

- [ ] **Step 1: Run the full vitpose + posekit inference test set**

Run:
```bash
conda run -n hydra-mps python -m pytest \
  tests/test_vitpose_posekit_load_backend.py \
  tests/test_vitpose_posekit_workers.py \
  tests/test_vitpose_posekit_gui_wiring.py \
  tests/test_posekit_workers_pose_backend.py \
  tests/test_posekit_main_window.py \
  -v
```
Expected: all PASS (no regressions in the existing posekit worker/main-window tests).

- [ ] **Step 2: Lint gate**

Run: `conda run -n hydra-mps make lint-moderate`
Expected: no new moderate findings in the three modified files. (If `make lint-moderate` is unavailable, run `conda run -n hydra-mps flake8 src/hydra_suite/core/inference/api.py src/hydra_suite/posekit/gui/workers.py src/hydra_suite/posekit/gui/main_window.py`.)

- [ ] **Step 3: Manual end-to-end (name the result in the PR, do not skip if a checkpoint is available)**

With a real fine-tuned ViTPose `best.pt`:
1. Launch `conda run -n hydra-mps posekit`, open a project with unlabeled frames.
2. Inference panel → backend combo → **ViTPose**; the checkpoint row appears (YOLO/SLEAP rows hidden).
3. Browse to `best.pt`; pick a tier (CPU or GPU).
4. **Predict Keypoints** on the current frame → a keypoint overlay appears.
5. **Apply Predictions** → keypoints populate the annotation.
6. **Predict Dataset** (small scope) → progresses and caches predictions.

If no checkpoint is available in the environment, state that explicitly in the PR — the reviewer decides whether to require it before merge. The automated gate (Steps 1–2) stands on its own for the code paths.

- [ ] **Step 4: Pre-PR housekeeping**

```bash
conda run -n hydra-mps make docs-check
```
Expected: docs build clean (no terminology violations introduced).

---

## Self-review notes

- **Spec coverage:** load_pose_backend branch → Task 1; worker plumbing + fallback policy → Task 2; GUI combo/widget/`_pred_backend`/toggle/model-picker/runtime-stage/settings/worker-construction → Task 3; testing strategy → Tasks 1–4; runtime-agnostic (no accelerated PoseKit code) → satisfied by construction (nothing added). Error handling (checkpoint failure re-raises, missing-artifact fallback) → Task 2 re-raise + existing resolver behavior.
- **Non-goals honored:** no `gui/models.py` change (no "Use Latest"); no changes to caching/overlay/apply paths; no trackerkit changes; no accelerated-specific code.
- **Coordination:** this supersedes the tracking plan's Task 7 posekit portion (which was skipped); Task 7's trackerkit portion is already done and untouched here.
