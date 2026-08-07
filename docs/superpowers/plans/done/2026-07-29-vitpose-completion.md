# ViTPose Integration Completion Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close every remaining gap that keeps ViTPose from being a first-class pose backend at full parity with YOLO-pose and SLEAP across PoseKit (inference + training + eval) and TrackerKit (tracking), plus docs and dependency hygiene.

**Architecture:** The ViTPose ML core (native inference, fine-tune training, ONNX/TensorRT/CoreML export, config round-trip, resolver stage) is already complete and tested. This plan fixes the *connective tissue*: the broken train→infer handoff, an undeclared dependency, a runtime flavor→tier collapse that silently defeats gpu_fast, cross-session persistence, post-training UI (Use-Latest / loss plot / evaluation), and total documentation absence. Registration stays at intra-pose parity (registry unification is a deferred separate spec: `docs/superpowers/specs/2026-07-29-model-registry-unification-design.md`).

**Tech Stack:** Python 3, PyQt, PyTorch, timm, huggingface_hub, pytest. Conda env `hydra-mps` (Apple Silicon).

## Global Constraints

- Environment: activate conda env `hydra-mps` before running anything (`conda activate hydra-mps`) on the MPS box. Pose/SLEAP tests require conda on PATH.
- Cross-platform verification: the MPS box (this machine, `hydra-mps`) is the primary dev/test box; CUDA verification runs on `rutalab@mehek.taild08eb9.ts.net` with `hydra-cuda`. gpu_fast / accelerated-runtime changes (Task 1, Task 9 auto_export) should be sanity-checked on CUDA before final merge, but per-task TDD runs on MPS.
- Line length: 88.
- Do NOT run `make format`. Match surrounding style manually.
- Commit as the configured git user. Do NOT add a `Co-Authored-By: Claude` trailer.
- Work from the worktree `.worktrees/vitpose-completion` (branch `feature/vitpose-completion`).
- Run tests from the worktree with `PYTHONPATH=$PWD/src` so tests import the worktree tree, not the installed package. Base suite has ~24 pre-existing unrelated failures — gate on the delta (your new/touched tests), not absolute green.
- Registration scope is decided: **pose parity only.** Do NOT add `model_registry.json` entries for pose backends in this plan. Task 10 only *verifies* parity by test.
- ViTPose is copied into `models/pose/ViTPose/` on import exactly like YOLO-pose/SLEAP — do not change that flow.
- Never fabricate model download URLs or SHA256 hashes. The catalog stays COCO-B-only; Task 8 adds a guard, not new checkpoints.
- Follow existing patterns in each file. These are large GUI files — make focused edits, do not restructure.

---

### Task 1: Fix gpu_fast flavor→tier collapse (`migrate_runtime_to_tier`)

The PoseKit GUI emits accelerated flavor strings `"tensorrt_cuda"` (CUDA gpu_fast) and `"coreml"` (Apple gpu_fast); TrackerKit's crops_worker emits `"coreml"` on Apple gpu_fast. `migrate_runtime_to_tier` recognizes only `{"onnx_cpu","onnx_cuda","onnx_coreml","tensorrt"}` for fast and `{"cuda","mps"}` for gpu, so `"tensorrt_cuda"` and `"coreml"` both fall through to `"cpu"` — silently running torch-CPU instead of the accelerated engine for ALL pose backends. This is the single highest-value correctness fix (unblocks the accelerated ViTPose runtimes that are already built).

**Files:**
- Modify: `src/hydra_suite/core/inference/config.py:16-32` (`migrate_runtime_to_tier`)
- Test: `tests/test_migrate_runtime_to_tier.py` (create)

**Interfaces:**
- Consumes: nothing.
- Produces: `migrate_runtime_to_tier(runtimes: set[str]) -> RuntimeTier` now maps `"tensorrt_cuda"` and `"coreml"` (and `"onnx_mps"`) to `"gpu_fast"`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_migrate_runtime_to_tier.py
from hydra_suite.core.inference.config import migrate_runtime_to_tier


def test_tensorrt_cuda_flavor_maps_to_gpu_fast():
    assert migrate_runtime_to_tier({"tensorrt_cuda"}) == "gpu_fast"


def test_coreml_flavor_maps_to_gpu_fast():
    assert migrate_runtime_to_tier({"coreml"}) == "gpu_fast"


def test_onnx_mps_flavor_maps_to_gpu_fast():
    assert migrate_runtime_to_tier({"onnx_mps"}) == "gpu_fast"


def test_plain_gpu_and_cpu_flavors_unchanged():
    assert migrate_runtime_to_tier({"cuda"}) == "gpu"
    assert migrate_runtime_to_tier({"mps"}) == "gpu"
    assert migrate_runtime_to_tier({"cpu"}) == "cpu"
    assert migrate_runtime_to_tier(set()) == "gpu"
    # mixed set takes highest tier
    assert migrate_runtime_to_tier({"cpu", "coreml"}) == "gpu_fast"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=$PWD/src python -m pytest tests/test_migrate_runtime_to_tier.py -v`
Expected: `test_tensorrt_cuda_flavor_maps_to_gpu_fast`, `test_coreml_...`, `test_onnx_mps_...` FAIL (return `"cpu"`); the "unchanged" test passes.

- [ ] **Step 3: Implement the fix**

Extend the `fast` set in `migrate_runtime_to_tier` (`config.py:26`) to include the GUI-emitted accelerated flavors:

```python
    # onnx_* / *_cuda / coreml entries cover both legacy-config migration and the
    # runtime-flavor strings the GUIs emit for gpu_fast (tensorrt_cuda, coreml, onnx_mps).
    fast = {
        "onnx_cpu",
        "onnx_cuda",
        "onnx_coreml",
        "onnx_mps",
        "tensorrt",
        "tensorrt_cuda",
        "coreml",
    }
    gpu = {"cuda", "mps"}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=$PWD/src python -m pytest tests/test_migrate_runtime_to_tier.py -v`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add tests/test_migrate_runtime_to_tier.py src/hydra_suite/core/inference/config.py
git commit -m "fix(runtime): map tensorrt_cuda/coreml/onnx_mps flavors to gpu_fast tier"
```

---

### Task 2: Declare `huggingface_hub` as an explicit dependency

`src/hydra_suite/posekit/core/vitpose_checkpoints.py:8` imports `huggingface_hub` at module top level, but it is not declared in `pyproject.toml` — it resolves only transitively via timm/ultralytics today. A clean install that resolves without it crashes PoseKit ViTPose checkpoint download on import.

**Files:**
- Modify: `pyproject.toml` (core dependencies list, near `timm` at line ~38)
- Test: `tests/test_vitpose_dependency_declared.py` (create)

**Interfaces:**
- Consumes: nothing.
- Produces: nothing (packaging metadata only).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_vitpose_dependency_declared.py
import tomllib
from pathlib import Path


def test_huggingface_hub_is_declared():
    root = Path(__file__).resolve().parents[1]
    data = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
    deps = data["project"]["dependencies"]
    names = [d.lower().replace("_", "-") for d in deps]
    assert any(n.startswith("huggingface-hub") for n in names), (
        "huggingface_hub is imported at module top level in "
        "posekit/core/vitpose_checkpoints.py but not declared in pyproject.toml"
    )
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=$PWD/src python -m pytest tests/test_vitpose_dependency_declared.py -v`
Expected: FAIL (assertion — huggingface-hub not in dependencies).

- [ ] **Step 3: Add the dependency**

In `pyproject.toml`, add to the `[project].dependencies` array, next to the existing `timm` entry, matching the surrounding version-pin style (unpinned unless the file pins others):

```toml
    "huggingface_hub",
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=$PWD/src python -m pytest tests/test_vitpose_dependency_declared.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add tests/test_vitpose_dependency_declared.py pyproject.toml
git commit -m "fix(deps): declare huggingface_hub (top-level import in vitpose_checkpoints)"
```

---

### Task 3: Fix the ViTPose train→infer handoff (BLOCKER)

`ViTPoseTrainingWorker` emits `{"run_dir": ..., "best": str(run_dir/"best.pt")}` (`training.py:399-404`), but the shared `_on_finished` handler reads `info.get("weights")` and probes `run_dir/"weights"/"best.pt"` / `.../last.pt` (`training.py:1539, 1560-1561`) — a `"weights"` key and a `weights/` subdir that the ViTPose trainer never produces (it writes `best.pt`/`last.pt` directly in `run_dir`). Result: `project.latest_pose_weights` is never set for ViTPose, the Open-Eval button never enables, and the freshly trained checkpoint is orphaned from inference. YOLO emits `{"weights": run_dir/"weights"/best.pt}` matching the handler; only ViTPose's output is dropped.

**Files:**
- Modify: `src/hydra_suite/posekit/gui/dialogs/training.py` (`_on_finished`, ~1530-1575)
- Test: `tests/test_vitpose_training_finished_handoff.py` (create)

**Interfaces:**
- Consumes: `ViTPoseTrainingWorker.finished` payload `{"run_dir": str, "best": str}` (`training.py:399-404`).
- Produces: after a ViTPose run finishes, `_on_finished` sets the latest-weights path from `info["best"]` (falling back to `run_dir/"best.pt"`), enabling Open-Eval and populating `project.latest_pose_weights`.

- [ ] **Step 1: Read the handler to confirm exact structure**

Read `training.py:1530-1595` (the `_on_finished` method) so your edit matches the real control flow (the YOLO `weights` path, the `run_dir/"weights"/best.pt` probe, the `_set_latest_weights` call, the eval-button enable). Do not guess — the diff must slot into the existing branches.

- [ ] **Step 2: Write the failing test**

The handler touches Qt widgets; test the *payload-resolution* logic without a live dialog by extracting a pure helper. Add this helper to `training.py` (module-level, near `_on_finished`):

```python
def resolve_finished_weights(info: dict) -> str:
    """Return the best-weights path from a training worker's finished payload.

    Supports both the YOLO/ultralytics shape ({"weights": ".../weights/best.pt"})
    and the ViTPose shape ({"run_dir": ..., "best": ".../best.pt"}). Returns "" if
    no existing weights file can be resolved.
    """
    from pathlib import Path

    weights = str(info.get("weights") or "").strip()
    if weights and Path(weights).exists():
        return weights
    best = str(info.get("best") or "").strip()
    if best and Path(best).exists():
        return best
    run_dir = info.get("run_dir")
    if run_dir:
        rd = Path(run_dir)
        for cand in (rd / "weights" / "best.pt", rd / "best.pt",
                     rd / "weights" / "last.pt", rd / "last.pt"):
            if cand.exists():
                return str(cand)
    return ""
```

Test:

```python
# tests/test_vitpose_training_finished_handoff.py
from pathlib import Path
from hydra_suite.posekit.gui.dialogs.training import resolve_finished_weights


def test_vitpose_payload_resolves_best_at_run_dir_root(tmp_path):
    best = tmp_path / "best.pt"
    best.write_bytes(b"x")
    info = {"run_dir": str(tmp_path), "best": str(best)}
    assert resolve_finished_weights(info) == str(best)


def test_vitpose_payload_falls_back_to_run_dir_best(tmp_path):
    best = tmp_path / "best.pt"
    best.write_bytes(b"x")
    info = {"run_dir": str(tmp_path)}  # no explicit "best"/"weights"
    assert resolve_finished_weights(info) == str(best)


def test_yolo_payload_still_resolves_weights_subdir(tmp_path):
    wdir = tmp_path / "weights"
    wdir.mkdir()
    best = wdir / "best.pt"
    best.write_bytes(b"x")
    info = {"weights": str(best), "run_dir": str(tmp_path)}
    assert resolve_finished_weights(info) == str(best)


def test_missing_weights_returns_empty(tmp_path):
    assert resolve_finished_weights({"run_dir": str(tmp_path)}) == ""
```

- [ ] **Step 3: Run test to verify it fails**

Run: `PYTHONPATH=$PWD/src python -m pytest tests/test_vitpose_training_finished_handoff.py -v`
Expected: FAIL with `ImportError`/`AttributeError` (`resolve_finished_weights` not defined).

- [ ] **Step 4: Implement**

Add the `resolve_finished_weights` helper (above) at module level, then rewire `_on_finished` to use it instead of the inline `weights = info.get("weights")` + `run_dir/"weights"/best.pt` probe. Concretely, replace the resolution block (~`training.py:1539` and the `if not weights:` fallback block at ~1558-1565) with:

```python
        weights = resolve_finished_weights(info)
        if weights:
            self._last_weights = weights
```

Keep the downstream behavior identical: the existing `if weights: ... self.btn_open_eval.setEnabled(True); self._set_latest_weights(weights)` and the `elif self._last_weights:` branch continue to work because `_last_weights`/`weights` are now correctly populated for ViTPose. Do not remove the `latest_pose_weights`-exists fallback branch.

- [ ] **Step 5: Run test to verify it passes**

Run: `PYTHONPATH=$PWD/src python -m pytest tests/test_vitpose_training_finished_handoff.py -v`
Expected: all PASS.

- [ ] **Step 6: Commit**

```bash
git add tests/test_vitpose_training_finished_handoff.py src/hydra_suite/posekit/gui/dialogs/training.py
git commit -m "fix(posekit): register ViTPose trained weights on finish (train->infer handoff)"
```

---

### Task 4: "Use Latest" for the PoseKit ViTPose predict checkpoint

The ViTPose predict field (`pred_vitpose_edit`, `main_window.py:480-484`) is Browse-only. YOLO has `btn_pred_weights_latest` (`:467, 4777`) and SLEAP has `btn_sleap_model_latest` (`:511, 4505`), both pulling from `project.latest_pose_weights`. With Task 3 fixed, a trained ViTPose `best.pt` now populates that field — add the one-click reuse button so training→inference is closed in the UI.

**Files:**
- Modify: `src/hydra_suite/posekit/gui/main_window.py` (ViTPose predict widget row ~480-484; add handler mirroring `_use_latest_pred_weights`/the YOLO latest handler ~4777)
- Test: `tests/test_vitpose_use_latest.py` (create)

**Interfaces:**
- Consumes: `project.latest_pose_weights` (set by Task 3).
- Produces: `_use_latest_pred_vitpose()` sets `pred_vitpose_edit` text to `project.latest_pose_weights` when it exists and is a `.pt` file.

- [ ] **Step 1: Read the YOLO "Use Latest" implementation**

Read the YOLO handler around `main_window.py:4777` (the method connected to `btn_pred_weights_latest`) to mirror its guard logic (existence check, `.pt` check, user-facing warning when absent).

- [ ] **Step 2: Write the failing test**

Test the pure guard as a static method so no live widget is needed:

```python
# tests/test_vitpose_use_latest.py
from hydra_suite.posekit.gui.main_window import MainWindow


def test_latest_vitpose_path_accepts_existing_pt(tmp_path):
    p = tmp_path / "best.pt"
    p.write_bytes(b"x")
    assert MainWindow._latest_vitpose_candidate(str(p)) == str(p)


def test_latest_vitpose_path_rejects_missing(tmp_path):
    assert MainWindow._latest_vitpose_candidate(str(tmp_path / "nope.pt")) == ""


def test_latest_vitpose_path_rejects_non_pt(tmp_path):
    p = tmp_path / "model.onnx"
    p.write_bytes(b"x")
    assert MainWindow._latest_vitpose_candidate(str(p)) == ""
```

- [ ] **Step 3: Run test to verify it fails**

Run: `PYTHONPATH=$PWD/src python -m pytest tests/test_vitpose_use_latest.py -v`
Expected: FAIL (`_latest_vitpose_candidate` not defined).

- [ ] **Step 4: Implement**

Add the static guard and the button + handler:

```python
    @staticmethod
    def _latest_vitpose_candidate(path: str) -> str:
        from pathlib import Path

        p = str(path or "").strip()
        if not p:
            return ""
        fp = Path(p)
        if fp.is_file() and fp.suffix == ".pt":
            return str(fp)
        return ""

    def _use_latest_pred_vitpose(self) -> None:
        latest = ""
        if hasattr(self.project, "latest_pose_weights"):
            latest = self._latest_vitpose_candidate(
                str(self.project.latest_pose_weights or "")
            )
        if latest:
            self.pred_vitpose_edit.setText(latest)
        else:
            self._warn("No latest ViTPose weights available. Train a model first.")
```

Add a "Use Latest" `QPushButton` in the ViTPose predict row (~`main_window.py:480-484`), connected to `_use_latest_pred_vitpose`, matching the YOLO row's button layout. Use the project's existing warning helper (match the name used by the YOLO handler, e.g. `self._warn(...)` or the actual helper).

- [ ] **Step 5: Run test to verify it passes**

Run: `PYTHONPATH=$PWD/src python -m pytest tests/test_vitpose_use_latest.py -v`
Expected: all PASS.

- [ ] **Step 6: Commit**

```bash
git add tests/test_vitpose_use_latest.py src/hydra_suite/posekit/gui/main_window.py
git commit -m "feat(posekit): Use-Latest button for ViTPose predict checkpoint"
```

---

### Task 5: Render the ViTPose training loss plot

`_update_loss_plot` (the real method is at `training.py:1657-1770`) reads only Ultralytics `results.csv`. The ViTPose trainer writes `metrics.csv` with header `epoch,train_loss,val_loss,pck@0.05,pck@0.1` (`training/train.py:89-93`). No `results.csv` is produced, so the ViTPose loss curve stays blank while progress works.

**CORRECTED (plan defect found during execution):** `_update_loss_plot` is NOT a two-series `(epochs, train, val)` renderer. It discovers *all* `*loss*` columns in `results.csv`, builds per-component `train_vals: Dict[str, list[float]]` / `val_vals: Dict[str, list[float]]`, creates one `QCheckBox` per component (`self.loss_component_checks`), and renders via `make_loss_plot_image(train_vals, val_vals, ...)` (`dialogs/utils.py:236`) — x-axis is row index, there is no separate epochs list. The fix must PRESERVE this multi-series/checkbox UI. So the extracted helper returns the SAME dict-keyed shape, and the ViTPose branch contributes a single component named `"loss"` that flows through the existing checkbox machinery as one curve.

**Files:**
- Modify: `src/hydra_suite/posekit/gui/dialogs/training.py` (`_update_loss_plot`, ~1657-1770; extract a pure parser)
- Test: `tests/test_vitpose_loss_plot_parse.py` (create)

**Interfaces:**
- Consumes: a run dir containing either `results.csv` (YOLO, multi-component) or `metrics.csv` (ViTPose, single `train_loss`/`val_loss`).
- Produces: `parse_loss_components(run_dir) -> tuple[dict[str, list[float]], dict[str, list[float]]]` returning `(train_vals, val_vals)` keyed by loss-component name, from whichever file exists; `({}, {})` when neither exists. For `metrics.csv` the single component is named `"loss"`.

- [ ] **Step 1: Read the real renderer**

Read `_update_loss_plot` (`training.py:1657-1770`) and `make_loss_plot_image` (`dialogs/utils.py:236`) to capture the EXACT existing `results.csv` column-discovery → `train_vals`/`val_vals` dict logic and the component-key derivation (how `"train/box_loss"` becomes the component key). You will transplant this verbatim into the helper's `results.csv` branch — do not reimplement it. Note the real key format so your regression test matches it.

- [ ] **Step 2: Write the failing test**

```python
# tests/test_vitpose_loss_plot_parse.py
from hydra_suite.posekit.gui.dialogs.training import parse_loss_components


def test_parses_vitpose_metrics_csv(tmp_path):
    (tmp_path / "metrics.csv").write_text(
        "epoch,train_loss,val_loss,pck@0.05,pck@0.1\n"
        "0,1.5,1.8,0.10,0.20\n"
        "1,0.9,1.1,0.30,0.45\n",
        encoding="utf-8",
    )
    train_vals, val_vals = parse_loss_components(tmp_path)
    assert train_vals == {"loss": [1.5, 0.9]}
    assert val_vals == {"loss": [1.8, 1.1]}


def test_returns_empty_when_no_csv(tmp_path):
    assert parse_loss_components(tmp_path) == ({}, {})


def test_results_csv_preserved_as_multi_component(tmp_path):
    # Mirror the REAL column/key format observed in Step 1. Adjust the header
    # and the expected component key(s) to match the transplanted logic exactly.
    (tmp_path / "results.csv").write_text(
        "epoch,train/box_loss,val/box_loss\n"
        "0,1.0,1.2\n"
        "1,0.5,0.7\n",
        encoding="utf-8",
    )
    train_vals, val_vals = parse_loss_components(tmp_path)
    # component key derived by the existing logic (e.g. "box_loss")
    assert list(train_vals.values())[0] == [1.0, 0.5]
    assert list(val_vals.values())[0] == [1.2, 0.7]
    assert train_vals.keys() == val_vals.keys()
```

- [ ] **Step 3: Run test to verify it fails**

Run: `PYTHONPATH=$PWD/src python -m pytest tests/test_vitpose_loss_plot_parse.py -v`
Expected: FAIL (`parse_loss_components` not defined).

- [ ] **Step 4: Implement**

Extract `parse_loss_components(run_dir)` returning `(train_vals, val_vals)` dicts:

```python
def parse_loss_components(run_dir):
    import csv
    from pathlib import Path

    rd = Path(run_dir)
    results = rd / "results.csv"
    metrics = rd / "metrics.csv"
    if results.exists():
        # TRANSPLANT the existing _update_loss_plot results.csv logic verbatim:
        # discover *loss* columns, build train_vals/val_vals dicts keyed by
        # component name. Return them.
        ...
        return train_vals, val_vals
    if metrics.exists():
        tr, vl = [], []
        with metrics.open(encoding="utf-8", newline="") as fh:
            for row in csv.DictReader(fh):
                tr.append(float(row["train_loss"]))
                vl.append(float(row["val_loss"]))
        return {"loss": tr}, {"loss": vl}
    return {}, {}
```

Then rewire `_update_loss_plot` to call `parse_loss_components(run_dir)` for its data, keeping the checkbox discovery (`keys = union of the returned dict keys`), the per-component `QCheckBox` creation, the checked-filter, and the `make_loss_plot_image(...)` call UNCHANGED. For ViTPose, `keys == {"loss"}` → one checkbox, one curve. Do not alter the plotting/rendering code or `make_loss_plot_image`'s contract.

- [ ] **Step 5: Run test to verify it passes**

Run: `PYTHONPATH=$PWD/src python -m pytest tests/test_vitpose_loss_plot_parse.py -v`
Expected: all PASS.

- [ ] **Step 6: Commit**

```bash
git add tests/test_vitpose_loss_plot_parse.py src/hydra_suite/posekit/gui/dialogs/training.py
git commit -m "feat(posekit): plot ViTPose training loss from metrics.csv"
```

---

### Task 6: ViTPose backend in the PoseKit evaluation dashboard

The eval dialog (`dialogs/evaluation.py`) predicts through `PoseInferenceService.predict` (`integrations/sleap/service.py:216`), which has a `sleap` branch and an `else` YOLO/ultralytics-subprocess branch (`:245, 298`) — no ViTPose branch. The backend combo lists only `["YOLO","SLEAP"]` (`:494`), the worker defaults `backend="yolo"` (`:75`), and `_open_eval` passes no backend (`training.py:1623-1628`). So a ViTPose checkpoint is evaluated through ultralytics and fails. Add a ViTPose path that reuses the already-working inference (`load_pose_backend`) and surface it in the dialog.

**Files:**
- Modify: `src/hydra_suite/integrations/sleap/service.py` (`predict`, add `backend == "vitpose"` branch ~after 245)
- Modify: `src/hydra_suite/posekit/gui/dialogs/evaluation.py` (combo `:494`; worker `backend`/validation `:75,106,139`)
- Modify: `src/hydra_suite/posekit/gui/dialogs/training.py` (`_open_eval` pass `backend="vitpose"` when the trained model was ViTPose)
- Test: `tests/test_vitpose_eval_predict.py` (create)

**Interfaces:**
- Consumes: `load_pose_backend` (`core/inference/api.py:41`) and the `PoseInferenceService` prediction-dict contract `{str(image_path): [(x, y, conf), ...]}` used by `_score_one_frame` (`evaluation.py:240`).
- Produces: `PoseInferenceService.predict(..., backend="vitpose")` returns the same `(preds_dict, err)` contract as the yolo/sleap branches, running ViTPose via `load_pose_backend`.

- [ ] **Step 1: Read the contract**

Read `service.py:216-314` (both existing branches) and `evaluation.py:127-149, 234-253` to confirm the exact preds-dict shape (`{str(path): [(x,y,conf)*K]}`), the `.pt`/existence guards, and how `merge_cache` is called. Read `_build_pose_backend`/`load_pose_backend` (`posekit/gui/workers.py`, `core/inference/api.py:41-135`) to reuse the exact ViTPose loading call (checkpoint path, `vitpose_batch`, runtime flavor→`compute_runtime`).

- [ ] **Step 2: Write the failing test**

Drive the new branch with a fake backend so no real checkpoint/GPU is needed — verify the ViTPose branch loads via `load_pose_backend` and returns the correct dict shape. Use monkeypatch:

```python
# tests/test_vitpose_eval_predict.py
import numpy as np
from pathlib import Path
from hydra_suite.integrations.sleap.service import PoseInferenceService


class _FakeBackend:
    def predict_batch(self, crops):
        # one instance, 2 keypoints, per image
        return [np.array([[1.0, 2.0, 0.9], [3.0, 4.0, 0.8]], dtype=np.float32)
                for _ in crops]


def test_vitpose_branch_returns_keypoint_dict(tmp_path, monkeypatch):
    ckpt = tmp_path / "best.pt"
    ckpt.write_bytes(b"x")
    img = tmp_path / "img0.png"
    img.write_bytes(b"x")

    monkeypatch.setattr(
        "hydra_suite.integrations.sleap.service.load_pose_backend",
        lambda **kw: _FakeBackend(),
        raising=False,
    )
    # stub image loading used by the vitpose branch to avoid decoding a fake PNG
    monkeypatch.setattr(
        "hydra_suite.integrations.sleap.service._load_images_for_vitpose",
        lambda paths: [np.zeros((8, 8, 3), dtype=np.uint8) for _ in paths],
        raising=False,
    )

    svc = PoseInferenceService(tmp_path, ["a", "b"])
    preds, err = svc.predict(
        model_path=ckpt, image_paths=[img], device="cpu", imgsz=256,
        conf=0.0, batch=1, backend="vitpose", cache_predictions=False,
    )
    assert err == ""
    assert list(preds.keys()) == [str(img)]
    assert len(preds[str(img)]) == 2  # K=2 keypoints
    assert preds[str(img)][0] == (1.0, 2.0, 0.9)
```

- [ ] **Step 3: Run test to verify it fails**

Run: `PYTHONPATH=$PWD/src python -m pytest tests/test_vitpose_eval_predict.py -v`
Expected: FAIL (no `vitpose` branch → falls into the yolo `.pt` subprocess path or errors).

- [ ] **Step 4: Implement the service branch**

In `PoseInferenceService.predict`, add after the `sleap` block (before the yolo fallthrough at `:289`):

```python
        if backend == "vitpose":
            if not model_path.exists() or not model_path.is_file():
                return None, f"Weights not found: {model_path}"
            if model_path.suffix != ".pt":
                return None, f"Invalid weights file: {model_path}"
            be = load_pose_backend(
                family="vitpose",
                model_path=str(model_path),
                compute_runtime=device,
                vitpose_batch=int(batch),
            )
            images = _load_images_for_vitpose([Path(p) for p in image_paths])
            preds: Dict[str, List[Tuple[float, float, float]]] = {}
            for path, kpts in zip(image_paths, be.predict_batch(images)):
                arr = np.asarray(kpts, dtype=np.float32)
                preds[str(path)] = [
                    (float(r[0]), float(r[1]), float(r[2])) for r in arr
                ]
                if progress_cb:
                    progress_cb(len(preds), len(image_paths))
            if cache_predictions:
                self.merge_cache(model_path, preds, backend=backend)
            return preds, ""
```

Add the module-level helper `_load_images_for_vitpose(paths)` (decode each image to an RGB uint8 ndarray via PIL/numpy, matching what `ViTPoseBackend.predict_batch` expects — confirm the expected input format from `core/individual/pose/backends/vitpose.py` and `_build_pose_backend`). Import `load_pose_backend` at the top of `service.py` (lazy import inside the branch if a top-level import risks a cycle). Confirm the actual `load_pose_backend` signature/kwargs against `core/inference/api.py:41` and adjust the call to match exactly (the audit shows `family`, `model_path`, `compute_runtime`, `vitpose_batch`).

- [ ] **Step 5: Wire the dialog**

- `evaluation.py:494`: add `"ViTPose"` to `backend_combo` items.
- `EvalWorker._validate_inputs` (`:106`): treat `vitpose` like the yolo `.pt` branch (file exists, `.pt`) — the `if self.backend != "sleap":` ultralytics-import guard must NOT run for vitpose (ViTPose does not need ultralytics). Change that guard to `if self.backend == "yolo":`.
- `training.py:1623-1628` `_open_eval`: pass `backend="vitpose"` when the just-trained model’s backend was ViTPose (thread the training backend through, mirroring how the trained-backend is already known in the dialog).

- [ ] **Step 6: Run tests**

Run: `PYTHONPATH=$PWD/src python -m pytest tests/test_vitpose_eval_predict.py -v`
Expected: PASS. Also re-run the existing eval tests if any: `PYTHONPATH=$PWD/src python -m pytest tests/ -k eval -v` and confirm no regressions in touched behavior.

- [ ] **Step 7: Commit**

```bash
git add tests/test_vitpose_eval_predict.py src/hydra_suite/integrations/sleap/service.py src/hydra_suite/posekit/gui/dialogs/evaluation.py src/hydra_suite/posekit/gui/dialogs/training.py
git commit -m "feat(posekit): evaluate ViTPose checkpoints in the eval dashboard"
```

---

### Task 7: Persist the TrackerKit ViTPose model selection across sessions

The config save block (`orchestrators/config.py:1834-1842`) writes `pose_yolo_model_dir` and `pose_sleap_model_dir` but no `pose_vitpose_model_dir`; the load block (`:1162-1170`) restores yolo/sleap slots and legacy-fallbacks into them but never the vitpose slot. So a selected ViTPose checkpoint is lost on save/reload. SLEAP and YOLO have full round-trip; ViTPose does not.

**Files:**
- Modify: `src/hydra_suite/trackerkit/gui/orchestrators/config.py` (save ~1834-1842; load ~1162-1170)
- Test: `tests/test_vitpose_trackerkit_persistence.py` (create)

**Interfaces:**
- Consumes: `MainWindow._pose_model_path_for_backend("vitpose")` and `_set_pose_model_path_for_backend(path, backend="vitpose")` (both already handle vitpose — TrackerKit audit confirmed `main_window.py:1085-1102`).
- Produces: config dict gains key `pose_vitpose_model_dir`; on load it is routed to the vitpose slot.

- [ ] **Step 1: Read both blocks**

Read `config.py:1160-1172` and `1828-1842` to match the exact `make_pose_model_path_relative` / `get_cfg` idioms and the legacy-fallback shape used for yolo/sleap.

- [ ] **Step 2: Write the failing test**

Assert the save dict includes the vitpose key and the load routes it. Use a lightweight stub of the MainWindow surface the orchestrator calls (mirror any existing config-orchestrator test in `tests/`; if one exists, follow its harness). Minimal form:

```python
# tests/test_vitpose_trackerkit_persistence.py
import inspect
from hydra_suite.trackerkit.gui.orchestrators import config as cfgmod


def test_save_serializes_vitpose_model_dir():
    src = inspect.getsource(cfgmod)
    assert "pose_vitpose_model_dir" in src, (
        "save/load must persist the vitpose model slot like yolo/sleap"
    )
    # both a write (relative path from the vitpose slot) and a read must exist
    assert src.count("pose_vitpose_model_dir") >= 2
    assert '_set_pose_model_path_for_backend' in src
    assert 'backend="vitpose"' in src or "backend='vitpose'" in src
```

(If the test harness in the repo supports constructing the orchestrator with a fake MainWindow, prefer a real save→load round-trip asserting the vitpose path survives; use the source-level guard only if no such harness exists.)

- [ ] **Step 3: Run test to verify it fails**

Run: `PYTHONPATH=$PWD/src python -m pytest tests/test_vitpose_trackerkit_persistence.py -v`
Expected: FAIL (`pose_vitpose_model_dir` absent).

- [ ] **Step 4: Implement**

Save block (after the `pose_sleap_model_dir` entry ~`config.py:1841`):

```python
                "pose_vitpose_model_dir": make_pose_model_path_relative(
                    self._mw._pose_model_path_for_backend("vitpose")
                ),
```

Load block (after the sleap restore ~`config.py:1170`):

```python
        vitpose_pose_model = str(get_cfg("pose_vitpose_model_dir", default="")).strip()
        if not vitpose_pose_model and pose_backend.lower() == "vitpose":
            vitpose_pose_model = legacy_pose_model
        self._mw._set_pose_model_path_for_backend(vitpose_pose_model, backend="vitpose")
```

- [ ] **Step 5: Run test to verify it passes**

Run: `PYTHONPATH=$PWD/src python -m pytest tests/test_vitpose_trackerkit_persistence.py -v`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add tests/test_vitpose_trackerkit_persistence.py src/hydra_suite/trackerkit/gui/orchestrators/config.py
git commit -m "fix(trackerkit): persist ViTPose model selection across sessions"
```

---

### Task 8: Pre-flight guard for variant/checkpoint mismatch

The training dialog offers variants B/L/H (`training.py:585`) but the catalog pins only `vitpose-b-coco` (`vitpose_checkpoints.py:56-65`; L/H/animal are documented TODOs). Selecting L or H with the only downloadable checkpoint (B) produces a silent backbone mismatch at load. Add a clear pre-flight error instead of a cryptic failure. Do NOT add fabricated checkpoints.

**Files:**
- Modify: `src/hydra_suite/posekit/core/vitpose_checkpoints.py` (add a `resolve_checkpoint` guard or a `check_variant_available` helper)
- Test: `tests/test_vitpose_variant_guard.py` (create)

**Interfaces:**
- Consumes: `CATALOG` (`vitpose_checkpoints.py:55`).
- Produces: `check_variant_available(variant: str) -> None` raises `ValueError` with a clear message naming the missing variant and the available ones when no catalog entry exists for a COCO checkpoint of that variant.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_vitpose_variant_guard.py
import pytest
from hydra_suite.posekit.core.vitpose_checkpoints import check_variant_available


def test_available_variant_passes():
    check_variant_available("b")  # vitpose-b-coco is in the catalog


def test_missing_variant_raises_with_guidance():
    with pytest.raises(ValueError) as ei:
        check_variant_available("h")
    msg = str(ei.value).lower()
    assert "h" in msg and "browse" in msg  # points user to Browse a checkpoint
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=$PWD/src python -m pytest tests/test_vitpose_variant_guard.py -v`
Expected: FAIL (`check_variant_available` not defined).

- [ ] **Step 3: Implement**

```python
def check_variant_available(variant: str) -> None:
    """Raise ValueError if no catalog checkpoint exists for this variant."""
    v = str(variant or "").strip().lower()
    available = sorted({k.split("-")[1] for k in CATALOG})  # e.g. {"b"}
    if v not in available:
        raise ValueError(
            f"No bundled ViTPose checkpoint for variant '{v}'. "
            f"Available auto-download variants: {', '.join(available)}. "
            f"Browse to a local {v}-variant checkpoint instead."
        )
```

Call it in the training-start path (`training.py::_start_vitpose_training`, ~`:1452`) before launching the worker, surfacing the error via the dialog's existing error path when the user chose auto-download (not Browse).

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=$PWD/src python -m pytest tests/test_vitpose_variant_guard.py -v`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add tests/test_vitpose_variant_guard.py src/hydra_suite/posekit/core/vitpose_checkpoints.py src/hydra_suite/posekit/gui/dialogs/training.py
git commit -m "feat(posekit): guard ViTPose training against unavailable variant checkpoints"
```

---

### Task 9: Core consistency cleanups (stage name, fail-loud gate, docstring)

Three small core-consistency items from the audit, grouped because each is a few lines:
1. `crops_worker.py:156` maps `vitpose` → `"yolo_pose"` resolver stage (harmless — resolver treats non-bgsub stages identically — but inconsistent with PoseKit's correct `"vitpose_pose"`).
2. `PoseViTPoseConfig` has no `auto_export` fail-loud flag; on export failure the vitpose branch silently degrades to native torch (`api.py:182-188`), unlike `OBBDirectConfig.auto_export` (config.py:97) which raises when an artifact is missing and auto_export is off.
3. Stale docstring `backends/vitpose.py:3` ("Native path only in Phase 1; ... Phase 2") — Phase 2 (CoreML+TensorRT runners) is wired.

**Files:**
- Modify: `src/hydra_suite/trackerkit/.../crops_worker.py:156`
- Modify: `src/hydra_suite/core/inference/config.py` (`PoseViTPoseConfig`, add `auto_export: bool = True`) and `core/inference/api.py:182-188` (respect it)
- Modify: `src/hydra_suite/core/individual/pose/backends/vitpose.py:3` (docstring)
- Test: `tests/test_vitpose_autoexport_gate.py` (create); extend stage assertion in `tests/test_vitpose_trackerkit_wiring.py` if appropriate

**Interfaces:**
- Consumes: `PoseViTPoseConfig` (config.py:333).
- Produces: `PoseViTPoseConfig.auto_export: bool = True`; when `False` and no accelerated artifact exists, the load path raises instead of silently using torch.

- [ ] **Step 1: Write the failing test (auto_export gate)**

```python
# tests/test_vitpose_autoexport_gate.py
from hydra_suite.core.inference.config import PoseViTPoseConfig


def test_auto_export_defaults_true_and_roundtrips():
    c = PoseViTPoseConfig(model_path="m.pt")
    assert c.auto_export is True
    from dataclasses import asdict
    d = asdict(c)
    assert d["auto_export"] is True
    c2 = PoseViTPoseConfig(**d)
    assert c2.auto_export is True
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=$PWD/src python -m pytest tests/test_vitpose_autoexport_gate.py -v`
Expected: FAIL (`auto_export` not a field).

- [ ] **Step 3: Implement all three**

- Add `auto_export: bool = True` to `PoseViTPoseConfig` (`config.py:333-338`). In `api.py:182-188`, when the accelerated artifact is missing/export fails AND `auto_export is False`, raise a clear error instead of setting `vitpose_flavor="native"`. When `auto_export is True`, keep current behavior (attempt export, warn+fallback on failure) to avoid changing the default UX.
- `crops_worker.py:156`: change `"sleap_pose" if backend_family == "sleap" else "yolo_pose"` to also map `vitpose` → `"vitpose_pose"`.
- Update the `backends/vitpose.py:3` docstring to state CoreML/TensorRT runners are wired.

- [ ] **Step 4: Run tests**

Run: `PYTHONPATH=$PWD/src python -m pytest tests/test_vitpose_autoexport_gate.py tests/test_vitpose_trackerkit_wiring.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add tests/test_vitpose_autoexport_gate.py src/hydra_suite/core/inference/config.py src/hydra_suite/core/inference/api.py src/hydra_suite/trackerkit/gui/crops_worker.py src/hydra_suite/core/individual/pose/backends/vitpose.py
git commit -m "chore(vitpose): vitpose_pose stage, auto_export gate, docstring refresh"
```

(Note: confirm the exact `crops_worker.py` path — the audit cited `trackerkit/.../crops_worker.py:156`; locate it with `git grep -n 'yolo_pose' src/hydra_suite/trackerkit`.)

---

### Task 10: Verify pose-parity model registration (no code change)

Per the settled decision, ViTPose registration stays at parity with YOLO-pose/SLEAP (copy into `models/pose/ViTPose/`, filename metadata, picker listing) — no `model_registry.json` entry for any pose backend. Add a regression test that locks this parity so future edits don't accidentally diverge ViTPose from the other pose backends.

**Files:**
- Test: `tests/test_vitpose_registration_parity.py` (create)

**Interfaces:**
- Consumes: `trackerkit/gui/model_utils.py::get_pose_models_directory`, `orchestrators/config.py` import flow.
- Produces: nothing (test only).

- [ ] **Step 1: Write the test**

```python
# tests/test_vitpose_registration_parity.py
import inspect
from hydra_suite.trackerkit.gui import model_utils
from hydra_suite.trackerkit.gui.orchestrators import config as cfgmod


def test_vitpose_pose_dir_parallels_yolo_and_sleap():
    for backend, leaf in [("yolo", "YOLO"), ("sleap", "SLEAP"), ("vitpose", "ViTPose")]:
        assert str(model_utils.get_pose_models_directory(backend)).endswith(leaf)


def test_pose_import_does_not_call_yolo_registry():
    # Pose parity decision: NO pose backend writes model_registry.json.
    src = inspect.getsource(cfgmod._import_pose_model_to_repository)
    assert "register_yolo_model" not in src, (
        "pose imports must stay registry-free (parity across yolo/sleap/vitpose); "
        "registry unification is a separate deferred spec"
    )
```

- [ ] **Step 2: Run test**

Run: `PYTHONPATH=$PWD/src python -m pytest tests/test_vitpose_registration_parity.py -v`
Expected: PASS immediately (documents current parity). If `_import_pose_model_to_repository` is not directly importable, adjust to `inspect.getsource(cfgmod)` and substring-check.

- [ ] **Step 3: Commit**

```bash
git add tests/test_vitpose_registration_parity.py
git commit -m "test(vitpose): lock pose-parity registration (no registry entry)"
```

---

### Task 11: Documentation

ViTPose is functional but undocumented: `grep vitpose docs/` (ex-superpowers) returns zero. Document it everywhere the other pose backends are documented, and fix the factually-stale runtime-integration stage list.

**Files:**
- Modify: `docs/developer-guide/runtime-integration.md:51-52` (add `vitpose_pose` to the stage list)
- Modify: `docs/user-guide/posekit.md` (ViTPose as a selectable inference + training backend; weight acquisition: auto HF download for COCO-B + Browse fallback + animal-checkpoint caveat)
- Modify: `docs/user-guide/compute-runtimes.md:15-16` (add ViTPose to the pose inference list)
- Modify: `docs/reference/ui-components-posekit.md:81` (backend selection includes ViTPose)
- Modify: `README.md` and `CLAUDE.md` (mention ViTPose where pose backends are enumerated)
- Modify: `docs/developer-guide/` (document the `python -m hydra_suite.core.individual.pose.vitpose.training --config run.json` CLI + the `RunConfig` schema fields from `training/config.py`)
- Test: `tests/test_vitpose_docs_present.py` (create — a lightweight doc-coverage guard)

**Interfaces:**
- Consumes: nothing.
- Produces: docs; a guard test.

- [ ] **Step 1: Write the failing guard test**

```python
# tests/test_vitpose_docs_present.py
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _read(p):
    return (ROOT / p).read_text(encoding="utf-8").lower()


def test_runtime_integration_lists_vitpose_pose_stage():
    assert "vitpose_pose" in _read("docs/developer-guide/runtime-integration.md")


def test_posekit_guide_documents_vitpose():
    assert "vitpose" in _read("docs/user-guide/posekit.md")


def test_compute_runtimes_lists_vitpose():
    assert "vitpose" in _read("docs/user-guide/compute-runtimes.md")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=$PWD/src python -m pytest tests/test_vitpose_docs_present.py -v`
Expected: FAIL (docs lack "vitpose").

- [ ] **Step 3: Write the docs**

Author the doc sections. For each pose-backend enumeration, add ViTPose parallel to YOLO/SLEAP. In `runtime-integration.md:51`, add `vitpose_pose` to the stage tuple listing. In `posekit.md`, add a "ViTPose Backend" subsection covering: selecting ViTPose in Predict/Predict-Dataset and Training, the auto HF download (`nielsr/vitpose-original-checkpoints`, COCO-B) vs Browse for a local/animal checkpoint, and that animal (AP-10K/APT-36K) checkpoints are not bundled. Document the training CLI + `RunConfig` fields (read `training/config.py` for the authoritative field list). Verify terminology rule (`posekit`, `hydra_suite.posekit`).

- [ ] **Step 4: Run test + docs build**

Run: `PYTHONPATH=$PWD/src python -m pytest tests/test_vitpose_docs_present.py -v` → PASS.
Run: `make docs-check` (build + terminology). Fix any strict-build breakage you introduced.

- [ ] **Step 5: Commit**

```bash
git add tests/test_vitpose_docs_present.py docs/ README.md CLAUDE.md
git commit -m "docs(vitpose): document ViTPose pose backend, weights, training CLI, stage list"
```

---

### Task 12: Centralize the ViTPose asset cache path in `paths.py` (Minor)

The ViTPose checkpoint cache defaults to an ad-hoc `~/.cache/vitpose-assets` (tests hard-code it; the app passes an ad-hoc `cache_dir`), bypassing `hydra_suite.paths`, which CLAUDE.md mandates as the single source for all path resolution. Add a `get_vitpose_cache_dir()` helper and route the app + catalog default through it.

**Files:**
- Modify: `src/hydra_suite/paths.py` (add `get_vitpose_cache_dir()`)
- Modify: `src/hydra_suite/posekit/core/vitpose_checkpoints.py` (use it as the default cache dir)
- Test: `tests/test_vitpose_cache_path.py` (create)

**Interfaces:**
- Consumes: `platformdirs`/`HYDRA_DATA_DIR` conventions already in `paths.py`.
- Produces: `get_vitpose_cache_dir() -> Path` under the data dir, honoring `HYDRA_DATA_DIR`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_vitpose_cache_path.py
import importlib


def test_vitpose_cache_honors_data_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("HYDRA_DATA_DIR", str(tmp_path))
    import hydra_suite.paths as paths
    importlib.reload(paths)
    d = paths.get_vitpose_cache_dir()
    assert str(d).startswith(str(tmp_path))
    assert "vitpose" in str(d).lower()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=$PWD/src python -m pytest tests/test_vitpose_cache_path.py -v`
Expected: FAIL (`get_vitpose_cache_dir` not defined).

- [ ] **Step 3: Implement**

Add to `paths.py`, mirroring the existing `get_models_dir`/data-dir helpers:

```python
def get_vitpose_cache_dir() -> Path:
    """Cache dir for downloaded ViTPose backbone checkpoints."""
    d = get_data_dir() / "vitpose-assets"
    d.mkdir(parents=True, exist_ok=True)
    return d
```

(Use the actual data-dir accessor name in `paths.py`.) In `vitpose_checkpoints.py`, default the cache dir to `get_vitpose_cache_dir()` when the caller passes none. Keep backward compatibility: an explicit `cache_dir` argument still overrides.

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=$PWD/src python -m pytest tests/test_vitpose_cache_path.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add tests/test_vitpose_cache_path.py src/hydra_suite/paths.py src/hydra_suite/posekit/core/vitpose_checkpoints.py
git commit -m "refactor(paths): centralize ViTPose asset cache dir via paths.py"
```

---

## Final Verification (after all tasks)

- [ ] Run the full ViTPose test set: `PYTHONPATH=$PWD/src python -m pytest tests/ -k vitpose -v` — all ViTPose tests pass, output pristine.
- [ ] Run the new non-vitpose-named tests: `tests/test_migrate_runtime_to_tier.py`. Confirm green.
- [ ] `make lint-moderate` on the touched files — no new issues.
- [ ] `make docs-check` — strict docs build passes.
- [ ] Manual smoke (user, out of band): fine-tune a ViTPose model in PoseKit → confirm loss plot renders, Open-Eval enables, "Use Latest" loads the checkpoint into Predict, eval dashboard scores it, and TrackerKit remembers the selection after restart.

## Self-Review Checklist (plan author ran this)

- **Spec coverage:** all 12 audit findings + registration decision map to Tasks 1-12/10. Blocker=Task 3; Important=Tasks 1,2,4,5,6,7,11; Minor=Tasks 8,9,12; parity=Task 10.
- **Dependencies:** Task 4 (Use-Latest) depends on Task 3 (populates `latest_pose_weights`) — ordered accordingly. Task 6 reuses `load_pose_backend` from the already-merged inference slice. Others are independent.
- **No fabricated data:** Task 8 guards rather than inventing checkpoints; catalog stays COCO-B.
- **Type consistency:** `resolve_finished_weights`, `parse_loss_series`, `check_variant_available`, `_latest_vitpose_candidate`, `get_vitpose_cache_dir`, `_load_images_for_vitpose` are each defined in exactly one task and referenced consistently.
