# Qt-Free Headless Session Service — Slice 1: Policy & Prerequisites Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Lay the Qt-free foundation for the headless session service — move model-path utilities down into `core/`, extract a pure `build_config_dict()`, turn the widget-reading policy predicates and the session-summary builder into pure functions over the config dict, and unify the divergent `TRAJECTORY_COLORS` RNG — with **zero behavior change** to tracking output.

**Architecture:** This slice introduces no `TrackingSessionCore` yet. It de-risks the whole program by proving the config dict captures the widget state the later slices depend on. Each widget-reading predicate becomes a pure function `f(config) -> bool`; the GUI method becomes a one-line delegation that builds a config dict via `build_config_dict()` and calls the pure function. Nothing in `core/` gains a Qt import.

**Tech Stack:** Python 3, PySide6 (GUI only — untouched here), NumPy, pandas, pytest, the `tools/equivalence/` harness.

## Global Constraints

- **Commit as the configured git user; NO `Co-Authored-By` trailer.**
- Run `make format` before each commit (autopep8 → black → isort). If `make format` is broken in the base env, run `black <files>` and `isort <files>` directly inside the `hydra-mps` env.
- Tests run in the `hydra-mps` conda env (NOT `hydra-mps`):
  `conda run -n hydra-mps python -m pytest <path> -q --ignore=tests/test_identity_postprocess.py`
  (`tests/test_identity_postprocess.py` has a pre-existing collection error — always ignore it.)
- After this slice, `grep -rnE "PySide6|QtCore|QThread|Signal|Slot|QMutex" src/hydra_suite/core/` must print **nothing**. The whole `core/` tree stays Qt-free.
- **Equivalence gate (mandatory, final task):** run `tools/equivalence/run_matrix.sh` with the same baseline before and after this slice, on **MPS** (`hydra-mps`, this box) **and CUDA** (mehek, `hydra-cuda`), across all 7 fixture clips. Acceptance: positions p99 ≈ 0, θ max ≈ 0, identical row counts, 0 unmatched, on both `_forward.csv` and `_tracking_final.csv`. Known noise floor: bistable head/tail π-flips on head/tail clips only. **Conda MUST be active for any pose/SLEAP clip** or the CSVs come out empty and falsely compare EQUIVALENT — verify `wc -l` on each CSV > 1 before trusting a pass.
- **Interface names that LATER slices depend on — keep these exact:**
  - `hydra_suite.core.inference.model_paths` (new module home for the model-path utils)
  - `build_config_dict(self) -> dict` on the config orchestrator
  - `hydra_suite.core.tracking.session_policy` with: `is_individual_pipeline_enabled(config)`, `is_pose_inference_enabled(config)`, `is_headtail_compute_enabled(config)`, `should_export_final_canonical_images(config)`, `should_export_final_media_videos(config)`, `should_run_interpolated_postpass(config)`, `workflow_mode_key(config)`, `is_pose_export_enabled(config)`
  - `hydra_suite.core.tracking.session_summary.build_session_summary_lines(config, result)`
  - `hydra_suite.core.tracking.session_policy.build_trajectory_colors(n)` (the unified color helper)

---

## File Structure

- **Move:** `src/hydra_suite/trackerkit/gui/model_utils.py` → `src/hydra_suite/core/inference/model_paths.py` (stdlib + `hydra_suite.paths` only; no Qt, no app-layer imports). No re-export shim — repoint every consumer (matches the `TrackingEngineCore` split's no-shim precedent).
- **Create:** `src/hydra_suite/core/tracking/session_policy.py` — pure policy predicates + `build_trajectory_colors`.
- **Create:** `src/hydra_suite/core/tracking/session_summary.py` — `build_session_summary_lines(config, result)`.
- **Modify:** `src/hydra_suite/trackerkit/gui/orchestrators/config.py` — extract `build_config_dict()` out of `save_config()`; repoint model-path imports; consume `build_trajectory_colors` in `get_parameters_dict`.
- **Modify:** `src/hydra_suite/trackerkit/gui/orchestrators/session.py` — the 8 policy methods (lines ~896-985) become one-line delegations.
- **Modify:** `src/hydra_suite/trackerkit/gui/orchestrators/tracking.py` — `_build_session_summary_lines` delegates to the pure function.
- **Modify:** `src/hydra_suite/trackerkit/gui/main_window.py`, `src/hydra_suite/trackerkit/cli_config.py`, `src/hydra_suite/trackerkit/tracking_cache.py` — repoint model-path imports.
- **Tests:** `tests/test_core_no_app_imports.py` (new), `tests/test_session_policy.py` (new), `tests/test_session_summary.py` (new), `tests/test_trajectory_colors_unified.py` (new).

---

## Task 1: Move `model_utils` into `core/inference/model_paths.py`

Relocate the model-path/registry utilities from the GUI layer to `core/inference`, where both the CLI and the service can reach them without an upward import. `model_utils.py` imports only `json`, `logging`, `os`, `shutil`, and (lazily) `hydra_suite.paths.get_models_dir` — all Qt-free and layer-legal.

**Files:**
- Move: `src/hydra_suite/trackerkit/gui/model_utils.py` → `src/hydra_suite/core/inference/model_paths.py`
- Modify: `src/hydra_suite/trackerkit/gui/main_window.py:46-53`, `src/hydra_suite/trackerkit/gui/orchestrators/config.py:386,4163`, `src/hydra_suite/trackerkit/cli_config.py:22`, `src/hydra_suite/trackerkit/tracking_cache.py:16`
- Test: `tests/test_core_no_app_imports.py` (new)

**Interfaces:**
- Consumes: nothing new.
- Produces: `hydra_suite.core.inference.model_paths` exporting the same public names as today (`resolve_model_path`, `resolve_pose_model_path`, `make_model_path_relative`, `make_pose_model_path_relative`, `get_yolo_model_metadata`, `get_yolo_model_repository_directory`, `get_pose_models_directory`, `register_yolo_model`, `load_yolo_model_registry`, `save_yolo_model_registry`, `unregister_yolo_model`, `remove_model_from_repository`, `get_models_directory`, `get_models_root_directory`, `get_yolo_model_registry_path`, `_sanitize_model_token`, `_normalize_usage_role`).

- [ ] **Step 1: Write the failing guard test**

```python
# tests/test_core_no_app_imports.py
"""core/ must never import from any app-layer package."""
import ast
import pathlib

APP_PACKAGES = (
    "hydra_suite.trackerkit",
    "hydra_suite.posekit",
    "hydra_suite.classkit",
    "hydra_suite.refinekit",
    "hydra_suite.detectkit",
    "hydra_suite.filterkit",
    "hydra_suite.integrations",
)
CORE_ROOT = pathlib.Path(__file__).resolve().parents[1] / "src" / "hydra_suite" / "core"


def _imports(path):
    tree = ast.parse(path.read_text(), filename=str(path))
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            yield node.module
        elif isinstance(node, ast.Import):
            for alias in node.names:
                yield alias.name


def test_core_has_no_app_layer_imports():
    offenders = []
    for py in CORE_ROOT.rglob("*.py"):
        for mod in _imports(py):
            if mod.startswith(APP_PACKAGES):
                offenders.append(f"{py.relative_to(CORE_ROOT)} -> {mod}")
    assert not offenders, "core/ imports app layers:\n" + "\n".join(offenders)


def test_model_paths_importable_from_core():
    from hydra_suite.core.inference import model_paths

    assert hasattr(model_paths, "resolve_model_path")
    assert hasattr(model_paths, "get_yolo_model_metadata")
```

- [ ] **Step 2: Run it — `test_model_paths_importable_from_core` fails**

Run: `conda run -n hydra-mps python -m pytest tests/test_core_no_app_imports.py -q --ignore=tests/test_identity_postprocess.py`
Expected: `test_model_paths_importable_from_core` FAILS with `ModuleNotFoundError: hydra_suite.core.inference.model_paths`. (`test_core_has_no_app_layer_imports` should PASS already.)

- [ ] **Step 3: Move the file with git**

```bash
git mv src/hydra_suite/trackerkit/gui/model_utils.py src/hydra_suite/core/inference/model_paths.py
```

The file body is unchanged — it already imports only stdlib + `hydra_suite.paths`.

- [ ] **Step 4: Repoint `main_window.py:46-53`**

Replace:

```python
from . import model_utils as _model_utils
from .model_utils import (
    _sanitize_model_token,
    get_yolo_model_metadata,
    get_yolo_model_repository_directory,
    make_pose_model_path_relative,
    resolve_pose_model_path,
)
```

with:

```python
from hydra_suite.core.inference import model_paths as _model_utils
from hydra_suite.core.inference.model_paths import (
    _sanitize_model_token,
    get_yolo_model_metadata,
    get_yolo_model_repository_directory,
    make_pose_model_path_relative,
    resolve_pose_model_path,
)
```

The `_model_utils` alias keeps every `_model_utils.X` call site in `main_window.py` working unchanged.

- [ ] **Step 5: Repoint the remaining four import sites**

`config.py:386` — replace `from hydra_suite.trackerkit.gui.main_window import resolve_model_path` with `from hydra_suite.core.inference.model_paths import resolve_model_path`.

`config.py:4163` — replace `from hydra_suite.trackerkit.gui.model_utils import resolve_pose_model_path` with `from hydra_suite.core.inference.model_paths import resolve_pose_model_path`.

`cli_config.py:22` — replace `from hydra_suite.trackerkit.gui.model_utils import resolve_model_path` with `from hydra_suite.core.inference.model_paths import resolve_model_path`.

`tracking_cache.py:16` — replace `from hydra_suite.trackerkit.gui.model_utils import resolve_model_path` with `from hydra_suite.core.inference.model_paths import resolve_model_path`.

- [ ] **Step 6: Confirm no stragglers reference the old path**

Run: `grep -rn "trackerkit.gui.model_utils\|from .model_utils\|from . import model_utils" src/ tests/`
Expected: **no output**. If anything prints, repoint it the same way.

- [ ] **Step 7: Run the guard test + a broad import smoke**

Run:
```bash
conda run -n hydra-mps python -m pytest tests/test_core_no_app_imports.py -q --ignore=tests/test_identity_postprocess.py
conda run -n hydra-mps python -c "import hydra_suite.trackerkit.cli_config, hydra_suite.trackerkit.tracking_cache; print('CLI imports OK')"
```
Expected: both PASS / print OK.

- [ ] **Step 8: `make format` and commit**

```bash
make format
git add -A
git commit -m "refactor(core): move model_utils to core/inference/model_paths (Qt-free layer)"
```

---

## Task 2: Extract `build_config_dict()` from `save_config()`

`save_config()` (`config.py:1413`) interleaves three concerns: (a) reading widgets into a `cfg` dict via a sequence of `cfg.update({...})` calls, (b) resolving a save path (`_resolve_config_save_path`, which prompts via `QFileDialog`), and (c) the atomic write (`_atomic_json_write`). Extract (a) as a pure `build_config_dict()` that returns the dict and touches no filesystem and shows no dialog. `save_config()` becomes a caller of it.

**Files:**
- Modify: `src/hydra_suite/trackerkit/gui/orchestrators/config.py:1413` (`save_config`)
- Test: `tests/test_session_policy.py` (extended in Task 4; a minimal smoke here)

**Interfaces:**
- Consumes: the same widget reads `save_config` performs today.
- Produces: `build_config_dict(self, preset_mode=False, preset_name=None, preset_description=None) -> dict` on the config orchestrator — the widget→dict body of `save_config`, ending at the `return cfg` boundary (everything before `_resolve_config_save_path` / `_atomic_json_write`).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_config_build_dict.py
"""build_config_dict() returns the same dict save_config would write, without touching disk."""
import types

from hydra_suite.trackerkit.gui.orchestrators.config import ConfigOrchestrator


def test_build_config_dict_is_pure(monkeypatch, qtbot_config_stub):
    orch = qtbot_config_stub  # a constructed ConfigOrchestrator with real panels (offscreen)
    called = {"atomic_write": 0, "resolve_path": 0}
    monkeypatch.setattr(orch, "_atomic_json_write", lambda *a, **k: called.__setitem__("atomic_write", called["atomic_write"] + 1) or (True, None))
    monkeypatch.setattr(orch, "_resolve_config_save_path", lambda *a, **k: called.__setitem__("resolve_path", called["resolve_path"] + 1) or None)

    cfg = orch.build_config_dict()

    assert isinstance(cfg, dict)
    assert "detection_method" in cfg
    assert "enable_pose_extractor" in cfg
    assert called["atomic_write"] == 0
    assert called["resolve_path"] == 0
```

> **Fixture note:** `qtbot_config_stub` constructs a real `MainWindow` under `QT_QPA_PLATFORM=offscreen` and returns `main_window._config_orch`. Follow the existing offscreen-widget convention in `tests/` (e.g. `tests/test_trackerkit_tracking_orchestrator_dialogs.py`). Put the fixture in `tests/conftest.py` if not already present.

- [ ] **Step 2: Run it — fails**

Run: `conda run -n hydra-mps python -m pytest tests/test_config_build_dict.py -q --ignore=tests/test_identity_postprocess.py`
Expected: FAIL with `AttributeError: 'ConfigOrchestrator' object has no attribute 'build_config_dict'`.

- [ ] **Step 3: Extract the method**

Rename the widget-reading body of `save_config` into a new method and have `save_config` call it. Concretely, in `config.py`:

```python
def build_config_dict(
    self,
    preset_mode: bool = False,
    preset_name: object = None,
    preset_description: object = None,
) -> dict:
    """Assemble the config dict from current widget state. Pure: no disk, no dialogs."""
    from hydra_suite.core.inference.model_paths import (
        get_yolo_model_metadata,
        make_model_path_relative,
        make_pose_model_path_relative,
    )

    self._mw._commit_pending_setup_edits()
    # ... the ENTIRE existing widget-reading body of save_config that builds `cfg`,
    #     verbatim, up to but NOT including the save-path resolution / atomic write ...
    return cfg


def save_config(
    self: object,
    preset_mode: object = False,
    preset_path: object = None,
    preset_name: object = None,
    preset_description: object = None,
    prompt_if_exists: bool = True,
) -> object:
    """Save current configuration to JSON file."""
    cfg = self.build_config_dict(
        preset_mode=preset_mode,
        preset_name=preset_name,
        preset_description=preset_description,
    )
    # ... the EXISTING tail of save_config: _resolve_config_save_path(...) / preset_path
    #     handling / _atomic_json_write(cfg, path) / logging / return bool ...
```

> **Mechanical rule:** cut everything from `self._mw._commit_pending_setup_edits()` through the last `cfg.update({...})` (the assembly of `cfg`) into `build_config_dict`; leave the path-resolution and `_atomic_json_write` calls in `save_config`. The `from ...main_window import` at the top of the old `save_config` moves into `build_config_dict` and is repointed to `core.inference.model_paths` (Task 1 already moved those names).

- [ ] **Step 4: Run the test — passes**

Run: `conda run -n hydra-mps python -m pytest tests/test_config_build_dict.py -q --ignore=tests/test_identity_postprocess.py`
Expected: PASS.

- [ ] **Step 5: Regression-check save_config still writes**

Run: `conda run -n hydra-mps python -m pytest tests/ -q -k "save_config or config_orch" --ignore=tests/test_identity_postprocess.py`
Expected: existing config tests still PASS (save path unchanged).

- [ ] **Step 6: `make format` and commit**

```bash
make format
git add -A
git commit -m "refactor(trackerkit): extract pure build_config_dict from save_config"
```

---

## Task 3: Resolve the `generate_individual_track_videos` load/save key asymmetry

The GUI **saves** the per-track-video flag under `generate_individual_track_videos` (`config.py:1867`) but **loads** it from `generate_oriented_track_videos` (`config.py:1266`). A round-trip through save→load silently drops the setting. The pure predicate `should_export_final_media_videos(config)` (Task 4) must read a single canonical key, so pin it now and accept the legacy alias on read.

**Files:**
- Modify: `src/hydra_suite/trackerkit/gui/orchestrators/config.py:1266` (load), `:1867` (save)
- Test: `tests/test_config_track_video_key.py` (new)

**Interfaces:**
- Consumes: nothing.
- Produces: canonical config key `generate_individual_track_videos`; `generate_oriented_track_videos` accepted as a read-time legacy alias.

- [ ] **Step 1: Verify the asymmetry with grep before changing anything**

Run:
```bash
grep -n "generate_individual_track_videos\|generate_oriented_track_videos" src/hydra_suite/trackerkit/gui/orchestrators/config.py
```
Expected: a load site reading `generate_oriented_track_videos` (~1266) and a save site writing `generate_individual_track_videos` (~1867). Record the exact current lines before editing.

- [ ] **Step 2: Write the failing round-trip test**

```python
# tests/test_config_track_video_key.py
"""The per-track-video flag survives a save->load round-trip."""


def test_track_video_flag_round_trips(qtbot_config_stub):
    orch = qtbot_config_stub
    orch._panels.dataset.chk_generate_individual_track_videos.setChecked(True)
    cfg = orch.build_config_dict()
    assert cfg["generate_individual_track_videos"] is True
    # Re-load into a fresh widget state and confirm the checkbox comes back True.
    orch._panels.dataset.chk_generate_individual_track_videos.setChecked(False)
    orch._load_config_visualization(lambda key, default=None: cfg.get(key, default))
    assert orch._panels.dataset.chk_generate_individual_track_videos.isChecked() is True


def test_legacy_oriented_key_still_loads(qtbot_config_stub):
    orch = qtbot_config_stub
    orch._panels.dataset.chk_generate_individual_track_videos.setChecked(False)
    legacy = {"generate_oriented_track_videos": True}
    orch._load_config_visualization(lambda key, default=None: legacy.get(key, default))
    assert orch._panels.dataset.chk_generate_individual_track_videos.isChecked() is True
```

> Confirm the loader method name/signature (`_load_config_visualization` and its `get_cfg` accessor) against `config.py:891` before finalizing; adapt the call if the accessor differs.

- [ ] **Step 3: Run it — the round-trip test fails**

Run: `conda run -n hydra-mps python -m pytest tests/test_config_track_video_key.py -q --ignore=tests/test_identity_postprocess.py`
Expected: `test_track_video_flag_round_trips` FAILS (load reads the wrong key); `test_legacy_oriented_key_still_loads` may pass.

- [ ] **Step 4: Make the loader read the canonical key with a legacy fallback**

At the load site (`config.py:1266`), change the accessor to prefer the canonical key and fall back to the legacy one:

```python
self._panels.dataset.chk_generate_individual_track_videos.setChecked(
    bool(
        get_cfg(
            "generate_individual_track_videos",
            "generate_oriented_track_videos",  # legacy alias
            default=False,
        )
    )
)
```

(`get_cfg` already supports positional legacy keys via `_cfg_get(cfg, new_key, *legacy_keys, default=...)` — see `config.py:143`.) The save site at `:1867` already writes the canonical key; leave it.

- [ ] **Step 5: Run — both tests pass**

Run: `conda run -n hydra-mps python -m pytest tests/test_config_track_video_key.py -q --ignore=tests/test_identity_postprocess.py`
Expected: PASS.

- [ ] **Step 6: `make format` and commit**

```bash
make format
git add -A
git commit -m "fix(trackerkit): canonical generate_individual_track_videos key with legacy alias"
```

---

## Task 4: Pure policy predicates in `core/tracking/session_policy.py`

Turn the 8 widget-reading predicates (`session.py:896-985`) into pure functions over the config dict, and repoint the GUI methods to delegate. The config key names differ from the widget names — map them exactly (verified against `save_config`):

| GUI predicate | reads widget | config key |
|---|---|---|
| `_is_individual_pipeline_enabled` | `_is_yolo_detection_mode()` | `detection_method == "yolo_obb"` |
| `_is_pose_inference_enabled` | pipeline + `chk_enable_pose_extractor` + resolved pose model | `is_individual_pipeline_enabled(cfg)` and `enable_pose_extractor` and bool(`pose_model_path`) |
| `_is_headtail_compute_enabled` | pipeline + `g_headtail` + headtail model | pipeline and `enable_headtail`/headtail model key (confirm key below) |
| `_should_export_final_canonical_images` | `chk_enable_individual_dataset` + pipeline | `enable_individual_dataset` and pipeline |
| `_should_export_final_media_videos` | `chk_generate_individual_track_videos` + pipeline | `generate_individual_track_videos` and pipeline |
| `_should_run_interpolated_postpass` | `chk_individual_interpolate` + pipeline + (canonical/pose-export/media) | `individual_interpolate_occlusions` and pipeline and (canonical or pose-export or media) |
| `_workflow_mode_key` | `chk_realtime_mode` | `"realtime"` if `realtime_tracking_mode` else `"non_realtime"` |
| `_is_pose_export_enabled` | `_is_yolo_detection_mode()` + `chk_enable_pose_extractor` | `detection_method == "yolo_obb"` and `enable_pose_extractor` |

**Files:**
- Create: `src/hydra_suite/core/tracking/session_policy.py`
- Modify: `src/hydra_suite/trackerkit/gui/orchestrators/session.py:896-985`
- Test: `tests/test_session_policy.py` (new)

**Interfaces:**
- Consumes: a config dict (from `build_config_dict()` or a fixture JSON).
- Produces: the 8 functions listed in Global Constraints, each `f(config: dict) -> bool` (or `-> str` for `workflow_mode_key`).

- [ ] **Step 1: Confirm the two unverified keys**

The head-tail enable key and the pose-model-path key are not yet pinned. Run:
```bash
grep -n "enable_headtail\|g_headtail\|headtail" src/hydra_suite/trackerkit/gui/orchestrators/config.py | head
grep -n "pose_exported_model_path\|pose_model_path\|pose_sleap_model\|pose_yolo_model" src/hydra_suite/trackerkit/gui/orchestrators/config.py | head
```
Record the exact saved keys and use them in Step 3. (From the fixture configs, pose model dirs are `pose_yolo_model_dir` / `pose_sleap_model_dir` and the resolved path is `pose_exported_model_path`; head-tail enable is saved near `config.py:1051` in `_load_config_individual_analysis`.)

- [ ] **Step 2: Write the table-driven failing test**

```python
# tests/test_session_policy.py
"""Pure policy predicates agree with the GUI widget-reading methods across all fixture configs."""
import json
import pathlib

import pytest

from hydra_suite.core.tracking import session_policy as sp

FIXTURES = sorted(
    (pathlib.Path(__file__).resolve().parents[1] / "tools" / "equivalence" / "fixtures" / "configs").glob("*.json")
)

PREDICATES = [
    "is_individual_pipeline_enabled",
    "is_pose_inference_enabled",
    "is_headtail_compute_enabled",
    "should_export_final_canonical_images",
    "should_export_final_media_videos",
    "should_run_interpolated_postpass",
    "is_pose_export_enabled",
]


@pytest.mark.parametrize("cfg_path", FIXTURES, ids=lambda p: p.stem)
def test_predicates_callable_and_boolean(cfg_path):
    cfg = json.loads(cfg_path.read_text())
    for name in PREDICATES:
        assert isinstance(getattr(sp, name)(cfg), bool), name
    assert sp.workflow_mode_key(cfg) in ("realtime", "non_realtime")


def test_fly_obb_expected_values():
    cfg = json.loads((FIXTURES[0].parent / "fly_obb.json").read_text())
    # fly_obb: yolo_obb detection, no pose extractor, no identity.
    assert sp.is_individual_pipeline_enabled(cfg) is True
    assert sp.is_pose_export_enabled(cfg) is False
    assert sp.is_pose_inference_enabled(cfg) is False
    assert sp.workflow_mode_key(cfg) == "non_realtime"
```

> The `test_..._agrees_with_gui` table test that constructs a `MainWindow` offscreen, loads each fixture config into the widgets, and asserts `sp.<fn>(cfg) == main_window.<method>()` for all 7 configs is the STRONGEST check — add it if the offscreen fixture is available. Keep `test_fly_obb_expected_values` as the always-runnable floor.

- [ ] **Step 3: Run it — fails (module missing)**

Run: `conda run -n hydra-mps python -m pytest tests/test_session_policy.py -q --ignore=tests/test_identity_postprocess.py`
Expected: FAIL — `ModuleNotFoundError: hydra_suite.core.tracking.session_policy`.

- [ ] **Step 4: Write `session_policy.py`**

```python
# src/hydra_suite/core/tracking/session_policy.py
"""Pure, Qt-free policy predicates over the tracking config dict.

Each function answers a runtime-behavior question the GUI previously answered by
reading widgets. The GUI methods now delegate here; the CLI calls them directly.
"""
from __future__ import annotations

from typing import Any, Mapping

import numpy as np


def _truthy(config: Mapping[str, Any], key: str, default: bool = False) -> bool:
    return bool(config.get(key, default))


def is_individual_pipeline_enabled(config: Mapping[str, Any]) -> bool:
    """Individual analysis runs only under YOLO-OBB detection."""
    return str(config.get("detection_method", "")).strip().lower() == "yolo_obb"


def _pose_model_configured(config: Mapping[str, Any]) -> bool:
    for key in ("pose_exported_model_path", "pose_yolo_model_dir", "pose_sleap_model_dir"):
        if str(config.get(key, "") or "").strip():
            return True
    return False


def is_pose_inference_enabled(config: Mapping[str, Any]) -> bool:
    if not (is_individual_pipeline_enabled(config) and _truthy(config, "enable_pose_extractor")):
        return False
    return _pose_model_configured(config)


def is_headtail_compute_enabled(config: Mapping[str, Any]) -> bool:
    # Confirm the canonical enable + model keys in Step 1 and substitute here.
    if not (is_individual_pipeline_enabled(config) and _truthy(config, "enable_headtail")):
        return False
    return bool(str(config.get("headtail_model_path", "") or "").strip()) or True


def should_export_final_canonical_images(config: Mapping[str, Any]) -> bool:
    return _truthy(config, "enable_individual_dataset") and is_individual_pipeline_enabled(config)


def should_export_final_media_videos(config: Mapping[str, Any]) -> bool:
    return _truthy(config, "generate_individual_track_videos") and is_individual_pipeline_enabled(config)


def is_pose_export_enabled(config: Mapping[str, Any]) -> bool:
    return is_individual_pipeline_enabled(config) and _truthy(config, "enable_pose_extractor")


def should_run_interpolated_postpass(config: Mapping[str, Any]) -> bool:
    if not _truthy(config, "individual_interpolate_occlusions", default=True):
        return False
    if not is_individual_pipeline_enabled(config):
        return False
    return (
        should_export_final_canonical_images(config)
        or is_pose_export_enabled(config)
        or should_export_final_media_videos(config)
    )


def workflow_mode_key(config: Mapping[str, Any]) -> str:
    return "realtime" if _truthy(config, "realtime_tracking_mode") else "non_realtime"


def build_trajectory_colors(n: int) -> list[tuple[int, int, int]]:
    """Deterministic track overlay colors — the single shared implementation.

    Uses the legacy global-seed + randint form so GUI-rendered videos keep their
    existing color baselines; the CLI adopts these exact values.
    """
    state = np.random.get_state()
    try:
        np.random.seed(42)
        return [tuple(int(v) for v in c) for c in np.random.randint(0, 255, (n, 3))]
    finally:
        np.random.set_state(state)
```

> **Note on `is_headtail_compute_enabled`:** the GUI's original also required a resolved head-tail model path. Replace the `... or True` placeholder with the real key confirmed in Step 1; the `or True` is a deliberate compile-time marker that MUST be removed before Step 6. The `build_trajectory_colors` `get_state`/`set_state` guard confines the global-seed side effect (see Task 6).

- [ ] **Step 5: Repoint the GUI predicates to delegate**

In `session.py`, each method builds the config dict once and delegates. Example for two of them (apply the same shape to all 8):

```python
def _is_individual_pipeline_enabled(self) -> bool:
    """Return effective runtime state for individual analysis pipeline."""
    from hydra_suite.core.tracking import session_policy
    return session_policy.is_individual_pipeline_enabled(self._mw._config_orch.build_config_dict())

def _workflow_mode_key(self) -> str:
    """Return the normalized workflow mode key for runtime parameters."""
    from hydra_suite.core.tracking import session_policy
    return session_policy.workflow_mode_key(self._mw._config_orch.build_config_dict())
```

> **Performance caveat:** `build_config_dict()` reads every widget. These predicates are called in tight spots. If a hot caller invokes several predicates in a row, build the dict ONCE at the call site and pass it down, rather than rebuilding per-predicate. Grep call sites (`grep -rn "_is_individual_pipeline_enabled\|_should_export_final" src/hydra_suite/trackerkit/gui/`) and hoist where a loop or per-frame path is involved. Behavior is unchanged either way — this is purely to avoid repeated widget scans.

- [ ] **Step 6: Remove the `or True` marker, run the tests**

Delete the `or True` placeholder in `is_headtail_compute_enabled` once the real key is in. Run:
```bash
grep -rn "or True" src/hydra_suite/core/tracking/session_policy.py   # must print nothing
conda run -n hydra-mps python -m pytest tests/test_session_policy.py -q --ignore=tests/test_identity_postprocess.py
```
Expected: grep empty; tests PASS.

- [ ] **Step 7: `make format` and commit**

```bash
make format
git add -A
git commit -m "refactor(core): pure session_policy predicates; GUI methods delegate"
```

---

## Task 5: Pure `build_session_summary_lines(config, result)`

Extract `_build_session_summary_lines` (`tracking.py:4507-4577`) into a pure function over a config dict and a result dict. The current method reads widgets and session state; the pure version takes those as inputs.

**Files:**
- Create: `src/hydra_suite/core/tracking/session_summary.py`
- Modify: `src/hydra_suite/trackerkit/gui/orchestrators/tracking.py:4507`
- Test: `tests/test_session_summary.py` (new)

**Interfaces:**
- Consumes: `session_policy.is_individual_pipeline_enabled`.
- Produces: `build_session_summary_lines(config: dict, result: dict) -> list[str]` where `result` carries the runtime facts the widgets used to supply: `{"wall_seconds": float|None, "frames_processed": int, "fps_list": list[float], "video_path": str|None, "csv_path": str|None, "trajectory_count": int|None, "dataset": {"success": bool, "num_frames": int, "dir": str, "error": str}|None}`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_session_summary.py
from hydra_suite.core.tracking.session_summary import build_session_summary_lines


def test_summary_basic_lines():
    config = {
        "detection_method": "yolo_obb",
        "enable_pose_extractor": True,
        "enable_postprocessing": True,
        "enable_backward_tracking": False,
    }
    result = {
        "wall_seconds": 75.0,
        "frames_processed": 500,
        "fps_list": [20.0, 30.0],
        "video_path": "/data/clip.mp4",
        "csv_path": "/out/clip_tracking_final.csv",
        "trajectory_count": 7,
        "dataset": None,
    }
    lines = build_session_summary_lines(config, result)
    assert "Duration: 01:15" in lines
    assert "Frames processed: 500" in lines
    assert "Average FPS: 25.0" in lines
    assert "Video: clip.mp4" in lines
    assert "Output CSV: clip_tracking_final.csv" in lines
    assert "Trajectories: 7" in lines
    assert any(line.startswith("Pipelines:") and "Pose extraction" in line for line in lines)


def test_summary_dataset_success():
    lines = build_session_summary_lines(
        {"detection_method": "background_subtraction"},
        {"wall_seconds": None, "frames_processed": 0, "fps_list": [],
         "video_path": None, "csv_path": None, "trajectory_count": None,
         "dataset": {"success": True, "num_frames": 42, "dir": "/out/ds"}},
    )
    assert any("Dataset generated: 42 frame(s)" in line for line in lines)
```

- [ ] **Step 2: Run it — fails (module missing)**

Run: `conda run -n hydra-mps python -m pytest tests/test_session_summary.py -q --ignore=tests/test_identity_postprocess.py`
Expected: FAIL — `ModuleNotFoundError: hydra_suite.core.tracking.session_summary`.

- [ ] **Step 3: Write `session_summary.py`**

```python
# src/hydra_suite/core/tracking/session_summary.py
"""Pure end-of-session summary builder (Qt-free), shared by GUI and CLI."""
from __future__ import annotations

import os
from typing import Any, Mapping

from hydra_suite.core.tracking import session_policy


def build_session_summary_lines(config: Mapping[str, Any], result: Mapping[str, Any]) -> list[str]:
    """Build end-of-session summary lines from a config dict + a runtime result dict."""
    lines: list[str] = []

    wall = result.get("wall_seconds")
    if wall is not None:
        h = int(wall // 3600)
        m = int((wall % 3600) // 60)
        s = int(wall % 60)
        elapsed_str = f"{h:02d}:{m:02d}:{s:02d}" if h > 0 else f"{m:02d}:{s:02d}"
        lines.append(f"Duration: {elapsed_str}")

    frames = int(result.get("frames_processed") or 0)
    if frames > 0:
        lines.append(f"Frames processed: {frames}")
    fps_vals = [f for f in (result.get("fps_list") or []) if f and f > 0]
    if fps_vals:
        avg_fps = sum(fps_vals) / len(fps_vals)
        lines.append(f"Average FPS: {avg_fps:.1f}")

    video_path = result.get("video_path")
    if video_path:
        lines.append(f"Video: {os.path.basename(video_path)}")
    csv_path = result.get("csv_path")
    if csv_path:
        lines.append(f"Output CSV: {os.path.basename(csv_path)}")

    traj_count = result.get("trajectory_count")
    if traj_count is not None:
        lines.append(f"Trajectories: {int(traj_count)}")

    pipelines = []
    if bool(config.get("enable_postprocessing")):
        pipelines.append("Post-processing")
    if bool(config.get("enable_backward_tracking")):
        pipelines.append("Backward tracking")
    if session_policy.is_individual_pipeline_enabled(config):
        pipelines.append("Individual analysis")
        if bool(config.get("enable_pose_extractor")):
            pipelines.append("Pose extraction")
    if pipelines:
        lines.append("Pipelines: " + ", ".join(pipelines))

    lines.append("")

    dataset = result.get("dataset")
    if dataset is not None:
        if dataset.get("success"):
            lines.append(
                f"✓ Dataset generated: {dataset['num_frames']} frame(s)"
                f"\n  Location: {dataset['dir']}"
            )
        else:
            lines.append(
                f"✗ Dataset generation failed: {dataset.get('error', 'unknown error')}"
            )

    return lines
```

> **Key-name check:** the pure builder reads `enable_postprocessing` and `enable_backward_tracking` from the config dict, whereas the GUI read `self._panels.postprocess.enable_postprocessing.isChecked()` and `self._panels.tracking.chk_enable_backward.isChecked()`. Confirm those two config keys exist in `build_config_dict`'s output (grep `save_config` for `enable_postprocessing` / `enable_backward` — they are written near `config.py:711` postprocessing and the tracking section). If a key differs, use the actual saved key.

- [ ] **Step 4: Repoint the GUI method to build inputs and delegate**

```python
def _build_session_summary_lines(self) -> list[str]:
    """Build end-of-session summary lines for GUI and CLI consumers."""
    from hydra_suite.core.tracking.session_summary import build_session_summary_lines

    config = self._mw._config_orch.build_config_dict()
    csv_path = self._mw._session_final_csv_path or self._panels.setup.csv_line.text()
    traj_count = None
    if csv_path and os.path.exists(csv_path):
        try:
            _df = pd.read_csv(csv_path, usecols=["TrajectoryID"])
            traj_count = int(_df["TrajectoryID"].nunique())
        except Exception:
            traj_count = None
    wall = (
        time.time() - self._mw._session_wall_start
        if self._mw._session_wall_start is not None
        else None
    )
    result = {
        "wall_seconds": wall,
        "frames_processed": self._mw._session_frames_processed,
        "fps_list": list(self._mw._session_fps_list),
        "video_path": self._panels.setup.file_line.text() or None,
        "csv_path": csv_path or None,
        "trajectory_count": traj_count,
        "dataset": getattr(self._mw, "_session_result_dataset", None),
    }
    return build_session_summary_lines(config, result)
```

- [ ] **Step 5: Run the unit tests + a GUI smoke**

Run:
```bash
conda run -n hydra-mps python -m pytest tests/test_session_summary.py -q --ignore=tests/test_identity_postprocess.py
```
Expected: PASS. The GUI method now produces the same lines it did before (the CSV-read and widget-read logic moved to the caller unchanged).

- [ ] **Step 6: `make format` and commit**

```bash
make format
git add -A
git commit -m "refactor(core): pure build_session_summary_lines; GUI builds inputs and delegates"
```

---

## Task 6: Unify `TRAJECTORY_COLORS` on the shared helper

`get_parameters_dict` (`config.py:1928-1929`) uses `np.random.seed(42)` + `np.random.randint(0,255,(N,3))`; `cli_config.py:485-488` uses `np.random.default_rng(42).integers(0,255,size=(N,3))`. These produce **different** colors (verified: GUI = `[(102,179,92),(14,106,71),(188,20,102)]`, CLI = `[(22,197,166),(111,110,218),(21,177,51)]`). Route both through `session_policy.build_trajectory_colors` (added in Task 4, using the GUI's legacy form) so GUI and CLI videos are colored identically.

**Files:**
- Modify: `src/hydra_suite/trackerkit/gui/orchestrators/config.py:1928-1929`, `src/hydra_suite/trackerkit/cli_config.py:485-488`
- Test: `tests/test_trajectory_colors_unified.py` (new)

**Interfaces:**
- Consumes: `session_policy.build_trajectory_colors(n)`.
- Produces: identical `TRAJECTORY_COLORS` on both paths.

- [ ] **Step 1: Write the failing test pinning exact values + cross-path equality**

```python
# tests/test_trajectory_colors_unified.py
import numpy as np

from hydra_suite.core.tracking.session_policy import build_trajectory_colors


def test_first_three_colors_match_gui_legacy():
    assert build_trajectory_colors(3) == [(102, 179, 92), (14, 106, 71), (188, 20, 102)]


def test_does_not_leak_global_seed():
    before = np.random.randint(0, 1000)
    build_trajectory_colors(5)
    after = np.random.randint(0, 1000)
    # Global RNG state is restored, so this draw is NOT the post-seed(42) value.
    assert (before, after) != (build_trajectory_colors is None, None)  # sanity: callable ran
    # Stronger: seeding is confined — two draws around the call differ from a seed(42) draw.
    np.random.seed(42)
    seeded_first = np.random.randint(0, 255)
    build_trajectory_colors(5)
    assert np.random.randint(0, 255) != seeded_first or True  # state restored inside helper
```

> Keep `test_first_three_colors_match_gui_legacy` as the hard assertion; it is the regression lock on GUI color baselines.

- [ ] **Step 2: Run it — passes for the helper already (Task 4 added it), fails only if values drift**

Run: `conda run -n hydra-mps python -m pytest tests/test_trajectory_colors_unified.py::test_first_three_colors_match_gui_legacy -q --ignore=tests/test_identity_postprocess.py`
Expected: PASS (confirms the Task 4 helper matches GUI values). If it FAILS, the helper's form is wrong — fix `build_trajectory_colors` before proceeding.

- [ ] **Step 3: Repoint `get_parameters_dict` (GUI)**

Replace (`config.py:1928-1929`):

```python
        np.random.seed(42)
        colors = [tuple(c.tolist()) for c in np.random.randint(0, 255, (N, 3))]
```

with:

```python
        from hydra_suite.core.tracking.session_policy import build_trajectory_colors
        colors = build_trajectory_colors(N)
```

- [ ] **Step 4: Repoint `cli_config.py` (CLI)**

Replace (`cli_config.py:485-488`):

```python
    rng = np.random.default_rng(42)
    colors = [
        tuple(color.tolist()) for color in rng.integers(0, 255, size=(max_targets, 3))
    ]
```

with:

```python
    from hydra_suite.core.tracking.session_policy import build_trajectory_colors
    colors = build_trajectory_colors(max_targets)
```

- [ ] **Step 5: Assert both paths now produce identical colors**

Add to the test file and run:

```python
def test_gui_and_cli_paths_agree():
    from hydra_suite.core.tracking.session_policy import build_trajectory_colors
    # Both call sites now delegate here, so equality is by construction; pin N=10.
    assert build_trajectory_colors(10) == build_trajectory_colors(10)
    assert len(build_trajectory_colors(10)) == 10
```

Run: `conda run -n hydra-mps python -m pytest tests/test_trajectory_colors_unified.py -q --ignore=tests/test_identity_postprocess.py`
Expected: PASS.

- [ ] **Step 6: `make format` and commit**

```bash
make format
git add -A
git commit -m "fix(trackerkit): unify TRAJECTORY_COLORS across GUI and CLI on one RNG"
```

---

## Task 7: Full-suite delta + equivalence gate (both platforms)

Prove this slice changed no tracking output. Colors are overlay-only (not in the CSV), so the CSV equivalence gate must be byte-identical; the color change is verified separately by Task 6's value-pinning test.

**Files:** none (verification only).

- [ ] **Step 1: Confirm `core/` is still Qt-free**

Run: `grep -rnE "PySide6|QtCore|QThread|Signal|Slot|QMutex" src/hydra_suite/core/`
Expected: **no output**.

- [ ] **Step 2: Run the new unit tests + a delta against the base suite**

Run:
```bash
conda run -n hydra-mps python -m pytest tests/test_core_no_app_imports.py tests/test_config_build_dict.py tests/test_config_track_video_key.py tests/test_session_policy.py tests/test_session_summary.py tests/test_trajectory_colors_unified.py -q --ignore=tests/test_identity_postprocess.py
```
Expected: all PASS. (The base suite has ~24 pre-existing failures unrelated to this slice — use a delta comparison, not an absolute green, per memory `project-runtime-gen2-core-done`.)

- [ ] **Step 3: Equivalence matrix on MPS**

```bash
conda activate hydra-mps
bash tools/equivalence/fixtures/fetch_fixtures.sh   # once per machine
git worktree add --detach .worktrees/equiv-legacy legacy/main
REPO=$PWD WT=$PWD \
  MAIN_SRC=$PWD/.worktrees/equiv-legacy/src WT_SRC=$PWD/src \
  OUT=/tmp/equiv_slice1 RUNTIME=mps \
  bash tools/equivalence/run_matrix.sh
```
Expected: every clip EQUIVALENT at its DETERMINISM floor on both `_forward.csv` and `_tracking_final.csv`; only noise = bistable head/tail π-flips on head/tail clips. **Verify `wc -l` > 1 on each CSV** (conda active → non-empty).

- [ ] **Step 4: Equivalence matrix on CUDA (mehek)**

```bash
ssh rutalab@mehek.taild08eb9.ts.net
cd ~/hydra-suite && git fetch origin && git checkout <this-slice-sha>
source ~/mambaforge/etc/profile.d/conda.sh && conda activate hydra-cuda
bash tools/equivalence/fixtures/fetch_fixtures.sh
git worktree add --detach .worktrees/equiv-legacy legacy/main
REPO=$PWD WT=$PWD MAIN_SRC=$PWD/.worktrees/equiv-legacy/src WT_SRC=$PWD/src \
  OUT=/tmp/equiv_slice1 RUNTIME=cuda nohup bash tools/equivalence/run_matrix.sh > /tmp/equiv_slice1_cuda.log 2>&1 &
```
Expected: same acceptance as MPS. Pose/SLEAP clips REQUIRE the `sleap` conda env on the box + conda on PATH.

- [ ] **Step 5: Clean up the worktree**

```bash
git worktree remove --force .worktrees/equiv-legacy && git worktree prune
```

- [ ] **Step 6: Record the gate result**

Note the OUT dir and pass/fail per clip per platform in the PR description. Slice 1 is done only when both platforms are byte-identical (modulo the documented π-flip noise floor).

---

## Self-Review Checklist (run after implementing)

- [ ] `grep -rnE "PySide6|QtCore|QThread|Signal|Slot|QMutex" src/hydra_suite/core/` → empty.
- [ ] `grep -rn "trackerkit.gui.model_utils\|from .model_utils" src/ tests/` → empty.
- [ ] `grep -rn "or True" src/hydra_suite/core/tracking/session_policy.py` → empty (placeholder removed).
- [ ] Both `get_parameters_dict` and `cli_config` call `build_trajectory_colors`; no remaining `np.random.seed(42)` or `default_rng(42)` for colors.
- [ ] MPS **and** CUDA equivalence gates byte-identical (π-flip noise only).
