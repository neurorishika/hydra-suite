# Model Registry Unification Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `model_registry.json` the single authoritative model inventory: one owning module in the Data layer, rich per-family metadata that pickers consume, all backends (incl. the 3 pose backends) registered on import, a pose publish role, and an interactive CLI to backfill pre-existing models — with the two old writer APIs retired outright (no shims).

**Architecture:** New `hydra_suite/data/model_registry.py` (Data layer — importable by app kits, Training, and the CLI without violating dependency direction) owns `model_registry.json`. The GUI-side (`trackerkit/gui/model_utils.py`) and Training-side (`training/model_publish.py`) registry functions are deleted and every call site repointed. Pickers list from the registry (source of truth); unregistered models trigger a non-blocking migrate notice. A CLI backfills existing models.

**Tech Stack:** Python 3, dataclasses, PyQt, pytest. Conda env `hydra-mps`.

**Design spec:** `docs/superpowers/specs/2026-07-29-model-registry-unification-design.md`

## Global Constraints

- Work in an isolated worktree under `.worktrees/`; branch `feature/model-registry-unification`. Main stays untouched.
- Environment: `source ~/miniforge3/etc/profile.d/conda.sh && conda activate hydra-mps` before running anything.
- Run tests from the worktree with `PYTHONPATH=$PWD/src`. Base suite has ~24 pre-existing unrelated failures (incl. a collection error in `tests/test_identity_postprocess.py`) — gate on the delta (your new/touched tests), not absolute green. Run vitpose-adjacent suites with `--continue-on-collection-errors`.
- Line length 88. Do NOT run `make format`. Match surrounding style manually.
- Commit as the configured git user. NO `Co-Authored-By: Claude` trailer.
- **Dependency rule (hard):** `hydra_suite/data/model_registry.py` MUST NOT import from any app-layer package (trackerkit/posekit/classkit/refinekit/detectkit/filterkit) or Integrations. It may import from `hydra_suite.paths` and stdlib only. App layers + Training + the CLI import IT.
- **Retire, don't shim:** after repointing callers, DELETE the old registry functions. No back-compat wrappers left behind.
- **Registry file:** both old writers target `get_models_dir()/model_registry.json`. `training/model_publish.py` additionally honors a `_use_project_root_override()` (project-root `models/`) test hook — the unified module MUST preserve an equivalent override so `model_publish` tests still work. Confirm the real override mechanism before finalizing path resolution.
- **Root file format:** `{"schema_version": <int>, "entries": {relpath: entry}}`. Bump `schema_version` to 3 for the unified schema; `load_registry` transparently upgrades v2-GUI, flat-legacy, and v2-`model_publish` entries on read.
- **Atomic writes:** `save_registry` writes to a temp file + `os.replace` (never a partial file on crash).

---

## Phase 1 — The unified data module (no consumers yet)

### Task 1: `ModelRegistryEntry` schema + load/save + legacy read

**Files:**
- Create: `src/hydra_suite/data/model_registry.py`
- Test: `tests/test_model_registry_core.py`

**Interfaces:**
- Produces:
  - `@dataclass ModelRegistryEntry` with fields: `task_family: str` (`obb`/`detect`/`segment`/`classify`/`pose`), `backend: str` (`yolo`/`sleap`/`vitpose`), `usage_role: str = ""`, `species: str = ""`, `notes: str = ""`, `added_at: str = ""`, `source_path: str = ""`, `stored_filename: str = ""`, `size: str = ""`, `num_keypoints: int = 0`, `skeleton_name: str = ""`, `num_classes: int = 0`, `class_names: list[str] = field(default_factory=list)`, `needs_review: bool = False`, `extra: dict = field(default_factory=dict)`. Methods `to_dict()` / `from_dict(d)`.
  - `SCHEMA_VERSION = 3`
  - `registry_path() -> Path` — `get_models_dir()/model_registry.json`, honoring the `model_publish` project-root override.
  - `load_registry() -> dict[str, ModelRegistryEntry]` — reads the file, upgrades any legacy entry via `read_legacy_entry`, returns `{relpath: ModelRegistryEntry}`; `{}` on missing/corrupt.
  - `save_registry(entries: dict[str, ModelRegistryEntry]) -> None` — atomic write of `{"schema_version": 3, "entries": {p: e.to_dict()}}`.
  - `read_legacy_entry(relpath: str, raw: dict) -> ModelRegistryEntry` — maps the GUI format (`{size, species, model_info, added_at, source_path, stored_filename, task_family, usage_role}`), the `model_publish` format (arch/description/classifier fields), and flat unknown keys (→ `extra`) into a `ModelRegistryEntry`. `model_info`→`notes`. Infers `task_family` from the relpath prefix (`obb/`, `detection/`, `classification/`, `pose/`) when absent, and `backend` from `pose/<Backend>/` or default `yolo`.

- [ ] **Step 1: Write failing tests**

```python
# tests/test_model_registry_core.py
import json
from hydra_suite.data.model_registry import (
    ModelRegistryEntry, SCHEMA_VERSION, load_registry, save_registry,
    read_legacy_entry, registry_path,
)


def test_entry_roundtrip():
    e = ModelRegistryEntry(task_family="pose", backend="vitpose", num_keypoints=17,
                           species="ant", notes="hi")
    assert ModelRegistryEntry.from_dict(e.to_dict()) == e


def test_save_then_load_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setenv("HYDRA_DATA_DIR", str(tmp_path))
    import hydra_suite.paths as paths, importlib
    importlib.reload(paths)
    import hydra_suite.data.model_registry as mr
    importlib.reload(mr)
    e = mr.ModelRegistryEntry(task_family="obb", backend="yolo", species="fly")
    mr.save_registry({"obb/m.pt": e})
    got = mr.load_registry()
    assert got["obb/m.pt"].species == "fly"
    raw = json.loads(mr.registry_path().read_text())
    assert raw["schema_version"] == mr.SCHEMA_VERSION


def test_read_legacy_gui_format():
    raw = {"size": "s", "species": "fly", "model_info": "v1",
           "added_at": "2026-01-01T00:00:00", "source_path": "/x/y.pt",
           "stored_filename": "y.pt", "task_family": "obb", "usage_role": "primary"}
    e = read_legacy_entry("obb/y.pt", raw)
    assert e.task_family == "obb" and e.backend == "yolo"
    assert e.notes == "v1" and e.size == "s" and e.usage_role == "primary"


def test_read_legacy_infers_family_and_backend_from_path():
    e = read_legacy_entry("pose/ViTPose/best.pt", {})
    assert e.task_family == "pose" and e.backend == "vitpose"


def test_read_legacy_unknown_keys_go_to_extra():
    e = read_legacy_entry("classification/z.pt", {"arch": "resnet", "weird": 1})
    assert e.extra.get("arch") == "resnet" and e.extra.get("weird") == 1
```

- [ ] **Step 2: Run tests, verify RED**

Run: `PYTHONPATH=$PWD/src python -m pytest tests/test_model_registry_core.py -v` → FAIL (module/symbols not defined).

- [ ] **Step 3: Implement `model_registry.py`**

Write the dataclass, `SCHEMA_VERSION = 3`, `registry_path()` (import `get_models_dir` from `hydra_suite.paths`; replicate the `model_publish` project-root override — READ `training/model_publish.py::get_models_root`/`_use_project_root_override` first and reuse the same mechanism/env hook so both resolve identically), `load_registry`/`save_registry` (atomic via temp + `os.replace`), and `read_legacy_entry`. Import stdlib + `hydra_suite.paths` ONLY — no app-layer imports.

- [ ] **Step 4: Run tests, verify GREEN**

Run: `PYTHONPATH=$PWD/src python -m pytest tests/test_model_registry_core.py -v` → PASS.

- [ ] **Step 5: Commit**

```bash
git add src/hydra_suite/data/model_registry.py tests/test_model_registry_core.py
git commit -m "feat(data): unified model_registry module — schema, load/save, legacy read"
```

---

### Task 2: `register_model` / `unregister_model` / `get_entry` / `list_entries`

**Files:**
- Modify: `src/hydra_suite/data/model_registry.py`
- Test: `tests/test_model_registry_ops.py`

**Interfaces:**
- Consumes: `load_registry`/`save_registry`/`ModelRegistryEntry` (Task 1).
- Produces:
  - `register_model(relpath: str, entry: ModelRegistryEntry) -> None` — upsert + save.
  - `unregister_model(relpath: str) -> bool` — remove if present + save; False if absent.
  - `get_entry(relpath: str) -> ModelRegistryEntry | None`.
  - `list_entries(task_family: str | None = None, backend: str | None = None) -> dict[str, ModelRegistryEntry]` — filtered view.
  - Keys are always models-root-relative POSIX paths — add `normalize_relpath(path) -> str` and use it in every op so absolute/relative/backslash inputs converge.

- [ ] **Step 1: Write failing tests**

```python
# tests/test_model_registry_ops.py
import importlib, pytest


@pytest.fixture
def mr(tmp_path, monkeypatch):
    monkeypatch.setenv("HYDRA_DATA_DIR", str(tmp_path))
    import hydra_suite.paths as paths
    importlib.reload(paths)
    import hydra_suite.data.model_registry as m
    importlib.reload(m)
    return m


def test_register_get_roundtrip(mr):
    e = mr.ModelRegistryEntry(task_family="pose", backend="sleap", species="ant")
    mr.register_model("pose/SLEAP/m", e)
    assert mr.get_entry("pose/SLEAP/m").species == "ant"


def test_register_upsert_overwrites(mr):
    mr.register_model("obb/a.pt", mr.ModelRegistryEntry(task_family="obb", backend="yolo", species="v1"))
    mr.register_model("obb/a.pt", mr.ModelRegistryEntry(task_family="obb", backend="yolo", species="v2"))
    assert mr.get_entry("obb/a.pt").species == "v2"
    assert len(mr.load_registry()) == 1


def test_unregister(mr):
    mr.register_model("obb/a.pt", mr.ModelRegistryEntry(task_family="obb", backend="yolo"))
    assert mr.unregister_model("obb/a.pt") is True
    assert mr.unregister_model("obb/a.pt") is False
    assert mr.get_entry("obb/a.pt") is None


def test_list_entries_filters(mr):
    mr.register_model("obb/a.pt", mr.ModelRegistryEntry(task_family="obb", backend="yolo"))
    mr.register_model("pose/ViTPose/b.pt", mr.ModelRegistryEntry(task_family="pose", backend="vitpose"))
    mr.register_model("pose/SLEAP/c", mr.ModelRegistryEntry(task_family="pose", backend="sleap"))
    assert set(mr.list_entries(task_family="pose")) == {"pose/ViTPose/b.pt", "pose/SLEAP/c"}
    assert set(mr.list_entries(backend="vitpose")) == {"pose/ViTPose/b.pt"}
```

- [ ] **Step 2: RED** — `pytest tests/test_model_registry_ops.py -v`.
- [ ] **Step 3: Implement** the four ops + `normalize_relpath` on top of load/save.
- [ ] **Step 4: GREEN.**
- [ ] **Step 5: Commit** — `feat(data): registry ops register/unregister/get/list_entries`.

---

### Task 3: `find_unregistered` + `entry_is_stale`

**Files:**
- Modify: `src/hydra_suite/data/model_registry.py`
- Test: `tests/test_model_registry_discovery.py`

**Interfaces:**
- Produces:
  - `find_unregistered(models_root: Path | None = None) -> list[str]` — scan the model dirs (`obb/`, `detection/`, `classification/`, `pose/YOLO`, `pose/SLEAP`, `pose/ViTPose`) for `.pt`/`.pth` files and SLEAP model dirs, return models-root-relative paths that have NO registry entry. Default root = `get_models_dir()`.
  - `entry_is_stale(relpath: str, models_root: Path | None = None) -> bool` — True if the registered path no longer exists on disk.

- [ ] **Step 1: Write failing tests** (create files under a tmp models root; register one, leave one unregistered; assert `find_unregistered` returns exactly the unregistered one; register a path then delete the file and assert `entry_is_stale` is True). Use the `HYDRA_DATA_DIR` + reload fixture pattern from Task 2.
- [ ] **Step 2: RED.**
- [ ] **Step 3: Implement** the scan (know the SLEAP case: a model is a directory, not a file) and stale check.
- [ ] **Step 4: GREEN.**
- [ ] **Step 5: Commit** — `feat(data): find_unregistered + entry_is_stale discovery helpers`.

---

## Phase 2 — Retire old APIs, repoint callers

### Task 4: Repoint Training-side (`model_publish`) to `data.model_registry`; delete old

**Files:**
- Modify: `src/hydra_suite/training/model_publish.py` (`_registry_path`, `load_model_registry`, `save_model_registry`, and the registry read/write inside `publish_trained_model` at ~515-537, 631, 662, 832-837)
- Modify/Migrate tests: `tests/test_model_registry_helpers.py`, `tests/test_model_publish_slice_geometry.py`

**Interfaces:**
- Consumes: `data.model_registry.{load_registry, register_model, get_entry, ModelRegistryEntry}`.
- Produces: `publish_trained_model` still returns `(registry_key, absolute_model_path)` and writes the same on-disk result, but via `register_model`. `load_model_registry`/`save_model_registry`/`_registry_path` DELETED from `model_publish`.

- [ ] **Step 1: Read** `publish_trained_model` fully (703-end) and the 4 registry-touch sites to see exactly what entry dict it builds (arch/description/classifier_v2/slice_geometry fields). Map those into `ModelRegistryEntry` (rich classifier/slice fields go into `extra` unless they have a first-class column). Note: `_task_usage_for_role`/`_repo_dir_for_role` stay.
- [ ] **Step 2: Write/adjust failing tests** — `test_model_registry_helpers.py` currently asserts `save_model_registry`/`load_model_registry` behavior; rewrite it to assert the same round-trip through `data.model_registry` (import the new module). Keep a test proving `publish_trained_model` produces a registry entry readable via `get_entry`. Run to confirm RED against current code where appropriate.
- [ ] **Step 3: Implement** — replace the registry helpers' bodies' callers with `data.model_registry` calls; delete `load_model_registry`/`save_model_registry`/`_registry_path`; rewrite `publish_trained_model`'s registry section to build a `ModelRegistryEntry` and call `register_model`. Preserve the project-root override (now centralized in `data.model_registry.registry_path`).
- [ ] **Step 4: GREEN** — run `test_model_registry_helpers.py` + `test_model_publish_slice_geometry.py` + `pytest -k model_publish`.
- [ ] **Step 5: Commit** — `refactor(training): route model_publish through data.model_registry; drop local registry`.

---

### Task 5: Repoint GUI-side (`model_utils` + config + main_window); delete old

**Files:**
- Modify: `src/hydra_suite/trackerkit/gui/model_utils.py` (delete `load_yolo_model_registry`, `save_yolo_model_registry`, `register_yolo_model`, `unregister_yolo_model`, `get_yolo_model_registry_path`, `_extract_registry_entries`, `_normalize_yolo_model_metadata` if now unused; rewrite `get_yolo_model_metadata` + the registry-clearing part of `remove_model_from_repository` via `data.model_registry`)
- Modify: `src/hydra_suite/trackerkit/gui/main_window.py:76-79` (the 4 re-exports — drop or repoint)
- Modify: `src/hydra_suite/trackerkit/gui/orchestrators/config.py:39, 3991` (import + the `register_yolo_model(rel_path, metadata)` call → build `ModelRegistryEntry` + `register_model`)
- Test: `tests/test_model_registry_gui_import.py`

**Interfaces:**
- Consumes: `data.model_registry.{register_model, get_entry, unregister_model, ModelRegistryEntry, normalize_relpath}`.
- Produces: `_import_yolo_model_to_repository` (config.py) registers via `register_model`; `remove_model_from_repository` clears via `unregister_model`; `get_yolo_model_metadata` reads via `get_entry`. Old `*_yolo_model_registry` symbols GONE.

- [ ] **Step 1: Read** the current `_import_yolo_model_to_repository` (config.py ~3868-3992) metadata dict and `remove_model_from_repository` (model_utils 306-345) to preserve behavior exactly (path safety checks in remove stay).
- [ ] **Step 2: Write failing test** — `test_model_registry_gui_import.py`: call `_import_yolo_model_to_repository` (or its metadata-building + register path) for an OBB model and assert `get_entry` returns an entry with the right `task_family`/`species`; call `remove_model_from_repository` and assert the entry is gone. Use a QApplication-less path if possible, else the MainWindow test harness. RED first.
- [ ] **Step 3: Implement** — repoint config.py:3991 to build a `ModelRegistryEntry(task_family=..., backend="yolo", size=..., species=..., notes=model_info, ...)` and call `register_model`; repoint `remove_model_from_repository` + `get_yolo_model_metadata`; delete the dead `model_utils` registry functions and the `main_window.py:76-79` re-exports (grep the codebase for any other reference first and repoint/remove).
- [ ] **Step 4: GREEN** + `grep -rn "register_yolo_model\|_yolo_model_registry" src/` returns nothing.
- [ ] **Step 5: Commit** — `refactor(trackerkit): route model imports through data.model_registry; delete legacy registry API`.

---

## Phase 3 — Extend registration to pose + publish role

### Task 6: Pose `TrainingRole` members + `publish_trained_model` pose support

**Files:**
- Modify: `src/hydra_suite/training/contracts.py` (`TrainingRole`: add `POSE_YOLO`, `POSE_SLEAP`, `POSE_VITPOSE`)
- Modify: `src/hydra_suite/training/model_publish.py` (`_repo_dir_for_role`, `_task_usage_for_role`, and any role-branching in `publish_trained_model` to handle pose → `pose/<Backend>/`, `task_family="pose"`, `backend` per role)
- Test: `tests/test_model_publish_pose.py`

**Interfaces:**
- Consumes: `data.model_registry`.
- Produces: `publish_trained_model(role=TrainingRole.POSE_VITPOSE, ...)` copies to `pose/ViTPose/` and registers `ModelRegistryEntry(task_family="pose", backend="vitpose", num_keypoints=..., ...)`. Accept `num_keypoints`/`skeleton_name` (thread a new optional kwarg or via existing `training_params`).

- [ ] **Step 1: Read** `_repo_dir_for_role`/`_task_usage_for_role` to see the mapping table; extend for the 3 pose roles.
- [ ] **Step 2: Write failing test** — publish a fake pose artifact with `role=POSE_VITPOSE`, assert it lands under `pose/ViTPose/` and `get_entry` shows `task_family="pose"`, `backend="vitpose"`, `num_keypoints` set. RED.
- [ ] **Step 3: Implement** the role additions + mappings.
- [ ] **Step 4: GREEN.**
- [ ] **Step 5: Commit** — `feat(training): pose TrainingRole members + pose publish support`.

---

### Task 7: GUI pose imports register all 3 backends; replace parity test

**Files:**
- Modify: `src/hydra_suite/trackerkit/gui/orchestrators/config.py` (`_import_pose_model_to_repository` ~4268-4415 — add a `register_model` call after the copy, for all three pose backends, with `task_family="pose"`, `backend∈{yolo,sleap,vitpose}`, `num_keypoints` where known)
- Delete + Replace: `tests/test_vitpose_registration_parity.py` → `tests/test_pose_registration.py`

**Interfaces:**
- Consumes: `data.model_registry.register_model`.
- Produces: importing any pose model writes a registry entry (reversing the old parity-lock decision).

- [ ] **Step 1: Read** `_import_pose_model_to_repository` to get the backend-key derivation + dest path + metadata dialog values already collected (species/type/info), and `num_keypoints` availability (may need project keypoint count — if unavailable at import, leave 0 and let the CLI/inspection fill it; do NOT block import).
- [ ] **Step 2: Write failing test** — `test_pose_registration.py`: importing a pose model (per backend) registers an entry with `task_family="pose"` and the right `backend`; assert `register_yolo_model`/parity-absence is no longer the contract. RED (no register call yet).
- [ ] **Step 3: Implement** the `register_model` call in the pose import path; delete `test_vitpose_registration_parity.py`.
- [ ] **Step 4: GREEN.**
- [ ] **Step 5: Commit** — `feat(trackerkit): register pose model imports (yolo/sleap/vitpose)`.

---

## Phase 4 — Pickers as source of truth (TrackerKit) + upgrade-cliff notice

### Task 8: Populate model combos from the registry; add migrate notice + stale handling

**Files:**
- Modify: `src/hydra_suite/trackerkit/gui/orchestrators/config.py` (`_populate_yolo_model_combo` ~3740-3790 and `_populate_pose_model_combo` ~3995-4025 — list from `list_entries(...)` instead of `os.listdir`; render metadata in the label; skip/grey stale entries)
- Modify: the panel/UI that hosts the combos to show a non-blocking notice when `find_unregistered()` is non-empty (locate the identity/detection panel that owns these combos)
- Test: `tests/test_registry_backed_pickers.py`

**Interfaces:**
- Consumes: `data.model_registry.{list_entries, find_unregistered, entry_is_stale, get_entry}`.
- Produces: combos list only registered models of the matching family/backend, labeled with species/date; unregistered models are absent but a notice is emitted.

- [ ] **Step 1: Read** both populate functions to see the exact combo item format (userData carries the path). Preserve `preferred_model_path` selection behavior. Determine where a non-blocking notice can live (a label/status line in the panel — match existing patterns like the SLEAP-env row).
- [ ] **Step 2: Write failing test** — build the combo population (a headless helper that returns the list of (label, path) it would add) fed a registry with 2 registered + 1 unregistered model; assert only registered appear, labels include metadata, and a `has_unregistered` flag/notice signal is set. Extract a pure `registry_items_for(task_family, backend)` helper so the assertion doesn't need a live Qt combo. RED.
- [ ] **Step 3: Implement** — swap the `os.listdir` bodies for `list_entries(...)`, add the `registry_items_for` helper + label formatting, skip stale entries, and wire the notice. Keep selection/preferred-path behavior.
- [ ] **Step 4: GREEN** + a quick manual note in the report (combos still select correctly).
- [ ] **Step 5: Commit** — `feat(trackerkit): registry-backed model pickers + migrate notice`.

---

## Phase 5 — CLI migration script

### Task 9: `scripts/migrate_model_registry.py` (interactive + `--auto`, legacy upgrade, stale prune)

**Files:**
- Create: `scripts/migrate_model_registry.py` (also runnable as `python -m hydra_suite.data.migrate_model_registry` — put the logic in `src/hydra_suite/data/migrate_model_registry.py` with a thin `scripts/` shim, so it's importable/testable)
- Test: `tests/test_migrate_model_registry.py`

**Interfaces:**
- Consumes: `data.model_registry.{find_unregistered, register_model, get_entry, load_registry, save_registry, entry_is_stale, ModelRegistryEntry, read_legacy_entry}`.
- Produces: `run_migration(models_root=None, auto=False, prompt_fn=input, prune_stale=False) -> MigrationSummary` — the pure core the CLI wraps; `main(argv)` parses `--auto`, `--prune-stale`.

- [ ] **Step 1: Write failing tests**

```python
# tests/test_migrate_model_registry.py
import importlib, pytest
from pathlib import Path


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.setenv("HYDRA_DATA_DIR", str(tmp_path))
    import hydra_suite.paths as paths
    importlib.reload(paths)
    import hydra_suite.data.model_registry as mr
    importlib.reload(mr)
    import hydra_suite.data.migrate_model_registry as mig
    importlib.reload(mig)
    root = paths.get_models_dir()
    (root / "obb").mkdir(parents=True, exist_ok=True)
    (root / "obb" / "m.pt").write_bytes(b"x")
    return mr, mig, root


def test_auto_backfills_unregistered_flagged_needs_review(env):
    mr, mig, root = env
    summary = mig.run_migration(auto=True)
    e = mr.get_entry("obb/m.pt")
    assert e is not None and e.task_family == "obb" and e.needs_review is True
    assert summary.registered == 1


def test_idempotent_rerun_registers_nothing_new(env):
    mr, mig, root = env
    mig.run_migration(auto=True)
    summary2 = mig.run_migration(auto=True)
    assert summary2.registered == 0


def test_interactive_uses_prompt_answers(env):
    mr, mig, root = env
    answers = iter(["ant", "primary", ""])  # species, role, notes (match your prompt order)
    e_summary = mig.run_migration(auto=False, prompt_fn=lambda _="": next(answers))
    e = mr.get_entry("obb/m.pt")
    assert e.species == "ant" and e.needs_review is False


def test_prune_stale_removes_missing_file_entries(env):
    mr, mig, root = env
    mr.register_model("obb/gone.pt", mr.ModelRegistryEntry(task_family="obb", backend="yolo"))
    summary = mig.run_migration(auto=True, prune_stale=True)
    assert mr.get_entry("obb/gone.pt") is None
    assert summary.pruned == 1
```

- [ ] **Step 2: RED** — `pytest tests/test_migrate_model_registry.py -v`.
- [ ] **Step 3: Implement** `run_migration` + `MigrationSummary` (registered/skipped/pruned/upgraded counts), filename-parse guesses for interactive pre-fill, `--auto` sets `needs_review=True`, legacy-entry upgrade (re-save through `load_registry`/`save_registry` bumps schema), stale prune. Add `main(argv)` with argparse and the `scripts/` shim. `prompt_fn` defaults to `input` so tests inject answers.
- [ ] **Step 4: GREEN.**
- [ ] **Step 5: Commit** — `feat(data): interactive model-registry migration CLI`.

---

## Final Verification (after all tasks)

- [ ] `grep -rn "register_yolo_model\|load_yolo_model_registry\|save_yolo_model_registry\|def load_model_registry\|def save_model_registry" src/` → nothing (old APIs fully retired).
- [ ] Full registry test set: `PYTHONPATH=$PWD/src python -m pytest tests/ -k "registry or model_publish or migrate or pose_registration" -v --continue-on-collection-errors` → all green (delta gate).
- [ ] `make lint-moderate` on touched files — no new issues.
- [ ] Import-cycle / layering check: `python -c "import hydra_suite.data.model_registry"` from a clean env, and confirm `model_registry.py` imports no app-layer package (grep its imports).
- [ ] Manual smoke (report, not automated): run `python -m hydra_suite.data.migrate_model_registry --auto` against a scratch `HYDRA_DATA_DIR` with a couple of dummy models; confirm entries appear; open TrackerKit combos conceptually list from the registry.

## Self-Review Checklist (plan author ran this)

- **Spec coverage:** unify APIs (Tasks 4,5) ✓; registry-driven pickers (Task 8) ✓; pose in registry + publish (Tasks 6,7) ✓; CLI backfill (Task 9) ✓; retire-not-shim (Tasks 4,5 delete) ✓; legacy read of both formats (Task 1) ✓; upgrade-cliff notice (Task 8) ✓; stale handling (Tasks 3,8,9) ✓.
- **Dependency direction:** `data/model_registry.py` imports only stdlib + `hydra_suite.paths` (Global Constraints + Task 1 Step 3 + final layering check).
- **Ordering:** module (P1) → callers repointed (P2) → pose extension (P3) → pickers (P4) → CLI (P5). Each phase builds on the prior. Task 8 (pickers) depends on entries existing (Tasks 5-7); Task 9 (CLI) depends on the full module (P1) + registration story.
- **Type consistency:** `ModelRegistryEntry`, `register_model`, `list_entries`, `find_unregistered`, `entry_is_stale`, `read_legacy_entry`, `normalize_relpath`, `run_migration`/`MigrationSummary` each defined once and referenced consistently.
- **Known reversal:** Task 7 deletes `test_vitpose_registration_parity.py` (the pose-parity lock is intentionally superseded).
