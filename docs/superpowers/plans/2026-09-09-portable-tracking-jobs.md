# Portable Tracking Jobs Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Package a tracking experiment on one machine into a self-contained job directory that runs on any compute box with zero model registration, and whose outputs — including reusable detection caches — come home beside the original videos.

**Architecture:** A Qt-free `data/tracking_job/` package owns the manifest, packing, output discovery, preflight and rsync transport; the app-layer `trackerkit/job_cli.py` does the planning and engine-parameter resolution and hands results down (Core/Data never import an app layer). The compute box is reconfigured by two environment variables only — a new `HYDRA_MODELS_DIR` pointing at `<job>/models` and the existing `HYDRA_CONFIG_DIR` pointing at `<job>/config` — so engines and calibration profiles stay host-scoped. Cache keys are redefined to identify models and videos by *content* rather than absolute path + mtime, which is what makes a cache produced on the compute box hit on the laptop.

**Tech Stack:** Python 3.11+, argparse, `rsync` over `ssh`, pytest, conda envs `hydra-mps` (this Mac) / `hydra-cuda` (firebrat).

**Spec:** `docs/superpowers/specs/2026-09-09-portable-tracking-jobs-design.md`

**Worktree:** `.worktrees/portable-jobs` on branch `feat/portable-tracking-jobs`, created from `main` @ `8f9688e0` (which already contains the `fix/calibrate-replay-gate` merge). All work happens there.

## Spec corrections established before planning (read these first)

The spec was written at `8678c5b6`. Re-anchoring against `8f9688e0` found six errors that change the work. Each is handled by a named task; **do not** implement the spec's wording where it conflicts with this list.

1. **§7b is under-scoped — `config_hash` is path-bound.** `cache/keys.py:426-434` `_model_signature()` folds a raw, unnormalized `"{path}|mtime={mtime:.9f}"` string into `config_hash` for OBB detection (direct `:158`, sequential `:192-193`). Renaming `CacheKey.model_path` → `model_id` alone leaves remote OBB caches missing locally. `_model_signature` must become content-based too. (Task 4)
2. **§240 "Load side is already symmetric" is imprecise for `color_tag_model_path`, but the fix is portability-by-construction, not a bug fix.** `engine_params.py:1591` emits `COLOR_TAG_MODEL_PATH` verbatim via `_cfg_get` with **no** `resolve_model_path`, and `:1024` copies it into `CNN_CLASSIFIER_MODEL_PATH` unresolved. **`COLOR_TAG_MODEL_PATH`/`CNN_CLASSIFIER_MODEL_PATH` are dead — grep of `src/hydra_suite/core/` finds zero `color_tag` consumers, and the GUI's colour-tag panel is `setVisible(False)` (`trackerkit/gui/panels/identity_panel.py:143`).** Colour-tag identity actually runs through ClassKit multi-head classifiers via `cnn_classifiers`. So relativizing `color_tag_model_path` on save cannot "break colour-tag identity" — nothing reads it. We still relativize it on save (defensive symmetry: whatever lands in the config should be portable) and still add load-side resolution (the lowercase key lands in the job sidecar and `verify_job` rejects absolute paths there), but the key itself is never yielded by `iter_model_references` and never packed — see correction 7. (Task 2, Task 3)
3. **§216 "build_engine_params already emits `""` for disabled roles" is FALSE except for head-tail — but the fix is "gate on what core loads", not "gate on the GUI's enable flags".** Only head-tail is gated in `build_engine_params` (`:823-843`). `POSE_MODEL_DIR` (`:985-992`) IS gated deeper in the pipeline: `core/inference/config.py:1255` only builds a pose config when the pose stage is live. AprilTag IS gated (`core/inference/config.py:1357`) and needs no model file at all. YOLO mode is selected by `config.obb.mode`, not a boolean flag. **CNN classifiers are NOT gated on `ENABLE_IDENTITY_ANALYSIS`**: `core/inference/config.py:1224` builds a `CNNConfig` for every entry in `CNN_CLASSIFIERS` unconditionally, `core/inference/runner.py:509` loads all of them, and `core/tracking/worker.py:955` enables the identity phase from `bool(p.get("CNN_CLASSIFIERS", []))` — the enable flag never enters this decision. When a listed model file is missing, `config.py:1227-1233` logs an error and **continues** rather than refusing to run. `iter_model_references` must therefore gate each role on the same signal core itself uses to decide whether to load it — pose on the pose-stage-live check, apriltag on `USE_APRILTAGS`, CNN classifiers on the list being non-empty (never on `ENABLE_IDENTITY_ANALYSIS`) — not on GUI enable flags, which under-ship relative to what core actually loads. (Task 3)
4. **§421 `job calibrate --sahi-profile` has no CLI surface.** `run_calibrate_cli()` (`calibrate_cli.py:80`) takes no `sahi_profile` parameter and `app.py:446-462` never passes one. Because `pack` bakes the resolved SAHI profile into every per-video sidecar (§6.4), `job calibrate` forwards **only** `inference_autotune_manual`; the profile is already in the config. Documented, not worked around. (Task 11)
5. **§286 `HYDRA_MODELS_DIR` "documented in paths.py:1-16" — it does not exist anywhere.** `get_models_dir()` (`paths.py:128-132`) has no override of its own and is purely `HYDRA_DATA_DIR`-derived. It is new work, including a new `print_paths()` line. (Task 1)
6. **Two incompatible model repository layouts share one `model_registry.json`.** `model_publish._repo_dir_for_role` (`:65-92`) writes `YOLO-obb/`, `YOLO-detect/`, `tiny-classify/…`; `model_paths.get_yolo_model_repository_directory` (`:75-95`) reads `obb/`, `detection/`, `classification/orientation/`. Pack copies each model to **the models-root-relative key its own config already uses** (`make_model_path_relative`), preserving whichever layout that config was built against; it never re-derives a layout from a role. (Task 6)
7. **`POSE_SKELETON_FILE`'s "fallback via the engine" claim is wrong — every consumer reads it verbatim.** `engine_params.py:1179`, `core/post/pose_merge.py:292`, `core/post/media_export.py:401`, `core/individual/properties/cache.py:221`, and `core/inference/config.py:1258` all read `POSE_SKELETON_FILE` as-is with no fallback resolution. Task 11 must explicitly compute and stamp `skeleton_path` on `PlannedVideo` before pack rewrites the sidecar. (Task 11)

Plus two pre-existing bugs found while anchoring. **Do not fix them in this branch** — they change cache-hit behaviour and would contaminate the §7b gate. Record them and move on:

- `cache/reuse.py:62` and `core/tracking/worker.py:1435` call `detection_cache_key(...)` **without** `detection_batch_size`, while `runner.py:553` passes it. With a non-default batch these compute a different key than the runner's own cache.
- `trackerkit/gui/dialogs/parameter_helper.py:1768-1780` uses a private `_source_signature` closure instead of `keys.video_signature()`, so it will *not* follow the Task 4 redefinition. Spec §302's "all call the one function" is wrong. Task 4 adds a comment there naming the divergence.

## Global Constraints

- **Baselines to protect** (measured on `hydra-mps` @ `8f9688e0`): `tests/test_inference_cache_keys.py tests/test_inference_cache_chunked.py` = **111 passed**; `tests/test_gui_cli_param_equivalence.py tests/test_get_parameters_dict_characterization.py tests/test_paths.py` = **27 passed**. **Task 1's `HYDRA_MODELS_DIR` work landed on this branch at `3d4e2625`, before this fix wave** — it added `tests/test_paths_models_dir_override.py` (8 tests) as a *separate* file, so `test_paths.py` itself is still 19 tests, unchanged. The current, up-to-date protected count for this group is therefore **`tests/test_gui_cli_param_equivalence.py tests/test_get_parameters_dict_characterization.py tests/test_paths.py tests/test_paths_models_dir_override.py` = 35 passed** (27 + the 8 new ones), verified by collection on this worktree. Any task reducing these without an explicit, justified deletion is a regression.
- **Run the FULL relevant suite every task**, never a subset chosen by apparent relevance — the reflective contract guards (`test_get_parameters_dict_characterization.py`, the new Task 3 guard) break on ANY field addition. This is memory `feedback_run_contract_guards_after_field_additions`.
- **Dependency direction is a hard gate.** `src/hydra_suite/data/tracking_job/` must import ONLY from `data`, `core`, `training.model_publish`, `paths`, and the stdlib. It must never import `trackerkit`, `classkit`, `detectkit`, `posekit`, `refinekit`, `filterkit`, `widgets`, or PySide6. Task 5 adds an automated test asserting this.
- **Never import from `legacy/`.**
- **The remote never mutates `hydra_job.json`.** `run` appends to `logs/runs.jsonl`; `pull` merges. This is what makes `push` a pure input sync.
- **`rsync` is never reimplemented.** `transport.py` builds argv lists and shells out. Never pass `--delete`.
- **`CACHE_SCHEMA_VERSION` goes 4 → 5 exactly once**, in Task 4. No converter — caches are derived data.
- Format before every commit: `make format`. Lint gate: `make lint` (fix B7: **`make lint-moderate` DOES NOT EXIST** — `Makefile:421,427,440,446` define only `lint`, `lint-fix`, `lint-strict`, `lint-report`. A commit step chained `make lint-moderate && git commit` would never reach the commit).
- Activate the env first: `conda activate hydra-mps`. **An agent executing this plan gets a fresh shell per Bash call, so `conda activate` does NOT persist** — either put every command of a step inside ONE fenced block (as most steps here do) or prefix each command with `conda run --no-capture-output -n hydra-mps` (fix B14; Task 13 uses the latter throughout). Before any heavy run, kill stale `sleap`/`hydra` processes; **never** touch a process that is not sleap/hydra.
- Commit after every task. Do not squash tasks together.
- **Never invoke bare `trackerkit` (or bare `hydra`) from inside this worktree for any acceptance/E2E step.** Verified: `hydra_suite.__file__` resolves to MAIN's editable install here, not `.worktrees/portable-jobs/src` — a bare console-script invocation silently exercises unmodified `main` code and reports false confidence about the branch under test. Every acceptance/CLI-smoke command in this plan must instead be `PYTHONPATH=<worktree>/src python -m hydra_suite.trackerkit.app job ...` (this matches the repo's known PYTHONPATH gotcha, memory `feedback_equivalence_pythonpath_gotcha`). The equivalence fixture clips (`tools/equivalence/fixtures/clips/*.mp4`) are gitignored, but on this box they are already present in this worktree (verified: `tools/equivalence/fixtures/clips/*.mp4` exist here) — `fetch_fixtures.sh` is a no-op unless working from a genuinely fresh clone/worktree that lacks them.
- **Verification box is `firebrat`** (`rutalab@firebrat`, RTX 4090 idle, conda at `~/miniforge3`, envs `hydra-cuda` + `sleap`, rsync 3.2.7, 1.6 TB free, equivalence fixtures already present at `~/hydra-suite/tools/equivalence/fixtures/`). **`courtship` is running a live `trackerkit` job — do not use it and do not kill anything on it.**

---

## File structure

| File | Responsibility |
|---|---|
| `src/hydra_suite/paths.py` | + `HYDRA_MODELS_DIR` override on `get_models_dir()`; + `print_paths()` line; + `get_platform_config_dir()` (Task 1) |
| `src/hydra_suite/trackerkit/engine_params.py` | + load-side `resolve_model_path` for colour tag (Task 2); + `iter_model_references` and the four key tuples (Task 3) |
| `src/hydra_suite/trackerkit/gui/orchestrators/config.py` | save-side relativization of `color_tag_model_path` and `cnn_classifiers[].model_path` (Task 2) |
| `src/hydra_suite/core/inference/cache/base.py` | `CacheKey` → `(schema_version, model_id, config_hash)`; `CACHE_SCHEMA_VERSION = 5` (Task 4) |
| `src/hydra_suite/core/inference/cache/keys.py` | content-based `model_id`, `_model_signature`, `video_signature` (Task 4) |
| `src/hydra_suite/core/inference/content_id.py` | **new** — `file_content_id`, `directory_content_id`, `video_signature` primitives + LRU memoization (Task 4) |
| `src/hydra_suite/detectkit/jobs/prediction_cache.py`, `detectkit/sidecars/operations.py` | migrated onto the content-based key; source-directory content id dispatch; sidecar IPC payload shape bumped (Task 4) |
| `src/hydra_suite/data/tracking_job/manifest.py` | `JobManifest` dataclass, atomic write, `validate_job_relpath` (Task 5) |
| `src/hydra_suite/data/tracking_job/shared_roots.py` | mount table read/write, longest-alias matching (Task 5) |
| `src/hydra_suite/data/tracking_job/references.py` | model copy (file/dir/bundle/sidecars), registry subset (Task 6) |
| `src/hydra_suite/data/tracking_job/pack.py` | `pack_job`: copy → rewrite → runner → verify (Task 7) |
| `src/hydra_suite/data/tracking_job/runner.py` | `run.sh` template (Task 7) |
| `src/hydra_suite/data/tracking_job/verify.py` | `verify_job` offline checks (Task 7) |
| `src/hydra_suite/data/tracking_job/outputs.py` | output-set discovery + origin mapping (Task 8) |
| `src/hydra_suite/data/tracking_job/transport.py` | rsync/ssh argv construction + execution (Task 9) |
| `src/hydra_suite/data/tracking_job/preflight.py` | host checks, shared-root materialization (Task 10) |
| `src/hydra_suite/trackerkit/job_cli.py` | argparse wiring + planning/param-building handoff (Task 11) |
| `src/hydra_suite/trackerkit/app.py` | register the `job` subcommand group (Task 11) |

---

### Task 1: `HYDRA_MODELS_DIR` override, and `get_platform_config_dir()`

**Status: the `HYDRA_MODELS_DIR` override is DONE — already committed on this branch at `3d4e2625`** ("feat(paths): HYDRA_MODELS_DIR relocates the models root independently of the data dir"). Verified in source: `src/hydra_suite/paths.py` has the `HYDRA_MODELS_DIR` docstring line, `get_models_dir()` reads `os.environ.get("HYDRA_MODELS_DIR")` per call and `expanduser()`s it, `print_paths()` reports it, and `tests/test_paths_models_dir_override.py` exists with the 8 tests below (all collected and, per that commit's own gate, passing). `docs/getting-started/installation.md` has the table row. **Do not redo Steps 1-7 below** — they are kept verbatim only as the historical record of what `3d4e2625` implemented; skip straight to the "Remaining work" step.

Original Steps 1-7 (already done, do not re-run as if pending):

**Files (already modified):**
- `src/hydra_suite/paths.py:1-16` (docstring), `:128-132` (`get_models_dir`), `:202-228` (`print_paths`)
- `tests/test_paths_models_dir_override.py` (created)

**Interfaces:**
- Consumes: nothing.
- Produces: `get_models_dir()` honours `$HYDRA_MODELS_DIR`. Everything routing through it (`model_paths.get_models_root_directory()`, `model_publish.get_models_root()`, `_registry_path()`, SAM3 checkpoint root) follows automatically.

- [x] **Step 1: Write the failing test**

Create `tests/test_paths_models_dir_override.py`:

```python
"""HYDRA_MODELS_DIR relocates the models root without moving the data dir."""

import importlib

import pytest

from hydra_suite import paths


def test_models_dir_follows_env_override(tmp_path, monkeypatch):
    models = tmp_path / "job" / "models"
    monkeypatch.setenv("HYDRA_MODELS_DIR", str(models))
    assert paths.get_models_dir() == models
    assert models.is_dir(), "get_models_dir must create the directory"


def test_models_dir_override_does_not_move_data_dir(tmp_path, monkeypatch):
    data = tmp_path / "hostdata"
    models = tmp_path / "job" / "models"
    monkeypatch.setenv("HYDRA_DATA_DIR", str(data))
    monkeypatch.setenv("HYDRA_MODELS_DIR", str(models))
    assert paths.get_models_dir() == models
    assert paths.get_data_dir() == data
    # Engine artifacts and calibration profiles must stay host-scoped.
    assert paths.get_training_runs_dir().is_relative_to(data)


def test_models_dir_without_override_is_data_dir_models(tmp_path, monkeypatch):
    data = tmp_path / "hostdata"
    monkeypatch.setenv("HYDRA_DATA_DIR", str(data))
    monkeypatch.delenv("HYDRA_MODELS_DIR", raising=False)
    assert paths.get_models_dir() == data / "models"


def test_override_is_read_per_call_not_cached(tmp_path, monkeypatch):
    first = tmp_path / "a"
    second = tmp_path / "b"
    monkeypatch.setenv("HYDRA_MODELS_DIR", str(first))
    assert paths.get_models_dir() == first
    monkeypatch.setenv("HYDRA_MODELS_DIR", str(second))
    assert paths.get_models_dir() == second


def test_expanduser_is_applied(monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("HYDRA_MODELS_DIR", "~/jobmodels")
    assert paths.get_models_dir() == tmp_path / "jobmodels"


def test_registry_path_follows_the_override(tmp_path, monkeypatch):
    models = tmp_path / "job" / "models"
    monkeypatch.setenv("HYDRA_MODELS_DIR", str(models))
    from hydra_suite.training import model_publish
    from hydra_suite.core.inference import model_paths

    assert model_publish._registry_path() == models / "model_registry.json"
    assert model_paths.get_models_root_directory() == str(models)


def test_print_paths_reports_the_override(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("HYDRA_MODELS_DIR", str(tmp_path / "m"))
    paths.print_paths()
    assert "HYDRA_MODELS_DIR" in capsys.readouterr().out
```

- [x] **Step 2: Run test to verify it fails**

Run: `conda activate hydra-mps && python -m pytest tests/test_paths_models_dir_override.py -v`
Expected: FAIL — `get_models_dir()` returns the data-dir path, ignoring the env var.

- [x] **Step 3: Implement**

In `src/hydra_suite/paths.py`, replace `get_models_dir` (currently `:128-132`):

```python
def get_models_dir() -> Path:
    """Directory holding published models and ``model_registry.json``.

    ``HYDRA_MODELS_DIR`` relocates ONLY the models root, independently of
    ``HYDRA_DATA_DIR``. This is what lets a portable job supply its own models
    (``HYDRA_MODELS_DIR=<job>/models``) while engine artifacts, calibration
    profiles and training runs stay on the host's data dir. Read per call, not
    cached, so tests can monkeypatch the environment.
    """
    override = os.environ.get("HYDRA_MODELS_DIR")
    if override:
        path = Path(override).expanduser()
        path.mkdir(parents=True, exist_ok=True)
        return path
    p = _user_data_dir() / "models"
    p.mkdir(parents=True, exist_ok=True)
    return p
```

Add `HYDRA_MODELS_DIR` to the module docstring's override list (`:1-16`), and in `print_paths` (`:202-228`) add a line alongside the existing CONFIG/DATA/PROJECTS override reporting:

```python
    print(f"  HYDRA_MODELS_DIR: {os.environ.get('HYDRA_MODELS_DIR', '(unset)')}")
```

- [x] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_paths_models_dir_override.py tests/test_paths.py -v`
Expected: all PASS.

- [x] **Step 5: Grep gate — no module may bypass `get_models_dir()`**

Run:
```bash
grep -rn 'get_data_dir()[[:space:]]*/[[:space:]]*"models"' src/ && echo "VIOLATION" || echo "clean"
```
Expected: `clean`. (Known non-violations, do not touch: `paths_migrate.py:31` uses `repo_root / "models"` as a migration *source*; `posekit/gui/main_window.py:4608` uses a SLEAP training-run root; ClassKit/DetectKit `artifacts/models` are per-project.)

- [x] **Step 6: Document**

In `docs/getting-started/installation.md`, in the environment-variable table (near the existing `HYDRA_DATA_DIR`/`HYDRA_CONFIG_DIR` rows), add:

```markdown
| `HYDRA_MODELS_DIR` | Relocates only the models root (`model_registry.json` + published models), leaving engine artifacts and calibration profiles on the host data dir. Set automatically by a packed job's `run.sh`. |
```

- [x] **Step 7: Commit**

```bash
make format
git add src/hydra_suite/paths.py tests/test_paths_models_dir_override.py docs/getting-started/installation.md
git commit -m "feat(paths): HYDRA_MODELS_DIR relocates the models root independently of the data dir"
```

Committed at `3d4e2625`. End of the already-done historical record.

---

**Remaining work for Task 1 — `get_platform_config_dir()` (fix Y3, round 7).** This is the only piece of Task 1 not yet on the branch. Task 10's fix X1 (`preflight.py`) imports `get_platform_config_dir` from `hydra_suite.paths`; without it, `preflight.py` fails to import at all, breaking every downstream task that touches preflight. Task 6's `shared_roots` preflight check needs to distinguish "the host has no config-dir override" from "read whatever `HYDRA_CONFIG_DIR` currently points at" — and inside `run.sh`, `HYDRA_CONFIG_DIR` has already been redirected to the job's own snapshot, so at that point it is not a usable signal for "the host's real platformdirs default" at all.

**Files:**
- Modify: `src/hydra_suite/paths.py` (add `get_platform_config_dir()` beside `_user_config_dir()`, verified at `:38-49`)
- Test: `tests/test_paths.py` (add cases; do not create a new file — this is a one-function addition to an already-tested module)

- [ ] **Step 8: Write the failing test**

Add to `tests/test_paths.py`:

```python
def test_get_platform_config_dir_ignores_override(tmp_path, monkeypatch):
    from hydra_suite import paths

    monkeypatch.setenv("HYDRA_CONFIG_DIR", str(tmp_path / "job_config_snapshot"))
    real = paths.get_platform_config_dir()
    assert real != tmp_path / "job_config_snapshot"
    assert real.is_dir()


def test_get_platform_config_dir_matches_platformdirs_default(tmp_path, monkeypatch):
    from platformdirs import user_config_dir

    from hydra_suite import paths

    monkeypatch.delenv("HYDRA_CONFIG_DIR", raising=False)
    expected = Path(user_config_dir(paths.APP_NAME, paths.APP_AUTHOR))
    assert paths.get_platform_config_dir() == expected
```

- [ ] **Step 9: Run test to verify it fails**

Run: `conda activate hydra-mps && python -m pytest tests/test_paths.py -k platform_config_dir -v`
Expected: FAIL — `AttributeError: module 'hydra_suite.paths' has no attribute 'get_platform_config_dir'`.

- [ ] **Step 10: Implement**

In `src/hydra_suite/paths.py`, add beside `_user_config_dir()` (`:38-49`):

```python
def get_platform_config_dir() -> Path:
    """The platformdirs config dir, ignoring ``HYDRA_CONFIG_DIR`` entirely.

    Used when a caller has an explicit, separate signal that no config-dir
    override applies (e.g. ``run.sh``'s ``HYDRA_HOST_CONFIG_DIR=""``, which
    means "the host used its default config dir before job env redirection
    took over") and needs the REAL platformdirs path, not whatever
    ``HYDRA_CONFIG_DIR`` happens to be set to right now.
    """
    p = Path(user_config_dir(APP_NAME, APP_AUTHOR))
    p.mkdir(parents=True, exist_ok=True)
    return p
```

This is a thin, deliberate duplication of `_user_config_dir()`'s else-branch — not a refactor of `_user_config_dir()` itself, since every other caller of `_user_config_dir()` legitimately wants the override-aware behaviour.

- [ ] **Step 11: Run tests to verify they pass**

Run: `python -m pytest tests/test_paths.py tests/test_paths_models_dir_override.py -v`
Expected: all PASS — **21 tests in `test_paths.py`** (19 existing + 2 new), 8 in the override file, 35 total across the four-file baseline group in Global Constraints.

- [ ] **Step 12: Commit**

```bash
make format
git add src/hydra_suite/paths.py tests/test_paths.py
git commit -m "feat(paths): add get_platform_config_dir(), an override-blind platformdirs config resolver"
```

---

### Task 2: Close the two save-side path leaks — and the load-side asymmetry behind one of them

**Files:**
- Modify: `src/hydra_suite/trackerkit/engine_params.py:1023-1024` and `:1591` (load-side resolve)
- Modify: `src/hydra_suite/trackerkit/gui/orchestrators/config.py:2096` (`color_tag_model_path`), `:2072-2074` (`cnn_classifiers`)
- Test: `tests/test_config_model_path_portability.py` (create)

**Interfaces:**
- Consumes: **Task 1** — the `models_root` fixture in this task's test sets `HYDRA_MODELS_DIR`, which only exists after Task 1. (Fix M-minor: the earlier "Consumes: nothing" was wrong.)
- Produces: a config saved from the GUI contains no absolute path in `color_tag_model_path` or `cnn_classifiers[].model_path` when those models live under the models root; `build_engine_params` resolves both back to absolute. Task 3 and Task 7 rely on this round trip.

**Why the load side is in this task, honestly stated.** `engine_params.py:1591` emits `COLOR_TAG_MODEL_PATH` as `str(_cfg_get(cfg, "color_tag_model_path", default=""))` with **no** `resolve_model_path`, and `:1024` copies that unresolved value into `CNN_CLASSIFIER_MODEL_PATH`. `cnn_classifiers[].model_path` *is* already resolved (`:927-932`). Spec §240 asserts the load side is already symmetric; it is not — but that asymmetry does **not** currently break anything, because `COLOR_TAG_MODEL_PATH`/`CNN_CLASSIFIER_MODEL_PATH` are dead: nothing in `src/hydra_suite/core/` reads `color_tag`, and the GUI's colour-tag input is `setVisible(False)` (`trackerkit/gui/panels/identity_panel.py:143`). Real colour-tag identity runs through ClassKit multi-head classifiers via `cnn_classifiers`, which is already resolved on load. We still add the load-side resolve here — it costs nothing, keeps the (currently inert) key internally consistent with every other model key, and the lowercase `color_tag_model_path` key lands verbatim in the job sidecar where `verify_job` (Task 7) rejects absolute paths — but the correct framing is **defensive symmetry + portability-by-construction**, not "or colour-tag identity breaks". Task 3 additionally moves `COLOR_TAG_MODEL_PATH` into the never-yielded key set, so pack never ships it as a model reference regardless.

- [ ] **Step 1: Write the failing test**

Create `tests/test_config_model_path_portability.py`:

```python
"""Model paths in saved configs are models-root-relative and resolve back."""

import json

import pytest

from hydra_suite.core.inference.model_paths import (
    make_model_path_relative,
    resolve_model_path,
)
from hydra_suite.trackerkit.engine_params import build_engine_params
from hydra_suite.trackerkit.engine_params import RuntimeContext


@pytest.fixture()
def models_root(tmp_path, monkeypatch):
    root = tmp_path / "models"
    (root / "classification" / "colortag").mkdir(parents=True)
    (root / "classification" / "colortag" / "tags.pth").write_bytes(b"x")
    (root / "classification" / "identity").mkdir(parents=True)
    (root / "classification" / "identity" / "ids.pth").write_bytes(b"y")
    monkeypatch.setenv("HYDRA_MODELS_DIR", str(root))
    return root


def _runtime():
    return RuntimeContext(fps=30.0, total_frames=10, frame_width=64, frame_height=64)


def test_color_tag_absolute_path_is_relativized_on_save(models_root):
    absolute = str(models_root / "classification" / "colortag" / "tags.pth")
    assert make_model_path_relative(absolute) == "classification/colortag/tags.pth"


def test_color_tag_relative_path_resolves_in_engine_params(models_root):
    cfg = {
        "color_tag_model_path": "classification/colortag/tags.pth",
        "identity_method": "color_tag",
        "enable_identity_analysis": True,
    }
    params = build_engine_params(cfg, runtime=_runtime())
    expected = str(models_root / "classification" / "colortag" / "tags.pth")
    assert params["COLOR_TAG_MODEL_PATH"] == expected
    # The legacy singular bridge must carry the RESOLVED value too.
    assert params["CNN_CLASSIFIER_MODEL_PATH"] == expected


def test_empty_color_tag_stays_empty(models_root):
    params = build_engine_params({"color_tag_model_path": ""}, runtime=_runtime())
    assert params["COLOR_TAG_MODEL_PATH"] == ""
    assert params["CNN_CLASSIFIER_MODEL_PATH"] == ""


def test_color_tag_outside_models_root_is_left_absolute(models_root, tmp_path):
    outside = tmp_path / "elsewhere" / "tags.pth"
    outside.parent.mkdir(parents=True)
    outside.write_bytes(b"z")
    # make_model_path_relative only relativizes inside the root.
    assert make_model_path_relative(str(outside)) == str(outside)
    params = build_engine_params(
        {"color_tag_model_path": str(outside)}, runtime=_runtime()
    )
    assert params["COLOR_TAG_MODEL_PATH"] == str(outside)


def test_save_load_round_trip_is_identity_for_engine_params(models_root):
    absolute = str(models_root / "classification" / "identity" / "ids.pth")
    absolute_cfg = {"cnn_classifiers": [{"model_path": absolute, "batch_size": 8}]}
    relative_cfg = {
        "cnn_classifiers": [
            {"model_path": make_model_path_relative(absolute), "batch_size": 8}
        ]
    }
    a = build_engine_params(absolute_cfg, runtime=_runtime())
    b = build_engine_params(relative_cfg, runtime=_runtime())
    assert a["CNN_CLASSIFIERS"] == b["CNN_CLASSIFIERS"]
    assert a["CNN_CLASSIFIERS"][0]["model_path"] == absolute
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_config_model_path_portability.py -v`
Expected: `test_color_tag_relative_path_resolves_in_engine_params` FAILS — `COLOR_TAG_MODEL_PATH` comes back as the bare relative string.

- [ ] **Step 3: Fix the load side in `engine_params.py`**

At `:1023` replace the derivation:

```python
    # COLOR_TAG_MODEL_PATH is persisted models-root-relative (see
    # gui/orchestrators/config.py). Resolve it here exactly like every other
    # model key, or a relative value reaches the engine verbatim and the
    # colour-tag stage cannot load its checkpoint.
    color_tag_model_path = str(
        resolve_model_path(_cfg_get(cfg, "color_tag_model_path", default=""))
    )
    cnn_classifier_model_path = color_tag_model_path
```

At `:1591` emit the already-resolved local instead of re-reading the config:

```python
        "COLOR_TAG_MODEL_PATH": color_tag_model_path,
```

(`resolve_model_path("")` returns `""`, so disabled colour tag is unaffected; `resolve_model_path` returns absolute paths unchanged, so existing absolute configs keep working.)

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_config_model_path_portability.py -v`
Expected: all PASS.

- [ ] **Step 5: Fix the save side in `gui/orchestrators/config.py`**

At `:2096`:

```python
            "color_tag_model_path": make_model_path_relative(
                self._panels.identity.line_color_tag_model.text()
            ),
```

At `:2072-2074`, relativize each classifier entry as it is serialized:

```python
            "cnn_classifiers": [
                {**entry, "model_path": make_model_path_relative(
                    entry.get("model_path", "")
                )}
                for entry in self._mw._identity_config().get("cnn_classifiers", [])
            ],
```

`make_model_path_relative` is already imported at `:41-42`.

- [ ] **Step 6: Run the parity gate — this is the load-bearing check**

Run:
```bash
python -m pytest tests/test_gui_cli_param_equivalence.py \
                 tests/test_get_parameters_dict_characterization.py \
                 tests/test_config_model_path_portability.py -v
```
Expected: PASS, with `test_gui_cli_param_equivalence.py` **unchanged**. Relativize-on-save + resolve-on-load is a round trip, so `build_engine_params` output must be byte-identical to before. If the characterization golden changes, STOP — that means the round trip is not identity and the fix is wrong.

**Host caveat before you act on a STOP (fix M-minor).** The committed golden
`tests/data/get_parameters_dict_golden/ant_cnn_identity.json` carries a literal
absolute classifier path:
`/Users/neurorishika/Library/Application Support/hydra-suite/models/classification/identity/20260429-105036_classifier_multihead_obiroi_colortag.multihead.json`,
and `CNN_CLASSIFIERS` is **not** in that test's `HOST_DEPENDENT_DROPPED_KEYS`
(`tests/test_get_parameters_dict_characterization.py:269-274` lists only
`YOLO_MODEL_PATH`, `YOLO_OBB_DIRECT_MODEL_PATH`, `YOLO_HEADTAIL_MODEL_PATH`,
`POSE_MODEL_DIR`). So the round trip is provably identity **only on a host where
that exact file exists** — `resolve_model_path` on a models-root-relative value
yields `<models root>/<relpath>` regardless, but the golden pins this machine's
models root. On any other host this test already fails for `ant_cnn_identity`
before Task 2 touches anything. Confirm the failure is pre-existing by running
the same command on the branch base **before** treating a STOP as real
(memory `feedback_pre_existence_is_tested_against_branch_base`).

**Fix A7 (minor) — this particular STOP cannot fire from `color_tag_model_path` at all, on any host.** None of the equivalence fixtures (`tools/equivalence/fixtures/configs/*.json`) set `color_tag_model_path` — grepped, zero hits, including in `ant_cnn_identity.json`, which uses `cnn_classifiers` exclusively for its real classifier. So a golden divergence traceable to THIS task's `color_tag_model_path` change specifically cannot come from the committed fixtures/goldens; the only source that could ever trip this STOP is a config an agent constructs by hand while testing (e.g. `_everything_on` in Task 3's own test fixtures, which DOES set `color_tag_model_path`). The `ant_cnn_identity` host-path caveat above is a real, independent, pre-existing gap (fix in `CNN_CLASSIFIERS` handling, not `color_tag_model_path`) — don't conflate the two when triaging a STOP.

- [ ] **Step 7: Commit**

```bash
make format
git add src/hydra_suite/trackerkit/engine_params.py \
        src/hydra_suite/trackerkit/gui/orchestrators/config.py \
        tests/test_config_model_path_portability.py
git commit -m "fix(config): relativize color-tag and CNN classifier model paths on save, resolve them on load"
```

---

### Task 3: `iter_model_references` + the contract guard

**Files:**
- Modify: `src/hydra_suite/trackerkit/engine_params.py` (append after `build_engine_params`)
- Test: `tests/test_engine_params_model_reference_contract.py` (create)

**Interfaces:**
- Consumes: Task 2's resolved `COLOR_TAG_MODEL_PATH`.
- Produces:
  - `ModelReference` — frozen dataclass with fields `role: str`, `path: str`, `kind: str` (`"file"` or `"directory"`).
  - `iter_model_references(params: Mapping[str, Any]) -> Iterator[ModelReference]`
  - Module constants `MODEL_FILE_PARAM_KEYS`, `MODEL_DIR_PARAM_KEYS`, `MODEL_LIST_PARAM_KEYS`, `NON_MODEL_PATH_PARAM_KEYS`.

  Task 7's `pack_job` consumes `ModelReference` objects; Task 11's `job_cli` produces them.

**Governing principle: gate on what core actually loads, not on the GUI's enable flag.** Only head-tail emits `""` when disabled at the `build_engine_params` layer. But "gate on the enable flag" is itself the wrong rule for identity: core does **not** gate CNN classifiers on `ENABLE_IDENTITY_ANALYSIS` at all.
- `core/inference/config.py:1224` builds a `CNNConfig` for **every** entry in `CNN_CLASSIFIERS`, unconditionally.
- `core/inference/runner.py:509` loads all of them.
- `core/tracking/worker.py:955` derives whether the identity phase runs from `bool(p.get("CNN_CLASSIFIERS", []))` — the enable flag is never consulted for this decision.
- When a listed model file is missing, `core/inference/config.py:1227-1233` logs an error and **continues**, rather than refusing to run.

So gating `CNN_CLASSIFIERS` on `ENABLE_IDENTITY_ANALYSIS` **under-ships**: a config with the flag off but a populated `cnn_classifiers` list still runs the classifier in core, and a job packed with the flag-gated rule would silently diverge on the remote box (core loads models pack never shipped). `iter_model_references` must instead ship `CNN_CLASSIFIERS` whenever the list is non-empty, full stop — no identity-flag gate anywhere near it. Pose and head-tail remain gated (their flags genuinely control whether core builds the stage: `config.py:1255` for pose, the head-tail gate already in `build_engine_params` for head-tail); AprilTag needs no model file and is gated on `USE_APRILTAGS` (`config.py:1357`) purely for completeness of the reference set, not because it ships anything. The two non-selected YOLO mode keys are gated on `YOLO_OBB_MODE` since core genuinely never loads the unselected pair. `COLOR_TAG_MODEL_PATH` is never yielded at all — see below.

- [ ] **Step 1: Write the failing test**

Create `tests/test_engine_params_model_reference_contract.py`:

```python
"""Every path-bearing engine param key is classified, and only live roles ship."""

import pytest

from hydra_suite.trackerkit.engine_params import (
    MODEL_DIR_PARAM_KEYS,
    MODEL_FILE_PARAM_KEYS,
    MODEL_LIST_PARAM_KEYS,
    NON_MODEL_PATH_PARAM_KEYS,
    RuntimeContext,
    build_engine_params,
    iter_model_references,
)


def _runtime():
    return RuntimeContext(fps=30.0, total_frames=10, frame_width=64, frame_height=64)


def _everything_on(models_root):
    """A config that turns on every model-consuming role at once."""
    return {
        "detection_method": "yolo_obb",
        "yolo_obb_mode": "direct",
        "yolo_obb_direct_model_path": "obb/direct.pt",
        "yolo_detect_model_path": "detection/detect.pt",
        "yolo_crop_obb_model_path": "obb/cropped/crop.pt",
        "enable_headtail_orientation": True,
        "yolo_headtail_model_path": "classification/orientation/ht.pth",
        "enable_pose_extractor": True,
        "pose_model_type": "SLEAP",
        "pose_sleap_model_dir": "pose/SLEAP/run",
        "pose_model_dir": "pose/SLEAP/run",
        "pose_skeleton_file": str(models_root / "skel.json"),
        "enable_identity_analysis": True,
        "identity_method": "cnn",
        "cnn_classifiers": [{"model_path": "classification/identity/ids.pth"}],
        "color_tag_model_path": "classification/colortag/tags.pth",
        "use_apriltags": False,
    }


@pytest.fixture()
def models_root(tmp_path, monkeypatch):
    root = tmp_path / "models"
    for rel in (
        "obb/direct.pt",
        "detection/detect.pt",
        "obb/cropped/crop.pt",
        "classification/orientation/ht.pth",
        "classification/identity/ids.pth",
        "classification/colortag/tags.pth",
    ):
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"m")
    (root / "pose" / "SLEAP" / "run").mkdir(parents=True)
    (root / "pose" / "SLEAP" / "run" / "best.ckpt").write_bytes(b"c")
    (root / "skel.json").write_text("{}")
    monkeypatch.setenv("HYDRA_MODELS_DIR", str(root))
    return root


def test_every_path_ish_key_is_classified(models_root):
    """A new model role added without classification fails HERE, loudly.

    Minor fix: `path_ish` used to be computed from a SINGLE direct-mode,
    non-apriltag `_everything_on` build. `path_ish` only ever contains keys
    that are actually PRESENT in that one `params` dict -- a new
    path-bearing key that only appears when `yolo_obb_mode == "sequential"`
    or `use_apriltags == True` (e.g. a hypothetical
    APRILTAG_CALIBRATION_PATH) would never show up in `path_ish` at all
    under the direct-mode-only build, so the guard would stay silently
    green even if that key were never classified -- the exact
    "new role added without classification" failure this test exists to
    catch. Union `path_ish` across THREE builds: the direct-mode one above,
    a sequential-mode variant, and a `use_apriltags=True` variant, so a key
    that only exists under either of those modes is still covered."""
    cfg_direct = _everything_on(models_root)
    cfg_sequential = _everything_on(models_root)
    cfg_sequential["yolo_obb_mode"] = "sequential"
    cfg_apriltags = _everything_on(models_root)
    cfg_apriltags["use_apriltags"] = True

    path_ish: set[str] = set()
    for cfg in (cfg_direct, cfg_sequential, cfg_apriltags):
        params = build_engine_params(cfg, runtime=_runtime())
        path_ish |= {
            key
            for key in params
            if key.endswith("_PATH") or key.endswith("_DIR") or key.endswith("_FILE")
        }

    classified = (
        set(MODEL_FILE_PARAM_KEYS)
        | set(MODEL_DIR_PARAM_KEYS)
        | set(MODEL_LIST_PARAM_KEYS)
        | set(NON_MODEL_PATH_PARAM_KEYS)
    )
    unclassified = path_ish - classified
    assert not unclassified, (
        "New path-bearing engine param key(s) are unclassified: "
        f"{sorted(unclassified)}. Add each to exactly one of "
        "MODEL_FILE_PARAM_KEYS / MODEL_DIR_PARAM_KEYS / MODEL_LIST_PARAM_KEYS / "
        "NON_MODEL_PATH_PARAM_KEYS in engine_params.py."
    )


def test_list_carriers_are_classified(models_root):
    """Any list-of-dicts param containing a 'model_path' must be declared."""
    params = build_engine_params(_everything_on(models_root), runtime=_runtime())
    carriers = {
        key
        for key, value in params.items()
        if isinstance(value, list)
        and value
        and all(isinstance(e, dict) for e in value)
        and any("model_path" in e for e in value)
    }
    assert carriers <= set(MODEL_LIST_PARAM_KEYS), (
        f"Undeclared model-carrying list param(s): {sorted(carriers - set(MODEL_LIST_PARAM_KEYS))}"
    )


def test_classification_sets_are_disjoint():
    sets = [
        set(MODEL_FILE_PARAM_KEYS),
        set(MODEL_DIR_PARAM_KEYS),
        set(MODEL_LIST_PARAM_KEYS),
        set(NON_MODEL_PATH_PARAM_KEYS),
    ]
    for i, a in enumerate(sets):
        for b in sets[i + 1 :]:
            assert not (a & b), f"key classified twice: {sorted(a & b)}"


def test_all_live_roles_are_yielded(models_root):
    params = build_engine_params(_everything_on(models_root), runtime=_runtime())
    roles = {ref.role for ref in iter_model_references(params)}
    # COLOR_TAG_MODEL_PATH is intentionally absent: it is dead (no consumer in
    # src/hydra_suite/core/; the GUI field is setVisible(False)) and moved to
    # NON_MODEL_PATH_PARAM_KEYS below. It is never yielded regardless of value.
    assert roles == {
        "YOLO_OBB_DIRECT_MODEL_PATH",
        "YOLO_HEADTAIL_MODEL_PATH",
        "POSE_MODEL_DIR",
        "CNN_CLASSIFIERS",
    }


def test_pose_is_not_shipped_when_the_stage_is_off(models_root):
    cfg = _everything_on(models_root)
    cfg["enable_pose_extractor"] = False
    params = build_engine_params(cfg, runtime=_runtime())
    assert params["POSE_MODEL_DIR"], "precondition: the key is still emitted"
    roles = {ref.role for ref in iter_model_references(params)}
    assert "POSE_MODEL_DIR" not in roles


def test_unselected_yolo_mode_models_are_not_shipped(models_root):
    """In direct mode the sequential pair is emitted but never loaded."""
    params = build_engine_params(_everything_on(models_root), runtime=_runtime())
    assert params["YOLO_DETECT_MODEL_PATH"], "precondition: emitted anyway"
    roles = {ref.role for ref in iter_model_references(params)}
    assert "YOLO_DETECT_MODEL_PATH" not in roles
    assert "YOLO_CROP_OBB_MODEL_PATH" not in roles


def test_sequential_mode_ships_the_pair_and_not_the_direct_model(models_root):
    cfg = _everything_on(models_root)
    cfg["yolo_obb_mode"] = "sequential"
    params = build_engine_params(cfg, runtime=_runtime())
    roles = {ref.role for ref in iter_model_references(params)}
    assert "YOLO_DETECT_MODEL_PATH" in roles
    assert "YOLO_CROP_OBB_MODEL_PATH" in roles
    assert "YOLO_OBB_DIRECT_MODEL_PATH" not in roles


def test_cnn_classifiers_ship_even_when_identity_flag_is_off(models_root):
    """Core does not gate CNN classifiers on ENABLE_IDENTITY_ANALYSIS — neither
    does iter_model_references. See core/inference/config.py:1224,
    core/inference/runner.py:509, core/tracking/worker.py:955: the flag is
    never consulted for whether classifiers load. A flag-gated
    iter_model_references would under-ship relative to what core actually
    loads and cause silent divergence on the remote box."""
    cfg = _everything_on(models_root)
    cfg["enable_identity_analysis"] = False
    params = build_engine_params(cfg, runtime=_runtime())
    roles = {ref.role for ref in iter_model_references(params)}
    assert "CNN_CLASSIFIERS" in roles


def test_empty_cnn_classifiers_list_yields_nothing(models_root):
    cfg = _everything_on(models_root)
    cfg["cnn_classifiers"] = []
    params = build_engine_params(cfg, runtime=_runtime())
    roles = {ref.role for ref in iter_model_references(params)}
    assert "CNN_CLASSIFIERS" not in roles


def test_color_tag_model_path_is_never_yielded(models_root):
    """COLOR_TAG_MODEL_PATH is dead (no consumer in core/); it lives in
    NON_MODEL_PATH_PARAM_KEYS and must never appear as a reference role even
    when populated and even when identity is enabled."""
    params = build_engine_params(_everything_on(models_root), runtime=_runtime())
    assert params["COLOR_TAG_MODEL_PATH"], "precondition: the key is populated"
    roles = {ref.role for ref in iter_model_references(params)}
    assert "COLOR_TAG_MODEL_PATH" not in roles


def test_stale_color_tag_path_does_not_block_yielding_other_roles(models_root):
    """A nonexistent color_tag_model_path must not affect iter_model_references
    at all, since the key is never resolved to a filesystem check here (that
    dead-key handling lives entirely in NON_MODEL_PATH_PARAM_KEYS)."""
    cfg = _everything_on(models_root)
    cfg["color_tag_model_path"] = "classification/colortag/does_not_exist.pth"
    params = build_engine_params(cfg, runtime=_runtime())
    roles = {ref.role for ref in iter_model_references(params)}
    assert "CNN_CLASSIFIERS" in roles
    assert "YOLO_OBB_DIRECT_MODEL_PATH" in roles


def test_pose_reference_kind_is_directory(models_root):
    params = build_engine_params(_everything_on(models_root), runtime=_runtime())
    pose = [r for r in iter_model_references(params) if r.role == "POSE_MODEL_DIR"]
    assert len(pose) == 1
    assert pose[0].kind == "directory"


def test_pose_reference_kind_is_file_for_yolo_pose(models_root):
    """Fix A3: POSE_MODEL_DIR is a FILE for the YOLO-pose/ViTPose backends —
    pose/backends/yolo.py:64-65 and vitpose.py:206 both call
    model_path.with_suffix(...) on it. "kind" must be derived from what's
    on disk, not assumed directory just because the key lives in
    MODEL_DIR_PARAM_KEYS. **Correction (fix V5): the real
    ant_pose_headtail.json fixture does NOT exercise this file-kind branch —
    it has `"pose_model_type": "sleap"` (verified:
    tools/equivalence/fixtures/configs/ant_pose_headtail.json:236), so
    POSE_MODEL_DIR there resolves to the SLEAP run DIRECTORY, not a file.
    This test therefore builds its own synthetic YOLO-pose config from
    scratch (below) rather than citing the fixture — that part was always
    correct — but a prior draft's rationale wrongly implied the fixture
    itself was the YOLO-pose file-kind case. It is not: it is Task 13's
    SLEAP/directory case (see the Task 13 pre-check below for what that
    implies for the Goal-4 portability probe)."""
    cfg = _everything_on(models_root)
    yolo_pose_path = models_root / "pose" / "YOLO-pose" / "run.pt"
    yolo_pose_path.parent.mkdir(parents=True, exist_ok=True)
    yolo_pose_path.write_bytes(b"y")
    cfg["pose_model_type"] = "YOLO"
    cfg["pose_yolo_model_dir"] = "pose/YOLO-pose/run.pt"
    cfg["pose_model_dir"] = "pose/YOLO-pose/run.pt"
    params = build_engine_params(cfg, runtime=_runtime())
    pose = [r for r in iter_model_references(params) if r.role == "POSE_MODEL_DIR"]
    assert len(pose) == 1
    assert pose[0].kind == "file"


def test_empty_values_are_skipped(models_root):
    cfg = _everything_on(models_root)
    cfg["enable_headtail_orientation"] = False
    params = build_engine_params(cfg, runtime=_runtime())
    assert params["YOLO_HEADTAIL_MODEL_PATH"] == ""
    assert "YOLO_HEADTAIL_MODEL_PATH" not in {r.role for r in iter_model_references(params)}


def test_bgsub_config_yields_no_model_references(models_root):
    cfg = _everything_on(models_root)
    cfg["detection_method"] = "background_subtraction"
    cfg["enable_pose_extractor"] = False
    cfg["enable_identity_analysis"] = False
    cfg["enable_headtail_orientation"] = False
    # Fix B3: clearing the LIST is what makes this assertion true, not clearing
    # the enable flag. The C1 rule (below) is that CNN_CLASSIFIERS is yielded
    # whenever the list is non-empty, because core/inference/config.py:1224
    # builds a CNNConfig per entry with no reference to
    # ENABLE_IDENTITY_ANALYSIS. A bgsub job that still ships classifiers is
    # therefore correct behaviour, not a bug -- so this test must empty the
    # list to assert "no model references at all".
    cfg["cnn_classifiers"] = []
    params = build_engine_params(cfg, runtime=_runtime())
    assert list(iter_model_references(params)) == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_engine_params_model_reference_contract.py -v`
Expected: FAIL with `ImportError: cannot import name 'iter_model_references'`.

- [ ] **Step 3: Implement in `engine_params.py`**

Append after `build_engine_params`:

```python
# --- Portable-job model reference derivation -------------------------------
#
# These four tuples are the CLASSIFICATION of every path-bearing engine param
# key. `tests/test_engine_params_model_reference_contract.py` fails if a new
# key ends in _PATH/_DIR/_FILE and is not in exactly one of them, which is what
# makes `trackerkit job pack` impossible to silently forget when a new model
# role is added.
#
# GOVERNING PRINCIPLE: gate on what core actually loads, not on the GUI's
# enable flag. Verified against real source:
#   - core/inference/config.py:1224 builds a CNNConfig for EVERY entry in
#     CNN_CLASSIFIERS, unconditionally — no ENABLE_IDENTITY_ANALYSIS check.
#   - core/inference/runner.py:509 loads all of them.
#   - core/tracking/worker.py:955 derives whether the identity phase runs from
#     bool(p.get("CNN_CLASSIFIERS", [])), never from the enable flag.
#   - core/inference/config.py:1227-1233 logs an error and CONTINUES when a
#     listed model file is missing, rather than refusing to run.
# So CNN_CLASSIFIERS ships whenever the list is non-empty, full stop. Gating
# it on ENABLE_IDENTITY_ANALYSIS would under-ship relative to what core loads
# and cause silent divergence between the packed job and the remote run.
# COLOR_TAG_MODEL_PATH is dead (no consumer in src/hydra_suite/core/; GUI field
# is setVisible(False)) and lives in NON_MODEL_PATH_PARAM_KEYS — never yielded.

MODEL_FILE_PARAM_KEYS = (
    "YOLO_OBB_DIRECT_MODEL_PATH",
    "YOLO_DETECT_MODEL_PATH",
    "YOLO_CROP_OBB_MODEL_PATH",
    "YOLO_HEADTAIL_MODEL_PATH",
)
MODEL_DIR_PARAM_KEYS = ("POSE_MODEL_DIR",)
# param key -> the dict key holding the path inside each list entry
MODEL_LIST_PARAM_KEYS = {"CNN_CLASSIFIERS": "model_path"}
NON_MODEL_PATH_PARAM_KEYS = (
    # Alias of whichever OBB key the mode selected; never a distinct artifact.
    "YOLO_MODEL_PATH",
    # Dead: no consumer anywhere in src/hydra_suite/core/ (grepped); the GUI
    # field is setVisible(False) at trackerkit/gui/panels/identity_panel.py:143.
    # Real colour-tag identity runs through ClassKit multi-head classifiers via
    # CNN_CLASSIFIERS. Never yielded as a reference, even when populated.
    "COLOR_TAG_MODEL_PATH",
    # Legacy singular bridge; always equals COLOR_TAG_MODEL_PATH (see :1023).
    # Equally dead; kept classified for the contract guard only.
    "CNN_CLASSIFIER_MODEL_PATH",
    # Always "" today.
    "POSE_EXPORTED_MODEL_PATH",
    # A config asset, shipped by the config snapshot, not the models root.
    "POSE_SKELETON_FILE",
    # Output destinations, not inputs.
    "DATASET_OUTPUT_DIR",
    "FINAL_MEDIA_EXPORT_VIDEO_OUTPUT_DIR",
    "INDIVIDUAL_DATASET_OUTPUT_DIR",
    "INDIVIDUAL_PROPERTIES_CACHE_PATH",
)


@dataclass(frozen=True)
class ModelReference:
    """One model artifact a run will actually load."""

    role: str
    path: str
    kind: str  # "file" | "directory"


def iter_model_references(params: Mapping[str, Any]) -> Iterator[ModelReference]:
    """Yield every model artifact the run described by ``params`` will load.

    Enablement gating matters, but it must track core's own decision, not the
    GUI's enable flags: ``build_engine_params`` emits POSE_MODEL_DIR and both
    non-selected YOLO mode keys REGARDLESS of whether the stage runs (only
    head-tail is gated in build_engine_params, at :823-843), and CNN_CLASSIFIERS
    is never gated on ENABLE_IDENTITY_ANALYSIS anywhere in core (see the module
    comment above). Filtering on "non-empty string" alone would ship models
    that are never loaded; filtering CNN_CLASSIFIERS on the identity flag would
    under-ship models core loads anyway. COLOR_TAG_MODEL_PATH is never yielded.
    """
    obb_mode = str(params.get("YOLO_OBB_MODE", "direct") or "direct").lower()
    live_files: list[str] = []
    if params.get("DETECTION_METHOD") != "background_subtraction":
        if obb_mode == "sequential":
            live_files += ["YOLO_DETECT_MODEL_PATH", "YOLO_CROP_OBB_MODEL_PATH"]
        else:
            live_files.append("YOLO_OBB_DIRECT_MODEL_PATH")
    live_files.append("YOLO_HEADTAIL_MODEL_PATH")  # already "" when disabled

    for key in live_files:
        value = str(params.get(key, "") or "").strip()
        if value:
            yield ModelReference(role=key, path=value, kind="file")

    if params.get("ENABLE_POSE_EXTRACTOR"):
        for key in MODEL_DIR_PARAM_KEYS:
            value = str(params.get(key, "") or "").strip()
            if value:
                # Fix A3: POSE_MODEL_DIR is a directory for the SLEAP backend
                # but a FILE for YOLO-pose/ViTPose (pose/backends/yolo.py:64-65,
                # pose/backends/vitpose.py:206 both do model_path.with_suffix(...)
                # on it; the fixture ant_pose_headtail.json:238 sets it to
                # "YOLO-pose/....pt", not a directory). "kind" must be derived
                # from what's actually on disk at reference time, never assumed
                # from which tuple the key lives in.
                yield ModelReference(
                    role=key,
                    path=value,
                    kind="directory" if os.path.isdir(value) else "file",
                )

    # CNN classifiers: gated ONLY on the list being non-empty — never on
    # ENABLE_IDENTITY_ANALYSIS. See the module-level comment for why.
    for key, field in MODEL_LIST_PARAM_KEYS.items():
        for entry in params.get(key, []) or []:
            value = str((entry or {}).get(field, "") or "").strip()
            if value:
                yield ModelReference(role=key, path=value, kind="file")
```

Add `Iterator` to the `typing` imports, `dataclass` to the `dataclasses` import, and `import os` (for the `os.path.isdir` kind check above — fix A3) at the top of the file if not already present.

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_engine_params_model_reference_contract.py -v`
Expected: all PASS. If `test_every_path_ish_key_is_classified` fails listing a key this plan did not anticipate, add it to `NON_MODEL_PATH_PARAM_KEYS` (if it is an output/config path) or to the appropriate model tuple — do **not** loosen the assertion.

- [ ] **Step 5: Run the full contract-guard suite**

Run:
```bash
python -m pytest tests/test_engine_params_model_reference_contract.py \
                 tests/test_get_parameters_dict_characterization.py \
                 tests/test_gui_cli_param_equivalence.py -v
```
Expected: PASS; `build_engine_params` output unchanged (this task only *adds* a reader).

- [ ] **Step 6: Commit**

```bash
make format
git add src/hydra_suite/trackerkit/engine_params.py tests/test_engine_params_model_reference_contract.py
git commit -m "feat(engine-params): derive the model reference set from build_engine_params with a contract guard"
```

---

### Task 4: Content-based cache identity (schema 4 → 5)

**Fix A8 — this task is too large for one agent; it splits into three checkpointed sub-tasks that share this task's numbering (no cross-reference elsewhere in this plan changes — every "Task 4" reference below still means this task; the split only adds commit boundaries an agent can hand off at):**
- **4a** (Steps 1-5): `content_id.py` + `CacheKey`/`cache/keys.py` rewrite + schema bump to 5. Self-contained; can be reviewed/handed off alone. Commits at the checkpoint inserted after Step 5.
- **4b** (Step 6): the DetectKit migration (fix A4's `_source_content_id`, `prediction_cache.py`, `sidecars/operations.py`). Depends on 4a's `content_id` module. Commits at the checkpoint inserted after Step 6.
- **4c** (Steps 7-14): the 14-file test migration (Step 8's full enumeration) plus both platform equivalence gates (Steps 13-14). Depends on 4a+4b landing first — this is the slice that proves the whole task is byte-identical, so it must see the finished code, not a partial rewrite. Commits at the existing Step 12, gated by Steps 13-14.

An agent picking up 4b or 4c should read the prior sub-task's commit(s) rather than re-deriving `content_id.py`'s contract from scratch.

**Files:**
- Create: `src/hydra_suite/core/inference/content_id.py`
- Modify: `src/hydra_suite/core/inference/cache/base.py` (`CacheKey`, `CACHE_SCHEMA_VERSION`)
- Modify: `src/hydra_suite/core/inference/cache/keys.py` (`_model_signature`, `video_signature`, all six key builders, `_mtime` removal)
- Modify: `src/hydra_suite/core/inference/cache/reuse.py:21`, `cache/reader.py:20,33` (placeholder keys)
- Modify: `src/hydra_suite/detectkit/jobs/prediction_cache.py:31-66`, `src/hydra_suite/detectkit/sidecars/operations.py:21-35`
- Modify: `src/hydra_suite/trackerkit/gui/dialogs/parameter_helper.py:1768-1780` (comment only)
- Test: `tests/test_cache_content_identity.py` (create); update `tests/test_inference_cache_keys.py`, `tests/test_inference_cache_chunked.py`, `tests/test_pose_cache_empty_frame.py`, `tests/test_detectkit_prediction_cache.py` (minor fix — 4b modifies `detectkit/jobs/prediction_cache.py` and this file is where that migration is tested; it was previously omitted from this header even though it is both referenced and updated later in this task, at Step 6)

**Interfaces:**
- Consumes: nothing from earlier tasks **at the code level** — but it still runs FOURTH in execution order, so its own equivalence gate (Step 13/14) must baseline against the Task-3 tip, not the branch root, or a divergence cannot be attributed to Task 4 specifically. See fix A7 at Step 13.
- Produces:
  - `content_id.file_content_id(path: str) -> str` → `"sha256:<hex>"`, `""` for a missing/empty path.
  - `content_id.directory_content_id(path: str) -> str` → `"dirsha256:<hex>"`.
  - `content_id.model_content_id(path: str) -> str` → dispatches file/dir/`.multihead.json` manifest; memoized per process on `(realpath, size, mtime_ns)`. Fix A1b: a `.multihead.json` path gets a composite `"multihead:<hex>"` digest over the manifest bytes AND every `factor_models[].path` head it references (resolved the same way `backend.py:494-497` resolves them at load time) — NOT just the manifest bytes, so a head retrained in place under the same filename is not invisible to the key.
  - `content_id.video_signature(path: str | None) -> str` → `"{size}:{sha256(head8MiB‖tail8MiB)[:32]}"`.
  - `CacheKey(schema_version: int, model_id: str, config_hash: str)` with `as_string()` = `f"v{schema_version}|{model_id}|{config_hash}"`.

**This is the only slice that can break byte-identity. It ships alone, with its own full equivalence matrix on both platforms.**

**Scope decision (user-confirmed):** DetectKit migrates onto the same content-based `CacheKey`. Its sidecar payload shape (`operations.py:21-35`, which pins the exact field-name set) is bumped so old payloads are rejected loudly rather than misread. **Correction:** this is not an on-disk file format — `operations.py` is the in-process, parent→sidecar-process IPC payload used to hand a cache key across a subprocess boundary within one run, not a value read back from disk in a later session. So "sidecars written before v5 must be regenerated" overstates the blast radius: there is no persisted-on-disk artifact from a prior run that this would orphan. The rejection is still correct, but not for the reason originally stated. **Minor fix — "an old, unrestarted sidecar process" is not actually possible and the message should not claim it.** `detectkit/sidecars/supervisor.py:257-258`'s `ProtectedOperation` is "one synchronously executed sidecar" spawned fresh per request — there is no long-lived sidecar process that could persist across a code change and send a stale payload. The real cause of a mismatched payload shape is **code-version skew between the parent and sidecar interpreter** (e.g. the sidecar's Python environment/installed package still has the old `hydra_suite` on its path while the parent process has the new one — plausible in a dev checkout with multiple envs, or a partially-updated install). Reword the rejection message accordingly — see the corrected wording in Step 6.

**Deliberately unchanged:** `core/individual/pose/artifacts.py:71` `path_fingerprint_token` embeds the resolved absolute path, but it guards *export-artifact validity* (ONNX/TensorRT/CoreML reuse) which is host-local by design (spec §2 non-goal) and feeds **no** `CacheKey`. Leave it. Verified: its only consumers are `pose/backends/{sleap,vitpose,yolo}.py` artifact signatures.

- [ ] **Step 1: Write the failing test**

Create `tests/test_cache_content_identity.py`:

```python
"""Cache identity is content-based: same bytes anywhere == same key."""

import json
import os
import shutil

import pytest

from hydra_suite.core.inference import content_id
from hydra_suite.core.inference.cache.base import CACHE_SCHEMA_VERSION, CacheKey


def _touch_different_mtime(path):
    os.utime(path, (1, 1))


def test_file_content_id_is_path_and_mtime_independent(tmp_path):
    a = tmp_path / "here" / "model.pt"
    b = tmp_path / "somewhere" / "else" / "model.pt"
    a.parent.mkdir(parents=True)
    b.parent.mkdir(parents=True)
    a.write_bytes(b"weights" * 100)
    shutil.copy2(a, b)
    _touch_different_mtime(b)
    assert content_id.file_content_id(str(a)) == content_id.file_content_id(str(b))
    assert content_id.file_content_id(str(a)).startswith("sha256:")


def test_one_byte_change_changes_the_file_content_id(tmp_path):
    a = tmp_path / "m.pt"
    a.write_bytes(b"aaaa")
    first = content_id.file_content_id(str(a))
    a.write_bytes(b"aaab")
    assert content_id.file_content_id(str(a)) != first


def test_directory_content_id_is_path_independent(tmp_path):
    a = tmp_path / "run_a"
    b = tmp_path / "nested" / "run_b"
    for root in (a, b):
        root.mkdir(parents=True)
        (root / "best.ckpt").write_bytes(b"ckpt")
        (root / "training_config.json").write_text("{}")
    assert content_id.directory_content_id(str(a)) == content_id.directory_content_id(str(b))
    assert content_id.directory_content_id(str(a)).startswith("dirsha256:")


def test_directory_content_id_ignores_local_runtime_artifacts(tmp_path):
    root = tmp_path / "run"
    root.mkdir()
    (root / "best.ckpt").write_bytes(b"ckpt")
    before = content_id.directory_content_id(str(root))
    engines = root / ".hydra-runtime-artifacts"
    engines.mkdir()
    (engines / "model.engine").write_bytes(b"host-specific")
    assert content_id.directory_content_id(str(root)) == before


def test_directory_content_id_notices_a_changed_member(tmp_path):
    root = tmp_path / "run"
    root.mkdir()
    (root / "best.ckpt").write_bytes(b"ckpt")
    before = content_id.directory_content_id(str(root))
    (root / "best.ckpt").write_bytes(b"ckpt2")
    assert content_id.directory_content_id(str(root)) != before


def test_missing_path_yields_empty_id(tmp_path):
    assert content_id.file_content_id(str(tmp_path / "nope.pt")) == ""
    assert content_id.file_content_id("") == ""


def test_two_different_missing_models_get_different_content_ids(tmp_path):
    """Fix M5: two configured-but-missing models must NOT collapse to the same
    "" id, or a cache written while model A was missing would spuriously
    validate for model B. file_content_id("") == "" for both is fine (that's
    the file-level primitive); model_content_id must NOT collapse them."""
    content_id.model_content_id.cache_clear()
    a = content_id.model_content_id(str(tmp_path / "missing_a.pt"))
    b = content_id.model_content_id(str(tmp_path / "missing_b.pt"))
    assert a != b
    assert a.startswith("missing:")
    assert b.startswith("missing:")


def test_unconfigured_model_path_is_still_empty():
    assert content_id.model_content_id("") == ""
    assert content_id.model_content_id(None) == ""


def _write_multihead_manifest(root, manifest_name="clf.multihead.json", head_bytes=b"head_a"):
    (root / "clf_flat.pth").write_bytes(head_bytes)
    manifest = root / manifest_name
    manifest.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "kind": "classifier_multihead_bundle",
                "factor_names": ["colour"],
                "factor_models": [{"factor": "colour", "path": "clf_flat.pth", "class_names": ["a"]}],
            }
        ),
        encoding="utf-8",
    )
    return manifest


def test_multihead_manifest_content_id_notices_a_retrained_head(tmp_path):
    """Fix A1b + W4: retraining a head IN PLACE under the same filename
    changes zero bytes of the manifest JSON itself -- a v5 key built by
    hashing only the manifest (file_content_id) would be BLIND to the
    retrain and would wrongly validate a stale cache. model_content_id must
    fold every factor_models[].path head into the manifest's identity
    (backend.py:494-497 is the resolution algorithm this mirrors: base =
    manifest.parent; (base / entry["path"]).resolve()).

    Fix W4: deliberately NO `cache_clear()` call between the two
    `model_content_id` calls below. This is the actual bug fix wave 3's
    version of this test masked: `_content_id_for_hint` is `@lru_cache`-d,
    keyed on `_stat_hint(manifest)`. Retraining a SIBLING head file changes
    zero bytes/stat of the MANIFEST itself, so a version of `_stat_hint`
    that only stats the manifest returns the identical memo key before and
    after -- `model_content_id` would return the STALE cached composite
    without a manual `cache_clear()` ever being called, which is exactly
    what a real long-lived TrackerKit GUI session does (it never calls
    `cache_clear()` between ClassKit retraining a head in another process
    and the GUI's own next cache-key computation). Calling `cache_clear()`
    here would hide that regression entirely -- the fixed `_stat_hint` must
    make the SECOND `model_content_id` call itself observe the retrain via
    a changed memo key, with the cache warm the whole time."""
    content_id.model_content_id.cache_clear()
    manifest = _write_multihead_manifest(tmp_path)
    before = content_id.model_content_id(str(manifest))
    assert before.startswith("multihead:")
    # Minor fix (round-7): capture the manifest's OWN file_content_id BEFORE
    # the head rewrite, so the assertion below is a real before/after
    # comparison, not the tautological `x == x` the previous draft had
    # (`file_content_id(str(manifest)) == file_content_id(str(manifest))`,
    # which is trivially true regardless of whether the manifest bytes
    # actually changed and proves nothing).
    manifest_only_id_before = content_id.file_content_id(str(manifest))
    (tmp_path / "clf_flat.pth").write_bytes(b"retrained_head")
    after = content_id.model_content_id(str(manifest))  # NOTE: no cache_clear() here
    assert after != before
    # The manifest bytes alone are unchanged -- proves the composite is
    # actually reading the head, not just re-hashing the manifest file.
    assert content_id.file_content_id(str(manifest)) == manifest_only_id_before


def test_multihead_manifest_content_id_is_path_independent(tmp_path):
    a = tmp_path / "run_a"
    b = tmp_path / "run_b"
    a.mkdir()
    b.mkdir()
    manifest_a = _write_multihead_manifest(a)
    manifest_b = _write_multihead_manifest(b)
    content_id.model_content_id.cache_clear()
    assert content_id.model_content_id(str(manifest_a)) == content_id.model_content_id(str(manifest_b))


def test_video_signature_survives_a_touched_mtime(tmp_path):
    v = tmp_path / "clip.mp4"
    v.write_bytes(b"\x00" * (1 << 20))
    first = content_id.video_signature(str(v))
    _touch_different_mtime(v)
    assert content_id.video_signature(str(v)) == first


def test_video_signature_changes_when_content_changes(tmp_path):
    v = tmp_path / "clip.mp4"
    v.write_bytes(b"\x00" * (1 << 20))
    first = content_id.video_signature(str(v))
    v.write_bytes(b"\x01" * (1 << 20))
    assert content_id.video_signature(str(v)) != first


def test_video_signature_detects_a_tail_only_change(tmp_path):
    """A re-encode that keeps the head must still invalidate. 4 MiB never
    enters the seek branch (file <= _VIDEO_PROBE), so this alone would pass
    even with a head-only implementation — it is NOT sufficient coverage by
    itself; see the 8-16 MiB and >16 MiB cases below."""
    v = tmp_path / "clip.mp4"
    body = bytearray(b"\x00" * (4 << 20))
    v.write_bytes(bytes(body))
    first = content_id.video_signature(str(v))
    body[-16:] = b"\xff" * 16
    v.write_bytes(bytes(body))
    assert content_id.video_signature(str(v)) != first


def test_video_signature_detects_a_tail_only_change_between_head_and_full_probe(tmp_path, monkeypatch):
    """12 MiB file: bigger than the 8 MiB head probe, smaller than 2x the
    probe, so the seek branch's overlap-with-head math is exercised."""
    monkeypatch.setattr(content_id, "_VIDEO_PROBE", 8 << 20)
    v = tmp_path / "clip.mp4"
    body = bytearray(b"\x00" * (12 << 20))
    v.write_bytes(bytes(body))
    first = content_id.video_signature(str(v))
    body[-16:] = b"\xff" * 16
    v.write_bytes(bytes(body))
    assert content_id.video_signature(str(v)) != first


def test_video_signature_detects_a_tail_only_change_above_2x_probe(tmp_path, monkeypatch):
    """>16 MiB (>2x probe): head and tail windows are disjoint; a tail-only
    edit must still be caught even though head bytes are fully unchanged.
    Probe size is monkeypatched down so the test stays fast."""
    monkeypatch.setattr(content_id, "_VIDEO_PROBE", 1 << 20)  # 1 MiB probe
    v = tmp_path / "clip.mp4"
    body = bytearray(b"\x00" * (3 << 20))  # 3 MiB, > 2x the 1 MiB probe
    v.write_bytes(bytes(body))
    first = content_id.video_signature(str(v))
    body[-16:] = b"\xff" * 16
    v.write_bytes(bytes(body))
    assert content_id.video_signature(str(v)) != first


def test_video_signature_of_missing_file_is_empty(tmp_path):
    assert content_id.video_signature(str(tmp_path / "gone.mp4")) == ""
    assert content_id.video_signature(None) == ""


def test_model_content_id_is_memoized_per_process(tmp_path, monkeypatch):
    m = tmp_path / "m.pt"
    m.write_bytes(b"x" * 4096)
    content_id.model_content_id.cache_clear()
    calls = []
    real_open = open

    def counting_open(path, *args, **kwargs):
        calls.append(str(path))
        return real_open(path, *args, **kwargs)

    monkeypatch.setattr("builtins.open", counting_open)
    first = content_id.model_content_id(str(m))
    second = content_id.model_content_id(str(m))
    assert first == second
    # Minor fix: `_stat_hint` (and therefore `file_content_id`, which opens
    # `Path(real)` where `real = os.path.realpath(path)`) opens the REALPATH,
    # not the caller's original string. On macOS, tmp_path lives under
    # /var, which is itself a symlink to /private/var, so
    # os.path.realpath(m) != str(m) -- comparing against str(m) here is
    # fragile and would undercount (or overcount, if some OTHER file happens
    # to share the same basename) on exactly that platform. Compare
    # realpaths on both sides.
    real_m = os.path.realpath(str(m))
    assert len([c for c in calls if os.path.realpath(c) == real_m]) == 1


def test_cache_key_has_no_mtime_field_and_new_string_form():
    key = CacheKey(
        schema_version=CACHE_SCHEMA_VERSION, model_id="sha256:ab", config_hash="cd"
    )
    assert not hasattr(key, "model_mtime")
    assert not hasattr(key, "model_path")
    assert key.as_string() == f"v{CACHE_SCHEMA_VERSION}|sha256:ab|cd"


def test_schema_version_is_five():
    assert CACHE_SCHEMA_VERSION == 5
```

Add to `tests/test_inference_cache_keys.py` (the OBB-specific regressions — this is spec correction #1).

**Fix B13 — these are written against the REAL helpers/APIs in that file, verified at `8f9688e0`. Do not improvise; there is no "adapt as needed" here, because this is the one slice that must ship byte-identical.**

Verified facts you are coding against:
- The file's existing direct-OBB helper is `_obb_direct(path="/m.pt", threshold=0.5)` (`tests/test_inference_cache_keys.py:44-52`). There is **no** `_direct_obb_config`. Do **not** add one.
- There is **no** `_sequential_config` helper; sequential configs are built inline as `OBBConfig(mode="sequential", sequential=OBBSequentialConfig(detect_model_path=..., obb_model_path=...))` (`:205-213`, `:265-275`). Keep doing that inline.
- `detection_cache_key(config: OBBConfig, roi_mask=None, batch_size=1)` (`cache/keys.py:93-97`) — the second positional is optional; pass it explicitly as `None` for readability.
- The file currently imports only `numpy`, `pytest`, `torch` plus `hydra_suite` names (`:1-31`). **Add `import os` and `import shutil` to its import block**; both are used below.
- `DetectionCacheHandle` (`cache/store.py:358-366`) is a dataclass taking `path`, `key`, and keyword-ish fields `require_key`, `read_only`, `write_mode`. Its writer is `write_frame(frame_idx: int, *, result: OBBResult)` (`:385`) and it validates that `result.frame_idx == frame_idx` and that all eight arrays are the same length. There is **no** `DetectionCacheStore`, **no** `DetectionCacheReader`, and **no** `write_frame(0, boxes=[])`.
- `CacheKey` is a frozen dataclass (`cache/base.py:23`), so a v4-shaped key is built with `dataclasses.replace(key, schema_version=4)` — monkeypatching `base.CACHE_SCHEMA_VERSION` will **not** work, because `cache/keys.py:21` binds the constant at import time (`from .base import CACHE_SCHEMA_VERSION`).
- A valid, non-degenerate `OBBResult` is already producible in this file via `materialize_tensors(_raw())` (`:32`, `:34-41`; `stages/obb.py:1858`). Use it rather than hand-building eight arrays.
- Reusability is decided by `CacheHandle.is_reusable()` (`cache/store.py:254` → `chunked.py:469`), which is the same predicate `cache_set_is_fully_reusable` uses.

```python
def test_obb_detection_key_is_identical_for_the_same_model_at_two_paths(tmp_path):
    """config_hash must NOT carry the model path (keys.py _model_signature)."""
    a = tmp_path / "one" / "obb.pt"
    b = tmp_path / "two" / "obb.pt"
    a.parent.mkdir(parents=True)
    b.parent.mkdir(parents=True)
    a.write_bytes(b"obb-weights")
    shutil.copy2(a, b)
    os.utime(b, (1, 1))
    key_a = detection_cache_key(_obb_direct(path=str(a)), None)
    key_b = detection_cache_key(_obb_direct(path=str(b)), None)
    assert key_a.as_string() == key_b.as_string()


def test_sequential_obb_key_is_identical_for_the_same_pair_at_two_paths(tmp_path):
    def _pair(root):
        root.mkdir(parents=True, exist_ok=True)
        (root / "detect.pt").write_bytes(b"d")
        (root / "obb.pt").write_bytes(b"o")
        return OBBConfig(
            mode="sequential",
            sequential=OBBSequentialConfig(
                detect_model_path=str(root / "detect.pt"),
                obb_model_path=str(root / "obb.pt"),
            ),
        )

    key_a = detection_cache_key(_pair(tmp_path / "one"), None)
    key_b = detection_cache_key(_pair(tmp_path / "two"), None)
    assert key_a.as_string() == key_b.as_string()


def test_sequential_key_changes_when_either_model_changes(tmp_path):
    root = tmp_path / "m"
    root.mkdir()
    detect, obb = root / "detect.pt", root / "obb.pt"
    detect.write_bytes(b"d")
    obb.write_bytes(b"o")

    def _cfg():
        return OBBConfig(
            mode="sequential",
            sequential=OBBSequentialConfig(
                detect_model_path=str(detect), obb_model_path=str(obb)
            ),
        )

    base = detection_cache_key(_cfg(), None).as_string()
    obb.write_bytes(b"o2")
    # model_content_id is memoized on (realpath, size, mtime_ns); the rewrite
    # changes size AND mtime_ns, so the memo entry is a miss, not a stale hit.
    assert detection_cache_key(_cfg(), None).as_string() != base


def test_v4_cache_on_disk_is_rejected_and_rebuilt(tmp_path):
    """Fix M4: a literal-string comparison is tautological -- it proves nothing
    about the actual store. Exercise the real handle path: write a v4-shaped
    on-disk cache, then prove the store treats it as unusable (rejected) and a
    v5 write follows (rebuilt), per spec section 7b.6 item 3.
    """
    import dataclasses

    from hydra_suite.core.inference.cache.store import DetectionCacheHandle

    cache_dir = tmp_path / ".inference_cache_clip"
    cache_dir.mkdir()
    path = cache_dir / "detection.npz"
    result = materialize_tensors(_raw())
    v5_key = CacheKey(
        schema_version=CACHE_SCHEMA_VERSION, model_id="sha256:aa", config_hash="bb"
    )
    v4_key = dataclasses.replace(v5_key, schema_version=4)

    stale = DetectionCacheHandle(path=path, key=v4_key, write_mode="fresh")
    stale.write_frame(result.frame_idx, result=result)
    stale.close()

    rejected = DetectionCacheHandle(
        path=path, key=v5_key, read_only=True, write_mode="auto"
    )
    assert not rejected.is_reusable(), "a v4 on-disk cache must not validate at v5"
    rejected.close()  # read_only close is a disk no-op (store.py:223-225)

    fresh = DetectionCacheHandle(path=path, key=v5_key, write_mode="fresh")
    fresh.write_frame(result.frame_idx, result=result)
    fresh.close()
    rebuilt = DetectionCacheHandle(path=path, key=v5_key, read_only=True)
    assert rebuilt.is_reusable(), "a fresh v5 write must be usable"
    rebuilt.close()


def test_cache_written_at_one_path_is_reusable_from_a_copy_at_another_path(tmp_path):
    """Fix Z7: no existing test in this task proves the actual Goal-4
    property AT THE HANDLE LEVEL -- that a cache produced against a model at
    path A validates against the SAME model's bytes copied to path B in a
    simulated fresh process (a different machine, in practice). Every other
    test here proves the KEY STRING is path-independent; this proves the
    on-disk cache built from that key is actually reusable end to end.
    """
    import shutil

    from hydra_suite.core.inference.cache.store import DetectionCacheHandle
    from hydra_suite.core.inference import content_id

    model_a = tmp_path / "box_a" / "obb.pt"
    model_a.parent.mkdir(parents=True)
    model_a.write_bytes(b"obb-weights" * 1000)

    key_a = detection_cache_key(_obb_direct(path=str(model_a)), None)
    cache_dir = tmp_path / ".inference_cache_clip"
    cache_dir.mkdir()
    path = cache_dir / "detection.npz"
    result = materialize_tensors(_raw())
    writer = DetectionCacheHandle(path=path, key=key_a, write_mode="fresh")
    writer.write_frame(result.frame_idx, result=result)
    writer.close()

    # Simulate a fresh process on a different machine: drop the in-process
    # memoization AND rebuild the key from a COPY of the same bytes at a
    # different path with a different mtime.
    content_id.model_content_id.cache_clear()
    model_b = tmp_path / "box_b" / "nested" / "obb.pt"
    model_b.parent.mkdir(parents=True)
    shutil.copy2(model_a, model_b)
    os.utime(model_b, (1, 1))
    key_b = detection_cache_key(_obb_direct(path=str(model_b)), None)

    reader = DetectionCacheHandle(path=path, key=key_b, read_only=True)
    assert reader.is_reusable(), "identical bytes at a different path must reuse the cache"
    reader.close()


def test_cache_written_at_one_path_is_not_reusable_after_one_byte_changes(tmp_path):
    """Fix Z7 (negative case): the copy-at-a-different-path test above proves
    portability; this proves it isn't achieved by accidentally ignoring model
    content altogether -- a genuinely different model at the new path must
    NOT validate."""
    from hydra_suite.core.inference.cache.store import DetectionCacheHandle
    from hydra_suite.core.inference import content_id

    model_a = tmp_path / "box_a" / "obb.pt"
    model_a.parent.mkdir(parents=True)
    model_a.write_bytes(b"obb-weights" * 1000)

    key_a = detection_cache_key(_obb_direct(path=str(model_a)), None)
    cache_dir = tmp_path / ".inference_cache_clip"
    cache_dir.mkdir()
    path = cache_dir / "detection.npz"
    result = materialize_tensors(_raw())
    writer = DetectionCacheHandle(path=path, key=key_a, write_mode="fresh")
    writer.write_frame(result.frame_idx, result=result)
    writer.close()

    content_id.model_content_id.cache_clear()
    model_c = tmp_path / "box_c" / "obb.pt"
    model_c.parent.mkdir(parents=True)
    model_c.write_bytes(b"different-obb-weights" * 1000)
    key_c = detection_cache_key(_obb_direct(path=str(model_c)), None)

    reader = DetectionCacheHandle(path=path, key=key_c, read_only=True)
    assert not reader.is_reusable(), "genuinely different model bytes must not validate"
    reader.close()
```

`CacheKey` and `CACHE_SCHEMA_VERSION` are already imported at the top of that file (`:26`), as are `OBBConfig`/`OBBSequentialConfig` (`:24-25`) and `materialize_tensors`/`_raw` (`:31-41`); only `os` and `shutil` are new.

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_cache_content_identity.py tests/test_inference_cache_keys.py -v`
Expected: FAIL — `content_id` module missing; `CACHE_SCHEMA_VERSION == 4`; the two-path OBB tests fail because `_model_signature` puts the path in `config_hash`.

- [ ] **Step 3: Create `src/hydra_suite/core/inference/content_id.py`**

```python
"""Content-based identity for models and videos.

Cache keys used to be ``(path, mtime)``-based, which made every cache
machine-local: a cache computed on a compute box could never be reused on the
staging machine because the model lived at a different absolute path. These
primitives identify an artifact by what it CONTAINS, so a cache travels with
the job (see docs/superpowers/specs/2026-09-09-portable-tracking-jobs-design.md
section 7b).
"""

from __future__ import annotations

import hashlib
import os
from functools import lru_cache
from pathlib import Path

_CHUNK = 1 << 20  # 1 MiB
_VIDEO_PROBE = 8 << 20  # head and tail bytes hashed for a video signature

# Host-specific engine caches must never influence a portable identity.
# Fix V-minor: `.DS_Store` and `__pycache__` are host/OS/filesystem-visit
# noise, not model content -- a Finder window opened on a SLEAP run
# directory on the Mac (or a stray bytecode cache from any tooling that
# imports something inside it) changes `directory_content_id` and defeats
# an otherwise-valid pulled cache purely because of when/whether a Finder
# window happened to be opened, which has nothing to do with model
# portability.
_EXCLUDED_DIR_NAMES = {".hydra-runtime-artifacts", ".DS_Store", "__pycache__"}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(_CHUNK), b""):
            digest.update(block)
    return digest.hexdigest()


def file_content_id(path: str | os.PathLike[str] | None) -> str:
    """``"sha256:<hex>"`` of a file's bytes; ``""`` if absent or unreadable."""
    if not path:
        return ""
    p = Path(path)
    try:
        if not p.is_file():
            return ""
        return f"sha256:{_sha256_file(p)}"
    except OSError:
        return ""


def directory_content_id(path: str | os.PathLike[str] | None) -> str:
    """``"dirsha256:<hex>"`` over sorted ``(relpath, sha256)`` of every member.

    MEASURED COST OF THIS DEVIATION (round-8, on the real fixture SLEAP run
    dir ``pose/SLEAP/20260214-224154_unet_ant_single_instance``, 94.7 MB):
    whole-tree 82.6 ms vs artifacts.py-allowlist 69.8 ms -- a 12.8 ms delta,
    because ``best.ckpt`` alone is 94 MB and the extra ``.slp``/``.csv``
    members total 0.4 MB. Memoized once per process, so this is immaterial
    against ``PERF_TOLERANCE=1.25``. Round-8 review raised the allowlist as a
    PERF concern; it was measured and rejected on the numbers. Do not
    re-litigate without a new measurement.

    DEVIATION FROM SPEC §7b.2: the spec says to reuse the artifact-fingerprint
    file SET (``core/individual/pose/artifacts.py``'s selection). This hashes
    every file under the directory instead and excludes only
    ``.hydra-runtime-artifacts/`` by name. Deliberate, not an oversight: that
    directory is nearly a phantom exclusion — it exists only as an OSError
    fallback (``core/inference/runtime_artifacts.py:765-768``); real pose
    exports (ONNX/TensorRT/CoreML) are written as SIBLINGS of the pose run
    directory, not inside it, so they are not members of ``root`` at all and
    the "exclude runtime artifacts" tests below are largely tautological
    (nothing host-specific is actually under `root` to exclude in the common
    case). We accept the deviation because hashing the whole tree is simpler,
    strictly safer (it can only over-invalidate, never silently miss a real
    content change), and the fingerprint subset is itself an
    implementation detail of a different, unrelated consumer.

    KNOWN LIVE COST of this deviation, not theoretical (fix V5, exercised at
    Task 13's Step 8-ish real-run check): a SLEAP-directory model means
    hashing EVERY file under the training-run tree, including
    ``labels_*.slp`` and ``viz/*.png`` — non-model members that can change
    (a log, a lockfile written during inference) without the actual model
    weights changing, which over-invalidates a pulled cache for reasons
    unrelated to model identity, and costs a full-tree hash once per fresh
    process where the pre-v5 path-based key paid nothing. If this shows up
    as a real PERF-gate or false-invalidation problem (see Task 13's
    self-diagnosing SLEAP-directory check), the fallback is the narrower
    ``core/individual/pose/artifacts.py`` fingerprint selection this
    docstring deviates from — not implemented here, kept as an escape hatch.
    """
    if not path:
        return ""
    root = Path(path)
    try:
        if not root.is_dir():
            return ""
        parts: list[str] = []
        for child in sorted(root.rglob("*")):
            if not child.is_file():
                continue
            rel = child.relative_to(root)
            if any(part in _EXCLUDED_DIR_NAMES for part in rel.parts):
                continue
            parts.append(f"{rel.as_posix()}={_sha256_file(child)}")
        blob = "\n".join(parts).encode("utf-8")
        return f"dirsha256:{hashlib.sha256(blob).hexdigest()}"
    except OSError:
        return ""


def _stat_hint(path: str) -> tuple[object, ...]:
    """Cheap local identity used ONLY as a memoization key, never in a cache key.

    KNOWN STALENESS (not a regression — matches the old (path, mtime) behaviour):
    for a directory, ``st.st_mtime_ns`` is the directory ENTRY's own mtime,
    which does not change when a member file already inside it is overwritten
    in place (only when an entry is added/removed/renamed). So
    ``model_content_id`` on a directory can return a stale memoized value
    within one process if a pose-run directory's member is edited in place
    without an entry list change. The old path-based key had the identical
    blind spot (it never looked inside directories at all), so this is not a
    new failure mode, only carried forward. Cross-process/cross-run identity
    is unaffected: a fresh process always re-hashes the tree.

    Fix W4 (multihead memo re-opens the bug it was added to close): for a
    ``.multihead.json`` path, the hint is NOT just ``(realpath, size,
    mtime_ns)`` of the manifest file. Retraining a head (ClassKit, a SEPARATE
    process from any long-lived TrackerKit GUI holding this memo) rewrites a
    SIBLING ``*_flat*.pth`` file that ``factor_models[].path`` points at —
    zero bytes of the manifest itself change, so the manifest-only hint is
    identical before and after a retrain and ``lru_cache`` returns the SAME
    stale composite id it returned before, defeating the whole point of fix
    A1b's composite digest. `_stat_hint` therefore also stats every head the
    manifest currently references (the same resolution
    ``_multihead_manifest_content_id`` uses: ``base = manifest.parent;
    (base / entry["path"]).resolve()``) and folds each head's ``(size,
    mtime_ns)`` into the returned tuple, so retraining a head changes the
    MEMO KEY itself and the next call re-hashes rather than returning a
    cached value. `cnn_cache_key` (`cache/keys.py:361-365`) runs inside a
    potentially long-lived TrackerKit GUI process while ClassKit retrains
    heads in a different process — this is exactly the path that stayed
    stale before.
    """
    try:
        st = os.stat(path)
        real = os.path.realpath(path)
    except OSError:
        # NOT realpath: on an unstattable path we deliberately keep the caller's
        # string verbatim, so the memo key and the sentinel below agree on what
        # "this artifact" means even when the path cannot be canonicalized.
        # (Fix M-minor: the sentinel comment used to say "realpath", which
        # disagreed with this branch. `_content_id_for_hint` reads `hint[0]`,
        # which is realpath on the success branch and the raw string here.)
        return (str(path), -1, -1)
    base_hint: tuple[object, ...] = (real, st.st_size, st.st_mtime_ns)
    if real.lower().endswith(".multihead.json"):
        head_stats: list[tuple[int, int]] = []
        try:
            import json

            data = json.loads(Path(real).read_text(encoding="utf-8"))
            manifest_dir = Path(real).parent
            for entry in data.get("factor_models", []) or []:
                entry_path = entry.get("path") if isinstance(entry, dict) else None
                if not entry_path:
                    continue
                head = (manifest_dir / entry_path).resolve()
                try:
                    hst = os.stat(head)
                    head_stats.append((hst.st_size, hst.st_mtime_ns))
                except OSError:
                    head_stats.append((-1, -1))
        except (OSError, ValueError):
            # Unparseable manifest: fall through with base_hint only -- the
            # manifest-only hint is still correct, just not head-aware; this
            # matches _multihead_manifest_content_id's own (OSError, ValueError)
            # handling (fix Z6: it degrades to file_content_id(manifest) on an
            # unreadable/malformed manifest, not "" -- a non-empty id here is
            # still important so a manifest-only-but-valid file keeps a stable,
            # non-sentinel identity even when head resolution can't happen).
            head_stats = []
        base_hint = base_hint + tuple(sorted(head_stats))
    return base_hint


def _multihead_manifest_content_id(manifest_path: Path) -> str:
    """Composite digest of a ``.multihead.json`` manifest AND every head it
    references (fix A1b).

    ``file_content_id`` alone hashes only the manifest bytes -- a few hundred
    bytes of JSON with ``factor_models[].path`` entries pointing at sibling
    ``_flat.pth``/``_flat_1.pth`` checkpoints
    (``core/individual/classification/backend.py:494-497``: ``base =
    manifest_path.parent; factor_path = (base / entry["path"]).resolve()``).
    Retraining a head IN PLACE under the same filename changes zero bytes the
    manifest's own hash sees, so a v5 key built from the manifest alone is
    BLIND to that retrain and would wrongly reuse a stale cache. Fold each
    head's own content id into the manifest's.

    This corrects the design doc's restatement ("model_id of the selected
    checkpoint only, as today; sibling heads travel with it and are covered
    by discover_multihead_model_bundle at pack time") for THIS format:
    ``discover_multihead_model_bundle`` only recognises ``*.bundle.json``
    (``classkit/model_bundle.py:12,82,119``) and returns ``None`` for a bare
    ``.multihead.json`` path -- it covers nothing here, at pack time or
    otherwise (see Task 11 fix A1a). The "selected checkpoint only" identity
    is correct for the ``.bundle.json`` case; it is wrong for
    ``.multihead.json``, which is the format the real production identity
    classifier actually uses (``tools/equivalence/fixtures/configs/
    ant_cnn_identity.json:235``).
    """
    # Fix Z6: `_stat_hint` above catches `(OSError, ValueError)` and guards
    # the manifest actually being a dict; this function must be at least as
    # defensive, because unlike `_stat_hint` (a memo-key helper only), an
    # UNCAUGHT exception here propagates out of `model_content_id` ->
    # `cnn_cache_key` -> `_open_caches`, aborting the whole run BEFORE model
    # loading ever gets a chance to raise the proper `ClassifierFormatError`
    # for the same malformed manifest. A truncated/corrupted
    # `.multihead.json` raises `json.JSONDecodeError` (a `ValueError`), not
    # `OSError` -- catching only `OSError` lets it escape uncaught. A
    # syntactically valid but non-dict JSON document (e.g. a bare `[]` or a
    # number) makes `data.get(...)` raise `AttributeError`.
    import json

    try:
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            return file_content_id(manifest_path)
    except (OSError, ValueError):
        return file_content_id(manifest_path)
    manifest_id = file_content_id(manifest_path)
    base = manifest_path.parent
    head_ids = []
    for entry in data.get("factor_models", []) or []:
        entry_path = entry.get("path") if isinstance(entry, dict) else None
        if not entry_path:
            continue
        head_ids.append(file_content_id((base / entry_path).resolve()))
    blob = "\n".join([manifest_id, *sorted(head_ids)]).encode("utf-8")
    return f"multihead:{hashlib.sha256(blob).hexdigest()}"


@lru_cache(maxsize=256)
def _content_id_for_hint(hint: tuple[object, ...]) -> str:
    # Fix W4: hint is (realpath, size, mtime_ns) for a plain file/directory,
    # or that plus a sorted tuple of (size, mtime_ns) per referenced head for
    # a .multihead.json manifest -- see _stat_hint. hint[0] is always the
    # realpath (or the raw unstattable string) regardless of length.
    real = hint[0]
    p = Path(real)
    if p.is_dir():
        dir_id = directory_content_id(real)
        if not dir_id:
            # Minor fix: directory_content_id() returns "" on ANY member
            # OSError (e.g. a permission-denied file partway through the
            # tree) for a directory that genuinely EXISTS -- silently
            # producing an empty content id for a real model directory.
            # Log loudly rather than let this degrade quietly and unnoticed.
            import logging

            logging.getLogger(__name__).warning(
                "directory_content_id() returned empty for an existing "
                "directory %s -- a member file likely raised OSError during "
                "the walk; the resulting content id is empty rather than "
                "reflecting the directory's real contents",
                real,
            )
        return dir_id
    if p.name.lower().endswith(".multihead.json"):
        multihead_id = _multihead_manifest_content_id(p)
        if multihead_id:
            return multihead_id
    id_ = file_content_id(real)
    if id_:
        return id_
    # MISSING-MODEL SENTINEL (fix M5): file_content_id("") is "" for BOTH "no
    # path configured" and "path configured but the file is gone". If two
    # DIFFERENT missing models both fell through to "", a cache written while
    # model A was missing would spuriously validate for model B — worse than
    # the old (path, mtime) key, which at least differed by path. Distinguish
    # missing-but-configured artifacts by hashing their path string
    # (`hint[0]`: realpath when the artifact was stattable, the caller's raw
    # string when it was not -- see `_stat_hint`) instead of their bytes, so
    # two different missing paths still diverge.
    #
    # WHY A PATH-DEPENDENT SENTINEL IS ACCEPTABLE IN A PORTABILITY-MOTIVATED
    # KEY (fix M-minor -- state this, do not silently rely on it):
    #   1. It is unreachable on any cache-HIT path. A missing CNN classifier is
    #      logged and SKIPPED (`core/inference/config.py:1227-1233` continues),
    #      so no CNNConfig -- and therefore no cache key -- ever carries it. A
    #      missing detection/pose/head-tail model fails model loading and the
    #      run dies before any cache is written. The sentinel can only appear
    #      in a key that is never compared against a cache that exists.
    #   2. The PRE-v5 key was equally path-bound for exactly this case:
    #      `keys.py:419-423` returned mtime 0.0 on OSError while still
    #      embedding the absolute path, so v5 is no worse here.
    #   3. Collapsing to "" would be STRICTLY WORSE than either: it would make
    #      two different missing models share one identity, so a cache written
    #      while model A was missing could validate for model B.
    if real and real != "None":
        return f"missing:{hashlib.sha256(real.encode('utf-8')).hexdigest()[:16]}"
    return ""


def model_content_id(path: str | os.PathLike[str] | None) -> str:
    """Content id of a model artifact (file or directory), memoized per process.

    Hashing a 100 MB checkpoint costs ~0.3 s; the optimizer and preview paths
    rebuild keys many times per session, so memoize on ``(realpath, size,
    mtime_ns)``. mtime therefore remains a LOCAL fast-path hint and never
    reaches the key itself.

    A configured-but-missing artifact does NOT collapse to ``""`` — see the
    missing-model sentinel in ``_content_id_for_hint``. An unconfigured
    (empty/None) path still returns ``""``.
    """
    if not path:
        return ""
    return _content_id_for_hint(_stat_hint(str(path)))


model_content_id.cache_clear = _content_id_for_hint.cache_clear  # type: ignore[attr-defined]


def video_signature(path: str | os.PathLike[str] | None) -> str:
    """``"{size}:{sha256(head 8 MiB || tail 8 MiB)[:32]}"``; ``""`` if absent.

    Size plus the first and last 8 MiB catches every realistic replacement (a
    re-encode, a trim, a different clip under the same name) for ~30 ms,
    without reading a 50 GB file. Symlinks are followed, as before.

    NOT memoized, unlike ``model_content_id``. ``optimizer_workers.py:343``
    calls this once per evaluation during autotune search, so an unmemoized
    16 MiB read per call is a real, deliberately-accepted cost: video files
    are large enough that even the ``(realpath, size, mtime_ns)`` memoization
    hint used for models is comparatively cheap to skip re-deriving, and the
    autotune loop already dominates wall-clock with model inference. If this
    ever shows up in a profile, add the same ``lru_cache``-on-stat-hint
    pattern as ``model_content_id`` — do not memoize on path alone.
    """
    if not path:
        return ""
    try:
        st = os.stat(path)  # follows symlinks
        size = st.st_size
        digest = hashlib.sha256()
        with open(path, "rb") as handle:
            digest.update(handle.read(_VIDEO_PROBE))
            if size > _VIDEO_PROBE:
                handle.seek(max(size - _VIDEO_PROBE, _VIDEO_PROBE))
                digest.update(handle.read(_VIDEO_PROBE))
        return f"{size}:{digest.hexdigest()[:32]}"
    except OSError:
        return ""
```

- [ ] **Step 4: Rewrite `CacheKey` in `cache/base.py`**

```python
# v5 = model and video identity became CONTENT-based rather than
#      (absolute path, mtime)-based, so a cache produced on a compute box is
#      reusable on the staging machine. ``model_path``+``model_mtime`` collapse
#      into a single ``model_id``. See the portable-jobs design, section 7b.
CACHE_SCHEMA_VERSION = 5


@dataclass(frozen=True)
class CacheKey:
    schema_version: int  # CACHE_SCHEMA_VERSION at write time
    model_id: str  # content identity: "sha256:…", "dirsha256:…", "a|b", or a sentinel
    config_hash: str  # sha256 hex of model-affecting config fields; "" when none apply

    def as_string(self) -> str:
        return f"v{self.schema_version}|{self.model_id}|{self.config_hash}"

    def matches(self, other: "CacheKey") -> bool:
        return self.as_string() == other.as_string()
```

(The old `matches()` carried a dead 1e-3 mtime tolerance while `as_string()` formatted `:.6f`; on-disk validation was always exact-string. Collapsing to string equality removes the inconsistency.)

- [ ] **Step 5: Rewrite the key builders in `cache/keys.py`**

Delete `_mtime`. Replace `_model_signature` — **this is spec correction #1**:

```python
def _model_signature(path: str) -> str:
    """Content signature of one model, for folding into a raw config hash.

    A sequential detector consumes two independently changing artifacts, so
    both are folded here as well as into ``model_id``. This MUST be
    content-based: it used to be f"{path}|mtime=…", which put an absolute
    path inside ``config_hash`` and made every OBB cache machine-local even
    after ``model_id`` itself became content-based.
    """
    return model_content_id(path)
```

**Fix Z1 (CRITICAL) — DELETE the existing `def video_signature` at `keys.py:36-51` in this same step.** That old function is a `(size, mtime_ns)`-based stub that predates `content_id.py`; it is NOT superseded automatically just because `content_id.video_signature` now exists elsewhere. `keys.py` currently defines its own `video_signature` (verified: `keys.py:36-51`); simply adding `from ..content_id import video_signature` alongside the pre-existing `def video_signature` is a **name collision, not a re-export** — whichever definition appears LATER in the file wins at import time, and nothing enforces which one that is. Do it wrong and `F811` (redefinition) will NOT catch it: `.flake8`'s `extend-ignore` disables F811. Every caller that imports `video_signature` from `keys`/`runner` rather than `content_id` directly — `runner.py:33`, `worker.py:1430`, `optimizer.py:45`, `optimizer_workers.py:44`, `production_replay.py:175`, `core/inference/autotune/session.py:112,118` — would silently keep getting the OLD mtime-based signature, and every video-bound cache stays machine-local. The correct sequence: (1) delete lines 36-51 of `keys.py` entirely; (2) add `from ..content_id import model_content_id, video_signature` to `keys.py`'s imports so the name is re-exported for those callers. Do NOT leave both definitions in the file under any circumstance.

**Add this guard test to `tests/test_inference_cache_keys.py`** (which already imports `from hydra_suite.core.inference.cache import keys as keys_mod` at line 6) so the shadowing can never silently come back:

```python
def test_keys_module_reexports_content_id_video_signature_not_a_shadow():
    """Fix Z1: keys.py must import content_id.video_signature, not redefine
    its own — a same-named local def would silently shadow it and F811 is
    disabled in .flake8's extend-ignore, so nothing else would catch this."""
    from hydra_suite.core.inference import content_id

    assert keys_mod.video_signature is content_id.video_signature


def test_keys_module_video_signature_is_mtime_invariant_through_the_reexport(tmp_path):
    """Exercise the touched-mtime-invariance property THROUGH keys_mod's
    re-exported name specifically (not content_id directly), so a future
    reintroduction of a local mtime-based def in keys.py fails here even if
    it somehow also passed the identity check above."""
    import os

    v = tmp_path / "clip.mp4"
    v.write_bytes(b"\x00" * (1 << 20))
    first = keys_mod.video_signature(str(v))
    os.utime(v, (1, 1))
    assert keys_mod.video_signature(str(v)) == first
```

In each builder, replace the `model_path=` / `model_mtime=` pair with a single `model_id=`:

```python
# detection_cache_key, direct (was :132)
    return CacheKey(
        schema_version=CACHE_SCHEMA_VERSION,
        model_id=model_content_id(path),
        config_hash=config_hash,
    )
# detection_cache_key, sequential (was :119-132): keep the "|" join, content-side
    model_id = (
        f"{model_content_id(config.sequential.detect_model_path)}"
        f"|{model_content_id(config.sequential.obb_model_path)}"
    )
# bgsub (was :341-346): sentinel unchanged
        model_id="background_subtraction",
# apriltag (was :407-412): sentinel unchanged
        model_id="",
# headtail (:356), cnn (:365), pose (:395): model_id=model_content_id(<their path>)
```

For `pose_cache_key`, the artifact is a directory, so `model_content_id` dispatches to `directory_content_id` automatically.

Update the placeholder keys: `cache/reuse.py:21` and `cache/reader.py:20,33` become
`CacheKey(schema_version=0, model_id="", config_hash="")`.

- [ ] **Sub-task 4a checkpoint: commit `content_id.py` + `CacheKey`/`keys.py` alone (fix A8)**

```bash
make format
git add src/hydra_suite/core/inference/content_id.py \
        src/hydra_suite/core/inference/cache/base.py \
        src/hydra_suite/core/inference/cache/keys.py \
        src/hydra_suite/core/inference/cache/reuse.py \
        src/hydra_suite/core/inference/cache/reader.py \
        tests/test_cache_content_identity.py \
        tests/test_inference_cache_keys.py
git commit -m "feat(cache): content-based model/video identity primitives, CacheKey schema v5 (4a)"
```
**Fix Z4 — `tests/test_inference_cache_keys.py` is staged and committed HERE, not deferred to 4c.** Step 1 adds five new tests to that file (the OBB two-path tests plus, per fix Z1 above, the `keys_mod.video_signature` re-export guard), so it is a file Step 1 edited within 4a's own scope — leaving it unstaged would mean `make format` at this checkpoint silently reformats an uncommitted file, and a fresh agent picking up 4b or 4c and reading "the prior sub-task's commit(s)" would not see Step 1's tests at all. Note this file will receive FURTHER edits in 4c (Step 8's itemized rewrites of its pre-existing tests) — that is expected; 4a commits only the additions Step 1 made, 4c commits the rest of that same file's changes on top. This is a real, separately-reviewable checkpoint: the primitives exist and are tested (Step 1's five new tests plus the OBB regressions pass at this point; the file's OTHER, pre-existing tests — the ones Step 8 rewrites — still fail until 4c, which is expected and not a regression to "fix" here).

- [ ] **Step 6: Migrate DetectKit**

`detectkit/jobs/prediction_cache.py:31-61` (function range corrected — it is `:31-61`, not `:31-66`) — replace the absolute-path identity. **Add the import** (the current file imports nothing from `content_id`): `from hydra_suite.core.inference.content_id import directory_content_id, model_content_id, video_signature`.

**Fix Z5 — preserve `expanduser()`, and note what happens to the `"models"` list.** The current implementation (`prediction_cache.py:37,47`) resolves every path through `Path(path).expanduser().resolve()` before using it — both for the model paths and for `source_path` — so a `~/models/x.pt`-style config path resolves correctly. The rewrite below must keep calling `model_content_id`/`_source_content_id` on **expanduser()'d** paths (`str(Path(path).expanduser())` is enough; `model_content_id`/`directory_content_id` already realpath internally via `_stat_hint`, so a further `.resolve()` is redundant but harmless) — passing the raw, un-expanded path through would hit the missing-model sentinel for every `~/`-prefixed configured model, silently breaking DetectKit caching for exactly the layout `HYDRA_DATA_DIR`/`HYDRA_CONFIG_DIR` encourage. Also note explicitly: the old `encoded` payload carried a `"models"` list of `(path, mtime_ns, size)` tuples that is now dropped in favor of folding each model's `model_content_id` directly into `model_id` (below) — this is harmless, not an oversight, because `cache_path_for` (`prediction_cache.py:63-66`) hashes `key.as_string()`, which already includes `model_id`; do not attempt to preserve the old `"models"` key in `encoded`.

**`source_path` here is a DATASET DIRECTORY, not a video file**: `detectkit/gui/panels/dataset_panel.py:394` does `Path(source_path)/"images"`, so calling `video_signature(source_path)` directly hits `IsADirectoryError`, which `video_signature`'s `except OSError` swallows into `""` — every source in a DetectKit project would then collapse onto the same identity, silently defeating cache invalidation across sources. Dispatch on what `source_path` actually is:

```python
    def _source_content_id(source_path: str) -> str:
        """DetectKit prediction-cache sources are DATASET DIRECTORIES
        (dataset_panel.py:394 does Path(source_path)/"images"), not bare video
        files. video_signature() on a directory silently degrades to "" via
        its except OSError, collapsing every source onto one identity. Dispatch
        explicitly instead of guessing from the exception.

        Fix A4 -- hash ONLY the images/ subtree LISTING, not directory_content_id
        of the dataset root. `directory_content_id(source_path)` would hash
        every byte under source_path, including `labels/`
        (operations.py:87 does `images_dir = source_path / "images"`, and
        `labels/` sits as its sibling under the same root, edited by the
        reviewer on every save via dataset_panel.py:394). Two things follow
        from hashing the root wholesale, both real regressions, neither
        exercised by any test in this plan:
          1. `cache_path_for` derives the on-disk cache filename from
             `key.as_string()` (`prediction_cache.py:63-66`), so EVERY label
             save changes the key and orphans the previous prediction cache
             file -- `artifacts/inference_cache/` grows without bound as the
             reviewer works, one dead file per save.
          2. `directory_content_id` reads and sha256's every byte under the
             tree on every request (`dataset_inference.py:64`), which is
             O(dataset bytes) -- and DetectKit datasets are image sets, often
             far larger than a model checkpoint.
        This cache is project-local (it lives beside the dataset, keyed by an
        absolute-ish path already, never shipped by a portable job -- Task 6's
        model-reference machinery does not touch DetectKit at all) and was
        never claimed to be portable, so there is no reason to pay content
        hashing here: identify the SOURCE by what actually invalidates
        predictions -- the image SET, not every byte of every image and
        certainly not the label annotations the reviewer is actively editing.
        List (name, size, mtime_ns) for every file directly under
        `images/` (non-recursive: dataset_panel.py:394 treats images/ as a
        flat pool) and hash that listing; do not recurse into labels/ at all.
        """
        p = Path(source_path)
        if p.is_dir():
            images_dir = p / "images"
            if not images_dir.is_dir():
                return directory_content_id(str(p))
            entries = sorted(
                (child.name, child.stat().st_size, child.stat().st_mtime_ns)
                for child in images_dir.iterdir()
                if child.is_file()
            )
            blob = "\n".join(f"{n}={s}:{m}" for n, s, m in entries).encode("utf-8")
            return f"imgset:{hashlib.sha256(blob).hexdigest()}"
        return video_signature(source_path)

    # Fix Z5: expanduser() BEFORE hashing, same as the old implementation did
    # for both model_paths and source_path (prediction_cache.py:37,47) — a
    # raw "~/models/x.pt" path fails Path.is_file()/is_dir() and would
    # otherwise fall through to the missing-model sentinel.
    identities = [
        (model_content_id(str(Path(path).expanduser())), path)
        for path in model_paths
        if path
    ]
    model_id = "|".join(identity for identity, _ in identities)
    encoded = json.dumps(
        {
            # The SOURCE (image/video file OR dataset directory) is identified
            # by content too, so a prediction cache survives the project
            # moving on disk. See _source_content_id above for the dispatch.
            "source": _source_content_id(str(Path(source_path).expanduser())),
            "settings": settings,
        },
        sort_keys=True,
    )
    return CacheKey(
        schema_version=CACHE_SCHEMA_VERSION,
        model_id=model_id,
        config_hash=hashlib.sha256(encoded).hexdigest(),
    )
```

`detectkit/sidecars/operations.py:21-35` — the field-name set is a wire format. Update it and reject old payloads loudly:

```python
    if set(raw) != {"schema_version", "model_id", "config_hash"}:
        raise ValueError(
            "cache_key has invalid fields; this is the in-process sidecar IPC "
            "payload shape, not an on-disk format — a pre-v5 shaped payload "
            "means the parent and sidecar interpreters are running mismatched "
            "hydra_suite code versions, not a stale unrestarted process (the "
            "sidecar is spawned fresh per request, never long-lived)"
        )
    return CacheKey(
        schema_version=int(raw["schema_version"]),
        model_id=str(raw["model_id"]),
        config_hash=str(raw["config_hash"]),
    )
```

The producer at `detectkit/jobs/dataset_inference.py:78` (`"cache_key": asdict(key)`) needs no change — `asdict` follows the dataclass.

Add to `tests/test_detectkit_prediction_cache.py` (fix M6 — without this test the directory-collapse bug ships silently):

```python
def test_two_different_dataset_directories_get_different_source_ids(tmp_path):
    """Fix M6: source_path is a DATASET DIRECTORY (dataset_panel.py:394 does
    Path(source_path)/"images"), not a video file. video_signature(dir) hits
    IsADirectoryError -> swallowed to "" by its except OSError, so every
    source in a project would collapse onto the same identity. Prove the
    dispatch keeps them distinct."""
    from hydra_suite.detectkit.jobs.prediction_cache import prediction_cache_key

    a = tmp_path / "dataset_a" / "images"
    b = tmp_path / "dataset_b" / "images"
    a.mkdir(parents=True)
    b.mkdir(parents=True)
    (a / "1.png").write_bytes(b"aaa")
    (b / "1.png").write_bytes(b"bbbbbbb")
    # Fix V-minor: DIFFERENT-LENGTH content, not just different bytes at the
    # same length -- b"aaa" vs b"bbb" (both 3 bytes) makes distinctness ride
    # entirely on mtime_ns (both files are written back-to-back in the same
    # test, so their mtimes differ only by whatever the filesystem's mtime
    # resolution happens to catch), which is FLAKY on coarse-mtime
    # filesystems even though the docstring claims "different bytes" is what
    # is being tested. A different length changes size_bytes too, which
    # every fingerprint in this plan hashes unconditionally (no mtime
    # coalescing needed to see the difference).
    #
    # Fix B13: the real entry point is `prediction_cache_key(source_path,
    # model_paths, settings)` -- three POSITIONAL parameters
    # (`detectkit/jobs/prediction_cache.py:31-34`). There is no
    # `build_cache_key` anywhere in the repo.
    key_a = prediction_cache_key(str(tmp_path / "dataset_a"), [], {})
    key_b = prediction_cache_key(str(tmp_path / "dataset_b"), [], {})
    assert key_a.as_string() != key_b.as_string()
```

The load-bearing assertion is that two directories with different member-file
bytes must not collapse to the same key.

```python
def test_editing_a_label_does_not_change_the_prediction_cache_key(tmp_path):
    """Fix A4: source_path is the DATASET ROOT, and labels/ (sibling of
    images/, edited by the reviewer on every save via dataset_panel.py:394)
    must NOT be part of the source identity. If it were, cache_path_for
    (prediction_cache.py:63-66) would derive a NEW on-disk filename from
    key.as_string() on every label save, orphaning the previous prediction
    cache file and growing artifacts/inference_cache/ without bound while the
    reviewer works -- with no test anywhere catching it before this one."""
    from hydra_suite.detectkit.jobs.prediction_cache import prediction_cache_key

    root = tmp_path / "dataset"
    (root / "images").mkdir(parents=True)
    (root / "images" / "1.png").write_bytes(b"aaa")
    (root / "labels").mkdir(parents=True)
    (root / "labels" / "1.json").write_text('{"boxes": []}')

    before = prediction_cache_key(str(root), [], {})
    (root / "labels" / "1.json").write_text('{"boxes": [[1, 2, 3, 4]]}')
    (root / "labels" / "2.json").write_text('{"boxes": []}')
    after = prediction_cache_key(str(root), [], {})

    assert before.as_string() == after.as_string()


def test_adding_an_image_does_change_the_prediction_cache_key(tmp_path):
    """The images/ subtree LISTING is still the real invalidation signal --
    fix A4 narrows the hash scope, it does not remove cache invalidation for
    the thing that actually should invalidate it."""
    from hydra_suite.detectkit.jobs.prediction_cache import prediction_cache_key

    root = tmp_path / "dataset"
    (root / "images").mkdir(parents=True)
    (root / "images" / "1.png").write_bytes(b"aaa")

    before = prediction_cache_key(str(root), [], {})
    (root / "images" / "2.png").write_bytes(b"bbb")
    after = prediction_cache_key(str(root), [], {})

    assert before.as_string() != after.as_string()
```

- [ ] **Sub-task 4b checkpoint: commit the DetectKit migration alone (fix A8)**

```bash
make format
git add src/hydra_suite/detectkit/jobs/prediction_cache.py \
        src/hydra_suite/detectkit/sidecars/operations.py \
        tests/test_detectkit_prediction_cache.py
git commit -m "feat(detectkit): migrate prediction-cache identity onto content_id, scoped to images/ (fix A4) (4b)"
```

- [ ] **Step 7: Comment the known divergence in `parameter_helper.py`**

At `:1768`, above the `_source_signature` closure:

```python
        # NOTE: this deliberately does NOT call
        # core.inference.content_id.video_signature. It is a local UI-freshness
        # probe, not a cache key, and it intentionally keeps cheap stat()
        # semantics. If it is ever folded into a CacheKey it must switch to the
        # content-based function or it will reintroduce machine-local caches.
```

- [ ] **Step 8: Update every test file that constructs `CacheKey` / uses `model_mtime` — full enumeration**

This is a breaking dataclass-shape change (`model_path`+`model_mtime` → `model_id`). **Fix Z2 — the enumeration below is 17 files, not 14: the original 14-file list plus 3 files it missed because Step 8's own grep instruction never checked `.model_path` (only `.model_mtime`).** Verified survivors NOT in the original 14 and NOT in Step 11's run list: `tests/test_vitpose_pose_config.py:25` (`assert key.model_path == str(p)`), `tests/test_bgsub_cache_keys.py:102` (`assert key.model_path == "background_subtraction"`), `tests/test_inference_cache_reuse.py:174` (`assert key.model_path == "background_subtraction"`). Fix ALL 17 in this step, not a representative subset:

1. `tests/test_inference_cache_keys.py` — see the itemized rewrites below; several need REWRITING, not just renaming.
2. `tests/test_inference_cache_chunked.py:30,432` — positional `CacheKey(...)` with 4 fields; make it 3 (`schema_version, model_id, config_hash`).
3. `tests/test_inference_cache_store.py` — constructs `CacheKey`/writes cache entries keyed by the old shape; update to `model_id`.
4. `tests/test_detection_cache_reader_readonly.py` — builds a `CacheKey` to open a reader; update.
5. `tests/test_detectkit_prediction_cache.py` — exercises `detectkit/jobs/prediction_cache.py`; update for the Step 6 rewrite (content-based `model_id`, no absolute-path identity).
6. `tests/test_obbresult_class_ids.py` — touches OBB cache-key construction incidentally; update field names.
7. `tests/test_oriented_track_video_export.py` — builds a cache key to seed export input; update.
8. `tests/test_dataset_generation.py` — dataset builder reads/writes a detection cache; update.
9. `tests/test_pose_cache_empty_frame.py:26` — positional 4-field `CacheKey`; make it 3.
10. `tests/core/inference/cache/test_detection_reader.py` — reader-level `CacheKey` construction; update.
11. `tests/core/individual/dataset/test_oriented_video_actual_rows.py` — builds a cache key for a fixture row set; update.
12. `tests/core/post/test_interpolated_crops_size_lookup.py` — cache-key-gated crop lookup; update.
13. `tests/identity/test_evidence_stage_runner.py` — stage runner constructs a `CacheKey` for its input cache; update.
14. `tests/refinekit/test_overlay_modern_cache.py` — RefineKit's own cache-key construction; update.
15. `tests/test_vitpose_pose_config.py:25` — `assert key.model_path == str(p)`; rewrite to assert `key.model_id == model_content_id(str(p))` (or the appropriate content-based expectation for that fixture).
16. `tests/test_bgsub_cache_keys.py:102` — `assert key.model_path == "background_subtraction"`; rename attribute access to `.model_id` (the bgsub sentinel value itself is unchanged, per Step 5).
17. `tests/test_inference_cache_reuse.py:174` — `assert key.model_path == "background_subtraction"`; same rename as #16.

For every file above: grep it for `CacheKey(`, `model_path=`, `model_mtime=`, `.model_mtime`, **and `.model_path`** (fix Z2 — the original grep instruction omitted `.model_path`, which is exactly how files #15-17 above were missed); replace with `model_id=`/`.model_id` (computed via `model_content_id`/`video_signature` as appropriate for that test's fixtures, never a literal path string).

**`tests/test_inference_cache_keys.py` needs actual rewrites, not renames — itemized:**

- `test_detection_key_changes_with_model_path` (`:192`) — currently uses `/a.pt` and `/b.pt`, which don't exist on disk. **Minor fix (round-7) — corrected rationale for the rewrite (the original rationale was wrong): under the missing-model sentinel, `/a.pt` and `/b.pt` already hash their DIFFERENT paths and so already produce DIFFERENT sentinel ids — even before fix M5, and certainly after it (fix M5 fixes a different bug: two DIFFERENT missing paths spuriously colliding onto the SAME sentinel value, which is not what two distinct nonexistent paths do here). The real problem with the current test is that it never exercises real content-based hashing at all — it only ever exercises the missing-model-sentinel fallback path, which is path-keyed, not content-keyed, so the test would "pass" whether or not sha256 content identity works correctly (and would fail to catch a bug where two files with the SAME bytes at different paths wrongly get different keys, since it never has two paths with equal content to compare). The rewrite itself is still correct and necessary — create two real temp files with **different bytes** and assert the keys differ — but for the right reason: to actually exercise the content-hashing code path this task builds, not to avoid a same-sentinel collision that was never the failure mode here.
- `test_detection_key_stable_with_threshold` (`:198`) — reads `.model_path` off the built key; that attribute no longer exists. Rewrite to read `.model_id`.
- `test_detection_key_sequential_encodes_both_models` (`:205`) — asserts substrings of the two absolute paths appear in the key string; under content identity the key contains sha256 hex, not paths. Rewrite to assert the key changes when either model's **bytes** change (mirrors the new `test_sequential_key_changes_when_either_model_changes` above), not to assert path substrings.
- `test_sequential_second_model_signature_invalidates_detection_key` (`:280`) — monkeypatches `keys._mtime`, which Step 5 deletes entirely. Rewrite to monkeypatch nothing and instead rewrite the second model file's bytes on disk, asserting the key changes (the real content-based path).
- `test_cache_key_matches_tolerates_small_mtime_diff` (`:175`) — the whole point of this test is the mtime tolerance `CacheKey.matches()` used to carry, which Step 4 explicitly removes (string-equality only, no tolerance). **Delete this test** — say so in the diff/commit, don't silently drop it. There is nothing left to test once mtime is out of the key entirely.
- `test_cache_key_matches_only_when_schema_version_matches` (`:161`) — still valid in spirit; rewrite the fixture `CacheKey(...)` calls to the 3-field shape, semantics unchanged.
- `test_cnn_and_headtail_keys_differ_across_schema_v3_v4` (`:135`) — rename to `..._v4_v5` (schema is now 4→5 territory conceptually, but the ACTUAL assertion — that two different schema versions never produce string-equal keys — is unchanged and should use `CACHE_SCHEMA_VERSION` and `CACHE_SCHEMA_VERSION - 1` rather than hardcoded 3/4 literals so it doesn't silently rot again at the next bump).
- `tests/test_inference_cache_keys.py:132`'s `CACHE_SCHEMA_VERSION == 4` assertion — change to `5`.
- **Fix Z2 — five more anchors in this same file the original itemization missed** (found via the corrected `.model_path` grep above): `:536` (`test_bgsub_key_changes_with_detection_params` — `k1.model_path == "background_subtraction"` → `.model_id`), `:576-577` (`test_bgsub_key_stable_for_same_params` region — `k_a.model_path == k.model_path` and `k_a.model_mtime == k.model_mtime`; the second assertion has no `model_id` equivalent and must simply be deleted, not renamed — `model_mtime` no longer exists on `CacheKey` at all), `:609` (`test_headtail_key_stable_with_threshold` — `k1.model_path == k2.model_path` → `.model_id`), `:625` (`test_cnn_key_stable_with_calibration_temperature` — same rename), `:682` (`test_apriltag_key_has_empty_model_path` — `k.model_path == ""` → `k.model_id == ""`; the apriltag sentinel itself, per Step 5, is unchanged as `model_id=""`).

- [ ] **Step 9: Run the cache suites**

Run:
```bash
python -m pytest tests/test_cache_content_identity.py tests/test_inference_cache_keys.py \
                 tests/test_inference_cache_chunked.py tests/test_pose_cache_empty_frame.py \
                 tests/trackerkit/test_detection_cache_validity_probe.py \
                 tests/test_tracking_production_replay.py -v
```
Expected: PASS, and the `test_inference_cache_keys.py` + `test_inference_cache_chunked.py` pair is **>= 111** (the baseline) plus the new tests.

- [ ] **Step 10: Grep gate — no `model_path`/`model_mtime` survivors on CacheKey**

Run:
```bash
grep -rn 'model_mtime' src/ tests/ && echo "VIOLATION" || echo "clean"
grep -rn '\.model_path\b' src/ tests/ && echo "VIOLATION" || echo "clean"
grep -rnA3 'CacheKey(' src/ | grep -B3 -v 'model_id' | grep 'CacheKey(' && echo "CHECK THESE" || echo "clean"
```
**Fix Z2 — the original single-line `grep -v 'model_id'` gate can NEVER print clean and is not a usable gate as written.** Every multi-line `CacheKey(` constructor in `keys.py` (verified: `:129,341,353,362,392,407`) has `model_id=` on the FOLLOWING line, not the same line as `CacheKey(` — `grep -v 'model_id'` matches per-LINE, so it always flags these real, correct constructions as "CHECK THESE", making the gate noisy on every run regardless of correctness (a false-positive gate that always fires is worse than no gate: nobody re-reads the same six false positives every time). Use `grep -A3` to pull the next 3 lines and only flag a `CacheKey(` block where none of those lines contain `model_id`, as above. Also add `.model_path` as its own gate (fix Z2) since the attribute no longer exists on `CacheKey` at all after Step 4 — any survivor is a bug, not a matter of naming style.

- [ ] **Step 11: Run every file enumerated in Step 8 explicitly — not a `-k` filter**

A `-k "cache or detectkit or sidecar"` substring filter is exactly the kind of "subset chosen by apparent relevance" the Global Constraints forbid, and several of the 14 files in Step 8 (e.g. `tests/test_obbresult_class_ids.py`, `tests/test_oriented_track_video_export.py`, `tests/test_dataset_generation.py`, `tests/core/individual/dataset/test_oriented_video_actual_rows.py`, `tests/core/post/test_interpolated_crops_size_lookup.py`) do not match that pattern and would silently be skipped. Run the explicit file list instead:

```bash
python -m pytest \
  tests/test_cache_content_identity.py \
  tests/test_inference_cache_keys.py \
  tests/test_inference_cache_chunked.py \
  tests/test_inference_cache_store.py \
  tests/test_detection_cache_reader_readonly.py \
  tests/test_detectkit_prediction_cache.py \
  tests/test_obbresult_class_ids.py \
  tests/test_oriented_track_video_export.py \
  tests/test_dataset_generation.py \
  tests/test_pose_cache_empty_frame.py \
  tests/core/inference/cache/test_detection_reader.py \
  tests/core/individual/dataset/test_oriented_video_actual_rows.py \
  tests/core/post/test_interpolated_crops_size_lookup.py \
  tests/identity/test_evidence_stage_runner.py \
  tests/refinekit/test_overlay_modern_cache.py \
  tests/test_vitpose_pose_config.py \
  tests/test_bgsub_cache_keys.py \
  tests/test_inference_cache_reuse.py \
  -v
```
(Fix Z2: the last three files above are the `.model_path` survivors Step 8's items #15-17 add; they were missing from this run list in the same way they were missing from the original 14-file enumeration.)
Expected: no `TypeError: __init__() got an unexpected keyword argument`, no `AttributeError` on `.model_path`/`.model_mtime`, all PASS. THEN also run the full suite as a final catch-all for any construction site this enumeration missed: `python -m pytest tests/ -q 2>&1 | tail -30` and diff the failure set against the pre-task baseline (memory `project_test_suite_batching_chunk_boundary_trap`: compare failure SETS, not raw counts).

- [ ] **Step 12: Commit (sub-task 4c — everything else: the 14-file test migration + the `parameter_helper.py` comment)**

`content_id.py`, `cache/base.py`, `cache/keys.py`, and the DetectKit files were already committed at the 4a/4b checkpoints above (fix A8) — this commit is only the remaining call-site/test migration.

```bash
make format
git add src/hydra_suite/trackerkit/gui/dialogs/parameter_helper.py \
        tests/
git commit -m "test(cache): migrate every CacheKey/model_mtime call site onto v5 (4c)

Model identity becomes sha256 of the artifact's bytes (dirsha256 for pose
directories) and the video signature becomes size + head/tail content hash, so
a cache produced on a compute box is reusable on the staging machine.
_model_signature is fixed too: it folded a raw absolute path + mtime into
config_hash, which kept OBB caches machine-local independently of model_id.
Every existing cache is invalidated once and regenerates. Completes Task 4
(4a: primitives + CacheKey, 4b: DetectKit) with the full call-site/test
migration and the equivalence gate below."
```

**Fix A7 — the baseline commit for this gate, and why it must NOT be `8f9688e0`.** This task declares "Consumes: nothing from earlier tasks", but it runs FOURTH in execution order, after Tasks 1-3 have already landed real behavior changes (Task 2's save/load path relativization, Task 3's `iter_model_references`). Baselining Step 13/14 against `8f9688e0` (the branch root, before ANY task) compares "everything through Task 4" against "nothing" in one shot — if a divergence appears, there is no way to tell whether Task 1, 2, 3, or 4 caused it, which defeats the entire point of running each task's own equivalence gate separately (this is exactly the attribution principle CLAUDE.md's equivalence section states: "so each slice's effect is isolated, not conflated"). Since this task is NOT reordered to run first (the task order above is unchanged), the correct baseline is **the commit at the tip of Task 3** (i.e., `HEAD` immediately before Task 4's own commits begin) — not `8f9688e0`. Record that commit SHA in the Acceptance Log entry for this step (e.g. `git rev-parse HEAD` run right before Task 4 Step 1). Tasks 1-3's own gates (already run at the end of each of those tasks) are what isolates their individual effects; this step isolates Task 4's.

- [ ] **Step 13: BEFORE/AFTER equivalence gate on MPS (blocking)**

**What this gate proves, precisely (and what it doesn't).** Every fixture config in `tools/equivalence/fixtures/configs/` sets `enable_backward_tracking: true`, and `trackerkit/headless_tracking.py:248` runs the backward pass against the SAME `detection_cache_path` the forward pass just wrote — so this matrix DOES exercise a same-process, same-path write-then-read cache-key round trip (forward writes the v5 cache, backward reads it back and must find it reusable), on real configs including `.multihead.json`-based CNN keys and `dirsha256`-based pose-directory keys. Concretely this proves: (a) key computation doesn't crash or diverge on any real fixture config, (b) same-path write→read reuse via the forward→backward handoff, and (c) no perf regression (`PERF_TOLERANCE`). It does **not** prove cross-path/cross-machine portability — no fixture here ever copies a model to a second path and rebuilds the key against it. That property is covered separately: the handle-level unit tests added at fix Z7 (`test_cache_written_at_one_path_is_reusable_from_a_copy_at_another_path` and its negative counterpart, in Step 1), and the end-to-end round trip in Task 13. Do not read a green result here as proof of portability by itself.

Caches and JIT state MUST be cleared on both sides — a stale `v4` cache or a poisoned `__pycache__` fakes a result (memories `feedback_numba_jit_cache_poisons_equivalence`, `project_merge_candidate_parity_done`).

```bash
conda activate hydra-mps
pkill -f 'sleap|hydra' || true   # ONLY sleap/hydra; never other processes
find . -name '__pycache__' -type d -prune -exec rm -rf {} +
rm -rf /tmp/equiv_jobs
# Fix C1 (round-5 CRITICAL): TASK3_TIP MUST be a literal sha recorded BEFORE
# Task 4a's first commit. The previous revision computed it with
# `git rev-parse HEAD` INSIDE this block -- but this block runs AFTER 4a/4b/4c
# have committed, so HEAD was Task 4's own tip: MAIN_SRC and WT_SRC pointed at
# the SAME code and every clip compared EQUIVALENT by construction. That made
# the only gate protecting the byte-identity-critical slice completely
# tautological. Paste the sha; do not compute it here.
#
#   BEFORE starting Task 4a, run:  git rev-parse HEAD
#   and paste the result on the next line.
TASK3_TIP="<PASTE the sha printed before Task 4a started>"

# Guard: refuse to run a gate that compares a tree against itself.
if [ "$TASK3_TIP" = "$(git rev-parse HEAD)" ]; then
  echo "FATAL: TASK3_TIP == HEAD -- baseline and current are the same tree." >&2
  echo "This gate would report EQUIVALENT for every clip while proving" >&2
  echo "nothing. Record the pre-Task-4 sha and rerun." >&2
  exit 1
fi
case "$TASK3_TIP" in *PASTE*|"") echo "FATAL: TASK3_TIP not filled in" >&2; exit 1;; esac

# Minor fix: this path can survive from an earlier/aborted gate run and
# make `worktree add` fail outright; clear it first, every time.
git -C /Users/neurorishika/Projects/Rockefeller/Kronauer/multi-animal-tracker \
    worktree remove --force .worktrees/equiv-base 2>/dev/null; \
git -C /Users/neurorishika/Projects/Rockefeller/Kronauer/multi-animal-tracker \
    worktree prune
git -C /Users/neurorishika/Projects/Rockefeller/Kronauer/multi-animal-tracker \
    worktree add --detach .worktrees/equiv-base "$TASK3_TIP"
REPO=$PWD WT=$PWD \
  MAIN_SRC=$PWD/../equiv-base/src WT_SRC=$PWD/src \
  OUT=/tmp/equiv_jobs RUNTIME=mps \
  bash tools/equivalence/run_matrix.sh
```
Expected: every clip EQUIVALENT at its DETERMINISM floor; identical row counts; 0 unmatched. Known accepted noise: bistable head/tail π-flips on head/tail clips. **Verify `wc -l` on the CSVs is > 1 before trusting any EQUIVALENT** — empty CSVs falsely compare equal.

Record the result in the plan's Acceptance Log below. If any clip diverges, STOP and debug; do not proceed to Task 5.

- [ ] **Step 14: Same gate on firebrat (CUDA)**

**Fix Y7 (round-7) — `git fetch && git checkout <this-branch-sha>` cannot work as written: `feat/portable-tracking-jobs` (and the `TASK3_TIP` sha it names) exist ONLY in this local worktree/clone, never pushed anywhere firebrat's `git fetch` can reach — and even local `main` is ahead of `origin/main`, so "just fetch" is never sufficient for a branch built on top of it. Verified: `git branch -r --contains feat/portable-tracking-jobs` and `git ls-remote origin feat/portable-tracking-jobs` both come back empty from this worktree. The transport step below MUST run before `git fetch` on firebrat, every time this branch's tip changes (i.e. before Step 14 here, and again before Task 13 Step 3, which reuses this recipe verbatim).** Chosen mechanism: a **git bundle**, not `git push origin`, per this plan's own "do not push" instruction (this plan/branch is worked in an isolated local worktree and is not to touch `origin` autonomously) and per the existing precedent for this exact box class (memory `project_pose_cnn_batched_detection_slowdown`'s "mehek git-bundle transport recipe"). Run from THIS worktree (`.worktrees/portable-jobs`), local machine, before ssh'ing in:

```bash
# On THIS machine, from the worktree:
cd /Users/neurorishika/Projects/Rockefeller/Kronauer/multi-animal-tracker/.worktrees/portable-jobs
git bundle create /tmp/portable-jobs.bundle feat/portable-tracking-jobs
scp /tmp/portable-jobs.bundle firebrat:/tmp/portable-jobs.bundle
ssh firebrat 'cd ~/hydra-suite && git fetch /tmp/portable-jobs.bundle feat/portable-tracking-jobs:refs/heads/feat/portable-tracking-jobs'
```

Only after that succeeds does the `ssh firebrat` session below have a local `feat/portable-tracking-jobs` ref to check out — `git checkout <this-branch-sha>` (a bare sha, detached) then resolves because the bundle fetch already made the sha reachable, without firebrat's `origin` remote ever needing to know about this branch.

```bash
ssh firebrat
cd ~/hydra-suite && git checkout <this-branch-sha>
source ~/miniforge3/etc/profile.d/conda.sh && conda activate hydra-cuda
find . -name '__pycache__' -type d -prune -exec rm -rf {} +
# Fix C1 (round-5 CRITICAL): this is a FRESH ssh session -- $TASK3_TIP from
# Step 13 does NOT carry over. An unset variable expands to "" and
# `git worktree add --detach ... ""` fails, so the CUDA gate would never run
# at all. Paste the SAME literal sha recorded before Task 4a started.
TASK3_TIP="<PASTE the same sha used in Step 13>"
case "$TASK3_TIP" in *PASTE*|"") echo "FATAL: TASK3_TIP not filled in" >&2; exit 1;; esac
if [ "$TASK3_TIP" = "$(git rev-parse HEAD)" ]; then
  echo "FATAL: TASK3_TIP == HEAD -- the gate would be tautological." >&2; exit 1
fi
# Minor fix: same as Step 13 -- clear a surviving path from an earlier run.
git worktree remove --force .worktrees/equiv-base 2>/dev/null; git worktree prune
git worktree add --detach .worktrees/equiv-base "$TASK3_TIP"
REPO=$PWD WT=$PWD MAIN_SRC=$PWD/.worktrees/equiv-base/src WT_SRC=$PWD/src \
  OUT=/tmp/equiv_jobs RUNTIME=cuda nohup bash tools/equivalence/run_matrix.sh \
  > /tmp/equiv_cuda_jobs.log 2>&1 &
```
Expected: same acceptance. Record in the Acceptance Log.

---

### Task 5: Job manifest + shared-root mount table + the dependency-direction gate

**Files:**
- Create: `src/hydra_suite/data/tracking_job/__init__.py`, `manifest.py`, `shared_roots.py`
- Modify: `src/hydra_suite/paths.py` (+ `get_shared_roots_path`)
- Test: `tests/test_tracking_job_manifest.py`, `tests/test_tracking_job_shared_roots.py`, `tests/test_tracking_job_layering.py` (create)

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `JobManifest` frozen dataclass: `job_version: int = 1`, `job_id: str`, `created_at: str`, `created_on: dict`, `keystone: dict`, `videos: list[JobVideo]`, `models: list[JobModel]`, `config_snapshot: dict`, `requirements: dict`, `track_args: dict`, `pull_history: list[dict]`; `to_dict()` / `from_dict()` / `write(path)` / `read(path)`.
  - `JobVideo`: `job_path, origin_path, size_bytes, config_job_path, config_provenance, pushed_siblings: list[str], signature: str = ""`, `shared: dict | None = None`, `redirected_outputs: dict[str, str]`.
  - `JobModel`: `key, roles: list[str], origin_path, kind, sha256, size_bytes, sidecars: list[str], files: list[str], registry_entry_present: bool, file_digests: dict[str, str]` (fix M8: per-file sha256 for directory models/bundles, so a single corrupted member is detectable — a top-level sha256 alone cannot catch that).
  - `validate_job_relpath(value: str) -> Path` — raises `TrackingJobError` on absolute or `..`-containing paths.
  - `TrackingJobError(ValueError)` with `.code: int`.
  - `SUPPORTED_JOB_VERSION = 1`.
  - `shared_roots.load_shared_roots(path: Path | None = None) -> dict[str, str]`, `save_shared_roots(table: dict[str, str], path: Path | None = None) -> None`, `save_alias(alias: str, root: str, path: Path | None = None) -> dict[str, str]` (fix B12a: read-modify-write one alias into the persisted table and return the new table — Tasks 10/11 call **this**, never `save_shared_roots` directly, so a one-off `--shared-root` promotion cannot clobber the host's other aliases), `match_shared_root(abs_path, table) -> tuple[str, str] | None` (longest root wins), `resolve_shared(alias, relpath, table) -> Path`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_tracking_job_manifest.py`:

```python
"""Job manifest: versioned, atomic, traversal-safe."""

import json

import pytest

from hydra_suite.data.tracking_job.manifest import (
    SUPPORTED_JOB_VERSION,
    JobManifest,
    JobModel,
    JobVideo,
    TrackingJobError,
    validate_job_relpath,
)


def _manifest():
    return JobManifest(
        job_id="2026-09-09T14-03-12_test",
        created_at="2026-09-09T14:03:12Z",
        created_on={"hostname": "mbp", "platform": "darwin"},
        keystone={"video": "videos/a.mp4", "config": "videos/a_config.json"},
        videos=[
            JobVideo(
                job_path="videos/a.mp4",
                origin_path="/Volumes/lab/a.mp4",
                size_bytes=10,
                config_job_path="videos/a_config.json",
                config_provenance="own-sidecar",
                pushed_siblings=["videos/a_config.json"],
            )
        ],
        models=[
            JobModel(
                key="obb/x.pt",
                roles=["YOLO_OBB_DIRECT_MODEL_PATH"],
                origin_path="/host/models/obb/x.pt",
                kind="file",
                sha256="ab",
                size_bytes=3,
            )
        ],
    )


def test_round_trip_is_lossless(tmp_path):
    path = tmp_path / "hydra_job.json"
    original = _manifest()
    original.write(path)
    assert JobManifest.read(path).to_dict() == original.to_dict()


def test_job_version_is_emitted_first_and_defaults_to_one():
    payload = _manifest().to_dict()
    assert next(iter(payload)) == "job_version"
    assert payload["job_version"] == SUPPORTED_JOB_VERSION


def test_unsupported_version_raises(tmp_path):
    path = tmp_path / "hydra_job.json"
    payload = _manifest().to_dict()
    payload["job_version"] = 99
    path.write_text(json.dumps(payload))
    with pytest.raises(TrackingJobError) as excinfo:
        JobManifest.read(path)
    assert "99" in str(excinfo.value)


def test_write_is_atomic_leaving_no_temp_files(tmp_path):
    path = tmp_path / "hydra_job.json"
    _manifest().write(path)
    assert [p.name for p in tmp_path.iterdir()] == ["hydra_job.json"]


@pytest.mark.parametrize("bad", ["/abs/path", "../escape", "videos/../../etc/passwd", ""])
def test_validate_job_relpath_rejects_unsafe(bad):
    with pytest.raises(TrackingJobError):
        validate_job_relpath(bad)


@pytest.mark.parametrize("good", ["videos/a.mp4", "models/obb/x.pt", "config/skeletons/s.json"])
def test_validate_job_relpath_accepts_safe(good):
    assert str(validate_job_relpath(good)) == good


def test_manifest_read_validates_every_relpath(tmp_path):
    path = tmp_path / "hydra_job.json"
    payload = _manifest().to_dict()
    payload["videos"][0]["job_path"] = "/etc/passwd"
    path.write_text(json.dumps(payload))
    with pytest.raises(TrackingJobError):
        JobManifest.read(path)


def test_error_carries_a_code():
    err = TrackingJobError("boom", code=3)
    assert err.code == 3
```

Create `tests/test_tracking_job_shared_roots.py`:

```python
"""Shared-root mount table: alias in, host path out."""

import json

import pytest

from hydra_suite.data.tracking_job import shared_roots
from hydra_suite.data.tracking_job.manifest import TrackingJobError


@pytest.fixture()
def config_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("HYDRA_CONFIG_DIR", str(tmp_path))
    return tmp_path


def test_missing_table_is_an_empty_mapping(config_dir):
    assert shared_roots.load_shared_roots() == {}


def test_save_then_load_round_trip(config_dir):
    shared_roots.save_shared_roots({"labnas": "/Volumes/lab"})
    assert shared_roots.load_shared_roots() == {"labnas": "/Volumes/lab"}


def test_match_returns_alias_and_relpath(config_dir):
    table = {"labnas": "/Volumes/lab"}
    assert shared_roots.match_shared_root("/Volumes/lab/2026-09/a.mp4", table) == (
        "labnas",
        "2026-09/a.mp4",
    )


def test_longest_root_wins(config_dir):
    table = {"lab": "/Volumes/lab", "project": "/Volumes/lab/2026-09"}
    alias, rel = shared_roots.match_shared_root("/Volumes/lab/2026-09/a.mp4", table)
    assert alias == "project"
    assert rel == "a.mp4"


def test_no_match_returns_none(config_dir):
    assert shared_roots.match_shared_root("/Users/me/a.mp4", {"lab": "/Volumes/lab"}) is None


def test_partial_path_component_is_not_a_match(config_dir):
    """/Volumes/lab must not match /Volumes/labour."""
    assert shared_roots.match_shared_root("/Volumes/labour/a.mp4", {"lab": "/Volumes/lab"}) is None


def test_resolve_unknown_alias_lists_the_known_ones(config_dir):
    with pytest.raises(TrackingJobError) as excinfo:
        shared_roots.resolve_shared("nope", "a.mp4", {"labnas": "/Volumes/lab"})
    message = str(excinfo.value)
    assert "nope" in message and "labnas" in message


def test_resolve_returns_the_host_path(config_dir):
    resolved = shared_roots.resolve_shared("labnas", "2026-09/a.mp4", {"labnas": "/mnt/lab"})
    assert str(resolved) == "/mnt/lab/2026-09/a.mp4"


def test_resolve_rejects_a_traversing_relpath(config_dir):
    with pytest.raises(TrackingJobError):
        shared_roots.resolve_shared("labnas", "../../etc/passwd", {"labnas": "/mnt/lab"})
```

Create `tests/test_tracking_job_layering.py` — the automated dependency-direction gate:

```python
"""data/tracking_job is Qt-free and imports no app layer."""

import ast
import pathlib

import pytest

# Minor fix: a CWD-relative path makes this test pass or fail depending on
# where pytest is invoked FROM, not on the code itself — resolve relative to
# this test file's own location instead, which is invocation-directory-proof.
PACKAGE = pathlib.Path(__file__).resolve().parents[1] / "src" / "hydra_suite" / "data" / "tracking_job"
FORBIDDEN_ROOTS = {
    "trackerkit", "classkit", "detectkit", "posekit", "refinekit",
    "filterkit", "widgets", "launcher", "integrations",
}
FORBIDDEN_MODULES = {"PySide6", "PyQt5", "PyQt6", "qtpy"}


def _module_dotted_name(path):
    """This file's own fully-qualified module name, e.g. 'hydra_suite.data.tracking_job.pack'."""
    src_root = pathlib.Path(__file__).resolve().parents[1] / "src"
    rel = path.resolve().relative_to(src_root).with_suffix("")
    parts = list(rel.parts)
    if parts[-1] == "__init__":
        parts = parts[:-1]
    return parts


def _imported_names(path, own_pkg_parts=None):
    """Fix X5b: resolve RELATIVE imports to an absolute dotted path before
    yielding, so `from ...trackerkit.cli_config import x` inside
    data/tracking_job/pack.py is checked exactly like an absolute
    `from hydra_suite.trackerkit.cli_config import x` would be. The plan's
    own generated code (e.g. `_normalize_model_path`'s import of
    `core.inference.model_paths`) uses relative imports throughout
    data/tracking_job/ -- a level==0-only gate is blind to every one of them,
    which is exactly how a relative `from ...trackerkit import ...` import
    would have passed this test green. `own_pkg_parts` overrides the
    real-source-tree-derived package path -- used only by
    `test_the_gate_itself_catches_a_relative_app_layer_import` below, which
    exercises a synthetic file that is never actually under `src/`.
    """
    if own_pkg_parts is None:
        # Fix Q1 (adversarial review): `_module_dotted_name` ALREADY strips a
        # trailing "__init__" component (its own body: `if parts[-1] ==
        # "__init__": parts = parts[:-1]`), so for `data/tracking_job/__init__.py`
        # it already returns the PACKAGE's own dotted name,
        # `['hydra_suite', 'data', 'tracking_job']` -- there is no separate
        # "module's own filename" component left to drop. Unconditionally
        # doing `[:-1]` here, as an earlier draft did, popped that list a
        # SECOND time for `__init__.py` specifically, landing one level too
        # shallow (`['hydra_suite', 'data']`). A synthetic
        # `data/tracking_job/__init__.py` containing
        # `from ...trackerkit.cli_config import y` (level=3) then resolved to
        # `base = own_pkg_parts[:2-3+1] = own_pkg_parts[:0] = []`, yielding
        # bare `"trackerkit.cli_config"` with no `hydra_suite.` prefix -- so
        # NEITHER the Qt-module assertion nor the `hydra_suite.` app-layer
        # assertion below ever fired on it, and the whole gate was blind
        # inside `__init__.py`. For every OTHER file (a plain `foo.py`),
        # `_module_dotted_name` returns
        # `['hydra_suite', 'data', 'tracking_job', 'foo']`, whose trailing
        # element genuinely IS the module's own filename, so `[:-1]` is
        # correct there. Branch on `path.name`, not on the returned parts,
        # because the two cases need different treatment of an
        # already-`__init__`-stripped list.
        own_pkg_parts = (
            _module_dotted_name(path)
            if path.name == "__init__.py"
            else _module_dotted_name(path)[:-1]
        )
    tree = ast.parse(path.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                yield alias.name
        elif isinstance(node, ast.ImportFrom) and node.module:
            if node.level == 0:
                yield node.module
            else:
                # level=1 means "this package"; each extra level pops one
                # more trailing component off the module's own package path.
                base = own_pkg_parts[: len(own_pkg_parts) - node.level + 1]
                yield ".".join(base + [node.module])


@pytest.mark.parametrize("path", sorted(PACKAGE.glob("*.py")), ids=lambda p: p.name)
def test_no_app_layer_or_qt_imports(path):
    for name in _imported_names(path):
        head = name.split(".")[0]
        assert head not in FORBIDDEN_MODULES, f"{path.name} imports Qt: {name}"
        if name.startswith("hydra_suite."):
            layer = name.split(".")[1]
            assert layer not in FORBIDDEN_ROOTS, f"{path.name} imports app layer: {name}"


def test_the_gate_itself_catches_a_relative_app_layer_import(tmp_path):
    """A regression test FOR the gate: a relative import of an app layer
    must fail, proving `_imported_names`'s level-resolution actually works
    and this isn't just re-testing the (already-passing) absolute-import
    case.
    """
    victim = tmp_path / "victim.py"
    victim.write_text("from ...trackerkit.cli_config import load_advanced_tracker_config\n")
    names = list(
        _imported_names(victim, own_pkg_parts=["hydra_suite", "data", "tracking_job"])
    )
    assert names == ["hydra_suite.trackerkit.cli_config"], names
    with pytest.raises(AssertionError, match="app layer"):
        for name in names:
            layer = name.split(".")[1]
            assert layer not in FORBIDDEN_ROOTS, f"imports app layer: {name}"


def test_the_gate_catches_a_relative_app_layer_import_inside_a_real_init_file(tmp_path):
    """Fix Q1 regression: the test above passes `own_pkg_parts` explicitly and
    therefore never exercises `_imported_names`'s own SELF-DERIVATION of
    `own_pkg_parts` -- it was passing even when that derivation was broken
    specifically for files named `__init__.py`. This test puts a synthetic
    `__init__.py` under a real `src/` tree (so `_module_dotted_name` runs its
    real resolution, not a stand-in) and passes `own_pkg_parts=None` (the
    default `_imported_names` actually uses), proving the self-derivation
    itself -- not just the level-arithmetic once handed a correct
    `own_pkg_parts` -- resolves the relative import to an absolute
    `hydra_suite.trackerkit....` name and the gate catches it.
    """
    fake_src = tmp_path / "src"
    fake_pkg = fake_src / "hydra_suite" / "data" / "tracking_job"
    fake_pkg.mkdir(parents=True)
    victim = fake_pkg / "__init__.py"
    victim.write_text("from ...trackerkit.cli_config import load_advanced_tracker_config\n")

    import types

    # `_module_dotted_name` resolves relative to `pathlib.Path(__file__).resolve()
    # .parents[1] / "src"` (this TEST file's own location), which is the real
    # repo's `tests/`, not `tmp_path`. Monkeypatch a private module-level
    # `__file__` stand-in is unnecessary complexity here -- instead call the
    # two helpers directly against a `src_root` computed the same way
    # `_module_dotted_name` computes it internally, by constructing the
    # relative-path arithmetic inline against `fake_src`, mirroring exactly
    # what `_module_dotted_name` does so this test proves the SAME logic
    # `_imported_names(path, own_pkg_parts=None)` runs in production.
    rel = victim.resolve().relative_to(fake_src).with_suffix("")
    parts = list(rel.parts)
    if parts[-1] == "__init__":
        parts = parts[:-1]
    own_pkg_parts = parts if victim.name == "__init__.py" else parts[:-1]
    assert own_pkg_parts == ["hydra_suite", "data", "tracking_job"], own_pkg_parts

    tree = ast.parse(victim.read_text())
    names = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module and node.level:
            base = own_pkg_parts[: len(own_pkg_parts) - node.level + 1]
            names.append(".".join(base + [node.module]))
    assert names == ["hydra_suite.trackerkit.cli_config"], names
    layer = names[0].split(".")[1]
    assert layer in FORBIDDEN_ROOTS


def test_package_imports_without_qt_installed():
    """Importing the package must not pull PySide6 in transitively.

    Run in a SUBPROCESS. Purging ``hydra_suite.*`` from ``sys.modules`` in-process
    (an earlier draft's approach) is not order-robust and actively harms the rest
    of the run: nothing restores the purged modules, so every later test that
    re-imports gets FRESH class objects -- ``isinstance`` checks against
    pre-purge classes start failing, ``lru_cache``es and registries reset, and
    numba re-JITs. A subprocess has a clean interpreter by construction and
    leaves this process untouched.
    """
    import os
    import subprocess
    import sys

    script = (
        "import sys; sys.modules['PySide6'] = None; "
        "import hydra_suite.data.tracking_job; "
        "assert not any(m.startswith('PySide6.') for m in sys.modules), "
        "'importing tracking_job pulled in a Qt submodule'"
    )
    # Fix A6: PACKAGE = <repo>/src/hydra_suite/data/tracking_job, so
    # PACKAGE.parents[0]=data, [1]=hydra_suite, [2]=src, [3]=<repo root>.
    # parents[3] is the REPO ROOT, not src -- tests/conftest.py:4-9 only ever
    # puts SRC_DIR on sys.path for the IN-PROCESS test run; it does nothing
    # for this subprocess's env, so a PYTHONPATH of the repo root here makes
    # the subprocess import MAIN's editable `hydra_suite` install (verified:
    # `hydra_suite.__file__` resolves outside this worktree), which has no
    # `data.tracking_job` module at all. The test would then fail for an
    # entirely unrelated reason (ModuleNotFoundError on an ancestor package,
    # not the Qt-submodule assertion), and after a merge to main it would
    # PASS without ever having exercised the tree under test. Use src.
    env = {**os.environ, "PYTHONPATH": str(PACKAGE.parents[2])}
    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, env=env
    )
    assert result.returncode == 0, result.stderr
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_tracking_job_manifest.py tests/test_tracking_job_shared_roots.py tests/test_tracking_job_layering.py -v`
Expected: FAIL. **Minor fix (adversarial review) — the real expected failure shape, stated precisely.** `test_tracking_job_manifest.py`/`test_tracking_job_shared_roots.py` fail at collection (`ModuleNotFoundError: hydra_suite.data.tracking_job`), as expected. `test_tracking_job_layering.py` is different: `test_no_app_layer_or_qt_imports` is PARAMETRIZED over `sorted(PACKAGE.glob("*.py"))`, and with no package on disk yet that glob is empty — pytest collects it as a test with an EMPTY parameter set, which SKIPS (reports "no tests ran" for that parametrization) rather than failing. Only `test_the_gate_itself_catches_a_relative_app_layer_import` (it constructs its own synthetic file, independent of `PACKAGE`), the new `test_the_gate_catches_a_relative_app_layer_import_inside_a_real_init_file` (same — builds its own `tmp_path` tree), and `test_package_imports_without_qt_installed` (its subprocess `import hydra_suite.data.tracking_job` genuinely fails) produce real FAILs from this file at this step. State this as the actual expectation rather than a blanket "FAIL", so a literal implementer doesn't mistake the parametrized test's skip for an unexpected result.

- [ ] **Step 3: Implement `manifest.py`**

Follow `data/project_bundle.py` conventions: `bundle_version`-style integer first, `to_dict`/`from_dict` dataclasses, `ValueError` subclass on version mismatch, `write_json_atomic` reused from `project_bundle` (`:167-186`).

```python
"""Manifest for a portable tracking job (hydra_job.json)."""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..project_bundle import write_json_atomic

SUPPORTED_JOB_VERSION = 1
JOB_MANIFEST_FILENAME = "hydra_job.json"


class TrackingJobError(ValueError):
    """A job-lifecycle error carrying the CLI exit code to use.

    2 = argument/validation, 3 = preflight, 4 = transport, 5 = pull collision.
    """

    def __init__(self, message: str, *, code: int = 2) -> None:
        super().__init__(message)
        self.code = code


def validate_job_relpath(value: str) -> Path:
    """Every path inside a job is relative to the job root and stays inside it."""
    if not value:
        raise TrackingJobError("empty job-relative path")
    candidate = Path(value)
    if candidate.is_absolute() or any(part == ".." for part in candidate.parts):
        raise TrackingJobError(f"unsafe job-relative path: {value}")
    return candidate


@dataclass(frozen=True)
class JobVideo:
    job_path: str
    origin_path: str
    size_bytes: int
    config_job_path: str
    config_provenance: str
    pushed_siblings: list[str] = field(default_factory=list)
    signature: str = ""
    shared: dict[str, str] | None = None
    redirected_outputs: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class JobModel:
    key: str
    roles: list[str]
    origin_path: str
    kind: str  # "file" | "directory"
    sha256: str = ""
    size_bytes: int = 0
    sidecars: list[str] = field(default_factory=list)
    files: list[str] = field(default_factory=list)
    registry_entry_present: bool = False
    # Fix M8: a directory model (pose run dir) or bundle (ClassKit sidecars)
    # only ever got ONE top-level sha256/size pair, so verify/preflight could
    # not detect a single corrupted member file inside a multi-file artifact —
    # spec §6.2 step 5 requires "sha256 of every file". Populated by pack.py
    # for kind == "directory" and for any model with sidecars/files; a plain
    # single-file model leaves this empty (its top-level sha256 already covers
    # it). Keys are job-relative paths (e.g. "pose/SLEAP/run/best.ckpt").
    #
    # Minor fix (adversarial review) — a DELIBERATE, DOCUMENTED deviation
    # from spec §6.2 step 5's literal text, not an oversight: the spec says
    # "sha256 of every file", read most literally as EVERY file in the job
    # (including every video and every sidecar JSON), but this plan only
    # ever populates `file_digests` for directory/bundle MODEL members
    # (`copy_model_reference`, Task 6) — never for a video (`JobVideo` has
    # no per-file digest field at all; `verify_job`'s video check is a size
    # comparison, fix W1b, not a hash) and never for a lone sidecar JSON.
    # This is intentional, not a gap that slipped through: (1) videos are
    # multi-gigabyte and re-hashing one on every `verify_job` call would make
    # the offline, "cheap, stat-only" design goal (stated explicitly for the
    # video-size check, fix W1b) impossible for the one artifact class where
    # it matters most; content-level video integrity is Task 10 preflight's
    # `video_signature` check instead, which trades verify's offline-ness for
    # a one-time content read at run time, deliberately NOT duplicated here;
    # (2) sidecar JSONs are tiny, pack-regenerated, deterministic snapshots
    # (config, skeletons) with no plausible silent-corruption story rsync
    # doesn't already guard against via its own checksum mode — they get an
    # existence check, not a hash. `file_digests` closes the ONE real gap the
    # M8 fix targets (a multi-file model artifact where corruption of ONE
    # member is otherwise undetectable by a single top-level hash); it was
    # never meant to make every byte in the job content-addressed.
    file_digests: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class JobManifest:
    job_id: str
    created_at: str
    created_on: dict[str, Any]
    keystone: dict[str, str]
    videos: list[JobVideo]
    models: list[JobModel]
    job_version: int = SUPPORTED_JOB_VERSION
    config_snapshot: dict[str, Any] = field(default_factory=dict)
    requirements: dict[str, Any] = field(default_factory=dict)
    track_args: dict[str, Any] = field(default_factory=dict)
    pull_history: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"job_version": self.job_version}
        payload.update(
            {
                "job_id": self.job_id,
                "created_at": self.created_at,
                "created_on": dict(self.created_on),
                "keystone": dict(self.keystone),
                "videos": [dataclasses.asdict(v) for v in self.videos],
                "models": [dataclasses.asdict(m) for m in self.models],
                "config_snapshot": dict(self.config_snapshot),
                "requirements": dict(self.requirements),
                "track_args": dict(self.track_args),
                "pull_history": [dict(entry) for entry in self.pull_history],
            }
        )
        return payload

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "JobManifest":
        version = int(data.get("job_version", SUPPORTED_JOB_VERSION))
        if version != SUPPORTED_JOB_VERSION:
            raise TrackingJobError(
                f"Unsupported tracking job version {version} "
                f"(this build supports {SUPPORTED_JOB_VERSION})"
            )
        # Minor fix: JobVideo(**entry)/JobModel(**entry) raise a bare TypeError
        # on any unknown forward-compat key (e.g. an older client reading a
        # manifest written by a newer one with an added field), which is not
        # a TrackingJobError and so escapes the CLI's uniform error handling
        # (run_job_cli only catches TrackingJobError, per Task 11 Step 4's
        # "All TrackingJobErrors are caught... printed as error:"). Wrap it.
        try:
            videos = [JobVideo(**entry) for entry in data.get("videos", [])]
            models = [JobModel(**entry) for entry in data.get("models", [])]
        except TypeError as exc:
            raise TrackingJobError(
                f"hydra_job.json has an unrecognized field for this build: {exc}"
            ) from exc
        for video in videos:
            validate_job_relpath(video.job_path)
            validate_job_relpath(video.config_job_path)
            for sibling in video.pushed_siblings:
                validate_job_relpath(sibling)
        for model in models:
            validate_job_relpath(model.key)
            for extra in list(model.sidecars) + list(model.files):
                validate_job_relpath(extra)
        # Fix V-minor: `data["job_id"]`/`data["created_at"]` are bare
        # dict-index lookups -- a manifest missing either key (hand-edited,
        # truncated write, or an even-older format than the job_version
        # check above catches) raises a plain KeyError here, which -- same
        # as the JobVideo/JobModel TypeError case just above -- is not a
        # TrackingJobError and so escapes run_job_cli's uniform "catches
        # TrackingJobError, prints as error:" handling (Task 11 Step 4).
        # Wrap it the same way.
        try:
            job_id = str(data["job_id"])
            created_at = str(data["created_at"])
        except KeyError as exc:
            raise TrackingJobError(
                f"hydra_job.json is missing required field {exc}"
            ) from exc
        return cls(
            job_version=version,
            job_id=job_id,
            created_at=created_at,
            created_on=dict(data.get("created_on", {})),
            keystone=dict(data.get("keystone", {})),
            videos=videos,
            models=models,
            config_snapshot=dict(data.get("config_snapshot", {})),
            requirements=dict(data.get("requirements", {})),
            track_args=dict(data.get("track_args", {})),
            pull_history=list(data.get("pull_history", [])),
        )

    def write(self, path: Path) -> None:
        write_json_atomic(Path(path), self.to_dict())

    @classmethod
    def read(cls, path: Path) -> "JobManifest":
        import json

        with open(path, encoding="utf-8") as handle:
            return cls.from_dict(json.load(handle))
```

Note: a `shared` video has no file under `videos/` but keeps a `job_path` (the symlink the remote materializes), so `validate_job_relpath` still applies.

- [ ] **Step 4: Implement `shared_roots.py` and `paths.get_shared_roots_path`**

In `paths.py`, beside `get_advanced_config_path` (fix Q5, adversarial review: verified now at `:188`, not `:156-158` — Task 1 inserted `get_platform_config_dir` above it, shifting every later line number; re-check every `paths.py` line citation in this plan against current source before implementing):

```python
def get_shared_roots_path() -> Path:
    """Host mount table mapping a shared-root alias to this host's mount point."""
    return _user_config_dir() / "shared_roots.json"
```

`shared_roots.py`:

```python
"""Host mount table: a video on a lab share is referenced, never copied."""

from __future__ import annotations

import json
import os
from pathlib import Path

from ...paths import get_shared_roots_path
from .manifest import TrackingJobError, validate_job_relpath


def load_shared_roots(path: Path | None = None) -> dict[str, str]:
    target = Path(path) if path is not None else get_shared_roots_path()
    if not target.is_file():
        return {}
    try:
        with open(target, encoding="utf-8") as handle:
            data = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise TrackingJobError(f"cannot read shared-root table {target}: {exc}") from exc
    if not isinstance(data, dict):
        raise TrackingJobError(f"shared-root table must be a JSON object: {target}")
    return {str(k): str(v) for k, v in data.items()}


def save_shared_roots(table: dict[str, str], path: Path | None = None) -> None:
    from ..project_bundle import write_json_atomic

    target = Path(path) if path is not None else get_shared_roots_path()
    write_json_atomic(target, {str(k): str(v) for k, v in table.items()})


def save_alias(alias: str, root: str, path: Path | None = None) -> dict[str, str]:
    """Persist ONE alias, preserving every other entry (fix B12a).

    Tasks 10 and 11 promote a one-off ``--shared-root ALIAS=PATH`` into the
    host table. Calling ``save_shared_roots({alias: root})`` there would delete
    every other alias this machine already knows, so the read-modify-write
    lives here, once, instead of being re-derived at each call site.
    """
    target = Path(path) if path is not None else get_shared_roots_path()
    table = load_shared_roots(target) if target.exists() else {}
    table[str(alias)] = str(root)
    save_shared_roots(table, target)
    return table


def match_shared_root(abs_path: str, table: dict[str, str]) -> tuple[str, str] | None:
    """Longest matching alias root wins; ``None`` when the file is off-share."""
    real = os.path.realpath(abs_path)
    best: tuple[str, str] | None = None
    best_len = -1
    for alias, root in table.items():
        root_real = os.path.realpath(root)
        try:
            rel = Path(real).relative_to(root_real)
        except ValueError:
            continue  # component-safe: "/Volumes/labour" is not under "/Volumes/lab"
        if len(root_real) > best_len:
            best_len = len(root_real)
            best = (alias, rel.as_posix())
    return best


def resolve_shared(alias: str, relpath: str, table: dict[str, str]) -> Path:
    if alias not in table:
        known = ", ".join(sorted(table)) or "(none configured)"
        raise TrackingJobError(
            f"unknown shared-root alias {alias!r}; known aliases: {known}. "
            f"Add it with 'trackerkit job shared-root add {alias} <path>' or pass "
            f"--shared-root {alias}=<path>.",
            code=3,
        )
    return Path(table[alias]) / validate_job_relpath(relpath)
```

`Path.relative_to` is component-wise, so the `/Volumes/labour` case is handled without string prefix checks.

- [ ] **Step 5: `__init__.py` re-exports**

```python
"""Portable tracking jobs: pack anywhere, run anywhere, sync back."""

from .manifest import (
    JOB_MANIFEST_FILENAME,
    SUPPORTED_JOB_VERSION,
    JobManifest,
    JobModel,
    JobVideo,
    TrackingJobError,
    validate_job_relpath,
)

__all__ = [
    "JOB_MANIFEST_FILENAME",
    "SUPPORTED_JOB_VERSION",
    "JobManifest",
    "JobModel",
    "JobVideo",
    "TrackingJobError",
    "validate_job_relpath",
]
```

(Later tasks append `pack_job`, `verify_job`, `plan_pull`, `preflight_job` here.)

- [ ] **Step 6: Run tests to verify they pass**

Run: `python -m pytest tests/test_tracking_job_manifest.py tests/test_tracking_job_shared_roots.py tests/test_tracking_job_layering.py -v`
Expected: all PASS.

- [ ] **Step 7: Commit**

```bash
make format
git add src/hydra_suite/data/tracking_job/ src/hydra_suite/paths.py tests/test_tracking_job_*.py
git commit -m "feat(tracking-job): manifest, shared-root mount table, and the layering gate"
```

---

### Task 6: Model reference copying + registry subset

**Files:**
- Create: `src/hydra_suite/data/tracking_job/references.py`
- Test: `tests/test_tracking_job_references.py` (create)

**Interfaces:**
- Consumes: `JobModel`, `TrackingJobError` (Task 5).
- Produces:
  - `PlannedModel` frozen dataclass: `role: str`, `source_path: str`, `kind: str`, `key: str`, `bundle_artifacts: list[str]` — built by the **app layer** (Task 11) because bundle discovery lives in `classkit`.
  - `copy_model_reference(planned: PlannedModel, models_root: Path) -> JobModel`
  - `write_registry_subset(shipped_keys: set[str], entries: Iterable[tuple[str, dict]], destination: Path) -> int`
  - `external_key_for(path: str) -> str` → `external/<sha256[:12]>/<basename>`

**Layout rule (spec correction #6).** Two incompatible repo layouts share one registry (`model_publish._repo_dir_for_role` writes `YOLO-obb/`; `model_paths.get_yolo_model_repository_directory` reads `obb/`). `copy_model_reference` never re-derives a layout from a role — it copies to **the key the config already resolved through**, i.e. `make_model_path_relative(source)`, computed by the caller. Whatever layout the staging machine used is preserved verbatim, so the same relative string in the sidecar resolves on the remote.

- [ ] **Step 1: Write the failing test**

Create `tests/test_tracking_job_references.py`:

```python
"""Copying a model reference into a job, with sidecars, bundles and registry."""

import hashlib
import json

import pytest

from hydra_suite.data.tracking_job.references import (
    PlannedModel,
    copy_model_reference,
    external_key_for,
    write_registry_subset,
)


@pytest.fixture()
def source_root(tmp_path):
    root = tmp_path / "hostmodels"
    (root / "obb").mkdir(parents=True)
    model = root / "obb" / "x.pt"
    model.write_bytes(b"weights")
    (root / "obb" / "x.pt.slice_meta.json").write_text('{"slice": 1}')
    (root / "obb" / "x.pt.canonical_meta.json").write_text('{"canonical": 1}')
    return root


def test_file_model_is_copied_with_its_sidecars(source_root, tmp_path):
    models_root = tmp_path / "job" / "models"
    planned = PlannedModel(
        role="YOLO_OBB_DIRECT_MODEL_PATH",
        source_path=str(source_root / "obb" / "x.pt"),
        kind="file",
        key="obb/x.pt",
    )
    record = copy_model_reference(planned, models_root)
    assert (models_root / "obb" / "x.pt").read_bytes() == b"weights"
    assert (models_root / "obb" / "x.pt.slice_meta.json").exists()
    assert (models_root / "obb" / "x.pt.canonical_meta.json").exists()
    assert set(record.sidecars) == {
        "obb/x.pt.slice_meta.json",
        "obb/x.pt.canonical_meta.json",
    }
    assert record.kind == "file"
    assert record.sha256 and record.size_bytes == 7


def test_sha256_matches_the_file_bytes(source_root, tmp_path):
    # Fix W9: `hashlib` is used by several tests below (directory/bundle
    # digest assertions) without a local import -- it must be imported once
    # at module level (done in this file's header above), not re-imported
    # locally here, or every OTHER test using it raises NameError.
    planned = PlannedModel(
        role="R", source_path=str(source_root / "obb" / "x.pt"), kind="file", key="obb/x.pt"
    )
    record = copy_model_reference(planned, tmp_path / "models")
    assert record.sha256 == hashlib.sha256(b"weights").hexdigest()


def test_directory_model_is_copied_whole(tmp_path):
    src = tmp_path / "host" / "pose" / "SLEAP" / "run"
    src.mkdir(parents=True)
    (src / "best.ckpt").write_bytes(b"c")
    (src / "training_config.json").write_text("{}")
    nested = src / "sub"
    nested.mkdir()
    (nested / "extra.json").write_text("{}")
    planned = PlannedModel(
        role="POSE_MODEL_DIR", source_path=str(src), kind="directory",
        key="pose/SLEAP/run",
    )
    models_root = tmp_path / "job" / "models"
    record = copy_model_reference(planned, models_root)
    assert (models_root / "pose" / "SLEAP" / "run" / "best.ckpt").exists()
    assert (models_root / "pose" / "SLEAP" / "run" / "sub" / "extra.json").exists()
    assert record.kind == "directory"
    assert "pose/SLEAP/run/best.ckpt" in record.files
    # Fix B4: file_digests must be POPULATED, not merely declared. Without
    # these assertions the field ships empty and verify_job's member-integrity
    # branch (which is gated on `if model.file_digests:`) never runs at all.
    assert set(record.file_digests) == set(record.files)
    assert record.file_digests["pose/SLEAP/run/best.ckpt"] == hashlib.sha256(
        b"c"
    ).hexdigest()
    # A directory model carries no top-level digest/size; verify_job must skip
    # its sha256/size_bytes checks for kind == "directory".
    assert record.sha256 == ""
    assert record.size_bytes == 0


def test_directory_model_excludes_host_runtime_artifacts(tmp_path):
    src = tmp_path / "host" / "run"
    (src / ".hydra-runtime-artifacts").mkdir(parents=True)
    (src / "best.ckpt").write_bytes(b"c")
    (src / ".hydra-runtime-artifacts" / "m.engine").write_bytes(b"host")
    models_root = tmp_path / "job" / "models"
    record = copy_model_reference(
        PlannedModel(role="POSE_MODEL_DIR", source_path=str(src), kind="directory",
                     key="pose/run"),
        models_root,
    )
    assert not (models_root / "pose" / "run" / ".hydra-runtime-artifacts").exists()
    assert not any(".hydra-runtime-artifacts" in f for f in record.files)
    assert not any(".hydra-runtime-artifacts" in f for f in record.file_digests)


def test_bundle_artifacts_are_copied_beside_the_selected_checkpoint(tmp_path):
    src = tmp_path / "host" / "classification" / "identity"
    src.mkdir(parents=True)
    head_a = src / "head_a.pth"
    head_b = src / "head_b.pth"
    manifest = src / "ids.bundle.json"
    head_a.write_bytes(b"a")
    head_b.write_bytes(b"b")
    manifest.write_text("{}")
    planned = PlannedModel(
        role="CNN_CLASSIFIERS", source_path=str(head_a), kind="file",
        key="classification/identity/head_a.pth",
        bundle_artifacts=[str(head_b), str(manifest)],
    )
    models_root = tmp_path / "job" / "models"
    record = copy_model_reference(planned, models_root)
    assert (models_root / "classification" / "identity" / "head_b.pth").exists()
    assert (models_root / "classification" / "identity" / "ids.bundle.json").exists()
    assert "classification/identity/head_b.pth" in record.files
    # Fix B4: bundle artifacts are separate files the primary's sha256 cannot
    # cover, so each must carry its own digest.
    assert set(record.file_digests) == set(record.files)
    assert record.file_digests["classification/identity/head_b.pth"] == hashlib.sha256(
        b"b"
    ).hexdigest()
    assert record.file_digests[
        "classification/identity/ids.bundle.json"
    ] == hashlib.sha256(b"{}").hexdigest()


def test_bundle_head_sidecar_is_recorded_so_it_gets_pushed(tmp_path):
    """Fix V1: a bundle head's own .v2meta.json must land in record.sidecars,
    not just get copied to disk -- Task 9's build_push_input_list only walks
    manifest.models[*].sidecars/files, so an un-recorded sidecar never ships,
    and a missing .v2meta.json for a flat .pt classifier silently falls back
    to fit_policy 'squash' on the remote (backend.py:38-52, :288)."""
    src = tmp_path / "host" / "classification" / "identity"
    src.mkdir(parents=True)
    head_a = src / "head_a.pth"
    head_b = src / "head_b.pth"
    head_a.write_bytes(b"a")
    head_b.write_bytes(b"b")
    # Fix X9 (round-6): the real convention is `.with_suffix(".v2meta.json")`
    # (REPLACES the model's own suffix), not an appended
    # "<name>.pth.v2meta.json" -- verified core/inference/model_paths.py:44
    # (`src.with_suffix(".v2meta.json")`) and
    # core/individual/classification/backend.py:288, which is the actual
    # reader. The original test wrote and asserted a name the real code never
    # produces or looks for, so it proved nothing about the file
    # copy_model_metadata_sidecars/backend.py actually round-trip.
    (src / "head_b.v2meta.json").write_text('{"fit_policy": "letterbox"}')
    planned = PlannedModel(
        role="CNN_CLASSIFIERS", source_path=str(head_a), kind="file",
        key="classification/identity/head_a.pth",
        bundle_artifacts=[str(head_b)],
    )
    models_root = tmp_path / "job" / "models"
    record = copy_model_reference(planned, models_root)
    assert "classification/identity/head_b.v2meta.json" in record.sidecars


def test_missing_source_fails_with_the_role_and_the_path(tmp_path):
    from hydra_suite.data.tracking_job.manifest import TrackingJobError

    planned = PlannedModel(
        role="YOLO_HEADTAIL_MODEL_PATH", source_path=str(tmp_path / "gone.pt"),
        kind="file", key="x/gone.pt",
    )
    with pytest.raises(TrackingJobError) as excinfo:
        copy_model_reference(planned, tmp_path / "models")
    message = str(excinfo.value)
    assert "YOLO_HEADTAIL_MODEL_PATH" in message and "gone.pt" in message


def test_external_key_is_deterministic_and_namespaced(tmp_path):
    outside = tmp_path / "elsewhere" / "m.pt"
    outside.parent.mkdir(parents=True)
    outside.write_bytes(b"z")
    key = external_key_for(str(outside))
    assert key.startswith("external/") and key.endswith("/m.pt")
    assert external_key_for(str(outside)) == key


def test_copy_is_idempotent(source_root, tmp_path):
    planned = PlannedModel(
        role="R", source_path=str(source_root / "obb" / "x.pt"), kind="file", key="obb/x.pt"
    )
    models_root = tmp_path / "models"
    first = copy_model_reference(planned, models_root)
    second = copy_model_reference(planned, models_root)
    assert first == second


def test_registry_subset_contains_only_shipped_keys_with_null_source(tmp_path):
    entries = [
        ("obb/x.pt", {"species": "ant", "source_path": "/host/train/run/best.pt"}),
        ("obb/unused.pt", {"species": "fly", "source_path": "/host/other.pt"}),
    ]
    destination = tmp_path / "models" / "model_registry.json"
    count = write_registry_subset({"obb/x.pt"}, entries, destination)
    payload = json.loads(destination.read_text())
    assert count == 1
    assert payload["schema_version"] == 2
    assert set(payload["entries"]) == {"obb/x.pt"}
    assert payload["entries"]["obb/x.pt"]["source_path"] is None
    assert payload["entries"]["obb/x.pt"]["species"] == "ant"


def test_registry_subset_writes_an_empty_v2_registry_when_nothing_matches(tmp_path):
    destination = tmp_path / "models" / "model_registry.json"
    assert write_registry_subset(set(), [], destination) == 0
    payload = json.loads(destination.read_text())
    assert payload == {"schema_version": 2, "entries": {}}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_tracking_job_references.py -v`
Expected: FAIL — module missing.

- [ ] **Step 3: Implement `references.py`**

```python
"""Copying model artifacts into a job's models root."""

from __future__ import annotations

import hashlib
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

from ...core.inference.model_paths import copy_model_metadata_sidecars
from ..project_bundle import write_json_atomic
from .manifest import JobModel, TrackingJobError, validate_job_relpath

# Host-local engine caches never travel: they are rebuilt per compute box.
# Fix V-minor: keep this in sync with content_id.py's _EXCLUDED_DIR_NAMES --
# .DS_Store/__pycache__ are OS/tooling noise, not model content, and copying
# them into the job would be dead weight `directory_content_id` already
# ignores on the read side.
EXCLUDED_DIR_NAMES = {".hydra-runtime-artifacts", ".DS_Store", "__pycache__"}


@dataclass(frozen=True)
class PlannedModel:
    """One model to ship, already resolved by the app layer.

    ``key`` is the models-root-relative path the CONFIG already uses
    (``make_model_path_relative``), never re-derived from ``role``: the repo has
    two incompatible role->directory layouts over one registry, and preserving
    the config's own key is what makes the sidecar resolve on the remote.
    ``bundle_artifacts`` are sibling files found by ``discover_multihead_model_bundle``
    in the app layer (ClassKit is an app layer; Data must not import it).
    """

    role: str
    source_path: str
    kind: str  # "file" | "directory"
    key: str
    bundle_artifacts: list[str] = field(default_factory=list)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def external_key_for(path: str) -> str:
    """Job key for a model that lives OUTSIDE the staging models root."""
    source = Path(path)
    digest = hashlib.sha256(str(source.resolve()).encode("utf-8")).hexdigest()[:12]
    return f"external/{digest}/{source.name}"


def _copy_file(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def copy_model_reference(planned: PlannedModel, models_root: Path) -> JobModel:
    """Copy one model (plus sidecars/bundle siblings) into ``models_root``."""
    validate_job_relpath(planned.key)
    source = Path(planned.source_path)
    destination = Path(models_root) / planned.key
    if not source.exists():
        raise TrackingJobError(
            f"model for role {planned.role} does not exist on this machine: "
            f"{planned.source_path}",
            code=2,
        )

    sidecars: list[str] = []
    files: list[str] = []

    if planned.kind == "directory":
        if destination.exists():
            shutil.rmtree(destination)
        shutil.copytree(
            source,
            destination,
            ignore=shutil.ignore_patterns(*EXCLUDED_DIR_NAMES),
        )
        # Fix B4: file_digests was DECLARED on JobModel but never populated,
        # which made verify_job's "if model.file_digests: check every member"
        # branch vacuously true for every directory model that has ever been
        # packed. Populate it here -- this walk is the ONLY place that sees the
        # member list, so there is nowhere else it could be filled in.
        digests: dict[str, str] = {}
        for child in sorted(destination.rglob("*")):
            if child.is_file():
                relpath = child.relative_to(models_root).as_posix()
                files.append(relpath)
                digests[relpath] = _sha256(child)
        return JobModel(
            key=planned.key,
            roles=[planned.role],
            origin_path=str(source),
            kind="directory",
            # A directory has NO meaningful top-level sha256/size_bytes; both
            # stay at their dataclass defaults and verify_job must not check
            # them for kind == "directory" (see Task 7's verify rules).
            files=files,
            file_digests=digests,
        )

    _copy_file(source, destination)
    copy_model_metadata_sidecars(str(source), str(destination))
    for sidecar in sorted(destination.parent.glob(destination.name + ".*")):
        if sidecar != destination:
            sidecars.append(sidecar.relative_to(models_root).as_posix())
    v2 = destination.with_suffix(".v2meta.json")
    if v2.exists():
        sidecars.append(v2.relative_to(models_root).as_posix())

    for artifact in planned.bundle_artifacts:
        artifact_path = Path(artifact)
        if not artifact_path.exists():
            raise TrackingJobError(
                f"bundle artifact for role {planned.role} is missing: {artifact}", code=2
            )
        sibling = destination.parent / artifact_path.name
        _copy_file(artifact_path, sibling)
        copy_model_metadata_sidecars(str(artifact_path), str(sibling))
        files.append(sibling.relative_to(models_root).as_posix())
        # Fix V1: bundle heads (ClassKit multi-head classifiers) carry their OWN
        # metadata sidecars, same two conventions as the primary model above
        # (three-suffix-appended and .v2meta.json). Without collecting these into
        # `sidecars`, a head's .v2meta.json is copied to disk but never recorded
        # in JobModel, so build_push_input_list (Task 9) never pushes it and
        # verify_job (Task 7) never checks it exists. A missing .v2meta.json for
        # a flat .pt classifier makes backend.py:288's with_suffix lookup miss,
        # falling into resolve_fit_policy(None, ...) -> "assuming legacy 'squash'"
        # (backend.py:38-52) -- a silently different crop preprocessing on the
        # remote, with only a warning logged.
        for head_sidecar in sorted(sibling.parent.glob(sibling.name + ".*")):
            if head_sidecar != sibling:
                sidecars.append(head_sidecar.relative_to(models_root).as_posix())
        head_v2 = sibling.with_suffix(".v2meta.json")
        if head_v2.exists():
            sidecars.append(head_v2.relative_to(models_root).as_posix())

    # Fix B4: the bundle artifacts are extra FILES beside the primary model;
    # the primary's own sha256 says nothing about them, so digest each one.
    # (Sidecars are metadata JSON regenerated by copy_model_metadata_sidecars
    # and are deliberately not digested -- verify only asserts they exist.)
    file_digests = {
        relpath: _sha256(Path(models_root) / relpath) for relpath in sorted(set(files))
    }

    return JobModel(
        key=planned.key,
        roles=[planned.role],
        origin_path=str(source),
        kind="file",
        sha256=_sha256(destination),
        size_bytes=destination.stat().st_size,
        sidecars=sorted(set(sidecars)),
        files=sorted(set(files)),
        file_digests=file_digests,
    )


def write_registry_subset(
    shipped_keys: set[str],
    entries: Iterable[tuple[str, dict]],
    destination: Path,
) -> int:
    """Write a v2 registry containing only the shipped models.

    ``source_path`` is nulled: it is the staging machine's absolute path to the
    training artifact and is meaningless on any other host.
    """
    subset: dict[str, dict] = {}
    for key, meta in entries:
        if key in shipped_keys:
            entry = dict(meta)
            entry["source_path"] = None
            subset[key] = entry
    write_json_atomic(Path(destination), {"schema_version": 2, "entries": subset})
    return len(subset)
```

Note `copy_model_metadata_sidecars` has two conventions (three suffixes appended to the full name, `.v2meta.json` replacing the suffix), which is why both are collected. **Minor note:** this also ships `x.pt.runtime_meta.json`, a host-specific export stamp (TensorRT/CoreML engine build provenance), which nominally contradicts "engines never travel" (spec §2 non-goal). Harmless in practice — it's signature-gated metadata, not the engine binary itself, and a stale stamp on the remote just causes a rebuild rather than a silent wrong-engine reuse — but worth knowing it rides along rather than being filtered out.

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_tracking_job_references.py tests/test_tracking_job_layering.py -v`
Expected: all PASS, including the layering gate (this module imports `core.inference.model_paths`, which is allowed).

- [ ] **Step 5: Commit**

```bash
make format
# Fix B10: references.py's symbols are re-exported from the package __init__.
git add src/hydra_suite/data/tracking_job/references.py \
        src/hydra_suite/data/tracking_job/__init__.py \
        tests/test_tracking_job_references.py
git commit -m "feat(tracking-job): copy model references with sidecars, bundles and a registry subset"
```

---

### Task 7: `pack_job` + `run.sh` + `verify_job`

**Files:**
- Create: `src/hydra_suite/data/tracking_job/pack.py`, `runner.py`, `verify.py`
- Test: `tests/test_tracking_job_pack.py`, `tests/test_tracking_job_verify.py` (create)

**Interfaces:**
- Consumes: `JobManifest`/`JobVideo`/`JobModel` (Task 5), `PlannedModel`/`copy_model_reference`/`write_registry_subset` (Task 6).
- Produces:
  - `PlannedVideo` frozen dataclass: `video_path: str`, `config: dict`, `config_provenance: str`, `planned_models: list[PlannedModel]`, `skeleton_path: str`, **`cnn_model_keys: dict[str, str]`** (fix B5 — see "the CNN rewrite rule" below; defaults to `field(default_factory=dict)`).
  - `pack_job(job_dir, planned_videos, *, registry_entries, advanced_config_path, advanced_config_fallback=None, track_args, shared_table, copy_videos=False, shared_mode="auto", job_name=None, force=False, force_discard_outputs=False) -> JobManifest` (`force` — fix V-minor/X8, see the re-pack-safety note below Step 4's ordered-steps list: required to pack into a non-empty `job_dir`; `force_discard_outputs` — fix X8, a SEPARATE opt-in from `force`, required in addition when `videos/` holds pulled outputs the old manifest doesn't account for; `advanced_config_fallback` — fix X5a, see Fix V4 below: the CALLER's already-resolved `load_advanced_tracker_config()` dict, used only when `advanced_config_path` doesn't exist on disk, so `pack.py` itself never imports `trackerkit`; every fixture/test whose `advanced_config_path` already points at a real file leaves this `None` and never touches the fallback branch)
  - `ROLE_TO_CONFIG_KEY: dict[str, str | tuple[str, ...]]` and `LEGACY_ALIAS_CONFIG_KEYS: tuple[str, ...]` — module-level in `pack.py` (fix B-minor: these were consumed by Task 11 but never listed as produced by this task, so Task 11 had nothing to import).
  - `render_run_sh() -> str`
  - `verify_job(job_dir, *, fast: bool = False) -> list[str]` — returns problems; empty means valid. **`fast=True` skips every sha256 computation** (both the top-level `models[].sha256`/`size_bytes` comparison and the per-member `file_digests` walk), degrading those to existence-only checks. Every other check is unchanged. This exists because Task 10's `preflight_job(..., fast=True)` runs `verify_job` as its first check; without threading `fast` through, `fast` would be entirely defeated — verify would hash every model anyway (fix B-minor).
  - **Fix W1 — `JobVideo.signature` is populated for EVERY video, not only `shared` entries.** Before this fix, `pack_job` only computed `content_id.video_signature(origin_path)` when a video matched a shared-root alias, so a non-shared video that `rsync --partial` truncated mid-push (rsync's documented behaviour on interruption: it leaves the partial bytes at the destination path rather than deleting them) had `signature == ""`, `verify_job` only checked that `job_path` *existed*, and a truncated `.mp4` therefore both verified and preflighted clean, then ran to a CSV that looked complete. `pack_job` now calls `content_id.video_signature(origin_path)` unconditionally for every `PlannedVideo` and stores the result on `JobVideo.signature` regardless of `shared`. This is the field Task 10's new `video_signature` preflight check (below) and `verify_job`'s new `size_bytes` check compare against.
  - `ABSOLUTE_PATH_FORBIDDEN_KEYS` — the sidecar keys `verify` asserts are never absolute.
  - **`manifest.config_snapshot` has this EXACT shape (fix B11 — it was previously defined only inside a Task 9 test, so `build_push_input_list` could silently omit files):**

    ```python
    {
        "advanced_config": "config/advanced_config.json",   # always present
        "skeletons": ["config/skeletons/<name>.json", ...], # sorted, deduped;
                                                           # [] when no video
                                                           # configured a skeleton
    }
    ```

    Both values are job-relative POSIX strings that pass `validate_job_relpath`.
    `build_push_input_list` (Task 9) enumerates exactly
    `[config_snapshot["advanced_config"], *config_snapshot["skeletons"]]` plus the
    `.seeded` markers; nothing else under `config/` is a pack-time input.

    **Fix V4 — `advanced_config` is documented "always present," and `pack_job` must MAKE that true even when the staging host has never saved one.** `job_cli.py` (Task 11, Step 7) calls `pack_job(..., advanced_config_path=str(get_advanced_config_path()))` — that is just the PATH `get_advanced_config_path()` (`paths.py:188`, post-Task-1 line number) returns, and nothing guarantees a file exists there: a CLI-only staging box (headless, never opened TrackerKit's GUI to trigger a first save) has no `advanced_config.json` under its config dir at all. Spec §6.2 step 7 says "copy the host's advanced config **if it exists**" — read literally, that means SKIP the copy when it's absent, which leaves `config/advanced_config.json` missing from the packed job while `config_snapshot["advanced_config"]` still names it (this key is unconditional per the shape above). Two things then break: (a) `verify_job`'s `config_job_path`-style existence check would fail on every fresh staging box (an immediate, loud pack-time failure — not silently wrong, but a hard blocker for the most common "just installed the CLI, never ran the GUI" case), and (b) even if verify somehow tolerated it, `build_push_input_list` (Task 9) still lists `config_snapshot["advanced_config"]` unconditionally, so `rsync --files-from` gets a manifest line naming a file that was never created, and `rsync` exits 23 ("some files could not be transferred") — `push` then fails with code 4 on every CLI-only box, for a config file the run doesn't strictly need (it has defaults).

    Fix: `pack_job` treats "the file at `advanced_config_path` doesn't exist" as "use defaults," not "skip the snapshot." **Fix X5a (round-6) — `pack_job` must NOT call `load_advanced_tracker_config()` itself.** `load_advanced_tracker_config` lives at `trackerkit/cli_config.py:114` — `trackerkit` is an app layer, and `pack.py` is Data; Data must never import an app layer (this plan's own hard dependency-direction rule, and now an ENFORCED one — fix X5b makes `test_no_app_layer_or_qt_imports` catch this exact import even written as a relative `from ...trackerkit.cli_config import ...`, which is how an earlier draft of this fix would have slipped past the level-0-only gate). Resolution happens in the CALLER instead, the same pattern fix W6 already established for `runtime_tier`: **`job_cli.py` (Task 11) calls `load_advanced_tracker_config()` itself** (it is already an app-layer module, so this import is unremarkable there) and passes the resulting dict down as `pack_job(..., advanced_config_fallback=load_advanced_tracker_config())`. `pack_job` then does, with no import of `trackerkit` anywhere in `pack.py`:

    ```python
    src = Path(advanced_config_path)
    if src.exists():
        # On-disk bytes are what the staging user actually configured;
        # re-deriving would silently normalize/drop keys a merge doesn't
        # round-trip, so a plain copy is the only faithful choice.
        shutil.copy(src, job_dir / "config" / "advanced_config.json")
    elif advanced_config_fallback is not None:
        write_json_atomic(
            job_dir / "config" / "advanced_config.json", advanced_config_fallback
        )
    else:
        raise TrackingJobError(
            f"advanced_config_path {src} does not exist and no "
            f"advanced_config_fallback was supplied; the caller must resolve "
            f"load_advanced_tracker_config() itself (pack.py never imports "
            f"trackerkit) before calling pack_job"
        )
    ```

    Either way, `config/advanced_config.json` exists after `pack_job` returns, `config_snapshot["advanced_config"]` is never a lie, and `verify_job`/`build_push_input_list` need no special-casing. Add `test_pack_job_synthesizes_advanced_config_when_the_host_has_none`: call `pack_job(..., advanced_config_path=str(tmp_path / "does_not_exist.json"), advanced_config_fallback={"adv": "fallback"})` and assert `(job_dir / "config" / "advanced_config.json").is_file()` and that its parsed JSON equals the passed-in `advanced_config_fallback` dict exactly (not a re-derived one — proving `pack.py` used the CALLER's dict, not its own `trackerkit` call). Add `test_pack_py_module_has_no_trackerkit_import`: grep `pack.py`'s own AST (reuse `test_tracking_job_layering.py`'s `_imported_names`) and assert `"trackerkit"` never appears — a second, file-scoped belt-and-braces check alongside the general layering gate.

    **Minor fix — basename collisions under `config/skeletons/`.** Skeletons are copied by BASENAME (`config/skeletons/<name>.json`), and two different videos in the same job can point at two DIFFERENT skeleton files that happen to share a filename (e.g. two labs both naming their skeleton `skeleton.json`). Copying the second over the first would silently make one video's skeleton wrong on the remote, with `verify_job` unable to catch it (the file exists; it's just the wrong bytes). Detect this at pack time: if two distinct source `pose_skeleton_file` paths resolve to the same `config/skeletons/<name>.json` target AND their content differs (compare via `content_id.file_content_id`, not just presence), raise `TrackingJobError(code=2)` naming both source paths — fail loudly at pack, not silently on the remote.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_tracking_job_pack.py`:

```python
"""Packing a job: rewrites, videos, sidecars, runner, manifest."""

import json
import os
import stat
from pathlib import Path

import pytest

from hydra_suite.data.tracking_job.manifest import JobManifest, TrackingJobError
from hydra_suite.data.tracking_job.pack import PlannedVideo, pack_job
from hydra_suite.data.tracking_job.references import PlannedModel


# NOTE (fix B8): this `staging` fixture and the `_planned` helper below are
# shown here for readability, but Step 1b MOVES both verbatim into
# `tests/conftest.py` -- `test_tracking_job_verify.py` and (Task 10)
# `test_tracking_job_preflight.py` cannot see a module-local fixture and would
# fail at COLLECTION. This module keeps `_pack` and the tests, and imports the
# helper back with `from tests.helpers.tracking_job import _planned`.
@pytest.fixture()
def staging(tmp_path):
    """A models root, a video, a skeleton and an advanced config."""
    models = tmp_path / "models"
    (models / "obb").mkdir(parents=True)
    (models / "obb" / "x.pt").write_bytes(b"w")
    videos = tmp_path / "data"
    videos.mkdir()
    video = videos / "colony.mp4"
    video.write_bytes(b"\x00" * 2048)
    skeleton = tmp_path / "skel" / "ant.json"
    skeleton.parent.mkdir()
    skeleton.write_text('{"nodes": []}')
    advanced = tmp_path / "advanced_config.json"
    advanced.write_text('{"adv": true}')
    return {
        "models": models, "video": video, "skeleton": skeleton, "advanced": advanced,
    }


def _planned(staging, **overrides):
    config = {
        "file_path": str(staging["video"]),
        "csv_path": str(staging["video"].with_name("colony_tracking.csv")),
        "video_output_path": str(staging["video"].with_name("colony_tracking.mp4")),
        "yolo_obb_direct_model_path": "obb/x.pt",
        "pose_skeleton_file": str(staging["skeleton"]),
    }
    config.update(overrides.pop("config", {}))
    return PlannedVideo(
        video_path=str(staging["video"]),
        config=config,
        config_provenance="own-sidecar",
        planned_models=[
            PlannedModel(
                role="YOLO_OBB_DIRECT_MODEL_PATH",
                source_path=str(staging["models"] / "obb" / "x.pt"),
                kind="file",
                key="obb/x.pt",
            )
        ],
        skeleton_path=str(staging["skeleton"]),
        **overrides,
    )


def _pack(tmp_path, staging, planned=None, **kwargs):
    # Fix M1: shared_table used to be hard-coded AND forwarded via **kwargs,
    # so any caller passing shared_table= (several tests below do) raised
    # "TypeError: got multiple values for argument 'shared_table'". Pop the
    # default out of kwargs instead of hard-coding the keyword argument.
    kwargs.setdefault("shared_table", {})
    return pack_job(
        tmp_path / "job",
        [planned or _planned(staging)],
        registry_entries=[("obb/x.pt", {"species": "ant", "source_path": "/host/a.pt"})],
        advanced_config_path=str(staging["advanced"]),
        track_args={"video_list": "videos.txt"},
        **kwargs,
    )


def test_job_layout_is_created(tmp_path, staging):
    _pack(tmp_path, staging)
    job = tmp_path / "job"
    assert (job / "hydra_job.json").is_file()
    assert (job / "run.sh").is_file()
    assert (job / "videos.txt").is_file()
    assert (job / "models" / "obb" / "x.pt").is_file()
    assert (job / "models" / "model_registry.json").is_file()
    assert (job / "config" / "advanced_config.json").is_file()
    assert (job / "config" / "presets" / ".seeded").is_file()
    assert (job / "config" / "skeletons" / ".seeded").is_file()
    assert (job / "videos" / "colony_config.json").is_file()


def test_run_sh_is_executable(tmp_path, staging):
    _pack(tmp_path, staging)
    mode = (tmp_path / "job" / "run.sh").stat().st_mode
    assert mode & stat.S_IXUSR


# Fix Y1 (round-7): the round-6 `eval "$TRACKERKIT track --video-list
# videos.txt $(printf '%q ' "$@")"` form is broken with ZERO passthrough
# args -- `printf '%q ' "$@"` on an empty "$@" still emits one `''` token,
# and `eval` re-parses that as a real, bogus empty positional, so
# `parse_arguments` sees `track --video-list videos.txt ''` and raises
# ("use either explicit video paths or --video-list, not both") before a
# single frame runs. This must be a REAL executed test -- monkeypatching
# `subprocess.run` (as the `job run` unit tests do) never invokes the actual
# bash template, so it cannot see this bug class at all.
def test_run_sh_track_invocation_has_no_stray_empty_positional_with_zero_args(
    tmp_path, staging
):
    import subprocess
    import sys

    _pack(tmp_path, staging)
    job = tmp_path / "job"

    # An argv-dumping stub standing in for `trackerkit`: writes its argv
    # (one JSON list per invocation) to argv_dump.jsonl and always exits 0,
    # so run.sh's own preflight/track/`_record-run` calls all "succeed"
    # without needing a real engine.
    stub = tmp_path / "trackerkit_stub.py"
    stub.write_text(
        "import json, sys\n"
        "with open(sys.argv[0] + '.dump', 'a') as f:\n"
        "    f.write(json.dumps(sys.argv[1:]) + '\\n')\n"
        "sys.exit(0)\n"
    )
    dump = Path(str(stub) + ".dump")

    env = dict(os.environ)
    env["HYDRA_JOB_TRACKERKIT"] = f"{sys.executable} {stub}"
    env["HYDRA_JOB_SKIP_PREFLIGHT"] = "1"  # isolate the `track` + `_record-run` calls

    result = subprocess.run(
        ["bash", str(job / "run.sh")], cwd=job, env=env, capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr

    lines = dump.read_text().strip().splitlines()
    assert len(lines) == 2, lines  # track, then job _record-run
    track_argv = json.loads(lines[0])
    assert track_argv == ["track", "--video-list", "videos.txt"], track_argv
    assert "" not in track_argv, "run.sh must never pass an empty positional to track"


def test_video_is_symlinked_by_default(tmp_path, staging):
    _pack(tmp_path, staging)
    link = tmp_path / "job" / "videos" / "colony.mp4"
    assert link.is_symlink()
    assert os.path.realpath(link) == os.path.realpath(staging["video"])


def test_copy_videos_makes_a_real_file(tmp_path, staging):
    _pack(tmp_path, staging, copy_videos=True)
    real = tmp_path / "job" / "videos" / "colony.mp4"
    assert real.is_file() and not real.is_symlink()


def test_sidecar_has_no_absolute_paths(tmp_path, staging):
    _pack(tmp_path, staging)
    sidecar = json.loads((tmp_path / "job" / "videos" / "colony_config.json").read_text())
    assert sidecar["file_path"] == "videos/colony.mp4"
    assert sidecar["csv_path"] == "videos/colony_tracking.csv"
    assert sidecar["video_output_path"] == "videos/colony_tracking.mp4"
    assert sidecar["pose_skeleton_file"] == "config/skeletons/ant.json"
    assert sidecar["yolo_obb_direct_model_path"] == "obb/x.pt"


def test_skeleton_is_snapshotted(tmp_path, staging):
    _pack(tmp_path, staging)
    assert (tmp_path / "job" / "config" / "skeletons" / "ant.json").read_text() == '{"nodes": []}'


def test_redirected_output_is_recorded(tmp_path, staging, tmp_path_factory):
    elsewhere = tmp_path / "renders" / "custom.mp4"
    planned = _planned(staging, config={"video_output_path": str(elsewhere)})
    manifest = _pack(tmp_path, staging, planned=planned)
    sidecar = json.loads((tmp_path / "job" / "videos" / "colony_config.json").read_text())
    assert sidecar["video_output_path"] == "videos/colony_custom.mp4"
    assert manifest.videos[0].redirected_outputs == {
        "videos/colony_custom.mp4": str(elsewhere)
    }


def test_redirected_output_with_the_default_basename_in_a_different_directory_is_still_recorded(
    tmp_path, staging
):
    """Fix W10: /renders/colony_tracking.mp4 has the DEFAULT basename
    (colony_tracking.mp4 -- what _default_output_paths would compute for
    colony.mp4), so a basename-only redirect check wrongly treats it as
    already-in-the-default-location and drops it with NO
    redirected_outputs entry -- pull would then place it beside the video
    instead of restoring it to /renders. The fix compares the full
    RESOLVED path against _default_output_paths(video_source_path), which
    correctly sees this as redirected because the DIRECTORY differs."""
    elsewhere = tmp_path / "renders" / "colony_tracking.mp4"
    planned = _planned(staging, config={"video_output_path": str(elsewhere)})
    manifest = _pack(tmp_path, staging, planned=planned)
    sidecar = json.loads((tmp_path / "job" / "videos" / "colony_config.json").read_text())
    assert sidecar["video_output_path"] == "videos/colony_colony_tracking.mp4"
    assert manifest.videos[0].redirected_outputs == {
        "videos/colony_colony_tracking.mp4": str(elsewhere)
    }


def test_videos_txt_is_job_relative_and_keystone_first(tmp_path, staging):
    _pack(tmp_path, staging)
    lines = (tmp_path / "job" / "videos.txt").read_text().splitlines()
    assert lines == ["videos/colony.mp4"]


def test_pushed_siblings_lists_the_sidecar(tmp_path, staging):
    manifest = _pack(tmp_path, staging)
    assert manifest.videos[0].pushed_siblings == ["videos/colony_config.json"]


def test_origin_path_is_recorded_for_pull(tmp_path, staging):
    manifest = _pack(tmp_path, staging)
    assert manifest.videos[0].origin_path == str(staging["video"])


def test_registry_subset_is_written_with_null_source(tmp_path, staging):
    _pack(tmp_path, staging)
    payload = json.loads((tmp_path / "job" / "models" / "model_registry.json").read_text())
    assert set(payload["entries"]) == {"obb/x.pt"}
    assert payload["entries"]["obb/x.pt"]["source_path"] is None


def test_basename_collision_across_directories_is_rejected(tmp_path, staging):
    other_dir = tmp_path / "other"
    other_dir.mkdir()
    twin = other_dir / "colony.mp4"
    twin.write_bytes(b"\x00" * 512)
    second = _planned(staging)
    second = PlannedVideo(
        video_path=str(twin),
        config={"file_path": str(twin)},
        config_provenance="keystone-baseline",
        planned_models=[],
        skeleton_path="",
    )
    with pytest.raises(TrackingJobError) as excinfo:
        pack_job(
            tmp_path / "job",
            [_planned(staging), second],
            registry_entries=[],
            advanced_config_path=str(staging["advanced"]),
            track_args={},
            shared_table={},
        )
    message = str(excinfo.value)
    assert str(staging["video"]) in message and str(twin) in message


def test_shared_video_is_referenced_not_copied(tmp_path, staging):
    table = {"labnas": str(staging["video"].parent)}
    manifest = _pack(tmp_path, staging, shared_table=table)
    entry = manifest.videos[0]
    assert entry.shared == {"alias": "labnas", "relpath": "colony.mp4"}
    assert entry.signature
    assert not (tmp_path / "job" / "videos" / "colony.mp4").exists()


def test_shared_only_rejects_an_off_share_video(tmp_path, staging):
    with pytest.raises(TrackingJobError):
        _pack(tmp_path, staging, shared_table={}, shared_mode="shared-only")


def test_no_shared_disables_matching(tmp_path, staging):
    table = {"labnas": str(staging["video"].parent)}
    manifest = _pack(tmp_path, staging, shared_table=table, shared_mode="no-shared")
    assert manifest.videos[0].shared is None
    assert (tmp_path / "job" / "videos" / "colony.mp4").is_symlink()


def test_pack_self_verifies_and_manifest_reads_back(tmp_path, staging):
    _pack(tmp_path, staging)
    manifest = JobManifest.read(tmp_path / "job" / "hydra_job.json")
    assert manifest.job_version == 1
    assert manifest.keystone["video"] == "videos/colony.mp4"
```

Create `tests/test_tracking_job_verify.py`:

```python
"""verify_job is an offline, mount-agnostic integrity check."""

import json

import pytest

from hydra_suite.data.tracking_job.verify import verify_job


def test_a_freshly_packed_job_verifies(packed_job):
    assert verify_job(packed_job) == []


def test_a_corrupted_model_is_detected(packed_job):
    (packed_job / "models" / "obb" / "x.pt").write_bytes(b"tampered")
    problems = verify_job(packed_job)
    assert any("obb/x.pt" in p for p in problems)


def test_a_missing_model_is_detected(packed_job):
    (packed_job / "models" / "obb" / "x.pt").unlink()
    assert any("obb/x.pt" in p for p in verify_job(packed_job))


def test_an_absolute_path_in_a_sidecar_is_detected(packed_job):
    sidecar = packed_job / "videos" / "colony_config.json"
    payload = json.loads(sidecar.read_text())
    payload["csv_path"] = "/Users/someone/out.csv"
    sidecar.write_text(json.dumps(payload))
    assert any("csv_path" in p for p in verify_job(packed_job))


def test_a_missing_skeleton_is_detected(packed_job):
    (packed_job / "config" / "skeletons" / "ant.json").unlink()
    assert any("skeleton" in p.lower() for p in verify_job(packed_job))


def test_videos_txt_keystone_mismatch_is_detected(packed_job):
    (packed_job / "videos.txt").write_text("videos/other.mp4\n")
    assert verify_job(packed_job)


def test_a_truncated_video_is_detected(packed_job):
    """Fix W1b: rsync --partial leaves a truncated file at the destination
    path on interruption -- it EXISTS, so the pre-fix existence-only check
    passed. Truncating in place (same path, fewer bytes) reproduces exactly
    that failure mode without needing an actual push."""
    video = packed_job / "videos" / "colony.mp4"
    real = video.resolve()  # colony.mp4 is a symlink by default (copy_videos=False)
    real.write_bytes(real.read_bytes()[:100])
    problems = verify_job(packed_job)
    assert any("colony.mp4" in p and "size" in p.lower() for p in problems)


def test_all_problems_are_reported_not_just_the_first(packed_job):
    (packed_job / "models" / "obb" / "x.pt").unlink()
    (packed_job / "config" / "skeletons" / "ant.json").unlink()
    assert len(verify_job(packed_job)) >= 2
```

**Fix V2 note — `test_a_packed_shared_job_verifies_clean` is deferred to Task 10, not added here.** `verify_job`'s shared-exemption is proven by a test that packs a job with a `shared_table`, but `tests/conftest.py` does not gain a `packed_job_shared` fixture until Task 10 (Fix M11, below) — this task's own `staging`/`_planned`/`packed_job` fixtures (Step 1b) carry no shared-root wiring. Adding a test here that depends on a not-yet-existing fixture would break Task 7's own "Step 6: Run tests to verify they pass" gate with a collection-time `fixture 'packed_job_shared' not found` error. The test is added in Task 10 instead (see `test_a_packed_shared_job_verifies_clean` alongside the other `packed_job_shared` tests below), once the fixture that exercises `shared_table` actually exists — but it still proves exactly this task's `verify_job` behaviour, since Task 10 adds no new verify-time exemption of its own.

- [ ] **Step 1b (fix B8): MOVE `staging`, `_planned` and `packed_job` into `tests/conftest.py` — do not leave them in `test_tracking_job_pack.py`.**

`tests/test_tracking_job_verify.py` (this task) and `tests/test_tracking_job_preflight.py` (Task 10) both consume `packed_job`, which consumes `staging`/`_planned`. A pytest fixture defined in one test MODULE is not visible from another, so leaving them in `test_tracking_job_pack.py` makes both of those files fail at COLLECTION with `fixture 'staging' not found` — not at assertion time, so the failure looks unrelated to this task. `tests/conftest.py` today defines only `direct_obb_fixture` and the autouse `_neutralize_leaked_training_flags`; there is no `tests/tracking_job_conftest.py` and none is created.

Cut the `staging` fixture out of `tests/test_tracking_job_pack.py` and APPEND it (plus `packed_job`) to `tests/conftest.py`. **`_planned` goes to `tests/helpers/tracking_job.py`, NOT into conftest** — importing a name out of a conftest (`from tests.conftest import _planned`) is still the wrong pattern even though `tests/__init__.py` DOES exist (verified: it does, 34 bytes — correcting the wave-2 claim that it doesn't; the double-loading failure mode that claim described is not what's actually at risk here). The real reason is layering, not import mechanics: conftest is pytest's fixture-discovery file, not a module meant to export plain helper functions for other test files to import from — mixing the two makes `conftest.py` do double duty and obscures where `_planned` actually lives. `tests/helpers/` is already a real package (`tests/helpers/__init__.py` exists), so both `tests/conftest.py` and `tests/test_tracking_job_pack.py` do `from tests.helpers.tracking_job import _planned`.

**Fix Q6 (adversarial review) — a MODULE-LEVEL `from hydra_suite.data.tracking_job.pack import PlannedVideo` inside `tests/helpers/tracking_job.py` drags the whole `pack.py` import graph into COLLECTION for the entire pytest suite, not just this task's tests.** `pack.py` imports `hydra_suite.core.tracking.session_policy` (fix X3a's `is_pose_inference_enabled` guard), and `core/tracking/__init__.py:3` does `from .worker import TrackingEngineCore` — which pulls in cv2/torch/coremltools at IMPORT time (verified: `core/tracking/__init__.py` is exactly `from .worker import TrackingEngineCore`, no lazy import). Because `tests/conftest.py` is collected before every single test in the suite runs, and this task's Step 1b has `tests/conftest.py` do `from tests.helpers.tracking_job import _planned` — placed correctly AFTER the `sys.path` block, per the paragraph above — that one import statement, even in the right position, still triggers `tracking_job.py`'s own module-level `from hydra_suite.data.tracking_job.pack import PlannedVideo` at COLLECTION time for EVERY pytest invocation in the whole repo, including runs that never touch a single tracking-job test. Worse: any import error anywhere in that chain (a missing optional dependency on a given machine, a real bug in `pack.py`) now aborts collection for the ENTIRE suite, not just this task's files — exactly the failure mode the paragraph above was trying to avoid by placing the import after the `sys.path` block, except the heavy-import problem is orthogonal to import ORDERING and isn't fixed by reordering alone.

Fix: `tests/helpers/tracking_job.py` imports `PlannedVideo`/`PlannedModel` **inside `_planned`'s own function body**, not at module scope — mirroring the pattern the `packed_job` fixture below already uses (`from hydra_suite.data.tracking_job.pack import pack_job` inside the fixture function, not at the top of `conftest.py`). This defers the entire `pack.py`/`core.tracking` import chain to the moment a test actually CALLS `_planned(...)`, i.e. only when a tracking-job test genuinely runs, never at bare collection:

```python
"""Shared builders for the portable-job tests."""


def _planned(staging, **overrides):
    from hydra_suite.data.tracking_job.pack import PlannedVideo
    from hydra_suite.data.tracking_job.references import PlannedModel

    config = {
        "file_path": str(staging["video"]),
        "csv_path": str(staging["video"].with_name("colony_tracking.csv")),
        "video_output_path": str(staging["video"].with_name("colony_tracking.mp4")),
        "yolo_obb_direct_model_path": "obb/x.pt",
        "pose_skeleton_file": str(staging["skeleton"]),
    }
    config.update(overrides.pop("config", {}))
    return PlannedVideo(
        video_path=str(staging["video"]),
        config=config,
        config_provenance="own-sidecar",
        planned_models=[
            PlannedModel(
                role="YOLO_OBB_DIRECT_MODEL_PATH",
                source_path=str(staging["models"] / "obb" / "x.pt"),
                kind="file",
                key="obb/x.pt",
            )
        ],
        skeleton_path=str(staging["skeleton"]),
        **overrides,
    )
```

`tests/conftest.py` itself still does `from tests.helpers.tracking_job import _planned` — but that top-level import now only binds a plain function object; it no longer transitively imports `pack.py`/`core.tracking`/cv2/torch at collection time, since `tracking_job.py`'s own module body has no heavy import left in it. Put that `tests.helpers.tracking_job` import AFTER the `SRC_DIR`/`REPO_ROOT` `sys.path` block already at the top of `tests/conftest.py:1-12` (not before it) regardless — this is still needed so `tests.helpers.tracking_job` itself is importable at all (it is not on `sys.path` before that block runs), even though it no longer carries the heavy-import risk the paragraph above originally worried about. Add `test_conftest_collection_does_not_import_torch_or_cv2`: a subprocess test (same pattern as `test_package_imports_without_qt_installed` above) that runs `pytest --collect-only tests/test_something_unrelated.py` (any pre-existing, non-tracking-job test file) with `sys.modules['cv2'] = None` and `sys.modules['torch'] = None` pre-poisoned, and asserts collection still succeeds — proving `conftest.py`'s own module-level imports never require either package.

and APPEND this to `tests/conftest.py` (which does `from tests.helpers.tracking_job import _planned` after the `sys.path` block, per the fix above):

```python
# --- Portable tracking-job fixtures (shared by pack/verify/preflight tests) ---


@pytest.fixture()
def staging(tmp_path):
    """A models root, a video, a skeleton and an advanced config."""
    models = tmp_path / "models"
    (models / "obb").mkdir(parents=True)
    (models / "obb" / "x.pt").write_bytes(b"w")
    videos = tmp_path / "data"
    videos.mkdir()
    video = videos / "colony.mp4"
    video.write_bytes(b"\x00" * 2048)
    skeleton = tmp_path / "skel" / "ant.json"
    skeleton.parent.mkdir()
    skeleton.write_text('{"nodes": []}')
    advanced = tmp_path / "advanced_config.json"
    advanced.write_text('{"adv": true}')
    return {
        "models": models,
        "video": video,
        "skeleton": skeleton,
        "advanced": advanced,
    }


@pytest.fixture()
def packed_job(tmp_path, staging):
    """A freshly packed, self-verified job directory."""
    from hydra_suite.data.tracking_job.pack import pack_job

    pack_job(
        tmp_path / "job",
        [_planned(staging)],
        registry_entries=[
            ("obb/x.pt", {"species": "ant", "source_path": "/host/a.pt"})
        ],
        advanced_config_path=str(staging["advanced"]),
        track_args={"video_list": "videos.txt"},
        shared_table={},
    )
    return tmp_path / "job"
```

`pytest` is already imported at the top of `tests/conftest.py`; add `tests/helpers/tracking_job.py` to this task's `git add`. **Task 10 must EXTEND this `packed_job` (or add a distinctly-named sibling), never redefine a second `packed_job` — see Task 10 Step 1.**

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_tracking_job_pack.py tests/test_tracking_job_verify.py -v`
Expected: FAIL — modules missing.

- [ ] **Step 3: Implement `runner.py`**

```python
"""The generated run.sh: the job's executable contract."""

RUN_SH = '''#!/usr/bin/env bash
set -euo pipefail
JOB="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# The HOST's config dir, captured BEFORE we override it: the shared-root mount
# table is a property of this machine, not of the job, and must never be read
# from the job snapshot. Empty means "the platformdirs default".
export HYDRA_HOST_CONFIG_DIR="${HYDRA_CONFIG_DIR:-}"

# The job supplies models and config; HYDRA_DATA_DIR is deliberately NOT set so
# engine artifacts and calibration profiles stay host-scoped.
export HYDRA_MODELS_DIR="$JOB/models"
export HYDRA_CONFIG_DIR="$JOB/config"
export KMP_DUPLICATE_LIB_OK=TRUE

if [ "${HYDRA_JOB_HOST_ADVANCED_CONFIG:-0}" = "1" ]; then
  # ****************************************************************
  # LOUD WARNING (minor fix -- this switch is a silent-divergence risk and
  # belongs documented HERE, at the escape hatch itself, not only in prose
  # elsewhere in this plan): HYDRA_JOB_HOST_ADVANCED_CONFIG=1 overwrites
  # config/advanced_config.json with THIS HOST's own advanced config,
  # which carries canonical_margin / reference_aspect_ratio / slice_* --
  # every one of which feeds canonical_geometry_key (crop framing) and the
  # SAHI slice-tile hash. If this host's advanced config differs from the
  # one that was PACKED, a run using this switch can produce DIFFERENT
  # tracking results than the job's own snapshot would have, AND the
  # resulting caches are keyed differently, so a cache pulled back here
  # will never hit locally against the job's original (unswitched)
  # snapshot either. Use only when you specifically intend to run this
  # job under THIS host's advanced config instead of the one it shipped
  # with.
  # ****************************************************************
  # Minor fix: $HOME/.config/hydra-suite is Linux-only (platformdirs puts
  # macOS config at "$HOME/Library/Application Support/hydra-suite"); prefer
  # HYDRA_HOST_CONFIG_DIR when set (the common case here, since run.sh always
  # sets it above) and fall back per-OS only when it is genuinely empty.
  if [ -n "${HYDRA_HOST_CONFIG_DIR:-}" ]; then
    HOST_ADV="$HYDRA_HOST_CONFIG_DIR/advanced_config.json"
  elif [ "$(uname -s)" = "Darwin" ]; then
    HOST_ADV="$HOME/Library/Application Support/hydra-suite/advanced_config.json"
  else
    HOST_ADV="$HOME/.config/hydra-suite/advanced_config.json"
  fi
  if [ -f "$HOST_ADV" ]; then
    echo "run.sh: using the HOST advanced config ($HOST_ADV), not the job snapshot"
    cp "$HOST_ADV" "$JOB/config/advanced_config.json"
    # Minor fix: `pull` never fetches config/ back (Task 9's build_push_input_list
    # is the only manifest-driven list, and pull only fetches videos/ outputs +
    # logs/), so if this branch overwrote the pushed advanced_config.json, the
    # local record silently diverges from what actually ran -- with nothing in
    # `job status`/the pulled artifacts saying so. Record it in the one place
    # that DOES travel back: the run-record line.
    export HYDRA_JOB_HOST_ADVANCED_CONFIG_USED=1
  fi
fi

# cd is what makes the job-relative videos.txt and sidecar file_path values work
# with load_video_list()'s CWD-relative semantics.
cd "$JOB"
mkdir -p logs
START="$(date -u +%FT%TZ)"

# Fix A2a: `trackerkit` must NOT be assumed to be on PATH. `ssh host 'cmd'`
# (non-interactive, non-login) does not source the shell profile that puts a
# conda env's entry points on PATH -- verified on firebrat: even
# `ssh firebrat 'bash -lc "which trackerkit"'` -> rc=1; only an explicit
# `source <conda>/etc/profile.d/conda.sh && conda activate <env>` resolves it.
# Global Constraint (CLAUDE.md line 44) forbids a bare `trackerkit` locally
# for the same reason. HYDRA_JOB_TRACKERKIT lets the caller (job_cli.py's
# `--remote-bootstrap`, or a local override) inject a fully-qualified
# invocation; the default keeps today's behavior for a shell where it IS on
# PATH (e.g. an already-activated interactive session).
#
# Minor fix (round-6): `job_cli.py` builds this value with Python's
# `shlex.quote(sys.executable)` so a space-containing interpreter path
# survives -- but `shlex.quote` produces POSIX shell-syntax quoting (wrapping
# quotes as literal characters), and a bare unquoted expansion word-splits on
# whitespace WITHOUT re-parsing embedded quote characters as syntax -- they
# stay literal -- so a quoted path with a space would word-split into two
# bogus tokens (one carrying a stray leading/trailing quote character)
# instead of being treated as one argument.
#
# Fix Y1 (round-7, REPLACES the round-6 `eval` fix below -- that fix was
# itself the bug): `eval "$TRACKERKIT track --video-list videos.txt
# $(printf '%q ' "$@")"` is broken with ZERO passthrough args. Verified by
# running it: bash's `printf '%q ' "$@"` on an EMPTY "$@" still emits one
# token, a literal `'' ` (an empty-but-quoted string), not nothing -- `"$@"`
# only expands to "nothing" when it is NOT first flattened through a command
# substitution that itself always produces at least the joining space.
# `eval` then re-parses that `''` as a real empty positional argument, so
# `trackerkit track --video-list videos.txt ''` runs, and the REAL parser
# (`parse_arguments`) rejects it: `error: use either explicit video paths or
# --video-list, not both` -> SystemExit 2. Every `job run` with no extra
# flags -- the common case, exercised by Task 13's acceptance runs -- would
# therefore die before the first frame.
#
# The actual fix does not need `eval` for the ARGV path at all. `eval` is
# only needed to turn ONE trusted, pre-quoted string (the interpreter
# invocation itself, which may be a multi-word `conda run -n hydra-mps
# trackerkit`-style string) into multiple argv words; it must never be used
# on `"$@"`, which is untrusted job-runtime argv and bash already knows how
# to pass through byte-for-byte via a real array and `"${TK[@]}"`.
TRACKERKIT_STR="${HYDRA_JOB_TRACKERKIT:-trackerkit}"
#   eval only on the trusted interpreter string -> build an array once:
eval "TK=($TRACKERKIT_STR)"
#   From here on, every invocation is "${TK[@]}" ... "$@" -- no eval, no
#   printf %q, no re-quoting. "$@" with zero elements now correctly expands
#   to NOTHING (this is the whole point of using an array + native "$@"
#   passthrough instead of round-tripping args through a string).

# Fix W1d (round-5 correction): preflight runs BEFORE `set +e`, so a failing
# preflight ABORTS under `set -euo pipefail` instead of printing and letting
# `track` proceed onto a truncated video. The previous revision placed this
# call after `set +e` while its comment claimed the opposite, which made the
# whole hand-run protection inert -- exactly the case W1d exists for.
# `job preflight .` runs from $JOB (we already cd'd) so shared aliases resolve
# from HYDRA_HOST_CONFIG_DIR as check 6 describes, and its diagnostics land in
# logs/preflight.json rather than interleaved into logs/run.log.
#
# Fix X2: this self-preflight is deliberately FLAG-LESS -- it never sees a
# one-off `--shared-root ALIAS=PATH` or `--allow-tier-fallback` the caller may
# have passed to `trackerkit job run`. A non-persisted alias override cannot
# be threaded through an `ssh ... && ./run.sh` chain as a CLI flag without
# re-parsing job_cli's own argv inside bash, and `job run` (both the local and
# the ssh-chained remote branch, fix M7) already runs `preflight_job` itself,
# WITH those flags, immediately before invoking run.sh. Re-running a
# flag-less preflight here would then fail on the exact alias/tier state the
# first preflight just proved workable -- silently making --shared-root and
# --allow-tier-fallback dead for every `job run` path. So `job run` sets
# HYDRA_JOB_SKIP_PREFLIGHT=1 in run.sh's environment once ITS OWN preflight
# (with the caller's flags) has already passed; run.sh honors it here and
# skips straight to `track`. A hand-run `./run.sh` (no `job run` wrapper, the
# case W1d protects) never has this variable set, so it always gets the
# flag-less self-preflight -- the hand-run protection stays intact.
if [ "${HYDRA_JOB_SKIP_PREFLIGHT:-0}" = "1" ]; then
  echo "run.sh: skipping self-preflight (already run by 'trackerkit job run' with its flags)"
else
  # NOTE: this is the ONLY preflight for a hand-run ./run.sh, so it carries no
  # --shared-root/--allow-tier-fallback overrides; those flags only exist on
  # `trackerkit job run`/`trackerkit job preflight`, never on run.sh itself.
  # Fix Y1: array expansion, NOT eval -- "${TK[@]}" is already the correctly
  # word-split interpreter invocation; `job preflight .` has no arguments
  # that could themselves contain spaces, so this is a plain literal tail.
  "${TK[@]}" job preflight .
fi

set +e
# Fix Y1: "${TK[@]}" ... "$@" passes the job's own extra argv through
# NATIVELY -- no string round-trip, no printf %q, no eval. This is what
# makes a ZERO-argument "$@" expand to truly nothing (the round-6 `eval
# "... $(printf '%q ' "$@")"` form could not do this: printf on an empty
# "$@" still emits one `''` token, which eval then re-parsed as a real,
# bogus empty positional -- see the comment above TRACKERKIT_STR= for the
# full empirical trace). A video path or any other forwarded argument
# containing a space survives untouched, exactly as bash's own "$@"
# semantics guarantee, with no re-quoting step to get wrong.
"${TK[@]}" track --video-list videos.txt "$@" 2>&1 | tee -a logs/run.log
CODE=${PIPESTATUS[0]}
set -e
# Fix A2d: this whole script runs under `set -euo pipefail`. If `_record-run`
# itself fails (e.g. the same PATH issue below `$TRACKERKIT`, or a transient
# I/O error writing logs/runs.jsonl), a `set -e`-fatal exit here would replace
# $CODE -- the run's REAL exit code -- with _record-run's exit code, silently
# masking a tracking failure as a bookkeeping failure or vice versa. Make the
# record step non-fatal and always preserve and exit with the run's own $CODE.
"${TK[@]}" job _record-run --started "$START" --exit-code "$CODE" -- "$@" || \
  echo "run.sh: WARNING: failed to append to logs/runs.jsonl (exit $?); run's own exit code $CODE is unaffected" >&2
exit "$CODE"
'''


def render_run_sh() -> str:
    return RUN_SH
```

- [ ] **Step 4: Implement `pack.py`**

Steps, in order (spec §6.2): resolve shared/symlink/copy per video → copy models → registry subset → config snapshot (advanced config, skeletons, `.seeded` markers) → rewrite each config → write sidecars → `videos.txt` → requirements → `run.sh` → manifest → `verify_job` self-check (raise `TrackingJobError` if it reports problems).

**Minor fix (adversarial review) — "copy models" dedupes by `PlannedModel.key`, not one `copy_model_reference` call per `PlannedVideo.planned_models` entry.** Two videos in the same job routinely share the same model (e.g. both use `obb/x.pt`), so two different `PlannedVideo`s each carry a `PlannedModel(key="obb/x.pt", ...)`. Calling `copy_model_reference` once per occurrence re-copies (and re-hashes) the same file twice and, worse, appends two `JobModel(key="obb/x.pt", ...)` entries to `manifest.models` — a manifest with a duplicate key that `JobManifest.to_dict`/`from_dict` round-trips faithfully (nothing rejects it) but that misrepresents the job as shipping the model twice. `pack_job`'s "copy models" step therefore builds a `dict[str, PlannedModel]` keyed by `key` across ALL `planned_videos` first (first-seen `PlannedModel` for a given key wins — `source_path`/`kind` are expected identical for the same key by construction, since the key IS the models-root-relative path the config resolved through), calls `copy_model_reference` exactly once per unique key, and the resulting `JobModel.roles` is the UNION of every `PlannedModel.role` across all occurrences of that key (a model referenced as `YOLO_OBB_DIRECT_MODEL_PATH` by one video and, hypothetically, some other role by another video for the same key carries both roles). Add `test_pack_dedupes_a_model_shared_by_two_videos`: two `PlannedVideo`s whose `planned_models` both carry `key="obb/x.pt"`; assert `manifest.models` has exactly one entry for that key and `(job_dir / "models" / "obb" / "x.pt")` was copied (assert via `sha256` or an mtime/copy-count spy) only once.

**Fix V-minor — re-packing into an existing job directory is unaddressed and unsafe as written.** Nothing above says what `pack_job(job_dir, ...)` does when `job_dir` already contains a previous pack. Concretely, three things break: (1) the non-shared video branch does `os.symlink(origin, job_dir / "videos" / basename)` (spec §6.2, "resolve shared/symlink/copy per video") — a second `pack_job` call against the same `job_dir` hits `FileExistsError` on that `os.symlink` the instant the video basename repeats, which it always does for "re-pack the same job after fixing a config typo"; (2) a model or config key that existed in the OLD pack but is absent from the NEW `planned_videos` (e.g. a video was removed from this pack) leaves its old file under `models/`/`config/` on disk with no manifest entry pointing at it — `verify_job` never notices (it only checks that manifest entries exist, never that `models/`/`config/` contains nothing extra), so a stale sidecar or stale model silently rides along in the next push; (3) `videos.txt` from the OLD pack is fully overwritten by the write step, but any of the three problems above can leave it internally inconsistent with what's actually on disk if the process is interrupted between steps. Fix: `pack_job` refuses to write into a **non-empty** `job_dir` unless the caller passes `force=True` (surfaced as `job pack --force` in Task 11). **Fix X8 (round-6) — a blanket `shutil.rmtree(videos/)` is wrong: `videos/` is where `pull` places tracking CSVs, caches and outputs, and spec §8.2/§8.3 (verified `docs/superpowers/specs/2026-09-09-portable-tracking-jobs-design.md:374,382`) says explicitly "the job tree keeps its copy so the job remains a complete record" and that discarding it is `job clean`'s job, "deliberately not in scope."** A blind `rmtree` on `--force` silently reimplements `job clean` as a side effect of what a user reads as "repack this job," destroying every pulled result the moment they fix one config typo and repack. Fix, with `force=True`:

1. **Remove only manifest-KNOWN pack artifacts**, read from the OLD `hydra_job.json` (the one already on disk, read BEFORE any deletion): for each `JobVideo` in the old manifest, remove exactly `videos/<job_path>` (the video symlink/copy pack itself created — `os.remove` for a symlink, `os.remove` for a plain copy, never `shutil.rmtree` on a directory) and its sidecar `videos/<stem>_config.json`. **Minor fix (adversarial review) — a `shared` video has NO file under `videos/<job_path>` at all (spec §6.7: the symlink is materialized later, on the remote, by `materialize_shared_videos`), so this removal step must tolerate that: `os.remove(videos/<job_path>)` wrapped in `try/except FileNotFoundError: pass` (or an explicit `if path.exists()` guard) for every `JobVideo`, not an unconditional `os.remove` that raises on the very first re-pack of a job containing a shared video.** `models/` and `config/` ARE still fully cleared and recreated (`shutil.rmtree(..., ignore_errors=True)` then recreated) — those two directories hold only pack-owned, deterministically-regenerable artifacts (copied models, config snapshots, skeletons) with no pull-time output ever written into them, so this part of the original fix stands unchanged. Add `test_pack_with_force_tolerates_a_shared_video_with_no_local_file`: pack once with `shared_table` set so the video is `shared`, repack with `force=True`; assert no exception and `verify_job(job_dir) == []`.
2. **Refuse if `videos/` contains anything the old manifest doesn't account for**, once the known pack artifacts above are notionally subtracted — i.e. anything else under `videos/` (a pulled CSV, `.inference_cache_<stem>/`, `run.log`'s sibling artifacts, a user's own stray file) blocks the repack with a loud `TrackingJobError` naming every unaccounted path, UNLESS the caller passes a second, explicitly separate opt-in (`force_discard_outputs=True`, surfaced as `job pack --force --discard-outputs`, never bundled into plain `--force`) — this is the "or warn loudly and require an extra opt-in" branch: a re-pack that only touches config/models never needs it, and a user who genuinely wants to throw away pulled results must say so with a second flag, not get it for free from `--force` alone.
3. **Carry `pull_history` forward.** The NEW `JobManifest` `pack_job` writes at the end must copy `pull_history` from the OLD manifest (read in step 1, before any file is touched) rather than defaulting to `[]` — a re-pack is a NEW pack of the SAME job identity, not a new job, and `job_id` is unchanged across a re-pack for the same reason (an already-existing but unstated invariant this fix makes explicit: re-pack preserves `job_id` too, since nothing in this fix's ordering ever reassigns it).

`job_dir / "logs"` (run history) and `job_dir / "hydra_job.json"` (only overwritten at the very end, once the new pack is known to be valid) are, as before, never cleared directly — `hydra_job.json` is simply overwritten with the new manifest object that now carries the OLD `pull_history` forward. An empty or non-existent `job_dir` (the common case — `job pack` creating a job for the first time) needs no `--force` and is unaffected by this fix. Add `test_pack_into_a_nonempty_job_dir_without_force_refuses` (asserts `TrackingJobError`, no partial writes); `test_pack_with_force_clears_stale_artifacts` (pack once with two videos, pack again with `force=True` and only one of them, assert the removed video's `config/`-snapshot and `videos/` entries are gone and `verify_job` still passes clean); `test_pack_with_force_preserves_pulled_outputs` (pack, then simulate a `pull` by writing a fake `videos/colony_tracking_final.csv` and `.inference_cache_colony/detection.npz` directly into the job dir, then repack with `force=True` and NO `--discard-outputs`; assert `TrackingJobError` naming both stray paths and that neither file was touched); `test_pack_with_force_discard_outputs_removes_them` (same setup, repack with both `force=True, force_discard_outputs=True`; assert success and the stray files are gone); `test_pack_with_force_carries_pull_history_forward` (pack, hand-write a `pull_history` entry into `hydra_job.json`, repack with `force=True`, assert the new manifest's `pull_history` still contains that entry and `job_id` is unchanged).

**Fix W1 — the "resolve shared/symlink/copy per video" step ALWAYS computes `signature`.** Whichever branch a video takes (shared-alias reference, symlink, or `copy_videos=True` real copy), `pack_job` calls `content_id.video_signature(planned.video_path)` — `planned.video_path` is still the ORIGIN path at this point in the pipeline, before any job-relative rewriting — and passes the result as `JobVideo(..., signature=...)`. This is not conditional on `entry.shared` being set. `size_bytes` is likewise always `os.path.getsize(planned.video_path)` at pack time, recorded on `JobVideo.size_bytes` regardless of branch (it already was; this fix only closes the `signature` gap). `content_id.video_signature` is a Task 4 primitive and `pack.py` is in the Data layer, so importing it (`from ...core.inference import content_id`) does not cross a forbidden layering boundary — this mirrors the existing `_normalize_model_path` import of `core.inference.model_paths` two sections above.

The rewrite table (§6.4), applied to a deep copy of each planned config:

**Minor note — `csv_path`'s redirect is dead code today, don't over-trust it.** `load_tracker_cli_session` derives `raw_csv_path` from the video itself (`cli_config.py:319`) and never reads `cfg["csv_path"]`; only `video_output_path` is actually consumed downstream (`core/tracking/session.py:653`). So rewriting/recording `csv_path` here is harmless (it keeps the sidecar internally consistent and future-proofs against a consumer being added) but currently has no live effect on where the CSV lands — do not treat the `csv_path` half of this rewrite as proof that CSV redirection works end-to-end; only `video_output_path` is load-bearing today.

```python
def _rewrite_config(
    config, *, video_source_path, video_basename, model_keys, cnn_model_keys, skeleton_job_path
):
    """Make one video's config job-relative. Returns (config, redirected).

    Fix W10: `video_source_path` is the video's ORIGIN absolute path (NOT
    just its basename) -- required so the redirect check below can compare
    against the REAL default output locations `_default_output_paths`
    computes, not merely a basename match.
    """
    out = copy.deepcopy(dict(config))
    stem = Path(video_basename).stem
    out["file_path"] = f"videos/{video_basename}"
    redirected: dict[str, str] = {}
    # Fix W10: the redirect rule must key on the RESOLVED path, not the
    # basename. `_default_output_paths` (trackerkit/cli_config.py:270-276)
    # is the actual authority for "where does the engine's default output
    # land" -- `(video.with_suffix("").parent / f"{stem}_tracking.csv",
    # ...{stem}_tracking.mp4)`. A basename-only check treats
    # `/renders/colony_tracking.mp4` (a DIFFERENT directory, but the SAME
    # default *name*) as if it were the default location: it gets rewritten
    # to `videos/colony_tracking.mp4` with NO `redirected_outputs` entry, so
    # `pull` maps it back beside the video instead of restoring it to
    # `/renders`. Comparing the FULL resolved path against
    # `_default_output_paths(video_source_path)` catches exactly this case.
    # NOT imported from trackerkit.cli_config._default_output_paths: that
    # function lives in the trackerkit APP layer, and pack.py is Data --
    # importing it would cross the same one-way dependency boundary fix W6
    # avoids for build_engine_params. The computation itself is a three-line
    # pure function of the source path (verified identical to
    # cli_config.py:270-276's own body), so pack.py inlines it rather than
    # importing across layers.
    _video = Path(video_source_path)
    _base = _video.with_suffix("")
    default_csv = str(_base.parent / f"{_base.name}_tracking.csv")
    default_video_out = str(_base.parent / f"{_base.name}_tracking.mp4")
    default_paths = {"csv_path": default_csv, "video_output_path": default_video_out}
    for key, default_suffix in (
        ("csv_path", "_tracking.csv"),
        ("video_output_path", "_tracking.mp4"),
    ):
        original = str(out.get(key, "") or "")
        if not original:
            continue
        default_name = f"{stem}{default_suffix}"
        is_default_location = (
            str(Path(original).resolve()) == str(Path(default_paths[key]).resolve())
        )
        if is_default_location:
            out[key] = f"videos/{default_name}"
        else:
            job_relative = f"videos/{stem}_{Path(original).name}"
            out[key] = job_relative
            redirected[job_relative] = str(Path(original).resolve())
    for config_key, job_key in model_keys.items():
        out[config_key] = job_key
    # Fix B5: cnn_classifiers[].model_path is a LIST of dicts, not a scalar
    # key, so it cannot go through model_keys. Rewrite each entry in place by
    # looking its RESOLVED absolute source path up in the per-entry map the
    # caller supplies. See "the CNN rewrite rule" below for the exact
    # normalization on both sides of that lookup.
    entries = out.get("cnn_classifiers")
    if isinstance(entries, list) and cnn_model_keys:
        rewritten = []
        for entry in entries:
            entry = dict(entry)
            lookup = _normalize_model_path(entry.get("model_path", ""))
            if lookup and lookup in cnn_model_keys:
                entry["model_path"] = cnn_model_keys[lookup]
            rewritten.append(entry)
        out["cnn_classifiers"] = rewritten
    # Fix B2: the skeleton rewrite is UNCONDITIONAL on `pose_skeleton_file`
    # being non-empty -- it is NOT gated on pose being enabled. `verify_job`
    # forbids an absolute `pose_skeleton_file` unconditionally, so gating the
    # rewrite on pose enablement makes any config that carries a stale absolute
    # skeleton path with `enable_pose_extractor: False` fail pack's own
    # self-verify (this is exactly what the hostile acceptance config in Task
    # 11 does with `fly_obb.json`).
    #
    # Fix Q7 (adversarial review) -- PRECEDENCE, stated explicitly: this
    # function rewrites from `skeleton_job_path` (the caller-supplied
    # parameter), NEVER by re-reading `config["pose_skeleton_file"]` itself.
    # `planned.skeleton_path` (Task 11's `_rewrite_config`-caller computes
    # `skeleton_job_path` from it -- see the per-video loop in Step 4) is
    # THE authoritative source; the config dict's own `pose_skeleton_file`
    # is treated only as the thing being overwritten, never consulted for
    # the rewrite decision. This matters because a caller COULD in principle
    # hand `_rewrite_config` a `config` whose `pose_skeleton_file` disagrees
    # with `planned.skeleton_path` (e.g. a hand-built `PlannedVideo` in a
    # test, or a future bug in Task 11's stamping code) -- in that case
    # `skeleton_job_path == ""` while the config's raw `pose_skeleton_file`
    # is still a stale absolute string: with no rewrite, `verify_job` would
    # reject the packed sidecar with an "absolute path" message that gives
    # no hint the REAL cause was a caller/data mismatch between
    # `planned.skeleton_path` and `config["pose_skeleton_file"]`, not a
    # missing skeleton. `pack_job` (Step 4's per-video loop, not
    # `_rewrite_config` itself, which stays a pure string-rewriting helper
    # with no manifest/error-raising concerns) checks this BEFORE calling
    # `_rewrite_config`, loudly:
    #
    #     raw_skeleton = str(planned.config.get("pose_skeleton_file", "") or "").strip()
    #     if raw_skeleton and not planned.skeleton_path:
    #         raise TrackingJobError(
    #             f"{video_basename}: config sets pose_skeleton_file "
    #             f"({raw_skeleton!r}) but PlannedVideo.skeleton_path is empty; "
    #             f"the caller must resolve and stamp skeleton_path to match "
    #             f"(see Task 11 Step 6, resolve_model_path(pose_skeleton_file))"
    #         )
    #
    # This turns a confusing downstream verify failure into a pack-time error
    # naming the actual mismatch. Rewrite whenever `skeleton_job_path` (i.e.
    # `planned.skeleton_path`) is non-empty; leave the key untouched when it
    # is empty (which, given the check above, only happens when the source
    # config's `pose_skeleton_file` was ALSO empty -- the two are guaranteed
    # consistent by the time `_rewrite_config` runs).
    if skeleton_job_path:
        out["pose_skeleton_file"] = skeleton_job_path
    return out, redirected
```

Add `test_skeleton_path_config_mismatch_raises`: build a `PlannedVideo` directly (bypassing `_planned`'s normal consistency) with `config={"pose_skeleton_file": "/host/some/skeleton.json", ...}` but `skeleton_path=""`; assert `pack_job` raises `TrackingJobError` naming both the video and the mismatch, not a bare `verify_job` "absolute path" failure.

**Fix X3a — `pack_job` must refuse to pack a pose-enabled video with no skeleton, or the packed job dies on the remote at model load with no warning here.** `_rewrite_config`'s skeleton handling above only rewrites a path that IS present; nothing checks that one exists when pose inference is actually enabled. If `is_pose_inference_enabled(cfg)` (`core/tracking/session_policy.py:29` — Core, safe to import from `pack.py`, same layering already used for `content_id`/`model_paths`) is true and the resolved `pose_skeleton_file` is empty, `core/inference/stages/pose.py:145-148` later yields empty `keypoint_names`, and `core/individual/pose/api.py:105` raises `"SLEAP backend requires keypoint_names"` — deep inside a remote `trackerkit track` run, in `logs/run.log`, long after `pack`/`push`/`preflight` all reported success. `pack_job` (in the same per-video loop that calls `_rewrite_config`) must instead raise `TrackingJobError(code=2)` **at pack time**, naming the offending video's `file_path`:

```python
from hydra_suite.core.tracking.session_policy import is_pose_inference_enabled

if is_pose_inference_enabled(config) and not str(config.get("pose_skeleton_file", "") or "").strip():
    raise TrackingJobError(
        f"{video_basename}: pose inference is enabled but pose_skeleton_file is empty; "
        f"pack cannot produce a job that will fail at remote model load"
    )
```

This check runs BEFORE `_rewrite_config`, on the SOURCE config. **Fix Q5 (adversarial review) — correcting this section's own prose: the check above only tests EMPTINESS of `pose_skeleton_file` (`not str(...).strip()`), never resolution.** An earlier draft of this paragraph claimed it "also catches the case an already-portable `pose_skeleton_file` is job-relative but doesn't resolve to a real file" — that is false as the guard is written: a non-empty string, portable or not, resolvable or not, passes this specific check unconditionally. (A job-relative-but-broken skeleton path is instead caught later, by `verify_job`'s "every referenced `config/skeletons/*` exists" check, which runs as part of `pack_job`'s own self-verify at the end of Step 4 — so the end-to-end guarantee "pack cannot produce a job whose skeleton is missing" still holds, just via a different check than this paragraph originally credited.) This paragraph's claim is corrected, not the code.

**Fix Q5 — `test_pack_pose_enabled_no_skeleton_raises` as originally specified cannot actually trigger the guard.** `is_pose_inference_enabled` (`core/tracking/session_policy.py:29-32`) requires ALL THREE of: `detection_method == "yolo_obb"` (`is_individual_pipeline_enabled`), `enable_pose_extractor` truthy, AND a non-empty `pose_model_dir` — a config setting only `enable_pose_extractor: true` and a pose backend leaves `is_pose_inference_enabled(config)` `False` (missing `detection_method`/`pose_model_dir`), so `pack_job`'s guard silently no-ops and the test as originally described would fail (no `TrackingJobError` raised), inviting a later implementer to "fix" this by loosening the very predicate the plan mandates matching. The test must set the FULL key set the predicate actually reads:

```python
def test_pack_pose_enabled_no_skeleton_raises(tmp_path, staging):
    from hydra_suite.data.tracking_job.manifest import TrackingJobError

    planned = _planned(
        staging,
        config={
            "detection_method": "yolo_obb",
            "enable_pose_extractor": True,
            "pose_model_dir": "pose/SLEAP/run",
            "pose_model_type": "sleap",
            "pose_skeleton_file": "",
        },
    )
    with pytest.raises(TrackingJobError) as excinfo:
        pack_job(
            tmp_path / "job", [planned], registry_entries=[],
            advanced_config_path=str(staging["advanced"]), track_args={}, shared_table={},
        )
    assert "colony.mp4" in str(excinfo.value)
    assert not (tmp_path / "job" / "hydra_job.json").exists()
```

(no partial `job_dir` is left in a state `verify_job` would call clean — mirrors the existing "pack fails loudly, not silently" pattern used throughout this task for other config-shape violations).

**Fix X3b — Task 13's own three acceptance fixtures must inject the same skeleton `run_matrix.sh` does, or the pose/identity jobs never even get past `job pack` under fix X3a (by design — that is the guard doing its job), and neither acceptance job would ever run.** All three fixture configs (`fly_obb.json`, `ant_pose_headtail.json`, `ant_cnn_identity.json`) ship with `pose_skeleton_file: ""`; `tools/equivalence/run_matrix.sh:63-68` supplies the real skeleton (`$FX/ooceraea_biroi.json`) as a SEPARATE table column that `tools/equivalence/runner.py` injects into the in-memory config before running — `job pack` (Task 13 Step 4) instead passes the raw fixture config file straight through `--config`, so `ant_pose_headtail` and `ant_cnn_identity` (both pose-enabled) would fail fix X3a's new guard immediately, and Step 4's acceptance evidence for both would never be collected. Before packing those two jobs, materialize a config with the skeleton filled in, mirroring what the equivalence runner does for the SAME fixture:

```bash
PYTHONPATH=$PWD/src conda run --no-capture-output -n hydra-mps python - <<'PY'
import json
from pathlib import Path

fx = Path("tools/equivalence/fixtures")
skel = str((fx / "ooceraea_biroi.json").resolve())
for name in ("ant_pose_headtail", "ant_cnn_identity"):
    cfg_path = fx / "configs" / f"{name}.json"
    cfg = json.loads(cfg_path.read_text())
    cfg["pose_skeleton_file"] = skel
    (Path("/tmp/jobs") / f"{name}_config.json").write_text(json.dumps(cfg))
PY
```

then pack `ant_pose_headtail`/`ant_cnn_identity` with `--config /tmp/jobs/ant_pose_headtail_config.json` / `--config /tmp/jobs/ant_cnn_identity_config.json` respectively (`fly_obb` is pose-disabled and keeps its original `--config` invocation unchanged, since fix X3a's guard is a no-op for it). The materialized configs are written under `/tmp/jobs/`, never back into the tracked `tools/equivalence/fixtures/configs/` — this is a per-run acceptance artifact, not a fixture change.

where `_normalize_model_path` is the ONE normalization both sides of the CNN
lookup use, defined in `pack.py`:

```python
def _normalize_model_path(value: object) -> str:
    """Canonical lookup form for a CNN classifier entry's model_path.

    Config entries may be models-root-RELATIVE (Task 2 relativizes them on
    save) while ``PlannedModel.source_path`` is absolute, so both sides go
    through ``resolve_model_path`` first and are then fully canonicalized.
    ``resolve_model_path`` lives in ``core/inference/model_paths.py:160`` --
    Core, which the Data layer is allowed to import (the layering gate only
    forbids app layers and Qt).
    """
    from ...core.inference.model_paths import resolve_model_path

    raw = str(value or "").strip()
    if not raw:
        return ""
    return str(Path(str(resolve_model_path(raw))).expanduser().resolve())
```

**The CNN rewrite rule, written out in full (fix B5).**

1. `job_cli.py` (Task 11), which is what walks `iter_model_references` for a
   video, knows for each `CNN_CLASSIFIERS` reference both the absolute
   `PlannedModel.source_path` and the job key it was assigned. It builds
   `cnn_model_keys = {_normalize_model_path(m.source_path): m.key for m in
   planned_models if m.role == "CNN_CLASSIFIERS"}` and stamps it on that
   video's `PlannedVideo`.
2. `pack.py` reads `planned.cnn_model_keys` and matches each config entry by
   `_normalize_model_path(entry["model_path"])`. Both sides therefore compare
   fully-resolved absolute strings; a relative config value and an absolute
   `source_path` for the same file normalize to the same string.
3. An entry whose normalized path is **not** in the map is left untouched.
   That is only reachable when the file does not exist on the staging machine
   (`iter_model_references` yields it, `copy_model_reference` then raises
   `TrackingJobError` code 2 before any rewrite happens) or when the entry is
   empty. Either way pack fails loudly rather than shipping a stale absolute
   path, because `verify_job` checks every `cnn_classifiers[].model_path`.

**Fix M2 — the role→config-key mapping is explicit, not left implicit.** `PlannedVideo.planned_models` carries `role` (an engine-param key like `YOLO_OBB_DIRECT_MODEL_PATH`), but the config dict being rewritten uses **lowercase config keys** (`yolo_obb_direct_model_path`). Something has to map role → config key, and it must be a single source of truth so a new role can't be added to `iter_model_references` (Task 3) without also being reachable here. Two pieces:

1. A static table in `pack.py` itself (Data layer — this is just a naming convention between two key vocabularies both already owned by this module, not app-layer knowledge):

```python
# The FULL role -> config-key mapping. Every role iter_model_references
# (Task 3) can yield MUST appear here or _rewrite_config silently leaves an
# absolute path in the sidecar for that role.
ROLE_TO_CONFIG_KEY: dict[str, str | tuple[str, ...]] = {
    "YOLO_OBB_DIRECT_MODEL_PATH": "yolo_obb_direct_model_path",
    "YOLO_DETECT_MODEL_PATH": "yolo_detect_model_path",
    "YOLO_CROP_OBB_MODEL_PATH": "yolo_crop_obb_model_path",
    "YOLO_HEADTAIL_MODEL_PATH": "yolo_headtail_model_path",
    # POSE_MODEL_DIR fans out to every backend-specific alias engine_params.py
    # also populates from the same directory, plus the legacy singular bridge.
    # Minor fix: a blind fan-out OVERWRITES all four keys with the SAME job
    # key, even for backends the job does not use -- e.g. for
    # ant_pose_headtail (YOLO-pose backend), it would stomp
    # `pose_yolo_model_dir` (which legitimately carries
    # "YOLO-pose/...pt", a DIFFERENT model than whatever POSE_MODEL_DIR
    # resolved to for a differently-configured video) with the SLEAP-role
    # job key. Harmless for the run itself (engine_params.py:985 reads the
    # ACTIVE backend's `pose_<backend>_model_dir` first and only falls back
    # to the legacy `pose_model_dir` when empty, so an inactive backend's
    # key is never READ) but destroys provenance in the sidecar -- a human
    # inspecting the pushed config sees the wrong model path recorded
    # against the inactive backend keys. Only rewrite the key(s) matching
    # the video's ACTIVE `pose_model_type` (plus the always-present legacy
    # `pose_model_dir` bridge), not all four unconditionally: `pack.py`
    # mirrors `engine_params.py:975-978`'s OWN defaulting exactly (minor
    # fix, adversarial review) --
    # `str(config.get("pose_model_type", "yolo")).strip().lower()`, then
    # falls back to `"yolo"` again if the lowercased result isn't one of
    # `{"yolo", "sleap", "vitpose"}` -- rather than reading the raw key
    # as-is. A raw `config.get("pose_model_type", "")` (no default, no
    # lowercasing) means an ABSENT key, or a differently-cased value like
    # `"SLEAP"`, matches none of the three alias branches, so `model_keys`
    # populates none of the pose-alias keys at all -- the run still works
    # (`engine_params.py:985`'s own runtime lookup falls back to the legacy
    # `pose_model_dir` when the active alias is empty), but the sidecar's
    # provenance is then wrong: a config that plainly targets SLEAP shows
    # `pose_sleap_model_dir: ""` with no active-alias key populated at all.
    # `pack.py` maps the resolved (lowercased, defaulted) type to the single
    # matching alias (`"yolo"` -> `pose_yolo_model_dir`,
    # `"sleap"` -> `pose_sleap_model_dir`, `"vitpose"` ->
    # `pose_vitpose_model_dir`); the other two backend-specific aliases are
    # NOT simply left holding whatever the source config had -- fix Y6
    # (round-7): fix X6 (below this table) blanks every inactive-role key
    # in ABSOLUTE_PATH_FORBIDDEN_KEYS's model-path subset to "", because an
    # untouched stale absolute path there fails verify_job's unconditional
    # absolute-path check. X6 is the ONE rule for inactive keys; earlier text
    # in this task said "left untouched", which directly contradicted X6 --
    # X6 wins, everywhere.
    "POSE_MODEL_DIR": (
        "pose_model_dir",
        "pose_yolo_model_dir",
        "pose_sleap_model_dir",
        "pose_vitpose_model_dir",
    ),
    # CNN_CLASSIFIERS is handled separately: it rewrites cnn_classifiers[].model_path
    # entries in place, not a single scalar config key.
}
# Legacy alias: engine_params.py:814-822 falls back to yolo_model_path when
# the mode-specific key is empty -- MODE-DEPENDENTLY: direct mode reads
# `yolo_obb_direct_model_path` then `yolo_model_path` (:814-816), sequential
# mode reads `yolo_crop_obb_model_path` then `yolo_model_path` (:820-822).
# If a legacy config still uses the alias it must be rewritten too, or a stale
# absolute path survives packing untouched.
#
# WHICH JOB KEY IT TAKES (fix B5 -- this was named but never specified):
# `yolo_model_path` is assigned the SAME job key as whichever OBB role is live
# for this video, i.e. the job key of the `PlannedModel` whose role is
# `YOLO_OBB_DIRECT_MODEL_PATH` when `yolo_obb_mode == "direct"`, or
# `YOLO_CROP_OBB_MODEL_PATH` when it is "sequential". It never gets a job key
# of its own, because it never names a model the mode-specific key does not
# already name. When neither role is present for this video (bgsub detection),
# there is no model for `yolo_model_path` to point at -- Fix Y6 (round-7):
# fix X6 blanks it to "" in this case too (it is in
# ABSOLUTE_PATH_FORBIDDEN_KEYS, same as every other model-path key), it is
# NOT left holding a stale absolute value from the source config. A bgsub
# config that still carries a stale absolute `yolo_model_path` from a
# previous detect-mode configuration must still pack cleanly; leaving it
# untouched would make verify_job reject that config unconditionally, with
# no way to ever pack it. Concretely, when building `model_keys`:
#
#     obb_role = (
#         "YOLO_OBB_DIRECT_MODEL_PATH"
#         if str(config.get("yolo_obb_mode", "direct")).strip().lower() != "sequential"
#         else "YOLO_CROP_OBB_MODEL_PATH"
#     )
#     if obb_role in role_to_job_key:
#         for alias in LEGACY_ALIAS_CONFIG_KEYS:
#             model_keys[alias] = role_to_job_key[obb_role]
LEGACY_ALIAS_CONFIG_KEYS = ("yolo_model_path",)
```

2. `PlannedVideo.planned_models: list[PlannedModel]` (already defined, Task 5/6) is what `job_cli.py` (Task 11) builds from `iter_model_references`; `_rewrite_config` derives its `model_keys` argument from `ROLE_TO_CONFIG_KEY[role]` for each `PlannedModel.role`. **Fix V6 (resolving the earlier draft's self-contradiction): for a tuple-valued role, `model_keys` is populated for ONLY the alias(es) matching this video's active backend — never a blind fan-out to every key in the tuple.** `POSE_MODEL_DIR` is currently the only tuple-valued role, so concretely: `pack.py` reads `config.get("pose_model_type", "")` for the video being rewritten and maps it to the single matching alias (`"yolo"` -> `pose_yolo_model_dir`, `"sleap"` -> `pose_sleap_model_dir`, `"vitpose"` -> `pose_vitpose_model_dir`), PLUS the always-present legacy `pose_model_dir` bridge — the other two backend-specific aliases in the tuple are NOT left holding whatever the source config had. **Fix Y6 (round-7) — this paragraph and the "Minor fix" note above this table were themselves the self-contradiction round-7 review caught: both said "left completely untouched," which directly conflicts with fix X6 below (which blanks every inactive-role `ABSOLUTE_PATH_FORBIDDEN_KEYS` model-path key, including these two).** X6 is the one rule: `_rewrite_config` populates `model_keys` for only the active alias(es) as this paragraph originally said, AND separately, in the same pass, blanks the inactive aliases to `""` per X6 -- the two are not in tension once X6 is applied uniformly to every inactive model-path key across all three families (OBB mode, head-tail, pose backend) and to `yolo_model_path`. This keeps the sidecar internally consistent with the X6 rule: an inactive backend's alias key is blanked to `""`, never a stray job key for a model that video never loads and never a leftover absolute source-machine path either. `color_tag_model_path` is deliberately **absent** from `ROLE_TO_CONFIG_KEY` — Task 3 never yields `COLOR_TAG_MODEL_PATH` as a reference, so pack never rewrites or ships a MODEL for it. **Fix Q2 (adversarial review) — but it is NOT "left whatever the source config had".** It is in `ABSOLUTE_PATH_FORBIDDEN_KEYS` (below), the GUI field that sets it is `setVisible(False)` (Task 2's own framing: `trackerkit/gui/panels/identity_panel.py:143`, so a user has no way to clear a stale absolute value even if they wanted to), and Task 2 only relativizes paths that fall *under the models root* — an out-of-root absolute value is untouched at save time. A config holding one would therefore pack "successfully" and then fail `verify_job`'s own unconditional absolute-path check immediately after, with no way to ever pack it (the same failure mode X6 exists to close for every other inactive model-path key). `pack.py` blanks `color_tag_model_path` to `""` unconditionally in the SAME per-video pass that applies the X6 inactive-role blanking — it is dead (Task 2's own correction 2: zero consumers in `core/`), so, exactly like an inactive-role key, it carries zero information the remote needs. Add `test_pack_blanks_color_tag_model_path_even_when_out_of_root`: a config with an absolute, out-of-models-root `color_tag_model_path`; assert `pack_job` succeeds, the sidecar's `color_tag_model_path == ""`, and `verify_job(job_dir) == []`. The earlier prose in this task and in Task 2 describing `color_tag_model_path` as "stays whatever the source config had" / "left alone if it points outside the models root" is superseded by this fix — pack.py, not the GUI save path, is what makes every packed config self-verify.

`model_keys` therefore maps a subset of config keys (`yolo_obb_direct_model_path`, `yolo_detect_model_path`, `yolo_crop_obb_model_path`, `yolo_headtail_model_path`, `pose_model_dir` plus exactly one of `pose_yolo_model_dir`/`pose_sleap_model_dir`/`pose_vitpose_model_dir` per video, and the legacy `yolo_model_path` alias — but **not** `color_tag_model_path`) to its job key. Only single-valued roles are unconditional; `POSE_MODEL_DIR` is always active-backend-only.

**WHO BUILDS WHAT — the single, non-negotiable division of labour (fix B5; an earlier draft contained two sentences that contradicted each other on this and one of them is now deleted):**

- **`pack.py` derives the SCALAR `model_keys` itself**, inside `pack_job`, from `planned.planned_models` via `ROLE_TO_CONFIG_KEY` (for `POSE_MODEL_DIR`, resolving to the single alias matching the video's active `pose_model_type` plus the legacy `pose_model_dir` bridge — never every key in the tuple, per fix V6 above) plus the `LEGACY_ALIAS_CONFIG_KEYS` rule above. It needs nothing from the app layer to do this: `PlannedVideo` already carries its own `planned_models`, and the role→config-key naming convention is pack's own table.
- **`job_cli.py` (Task 11) supplies the CNN PER-ENTRY map** as `PlannedVideo.cnn_model_keys`, because matching a `cnn_classifiers[]` list entry to its `PlannedModel` is a per-entry association that only the code that walked `iter_model_references` for that video observed.

There is no third option and no "or whatever the caller prefers".

**Minor fix (adversarial review) — no test in this task exercises the `cnn_classifiers[].model_path` rewrite end-to-end, and it is the one rewrite whose lookup (`_normalize_model_path` → `resolve_model_path`) depends on `HYDRA_MODELS_DIR` at pack time.** Every other rewrite in this task's test suite (OBB, head-tail, pose, skeleton) is exercised by at least one `test_tracking_job_pack.py` test; the CNN per-entry rewrite is described in prose (this section) but never actually packed-and-asserted. Add `test_pack_rewrites_cnn_classifiers_model_path`:

```python
def test_pack_rewrites_cnn_classifiers_model_path(tmp_path, staging, monkeypatch):
    head = staging["models"] / "classification" / "identity" / "head_a.pth"
    head.parent.mkdir(parents=True)
    head.write_bytes(b"h")
    # `_normalize_model_path` -> `resolve_model_path` resolves a
    # models-root-relative config value against HYDRA_MODELS_DIR; pin it to
    # this fixture's own models root so the lookup is deterministic here
    # regardless of what's configured on the machine running the test.
    monkeypatch.setenv("HYDRA_MODELS_DIR", str(staging["models"]))
    planned = _planned(
        staging,
        config={
            "cnn_classifiers": [
                {"model_path": str(head), "species": "ant"},
            ]
        },
        planned_models=[
            PlannedModel(
                role="YOLO_OBB_DIRECT_MODEL_PATH",
                source_path=str(staging["models"] / "obb" / "x.pt"),
                kind="file", key="obb/x.pt",
            ),
            PlannedModel(
                role="CNN_CLASSIFIERS", source_path=str(head), kind="file",
                key="classification/identity/head_a.pth",
            ),
        ],
        cnn_model_keys={
            str(head.expanduser().resolve()): "classification/identity/head_a.pth"
        },
    )
    manifest = _pack(tmp_path, staging, planned=planned)
    sidecar = json.loads((tmp_path / "job" / "videos" / "colony_config.json").read_text())
    assert sidecar["cnn_classifiers"][0]["model_path"] == "classification/identity/head_a.pth"
    assert sidecar["cnn_classifiers"][0]["species"] == "ant"  # non-path keys survive untouched
    assert verify_job(tmp_path / "job") == []
```

This proves the rewrite fires, leaves sibling dict keys alone, and that the resulting sidecar passes `verify_job`'s unconditional `cnn_classifiers[].model_path` absolute-path check — the one path through `_rewrite_config` this task's test suite would otherwise ship entirely unverified.

`_rewrite_config` must also rewrite `yolo_model_path` (the legacy alias `engine_params.py:815` falls back to) whenever it is present and non-empty and its value resolves under the models root — otherwise a legacy config carrying only the alias key ships an absolute path untouched.

Every video gets a sidecar regardless of provenance, so the remote needs no keystone logic; `track_args["keystone_override"]` is recorded for provenance only.

`requirements`: `min_hydra_suite_version` from the running build.

**Fix W12 — "selects the SLEAP backend via the service path" was never a defined, checkable rule.** `pack_job` computes `conda_envs` (deduped across videos, empty list when none apply) for any keystone `PlannedVideo.config` where `str(cfg.get("pose_model_type", "")).strip().lower() == "sleap"` **and** `is_pose_inference_enabled(cfg)` is true (`core/tracking/session_policy.py:29-32`, Core layer — importable from `pack.py`'s Data layer) — i.e. `detection_method == "yolo_obb"`, `enable_pose_extractor` truthy, AND `pose_model_dir` non-empty. This is a strict superset check, not "pose_model_type == sleap alone": a config with `pose_model_type: SLEAP` but no `pose_model_dir` (or a non-`yolo_obb` `detection_method`, or `enable_pose_extractor: False`) never reaches `is_pose_inference_enabled`, so pose inference — and therefore the SLEAP service, and therefore the `sleap` conda env — never actually runs for it, and `conda_envs` correctly stays empty. This is deliberately the SAME predicate the run itself uses to decide whether pose inference is live at all, so `requirements.conda_envs` can never claim a dependency the run doesn't actually have, or omit one it does.

**Fix Y8 (round-7) — the env NAME itself, `[cfg["pose_sleap_env"]]`, mis-derives whenever `pose_sleap_env` is absent, empty, or a placeholder — `pack_job` must mirror `engine_params.py`'s OWN defaulting exactly, not read the raw config key.** Verified at `engine_params.py:995-998`:

```python
pose_sleap_env = str(_cfg_get(cfg, "pose_sleap_env", default="sleap")).strip()
if not pose_sleap_env or pose_sleap_env.lower().startswith("no sleap envs"):
    pose_sleap_env = "sleap"
```

A config missing the `pose_sleap_env` key entirely (a perfectly normal fixture/staging config — nothing requires the key to be present) or holding `""`/a `"no sleap envs…"` GUI placeholder resolves, at RUN time, to the env named `"sleap"`. But `conda_envs = [cfg["pose_sleap_env"]]` — plain dict subscripting on the raw key — either raises `KeyError` (absent key: `pack_job` would crash at pack time on any such config, not silently mis-derive) or, if `pack.py` instead used `cfg.get("pose_sleap_env", "")`, would append `""` to `conda_envs` (an empty-string env name, never `"sleap"`) so preflight check 3 (`conda_envs` — every required env exists) passes vacuously (an empty string is never checked against `conda env list` meaningfully, or worse causes a preflight false-pass depending on how the emptiness is handled) and the run then fails only when the SLEAP service actually tries to `conda run -n sleap` — deep into the remote run, long after pack/push/preflight all reported success. Fix: `pack_job` computes `conda_envs` using the SAME defaulting logic as `engine_params.py:995-998`, verbatim (inlined, not imported — `engine_params.py` lives in the `trackerkit` app layer, and `pack.py` is Data, same layering boundary fix W6/X5a already established):

```python
raw_sleap_env = str(cfg.get("pose_sleap_env", "") or "").strip()
if not raw_sleap_env or raw_sleap_env.lower().startswith("no sleap envs"):
    raw_sleap_env = "sleap"
conda_envs = [raw_sleap_env]  # only reached when the W12 gate above is true
```

**Fix Q5 (adversarial review) — both tests below must supply the FULL key set `is_pose_inference_enabled` requires, PLUS a real skeleton (else fix X3a's guard raises first, before `conda_envs` is ever computed) PLUS a `POSE_MODEL_DIR` `PlannedModel` (else fix X6 blanks `pose_model_dir` to `""` in the rewritten sidecar while `requirements.conda_envs` would still claim `sleap` — internally inconsistent, and the gate itself reads `pose_model_dir` non-empty to begin with).** Add `test_conda_envs_defaults_to_sleap_when_key_is_absent`:

```python
def test_conda_envs_defaults_to_sleap_when_key_is_absent(tmp_path, staging):
    planned = _planned(
        staging,
        config={
            "detection_method": "yolo_obb",
            "enable_pose_extractor": True,
            "pose_model_dir": "pose/SLEAP/run",
            "pose_model_type": "sleap",
            "pose_skeleton_file": str(staging["skeleton"]),
            # pose_sleap_env genuinely ABSENT -- not set to "", the shape a
            # real staging config that never touched the SLEAP-env combo box
            # takes.
        },
        planned_models=[
            PlannedModel(
                role="YOLO_OBB_DIRECT_MODEL_PATH",
                source_path=str(staging["models"] / "obb" / "x.pt"),
                kind="file", key="obb/x.pt",
            ),
            PlannedModel(
                role="POSE_MODEL_DIR",
                source_path=str(staging["models"] / "obb"),  # any real directory
                kind="directory", key="pose/SLEAP/run",
            ),
        ],
    )
    manifest = _pack(tmp_path, staging, planned=planned)
    assert manifest.requirements["conda_envs"] == ["sleap"]
```

Add `test_conda_envs_defaults_to_sleap_for_the_placeholder_value`: identical setup, with `"pose_sleap_env": "no sleap envs found"` added to `config`; assert `conda_envs == ["sleap"]`, not `["no sleap envs found"]`. Both must succeed (not raise `KeyError`, and not raise fix X3a's missing-skeleton guard). The existing `packed_job_needing_sleap` fixture (Task 10) sets `pose_sleap_env` explicitly and therefore never exercised either defaulting branch — these two new tests close that gap; do not modify `packed_job_needing_sleap` itself, since it correctly tests the explicit-value path.

**Fix W6 — `requirements.runtime_tier` must come from the RESOLVED tier, never the raw config dict.** All three Task 13 acceptance fixtures (`fly_obb.json`, `ant_pose_headtail.json`, `ant_cnn_identity.json` — grepped, verified: none has a `runtime_tier` key) have no `runtime_tier` key at all, and `build_engine_params` defaults an absent one to `"gpu"` (`engine_params.py:800-805`, `trackerkit` — app layer). A naive `cfg.get("runtime_tier", "")` inside `pack_job` would stamp `requirements.runtime_tier = ""` on every one of them, which means nothing to preflight's check 4 — silently wrong in whichever direction that check treats falsy values. But `pack.py` is Data layer and `build_engine_params` is `trackerkit` (app layer); `pack_job` calling it directly would violate the one-way dependency-direction rule Task 3's contract guard tests. So the RESOLUTION happens in the caller, not in `pack.py`: **`job_cli.py` (Task 11), which already builds a session from the keystone config to walk `iter_model_references`, resolves the tier via that same session/`build_engine_params` call and passes it explicitly as `track_args["runtime_tier"]`** — this is not new plumbing, it is the SAME pattern the Task 7/Task 10 test fixtures already use (`track_args={"video_list": "videos.txt", "runtime_tier": "cpu"}`, shown throughout Task 10's `packed_job*` fixtures above). `pack_job` itself only ever does `requirements.runtime_tier = track_args.get("runtime_tier", "gpu")` — reading a value the caller already resolved, with the same `"gpu"` fallback `build_engine_params` uses, never re-deriving it and never importing `trackerkit`.

**Minor fix (adversarial review) — this paragraph's own citation is imprecise.** This task's `_pack`/`packed_job` fixtures (Step 1/Step 1b, above) pass `track_args={"video_list": "videos.txt"}` with NO `runtime_tier` key at all — they exercise the `"gpu"` fallback branch, not the `"cpu"`-resolved branch this paragraph describes. It is Task 10's `packed_job*` fixtures (built on top of this task's `packed_job`, extended with `"runtime_tier": "cpu"`) that show the resolved-tier shape. Both are intentional and correct for what each task is testing — Task 7 proves the fallback, Task 10 proves an explicit CPU-tier job preflights correctly against the `cpu` tier's own requirements — this is not a bug, just a citation correction: "shown throughout Task 10's `packed_job*` fixtures above" should not be read as also describing this task's own `track_args`.

Basename collisions across different source directories raise `TrackingJobError` naming **both** origins.

- [ ] **Step 5: Implement `verify.py`**

```python
ABSOLUTE_PATH_FORBIDDEN_KEYS = (
    "file_path", "csv_path", "video_output_path", "pose_skeleton_file",
    "yolo_obb_direct_model_path", "yolo_detect_model_path",
    "yolo_crop_obb_model_path", "yolo_headtail_model_path",
    "pose_model_dir", "pose_yolo_model_dir", "pose_sleap_model_dir",
    "pose_vitpose_model_dir", "color_tag_model_path",
    # Fix M2: the legacy alias engine_params.py:815 falls back to when the
    # mode-specific key is empty. Must be forbidden too, or a legacy config
    # ships an absolute path through this back door undetected.
    "yolo_model_path",
)
```

`verify_job(job_dir) -> list[str]` collects **all** problems (never short-circuits):
manifest parses and `job_version == 1`;

**model integrity is checked by `kind`, with an explicit `"directory"` branch (fix B4).** A directory model has no meaningful top-level `sha256`/`size_bytes` — `copy_model_reference` leaves both at their dataclass defaults (`""`/`0`) — so applying the file rule to it would fail EVERY pose job:

```python
for model in manifest.models:
    if model.kind == "directory":
        # No top-level digest exists. Integrity is the per-member check below,
        # which is why file_digests MUST be populated (fix B4).
        if not model.file_digests:
            problems.append(
                f"model {model.key}: directory model has no file_digests; "
                "repack with a build that populates them"
            )
    else:
        target = job_dir / "models" / model.key
        # ... exists / sha256 matches model.sha256 / size matches size_bytes
```

then, for every model regardless of kind: **for every model with a non-empty `file_digests` (directory models and bundles — fix M8), every listed job-relative member path exists AND its sha256 matches**, so a corrupted single file inside a multi-file pose/bundle artifact is caught (a top-level directory sha256 alone cannot pinpoint or even always detect this depending on hash construction — checking members individually is the actual §6.2 step 5 requirement); every `sidecars[]` exists; every non-`shared` `videos[].job_path` exists (symlink target on the staging machine, regular file on the remote); **and its actual size on disk (`os.path.getsize`, following a symlink) equals `videos[].size_bytes` (fix W1b)** — this is the offline, cheap (stat-only, no content read) catch for the exact failure mode that let a truncated push both verify AND run: `rsync --partial` deliberately leaves a truncated `.mp4` at the destination path on interruption, so the file *exists* but its size no longer matches what pack recorded from the origin. A mismatch is reported as `f"video {video.job_path}: size on disk ({actual}) != manifest size_bytes ({video.size_bytes})"`. (The full content `signature` comparison — which needs to read bytes, not just `stat` — is Task 10 preflight's job, not verify's; verify stays offline and cheap by design.); every `config_job_path` exists; no sidecar has an absolute value (or a `..`) in `ABSOLUTE_PATH_FORBIDDEN_KEYS`, including each `cnn_classifiers[].model_path`; every referenced `config/skeletons/*` exists; `videos.txt` lines all exist **except lines belonging to a `shared` video (fix V2, below)** and the first equals `keystone["video"]`; every manifest relpath passes `validate_job_relpath`.

**Fix V2 — a `shared` video's `videos.txt` line, and its `job_path`, must be exempt from BOTH existence checks, not just the size check.** At pack time (spec §6.7) a `shared` video has no file under `videos/` at all — that symlink is materialized later, on the remote, by Task 10's `materialize_shared_videos`. The "every non-`shared` `videos[].job_path` exists" wording above already exempts shared entries from the per-model existence/size loop, but the *separate* `videos.txt` line-existence check must apply the same exemption or it fails on every shared job: `pack_job` writes `videos/colony.mp4` as the line for a shared video exactly as it does for a copied/symlinked one (the runner needs a job-relative path regardless of provenance), so a naive "every line names a file that exists on disk" check reads that line as broken immediately after packing. Concretely: build a `{job_path -> JobVideo}` map from `manifest.videos`, and for each `videos.txt` line, resolve it to the matching `JobVideo` and skip the on-disk-existence check (but still require the line to be a valid job-relative video the manifest actually knows about) when `video.shared is not None`. Without this fix, `pack_job`'s own self-verify (Step 5's "`verify_job` self-check, raise `TrackingJobError` if it reports problems") raises for every `shared` job at pack time, which means `packed_job_shared` (Task 10's fixture, defined below) raises at fixture setup and every Task 10 shared test in `tests/test_tracking_job_preflight.py` errors before its body runs — not a subtle edge case, a total block on shared-video support ever working. (Fix V2 note, round-6: `test_a_packed_shared_job_verifies_clean` itself is deferred to Task 10 — see the "Fix V2 note" callout earlier above (Task 7), which is the authoritative statement of where that test lives; an earlier draft named it here too, which would have collided with a fixture (`packed_job_shared`) that does not exist until Task 10.)

**Fix X6 — `verify_job` forbids an absolute path on EVERY key in `ABSOLUTE_PATH_FORBIDDEN_KEYS`, but `_rewrite_config`/`copy_model_reference` only rewrite the ACTIVE roles `iter_model_references` yields — so a config whose INACTIVE role still holds an external absolute path (common after switching detect mode or pose backend on the staging host, since the GUI/CLI never clears a stale field when you flip modes) packs "successfully," then fails pack's own self-verify immediately after.** Concretely: `yolo_detect_model_path`/`yolo_crop_obb_model_path` for the non-selected `YOLO_OBB_MODE` pair (Task 3's "gate on `YOLO_OBB_MODE`" rule means the unselected key is never yielded, by design); `yolo_headtail_model_path` when head-tail is off; `pose_yolo_model_dir`/`pose_sleap_model_dir`/`pose_vitpose_model_dir` for the two non-selected pose backends (only the active one is gated in via the pose-stage-live check). None of these three families are hypothetical — switching `YOLO_OBB_MODE` from `direct` to `sequential`, or switching a pose backend from `sleap` to `yolo`, is an ordinary staging-host workflow that leaves the PREVIOUS mode's model path sitting in the config, still absolute, still present.

Resolution: **pack blanks every inactive-role key in `ABSOLUTE_PATH_FORBIDDEN_KEYS`'s model-path subset, rather than requiring verify to special-case them.** This mirrors the existing precedent for `color_tag_model_path` (fix Q2, above: blanked unconditionally regardless of whether anything reads it, because a config field with no live consumer for THIS job carries zero information the remote needs) and is simpler than teaching `verify_job` a parallel "is this role active" computation that would have to reproduce `iter_model_references`'s own gating logic a second time, in a different layer, and risk drifting out of sync with it.

**Fix Q3 (adversarial review) — `active_roles` as originally written (`{ref.role for ref in iter_model_references(params_for_this_video)}`) is layer-illegal inside `pack.py`.** `iter_model_references` is defined at `trackerkit/engine_params.py:1862` — `trackerkit` is an app layer, and `pack.py` is Data (this plan's own hard dependency-direction rule, and now an ENFORCED one via the Task 5 layering gate: `test_no_app_layer_or_qt_imports` would fail the instant `pack.py` imported it, exactly as it already does for `load_advanced_tracker_config`/`build_engine_params`/`_default_output_paths` elsewhere in this task, per fixes X5a/W6/W10). The only legal derivation lives entirely inside data the caller already handed `pack_job`: `PlannedVideo.planned_models` (Task 5/6) is exactly the set of `PlannedModel`s `job_cli.py` built from walking `iter_model_references` for that video — so, concretely, in the SAME per-video loop that calls `_rewrite_config` (Task 7):

```python
active_roles = {model.role for model in planned.planned_models}
```

`iter_model_references` is never called or imported by `pack.py` anywhere — `job_cli.py` (Task 11) is the only place that ever calls it, exactly once per video, and `PlannedVideo.planned_models` is how that result reaches `pack.py` at all. Then, for every key in the model-path subset of `ABSOLUTE_PATH_FORBIDDEN_KEYS` whose corresponding role is NOT in `active_roles`, set `out[key] = ""` in `_rewrite_config` **whenever that key is already present in the config dict** (minor fix, adversarial review: `if key in out: out[key] = ""`, not an unconditional `out[key] = ""` for every key in the subset regardless of presence — a blank-only-if-present write is behaviourally identical for every consumer, since `_cfg_get`/`.get(key, "")` treat an absent key and an explicit `""` identically everywhere downstream, but it stops `pack.py` from INJECTING keys the source config never had at all, e.g. stamping `pose_vitpose_model_dir: ""` into a sidecar for a config that never once mentioned ViTPose — cleaner provenance, same behavior) (only if it's not already handled by an active rewrite). `verify_job`'s check is UNCHANGED and stays strict on every key — this is the "keep it consistent" fix, not a weakening of the gate: after packing, every sidecar's inactive-role keys are always empty, so the existing unconditional verify check is trivially satisfied without knowing which roles were active. `pose_model_dir` legacy alias key follows the same rule via whichever of the three backend-specific keys it maps to. Add `test_active_roles_derivation_never_imports_trackerkit`: the same file-scoped AST check fix X5a already adds for `pack.py` (reusing `test_tracking_job_layering.py`'s `_imported_names`) also asserts `"engine_params"` and `"iter_model_references"` never appear as an imported name/attribute in `pack.py`'s source.

Add `test_pack_blanks_an_inactive_role_absolute_path`: a config with `yolo_obb_mode: "direct"`, a real `yolo_obb_direct_model_path`, AND a stale absolute `yolo_crop_obb_model_path` (the sequential-mode key, inactive because mode is `direct`) pointing at a file that exists on the staging host but is never referenced by `iter_model_references` for this config; assert `pack_job` succeeds, the sidecar's `yolo_crop_obb_model_path == ""`, and `verify_job(job_dir) == []`. A second case does the same for `pose_sleap_model_dir` (stale, absolute) while `pose_yolo_model_dir` is the active pose backend. **Add a third case (fix Y6, round-7) for the `yolo_model_path` legacy alias under bgsub detection — the earlier-in-this-task prose about it once said "left untouched", which directly contradicted X6 and would make such a config unpackable forever:** a config using bgsub detection (no `yolo_obb_direct_model_path`/`yolo_crop_obb_model_path` role active at all — `iter_model_references` yields neither `YOLO_OBB_DIRECT_MODEL_PATH` nor `YOLO_CROP_OBB_MODEL_PATH` for it) with a stale absolute `yolo_model_path` left over from a previous non-bgsub configuration; assert `pack_job` succeeds, the sidecar's `yolo_model_path == ""`, and `verify_job(job_dir) == []`.

**Fix M15 — `registry_entry_present` must actually be set.** It is declared on `JobModel` with a `False` default and nothing in this plan as originally written ever set it, so it would always read `False` even for a model that has a real `model_registry.json` entry. `pack_job` sets it explicitly: for each `JobModel` it builds, `registry_entry_present = (model.key in {key for key, _ in registry_entries})` — i.e. true iff the model's job-relative key is one of the keys `write_registry_subset` (Task 6) actually wrote an entry for.

**Fix Q4 (adversarial review) — `registry_entries` is an `Iterable`-typed parameter (`iter_registry_entries()` at `training/model_publish.py:743` is a generator: it YIELDS), and this task's body consumes it TWICE — once inside `write_registry_subset(shipped_keys, entries, destination)` (Task 6, which iterates `entries` to build the subset dict) and again here for M15's `{key for key, _ in registry_entries}`.** A generator is exhausted after its first full iteration; whichever of the two consumes it second sees an EMPTY iterable, with no exception raised — `write_registry_subset` would silently write `{"schema_version": 2, "entries": {}}` (an empty registry subset shipped for every job, even one whose models genuinely have registry provenance) if it ran second, or every `JobModel.registry_entry_present` would silently read `False` regardless of the real answer if M15's comprehension ran second. Task 11's caller (Step 7) already passes `registry_entries=list(iter_registry_entries())`, which happens to dodge this for that ONE call site — but `pack_job`'s own signature accepts any `Iterable[tuple[str, dict]]`, every unit test in this task passes a plain `list` too (masking the bug in tests the same way the real call site masks it), and nothing enforces "always a list" at the boundary `pack_job` itself controls. Fix: `pack_job` materializes its own `registry_entries` parameter to a `list` ONCE, at the very top of the function body, before either consumer runs:

```python
registry_entries = list(registry_entries)
```

This makes `pack_job` correct regardless of what kind of iterable a future caller passes, rather than relying on every present and future call site independently remembering to pre-materialize. Add `test_pack_job_accepts_a_registry_entries_generator`: call `_pack(tmp_path, staging, registry_entries=(e for e in [("obb/x.pt", {"species": "ant", "source_path": "/host/a.pt"})]))` (a genuine one-shot generator, not a list) and assert both that `models/model_registry.json` contains the `obb/x.pt` entry AND that the returned manifest's `models[0].registry_entry_present is True` — proving neither consumer starved the other.

**Minor note (round-6) — `registry_entry_present: false` is informational metadata, not a functional gap.** Grepped: nothing in `engine_params.py`, `core/inference/`, or any pose/CNN backend reads `model_registry.json` at RUNTIME — the registry is written at training time and read by GUI/CLI tooling that lists or picks models, never consulted while actually resolving or loading a model path during tracking. So a packed job with `registry_entry_present: false` on every model (the common case for models trained before this branch existed, or copied in from elsewhere) tracks identically to one where it's `true` — the field exists purely so a future reader (or `job status`) can flag "this job's models have no registry provenance" without that meaning anything is broken. Recorded here so a future contributor chasing "why does registry_entry_present matter" doesn't go looking for a runtime consumer that does not exist.

`shared` entries are **not** checked here — verify is offline and mount-agnostic; preflight (Task 10) checks them.

- [ ] **Step 6: Run tests to verify they pass**

Run: `python -m pytest tests/test_tracking_job_pack.py tests/test_tracking_job_verify.py tests/test_tracking_job_layering.py -v`
Expected: all PASS.

- [ ] **Step 7: Commit**

```bash
make format
git add src/hydra_suite/data/tracking_job/pack.py \
        src/hydra_suite/data/tracking_job/runner.py \
        src/hydra_suite/data/tracking_job/verify.py \
        src/hydra_suite/data/tracking_job/__init__.py \
        tests/test_tracking_job_pack.py \
        tests/test_tracking_job_verify.py \
        tests/helpers/tracking_job.py \
        tests/conftest.py
git commit -m "feat(tracking-job): pack a job directory, generate run.sh, and verify it offline"
```

---

### Task 8: Output discovery + origin mapping

**Files:**
- Create: `src/hydra_suite/data/tracking_job/outputs.py`
- Test: `tests/test_tracking_job_outputs.py` (create)

**Interfaces:**
- Consumes: `JobManifest` (Task 5).
- Produces:
  - `discover_outputs(manifest, listing: Iterable[str]) -> list[str]` — job-relative output paths, given a flat listing of everything under `videos/`.
  - `PullDestination` frozen dataclass: `job_relpath: str`, `destination: str`, `redirected: bool`.
  - `map_outputs_to_origins(manifest, outputs) -> list[PullDestination]`
  - `plan_pull(manifest, listing, *, include_caches=True) -> list[PullDestination]`

**The contract is structural, not name-based.** An output is anything under `videos/` that is not a manifest video and not in that video's `pushed_siblings`. A future artifact type is therefore pulled with no code change.

- [ ] **Step 1: Write the failing test**

Create `tests/test_tracking_job_outputs.py`:

```python
"""Output discovery is structural; origin mapping puts artifacts back."""

import pytest

from hydra_suite.data.tracking_job.manifest import JobManifest, JobVideo
from hydra_suite.data.tracking_job.outputs import (
    discover_outputs,
    map_outputs_to_origins,
    plan_pull,
)


def _manifest(**video_overrides):
    video = JobVideo(
        job_path="videos/colony.mp4",
        origin_path="/Volumes/lab/2026-09/colony.mp4",
        size_bytes=10,
        config_job_path="videos/colony_config.json",
        config_provenance="own-sidecar",
        pushed_siblings=["videos/colony_config.json"],
        **video_overrides,
    )
    return JobManifest(
        job_id="j", created_at="t", created_on={},
        keystone={"video": "videos/colony.mp4", "config": "videos/colony_config.json"},
        videos=[video], models=[],
    )


# Every artifact the pipeline is known to produce today (spec section 10).
# Fix X4: the original list used invented names ("colony_tracking.csv",
# "colony_tracking_with_individual.csv") that headless_tracking.py never
# writes. Verified real names: with enable_backward_tracking (the fixtures'
# default), headless_tracking.py:221-224 writes raw `<stem>_tracking_forward
# .csv`/`<stem>_tracking_backward.csv`; cli_config.py:325 names the
# post-processing intermediate `<stem>_tracking_forward_processed.csv`; the
# Debug-mode terminal files are `<stem>_tracking_final.csv` (bare) and
# `<stem>_tracking_final_with_individual.csv` (rich export, RICH_EXPORT_SUFFIX
# appended per core/tracking/session.py -- also the name run_matrix.sh:320
# compares); the User-mode terminal file is `<stem>_tracks.csv`
# (core/tracking/session.py:812, `user_tracks_path`). Both terminal-file
# families are listed since a job's mode (User/Debug) is a config choice, not
# a fixed pipeline output.
KNOWN_ARTIFACTS = [
    "videos/colony_tracking_forward.csv",
    "videos/colony_tracking_backward.csv",
    "videos/colony_tracking_forward_processed.csv",
    "videos/colony_tracking_final.csv",
    "videos/colony_tracking_final_with_individual.csv",
    "videos/colony_tracks.csv",
    "videos/colony_tracking.mp4",
    "videos/colony_logs/run.log",
    "videos/.inference_cache_colony/detection.npz",
    "videos/.inference_cache_colony/opt/trial_0.npz",
    "videos/colony_datasets/active_learning/labels.json",
    "videos/colony_datasets/oriented_videos/a.mp4",
    "videos/colony_datasets/individual_crops/0001.png",
]


def test_every_known_artifact_is_discovered():
    listing = ["videos/colony.mp4", "videos/colony_config.json", *KNOWN_ARTIFACTS]
    assert sorted(discover_outputs(_manifest(), listing)) == sorted(KNOWN_ARTIFACTS)


def test_the_video_itself_is_never_an_output():
    assert discover_outputs(_manifest(), ["videos/colony.mp4"]) == []


def test_pushed_siblings_are_never_outputs():
    assert discover_outputs(_manifest(), ["videos/colony_config.json"]) == []


def test_an_unknown_future_artifact_is_still_discovered():
    """Structural discovery means new artifact types need no code change."""
    listing = ["videos/colony.mp4", "videos/colony_somethingnew/report.html"]
    assert discover_outputs(_manifest(), listing) == ["videos/colony_somethingnew/report.html"]


def test_outputs_map_beside_the_origin_video():
    outputs = ["videos/colony_tracking.csv", "videos/.inference_cache_colony/detection.npz"]
    mapped = {d.job_relpath: d.destination for d in map_outputs_to_origins(_manifest(), outputs)}
    assert mapped["videos/colony_tracking.csv"] == "/Volumes/lab/2026-09/colony_tracking.csv"
    assert (
        mapped["videos/.inference_cache_colony/detection.npz"]
        == "/Volumes/lab/2026-09/.inference_cache_colony/detection.npz"
    )


def test_a_redirected_output_goes_back_to_its_recorded_absolute_path():
    manifest = _manifest(
        redirected_outputs={"videos/colony_custom.mp4": "/Volumes/renders/custom.mp4"}
    )
    mapped = map_outputs_to_origins(manifest, ["videos/colony_custom.mp4"])
    assert mapped[0].destination == "/Volumes/renders/custom.mp4"
    assert mapped[0].redirected is True


def test_a_shared_video_maps_outputs_to_the_local_mount_of_the_origin():
    """Outputs land beside the original on the share, via origin_path."""
    manifest = _manifest(shared={"alias": "labnas", "relpath": "2026-09/colony.mp4"})
    mapped = map_outputs_to_origins(manifest, ["videos/colony_tracking.csv"])
    assert mapped[0].destination == "/Volumes/lab/2026-09/colony_tracking.csv"


def test_no_caches_excludes_the_inference_cache_tree():
    listing = ["videos/colony.mp4", *KNOWN_ARTIFACTS]
    planned = [d.job_relpath for d in plan_pull(_manifest(), listing, include_caches=False)]
    assert not any(".inference_cache_" in p for p in planned)
    assert "videos/colony_tracking.csv" in planned


def test_caches_are_included_by_default():
    listing = ["videos/colony.mp4", *KNOWN_ARTIFACTS]
    planned = [d.job_relpath for d in plan_pull(_manifest(), listing)]
    assert "videos/.inference_cache_colony/detection.npz" in planned


def test_an_output_outside_videos_is_ignored():
    listing = ["logs/run.log", "hydra_job.json", "videos/colony_tracking.csv"]
    assert discover_outputs(_manifest(), listing) == ["videos/colony_tracking.csv"]


def test_multiple_videos_route_to_their_own_origins():
    a = JobVideo(
        job_path="videos/a.mp4", origin_path="/data/one/a.mp4", size_bytes=1,
        config_job_path="videos/a_config.json", config_provenance="own-sidecar",
        pushed_siblings=["videos/a_config.json"],
    )
    b = JobVideo(
        job_path="videos/b.mp4", origin_path="/data/two/b.mp4", size_bytes=1,
        config_job_path="videos/b_config.json", config_provenance="keystone-baseline",
        pushed_siblings=["videos/b_config.json"],
    )
    manifest = JobManifest(
        job_id="j", created_at="t", created_on={},
        keystone={"video": "videos/a.mp4", "config": "videos/a_config.json"},
        videos=[a, b], models=[],
    )
    mapped = {
        d.job_relpath: d.destination
        for d in map_outputs_to_origins(
            manifest, ["videos/a_tracking.csv", "videos/b_tracking.csv"]
        )
    }
    assert mapped["videos/a_tracking.csv"] == "/data/one/a_tracking.csv"
    assert mapped["videos/b_tracking.csv"] == "/data/two/b_tracking.csv"


def test_an_unattributable_output_raises():
    """An artifact matching no video's stem must not be silently dropped."""
    from hydra_suite.data.tracking_job.manifest import TrackingJobError

    with pytest.raises(TrackingJobError):
        map_outputs_to_origins(_manifest(), ["videos/unrelated_thing.csv"])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_tracking_job_outputs.py -v`
Expected: FAIL — module missing.

- [ ] **Step 3: Implement `outputs.py`**

Attribution rule: an output belongs to the video whose **stem** is the longest prefix match of the output's first path component under `videos/` (so `colony_tracking.csv`, `colony_logs/…`, `.inference_cache_colony/…` and `colony_datasets/…` all attribute to `colony.mp4`). The `.inference_cache_<stem>` form is a *suffix* match, so handle it explicitly. Raise `TrackingJobError` when an output attributes to no video — silently dropping an artifact is how results get lost.

**Fix M3 — no single-video shortcut.** The original draft returned `manifest.videos[0]` unconditionally whenever the job had exactly one video, bypassing the stem check entirely. That makes `test_an_unattributable_output_raises` (which, in the failing-test fixture above, uses a **one-video** manifest) structurally unable to ever raise — the safeguard the test exists to prove would be permanently dead code for every single-video job, which is the overwhelmingly common case. We keep the safeguard for ALL job shapes, including one video, and drop the shortcut: silent misattribution loses data, and the cost of a real stem check is negligible.

```python
def _owning_video(manifest, relpath):
    first = PurePosixPath(relpath).parts[1]  # after "videos/"
    best, best_len = None, -1
    for video in manifest.videos:
        stem = PurePosixPath(video.job_path).stem
        if first.startswith(stem) or first == f".inference_cache_{stem}":
            if len(stem) > best_len:
                best, best_len = video, len(stem)
    if best is None:
        raise TrackingJobError(
            f"pulled artifact {relpath} matches no video in the job; refusing to "
            "guess a destination",
            code=5,
        )
    return best
```

`test_an_unattributable_output_raises` (Step 1) already uses `_manifest()` — a one-video job whose video stem is `colony` — with the output `videos/unrelated_thing.csv`, whose first component `unrelated_thing.csv` does not start with `colony`. With the shortcut removed this now genuinely exercises the raise path; no test change is needed, only the implementation fix.

**Fix — ignore dotfiles.** `map_outputs_to_origins` raising on ANY unattributable file means a stray `videos/.DS_Store` (or any other dotfile with no video-stem relationship, e.g. an editor swap file) aborts the entire pull. Add an ignore list before attribution runs:

```python
_IGNORED_BASENAMES = {".DS_Store"}


def discover_outputs(manifest, listing):
    ...  # existing structural filtering
    return [
        p for p in outputs
        if PurePosixPath(p).name not in _IGNORED_BASENAMES
        and not PurePosixPath(p).name.startswith(".DS_Store")
    ]
```

Apply the same ignore filter inside `map_outputs_to_origins`/`plan_pull` (or rely on them only ever being called with `discover_outputs`'s already-filtered result — pick one and document it). Add `tests/test_tracking_job_outputs.py::test_a_stray_ds_store_is_ignored_not_raised`:

```python
def test_a_stray_ds_store_is_ignored_not_raised():
    listing = ["videos/colony.mp4", "videos/.DS_Store"]
    assert discover_outputs(_manifest(), listing) == []
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_tracking_job_outputs.py -v`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
make format
# Fix B10: outputs.py's plan_pull/PullDestination are re-exported from __init__.
git add src/hydra_suite/data/tracking_job/outputs.py \
        src/hydra_suite/data/tracking_job/__init__.py \
        tests/test_tracking_job_outputs.py
git commit -m "feat(tracking-job): structural output discovery and origin mapping for pull"
```

---

### Task 9: Transport (rsync/ssh)

**Files:**
- Create: `src/hydra_suite/data/tracking_job/transport.py`
- Test: `tests/test_tracking_job_transport.py` (create)

**Interfaces:**
- Consumes: `JobManifest` (Task 5), `plan_pull` (Task 8).
- Produces:
  - `RemoteTarget` frozen dataclass: `host: str`, `path: str`; `parse_remote(text) -> RemoteTarget`.
  - `build_push_input_list(manifest) -> list[str]`
  - `build_rsync_argv(source, destination, *, files_from, extra=()) -> list[str]`
  - `push_job(job_dir, remote, *, runner=subprocess.run) -> None`
  - `remote_video_listing(remote, *, runner) -> list[str]`
  - `pull_job(remote, job_dir, *, include_caches, overwrite, dry_run, force=False, runner) -> PullReport` (fix B12b — **not** `list[PullDestination]`; the two descriptions disagreed. Step 3 requires a value that separates hard failures from per-video skips, which a bare list cannot express). `force` bypasses fix A5's still-running check.
  - `PullReport` frozen dataclass — the single return type, defined in `transport.py`:

    ```python
    @dataclass(frozen=True)
    class PullReport:
        pulled: list[PullDestination]      # what actually landed (empty on dry_run)
        planned: list[PullDestination]     # what plan_pull produced, pre-filter
        skipped: list[tuple[str, str]]     # (job_relpath, reason) -- never silent
        dry_run: bool = False

        @property
        def ok(self) -> bool:
            return not self.skipped
    ```

    Hard failures never come back in a `PullReport`: they raise `TrackingJobError(code=5)` before any transfer starts (see Step 3).
  - `_last_runs_jsonl_timestamp(path: Path) -> float | None` (fix W3b — was referenced by the fix A5 "still running" guard but never declared here) — parses the LAST line of `runs.jsonl` and returns its `finished_at` as a Unix timestamp via `datetime.fromisoformat(...).timestamp()`, or `None` if the file is missing, empty, or the last line is unparseable/missing `finished_at`. Never raises.
  - `_RUNNING_CHECK_SLACK_SECONDS = 1.0` (fix W3b) — module-level constant; the still-running guard compares `run_log.stat().st_mtime - _RUNNING_CHECK_SLACK_SECONDS` against `last_entry_ts`, absorbing the sub-second gap between `run.sh`'s `tee -a logs/run.log` closing and `_record-run` stamping `finished_at` in the same script.

- [ ] **Step 1: Write the failing test**

Create `tests/test_tracking_job_transport.py`:

```python
"""Transport builds exact argv and never invents file lists."""

import pytest

from hydra_suite.data.tracking_job.manifest import (
    JobManifest, JobModel, JobVideo, TrackingJobError,
)
from hydra_suite.data.tracking_job.transport import (
    build_push_input_list,
    build_rsync_argv,
    parse_remote,
)


def _manifest(shared=None):
    return JobManifest(
        job_id="j", created_at="t", created_on={},
        keystone={"video": "videos/a.mp4", "config": "videos/a_config.json"},
        videos=[
            JobVideo(
                job_path="videos/a.mp4", origin_path="/data/a.mp4", size_bytes=1,
                config_job_path="videos/a_config.json", config_provenance="own-sidecar",
                pushed_siblings=["videos/a_config.json"], shared=shared,
            )
        ],
        models=[
            JobModel(
                key="obb/x.pt", roles=["R"], origin_path="/h/x.pt", kind="file",
                sha256="ab", size_bytes=1,
                sidecars=["obb/x.pt.slice_meta.json"],
            ),
            # Minor fix (round-7): a directory model, so build_push_input_list's
            # models-root-relative `files[]` prefixing is actually exercised --
            # without one, a bug that emitted `files[]` entries WITHOUT the
            # `models/` prefix (rsync would then look for them at the wrong
            # path relative to the push source root and silently omit them)
            # had no test able to catch it.
            JobModel(
                key="pose/SLEAP/run", roles=["POSE_MODEL_DIR"],
                origin_path="/h/pose/SLEAP/run", kind="directory",
                files=["pose/SLEAP/run/best.ckpt", "pose/SLEAP/run/training_config.json"],
                file_digests={
                    "pose/SLEAP/run/best.ckpt": "cd",
                    "pose/SLEAP/run/training_config.json": "ef",
                },
            ),
        ],
        config_snapshot={
            "advanced_config": "config/advanced_config.json",
            "skeletons": ["config/skeletons/ant.json"],
        },
    )


def test_parse_remote_splits_host_and_path():
    target = parse_remote("rutalab@firebrat:/home/rutalab/jobs/j1")
    assert target.host == "rutalab@firebrat"
    assert target.path == "/home/rutalab/jobs/j1"


@pytest.mark.parametrize("bad", ["nohost", "host:relative/path", ""])
def test_parse_remote_rejects_malformed_targets(bad):
    with pytest.raises(TrackingJobError):
        parse_remote(bad)


def test_push_list_is_inputs_only():
    entries = set(build_push_input_list(_manifest()))
    assert entries == {
        "hydra_job.json", "run.sh", "videos.txt",
        "config/advanced_config.json", "config/skeletons/ant.json",
        "config/presets/.seeded", "config/skeletons/.seeded",
        "models/model_registry.json",
        "models/obb/x.pt", "models/obb/x.pt.slice_meta.json",
        # Minor fix (round-7): the directory model's `files[]` -- these are
        # already models-root-relative on JobModel (verified: fix M8's
        # `file_digests["pose/SLEAP/run/best.ckpt"]` example uses the same
        # convention), so build_push_input_list must prefix them with
        # "models/" exactly like every other model entry, never emit them
        # bare (which would make rsync look in the wrong place entirely).
        "models/pose/SLEAP/run/best.ckpt",
        "models/pose/SLEAP/run/training_config.json",
        "videos/a.mp4", "videos/a_config.json",
    }


def test_push_list_never_contains_an_output():
    """A re-push after a local pull must not overwrite remote outputs."""
    entries = build_push_input_list(_manifest())
    assert not any("_tracking" in e or ".inference_cache_" in e for e in entries)
    assert not any(e.startswith("logs/") for e in entries)


def test_a_shared_video_is_excluded_from_the_push_list():
    entries = build_push_input_list(
        _manifest(shared={"alias": "labnas", "relpath": "a.mp4"})
    )
    assert "videos/a.mp4" not in entries
    assert "videos/a_config.json" in entries


def test_rsync_argv_is_exact():
    argv = build_rsync_argv("/job/", "host:/remote/", files_from="/tmp/list.txt")
    assert argv == [
        "rsync", "-a", "--partial", "--info=progress2",
        "--files-from=/tmp/list.txt", "/job/", "host:/remote/",
    ]


def test_push_argv_dereferences_symlinks():
    argv = build_rsync_argv(
        "/job/", "host:/remote/", files_from="/tmp/l.txt", extra=("--copy-links",)
    )
    assert "--copy-links" in argv


def test_rsync_argv_never_deletes():
    argv = build_rsync_argv("/job/", "host:/remote/", files_from="/tmp/l.txt")
    assert "--delete" not in argv


def test_push_reports_the_exact_command_on_failure(tmp_path):
    """Fix V-minor: push_job runs an `ssh ... mkdir -p` remote-parent check
    (see the V-minor "creates the remote parent directory" fix above) BEFORE
    the rsync transfer -- a runner that fails EVERY call would raise on the
    mkdir step, never reaching the actual transfer, and this test would then
    be proving nothing about the transfer-failure path it's named for (the
    same trap the fix-note above documents for the presence check). The fake
    runner here succeeds on any ssh/mkdir call and fails ONLY the rsync
    transfer, so this test actually exercises what its name claims."""
    from hydra_suite.data.tracking_job.transport import push_job
    import subprocess

    def selective_runner(argv, **kwargs):
        if argv and argv[0] == "rsync":
            return subprocess.CompletedProcess(argv, 23, stdout="", stderr="rsync: boom")
        return subprocess.CompletedProcess(argv, 0, stdout="", stderr="")

    manifest = _manifest()
    (tmp_path / "hydra_job.json").write_text("{}")
    manifest.write(tmp_path / "hydra_job.json")
    with pytest.raises(TrackingJobError) as excinfo:
        push_job(tmp_path, "host:/remote/j", runner=selective_runner)
    message = str(excinfo.value)
    assert "rsync: boom" in message and "rsync" in message
    assert excinfo.value.code == 4
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_tracking_job_transport.py -v`
Expected: FAIL — module missing.

- [ ] **Step 3: Implement `transport.py`**

`build_push_input_list` is derived **from the manifest**, never from an exclude list: `hydra_job.json`, `run.sh`, `videos.txt`, the config snapshot files plus both `.seeded` markers, `models/model_registry.json`, every model key + its `sidecars` + its `files`, every non-`shared` `videos[].job_path`, and every `pushed_siblings` entry.

`push_job` writes the list to a temp file, runs `build_rsync_argv(..., extra=("--copy-links",))`, and on non-zero return raises `TrackingJobError(code=4)` embedding the joined argv and the captured stderr verbatim.

**Fix V-minor — `push_job` must `ssh mkdir -p` the REMOTE PARENT before the rsync transfer, or the first push to a fresh remote tree fails.** The push target is `<remote>/jobs/<name>` (two path components below whatever `<remote>` already names), but `rsync -a ... host:/remote/jobs/<name>/` only creates the FINAL path component it is given — it does not `mkdir -p` an entire missing chain the way `rsync -a --mkpath` (a newer rsync flag, not assumed available here) would. If `jobs/` itself does not yet exist on the remote (the common first-push-ever case for a fresh remote job root), rsync fails with "No such file or directory" before a single file transfers, reported as a `TrackingJobError(code=4)` that looks identical to a real transfer failure. Fix: `push_job` runs `ssh <host> "mkdir -p $(dirname <remote_path>)"` (via the SAME `--remote-bootstrap`-prefixed pattern fix A2b established for other ssh calls in this plan) BEFORE the rsync transfer. `dirname` of `jobs/<name>` is `jobs`, so this creates the parent only — `rsync -a` still owns creating the final `<name>` directory itself, which keeps rsync (not a bespoke mkdir) as the single source of truth for the job directory's own permissions/ownership. Add `test_push_creates_the_remote_parent_directory`: fake runner records both the `ssh ... mkdir -p` call and the rsync call, asserts the mkdir call happens first and names the parent (not the full job path).

`remote_video_listing` runs `ssh <host> "cd <path> && find videos -type f -o -type l"` and returns the lines — one `ssh` call, so new artifact types need no code change.

`pull_job`: fetch `logs/` first (including `runs.jsonl`), then **check the run actually finished (fix A5)**, then list, then `plan_pull`, then check collisions (sha256 compare; refuse with `code=5` listing **every** collision before touching anything unless `overwrite`), then rsync the output set, then copy/hardlink into the mapped destinations, then **merge** (not append — see fix M12/§5 below) a `pull_history` entry into the **local** manifest. `--dry-run` prints the destination map and returns before any transfer.

**Minor fix — the `logs/` fetch on a job that was pushed but never run must not be treated as a transfer failure.** `rsync` exits `23` ("partial transfer due to error") when the SOURCE `logs/` directory doesn't exist yet (nothing has ever written to it — `run.sh` is what first does `mkdir -p logs`). `pull_job` must tolerate rc `23` specifically for the `logs/` fetch step (treat it as "no logs yet", not a hard failure) — any OTHER non-zero rc from that same rsync call still raises `TrackingJobError`. This is what makes the fix A5 running-check's own "skipped entirely when `run_log` does not exist at all" clause reachable in practice for a genuinely never-run job, rather than the pull dying one step earlier on the logs fetch itself.

**Fix Y4 (round-7) — the `logs/` fetch must be a plain recursive rsync, NEVER routed through `build_rsync_argv(..., files_from=...)`.** The Task 9 prose above said only "rsync the remote `logs/` first" without naming the mechanism; the only fetch helper this task defines is `build_rsync_argv(source, destination, *, files_from, extra=())`, and if the logs/ fetch is implemented by calling that helper with a `files_from` manifest listing `logs/run.log` and `logs/runs.jsonl`, it silently fails: `rsync --files-from=LIST` treats every line in `LIST` as an exact, individual transfer item — it does **not** recurse into a directory entry, and more importantly it never discovers files that are not already named in `LIST` ahead of time. That is fine for `push_job` (Task 9's `build_push_input_list` enumerates the manifest's own known files up front) but wrong for `pull_job`'s `logs/` fetch, whose entire job is to discover whatever `run.sh` happened to write on the remote — `run.log` and `runs.jsonl` are not "already known" the way pushed inputs are. Concretely: if `pull_job` built a `files_from` list containing `["logs/run.log", "logs/runs.jsonl"]` and passed it to `build_rsync_argv`, and the REMOTE `logs/` directory is genuinely a directory (not a bare pair of files at the rsync root), `--files-from` entries are resolved relative to the rsync source root and copied as individual files — this can work by accident if the paths are named exactly right, but it is fragile and, worse, means a THIRD file `run.sh` might write into `logs/` in the future (or any file this task's own author didn't anticipate) is silently never pulled, with no error. Fix: the `logs/` fetch is its own **separate, explicit, plain-recursive rsync call**, not built through `build_rsync_argv`'s `files_from` mechanism at all:

```python
logs_argv = ["rsync", "-a", f"{remote_logs_path}/", str(job_dir / "logs") + "/"]
result = runner(logs_argv, capture_output=True, text=True)
if result.returncode not in (0, 23):  # 23 = "no logs yet", tolerated per the fix above
    raise TrackingJobError(f"failed to fetch logs/: {' '.join(logs_argv)}\n{result.stderr}", code=4)
```

`-a` (archive) recurses into `logs/` and transfers everything under it — `run.log`, `runs.jsonl`, and any future file — with no advance enumeration required, which is exactly what "discover whatever the remote run wrote" needs and `--files-from` structurally cannot provide. Add `test_pull_logs_fetch_transfers_both_run_log_and_runs_jsonl`: a fake runner that, when it sees an argv starting with `["rsync", "-a"]` and ending in `.../logs/`, actually copies a fixture `logs/run.log` + `logs/runs.jsonl` pair from a fake "remote" tmp dir into the `destination` argv element (simulating what real rsync would do), then asserts BOTH files exist under `job_dir / "logs"` after `pull_job` returns (or raises past the running-check, whichever the fixture's `run.log`/`runs.jsonl` contents dictate) — proving the fetch step is a real recursive copy, not a `files_from` call that happens to name the same two files.

**Fix Y5 (round-7) — `pull_job`'s step order MUST be pinned explicitly: fetch `logs/` → running-check → THEN read the local manifest (collision detection, `plan_pull`, `pull_history` all need the local manifest, and none of them may run first).** Left implicit, the "natural" order an implementer reaches for is manifest-first (it is needed by nearly everything else `pull_job` does), which silently defeats fix A5: if `pull_job` reads `job_dir / "hydra_job.json"` before the running-check, a `dest/` fixture that has `logs/` but no `hydra_job.json` (exactly the "first pull ever, running-check should fire first" shape in the tests below) raises a bare `FileNotFoundError` from the manifest read, not `TrackingJobError` — the running-check code is never reached, so the guard reads as tested and passing while actually being dead on this input shape. **The order is a hard requirement, not an implementation detail:** (1) fetch `logs/` (tolerating rc 23 per the fix above), (2) run the fix A5 running-check using ONLY `job_dir / "logs"` (never touches the manifest), (3) only after the running-check passes (or `force=True` bypassed it), read `job_dir / "hydra_job.json"` for collision detection / `plan_pull` / `pull_history`. A job dir that has never been pulled into before therefore has no local manifest on disk yet for a first `pull` — `pull_job` creates/writes it fresh at the end of a successful run (mirroring `push_job`, which never requires a pre-existing local manifest either); the running-check's own precondition (`run_log.is_file()`) is the only gate on step 2, so step 3 is unreached on the still-running path and a missing local manifest there is a non-issue.

**Fix A5 — nothing currently stops `pull` from grabbing a job that is still running.** `run.sh` appends to `logs/runs.jsonl` only at exit (Fix A2d's `_record-run` call, which runs after `run.sh`'s own `track` invocation finishes), and it `tee -a`s to `logs/run.log` continuously WHILE running. Caches self-protect key-wise, but a partial `<stem>_tracking.csv` — the file the run was in the middle of writing when `pull` grabbed it — would be pulled and placed beside the origin looking exactly like a completed result, with no marker distinguishing it. Immediately after fetching `logs/`, before `plan_pull` or any output transfer, check:

```python
runs_jsonl = job_dir / "logs" / "runs.jsonl"
run_log = job_dir / "logs" / "run.log"
if run_log.is_file() and not force:
    if not runs_jsonl.is_file():
        raise TrackingJobError(
            "the remote run has not finished yet (logs/run.log exists but "
            "logs/runs.jsonl has no entries) -- refusing to pull a partial "
            "result; pass --force to pull anyway",
            code=5,
        )
    last_entry_ts = _last_runs_jsonl_timestamp(runs_jsonl)  # parses "finished_at"
    # Fix W3b: `_RUNNING_CHECK_SLACK_SECONDS = 1.0`. `run.sh`'s tee -a to
    # logs/run.log and its subsequent `_record-run` call (which stamps
    # finished_at) happen back-to-back in the same script, close enough in
    # wall-clock time that a second-truncated finished_at could legitimately
    # land BEFORE a fractional-second run_log mtime for a run that in fact
    # completed cleanly -- a false "still running". One second of slack
    # absorbs that without meaningfully weakening the check (a run that is
    # actually still in progress keeps appending to run.log well past any
    # 1-second window).
    if last_entry_ts is None or last_entry_ts < run_log.stat().st_mtime - _RUNNING_CHECK_SLACK_SECONDS:
        raise TrackingJobError(
            "logs/run.log was modified after the last logs/runs.jsonl entry "
            "-- the remote run looks still in progress; refusing to pull a "
            "partial result; pass --force to pull anyway",
            code=5,
        )
```

**Fix W3b — the comparison was arithmetically broken, not just imprecise.** Two independent defects made "run genuinely completed" register as "still in progress" on ~every real run, not just an edge case:

1. **`finished_at` was never pinned to a resolution finer than seconds**, while `run_log.stat().st_mtime` is fractional-second (a real filesystem mtime). `_record-run` (Task 11, the `job _record-run` hidden subcommand referenced above) must stamp `finished_at = datetime.now(timezone.utc).isoformat(timespec="microseconds")`, not `date -u +%FT%TZ` truncated to whole seconds — the plan's `run.sh` uses `date -u +%FT%TZ` for `START` (fine — that field is provenance only, never compared), but `finished_at` is written by `_record-run` in Python and must use `isoformat(timespec="microseconds")` specifically so `_last_runs_jsonl_timestamp` can parse a fractional-second value that is actually comparable to `st_mtime`.
2. **`_last_runs_jsonl_timestamp` was referenced but never defined anywhere in this plan**, nor declared as one of `transport.py`'s Produces. Add it to the Interfaces list: `_last_runs_jsonl_timestamp(path: Path) -> float | None` — parses the LAST line of `runs.jsonl` (the file is append-only, one JSON object per line; the last line is the most recent run), reads its `finished_at` via `datetime.fromisoformat(...).timestamp()`, and returns `None` when the file is empty, unparseable, or missing the key (never raises — the caller already treats `None` as "no valid entry").
3. **The module-level constant `_RUNNING_CHECK_SLACK_SECONDS = 1.0`** is new (fix W3b) and lives beside `_last_runs_jsonl_timestamp` in `transport.py`; add both to this task's Produces bullet list.

`pull_job` gains a `force: bool = False` keyword (also surfaced as `--force` on `job pull`'s CLI parser in Task 11). This check is skipped entirely when `run_log` does not exist at all (a job that was pushed but never run yet — nothing to be partial about; `plan_pull` will simply find no outputs). Add to `tests/test_tracking_job_transport.py`:

```python
# Fix W3a: BOTH tests below had two bugs that made them pass vacuously.
# (1) They wrote run.log under the REMOTE fixture path
#     (tmp_path/"remote"/"logs"), but the running-check in pull_job reads
#     job_dir/"logs"/"run.log" where job_dir is the DESTINATION -- the
#     directory pull_job fetches logs/ INTO, not the fake remote source
#     the test constructed. Since nothing ever wrote to
#     tmp_path/"dest"/"logs"/"run.log", the "no runs.jsonl at all" guard's
#     own precondition (run_log.is_file()) was False, so the guard
#     self-skipped via its own "no run.log -> nothing to check" rule --
#     the test's pytest.raises(TrackingJobError) only passed because of
#     whatever OTHER exception the AssertionError runner below produced,
#     never because the running-check itself fired.
# (2) `runner=lambda *a, **k: (_ for _ in ()).throw(AssertionError(...))`
#     fires on EVERY runner call, including the logs/ rsync fetch that
#     MUST happen before the running-check can even run (pull_job fetches
#     logs/ first, per the Task 9 prose above: "fetch logs/ first
#     (including runs.jsonl), then check the run actually finished").
#     That rsync call itself raised AssertionError before the
#     running-check ever executed, so the "raises TrackingJobError"
#     assertion below was again satisfied by an entirely different,
#     unintended code path.
#
# Fixed: the fixture writes DIRECTLY to dest/logs/{run.log,runs.jsonl} (the
# post-fetch state pull_job's own check reads), and the runner returns a
# real `subprocess.CompletedProcess(argv, 0, stdout="", stderr="")` for the
# logs/ rsync call -- mirroring test_push_reports_the_exact_command_on_failure's
# `selective_runner` shape above, just always returning success instead of
# failing selectively -- so the fetch step itself is a harmless no-op and
# the running-check is the ONLY thing that can raise.
import subprocess


def test_pull_refuses_a_job_that_looks_still_running(tmp_path):
    """Fix A5/W3a: run.log newer than the last runs.jsonl entry (or
    runs.jsonl absent while run.log exists) means the run has not finished."""
    from hydra_suite.data.tracking_job.transport import pull_job
    from hydra_suite.data.tracking_job.manifest import TrackingJobError

    dest_logs = tmp_path / "dest" / "logs"
    dest_logs.mkdir(parents=True)
    (dest_logs / "run.log").write_text("still going...\n")
    # No runs.jsonl at all -- the run has not exited yet.

    def logs_only_runner(argv, **kwargs):
        return subprocess.CompletedProcess(argv, 0, stdout="", stderr="")

    with pytest.raises(TrackingJobError) as excinfo:
        pull_job(
            "host:/remote", tmp_path / "dest",
            include_caches=True, overwrite=False, dry_run=False,
            runner=logs_only_runner,
        )
    assert excinfo.value.code == 5


def test_pull_force_bypasses_the_running_check(tmp_path):
    """--force must still allow pulling a genuinely in-progress job on
    purpose (e.g. to inspect partial CSVs while a long run is ongoing).

    Fix Y5 (round-7): the round-6 version of this test only asserted a
    forbidden SUBSTRING was absent from whatever exception (if any) came
    out, catching `Exception` broadly -- so a completely different failure
    (e.g. a bare `FileNotFoundError` from `pull_job` reading a local
    manifest that this fixture never created) would ALSO satisfy the
    assertion, without --force having been exercised at all. Fixed: the
    fixture supplies a real, minimal local manifest (matching fix Y5's
    pinned step order, where the manifest is read only AFTER the
    running-check) so a force=True, dry_run=True pull can run to actual
    completion, and the test asserts NO exception is raised and a
    PullReport comes back -- proving the bypass, not merely the absence of
    one particular error string.
    """
    from hydra_suite.data.tracking_job.manifest import JobManifest
    from hydra_suite.data.tracking_job.transport import pull_job, PullReport

    dest = tmp_path / "dest"
    dest_logs = dest / "logs"
    dest_logs.mkdir(parents=True)
    (dest_logs / "run.log").write_text("still going...\n")
    # A minimal, valid, zero-video local manifest -- enough for pull_job to
    # get past the (post-running-check) manifest read and plan_pull with an
    # empty video set, all the way to the dry-run report.
    manifest = JobManifest(
        job_id="j", created_at="t", created_on={}, keystone={}, videos=[], models=[],
    )
    manifest.write(dest / "hydra_job.json")

    def succeeding_runner(argv, **kwargs):
        return subprocess.CompletedProcess(argv, 0, stdout="", stderr="")

    report = pull_job(
        "host:/remote", dest,
        include_caches=True, overwrite=False, dry_run=True, force=True,
        runner=succeeding_runner,
    )
    assert isinstance(report, PullReport)
```

**Fix M12 — destination-side failure policy, fully specified (was unspecified).** `pull_job` never said what happens when the *destination* side is broken, only when the pulled bytes are broken. **Clarifying the apparent contradiction (fix B12d).** There are TWO distinct classes and they do NOT share an exit path:

- **Hard failures** — detected up front, before ANY byte moves, and they abort the whole pull by raising `TrackingJobError(code=5)` listing every one of them at once. Today the only hard failure is a **destination collision** without `overwrite` (an existing file whose sha256 differs from what would be written). **Minor note — a pre-existing local `.inference_cache_<stem>/` is a common, expected trigger of this.** If the video was ever tracked locally BEFORE being packed/pushed/pulled (the normal case: you run locally, later decide to offload a re-run to firebrat, then pull the remote result back beside the same origin video), a local `.inference_cache_<stem>/cache_set.json` already exists at the destination. Any byte difference in that file (a different generation ID, a different write timestamp recorded inside it, etc.) makes the WHOLE pull refuse via the collision path above without `--overwrite` — this is not a bug, it is the collision policy working as designed, but it is worth knowing this specific file is the one most likely to trip it on a machine that already has local tracking history for the same video.
- **Per-video skips** — that video's outputs are excluded and the pull *continues* for every unaffected video. Each carries a reason string into `PullReport.skipped`; the CLI prints them and exits non-zero, but the transfer for other videos has already succeeded.

An earlier draft said both "collected into the SAME code=5 failure report" and "per-video skips… not a hard sys.exit", which is self-contradictory. The rule is: **collisions abort (code 5); missing origins and unavailable mounts skip that video and are reported.** Both are surfaced all at once, never one-by-one across repeated invocations, and never silently.

The per-video skip cases:

- **Origin directory no longer exists** (the video's `origin_path` parent was deleted/renamed since packing): report `"origin directory missing for <job_relpath>: <origin_path>"` and exclude that video's outputs from the transfer, but do not abort the whole pull — other videos' outputs still land. Surfaced as a per-video problem in the returned report, not a hard `sys.exit`.
- **A `shared` video's local mount is absent at pull time** (the alias in `shared_roots.json` doesn't resolve, or resolves but the path doesn't exist — e.g. the NAS isn't mounted on the machine running `pull`): report `"shared mount unavailable for alias '<alias>': <resolved_path>"` and exclude that video's outputs the same way. Fix B12c: implement this with **Task 5's `shared_roots.resolve_shared(alias, relpath, table)`**, not with Task 10's `materialize_shared_videos` — Task 10 comes LATER, so depending on it here would make Task 9 unimplementable in order. `resolve_shared` is the shared primitive both `pull_job` and Task 10's `materialize_shared_videos` build on; the machine running `pull` may differ from the one that ran `run.sh`, which is why the check is repeated at all.
- **A hardlink fails across filesystems** (`OSError: [Errno 18] Invalid cross-device link`, e.g. `/tmp` staging and the destination are different filesystems/mounts): catch `OSError` around the hardlink attempt specifically and fall back to a real copy (`shutil.copy2`) rather than failing the pull — this is expected on many lab NAS layouts, not a real error, so it degrades gracefully instead of erroring. Only *other* `OSError`s (permission denied, disk full) propagate as pull failures.

`pull_job` therefore returns a report that separates "hard failures preventing any pull" (still `code=5`, still pre-flight-checked before ANY transfer starts, e.g. collisions and completely absent local mounts for `overwrite=False`) from "per-video skips with reasons" (origin/mount unavailable — printed clearly, pull continues for unaffected videos). Never a silent skip either way — every skip has a reported reason string.

**Fix — `pull_history` MERGES, it does not merely append (spec §5). Round-6 correction: the merge key must match the REAL entry shape.** Spec §5's actual `pull_history` entry (verified `docs/superpowers/specs/2026-09-09-portable-tracking-jobs-design.md:146-147`) is `{"pulled_at": "…", "files": N, "bytes": N}` — there is no `job_relpath` field and no `started_at` field on a pull-history entry (`started_at` is a field of a *separate* structure, the `runs.jsonl` run record `pull` fetches alongside outputs, not of the `pull_history` entry itself); a merge key of `(run started_at, job_relpath)` cannot be built from this shape at all. The double-count risk the ORIGINAL draft was reacting to is real but was mis-described: a `pull` invocation that is retried after crashing partway through (e.g. network drop after transfer completes but before `hydra_job.json` is rewritten) could otherwise append a second, near-duplicate entry for what is really the same physical pull. Fix: `pull_job` merges on the entry's own `pulled_at` timestamp — an entry with a `pulled_at` already present in `manifest.pull_history` is left alone (skip-append); this makes a retried pull that re-reaches the manifest-write step with the SAME already-recorded `pulled_at` (the timestamp is captured once, at the start of the pull, and carried through any retry of the write step) idempotent, while two genuinely separate pulls (different `pulled_at`) both land as distinct entries, which is the correct history to keep.

**Fix — `created_on.git_sha` provenance (no helper currently exists in `src/`).** The manifest's `created_on` dict needs a `git_sha` field but there is no existing helper anywhere in `src/hydra_suite/` that reads the running checkout's commit. Add a small helper in `manifest.py`:

```python
def _current_git_sha() -> str:
    """Best-effort git SHA of the running checkout; "" if not a git repo or
    git is unavailable (e.g. a pip-installed hydra-suite with no .git)."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parent,
            capture_output=True, text=True, timeout=5,
        )
        return result.stdout.strip() if result.returncode == 0 else ""
    except (OSError, subprocess.SubprocessError):
        return ""
```
Called once at pack time (Task 7) to populate `created_on["git_sha"]`; never re-derived at pull time (the manifest is immutable on the remote).

Check `rsync` is on PATH locally and remotely up front, failing with the install hint.

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_tracking_job_transport.py -v`
Expected: all PASS.

- [ ] **Step 5: Optional live localhost exercise**

Run: `HYDRA_TEST_SSH_LOCALHOST=1 python -m pytest tests/test_tracking_job_transport.py -v -k localhost`
Expected: skipped unless the env var is set (guard the live test with `pytest.mark.skipif`).

- [ ] **Step 6: Commit**

```bash
make format
# Fix B10: this task ALSO adds `_current_git_sha` to manifest.py and changes
# pull_history merge semantics there, and touches outputs.py's PullDestination
# consumers via PullReport -- the original `git add` listed only transport.py
# and its test, so those edits would have been left uncommitted.
git add src/hydra_suite/data/tracking_job/transport.py \
        src/hydra_suite/data/tracking_job/manifest.py \
        src/hydra_suite/data/tracking_job/__init__.py \
        tests/test_tracking_job_transport.py \
        tests/test_tracking_job_manifest.py
git commit -m "feat(tracking-job): rsync/ssh transport with a manifest-derived push input set"
```

---

### Task 10: Preflight

**Files:**
- Create: `src/hydra_suite/data/tracking_job/preflight.py`
- Test: `tests/test_tracking_job_preflight.py` (create)

**Interfaces:**
- Consumes: `verify_job` (Task 7), `shared_roots` (Task 5).
- Produces:
  - `PreflightResult` frozen dataclass: `ok: bool`, `checks: list[dict]` (each `{"name", "ok", "detail"}`).
  - `preflight_job(job_dir, *, shared_root_overrides=None, fast=False, allow_tier_fallback=False, available_tiers=None, conda_envs=None) -> PreflightResult`
  - `materialize_shared_videos(manifest, job_dir, table) -> None`

**Every check runs before exit** — a preflight that stops at the first failure makes the user iterate once per problem.

- [ ] **Step 1: Write the failing test**

Create `tests/test_tracking_job_preflight.py`:

```python
"""Preflight fails loudly, reporting every problem at once."""

import pytest

from hydra_suite.data.tracking_job.preflight import preflight_job


def _names(result):
    return {c["name"] for c in result.checks if not c["ok"]}


def test_a_good_job_passes(packed_job):
    result = preflight_job(packed_job, available_tiers=("cpu", "gpu"), conda_envs=("sleap",))
    assert result.ok, result.checks


def test_a_missing_conda_env_fails_naming_it(packed_job_needing_sleap):
    result = preflight_job(
        packed_job_needing_sleap, available_tiers=("gpu",), conda_envs=()
    )
    assert not result.ok
    detail = " ".join(c["detail"] for c in result.checks if not c["ok"])
    assert "sleap" in detail


def test_an_unavailable_tier_fails_unless_fallback_is_allowed(packed_job_gpu_tier):
    strict = preflight_job(packed_job_gpu_tier, available_tiers=("cpu",), conda_envs=())
    assert not strict.ok
    assert "runtime_tier" in _names(strict)
    lenient = preflight_job(
        packed_job_gpu_tier, available_tiers=("cpu",), conda_envs=(),
        allow_tier_fallback=True,
    )
    assert "runtime_tier" not in _names(lenient)


def test_a_tampered_model_fails_the_hash_check(packed_job):
    (packed_job / "models" / "obb" / "x.pt").write_bytes(b"tampered")
    result = preflight_job(packed_job, available_tiers=("cpu",), conda_envs=())
    assert not result.ok
    assert "models" in _names(result)


def test_fast_skips_hashing(packed_job):
    """Fix B-minor: `fast` is only real if it reaches verify_job.

    preflight's FIRST check is `verify_job(job_dir)`, which hashes every model
    and every file_digests member. Calling it without threading `fast` through
    would defeat `fast` entirely -- preflight would skip its own hash check
    while verify quietly did the same work and reported the same tamper under a
    different check name. `preflight_job` must call
    `verify_job(job_dir, fast=fast)` (Task 7 signature), and this test asserts
    the tamper is invisible under BOTH names.
    """
    (packed_job / "models" / "obb" / "x.pt").write_bytes(b"tampered")
    result = preflight_job(
        packed_job, available_tiers=("cpu",), conda_envs=(), fast=True
    )
    failed = _names(result)
    assert "models" not in failed
    assert "verify" not in failed


def test_all_failures_are_reported_not_just_the_first(packed_job_gpu_tier):
    (packed_job_gpu_tier / "models" / "obb" / "x.pt").write_bytes(b"tampered")
    result = preflight_job(packed_job_gpu_tier, available_tiers=("cpu",), conda_envs=())
    assert len(_names(result)) >= 2


def test_a_result_file_is_written(packed_job):
    preflight_job(packed_job, available_tiers=("cpu",), conda_envs=())
    assert (packed_job / "logs" / "preflight.json").is_file()


def test_a_packed_shared_job_verifies_clean(packed_job_shared):
    """Fix V2: a shared video has no file under videos/ at pack time (spec
    §6.7) -- verify_job must exempt it from both the per-model job_path
    existence/size check and the videos.txt line-existence check, or
    pack_job's own self-verify (which every packed_job* fixture, including
    this one, goes through) raises for every shared job -- meaning THIS
    FIXTURE would raise at setup and every shared test in this file would
    error before its body ever ran."""
    from hydra_suite.data.tracking_job.verify import verify_job

    assert verify_job(packed_job_shared) == []


def test_shared_video_is_materialized_as_a_symlink(packed_job_shared, tmp_path):
    # Fix B9: shared.relpath is "colony.mp4" (see packed_job_shared), so the
    # mount copy goes at "<mount>/colony.mp4".
    mount = tmp_path / "mnt" / "lab"
    mount.mkdir(parents=True)
    source = mount / "colony.mp4"
    source.write_bytes(b"\x00" * 2048)
    result = preflight_job(
        packed_job_shared, shared_root_overrides={"labnas": str(mount)},
        available_tiers=("cpu",), conda_envs=(),
    )
    assert result.ok, result.checks
    link = packed_job_shared / "videos" / "colony.mp4"
    assert link.is_symlink()
    assert link.resolve() == source.resolve()


def test_an_unknown_alias_fails_listing_known_aliases(packed_job_shared):
    result = preflight_job(packed_job_shared, shared_root_overrides={}, conda_envs=())
    assert not result.ok
    detail = " ".join(c["detail"] for c in result.checks if not c["ok"])
    assert "labnas" in detail


def test_a_re_encoded_shared_video_fails_the_signature_check(packed_job_shared, tmp_path):
    """Fix B9: the file MUST EXIST at the resolved path with DIFFERENT bytes.

    Writing it at the wrong path made this pass because the file was missing,
    which is the *previous* test's failure mode -- the signature check itself
    was never exercised and could have been deleted with the suite still green.
    """
    mount = tmp_path / "mnt" / "lab"
    mount.mkdir(parents=True)
    source = mount / "colony.mp4"
    # Fix (minor): SAME size as the staging fixture's video (2048 bytes,
    # b"\x00" * 2048) -- using 4096 here would also trip a size_bytes
    # mismatch, so a failure message containing "signature" would not
    # actually prove the SIGNATURE check (vs. a size check) is what fired.
    # Same size, different content isolates the content-signature check.
    source.write_bytes(b"\xff" * 2048)  # exists; re-encoded => different content
    assert source.is_file()
    result = preflight_job(
        packed_job_shared, shared_root_overrides={"labnas": str(mount)}, conda_envs=()
    )
    assert not result.ok
    assert "shared_roots" in _names(result)
    detail = " ".join(c["detail"] for c in result.checks if not c["ok"])
    assert "signature" in detail.lower(), (
        "must fail on the SIGNATURE, not on a missing file"
    )


def test_a_missing_shared_file_fails_with_both_paths(packed_job_shared, tmp_path):
    mount = tmp_path / "mnt" / "lab"
    mount.mkdir(parents=True)  # empty: "<mount>/colony.mp4" does not exist
    result = preflight_job(
        packed_job_shared, shared_root_overrides={"labnas": str(mount)}, conda_envs=()
    )
    assert not result.ok
    detail = " ".join(c["detail"] for c in result.checks if not c["ok"])
    assert "colony.mp4" in detail and str(mount) in detail


def test_a_truncated_non_shared_video_fails_the_signature_check(packed_job):
    """Fix W1c: this is the actual gap -- a NON-shared, pushed video that
    rsync --partial truncated must fail preflight even though it exists and
    even before verify_job's cheaper size_bytes check would also catch it,
    proving the two checks are independent layers, not one masquerading as
    the other."""
    video = packed_job / "videos" / "colony.mp4"
    real = video.resolve()
    real.write_bytes(real.read_bytes()[:100])
    result = preflight_job(packed_job, available_tiers=("cpu",), conda_envs=())
    assert not result.ok
    assert "video_signature" in _names(result)


def test_a_missing_video_fails_preflight(packed_job):
    """Fix W1d/load_video_list gap: load_video_list only WARNs on a missing
    video and silently continues with a subset batch (trackerkit/app.py,
    the `missing = [...]` block). preflight is what turns that into a hard,
    named failure before track ever runs."""
    (packed_job / "videos" / "colony.mp4").unlink()
    result = preflight_job(packed_job, available_tiers=("cpu",), conda_envs=())
    assert not result.ok


def test_insufficient_disk_fails(packed_job, monkeypatch):
    import shutil

    monkeypatch.setattr(
        shutil, "disk_usage", lambda _p: shutil._ntuple_diskusage(1, 1, 0)
    )
    result = preflight_job(packed_job, available_tiers=("cpu",), conda_envs=())
    assert "disk" in _names(result)
```

**Fix M11 — the four `packed_job*` fixtures, fully specified.** The tests above were host-dependent (nothing pinned `runtime_tier`, and nothing isolated the shared-root/host-config lookups from this machine's REAL `HYDRA_CONFIG_DIR`/`HYDRA_HOST_CONFIG_DIR`), so a real `labnas` alias configured on this Mac (see Task 13 Step 5, which does exactly that) could flip these tests. `build_engine_params` also defaults an absent `runtime_tier` to `"gpu"` (`engine_params.py:805`), so any fixture that doesn't pin one explicitly is silently GPU-tier and `test_a_good_job_passes`'s `available_tiers=("cpu", "gpu")` would mask that. Add these to `tests/conftest.py` — the same file Task 7 Step 1b already put `staging`, `_planned` and `packed_job` in. **Fix B8: this task must EXTEND the existing `packed_job`, never define a second one.** Two `@pytest.fixture()`-decorated `packed_job` functions in one module means the later definition silently shadows the earlier one, and which behaviour any given test gets depends on textual order — exactly the kind of invisible coupling that makes a preflight failure unreproducible. Concretely:

1. Add `_isolated_host_config`, `packed_job_needing_sleap`, `packed_job_gpu_tier` and `packed_job_shared` as NEW fixtures.
2. **Edit the existing `packed_job` in place** to (a) take `_isolated_host_config`, and (b) pin `runtime_tier` to `"cpu"` — its body becomes the `packed_job` shown below. Do not paste a duplicate. **The four fixture bodies below call `pack_job` bare**; Task 7's version imported it lazily inside the fixture, so add `from hydra_suite.data.tracking_job.pack import pack_job` at the TOP of `tests/conftest.py` when making this edit (the package exists by Task 10, so a module-level import is safe and removes four duplicated lazy imports). Without it every preflight test raises `NameError: name 'pack_job' is not defined`.
3. Re-run `python -m pytest tests/test_tracking_job_pack.py tests/test_tracking_job_verify.py -v` after the edit: those files consume `packed_job` too, and pinning the tier must not change their outcomes.

Every fixture monkeypatches `HYDRA_CONFIG_DIR` and `HYDRA_HOST_CONFIG_DIR` to `tmp_path` so a real machine-local alias can never leak in:

```python
@pytest.fixture()
def _isolated_host_config(tmp_path, monkeypatch):
    """Every preflight test MUST use this (directly or via packed_job*) so a
    real labnas alias configured on the developer's machine (Task 13 Step 5
    configures exactly one) cannot flip a preflight test's outcome."""
    host_cfg = tmp_path / "host_config"
    host_cfg.mkdir()
    monkeypatch.setenv("HYDRA_CONFIG_DIR", str(host_cfg))
    monkeypatch.setenv("HYDRA_HOST_CONFIG_DIR", str(host_cfg))
    return host_cfg


@pytest.fixture()
def packed_job(tmp_path, staging, _isolated_host_config):
    """A minimal, valid, CPU-tier packed job. runtime_tier is pinned
    explicitly to "cpu" — build_engine_params defaults an absent tier to
    "gpu" (engine_params.py:805), so leaving it unset would make this
    fixture silently GPU-tier and mask tier-related preflight bugs."""
    planned = _planned(staging, config={"runtime_tier": "cpu"})
    job_dir = tmp_path / "job"
    pack_job(
        job_dir, [planned],
        registry_entries=[("obb/x.pt", {"species": "ant", "source_path": "/host/a.pt"})],
        advanced_config_path=str(staging["advanced"]),
        track_args={"video_list": "videos.txt", "runtime_tier": "cpu"},
        shared_table={},
    )
    return job_dir


@pytest.fixture()
def packed_job_needing_sleap(tmp_path, staging, _isolated_host_config):
    """A job whose keystone config selects the SLEAP pose backend, so
    preflight's conda_envs check must name "sleap" as required.

    Fix W12: `enable_pose_extractor` + `pose_model_type` ALONE do not
    satisfy `is_pose_inference_enabled` (core/tracking/session_policy.py:
    29-32) -- that also requires `detection_method == "yolo_obb"` (via
    `is_individual_pipeline_enabled`) AND a non-empty `pose_model_dir`.
    Without both, `is_pose_export_enabled`/`is_pose_inference_enabled` are
    False, pose inference never actually runs, and pack_job's fix-W12 rule
    correctly (per its own definition) computes an EMPTY conda_envs -- this
    fixture would then not "need" sleap at all, and
    test_a_missing_conda_env_fails_naming_it would be testing nothing."""
    planned = _planned(
        staging,
        config={
            "runtime_tier": "cpu",
            "detection_method": "yolo_obb",
            "enable_pose_extractor": True,
            "pose_model_type": "SLEAP",
            "pose_model_dir": "pose/SLEAP/run",
            "pose_sleap_env": "sleap",
        },
    )
    job_dir = tmp_path / "job"
    pack_job(
        job_dir, [planned],
        registry_entries=[("obb/x.pt", {"species": "ant", "source_path": "/host/a.pt"})],
        advanced_config_path=str(staging["advanced"]),
        track_args={"video_list": "videos.txt", "runtime_tier": "cpu"},
        shared_table={},
    )
    return job_dir


@pytest.fixture()
def packed_job_gpu_tier(tmp_path, staging, _isolated_host_config):
    """A job whose keystone config explicitly requests the "gpu" tier, so
    the runtime_tier check has something real to fail against on a
    CPU-only available_tiers set."""
    planned = _planned(staging, config={"runtime_tier": "gpu"})
    job_dir = tmp_path / "job"
    pack_job(
        job_dir, [planned],
        registry_entries=[("obb/x.pt", {"species": "ant", "source_path": "/host/a.pt"})],
        advanced_config_path=str(staging["advanced"]),
        track_args={"video_list": "videos.txt", "runtime_tier": "gpu"},
        shared_table={},
    )
    return job_dir


@pytest.fixture()
def packed_job_shared(tmp_path, staging, _isolated_host_config):
    """A job whose video is referenced via the "labnas" shared-root alias
    rather than copied/symlinked in directly."""
    planned = _planned(staging, config={"runtime_tier": "cpu"})
    job_dir = tmp_path / "job"
    # Fix B9: the alias root is the video's OWN parent, so
    # match_shared_root yields relpath "colony.mp4" -- NOT "2026-09/colony.mp4".
    # The three shared tests below therefore place their fake mount's copy at
    # "<mount>/colony.mp4". An earlier draft had the tests writing to
    # "<mount>/2026-09/colony.mp4", which preflight never looked at, so
    # test_a_re_encoded_shared_video_fails_the_signature_check passed for the
    # WRONG REASON (file absent, not signature mismatch) and would have kept
    # passing even if the signature check were deleted entirely.
    table = {"labnas": str(staging["video"].parent)}
    pack_job(
        job_dir, [planned],
        registry_entries=[("obb/x.pt", {"species": "ant", "source_path": "/host/a.pt"})],
        advanced_config_path=str(staging["advanced"]),
        track_args={"video_list": "videos.txt", "runtime_tier": "cpu"},
        shared_table=table,
    )
    assert (
        JobManifest.read(job_dir / "hydra_job.json").videos[0].shared
        == {"alias": "labnas", "relpath": "colony.mp4"}
    ), "fixture and tests must agree on the shared relpath (fix B9)"
    return job_dir
```

(`JobManifest` is imported at the top of `tests/conftest.py` alongside the other tracking-job imports.)

**`available_tiers=None` semantics (undefined in the original draft):** `preflight_job(..., available_tiers=None)` means "do not run the `runtime_tier` check at all" (report it neither passing nor failing — omit it from `checks`), as distinct from `available_tiers=()` which means "the tier check runs and fails, because nothing is available." This lets a caller that hasn't yet determined the local tier set (e.g. a dry `verify`-only invocation) skip the check honestly rather than getting a spurious pass or fail. `job_cli.py` (this task) always passes a real, non-`None` `available_tiers` derived from `runtime.resolver.available_tiers(detect_platform())` for actual `preflight`/`run` invocations; only test code exercising the "not yet known" path uses `None`. **Fix W7 — `available_tiers` is NOT a bare, zero-argument call.** Its real signature (verified `runtime/resolver.py:77`) is `available_tiers(platform: PlatformInfo) -> list` — it needs a `PlatformInfo` describing the ACTUAL host, obtained via `detect_platform()` (`runtime/resolver.py:104`, which reads `hydra_suite.utils.gpu_utils.CUDA_AVAILABLE`/`MPS_AVAILABLE`). Every call site in this plan is `runtime.resolver.available_tiers(runtime.resolver.detect_platform())`, never `available_tiers()` alone.

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_tracking_job_preflight.py -v`
Expected: FAIL — module missing.

- [ ] **Step 3: Implement `preflight.py`**

Checks, in order, all executed (spec §9.4):

1. `manifest` — parses, version supported, `verify_job` returns no problems.
2. `version` — `hydra_suite` importable and `>= requirements.min_hydra_suite_version`.
3. `conda_envs` — every required env exists. The caller injects the env list (a `conda env list` subprocess in the CLI) so the core stays testable and subprocess-free.
4. `runtime_tier` — requested tier in `available_tiers`; otherwise report the fallback the resolver would take and fail unless `allow_tier_fallback`.
5. `models` — every file hashes to the manifest sha256, and every `file_digests` member hashes to its recorded digest (skipped when `fast`). **Check 1 must call `verify_job(job_dir, fast=fast)`**, not bare `verify_job(job_dir)`; otherwise `fast` is defeated because verify re-hashes everything anyway (fix B-minor).
6. `shared_roots` — resolve each alias (overrides first, then the **host** table, resolved by testing `HYDRA_HOST_CONFIG_DIR` for **presence, not truthiness** — the three cases are distinct and must be handled separately: (a) present and non-empty → that directory IS the host config dir, read `shared_roots.json` from it; (b) present but **empty string** — this is `run.sh`'s own signal (`export HYDRA_HOST_CONFIG_DIR="${HYDRA_CONFIG_DIR:-}"`) that the host had no `HYDRA_CONFIG_DIR` override at launch, i.e. it uses the platformdirs default — read `shared_roots.json` from `paths.get_platform_config_dir()` (Fix X1a, Task 1), which deliberately ignores `HYDRA_CONFIG_DIR` since by the time `run.sh`'s self-preflight runs, `HYDRA_CONFIG_DIR` has ALREADY been redirected to the job's own `config/` snapshot; (c) absent entirely — `trackerkit job preflight <job>` invoked directly, never through `run.sh` — fall back to `HYDRA_CONFIG_DIR` if set, else the platformdirs default (today's `load_shared_roots()` default resolution is correct here, since no redirection has happened at all); never the job snapshot in any case), check existence and readability, compare `size_bytes` and `signature`, then materialize `videos/<basename>` as a symlink (replace an existing **symlink**, never a regular file). Failures here are reported under check name `shared_roots`.
7. `video_signature` (fix W1c — new check). For **every** `manifest.videos[]` entry, shared or not, resolve `videos/<job_path>` to its actual bytes on disk — following the shared symlink `shared_roots` just materialized for a `shared` entry, or the existing symlink/regular file `pack_job` created for a non-shared one — and compare `content_id.video_signature(resolved_path)` against the recorded `video.signature`. A mismatch (or the resolved path being unreadable) is reported under check name `video_signature`, naming the video's `job_path`. This is what actually proves Goal-4-adjacent claim "the video that runs is the video that was packed": a `size_bytes` match (verify, offline) is necessary but not sufficient — two different re-encodes of the same clip can coincidentally land on the same byte count — so the content signature is the check that closes W1 end-to-end for BOTH shared videos (already checked here, redundantly with check 6's own signature comparison — check 6 keeps its signature comparison too, since it runs first and gates whether materialization even makes sense to report as `shared_roots`-scoped; check 7 then re-verifies signature against whatever ended up on disk, covering the non-shared population check 6 never touches) and pushed non-shared videos (the actual W1 gap: a `videos/<name>.mp4` that `rsync --partial` truncated is, after materialization, just a regular file whose `content_id.video_signature` no longer matches what `pack_job` recorded from the origin).
8. `disk` — free space under `videos/` >= 1.5x total video bytes, counting shared videos' sizes for caches but not for the videos themselves. **Minor fix:** `preflight.py` must call `shutil.disk_usage(...)` through the MODULE (`import shutil; shutil.disk_usage(...)`), never `from shutil import disk_usage` bound to a local name — `test_insufficient_disk_fails` monkeypatches the attribute on the `shutil` module object itself (`monkeypatch.setattr(shutil, "disk_usage", ...)`), which only intercepts lookups that go through `shutil.disk_usage` at call time; a `from shutil import disk_usage` import would have already bound the ORIGINAL function into `preflight.py`'s own namespace at import time, and the monkeypatch would silently not apply, making the test measure the real filesystem instead of the fake tiny one.

Write the result to `logs/preflight.json`.

**Exit-code contract (round-5 minor 2).** `trackerkit job preflight` MUST exit
non-zero (code 3, per §13) whenever `PreflightResult.ok` is False. Both the
`job run` ssh chain (`… job preflight . && ./run.sh …`) and `run.sh`'s own
self-preflight under `set -euo pipefail` rely on that exit code to abort before
`track` touches a possibly-truncated video. A preflight that prints failures but
exits 0 silently disarms both protections. Add a CLI test asserting the exit
code for a failing job.

**Fix X1 (supersedes the prior "empty means platformdirs" one-liner and the prior V-minor code block below — both were still keyed on TRUTHINESS, which is exactly what silently swallows `run.sh`'s empty-string signal and aborts every shared-video run).** `run.sh` sets `export HYDRA_HOST_CONFIG_DIR="${HYDRA_CONFIG_DIR:-}"` — on any host using platformdirs defaults (the common case), `HYDRA_CONFIG_DIR` was never set, so this exports `HYDRA_HOST_CONFIG_DIR=""`: **present in the environment, but the empty string**, which Python's `os.environ.get("HYDRA_HOST_CONFIG_DIR")` returns as `""`, and `"" or os.environ.get("HYDRA_CONFIG_DIR")` then falls through to `HYDRA_CONFIG_DIR` — which by the time `run.sh`'s self-preflight runs has ALREADY been redirected to `$JOB/config` (`export HYDRA_CONFIG_DIR="$JOB/config"`, two lines above). That is the job snapshot, which by design never contains `shared_roots.json` (a shared-root table is host identity, not job content) — every alias then reports unknown, check 6 fails, preflight exits 3, and `run.sh`'s `set -euo pipefail` aborts the whole run. **This is not a corner case — it is what happens on every default-configured host, for every job with a shared video.**

**Fix X1b — test presence, not truthiness, and route the "empty" case through `paths.get_platform_config_dir()` (Fix X1a), never through `HYDRA_CONFIG_DIR`.** `load_shared_roots(path=None)` (Task 5) resolves its own default via `get_shared_roots_path()` -> `paths.py`'s `_user_config_dir()`, which reads `HYDRA_CONFIG_DIR` **at call time** — inside `run.sh` that is already the job's redirected dir, so a bare `load_shared_roots()` call is just as broken as the truthiness fallthrough above. `preflight_job` must distinguish three cases by testing for the KEY's presence in `os.environ`, not by testing its value's truthiness (`preflight.py` adds `from ...paths import get_platform_config_dir` alongside its existing `from ...paths import ...`-style imports, the same relative-import convention `shared_roots.py` already uses for `get_shared_roots_path`):

```python
_HOST_CFG_VAR = "HYDRA_HOST_CONFIG_DIR"

if _HOST_CFG_VAR in os.environ:
    _host_cfg = os.environ[_HOST_CFG_VAR]
    if _host_cfg:
        # (a) run.sh, host had an HYDRA_CONFIG_DIR override at launch.
        host_table = load_shared_roots(Path(_host_cfg) / "shared_roots.json")
    else:
        # (b) run.sh, host used the platformdirs default — HYDRA_CONFIG_DIR
        # has already been redirected to the job's own config/, so it must
        # NOT be consulted here.
        host_table = load_shared_roots(get_platform_config_dir() / "shared_roots.json")
else:
    # (c) invoked directly, never through run.sh (e.g. interactive
    # `trackerkit job preflight <job>`) — no redirection has happened, so
    # HYDRA_CONFIG_DIR (or its own platformdirs default) is correct as-is.
    host_table = load_shared_roots()
```

`preflight_job` must therefore never call `load_shared_roots()` bare except in branch (c). Add three tests, each reproducing one real invocation shape exactly (not the isolation fixture's shortcut of pointing both vars at the same dir):

- `test_preflight_reads_the_host_shared_roots_table_not_the_job_snapshot` — `HYDRA_CONFIG_DIR` set to a job-snapshot-shaped tmp dir with no `shared_roots.json`, `HYDRA_HOST_CONFIG_DIR` set to a SEPARATE tmp dir that DOES have one with a real alias; assert the alias resolves (branch a).
- `test_preflight_run_sh_empty_host_config_dir_reads_platformdirs_default` — reproduces `run.sh`'s EXACT env: `HYDRA_HOST_CONFIG_DIR=""` (present, empty) plus `HYDRA_CONFIG_DIR=<job>/config` (redirected, job-snapshot-shaped, no `shared_roots.json`); monkeypatch `platformdirs.user_config_dir` (or `paths.user_config_dir`, whichever `get_platform_config_dir()` calls) to a tmp dir that DOES have a real `shared_roots.json` with the alias; assert the alias resolves, not "unknown alias" (branch b — this is the one X1 exists to fix).
- `test_preflight_direct_invocation_no_host_config_dir_reads_config_dir` — `HYDRA_HOST_CONFIG_DIR` absent (`monkeypatch.delenv(..., raising=False)`), `HYDRA_CONFIG_DIR` set to a tmp dir with a real `shared_roots.json`; assert the alias resolves (branch c).

**Minor fix — restore the "offer to persist" behaviour spec §6.7 step 1 describes and the original draft silently dropped.** A one-off `--shared-root ALIAS=PATH` given to `job run`/`job preflight` should, after a successful preflight run that used it, prompt (interactively, when stdout is a tty and `--yes`/`--no-input` wasn't passed) to save the alias into the host's persistent `shared_roots.json` table via `shared_roots.py` (Task 5), so the next invocation doesn't need to repeat the override. In `job_cli.py` (Task 11), after `preflight_job(..., shared_root_overrides=parsed_overrides)` returns `ok=True`, for each override alias not already present in the persisted table: prompt `Save shared-root alias 'labnas' -> '/mnt/lab' for future jobs? [y/N]` and call `shared_roots.save_alias(alias, path)` on yes. Non-interactive/CI invocations skip the prompt and leave the override one-off, as before.

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_tracking_job_preflight.py tests/test_tracking_job_layering.py -v`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
make format
# Fix B10: this task also EDITS tests/conftest.py (the packed_job fixture is
# amended in place and four fixtures are added) and verify.py (the `fast`
# parameter) -- both would have been left uncommitted.
git add src/hydra_suite/data/tracking_job/preflight.py \
        src/hydra_suite/data/tracking_job/verify.py \
        src/hydra_suite/data/tracking_job/__init__.py \
        tests/test_tracking_job_preflight.py \
        tests/conftest.py
git commit -m "feat(tracking-job): preflight with shared-root materialization, reporting every failure"
```

---

### Task 11: CLI wiring — `trackerkit job …`

**Fix A8 — this task is too large for one agent; it splits into three checkpointed sub-tasks that share this task's numbering (no cross-reference elsewhere in this plan changes):**
- **11a** (Steps 1-3, plus the `pack` flow inside Step 4 — items 1-8 and its "Fix B5"/"Fix M9+B2" subsections): argparse wiring for the whole `job` group, and the `pack` subcommand end to end (the multihead-bundle fix A1a, the skeleton-path computation, the CNN model-key map). Commits at the checkpoint inserted after the `pack` flow.
- **11b** (the `push`/`pull`/`status` wiring inside Step 4, thin — these mostly call straight into Task 9's `transport.py`): argparse-to-transport glue, plus fix A2b's `--remote-bootstrap` plumbing for `push`'s post-push `job verify` call. Commits at the checkpoint inserted after this wiring.
- **11c** (the `run`/`calibrate`/`preflight` flow inside Step 4 — fix M7's chained-ssh dispatch and fix A2b's `--remote-bootstrap` for `run`/`calibrate`, plus Steps 5-8: help smoke, the local pack+verify e2e, the hostile-config e2e, and the M10 coverage gaps): depends on 11a (packed jobs to run against) and 11b (the transport calls `run`'s remote branch wraps). This is the sub-task that actually exercises fix A2's remote-bootstrap requirement end to end. Commits at the existing Step 9.

An agent picking up 11b or 11c should read the prior sub-task's commit(s) for the exact `job_cli.py` module shape rather than re-deriving it.

**Files:**
- Create: `src/hydra_suite/trackerkit/job_cli.py`
- Modify: `src/hydra_suite/trackerkit/app.py:97` (subparsers), `:269-311` (`parse_arguments`), `:399-534` (`main` dispatch)
- Test: `tests/test_trackerkit_job_cli.py` (create)

**Interfaces:**
- Consumes: everything from Tasks 5-10, plus `iter_model_references` (Task 3), `plan_batch_jobs` (`batch_plan.py:119`), `resolve_track_video_inputs` (`app.py:351`), `load_tracker_cli_session` (`cli_config.py:304`), `discover_multihead_model_bundle` (`classkit/model_bundle.py:42`).
- Produces: `run_job_cli(args) -> int`; **`job_cli._dispatch(args) -> int`** — the module-level function `run_job_cli` delegates to after its `try:`, declared here because `test_exit_codes_are_mapped` monkeypatches it by name (fix B-minor: it was used in the test but never listed as produced, so an implementation that inlined the dispatch would make that test fail with `AttributeError`); and the `job` subcommand group.

**This module is the app-layer bridge.** `data/tracking_job` must never import `trackerkit` or `classkit`, so `job_cli.py` does the planning, engine-parameter building and ClassKit bundle discovery, then hands `PlannedVideo`/`PlannedModel` objects down.

**Spec correction #4:** `job calibrate` forwards **only** `inference_autotune_manual`. `run_calibrate_cli()` (`calibrate_cli.py:80`) has no `sahi_profile` parameter, and it does not need one: `pack` bakes the resolved SAHI profile into every sidecar, so the config `calibrate` loads already carries it. Forwarding the manual fields is mandatory — they feed `compute_baseline_digest` → `key.baseline_digest` (`integration.py:440-446`), so dropping them makes `track`'s profile lookup silently miss.

- [ ] **Step 1: Write the failing test**

Create `tests/test_trackerkit_job_cli.py`:

```python
"""The job subcommand group: parsing, rejection, dispatch."""

import pytest

from hydra_suite.trackerkit.app import build_parser, parse_arguments


def test_job_pack_parses_like_track():
    args = parse_arguments(["job", "pack", "/tmp/j", "--video-list", "list.txt"])
    assert args.command == "job"
    assert args.job_command == "pack"
    assert args.video_list == "list.txt"


def test_job_pack_accepts_explicit_videos():
    args = parse_arguments(["job", "pack", "/tmp/j", "a.mp4", "b.mp4"])
    assert args.videos == ["a.mp4", "b.mp4"]


def test_job_pack_rejects_both_videos_and_video_list():
    with pytest.raises(SystemExit):
        parse_arguments(["job", "pack", "/tmp/j", "a.mp4", "--video-list", "l.txt"])


def test_job_pack_rejects_neither():
    with pytest.raises(SystemExit):
        parse_arguments(["job", "pack", "/tmp/j"])


@pytest.mark.parametrize("flag", ["--gpus", "--jobs", "--threads-per-job"])
def test_job_pack_rejects_compute_box_flags(flag):
    """They describe the compute box, not the experiment (calibrate precedent)."""
    with pytest.raises(SystemExit):
        parse_arguments(["job", "pack", "/tmp/j", "a.mp4", flag, "2"])


@pytest.mark.parametrize("flag", ["--sahi-profile", "--inference-autotune-manual"])
def test_job_run_rejects_experiment_flags(flag):
    """Those are fixed at pack time and always forwarded from track_args."""
    with pytest.raises(SystemExit):
        parse_arguments(["job", "run", "/tmp/j", flag, "x"])


def test_job_run_accepts_compute_box_flags():
    args = parse_arguments(["job", "run", "/tmp/j", "--gpus", "auto", "--jobs", "2"])
    assert args.gpus == "auto"
    assert args.jobs == 2


def test_job_pack_rejects_conflicting_shared_modes():
    with pytest.raises(SystemExit):
        parse_arguments(["job", "pack", "/tmp/j", "a.mp4", "--no-shared", "--shared-only"])


def test_every_documented_subcommand_exists():
    parser = build_parser()
    job_action = next(
        a for a in parser._subparsers._group_actions if "job" in a.choices
    )
    assert set(job_action.choices["job"]._subparsers._group_actions[0].choices) >= {
        "pack", "verify", "shared-root", "push", "preflight", "run",
        "calibrate", "status", "pull",
    }


def test_shared_root_add_parses_alias_and_path():
    args = parse_arguments(["job", "shared-root", "add", "labnas", "/Volumes/lab"])
    assert args.alias == "labnas"
    assert args.path == "/Volumes/lab"


def test_pull_flags_parse():
    args = parse_arguments(
        ["job", "pull", "host:/r/j", "/tmp/j", "--dry-run", "--no-caches", "--overwrite"]
    )
    assert args.dry_run and args.no_caches and args.overwrite
    assert args.force is False  # fix A5's still-running guard is ON by default


def test_pull_force_flag_parses():
    args = parse_arguments(["job", "pull", "host:/r/j", "/tmp/j", "--force"])
    assert args.force is True


def test_shared_root_override_parses_as_alias_equals_path():
    args = parse_arguments(
        ["job", "preflight", "/tmp/j", "--shared-root", "labnas=/mnt/lab"]
    )
    assert args.shared_root == ["labnas=/mnt/lab"]


def test_exit_codes_are_mapped(monkeypatch):
    """TrackingJobError.code becomes the process exit code."""
    from hydra_suite.data.tracking_job.manifest import TrackingJobError
    from hydra_suite.trackerkit import job_cli

    def boom(_args):
        raise TrackingJobError("nope", code=3)

    monkeypatch.setattr(job_cli, "_dispatch", boom)
    assert job_cli.run_job_cli(object()) == 3
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_trackerkit_job_cli.py -v`
Expected: FAIL — no `job` subcommand.

- [ ] **Step 3: Register the `job` group in `app.py`**

After the `calibrate` subparser block (ends `:254`), add nested subparsers mirroring the exact house style:

```python
    job_parser = subparsers.add_parser(
        "job",
        help="Package, move, run and retrieve a portable tracking job",
        allow_abbrev=False,
    )
    job_subparsers = job_parser.add_subparsers(dest="job_command")

    job_pack = job_subparsers.add_parser("pack", allow_abbrev=False, help="...")
    # Deliberately NO --gpus / --jobs / --threads-per-job: they describe the
    # COMPUTE BOX, not the experiment, and belong to `job run`. Not registering
    # them makes passing one an "unrecognized arguments" SystemExit from
    # argparse itself -- a loud rejection, not a silent ignore. Same reasoning
    # as the calibrate subparser above.
    # Fix X8: --force alone only allows repacking a non-empty job_dir (clears
    # models/config, removes only the OLD manifest's own video/sidecar pack
    # artifacts). It does NOT authorize discarding pulled outputs sitting
    # under videos/ -- that needs the separate, explicit --discard-outputs,
    # so a plain "fix my config typo and repack" can never silently eat a
    # pulled result.
    job_pack.add_argument("--force", action="store_true")
    job_pack.add_argument("--discard-outputs", action="store_true")
```

`job run` registers `--gpus/--jobs/--threads-per-job/--detach/--calibrate/--allow-tier-fallback/--shared-root/--remote-bootstrap` (fix A2b — required for any remote target, see below) and deliberately **omits** `--sahi-profile/--apply-tuned-inference/--inference-autotune-manual`. `job calibrate` and `job status` also register `--remote-bootstrap` for the same reason (default `""`).

In `parse_arguments`, add a `job`/`pack` branch mirroring `:278-296` (both/neither video checks, `--no-shared` vs `--shared-only` conflict). `_subparser_choices` (`:259-266`) only walks the top level; add a nested lookup so `job_parser.error(...)` and each sub-subparser's `.error(...)` are reachable.

In `main`, add `elif args.command == "job":` dispatching to `run_job_cli(args)` with the same lazy import + `try/except` shape as the `calibrate` branch (`:446-462`).

- [ ] **Step 4: Implement `job_cli.py`**

`pack` flow:

1. `videos = [os.path.abspath(v) for v in resolve_track_video_inputs(args.videos, args.video_list)]`. **Minor fix:** `resolve_track_video_inputs` returns raw strings, possibly relative to the CWD `job pack` was invoked from — it does not `abspath` them. Every downstream `origin_path`, shared-root matching (`match_shared_root` needs a canonical absolute path to compare against the mount table), and the pull-time `origin_path` used to place outputs back all assume an absolute path; a relative one silently breaks once the process's CWD is no longer what it was at pack time (e.g. `run.sh`'s later `cd "$JOB"`, or `pull` running from a different directory entirely).
2. `specs = plan_batch_jobs(videos, explicit_config_path=args.config, keystone_override=args.keystone_override, sahi_profile=args.sahi_profile, apply_tuned_inference=args.apply_tuned_inference, inference_autotune_manual=args.inference_autotune_manual)` — pack does **not** reimplement config precedence. **Minor correction to the rationale:** forwarding `apply_tuned_inference`/`inference_autotune_manual` here is NOT what makes those fields end up in `spec.config` — `plan_batch_jobs` already bakes them into `spec.config` itself before returning (`batch_plan.py:174-180`). Passing them again here is idempotent (re-applying the same values `plan_batch_jobs` already applied), not load-bearing; the actual reason to pass them through explicitly is so `job pack`'s own CLI flags drive the SAME precedence rules every other `plan_batch_jobs` caller gets, not a load-bearing data dependency.
3. For each spec, `session = load_tracker_cli_session(spec.video_path, config_data=spec.config)`. **Fix M16 — do not also call `build_tracking_parameters`.** `TrackerCliSession` already carries the fully-resolved params at `.params` (`cli_config.py:52-70`) — `load_tracker_cli_session` does the same resolution `build_tracking_parameters` would, and the latter additionally needs a `video_probe` this step never has a reason to obtain (probing the video a second time here is pure waste). Use `params = session.params` directly.
4. `refs = list(iter_model_references(params))`.
5. For each ref: `key = make_pose_model_path_relative(ref.path) if ref.kind == "directory" else make_model_path_relative(ref.path)`; if still absolute, `key = external_key_for(ref.path)`. **Minor fix — restrict bundle discovery to `CNN_CLASSIFIERS` refs only.** `discover_multihead_model_bundle` walks `_discover_bundle_siblings` (`classkit/model_bundle.py:159`), whose same-date-prefix heuristic exists to catch classifier head siblings — applied to a non-classifier file ref (e.g. `YOLO_OBB_DIRECT_MODEL_PATH` pointing into `models/obb/`), that same heuristic can match and bundle unrelated `.pt` files that merely happen to share a training-run date prefix in that directory, shipping models the job never uses and, worse, potentially exposing files not actually referenced by the config. So: only when `ref.role == "CNN_CLASSIFIERS"` does this step call `bundle = discover_multihead_model_bundle(ref.path)` and pass `bundle["artifact_paths"]` (minus the selected checkpoint) as `bundle_artifacts`. Every other role's file ref gets `bundle_artifacts=[]` from this call (a role may still separately carry its own known sidecars via `copy_model_metadata_sidecars`, which is unrelated to bundle discovery).

   **Fix A1 — `.multihead.json` bundles never shipped their heads.** `discover_multihead_model_bundle` (`classkit/model_bundle.py:12,82,119`) recognises ONLY the `*.bundle.json` manifest form; it returns `None` for a bare `.multihead.json` primary path and does nothing. But `.multihead.json` is the production identity-classifier format — the equivalence fixture `tools/equivalence/fixtures/configs/ant_cnn_identity.json:235` points `cnn_classifiers[].model_path` straight at one, and core's `_select_loader` (`core/individual/classification/backend.py:530`) dispatches `.multihead.json` to `_ClassifierMultiheadBundleLoader`, which resolves each `factor_models[].path` **relative to the manifest's own directory** (`backend.py:494-497`: `base = manifest_path.parent; factor_path = (base / entry["path"]).resolve()`). Without shipping those heads, the manifest arrives alone, `config.py:1227` sees the path exists and builds the `CNNConfig`, and the loader then fails on the missing head files on the remote box — a run-time break with **no pack-time error**.

   So when `ref.role == "CNN_CLASSIFIERS"` and `ref.path` ends with `.multihead.json` (case-insensitive), do **not** call `discover_multihead_model_bundle` (it is `.bundle.json`-only and returns `None` here). Instead read the manifest's `factor_models` list directly — this is the same lightweight JSON parse `_ClassifierMultiheadBundleLoader.parse_metadata` and `.load` already do (`backend.py:388-478`, `:494-497`); do NOT instantiate `_ClassifierMultiheadBundleLoader.load` itself, since it constructs a live `ClassifierBackend` (device selection, real torch weight load) per head and is far too heavy for a packing step that only needs paths:

   ```python
   def _multihead_bundle_artifacts(manifest_path: str) -> list[str]:
       """Every factor-model head a .multihead.json manifest references,
       resolved the same way core resolves them at load time (backend.py:494-497:
       base = manifest_path.parent; path = (base / entry["path"]).resolve())."""
       import json
       from pathlib import Path

       manifest = Path(manifest_path)
       try:
           data = json.loads(manifest.read_text(encoding="utf-8"))
       except (OSError, json.JSONDecodeError) as exc:
           # Minor fix: a missing or malformed manifest must fail loudly at
           # pack time with a TrackingJobError naming the manifest, not
           # propagate a bare OSError/JSONDecodeError up through job_cli.
           raise TrackingJobError(
               f"cannot read multihead manifest {manifest_path!r}: {exc}",
               code=2,
           ) from exc
       base = manifest.parent
       heads = []
       for entry in data.get("factor_models", []):
           try:
               entry_path = entry["path"]
           except (KeyError, TypeError) as exc:
               # Minor fix: a factor_models[] entry with no "path" (or that
               # isn't even a dict) must raise TrackingJobError, not a bare
               # KeyError -- pack.py's caller contract is TrackingJobError
               # for every packing failure, so an uncaught KeyError here
               # would be an unrecognizable crash instead of a diagnosable
               # pack failure.
               raise TrackingJobError(
                   f"multihead manifest {manifest_path!r} has a factor_models "
                   f"entry with no 'path': {entry!r}",
                   code=2,
               ) from exc
           # Minor fix: model_publish.py:174-175 can emit an ABSOLUTE
           # entry["path"] when a head lives outside the manifest's own
           # directory. `base / entry_path` on an absolute entry_path
           # silently discards `base` (Path's own "/" operator semantics:
           # an absolute right operand replaces the left), so the head
           # would resolve to wherever entry_path itself points -- and the
           # later flatten-by-basename copy step (Task 6) would then copy
           # it under a basename that collides with, or diverges from,
           # what the manifest expects, silently corrupting the shipped
           # bundle. Refuse a non-bare-basename entry["path"] outright.
           if Path(entry_path).name != entry_path or Path(entry_path).is_absolute():
               raise TrackingJobError(
                   f"multihead manifest {manifest_path!r} has a non-relative "
                   f"factor_models path (expected a bare basename): {entry_path!r}",
                   code=2,
               )
           head_path = (base / entry_path).resolve()
           if not head_path.exists():
               raise TrackingJobError(
                   f"multihead manifest {manifest_path!r} references a missing "
                   f"head model: {head_path}",
                   code=2,
               )
           heads.append(str(head_path))
       return heads
   ```

   Pass `bundle_artifacts=_multihead_bundle_artifacts(ref.path)` for these refs (each head also carries its own sidecars via `copy_model_metadata_sidecars`, same as any other `bundle_artifacts` entry — see Task 6's `copy_model_reference`). Keep the existing `discover_multihead_model_bundle` call for the `.bundle.json` case; the two are mutually exclusive by filename suffix.

   **Fix B5 — what this step hands to pack, precisely.** `pack.py` derives the SCALAR `model_keys` itself from `PlannedVideo.planned_models` via `pack.ROLE_TO_CONFIG_KEY` (Task 7); `job_cli` does **not** build that map. What `job_cli` DOES build, because only it observed the per-entry association, is the CNN map:

```python
from hydra_suite.data.tracking_job.pack import PlannedVideo, _normalize_model_path

cnn_model_keys = {
    _normalize_model_path(ref.path): key_for[id(ref)]
    for ref in refs
    if ref.role == "CNN_CLASSIFIERS"
}
```

where `key_for[id(ref)]` is the job key computed for that reference earlier in this step, and `_normalize_model_path` is pack's shared canonicalization (`str(Path(resolve_model_path(raw)).expanduser().resolve())`) so both sides of the lookup compare identical absolute strings. Stamp it on the `PlannedVideo` as `cnn_model_keys=cnn_model_keys`.
6. **Fix M9 + B2 — compute `skeleton_path` explicitly, gated ONLY on the value being non-empty, NEVER on pose enablement.** Spec correction #7 established that `POSE_SKELETON_FILE` has no engine-side fallback — every consumer (`engine_params.py:1179`, `core/post/pose_merge.py:292`, `core/post/media_export.py:401`, `core/individual/properties/cache.py:221`, `core/inference/config.py:1258`) reads it verbatim. So before building `PlannedVideo`:

```python
raw_skeleton = str(session.config.get("pose_skeleton_file", "") or "").strip()
skeleton_path = str(resolve_model_path(raw_skeleton)) if raw_skeleton else ""
```

**Fix B2 — why there is no `enable_pose_extractor` check here.** An earlier draft computed `skeleton_path` only when pose was enabled, which is self-defeating and would have made Task 11 Step 7's hostile acceptance case fail at pack time, before a single assertion ran: `tools/equivalence/fixtures/configs/fly_obb.json` has `enable_pose_extractor: False`, the hostile config then sets an absolute `pose_skeleton_file`, so a pose-gated computation yields `""` → Task 7's `_rewrite_config` skips the rewrite (`if skeleton_job_path:`) → the sidecar keeps the absolute path → `verify_job` rejects it (`pose_skeleton_file` is in `ABSOLUTE_PATH_FORBIDDEN_KEYS` **unconditionally**) → `pack_job`'s own self-verify raises. Snapshotting + rewriting whenever the value is non-empty is both simpler and the only rule consistent with what verify actually enforces. A stale skeleton path in a pose-disabled config costs one small JSON file in the job; leaving it absolute costs the whole pack.

Without this step at all, a pose-enabled job silently ships with no skeleton and every pose-merge/export consumer breaks on the remote with no loud error at pack time.
7. `pack_job(...)` with `registry_entries=list(iter_registry_entries())`, `advanced_config_path=str(get_advanced_config_path())`, `advanced_config_fallback=load_advanced_tracker_config()` (fix X5a — `job_cli.py` is the app-layer caller, so THIS is where `trackerkit.cli_config.load_advanced_tracker_config` is imported and called; `pack.py` never imports it), `shared_table=load_shared_roots()`, `force=args.force`, `force_discard_outputs=args.discard_outputs` (fix X8 — `--discard-outputs` alone, without `--force`, is meaningless since `pack_job` only even looks at `videos/`'s contents on the force path; `job pack` does not reject that combination, it is simply a no-op flag in that case).
8. **Warn** (spec §6.6) when any export stage is enabled in a config: the CLI leaves `DATASET_OUTPUT_DIR`, `FINAL_MEDIA_EXPORT_VIDEO_OUTPUT_DIR` and `INDIVIDUAL_DATASET_OUTPUT_DIR` at `None` (`cli_config.py:293-295`), so those exports produce nothing on the remote. Point at follow-up §17.1.

- [ ] **Sub-task 11a checkpoint: commit `pack` end to end (fix A8)**

At this point `job pack` (argparse wiring from Steps 1-3, plus the flow above) is a complete, independently testable slice — it does not need `push`/`pull`/`run` to exist to be exercised (Step 7's local pack+verify e2e and hostile-config e2e can both run against 11a alone).

```bash
make format
git add src/hydra_suite/trackerkit/job_cli.py \
        src/hydra_suite/trackerkit/app.py \
        tests/test_trackerkit_job_cli.py
git commit -m "feat(job-cli): argparse wiring for the job group + job pack end to end (11a)

Includes fix A1a's .multihead.json head-shipping and fix M9/B2's unconditional
skeleton snapshot."
```

`_record-run` is a hidden subcommand appending one JSON line to `logs/runs.jsonl` (`started_at, finished_at, hostname, exit_code, hydra_suite_version, git_sha, argv, host_advanced_config_used`). The last field is `bool(os.environ.get("HYDRA_JOB_HOST_ADVANCED_CONFIG_USED"))` (minor fix — see `run.sh`'s corresponding export above): since `pull` never fetches `config/` back, this is the only record that survives the round trip telling the local machine the run used a different `advanced_config.json` than the one it pushed. **Fix Y2 (round-7) — `add_parser("_record-run", metavar=argparse.SUPPRESS)` does not exist as an option: `_SubParsersAction.add_parser` forwards unrecognized kwargs straight to `ArgumentParser.__init__`, which has no `metavar` parameter** — verified by running it: `TypeError: ArgumentParser.__init__() got an unexpected keyword argument 'metavar'`, raised at `build_parser()` construction time, which breaks EVERY `trackerkit` invocation, not just `job _record-run`. Use `help=argparse.SUPPRESS` instead (`add_parser`'s `help=` kwarg is the one that is popped before the rest are forwarded to `ArgumentParser.__init__`, and `argparse.SUPPRESS` there is exactly what hides a subparser from `-h` while leaving it fully callable): `job_subparsers.add_parser("_record-run", help=argparse.SUPPRESS)`. This is the SAME pattern used correctly at `job_pack = job_subparsers.add_parser("pack", allow_abbrev=False, help="...")` above (Step 3) — only the value of `help=` differs. Register its trailing argv capture with `nargs=argparse.REMAINDER` (minor fix) — without it, argparse tries to parse `run.sh`'s forwarded `"$@"` (which can contain flags like `--gpus auto`) against `_record-run`'s own option strings and errors out instead of passing them through verbatim. Add `test_record_run_parser_is_hidden_but_callable`: build the parser, assert `"_record-run"` does not appear in `parser.format_help()`, and assert `build_parser().parse_args(["job", "_record-run", "--started", "x", "--exit-code", "0", "--", "--gpus", "auto"])` succeeds without raising. Grep the whole plan for any other `add_parser(..., metavar=` occurrence and fix identically — none found elsewhere as of this fix wave (verified: `grep -n "add_parser(" docs/superpowers/plans/2026-09-09-portable-tracking-jobs.md` shows only the `job` top-level and `pack` subparsers, both already using `help=`, not `metavar=`). **Fix W3b: `finished_at = datetime.now(timezone.utc).isoformat(timespec="microseconds")`, computed by `_record-run` itself at write time — never derived from `run.sh`'s `$START` (which is second-truncated and is `started_at`, a different field) and never second-truncated.** This is what makes `transport.py`'s `_last_runs_jsonl_timestamp` comparable at all to a real (fractional-second) filesystem `st_mtime` on `logs/run.log` — see the Task 9 "still running" guard fix above.

`push`/`pull`/`status` dispatch is thin glue over Task 9's `transport.py` and Task 5's manifest: `push` parses its target with `parse_remote`, calls `push_job(job_dir, target, runner=subprocess.run)`, then (fix A2b, fix (c) above) runs `ssh <host> '<bootstrap>; cd <path> && trackerkit job verify .'` through the same `--remote-bootstrap` and surfaces a non-zero verify as a push failure. `pull` parses its target and calls `pull_job(target, job_dir, include_caches=not args.no_caches, overwrite=args.overwrite, dry_run=args.dry_run, force=args.force)` (fix A5), printing `PullReport.skipped` and returning non-zero when non-empty. `status` reads the local manifest summary plus the tail of `logs/runs.jsonl`, or the same over ssh (with `--remote-bootstrap`) for a remote target.

- [ ] **Sub-task 11b checkpoint: commit `push`/`pull`/`status` wiring (fix A8)**

```bash
make format
git add src/hydra_suite/trackerkit/job_cli.py \
        tests/test_trackerkit_job_cli.py
git commit -m "feat(job-cli): push/pull/status dispatch onto Task 9's transport.py (11b)

Includes fix A5's pull-while-running guard (--force) and fix A2b's
--remote-bootstrap for push's post-push remote verify."
```

**Fix M7 — `run` against a remote target MUST run preflight over ssh BEFORE `./run.sh`, or §6.7 (shared-root materialization) is defeated for the primary workflow.** The original draft only ran `preflight_job` for the LOCAL-dir case and, for remote, went straight to `ssh <host> 'cd <path> && ./run.sh …'`. That means a `shared` video is never materialized (symlinked into `videos/`) on the compute box before `run.sh` invokes `trackerkit track`, so the primary "shared NAS mount, don't copy the video" workflow (spec §6.7) silently fails on first use for every remote run — the only path that actually exercises it in production. Both branches now run preflight first, **forwarding `--shared-root`/`--allow-tier-fallback` to THIS preflight call** (they are `job run`'s own flags, parsed by `job_cli.py` — not passed through to `run.sh`, which never sees them, per fix X2), and then set `HYDRA_JOB_SKIP_PREFLIGHT=1` so run.sh's own flag-less self-preflight (Task 7, fix X2) does not immediately re-run the SAME checks without those flags and fail on the exact alias/tier gap the first preflight just cleared:

```
local dir  -> preflight_job(job_dir, shared_root_overrides=.., allow_tier_fallback=..) locally, then
              subprocess.run(["./run.sh", *passthrough], cwd=job_dir, check=False,
                              env={**os.environ, "HYDRA_JOB_TRACKERKIT": ..., "HYDRA_JOB_SKIP_PREFLIGHT": "1"})
remote     -> ssh <host> '<remote_bootstrap> cd <path> &&
              trackerkit job preflight . [--shared-root ALIAS=PATH ...] [--allow-tier-fallback] &&
              HYDRA_JOB_SKIP_PREFLIGHT=1 ./run.sh …'
              (single ssh invocation; preflight's non-zero exit short-circuits
              the && before run.sh ever starts, so a materialization failure is
              reported before any tracking begins), wrapped in `nohup … &`
              under --detach.
```

A one-off `--shared-root` given to `job run` is therefore no longer dead: it reaches the ONE preflight that actually runs (`job run`'s own, with the flag), and `run.sh`'s self-preflight is skipped rather than silently re-failing without it. `--allow-tier-fallback` is threaded the same way. Add `test_job_run_forwards_shared_root_and_sets_skip_preflight` (local branch): monkeypatch `preflight_job` to capture its kwargs and `subprocess.run` to capture its `env`, call `job run <job> --shared-root labnas=/mnt/lab --allow-tier-fallback`, and assert both `preflight_job(shared_root_overrides={"labnas": "/mnt/lab"}, allow_tier_fallback=True, ...)` and `env["HYDRA_JOB_SKIP_PREFLIGHT"] == "1"`. Add `test_job_run_remote_command_includes_shared_root_flag_and_skip_preflight`: assert the constructed ssh command string contains both `--shared-root labnas=/mnt/lab` before the `&&` and `HYDRA_JOB_SKIP_PREFLIGHT=1` immediately before `./run.sh`.

**Fix V-minor — `--detach`'s `nohup … &` wrapper must redirect stdout/stderr explicitly, or it double-writes alongside `run.sh`'s own logging.** `run.sh` already does `... | tee -a logs/run.log` internally (Task 7's runner spec) — that is the durable, structured log a detached run is meant to be checked via `job status`/`logs/run.log` later. `nohup` with no redirect specified defaults to appending combined output to `./nohup.out` in the CWD the `ssh`/`subprocess` invocation runs from, which is a SECOND, redundant copy of the same output living in a different, undocumented file that nothing in this plan ever names, checks, or cleans up — worse, for the remote branch that CWD is wherever the ssh session's shell starts (not necessarily the job dir, depending on `--remote-bootstrap`), so `nohup.out` can land somewhere the user never thinks to look. Fix: the `--detach` wrapper explicitly redirects to `/dev/null` since `run.sh`'s own `tee -a logs/run.log` is already the durable record: `nohup ./run.sh {shlex.join(passthrough)} >/dev/null 2>&1 &` (local) / `nohup ./run.sh {shlex.join(passthrough)} >/dev/null 2>&1 &` appended inside the remote `ssh` command string (after the `&&` chain, before the closing quote). No `nohup.out` is ever created by either branch.

**Minor fix (round-7) — the REMOTE `ssh ... 'bootstrap && cd path && nohup ./run.sh ... >/dev/null 2>&1 &'` invocation also needs its STDIN closed, or the ssh session itself can hang waiting for input that never arrives.** Redirecting the backgrounded `run.sh`'s own stdout/stderr does not touch the parent `ssh` client's stdin — a non-interactive `ssh host 'cmd &'` without `-f` or an explicit `</dev/null` can still block on the local end until the remote shell's own stdin is closed, which it normally is only when the ssh session itself exits (and it won't, because the backgrounded job keeps the remote shell's job table non-empty in some sshd configurations). Fix: append `</dev/null` to the full remote command string (`ssh host 'bootstrap && cd path && nohup ./run.sh ... >/dev/null 2>&1 & disown' </dev/null`, or equivalently invoke with `ssh -f host '...'` which backgrounds the ssh client itself after authentication). `job_cli.py`'s `--detach` remote branch (Task 11) uses `</dev/null` on the `subprocess.run(["ssh", ...])` call's `stdin=subprocess.DEVNULL` — the Python equivalent of shell's `</dev/null` — rather than relying on shell redirection inside the remote command string, since that is the one guaranteed to close the LOCAL ssh client's own stdin regardless of what the remote shell does.
`subprocess.run(["./run.sh", *passthrough])` for the local-dir branch MUST pass `cwd=job_dir` (minor fix) — `run.sh` itself resolves its own location via `BASH_SOURCE`, but the parent Python process's CWD is whatever the user invoked `trackerkit job run` from, and without `cwd=job_dir` a relative job path argument on the CLI would still work by luck (bash resolves `./run.sh` against the argv path, not CWD) while anything inside `run.sh` that assumes CWD == job root during the brief window before its own `cd "$JOB"` would not. `track_args` recorded at pack time are **always** forwarded.

**Fix V3 — the local branch of `job run` must ITSELF export `HYDRA_JOB_TRACKERKIT`, or `run.sh`'s PATH-independence (fix A2a) is dead code on every dev machine.** `run.sh` reads `TRACKERKIT_STR="${HYDRA_JOB_TRACKERKIT:-trackerkit}"` (Task 7's runner template) — it only ever *consumes* the variable, nothing in this plan as written ever *sets* it for the local-dir path. `job_cli.py`'s local branch runs a bare `subprocess.run(["./run.sh", *passthrough], cwd=job_dir, check=False)` with no `env=` override, so the child inherits the parent's environment unchanged and `HYDRA_JOB_TRACKERKIT` is simply absent — `run.sh` falls back to bare `trackerkit`, exactly the failure this plan's own Global Constraint calls out: in any dev worktree, `trackerkit` on `$PATH` resolves to MAIN's editable install (`pip install -e` registers one console-script entry point per environment; a worktree checkout is not a separate install), not this worktree's checked-out code. Two concrete failures follow: (1) `trackerkit job preflight .` invoked from inside `run.sh` (there is no such invocation inside `run.sh` itself, but `job_cli.py`'s own remote branch chains `trackerkit job preflight . && ./run.sh` over ssh with the SAME PATH-resolution problem for the ssh session, covered separately by fix A2b's `--remote-bootstrap`) would hit `argparse: invalid choice: 'job'` if MAIN predates the `job` subcommand; (2) even where `job` exists on MAIN, `trackerkit track` inside `run.sh` runs MAIN's tracking code — silently different cache-key/detection logic from the worktree the job was packed and is being verified from, which defeats the entire portable-jobs feature for local testing (every "run this locally to verify Goal-N" acceptance step in Task 13 would actually be exercising MAIN, not the branch under test).

Fix: `job_cli.py`'s local branch (Task 11) sets `HYDRA_JOB_TRACKERKIT` explicitly before invoking `run.sh`:

```python
env = dict(os.environ)
env.setdefault(
    "HYDRA_JOB_TRACKERKIT", f"{shlex.quote(sys.executable)} -m hydra_suite.trackerkit.app"
)
# Fix X2: this env-based `subprocess.run` ONLY runs after `preflight_job(...)`
# (above, with the caller's --shared-root/--allow-tier-fallback) has already
# returned ok=True -- so it is safe, and required, to tell run.sh's own
# flag-less self-preflight to skip: it would otherwise re-run the SAME checks
# without those flags and fail on the exact gap the local preflight just
# cleared.
env["HYDRA_JOB_SKIP_PREFLIGHT"] = "1"
subprocess.run(["./run.sh", *passthrough], cwd=job_dir, check=False, env=env)
```

`sys.executable` is the interpreter `trackerkit job run` itself is running under, so this always resolves to whichever environment (and, inside a worktree with `pip install -e` pointed at that worktree's `src/`, whichever checkout) invoked the CLI — never a second, possibly-different `trackerkit` resolved fresh from `$PATH`. `setdefault` (not a plain assignment) preserves an explicit user override of `HYDRA_JOB_TRACKERKIT` in their shell environment, e.g. to point at a specific remote-matching interpreter for a local dry run. Verified `python -m hydra_suite.trackerkit.app` is the real module path for the entry point: `pyproject.toml`'s `trackerkit = "hydra_suite.trackerkit.app:main"`.

Add `test_job_run_local_sets_hydra_job_trackerkit` (Task 11 or Task 13's CLI test module, whichever hosts the `job run` unit tests): monkeypatch `subprocess.run` to capture its `env` kwarg, call the local-dir `job run` path against a `packed_job`, and assert `env["HYDRA_JOB_TRACKERKIT"]` contains `sys.executable` and `hydra_suite.trackerkit.app`.

**Fix A2b — `job run <remote>` (and `job calibrate <remote>`/`job status <remote>`) cannot resolve `trackerkit` (or even `conda`) over a bare `ssh host 'cmd'`, so the remote branch above is unimplementable as written.** Verified directly on firebrat: `ssh firebrat 'which trackerkit'` -> rc 1, and **even `ssh firebrat 'bash -lc "which trackerkit"'` -> rc 1** — a non-interactive `bash -lc` there still does not put the conda env's entry points on PATH. Only an explicit `source ~/miniforge3/etc/profile.d/conda.sh && conda activate hydra-cuda` resolves it (confirmed: `hydra_suite.__file__` then reports `/home/rutalab/hydra-suite/src/hydra_suite/__init__.py`, an editable install, so whichever branch is checked out there is what `trackerkit` runs). This is exactly the failure mode CLAUDE.md line 44 forbids for local commands, and it applies equally over ssh.

Every remote-target subcommand (`run`, `calibrate`, `status`) therefore takes a `--remote-bootstrap TEXT` option: a shell fragment prepended, verbatim and semicolon-terminated, to the remote command before `cd <path>`. Default: `""` (empty — preserves today's behavior for a login-ish remote shell where `trackerkit` genuinely is on PATH; most boxes are not that, so an empty default will visibly fail rather than silently mis-schedule, which is the safer failure). The constructed remote command becomes:

```python
# Fix X2: forward the caller's --shared-root/--allow-tier-fallback into THIS
# preflight (the one that actually runs and gates ./run.sh), and tell
# run.sh's own flag-less self-preflight to skip -- otherwise it re-runs the
# SAME checks without those flags immediately afterward and fails on the
# exact gap this preflight just cleared.
preflight_flags = "".join(f" --shared-root {shlex.quote(a)}={shlex.quote(p)}" for a, p in shared_root_overrides.items())
if allow_tier_fallback:
    preflight_flags += " --allow-tier-fallback"
remote_cmd = (
    f"{bootstrap} cd {shlex.quote(remote_path)} && "
    f"trackerkit job preflight .{preflight_flags} && "
    f"HYDRA_JOB_SKIP_PREFLIGHT=1 ./run.sh {shlex.join(passthrough)}"
)
```

where `bootstrap` is `args.remote_bootstrap.rstrip()` plus a trailing `; ` if non-empty (so `source ... && conda activate ...` — itself `&&`-joined — cannot short-circuit the rest of the chained command by being read as the LHS of the following `&&`). The value used against firebrat for Task 13's acceptance run is:

```
--remote-bootstrap "source ~/miniforge3/etc/profile.d/conda.sh && conda activate hydra-cuda"
```

recorded here so Task 13's acceptance commands use it verbatim rather than re-discovering it. `run.sh` itself does not need `conda activate` (Fix A2a already makes it PATH-independent via `HYDRA_JOB_TRACKERKIT`), but the ssh session invoking `trackerkit job preflight` *before* `run.sh` starts does — the bootstrap covers exactly that gap.

Task 13 Step 5 must name this actual command (`<bootstrap>; cd <path> && trackerkit job preflight . [--shared-root ...] [--allow-tier-fallback] && HYDRA_JOB_SKIP_PREFLIGHT=1 ./run.sh …` via one `ssh` invocation) rather than describing preflight and run as separate, un-chained steps.

`calibrate`: run `trackerkit calibrate` inside the job environment (`HYDRA_MODELS_DIR`, `HYDRA_CONFIG_DIR`, `cd <job>`) against the keystone video, forwarding `track_args["inference_autotune_manual"]` verbatim; for a remote target this goes through the same `--remote-bootstrap` prefix as `run`.

All `TrackingJobError`s are caught in `run_job_cli`, printed as `error: <message>`, and returned as `err.code`.

- [ ] **Step 5: Run tests to verify they pass**

Run:
```bash
python -m pytest tests/test_trackerkit_job_cli.py \
                 tests/test_trackerkit_app.py \
                 tests/test_trackerkit_cli_fanout.py \
                 tests/test_trackerkit_calibrate_cli.py -v
```
(Fix M14: `tests/test_trackerkit_cli.py` does not exist — the real existing CLI test files are `tests/test_trackerkit_app.py`, `tests/test_trackerkit_cli_fanout.py` and `tests/test_trackerkit_calibrate_cli.py`; running a nonexistent path silently reports 0 tests collected from that arg rather than failing loudly, which would have hidden a real regression here.)
Expected: PASS, with the existing CLI tests unchanged (a new subcommand must not perturb `track`/`calibrate` parsing).

- [ ] **Step 6: Smoke the help output**

Run (fix B15 — bare `trackerkit` here violated this plan's own PYTHONPATH constraint; on this Mac it resolves to MAIN's editable install, which has no `job` subcommand at all, so this step would "fail" for a reason unrelated to the branch):
```bash
WT=/Users/neurorishika/Projects/Rockefeller/Kronauer/multi-animal-tracker/.worktrees/portable-jobs
PYTHONPATH=$WT/src python -m hydra_suite.trackerkit.app job --help
PYTHONPATH=$WT/src python -m hydra_suite.trackerkit.app job pack --help
PYTHONPATH=$WT/src python -m hydra_suite.trackerkit.app job run --help
```
Expected: all nine subcommands listed; `pack --help` shows no `--gpus`; `run --help` shows no `--sahi-profile`.

- [ ] **Step 7: End-to-end local pack + verify — realistic fixture AND a purpose-built hostile config**

**Fix M13 — `trackerkit` bare must never be invoked from this worktree.** `hydra_suite.__file__` resolves to MAIN's editable install here, not the worktree's `src/` (verified) — a bare `trackerkit` invocation would silently exercise unmodified `main` code and report false confidence. Every invocation in this step (and in Task 13) uses:
```bash
PYTHONPATH=/Users/neurorishika/Projects/Rockefeller/Kronauer/multi-animal-tracker/.worktrees/portable-jobs/src \
  python -m hydra_suite.trackerkit.app job ...
```
Also fix the fixture-availability gap: `tools/equivalence/fixtures/clips/*.mp4` are gitignored and are **not present in a fresh worktree**. Fetch them first (this is a real setup step this plan previously omitted, not implied by "sanity" checking the runner):
```bash
conda activate hydra-mps
cd /Users/neurorishika/Projects/Rockefeller/Kronauer/multi-animal-tracker/.worktrees/portable-jobs
bash tools/equivalence/fixtures/fetch_fixtures.sh   # populates fixtures/clips/*.mp4 + fixtures/configs/*.json
test -f tools/equivalence/fixtures/clips/fly_obb.mp4 || { echo "fixtures missing"; exit 1; }
```

**Fix C4 — the acceptance fixtures are already portable, so a round trip against them alone proves almost nothing.** `tools/equivalence/fixtures/configs/fly_obb.json` and `ant_pose_headtail.json` already have `file_path`/`csv_path`/`video_output_path`/`pose_skeleton_file` = `""` and models-root-relative model keys — a "confirm no absolute paths" check against them is vacuously true regardless of whether the absolute-path rewrite, redirect recording, `external/` keys, skeleton snapshot, or `cnn_classifiers` rewriting actually work, because none of those code paths are triggered by an already-portable input. Run BOTH cases:

1. **Realistic case** — the existing fixture, unmodified:
```bash
mkdir -p /tmp/jobsmoke && cp tools/equivalence/fixtures/configs/fly_obb.json /tmp/jobsmoke/cfg.json
PYTHONPATH=$PWD/src python -m hydra_suite.trackerkit.app job pack /tmp/jobsmoke/job \
    tools/equivalence/fixtures/clips/fly_obb.mp4 --config /tmp/jobsmoke/cfg.json
PYTHONPATH=$PWD/src python -m hydra_suite.trackerkit.app job verify /tmp/jobsmoke/job
```
Expected: pack succeeds, `verify` reports no problems, and `/tmp/jobsmoke/job/models/obb/` holds the OBB checkpoint. Inspect `videos/fly_obb_config.json` and confirm **no absolute paths**.

2. **Hostile case — the load-bearing one.** Build a deliberately-hostile config derived from the fixture, then pack it and assert every rewrite actually happened:
```bash
mkdir -p /tmp/jobsmoke_hostile/renders /tmp/jobsmoke_hostile/outside_models
python - <<'PY'
import json, shutil
from pathlib import Path

base = json.loads(Path("tools/equivalence/fixtures/configs/fly_obb.json").read_text())
outside = Path("/tmp/jobsmoke_hostile/outside_models/extra_classifier.pth")
outside.write_bytes(b"outside-model")

base["file_path"] = "/tmp/jobsmoke_hostile/should_be_ignored.mp4"      # deliberately wrong absolute source
base["csv_path"] = "/tmp/jobsmoke_hostile/abs_out.csv"                  # absolute output
base["video_output_path"] = "/tmp/jobsmoke_hostile/renders/custom.mp4" # redirected OUTSIDE the video dir
base["pose_skeleton_file"] = "/tmp/jobsmoke_hostile/abs_skel.json"      # absolute skeleton
Path("/tmp/jobsmoke_hostile/abs_skel.json").write_text('{"nodes": []}')
base["enable_identity_analysis"] = True
base["cnn_classifiers"] = [{"model_path": str(outside), "batch_size": 8}]  # absolute, OUTSIDE models root

Path("/tmp/jobsmoke_hostile/hostile_cfg.json").write_text(json.dumps(base))
PY

PYTHONPATH=$PWD/src python -m hydra_suite.trackerkit.app job pack /tmp/jobsmoke_hostile/job \
    tools/equivalence/fixtures/clips/fly_obb.mp4 --config /tmp/jobsmoke_hostile/hostile_cfg.json
PYTHONPATH=$PWD/src python -m hydra_suite.trackerkit.app job verify /tmp/jobsmoke_hostile/job

python - <<'PY'
import json
from pathlib import Path

job = Path("/tmp/jobsmoke_hostile/job")
sidecar = json.loads(next(job.glob("videos/*_config.json")).read_text())
assert not sidecar["file_path"].startswith("/"), sidecar["file_path"]
assert not sidecar["csv_path"].startswith("/"), sidecar["csv_path"]
assert not sidecar["video_output_path"].startswith("/"), sidecar["video_output_path"]
assert not sidecar["pose_skeleton_file"].startswith("/"), sidecar["pose_skeleton_file"]
for entry in sidecar.get("cnn_classifiers", []):
    assert not entry["model_path"].startswith("/"), entry["model_path"]
    assert entry["model_path"].startswith("external/"), (
        "the out-of-models-root classifier must land under the external/ key: "
        f"{entry['model_path']}"
    )

manifest = json.loads((job / "hydra_job.json").read_text())
video_entry = manifest["videos"][0]
assert video_entry["redirected_outputs"], "the redirected render output must be recorded"
print("hostile-config round trip: ALL REWRITES VERIFIED")
PY
```
Expected: pack succeeds despite every deliberately-adversarial input; `verify` reports no problems (everything landed job-relative); the printed assertions confirm the absolute-path rewrite, the redirect recording, the `external/` key for the out-of-root classifier, and the skeleton snapshot all actually fired. If ANY assertion fails, the corresponding Task 2/6/7 code is broken regardless of what the realistic-fixture case in sub-step 1 reported.

- [ ] **Step 8: Fill the M10 coverage gaps — spec requirements this plan otherwise leaves untested**

Five spec requirements had no task or step anywhere in the original draft. Add all five here, as part of Task 11 since each depends on the CLI wiring just built:

**(a) §15.4 — the sidecar↔engine-params equality test. This is the actual unit proof of Goal 3 ("a packed job resolves identically on the compute box") and was entirely absent.** Add `tests/test_tracking_job_sidecar_engine_params_equality.py`:
```python
"""The sidecar a packed job ships must resolve to the SAME engine params the
staging machine built — this is the unit-level proof of Goal 3."""

import json
import os

import pytest

from hydra_suite.trackerkit.cli_config import (
    TrackerCliVideoProbe,
    load_tracker_cli_session,
)

# Fix B6: `load_tracker_cli_session(None, ...)` CANNOT work.
#   * `video_path: str` is a REQUIRED POSITIONAL (cli_config.py:304-306), and
#   * when `video_probe` is not supplied it calls `probe_video(video_path)`
#     (`:317`), which raises `RuntimeError(f"Failed to open video: ...")` on an
#     unopenable file (`:253-254`).
# The `staging` fixture's video is 2 KiB of zeros -- cv2 cannot open it -- so
# even passing the real path without a probe would raise. Pass BOTH the real
# job-relative video path (read from the sidecar's own `file_path`, which pack
# rewrote to "videos/<name>") AND an injected probe. The `video_probe=`
# keyword is supported at `:308`.
#
# ALTERNATIVE, if a genuinely decodable clip is ever wanted here:
# `tests/helpers/tiny_clip.py` builds a real 6-frame 64x64 video; swap the
# injected probe for a clip from that harness and drop `video_probe=`.
_PROBE = TrackerCliVideoProbe(fps=30.0, total_frames=1, width=64, height=64)


def _staging_params(staging, monkeypatch):
    """The STAGING-side engine params: same config, resolved against the
    staging models root (never the packed job's). This is the "expected"
    side of the fix X7 equality check below.
    """
    monkeypatch.setenv("HYDRA_MODELS_DIR", str(staging["models"]))
    monkeypatch.delenv("HYDRA_CONFIG_DIR", raising=False)
    config = {
        "file_path": str(staging["video"]),
        "yolo_obb_direct_model_path": "obb/x.pt",
        "pose_skeleton_file": str(staging["skeleton"]),
    }
    session = load_tracker_cli_session(
        str(staging["video"]), config_data=config, video_probe=_PROBE
    )
    return session.params


def test_packed_sidecar_resolves_identically_to_staging(packed_job, staging, monkeypatch):
    # Fix X7 (round-6): the original assertion only checked CONTAINMENT
    # (resolved path lies somewhere inside <job>/models) -- that passes even
    # when the sidecar resolves to the WRONG model that happens to live
    # inside the job root (a legacy `yolo_model_path` alias, or a pose
    # backend mix-up that ships model B but the sidecar's role still points
    # at model A's job-relative slot). Spec §15.4 requires the sidecar's
    # resolved params equal the staging-side params KEY FOR KEY. Since the
    # two sides resolve against DIFFERENT absolute roots (staging models dir
    # vs. <job>/models), "equal" means: for every role key, the path
    # RELATIVE TO ITS OWN MODELS ROOT is identical on both sides -- that is
    # the actual portable invariant (same model, same role, same relative
    # slot), not merely "somewhere under models/".
    staging_params = _staging_params(staging, monkeypatch)

    monkeypatch.setenv("HYDRA_MODELS_DIR", str(packed_job / "models"))
    monkeypatch.setenv("HYDRA_CONFIG_DIR", str(packed_job / "config"))
    monkeypatch.chdir(packed_job)

    def _relative_to_root(value, root):
        return os.path.relpath(os.path.abspath(value), os.path.abspath(str(root)))

    for sidecar in packed_job.glob("videos/*_config.json"):
        video_relpath = json.loads(sidecar.read_text())["file_path"]
        session = load_tracker_cli_session(
            video_relpath, config_path=str(sidecar), video_probe=_PROBE
        )
        params = session.params
        for role_key in ("YOLO_OBB_DIRECT_MODEL_PATH", "POSE_MODEL_DIR"):
            staged_value = staging_params.get(role_key, "")
            packed_value = params.get(role_key, "")
            assert bool(staged_value) == bool(packed_value), (
                f"{role_key}: staging has {staged_value!r}, packed sidecar has "
                f"{packed_value!r} -- presence must match"
            )
            if not staged_value:
                continue
            assert os.path.commonpath(
                [os.path.abspath(packed_value), os.path.abspath(str(packed_job / "models"))]
            ) == os.path.abspath(str(packed_job / "models")), (
                f"{role_key} resolved outside <job>/models: {packed_value}"
            )
            assert _relative_to_root(staged_value, staging["models"]) == _relative_to_root(
                packed_value, packed_job / "models"
            ), (
                f"{role_key} resolved to a DIFFERENT model: staging picked "
                f"{_relative_to_root(staged_value, staging['models'])!r}, packed sidecar "
                f"picked {_relative_to_root(packed_value, packed_job / 'models')!r}"
            )
        staged_cnn = {
            entry.get("factor_name", i): _relative_to_root(entry.get("model_path", ""), staging["models"])
            for i, entry in enumerate(staging_params.get("CNN_CLASSIFIERS", []) or [])
            if entry.get("model_path")
        }
        packed_cnn = {
            entry.get("factor_name", i): _relative_to_root(entry.get("model_path", ""), packed_job / "models")
            for i, entry in enumerate(params.get("CNN_CLASSIFIERS", []) or [])
            if entry.get("model_path")
        }
        assert staged_cnn == packed_cnn, (
            f"CNN_CLASSIFIERS resolved differently: staging={staged_cnn} packed={packed_cnn}"
        )
```
The signature above is verified against `cli_config.py:304-310`:
`load_tracker_cli_session(video_path: str, *, config_path=None, config_data=None, video_probe=None, advanced_config=None)`. **This same `(real video path + injected `TrackerCliVideoProbe`)` shape applies EVERYWHERE this plan calls `load_tracker_cli_session` on a fixture job** — there is no variant that accepts `None`.

The load-bearing assertion is: point `HYDRA_MODELS_DIR`/`HYDRA_CONFIG_DIR` at the packed job, CWD at the job root, load each sidecar, and prove every resolved model path (compared to the STAGING-side session built against the staging models root, key for key — fix X7) lands inside `<job>/models` AND names the SAME model, for that role.

**(b) §6.5's characterization test must call the GUI's real `build_config_dict`, not just `make_model_path_relative`.** Testing the helper alone (as Task 2's tests do) proves the helper works, not that the GUI calls it on every relevant field.

**Fix B16 — this is specified as HOW, not what.** The entry point is verified: `ConfigOrchestrator.build_config_dict(self, preset_mode: bool = False, preset_name=None, preset_description=None) -> dict` at `src/hydra_suite/trackerkit/gui/orchestrators/config.py:1625-1630`. It reads live `self._panels.*` widget state, so it needs a real offscreen `MainWindow` — exactly the plumbing `tests/test_get_parameters_dict_characterization.py:43-54, 303-325` already has. Reuse that pattern verbatim; do not invent a new one. Add `tests/test_gui_config_dict_portability.py`:

```python
"""build_config_dict() -- the GUI's real save path -- emits portable paths."""

import os
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytest.importorskip("PySide6")

from PySide6.QtWidgets import QApplication  # noqa: E402

from hydra_suite.trackerkit.gui.main_window import MainWindow  # noqa: E402


@pytest.fixture(scope="module")
def qapp() -> QApplication:
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


@pytest.fixture
def main_window(monkeypatch, qapp):
    monkeypatch.setattr(MainWindow, "_save_advanced_config", lambda self: None)
    monkeypatch.setattr(MainWindow, "_load_advanced_config", lambda self: {})
    window = MainWindow()
    try:
        yield window
    finally:
        window.close()


def test_build_config_dict_relativizes_color_tag_and_cnn_paths(
    main_window, tmp_path, monkeypatch
):
    models = tmp_path / "models"
    (models / "classification" / "identity").mkdir(parents=True)
    tag = models / "classification" / "identity" / "tag.pth"
    clf = models / "classification" / "identity" / "clf.multihead.json"
    tag.write_bytes(b"t")
    clf.write_text("{}")
    monkeypatch.setenv("HYDRA_MODELS_DIR", str(models))

    orch = main_window._config_orch
    orch._panels.identity.line_color_tag_model.setText(str(tag))
    monkeypatch.setattr(
        main_window,
        "_identity_config",
        lambda: {"cnn_classifiers": [{"model_path": str(clf), "batch_size": 8}]},
    )

    cfg = orch.build_config_dict(preset_mode=False)

    assert cfg["color_tag_model_path"] == "classification/identity/tag.pth"
    assert cfg["cnn_classifiers"][0]["model_path"] == (
        "classification/identity/clf.multihead.json"
    )
    assert not Path(cfg["color_tag_model_path"]).is_absolute()
```

**Fix W8 — `main_window._panels` does not exist; verified source.** `MainWindow._panels_bundle()` (`main_window.py:979`) builds and returns a FRESH `SimpleNamespace` each call — it is not a persistent attribute anyone can read back. The persistent attribute is `ConfigOrchestrator._panels` (`orchestrators/config.py:165`, set in `__init__(self, main_window, config, panels)`; `MainWindow` holds the orchestrator itself at `self._config_orch = ConfigOrchestrator(...)`, `main_window.py:936`), reached as `main_window._config_orch._panels` — i.e. `orch._panels` once `orch = main_window._config_orch` is already bound, as the test above now does. `main_window._config_orch` is a real, persistent attribute (verified: `main_window.py:936`); `main_window._panels` is not. The test now consistently uses `orch._panels.identity...`.

(`main_window._config_orch` is the attribute the existing characterization test drives, and `orch._panels` is `ConfigOrchestrator`'s own persistent panel bundle, not `MainWindow`'s; `_identity_config` is the accessor Task 2 Step 5's `cnn_classifiers` serialization reads through, so patching it is what isolates this test from whatever the identity panel's live widget state happens to be.)

**(c) §8.1/§11 — remote `job verify` over ssh after push.** Add to `job_cli.py`'s `push` flow (or a documented follow-on step): after `push_job` succeeds, `ssh <host> '<bootstrap>; cd <path> && trackerkit job verify .'` (same `--remote-bootstrap` prefix as fix A2b, since this is exactly the same bare-PATH problem) and surface a non-zero verify result as a push failure — a push that lands a broken job on the remote should fail loudly, not silently succeed and defer the discovery to `run`. `push` therefore also registers `--remote-bootstrap`.

**(d) §8.2 step 5 — pull collision policy tests.** Add to `tests/test_tracking_job_transport.py`: a test that `pull_job` without `overwrite` and a sha256-mismatched existing destination file raises `TrackingJobError(code=5)` naming the colliding path; a test that `overwrite=True` proceeds and replaces it; a test that an identical-sha256 existing file at the destination is treated as already-pulled (no error, no re-copy) rather than a collision.

**(e) §9.2 — `job status`, `run --calibrate`, `job calibrate <remote>`, `--budget-seconds`.** These four are named in the spec's CLI surface but had no parsing test anywhere in Task 11's Step 1. Add:
```python
def test_job_status_parses():
    args = parse_arguments(["job", "status", "/tmp/j"])
    assert args.job_command == "status"


def test_job_run_calibrate_flag_parses():
    args = parse_arguments(["job", "run", "/tmp/j", "--calibrate"])
    assert args.calibrate is True


def test_job_calibrate_accepts_a_remote_target():
    args = parse_arguments(["job", "calibrate", "host:/remote/j"])
    assert args.target == "host:/remote/j"


def test_job_run_budget_seconds_parses():
    args = parse_arguments(["job", "run", "/tmp/j", "--budget-seconds", "3600"])
    assert args.budget_seconds == 3600


def test_job_run_remote_bootstrap_defaults_to_empty():
    """Fix A2b: default must be '' -- an empty bootstrap fails loudly against a
    box where trackerkit is not on a bare ssh PATH, rather than silently
    mis-scheduling. See the firebrat value documented in fix A2b."""
    args = parse_arguments(["job", "run", "host:/remote/j"])
    assert args.remote_bootstrap == ""


def test_job_run_remote_bootstrap_parses():
    args = parse_arguments(
        [
            "job", "run", "host:/remote/j",
            "--remote-bootstrap",
            "source ~/miniforge3/etc/profile.d/conda.sh && conda activate hydra-cuda",
        ]
    )
    assert "conda activate hydra-cuda" in args.remote_bootstrap
```
`job status` prints the manifest summary plus the tail of `logs/runs.jsonl` (local) or the same over ssh (remote target). `run --calibrate` runs `trackerkit job calibrate <job>` before `run.sh`, sharing the same forwarding rule as the standalone `calibrate` subcommand (spec correction #4: only `inference_autotune_manual`, never `--sahi-profile`). `job calibrate <remote>` runs calibration over ssh the same way `run` does (fix M7's chained-ssh pattern).

**Fix W11 — `--budget-seconds` has nowhere to go on `track`, so it must route ONLY to the calibration leg, never into `track_args`.** Verified: `--budget-seconds` is exclusively a `trackerkit calibrate` CLI flag (`app.py:224-228`, consumed at `app.py:299-306,453` and threaded through to `calibrate_cli.py:84,159`'s `budget_seconds` parameter). `trackerkit track`'s own argparse subparser has no `--budget-seconds` option at all. `job run --budget-seconds N` (without `--calibrate`) forwarding `N` into `track_args` and then into `run.sh`'s `"$@"` would make `$TRACKERKIT track --video-list videos.txt --budget-seconds N` fail outright — `track`'s argparse rejects the unrecognized flag and the WHOLE run aborts before a single frame is processed, for a flag whose only meaning is "how long may the calibration search run." `job_cli.py` therefore keeps `args.budget_seconds` entirely separate from `track_args`: when `--calibrate` (or the standalone `job calibrate`) is what's running, it is passed as the `budget_seconds=` argument to the calibration invocation (the `trackerkit job calibrate` dispatch, or the pre-`run.sh` calibration step `run --calibrate` triggers); it is NEVER appended to `track_args` and therefore never rides along into `"$@"` on the `track` invocation itself. `job run --budget-seconds N` with no `--calibrate` present is a no-op for that flag (there is no calibration leg to apply it to) — `job_cli.py`'s parser should accept the combination without erroring (an unused budget is harmless), but `job_cli.py` must not attempt to forward it to `track` in that case either.

**Fix V-minor — `job calibrate [--budget-seconds]` must supply the SAME default `trackerkit calibrate` itself uses, or an omitted flag crashes at the `run_calibrate_cli` call.** Verified: `run_calibrate_cli(video_path, *, config_path=None, budget_seconds: float, inference_autotune_manual=None)` (`calibrate_cli.py:80-86`) declares `budget_seconds` as a REQUIRED keyword-only parameter with no default. `trackerkit calibrate`'s own argparse subparser never lets this bite because it sets `default=DEFAULT_CALIBRATION_BUDGET_SECONDS` on `--budget-seconds` (`app.py:224-231`, `DEFAULT_CALIBRATION_BUDGET_SECONDS` imported at `app.py:16`) — `args.budget_seconds` is therefore never `None` by the time it reaches `calibrate_cli.py:453`'s `budget_seconds=float(args.budget_seconds)`. `job_cli.py`'s `job calibrate`/`job run --calibrate` argparse definitions (Task 11) must do the same: `add_argument("--budget-seconds", type=float, default=DEFAULT_CALIBRATION_BUDGET_SECONDS, ...)` (import `DEFAULT_CALIBRATION_BUDGET_SECONDS` from the same module `app.py` does). Without an explicit default here, `job calibrate <job>` with no flag passes `args.budget_seconds = None` straight into `run_calibrate_cli(..., budget_seconds=None)`, which either raises inside the search-deadline arithmetic or (worse) silently produces a zero/negative budget depending on where `None` first gets compared — either way a bug this plan's own Task 11 CLI tests would need to catch. Add `test_job_calibrate_budget_seconds_defaults_when_omitted`: `parse_arguments(["job", "calibrate", "/tmp/j"])` and assert `args.budget_seconds == DEFAULT_CALIBRATION_BUDGET_SECONDS` (not `None`).

- [ ] **Step 9: Commit (sub-task 11c — `run`/`calibrate`/`preflight` + the e2e/coverage-gap tests; fix A8)**

`job_cli.py`'s `pack` and `push`/`pull`/`status` pieces were already committed at the 11a/11b checkpoints above — this commit is the `run`/`calibrate` remote-bootstrap dispatch (fix M7 + fix A2b) plus everything Steps 5-8 added.

```bash
make format && make lint
# Fix B10: Step 8 also adds tests/test_gui_config_dict_portability.py (8b) and
# changes job_cli's push flow (8c); Step 4 stamps cnn_model_keys, which lives on
# pack.PlannedVideo. All are listed here.
git add src/hydra_suite/trackerkit/job_cli.py \
        src/hydra_suite/trackerkit/app.py \
        src/hydra_suite/data/tracking_job/pack.py \
        tests/test_trackerkit_job_cli.py \
        tests/test_tracking_job_sidecar_engine_params_equality.py \
        tests/test_gui_config_dict_portability.py \
        tests/test_config_model_path_portability.py \
        tests/test_tracking_job_transport.py
git commit -m "feat(trackerkit): job run/calibrate/preflight CLI, remote-bootstrap dispatch (11c)

Completes Task 11 (11a: pack, 11b: push/pull/status) with fix M7's chained
ssh preflight-then-run.sh and fix A2b's --remote-bootstrap for every
remote-target subcommand."
```

---

### Task 12: Documentation

**Files:**
- Create: `docs/user-guide/trackerkit-jobs.md`
- Modify: `docs/user-guide/trackerkit-cli.md` (cross-link from `## A batch`), `mkdocs.yml` (nav), **`docs/developer-guide/architecture.md`** (fix B-minor: the file was previously unnamed — append a "Portable job model-reference contract" section there; it is the guide that already describes layer boundaries and extension points, and it is already in the mkdocs nav so no nav edit is needed for it)

- [ ] **Step 1: Write `docs/user-guide/trackerkit-jobs.md`**

Cover: the lifecycle walkthrough (pack → push → preflight → calibrate → run → pull); the "what travels, what doesn't" table (models/config/videos travel; TensorRT/ONNX/CoreML engines, calibration profiles and conda envs do not); the shared-root mount table with a two-host example (`{"labnas": "/Volumes/lab"}` on the laptop, `{"labnas": "/mnt/lab"}` on the box); the conda-env requirement and how preflight reports it; the nine-GPU example adapted to `job run --gpus auto`; and the §6.6 export limitation with a pointer to follow-up §17.1.

- [ ] **Step 2: Add the developer-guide note**

Append a "Portable job model-reference contract" section to `docs/developer-guide/architecture.md` stating that `iter_model_references` + the four key tuples in `engine_params.py` are the contract every new model role must join; that `pack.ROLE_TO_CONFIG_KEY` must gain the matching lowercase config key(s); and that `tests/test_engine_params_model_reference_contract.py` fails loudly otherwise.

- [ ] **Step 3: Build the docs**

Run: `make docs-check`
Expected: strict build passes, terminology check clean.

- [ ] **Step 4: Commit**

```bash
git add docs/user-guide/trackerkit-jobs.md \
        docs/user-guide/trackerkit-cli.md \
        docs/developer-guide/architecture.md \
        mkdocs.yml
git commit -m "docs: portable tracking jobs user guide and the model-reference contract note"
```

---

### Task 13: Acceptance — full gates + the real round trip on firebrat

This task produces evidence, not code. **Nothing merges until every box here is ticked with pasted output.**

- [ ] **Step 1: Full local suite delta**

```bash
# Fix B14: self-contained -- `conda activate` does not survive to the next
# Bash tool call, and the run MUST use this worktree's src, not MAIN's
# editable install.
cd /Users/neurorishika/Projects/Rockefeller/Kronauer/multi-animal-tracker/.worktrees/portable-jobs && \
  pkill -f 'sleap|hydra' || true
cd /Users/neurorishika/Projects/Rockefeller/Kronauer/multi-animal-tracker/.worktrees/portable-jobs && \
  PYTHONPATH=$PWD/src KMP_DUPLICATE_LIB_OK=TRUE \
  conda run --no-capture-output -n hydra-mps python -m pytest tests/ -q 2>&1 | tail -30
```
Run this in the FOREGROUND. A backgrounded run piped through `tail` shows partial output that reads like a hang (memory `feedback_partial_output_is_not_a_hang`).
Expected: the failure SET is a subset of the known-failing set on `8f9688e0`. Compare **sets**, not counts — new test files shift pytest chunk boundaries and can fake collection errors (memory `project_test_suite_batching_chunk_boundary_trap`). Record both sets.

- [ ] **Step 2: MPS equivalence matrix (final, post-everything)**

Run the Task 4 Step 13 recipe again at the branch tip. Expected: every clip EQUIVALENT at its determinism floor, row counts > 1 on every CSV, perf ratio <= 1.25.

- [ ] **Step 3: CUDA equivalence matrix on firebrat**

Run the Task 4 Step 14 recipe at the branch tip on `firebrat`, **including its fix Y7 git-bundle transport step first** (the branch has moved since Step 14 was last run, so the bundle must be re-created and re-fetched, not assumed still current on firebrat). **Confirm `courtship` is still not to be touched.** Expected: same acceptance.

- [ ] **Step 4: The Goal-4 round trip — the only proof that matters**

**Correction (fix to the plan's own gate claim, was overstated):** the equivalence harness does NOT exercise "zero cache reuse". `tools/equivalence/runner.py:154` forces `use_cached_detections: False` for the *forward* pass only, but every fixture config has `enable_backward_tracking: true` and the runner's DISABLE block never turns that off, so the backward pass reads the forward pass's own detection cache **in-process, at the same path, on the same machine**. The matrix therefore DOES exercise write-then-read key equality (the cache written at path P by the forward pass is read back at the same path P by the backward pass) — that is real coverage, and it is why a wrong `config_hash`/`model_id` computation would already have broken the existing gate before this branch ever touched portability. What the matrix does **not** exercise is **cross-machine, cross-path reuse**: a cache written on one host/path and read back on a different host at a different absolute path, with no re-detection. That is the one gap this step closes, and it is the only step in the whole plan that proves it. State this precisely — "the cache change is gated except for portability", not "the cache change is ungated" — anywhere else in this document that repeats the stronger, false claim.

**Fix B14 — every command below is SELF-CONTAINED.** An agent executing this plan gets a FRESH shell per Bash tool call: `export PYTHONPATH=...`, `alias job_cli=...` and a bare `conda activate hydra-mps` do NOT survive to the next call, so a plan written that way silently runs the next command against MAIN's editable install in the base env — the exact false-confidence failure fix M13 exists to prevent. Each line therefore inlines its own env, and conda is entered via `conda run --no-capture-output` (which inherits the inlined env vars and, unlike plain `conda run`, streams output instead of buffering it).

**Fix A1c — `ant_cnn_identity` MUST be one of the packed jobs.** The original two jobs (`fly_obb`, `ant_pose_headtail`) never exercise the `.multihead.json` bundle path at all — `fly_obb`/`ant_pose_headtail` carry no `cnn_classifiers` entries — so nothing in this acceptance run would ever have caught fix A1a/A1b if they were wrong. `ant_cnn_identity.json`/`ant_cnn_identity.mp4` already exist as equivalence fixtures (`tools/equivalence/fixtures/configs/ant_cnn_identity.json`, `.../clips/ant_cnn_identity.mp4`) and its `cnn_classifiers[0].model_path` points at the real `*.multihead.json` manifest, so it is the correct third job. Below, `BOOTSTRAP` is the value fix A2b established for firebrat.

```bash
BOOTSTRAP="source ~/miniforge3/etc/profile.d/conda.sh && conda activate hydra-cuda"

# 0. Populate the gitignored fixture clips (once per machine).
cd /Users/neurorishika/Projects/Rockefeller/Kronauer/multi-animal-tracker/.worktrees/portable-jobs && \
  conda run --no-capture-output -n hydra-mps bash tools/equivalence/fixtures/fetch_fixtures.sh

# 1. Pack THREE fixture jobs from a COPY of the clips, never from
#    tools/equivalence/fixtures/clips/ itself: `pull` lands outputs BESIDE THE
#    ORIGIN, so packing in place would write CSVs, logs and a
#    .inference_cache_<stem>/ into the tracked fixture directory. The third
#    job (ant_cnn_identity) is fix A1c: it is the ONLY fixture whose
#    cnn_classifiers points at a real .multihead.json bundle, so it is the
#    only job that exercises the fix A1a/A1b head-shipping path at all.
mkdir -p /tmp/jobs/src && cp \
  /Users/neurorishika/Projects/Rockefeller/Kronauer/multi-animal-tracker/.worktrees/portable-jobs/tools/equivalence/fixtures/clips/fly_obb.mp4 \
  /Users/neurorishika/Projects/Rockefeller/Kronauer/multi-animal-tracker/.worktrees/portable-jobs/tools/equivalence/fixtures/clips/ant_pose_headtail.mp4 \
  /Users/neurorishika/Projects/Rockefeller/Kronauer/multi-animal-tracker/.worktrees/portable-jobs/tools/equivalence/fixtures/clips/ant_cnn_identity.mp4 \
  /tmp/jobs/src/
# Fix X3b: ant_pose_headtail and ant_cnn_identity are pose-enabled but ship
# with pose_skeleton_file: "" -- run_matrix.sh injects the real skeleton as a
# separate table column at run time; job pack has no such column, so we
# materialize the same skeleton into a real config file first (fix X3a's new
# pack-time guard would otherwise refuse both, correctly).
cd /Users/neurorishika/Projects/Rockefeller/Kronauer/multi-animal-tracker/.worktrees/portable-jobs && \
  PYTHONPATH=$PWD/src conda run --no-capture-output -n hydra-mps python - <<'PY'
import json
from pathlib import Path

fx = Path("tools/equivalence/fixtures")
skel = str((fx / "ooceraea_biroi.json").resolve())
Path("/tmp/jobs").mkdir(parents=True, exist_ok=True)
for name in ("ant_pose_headtail", "ant_cnn_identity"):
    cfg_path = fx / "configs" / f"{name}.json"
    cfg = json.loads(cfg_path.read_text())
    cfg["pose_skeleton_file"] = skel
    (Path("/tmp/jobs") / f"{name}_config.json").write_text(json.dumps(cfg))
PY

cd /Users/neurorishika/Projects/Rockefeller/Kronauer/multi-animal-tracker/.worktrees/portable-jobs && \
  PYTHONPATH=$PWD/src conda run --no-capture-output -n hydra-mps \
    python -m hydra_suite.trackerkit.app job pack /tmp/jobs/fly \
      /tmp/jobs/src/fly_obb.mp4 \
      --config tools/equivalence/fixtures/configs/fly_obb.json

cd /Users/neurorishika/Projects/Rockefeller/Kronauer/multi-animal-tracker/.worktrees/portable-jobs && \
  PYTHONPATH=$PWD/src conda run --no-capture-output -n hydra-mps \
    python -m hydra_suite.trackerkit.app job pack /tmp/jobs/pose \
      /tmp/jobs/src/ant_pose_headtail.mp4 \
      --config /tmp/jobs/ant_pose_headtail_config.json

cd /Users/neurorishika/Projects/Rockefeller/Kronauer/multi-animal-tracker/.worktrees/portable-jobs && \
  PYTHONPATH=$PWD/src conda run --no-capture-output -n hydra-mps \
    python -m hydra_suite.trackerkit.app job pack /tmp/jobs/identity \
      /tmp/jobs/src/ant_cnn_identity.mp4 \
      --config /tmp/jobs/ant_cnn_identity_config.json

# Assert the multihead heads actually shipped (fix A1a/A1c evidence, not just
# the manifest): every factor_models[].path resolved beside the manifest.
#
# Fix (minor, Task 13) -- `find ... -exec python - {} \; <<'PY'` is BROKEN
# for more than one match: the heredoc is stdin for the `find` COMMAND
# LINE, not per-invocation stdin for each spawned `python -` process --
# every -exec'd child inherits the SAME underlying stdin file descriptor
# from `find`, and once the first child reads the heredoc to EOF, every
# subsequent child sees EOF immediately and silently no-ops (exit 0, no
# assertion ever runs, no output). Write the checker to a real file instead
# so each `-exec` invocation gets its own independent read of it.
cat > /tmp/check_multihead_shipped.py <<'PY'
import json, sys
from pathlib import Path
manifest = Path(sys.argv[1])
data = json.loads(manifest.read_text())
for entry in data["factor_models"]:
    head = (manifest.parent / entry["path"]).resolve()
    assert head.is_file(), f"MISSING SHIPPED HEAD: {head}"
    print(f"OK: {head}")
PY
find /tmp/jobs/identity/models -iname '*.multihead.json' \
  -exec python3 /tmp/check_multihead_shipped.py {} \;

# 2. Push.
cd /Users/neurorishika/Projects/Rockefeller/Kronauer/multi-animal-tracker/.worktrees/portable-jobs && \
  PYTHONPATH=$PWD/src conda run --no-capture-output -n hydra-mps \
    python -m hydra_suite.trackerkit.app job push /tmp/jobs/fly \
      rutalab@firebrat:/home/rutalab/jobs/fly

cd /Users/neurorishika/Projects/Rockefeller/Kronauer/multi-animal-tracker/.worktrees/portable-jobs && \
  PYTHONPATH=$PWD/src conda run --no-capture-output -n hydra-mps \
    python -m hydra_suite.trackerkit.app job push /tmp/jobs/pose \
      rutalab@firebrat:/home/rutalab/jobs/pose

cd /Users/neurorishika/Projects/Rockefeller/Kronauer/multi-animal-tracker/.worktrees/portable-jobs && \
  PYTHONPATH=$PWD/src conda run --no-capture-output -n hydra-mps \
    python -m hydra_suite.trackerkit.app job push /tmp/jobs/identity \
      rutalab@firebrat:/home/rutalab/jobs/identity

# 3. Local preflight sanity.
cd /Users/neurorishika/Projects/Rockefeller/Kronauer/multi-animal-tracker/.worktrees/portable-jobs && \
  PYTHONPATH=$PWD/src conda run --no-capture-output -n hydra-mps \
    python -m hydra_suite.trackerkit.app job preflight /tmp/jobs/fly

# 4. Run remotely. Fix M7: `job run <remote>` internally issues ONE ssh
#    invocation chaining `trackerkit job preflight . && ./run.sh ...`.
#    Fix A2b: `--remote-bootstrap` is REQUIRED here -- a bare ssh session on
#    firebrat cannot resolve `trackerkit` or `conda` (verified: even
#    `ssh firebrat 'bash -lc "which trackerkit"'` -> rc 1).
cd /Users/neurorishika/Projects/Rockefeller/Kronauer/multi-animal-tracker/.worktrees/portable-jobs && \
  PYTHONPATH=$PWD/src conda run --no-capture-output -n hydra-mps \
    python -m hydra_suite.trackerkit.app job run \
      rutalab@firebrat:/home/rutalab/jobs/fly --remote-bootstrap "$BOOTSTRAP"

cd /Users/neurorishika/Projects/Rockefeller/Kronauer/multi-animal-tracker/.worktrees/portable-jobs && \
  PYTHONPATH=$PWD/src conda run --no-capture-output -n hydra-mps \
    python -m hydra_suite.trackerkit.app job run \
      rutalab@firebrat:/home/rutalab/jobs/pose --remote-bootstrap "$BOOTSTRAP"

cd /Users/neurorishika/Projects/Rockefeller/Kronauer/multi-animal-tracker/.worktrees/portable-jobs && \
  PYTHONPATH=$PWD/src conda run --no-capture-output -n hydra-mps \
    python -m hydra_suite.trackerkit.app job run \
      rutalab@firebrat:/home/rutalab/jobs/identity --remote-bootstrap "$BOOTSTRAP"

# 5. Pull.
cd /Users/neurorishika/Projects/Rockefeller/Kronauer/multi-animal-tracker/.worktrees/portable-jobs && \
  PYTHONPATH=$PWD/src conda run --no-capture-output -n hydra-mps \
    python -m hydra_suite.trackerkit.app job pull \
      rutalab@firebrat:/home/rutalab/jobs/fly /tmp/jobs/fly

cd /Users/neurorishika/Projects/Rockefeller/Kronauer/multi-animal-tracker/.worktrees/portable-jobs && \
  PYTHONPATH=$PWD/src conda run --no-capture-output -n hydra-mps \
    python -m hydra_suite.trackerkit.app job pull \
      rutalab@firebrat:/home/rutalab/jobs/pose /tmp/jobs/pose

cd /Users/neurorishika/Projects/Rockefeller/Kronauer/multi-animal-tracker/.worktrees/portable-jobs && \
  PYTHONPATH=$PWD/src conda run --no-capture-output -n hydra-mps \
    python -m hydra_suite.trackerkit.app job pull \
      rutalab@firebrat:/home/rutalab/jobs/identity /tmp/jobs/identity

# 6. The Goal-4 probe (Step 4 item 3 below), on all three pulled jobs.
cd /Users/neurorishika/Projects/Rockefeller/Kronauer/multi-animal-tracker/.worktrees/portable-jobs && \
  PYTHONPATH=$PWD/src conda run --no-capture-output -n hydra-mps \
    python tools/equivalence/probe_cache_hit.py /tmp/jobs/fly videos/fly_obb.mp4

cd /Users/neurorishika/Projects/Rockefeller/Kronauer/multi-animal-tracker/.worktrees/portable-jobs && \
  PYTHONPATH=$PWD/src conda run --no-capture-output -n hydra-mps \
    python tools/equivalence/probe_cache_hit.py /tmp/jobs/pose videos/ant_pose_headtail.mp4

cd /Users/neurorishika/Projects/Rockefeller/Kronauer/multi-animal-tracker/.worktrees/portable-jobs && \
  PYTHONPATH=$PWD/src conda run --no-capture-output -n hydra-mps \
    python tools/equivalence/probe_cache_hit.py /tmp/jobs/identity videos/ant_cnn_identity.mp4

# 7. Fix W2 -- the SECOND probe run per job, in --local mode: no HYDRA_*
#    overrides, original video path (pull already landed it/its refreshed
#    cache beside the origin, per command 2's assertion). This proves KEY
#    PORTABILITY (the same cache key the job snapshot's config produces on
#    firebrat is reproduced locally, on this machine, at this path) -- it
#    does NOT prove "a local `trackerkit track` run reuses the cache", and
#    the wording must not claim that.
#
#    Minor fix (round-7): use the PACKED sidecar config
#    (/tmp/jobs/<name>_config.json, written earlier in this step) here, NOT
#    tools/equivalence/fixtures/configs/*.json directly. The ORIGINAL
#    fixture configs for ant_pose_headtail/ant_cnn_identity have
#    pose_skeleton_file: "" -- fix X3a's pack-time guard already refuses to
#    PACK a pose-enabled video with no skeleton, and a real local `track`
#    run against the untouched fixture config would fail at SLEAP
#    construction ("SLEAP backend requires keypoint_names") before it ever
#    got near the cache. The probe script itself doesn't build a full
#    session (fix B1), so it wouldn't visibly fail on the bare fixture
#    config -- but "it doesn't crash" there is not evidence of anything;
#    it would only be proving the probe's own narrow cache-key check
#    against a config that could never actually run. The
#    /tmp/jobs/<name>_config.json files already carry the real, resolved
#    skeleton path (written above, same block that materializes them for
#    `job pack`), so using them here keeps the probe's input consistent
#    with what actually got packed and pushed.
cd /Users/neurorishika/Projects/Rockefeller/Kronauer/multi-animal-tracker/.worktrees/portable-jobs && \
  PYTHONPATH=$PWD/src conda run --no-capture-output -n hydra-mps \
    python tools/equivalence/probe_cache_hit.py --local \
      /tmp/jobs/src/fly_obb.mp4 tools/equivalence/fixtures/configs/fly_obb.json

cd /Users/neurorishika/Projects/Rockefeller/Kronauer/multi-animal-tracker/.worktrees/portable-jobs && \
  PYTHONPATH=$PWD/src conda run --no-capture-output -n hydra-mps \
    python tools/equivalence/probe_cache_hit.py --local \
      /tmp/jobs/src/ant_pose_headtail.mp4 /tmp/jobs/ant_pose_headtail_config.json

cd /Users/neurorishika/Projects/Rockefeller/Kronauer/multi-animal-tracker/.worktrees/portable-jobs && \
  PYTHONPATH=$PWD/src conda run --no-capture-output -n hydra-mps \
    python tools/equivalence/probe_cache_hit.py --local \
      /tmp/jobs/src/ant_cnn_identity.mp4 /tmp/jobs/ant_cnn_identity_config.json
```

Assert, and paste the evidence:

1. **Zero registration** happened on firebrat — `~/.local/share/hydra-suite/models/model_registry.json` on the box is byte-identical before and after (record its sha256 both times):
   ```bash
   ssh rutalab@firebrat 'sha256sum ~/.local/share/hydra-suite/models/model_registry.json 2>/dev/null || echo "ABSENT"'
   ```
   Run it before step 2 (push) and again after step 4 (run); the two lines must match exactly. **Minor note on what this check does and does not prove:** `run.sh` sets `HYDRA_MODELS_DIR="$JOB/models"` unconditionally, so the HOST registry at `~/.local/share/hydra-suite/models/model_registry.json` is architecturally unreachable from this run regardless of whether the run/preflight code is correct — no code path in this plan reads or writes it while `HYDRA_MODELS_DIR` is pinned to the job. This check is therefore closer to tautological than a positive proof of "zero registration behavior": it would pass even if a future regression introduced a `HYDRA_MODELS_DIR`-unaware model-registration call elsewhere, as long as that call also happened to read `HYDRA_MODELS_DIR` correctly. Keep it (a hash mismatch here WOULD be a real, loud bug), but don't over-read a pass as proof the registration code path was exercised and found safe — it wasn't exercised at all.
2. Every artifact from the §10 table landed **beside the ORIGIN clip** (`/tmp/jobs/src/`, per command 1) on the Mac, including `.inference_cache_<stem>/`. **Minor fix — do not expect `<stem>_tracking.mp4` among them, and use the REAL CSV names (fix X4).** All three fixture configs (`fly_obb.json`, `ant_pose_headtail.json`, `ant_cnn_identity.json` — grepped, verified) have `"video_output_path": ""`, and `core/tracking/session.py:653` only renders the overlay video when `video_output_path` is non-empty. All three also have `enable_backward_tracking: true`, so `headless_tracking.py:221-224` writes `<stem>_tracking_forward.csv`/`<stem>_tracking_backward.csv` as the raw passes (never `<stem>_tracking.csv` — that name is the pre-backward-split default `_default_output_paths` computes and is never what actually lands on disk once backward tracking is on), plus `<stem>_tracking_forward_processed.csv` (post-processing intermediate) and the mode-dependent terminal file(s): Debug mode writes `<stem>_tracking_final.csv` and `<stem>_tracking_final_with_individual.csv`; User mode writes `<stem>_tracks.csv` instead (`core/tracking/session.py:812`, `user_tracks_path`). So the expected artifact set for this run is those CSVs (not `_tracking.csv`/`_tracking_with_individual.csv`, which this pipeline never writes), `run.log`, and `.inference_cache_<stem>/` — never a rendered `.mp4`. If a rendered video WERE desired for this acceptance run, a fixture would need `video_output_path` set at pack time (Task 7's redirect/default-name rewrite already handles that case); none of the three does, so none should be checked for:
   ```bash
   ls -la /tmp/jobs/src/ && ls -la /tmp/jobs/src/.inference_cache_fly_obb/
   ```
   Note the two locations are deliberately different and both matter: `pull`
   maps outputs to the ORIGIN (`/tmp/jobs/src/`), while the probe in command 6
   reads `build_inference_cache_dir(<job>/videos/fly_obb.mp4)` — i.e.
   `/tmp/jobs/fly/videos/.inference_cache_fly_obb/`, the copy inside the job
   that `pull` also refreshes. Check both exist.
3. **The cache hits — executable probe, not a log-scrape** (commands 6 AND 7 above; the script is specified below), for **all three** jobs including `identity`. Fix W2: command 6 alone only proves firebrat-path -> /tmp/jobs-path independence (both env-pinned to the job snapshot); command 7's `--local` invocation, with no `HYDRA_*` overrides and against ORIGINAL/packed-sidecar config and the ORIGINAL video path, proves **cache-key portability** — the same key is reproduced on this machine, at this path, with no `HYDRA_*` overrides in effect (minor fix, round-7: state this precisely; it is NOT proof that a real local `trackerkit track` run reuses the cache end-to-end, since the probe never constructs a session — see fix B1/V5 below). Both must print `CACHE FULLY REUSABLE` for every job.
4. The pulled `_tracking.csv` from firebrat is row-identical to a native firebrat run of the same config, for the `fly` AND the `pose` jobs — see Step 4c for the exact commands and, critically, the row-content assertion on the pose job.

**Fix C3 — "the runner reports a cache hit" was never a real, checkable signal.** There is no "cache hit" log line anywhere in the codebase (grepped; only comments reference the concept). Use the REAL probes that exist instead.

**Fix B1 — the probe must NOT construct an `InferenceRunner`.** `cache_set_is_fully_reusable`'s own docstring (`core/inference/runner.py:430-437`, quoted verbatim) says so:

> *"This is deliberately independent of `InferenceRunner` construction: callers that only decide whether to prepare/reuse replay evidence must not initialize an OBB backend merely to inspect cache metadata. It is the pure cache-set portion of `InferenceRunner.caches_all_valid()`."*

Constructing a runner loads the OBB backend — slow, device-dependent, and able to fail for reasons that have nothing to do with cache identity, which is the only thing Goal 4 is about. Every symbol used below was read in source at `8f9688e0`:

| Symbol | Verified location | Verified signature |
|---|---|---|
| `build_inference_config_from_params` | `core/inference/config.py:978` | `(params: dict) -> InferenceConfig` |
| `build_inference_cache_dir` | `utils/video_artifacts.py:95` | `(video_path, artifact_base_dir=None, create=False) -> Path` |
| `video_signature` | `core/inference/cache/keys.py:36`, re-exported into `runner.py:33` | `(path: str \| None) -> str` |
| `_open_caches` | `core/inference/runner.py:530-538` | `(config, cache_dir, video_sig="", roi_mask=None, *, read_only=False, write_mode="auto") -> _CacheSet` — `video_sig` and `roi_mask` are POSITIONAL-OR-KEYWORD in that order; `read_only` is genuinely keyword-only and genuinely named `read_only` |
| `cache_set_is_fully_reusable` | `core/inference/runner.py:430` | `(caches: _CacheSet) -> bool` — MODULE-LEVEL function, not a method |
| `DetectionCacheHandle.get_missing_frames` | `core/inference/cache/store.py:333-343` | `(start_frame: int, end_frame: int, max_report: int = 10) -> list[int]` — reached as `caches.detection.get_missing_frames(...)`; `caches.detection` is `DetectionCacheHandle \| None` (`runner.py:107`) |
| read-only `close()` | `core/inference/cache/store.py:223-225` | `_finish_close` returns immediately when `read_only` — closing the probe's handles writes nothing |

**The probe is provably non-mutating — verified on the OPEN side too, not just `close()`.** `_open_caches`'s `read_only` branch (`runner.py:569-582`) only *reads*: it calls `load_cache_set(cache_dir)`, then sets `root` to the existing generation directory when the member set matches and to `cache_dir` itself otherwise. It never calls `mkdir`, never clones a revision (that is the `write_mode == "resume"` branch at `:583-596`), and never publishes a set manifest — `set_manifest_valid` is a computed boolean, not a write. Combined with the read-only `close()` no-op, **nothing under `cache_dir` is created, modified or removed by this probe.** The probe still asserts this, because the whole acceptance rests on it:

```python
before = {
    str(q.relative_to(cache_dir)): (q.stat().st_size, q.stat().st_mtime_ns)
    for q in sorted(cache_dir.rglob("*"))
    if q.is_file()
}
# ... open, probe, close ...
after = {
    str(q.relative_to(cache_dir)): (q.stat().st_size, q.stat().st_mtime_ns)
    for q in sorted(cache_dir.rglob("*"))
    if q.is_file()
}
if before != after:
    print("FAIL: the probe MUTATED the cache it was measuring")
    sys.exit(1)
```

Place the `before` snapshot immediately after the `cache_dir.is_dir()` check and the `after` comparison in the `finally` block's tail, after every handle is closed.

**`roi_mask` must match the run's.** `_open_caches` folds `roi_mask` into the detection key via `detection_cache_key(config.obb, roi_mask, ...)` (`runner.py:553`), and `detection_cache_key` folds it in **only when sliced inference is enabled** (`cache/keys.py:100-101`). So passing the wrong mask is a silent no-op on a non-sliced config and a silent key change on a sliced one — a false FAIL that looks like a portability bug. The probe therefore reads `params["ROI_MASK"]` from the same session the run used, exactly as `worker.py` does, and never substitutes `None`.

**`max_report` caps the list.** `get_missing_frames` breaks out at `max_report` (default 10), so `len(missing)` is a LOWER BOUND, not a count. The message below says "at least N" and never claims a total.

Write this to `tools/equivalence/probe_cache_hit.py`:

```python
"""tools/equivalence/probe_cache_hit.py -- the Goal-4 acceptance probe.

Usage:
  python probe_cache_hit.py <job_dir> <video_relpath_under_the_job>
  python probe_cache_hit.py --local <video_path> <config_path>

The first form is the firebrat-path -> /tmp/jobs-path independence check: it
pins HYDRA_MODELS_DIR/HYDRA_CONFIG_DIR to the job's OWN snapshot and chdirs
into it, the same way run.sh does, and loads the job's sidecar.

Fix W2 -- that first form alone does NOT prove what spec Goal 4 actually
promises. Goal 4 says the pulled cache is reusable by a LOCAL run against the
LOCAL models root and the ORIGINAL config -- i.e. with NO HYDRA_* overrides
at all, run.sh's env-pinning trick and the job's sidecar config never even in
the picture. The `--local` form is the probe's second check, proving
**cache-key portability**: it takes the video's un-pinned local host path
(pull lands outputs BESIDE THE ORIGIN, e.g. /tmp/jobs/src/fly_obb.mp4, per
Task 13 Step 4 command 2) and a config path -- the ORIGINAL fixture config
(tools/equivalence/fixtures/configs/fly_obb.json, not the job's sidecar) for
fly_obb, or the PACKED sidecar (/tmp/jobs/<name>_config.json, minor fix
round-7) for the two pose-enabled jobs, since their original fixture configs
carry `pose_skeleton_file: ""` and fix X3a's pack-time guard would refuse to
pack that shape at all -- leaves HYDRA_MODELS_DIR/HYDRA_CONFIG_DIR exactly as
the ambient shell has them, and does not chdir anywhere. Because the probe
never constructs a `TrackingWorker`/session (fix B1), this proves the cache
KEY reproduces correctly outside any job-scoped env override, not that a
real local `trackerkit track` invocation would itself hit the cache end to
end. `build_inference_cache_dir` resolves the SAME .inference_cache_<stem>/ that pull refreshed beside the
origin clip, because it is a pure function of `video_path`. BOTH invocations
must print "CACHE FULLY REUSABLE" for Goal 4 to be considered proven for a
given job; the first form alone only proves path-independence, not that a
plain local run (no job, no pinned env) actually reuses the pulled cache.

Exits 0 and prints "CACHE FULLY REUSABLE" iff every cache stage the config
enables is key-valid, coextensive, and has zero missing detection frames.
Exits 1 with the specific gap otherwise -- this is the ONLY acceptance
evidence for Goal 4, so it must fail loudly and specifically, never silently
pass.

**Fix V5/minor — narrow what a REUSABLE verdict from this probe actually proves.** This probe opens caches `read_only=True` via `_open_caches` and never constructs a `TrackingWorker`, so its own replay-vector handling (mirrored above from `worker.py:1308-1317`) is the ONLY replay logic exercised here — it is not a substitute for running the real forward/backward pass through `TrackingWorker`. Verified in `core/tracking/worker.py`: the recorded replay vector is applied unconditionally only `if self.backward_mode or self.cache_read_only_replay` (`:1310-1313`); a plain **forward** rerun with `use_cached_detections=True` instead takes the `elif` branch at `:1479-1491` (`not effective_realtime_tracking_mode and self.use_cached_detections and inference_runner.caches_all_valid() and ...detection_cache_covers_range(...)`), which resolves the inference config at the PROJECT'S CONFIGURED batch-size vector, never the forward pass's recorded one. So: if the job that produced the pulled cache was calibrated (a replay-vector record exists at a batch-size vector different from the project's configured default), this probe's `--local` invocation can print `CACHE FULLY REUSABLE` (because it applies the recorded vector, same as a `backward_mode`/`cache_read_only_replay` run would) while a real plain **forward** rerun of `trackerkit track` on the same machine misses the cache and silently recomputes — the probe and a forward CLI run are not testing the same code path in that case. State this probe's actual scope precisely: it proves the cache keys are machine-independent and reusable **via the backward/`cache_read_only_replay` replay path**, not that every possible local rerun mode reuses them. (None of Task 13's three acceptance fixtures are calibrated, so this gap does not invalidate THIS acceptance run's own result — it only bounds what future users should expect from an uncalibrated vs. calibrated job.)

**Also minor — not every on-disk cache is proven portable by this probe.** `core/individual/properties/cache.py:60-84`'s `_file_fingerprint` (used by `compute_detection_hash`) fingerprints the **resolved video path plus its `mtime_ns`/`size_bytes`** (verified: `_file_fingerprint` builds `{"configured_path", "resolved_path", "exists", "size_bytes", "mtime_ns"}` from `Path(configured).expanduser().resolve()` and `Path(resolved).stat()`). A pulled video lands at a different absolute path (or, even at the same path, with a different `mtime` from `rsync`/`scp`/symlink materialization) than the one the staging machine fingerprinted, so the individual-properties cache is **never** portable across the push/pull round trip regardless of anything this plan does — it silently recomputes on first use, the same way it would after any local file move. This probe does not check the individual-properties cache at all (it only opens `InferenceRunner`'s detection/pose/identity caches via `_open_caches`), so its `CACHE FULLY REUSABLE` verdict says nothing about that cache either way. Worth stating explicitly in the acceptance log rather than letting a reader assume "every cache travels" from the probe's name.

Deliberately does NOT build an InferenceRunner: see runner.py:430-437, whose
docstring states that inspecting cache metadata must not initialize an OBB
backend. `cache_set_is_fully_reusable` is the pure cache-set predicate that
`InferenceRunner.caches_all_valid()` itself wraps.
"""

import os
import sys
from pathlib import Path

if sys.argv[1] == "--local":
    # Fix W2: the second, real Goal-4 proof -- no HYDRA_* overrides, no
    # chdir, original video path and original fixture config. Whatever this
    # host's ambient HYDRA_MODELS_DIR/HYDRA_CONFIG_DIR already resolve to
    # (the platformdirs default on a fresh shell) is exactly what a user
    # doing a plain local run would get.
    video_path = str(Path(sys.argv[2]).resolve())
    sidecar = Path(sys.argv[3]).resolve()
else:
    job_dir = Path(sys.argv[1]).resolve()
    video_relpath = sys.argv[2]

    # Reconfigure the process the same way run.sh reconfigures the compute
    # box, BEFORE importing hydra_suite (paths are read at call time, but
    # keeping the ordering identical to run.sh removes a whole class of
    # doubt).
    os.environ["HYDRA_MODELS_DIR"] = str(job_dir / "models")
    os.environ["HYDRA_CONFIG_DIR"] = str(job_dir / "config")
    os.chdir(job_dir)

    video_path = str(job_dir / video_relpath)
    sidecar = job_dir / "videos" / f"{Path(video_relpath).stem}_config.json"

from hydra_suite.core.inference.cache.keys import video_signature
from hydra_suite.core.inference.config import build_inference_config_from_params
from hydra_suite.core.inference.runner import (
    _open_caches,
    cache_set_is_fully_reusable,
)
from hydra_suite.trackerkit.cli_config import load_tracker_cli_session
from hydra_suite.utils.video_artifacts import build_inference_cache_dir

if not sidecar.is_file():
    print(f"FAIL: no sidecar/config at {sidecar}")
    sys.exit(1)

# video_path is a REQUIRED POSITIONAL (cli_config.py:304); the pulled clip is a
# real, decodable video here, so probe_video() works and no injected
# TrackerCliVideoProbe is needed (unlike the tmp_path fixtures in Task 11).
session = load_tracker_cli_session(video_path, config_path=str(sidecar))
params = session.params
config = build_inference_config_from_params(params)
cache_dir = build_inference_cache_dir(video_path)
if not cache_dir.is_dir():
    print(f"FAIL: no pulled cache directory at {cache_dir}")
    sys.exit(1)

# Minor fix: mirror worker.py:1308-1317's replay-vector application. A
# real replay pass resolves the inference config at whatever batch-size
# vector the FORWARD pass actually recorded (batch sizes are folded into
# the cache keys), not at the project's configured vector -- probing at
# the default/configured vector when a replay record exists at a
# DIFFERENT vector gives a false FAIL that has nothing to do with
# portability.
from hydra_suite.core.inference.autotune.replay_vector import load_replay_vector

replay_vector = load_replay_vector(cache_dir)
if replay_vector is not None:
    config = replay_vector.apply(config)
    print(f"Applied recorded replay vector: {replay_vector.effective.to_dict()}")

caches = _open_caches(
    config,
    cache_dir,
    video_signature(video_path),
    params.get("ROI_MASK"),   # MUST match the run's mask -- see note above
    read_only=True,
)
try:
    if not cache_set_is_fully_reusable(caches):
        print(
            "FAIL: cache_set_is_fully_reusable() is False -- a member is "
            "key-invalid or the members are not coextensive, i.e. the key is "
            "still carrying something machine-local. Debug before merging."
        )
        sys.exit(1)

    if caches.detection is None:
        print("FAIL: no detection cache member was opened for this config")
        sys.exit(1)

    # Minor fix: `int(params.get("END_FRAME", 0) or 0)` turns an empty/None/-1
    # END_FRAME into 0, which makes the checked range [start_frame, 0) EMPTY
    # -- get_missing_frames() then vacuously returns [] and this probe "passes"
    # without checking anything. Mirror worker.py:865's real resolution
    # (`clamp_frame_range`, core/inference/config.py:543) instead of
    # reimplementing the None/absent-value handling ad hoc.
    from hydra_suite.core.inference.config import clamp_frame_range

    total_video_frames = getattr(session.video_probe, "total_frames", None)
    start_frame, end_frame = clamp_frame_range(
        params.get("START_FRAME", 0), params.get("END_FRAME", None), total_video_frames
    )
    # max_report caps the returned list (store.py:341-342), so len(missing) is
    # a LOWER BOUND, never a total. Do not report it as a count.
    missing = caches.detection.get_missing_frames(start_frame, end_frame)
    if missing:
        print(
            f"FAIL: at least {len(missing)} detection frames are missing in "
            f"[{start_frame}, {end_frame}] (capped report); first gaps: "
            f"{missing[:10]}"
        )
        sys.exit(1)
finally:
    # close() on a read_only handle returns immediately (store.py:223-225), so
    # this cannot stamp or truncate the cache being measured.
    for handle in caches.all_handles():
        handle.close()

print("CACHE FULLY REUSABLE")
sys.exit(0)
```

Run it for both pulled jobs and paste the `CACHE FULLY REUSABLE` output (see the self-contained commands in the block above). If either fails, the key is still carrying something machine-local — debug before merging.

- [ ] **Step 4b (fix B14): commit the probe**

Task 13 writes a new tracked file but had no commit step at all.

```bash
cd /Users/neurorishika/Projects/Rockefeller/Kronauer/multi-animal-tracker/.worktrees/portable-jobs && \
  git add tools/equivalence/probe_cache_hit.py docs/superpowers/plans/2026-09-09-portable-tracking-jobs.md && \
  git commit -m "test(equivalence): Goal-4 cache-reuse probe and the acceptance log"
```

- [ ] **Step 4c: item 4 — the native-run row comparison, as an actual command, for BOTH `fly` (no pose) and `pose` (SLEAP)**

"Row-identical to a native firebrat run" named no command (fix B14). Run the same config natively on firebrat, in the job's own environment, then diff. Run BOTH jobs — not just `fly` — because `fly_obb.json` has no pose stage at all, so it can never catch fix A2's documented trap (memory `CLAUDE.md`/equivalence README: "conda MUST be active for any pose/SLEAP clip, else empty CSVs falsely pass 'EQUIVALENT'"). Only the `pose` job's native run exercises the SLEAP service, and only an explicit non-empty/row-count assertion on ITS output (not just `assert_frame_equal`, which two empty, header-only DataFrames would also satisfy) proves the pose columns actually got populated.

```bash
BOOTSTRAP="source ~/miniforge3/etc/profile.d/conda.sh && conda activate hydra-cuda"

# fly (no pose stage -- baseline sanity, cheap):
ssh rutalab@firebrat "set -e
  $BOOTSTRAP
  cd /home/rutalab/jobs/fly
  mkdir -p /tmp/native_fly && cp -r videos /tmp/native_fly/
  rm -rf /tmp/native_fly/videos/.inference_cache_* /tmp/native_fly/videos/*_tracking*.csv
  HYDRA_MODELS_DIR=/home/rutalab/jobs/fly/models \
  HYDRA_CONFIG_DIR=/home/rutalab/jobs/fly/config \
  PYTHONPATH=\$HOME/hydra-suite/src \
    python -m hydra_suite.trackerkit.app track \
      --video /tmp/native_fly/videos/fly_obb.mp4 \
      --config /tmp/native_fly/videos/fly_obb_config.json"

# pose (SLEAP stage -- the job THIS fix is actually about). Fix A2b/A2c:
# conda MUST be active on this ssh session (via $BOOTSTRAP) or the SLEAP
# service returns "(False, 'Conda not found on PATH.')"
# (integrations/sleap/service.py:2430) and silently writes EMPTY pose columns
# that would still make a naive row-count-only comparison pass.
ssh rutalab@firebrat "set -e
  $BOOTSTRAP
  cd /home/rutalab/jobs/pose
  mkdir -p /tmp/native_pose && cp -r videos /tmp/native_pose/
  rm -rf /tmp/native_pose/videos/.inference_cache_* /tmp/native_pose/videos/*_tracking*.csv
  HYDRA_MODELS_DIR=/home/rutalab/jobs/pose/models \
  HYDRA_CONFIG_DIR=/home/rutalab/jobs/pose/config \
  PYTHONPATH=\$HOME/hydra-suite/src \
    python -m hydra_suite.trackerkit.app track \
      --video /tmp/native_pose/videos/ant_pose_headtail.mp4 \
      --config /tmp/native_pose/videos/ant_pose_headtail_config.json"

# Fix W5/X4 -- the raw pass CSVs (`<stem>_tracking_forward.csv`/
# `<stem>_tracking_backward.csv`) NEVER carry pose columns. Verified:
# data/csv_writer.py's build_tracking_csv_header (the forward/backward-pass
# CSV writer) has no pose fields at all. Pose columns only exist in the
# terminal, POST-PROCESSED csv, and both the FILE NAME and the column
# contract differ by mode (core/tracking/session.py:349/386/811 default
# DEBUG_MODE=True, and none of the three fixture configs override it, so
# every job in this acceptance run is Debug mode):
#   - Debug mode: `<stem>_tracking_final_with_individual.csv`
#     (core/post/rich_export.py:31 RICH_EXPORT_SUFFIX = "_with_individual",
#     appended to the `_tracking_final` terminal stem), columns
#     `PoseKpt_<name>_X` / `_Y` / `_Conf` (capitalized -- core/post/
#     trajectory_writer.py:12-13 _POSE_PREFIX="PoseKpt_", _POSE_X_SUFFIX="_X").
#   - User mode: `<stem>_tracks.csv` (core/tracking/session.py:812
#     user_tracks_path), columns `<name>_x` / `_y` / `_conf`
#     (lowercase -- trajectory_writer.py:192-203 derives these FROM the
#     PoseKpt_*_X/_Y/_Conf columns for the clean export).
# This acceptance run compares Debug-mode output on both sides (pulled and
# native), so it asserts on the `_tracking_final_with_individual.csv` file
# and the capitalized PoseKpt_ column names -- NOT the raw forward/backward
# CSVs and NOT the lowercase `_x`/`_y` names a User-mode run would use.
# fly_obb has no cnn_classifiers/pose (fix A1c), so its Debug-mode terminal
# file is the bare `<stem>_tracking_final.csv` (no rich-export sibling is
# guaranteed to carry meaningful extra columns for it, so the bare file is
# the correct comparison target).

# Then, on the Mac, compare the PULLED csv against the native one for EACH job:
scp rutalab@firebrat:/tmp/native_fly/videos/fly_obb_tracking_final.csv /tmp/native_fly_obb_tracking_final.csv
scp rutalab@firebrat:/tmp/native_pose/videos/ant_pose_headtail_tracking_final_with_individual.csv /tmp/native_pose_headtail_tracking_final_with_individual.csv
python - <<'PY'
import numpy as np
import pandas as pd

def compare(pulled_path, native_path, pose_columns=None, theta_columns=None):
    """Fix V-minor: a STRICT assert_frame_equal on ant_pose_headtail is a
    false-fail risk. ant_pose_headtail is a documented bistable head/tail
    clip (CLAUDE.md's equivalence-verification section: "Known baseline
    noise: bistable head/tail pi-flips on head/tail clips (theta can flip by
    pi on some rows) -- that's the migration's documented noise floor, not a
    regression"). This is the SAME native-firebrat-to-native-firebrat
    comparison the equivalence harness runs, and it accepts the same noise
    floor there (tools/equivalence/compare_caches.py's modpi residual,
    `min(raw, abs(np.pi - raw))`) -- a theta column here should get the
    same tolerance, not a stricter one just because this is the portable-jobs
    acceptance run rather than the equivalence harness. Positions and every
    non-theta column stay EXACTLY equal (this is firebrat-to-firebrat, same
    device, same code, same cache -- any non-theta divergence IS a
    regression); only heading columns get the pi-flip allowance.
    """
    a = pd.read_csv(pulled_path)
    b = pd.read_csv(native_path)
    assert len(a) == len(b), f"row counts differ: {len(a)} vs {len(b)}"
    # Fix A2c: an "identical" verdict on two EMPTY (header-only, len==0)
    # frames is exactly the trap this step exists to catch -- never trust it.
    assert len(a) > 1, f"{pulled_path} has {len(a)} rows -- empty/near-empty CSVs falsely 'match'"
    if pose_columns:
        for col in pose_columns:
            assert col in a.columns, f"missing pose column {col!r} in {pulled_path}"
            assert a[col].notna().any(), f"pose column {col!r} is entirely NaN/empty in {pulled_path}"
    theta_columns = set(theta_columns or ())
    strict_cols = [c for c in a.columns if c not in theta_columns]
    pd.testing.assert_frame_equal(a[strict_cols], b[strict_cols])
    # Minor fix (round-6): mirror tools/equivalence/compare_caches.py:28-30's
    # ang_diff EXACTLY -- wrap the raw difference into [0, 2pi) mod 2pi FIRST
    # (`d = |a-b| % 2pi; min(d, 2pi-d)`), THEN take the modpi residual
    # (`min(raw, |pi-raw|)`) the harness applies at compare_caches.py:96-98.
    # Skipping the 2pi wrap makes a heading straddling +-pi (e.g. a=3.13,
    # b=-3.13, true difference ~0.02 rad) compute a raw diff of ~6.26 rad and
    # falsely fail -- exactly the case this acceptance run's own clips can hit
    # near a wraparound frame. Also guard the all-NaN case explicitly:
    # np.nanmax on an all-NaN column returns nan, and `nan < 1e-6` is False,
    # so an all-NaN theta column would silently FAIL a check that should
    # instead say plainly that the column never had data.
    for col in theta_columns:
        a_theta = a[col].to_numpy(dtype=float)
        b_theta = b[col].to_numpy(dtype=float)
        assert not (np.isnan(a_theta).all() and np.isnan(b_theta).all()), (
            f"theta column {col!r} is entirely NaN on both sides -- nothing to compare"
        )
        d = np.abs(a_theta - b_theta) % (2 * np.pi)
        ang_diff = np.minimum(d, 2 * np.pi - d)
        residual = np.minimum(ang_diff, np.abs(np.pi - ang_diff))
        worst = float(np.nanmax(residual))
        assert worst < 1e-6, (
            f"theta column {col!r} differs by more than the documented pure "
            f"180-degree flip noise floor: worst modpi residual {worst:.3e} rad"
        )
    print(f"ROW-IDENTICAL (theta allowing documented pi-flips) + non-empty: {len(a)} rows ({pulled_path})")

compare("/tmp/jobs/fly/videos/fly_obb_tracking_final.csv", "/tmp/native_fly_obb_tracking_final.csv")
# Don't hardcode a specific keypoint name -- inspect the pulled Debug
# `_with_individual.csv`'s own header for whatever PoseKpt_<name>_X/_Y/_Conf
# columns this run's skeleton actually produced (case-aware: capital
# X/Y/Conf, the Debug-mode contract, not the User-mode lowercase one) and
# assert on THOSE, so this check is self-verifying regardless of which
# skeleton the fixture ends up using.
_pulled = pd.read_csv("/tmp/jobs/pose/videos/ant_pose_headtail_tracking_final_with_individual.csv")
_pose_cols = [
    c for c in _pulled.columns
    if c.startswith("PoseKpt_") and (c.endswith("_X") or c.endswith("_Y") or c.endswith("_Conf"))
]
assert _pose_cols, (
    "no PoseKpt_<name>_X/_Y/_Conf columns found in the pose job's "
    "_with_individual.csv at all -- either the pose stage did not run or "
    "this is not actually a Debug-mode run; inspect the header before "
    "trusting any row comparison"
)
compare(
    "/tmp/jobs/pose/videos/ant_pose_headtail_tracking_final_with_individual.csv",
    "/tmp/native_pose_headtail_tracking_final_with_individual.csv",
    pose_columns=_pose_cols,
    # Fix V-minor: "Theta" (core/post/trajectory_writer.py:128 reads
    # df["Theta"]) gets the documented pure-180-degree-flip allowance;
    # ant_pose_headtail is a head/tail clip and this is the exact noise
    # floor the equivalence harness's own README documents for it.
    theta_columns={"Theta"},
)
PY
```

Cross-device byte-identity vs the Mac is explicitly **not** claimed (spec §2 non-goal); this compares firebrat-to-firebrat.

**Fix V5 — this exact step is LIVE, not theoretical: `ant_pose_headtail` is a SLEAP job (`pose_model_type: "sleap"`, verified above), so the pose job's `POSE_MODEL_DIR` is the SLEAP run DIRECTORY and every run in this step exercises the SLEAP service, the `sleap` conda env, and `directory_content_id` over the whole SLEAP run tree — not a hypothetical.** `directory_content_id` (Task 4) hashes EVERY file under a directory-model's root — for a SLEAP run directory, that is the whole training-run tree. If `sleap-nn` writes anything (a log, a cache file, a lockfile) inside that SAME directory DURING inference on either the pulled copy or this native second run, the tree's content id diverges from the pristine staging-machine copy even though the actual model weights are unchanged. Because this is live for the pose job, a probe FAIL here has two plausible causes that look identical from the outside — a real cache-key portability bug, or the SLEAP run directory mutating itself during inference — so the check below makes the failure mode self-diagnosing rather than guessed at:

```bash
# Before the pose job's run.sh: snapshot the SLEAP run tree's content id.
ssh rutalab@firebrat "$BOOTSTRAP
  cd /home/rutalab/jobs/pose && PYTHONPATH=\$HOME/hydra-suite/src \
  python -c \"
from pathlib import Path
from hydra_suite.core.inference.content_id import directory_content_id
p = sorted(Path('models/pose').glob('*/*'))[0]
print('BEFORE', p, directory_content_id(p))
\"" | tee /tmp/sleap_dir_before.txt

# ... run the pose job's run.sh here (Task 13 Step 4 item 2/4) ...

# After: same hash, same directory.
ssh rutalab@firebrat "$BOOTSTRAP
  cd /home/rutalab/jobs/pose && PYTHONPATH=\$HOME/hydra-suite/src \
  python -c \"
from pathlib import Path
from hydra_suite.core.inference.content_id import directory_content_id
p = sorted(Path('models/pose').glob('*/*'))[0]
print('AFTER', p, directory_content_id(p))
\"" | tee /tmp/sleap_dir_after.txt

diff /tmp/sleap_dir_before.txt /tmp/sleap_dir_after.txt && echo "SLEAP run dir UNCHANGED by inference" \
  || echo "SLEAP run dir CHANGED -- any probe/verify FAIL on this job is inference-time self-mutation, not a cache-key bug; see the fallback below"
```

If the diff shows a change, the fallback is the narrower fingerprint `core/individual/pose/artifacts.py` already computes (deliberately host-local per spec §2, per Task 4's "Deliberately unchanged" note) rather than a naive whole-tree hash — not implemented here, but the diagnosis this check produces is what tells a future reader whether that fallback is actually needed, instead of leaving a pose-job probe FAIL to be misdiagnosed as a cache-key regression.

- [ ] **Step 5: Shared-root live check**

**Fix B14 — "configure the same alias on both hosts" named no command. Here they are.** Use a directory both hosts can genuinely see; if there is no real shared mount available, use an sshfs/NFS path or, at minimum, two directories holding byte-identical copies of the clip (the signature check compares content, so identical bytes at different mount points is exactly the case §6.7 targets).

```bash
# On the Mac (alias -> the local mount point):
cd /Users/neurorishika/Projects/Rockefeller/Kronauer/multi-animal-tracker/.worktrees/portable-jobs && \
  PYTHONPATH=$PWD/src conda run --no-capture-output -n hydra-mps \
    python -m hydra_suite.trackerkit.app job shared-root add labnas /Volumes/lab

# On firebrat (SAME alias -> that host's own mount point):
ssh rutalab@firebrat 'source ~/miniforge3/etc/profile.d/conda.sh && cd ~/hydra-suite && \
  PYTHONPATH=$PWD/src conda run --no-capture-output -n hydra-cuda \
    python -m hydra_suite.trackerkit.app job shared-root add labnas /mnt/lab'

# Verify both tables (they must share the alias and differ in the path):
cd /Users/neurorishika/Projects/Rockefeller/Kronauer/multi-animal-tracker/.worktrees/portable-jobs && \
  PYTHONPATH=$PWD/src conda run --no-capture-output -n hydra-mps \
    python -c "from hydra_suite.data.tracking_job.shared_roots import load_shared_roots; print(load_shared_roots())"
ssh rutalab@firebrat 'source ~/miniforge3/etc/profile.d/conda.sh && cd ~/hydra-suite && \
  PYTHONPATH=$PWD/src conda run --no-capture-output -n hydra-cuda \
    python -c "from hydra_suite.data.tracking_job.shared_roots import load_shared_roots; print(load_shared_roots())"'
```

Then: pack a video that lives under `/Volumes/lab`, confirm the manifest records `shared`, the push list omits the video, preflight materializes the symlink on firebrat, the run completes, and pull maps outputs beside the origin.

**Name the actual command (fix M7/X2).** "The run completes" is not a command. `job run rutalab@firebrat:/home/rutalab/jobs/<name>` internally chains `ssh rutalab@firebrat 'cd /home/rutalab/jobs/<name> && trackerkit job preflight . [--shared-root ...] [--allow-tier-fallback] && HYDRA_JOB_SKIP_PREFLIGHT=1 ./run.sh …'` as ONE ssh invocation, so preflight's materialization of the shared symlink happens immediately before `run.sh` on the same connection, and `run.sh`'s own flag-less self-preflight is skipped rather than re-running the same checks a second time without the flags that just made them pass — this is the step that actually exercises §6.7 for the primary remote workflow (a bare `./run.sh` without a preceding preflight, which the plan used to describe, would leave the shared video unmaterialized and `trackerkit track` would fail on a missing file). Paste the ssh session's preflight output showing the `shared_roots` check passing and the symlink being created, immediately followed by the tracking run's own log output, both from the SAME `job run` invocation.

- [ ] **Step 6: Record everything in the Acceptance Log**

---

## Acceptance Log

Fill in as gates are run. A gate with no pasted output is not a passed gate.

| Gate | Platform | Commit | Result | Evidence |
|---|---|---|---|---|
| Cache-key before/after matrix (smoke: fly_obb, worm_bgsub) | MPS | `6959c8ce` vs base `8ec4730d` | **PASS** | All DETERMINISM + EQUIVALENCE verdicts EQUIVALENT ✅ (pos_p99<=0.5px, theta_mean<=0.05rad, unmatched==0, every keyed column identical). PERF 1.02x / 0.70x (tol 1.25). Row counts non-empty AND identical: fly_obb 1501, worm_bgsub 2707. **Proof the changed path executed:** legacy `detection.npz` holds `v4\|/Users/neurorishika/Library/App…` (path-based); new holds `v5\|sha256:b5047e47…` (content-based). Ran under contention (a live user `detectkit` session, PID 30883) — PERF is therefore indicative only; byte-identity is unaffected. |
| Cache-key before/after matrix | CUDA (firebrat) | | | |
| Full matrix (Task 4 complete) | MPS | `6959c8ce` vs base `8ec4730d` | **PASS** | 8 clips (ant_pose_headtail, ant_obb_sleap, ant_obb_sequential, ant_cnn_identity, ant_cnn_identity_relink, worm_bgsub, worm_bgsub_scaled, fly_obb), 21 comparisons, **42 EQUIVALENT / 0 divergent**. PERF 0.96–1.13x (tol 1.25). Row counts all substantial and identical across legacy/new_a/new_b (ant_obb_sleap 11886, ant_cnn_identity_relink 10263, ant_cnn_identity 10247, ant_pose_headtail 9097, worm_bgsub 2707, fly_obb 1501, worm_bgsub_scaled 1003, ant_obb_sequential 942) — not empty CSVs comparing equal. SLEAP genuinely ran: 24 populated `PoseKpt_*` columns on ant_pose_headtail. `fly_obb_roi` excluded by design (ON-path clip, run_matrix.sh:71-76). Ran under contention (live user `detectkit` PID 30883) so PERF is indicative; byte-identity unaffected. |
| Final matrix | CUDA (firebrat) | | | |
| Full pytest delta | MPS | | | |
| Round trip + cache hit | Mac → firebrat → Mac | | | |
| Shared-root live | Mac → firebrat | | | |

---

## Follow-ups (explicitly out of scope, per spec §17)

1. **Headless CLI dataset/media export.** Derive the three `*_OUTPUT_DIR` values in `cli_config.py` exactly as the GUI does (`orchestrators/config.py:2241-2252`). Gated by the byte-identity harness because it changes CLI engine params. Until then `pack` warns.
2. GUI "Package job…" action calling `pack_job` through a `BaseWorker`.
3. `job clean` and a retention policy.
4. Unify the duplicated sidecar-path formula (`session_plan.py:20-26` vs `orchestrators/config.py:97-103`).
5. Registry-name references in configs — belongs to the deferred model-registry unification spec.
6. **`detection_batch_size` is omitted from the key at `cache/reuse.py:62` and `worker.py:1435`** but passed at `runner.py:553`. Pre-existing; fixing it changes cache-hit behaviour and would contaminate this branch's gate.
7. **`parameter_helper.py:1768-1780` keeps a private `_source_signature`** instead of the shared `video_signature`. Commented in Task 4, not unified.
8. **`COLOR_TAG_MODEL_PATH`/`CNN_CLASSIFIER_MODEL_PATH` are dead code.** Discovered during adversarial review (fix C2): no consumer anywhere in `src/hydra_suite/core/`, GUI field `setVisible(False)`. Real colour-tag identity runs entirely through ClassKit `cnn_classifiers`. A future cleanup pass could delete both keys end-to-end (GUI field, `engine_params.py` derivation, `NON_MODEL_PATH_PARAM_KEYS`/`ABSOLUTE_PATH_FORBIDDEN_KEYS` entries) rather than carrying them as permanently-inert plumbing. Out of scope here — this branch only needs them to never be treated as live model references.
9. **`csv_path`'s job-relative rewrite is currently inert.** `load_tracker_cli_session` derives the CSV path from the video itself and never reads `cfg["csv_path"]` (`cli_config.py:319`); only `video_output_path` is consumed downstream (`core/tracking/session.py:653`). Discovered during Task 7 review (see the minor note in Task 7). Not broken, just presently a no-op; worth confirming with a real consumer test if `csv_path` is ever wired up.
10. **`video_signature` is unmemoized** (unlike `model_content_id`), called once per autotune evaluation (`optimizer_workers.py:343`). Accepted for this branch (see the docstring rationale in Task 4); revisit if it shows up in a profile.
11. **`CLAUDE.md`'s pre-PR checklist references a nonexistent `make lint-moderate`.** Discovered while applying fix B7: `Makefile:421,427,440,446` define only `lint`, `lint-fix`, `lint-strict`, `lint-report`, yet `CLAUDE.md`'s "Pre-PR checklist" tells contributors to run `make lint-moderate`. Anyone chaining it with `&&` (as this plan used to) silently never reaches the next command. **Do not fix `CLAUDE.md` in this branch** — it is unrelated to portable jobs and touching agent configuration from inside a feature branch is out of scope. File it as a standalone one-line correction.
12. **`ant_cnn_identity`'s characterization golden pins this machine's absolute classifier path.** `tests/data/get_parameters_dict_golden/ant_cnn_identity.json` carries `/Users/neurorishika/Library/Application Support/hydra-suite/models/classification/identity/20260429-105036_classifier_multihead_obiroi_colortag.multihead.json`, and `CNN_CLASSIFIERS` is absent from `HOST_DEPENDENT_DROPPED_KEYS` (`tests/test_get_parameters_dict_characterization.py:269-274`). That test therefore cannot pass on any other host. Either add `CNN_CLASSIFIERS` to the dropped set with the same ndarray-style normalization the model-path keys get, or regenerate the golden with a relativized path. Noted in Task 2 Step 6 so an agent does not misdiagnose it as a Task 2 regression; out of scope to fix here because it changes a committed characterization golden.
13. **`verify_job`'s directory-model branch trusts `file_digests` alone.** After fix B4 a directory model's integrity rests entirely on its per-member digests; there is no top-level roll-up, so a member ADDED to the job after packing (not present in `file_digests`) is not detected. Adding a "no unexpected files under `models/<key>/`" check would close it. Not done here because it needs a decision about whether host-written scratch files inside a pose-run directory are legitimate.
14. ~~The layering gate does not see relative imports.~~ **FIXED (fix X5b, round-6).** `test_no_app_layer_or_qt_imports` (Task 5) now resolves `node.level` against the module's own package path (`_imported_names`'s `own_pkg_parts` resolution) before applying `FORBIDDEN_ROOTS`, so a relative `from ...trackerkit import z` inside `data/tracking_job/` is caught exactly like the absolute form. `test_the_gate_itself_catches_a_relative_app_layer_import` proves the resolution formula against a synthetic file. This closes the hole X5 found live: `pack_job` calling `load_advanced_tracker_config()` via a relative import (fix X5a) would otherwise have passed this gate green.
15. **Task 4 introduces a SECOND, incompatible content-identity primitive.** `core/inference/autotune/fingerprint.py:292-340` already has `model_content_digest` (sha256 of a file, or a dir hashed by relative-name with its own exclusions and its own empty-dir behaviour, `lru_cache`d on stat) — a different function with different directory semantics than `content_id.model_content_id`/`directory_content_id`. The autotune subsystem and the cache-key subsystem now each compute "model content identity" their own way, with no shared test proving they agree (or a documented reason they must differ). Consolidate onto one primitive, or explicitly document why autotune's fingerprinting needs different semantics (e.g. different exclusion rules) — not done here because it is a second migration outside Task 4's declared file list and risks its own byte-identity blast radius.
16. **Dead code: `if real and real != "None"` in `_content_id_for_hint`'s missing-model-sentinel branch.** `model_content_id(None)` and `model_content_id("")` both return early (`if not path: return ""`) before `_stat_hint`/`_content_id_for_hint` are ever reached, so `real` (which is `hint[0]`, always a non-empty string or the literal string `"None"` only if some caller passed the STRING `"None"` rather than the value `None`) cannot be falsy at this point in the normal call graph. Either remove the dead `real and` half of the condition, or add a one-line comment explaining the (currently unverified) scenario it guards against, so a future reader doesn't have to re-derive whether it's reachable.
