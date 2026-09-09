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

- **Baselines to protect** (measured on `hydra-mps` @ `8f9688e0`): `tests/test_inference_cache_keys.py tests/test_inference_cache_chunked.py` = **111 passed**; `tests/test_gui_cli_param_equivalence.py tests/test_get_parameters_dict_characterization.py tests/test_paths.py` = **27 passed**. Any task reducing these without an explicit, justified deletion is a regression.
- **Run the FULL relevant suite every task**, never a subset chosen by apparent relevance — the reflective contract guards (`test_get_parameters_dict_characterization.py`, the new Task 3 guard) break on ANY field addition. This is memory `feedback_run_contract_guards_after_field_additions`.
- **Dependency direction is a hard gate.** `src/hydra_suite/data/tracking_job/` must import ONLY from `data`, `core`, `training.model_publish`, `paths`, and the stdlib. It must never import `trackerkit`, `classkit`, `detectkit`, `posekit`, `refinekit`, `filterkit`, `widgets`, or PySide6. Task 5 adds an automated test asserting this.
- **Never import from `legacy/`.**
- **The remote never mutates `hydra_job.json`.** `run` appends to `logs/runs.jsonl`; `pull` merges. This is what makes `push` a pure input sync.
- **`rsync` is never reimplemented.** `transport.py` builds argv lists and shells out. Never pass `--delete`.
- **`CACHE_SCHEMA_VERSION` goes 4 → 5 exactly once**, in Task 4. No converter — caches are derived data.
- Format before every commit: `make format`. Lint gate: `make lint-moderate`.
- Activate the env first: `conda activate hydra-mps`. Before any heavy run, kill stale `sleap`/`hydra` processes; **never** touch a process that is not sleap/hydra.
- Commit after every task. Do not squash tasks together.
- **Never invoke bare `trackerkit` (or bare `hydra`) from inside this worktree for any acceptance/E2E step.** Verified: `hydra_suite.__file__` resolves to MAIN's editable install here, not `.worktrees/portable-jobs/src` — a bare console-script invocation silently exercises unmodified `main` code and reports false confidence about the branch under test. Every acceptance/CLI-smoke command in this plan must instead be `PYTHONPATH=<worktree>/src python -m hydra_suite.trackerkit.app job ...` (this matches the repo's known PYTHONPATH gotcha, memory `feedback_equivalence_pythonpath_gotcha`). The equivalence fixture clips (`tools/equivalence/fixtures/clips/*.mp4`) are also gitignored and absent from a fresh worktree — run `bash tools/equivalence/fixtures/fetch_fixtures.sh` before any step that references them.
- **Verification box is `firebrat`** (`rutalab@firebrat`, RTX 4090 idle, conda at `~/miniforge3`, envs `hydra-cuda` + `sleap`, rsync 3.2.7, 1.6 TB free, equivalence fixtures already present at `~/hydra-suite/tools/equivalence/fixtures/`). **`courtship` is running a live `trackerkit` job — do not use it and do not kill anything on it.**

---

## File structure

| File | Responsibility |
|---|---|
| `src/hydra_suite/paths.py` | + `HYDRA_MODELS_DIR` override on `get_models_dir()`; + `print_paths()` line (Task 1) |
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

### Task 1: `HYDRA_MODELS_DIR` override

**Files:**
- Modify: `src/hydra_suite/paths.py:1-16` (docstring), `:128-132` (`get_models_dir`), `:202-228` (`print_paths`)
- Test: `tests/test_paths_models_dir_override.py` (create)

**Interfaces:**
- Consumes: nothing.
- Produces: `get_models_dir()` honours `$HYDRA_MODELS_DIR`. Everything routing through it (`model_paths.get_models_root_directory()`, `model_publish.get_models_root()`, `_registry_path()`, SAM3 checkpoint root) follows automatically.

- [ ] **Step 1: Write the failing test**

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

- [ ] **Step 2: Run test to verify it fails**

Run: `conda activate hydra-mps && python -m pytest tests/test_paths_models_dir_override.py -v`
Expected: FAIL — `get_models_dir()` returns the data-dir path, ignoring the env var.

- [ ] **Step 3: Implement**

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

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_paths_models_dir_override.py tests/test_paths.py -v`
Expected: all PASS.

- [ ] **Step 5: Grep gate — no module may bypass `get_models_dir()`**

Run:
```bash
grep -rn 'get_data_dir()[[:space:]]*/[[:space:]]*"models"' src/ && echo "VIOLATION" || echo "clean"
```
Expected: `clean`. (Known non-violations, do not touch: `paths_migrate.py:31` uses `repo_root / "models"` as a migration *source*; `posekit/gui/main_window.py:4608` uses a SLEAP training-run root; ClassKit/DetectKit `artifacts/models` are per-project.)

- [ ] **Step 6: Document**

In `docs/getting-started/installation.md`, in the environment-variable table (near the existing `HYDRA_DATA_DIR`/`HYDRA_CONFIG_DIR` rows), add:

```markdown
| `HYDRA_MODELS_DIR` | Relocates only the models root (`model_registry.json` + published models), leaving engine artifacts and calibration profiles on the host data dir. Set automatically by a packed job's `run.sh`. |
```

- [ ] **Step 7: Commit**

```bash
make format
git add src/hydra_suite/paths.py tests/test_paths_models_dir_override.py docs/getting-started/installation.md
git commit -m "feat(paths): HYDRA_MODELS_DIR relocates the models root independently of the data dir"
```

---

### Task 2: Close the two save-side path leaks — and the load-side asymmetry behind one of them

**Files:**
- Modify: `src/hydra_suite/trackerkit/engine_params.py:1023-1024` and `:1591` (load-side resolve)
- Modify: `src/hydra_suite/trackerkit/gui/orchestrators/config.py:2096` (`color_tag_model_path`), `:2072-2074` (`cnn_classifiers`)
- Test: `tests/test_config_model_path_portability.py` (create)

**Interfaces:**
- Consumes: nothing from earlier tasks.
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
    """A new model role added without classification fails HERE, loudly."""
    params = build_engine_params(_everything_on(models_root), runtime=_runtime())
    classified = (
        set(MODEL_FILE_PARAM_KEYS)
        | set(MODEL_DIR_PARAM_KEYS)
        | set(MODEL_LIST_PARAM_KEYS)
        | set(NON_MODEL_PATH_PARAM_KEYS)
    )
    path_ish = {
        key
        for key in params
        if key.endswith("_PATH") or key.endswith("_DIR") or key.endswith("_FILE")
    }
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
                yield ModelReference(role=key, path=value, kind="directory")

    # CNN classifiers: gated ONLY on the list being non-empty — never on
    # ENABLE_IDENTITY_ANALYSIS. See the module-level comment for why.
    for key, field in MODEL_LIST_PARAM_KEYS.items():
        for entry in params.get(key, []) or []:
            value = str((entry or {}).get(field, "") or "").strip()
            if value:
                yield ModelReference(role=key, path=value, kind="file")
```

Add `Iterator` to the `typing` imports and `dataclass` to the `dataclasses` import at the top of the file if not already present.

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

**Files:**
- Create: `src/hydra_suite/core/inference/content_id.py`
- Modify: `src/hydra_suite/core/inference/cache/base.py` (`CacheKey`, `CACHE_SCHEMA_VERSION`)
- Modify: `src/hydra_suite/core/inference/cache/keys.py` (`_model_signature`, `video_signature`, all six key builders, `_mtime` removal)
- Modify: `src/hydra_suite/core/inference/cache/reuse.py:21`, `cache/reader.py:20,33` (placeholder keys)
- Modify: `src/hydra_suite/detectkit/jobs/prediction_cache.py:31-66`, `src/hydra_suite/detectkit/sidecars/operations.py:21-35`
- Modify: `src/hydra_suite/trackerkit/gui/dialogs/parameter_helper.py:1768-1780` (comment only)
- Test: `tests/test_cache_content_identity.py` (create); update `tests/test_inference_cache_keys.py`, `tests/test_inference_cache_chunked.py`, `tests/test_pose_cache_empty_frame.py`

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces:
  - `content_id.file_content_id(path: str) -> str` → `"sha256:<hex>"`, `""` for a missing/empty path.
  - `content_id.directory_content_id(path: str) -> str` → `"dirsha256:<hex>"`.
  - `content_id.model_content_id(path: str) -> str` → dispatches file/dir; memoized per process on `(realpath, size, mtime_ns)`.
  - `content_id.video_signature(path: str | None) -> str` → `"{size}:{sha256(head8MiB‖tail8MiB)[:32]}"`.
  - `CacheKey(schema_version: int, model_id: str, config_hash: str)` with `as_string()` = `f"v{schema_version}|{model_id}|{config_hash}"`.

**This is the only slice that can break byte-identity. It ships alone, with its own full equivalence matrix on both platforms.**

**Scope decision (user-confirmed):** DetectKit migrates onto the same content-based `CacheKey`. Its sidecar payload shape (`operations.py:21-35`, which pins the exact field-name set) is bumped so old payloads are rejected loudly rather than misread. **Correction:** this is not an on-disk file format — `operations.py` is the in-process, parent→sidecar-process IPC payload used to hand a cache key across a subprocess boundary within one run, not a value read back from disk in a later session. So "sidecars written before v5 must be regenerated" overstates the blast radius: there is no persisted-on-disk artifact from a prior run that this would orphan. The rejection is still correct (a stale in-flight payload from an old, unrestarted sidecar process should fail loudly rather than being silently misread as v5), but soften the message accordingly — see the corrected wording in Step 6.

**Deliberately unchanged:** `core/individual/pose/artifacts.py:71` `path_fingerprint_token` embeds the resolved absolute path, but it guards *export-artifact validity* (ONNX/TensorRT/CoreML reuse) which is host-local by design (spec §2 non-goal) and feeds **no** `CacheKey`. Leave it. Verified: its only consumers are `pose/backends/{sleap,vitpose,yolo}.py` artifact signatures.

- [ ] **Step 1: Write the failing test**

Create `tests/test_cache_content_identity.py`:

```python
"""Cache identity is content-based: same bytes anywhere == same key."""

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
    assert len([c for c in calls if c == str(m)]) == 1


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

Add to `tests/test_inference_cache_keys.py` (the OBB-specific regressions — this is spec correction #1):

```python
def test_obb_detection_key_is_identical_for_the_same_model_at_two_paths(tmp_path):
    """config_hash must NOT carry the model path (keys.py _model_signature)."""
    from hydra_suite.core.inference.cache.keys import detection_cache_key

    a = tmp_path / "one" / "obb.pt"
    b = tmp_path / "two" / "obb.pt"
    a.parent.mkdir(parents=True)
    b.parent.mkdir(parents=True)
    a.write_bytes(b"obb-weights")
    shutil.copy2(a, b)
    os.utime(b, (1, 1))
    key_a = detection_cache_key(_direct_obb_config(str(a)), None)
    key_b = detection_cache_key(_direct_obb_config(str(b)), None)
    assert key_a.as_string() == key_b.as_string()


def test_sequential_obb_key_is_identical_for_the_same_pair_at_two_paths(tmp_path):
    from hydra_suite.core.inference.cache.keys import detection_cache_key

    def _pair(root):
        root.mkdir(parents=True, exist_ok=True)
        (root / "detect.pt").write_bytes(b"d")
        (root / "obb.pt").write_bytes(b"o")
        return str(root / "detect.pt"), str(root / "obb.pt")

    d1, o1 = _pair(tmp_path / "one")
    d2, o2 = _pair(tmp_path / "two")
    key_a = detection_cache_key(_sequential_config(d1, o1), None)
    key_b = detection_cache_key(_sequential_config(d2, o2), None)
    assert key_a.as_string() == key_b.as_string()


def test_sequential_key_changes_when_either_model_changes(tmp_path):
    from hydra_suite.core.inference.cache.keys import detection_cache_key

    root = tmp_path / "m"
    root.mkdir()
    detect, obb = root / "detect.pt", root / "obb.pt"
    detect.write_bytes(b"d")
    obb.write_bytes(b"o")
    base = detection_cache_key(_sequential_config(str(detect), str(obb)), None).as_string()
    obb.write_bytes(b"o2")
    assert detection_cache_key(_sequential_config(str(detect), str(obb)), None).as_string() != base


def test_v4_cache_on_disk_is_rejected_and_rebuilt(tmp_path):
    """Fix M4: a literal-string comparison is tautological — it proves nothing
    about the actual store. Exercise the real reader/store path: write a v4
    on-disk cache entry, then prove the store treats it as absent (rejected)
    and a v5 write follows (rebuilt), per spec §7b.6 item 3 ("rejected AND
    rebuilt")."""
    from hydra_suite.core.inference.cache import base as cache_base
    from hydra_suite.core.inference.cache.reader import DetectionCacheReader
    from hydra_suite.core.inference.cache.store import DetectionCacheStore

    cache_dir = tmp_path / ".inference_cache_clip"
    cache_dir.mkdir()
    stale_v4 = cache_base.CacheKey(schema_version=4, model_id="/abs/path/model.pt", config_hash="bb")
    store_v4 = DetectionCacheStore(cache_dir, cache_key=stale_v4)
    store_v4.write_frame(0, boxes=[])
    store_v4.close()

    v5_key = cache_base.CacheKey(schema_version=cache_base.CACHE_SCHEMA_VERSION, model_id="sha256:aa", config_hash="bb")
    reader = DetectionCacheReader(cache_dir, cache_key=v5_key)
    assert not reader.is_valid_for(v5_key), "a v4 key must not validate a v5 reader"

    store_v5 = DetectionCacheStore(cache_dir, cache_key=v5_key)
    store_v5.write_frame(0, boxes=[])
    store_v5.close()
    rebuilt = DetectionCacheReader(cache_dir, cache_key=v5_key)
    assert rebuilt.is_valid_for(v5_key), "a fresh v5 write must be usable"
```

(Adapt the exact `DetectionCacheReader`/`DetectionCacheStore` constructor and validity-check method names to whatever `cache/reader.py`/`cache/store.py` actually expose — the point non-negotiable here is that the assertion goes through the real store/reader round trip, not a bare string comparison of two `CacheKey.as_string()` values.)

Write `_direct_obb_config` / `_sequential_config` helpers in that test file mirroring the existing fixtures already there.

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
_EXCLUDED_DIR_NAMES = {".hydra-runtime-artifacts"}


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


def _stat_hint(path: str) -> tuple[str, int, int]:
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
    """
    try:
        st = os.stat(path)
        return (os.path.realpath(path), st.st_size, st.st_mtime_ns)
    except OSError:
        return (str(path), -1, -1)


@lru_cache(maxsize=256)
def _content_id_for_hint(hint: tuple[str, int, int]) -> str:
    real = hint[0]
    p = Path(real)
    if p.is_dir():
        return directory_content_id(real)
    id_ = file_content_id(real)
    if id_:
        return id_
    # MISSING-MODEL SENTINEL (fix M5): file_content_id("") is "" for BOTH "no
    # path configured" and "path configured but the file is gone". If two
    # DIFFERENT missing models both fell through to "", a cache written while
    # model A was missing would spuriously validate for model B — worse than
    # the old (path, mtime) key, which at least differed by path. Distinguish
    # missing-but-configured artifacts by hashing their (unreachable) realpath
    # instead of their bytes, so two different missing paths still diverge.
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

Import `model_content_id` and `video_signature` from `..content_id`. Re-export `video_signature` from `keys` so existing importers (`runner.py:33`, `worker.py:1430`, `optimizer.py:45`, `optimizer_workers.py:44`, `production_replay.py:175`, and `core/inference/autotune/session.py:112,118` — omitted from the original importer list, must be included) keep working unchanged.

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

- [ ] **Step 6: Migrate DetectKit**

`detectkit/jobs/prediction_cache.py:31-61` (function range corrected — it is `:31-61`, not `:31-66`) — replace the absolute-path identity. **`source_path` here is a DATASET DIRECTORY, not a video file**: `detectkit/gui/panels/dataset_panel.py:394` does `Path(source_path)/"images"`, so calling `video_signature(source_path)` directly hits `IsADirectoryError`, which `video_signature`'s `except OSError` swallows into `""` — every source in a DetectKit project would then collapse onto the same identity, silently defeating cache invalidation across sources. Dispatch on what `source_path` actually is:

```python
    def _source_content_id(source_path: str) -> str:
        """DetectKit prediction-cache sources are DATASET DIRECTORIES
        (dataset_panel.py:394 does Path(source_path)/"images"), not bare video
        files. video_signature() on a directory silently degrades to "" via
        its except OSError, collapsing every source onto one identity. Dispatch
        explicitly instead of guessing from the exception."""
        p = Path(source_path)
        if p.is_dir():
            return directory_content_id(str(p))
        return video_signature(source_path)

    identities = [
        (model_content_id(path), path) for path in model_paths if path
    ]
    model_id = "|".join(identity for identity, _ in identities)
    encoded = json.dumps(
        {
            # The SOURCE (image/video file OR dataset directory) is identified
            # by content too, so a prediction cache survives the project
            # moving on disk. See _source_content_id above for the dispatch.
            "source": _source_content_id(str(source_path)),
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
            "payload shape, not an on-disk format — an old, unrestarted "
            "sidecar process is sending a pre-v5 payload and must be restarted"
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
    from hydra_suite.detectkit.jobs.prediction_cache import build_cache_key

    a = tmp_path / "dataset_a" / "images"
    b = tmp_path / "dataset_b" / "images"
    a.mkdir(parents=True)
    b.mkdir(parents=True)
    (a / "1.png").write_bytes(b"aaa")
    (b / "1.png").write_bytes(b"bbb")
    key_a = build_cache_key(str(tmp_path / "dataset_a"), model_paths=[], settings={})
    key_b = build_cache_key(str(tmp_path / "dataset_b"), model_paths=[], settings={})
    assert key_a.config_hash != key_b.config_hash or key_a.as_string() != key_b.as_string()
```

(Adapt `build_cache_key`'s actual name/signature to whatever `prediction_cache.py` exposes as its public entry point — the load-bearing assertion is that two directories with different member-file bytes must not collapse to the same key.)

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

This is a breaking dataclass-shape change (`model_path`+`model_mtime` → `model_id`). 14 test files construct `CacheKey` or otherwise depend on the old shape. Fix ALL of them in this step, not a representative subset:

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

For every file above: grep it for `CacheKey(`, `model_path=`, `model_mtime=`, and `.model_mtime`; replace with `model_id=` (computed via `model_content_id`/`video_signature` as appropriate for that test's fixtures, never a literal path string).

**`tests/test_inference_cache_keys.py` needs actual rewrites, not renames — itemized:**

- `test_detection_key_changes_with_model_path` (`:192`) — currently uses `/a.pt` and `/b.pt`, which don't exist on disk, so under content identity both resolve to the SAME missing-model sentinel and the keys become spuriously EQUAL (this is exactly the M5 bug this task also fixes). Rewrite to create two real temp files with **different bytes** and assert the keys differ.
- `test_detection_key_stable_with_threshold` (`:198`) — reads `.model_path` off the built key; that attribute no longer exists. Rewrite to read `.model_id`.
- `test_detection_key_sequential_encodes_both_models` (`:205`) — asserts substrings of the two absolute paths appear in the key string; under content identity the key contains sha256 hex, not paths. Rewrite to assert the key changes when either model's **bytes** change (mirrors the new `test_sequential_key_changes_when_either_model_changes` above), not to assert path substrings.
- `test_sequential_second_model_signature_invalidates_detection_key` (`:280`) — monkeypatches `keys._mtime`, which Step 5 deletes entirely. Rewrite to monkeypatch nothing and instead rewrite the second model file's bytes on disk, asserting the key changes (the real content-based path).
- `test_cache_key_matches_tolerates_small_mtime_diff` (`:175`) — the whole point of this test is the mtime tolerance `CacheKey.matches()` used to carry, which Step 4 explicitly removes (string-equality only, no tolerance). **Delete this test** — say so in the diff/commit, don't silently drop it. There is nothing left to test once mtime is out of the key entirely.
- `test_cache_key_matches_only_when_schema_version_matches` (`:161`) — still valid in spirit; rewrite the fixture `CacheKey(...)` calls to the 3-field shape, semantics unchanged.
- `test_cnn_and_headtail_keys_differ_across_schema_v3_v4` (`:135`) — rename to `..._v4_v5` (schema is now 4→5 territory conceptually, but the ACTUAL assertion — that two different schema versions never produce string-equal keys — is unchanged and should use `CACHE_SCHEMA_VERSION` and `CACHE_SCHEMA_VERSION - 1` rather than hardcoded 3/4 literals so it doesn't silently rot again at the next bump).
- `tests/test_inference_cache_keys.py:132`'s `CACHE_SCHEMA_VERSION == 4` assertion — change to `5`.

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
grep -rn 'CacheKey(' src/ | grep -v 'model_id' && echo "CHECK THESE" || echo "clean"
```
Expected: `clean` for the first. Every `CacheKey(` construction must name `model_id`.

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
  -v
```
Expected: no `TypeError: __init__() got an unexpected keyword argument`, no `AttributeError` on `.model_path`/`.model_mtime`, all PASS. THEN also run the full suite as a final catch-all for any construction site this enumeration missed: `python -m pytest tests/ -q 2>&1 | tail -30` and diff the failure set against the pre-task baseline (memory `project_test_suite_batching_chunk_boundary_trap`: compare failure SETS, not raw counts).

- [ ] **Step 12: Commit**

```bash
make format
git add src/hydra_suite/core/inference/content_id.py \
        src/hydra_suite/core/inference/cache/ \
        src/hydra_suite/detectkit/jobs/prediction_cache.py \
        src/hydra_suite/detectkit/sidecars/operations.py \
        src/hydra_suite/trackerkit/gui/dialogs/parameter_helper.py \
        tests/
git commit -m "feat(cache): content-based model and video identity, schema v5

Model identity becomes sha256 of the artifact's bytes (dirsha256 for pose
directories) and the video signature becomes size + head/tail content hash, so
a cache produced on a compute box is reusable on the staging machine.
_model_signature is fixed too: it folded a raw absolute path + mtime into
config_hash, which kept OBB caches machine-local independently of model_id.
DetectKit migrates onto the same key, with an explicit file/directory dispatch
for its dataset-directory sources (they are not video files); its sidecar IPC
payload shape rejects pre-v5 payloads loudly. Every existing cache is
invalidated once and regenerates."
```

- [ ] **Step 13: BEFORE/AFTER equivalence gate on MPS (blocking)**

Caches and JIT state MUST be cleared on both sides — a stale `v4` cache or a poisoned `__pycache__` fakes a result (memories `feedback_numba_jit_cache_poisons_equivalence`, `project_merge_candidate_parity_done`).

```bash
conda activate hydra-mps
pkill -f 'sleap|hydra' || true   # ONLY sleap/hydra; never other processes
find . -name '__pycache__' -type d -prune -exec rm -rf {} +
rm -rf /tmp/equiv_jobs
git -C /Users/neurorishika/Projects/Rockefeller/Kronauer/multi-animal-tracker \
    worktree add --detach .worktrees/equiv-base 8f9688e0
REPO=$PWD WT=$PWD \
  MAIN_SRC=$PWD/../equiv-base/src WT_SRC=$PWD/src \
  OUT=/tmp/equiv_jobs RUNTIME=mps \
  bash tools/equivalence/run_matrix.sh
```
Expected: every clip EQUIVALENT at its DETERMINISM floor; identical row counts; 0 unmatched. Known accepted noise: bistable head/tail π-flips on head/tail clips. **Verify `wc -l` on the CSVs is > 1 before trusting any EQUIVALENT** — empty CSVs falsely compare equal.

Record the result in the plan's Acceptance Log below. If any clip diverges, STOP and debug; do not proceed to Task 5.

- [ ] **Step 14: Same gate on firebrat (CUDA)**

```bash
ssh firebrat
cd ~/hydra-suite && git fetch && git checkout <this-branch-sha>
source ~/miniforge3/etc/profile.d/conda.sh && conda activate hydra-cuda
find . -name '__pycache__' -type d -prune -exec rm -rf {} +
git worktree add --detach .worktrees/equiv-base 8f9688e0
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
  - `shared_roots.load_shared_roots() -> dict[str, str]`, `save_shared_roots(mapping)`, `match_shared_root(abs_path, table) -> tuple[str, str] | None` (longest root wins), `resolve_shared(alias, relpath, table) -> Path`.

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


def _imported_names(path):
    tree = ast.parse(path.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                yield alias.name
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            yield node.module


@pytest.mark.parametrize("path", sorted(PACKAGE.glob("*.py")), ids=lambda p: p.name)
def test_no_app_layer_or_qt_imports(path):
    for name in _imported_names(path):
        head = name.split(".")[0]
        assert head not in FORBIDDEN_MODULES, f"{path.name} imports Qt: {name}"
        if name.startswith("hydra_suite."):
            layer = name.split(".")[1]
            assert layer not in FORBIDDEN_ROOTS, f"{path.name} imports app layer: {name}"


def test_package_imports_without_qt_installed(monkeypatch):
    """Importing the package must not pull PySide6 in transitively."""
    import sys

    for module in [m for m in sys.modules if m.startswith("hydra_suite.data.tracking_job")]:
        del sys.modules[module]
    monkeypatch.setitem(sys.modules, "PySide6", None)
    import hydra_suite.data.tracking_job  # noqa: F401
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_tracking_job_manifest.py tests/test_tracking_job_shared_roots.py tests/test_tracking_job_layering.py -v`
Expected: FAIL — the package does not exist.

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
        return cls(
            job_version=version,
            job_id=str(data["job_id"]),
            created_at=str(data["created_at"]),
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

In `paths.py`, beside `get_advanced_config_path` (`:156-158`):

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
    import hashlib

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
EXCLUDED_DIR_NAMES = {".hydra-runtime-artifacts"}


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
        for child in sorted(destination.rglob("*")):
            if child.is_file():
                files.append(child.relative_to(models_root).as_posix())
        return JobModel(
            key=planned.key,
            roles=[planned.role],
            origin_path=str(source),
            kind="directory",
            files=files,
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

    return JobModel(
        key=planned.key,
        roles=[planned.role],
        origin_path=str(source),
        kind="file",
        sha256=_sha256(destination),
        size_bytes=destination.stat().st_size,
        sidecars=sorted(set(sidecars)),
        files=sorted(set(files)),
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
git add src/hydra_suite/data/tracking_job/references.py tests/test_tracking_job_references.py
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
  - `PlannedVideo` frozen dataclass: `video_path: str`, `config: dict`, `config_provenance: str`, `planned_models: list[PlannedModel]`, `skeleton_path: str`.
  - `pack_job(job_dir, planned_videos, *, registry_entries, advanced_config_path, track_args, shared_table, copy_videos=False, shared_mode="auto", job_name=None) -> JobManifest`
  - `render_run_sh() -> str`
  - `verify_job(job_dir) -> list[str]` — returns problems; empty means valid.
  - `ABSOLUTE_PATH_FORBIDDEN_KEYS` — the sidecar keys `verify` asserts are never absolute.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_tracking_job_pack.py`:

```python
"""Packing a job: rewrites, videos, sidecars, runner, manifest."""

import json
import os
import stat

import pytest

from hydra_suite.data.tracking_job.manifest import JobManifest, TrackingJobError
from hydra_suite.data.tracking_job.pack import PlannedVideo, pack_job
from hydra_suite.data.tracking_job.references import PlannedModel


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


def test_all_problems_are_reported_not_just_the_first(packed_job):
    (packed_job / "models" / "obb" / "x.pt").unlink()
    (packed_job / "config" / "skeletons" / "ant.json").unlink()
    assert len(verify_job(packed_job)) >= 2
```

Add a shared `packed_job` fixture in `tests/conftest.py` (or a new `tests/tracking_job_conftest.py` imported by both) that packs the Task 7 fixture job into `tmp_path` and returns the job dir.

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
  fi
fi

# cd is what makes the job-relative videos.txt and sidecar file_path values work
# with load_video_list()'s CWD-relative semantics.
cd "$JOB"
mkdir -p logs
START="$(date -u +%FT%TZ)"
set +e
trackerkit track --video-list videos.txt "$@" 2>&1 | tee -a logs/run.log
CODE=${PIPESTATUS[0]}
set -e
trackerkit job _record-run --started "$START" --exit-code "$CODE" -- "$@"
exit "$CODE"
'''


def render_run_sh() -> str:
    return RUN_SH
```

- [ ] **Step 4: Implement `pack.py`**

Steps, in order (spec §6.2): resolve shared/symlink/copy per video → copy models → registry subset → config snapshot (advanced config, skeletons, `.seeded` markers) → rewrite each config → write sidecars → `videos.txt` → requirements → `run.sh` → manifest → `verify_job` self-check (raise `TrackingJobError` if it reports problems).

The rewrite table (§6.4), applied to a deep copy of each planned config:

**Minor note — `csv_path`'s redirect is dead code today, don't over-trust it.** `load_tracker_cli_session` derives `raw_csv_path` from the video itself (`cli_config.py:319`) and never reads `cfg["csv_path"]`; only `video_output_path` is actually consumed downstream (`core/tracking/session.py:653`). So rewriting/recording `csv_path` here is harmless (it keeps the sidecar internally consistent and future-proofs against a consumer being added) but currently has no live effect on where the CSV lands — do not treat the `csv_path` half of this rewrite as proof that CSV redirection works end-to-end; only `video_output_path` is load-bearing today.

```python
def _rewrite_config(config, *, video_basename, model_keys, skeleton_job_path):
    """Make one video's config job-relative. Returns (config, redirected)."""
    out = copy.deepcopy(dict(config))
    stem = Path(video_basename).stem
    out["file_path"] = f"videos/{video_basename}"
    redirected: dict[str, str] = {}
    for key, default_suffix in (
        ("csv_path", "_tracking.csv"),
        ("video_output_path", "_tracking.mp4"),
    ):
        original = str(out.get(key, "") or "")
        if not original:
            continue
        default_name = f"{stem}{default_suffix}"
        if Path(original).name == default_name:
            out[key] = f"videos/{default_name}"
        else:
            job_relative = f"videos/{stem}_{Path(original).name}"
            out[key] = job_relative
            redirected[job_relative] = str(Path(original).resolve())
    for config_key, job_key in model_keys.items():
        out[config_key] = job_key
    if skeleton_job_path:
        out["pose_skeleton_file"] = skeleton_job_path
    return out, redirected
```

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
    "POSE_MODEL_DIR": (
        "pose_model_dir",
        "pose_yolo_model_dir",
        "pose_sleap_model_dir",
        "pose_vitpose_model_dir",
    ),
    # CNN_CLASSIFIERS is handled separately: it rewrites cnn_classifiers[].model_path
    # entries in place, not a single scalar config key.
}
# Legacy alias: engine_params.py:815 falls back to yolo_model_path when the
# mode-specific key is empty. If a legacy config still uses it, it must be
# rewritten too or a stale absolute path survives packing untouched.
LEGACY_ALIAS_CONFIG_KEYS = ("yolo_model_path",)
```

2. `PlannedVideo.planned_models: list[PlannedModel]` (already defined, Task 5/6) is what `job_cli.py` (Task 11) builds from `iter_model_references`; `_rewrite_config` derives its `model_keys` argument from `ROLE_TO_CONFIG_KEY[role]` for each `PlannedModel.role`, expanding tuple-valued roles to every config key in the tuple. `color_tag_model_path` is deliberately **absent** from `ROLE_TO_CONFIG_KEY` — Task 3 never yields `COLOR_TAG_MODEL_PATH` as a reference, so pack never rewrites or ships it; it stays whatever the source config had (which Task 2's save-side relativization already made portable-by-construction, or leaves alone if it points outside the models root).

`model_keys` therefore maps each config key (`yolo_obb_direct_model_path`, `yolo_detect_model_path`, `yolo_crop_obb_model_path`, `yolo_headtail_model_path`, `pose_model_dir`, `pose_yolo_model_dir`, `pose_sleap_model_dir`, `pose_vitpose_model_dir`, and the legacy `yolo_model_path` alias — but **not** `color_tag_model_path`) to its job key; `cnn_classifiers[].model_path` entries are rewritten in place. It is built by the caller (Task 11) from the `PlannedModel` list via `ROLE_TO_CONFIG_KEY`, because only the app layer knows which `PlannedVideo` a given role's `PlannedModel` belongs to (pack.py owns the key-name table; job_cli.py owns the per-video wiring).

`_rewrite_config` must also rewrite `yolo_model_path` (the legacy alias `engine_params.py:815` falls back to) whenever it is present and non-empty and its value resolves under the models root — otherwise a legacy config carrying only the alias key ships an absolute path untouched.

Every video gets a sidecar regardless of provenance, so the remote needs no keystone logic; `track_args["keystone_override"]` is recorded for provenance only.

`requirements`: `conda_envs = [cfg["pose_sleap_env"]]` when any video's config selects the SLEAP backend via the service path; `runtime_tier` from the keystone config; `min_hydra_suite_version` from the running build.

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
manifest parses and `job_version == 1`; every `models[]` file exists with matching sha256 and size; **for every model with a non-empty `file_digests` (directory models and bundles — fix M8), every listed job-relative member path exists AND its sha256 matches**, so a corrupted single file inside a multi-file pose/bundle artifact is caught (a top-level directory sha256 alone cannot pinpoint or even always detect this depending on hash construction — checking members individually is the actual §6.2 step 5 requirement); every `sidecars[]` exists; every non-`shared` `videos[].job_path` exists (symlink target on the staging machine, regular file on the remote); every `config_job_path` exists; no sidecar has an absolute value (or a `..`) in `ABSOLUTE_PATH_FORBIDDEN_KEYS`, including each `cnn_classifiers[].model_path`; every referenced `config/skeletons/*` exists; `videos.txt` lines all exist and the first equals `keystone["video"]`; every manifest relpath passes `validate_job_relpath`.

**Fix M15 — `registry_entry_present` must actually be set.** It is declared on `JobModel` with a `False` default and nothing in this plan as originally written ever set it, so it would always read `False` even for a model that has a real `model_registry.json` entry. `pack_job` sets it explicitly: for each `JobModel` it builds, `registry_entry_present = (model.key in {key for key, _ in registry_entries})` — i.e. true iff the model's job-relative key is one of the keys `write_registry_subset` (Task 6) actually wrote an entry for.

`shared` entries are **not** checked here — verify is offline and mount-agnostic; preflight (Task 10) checks them.

- [ ] **Step 6: Run tests to verify they pass**

Run: `python -m pytest tests/test_tracking_job_pack.py tests/test_tracking_job_verify.py tests/test_tracking_job_layering.py -v`
Expected: all PASS.

- [ ] **Step 7: Commit**

```bash
make format
git add src/hydra_suite/data/tracking_job/{pack,runner,verify}.py tests/test_tracking_job_{pack,verify}.py tests/conftest.py
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
KNOWN_ARTIFACTS = [
    "videos/colony_tracking.csv",
    "videos/colony_tracking.mp4",
    "videos/colony_tracking_forward_processed.csv",
    "videos/colony_tracks.csv",
    "videos/colony_tracking_with_individual.csv",
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
git add src/hydra_suite/data/tracking_job/outputs.py tests/test_tracking_job_outputs.py
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
  - `pull_job(remote, job_dir, *, include_caches, overwrite, dry_run, runner) -> list[PullDestination]`

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
            )
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
    from hydra_suite.data.tracking_job.transport import push_job
    import subprocess

    def failing_runner(argv, **kwargs):
        return subprocess.CompletedProcess(argv, 23, stdout="", stderr="rsync: boom")

    manifest = _manifest()
    (tmp_path / "hydra_job.json").write_text("{}")
    manifest.write(tmp_path / "hydra_job.json")
    with pytest.raises(TrackingJobError) as excinfo:
        push_job(tmp_path, "host:/remote/j", runner=failing_runner)
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

`remote_video_listing` runs `ssh <host> "cd <path> && find videos -type f -o -type l"` and returns the lines — one `ssh` call, so new artifact types need no code change.

`pull_job`: fetch `logs/` first (including `runs.jsonl`), then list, then `plan_pull`, then check collisions (sha256 compare; refuse with `code=5` listing **every** collision before touching anything unless `overwrite`), then rsync the output set, then copy/hardlink into the mapped destinations, then **merge** (not append — see fix M12/§5 below) a `pull_history` entry into the **local** manifest. `--dry-run` prints the destination map and returns before any transfer.

**Fix M12 — destination-side failure policy, fully specified (was unspecified).** `pull_job` never said what happens when the *destination* side is broken, only when the pulled bytes are broken. All of the following are collected into the SAME `code=5` failure report as collisions — never silently skipped, and reported all at once, not one-by-one across repeated invocations:

- **Origin directory no longer exists** (the video's `origin_path` parent was deleted/renamed since packing): report `"origin directory missing for <job_relpath>: <origin_path>"` and exclude that video's outputs from the transfer, but do not abort the whole pull — other videos' outputs still land. Surfaced as a per-video problem in the returned report, not a hard `sys.exit`.
- **A `shared` video's local mount is absent at pull time** (the alias in `shared_roots.json` doesn't resolve, or resolves but the path doesn't exist — e.g. the NAS isn't mounted on the machine running `pull`): report `"shared mount unavailable for alias '<alias>': <resolved_path>"` and exclude that video's outputs the same way. This is a pull-time re-check of the same alias resolution `preflight`/`materialize_shared_videos` (Task 10) already does, since the machine running `pull` may differ from the one that ran `run.sh`.
- **A hardlink fails across filesystems** (`OSError: [Errno 18] Invalid cross-device link`, e.g. `/tmp` staging and the destination are different filesystems/mounts): catch `OSError` around the hardlink attempt specifically and fall back to a real copy (`shutil.copy2`) rather than failing the pull — this is expected on many lab NAS layouts, not a real error, so it degrades gracefully instead of erroring. Only *other* `OSError`s (permission denied, disk full) propagate as pull failures.

`pull_job` therefore returns a report that separates "hard failures preventing any pull" (still `code=5`, still pre-flight-checked before ANY transfer starts, e.g. collisions and completely absent local mounts for `overwrite=False`) from "per-video skips with reasons" (origin/mount unavailable — printed clearly, pull continues for unaffected videos). Never a silent skip either way — every skip has a reported reason string.

**Fix — `pull_history` MERGES, it does not merely append (spec §5).** Since the remote never mutates `hydra_job.json` (Global Constraint) but multiple pulls can happen from different pull-capable machines, appending blindly would let two pulls of the SAME `runs.jsonl` entries double-count in `pull_history`. `pull_job` merges by keying on `(run started_at, job_relpath)` — an entry already present (matching key) is left alone; only genuinely new entries are appended.

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
git add src/hydra_suite/data/tracking_job/transport.py tests/test_tracking_job_transport.py
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
    (packed_job / "models" / "obb" / "x.pt").write_bytes(b"tampered")
    result = preflight_job(
        packed_job, available_tiers=("cpu",), conda_envs=(), fast=True
    )
    assert "models" not in _names(result)


def test_all_failures_are_reported_not_just_the_first(packed_job_gpu_tier):
    (packed_job_gpu_tier / "models" / "obb" / "x.pt").write_bytes(b"tampered")
    result = preflight_job(packed_job_gpu_tier, available_tiers=("cpu",), conda_envs=())
    assert len(_names(result)) >= 2


def test_a_result_file_is_written(packed_job):
    preflight_job(packed_job, available_tiers=("cpu",), conda_envs=())
    assert (packed_job / "logs" / "preflight.json").is_file()


def test_shared_video_is_materialized_as_a_symlink(packed_job_shared, tmp_path):
    mount = tmp_path / "mnt" / "lab"
    (mount / "2026-09").mkdir(parents=True)
    source = mount / "2026-09" / "colony.mp4"
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
    mount = tmp_path / "mnt" / "lab"
    (mount / "2026-09").mkdir(parents=True)
    (mount / "2026-09" / "colony.mp4").write_bytes(b"\xff" * 4096)  # wrong content
    result = preflight_job(
        packed_job_shared, shared_root_overrides={"labnas": str(mount)}, conda_envs=()
    )
    assert not result.ok
    assert "shared_roots" in _names(result)


def test_a_missing_shared_file_fails_with_both_paths(packed_job_shared, tmp_path):
    mount = tmp_path / "mnt" / "lab"
    mount.mkdir(parents=True)
    result = preflight_job(
        packed_job_shared, shared_root_overrides={"labnas": str(mount)}, conda_envs=()
    )
    assert not result.ok


def test_insufficient_disk_fails(packed_job, monkeypatch):
    import shutil

    monkeypatch.setattr(
        shutil, "disk_usage", lambda _p: shutil._ntuple_diskusage(1, 1, 0)
    )
    result = preflight_job(packed_job, available_tiers=("cpu",), conda_envs=())
    assert "disk" in _names(result)
```

**Fix M11 — the four `packed_job*` fixtures, fully specified.** The tests above were host-dependent (nothing pinned `runtime_tier`, and nothing isolated the shared-root/host-config lookups from this machine's REAL `HYDRA_CONFIG_DIR`/`HYDRA_HOST_CONFIG_DIR`), so a real `labnas` alias configured on this Mac (see Task 13 Step 5, which does exactly that) could flip these tests. `build_engine_params` also defaults an absent `runtime_tier` to `"gpu"` (`engine_params.py:805`), so any fixture that doesn't pin one explicitly is silently GPU-tier and `test_a_good_job_passes`'s `available_tiers=("cpu", "gpu")` would mask that. Add these to `tests/conftest.py` (or a shared `tracking_job_conftest.py`), every one monkeypatching `HYDRA_CONFIG_DIR` and `HYDRA_HOST_CONFIG_DIR` to `tmp_path` so a real machine-local alias can never leak in:

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
    preflight's conda_envs check must name "sleap" as required."""
    planned = _planned(
        staging,
        config={
            "runtime_tier": "cpu",
            "enable_pose_extractor": True,
            "pose_model_type": "SLEAP",
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
    table = {"labnas": str(staging["video"].parent)}
    pack_job(
        job_dir, [planned],
        registry_entries=[("obb/x.pt", {"species": "ant", "source_path": "/host/a.pt"})],
        advanced_config_path=str(staging["advanced"]),
        track_args={"video_list": "videos.txt", "runtime_tier": "cpu"},
        shared_table=table,
    )
    return job_dir
```

**`available_tiers=None` semantics (undefined in the original draft):** `preflight_job(..., available_tiers=None)` means "do not run the `runtime_tier` check at all" (report it neither passing nor failing — omit it from `checks`), as distinct from `available_tiers=()` which means "the tier check runs and fails, because nothing is available." This lets a caller that hasn't yet determined the local tier set (e.g. a dry `verify`-only invocation) skip the check honestly rather than getting a spurious pass or fail. `job_cli.py` (this task) always passes a real, non-`None` `available_tiers` derived from `runtime.resolver.available_tiers()` for actual `preflight`/`run` invocations; only test code exercising the "not yet known" path uses `None`.

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_tracking_job_preflight.py -v`
Expected: FAIL — module missing.

- [ ] **Step 3: Implement `preflight.py`**

Checks, in order, all executed (spec §9.4):

1. `manifest` — parses, version supported, `verify_job` returns no problems.
2. `version` — `hydra_suite` importable and `>= requirements.min_hydra_suite_version`.
3. `conda_envs` — every required env exists. The caller injects the env list (a `conda env list` subprocess in the CLI) so the core stays testable and subprocess-free.
4. `runtime_tier` — requested tier in `available_tiers`; otherwise report the fallback the resolver would take and fail unless `allow_tier_fallback`.
5. `models` — every file hashes to the manifest sha256 (skipped when `fast`).
6. `shared_roots` — resolve each alias (overrides first, then the **host** table, read from `HYDRA_HOST_CONFIG_DIR` when set — that's the `run.sh` path, where `HYDRA_CONFIG_DIR` has already been redirected to the job's own `config/` — **or from `HYDRA_CONFIG_DIR` directly when `HYDRA_HOST_CONFIG_DIR` is unset**, which is the "preflight invoked directly on a host, not via run.sh" path (e.g. `trackerkit job preflight <job>` run interactively before ever exporting the job's env), so that path doesn't wrongly read the job's own (still-job-scoped) config dir as if it were the host's; falling back to the platformdirs default only if neither is set — never the job snapshot), check existence and readability, compare `size_bytes` and `signature`, then materialize `videos/<basename>` as a symlink (replace an existing **symlink**, never a regular file).
7. `disk` — free space under `videos/` >= 1.5x total video bytes, counting shared videos' sizes for caches but not for the videos themselves.

Write the result to `logs/preflight.json`.

The `HYDRA_HOST_CONFIG_DIR` fallback matters: `run.sh` sets it to `${HYDRA_CONFIG_DIR:-}`, which is **empty** on any host using defaults, so an empty value must mean "the platformdirs config dir", not "no table".

**Minor fix — restore the "offer to persist" behaviour spec §6.7 step 1 describes and the original draft silently dropped.** A one-off `--shared-root ALIAS=PATH` given to `job run`/`job preflight` should, after a successful preflight run that used it, prompt (interactively, when stdout is a tty and `--yes`/`--no-input` wasn't passed) to save the alias into the host's persistent `shared_roots.json` table via `shared_roots.py` (Task 5), so the next invocation doesn't need to repeat the override. In `job_cli.py` (Task 11), after `preflight_job(..., shared_root_overrides=parsed_overrides)` returns `ok=True`, for each override alias not already present in the persisted table: prompt `Save shared-root alias 'labnas' -> '/mnt/lab' for future jobs? [y/N]` and call `shared_roots.save_alias(alias, path)` on yes. Non-interactive/CI invocations skip the prompt and leave the override one-off, as before.

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_tracking_job_preflight.py tests/test_tracking_job_layering.py -v`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
make format
git add src/hydra_suite/data/tracking_job/preflight.py tests/test_tracking_job_preflight.py
git commit -m "feat(tracking-job): preflight with shared-root materialization, reporting every failure"
```

---

### Task 11: CLI wiring — `trackerkit job …`

**Files:**
- Create: `src/hydra_suite/trackerkit/job_cli.py`
- Modify: `src/hydra_suite/trackerkit/app.py:97` (subparsers), `:269-311` (`parse_arguments`), `:399-534` (`main` dispatch)
- Test: `tests/test_trackerkit_job_cli.py` (create)

**Interfaces:**
- Consumes: everything from Tasks 5-10, plus `iter_model_references` (Task 3), `plan_batch_jobs` (`batch_plan.py:119`), `resolve_track_video_inputs` (`app.py:351`), `load_tracker_cli_session` (`cli_config.py:304`), `discover_multihead_model_bundle` (`classkit/model_bundle.py:42`).
- Produces: `run_job_cli(args) -> int`, and the `job` subcommand group.

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
```

`job run` registers `--gpus/--jobs/--threads-per-job/--detach/--calibrate/--allow-tier-fallback/--shared-root` and deliberately **omits** `--sahi-profile/--apply-tuned-inference/--inference-autotune-manual`.

In `parse_arguments`, add a `job`/`pack` branch mirroring `:278-296` (both/neither video checks, `--no-shared` vs `--shared-only` conflict). `_subparser_choices` (`:259-266`) only walks the top level; add a nested lookup so `job_parser.error(...)` and each sub-subparser's `.error(...)` are reachable.

In `main`, add `elif args.command == "job":` dispatching to `run_job_cli(args)` with the same lazy import + `try/except` shape as the `calibrate` branch (`:446-462`).

- [ ] **Step 4: Implement `job_cli.py`**

`pack` flow:

1. `videos = resolve_track_video_inputs(args.videos, args.video_list)`.
2. `specs = plan_batch_jobs(videos, explicit_config_path=args.config, keystone_override=args.keystone_override, sahi_profile=args.sahi_profile, apply_tuned_inference=args.apply_tuned_inference, inference_autotune_manual=args.inference_autotune_manual)` — pack does **not** reimplement config precedence.
3. For each spec, `session = load_tracker_cli_session(spec.video_path, config_data=spec.config)`. **Fix M16 — do not also call `build_tracking_parameters`.** `TrackerCliSession` already carries the fully-resolved params at `.params` (`cli_config.py:52-70`) — `load_tracker_cli_session` does the same resolution `build_tracking_parameters` would, and the latter additionally needs a `video_probe` this step never has a reason to obtain (probing the video a second time here is pure waste). Use `params = session.params` directly.
4. `refs = list(iter_model_references(params))`.
5. For each ref: `key = make_pose_model_path_relative(ref.path) if ref.kind == "directory" else make_model_path_relative(ref.path)`; if still absolute, `key = external_key_for(ref.path)`. For file refs, `bundle = discover_multihead_model_bundle(ref.path)` and pass `bundle["artifact_paths"]` (minus the selected checkpoint) as `bundle_artifacts`. Build the `role -> job_key` map, then expand it to the `config_key -> job_key` map for `_rewrite_config` via `pack.ROLE_TO_CONFIG_KEY` (Task 7, fix M2) — for each `ref.role`, look up its config key(s) (a tuple for `POSE_MODEL_DIR`'s fan-out) and assign each the same `key`.
6. **Fix M9 — compute `skeleton_path` explicitly; it is never derived automatically.** Spec correction #7 established that `POSE_SKELETON_FILE` has no engine-side fallback — every consumer (`engine_params.py:1179`, `core/post/pose_merge.py:292`, `core/post/media_export.py:401`, `core/individual/properties/cache.py:221`, `core/inference/config.py:1258`) reads it verbatim. So before building `PlannedVideo`, resolve `skeleton_path = str(resolve_model_path(session.config.get("pose_skeleton_file", "")))` when pose is enabled and the value is non-empty; otherwise `skeleton_path = ""`. This is what `pack.py`'s skeleton-snapshot step (Task 7 Step 4) consumes — without it, a pose-enabled job silently ships with no skeleton and every pose-merge/export consumer breaks on the remote with no loud error at pack time.
7. `pack_job(...)` with `registry_entries=list(iter_registry_entries())`, `advanced_config_path=str(get_advanced_config_path())`, `shared_table=load_shared_roots()`.
8. **Warn** (spec §6.6) when any export stage is enabled in a config: the CLI leaves `DATASET_OUTPUT_DIR`, `FINAL_MEDIA_EXPORT_VIDEO_OUTPUT_DIR` and `INDIVIDUAL_DATASET_OUTPUT_DIR` at `None` (`cli_config.py:293-295`), so those exports produce nothing on the remote. Point at follow-up §17.1.

`_record-run` is a hidden subcommand appending one JSON line to `logs/runs.jsonl` (`started_at, finished_at, hostname, exit_code, hydra_suite_version, git_sha, argv`). Add `metavar=argparse.SUPPRESS` so it stays out of help, and register its trailing argv capture with `nargs=argparse.REMAINDER` (minor fix) — without it, argparse tries to parse `run.sh`'s forwarded `"$@"` (which can contain flags like `--gpus auto`) against `_record-run`'s own option strings and errors out instead of passing them through verbatim.

**Fix M7 — `run` against a remote target MUST run preflight over ssh BEFORE `./run.sh`, or §6.7 (shared-root materialization) is defeated for the primary workflow.** The original draft only ran `preflight_job` for the LOCAL-dir case and, for remote, went straight to `ssh <host> 'cd <path> && ./run.sh …'`. That means a `shared` video is never materialized (symlinked into `videos/`) on the compute box before `run.sh` invokes `trackerkit track`, so the primary "shared NAS mount, don't copy the video" workflow (spec §6.7) silently fails on first use for every remote run — the only path that actually exercises it in production. Both branches now run preflight first:

```
local dir  -> preflight_job(job_dir, ...) locally, then
              subprocess.run(["./run.sh", *passthrough], cwd=job_dir, check=False)
remote     -> ssh <host> 'cd <path> && trackerkit job preflight . && ./run.sh …'
              (single ssh invocation; preflight's non-zero exit short-circuits
              the && before run.sh ever starts, so a materialization failure is
              reported before any tracking begins), wrapped in `nohup … &`
              under --detach.
```
`subprocess.run(["./run.sh", *passthrough])` for the local-dir branch MUST pass `cwd=job_dir` (minor fix) — `run.sh` itself resolves its own location via `BASH_SOURCE`, but the parent Python process's CWD is whatever the user invoked `trackerkit job run` from, and without `cwd=job_dir` a relative job path argument on the CLI would still work by luck (bash resolves `./run.sh` against the argv path, not CWD) while anything inside `run.sh` that assumes CWD == job root during the brief window before its own `cd "$JOB"` would not. `track_args` recorded at pack time are **always** forwarded.

Task 13 Step 5 must name this actual command (`trackerkit job preflight . && ./run.sh …` via one `ssh` invocation) rather than describing preflight and run as separate, un-chained steps.

`calibrate`: run `trackerkit calibrate` inside the job environment (`HYDRA_MODELS_DIR`, `HYDRA_CONFIG_DIR`, `cd <job>`) against the keystone video, forwarding `track_args["inference_autotune_manual"]` verbatim.

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

Run: `trackerkit job --help && trackerkit job pack --help && trackerkit job run --help`
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

import os

import pytest

from hydra_suite.trackerkit.cli_config import load_tracker_cli_session


def test_packed_sidecar_resolves_identically_to_staging(packed_job, monkeypatch):
    monkeypatch.setenv("HYDRA_MODELS_DIR", str(packed_job / "models"))
    monkeypatch.setenv("HYDRA_CONFIG_DIR", str(packed_job / "config"))
    monkeypatch.chdir(packed_job)

    for sidecar in packed_job.glob("videos/*_config.json"):
        session = load_tracker_cli_session(None, config_path=str(sidecar))
        params = session.params
        for role_key in ("YOLO_OBB_DIRECT_MODEL_PATH", "POSE_MODEL_DIR"):
            value = params.get(role_key, "")
            if value:
                assert os.path.commonpath(
                    [os.path.abspath(value), os.path.abspath(str(packed_job / "models"))]
                ) == os.path.abspath(str(packed_job / "models")), (
                    f"{role_key} resolved outside <job>/models: {value}"
                )
        for entry in params.get("CNN_CLASSIFIERS", []) or []:
            path = entry.get("model_path", "")
            if path:
                assert os.path.abspath(path).startswith(
                    os.path.abspath(str(packed_job / "models"))
                ), f"CNN_CLASSIFIERS entry resolved outside <job>/models: {path}"
```
(Adapt `load_tracker_cli_session`'s exact `config_path=`/keyword-argument shape to what `cli_config.py` actually exposes — the load-bearing assertion is: point `HYDRA_MODELS_DIR`/`HYDRA_CONFIG_DIR` at the packed job, CWD at the job root, load each sidecar, and prove every resolved model path lands inside `<job>/models`, matching what the staging-side `params` used for that role.)

**(b) §6.5's characterization test must call the GUI's `build_config_dict`, not just `make_model_path_relative`.** Add to `tests/test_config_model_path_portability.py` (Task 2) or a new file: build a full GUI config dict via `trackerkit.gui.orchestrators.config.build_config_dict` (or whatever the actual entry point is named) with an absolute `color_tag_model_path`/`cnn_classifiers` populated, assert the SERIALIZED dict — not a call to the relativization helper in isolation — comes back job-relative. Testing `make_model_path_relative` alone (as the original Task 2 tests do) proves the helper works, not that the GUI actually calls it on every relevant field.

**(c) §8.1/§11 — remote `job verify` over ssh after push.** Add to `job_cli.py`'s `push` flow (or a documented follow-on step): after `push_job` succeeds, `ssh <host> 'cd <path> && trackerkit job verify .'` (through the same `PYTHONPATH`-safe entry point) and surface a non-zero verify result as a push failure — a push that lands a broken job on the remote should fail loudly, not silently succeed and defer the discovery to `run`.

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
```
`job status` prints the manifest summary plus the tail of `logs/runs.jsonl` (local) or the same over ssh (remote target). `run --calibrate` runs `trackerkit job calibrate <job>` before `run.sh`, sharing the same forwarding rule as the standalone `calibrate` subcommand (spec correction #4: only `inference_autotune_manual`, never `--sahi-profile`). `job calibrate <remote>` runs calibration over ssh the same way `run` does (fix M7's chained-ssh pattern). `--budget-seconds` is forwarded verbatim into `track_args` and ultimately to the calibration/autotune budget the engine already accepts.

- [ ] **Step 9: Commit**

```bash
make format && make lint-moderate
git add src/hydra_suite/trackerkit/job_cli.py src/hydra_suite/trackerkit/app.py tests/test_trackerkit_job_cli.py \
        tests/test_tracking_job_sidecar_engine_params_equality.py tests/test_config_model_path_portability.py \
        tests/test_tracking_job_transport.py
git commit -m "feat(trackerkit): trackerkit job pack/push/run/pull/verify/preflight/calibrate/status CLI"
```

---

### Task 12: Documentation

**Files:**
- Create: `docs/user-guide/trackerkit-jobs.md`
- Modify: `docs/user-guide/trackerkit-cli.md` (cross-link from `## A batch`), `mkdocs.yml` (nav), `docs/developer-guide/` (a note on `iter_model_references`)

- [ ] **Step 1: Write `docs/user-guide/trackerkit-jobs.md`**

Cover: the lifecycle walkthrough (pack → push → preflight → calibrate → run → pull); the "what travels, what doesn't" table (models/config/videos travel; TensorRT/ONNX/CoreML engines, calibration profiles and conda envs do not); the shared-root mount table with a two-host example (`{"labnas": "/Volumes/lab"}` on the laptop, `{"labnas": "/mnt/lab"}` on the box); the conda-env requirement and how preflight reports it; the nine-GPU example adapted to `job run --gpus auto`; and the §6.6 export limitation with a pointer to follow-up §17.1.

- [ ] **Step 2: Add the developer-guide note**

A short section stating that `iter_model_references` + the four key tuples in `engine_params.py` are the contract every new model role must join, and that `tests/test_engine_params_model_reference_contract.py` fails loudly otherwise.

- [ ] **Step 3: Build the docs**

Run: `make docs-check`
Expected: strict build passes, terminology check clean.

- [ ] **Step 4: Commit**

```bash
git add docs/ mkdocs.yml
git commit -m "docs: portable tracking jobs user guide and the model-reference contract note"
```

---

### Task 13: Acceptance — full gates + the real round trip on firebrat

This task produces evidence, not code. **Nothing merges until every box here is ticked with pasted output.**

- [ ] **Step 1: Full local suite delta**

```bash
conda activate hydra-mps
pkill -f 'sleap|hydra' || true
python -m pytest tests/ -q 2>&1 | tail -30
```
Expected: the failure SET is a subset of the known-failing set on `8f9688e0`. Compare **sets**, not counts — new test files shift pytest chunk boundaries and can fake collection errors (memory `project_test_suite_batching_chunk_boundary_trap`). Record both sets.

- [ ] **Step 2: MPS equivalence matrix (final, post-everything)**

Run the Task 4 Step 13 recipe again at the branch tip. Expected: every clip EQUIVALENT at its determinism floor, row counts > 1 on every CSV, perf ratio <= 1.25.

- [ ] **Step 3: CUDA equivalence matrix on firebrat**

Run the Task 4 Step 14 recipe at the branch tip on `firebrat`. **Confirm `courtship` is still not to be touched.** Expected: same acceptance.

- [ ] **Step 4: The Goal-4 round trip — the only proof that matters**

The equivalence harness forces `use_cached_detections: False` (`tools/equivalence/runner.py:154`), so it exercises **zero** cache reuse. This step is the one that proves a remote cache hits locally.

**Fix C3 — "the runner reports a cache hit" was never a real, checkable signal.** There is no "cache hit" log line anywhere in the codebase (grepped; only comments reference the concept). The step as originally drafted had no executable acceptance criterion for the single most important claim in this whole plan. Use the REAL probes that exist instead: `InferenceRunner.caches_all_valid()` (`runner.py:1043`), `InferenceRunner.detection_cache_missing_frames()` (`runner.py:1076`), and `cache_set_is_fully_reusable(caches)` (`runner.py:430` -- a MODULE-LEVEL function taking a `_CacheSet`, NOT a method; `caches_all_valid()` is already its runner-level wrapper, so the probe uses that).

```bash
# All invocations use the worktree's src explicitly (fix M13 — bare `trackerkit`
# resolves to MAIN's editable install, not this worktree, on this Mac).
export PYTHONPATH=/Users/neurorishika/Projects/Rockefeller/Kronauer/multi-animal-tracker/.worktrees/portable-jobs/src
alias job_cli='python -m hydra_suite.trackerkit.app job'

# On the Mac: pack two fixture jobs (one pose/SLEAP, one pure OBB).
conda activate hydra-mps
cd /Users/neurorishika/Projects/Rockefeller/Kronauer/multi-animal-tracker/.worktrees/portable-jobs
bash tools/equivalence/fixtures/fetch_fixtures.sh   # populate gitignored fixtures/clips first
job_cli pack /tmp/jobs/fly tools/equivalence/fixtures/clips/fly_obb.mp4 \
    --config tools/equivalence/fixtures/configs/fly_obb.json
job_cli pack /tmp/jobs/pose tools/equivalence/fixtures/clips/ant_pose_headtail.mp4 \
    --config tools/equivalence/fixtures/configs/ant_pose_headtail.json

job_cli push /tmp/jobs/fly  rutalab@firebrat:/home/rutalab/jobs/fly
job_cli push /tmp/jobs/pose rutalab@firebrat:/home/rutalab/jobs/pose

job_cli preflight /tmp/jobs/fly        # local sanity
# Fix M7: run over ssh runs preflight-then-run.sh as ONE chained ssh invocation
# on the remote (this is what `job run <remote>` now does internally).
job_cli run rutalab@firebrat:/home/rutalab/jobs/fly
job_cli run rutalab@firebrat:/home/rutalab/jobs/pose

job_cli pull rutalab@firebrat:/home/rutalab/jobs/fly  /tmp/jobs/fly
job_cli pull rutalab@firebrat:/home/rutalab/jobs/pose /tmp/jobs/pose
```

Assert, and paste the evidence:
1. **Zero registration** happened on firebrat — `~/.local/share/hydra-suite/models/model_registry.json` on the box is byte-identical before and after (record its sha256 both times).
2. Every artifact from the §10 table landed beside the original clip on the Mac, including `.inference_cache_<stem>/`.
3. **The cache hits — executable probe, not a log-scrape.** For each pulled job, open the pulled `.inference_cache_<stem>/` with the SAME config used to produce it (same models, same job-relative paths, resolved via the pulled sidecar) and run this probe script:

```python
"""tools/equivalence/probe_cache_hit.py — the Goal-4 acceptance probe.

Usage: python probe_cache_hit.py <job_dir> <video_relpath_under_videos>
Exits 0 and prints "CACHE FULLY REUSABLE" iff every cache stage the config
enables is valid and has zero missing frames. Exits 1 with the specific
gap otherwise -- this is the ONLY acceptance evidence for Goal 4, so it must
fail loudly and specifically, never silently pass.
"""
import os
import sys
from pathlib import Path

job_dir = Path(sys.argv[1]).resolve()
video_relpath = sys.argv[2]

os.environ["HYDRA_MODELS_DIR"] = str(job_dir / "models")
os.environ["HYDRA_CONFIG_DIR"] = str(job_dir / "config")
os.chdir(job_dir)

from hydra_suite.trackerkit.cli_config import load_tracker_cli_session
from hydra_suite.core.inference.config import build_inference_config_from_params
from hydra_suite.core.inference.runner import InferenceRunner
from hydra_suite.utils.video_artifacts import build_inference_cache_dir

sidecar = job_dir / "videos" / f"{Path(video_relpath).stem}_config.json"
session = load_tracker_cli_session(str(job_dir / video_relpath), config_path=str(sidecar))
# InferenceRunner takes an InferenceConfig, NOT the raw params dict, and
# caches_all_valid() returns False outright unless cache_dir is supplied --
# which would fake a FAIL. Build both exactly as worker.py:1328 does.
params = session.params
runner = InferenceRunner(
    build_inference_config_from_params(params),
    cache_dir=build_inference_cache_dir(str(job_dir / video_relpath)),
    video_path=str(job_dir / video_relpath),
    cache_only=True,  # read-only probe: never write or recompute
    roi_mask=params.get("ROI_MASK"),
)

if not runner.caches_all_valid():
    print("FAIL: caches_all_valid() is False -- the key is still carrying "
          "something machine-local; debug before merging.")
    sys.exit(1)

# detection_cache_missing_frames REQUIRES the frame range (runner.py:1076);
# calling it bare is a TypeError.
start_frame = int(params.get("START_FRAME", 0) or 0)
end_frame = int(params.get("END_FRAME", 0) or 0)
missing = runner.detection_cache_missing_frames(start_frame, end_frame)
if missing:
    print(f"FAIL: {len(missing)} frames report missing from the detection "
          f"cache: {missing[:10]}...")
    sys.exit(1)

print("CACHE FULLY REUSABLE")
sys.exit(0)
```
(Adapt `InferenceRunner`'s exact constructor signature and `load_tracker_cli_session`'s `config_path=` keyword to what those modules actually expose — the load-bearing content is calling all three real probes named above against the pulled cache and failing loudly on any gap, not scraping a nonexistent log line.)

Run it for both pulled jobs and paste the `CACHE FULLY REUSABLE` output. If either fails, the key is still carrying something machine-local — debug before merging.

4. The pulled `_tracking.csv` from firebrat is row-identical to a native firebrat run of the same config (same host, same models, same config). Cross-device byte-identity vs the Mac is explicitly **not** claimed (spec §2 non-goal).

- [ ] **Step 5: Shared-root live check**

Configure the same alias on both hosts pointing at a directory each can see, pack a video under it, confirm the manifest records `shared`, the push list omits the video, preflight materializes the symlink on firebrat, the run completes, and pull maps outputs beside the origin.

**Name the actual command (fix M7).** "The run completes" is not a command. `job run rutalab@firebrat:/home/rutalab/jobs/<name>` internally chains `ssh rutalab@firebrat 'cd /home/rutalab/jobs/<name> && trackerkit job preflight . && ./run.sh …'` as ONE ssh invocation, so preflight's materialization of the shared symlink happens immediately before `run.sh` on the same connection — this is the step that actually exercises §6.7 for the primary remote workflow (a bare `./run.sh` without a preceding preflight, which the plan used to describe, would leave the shared video unmaterialized and `trackerkit track` would fail on a missing file). Paste the ssh session's preflight output showing the `shared_roots` check passing and the symlink being created, immediately followed by the tracking run's own log output, both from the SAME `job run` invocation.

- [ ] **Step 6: Record everything in the Acceptance Log**

---

## Acceptance Log

Fill in as gates are run. A gate with no pasted output is not a passed gate.

| Gate | Platform | Commit | Result | Evidence |
|---|---|---|---|---|
| Cache-key before/after matrix | MPS | | | |
| Cache-key before/after matrix | CUDA (firebrat) | | | |
| Final matrix | MPS | | | |
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
