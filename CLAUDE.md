# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build & Environment Setup

### Quick Install (pip, CPU only)

```bash
pip install hydra-suite
```

For GPU variants:

```bash
# NVIDIA GPU
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install hydra-suite[cuda]

# Apple Silicon
pip install hydra-suite[mps]
```

### Full Environment (conda)

Platform-specific conda environments are used. Choose one:

```bash
# Create environment (pick your platform)
make setup            # CPU / NumPy+Numba
make setup-mps        # Apple Silicon (M1/M2/M3/M4)
make setup-cuda       # NVIDIA GPU (CUDA 12 or 13)

# After activating the environment, install runtime packages
make install          # or install-mps / install-cuda

# Install dev/audit tools
make install-dev

# Install docs tools
make docs-install
```

Environment names: `hydra-suite`, `hydra-suite-mps`, `hydra-suite-cuda`.

## Running Tests

```bash
make pytest                # Run all tests
python -m pytest tests/test_<name>.py         # Run a single test file
python -m pytest tests/test_<name>.py::test_fn  # Run a single test function
make test-cov              # Tests + terminal coverage
make test-cov-html         # Tests + HTML coverage (htmlcov/index.html)
```

pytest is configured in `pyproject.toml`. Test files are in `tests/`. Benchmarks are excluded by default (`-m "not benchmark"`).

## Launching the Applications

```bash
hydra          # HYDRA Suite launcher (main entry point)
trackerkit     # MAT tracking GUI
posekit        # PoseKit pose-labeling GUI
filterkit      # FilterKit tool
classkit       # ClassKit labeler
refinekit      # RefineKit interactive proofreading
detectkit      # DetectKit detection tool
```

## Code Quality

```bash
make format            # autopep8 → black → isort (do this before committing)
make format-check      # Check formatting without changing files
make lint              # Moderate-severity flake8 (recommended gate)
make lint-fix          # Auto-fix with ruff then reformat
make lint-strict       # Strict flake8
make lint-report       # Side-by-side issue counts at all three levels
```

**Pre-PR checklist** (from `docs/developer-guide/contributing.md`):
```bash
make commit-prep       # format (black + isort)
make lint-moderate     # catch serious issues
make docs-check        # docs-build + quality metrics + terminology check
```

For large PRs, also run `make audit` (dead-code + dep-graph + type-check + coverage).

## Documentation

```bash
make docs-serve        # Live preview at http://127.0.0.1:8000
make docs-build        # Strict mkdocs build
make docs-check        # Build + quality metrics + terminology check
make techref-build     # Build LaTeX technical reference
```

Terminology rule: use `posekit` (CLI entry point) and `hydra_suite.posekit` (not legacy names).

## Code Health Tools

```bash
make dead-code         # Find unused code (vulture, ≥80% confidence)
make dead-code-whitelist  # Generate vulture_whitelist.py for false-positive review
make dep-graph         # Visual SVG dependency graph → hydra_suite.svg
make dep-graph-text    # Text module map via pyreverse
make type-check        # mypy static type checking (lenient mode)
make audit             # Full sweep: all of the above + coverage
```

See `to_fix.md` for known dead-code findings and the rationale for false-positive exclusions (Qt dynamic dispatch, PyTorch `forward`, `classkit/` stubs, etc.).

---

## Design Principles

These principles exist to prevent God-object accumulation and cross-kit duplication. Apply them whenever adding features or refactoring.

### No God Objects

- `MainWindow` is a **thin coordinator only**: it instantiates panels, connects signals, and delegates to a typed config object. It holds no business logic.
- If a class exceeds ~500 lines, it is doing too much. Extract focused sub-modules.
- Workers, dialogs, and config schemas must live in separate files — never embedded in `main_window.py`.

### Shared Abstractions (use these, don't reinvent)

| Abstraction | Location | When to use |
|---|---|---|
| `BaseWorker(QThread)` | `widgets/workers.py` | Every background task — provides `progress`, `status`, `error`, `finished` signals and error-safe `run()` |
| `BaseDialog(QDialog)` | `widgets/dialogs.py` | Every modal dialog — handles button box, modal setup, accept/reject boilerplate |
| `WelcomePage` | `widgets/welcome_page.py` | Splash/welcome screen for each kit |
| Typed config schema | `<kit>/config/schemas.py` | Kit runtime state — dataclass with `to_dict`/`from_dict`; never scatter state as `self.flag = True` on `MainWindow` |

### Kit GUI Structure (standard layout)

Every kit must follow this layout:

```
<kit>/
    app.py              # entry point: constructs QApplication and MainWindow
    gui/
        __init__.py
        main_window.py  # thin coordinator (~200 lines max)
        panels/         # focused UI sub-modules
        dialogs/        # one file per dialog class
        widgets/        # kit-local reusable widgets (if any)
    config/
        schemas.py      # typed dataclass config for kit state
```

### Cross-Kit Rules

- When fixing a pattern in one kit, check if the same fix applies to all kits — patterns are shared.
- Never copy-paste worker boilerplate: use `BaseWorker`.
- Never copy-paste dialog boilerplate: use `BaseDialog`.
- Config state lives in `<kit>/config/schemas.py`, not in widget attributes.

### Dependency Direction

App layers → Core / Runtime / Data / Training / Utils → (no upward imports)

- App layers (trackerkit, posekit, classkit, refinekit, detectkit, filterkit) may import Core/Runtime/Data/Training/Utils.
- Core/Runtime/Data/Training/Utils must **never** import from any app layer.
- `widgets/` shared layer is importable by all app layers but imports nothing from app layers.

---

## Architecture

### System Layers

| Layer | Package | Role |
|---|---|---|
| Launcher | `hydra_suite.launcher` | `hydra` entry point; routes to individual kits |
| MAT / TrackerKit | `hydra_suite.trackerkit` | Multi-animal tracking GUI |
| PoseKit | `hydra_suite.posekit` | Pose-labeling application |
| ClassKit | `hydra_suite.classkit` | Classification/embedding toolkit |
| RefineKit | `hydra_suite.refinekit` | Interactive proofreading |
| FilterKit | `hydra_suite.filterkit` | FilterKit tool |
| DetectKit | `hydra_suite.detectkit` | Detection tool |
| Shared Widgets | `hydra_suite.widgets` | Cross-kit UI components: BaseWorker, BaseDialog, WelcomePage |
| Integrations | `hydra_suite.integrations` | External tool bridges (SLEAP, X-AnyLabeling) |
| Core | `hydra_suite.core` | Detection, Kalman filter, assignment, post-processing, identity |
| Runtime | `hydra_suite.runtime` | Compute runtime selection and GPU utilities |
| Data | `hydra_suite.data` | CSV export, detection cache, dataset generation/merge |
| Training | `hydra_suite.training` | Dataset builders, training runner, registry, service |
| Utils | `hydra_suite.utils` | Geometry, image processing, batching, prefetch |
| Resources | `hydra_suite.resources` | Bundled read-only assets (brand icons, default configs, skeletons) |
| Paths | `hydra_suite.paths` | Central path resolution: bundled assets via `importlib.resources`, user dirs via `platformdirs` |

**Key boundary rules:**

- Dependency flows downward: App layers (MAT, PoseKit, ClassKit, RefineKit, FilterKit, DetectKit) may import from Core, Runtime, Data, Training, and Utils, but never the reverse.
- Core, Runtime, Data, Training, and Utils must not import from any app-layer package or from Integrations.
- Integrations bridges external tools and may import from Core/Runtime/Data/Utils but not from app layers.
- `widgets/` shared layer may be imported by all app layers; it must not import from any app layer.
- Data layer must be reusable from both GUI and scripts.
- All path resolution must go through `hydra_suite.paths`. No module should use `Path(__file__).parents[N]` to navigate to repo root.
- Bundled read-only assets are accessed via `importlib.resources` through the `paths` module.
- User-writable data (models, training, config) goes to platform-appropriate directories via `platformdirs`.
- Users can override data/config directories with `HYDRA_DATA_DIR` and `HYDRA_CONFIG_DIR` environment variables.

### Data and Config Directories

All apps share the same data directories via `hydra_suite.paths`:

```python
from hydra_suite.paths import get_models_dir, get_presets_dir, get_skeleton_dir
```

Default locations (via `platformdirs`):

| | macOS | Linux |
|---|---|---|
| Config | `~/Library/Application Support/hydra-suite/` | `~/.config/hydra-suite/` |
| Data | `~/Library/Application Support/hydra-suite/` | `~/.local/share/hydra-suite/` |

Override with environment variables:

```bash
export HYDRA_DATA_DIR=/mnt/lab-shared/hydra-data      # models, training runs
export HYDRA_CONFIG_DIR=/path/to/project-config      # presets, skeletons, advanced config
```

Debug current paths: `python -c "from hydra_suite.paths import print_paths; print_paths()"`

### MAT Tracking Pipeline

1. Video frames read (optionally prefetched via `utils`)
2. Detector generates measurements (`core/detectors/engine.py`)
3. Kalman filter predicts and update track state (`core/filters/kalman.py`)
4. Hungarian or greedy assignment (`core/assigners/hungarian.py`)
5. TrackingWorker emits frame/status/metrics to GUI (`core/tracking/worker.py`)
6. Trajectories written to CSV (`data/csv_writer.py`)
7. Optional backward pass reuses detection cache (`.npz`)
8. Post-processing resolves, relinks, and interpolates (`core/post/processing.py`)

Detection cache (`.npz`) enables backward tracking, reproducible reruns, and pose precompute. Backward mode refuses to run without a valid cache.

### Kalman State Vector

```
state = [x, y, theta, vx, vy]   # position, heading, velocity
measurement = [x, y, theta]
```

Process noise is anisotropic (longitudinal vs. lateral). Young tracks have attenuated velocity (`KALMAN_MATURITY_AGE`, `KALMAN_INITIAL_VELOCITY_RETENTION`).

### Identity / Runtime System

All compute-heavy methods use a single `compute_runtime` setting. Runtime support logic is centralized in:

- `src/hydra_suite/runtime/compute_runtime.py`
- `src/hydra_suite/utils/gpu_utils.py`

Canonical runtimes: `cpu`, `mps`, `cuda`, `onnx_cpu`, `onnx_cuda`, `tensorrt`.

When adding a new model/method: define a pipeline key, add capability rules in `_pipeline_supports_runtime()`, add runtime translation, wire UI intersection gating. See `docs/developer-guide/runtime-integration.md` for the full checklist.

### Extension Points

- **New detector**: implement `detect_objects`-compatible output in `core/detectors/engine.py`
- **New identity method**: extend `core/identity/analysis.py`; preserve the crop extraction metadata and output contract
- **New runtime pipeline**: follow the checklist in `docs/developer-guide/runtime-integration.md`

### Key Source Files for Auditing Behavior

- `src/hydra_suite/core/tracking/worker.py` — top-level orchestrator
- `src/hydra_suite/core/filters/kalman.py`
- `src/hydra_suite/core/assigners/hungarian.py`
- `src/hydra_suite/core/post/processing.py`
- `src/hydra_suite/core/identity/runtime_api.py`
- `src/hydra_suite/core/identity/classification/backend.py` — shared classifier loader
- `src/hydra_suite/core/identity/classification/errors.py` — classifier error hierarchy
- `src/hydra_suite/runtime/compute_runtime.py`
- `src/hydra_suite/paths.py` — central path resolution (all asset/data paths)
- `src/hydra_suite/widgets/workers.py` — BaseWorker base class (in-progress)
- `src/hydra_suite/widgets/dialogs.py` — BaseDialog base class (in-progress)

### `legacy/` Policy

Superseded code is moved to `legacy/` for one release cycle before deletion. `legacy/` is excluded from tests, mypy, and formatters. Never import from `legacy/` in `src/` or `tests/`.

### PoseKit Pipeline

1. Image set + project metadata loaded
2. Annotation state edited in UI (`posekit/gui/`)
3. Labels persisted to YOLO pose format
4. Optional model-assisted inference (YOLO pose, SLEAP) and split-generation steps

PoseKit inference uses the same `compute_runtime` system and the same ONNX/TensorRT artifact auto-management pattern as MAT.

---

## Active Refactoring Context

The codebase is currently mid-way through a **Simplification Sprint** (see `docs/superpowers/specs/2026-04-04-codebase-simplification-design.md`). Four sequential slices:

| Slice | Goal | Status |
| --- | --- | --- |
| 1 — Worker Pattern | All 15 `QThread` workers inherit `BaseWorker` from `widgets/workers.py` | In progress |
| 2 — Config Schemas | Each kit gets `<kit>/config/schemas.py`; `MainWindow` initializes `self.config` | Pending |
| 3 — Dialog Pattern | `BaseDialog` in `widgets/dialogs.py`; `classkit/gui/dialogs.py` split into 11 files | Pending |
| 4 — Monolith Split | `trackerkit/gui/main_window.py` (19k lines) decomposed into panels + workers + dialogs | Pending |

**When working on any kit GUI**: check whether a shared abstraction (`BaseWorker`, `BaseDialog`, typed schema) already exists or is planned before writing new boilerplate.

No public CLI entry points or inter-kit APIs change during this sprint.
