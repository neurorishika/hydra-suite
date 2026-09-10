# Architecture

## System Layers

- `hydra_suite.launcher`: HYDRA launcher entry point and tool selector.
- `hydra_suite.trackerkit`: TrackerKit multi-animal tracking GUI — decomposed into orchestrators, panels, workers, and widgets.
- `hydra_suite.posekit`: pose-labeling application and related dialogs/inference flows.
- `hydra_suite.classkit`: classification/embedding toolkit.
- `hydra_suite.detectkit`: detection model training tool.
- `hydra_suite.refinekit`: interactive proofreading.
- `hydra_suite.filterkit`: data sieve tool.
- `hydra_suite.widgets`: shared UI components (BaseWorker, BaseDialog, WelcomePage, recents).
- `hydra_suite.integrations`: external tool bridges (SLEAP, X-AnyLabeling).
- `hydra_suite.core`: detection, filtering, assignment, post-processing, worker orchestration.
- `hydra_suite.runtime`: compute runtime selection and GPU utilities.
- `hydra_suite.data`: CSV/cache/dataset generation and merging.
- `hydra_suite.training`: dataset builders, training runner, registry, model publishing.
- `hydra_suite.utils`: shared helpers (GPU detection, geometry, image processing, batching, prefetch).
- `hydra_suite.paths`: central path resolution — bundled assets via `importlib.resources`, user dirs via `platformdirs`.
- `hydra_suite.resources`: bundled read-only assets (brand icons, default configs, skeletons).

## Design Intent

- Keep high-throughput frame processing in worker-style components.
- Keep GUI orchestration separated from algorithm implementations.
- Keep data export/cache utilities outside core tracking logic.

## Runtime Entry Paths

- HYDRA Launcher: `hydra_suite.launcher.app:main`
- TrackerKit: `hydra_suite.trackerkit.app:main`
- PoseKit: `hydra_suite.posekit.gui.main:main`
- ClassKit: `hydra_suite.classkit.app:main`
- DetectKit: `hydra_suite.detectkit.app:main`
- FilterKit: `hydra_suite.filterkit.app:main`
- RefineKit: `hydra_suite.refinekit.app:main`

## Key Operational Boundaries

- Core tracking should not depend on GUI widget internals.
- Data layer should stay reusable from both GUI and scripts.
- PoseKit is a separate app surface with its own workflow but shared environment/tooling.
- All path resolution goes through `hydra_suite.paths`. No module uses `Path(__file__).parents[N]` to navigate to repo root.
- Bundled read-only assets are in `hydra_suite.resources` and accessed via `importlib.resources`.
- User-writable data (models, training runs, config) lives in platform directories, never inside the installed package.
- Users can override directories with `HYDRA_DATA_DIR` and `HYDRA_CONFIG_DIR` environment variables.

## Data and Config Directories

All apps share paths via `hydra_suite.paths`. This module resolves user-writable directories (models, training, presets, skeletons) and bundled read-only assets (brand icons, default configs).

**Key functions:**

| Function | Returns |
|----------|---------|
| `get_models_dir()` | `<data>/models/` — trained model storage |
| `get_training_runs_dir()` | `<data>/training/runs/` — training run registry |
| `get_training_workspace_dir(subdir)` | `<data>/training/<subdir>/` — training workspace |
| `get_presets_dir()` | `<config>/presets/` — config presets (seeded from bundled defaults) |
| `get_skeleton_dir()` | `<config>/skeletons/` — skeleton definitions (seeded from bundled defaults) |
| `get_advanced_config_path()` | `<config>/advanced_config.json` |
| `get_brand_icon_bytes(name)` | Bytes of a bundled SVG/PNG icon |
| `get_brand_qicon(name)` | `QIcon` from bundled icon (lazy Qt import) |
| `print_paths()` | Prints all resolved paths to stdout |

**Environment variable overrides:**

| Variable | Overrides | Default |
|----------|-----------|---------|
| `HYDRA_DATA_DIR` | Models, training runs | `platformdirs.user_data_dir()` |
| `HYDRA_CONFIG_DIR` | Presets, skeletons, advanced config | `platformdirs.user_config_dir()` |

When adding a new module that needs to read models, configs, or assets, always import from `hydra_suite.paths` — never construct paths relative to `__file__`.

## TrackerKit Decomposition

TrackerKit's `main_window.py` has been decomposed from a monolith into focused subpackages:

### Orchestrators (`trackerkit/gui/orchestrators/`)

Business logic delegates extracted from MainWindow:

- `config.py` — configuration load/save, parameter synchronization
- `session.py` — session lifecycle, video loading, output management
- `tracking.py` — tracking execution, forward/backward passes, worker coordination

### Panels (`trackerkit/gui/panels/`)

Each UI tab is a self-contained panel class:

- `setup_panel.py` — video, FPS, output paths, resize, runtime settings
- `detection_panel.py` — detection backend, thresholds, morphology, YOLO settings
- `tracking_panel.py` — assignment, Kalman, lifecycle, motion logic
- `identity_panel.py` — crop extraction, identity method settings
- `postprocess_panel.py` — cleanup, interpolation, merge, video output
- `dataset_panel.py` — dataset export configuration

### Workers (`trackerkit/gui/workers/`)

Background tasks running on `QThread` (inheriting `BaseWorker`):

- `crops_worker.py` — crop extraction for identity analysis
- `preview_worker.py` — detection preview rendering
- `merge_worker.py` — forward/backward merge operations
- `dataset_worker.py` — dataset generation
- `video_worker.py` — video rendering

### Widgets (`trackerkit/gui/widgets/`)

TrackerKit-specific reusable UI primitives:

- `collapsible.py` — collapsible section container
- `help_label.py` — inline help text widget
- `loss_plot_widget.py` — training loss visualization
- `stacked_page.py` — stacked page navigation
- `tooltip_button.py` — button with rich tooltip

## Portable job model-reference contract

Portable tracking jobs (`trackerkit job pack`, see
`docs/user-guide/trackerkit-jobs.md`) derive the set of files a job needs
from the engine parameter builder rather than a hand-maintained list.
`iter_model_references(params)` (`trackerkit/engine_params.py`) walks every
key in the four classification tuples in that same module —
`MODEL_DIR_PARAM_KEYS`, `MODEL_FILE_PARAM_KEYS`, `MODEL_LIST_PARAM_KEYS`,
`NON_MODEL_PATH_PARAM_KEYS` — to decide which resolved paths are model
references pack must ship, and which path-shaped keys are not models at all.

**Any new model-consuming role must join this contract in two places:**

1. Its engine-param key must be added to the correct tuple in
   `engine_params.py`, so `iter_model_references` classifies it.
2. `pack.ROLE_TO_CONFIG_KEY` (`data/tracking_job/pack.py`) must gain the
   matching lowercase config key(s) for that role, so the packer can
   rewrite the config-side reference to the job-relative path.

`tests/test_engine_params_model_reference_contract.py` fails loudly if a
role is missing from either side — it turns on every model-consuming role
at once and asserts every live model path is classified and every role that
should produce a reference does.
