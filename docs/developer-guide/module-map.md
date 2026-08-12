# Module Map

## Entrypoints

- `hydra_suite.launcher.app` — HYDRA launcher / tool selector
- `hydra_suite.trackerkit.app` — TrackerKit multi-animal tracking
- `hydra_suite.posekit.gui.main` — PoseKit pose labeling
- `hydra_suite.classkit.app` — ClassKit classification / embedding
- `hydra_suite.detectkit.app` — DetectKit detection model training
- `hydra_suite.filterkit.app` — FilterKit dataset filtering
- `hydra_suite.refinekit.app` — RefineKit interactive proofreading

## Core

- `hydra_suite.core.tracking.worker`
- `hydra_suite.core.detectors.engine`
- `hydra_suite.core.background.model`
- `hydra_suite.core.filters.kalman`
- `hydra_suite.core.assigners.hungarian`
- `hydra_suite.core.post.processing`
- `hydra_suite.core.individual.analysis`

## Data

- `hydra_suite.data.csv_writer`
- `hydra_suite.data.detection_cache`
- `hydra_suite.data.dataset_generation`
- `hydra_suite.data.dataset_merge`

## Runtime

- `hydra_suite.runtime.compute_runtime`

## TrackerKit GUI

- `hydra_suite.trackerkit.gui.main_window` — thin coordinator
- `hydra_suite.trackerkit.gui.orchestrators.config` — configuration management
- `hydra_suite.trackerkit.gui.orchestrators.session` — session lifecycle
- `hydra_suite.trackerkit.gui.orchestrators.tracking` — tracking execution
- `hydra_suite.trackerkit.gui.panels.setup_panel`
- `hydra_suite.trackerkit.gui.panels.detection_panel`
- `hydra_suite.trackerkit.gui.panels.tracking_panel`
- `hydra_suite.trackerkit.gui.panels.identity_panel`
- `hydra_suite.trackerkit.gui.panels.postprocess_panel`
- `hydra_suite.trackerkit.gui.panels.dataset_panel`
- `hydra_suite.trackerkit.gui.workers.crops_worker`
- `hydra_suite.trackerkit.gui.workers.preview_worker`
- `hydra_suite.trackerkit.gui.workers.merge_worker`
- `hydra_suite.trackerkit.gui.workers.dataset_worker`
- `hydra_suite.trackerkit.gui.workers.video_worker`
- `hydra_suite.trackerkit.gui.widgets.*`
- `hydra_suite.trackerkit.gui.dialogs.*`
- `hydra_suite.trackerkit.gui.model_utils`

## PoseKit

- `hydra_suite.posekit.gui.main`
- `hydra_suite.posekit.gui.main_window`
- `hydra_suite.posekit.gui.canvas`
- `hydra_suite.posekit.gui.models`
- `hydra_suite.posekit.gui.project`
- `hydra_suite.posekit.gui.workers`
- `hydra_suite.posekit.gui.dialogs.*`
- `hydra_suite.posekit.config.schemas`
- `hydra_suite.posekit.core.extensions`
- `hydra_suite.posekit.inference.worker`

## DetectKit

- `hydra_suite.detectkit.app`
- `hydra_suite.detectkit.config.schemas`
- `hydra_suite.detectkit.gui.panels.*`

## Shared Widgets

- `hydra_suite.widgets.workers` — `BaseWorker` base class for all threaded tasks
- `hydra_suite.widgets.dialogs` — `BaseDialog` base class for modal dialogs
- `hydra_suite.widgets.welcome_page` — `WelcomePage` splash screen
- `hydra_suite.widgets.recents` — recent files management

## Integrations and Utils

- `hydra_suite.integrations.sleap.service`
- `hydra_suite.integrations.xanylabeling.cli`
- `hydra_suite.utils.*`

## Paths and Resources

- `hydra_suite.paths` — central path resolver; all apps import from here for models, configs, assets
  - `get_models_dir()`, `get_presets_dir()`, `get_skeleton_dir()`, `get_training_runs_dir()`
  - `get_brand_icon_bytes(name)`, `get_brand_qicon(name)`
  - `print_paths()` — debug helper to show all resolved paths
  - Respects `HYDRA_DATA_DIR` and `HYDRA_CONFIG_DIR` environment variable overrides
- `hydra_suite.paths_migrate` — one-time migration helper for repo-to-user-dir data
- `hydra_suite.resources` — bundled read-only assets package
- `hydra_suite.resources.brand` — SVG/PNG brand icons
- `hydra_suite.resources.configs` — default config presets
- `hydra_suite.resources.configs.skeletons` — skeleton keypoint definitions
