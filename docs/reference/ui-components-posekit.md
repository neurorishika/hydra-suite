# UI Components Reference (PoseKit)

This reference describes PoseKit UI components in clean workflow order.

## How To Use This Page

- Configure project setup once, then iterate primarily in the main workspace.
- Use model-assist tools only after annotation conventions are stable.
- Keep train/eval/active-learning runs versioned by output folder and seed.

## Main Workspace Layout

| Area | Role | Key features |
|---|---|---|
| Left pane (frame management) | Curate labeling workload. | Labeling frames list, all frames list, search, sorting, random add, smart select, delete selected. |
| Center pane (canvas) | Perform keypoint annotation. | Image canvas, zoom, drag, keypoint placement, prediction overlays, metadata tags/notes. |
| Right pane (tool groups) | Configure annotation, display, navigation, model actions. | Annotation controls, display tuning, navigation shortcuts, training/evaluation/active-learning entry points. |
| Status bar | Runtime feedback. | Save state, progress bar, SLEAP status. |

## Project Setup Dialog (Wizard / Project Settings)

### Purpose
Define project paths, class/keypoint schema, skeleton, and migration behavior.

### Controls

| Control | Role | Value selection guidance | Common failure mode |
|---|---|---|---|
| Output root | Project artifact root. | Use per-project folder for reproducibility. | Mixing multiple projects in one root. |
| Labels directory | YOLO pose label storage location. | Keep under output root unless integrating external datasets. | Label path drift between sessions. |
| Autosave option | Save-on-navigation behavior. | Enable for most workflows. | Disabled autosave causing accidental data loss. |
| BBox pad fraction | Bounding box expansion from keypoints. | Increase only if extremities clip. | Excessive padding reducing localization precision. |
| Classes list | Category definitions. | Keep minimal and stable unless multi-class task is required. | Mid-project class changes without migration plan. |
| Keypoints/skeleton editor | Anatomical schema definition. | Finalize early; version changes deliberately. | Frequent schema churn invalidating older labels. |
| Migration controls | Existing-label remapping strategy. | Name-based if names are stable; index-based for append-only changes. | Wrong mapping corrupting historical labels. |

## Annotation Group

### Purpose
Control labeling semantics and progression behavior.

| Control | Role | How to choose values | Common failure mode |
|---|---|---|---|
| Class selector | Assigns class for current frame. | Keep consistent for single-class projects. | Unintended class flips during rapid labeling. |
| Keypoint list | Active keypoint selection and status. | Follow canonical anatomical order. | Skipped landmarks due to order confusion. |
| Mode (frame/keypoint) | Navigation model for labeling. | Frame mode for speed; keypoint mode for consistency checks. | Mode mismatch slowing workflow. |
| Click visibility | Default visibility code for placement. | Use occluded only when landmark is inferable; missing otherwise. | Visibility misuse reducing training signal quality. |

## Display Group

### Purpose
Tune visualization clarity without changing underlying labels.

| Control | Role | How to choose values | Common failure mode |
|---|---|---|---|
| Enhance contrast + settings | CLAHE/sharpening for visibility. | Use conservatively; confirm no hallucinated structure. | Over-enhancement masking true image quality issues. |
| Show predictions / confidence | Overlay model output for assisted labeling. | Enable during correction passes. | Trusting predictions without manual QA. |
| Autosave delay | Delayed write frequency. | Keep short enough for crash safety. | Delay too long increases potential data loss window. |
| Keypoint/edge opacity | Overlay visibility tuning. | Lower when dense overlays hide anatomy. | Overly faint overlays causing placement errors. |
| Point/text size | Annotation readability scaling. | Increase for high-res imagery or dense skeletons. | Oversized labels obscuring landmarks. |
| Fit to view | Reset viewport to full frame. | Use after heavy zoom/pan edits. | Remaining zoomed and missing out-of-view mistakes. |

## Navigation Group

### Purpose
Accelerate frame/keypoint traversal and save cadence.

| Control | Role | Notes |
|---|---|---|
| Prev/Next frame | Sequential frame traversal. | Works with A/D hotkeys. |
| Save | Immediate save. | Use before long operations. |
| Next unlabeled | Jump to next incomplete frame. | Critical for completion sweeps. |

## Model Group

### Purpose
Run model-assisted annotation, training, and evaluation workflows.

| Subsection | Includes | Value-selection guidance | Common failure mode |
|---|---|---|---|
| Backend selection | YOLO vs SLEAP vs ViTPose backend. | Match backend to model family. | Backend/model mismatch. |
| Prediction controls | Min confidence, current-frame predict, dataset predict, apply predictions, cache clear. | Use higher confidence for conservative adoption. | Bulk apply without review in difficult frames. |
| YOLO model controls | Weights path, browse/use latest. | Pin known-good weights per project version. | Using stale or incompatible weights. |
| SLEAP service controls | Conda env, model dir, device, start/stop service. | Validate env/model before long runs. | Service startup issues from bad env resolution. |
| Training/eval/active learning launches | Dialog entry points. | Use after baseline labels and split quality are confirmed. | Running training from inconsistent labels/splits. |

## Project Group

### Purpose
Manage schema and export artifacts.

| Control | Role | Notes |
|---|---|---|
| Skeleton editor | Update keypoint topology. | Prefer infrequent, versioned schema changes. |
| Project settings | Reopen setup/migration controls. | Use for controlled updates only. |
| Export dataset + splits | Prepare training-ready data layout. | Validate split manifests before training. |

## Smart Select Dialog

### Purpose
Select diverse, high-value unlabeled frames with embedding and clustering support.

| Control group | Includes | Guidance |
|---|---|---|
| Scope/filtering | Scope, exclusion toggles | Keep scope explicit to avoid hidden frame subsets. |
| Embedding config | Model, device, batch, max side, enhancement | Tune for stable embeddings first, speed second. |
| Selection config | N frames, clusters, min/cluster, strategy, threshold | Use cluster coverage to avoid redundancy. |

## Dataset Split Dialog

### Purpose
Create reproducible train/val/test (or k-fold) splits with cluster-aware options.

| Control | Guidance |
|---|---|
| Train/val/test fractions | Maintain enough validation/test signal for reliable comparisons. |
| K-fold count | Use when dataset size is limited and repeated validation is needed. |
| Min per cluster | Protect minority clusters from being dropped. |
| Random seed | Fix seed for reproducible experiment comparisons. |
| Split name | Use semantic names tied to experiment versions. |

## Training Runner Dialog

### Purpose
Launch and monitor model training/fine-tuning jobs.

| Control group | Includes | Guidance |
|---|---|---|
| Backend/model | Backend (YOLO/SLEAP/ViTPose), base weights/model dir | Keep backend/model family aligned. For ViTPose: pick the auto-download COCO-B catalog entry, or Browse to a local/animal checkpoint. |
| Training hyperparameters | Batch size, auto-batch, epochs, patience, image size, device | Increase complexity only after baseline convergence is stable. |
| Data controls | Train fraction, seed, ignore occluded, auxiliary datasets | Keep split/seed fixed for fair model comparisons. |
| SLEAP export options | Env, output `.slp`, include aux, embed media | Validate export format before training job launch. |

## Evaluation Dashboard

### Purpose
Compare predictions against labels with keypoint-level and frame-level diagnostics.

| Control | Role | Guidance |
|---|---|---|
| Backend + model path | Evaluation inference source (YOLO, SLEAP, or ViTPose). | Lock to specific model version for reproducible reports. |
| PCK and OKS thresholds | Metric sensitivity settings. | Use consistent thresholds across model comparisons. |
| Output directory | Result artifacts location. | Version by date/run id to track history. |
| Worst-frame tables | Error triage list. | Feed difficult frames back into labeling queue. |

## Active Learning Dialog

### Purpose
Suggest high-value frames based on uncertainty/disagreement/error signals.

| Control group | Includes | Guidance |
|---|---|---|
| Strategy | Uncertainty / disagreement / error-focused methods | Start with one strategy, compare lift before combining. |
| Backend/inference config | Device, image size, confidence, batch, cache | Keep inference settings aligned with evaluation settings. |
| Scope + N suggestions | Candidate pool and output count | Keep suggestion batches small and reviewable. |
| Evaluation CSV / keypoint focus | Error-targeted sampling | Use when correcting specific keypoint failure modes. |

## Recommended PoseKit Workflow

1. Setup project schema and paths.
2. Label pilot subset and validate consistency.
3. Run smart selection to expand diverse coverage.
4. Train baseline and evaluate worst frames.
5. Iterate with active learning and periodic schema-safe exports.
