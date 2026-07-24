# Design spec: ViTPose fine-tuning as a PoseKit subprocess (Spec 4)

**Date:** 2026-07-17
**Status:** design — approved shape, not yet a task-by-task plan.
**Worktree:** `.worktrees/vitpose-finetune` (branch `vitpose-finetune`, off `main`).
**Parent roadmap:** `docs/superpowers/specs/2026-07-16-vitpose-backend-roadmap.md` (Spec 4 section).
**Depends on:** Spec 1 (native ViTPose port), merged to `main`. The leaf package
`core/identity/pose/vitpose/` (model, transforms, decode, weights loader,
`build_vitpose`) is the substrate this spec builds on.
**Explicitly does NOT depend on:** Spec 2 (runtime layer) or Spec 3 (inference
backend registry). This spec stops at a validated checkpoint; using that
checkpoint for inference/preview/tracking is Spec 3's job.

## Goal

Add ViTPose fine-tuning to hydra as a **first-class PoseKit training backend**,
parallel to the existing YOLO-pose and SLEAP options, that takes a user's
labeled PoseKit project plus a pretrained ViTPose checkpoint and produces a
fine-tuned checkpoint with validation metrics — entirely within the app's own
Python environment, launched as a subprocess.

## Scope decisions (settled during brainstorming)

These were each decided explicitly; the plan must honor them.

1. **Target = custom animal, arbitrary K.** The user brings a COCO-keypoints
   dataset with any keypoint count K and their own skeleton. Final-layer is
   re-initialized to K; backbone loads strict.
2. **Train + validate only.** Deliverable is a validated checkpoint + metrics +
   val-prediction visualizations. Loading it for inference needs the Spec 3
   backend registry and is out of scope.
3. **Native torch, in-env subprocess.** ViTPose runs in the app's own
   environment (Spec 1 removed the mmcv/mmpose dependency), so the training
   subprocess is `python -m …vitpose.training` — **no `conda run` wrapping**
   (that is the SLEAP-only path).
4. **No MoE / no ViTPose+.** Rationale: MoE routing is not learned — an expert
   is selected by a hard integer `dataset_index`, so it only helps when *jointly
   training multiple distinct labeled datasets with heterogeneous keypoint
   conventions* (cross-species). hydra is structurally per-project (one project
   → one skeleton → one checkpoint; at track time you always know which project
   you are on), so the multi-model alternative — a separate fine-tuned model per
   project/species — always dominates MoE. Cross-species generalization is not
   hydra's job. Classic **B/L/H only**. If a cross-species research need ever
   materializes it is a separate spec built on architecture Spec 1 already ships
   (`build_vitpose_moe` exists and is untouched here).
5. **No flip augmentation, no symmetry schema change.** PoseKit's skeleton model
   has no left/right symmetry field, and flip requires known `flip_pairs`.
   Rather than extend the skeleton schema, flip is disabled entirely; we rely on
   scale/rotation/photometric augmentation. (This is the roadmap's safe default:
   "disable flip entirely if no L/R symmetry".)
6. **Not routed through `training/` contracts.** Pose training is dialog-local by
   existing precedent — YOLO-pose training builds a `task=pose` command and
   `Popen`s it directly; `TrainingRole`/`TrainingRunSpec` cover only OBB and
   classify roles. ViTPose follows the pose precedent, not the classify plumbing.

## Architecture: three layers, correctly placed

The work splits into three layers along the existing dependency boundaries. The
governing constraints are (a) the Spec-1 leaf imports nothing from `hydra_suite`
(AST-verified), and (b) `core/` must not import `posekit/` (verified clean today;
`build_coco_keypoints_dataset` lives in `posekit/core/extensions.py`).

| Layer | Location | Responsibility | May import |
|---|---|---|---|
| **Payload** | `core/identity/pose/vitpose/training/` (new subpackage of the leaf) | model setup, target encoding, loss, train loop, validation, CLI. Runs in the subprocess. | leaf siblings (`..model`, `..transforms`, `..decode`, `..weights`) + torch/numpy/cv2. **Nothing from `hydra_suite`.** |
| **Orchestration** | `posekit/core/vitpose_training.py` + `posekit/core/vitpose_checkpoints.py` (new, non-Qt) | build COCO dataset, resolve/download base checkpoint, write+validate `run.json`, build the subprocess command, parse progress lines. | `posekit/core` (dataset builder) + the leaf. **No Qt.** |
| **UI** | `ViTPoseTrainingWorker` in `posekit/gui/dialogs/training.py` (extend existing dialog) | `Popen` the command in-env, stream stdout → log pane, SIGTERM on cancel, drive progress bar. | Qt + orchestration. |

### Why the payload lives *inside* the leaf, not a sibling package

Because it is PoseKit-free (PoseKit passes a COCO dir + `run.json` path — the
payload never imports PoseKit), it satisfies leaf purity from inside the package.
Keeping it in `vitpose/training/` keeps all ViTPose code cohesive and lets the
leaf-purity AST check do double duty: extend that check to cover `training/` so
it *mechanically enforces* the PoseKit-free contract — the moment training
attempts `import hydra_suite.posekit…`, the check fails. The leaf's identity
shifts from "deployable inference model" to "self-contained ViTPose package
(model + training), free of `hydra_suite` coupling."

**Hard requirement:** `vitpose/__init__.py` must **not** eagerly import
`training/`. A pure-inference consumer (`from …vitpose import build_vitpose`)
must never load the training loop or its cv2-augmentation surface. `training/` is
imported only when referenced explicitly or run as `python -m …vitpose.training`.

### Why orchestration is NOT in `core/identity/pose/backends/`

`backends/{sleap,yolo}.py` are the **inference + auto-export** layer
(`SleapExportedBackend`, `YoloNativeBackend`, `auto_export_*`). They hold **zero
training code** (verified). The true ViTPose parallel to them is
`backends/vitpose.py` — the inference backend + auto-export caching wrapper —
which the roadmap earmarks for **Spec 3**. Training orchestration additionally
*cannot* live under `core/` at all, because it must call
`build_coco_keypoints_dataset` in `posekit`, and `core/` must not import
`posekit`. So orchestration lands in `posekit/core/`, beside the existing
`build_yolo_pose_dataset` / `build_coco_keypoints_dataset` — the layer that
already owns "what we offer users to train from."

## Data contract

The boundary between orchestration and payload is exactly two files:

- **A COCO-keypoints dataset directory** — precisely what
  `build_coco_keypoints_dataset` already emits: `images/` + `annotations.json`
  (with `keypoint_names`, `skeleton_edges`, and K keypoints per instance).
- **`run.json`** — a validated config carrying: resolved absolute init-checkpoint
  path, variant (`b|l|h`), K, input size, split ratios, hyperparameters (below),
  output dir, device, seed, resume-from (optional).

Nothing else crosses the boundary. The payload never imports PoseKit and is
independently runnable and testable.

## Payload components (`core/identity/pose/vitpose/training/`)

Each file is small and single-purpose:

- **`config.py`** — `run.json` schema as a dataclass, with validation (reject bad
  variant, non-positive K, missing checkpoint, empty split, unknown keys).
- **`dataset.py`** — COCO-keypoints → `torch.utils.data.Dataset`. Per instance:
  `box2cs` + `get_warp_matrix` affine crop (reusing the leaf transforms),
  augmentation = scale + rotation + photometric. **No flip.**
- **`targets.py`** — **UDP Gaussian heatmap encoder** (`sigma=2`). This is the
  one genuinely new numerical component (Spec 1 built *decode*; training needs
  the *encode* inverse). It is the primary correctness risk and gets a dedicated
  encode→decode round-trip test (≈ identity within the Spec-1 UDP dtype floor).
- **`model_setup.py`** — `build_vitpose(variant, K)`; load init checkpoint with
  **backbone strict, head re-init to K**, asserting the *only* missing/unexpected
  keys are `keypoint_head.final_layer.*`. Build layer-wise LR-decay param groups:
  `num_layers = depth + 2`; `pos_embed`/`patch_embed` → id 0, `blocks.{i}` →
  `i+1`, head → `num_layers-1`; `lr_scale = decay ** (num_layers - layer_id - 1)`.
  The `+2` is called out explicitly as an error-prone constant with its own test.
- **`loss.py`** — `JointsMSELoss(use_target_weight=True)`.
- **`train.py`** — the loop: AdamW (`lr=5e-4`, `wd=0.1`), grad-clip 1.0,
  warmup + cosine schedule, AMP, checkpoint every N epochs, **resume from
  checkpoint**, structured per-epoch metric emission to stdout.
- **`validate.py`** — decode predictions via the leaf `decode_udp_torch`; compute
  **PCK@0.05 and PCK@0.1** (bbox-normalized) + val loss.
- **`__main__.py`** — CLI: `python -m …vitpose.training --config run.json`.

### Recipe defaults (small custom data; all overridable via `run.json`)

AdamW `lr=5e-4`, `wd=0.1`, grad-clip 1.0, `JointsMSELoss(use_target_weight=True)`,
Gaussian targets `sigma=2` UDP-encoded, drop_path ~0.1 (not 0.3), epochs ~20–50
(not 210), input size `256×192` (H×W, ViTPose default), batch tuned to GPU.
Layer-wise LR decay `layer_decay=0.75` per `model_setup.py`.

### Validation metric: PCK, not OKS

COCO OKS requires per-keypoint sigmas calibrated on human anatomy; they do not
exist for an arbitrary animal skeleton. **PCK** (percentage of correct keypoints,
distance normalized by bbox size) is the animal-pose standard and needs no
sigmas. Report PCK@0.05 and PCK@0.1; select `best.pt` on **PCK@0.05**.

## Base-checkpoint catalog (`posekit/core/vitpose_checkpoints.py`)

Mirrors how YOLO offers `get_yolo_pose_base_models()` in an editable combo and
auto-downloads on first use. ViTPose has no ultralytics auto-download, so we
carry our own SHA-pinned fetcher (the core of `tools/vitpose/fetch_assets.py`,
factored into a shared `download_pinned(url, sha256, dest)` the catalog resolver
and the tool both call — one source of truth).

- **Catalog**: `name → {url, sha256, variant, num_keypoints, description}`,
  seeded with:
  - **ViTPose-B/L/H (COCO)** — the human-pose weights Spec 1 already pins and
    validated end-to-end. Guaranteed to load.
  - **ViTPose AP-10K / APT-36K (animal)** — same classic ViT backbone,
    animal-pretrained; the recommended starts for insects. Each ships with a
    pinned URL+SHA and a load test (backbone loads strict, head re-init to K).
    **No unvalidated entry ships.**
- **`resolve_checkpoint(name_or_path) -> Path`**: catalog name → download-if-
  absent to the app checkpoint cache, SHA256-verify (atomic write, clear error on
  mismatch), return local path; filesystem path → pass through.
- **Layering:** orchestration resolves name → concrete local path *before*
  writing `run.json`; the payload only ever opens a path and never downloads.

## PoseKit UI integration (`posekit/gui/dialogs/training.py`)

The seam already exists: the backend combo already lists
`["YOLO Pose", "ViTPose (soon)", "SLEAP"]`.

- Activate the placeholder: `"ViTPose (soon)"` → `"ViTPose"`.
- Selecting it swaps the model row to a variant picker (`b|l|h`) + an **editable
  init-checkpoint combo** = catalog names (default `ViTPose-B (COCO)`, animal
  entries labeled) **+ a "Browse…" local-file option** — identical pattern to the
  YOLO model combo. It reuses the existing device / epoch / batch controls.
- **`ViTPoseTrainingWorker`** (parallel to the YOLO-pose worker): on start it
  calls orchestration to build the COCO dataset (feeding the project's
  `keypoint_names` / `skeleton_edges` / `class_names`), resolve the checkpoint,
  and write `run.json`; then `Popen`s `python -m …vitpose.training` **in-env**,
  streams stdout line-by-line to the log pane, and sends **SIGTERM** on cancel
  via the same termination path the YOLO worker uses.
- **Progress via stdout only** (no extra IPC): the payload prints structured
  one-line records (`epoch`, `train_loss`, `val_loss`, `pck@0.05`, `pck@0.1`);
  the worker parses them to drive the progress bar and log.

## Outputs (land in the run dir, like YOLO's `results/`)

- `metrics.csv` — per-epoch train/val loss + PCK (curve inspectable).
- `best.pt` (selected on PCK@0.05) and `last.pt` checkpoints.
- `run.json` — the resolved config (reproducibility + resume).
- A handful of **val-prediction overlay PNGs** (cv2-drawn keypoints on sample val
  crops) so a user can eyeball quality without any inference backend — the
  "validate" half of "train + validate only."

## Error handling

Fail fast and loud, one structured stdout line per failure the worker can
surface: checkpoint variant/shape mismatch; K disagreeing with annotations;
re-init touching keys other than `final_layer.*`; empty split; SHA mismatch on
download; CUDA OOM (report a suggested smaller batch size).

## Testing strategy

- **Unit, headless (the bulk):**
  - UDP encoder round-trip (encode→decode ≈ identity within the Spec-1 float floor).
  - `model_setup` asserts re-init touches *only* `final_layer.*`.
  - Layer-wise LR groups produce the expected `+2`-based scales.
  - PCK computed correctly against a hand-built fixture.
  - `run.json` schema validation rejects malformed configs.
  - `resolve_checkpoint` passes through a local path and rejects a SHA mismatch
    (download mocked); each catalog entry has a backbone-strict load test.
- **Integration, tiny:** a **2–3 epoch overfit run on ~8 frames** must drive
  train loss down and PCK up — proves targets, loss, optimizer, and loop are
  wired coherently. This is Spec 4's objective gate, the analogue of Spec 1's
  parity gates: fast and it catches "silently trains on garbage targets."
- **Orchestration (non-Qt):** builds the dataset, resolves the checkpoint, writes
  a valid `run.json`, and constructs the correct subprocess command — all without
  launching real training.
- **PoseKit worker (Qt):** mock `Popen`; assert it wires dataset + `run.json` +
  command correctly and that SIGTERM reaches the child on cancel. Mirrors how the
  YOLO worker is tested.

## Explicit non-goals

No MoE / ViTPose+. No flip augmentation or symmetry schema change. No inference-
backend registration, preview, or tracking use (Spec 3). No `training/`
contracts plumbing. No cross-species / multi-dataset joint training. No new
skeleton-model fields in PoseKit.

## Interfaces this spec produces (for later specs)

- `python -m hydra_suite.core.identity.pose.vitpose.training --config run.json`
  — the fine-tuning entry point; emits `best.pt` in the run dir.
- `posekit.core.vitpose_checkpoints.resolve_checkpoint(name_or_path) -> Path`
  and the catalog — reusable by Spec 3's inference backend if it wants to offer
  the same base weights.
- `download_pinned(url, sha256, dest)` — the shared SHA-pinned fetcher.
- A `best.pt` whose `state_dict` matches `build_vitpose(variant, K)` — the
  artifact Spec 3's `backends/vitpose.py` will load and export.
