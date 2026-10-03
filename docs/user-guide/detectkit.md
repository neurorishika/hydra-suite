# DetectKit

DetectKit is the detection model training tool, launched via `detectkit`.

## Purpose

Train, evaluate, and manage YOLO detection models for use in TrackerKit.

## Launch

```bash
detectkit
```

## Workflow

1. Curate detection training datasets from TrackerKit exports or external sources.
2. Configure YOLO training parameters (epochs, batch size, image size).
3. Launch training and monitor loss curves.
4. Evaluate model performance on validation sets.

## Key Features

- Dataset panel for assembling and inspecting training data
- Training panel with configurable hyperparameters
- Validation-set evaluation with precision, recall, mAP50, and mAP50-95
- Side-by-side comparison of completed training runs
- Integration with TrackerKit's dataset generation exports

### Evaluate trained models

Open **Evaluate** from the main toolbar or **Evaluate…** in the Dataset panel.
DetectKit lists completed YOLO training runs that still have both their model
artifact and derived dataset. Select one or more runs, then choose **Evaluate
Selected** or **Evaluate All Available**.

Each model is evaluated on the held-out `val` split recorded for that training
run. Detection and OBB runs report box metrics; segmentation runs report mask
metrics. Results include precision, recall, mAP50, mAP50-95, and measured
inference time. Evaluation history is saved with the project so runs can be
compared again after reopening it. SAM3 concept runs use a different evaluation
stack and are shown as unavailable in this YOLO evaluator.

### Dataset panel

Dataset curation changes are recoverable. Removing selected images with
**Delete** or **Backspace** moves the images and matching labels into the
project's `artifacts/recovery/` folder. The frame-, source-, and whole-project
**Clear labels** actions save exact copies there before truncating the working
label files. This also protects linked sources that live outside the project:
their recovery payload is owned by the project that initiated the change.

Use **Undo last dataset change** or the platform Undo shortcut (Command-Z on
macOS, Ctrl-Z elsewhere) to restore the newest operation. Undo will not
overwrite an image recreated at the same path or labels edited after they were
cleared; resolve that conflict manually and try again. Nested images are shown
by their paths relative to the source's `images/` directory, with the absolute
path available in the row tooltip.

## SAM2 and SAM3 escalation

The **Escalation** group in the Tools panel upgrades and extends labels with
Segment Anything models. Both run on every device (CUDA, Apple MPS, CPU) and
are installed by default:

- **Escalate to segment (SAM2)** turns existing box labels into polygon masks.
- **Semantic escalation (SAM3)** finds every instance of a text prompt (e.g.
  `ant`), including animals missing from your labels.

Results are staged for frame-by-frame review; nothing changes until you
accept. Both are also available headless as `detectkit escalate sam2` /
`detectkit escalate sam3`. Stock SAM3 weights are licence-gated and need a
one-time Hugging Face login, and SAM3 *training* is an optional CUDA-only
add-on. See [SAM2 and SAM3: install and run](sam-install-and-run.md) for
devices, weights, the CLI and training setup, and
[Semantic Escalation (SAM3)](detectkit-semantic-escalation.md) for the SAM3
workflow.
