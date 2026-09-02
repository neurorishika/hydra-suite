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

The dataset panel's "Clear labels" actions (frame, source, and whole-project
scope) are destructive and irreversible: they truncate the matching label
file(s) to empty rather than deleting them, so the files remain on disk but
lose their annotations. Images themselves are never touched.
