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
- Integration with TrackerKit's dataset generation exports

### Dataset panel

The dataset panel's "Clear labels" actions (frame, source, and whole-project
scope) are destructive and irreversible: they truncate the matching label
file(s) to empty rather than deleting them, so the files remain on disk but
lose their annotations. Images themselves are never touched.
