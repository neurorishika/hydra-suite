# ClassKit

ClassKit is the classification and embedding toolkit, launched via `classkit`.

## Purpose

Build identity classifiers from animal crops using embedding models, clustering, and active learning.

## Launch

```bash
classkit
```

## Workflow

1. Create or open a project with source image directories.
2. Ingest and embed crops using a backbone model.
3. Cluster embeddings and visualize with UMAP.
4. Label identity classes manually or via AprilTag auto-labeling.
5. Train a classification head and evaluate results.
6. Export labeled datasets for downstream use.

## Key Features

- Embedding extraction with configurable backbone models
- UMAP-based dimensionality reduction and visualization
- FAISS-powered similarity search and clustering
- AprilTag auto-labeling for marker-based identity assignment
- Active learning for efficient labeling
- Export to Parquet/CSV, ImageFolder, and Ultralytics classification formats

## Training Augmentation Defaults

The training dialog's *Space and Augmentations* tab starts with conservative
defaults tuned for canonical animal crops:

| Setting | Default | Why |
|---|---|---|
| Horizontal / vertical flip | 0 | Left/right are class labels for heading models; use *label expansion* for label-aware mirroring. |
| Hue jitter | 0.01 | Colour-tag identity depends on hue; keep it minimal. |
| Saturation / brightness / contrast | 0.10 | Mild lighting variation. |
| Scale jitter | 0.20 | Window size varies 0.8x-1.2x about the crop centre, so the model tolerates the animal appearing larger or smaller. |
| Aspect ratio jitter | 0.20 | The crop window's width:height ratio varies 0.83x-1.2x at constant area. The crop centre never moves and the animal is never stretched. |

Scale and aspect jitter apply to train samples only. For YOLO-classify they are
written as extra pre-fitted copies per training image
(`canonical_aug_copies`, default 3); validation and test images stay clean.

