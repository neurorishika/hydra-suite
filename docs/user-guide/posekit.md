# PoseKit

PoseKit is now organized under `hydra_suite.posekit` and launched via `posekit`.

## Full UI Control Reference

For complete PoseKit interface documentation (every form control, question labels, defaults, value-selection guidance, and failure modes), see:

- `/reference/ui-components-posekit/`

## Capabilities

- Project setup wizard (classes/keypoints/skeleton)
- Frame-by-frame keypoint annotation
- Autosave with safer write semantics
- Metadata/tagging and smart frame selection
- Split generation and training/evaluation dialogs

## SLEAP Backend Setup

For SLEAP env creation and ONNX/TensorRT integration setup, see:

- `/getting-started/integrations/`

## ViTPose Backend

ViTPose is a third selectable pose backend alongside YOLO-pose and SLEAP, available in:

- **Predict Keypoints** (current-frame inference) and **Predict Dataset** (batch
  inference over the labeled frame set) in the Model group.
- **Training Runner** — fine-tune a ViTPose checkpoint on the project's YOLO-pose
  dataset; supports Use-Latest resume, a loss-curve plot, and the same evaluation
  dashboard as the other backends.

### Weight acquisition

- **Auto download:** the COCO-B catalog entry (`vitpose-b-coco`) downloads
  `vitpose-b.pth` from the `nielsr/vitpose-original-checkpoints` Hugging Face repo on
  first use, verifies it against a pinned SHA-256, and caches it locally. This is a
  human-COCO-17-keypoint checkpoint — a reasonable general-purpose starting point, not
  an animal-tuned model.
- **Browse:** point at any local `.pth` checkpoint — this is how you load an
  animal-pretrained checkpoint (for example fine-tuned on AP-10K or APT-36K) or a
  checkpoint you trained yourself with the training CLI below. Animal (AP-10K/APT-36K)
  checkpoints are **not** bundled or auto-downloadable; Browse is the only path to them
  today.

### Training

ViTPose can also be trained/fine-tuned outside the GUI via its standalone CLI:

```bash
python -m hydra_suite.core.identity.pose.vitpose.training --config run.json
```

See [Runtime Integration Guide: ViTPose Training CLI](../developer-guide/runtime-integration.md#vitpose-training-cli)
for the full `RunConfig` field reference (checkpoint/variant selection, dataset/output
paths, device, epochs, batch size, and other hyperparameters).

## Key UX Concepts

- Keypoint order is semantic and must remain stable.
- Visibility flags encode present/occluded/missing states.
- Project settings changes can require label migration.

## Core Hotkeys

- `A/D`: previous/next frame
- `Q/E`: previous/next keypoint
- `Ctrl+S`: save
- `V/O/N`: visible/occluded/missing mode

## Output Expectations

- YOLO pose labels are normalized text files per image.
- Project metadata and optional posekit artifacts are stored in project output directories.

## Recommended Workflow

1. Finalize keypoint spec before large-scale labeling.
2. Label a pilot subset and run sanity checks.
3. Generate split files and train a baseline.
4. Use model-assisted passes and active learning to iterate.
