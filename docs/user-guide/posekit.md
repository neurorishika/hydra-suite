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
python -m hydra_suite.core.individual.pose.vitpose.training --config run.json
```

See [Runtime Integration Guide: ViTPose Training CLI](../developer-guide/runtime-integration.md#vitpose-training-cli)
for the full `RunConfig` field reference (checkpoint/variant selection, dataset/output
paths, device, epochs, batch size, and other hyperparameters).

## Key UX Concepts

- Keypoint order is semantic and must remain stable.
- Visibility flags encode present/occluded/missing states.
- Project settings changes can require label migration.

## Smart Select

**Smart Select…** (left panel) picks the next frames to label with the same
methodology as ClassKit's candidate batches:

1. Every frame is embedded (DINOv2 by default; cached per model, so later runs
   are instant) and clustered with K-means (about √N clusters by default).
2. A batch is built from a mix of signals, in this order:

    | Slot | Default | Picks |
    |---|---|---|
    | Near model errors | 20% | Unlabeled frames that look like labeled frames the model gets wrong (mean keypoint error / object size above *Error above*, 0.05 by default) |
    | Model unsure | 25% | Frames with the lowest mean keypoint confidence |
    | Rare clusters | 10% | Frames from small or barely-labeled clusters (weight `size^-α`) |
    | Fill gaps | 30% | Frames that most reduce the distance from unlabeled frames to their nearest labeled one |
    | Typical | 10% | Frames from dense clusters with few labels |
    | Random checks | the rest | Random frames |

    The model slots use the cached predictions of the model selected in the
    Model panel (run **Predict Dataset…** first). Without predictions their
    share goes to *Fill gaps*.
3. The candidate table lists the batch in order with the reason for each pick.
   Remove rows with Delete, or open the **Embedding Explorer** to see every
   frame on a UMAP map with the batch highlighted and click points in or out.
4. **Start Labeling** adds the batch to the labeling set, switches the list to the
   *Smart Select batch* sort and opens the first unlabeled frame; Next / Next
   unlabeled follow the batch order (Next unlabeled wraps around to open frames
   earlier in the batch). **Add to Labeling Set** adds it without changing the
   sort. The batch order is saved with the project, and reopening a project with
   an unfinished batch switches back to the batch sort.

In Frame Mode, Smart Select picks whole source frames and adds every individual on
each picked frame; a frame that is partly labeled says so in the Detail column.
Frame Mode groups images by the `did<detection id>` crop names that FilterKit
exports; with other file names every image counts as its own frame, and the
dialog says so.

The labeling set is saved by image path, so it (and the batch) survives images
being added or removed between sessions.

Images that can't be read are skipped (the status line counts them) and are never
picked; they are remembered, so later runs don't retry them until the file
changes. Choosing HDBSCAN as the clustering method makes the cluster count a
minimum cluster size; it is slow above ~10,000 frames.

Clustering also writes `posekit/clusters/clusters.csv` for every frame, which the
*Cluster id* sort and the cluster-stratified export split use.

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

## Training Augmentation

The YOLO-pose and ViTPose training dialogs keep the crop centre fixed. Besides the
Ultralytics controls (HSV, rotation) it has two window augmentations:

| Setting | Default | Effect |
|---|---|---|
| scale jitter | 0.20 | The crop window is resized 0.8x-1.2x about its centre. |
| aspect jitter | 0.20 | The window's width:height ratio varies 0.83x-1.2x at constant area; the animal is never stretched. |

Keypoints, boxes and classes are transformed with the image; keypoints pushed
outside the window become not-visible, and a sample whose animal would be
mostly cut away is left un-jittered. Ultralytics' own `translate` (shifts the
centre) and `scale` (isotropic zoom) now default to 0. Training launches through
`hydra_suite.training.ultralytics_entrypoint`, which installs the window
augmentation from the `HYDRA_WINDOW_SCALE_JITTER` / `HYDRA_WINDOW_ASPECT_JITTER`
environment variables. Mosaic (an Ultralytics default) still tiles several crops
and is independent of these settings.

**ViTPose** (native trainer) has the same two controls in its *ViTPose Config*
group, with the same defaults and meaning. They are written to `run.json` as
`scale_jitter` / `aspect_jitter`; a `run.json` without `scale_jitter` keeps the
legacy behaviour (random zoom of about +/-30%, no aspect jitter), while an
explicit value, including 0, switches to the centred window jitter. The window
is re-cropped before the usual full-extent box and fit to the model input, with
black padding, and keypoints pushed outside it become not-visible. Rotation
(+/-40 degrees, 60% of samples) and photometric jitter are unchanged. SLEAP
training is external and does not get these controls.

