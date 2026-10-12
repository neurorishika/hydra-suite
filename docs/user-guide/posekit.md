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
  inference over the labeled frame set) in the Model group. ViTPose is the
  default backend for prediction and training.
- **Training Runner** — fine-tune a ViTPose checkpoint on the project's YOLO-pose
  dataset; supports Use-Latest resume, a loss-curve plot, and the same evaluation
  dashboard as the other backends.

### Weight acquisition

- **Auto download (default): AP-10K.** The `vitpose-b-ap10k` catalog entry is
  ViTPose-B trained on AP-10K (animal pose, 54 species). It downloads
  `torch/ap10k/vitpose-b-ap10k.pth` from the `JunkyByte/easy_ViTPose` Hugging Face
  mirror on first use (the upstream release hosts it on OneDrive only), verifies it
  against a pinned SHA-256, and caches it locally.
- **Auto download: AP-10K L and H.** `vitpose-l-ap10k` (1.2 GB) and
  `vitpose-h-ap10k` (2.4 GB) come from the same mirror, also hash-pinned. Choosing a
  catalog checkpoint sets the model size (B/L/H) to match.
- **Auto download: COCO-B.** The `vitpose-b-coco` entry downloads the human
  COCO-17 checkpoint `vitpose-b.pth` from `nielsr/vitpose-original-checkpoints`.
- **Browse:** point at any local `.pth` checkpoint, for example an APT-36K
  checkpoint or one you trained yourself with the training CLI below.

### Training

With ViTPose selected, the Training Runner shows the full recipe. *Recommended
settings* restores these defaults (an mmpose ViTPose / RTMPose-style fine-tune):

| Group | Setting | Default | What it does |
|---|---|---|---|
| Augmentation | rotation / rotation prob | 40° / 0.6 | Random rotation, N(0, r/2) clipped to ±r |
| | brightness, contrast, saturation | 0.3 each | Colour jitter (± fraction) |
| | hue | 0.02 | Hue shift (fraction of the colour circle) |
| | blur prob, noise prob | 0.1, 0.1 | Gaussian/median blur; sensor noise |
| | occlusion prob | 0.3 | One random rectangle over the crop, so hidden keypoints are learnt from the rest of the animal |
| | scale / aspect jitter, flips | 0.2 / 0.2, 0.5 / 0.5 | As for YOLO (flips need the Skeleton Editor's left/right pairs) |
| Optimisation | learning rate, weight decay | 5e-4, 0.1 | AdamW |
| | layer decay, drop path | Auto | The variant's ViTPose values |
| | warmup epochs | 3 | Linear warmup, then cosine decay (per batch) |
| | early stopping patience | 40 | Stop after 40 epochs without a better validation score |
| | loader workers | min(8, CPUs) | Parallel data loading |
| | EMA, mixed precision | on | Averaged weights are validated and saved; bf16 on NVIDIA GPUs |
| Validation | fraction, split by, seed | 0.2, source frame, 0 | Whole source frames are held out, so validation crops are not near-copies of training crops |
| | ignore occluded | off | Occluded keypoints get no loss |
| | flip test | on | Validate on the crop and its mirror, averaged |

Epochs, batch size and device are in the Config group (shared with YOLO), as are
auxiliary datasets, which ViTPose now uses too. The loss plot follows the run,
and the log reports PCK@0.05, PCK@0.1 and the mean normalised error (NME: pixel
error / √animal box area) every epoch. `best.pt` is the epoch with the highest
PCK@0.05 (lower NME breaks ties). The settings are kept between sessions.

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

## Training Defaults

- **Input size from the crops.** *Auto from crop size* (on by default) sizes the
  model input from the crops' pixel dimensions: YOLO trains at the smallest multiple
  of 32 that holds the crops' long side (192 for 192x128 crops, instead of 640), and
  ViTPose gets an input with the crops' aspect ratio at about one input pixel per crop
  pixel (*Detail* scales it; 160x256 HxW for 192x128 crops). Prediction always uses
  the size a model was trained at.
- **Flips.** Training mirrors crops left-right and up-down (probability 0.5 each).
  A mirror turns a left antenna into a right one, so the skeleton needs its
  left/right pairs: set them in the **Skeleton Editor** (*Suggest from names* pairs
  `left_*`/`right_*`, `l_*`/`r_*`). They are saved in the project and in skeleton
  files (`flip_pairs`). If any keypoint whose name marks a side (`left_*`, `l_*`,
  `antL`, `L1`, `lf_*`, ...) is not in a pair, flips stay off and the Training
  Runner names the keypoints (tick *Flip with unpaired side-like names* in Config
  when they are really midline keypoints); a skeleton with only midline keypoints
  flips without swapping. Loading a skeleton file saved without pairs keeps the current pairs only
  for keypoints it still has, matched by name.
- **Mosaic off.** Ultralytics' mosaic tiles four crops into one image, which puts
  the labelled animal off-centre next to unlabelled ones; it is off by default and
  can be switched on under Augmentations.

## Crops With Several Animals

A crop belongs to the animal at its centre; neighbours reaching into the frame are
not labelled. When a YOLO-pose model finds several animals in a crop (up to 10 are
kept), PoseKit and TrackerKit report the pose nearest the crop centre among the
confident detections (confidence at least 0.1 and at least a quarter of the best
one), not simply the most confident, so a clearer neighbour no longer replaces the
crop's own animal while near-zero fragments are ignored. ViTPose predicts one pose
per crop. TrackerKit pose caches made with YOLO-pose before this change are
recomputed.

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

