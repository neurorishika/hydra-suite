# PoseKit training defaults: flips, crop-sized inputs, AP-10K, central pose

**Status:** In progress on `feat/posekit-training-defaults`.

## Problem

On the ant pose project (8,633 crops of 192x128, crowded colonies):

- YOLO-pose datasets carried no `flip_idx`, so Ultralytics silently disabled
  every flip; the ViTPose trainer had no flips at all.
- Training defaulted to 640px YOLO inputs (3.3x upscaling) and a portrait
  256x192 ViTPose input for landscape crops (half of each input was padding).
- Ultralytics mosaic (on by default) put the labelled animal off-centre next
  to unlabelled neighbours.
- Predictions kept the most confident pose (`max_det=1`), so a clearer
  neighbour replaced the crop's own animal; the same in TrackerKit.
- ViTPose fine-tuning started from human COCO; YOLO was the default backend.

## Decisions (from the user)

1. Left/right keypoint pairs are chosen in the Skeleton Editor, saved in the
   skeleton file and the project (`flip_pairs`), and used by training.
2. Training sizes come from an auto-sizer driven by the crop sizes.
3. Mosaic is an option, off by default.
4. ViTPose-B AP-10K is the default init checkpoint; ViTPose the default backend.
5. The reported pose is always the one nearest the crop centre when a model
   returns several.

## Design

- `core/pose_flip.py` (numpy-free, light): pair validation, `flip_index`,
  `suggest_flip_pairs` (left/right, l/r naming), remapping on keypoint
  move/remove.
- `core/pose_instances.py` (numpy): `instance_centers` (box centre, else
  confident-keypoint mean) and `central_instance_index`: among detections
  with box confidence >= max(0.1, 0.25 x best) (pose models run at conf 1e-4,
  so unfiltered near-zero fragments at the centre would win), the one nearest
  the crop centre; confidence breaks sub-pixel ties. The rule version and
  per-crop limit are part of the TrackerKit YOLO pose cache key; PoseKit's
  prediction cache is bumped to v4 and only stores model-size predictions. `MAX_POSES_PER_CROP = 10` replaces the
  per-crop limit of 1 in PoseKit prediction, `PoseRuntimeConfig` and
  TrackerKit's `PoseYOLOConfig`.
- `posekit/core/training_options.plan_flips`: flips run only when every
  side-named keypoint is paired (`core/pose_flip` side rules: left/right
  words, l/r tokens, camelCase L/R, L1, lf/FL, and l/r glued to a word when
  the mirrored name exists; Suggest from names uses the same rules);
  otherwise flips are off with a note naming the keypoints. A midline-only
  skeleton flips with the identity mapping.
- `YoloNativeBackend` predicts at the model's training size (from the `.pt`,
  or the `.pt` beside an exported artifact), so ONNX/TensorRT artifacts
  exported at that size stay inside their dynamic profile.
- `training/pose_geometry_measure.measure_crop_geometry`: YOLO `imgsz` = the
  p90 crop long side rounded up to 32 (64..640); ViTPose `[H, W]` = median crop
  x 1.25 box padding x Detail on the 32 grid (cap 384). Prediction passes no
  `imgsz`, so Ultralytics uses the size stored in the checkpoint.
- ViTPose `RunConfig` gains `flip_pairs`, `fliplr`, `flipud`; the dataset
  mirrors image + keypoints (`x -> w - x`) and swaps pairs; validation never
  flips.
- Training Runner: ViTPose + AP-10K default; Config stays visible for ViTPose
  (it reads epochs/batch/device) with YOLO-only rows hidden; flip
  probabilities (0.5/0.5) for both backends; mosaic checkbox (off), passed
  even with augmentations off so Ultralytics' defaults don't apply.

## Bugs on main fixed along the way

- PoseKit YOLO training crashed at start: the SAHI scale-balance hook read
  PoseKit's list-shaped `manifest.json` as a dict.
- ViTPose training could not read PoseKit's COCO export (`images/x.png` was
  joined under `images/` again).
- PoseKit's YOLO prediction subprocess keyed results by Ultralytics'
  `image0.jpg` placeholders, so every batch overwrote the last and no key
  matched a real image (Evaluation / Active Learning with YOLO).
- The Training Runner and Evaluation dashboard reset a ViTPose prediction
  backend to YOLO.
