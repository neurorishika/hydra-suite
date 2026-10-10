# DetectKit SAHI Sliced-Training Runbook

## Why

Full-frame downscaling at camera resolution often merges crowded ant clusters into single bounding boxes, especially at high colony densities. SAHI (Sliced Aided Hyper Inference) solves this by cutting the image into overlapping tiles during both training and inference, allowing the OBB model to learn and detect objects at a natural tile-scale rather than at an undersized full-frame resolution. This runbook shows you how to train a model specifically on sliced data so inference can later apply SAHI to reliably separate clusters.

Every control named below belongs to the shared SAHI settings widget. That widget is described once, with every label, badge and resolution rule, in [SAHI Settings](../user-guide/sahi-settings.md).

## Prerequisites

You have a DetectKit project with:
- OBB-labeled or polygon-labeled data sources already configured
- The OBB-direct role set up with a known `imgsz_obb_direct` (e.g., 640 px)
- At least one training dataset with labeled ants in crowded frames

## Configure Sliced Training

1. **Open the training dialog** in the DetectKit GUI and navigate to the "Training" tab.

2. **Locate the "Sliced dataset / inference (SAHI)" group box.** This is the settings panel that governs both sliced training data generation and the preview mode.

3. **Enable sliced training:**
   - Check the "Enable sliced training + preview" checkbox.
   - Leave **Tile strategy** at **Fit to animal size** (the default, stored as `auto_object`). DetectKit measures the body size from all labels during the build, so it is not a manual setting. The **Body size** row shows "Last build: X px, measured automatically from labels" once a build has run.

4. **Configure tile sizing** (these control how crowded frames are split):
   - Set **Object scales** to `0.05, 0.1, 0.15, 0.2` (the default). Each value is an object scale: the animal's size as a fraction of the tile (body size ÷ tile size). With "Fit to animal size", each scale produces tiles of body size ÷ scale, and the **Tile size** row shows the resulting range (for example `→ 240–960 px over 4 scales`).
     - A larger fraction means a smaller tile and more aggressive crowd-splitting.
     - Scales are stored as fractions only, so changing the model input size does not require translating pixel targets yourself.
   - Leave **Tile overlap** at `0.2` (the default: neighbouring tiles share 20% of a tile). Next to it the widget shows the **whole-animal minimum**, the largest scale + 0.05 (`0.25` for the default set). Below that value an animal at a tile seam can be cut in every tile. The widget then warns and offers **Raise to 0.25**, but it never changes your overlap on its own.
   - Use the live tile-layout preview beside the controls to see the resulting grid over the project’s labelled frame sizes at their native dimensions. Click it to cycle through the image-size distribution; all configured object-scale targets are shown together. Before the first build it is explicitly illustrative; afterward it uses the label-derived body measurement.

5. **Configure negative sampling and merging** (click **Advanced** to show these rows):
   - Set **Minimum retained object area** to `0.25` (default, tiles with < 25% of the object's area are suppressed during slicing to avoid training on severely clipped animals).
   - **Below the floor** is fixed at "Drop fragment" for YOLO: a fragment under the minimum area loses its label.
   - Set **Empty-tile sampling fraction** to `0.15` (default, 15% of background-only tiles are kept, strengthening non-object detection).
   - Leave **Mix full frames** checked (default, ensures the model also learns full-frame context).
   - Set **Merge threshold** to `0.5` (default, overlapping predictions from adjacent tiles are merged when their overlap, IoS by default, exceeds this).
   - Leave **Balance multi-scale training loss** enabled. Every tile remains in the epoch, while `Balance strength` controls inverse-frequency loss weighting: `0.5` is square-root balancing and `1.0` gives exact balance among tile-size groups. Full-frame examples retain their normal weight.
   - This balance mode applies to single-process training. Distributed (DDP) runs retain Ultralytics' standard loader and log that balancing was skipped.

6. **Choose a different tile strategy only when needed:**
   - **Use model input size** (`auto_model`) makes each tile the model input size. The object-scale controls stay visible but are disabled, because labels are not used to set tile size.
   - **Custom tile size** (`custom`) enables the width and height spins. Choose it only when a known camera or acquisition geometry requires a fixed tile size.

## Build + Train

1. **Build the dataset:**
   - Click the **Build datasets** button in the training dialog.
   - The build log will show a new line: `Sliced dataset: <path-to-sliced-data>`.
   - This confirms that sliced tiles have been generated and written to disk.

2. **Train the OBB-direct role:**
   - Set the training role to **OBB-direct** (the role that learns to detect ants at their native scale).
   - Click **Train**.
   - The training will proceed over the sliced dataset; the model will learn detection patterns at the selected relative object scales.
   - Training time may increase due to the larger number of tiles per frame, but the model convergence often improves in crowded-frame scenarios.

## Validate

1. **Enable sliced inference in preview:**
   - In the training dialog, ensure "Enable sliced training + preview" is still checked (this same checkbox gates both training-data generation and preview slicing).
   - Open the **Preview** panel and navigate to a crowded frame (one where ants were previously merged).
   - The preview should now apply SAHI (tile-based inference) automatically.
   - With a published model selected, the preview tiles the way TrackerKit would. Geometry, scale, overlap and merge settings come from the model's sidecar (its primary calibration profile, else its training geometry). Only the enable toggle, the input size and the body size come from the project. The status bar names the source, for example "SAHI preview: training geometry". The preview serves **one** operating scale, the median of the training set, not every training scale.

2. **Inspect cluster separation:**
   - Compare the sliced preview output to the non-sliced baseline (turn off the checkbox to disable slicing temporarily).
   - On crowded frames, you should see individual bounding boxes that were previously merged; the model should now separate clusters.
   - If clusters are still merged, check that:
     - Object scales match your typical object scale (decrease the fractions if tiles are too small or fragmented).
     - Overlap is not too small (0.2 is typical; lower overlap can miss objects at tile edges).
     - Minimum retained object area is not too aggressive (0.25 deliberately rejects severely partial objects).

3. **Run the scale-sweep validation:**
   - Re-run the collaborator's detection-vs-scale curve experiment on validation frames.
   - The sliced model should show a **flattened detection curve** — detection accuracy no longer degrades at high densities.
   - If the curve still shows density-dependent degradation, consider:
     - Retraining with larger object-scale fractions to generate more aggressive tiling.
     - Increasing negative tile fraction to improve false-negative detection in sparse regions.

4. **Check model metadata:**
   - After training completes, the published model file (e.g., `best.pt`) has a sidecar file named after the full model filename: `best.pt.slice_meta.json`.
   - The sidecar is `schema_version: 3` with `model_family: "yolo"`. Its `training_geometry` keeps the build manifest's keys verbatim (`geometry_mode`, `target_sizes`, `object_tile_fraction`, `reference_body_px`, `overlap`, `min_area_ratio`, `imgsz`, and so on), so older readers still work. It adds the canonical keys `object_tile_fractions`, `trained_body_px` and `fragment_policy`. Calibration profiles are stored in the same file.
   - Keep this metadata file with the model. TrackerKit and the DetectKit preview both read it.

## Ship Back

1. **Prepare the model for delivery:**
   - Export the trained OBB-direct model from DetectKit as usual.
   - Ensure the `<model>.pt.slice_meta.json` sidecar is included in the delivery package.

2. **Document the slicing configuration:**
   - Include a note in your delivery that specifies:
     - Geometry mode used (`auto_object` in this runbook).
     - Object scales as model-input fractions (e.g., 0.05, 0.10, 0.15, 0.20).
     - The label-derived reference body pixel size recorded in the sidecar.
     - Overlap and min-area-ratio values.
   - Example note: *"Model trained with SAHI labelled-object tiling; object scales 0.05, 0.10, 0.15, 0.20 of model input; overlap 0.2; reference_body_px measured from labels. Sidecar slice_meta.json included."*

3. **Prepare for TrackerKit inference:**
   - When you select the model in TrackerKit's detection panel (direct mode), TrackerKit reads the sidecar and fills in the SAHI controls from the model's primary profile, else its training geometry. `trackerkit track` does the same headless, and `--sahi-profile` selects a calibration profile explicitly. See [SAHI calibration profiles](../user-guide/detectkit-sahi-calibration.md).
   - Ensure the sidecar is preserved in all downstream storage and version-control systems. Without it, TrackerKit falls back to its saved or default SAHI settings.

---

## Troubleshooting

### Preview shows no improvement or wrong tiling

- **Issue:** Clusters still appear merged, or tiles seem misaligned.
- **Check:** Confirm "Enable sliced training + preview" is checked. With "Fit to animal size", rebuild the sliced dataset so DetectKit can measure the body size from labels. Alternatively, switch to "Custom tile size" and set the dimensions explicitly. If a published model is selected, the preview follows that model's sidecar rather than these training settings. Check the "SAHI preview:" status message, or open the inference settings dialog to override it.

### Training is much slower

- **Expected behavior:** Sliced training processes more tiles, so training time increases. This is normal.
- **Optimization:** If training time is prohibitive, lower the object-scale fractions to generate fewer, larger tiles, but this may reduce cluster separation.

### Scale-sweep curve still shows degradation at high density

- **Diagnosis:** The model may not have learned sufficient crowd-splitting.
- **Solutions:**
  - Retrain with larger object-scale fractions to force more aggressive tiling.
  - Increase "Empty-tile sampling fraction" (e.g., 0.25) to improve sparse-frame accuracy.
  - Verify that the training dataset contains representative crowded frames; if training data is mostly sparse, the model won't learn crowd-splitting.
  - Check that overlap (0.2) and minimum retained object area (0.25) are not too conservative.

### Model runs but inference is not sliced

- **Diagnosis:** SAHI is off in TrackerKit, or the sidecar did not travel with the model.
- **Action:** Confirm `<model>.pt.slice_meta.json` sits next to the model file, then check "Enable sliced inference (SAHI)" in TrackerKit's detection panel (direct mode). The SAHI controls fill in from the sidecar.
