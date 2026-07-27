# DetectKit SAHI Sliced-Training Runbook

## Why

Full-frame downscaling at camera resolution often merges crowded ant clusters into single bounding boxes, especially at high colony densities. SAHI (Sliced Aided Hyper Inference) solves this by cutting the image into overlapping tiles during both training and inference, allowing the OBB model to learn and detect objects at a natural tile-scale rather than at an undersized full-frame resolution. This runbook shows you how to train a model specifically on sliced data so inference can later apply SAHI to reliably separate clusters.

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
   - Leave geometry mode at the default `auto_object` (this mode automatically tiles around ant bounding boxes).

4. **Configure tile sizing** (these control how crowded frames are split):
   - Set **Target sizes (px, comma)** to `200, 300, 400` (default).
     - Explanation: larger target apparent size → smaller tiles → more aggressive crowd-splitting.
     - The model will be trained to detect ants at these apparent sizes; SAHI inference will request tiles scaled to match these training sizes, so the learned detector sees familiar scales at inference time.
   - Set **Object tile fraction** to `0.15` (default, controls the fraction of the tile occupied by a typical object).
   - Set **Overlap** to `0.2` (default, creates a 20% border overlap between adjacent tiles to reduce edge artifacts).

5. **Configure negative sampling and merging:**
   - Set **Min area ratio** to `0.1` (default, tiles with < 10% of the object's area are suppressed during slicing).
   - Set **Negative tile fraction** to `0.15` (default, 15% of tiles generated will be background-only, strengthening non-object detection).
   - Leave **Mix full frames** checked (default, ensures the model also learns full-frame context).
   - Set **Merge threshold** to `0.5` (default, overlapping predictions from adjacent tiles are merged when IoU exceeds this).

6. **Optional: Set Reference body px** if `auto_object` mode gives poor results in preview:
   - Leave at `0.0` (auto/imgsz) if you want automatic sizing, but note that preview fidelity depends on accurate object size estimation.
   - If preview looks wrong (e.g., tiles are too large or too small), manually set this to your typical ant's pixel width in a native-resolution frame (e.g., 20–30 px for small ants).
   - Alternatively, use the `custom` geometry mode and set explicit "Custom tile W" and "Custom tile H" values.

## Build + Train

1. **Build the dataset:**
   - Click the **Build datasets** button in the training dialog.
   - The build log will show a new line: `Sliced dataset: <path-to-sliced-data>`.
   - This confirms that sliced tiles have been generated and written to disk.

2. **Train the OBB-direct role:**
   - Set the training role to **OBB-direct** (the role that learns to detect ants at their native scale).
   - Click **Train**.
   - The training will proceed over the sliced dataset; the model will learn detection patterns at the target tile scales (200, 300, 400 px).
   - Training time may increase due to the larger number of tiles per frame, but the model convergence often improves in crowded-frame scenarios.

## Validate

1. **Enable sliced inference in preview:**
   - In the training dialog, ensure "Enable sliced training + preview" is still checked (this same checkbox gates both training-data generation and preview slicing).
   - Open the **Preview** panel and navigate to a crowded frame (one where ants were previously merged).
   - The preview should now apply SAHI (tile-based inference) automatically.

2. **Inspect cluster separation:**
   - Compare the sliced preview output to the non-sliced baseline (turn off the checkbox to disable slicing temporarily).
   - On crowded frames, you should see individual bounding boxes that were previously merged; the model should now separate clusters.
   - If clusters are still merged, check that:
     - Target sizes match your typical object scale (increase target sizes if tiles are too small).
     - Overlap is not too small (0.2 is typical; lower overlap can miss objects at tile edges).
     - Min area ratio is not too aggressive (0.1 allows small partial objects).

3. **Run the scale-sweep validation:**
   - Re-run the collaborator's detection-vs-scale curve experiment on validation frames.
   - The sliced model should show a **flattened detection curve** — detection accuracy no longer degrades at high densities.
   - If the curve still shows density-dependent degradation, consider:
     - Retraining with smaller target sizes (e.g., 150, 250, 350) to generate more aggressive tiling.
     - Increasing negative tile fraction to improve false-negative detection in sparse regions.

4. **Check model metadata:**
   - After training completes, the published model file (e.g., `model.pt`) will have a sidecar file `model.slice_meta.json`.
   - This JSON file records the trained slicing geometry: `geometry_mode`, `target_sizes`, `reference_body_px`, `overlap`, `min_area_ratio`, etc.
   - Keep this metadata file with the model — it documents the exact slicing configuration used during training.

## Ship Back

1. **Prepare the model for delivery:**
   - Export the trained OBB-direct model from DetectKit as usual.
   - Ensure the `<model>.slice_meta.json` sidecar is included in the delivery package.

2. **Document the slicing configuration:**
   - Include a note in your delivery that specifies:
     - Geometry mode used (`auto_object` in this runbook).
     - Target sizes in pixels (e.g., 200, 300, 400).
     - Reference body pixel size if manually set (or "auto" if left at 0.0).
     - Overlap and min-area-ratio values.
   - Example note: *"Model trained with SAHI auto_object tiling; target sizes 200, 300, 400 px; overlap 0.2; reference_body_px=auto. Sidecar slice_meta.json included."*

3. **Prepare for TrackerKit inference:**
   - The `slice_meta.json` sidecar will later be read by TrackerKit's SAHI inference pipeline to auto-configure slicing for validation and production inference.
   - **Note:** TrackerKit's automatic sidecar reading is a separate future specification; for now, the metadata is documentation. When that spec is complete, the TrackerKit inference engine will automatically match the SAHI parameters to this model's trained geometry.
   - Ensure the sidecar is preserved in all downstream storage and version-control systems.

---

## Troubleshooting

### Preview shows no improvement or wrong tiling

- **Issue:** Clusters still appear merged, or tiles seem misaligned.
- **Check:** Confirm "Enable sliced training + preview" is checked. If using `auto_object`, set "Reference body px" to your typical object's pixel size (e.g., 20–30 px). Alternatively, switch to `custom` mode and explicitly set tile dimensions.

### Training is much slower

- **Expected behavior:** Sliced training processes more tiles, so training time increases. This is normal.
- **Optimization:** If training time is prohibitive, you can reduce "Target sizes" (e.g., 250, 350 instead of 200, 300, 400) to generate fewer tiles, but this may reduce cluster separation.

### Scale-sweep curve still shows degradation at high density

- **Diagnosis:** The model may not have learned sufficient crowd-splitting.
- **Solutions:**
  - Retrain with smaller target sizes (e.g., 150, 250, 350) to force more aggressive tiling.
  - Increase "Negative tile fraction" (e.g., 0.25) to improve sparse-frame accuracy.
  - Verify that the training dataset contains representative crowded frames; if training data is mostly sparse, the model won't learn crowd-splitting.
  - Check that overlap (0.2) and min-area-ratio (0.1) are not too conservative.

### Model runs but inference is not sliced

- **Diagnosis:** The downstream inference system (TrackerKit) may not be reading or applying SAHI parameters.
- **Action:** Until TrackerKit's automatic sidecar reading is implemented, manually configure SAHI in your inference system using the parameters recorded in `slice_meta.json`.
