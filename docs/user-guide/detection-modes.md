# Detection Modes

## Background Subtraction

Best when animals move on stable backgrounds.

### What It Means

Foreground is inferred by frame-to-background difference, then morphology/refinement is applied.

### Key Controls and Tradeoffs

- `SUBTRACTION_THRESHOLD`
  - Lower: more sensitive, more noise.
  - Higher: cleaner, may miss faint targets.
- `ENABLE_ADAPTIVE_BACKGROUND`
  - Helps with slow lighting drift.
  - Can absorb stationary animals if too aggressive.
- Morphology (`MORPH_KERNEL_SIZE`, split/dilation toggles)
  - Larger kernels smooth noise but can merge close animals.

### Use When

- Arena is static.
- Lighting is controlled or slowly changing.
- Target count is moderate and movement is visible.

## YOLO OBB

Best for complex backgrounds or weak motion contrast.

### What It Means

A model predicts oriented boxes per frame; detections feed the same tracking pipeline.

### Key Controls and Tradeoffs

- `YOLO_CONFIDENCE_THRESHOLD`
  - Lower: catches more objects, includes more false positives.
  - Higher: precision improves, recall may drop.
- `YOLO_IOU_THRESHOLD`
  - Controls suppression overlap behavior.
- `YOLO_DEVICE`, TensorRT options
  - Throughput and startup complexity vary by platform.

### Use When

- Targets can be stationary.
- Background subtraction is unstable.
- You have a suitable OBB model.

### Sliced Inference (SAHI)

In direct mode, "Enable sliced inference (SAHI)" tiles each frame so that
small animals keep more pixels. When the model carries a
`.slice_meta.json` sidecar, the tiling controls fill in from it (its
calibration profile, else its training geometry). See [SAHI
Settings](sahi-settings.md) for every control.

## Practical Selection Matrix

| Scenario | Preferred Mode | Why |
|---|---|---|
| Static arena, moving insects | Background subtraction | Simple and fast |
| Stationary animals / cluttered scene | YOLO OBB | Learned visual cues |
| Very large videos with limited GPU | Background subtraction + resize | Better throughput control |
| Heterogeneous data across setups | YOLO OBB | More robust across conditions |
