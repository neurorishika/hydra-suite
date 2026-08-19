# Individual Analysis

Individual analysis tools extract per-track crops and metadata for identity-focused workflows.

## What It Provides

- Crop extraction around detections/trajectories.
- Identity-oriented dataset output structure.
- Optional color/marker-based workflows depending on method configuration.

## Main Controls

- `ENABLE_IDENTITY_ANALYSIS`
- `IDENTITY_METHOD`
- Crop framing (`canonical_margin`, `apriltag_crop_padding` for AprilTag)
- Output format and destination options

## Feature Meaning

- Crop framing for every model- and dataset-facing crop is driven by one dial,
  `canonical_margin` (in `ADVANCED_CONFIG`, default `1.3`), together with
  `REFERENCE_BODY_SIZE` and `reference_aspect_ratio`, which define a fixed canonical
  canvas. A larger margin improves context but increases storage and may include
  neighbors; a smaller margin is efficient but risks truncation near boundaries.
- AprilTag crops are the one exception: they are plain axis-aligned patches (a tag is
  a rigid printed square that any rotation or rescale degrades), controlled by
  `apriltag_crop_padding` (default `0.0` = the detection's exact axis-aligned extent;
  range `-0.5` to `2.0`). Sufficiently negative values can shrink the padded box to
  zero area and empty the crop entirely -- this now logs a one-time warning telling
  you to raise the value.
- Method choice should reflect your marker protocol (none/color/apriltag/custom).

## Typical Uses

- Build identity classifier training sets.
- Diagnose identity-switch failure regions.
- Export curated individual-level clips/crops.
