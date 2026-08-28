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

## Identity Classifiers and Non-Identifying Classes

Not every CNN classifier configured for individual analysis feeds identity
resolution. Only classifiers marked **Unique identifier** in the identity
panel contribute to the identity catalog, the Hungarian assignment that keeps
identities unique, and the relink veto that stitches fragments across
occlusions. Every other classifier (behavior, sex, caste, ...) is exported as
ordinary columns and influences nothing about identity. In the clean
`<video>_tracks.csv` those appear as `<label>_class` / `<label>_conf` (one
pair per factor for a multi-factor model); identity classifiers do not, since
their result is the `identity` / `identity_id` block.

When more than one classifier is marked as a unique identifier, they
**combine**: thorax colour x abdomen shape produces one composite identity
per combination (`red_notch`, `blue_plain`, ...), not two catalogs competing
for the same track.

Some class values are not identity information at all -- an untagged animal
classified `notag`, a smudged or ambiguous tag read. Declaring such a value
as **non-identifying** (per classifier, in the identity panel) removes every
composite it produces from the catalog entirely, so any number of tracks can
carry that label at once without displacing a genuinely identified animal.
Marks can be declared at three granularities, and the choice matters more
than it looks:

| Mark | Matches | Use when |
|---|---|---|
| `notag` | that value on **any** axis of that model | the class carries no identity wherever it appears |
| `front:notag` | that value on the named factor only | only one tag position is unreadable |
| `notag_notag` | that whole composite label | only the *fully* unmarked animal is non-identifying |

A bare mark is the aggressive form: marking `notag` on a two-tag scheme drops
`red_notag` and `notag_blue` from the catalog too, so an animal with **one**
readable tag also stops resolving to an identity. If a half-readable animal
should still be identified, mark the whole composite (`notag_notag`) instead,
which removes only the fully unmarked combination. (Per-axis evidence from the
readable tag is retained for the relink veto either way -- it is catalog
membership, not evidence, that the mark removes.)

Tracks whose every axis reads a declared non-identifying value get a
descriptive label instead of "unknown": `IdentityFinalLabel` is set to the
observed composite (e.g. `notag_notag`), `IdentityFinalSource` is
`nonidentifying`, and `IdentityFinalID` stays `0` -- the unknown slot -- so no
downstream consumer keying on a resolved identity slot can mistake the shared
label for a real identity.

The slot travels with the label into the plain `<video>_tracks.csv` as
`identity_id` (alongside `identity`, `identity_confidence`, and
`identity_source`). **Group by `identity_id`, not by `identity`**: five rows
labelled `notag_notag` all carry `identity_id == 0` and are five different
animals, distinguished by `id` (the trajectory), not one animal seen five
times.

The honest limitation: these animals get no
identity resolution or identity-based fragment stitching across occlusions;
relinking for them falls back to spatial and motion gates alone.

See the [per-detection processing schematic](../schematics/trackerkit_pipeline.md)
for how this fits into the wider detection-to-identity pipeline.

## Final-Output Vocabulary

The debug `*_tracking_final_with_individual.csv` export carries a fixed,
explicit vocabulary for identity provenance -- no blank/NaN values:

- **`IdentityFinalSource`** is never blank. A row that carries no identity
  provenance (no classifier ran, no cache evidence, no relink veto) reads the
  literal string `"none"`, not an empty cell -- readers should treat blank as
  legacy data from before this vocabulary shipped, never as a valid current
  value.
- **`IdentityFinalSmoothedLabel` / `IdentityFinalSmoothedConfidence`** are an
  ungated *record* of the cache's forward-backward smoothed evidence for
  every row that has any (its argmax label and posterior), not a
  display value -- they are no longer threshold-gated by
  `IDENTITY_DISPLAY_THRESHOLD`. A row with no cache evidence (no
  `DetectionID`, e.g. a filled/interpolated row) reads
  `IdentityFinalSmoothedLabel="unknown"` with confidence `0.0`, which is a
  fact about that row lacking evidence, not a low-confidence classification.
- **`IdentityFinalConflictResolved`** is always boolean (`True`/`False`),
  never NaN.
- **Trajectories are dense.** A written trajectory has no interior NaN
  position/orientation and no gap in its frame sequence -- interior gaps are
  interpolated and filled (kept as `State="occluded"` rows with empty
  `DetectionID` so a consumer can still exclude them from evidence). A
  trajectory's leading and trailing runs that carry no detection and no
  position are dropped outright rather than left as dangling NaN rows, so
  every trajectory's first and last row has a real position.

Rendered/exported video overlays label each track by its resolved
`IdentityFinalLabel` (falling back to `IdentityFinalSmoothedLabel`, then
`UniqueIdentityKey`) using this priority chain, tier by tier:

1. `IdentityFinalLabel`, if the row has one and it is not the literal string
   `"unknown"`.
2. Otherwise `IdentityFinalSmoothedLabel`, under the same rule.
3. Otherwise `UniqueIdentityKey` -- the raw per-frame classifier/tag
   evidence -- but **only** when neither of the first two tiers had *any*
   value, informative or not.

A blank/missing value at a tier defers to the next tier, but the literal
value `"unknown"` does **not** -- it is treated as the resolved answer for
that track (this trajectory genuinely has no identity, not "we don't know
yet"), so the overlay stops there and falls back to a plain `TrajectoryID`
label/color instead of continuing down the chain to raw evidence. This
matters because `"unknown"` is a common, expected value: any trajectory the
fragment solver could not confidently resolve is written with
`IdentityFinalLabel="unknown"` (see above), and a real run's unresolved
fraction can be substantial. Without this rule, an unresolved track's video
label/color would silently come from raw, un-smoothed per-frame evidence --
the exact frame-to-frame flicker this priority chain exists to eliminate --
rather than from a stable value. The on-video label matches what the CSV
reports for that track whenever a tier resolved with an informative value;
for a genuinely unresolved track, both the CSV (`"unknown"`) and the video
(plain `TrajectoryID`) consistently signal "not identified," just with
different literal tokens.

## Typical Uses

- Build identity classifier training sets.
- Diagnose identity-switch failure regions.
- Export curated individual-level clips/crops.
