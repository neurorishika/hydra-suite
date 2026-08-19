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
ordinary columns and influences nothing about identity.

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

## Typical Uses

- Build identity classifier training sets.
- Diagnose identity-switch failure regions.
- Export curated individual-level clips/crops.
