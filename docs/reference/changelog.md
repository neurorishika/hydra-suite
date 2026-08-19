# Changelog

The canonical changelog is maintained in the repository root:

- [CHANGELOG.md](https://github.com/neurorishika/hydra-suite/blob/main/CHANGELOG.md)

For documentation-specific migrations in this branch:

- Added MkDocs Material docs system.
- Reorganized docs into audience-based tracks.
- Added mkdocstrings API reference pages.
- Standardized command terminology to `posekit`.

## Identity heads, composite catalogs, non-identifying classes (unreleased)

- **Breaking:** `UniqueIdentityKey` now contains identity-head sources only.
  Non-identity classifiers (behavior, sex, caste, ...) no longer appear in
  this column. Downstream parsers of `UniqueIdentityKey` will see fewer
  sources per row than before.
- **Breaking:** configurations with two or more classifiers marked "Unique
  identifier" now produce a cross-product catalog (one composite identity per
  axis combination) instead of a union of separate catalogs. Prior results
  for such multi-identity-model configurations were incorrect (each model
  competed for the same Hungarian columns); the new cross-product behavior is
  the correct one and results will differ.
- **New, opt-in:** identity classifiers can declare `non_identifying_classes`
  (bare class, `factor:class`, or whole composite label) to exclude those
  composites from the catalog. Tracks that only ever show non-identifying
  evidence are labelled with the composite (e.g. `notag_notag`) but keep
  `IdentityFinalID == 0` and `IdentityFinalSource == "nonidentifying"` -- they
  are tracked and labelled, never identity-resolved. No fixture declares this
  option, so it is a provable no-op for all existing configurations
  (byte-identical equivalence gate). See
  [Individual Analysis](../user-guide/individual-analysis.md#identity-classifiers-and-non-identifying-classes)
  for usage.
- **New column:** the clean `<video>_tracks.csv` gains `identity_id`, the
  resolved catalog slot that already existed in the rich export as
  `IdentityFinalID` (`0` = unknown / non-identifying). The `identity` label
  alone is not unique -- a non-identifying label is shared by every track
  carrying it -- so group by `identity_id`, not by `identity`. Readers that
  select columns by name are unaffected; readers that assume a fixed column
  count must be updated.
- **New warning:** two classifiers marked "Unique identifier" that predict the
  same class vocabulary are now flagged at resolve time. They are multiplied
  into composites (`ant1_ant1`), not fused into one axis -- redundant identity
  voters remain unsupported -- and previously failed silently below the
  256-entry catalog-size warning.
