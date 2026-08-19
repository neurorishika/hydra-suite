# Changelog

The canonical changelog is maintained in the repository root:

- [CHANGELOG.md](https://github.com/neurorishika/hydra-suite/blob/main/CHANGELOG.md)

For documentation-specific migrations in this branch:

- Added MkDocs Material docs system.
- Reorganized docs into audience-based tracks.
- Added mkdocstrings API reference pages.
- Standardized command terminology to `posekit`.

## Identity heads, composite catalogs, non-identifying classes (unreleased)

- **Fixed (data loss):** the rich export's per-detection CNN columns
  (`CNN_<label>[_<factor>]_Class` / `_Conf`) had been silently absent since the
  Gen-2 inference migration. They were merged from a V3 `CNNIdentityCache`
  whose writer was replaced by the CNN stage's own cache; the reader was
  `os.path.exists`-guarded, so it degraded to a no-op instead of failing. Any
  run with a CNN classifier therefore produced a rich CSV with no CNN columns,
  no `IdentityEvidenceTopLabel` / `IdentityEvidenceConfidence`, and no
  `UniqueIdentityKey`. All are restored. On the identity fixture this adds 7
  columns to `<video>_tracking_final_with_individual.csv` and populates them on
  7,475 of 8,099 rows (the rest are interpolated rows with no detection);
  `IdentityEvidenceSources` and `IdentityEvidenceConflictFlag` now report the
  CNN source. Positions, `TrajectoryID`, and the `IdentityFinal*` family are
  unchanged. **Anyone who built analyses on the post-migration rich CSV was
  working without identity evidence columns and should re-export.**

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
- **Fixed:** non-identity classifiers (behavior, sex, caste, ...) now reach the
  clean `<video>_tracks.csv` as `<label>[_<factor>]_class` / `_conf`. In User
  mode the clean CSV is the *only* export written, and it carried no classifier
  output at all -- so a classifier configured purely as output ran, cost
  inference time, and was discarded at export. The design's data contract
  ("they are output, not identity, and still fully exported") had held only in
  Debug mode. Identity classifiers are deliberately excluded: their channel is
  the resolved `identity`/`identity_id`, and their per-frame calls are evidence
  that belongs in the Debug export.
- **New column:** the clean `<video>_tracks.csv` gains `identity_id`, the
  resolved catalog slot that already existed in the rich export as
  `IdentityFinalID` (`0` = unknown / non-identifying). The `identity` label
  alone is not unique -- a non-identifying label is shared by every track
  carrying it -- so group by `identity_id`, not by `identity`. Readers that
  select columns by name are unaffected; readers that assume a fixed column
  count must be updated.
