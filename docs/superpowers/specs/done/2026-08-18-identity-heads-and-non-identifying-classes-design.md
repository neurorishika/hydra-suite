# Identity heads, cross-product catalogs, and non-identifying classes

**Date:** 2026-08-18
**Status:** Shipped — merged to main.

> **Post-implementation note.** Slices 1-3 shipped as designed. Verifying them
> on real video surfaced a defect this spec did not anticipate and could not
> have been satisfied without: the rich export's per-detection
> `CNN_<label>[_<factor>]_Class`/`_Conf` columns had been silently absent since
> the Gen-2 inference migration, so `IdentityEvidenceTopLabel`,
> `UniqueIdentityKey`, and this design's non-identifying-class report were all
> reading columns that no longer existed. Slice 1's scoping was correct but
> inert, and Slice 3's reporting half never fired. The export was restored in
> the same branch (see the changelog); the checkboxes in the paired plan were
> never ticked during subagent execution and should not be read as a coverage
> record — the gates are.
**Scope:** `core/individual/identity/`, `core/individual/postprocess_df.py`,
`core/post/identity_postprocess.py`, `core/tracking/worker.py` (remap only),
`trackerkit/config/identity_schema.py`, `trackerkit/gui/panels/identity_panel.py`

## Problem

Users running colortag identity report three distinct defects. They are
entangled — the third cannot be implemented correctly on top of the first two —
so this design fixes all three in one program, in dependency order.

### Defect 1 — non-identity classifiers pollute identity (bug, unreported)

A `CNN_CLASSIFIERS` entry with `unique_identifier=False` (behavior, sex, caste)
correctly contributes nothing to the identity *catalog* (`resolve.py:63`) and
nothing to *evidence fusion* (`worker.py:2721`, gated on `is_identity_provider`).
But two downstream derivations scan **every** column matching
`^CNN_(.+)_Class$` with no filter:

- `postprocess_df.py:67` → `IdentityEvidenceTopLabel`, `IdentityEvidenceConfidence`,
  `IdentityEvidenceSources`, `IdentityEvidenceConflictFlag`
- `identity_postprocess.py:152` (`derive_unique_identity_key_series`) →
  `UniqueIdentityKey`

Consequences:

- `IdentityEvidenceTopLabel` is a plain argmax-by-confidence across all heads, so
  a behavior classifier at 0.98 outranks the identity classifier at 0.80 and
  becomes the row's reported "top identity evidence".
- `UniqueIdentityKey` bundles non-identity classes into the identity key.
  `processing.py:3721` reads that column into the relink veto
  (`identity_sources_conflict`), so **an animal that changes behavior across an
  occlusion gap registers as an identity conflict and its legitimate relink is
  refused.** Relink runs on `with_pose_df` *after*
  `apply_identity_postprocessing_to_df` has written the column
  (`rich_export.py:306` then `:399`), so this path is live, not dormant.

The defect scales with the number of non-identity heads, and users are expected
to run several.

`rich_export.py:100,167` also scan all CNN class columns, but only for fill-rate
logging — enumerating every head there is correct. No change.

### Defect 2 — multiple identity classifiers union instead of cross-product

`resolve_catalog_spec` runs `itertools.product` **within** one model's factors,
then appends each identity model's resulting labels into one flat list, deduped
by display string. Two identity models (e.g. a thorax-tag model and an
abdomen-tag model) therefore yield `{red, blue, square, circle}` — four mutually
exclusive identities competing for the same Hungarian slots — instead of the
correct `{red_square, red_circle, blue_square, blue_circle}`.

A naive fix to `resolve.py` alone would be catastrophic rather than merely wrong.
Each classifier builds a **phase-local** catalog from its own factors
(`build_phase_catalog_labels`), and `worker.py:1927
_remap_source_log_probs_to_catalog` maps that phase basis into the global catalog
by **exact label string match**, dropping anything that misses
(`if not catalog.contains(label): continue`). Against a cross-product global
catalog, model A's phase label `red` matches no global label, so *all* evidence
from *all* identity models would be silently discarded and every animal would
resolve to unknown. The remap must be generalized in the same slice.

### Defect 3 — non-identifying classes are treated as unique identities (the request)

Colortag schemes include non-identifying classes: an untagged animal classifies
as `notag`, and a composite such as `notag_notag` is not an identity at all.
Today `resolve_catalog_spec` puts `notag_notag` in the catalog like any other
label, so every exclusivity mechanism applies to it: one Hungarian column, the
`blocked_labels` mask, commit-blocking in `_update_commitment`, swap detection,
and the offline `_has_collision` veto. Exactly one track can be `notag_notag` at
a time; the remaining untagged animals are pushed onto wrong real identities.

Users want untagged animals to (a) read as `notag_notag` rather than `unknown` —
which is a meaningfully different statement, "classified as untagged" vs "the
classifier could not tell" — and (b) never be constrained, merged, or swapped on
account of the shared label. They explicitly do not expect identity resolution
for them.

## Goals

1. A classifier that is not an identity provider has **zero** influence on any
   identity column, key, or decision, while remaining fully exported.
2. Multiple identity providers compose as a **cross-product** of factor axes,
   with evidence from each provider reaching the composite catalog intact.
3. Classes and composites can be declared **non-identifying**: excluded from the
   identity domain entirely, reported by name, never exclusive, never merged.

## Non-goals

- **Redundant identity voters.** Two identity models sharing a class vocabulary
  (both predicting `ant1..ant8`, meant as independent votes on one axis) are
  unsupported and will produce nonsensical `ant1_ant1` composites. Confirmed
  with the user that nobody runs this. If it is ever needed, the extension is a
  per-factor *axis name* in config, where factors sharing a name fuse instead of
  multiplying — additive, and it does not invalidate anything here.
- **Capacity-N shared labels** ("exactly 3 untagged animals"). Non-identifying
  classes have unbounded occupancy. Enforcing a count would require
  parameterizing the Hungarian solver, the blocking set, the commitment rule,
  the swap detector, and the collision veto — five correctness-critical sites —
  for a constraint that degrades results whenever the declared count is wrong.
- **Identity resolution for untagged animals.** No fragment stitching across
  occlusions by identity, no relink bonus. This is the deliberate trade that
  keeps Slice 3 small; the user accepted it explicitly.
- Realtime GUI overlay text for non-identifying tracks (they display as unknown
  live; the resolved label appears in post-processing output). Deferred.

## Design

Three slices, implemented and gated in order. Each is independently revertible.

### Slice 1 — Identity-head scoping

**Definition.** An *identity head* is a `CNN_CLASSIFIERS` entry with
`unique_identifier=True`. Its `label` field determines its column prefix:
`CNN_<label>_Class` (flat) or `CNN_<label>_<factor>_Class` (multi-factor), per
`properties/export.py:111 build_cnn_output_columns`.

**New module** `core/individual/identity/heads.py` (Core: stdlib only, no Qt, no
app-layer imports):

```python
def identity_head_labels(cnn_classifiers) -> tuple[str, ...]
def identity_class_columns(columns, head_labels) -> list[str]
```

`identity_class_columns` matches by *known head label* rather than by regex
capture — `col == f"CNN_{lbl}_Class"` or (`col.startswith(f"CNN_{lbl}_")` and
`col.endswith("_Class")`). This avoids the existing ambiguity where a head label
containing `_` is indistinguishable from a `<label>_<factor>` pair under
`^CNN_(.+)_Class$`.

**Call sites changed:**

- `postprocess_df._annotate_identity_summary_columns` — restrict
  `cnn_class_columns` to identity heads. `params` is already in scope in the
  enclosing `apply_identity_postprocessing_to_df`.
- `identity_postprocess.derive_unique_identity_key_series(df, identity_heads=None)`
  — new optional parameter. `None` preserves today's all-columns behavior (the
  documented legacy path, keeping `tests/test_unique_identity_key_derivation.py`
  meaningful); `postprocess_df.py:357` passes the resolved set.

`_fragment_unique_identity_sources` and the relink veto read the corrected
`UniqueIdentityKey` column and need no change — they inherit the fix.

**Degradation rule.** If `params` carries no `CNN_CLASSIFIERS` key at all (e.g.
re-running post-processing over a CSV without engine config), fall back to the
legacy all-columns behavior. If the key is present but no entry is marked
`unique_identifier`, the identity-head set is empty and no CNN column feeds
identity — which is the correct reading of that configuration, and matches the
fusion gate at `worker.py:2721`.

**Behavior change.** Results change only for configurations that run at least one
non-identity classifier alongside identity. Single-classifier configs — including
every equivalence fixture — are byte-identical.

### Slice 2 — Cross-product catalog across identity providers

**`resolve.py`.** Collect *axes* across all identity-providing models, in model
config order then factor order: each non-empty factor of each identity model
contributes one axis, keyed `(model_label, factor_name)`. The catalog is the
cartesian product over **all** axes; `display_label` is `"_"`-joined in axis
order, exactly as today's within-model product joins. `CatalogEntry.factors`
records qualified pairs `(f"{model_label}:{factor_name}", class_name)`.

Qualifying the factor names is safe: `CatalogEntry.factors` has no runtime math
consumer — it is written to `spec.to_dict` for provenance and read back by
`from_dict`, and the module docstring's whole point is that nothing decodes
identities by splitting the joined string. Slice 3 uses these qualified names for
axis-scoped class marking.

Single-identity-model configs are unaffected: one model's axes are its factors,
so the product, the display labels, and the ordering are what they are today.

**Catalog size.** The domain now grows multiplicatively (three models × four
classes = 64 entries), and the Hungarian cost matrix is N×(K+N). Emit a loud
warning above 256 entries naming the contributing axes. No hard cap — a warning
that names the cause is more useful than a failure.

**`worker.py:1927 _remap_source_log_probs_to_catalog`.** Replace exact-label
matching with structured distribution. Build once, from the resolved catalog
spec, a per-source lookup `phase_label -> [global_catalog_indices]`: a global
entry is reachable from a phase label iff the entry's classes on *that model's*
axes, joined in axis order, equal the phase label. Then, per detection: set each
reachable global index to the phase probability, leave unreachable entries at the
existing `1e-300` floor, renormalize, take logs.

This is the same semantics `substrate._factor_log_prob` already uses for the
composite branch (assign, then normalize — not divide-among), lifted one level up
from within-model factors to across-model phases.

**Single-model byte-identity.** With one identity model the phase labels equal
the global labels, so every phase label reaches exactly one global index and
"assign then renormalize over a 1e-300 floor" is arithmetically identical to
today's `remapped[idx] += prob` over the same floor. A unit test asserts equality
against the current implementation's output on a single-model catalog.

**Risk note.** This is the slice that can silently zero out all identity
evidence. It gets a dedicated test asserting the fused posterior is
*non-degenerate* (argmax lands on the correct composite, not on unknown) for a
two-model catalog — a failure mode that would otherwise look like "identity just
stopped working" with no error anywhere.

### Slice 3 — Non-identifying classes

**Config.** Each identity classifier entry gains
`non_identifying_classes: list[str]`. Each item is one of:

| Form | Matches |
|---|---|
| `notag` | that class in any axis of that model |
| `tag1:notag` | that class in the named factor of that model |
| `notag_notag` | that whole global composite display label |

The mixed forms are deliberate: the user's cases are case-by-case. `notag` and
`tag1:notag` are sugar that expands into composite exclusions at resolve time, so
**the engine only ever knows per-entry exclusion** — there is no second code path.

Persisted on `IdentityModelConfig` (`trackerkit/config/identity_schema.py`) as
`non_identifying_classes: tuple[str, ...] = ()`.

*As built (correction):* that dataclass field is **not** the live path, and
`build_engine_params` threads nothing of its own. `IdentityConfig.from_engine_config`
never populates `models`, so nothing reads `IdentityModelConfig.non_identifying_classes`
at runtime. The marks travel with each classifier entry instead:
`identity_panel.CNNClassifierRow.to_config` writes them into the saved config's
`cnn_classifiers` list, and `build_engine_params` passes that list through verbatim
into `CNN_CLASSIFIERS`, where `resolve.non_identifying_marks` reads them. The
dataclass field is kept for serialization round-tripping only; a `models`-populating
loader would have to be written for it to become live, and this program deliberately
did not write one.

**`resolve.py`.** After the cross-product, drop every entry that matches any
declared form. The catalog shrinks; indices stay contiguous.

**This is the entire engine-side change.** The Hungarian solver, `blocked_labels`,
`_update_commitment`, `_detect_and_execute_swaps`, and the offline
`_has_collision` veto are *untouched* — not conditionalized, not parameterized.
Excluded labels are simply not in the domain, so no exclusivity can be applied to
them, any number of untagged tracks coexist, and none are ever merged or swapped
onto each other. The evidence path already no-ops on out-of-catalog classes
(`worker.py:2727`, `if _cn and _cat.contains(_cn)`), so nothing else is needed to
make the evidence side safe.

**Empty-catalog guard.** If every entry is excluded, log a loud warning and leave
the spec empty; downstream already handles "catalog_spec had no entries" (the
`IdentityCatalog.from_labels` empty-list raise must not be reached).

**Reporting.** New helper in `postprocess_df`, run after the fragment solver and
before `_mirror_realtime_and_tag_into_final`. For each trajectory with no
resolved `IdentityFinalLabel`, compose the identity heads' per-frame classes in
catalog axis order, take the modal composite across the trajectory, and if that
composite is a declared non-identifying one, stamp:

| Column | Value |
|---|---|
| `IdentityFinalLabel` | the composite, e.g. `notag_notag` |
| `IdentityFinalID` | `0` (the unknown slot) |
| `IdentityFinalSource` | `"nonidentifying"` (new `IdentityFinalSource` constant) |
| `IdentityFinalConfidence` | mean over frames of the min confidence across axes |

The label is descriptive; **the ID stays the unknown slot**. That is the load-bearing
invariant: any consumer keying on a resolved identity slot uses `IdentityFinalID`
and can never mistake a shared label for a real identity. The composite is built
from identity heads only (Slice 1's set), never from `IdentityEvidenceTopLabel`,
whose argmax could otherwise be hijacked by a non-identity head.

Confidence uses the min across axes so a composite is only as trustworthy as its
weakest tag call.

**Relink guard.** Non-identifying class values are dropped from the
identity-sources dict that feeds `identity_sources_conflict`. Otherwise
`notag == notag` counts as *agreement* in `_compare_identity_sources`'s grouped
tally and can out-vote a genuine conflict on another axis. Agreement alone never
creates a link — the spatial and velocity gates still apply — so this is a
correctness guard, not a behavior expansion.

**GUI.** `identity_panel.py`: for a model marked "Unique identifier", show its
classes grouped by factor with a "non-identifying" checkbox each, plus a
free-text row for whole composites. Persisted through `IdentityModelConfig`.

## Data contract

- No new CSV columns. One new *value* in the existing `IdentityFinalSource`
  vocabulary: `"nonidentifying"`.
- `UniqueIdentityKey` narrows: it now carries identity heads only, minus
  non-identifying values. This is the fix, and it is a visible format change for
  anyone parsing that column downstream — call it out in the changelog.
- Non-identity classifiers' `CNN_<label>_Class` / `_Conf` columns are unchanged
  and still fully exported. They are output, not identity.
- The `IdentityFinalID == 0` pin protects consumers of the **rich** export
  (`<video>_tracking_final_with_individual.csv`), which is the only export
  carrying `IdentityFinalID`. `<video>_tracks.csv` carries `identity`,
  `identity_confidence`, and `identity_source` only, so readers of that file
  distinguish several untagged animals sharing one descriptive label by
  `identity_source == nonidentifying` alone. The CSV schema is deliberately
  unchanged.

## Testing and gates

Per-slice unit tests, plus the standard equivalence matrix on MPS (this box) and
CUDA (mehek) after each slice, run against the same baseline so each slice's
effect is attributable.

**Slice 1**
- With a behavior head present: `IdentityEvidenceTopLabel` and `UniqueIdentityKey`
  ignore it; a behavior change across a gap no longer vetoes relink.
- Absent `CNN_CLASSIFIERS` → legacy all-columns fallback.
- Present but no identity head → no CNN column feeds identity.
- Equivalence: byte-identical (fixtures have at most one identity classifier).

**Slice 2**
- Two identity models → catalog is the 4-entry product in the specified order.
- Remap on a single-model catalog equals the current implementation exactly.
- Remap on a two-model catalog is non-degenerate: fused posterior's argmax is the
  correct composite.
- Catalog-size warning fires above 256 entries.
- Equivalence: byte-identical.

**Slice 3**
- Marked classes are absent from the catalog; the surviving domain is correct for
  each of the three marking forms.
- N concurrent untagged tracks: all N keep distinct `TrajectoryID`s, none are
  merged, none are pushed onto real identities, all stamp
  `IdentityFinalLabel == "notag_notag"` with `IdentityFinalID == 0`.
- All-excluded catalog degrades with a warning, no raise.
- `notag == notag` does not count as relink agreement.
- Equivalence with `non_identifying_classes = []`: byte-identical (the feature is
  opt-in, so "off" is a provable no-op).

## Known limitations

- **The Hungarian Bayesian-cost CNN term ignores composite catalogs.**
  `worker.py`'s per-frame identity-cost term exact-matches `class_names[0]`
  against the catalog, so it contributes nothing whenever the catalog entry
  is a composite rather than a bare class name. This is pre-existing (it was
  already broken for a single multi-factor model), but Slice 2 makes
  composite catalogs reachable from two *single-factor* models too, widening
  the set of configurations in which the term silently contributes nothing.
  Not fixed here: the term is an association hint, the identity decision
  itself comes from the evidence path (which does remap correctly), and
  changing it would perturb assignment on every existing config.
- **`IdentityFinalID` is not exported to `<video>_tracks.csv`.** See the
  Data contract section.

## Risks

| Risk | Mitigation |
|---|---|
| Slice 2's remap silently zeroes all identity evidence | Dedicated non-degeneracy test; single-model equality test against current output |
| Existing multi-identity-model users get different results | Intentional — their current results are wrong. Changelog entry naming the union→cross-product change |
| `UniqueIdentityKey` narrowing breaks downstream parsers | Changelog; the column is derived, not authored, so it can be rebuilt |
| Catalog explosion with many axes | Warning above 256 entries naming the contributing axes |
| Untagged animals lose identity-based fragment stitching | Accepted trade, stated in Non-goals; they never had correct stitching anyway |

## Changelog

No root `CHANGELOG.md` exists in this repository; these notes record the
breaking changes shipped by this program (Tasks 1-11) for whoever writes the
release notes.

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
  `IdentityFinalID == 0` and `IdentityFinalSource == "nonidentifying"` — they
  are tracked and labelled, never identity-resolved. No fixture declares this
  option, so it is a provable no-op for all existing configurations
  (byte-identical equivalence gate).
