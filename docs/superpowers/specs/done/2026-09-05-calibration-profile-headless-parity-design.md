# DetectKit calibration profiles: headless parity and sidecar durability

**Status:** Shipped — merged to main
**Date:** 2026-09-05
**Audit this argues from:** `docs/superpowers/specs/2026-09-05-calibration-profile-headless-parity-audit.md`
(read-only Fable audit of `main` @ b8c424e7; every claim below carries its file:line there)

## Problem

DetectKit's direct-calibration wizard produces named operating profiles stored in
`<model>.pt.slice_meta.json` (v2). TrackerKit's **GUI** consumes them completely:
every profile appears in a combo, the explicit primary auto-applies on model
selection, all twelve profile-owned knobs reach `InferenceConfig`, user edits flip
to "Custom" without being clobbered, and the saved session records both the profile
id and an effective-settings snapshot.

Three things are missing.

1. **The headless path ignores profiles entirely.** `build_engine_params`
   (`trackerkit/engine_params.py:919-935`) sources overlap, tile fraction, tile
   width/height, `trained_body_px` and all four merge keys **only** from
   `advanced_config`. The GUI never persists `advanced_config.json`
   (`_save_advanced_config`, `gui/orchestrators/config.py:2556`, has no live
   caller), and the CLI never passes one (`cli_config.py:198-216`). The builder
   contains no reference to `slice_profile_id` or `slice_profile_settings`,
   although the per-video config written by the GUI carries both
   (`gui/orchestrators/config.py:1699-1702`). A config saved from a calibrated
   GUI session, re-run through `trackerkit track`, silently tiles and merges with
   defaults. This violates the project's stated GUI/CLI parity convention, and it
   is exactly the class of divergence the shared param-builder work existed to
   eliminate.
2. **The sidecar gets lost at two hand-off points.** `publish_trained_model` writes
   the sidecar only when `slice_geometry` is truthy
   (`training/model_publish.py:872-878`), and the registry `slice_profiles` stamp
   sits inside the same gate (`:930-935`), so publishing a calibrated but
   non-sliced artifact drops its profiles. TrackerKit's "Add model" import copies
   only the weights (`gui/orchestrators/config.py:3688`), so a `.pt` + sidecar
   pair shared between machines arrives without profiles or canonical geometry.
3. **One GUI honesty hole.** `spin_yolo_confidence` has no connection to
   `_mark_slice_profile_custom` (`gui/panels/detection_panel.py:1057-1081`), so
   editing the threshold leaves the panel and the saved config claiming a profile
   the run did not use.

## Goals

- `build_engine_params` resolves calibration profiles with **the same ladder the
  GUI uses**, so GUI and CLI produce identical params for the same config.
- A batch user can pick a profile without hand-editing JSON.
- A profile, once calibrated, survives publish and import.
- The run records which operating point it actually used.

## Non-goals (deferred, with reasons)

- **`settings.runtime_tier` as a profile field.** The wizard's design is that the
  user calibrates on the tier they track with; a profile that carries a tier can
  apply settings whose measured s/frame no longer describe the run. This is a
  spec change to the calibration format, not a wiring gap. Separate proposal.
- **Live sidecar reload, profile counts in the model combo, sequential-mode
  stage-1 profiles.** Polish, not correctness. Separate plan.
- **Making the CLI honour the GUI's *other* advanced-config keys** (`obb_seg_*`,
  etc.). Out of scope; they remain machine-global on the CLI and this must be
  documented as a known residual so nobody misreads the fix as broader.

## Design

### D1 — The resolution ladder is one ladder, defined once

The GUI's ladder, as implemented in `_apply_slice_meta_values`
(`detection_panel.py:2642-2765`) and `apply_slice_meta_for_model` (`:2765-2823`):

| Saved `slice_profile_id` | Snapshot present | Applied |
|---|---|---|
| `"__training__"` | either | training geometry (`slice_meta_to_panel_values(meta, "__training__")`) |
| `"__custom__"` | yes | the snapshot (`slice_meta_values_from_settings`) |
| `"__custom__"` | no | primary, else training |
| a live profile id | either | that profile |
| an id absent from this sidecar | yes | the snapshot |
| an id absent from this sidecar | no | primary, else training |
| `""` / missing | either | primary, else training |

Two facts a naive implementation gets wrong, both load-bearing:

- `profile_by_id` special-cases only `"__training__"`. A raw `"__custom__"` id
  falls through to the **primary**, which is why the GUI checks for it explicitly
  (`detection_panel.py:2671-2678`). The builder must make the same check.
- **Amended after adversarial review.** A resolved profile does **not** override
  `slice_enabled`, `slice_geometry_mode` or `yolo_confidence_threshold`. Those
  three are persisted by `build_config_dict` (`gui/orchestrators/config.py:1697,
  1698, 1730`) and already hold whatever the profile put in the widgets when the
  session was saved. The GUI's restore order confirms it for confidence: the
  profile is applied via model selection (`:530-533`) and the spin is then set
  from the config (`:614-616`) — the config wins. The overlay therefore supplies
  only the nine keys that live solely in `advanced_config` (`overlap`,
  `object_tile_fraction`, `slice_width`, `slice_height`, `trained_body_px`, and
  the four `merge_*`), and **warns** when the config's confidence disagrees with
  the claimed profile's measured value. This also makes the overlay a no-op
  inside the GUI, which matters because `get_parameters_dict()` is itself a
  `build_engine_params` call (`:2153-2160`).
- The **training-geometry rung applies**: `_training_values`
  (`slice_meta.py:287-320`) returns real `overlap` / `object_tile_fraction` /
  `trained_body_px`, and that is the sidecar every sliced-training publish
  writes. Its `enabled=True` is *not* propagated — `slice_enabled` comes from
  the config, per the rule above.

Unclaimed merge keys (`None` from either translator) mean **the default**, not
"whatever was there" — `SLICE_MERGE_DEFAULTS`, matching `detection_panel.py:2723-2732`.
Confidence is the exception: it is the user's global YOLO threshold, applied only
when a profile explicitly claims one (`:2739-2743`).

The overlay must degrade to a no-op on a missing/corrupt sidecar and must never
raise: `build_engine_params` is called in tests with paths that do not exist.

### D2 — `--sahi-profile` selects by name or id

`trackerkit track --sahi-profile <name|id>` rewrites the resolved config's
`slice_profile_id` before the builder runs, and applies to the keystone config in
a batch. Semantics, stated because they differ deliberately:

- **explicit flag, profile not found in the model's sidecar → hard error.** The
  user named an operating point; running a different one silently is the bug this
  whole design exists to remove.
- **config's saved id not found → ladder fallback + a warning log.** An old config
  must keep running.

Name matching is exact and case-sensitive against `available_slice_profiles`;
ambiguous names resolve to ids only.

### D3 — Sidecars travel with the model, always

One shared helper (`copy_model_metadata_sidecars`, promoted out of
`detectkit/gui/project.py:193-207` to a location both kits may import) copies
`.slice_meta.json`, `.canonical_meta.json`, `.runtime_meta.json` and
`.v2meta.json`. Used by TrackerKit's import path and by publish's
non-`slice_geometry` branch, which must also stamp `slice_profiles` in the
registry so the inventory does not disagree with the sidecar.

### D4 — Provenance

Headless session start logs the resolved profile name, its id, and the resolution
rung (`requested` / `primary` / `saved_settings` / `training`), so a batch log
answers "which operating point ran?" without re-deriving it. No new params keys:
a key nobody consumes is noise and would force a golden regeneration.

## Verification contract

- **Equivalence must stay byte-identical on MPS and CUDA.** Verified inert by
  construction: no fixture config in `tools/equivalence/fixtures/configs/` contains
  any `slice_profile*` key (grep: 0 hits in all ten), and no fixture model carries
  a `.slice_meta.json`. The overlay therefore cannot fire on the matrix. If a
  future fixture gains either, this reasoning expires.
- **The characterization golden must grow a non-default SAHI case.** The audit's
  root observation is that `tests/data/get_parameters_dict_golden/` covers only
  default SAHI, which is why F1 survived the GUI/CLI unification gate. A golden
  that exercises a profile overlay is the durable guard, not the fix itself.
- `tests/test_profile_cache_keys.py` semantics must hold unchanged: geometry,
  overlap and merge settings split the detection cache; confidence does not.
