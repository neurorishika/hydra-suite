# Classifier compatibility

This page documents the contract between ClassKit-trained classifier
artifacts and TrackerKit's head-tail and CNN-identity consumers.

## Artifacts

All ClassKit-trained classifier checkpoints carry `schema_version: 2` and
these fields:

| Field | Type | Flat | Multi-head |
|---|---|---|---|
| `schema_version` | int (== 2) | yes | yes |
| `arch` | str | yes | yes |
| `input_size` | `[H, W]` (2-int sequence) | yes | yes |
| `factor_names` | list[str] | `["flat"]` | length K |
| `class_names_per_factor` | list[list[str]] | `[[...]]` | length K |
| `class_names` | list[str] | yes (== `class_names_per_factor[0]`) | absent |
| `monochrome` | bool | yes | yes |

Multi-head YOLO bundles are instead described by a sidecar
`*.multihead.json` manifest; see the spec.

### `multihead_custom_shared` mode

Trains a single torchvision backbone with N parallel MLP heads (one per
factor) in a shared-trunk multi-head classifier. Produces a single `.pth`
artifact whose v2 schema carries `factor_names` + `class_names_per_factor`
+ `head_kind="multihead_shared_trunk"`. Unlike `multihead_custom`, no
`.multihead.json` sidecar is emitted — the checkpoint is self-describing.
Available on multi-factor schemes (e.g. 2- or 3-factor color tags);
single-factor schemes still use `flat_custom`.

## Registry

The registry is stored at `{models_root}/model_registry.json` in the v2
root shape:

```json
{
  "schema_version": 2,
  "entries": {
    "classification/identity/....pth": { "schema_version": 2, ... }
  }
}
```

Use `model_publish.iter_registry_entries` to read; never parse the JSON
directly from UI code.

## Consumers

- Head-tail accepts flat classifiers whose labels normalize to a non-empty
  subset of `{up, down, left, right, unknown}`. Any backbone.
- CNN identity accepts flat and multi-head artifacts. Multi-head imports
  require a `scoring_mode` (`"atomic"` or `"per_head_average"`).

## Fit policy

A classifier artifact records the **Layer-2 fit policy** it was trained
under: how a canonical crop of arbitrary aspect ratio is mapped onto the
model's square input. `ClassifierMetadata.fit_policy` is one of:

| Policy | Meaning |
|---|---|
| `letterbox` | Aspect-preserving resize onto a zero-padded canvas. What `training/canonical_transform.py` (`CanonicalFitTransform`) produces, and what every model published since 2026-08-05 is trained with. |
| `squash` | Legacy anisotropic `Resize((sz, sz))` — the crop is stretched to fill the square. What training produced *before* commit `3a2163ac` (2026-08-05). |
| `native` | The backend applies its own transform (ultralytics YOLO); crops are handed over untouched. |

Every classifier consumer goes through one shared Layer-2 function,
`core.canonicalization.fit.fit_crops_for_policy(crops, model_hw, policy)`.
Nothing preprocesses crops for a classifier by hand.

**The legacy rule.** An artifact with no `fit_policy` key predates
stamping, so `resolve_fit_policy` resolves it to `squash` and logs a
warning naming the file. This is not cosmetic: feeding a `squash`-trained
model letterboxed crops leaves a large fraction of the input black, and on
the April 2026 colour-tag model that collapsed joint max-probability from
0.946 to 0.141 — evidence indistinguishable from noise. Stamp such an
artifact explicitly (which also silences the warning):

```bash
python scripts/stamp_fit_policy.py <model.pth | bundle.multihead.json> \
    --policy letterbox|squash        # --dry-run to preview; writes a .bak sibling
```

An unrecognized value is a hard `ClassifierFormatError`, never a silent
fallback.

## Compute runtimes

Both consumers receive a `ResolvedBackend` (from
`runtime/resolver.py`) via `ClassifierBackend`; the same runtime tier
resolves identically for the `cnn_identity` and `head_tail` stages. See
[Runtime integration](runtime-integration.md).
