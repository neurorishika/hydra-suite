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

## Compute runtimes

Both consumers use `compute_runtime` dispatch via `ClassifierBackend`.
`_pipeline_supports_runtime` returns the same capability set for
`cnn_identity` and `head_tail`.
