# Active-Learning Escalated Multi-Format Export — Design

**Date:** 2026-08-17
**Status:** Design approved, pending implementation plan
**Scope:** TrackerKit + DetectKit active-learning dataset generation

## Problem

TrackerKit's active-learning (AL) dataset export writes YOLO OBB labels and nothing
else. When tracking runs a segmentation model, the mask contours the model produced
are discarded and the export degrades to oriented boxes. Downstream, DetectKit and
the training layer already understand a geometry-level ladder
(`polygon > obb > aabb`), so the information loss is gratuitous.

Auditing the export path to add escalation surfaced a second problem: TrackerKit's AL
implementation has diverged from DetectKit's, and the TrackerKit side carries
correctness bugs that corrupt both frame selection and label geometry.

This design covers both: escalated multi-format export, and convergence of the two AL
implementations onto one shared core.

## Current state

Two AL implementations exist and have diverged.

| | TrackerKit path | DetectKit path |
|---|---|---|
| Entry | `core/tracking/session.py:374 _run_dataset_generation`, `trackerkit/gui/workers/dataset_worker.py` | `detectkit/jobs/al_worker.py:run_active_learning` |
| Scoring | `data/dataset_generation.py:FrameQualityScorer` | `data/al/` directly |
| Candidate pool | none | `build_candidate_pool` (pHash dedup) |
| Label writer | `_format_obb_corners` / `_compute_obb_corners`, OBB only, class hardcoded `0` | `_write_geometry_label`, polygon-first |
| Output contract | loose folder + `metadata.json` | registered `OBBSource(level=…)` |

The shared core (`data/al/{signals,acquisition,candidate_pool,frame_source}.py`) is
sound. The problems are in the TrackerKit adapter, the exporter, and the
half-wired level plumbing on both sides.

### Existing machinery this design reuses

The escalation ladder is already built and has exactly one consumer:

- `OBBConfig.emit_native_geometry` (`core/inference/config.py:203`) — export-only opt-in.
- `OBBResult.polygons` (`core/inference/result.py:35`) — native contours in frame
  pixel space, populated by the segment extractor from `masks.xy`
  (`core/inference/stages/obb.py:1039`). Never serialized to the `.npz` cache.
- `GeometryLevel` + `scan_source_levels` (`training/geometry_levels.py`).
- `derive_detect_dataset_from_obb` (`training/dataset_builders.py:406`).
- `_write_geometry_label` (`detectkit/jobs/al_worker.py:118`) — polygon-first writer.

TrackerKit's exporter opts into none of it.

### Audit findings

Correctness bugs:

1. **`edge_score` is corrupt on the tracker path.** `dataset_generation.py:105` calls
   `score_crowd(obb_corners, frame_shape=(1, 1))` with pixel-space corners. Inside,
   `dx = min(x, w - x)` with `w=1` goes strongly negative, so
   `edge_norm = max(0, 1 - margin_px / 1.0)` returns values around 1900 instead of
   `[0, 1]`. After `_minmax` it survives as a channel ranking frames by how far
   right/down any detection sits.
2. **Unconstrained detection matching.** `_match_yolo_detection` uses a hardcoded
   50 px radius, independent of `REFERENCE_BODY_SIZE` and `RESIZE_FACTOR`, and is
   greedy with no mutual exclusion — two CSV rows can bind the same detection and
   emit duplicate boxes.
3. **Fabricated geometry is indistinguishable from real geometry.** On no match,
   `_write_frame_annotations` invents `w = ref * 2.2, h = ref * 0.8`. The
   `dimension_source` flag reaches `metadata.json` only; the `.txt` label looks real.
4. **Lost and interpolated tracks are exported as ground truth.** The only row filter
   is `pd.isna(X/Y)`. Since AL selects frames *because* tracking struggled, this
   systematically injects wrong boxes where the model is weakest.
5. **Non-OBB detection methods export a fully synthetic dataset.**
   `_init_detection_runner` returns `None` unless `DETECTION_METHOD == "yolo_obb"`,
   so bgsub tracking exports 100% reference-size boxes with no warning.
6. **Multi-class collapse.** Both writers hardcode class `0` while
   `OBBResult.class_ids` exists and DetectKit projects are multi-class.
7. **Segmentation models cannot reach the exporter.** `build_obb_only_config`
   (`config.py:1054`) never forwards `YOLO_OBB_DIRECT_TASK`, so the export runner is
   always built as `model_task="obb"`. A segmentation checkpoint hits the loud
   task-mismatch check at `stages/obb.py:358`.

Orphaned and regressed signals — the refactor that introduced `data/al/` left the
legacy scorer in place and wired only part of it:

8. **The fragmentation signal is computed and discarded.**
   `_score_fragmented_detections` detects one animal split into two detections
   (proximity + polygon overlap + both-boxes-suspiciously-small), weighted 0.3 in
   legacy. It now feeds only the dead scalar score. Its nominal replacement, the
   `crowd` channel, is max pairwise polygon overlap — animals *touching*, a different
   phenomenon. The `METRIC_FRAGMENTED_DETECTIONS` checkbox gates the `crowd` weight
   (`dataset_generation.py:58`), so the control does not do what its label says.
9. **Absolute thresholds became relative.** Legacy metrics returned 0 unless the frame
   was actually bad. `_composite_score` min-max normalizes every channel within the
   run, so a cleanly tracked video still yields a full ranking and exports its
   least-good frames. `DATASET_MIN_SELECTION_SCORE` therefore cannot work as its
   tooltip advertises, and scores are not comparable across videos.
10. **Count asymmetry lost.** Legacy weighted under-counting (0.3) twice as hard as
    over-counting (0.15). `score_count_deviation` is symmetric.
11. **`margin` is dead.** Computed and stored; `AcquisitionWeights` has no margin
    channel. Its only input, `DATASET_CONF_THRESHOLD`, is hardcoded to `0.5` at
    `engine_params.py:1214` and is not user-settable.
12. **bgsub confidences are all `NaN`** (`core/background/measure.py:216`), and
    `_channel_array` maps NaN to 0. On bgsub the `uncertainty` channel (weight 0.30)
    is silently dead and the remaining channels are diluted rather than renormalized.
13. **`ALRequest.export_level` is never set** by DetectKit's AL dialog; it is
    permanently `"obb"`.

Design smells:

14. **~220 lines of dead-weight computation per frame.** `score_frame` runs both the
    `ALSignals` pipeline and the entire legacy scalar pipeline, including the O(n²)
    `_score_fragmented_detections` loop. `get_worst_frames` reads only
    `frame_signals`. Eight of the sixteen tests in `tests/test_dataset_generation.py`
    pin the dead path; the live path is barely covered.
15. **Double inference with divergent thresholds.** The scorer reads cached OBBs; the
    exporter spins up a second `InferenceRunner` at conf 0.05 / iou 0.5 and
    re-detects. Defensible ("detect everything for annotation") but undocumented in
    the output, and it means labels can disagree with what tracking saw.
16. **`include_context` fights the diversity window.** ±1 frames triples the set with
    unscored near-duplicates; no perceptual dedup on this path. Context frames outside
    the tracked range silently produce empty label files.
17. **O(n²) pandas and per-frame seeking.** `df[df["FrameID"] == frame_id]` inside
    both the scoring and writing loops is a full scan per frame;
    `cap.set(POS_FRAMES)` seeks per frame. `valid_batch_indices` is threaded into
    `_detect_batch` and never used.
18. **No DetectKit handshake.** Output records no level, registers no `OBBSource`, and
    stamps no model/threshold/preset provenance, so a round is not reproducible. The
    generated `README.md` still instructs users to open X-AnyLabeling.

## Design

### 1. Output layout

One AL round produces up to three sibling **source roots**, each independently a
valid DetectKit source:

```
<video>_datasets/active_learning/al_20260817_141230/
  polygon/  images/  labels/  classes.txt  source.json    level=polygon, authoritative
  obb/      images/  labels/  classes.txt  source.json    level=obb,  derived_from=polygon
  aabb/     images/  labels/  classes.txt  source.json    level=aabb, derived_from=polygon
```

Images in derived roots are **hardlinks** to the authoritative root's images, so disk
cost stays at roughly 1× regardless of level count.

Rationale for sibling roots rather than sibling `labels_*/` directories inside one
root: DetectKit hardcodes `<source_root>/labels` at roughly ten call sites
(`detectkit/gui/utils.py`, `gui/source_import.py`, `gui/dialogs/source_validation.py`,
`gui/panels/dataset_panel.py`, `jobs/sam2_escalation.py`). Sibling label directories
would be invisible to all of them. Sibling roots need zero DetectKit changes.

Each root is registered as an `OBBSource` with its `level`,
`source_kind="trackerkit_al"`, and — for derived roots — `derived_from` and
`reviewed=False`, so the authoritative root stays the single point of human review. A
"regenerate derived levels" action re-derives the two lower roots from the
authoritative one, containing drift.

### 2. Level honesty

The export never claims a level the model did not produce.

| Detection source | Native geometry | Roots written |
|---|---|---|
| YOLO segment (direct or sequential stage-2) | mask contours via `masks.xy` | polygon, obb, aabb |
| YOLO OBB | rotated quad | obb, aabb |
| YOLO detect | axis-aligned box | aabb |
| bgsub | foreground contours | polygon, obb, aabb |

A rotated quad is **not** written as a polygon-level root: it carries no contour
information, and `scan_source_levels` already treats 9-field lines as ambiguous
`four_point`. Downward derivation (polygon → `minAreaRect` → obb → aabb) is lossless
to its target; upward derivation is out of scope.

### 3. Shared modules

New Qt-free code under `data/al/`, consumed by both AL paths:

- **`data/al/escalation.py`** — the single geometry-escalation authority. A
  `LabelRecord` per detection (`class_id`, `conf`, `polygon | None`, `corners`,
  `native_level`), `records_from_obb_result(obb, level)`, and
  `derive_down(records, target_level)`. Level conversion is written exactly once.
- **`data/al/labels.py`** — `write_label_file(path, records, frame_size, level)`.
  This is `_write_geometry_label` moved down out of `detectkit/jobs/al_worker.py`;
  DetectKit then imports it (app → data, correct direction).
- **`data/al/export.py`** — `export_al_dataset(...)`: writes the three-root layout,
  hardlinks images, stamps `source.json`, returns a manifest. Both
  `data/dataset_generation.py:export_dataset` and
  `detectkit/jobs/al_worker.py:run_active_learning` collapse onto it.

`GeometryLevel` and `classify_label_line` move from `training/geometry_levels.py` to
`utils/geometry_levels.py` (bottom layer, importable by Data and Training alike), and
are re-exported from the old path for back-compat. `scan_source_levels` stays in
Training — it is ingestion logic.

### 4. Acquisition scoring: port before delete

The legacy scalar pipeline is not pure dead weight. Three behaviours are ported into
`data/al/` as first-class channels **before** the legacy code is deleted.

**4a. Absolute floors replace min-max normalization.** `_minmax` is removed from
`_composite_score`. Every `score_*` function returns an absolute severity in `[0, 1]`
that is exactly `0` when the frame is not problematic:

| Channel | Absolute definition |
|---|---|
| `uncertainty` | `0` if `mean_conf >= conf_floor`, else `(conf_floor - mean_conf) / conf_floor` |
| `count` | asymmetric — under: `(expected - n) / expected`; over: `min((n - expected) / expected, 1) * 0.5` |
| `fragmentation` | ported heuristic, already absolute with its 0.45 suspicion gate |
| `crowd` | max pairwise polygon overlap ratio |
| `edge` | border proximity, computed against the **real** frame shape |
| `assignment` | `1 - mean(assignment_confidence)`, or `min(mean(cost) / 50, 1)` |
| `track_loss` | `min(lost / max_targets, 1)` |
| `position_uncertainty` | `min(mean(uncertainty) / 50, 1)` |
| `nms_instability` | `1 - mean(set IoU)` under threshold perturbation (DetectKit path) |

The weighted composite is then in `[0, 1]` and comparable across videos, making
`DATASET_MIN_SELECTION_SCORE` a real gate. A cleanly tracked video legitimately
exports few or no frames; when the selection comes back empty, the UI reports the
per-channel maxima observed so the user can see *why* rather than seeing a bare error.

This changes selection on the DetectKit path too (its `balanced` and
`uncertainty_heavy` presets score absolutely as well). That is intended: one scoring
vocabulary across both kits.

**4b. `fragmentation` becomes a real channel.** The `_score_fragmented_detections`
heuristic moves into `data/al/signals.py` as `score_fragmentation`, joins `ALSignals`,
and gains its own weight in `AcquisitionWeights` (0.30 in `tracker_default`, mirroring
its legacy weight). `METRIC_FRAGMENTED_DETECTIONS` is repointed at it, so the checkbox
finally controls what its label claims, and `crowd` gets its own weight and control.

**4c. `count` becomes asymmetric**, as in 4a.

Only after these three land is the legacy pipeline (`_score_confidence` through
`_score_fragmented_detections`, plus `frame_scores`) deleted and the eight tests that
pin it repointed at `frame_signals` / `select`. `score_frame` and `get_worst_frames`
keep their signatures.

Also resolved here: `score_crowd` receives the real frame shape (finding 1);
`DATASET_CONF_THRESHOLD` becomes the user-settable `uncertainty` floor (surfaced in
the panel) and the unused `margin` field is removed from `ALSignals` and
`score_uncertainty`, which then returns `mean_confidence` alone (finding 11); on bgsub the `uncertainty` weight is
zeroed **explicitly and the remainder renormalized**, rather than silently diluted
(finding 12).

### 5. Strict labels and accounting

- A CSV row is exported only if it binds to a real detection. Dropped:
  `State == "lost"`, interpolated rows, and unmatched rows. Fabricated
  `ref * 2.2 × ref * 0.8` boxes are gone.
- Nearest-center-within-50 px is replaced by mutual-exclusion assignment
  (`core/assigners/hungarian.py`) gated by a radius scaled to `REFERENCE_BODY_SIZE`,
  fixing both duplicate binding and wrong-scale matching.
- Real `class_id`s from `OBBResult.class_ids`; `classes.txt` from the project's
  `class_names`.
- `source.json` records per round and per frame: rows total, exported, dropped-lost,
  dropped-unmatched; plus model path, model task, thresholds, acquisition preset and
  weights, native level, and which frame ids were *selected* versus *context*. An AL
  round becomes reproducible and auditable.

### 6. Inference plumbing

Small, additive, all opt-in so the tracking hot path stays byte-identical:

1. `build_obb_only_config` forwards `model_task` and `emit_native_geometry`
   (fixes finding 7).
2. `_init_detection_runner` stops gating on `DETECTION_METHOD == "yolo_obb"`, reads
   the real task, and builds a bgsub export runner instead of returning `None`
   (fixes finding 5).
3. `core/background/measure.py:detect_objects` gains an opt-in contour return,
   threaded through the same size-filter and cap subselections; `run_bgsub` populates
   `OBBResult.polygons` under `emit_native_geometry`. The contours are already
   computed at `measure.py:196` and discarded, so this adds no work to the hot path.

### 7. Candidate-pool dedup on the tracker path

`build_candidate_pool` runs over the **selected** frames plus their context, not the
whole video. pHash over 100k frames is prohibitive; over a few hundred it is free, and
that is exactly where the `include_context` near-duplicate problem lives (finding 16).

### 8. UI

**TrackerKit `DatasetPanel`:**

1. **Export level status — read-only, derived** from the detection config and updated
   live: segmentation → "polygon + obb + aabb"; OBB → "obb + aabb — polygon requires a
   segmentation model"; detect → "aabb only"; bgsub → "polygon + obb + aabb (from
   foreground contours)". The level-honesty rule becomes visible before the run.
2. **Level checkboxes** — which achievable roots to materialize; all on by default,
   greyed where unachievable.
3. **Class names** — `line_dataset_class_name` becomes an ordered comma-separated list
   (index = class id). Single-class users type what they type today.
4. **DetectKit target** — optional "Register into DetectKit project" with browse.
   Empty means standalone output plus `source.json`, which DetectKit's importer adopts
   with levels already known.
5. **Dedup controls** — method (`phash`/`ahash`/`dhash`/`histogram`/`none`) and
   threshold, mirroring `CandidatePoolConfig`, beside the diversity window.
6. **Quality-metric controls corrected** — separate `fragmentation` and `crowd`
   checkboxes (per 4b); `METRIC_HIGH_UNCERTAINTY` default reconsidered.
7. **Min-selection-score tooltip rewritten** — under absolute scoring it is now a real
   cross-video gate, and the tooltip should say so.
8. **Results summary** — `finished_signal` widens from `Signal(str, int)` to carry the
   manifest, so the panel reports the three roots and their levels, frames exported,
   and the strict-drop counts.
9. **bgsub notice** — inline note that the uncertainty channel is unavailable
   (all-NaN confidences) and remaining weights were renormalized.
10. **Stale text** — panel help and the generated `README.md` retarget from
    X-AnyLabeling to DetectKit.

**DetectKit AL dialog** gets the same level status and checkboxes, wiring the dead
`export_level` field (finding 13), so both kits present one control vocabulary.

**Plumbing:** new knobs (`dataset_export_levels`, `dataset_dedup_method`,
`dataset_dedup_threshold`, `dataset_class_names`, `dataset_detectkit_project`) live in
`trackerkit/config/schemas.py` and flow through the shared `build_engine_params` — not
around it — so CLI and GUI stay unified.

## Error handling

- **Empty selection under absolute scoring** is a legitimate outcome, not an error.
  Report the per-channel maxima observed and the configured floors.
- **Level unavailable**: requesting polygon from an OBB model is refused at config
  time with the reason, not silently downgraded.
- **Contour extraction failure** (empty `findContours` result) falls back to the
  detection's `corners` for that instance, as the segment extractor already does at
  `stages/obb.py:1046`.
- **Hardlink failure** (cross-device, unsupported filesystem) falls back to copying,
  with a logged warning.
- **Partial writes**: roots are written to a temporary directory and moved into place
  so a cancelled round never registers a half-written source.

## Testing

- Unit: `derive_down` level round-trip properties; level honesty (an OBB model must
  never produce a polygon root); strict-drop accounting; hardlink layout with copy
  fallback; `source.json` schema.
- Unit: each absolute `score_*` returns exactly 0 on a clean frame and rises
  monotonically with severity; asymmetric count weighting; ported fragmentation
  heuristic matches the legacy scorer's output on the existing fixtures.
- Integration: DetectKit round-trip — `scan_source_levels` on each of the three roots
  resolves to the stamped level.
- Regression: the eight tests currently pinning the legacy scalar path are repointed
  at `frame_signals` / `select` rather than deleted.
- **Equivalence harness runs as a no-op gate.** Every change here is export-only or
  behind an opt-in flag, so tracking output must remain byte-identical on MPS and
  CUDA. Frame *selection* deliberately changes and is covered by a committed
  characterization golden instead — the durable-guard pattern established by the
  shared param-builder work, since a post-collapse oracle would be tautological.

## Out of scope

- Upward level derivation (inferring contours from boxes). SAM2 mask-priming already
  covers that need in DetectKit (`jobs/sam2_escalation.py`).
- Relocating `FilterKitCore` out of the app layer — the documented carve-out in
  `data/al/candidate_pool.py` stays until the Simplification Sprint lands it.
- Pose or identity dataset export; this design covers detection geometry only.
- The double-inference pass (finding 15) is documented in `source.json` rather than
  removed; unifying it with the cached detections is a separate performance slice.
