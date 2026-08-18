# Dataset Generation

Dataset generation supports active learning (AL) loops for improving detector
models: TrackerKit scores tracked frames for how much they would help
retraining, selects a diverse subset, and exports them as one or more
DetectKit-importable dataset roots.

## What It Does

- Scores every tracked frame using absolute-severity quality channels (below).
- Selects challenging, diverse frames for annotation (rank + spacing, with an
  optional near-duplicate dedup pass over the selection).
- Exports up to three sibling **source roots** — `polygon/`, `obb/`, `aabb/` —
  each a directly importable DetectKit source.

## Output Layout

One AL round writes a directory such as:

```
<video>_datasets/active_learning/al_20260817_141230/
  polygon/  images/  labels/  classes.txt  source.json    level=polygon, authoritative
  obb/      images/  labels/  classes.txt  source.json    level=obb,  derived_from=polygon
  aabb/     images/  labels/  classes.txt  source.json    level=aabb, derived_from=polygon
  manifest.json
```

The highest level the model actually produced is the **authoritative** root;
lower levels are **derived** from it by lossless downward conversion
(polygon → `minAreaRect` → obb → aabb). Images in derived roots are
**hardlinks** to the authoritative root's images (falling back to a copy on
filesystems that don't support hardlinks), so writing extra levels costs
almost no additional disk space.

Each root is independently a valid DetectKit source: it has its own
`images/`, `labels/`, `classes.txt`, and `source.json`, and can be opened or
registered in DetectKit on its own. `source.json` per root records the level,
whether it is authoritative or derived, its `derived_from` level, class
names, and provenance (model path/task, thresholds, acquisition preset and
weights). `manifest.json` at the round level lists every root written plus
round totals (frames exported, dropped-lost, dropped-unmatched, objects) and
which frame ids were selected versus context.

### Which levels a detection source can reach

The export never claims a level the underlying model did not produce:

| Detection source | Native geometry | Roots written |
|---|---|---|
| YOLO segment (direct or sequential stage-2) | mask contours | polygon, obb, aabb |
| YOLO OBB | rotated quad | obb, aabb |
| YOLO detect | axis-aligned box | aabb |
| Background subtraction | foreground contours | polygon, obb, aabb |

A rotated quad is never written as a polygon root — it carries no contour
information, so requesting polygon output from an OBB or detect model is
refused (or, in the panel, simply greyed out) rather than silently
downgraded or fabricated.

## Quality Metrics (Absolute Severities)

Every channel returns an **absolute** severity in `[0, 1]` that is exactly
`0.0` when the frame is not a problem, not a within-run rank. This makes
scores comparable across videos, and it means a cleanly tracked video can
legitimately produce few or even zero candidate frames — that is expected
behaviour, not a bug.

- **Uncertainty** — low mean detection confidence relative to a floor.
- **Count deviation** — detected object count vs. expected target count
  (asymmetric: missing an animal scores worse than a spurious extra box).
- **Crowd** — animals genuinely overlapping/touching (max pairwise polygon
  overlap).
- **Fragmentation** — one animal apparently split into two nearby, smaller
  detections. This is a distinct signal from crowd (see
  `docs/developer-guide/confidence-metrics.md`).
- **Edge** — a detection sitting close to the frame border.
- **Assignment / track loss / position uncertainty** — tracker-only channels
  from the Kalman/assignment pipeline.

### When selection comes back empty

If no frame's composite score clears `DATASET_MIN_SELECTION_SCORE`, the run
does not silently export nothing — it reports the **highest severity
observed per channel** across the whole video, so you can see which signal
came closest and by how much, rather than a bare "no frames found" error.

**Caveat:** `DATASET_MIN_SELECTION_SCORE` defaults to `0.0`, and selection
gates on `score >= min_score`. A frame that scores exactly `0.0` on every
channel therefore still clears the default gate and is still eligible to be
selected — the "a clean video legitimately exports nothing" behaviour only
materializes once you raise this threshold above `0.0`. At the default,
absolute scoring makes selection *honest* (a clean video ranks its frames
near zero instead of manufacturing a full-strength ranking), but it does not
by itself make selection *empty* — raise `DATASET_MIN_SELECTION_SCORE` if you
want "nothing worth reviewing" to actually export nothing.

## Strict Labels and Dropped Rows

A tracking row is exported as a label only when it binds one-to-one (within
a radius scaled to `REFERENCE_BODY_SIZE`) to a real detection from the export
pass. Rows are **dropped, not fabricated**, when:

- the track's `State` is `lost` (interpolated/coasted, not a real
  detection), or
- the row cannot be matched to any detection within the matching radius.

These drops are never silently absorbed into a placeholder box. Counts of
`dropped_lost` and `dropped_unmatched` (plus `frames_exported` and `objects`)
appear in each root's `source.json` totals and in `manifest.json`, and are
summarized to the user at the end of the export in the finish message /
dataset panel summary.

### Known limitation: detections with no tracking row are dropped

Matching is mutual-exclusion in **both** directions. A row with no detection
is dropped (above), and so is a **detection with no row** — which means an
animal the tracker never picked up can be visible in an exported image while
carrying no label. That direction is dropped *silently*: `dropped_unmatched`
counts only the mirror case (rows with no detection), and there is no counter
for detections with no row. Review each exported image before training; this
is inherent to binding labels to tracking output and is not something the
export can detect on your behalf. The caveat is also stated in every root's
`source.json` provenance note.

### Frames where nothing survived are not exported

A frame whose export detection pass produced no surviving label is **skipped**,
not written with an empty label file. YOLO reads an empty `.txt` as "this
image contains no objects", so exporting one would assert, as ground truth,
that a frame the exporter simply failed on is empty background — exactly the
frames active learning picks. Skipped frames are counted in `manifest.json`
under `frames_skipped_no_records`, with their ids in
`skipped_frame_ids_no_records`, and per-frame detection failures under
`detection_failed`. If **no** frame in a round carries any geometry, the
export fails loudly rather than writing an empty dataset.

### Multi-class checkpoints

Class ids come from the detection model; class names come from your project.
If the model emits an id beyond the names you supplied, `classes.txt` is
padded with `class_<n>` placeholders so every label id indexes a real line,
and the padded names are recorded in `class_names_autofilled` in both
`manifest.json` and each root's `source.json`. Rename them in DetectKit.

## When to Use

- Detector underperforms on specific lighting or behaviors.
- You need focused retraining data instead of random frame sampling.

## Tradeoffs

- Aggressive selection (a low min-selection-score, or a preset weighted
  toward one channel) can bias toward outliers.
- Conservative selection may miss edge cases.

## Practical Loop

1. Run TrackerKit and generate candidate frames — DatasetPanel shows which
   levels are achievable for the current detection method before you run.
2. Import the authoritative root (and any derived roots you want) into
   DetectKit, validate/annotate the selected frames there.
3. Retrain the YOLO model.
4. Re-run TrackerKit on representative videos.
5. Compare confidence and identity continuity metrics.
