# DetectKit Geometry Levels — Polygon-First Label Model — Design

**Date:** 2026-07-24
**Status:** Design approved; pending spec review → implementation plan
**Scope:** DetectKit label model, source import/validation, training role matrix,
X-AnyLabeling round-trip, TrackerKit active-learning export
**Depends on:** SAHI sliced inference (`2026-07-23-sahi-sliced-inference-design.md`) merged.
This design is independent of SAHI at the code level and may be implemented in
parallel, but lands after it.

## 1. Motivation

DetectKit cannot train segmentation or detection-only models, and the reason is not
a missing feature — it is that **segmentation information is destroyed at the front
door**.

Every imported source is materialized to YOLO-OBB 8-coordinate polygons
(`source_import.py:260 _format_obb_line`), and COCO `segmentation` is collapsed to a
rotated rect at import time (`source_import.py:350 _points_to_min_area_rect`). All
training roles then derive *downward* from that merged OBB dataset
(`dataset_builders.py`: `obb_direct` passthrough, `seq_detect` OBB→AABB,
`seq_crop_obb` crop). The X-AnyLabeling round-trip is hardcoded `--mode obb`
(`dataset_panel.py:637`).

Since the direct inference path now consumes `detect` and `segment` checkpoints as
OBB sources, the pipeline can *use* models DetectKit cannot *train*. This design
closes that gap by making the canonical label model richer than OBB.

## 2. Programme context

This is piece **A** of a three-piece programme, agreed in the order below. Each piece
gets its own spec → plan → implementation cycle.

| Piece | Scope | Order |
|---|---|---|
| SAHI | Sliced inference (already specced and planned) | 1st |
| **A** | **This document — polygon-first label model** | 2nd (may run parallel to SAHI) |
| C | Inference **region-source unification** | 3rd |
| B | SAM2 point-prompt escalation of OBB/detect → segment | 4th |

**Why C precedes B, and why C's charter is broad.** SAHI's `run_direct_sliced` is
structurally identical to `_run_sequential`: sub-regions → chunked predict →
extract-with-offset-remap → merge (SAHI already reuses `merge_obb_results` from
`obb.py`). The two differ only in where sub-regions come from — a fixed grid vs.
stage-1 proposal boxes. C therefore unifies on a **region-source abstraction**
(`regions → executor → extract-with-transform → merge`, where a region source is a
slice grid, stage-1 proposals, or the trivial whole frame). Two things blocked today
fall out of that one refactor:

- **Sequential-mode slicing**, an explicit SAHI v1 non-goal.
- **`seq_crop_segment`**, blocked because `_run_sequential` (`obb.py:652`) calls
  `extract_obb_result()` unconditionally (which reads `result.obb`), while
  `_extract_obb_from_masks` (`obb.py:899`) has no `offset`/`scale` parameters at all.
  Compounding this, `_assert_direct_task_matches_checkpoint` is direct-mode only, so
  a segment checkpoint used as stage-2 today yields `result.obb is None` on every
  frame — zero detections for a whole video, with no error.

C must land **after** SAHI: the SAHI plan couples to `obb.py`'s private extractor API
(`_resolve_imgsz`, `_extract_obb_from_boxes`, `_extract_obb_from_masks`,
`extract_obb_result`, `_apply_raw_detection_cap`, `merge_obb_results`,
`_empty_obb_result`, `_RawOBBTensors`) and cites line numbers task-by-task.

**Non-goals for this design:** SAM2 escalation (piece B), any change to the inference
region-source architecture (piece C), exposing `seq_crop_segment` in the training
dialog (deferred to C), and any change to class handling, splits, or training
hyperparameters.

## 3. Data model: `GeometryLevel`

One new concept, totally ordered by information content:

```
aabb  <  obb  <  polygon
```

Derivation **downward** is automatic and lossless-to-the-target
(polygon → `minAreaRect` → OBB → AABB). Derivation **upward** is impossible without
new information; that is piece B.

### 3a. On-disk encoding is unchanged in syntax

Every label line stays `class_id` followed by a normalized point list, exactly as
`_format_obb_line` writes today. An `aabb`-level source stores **axis-aligned quads**,
not `cx cy w h`. Consequences:

- Exactly one label parser, one canvas draw path, one merge path.
- YOLO-obb and YOLO-seg share this syntax, so `segment_direct` is a passthrough
  build; only the detect roles rewrite to `cx cy w h`.
- The recorded level is what keeps us honest that an `aabb` source's orientation is
  not real information.

`OBBCanvas._draw_detections` (`canvas.py:136`) already builds a `QPolygonF` from any
`polygon_px` with `len >= 3`, and is view-only (editing is delegated to
X-AnyLabeling). Variable-length polygons therefore render with no canvas change; only
the label→`polygon_px` parser stops assuming 8 coordinates.

### 3b. Where the level is stored

On the `OBBSource` entry in the **project file** — not in a sidecar inside the source
directory.

Rationale: sources may be *linked* rather than owned (`detectkit_project_is_portable`,
`project.py:219`), and a linked source directory is not reliably writable.
Materialized source directories are regenerated from the project, so the project file
is the single source of truth and there is no dual-write to keep consistent.

### 3c. Import stops downgrading

`_points_to_min_area_rect` becomes a *derivation* applied at training time, not an
import-time transform. Import levels:

| Input | Level |
|---|---|
| COCO with `segmentation` | `polygon` |
| COCO with `bbox` only | `aabb` |
| YOLO-detect | `aabb` |
| YOLO-OBB | `obb` |

## 4. Validation: sources are homogeneous

**Rule: a source has exactly one level. Mixed evidence is an error the user resolves,
never something the system guesses at.**

Level detection scans labels and classifies each file:

- any line with **>4 points** → unambiguous `polygon` evidence;
- **exactly-4-point** files → ambiguous (an OBB and a quad polygon are the same eight
  numbers) → resolved by the declared/intended level;
- `cx cy w h` input → `aabb`.

If a source contains both polygon-evidence files and 4-point-only files,
`DetectKitSourceValidationDialog` (`dialogs/source_validation.py:69`) **blocks** and
asks the user to resolve it: escalate the remainder, or explicitly confirm the quads
are genuine contours.

The confirm-override exists because a legitimately quad-shaped contour in a polygon
source is possible and would otherwise be unresolvable. It is an explicit user
assertion recorded on the source, not a silent guess.

## 5. Training roles

Six roles; each declares a minimum level; downward derivation is automatic.

| Role | YOLO task | Min level | Builder |
|---|---|---|---|
| `obb_direct` | obb | `obb` | passthrough (poly→`minAreaRect`) |
| `detect_direct` *(new)* | detect | `aabb` | `derive_detect_*` |
| `segment_direct` *(new)* | segment | `polygon` | passthrough as YOLO-seg |
| `seq_detect` | detect | `aabb` | `derive_detect_*` |
| `seq_crop_obb` | obb | `obb` | existing crop builder |
| `seq_crop_segment` *(new, hidden until C)* | segment | `polygon` | crop + clip polygon to crop space |

`detect_direct` and `seq_detect` train the same YOLO task but remain distinct roles:
they are different artifacts with different `imgsz` and different consumers, and the
project already keys checkpoints by role, with inference resolving models by role
(`project.py:687`).

`seq_crop_segment` is defined in the taxonomy, the level gating, and the dataset
builders **now**, so no data-model migration is needed when C lands — but it is not
offered in the training dialog until C makes it runnable. Six roles in the taxonomy,
five selectable initially.

### 5a. Builders

The existing derivation chain survives intact:

- `derive_detect_dataset_from_obb` (`dataset_builders.py:352`) gains a polygon input
  case.
- `derive_crop_obb_dataset_from_obb` (`:539`) gains a polygon sibling that clips the
  contour to the crop rectangle and re-normalizes to crop space. A contour extending
  past the crop is clipped, not dropped; an object whose clipped area falls below the
  builder's existing minimum-object threshold is dropped, matching how the current
  crop builder treats degenerate objects.
- `segment_direct` is a passthrough (shared line syntax).

### 5b. Mixed-level merges

A merged training dataset's level is `min()` across the selected sources. Adding one
`obb`-level source therefore disables `segment_direct`.

**This is a deliberate behavior change worth calling out.** The same rule means that
including an `aabb`-level source (YOLO-detect, or COCO with `bbox` only) drops the
merge to `aabb` and blocks **`obb_direct` and `seq_crop_obb` as well** — you cannot
train oriented boxes from axis-aligned ones. Today those sources are silently
upgraded to axis-aligned quads at import and the OBB roles are offered anyway,
training on orientations that were never annotated. Blocking is the correct behavior;
users who relied on the old permissiveness will see roles disappear, and the message
names the source responsible.

When a role is blocked, the training dialog states **"segment unavailable: source X is
obb-level"**, naming the blocking source — it does not silently grey the checkbox.
That message is the designed extension point where piece B later grows an
**"Escalate X to segment with SAM2"** action.

## 6. X-AnyLabeling round-trip

The hardcoded `--mode obb` becomes a mode derived from the source level:
`aabb → rectangle`, `obb → obb`, `polygon → polygon`.

The user may also deliberately select a **higher** mode than the source's current
level. That is the manual escalation path — the hand-operated version of piece B —
and it is offered explicitly rather than occurring as a side effect.

**Sync-back is where intent meets evidence.** `_sync_xal_stage_back`
(`dataset_panel.py:437`) today deletes the source's `labels/` and copies the staged
directory over it wholesale. It instead validates first:

1. Recompute the level from the staged labels.
2. Resolve the 4-point ambiguity using the launch mode as declared intent.
3. Run the §4 homogeneity check.
4. Only then copy back and update the source's level in the project.

A mixed source never lands on disk.

**Open verification (implementation-time, not a design gap):** the exact mode
vocabulary accepted by the external `xanylabeling convert` CLI
(`integrations/xanylabeling/cli.py:33`) must be confirmed against an installed
environment before the flags are wired. The current `obb` mode is the known-good
anchor.

## 7. TrackerKit active-learning export

Export must emit the richest geometry the model produced, so an exported source lands
at the level its detector supports rather than being flattened to OBB.

**Approach: re-detect at export time.** AL export already selects ~N worst frames from
a whole video, and `_init_detection_runner` (`dataset_generation.py:413`) already
builds an inference config and re-runs detection for dataset generation. Cost is N
frames of inference, not the whole video.

Rejected alternative: carrying polygons through the tracking pipeline and the `.npz`
detection cache. That is variable-length data in a fixed-width array world, changes
the cache schema (a byte-identical-parity risk on freshly merged code), and costs
memory on every frame to serve an export of a few dozen.

**Accepted tradeoff:** exported annotations come from a fresh detector pass, so they
may differ slightly from the tracked detections that *scored* the frame as
interesting (different confidence threshold, no tracking-side filtering). This is
correct for the use case: the information wanted at export time is what the model says
about the frame, not what the tracker believed, and the tracker's OBB is derivable
from the polygon anyway.

### 7a. Mechanism

To keep the "all detection goes through `InferenceRunner`" rule intact — no
ultralytics side-channel — `OBBResult` (`result.py:20`) gains one optional field:

```python
polygons: list[np.ndarray] | None = None   # native contours, export-only
```

populated **only** when the config sets `emit_native_geometry`, which the tracking
path never sets. Therefore: no hot-path memory cost, no `.npz` cache schema change,
no parity risk. The field is consumed in-process by the exporter and is never
serialized.

This is not the rejected alternative above in smaller form. The rejected design made
polygons a *pipeline-wide* concern — serialized into the detection cache and read
back by the exporter from tracked state. Here the field is opt-in, lives only for the
duration of one export-time detection call, and no consumer other than the exporter
ever observes it as non-`None`.

`_write_yolo_obb_label` (`al_worker.py:112`) generalizes to write a point list, and
the exported DetectKit source is stamped with the level matching the model's task:
`segment → polygon`, `obb → obb`, `detect → aabb`.

## 8. Migration

**No-op on disk.** A source with no recorded level reads as `obb`, which is precisely
what every existing project already is. No label files are rewritten and no re-import
is required.

## 9. Error handling

Three loud failures, following the codebase's established preference for failing
loudly over silently producing nothing (cf. `_assert_direct_task_matches_checkpoint`,
`obb.py:329`):

- A role requested above the merged dataset's level is refused, naming the specific
  blocking source.
- A source failing the homogeneity check blocks at validation.
- A derivation producing zero valid objects for an image reports that image rather
  than silently emitting an empty label file.

## 10. Testing

- **Level detection** across all four input kinds: YOLO-obb, YOLO-detect,
  COCO-with-segmentation, COCO-bbox-only.
- **Homogeneity validation**, including the explicit confirm-override.
- **Each derivation path:** poly→obb, poly→aabb, obb→aabb, poly→crop-polygon.
- **Round-trip level recompute**, with launch intent resolving the 4-point case.
- **Export geometry per model task:** segment→polygon, obb→obb, detect→aabb.
- **Regression gate (the one that matters):** an existing `obb`-only project must
  produce **byte-identical merged and derived datasets** before and after this
  change. This is the guarantee that a data-model change did not perturb anyone's
  existing training data.
