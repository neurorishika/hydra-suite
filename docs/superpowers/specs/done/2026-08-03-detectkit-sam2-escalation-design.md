# DetectKit Phase B — SAM2 Mask Priming (escalate-all) — Design

Date: 2026-08-03
Status: Design (approved for planning)
Programme: DetectKit geometry programme (SAHI → A → C → **B**). A, A.5, and C are
merged; B is the final piece.

## 1. Goal

Let a user turn an existing OBB/detect-labeled dataset into segmentation
(polygon) labels **without hand-drawing every contour**. An offline batch runs
SAM2 over the boxes the user already drew, primes a mask per detection, and
writes a **new, reviewable** segment source. The user fixes the primed masks in
X-AnyLabeling and marks the source reviewed; only then does it count for
training.

This unblocks the `polygon` `GeometryLevel` (and the `seq_crop_segment` /
`segment_direct` roles that require it) from datasets that only have OBB/box
labels today. Upward derivation (obb → polygon) "needs new information and is out
of scope" for the existing downward projectors
(`training/sliced_dataset.py:59`); SAM2 is that new information.

## 2. Scope

**In scope (MVP):** a dataset-wide "escalate all eligible sources to segment"
bulk run, producing new derived sources the user reviews.

**Explicitly out of scope (not this spec):**
- Interactive per-object mask priming in the canvas (real-time click-to-mask).
- In-place mutation of original sources (we never touch originals).
- A new X-AnyLabeling bridge (the existing polygon→`segment` round-trip is reused).
- SAM2 under the runtime **tier** system (TensorRT/CoreML export); SAM2 runs
  torch-only on the resolved device. Prompt models do not fit `InferenceRunner`'s
  `predict(frame)` contract.
- COCO-JSON import (segmentation is round-tripped as YOLO normalized-contour
  `.txt`, which is what we write).

## 3. Decisions (locked during brainstorming)

1. **MVP workflow:** escalate-**all** bulk pipeline (dataset-wide), not per-object
   interactive and not single-source-only.
2. **Output & trust model:** write masks into a **new derived source**
   (`<name>_seg`), leaving originals byte-untouched. The derived source starts
   `polygon`-level but `reviewed=False`; a user "Mark reviewed" action (after
   fixing in X-AnyLabeling) is what makes it count for training. Fully
   reversible — deleting the derived source is complete undo.
3. **Packaging:** SAM2 is a **hard dependency** with **managed weights**
   (auto-downloaded from the HF hub into the models dir, mirroring
   `vitpose_checkpoints`; `huggingface_hub` is already a dependency). SAM2 is
   **lazy-imported** so non-SAM workflows never pay its load cost.
4. **Prompt strategy:** per detection, **AABB box + positive center point +
   negative points at overlapping neighbors** (bleed suppression); take SAM2's
   highest predicted-IoU mask.
5. **User-selectable SAM2 version:** the escalate dialog exposes a version
   dropdown (SAM2.1 Hiera `tiny` / `small` / `base_plus` / `large`, default
   `base_plus`). Only the chosen variant is downloaded/cached; the choice is
   persisted on the derived source and shown in the run summary.

## 4. Data flow

```
escalate dialog: pick sources + SAM2 variant + params
  └─> Sam2EscalationWorker (BaseWorker):
        ensure/download chosen SAM2 checkpoint (managed weights)
        load Sam2SegmentExecutor on the resolved torch device
        for each selected source (eligible = level != polygon):
          for each image in source:
            read the image's OBB/box label lines -> boxes[]
            prompts = build_prompts(boxes, image_hw)   # box + center + neg pts
            for each detection:
              mask = executor.segment(image, prompt)   # best-IoU mask
              contour = mask_to_contour(mask)           # largest ext contour, simplified
              if contour is empty/low-IoU: contour = obb_rectangle_polygon(box)  # fallback, counted
              write "0 <normalized contour>" via _write_geometry_label(...)
          copy images into <name>_seg/images, labels into <name>_seg/labels
        register OBBSource(<name>_seg, level="polygon", reviewed=False,
                           derived_from=<name>, sam2_variant=<variant>)
  └─> run summary: "N primed, K fell back to OBB rectangle — review K first"

user: open <name>_seg in X-AnyLabeling (existing segment round-trip) -> fix masks
      -> "Mark reviewed" -> reviewed=True -> source now eligible for training builds
```

## 5. Components

### 5.1 SAM2 executor — `core/inference/sam2/executor.py` (new)
`Sam2SegmentExecutor`: a **standalone** prompt-in/mask-out executor, deliberately
outside `InferenceRunner`/`load_obb_executor` (those are `predict(frame)`-shaped,
no prompt inputs — recon confirmed). Responsibilities:
- Lazy-import `sam2`; construct the predictor from a variant's checkpoint + model
  config.
- Resolve the torch device from the host (`utils/gpu_utils` /
  `resolved_backend_for`): cuda → mps → cpu. No TensorRT/CoreML export.
- `segment(image_bgr, prompt) -> (mask: np.bool_ HxW, iou: float)`: set the image
  once per image (SAM2 caches the image embedding), then run each detection's
  prompt against it; return the highest predicted-IoU mask of SAM2's multimask
  output.
- `set_image(image)` / per-detection `predict(...)` split so the expensive image
  embedding is computed once per image, not once per detection.

### 5.2 Checkpoint management — `core/inference/sam2/checkpoints.py` (new)
Mirrors `core/identity/pose/vitpose/…checkpoints`. Holds the variant registry:

| variant key            | HF repo / file                      | model config |
|------------------------|-------------------------------------|--------------|
| `sam2.1-hiera-tiny`      | facebook/sam2.1-hiera-tiny (or equiv) | (bundled cfg) |
| `sam2.1-hiera-small`     | …                                   | …            |
| `sam2.1-hiera-base_plus` | … (default)                         | …            |
| `sam2.1-hiera-large`     | …                                   | …            |

`ensure_checkpoint(variant) -> Path`: return the cached checkpoint, downloading
from HF into the models dir on first use of that variant. Offline + uncached
chosen variant → raise a clear "variant `<x>` not downloaded" error naming the
variant (fail fast, before the run starts). Exact HF repo ids/filenames verified
at implementation time.

### 5.3 Prompt geometry — `detectkit/jobs/sam2_prompts.py` (new)
Pure, SAM2-free, unit-testable:
```
build_prompts(boxes: list[Box], image_hw) -> list[Prompt]
  Prompt = { box_xyxy, positive_points:[(x,y)], negative_points:[(x,y)] }
```
- `box_xyxy`: axis-aligned bbox of the detection (OBB → its AABB; aabb → itself).
- `positive_points`: the OBB/box center.
- `negative_points`: centers of neighbors whose AABB overlaps this detection's
  AABB (dense-scene bleed suppression). Overlap predicate is a pure geometry test
  over the image's full box set (no config `iou_threshold` coupling — same
  discipline as the C merge gate).

### 5.4 Mask → contour — `core/inference/sam2/masks.py` (new)
`mask_to_contour(mask: np.bool_) -> np.ndarray (P,2) | None`: largest **external**
contour via cv2 `findContours`, simplified with `approxPolyDP` (epsilon a small
fraction of perimeter; enforce a minimum vertex count). Single contour only
(YOLO-seg has no holes). `None` when the mask is empty/degenerate → caller applies
the OBB-rectangle fallback.

### 5.5 Escalation worker — `detectkit/jobs/sam2_escalation_worker.py` (new)
`Sam2EscalationWorker(BaseWorker)` (`widgets/workers.py:6`), mirroring the
structure of `jobs/al_worker.py` but reading **existing** labels rather than
running a detector:
- Iterates selected sources → images → reads label lines → `build_prompts` →
  `executor.segment` (one `set_image` per image) → `mask_to_contour` → writes
  contour labels via the existing `_write_geometry_label` (`al_worker.py:116`,
  already accepts a raw `(P,2)` polygon).
- Copies images + writes labels into `<name>_seg/`; registers the derived
  `OBBSource`.
- Emits `status`/`progress`; accumulates a per-run summary (primed vs
  fell-back counts).
- The SAM2 executor is **injected** (constructor arg) so tests can pass a fake.

### 5.6 Escalate dialog — `detectkit/gui/dialogs/escalate_sam2_dialog.py` (new)
`BaseDialog` (`widgets/dialogs.py`): source multi-select (select-all default,
already-polygon sources shown disabled), **SAM2 version dropdown** (§3.5), prompt
params (neighbor-negative on/off, contour-simplify epsilon), Run + progress +
summary. Blocks the run (with a clear message) when SAM2/weights are unavailable.

### 5.7 Source model — `detectkit/gui/models.py:27` (`OBBSource`)
Add:
- `reviewed: bool = True` — default `True` so **existing sources are unaffected**;
  escalation sets `False` on the new source.
- `derived_from: str | None = None` — provenance (the origin source name).
- `sam2_variant: str | None = None` — which SAM2 version primed it.

All three carried through `to_dict`/`from_dict` (`models.py:47`/`:60`).

### 5.8 Training gating — `dataset_builders.py` / `dialogs/training_dialog.py`
- Dataset builds **exclude** `reviewed=False` sources, with a clear message
  ("`<name>_seg` is unreviewed — review in X-AnyLabeling and Mark reviewed").
- Reuse the level machinery (`merged_level_and_blocker`,
  `blocked_roles_for_level`); the review gate is an additional filter, not a
  replacement.
- The role-gating tooltip (`training_dialog.py:1504`) gains a secondary
  "Escalate '{who}' to segment (SAM2)" entry point (single-source escalation via
  the same worker) alongside the primary dataset-wide button.

### 5.9 Review actions — reuse existing round-trip
- "Open in X-AnyLabeling" already maps `polygon → --mode segment`
  (`dataset_panel.py:58` `xal_mode_for_level`, launch at `:697`). No new bridge.
- New **"Mark reviewed"** action flips `reviewed=True` on the derived source.

## 6. Error handling & edge cases

| Case | Behavior |
|---|---|
| SAM2 empty / low-IoU mask for a box | Fall back to the OBB-as-polygon rectangle (never drop a detection); count it in the summary so the user reviews those first. |
| `aabb`-level source | Included; box prompt = the box itself, center point = box center. |
| Already-`polygon` source | Skipped (nothing to escalate); shown disabled in the dialog. |
| SAM2 not importable / chosen variant not cached (offline) | Escalate action/run blocked with a clear message naming the variant; fail fast before processing. |
| Re-run over an existing `<name>_seg` | Guarded overwrite (confirm); originals never touched. |
| Image with zero labels | Skipped (no prompts, no output line) — same as an empty AL frame. |

## 7. Testing

Pure unit (no SAM2, no network):
- `build_prompts`: box→AABB, center positive point, neighbor-negative geometry
  (touching vs disjoint boxes).
- `mask_to_contour`: synthetic binary mask → expected largest-contour polygon;
  empty mask → `None`.
- `checkpoints.ensure_checkpoint`: cached-path resolution + offline-uncached
  raises the named error (HF download mocked).
- `OBBSource` `reviewed`/`derived_from`/`sam2_variant` `to_dict`/`from_dict`
  round-trip; default `reviewed=True` for legacy dicts.
- Training gating: an unreviewed polygon source is excluded from the build with
  the expected message.

Worker orchestration (injected **fake** SAM2 executor, no weights):
- Reads labels → builds prompts → writes `<name>_seg` with contour labels,
  `reviewed=False`, `derived_from`/`sam2_variant` set; empty-mask fallback path
  produces the OBB rectangle and increments the fell-back counter.

Guarded real-SAM2 smoke (`pytest.importorskip("sam2")` + a small checkpoint,
`skipif` off-platform), mirroring the CoreML export smoke — exercises a real
`Sam2SegmentExecutor.segment` on one image + box.

## 8. Tradeoffs noted

- **Hard dependency** bloats every install (including CPU-only/GUI-less) with
  SAM2 + its deps; accepted per the packaging decision. Mitigated by lazy import
  (load cost only when escalation runs).
- Derived source **copies** images into `<name>_seg/` (matches AL; costs disk).
  Symlinking is a future optimization, not in this spec.

## 9. Future work (not this spec)

- Interactive per-object canvas priming (click-to-mask).
- Symlinked (zero-copy) derived-source images.
- Escalating a source **in place** with a backup, for users who prefer one source.
- SAM2 under the gpu_fast tier (TensorRT/CoreML) if batch escalation becomes a
  throughput bottleneck.
