# ViTPose dataset-driven auto-sizing (Slice 2) — design

Date: 2026-08-04. Branch: `feat/vitpose-geometry`, on top of Slice 1.

Slice 1 spec: `docs/superpowers/specs/2026-08-03-vitpose-per-checkpoint-geometry-design.md`.

## Why

Slice 1 made ViTPose's input geometry a per-checkpoint property, but the only ways
to reach a non-default geometry today are hand-editing `run.json` or loading an
externally trained checkpoint. Nobody will do either.

The size that actually suits a dataset is a property of the dataset: how large the
animals are in pixels, and what shape they are. That is measurable. ClassKit already
does this for classifier inputs and DetectKit for slice tiles; PoseKit has no
equivalent — its `imgsz` control is a bare spin box defaulting to 640
(`posekit/gui/dialogs/training.py:737-740`).

This slice measures the labelled data and proposes a geometry, with the operator
free to override it.

## What changed from Slice 1's "Slice 2 notes"

Those notes were written before the training dialog's data flow was mapped. Two
corrections:

1. **The source of truth is the PoseKit label store, not the COCO dataset.**
   `build_coco_keypoints_dataset` runs *inside* `ViTPoseTrainingWorker.run()`
   (`posekit/gui/dialogs/training.py:356`), i.e. on a worker thread after the operator
   clicks Start. At dialog time no `annotations.json` exists, so a suggestion shown in
   the dialog must come from the YOLO-pose `.txt` labels the project already has.

2. **The `measured_input_size` manifest stamp is cut.** The manifest is written after
   the size has already been chosen, so it could only record provenance — and the
   chosen size is already recorded twice, in `run.json` and in every saved checkpoint's
   `input_size`. A third copy nobody reads back is unused surface.

## Non-goals

- No change to how training consumes the geometry. Slice 1 already built that:
  `RunConfig.input_size` exists, validates, and flows to the checkpoint stamp.
- No multi-source label routing fix (see Known inconsistency).
- No auto-measurement on dialog open. Measurement is I/O; it happens on demand.
- No change to the ViTPose default. A project that never presses the button trains
  exactly as it does today.

---

## 1. The estimator

New file `src/hydra_suite/training/pose_geometry_measure.py`. Pure measurement — no
Qt, and **no imports from any app layer**, per the repo's dependency rule that
Training must not import from PoseKit. That rule has a concrete benefit here:
PoseKit's own `load_yolo_pose_label` (`posekit/core/extensions.py:516`) parses only the
**first line** of a label file and would silently under-count every multi-animal frame.
This module parses all lines.

```python
@dataclass(frozen=True)
class PoseSizeStats:
    sample_count: int          # instances measured, not files read
    frames_scanned: int
    frames_skipped: int        # unreadable image or no usable instance
    median_aspect: float       # width / height of the keypoint bbox
    median_long_px: float
    p90_long_px: float
    suggested_hw: list[int]    # [H, W], multiples of 32, within [64, 384]
    clamped: bool              # the raw suggestion exceeded the cap

def measure_pose_geometry(
    image_paths: Sequence[Path],
    labels_dir: Path,
    num_keypoints: int,
    *,
    detail: float = 1.0,
    max_images: int = 500,
    seed: int = 0,
) -> PoseSizeStats
```

### Algorithm

1. **Enumerate.** A frame is labelled iff `labels_dir / f"{stem}.txt"` exists and is
   non-empty — the convention `list_labeled_indices`
   (`posekit/core/extensions.py:743-755`) and `build_coco_keypoints_dataset`
   (`extensions.py:1176`) both already use.
2. **Subsample** deterministically to `max_images` with `random.Random(seed)`, the
   pattern `analyze_obb_sizes` uses (`training/dataset_inspector.py:365-367`).
3. **Read dimensions** with `PIL.Image.open(path).size`, which reads the header only.
   Every existing consumer full-decodes with `cv2.imread`; header reads are what make
   an in-dialog button responsive over hundreds of frames.
4. **Parse every line.** A line is usable when it has at least `5 + 3 * num_keypoints`
   fields (the check `_parse_label_lines` uses, `extensions.py:1198-1216`). Coordinates
   are normalized; multiply by the image's pixel dimensions.
5. **Per instance**, take the axis-aligned bounding box of keypoints with `v > 0`.
   Invisible keypoints are excluded — they carry no position information, and including
   them at their stored coordinates would bias every box toward the origin.
6. **Aggregate:** `median_aspect = median(w / h)`, `median_long_px = median(max(w, h))`,
   `p90_long_px = 90th percentile of max(w, h)`.
7. **Reconstruct a coherent pair** rather than taking independent medians of width and
   height, which can describe an animal that does not exist:
   ```
   long = median_long_px * detail
   if median_aspect >= 1.0:   W, H = long, long / median_aspect
   else:                      H, W = long, long * median_aspect
   ```
8. **Snap and clamp.** Round each dimension to the nearest multiple of 32 (ClassKit's
   rule for rescaled values, `classkit/gui/dialogs/training.py:97-104`), then clamp to
   `[64, 384]`. Set `clamped=True` if either dimension was capped.

### Why no extra padding factor

Inference applies `PADDING_FACTOR = 1.25` inside `box2cs` before warping, so the model
already sees more than the bare keypoint extent. Sizing the input to the measured
animal is therefore correct; the `detail` multiplier exists for operator preference,
not to compensate for padding.

### Why the cap is 384

ViT attention cost scales with token count: 192x256 is 192 tokens, 256x256 is 256,
384x384 is 576, 512x512 is 1024. Capping the *suggestion* at 384 keeps the tool from
silently proposing a model that trains several times slower. Larger values remain
reachable by typing them into the control — the cap constrains the suggestion, not
the operator.

### Error handling

| Condition | Behaviour |
|---|---|
| No labelled frames found | `ValueError` naming `labels_dir` |
| Image unreadable / not an image | skip the frame, increment `frames_skipped` |
| Line with too few fields | skip that line |
| Instance with fewer than 2 visible keypoints | skip the instance (no usable extent) |
| Degenerate box (`w == 0` or `h == 0`) | skip the instance (aspect undefined) |
| Every frame skipped, `sample_count == 0` | `ValueError` saying labels were found but none were usable |
| `detail <= 0` | `ValueError` |

Never fall back to a default on bad input. A silently defaulted suggestion looks like
a measurement and is not one.

---

## 2. The control

One `vitpose_layout.addRow(...)` inserted after `posekit/gui/dialogs/training.py:700`,
inside the existing `vitpose_group`. That group's visibility is already toggled
wholesale by `_update_backend_ui` (`training.py:1182`), so the new row needs no
show/hide wiring.

Contents:
- **Input size** — two `QSpinBox`es (H and W), single-step 32, range `[64, 1024]`.
  The range deliberately exceeds the auto cap of 384: the operator may type a larger
  value deliberately; the tool just will not propose one.
- **Detail** — `QDoubleSpinBox`, range `0.25`–`4.0`, step `0.25`, default `1.0`,
  suffix `x`. Mirrors ClassKit's `_auto_size_scale_spin`
  (`classkit/gui/dialogs/training.py:848-856`).
- **Auto from dataset** — `QPushButton`. Disabled when the project has no labelled
  frames.
- **Summary label** — after a measurement, reports sample count, median aspect,
  median and p90 long side, and says so explicitly when the suggestion was clamped.
  The p90 is the number that tells the operator whether the median is representative
  or whether the dataset has a long tail of large individuals.

Behaviour: the button runs the measurement synchronously under a wait cursor and
writes the result into the two spin boxes. Hundreds of header reads take well under a
second, so a `BaseWorker` thread would add failure modes without buying responsiveness.
The chosen size persists through the dialog's existing `_apply_settings` /
`_save_settings` (`training.py:1102`, `:1365`). Defaults on first open are ViTPose's
current default geometry, so a project that ignores the button behaves exactly as today.

---

## 3. Threading the value

Small, because Slice 1 built the far end:

- `ViTPoseTrainingWorker.__init__` (`training.py:311-341`) gains an `input_size`
  parameter and stores it.
- Its `params` dict (`training.py:364-372`) gains `"input_size"`.
- `_start_vitpose_training` (`training.py:1600-1614`) passes the spin-box values as
  `[H, W]`.
- `prepare_run` (`posekit/core/vitpose_training.py:17-27`) needs no change — it copies
  `params` wholesale into `validate_run_config`.
- `RunConfig.input_size` and its validation already exist from Slice 1.

The key must be exactly `input_size`; `validate_run_config` raises on unknown keys.

---

## 4. Known inconsistency, deliberately not fixed here

The training dialog reads labels from a flat `project.labels_dir`
(`training.py:1420`, `:1562`, `:1602`), while the main window routes per-source through
`_source_map` (`posekit/gui/main_window.py:1685-1694`). For a multi-source project the
dialog already ignores the additional sources — both when counting labelled frames and
when training.

The estimator matches the dialog's flat behaviour. That is the consistent choice: the
estimate then describes exactly the data that will actually be trained on. Making the
estimator smarter than the trainer would produce a suggestion for data the run never
sees.

Fixing the routing is a real change to what multi-source projects train on, needs its
own testing, and is out of scope here. Recorded so it is not mistaken for a defect
introduced by this slice.

---

## 5. Testing

**Estimator** — real coverage, no Qt, synthetic fixtures written to `tmp_path`:
- A known geometry recovers a known suggestion (animals of a known pixel extent in
  images of a known size).
- Multi-instance frames: a file with three lines contributes three instances, proving
  the module does not inherit `load_yolo_pose_label`'s first-line-only behaviour.
- Keypoints with `v == 0` are excluded from the bbox.
- `detail` scales the suggestion; the result stays on a multiple of 32.
- A dataset of very large animals sets `clamped=True` and caps at 384.
- Determinism: two calls on the same inputs give identical results.
- Aspect orientation: a dataset of tall animals yields `H > W`, and a dataset of wide
  animals yields `W > H` — the check that catches an inverted reconstruction.
- Empty and all-unusable label sets raise `ValueError`.

**Threading** — Qt-free. Construct `ViTPoseTrainingWorker` directly with an
`input_size` and assert its `params` dict carries `input_size` unchanged, and that
`validate_run_config` accepts that dict. This exercises the whole chain without
building a dialog, which matters because this repo has known modal-dialog hangs that
prevent the full suite from completing.

**GUI** — minimal. The dialog is not constructed in tests.

---

## Risks

| Risk | Mitigation |
|---|---|
| Measurement slow on a large project | Header reads, not decodes; deterministic cap of 500 frames |
| Median unrepresentative on a long-tailed dataset | p90 reported beside it so the operator can see the tail |
| Suggestion proposes an expensive model | Capped at 384 with an explicit "clamped" report |
| Aspect reconstruction inverted | Covered by the tall-vs-wide orientation test |
| Operator assumes the suggestion is authoritative | Summary states what was measured and how many samples it came from |
