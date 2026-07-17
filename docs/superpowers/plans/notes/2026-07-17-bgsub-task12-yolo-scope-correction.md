# Correction note: bg-sub plan Task 12/13 — the two "bg-sub" call sites are YOLO-only

**Applies to:** `docs/superpowers/plans/2026-07-16-bgsub-inference-stage.md`, Task 12 Steps 2–3 and Task 13 Step 3.
**Status:** correction to plan text. Apply while executing; do not follow Task 12 Steps 2–3 as written.

## The problem

Task 12 says it "only touches the bg-sub branch" and directs replacing `create_detector` with
`BackgroundMeasurer` at `data/dataset_generation.py:421` and
`core/tracking/optimization/optimizer_workers.py:372`.

**Both sites are YOLO-only. Neither has a bg-sub branch.**

- `dataset_generation.py:413-428` — `_init_yolo_detector` early-returns `None` unless
  `DETECTION_METHOD == "yolo_obb"` (`:417-418`). The string `"background_subtraction"` at `:417` is
  only the `.get()` default inside that guard. Every downstream consumer is the YOLO detector:
  `:531` `detect_objects_batched`, `:552` `detect_objects`, `:579` `hasattr(detector,
  "detect_objects_batched")`, and `:927-931` `detector.use_tensorrt` / `detector.tensorrt_batch_size`
  (both `YOLOOBBDetector` attributes). Grep confirms zero `ObjectDetector` references in the file.
- `optimizer_workers.py:330-338` — `DetectionCacheBuilderWorker`'s docstring: "runs YOLO detection on
  a frame range and writes a DetectionCache." Its `create_detector` at `:372` feeds
  `detector.detect_objects_batched(...)` at `:441`.

Following Task 12 as written swaps a YOLO detector for a `BackgroundMeasurer`, which has no
`detect_objects_batched`, no `use_tensorrt`, and no `tensorrt_batch_size` — breaking `:441` and
`:531`/`:927-931` at runtime. The instruction to "update `detect_objects` unpacking to the 4-tuple"
is also wrong here: `:552`'s 5-tuple is `YOLOOBBDetector.detect_objects`, which is untouched by this
plan. The 4-tuple is `BackgroundMeasurer`'s new contract only.

This then blocks Task 13 Step 3, whose gate is `grep -rn "create_detector" src/ tests/` → no output.
Task 13 Step 1 deletes `factory.py` (the only definition of `create_detector`), but nothing has given
these two YOLO sites a replacement, so the gate cannot pass. That same grep is the start gate for
`2026-07-16-legacy-batching-vestige-removal.md` (`:16-18`), so this stalls the whole chain.

## The fix

Construct `YOLOOBBDetector` directly at both sites. `create_detector` was only a dispatcher
(`factory.py:19-29`: `yolo_obb` → `YOLOOBBDetector`, else → `ObjectDetector`); at a site where the
method is statically known to be `yolo_obb`, calling the class directly is behavior-identical. This
is consistent with Task 13 Step 4's stated endpoint, which keeps `yolo_detector.py` in
`core/detectors/`.

**Task 12 Step 2 — `dataset_generation.py`.** Replace the step text with: in `_init_yolo_detector`,
change the import at `:415` from `from ..core.detectors import create_detector` to
`from ..core.detectors.yolo_detector import YOLOOBBDetector`, and `:421` from
`create_detector(params)` to `YOLOOBBDetector(params)`. Keep the `DETECTION_METHOD` guard as-is.
**Do not touch `:531`, `:552`, `:579`, or `:927-931`** — they are YOLO paths and their tuple contracts
do not change.

**Task 12 Step 3 — `optimizer_workers.py`.** Replace the step text with: change the function-local
import at `:367` to `from hydra_suite.core.detectors.yolo_detector import YOLOOBBDetector` and `:372`
to `YOLOOBBDetector(self.params)`. Leave `:441` alone, as the plan already says.

**Task 13 Step 3 — amend the gate.** `grep -rn "create_detector\|bg_detector\|bg_optimizer" src/ tests/`
expecting no output is still correct after the above. But add, to keep the endpoint honest:

```bash
grep -rn "YOLOOBBDetector" src/
```
Expected: `core/detectors/__init__.py`, `core/detectors/yolo_detector.py`,
`data/dataset_generation.py`, `core/tracking/optimization/optimizer_workers.py`. These last two are
**known, accepted follow-on debt** — they are the call sites that need the one-shot detect API that
`core/inference` does not yet expose. Do not try to migrate them to `InferenceRunner` in this plan:
the runner requires an `InferenceConfig` + video + cache and eagerly loads every model, which fits
neither a per-frame dimension-extraction loop nor a cache-builder. That migration belongs to the
`core/detectors` retirement project, together with relocating `_direct_obb_runtime.py`.

## Scope guard

This correction does not widen the plan. It keeps two YOLO call sites working exactly as they do
today while `create_detector` and the bg-sub half of `core/detectors` are removed as planned. The
`__init__.py` edit in Task 13 Step 1 still drops `create_detector` from imports and `__all__`;
`YOLOOBBDetector` stays exported.
