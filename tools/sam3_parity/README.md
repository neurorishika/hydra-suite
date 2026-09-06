# SAM3 spike-parity: Task 0 comparison tool

`compare_models.py` implements the plan's Task 0 Steps 1-5 (paired per-frame
comparison, significance criterion, matched-recall interpolation, AP/PR
curve, adjudication helpers) as pure, unit-tested functions, AND wires them
end to end to the live calibration harness in `_run_live_comparison`: it
loads frames + labels from a COCO json, runs `calibration.calibrate()` for
each of two labelers, recomputes per-frame extras/missed at a fixed
confidence from the cached raw candidates, builds `OperatingPoint` sweeps,
and writes `baseline.json`. All of that orchestration is exercised end to
end -- with a fake, dependency-injected labeler, no `sam3`, no GPU -- by
`test_run_live_comparison_wiring_produces_well_formed_baseline` and
`test_run_live_comparison_raises_on_no_common_frames` in
`tests/test_sam3_parity_compare.py`; the Step 1-5 pure functions have 31
further tests of their own.

The ONLY step that genuinely requires `sam3` to be importable is
`_default_labeler_factory`, which constructs a real `Sam3SemanticLabeler`
from a checkpoint via `Sam3SemanticLabeler.from_variant`. That import is
deferred inside the function body, so nothing else in this module needs
`sam3` at import time, and `main()`/`_build_arg_parser()` always use the
real factory (no CLI flag currently overrides it -- only tests inject a
fake, via `_run_live_comparison`'s `labeler_factory` parameter).

Step 6 HAS BEEN RUN. On 2026-09-05, on `courtship` (RTX 4090) under
`hydra-cuda`, against the 16 held-out validation frames (576 tiles, 805
instances) and both real published checkpoints; the resulting
`baseline.json` is committed here (commit `82327fd6`). It cannot be
reproduced on THIS machine (macOS, triton-blocked, no `sam3`).

**Read `baseline.json`'s `caveats` block and
`docs/superpowers/specs/2026-09-05-sam3-spike-parity-measurement-findings.md`
before quoting any number from it.** In short:

* **The outcome is NULL.** Paired per-frame difference +0.4375 extras/frame,
  95 % CI [-0.4375, +1.4375], sign p = 0.34 -- the pre-registered criterion
  (CI excludes zero) is not met. On AP, ours 0.530 beats the spike's 0.472.
* **One held-out frame, `f008975`, is in the SPIKE'S OWN TRAINING SET** and
  supplies most of the point estimate (+6, next largest -3). It was NOT
  dropped -- a post-hoc frame-set change is a protocol deviation. Excluding
  it: +0.0667, CI [-0.600, +0.667]; still null.
* **The matcher is biased.** `calibration.match_one_to_one`'s containment gate
  tests a vertex-mean "centroid" that falls outside 15.9 % (128/805) of ant
  outlines, vetoing near-perfect masks. Under a plain IoU >= 0.5 matcher on the
  SAME predictions recall goes 0.868 -> 0.962 (ours) and 0.852 -> 0.935
  (spike), and extras/tile 0.20 -> 0.07. `baseline.json` was deliberately NOT
  re-run under a corrected matcher: the criterion was pre-registered against
  the shipped harness.
* The two checkpoints also differ by **~19x in training data**, so this is not
  an adapter-surface comparison.

To re-run it (on a GPU box, courtship or mehek) -- note this produces a NEW
measurement, it does not amend the pre-registered one:

Use the env that has ULTRALYTICS, not the `sam3` training sidecar. Inference
goes through `ultralytics.models.sam.SAM3SemanticPredictor`
(`core/inference/semantic/sam3.py::from_variant`), which the `hydra-sam3`
sidecar does not have -- that env exists for LoRA TRAINING against Meta's
`sam3` package. On courtship the right env is `hydra-cuda` (it also needs
`scikit-learn`, pulled in by the `data.al` import chain).

```bash
conda activate hydra-cuda
KMP_DUPLICATE_LIB_OK=TRUE python tools/sam3_parity/compare_models.py \
  --checkpoint-a /path/to/ours.pt \
  --checkpoint-b ~/sam3_spike/work/checkpoints/fold_all_r16/adapters.pt \
  --frames-dir /path/to/held_out_frames \
  --coco-json /path/to/held_out_frames/_annotations.coco.json \
  --prompt "ant" \
  --reference-body-px 97 \
  --seam-margin-px 8 \
  --merge-iou 0.5 \
  --compare-confidence 0.5 \
  --target-recall 0.9 \
  --out tools/sam3_parity/baseline.json
```

`--compare-confidence` (Step 2) and `--target-recall` (Step 3) are stated
here rather than swept, per the plan's requirement to fix the significance
criterion and the matched-recall procedure before running. Pick
`--tile-fraction` from the trained/deployed SAHI configuration (omit for
full-frame, no tiling).

Use the 16-frame held-out validation split (576 tiles, 805 instances,
verified present in `_annotations.coco.json`) named in the plan. `--out`
defaults to `tools/sam3_parity/baseline.json` in this directory; commit that
file (with the exact arguments recorded inside it -- `checkpoint_a/b`,
`prompt`, `reference_body_px`, `tile_fraction`, `seam_margin_px`,
`merge_iou`, `compare_confidence`, `target_recall`, and the resolved
`frames` list) once it has been produced on real checkpoints. The committed
copy also carries a hand-added `caveats` block; the numeric keys are exactly
as the tool emitted them and must not be edited.

`_run_live_comparison`'s orchestration (load frames from COCO -> run
`calibrate()` per model -> per-frame re-threshold -> paired stats -> AP/PR
-> matched-recall -> write `baseline.json`) is fully implemented and
covered by a smoke test (see above); nothing needs to be "filled in" beyond
having `sam3` importable and real checkpoints. `main()` uses the real
`_default_labeler_factory` unconditionally -- there is no CLI flag to
substitute a fake labeler outside of tests.

Adjudicating whether extras are clutter or unlabelled ants (plan Step 5) is
NOT automated by this CLI: `extras_unique_to_a` / `unmatched_predictions`
are available as library functions for that inspection, but no `--out`
artifact currently records which polygons they flag. A human still has to
call them (or write a small script that does, against the calibration
preview data) and look at the results.
