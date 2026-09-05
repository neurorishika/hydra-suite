# SAM3 spike-parity: Task 0 comparison tool

`compare_models.py` implements the plan's Task 0 Steps 1-5 (paired per-frame
comparison, significance criterion, matched-recall interpolation, AP/PR
curve, adjudication helpers) as pure, unit-tested functions -- see
`tests/test_sam3_parity_compare.py`.

Step 6 (run on the 16 held-out validation frames and commit `baseline.json`)
is deferred: it requires a working `sam3` install and both published
checkpoints, neither of which is available on this (macOS, triton-blocked)
machine. Run it on a GPU box (courtship or mehek) with the `hydra-sam3`
sidecar env active:

```bash
conda activate hydra-sam3
KMP_DUPLICATE_LIB_OK=TRUE python tools/sam3_parity/compare_models.py \
  --checkpoint-a /path/to/ours.pt \
  --checkpoint-b ~/sam3_spike/work/checkpoints/fold_all_r16/adapters.pt \
  --frames-dir /path/to/held_out_frames \
  --coco-json /path/to/held_out_frames/_annotations.coco.json \
  --prompt "ant" \
  --reference-body-px 97 \
  --seam-margin-px 8 \
  --merge-iou 0.5 \
  --out tools/sam3_parity/baseline.json
```

Use the 16-frame held-out validation split (576 tiles, 805 instances,
verified present in `_annotations.coco.json`) named in the plan. `--out`
defaults to `tools/sam3_parity/baseline.json` in this directory; commit that
file (with the exact arguments used, which the tool records alongside the
results) once it has been produced on real checkpoints.

`_run_live_comparison` in `compare_models.py` currently raises
`NotImplementedError` at the one seam that requires `sam3` to be importable
(constructing each `SemanticLabeler` from its checkpoint via
`Sam3SemanticLabeler.from_variant`); fill that seam in on the GPU box, then
call the pure functions in this module the same way the module docstring
describes and the tests exercise.
