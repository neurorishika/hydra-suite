# Step 0 — SAM3 evidence preservation

**Status:** implemented, tests green, committed on `feat/sam3-evidence-preservation`
(worktree `.worktrees/sam3-evidence`). Not pushed, not merged.

## What was built

### 1. Per-epoch validation series (`val_series.jsonl`)
- `_evaluate_split(...)` — extracted from `_evaluate_and_write`; returns
  `val_loss_mean`, `val_batches`, `val_terms_mean` (per-term breakdown averaged over
  batches, taken from the same `loss_dict` the terminal path already computes) and
  `elapsed_s` (the pass's own measured wall clock).
- `_record_epoch_validation(...)` — called at the end of every epoch except the last;
  appends one JSON object per line to `<run_dir>/val_series.jsonl`, and `emit_log`s the
  number so it is visible while the run is happening.
- The FINAL epoch's record is appended by the terminal `_evaluate_and_write` from the
  *same* computation that writes `val_stats.json`, so (a) no epoch is evaluated twice and
  (b) the last row of the series can never disagree with the terminal artifact.
- `val_stats.json` keeps its historical keys (`val_loss_mean`, `val_batches`, `note`) and
  gains `val_terms_mean`. Additive only; `train.py:372` only checks existence.
- **Behaviour preservation is proven, not asserted:** the mid-run pass saves and restores
  python/numpy/torch/torch.cuda RNG state and restores `model.train()`, so the next epoch
  draws exactly the numbers it would have drawn without the call. Train shuffling was
  already seeded per-epoch (`spec.seed + epoch`); the restore covers incidental global-RNG
  consumption inside transforms. Weights, steps, optimizer, scheduler and selection are
  untouched. Selection remains last-epoch.
- The anti-correlation warning ("do not wire selection onto this series") is stated at
  both places a future reader would be tempted: the `_record_epoch_validation` docstring
  and the `_evaluate_split` docstring.
- Cadence knob: `HYDRA_SAM3_VAL_EVERY` (default 1 = every epoch), read via `val_cadence()`.
  An env var rather than a `Sam3LoraParams` field, so there is no spec.json / dialog /
  params-threading ripple. The final epoch is recorded regardless of cadence.

### 2. Retention budget replaces the count cap
- `KEEP_EPOCH_CHECKPOINTS = 3` and `prune_epoch_checkpoints` are **gone**.
- `checkpoint_budget_bytes(directory)` — `CHECKPOINT_BUDGET_FRACTION_OF_FREE` (0.25, a
  policy fraction, not a measurement) × (`shutil.disk_usage(dir).free` + bytes already
  held by existing epoch checkpoints, which are reclaimable). No hardcoded byte figure.
- `plan_checkpoint_retention(paths, *, free_bytes, adapter_bytes, budget_bytes=None)` —
  pure, injectable, returns paths to delete. Adapter size is measured from the files
  themselves at the call site (`max(st_size)`), never assumed.
- Policy when the budget binds: keep the FIRST and LAST checkpoints unconditionally, then
  repeatedly drop the middle-most survivor. Never a sliding window; the earliest epochs —
  the stall evidence — are never the first casualty, and the checkpoint just written is
  never deleted.
- `enforce_checkpoint_budget(directory)` applies the plan, deletes each `.complete.json`
  marker with its artifact, and `emit_log`s LOUDLY with the measured numbers that forced
  the prune.
- The old docstring's rationale (user-supplied epoch counts; disk exhaustion mid-run would
  destroy the artifact the feature exists to preserve; 200-epoch run on a full disk) is
  preserved verbatim-in-spirit in the new module-level comment. The mechanism is replaced,
  the concern is not refuted.

## Tests
- Before: **408 passed, 5 skipped** (`-k sam3`).
- After: **417 passed, 5 skipped**.
- TDD: 11 new tests written first, all 11 observed failing, then implemented. The two old
  sliding-window tests were rewritten to the new policy (keeping their marker-deletion and
  safe-on-missing-directory assertions). New coverage: budget-allows-keep-all,
  thin-middle-keep-ends, never-delete-just-written, budget-is-derived-not-hardcoded,
  markers+loud-log, missing-dir safety, JSONL append, cadence env parsing (incl. garbage),
  RNG save/restore present, loss decomposition + elapsed_s recorded, anti-correlation
  comment present. `import sam3` is unavailable on macOS, so the live-model paths are
  covered by source/stub assertions, as the existing file already does.

## Measured per-epoch validation cost
**Not measured — blocked.** The pass is CUDA-only (`sam3` cannot be imported on this box)
and mehek is running a GPU study that I was instructed not to touch.
Derived estimate from the prior run: 10 epochs / 2h12m ≈ 13.2 min per training epoch over
588 batches (forward + backward + optimizer). Validation is forward-only over the 147-tile
valid split, so **order 1–3 min per epoch, ≈8–20% added wall clock** — non-trivial, which
is what justifies the cadence knob rather than a silent cost.
The feature is self-measuring: every record carries `elapsed_s`, so the first real run
replaces this estimate with a measurement, per epoch, without any further instrumentation.

## Not done / out of scope
- D15 (does Requirement B apply to YOLO/Ultralytics training?) and the P1/YOLO retention
  check for the same class of constant — spec follow-ups, deliberately not touched.
- D13 (per-epoch detection-quality metric and stopping on it) — blocked on D12.
- No end-to-end GPU run of the new code; the validation-series and budget paths have not
  executed against a live `sam3` model.
