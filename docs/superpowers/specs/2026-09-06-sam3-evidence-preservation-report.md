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
  The effective cadence is stamped into every series record and into `val_stats.json`, so a
  cadence gap can never be mistaken for a crash-restart gap.
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
- **Total fail-open contract: no filesystem operation in the retention path may raise.**
  Measurement AND deletion. Guards stay narrowly scoped to `OSError`, so a
  `KeyError`/`TypeError` from the planner still surfaces as the logic bug it is.
- `enforce_checkpoint_budget(directory)` applies the plan, deletes each `.complete.json`
  marker with its artifact, and `emit_log`s LOUDLY with the measured numbers that forced
  the prune. Each deletion is INDEPENDENT: a failure logs and continues rather than
  aborting the rest, because aborting on the first failure leaves more disk consumed with
  no compensating benefit -- a partially thinned directory is a valid state for best-effort
  housekeeping, a dead run is not. The marker is removed only once its artifact is actually
  gone, so a marker still cannot outlive what it vouches for, and only paths actually
  deleted are returned/reported.
- The old docstring's rationale (user-supplied epoch counts; disk exhaustion mid-run would
  destroy the artifact the feature exists to preserve; 200-epoch run on a full disk) is
  preserved verbatim-in-spirit in the new module-level comment. The mechanism is replaced,
  the concern is not refuted.

## Tests
- Before: **408 passed, 5 skipped** (`-k sam3`).
- After: **425 passed, 5 skipped**.
- TDD: 19 new tests written first, all 11 observed failing, then implemented. The two old
  sliding-window tests were rewritten to the new policy (keeping their marker-deletion and
  safe-on-missing-directory assertions). New coverage: budget-allows-keep-all,
  thin-middle-keep-ends, never-delete-just-written, budget-is-derived-not-hardcoded,
  markers+loud-log, missing-dir safety, JSONL append, cadence env parsing (incl. garbage),
  RNG save/restore present, loss decomposition + elapsed_s recorded, anti-correlation
  comment present, unmeasurable-budget-retains-everything, stat-failure-skips-pruning, unlink-PermissionError-skips-pruning,
  one-unlink-failure-does-not-abort-the-others, guards-stay-scoped-to-OSError.
  The unlink guard was mutation-checked the same way: reverting the loop to its unguarded
  form fails 2 tests.
- The RNG restore is pinned BEHAVIOURALLY, not by source grep: a stub model plus an
  `_evaluate_split` stub that burns the python/numpy/torch streams (both on the success and
  the raising path). Mutation-checked -- deleting any one of the three restore lines fails
  2 tests. **Known limit:** `torch.cuda.set_rng_state_all` is unexercised on this box (no
  CUDA) and remains read-verified only, until a CUDA run exercises it. `import sam3` is unavailable on macOS, so the live-model paths are
  covered by source/stub assertions, as the existing file already does.

## Filesystem-call audit on the training path

The property being asserted is *no filesystem operation reachable from
`_write_epoch_checkpoint` may raise into `run_training`* -- not "the stat calls are
guarded". Two earlier passes fixed the instance named in review rather than the property,
which is how the unguarded `unlink` survived. Every reachable call, and why each is safe:

| Call | Status |
|---|---|
| `directory.mkdir(parents=True, exist_ok=True)` | Pre-existing, deliberately fatal (see below) |
| `_write_validated_adapter_artifact`: `temporary.open`, `torch.save`, `os.fsync`, `torch.load`, `os.replace`, `os.open`/`os.fsync`/`os.close` on the dir fd, `write_completion_marker`, `temporary.unlink` in `finally` | Pre-existing, deliberately fatal, unchanged by this branch |
| `enforce_checkpoint_budget`: `directory.is_dir()` | Cannot raise -- `Path.is_dir()` returns `False` on `OSError` |
| `enforce_checkpoint_budget`: `directory.glob("epoch_*.pt")`, `path.stat().st_size` | Inside the `OSError` guard -> retain everything, log |
| `checkpoint_budget_bytes`: `directory.is_dir()` | Cannot raise (as above) |
| `checkpoint_budget_bytes`: `shutil.disk_usage`, `glob`, `path.stat()` | Inside the `OSError` guard -> return `None`, retain everything, log |
| `plan_checkpoint_retention` | Pure; no I/O at all |
| `enforce_checkpoint_budget`: `path.unlink`, `marker.unlink` | Inside per-deletion `OSError` guards -> log and continue |

The `mkdir` and artifact-write calls are left fatal ON PURPOSE and are not a regression:
they *are* the salvage write, and a run that silently swallowed their failure would believe
it has a checkpoint it does not have. Retention is different in kind -- it is housekeeping
that deletes things, so its failure mode must be "keep more than intended", never "end the
run". That asymmetry is the whole design.

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
- A reused `run_dir` would accumulate `val_series.jsonl` rows across attempts (a
  crashed-then-rerun training appends a second epoch-1 record). Launcher run dirs are
  per-run, so this is theoretical; no dedup logic was added.
- No end-to-end GPU run of the new code; the validation-series and budget paths have not
  executed against a live `sam3` model.
