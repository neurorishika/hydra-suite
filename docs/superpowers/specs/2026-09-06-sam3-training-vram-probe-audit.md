# SAM3 training: GUI/CLI parity, OOM safety, and auto batch sizing — audit

**Date:** 2026-09-06. **Repo state audited:** `main` @ `65f85669`, plus the unmerged
branch `codex/measured-auto-batch` (7 commits ahead, owned by another session — audited
read-only via `git show`/`git diff main...`, never checked out).

**Method note.** Everything below is either VERIFIED (read in the tree at the cited
file:line, or read out of a git object on the named branch) or INFERRED (reasoning from
verified code). Nothing was executed: `import sam3` is unavailable on this host and the
CUDA box was off-limits. Runtime numbers for the live run come from the coordinator and
are labelled as REPORTED, not verified by me.

---

## Part 1 — GUI vs CLI parity

### Verdict

**Structurally parity-safe.** Both entry points converge on one builder: the GUI
constructs a `DetectTrainingPlan`, calls `plan.validate()` and `plan.role_entries()`
(`src/hydra_suite/detectkit/gui/dialogs/training_dialog.py:2956-2993`), which is exactly
what `detectkit train` does (`src/hydra_suite/detectkit/cli.py:215`, `:273`). This is
**not** the `build_engine_params` failure class — there is no parallel field mapping on
the SAM3 path. Four small divergences remain, listed below; only two are worth acting on.

### Why the shape is safe (VERIFIED)

- `Sam3TrainingPanel._build_ui` ends with `self.set_params(Sam3LoraParams())`
  (`sam3_training_panel.py:492`). Every widget's initial value is therefore *seeded from
  the dataclass default*, so a default can never drift between the panel and the
  contract — the classic divergence is structurally impossible here.
- `Sam3TrainingPanel.params()` (`sam3_training_panel.py:496-560`) constructs
  `Sam3LoraParams` with **all 31 fields** explicitly, including `adapt_scoring_head`
  (round-tripped through a plain attribute rather than a widget, deliberately —
  see the comment at `:539-546`).
- The CLI validator coerces the same 31 fields
  (`detectkit/config/training.py:495-549`: 9 bool, 8 int, 9 number, 4 string, plus
  `negative_prompts`), and `_construct_dataclass` (`training.py:~150`) **rejects unknown
  keys**, so a typo'd `sam3.*` key is a hard error, not a silent no-op.
- `label_quality_acknowledged`: gated on **both** sides — GUI checkbox
  (`sam3_training_panel.py:485`, read at `:557`) and CLI
  `plan.validate()` (`config/training.py:644-647`). No asymmetry.
- SAM3 `imgsz`: 640 default on both sides (`config/training.py:290` for the CLI;
  `training_dialog.py:_imgsz_for_role` falls through to `return 640` for
  `SEMANTIC_SAM3`, which has no spin box).

### Field table (all 31 `Sam3LoraParams` fields)

| Field | GUI widget | CLI plan key | Default source | Divergence |
|---|---|---|---|---|
| `prompt` | `prompt_edit` | `sam3.prompt` | dataclass | — |
| `negative_prompts` | `negative_prompts_edit` | `sam3.negative_prompts` | dataclass | — |
| `rank`, `alpha`, `dropout` | 3 spins | ✔ | dataclass | — |
| `lr`, `epochs`, `batch`, `grad_accum` | 4 spins | ✔ | dataclass | **`batch: -1` — see below** |
| `mixed_precision` | combo (`bf16` only) | ✔ (any string) | dataclass | CLI accepts `fp32`; GUI cannot express it. Deliberate (legacy-plan loading, `config/training.py:696`). |
| `num_negatives` | spin | ✔ | dataclass | — |
| `host_reserve_gb`, `host_reserve_fraction`, `cuda_safety_fraction`, `host_limit_headroom_fraction`, `watchdog_poll_seconds` | 5 spins ("Resource safety") | ✔ | dataclass | — |
| `adapt_vision_encoder` … `adapt_mask_decoder` (6) | 6 checkboxes | ✔ | dataclass | — |
| `adapt_scoring_head` | **no widget**, round-tripped | ✔ | dataclass | Headless-only by design and documented at the construction site. Acceptable. |
| `geometry_mode`, `object_tile_fraction`, `slice_width`, `slice_height`, `tile_overlap`, `keep_empty_tiles` | 6 widgets | ✔ | dataclass | — |
| `label_quality_acknowledged` | checkbox | ✔ | dataclass | — |
| `env_name` | `env_edit` | ✔ | **NOT the dataclass default** | **Divergence D1, below** |

### Divergences

- **D1 (real, small). `env_name` is never empty from the GUI.**
  `set_params` writes `p.env_name or DEFAULT_SAM3_ENV` (`sam3_training_panel.py:559`), so
  a GUI run always records a concrete env name. The dataclass default is `""`, which
  `resolve_sam3_env` resolves *at run time*, honouring `HYDRA_SAM3_ENV`
  (`training/sam3_lora/env.py`). Consequence: on a box where `HYDRA_SAM3_ENV` points at a
  non-default env, a CLI plan honours it and the GUI silently overrides it with the
  compiled-in default. INFERRED consequence; the code paths are VERIFIED.
- **D2 (minor, corrected on inspection). `device`.** GUI:
  `combo_device.currentText().strip() or "cpu"` (`training_dialog.py:2981`); CLI default
  `"auto"`. VERIFIED `_build_device_options` (`training_dialog.py:2489-2507`) enumerates
  `cuda`/`cuda:N`/`mps`/`cpu` and always appends `cpu`, so on a CUDA box the combo's first
  item is `cuda:0` (or `cuda` with >1 GPU) and the `or "cpu"` branch is unreachable. The
  GUI therefore never emits `"auto"` — it emits a concrete device, which
  `preflight._probe_cuda_device` (`preflight.py:521`) handles identically. The only
  residual is that a GUI user on a CPU-only box can start a SAM3 run that preflight then
  refuses, where the CLI's `"auto"` produces the same refusal by a different route. Not
  worth fixing; recorded so it is not re-discovered as a bug.
- **D3 (intentional, already announced). `publish_policy`.** GUI hard-codes
  `auto_import=True` (`training_dialog.py:2987`); the CLI defaults to `False` and now
  prints a banner saying so (`detectkit/cli.py:221-233`).
- **D4 (dead code, remove).** `TrainingDialog._sam3_spec_for`
  (`training_dialog.py:2571-2593`) is a *second* SAM3 spec builder that bypasses
  `plan.validate()` and `build_role_entries` and hard-codes
  `TrainingHyperParams(epochs=params.epochs)` with no `imgsz`, `device` or
  `publish_policy`. VERIFIED: its only references are in
  `tests/test_sam3_dialog_wiring.py:96,112` — no production caller. It is precisely the
  divergent-second-builder the convention exists to prevent, currently kept alive by a
  test. Delete both.

---

## Part 2 — OOM safety

### The guard chain (VERIFIED)

1. **Plan validation** — `DetectTrainingPlan.validate()` (`config/training.py:637-712`)
   bounds every safety knob before anything allocates.
2. **Admission preflight** — `preflight.assess_preflight` (`preflight.py:~880`) composes
   per-phase `PhaseEstimate`s (`model_load` / `training` / `validation` / `publish`) and
   checks them against a `ResourceObservation` derived from `nvidia-smi`
   (`preflight._probe_cuda_device`, `:521` — deliberately torch-free).
   - Device side: `training_device_peak = _MEASURED_BF16_DEVICE_PEAK_BYTES *
     precision_multiplier + max(0, batch-1) * _EXTRA_BATCH_DEVICE_BYTES + …`
     (`preflight.py:758-764`), with `_MEASURED_BF16_DEVICE_PEAK_BYTES = 12 GiB`
     (`preflight.py:94`) and `_EXTRA_BATCH_DEVICE_BYTES = 18 GiB` (`preflight.py:101`).
   - `accelerator_safety_fraction = 0.85` (`runtime/resource_budget.py:156`, applied at
     `:335`), capped in preflight at `_MAXIMUM_CUDA_SAFETY_FRACTION = 0.90`
     (`preflight.py:126`).
3. **Host containment** — `containment_host_limits` (`preflight.py:648-658`) turns the
   estimated host peak into a soft/hard pair via `host_limit_headroom_fraction` (1.25),
   handed to `runtime/resource_limits.py`, which launches the sidecar under
   `systemd-run … --property=MemoryMax=<hard>` (`resource_limits.py:187,194`) when
   available (`:105`).
4. **Immutability across the run** — `train.py:226-233` refuses to let a live re-assessment
   raise the hard limit above the one admission granted.
5. **Watchdog** — parent-side polling at `params.watchdog_poll_seconds`
   (`sam3_lora/train.py:202`, `:281`, `:313`), supervising the contained sidecar.

### The 128 GB host-RAM crash: fixed by a GUARD, not by an accident (VERIFIED)

The dataloader is structurally streaming, not incidentally small:

- `build_descriptors` (`sam3_lora/dataloader.py:158-200`) returns only `TileDescriptor`s —
  polygons and paths, explicitly "no OpenCV/PIL images, float tensors, dense masks, or
  collated batches" (`:161-166`). Memory scales with annotation metadata alone.
- `collate_epoch_batches` (`:310-322`) shuffles **indices**, then delegates to
  `collate_batches` (`:285-307`), a generator that decodes one tile at a time
  (`load_datapoints` → `cv2.imread`, `:203-220`), yields one batch, and `del`s it.
- The consumer honours the laziness: `cli.py:680` iterates `collate_epoch_batches`
  directly and `cli.py:805-810` iterates `collate_batches` for validation. There is no
  `list(...)` over either. VERIFIED by grep.

So the epoch can no longer be materialised whatever the settings are. On top of that the
`MemoryMax` cgroup cap is a genuine second guard: it converts any regression here into a
kill with an attributable exit code instead of a machine-wide OOM.

### Is `_MEASURED_BF16_DEVICE_PEAK_BYTES = 12 GiB` still defensible?

**No.** Honest arithmetic against the live run (peaks REPORTED by the coordinator, same
metric `torch.cuda.max_memory_reserved`, never reset, so each is a true running max):

| Source | Peak | Under-measures the next by |
|---|---|---|
| Inherited constant, never measured | 29.00 GiB | (a 3.7× over-estimate, correctly retired) |
| `tools/sam3/measure_bf16_peak.py`, N=5 **and** N=30 | 7.72 GiB | 41% low vs final |
| Live run, step 70 | 10.22 GiB | |
| Live run, step 310 | 11.59 GiB | |
| Live run, step 320 | **12.99 GiB** | |

- vs the base constant (12 GiB): **the observation exceeds it.**
- vs preflight's *composed* `training.accelerator_peak_bytes` = 13,275,856,896 B =
  **12.364 GiB**: the observation exceeds the whole admission estimate by **5.1%**.
- Headroom is therefore **negative**, not the ~55% the comment at `preflight.py:80-81`
  still claims. That comment is now factually wrong in the tree and should be corrected
  regardless of what else is done.

**No OOM occurred and none is predicted on this card.** Admission demands
`12.364 / 0.85 = 14.55 GiB` free, and 12.99 still fits under that. But the run survived on
slack whose stated purpose is absorbing allocator noise (2.18 GiB intended, 1.56 GiB
consumed), not covering a central estimate that is 5% too low. That is a correct outcome
produced by the wrong mechanism: it is luck with a documented margin, and the next
configuration — a denser dataset, a bigger `rank`, `batch > 1`, or a smaller card — has no
reason to be lucky.

**What the constant should be derived from.** Not a constant at all. It should be the
`max` over a *completed* run, recorded per `(GPU model, adapter surface, dataset density,
sidecar env hash)` in the profile store the branch already builds
(`runtime/memory_profiles.py` + `sam3_lora/autobatch.py`'s fingerprint), with the source
constant demoted to a coarse **floor** used only when the store has no matching record —
which is exactly the design Task 4 of the auto-batch plan describes, and exactly the task
that has not landed. Until then, the interim honest value is
`ceil(observed_full_run_max × inflation)`; on the only evidence available that is
`12.99 × 1.25 ≈ 16 GiB`, and it should carry a comment saying it is a placeholder for a
store lookup.

### The probe's stopping rule

**What is wrong with the current rule.** `tools/sam3/measure_bf16_peak.py` runs the real
training entry point and aborts after `--optimizer-steps` (default 5, `:44`, enforced at
`:102-108`). Its own docstring asserts the premise that has now been refuted: "peak VRAM
is reached early and then stays flat" (`:4-6`). Because the production loop **shuffles**
(`collate_epoch_batches`, `dataloader.py:319-320`), N fixed steps is a random sample of N
tiles from the head of one permutation. N=5 and N=30 agreeing was two samples from the
same easy prefix, not a plateau.

**Why no flatness rule can be recommended.** The trace was flat at 10.22 GiB for 230
consecutive steps *across an epoch boundary*, then rose twice within 10 steps. Any
"stable for K steps" or "N steps after the last new maximum" rule fails at every plausible
K. Time-based rules are dead.

**Recommended rule — content-derived, plus a calibration factor, plus a live tripwire.**

1. **Probe the corpus's worst tiles deliberately, not a shuffled stream.** The dataset
   builder already knows per-tile instance counts — the branch even records
   `max_active_instances_per_tile` and a new `p95_active_instances_per_tile`
   (`preflight.py`, added on `codex/measured-auto-batch`). Order descending and measure
   the top tiles. Sampling more of a shuffled stream only improves the odds of hitting the
   tail; it never bounds it.
   *Status: the landed probe already does this* — `cli._densest_first`
   (branch `cli.py:865-874`) sorts by `len(d.instances)` descending and
   `run_probe_measurement` (`:888-1000`) steps on that order.
2. **Rank instance-count density by mask work, not instance count alone.** `len(instances)`
   is a proxy; the allocation driver is (instances × tile pixels), and preflight already
   uses `_MASK_DEVICE_BYTES_PER_PIXEL` for exactly that. Sort by
   `sum(polygon_bbox_area)` or `instances × tile_area`, and take the top-`max(8, batch×4)`
   tiles rather than the top `batch`.
3. **Calibrate probe→run, and publish the factor.** Even densest-first, the landed probe
   reported **7.34 GiB at batch 1** (branch `contracts.py:218-228`, measured on mehek
   2026-09-06) against the live run's **12.99 GiB** — a **1.77× under-measurement of the
   same nominal configuration**. Until that gap is explained, any probe number must be
   multiplied by a stored, dated inflation factor (≥1.8 on current evidence) *before*
   `select_batch` sees it, on top of the existing `MEASURED_SAFETY_FRACTION = 0.8`.
4. **Explain the gap before trusting the probe** — one cheap diagnostic decides the fix.
   The live run's `max_memory_reserved` is 12.99; compare it against
   `max_memory_allocated` on the same run. If **allocated** is close to the probe's
   figure and only **reserved** grew, the driver is the caching allocator's fragmentation
   ratchet over hundreds of steps — which no content-derived probe on a virgin allocator
   can ever see, and the right fixes are `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`
   plus a fragmentation multiplier on the probe figure. If **allocated** also grew to
   ~12.99, the driver really is content and the density metric in step 2 is wrong (the
   probe is not finding the true worst tile). This is the single highest-information
   measurement available and it costs one line in the training loop.
   I could NOT run it. Flagged as unverified.
5. **Add a live tripwire regardless.** Every K optimizer steps, read
   `torch.cuda.max_memory_reserved()` and compare it to the admitted
   `training.accelerator_peak_bytes`. On exceedance, log loudly, record the new max to the
   profile store, and (optionally) stop at the next safe boundary. This converts the
   residual risk from "OOM at hour three with nothing learned" into "a diagnosis and a
   better number in the store", which is what makes the estimate converge instead of
   being re-guessed. One-line-ish, and it is the cheapest item in this whole report.
6. **Make the offline tool honest.** `measure_bf16_peak.py` should default to a full run
   (no `--optimizer-steps`), record the running max continuously, and label any truncated
   figure `"lower_bound": true` in the store record so nothing downstream can mistake it
   for a peak. Its docstring's flatness claim must go.

---

## Part 3 — "Efficient max batch sizing"

Plan: `docs/superpowers/plans/2026-09-04-measured-auto-batch-sizing.md` (revision 4,
user-authored, authoritative). Implementation branch: `codex/measured-auto-batch`,
**7 commits, unmerged**, +3,215/−120 across 13 files.

### Landed (on the branch, not on `main`)

- **Task 1 — batch-curve fit and conservative selection.** `cb95363f` + `56b18369`.
  `runtime/memory_profiles.py`: `fit_batch_curve`, `select_batch`
  (branch `:249-304`), `merge_records` (`:319`), `profile_store_path` (`:307`),
  `MEASURED_SAFETY_FRACTION = 0.8` (`:30`). `select_batch`'s requirement is
  `max(fitted, every observed peak at batch ≤ n)` and it never exceeds the largest
  observed rung — a sound envelope *given sound inputs*.
- **Task 2 — SAM3 workload fingerprint.** `25f66e2f` + `e3dc8a5d`.
  New `training/sam3_lora/autobatch.py` (702 lines), sidecar-env package hash without
  importing the env; `p95_active_instances_per_tile` added to the dataset profile.
- **Task 3 — parent-orchestrated probe phase.** `a9862605` + `d985a78a` + `b136322e`.
  `PROBE_CANDIDATES = (1,2,4,8)`, `MAX_AUTO_BATCH = 8`, `PROBE_STEPS = 2`
  (branch `autobatch.py:428-433`); a fresh contained sidecar per candidate
  (`train.py:231` `_run_probe_candidate`); ladder-termination taxonomy
  (`complete`/`oom`/`refused`/`host_limit`, only the last treated as transient);
  `cli.py --probe --probe-batch N` (`:1009-1034`) and `run_probe_measurement`
  (`:888-1000`); `assess_probe_preflight` with a `probe_floor=True` estimate that
  deliberately *bypasses* `_MEASURED_BF16_DEVICE_PEAK_BYTES` so the probe is not gated by
  the constant it exists to replace; `batch: -1` accepted in `Sam3LoraParams` and the plan
  validator (branch `config/training.py:649-655`).

### Remaining

- **Task 4 — the measured preflight formula.** NOT landed. VERIFIED: the branch's
  `preflight.py` contains no reference to `MemoryProfileStore` / `memory_profiles`; the
  training-path estimate still uses the 12 GiB constant + 18 GiB/extra-batch
  (`preflight.py:758-764`). Given Part 2, this is now the most consequential remaining
  task, not a polish item.
- **Task 5 — YOLO pre-launch batch resolution.** NOT landed (no `jobs/` or Ultralytics
  file touched on the branch). Note `TrainingHyperParams.batch = -1` is already *accepted*
  by the plan validator (`config/training.py:734`) and the GUI already emits it
  (`training_dialog.py:2947`) — so today `-1` is handed straight to Ultralytics'
  own autobatch, which is the inherited-analytic path the plan wants replaced.
- **Task 6 — expose SAM3 auto batch in the panel.** NOT landed;
  `sam3_training_panel.py` is untouched on the branch. Consequence: **`batch: -1` for
  SAM3 is CLI-only.** `batch_spin.setRange(1, 256)` (`sam3_training_panel.py:392`) cannot
  express `-1`, and `set_params` would silently clamp a loaded `-1` to 1 — a *new*
  GUI/CLI divergence introduced by the branch, and a silent-clamp round-trip bug on top.
- **Verification block** — GPU acceptance on mehek, `make lint-moderate`,
  `make docs-check`, final branch-diff review, merge, plan `**Status:**` update. All open.

### CRITICAL — does the landed auto-batch search interact correctly with Part 2?

**No. It would have been fooled by this exact trace.**

- The candidate measurement is `PROBE_STEPS = 2` optimizer steps
  (branch `autobatch.py:433`, consumed at `cli.py:947-963`). Two steps is chosen to
  materialise Adam's lazy `exp_avg`/`exp_avg_sq`, which is a real and correct reason — but
  it is a *floor* argument, not a peak argument.
- The probe is fed densest-first (`cli._densest_first`), which is the right instinct and
  is genuinely better than the shuffled N-step tool. **It still under-measured.** The
  branch's own recorded numbers (`contracts.py:218-228`): batch 1 → **7.34 GiB**,
  batch 2 → 10.16, batch 4 → 13.98. The live batch-1 run reached **12.99 GiB**. Same
  card, same env, same adapter surface, same tile size, same metric. **1.77×.**
  (Caveat, UNVERIFIED: I could not confirm the probe's dataset is byte-identical to the
  live run's. If it is not, that is itself the finding — the fingerprint is supposed to
  make the two records non-interchangeable, and confirming or refuting it is one `diff`.)
- Propagating that error through `select_batch`: with records (1, 7.34), (2, 10.16),
  (4, 13.98) the fit is ≈ `5.1 + 2.22·n` GiB. On a 24 GB card with ~23 GiB free the
  budget is `0.8 × 23 = 18.4` GiB, so batch **4** is selected (requirement 13.98).
  Inflated by the observed 1.77× the true requirement at batch 4 is ≈ 24.7 GiB —
  **an OOM, hours in.** On a 47 GB card it selects 8 against a true requirement near
  40 GiB, which clears only because the card is large. The 0.8 safety fraction covers a
  25% error; the observed error is 77%.
- Aggravating factor: the error is *multiplicative in batch*, so the absolute miss grows
  with the batch the search is trying to justify. An auto-batch feature whose measurement
  error scales with its own output is worse than no auto-batch, because it converts a
  conservative default (`batch: 1`) into a confident wrong answer.
- The mitigations already on the branch are real but insufficient: fail-closed when no
  candidate survives (`train.py:517-524`), per-candidate containment so a probe OOM is a
  *measurement*, and `select_batch` never extrapolating past the largest observed rung.
  All of these protect against a probe that OOMs. None protects against a probe that
  *succeeds with a number that is too small* — which is the actual failure mode.

**Recommendation to the owning session:** do not merge the branch until either
(a) `PROBE_STEPS` is replaced by the content-derived + calibrated rule in Part 2 (items 2-4),
or (b) a stored, dated probe→run inflation factor is applied inside `resolve_batch`
before `select_batch`, with `MAX_AUTO_BATCH` temporarily reduced until a full-run record
exists for at least one fingerprint. Item (b) is small and unblocks the merge; item (a) is
the durable fix. Either way, Part 2 item 5 (the live tripwire) should ship with it, because
it is what makes a wrong selection visible instead of fatal.

---

## Part 4 — Ranked gaps to "functional SAM3 training via GUI and CLI, no OOM, efficient max batch"

| # | Gap | Size | Owned by other session? |
|---|---|---|---|
| 1 | Auto-batch probe under-measures by 1.77× (`PROBE_STEPS = 2`); selection will pick a batch that OOMs later, and the error scales with batch. Blocks merge of `codex/measured-auto-batch`. | substantial | Yes — branch is theirs; this finding is not yet reflected in it |
| 2 | `_MEASURED_BF16_DEVICE_PEAK_BYTES = 12 GiB` and the composed 12.364 GiB estimate are **below** a live observation of 12.99 GiB; the run survived only on the 0.85 noise margin. Raise the constant now (interim ≈16 GiB) and fix the false "~55% headroom" comment at `preflight.py:80-81`. | one-line (constant) + small (comment/justification) | No — free to take, but coordinate: Task 4 will supersede it |
| 3 | Task 4 — preflight must read the measured profile store instead of the constant. This is what makes #2 stop recurring. | substantial | Plan-owned, not started by them |
| 4 | Diagnose reserved-vs-allocated on the live run to decide whether the gap is allocator fragmentation or content; add the live `max_memory_reserved` tripwire either way. | one-line (diagnostic) + small (tripwire) | No |
| 5 | Task 6 — SAM3 `batch: -1` is CLI-only; the panel's `setRange(1, 256)` cannot express it and silently clamps a loaded `-1` to 1. A *new* GUI/CLI divergence and a round-trip bug the branch introduces. | small | Plan-owned, not started |
| 6 | Task 5 — YOLO `batch: -1` still routes to Ultralytics' inherited autobatch rather than a measured pre-launch resolution. | substantial | Plan-owned, not started |
| 7 | `measure_bf16_peak.py` defaults to 5 steps and its docstring asserts the refuted flatness premise; any figure it writes to the store is an unlabelled lower bound. | small | No |
| 8 | Dead second SAM3 spec builder `TrainingDialog._sam3_spec_for` + the two tests keeping it alive. | small | No |
| 9 | D1 — GUI always pins `env_name`, overriding `HYDRA_SAM3_ENV`. (D2, the `device` default, was checked and is a non-issue — see Part 1.) | one-line | No |

**Bottom line against the user's goal.** GUI/CLI parity is essentially *already there* and
is the strongest part of this subsystem — gaps 5, 8 and 9 (D1 only) are the only parity work left and
they are all small. "No OOM errors" is *not* there: the admission estimate is currently
below a measured peak, and it has been below every successive measurement this project has
taken. "Efficient max batch sizing" is ~60% built on an unmerged branch, and the built part
would ship a confidently wrong number.

---

## Could not verify

- Nothing was executed; `import sam3` is unavailable on macOS and the CUDA box was
  off-limits by instruction. All GiB figures for live runs and for the branch's probe are
  read from source comments or supplied by the coordinator.
- Whether the branch's probe measurement (7.34 GiB @ batch 1) and the live run
  (12.99 GiB @ batch 1) used the same prepared dataset. This is the pivotal control for
  the Critical finding in Part 3 and should be checked first.
- Whether `systemd-run` is actually available on mehek (`resource_limits.py:105` degrades
  silently if not), i.e. whether the host `MemoryMax` cap is live or a no-op there.
