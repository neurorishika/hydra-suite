# Measured auto batch sizing implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development
> (recommended) or superpowers:executing-plans to implement this plan task-by-task.
> Steps use checkbox (`- [ ]`) syntax for tracking.

**Status:** revision 4, 2026-09-04. Revision 2 answered the first
implementation-readiness audit (framework reuse, parent-side resolution, GUI,
branch convention, dead-task removal, merge workflow). This revision answers the
second audit: probe admission policy, workload fingerprint, cache-write
ordering, a schema-legal provenance location, an exact measured preflight
formula, and a YOLO pre-launch resolution phase. Revision 4 closes the
follow-ons those answers exposed: computing the sidecar-env half of the
fingerprint without importing it, headroom and extrapolation rules on the
measured preflight path, a non-lossy file channel for the YOLO child's result,
device observation ordering, and how a probe child learns its candidate.

**Goal:** Replace inherited/analytic batch-memory constants with numbers this
codebase measures on the machine it is actually running on, and expose that
measurement as `batch: -1` auto sizing for SAM3 and YOLO.

**Architecture:** Reuse the repository's existing measurement framework
(`hydra_suite.runtime.memory_profiles`) — do **not** build a second one. Both
training roles get the same shape: a **contained pre-launch resolution phase**
owned by the parent, which produces one positive integer plus its provenance,
writes both into the run's artifacts, and only then builds immutable limits and
launches training. No training child ever sees `-1`.

**Tech Stack:** PyTorch CUDA memory stats, Ultralytics `autobatch`, existing
`hydra_suite.runtime.memory_profiles` / `resource_budget` /
`training.sam3_lora.preflight` / `process_supervisor`.

**Spec:** No separate design doc. The motivating evidence is in
`docs/superpowers/specs/2026-09-04-sam3-spike-vs-build-consolidation.md`
(inherited-constant class of bug) and commit `7a95cf25`, which replaced one
such constant with a measurement.

---

## Global Constraints

- **Never bake a measurement into a source constant.** A measurement is valid
  for one (GPU model, workload) pair. `_MEASURED_BF16_DEVICE_PEAK_BYTES = 29 GiB`
  was inherited from a different machine and configuration, was 3.7x the real
  figure, and silently excluded every card below ~32 GiB. Key measurements on a
  workload fingerprint; treat any in-source constant as a fallback only.
- **A measurement gate must never be guarded by the estimate it replaces.**
  Admission for the *probe* uses a probe-specific policy, not the fallback
  envelope the probe exists to correct.
- **Do not create a parallel measurement framework.** `memory_profiles.py`
  already defines `ProfileIdentity`, `PressureSettings`, `MemoryMeasurement`,
  `ProbePlan`, `recommend_batch_size` and the atomic, schema-versioned
  `MemoryProfileStore`. A second type named `MemoryMeasurement` would fragment
  cache invalidation and telemetry. Extend that module.
- **Measure `max_memory_reserved`, not `max_memory_allocated`.** The caching
  allocator does not return blocks to the driver between steps, so *reserved*
  is what the device must actually have free. (`MemoryMeasurement` carries
  both fields; populate both, select on reserved.)
- **A probe must measure the peak it certifies.** Forward + backward is not the
  peak: lazy Adam state and the first `optimizer.step()` are part of it. Probe
  through a real optimizer step, on the densest available tiles.
- **A probe must never leave the process in a worse state than it found it.**
  Satisfied structurally: every probe is its own contained child process, so
  there is no parent state to restore and no probe optimizer state to leak into
  training.
- **Auto sizing proposes; it does not silently change training dynamics.**
  Batch size changes gradient noise and therefore the trained model. Every
  auto-selected value must be logged prominently, written into the run's
  artifacts alongside its provenance, and never changed mid-run.
- **Fail closed.** If nothing fits — not even batch 1 — refuse the run. Never
  cache a configuration that OOMed at batch 1, and never fall back to "1" as if
  it had succeeded.
- **Preflight stays offline and millisecond-fast.** It may READ the profile
  store; it must not run a probe or a network call.
- **Cancellation is not an exception path to skip.** A probe phase must honour
  `should_cancel`, tear its child down through the same supervisor machinery as
  training, and on cancellation merge nothing and launch nothing.
- **GPU etiquette:** `courtship.taild08eb9.ts.net` may be running SAM3
  training (unit `sam3-4090`) and `mehek.taild08eb9.ts.net` is shared. Check
  `systemctl --user is-active sam3-4090` and `nvidia-smi` before any GPU work,
  never kill a process that is not yours, and prefer short probes.
- **Isolation:** work in a git worktree branched from local HEAD
  (`git worktree add .worktrees/auto-batch -b codex/measured-auto-batch HEAD`).
  The `codex/` branch prefix is this repo's convention for plan-driven work
  (see `docs/superpowers/plans/done/2026-09-03-detectkit-oom-hardening-agent-instructions.md`).
  Never `git stash`, never `git clean -fdx`, never `rm -rf`.
- Run tests with `conda activate hydra-mps` and `KMP_DUPLICATE_LIB_OK=TRUE`.
  Most of this plan is testable on CPU with fakes; GPU work is called out
  explicitly per task.

---

## File Structure

- **Modify** `src/hydra_suite/runtime/memory_profiles.py` — batch-curve fit,
  conservative selection, canonical store location, record replacement.
- **Create** `src/hydra_suite/training/sam3_lora/autobatch.py` — SAM3 workload
  fingerprint, candidate ladder, selection, and the probe entry point invoked in
  the sidecar. May import `torch`, never Qt.
- **Modify** `src/hydra_suite/training/sam3_lora/train.py` — parent-side probe
  phase between probe admission and the final preflight.
- **Modify** `src/hydra_suite/training/sam3_lora/cli.py` — a `--probe` mode; the
  training path is unchanged and keeps reading a positive `params.batch`.
- **Modify** `src/hydra_suite/training/sam3_lora/preflight.py` — probe admission
  policy, and a measured device-peak formula that replaces the estimate stack.
- **Modify** `src/hydra_suite/training/contracts.py` and
  `src/hydra_suite/detectkit/config/training.py` — allow `sam3.batch = -1`.
- **Create** `src/hydra_suite/training/yolo_autobatch.py` — YOLO pre-launch
  resolution phase (a short contained child that reports a resolved batch).
- **Modify** `src/hydra_suite/training/ultralytics_supervisor.py` — consume the
  resolved batch; stop clamping `-1` silently.
- **Modify** `src/hydra_suite/detectkit/gui/panels/sam3_training_panel.py` —
  expose auto batch.
- **Tests**: `tests/test_memory_profiles.py` (extend),
  `tests/test_sam3_autobatch.py`, `tests/test_sam3_train_probe_phase.py`,
  `tests/test_yolo_autobatch.py`, `tests/test_detectkit_training_cli.py`, plus
  additions to `tests/test_sam3_preflight.py` and
  `tests/test_ultralytics_supervisor.py`.

---

## Task 1: Batch-curve fit, selection, and record replacement

**Files:**
- Modify: `src/hydra_suite/runtime/memory_profiles.py`
- Test: `tests/test_memory_profiles.py`

**Why not a new module:** `MemoryMeasurement` (`memory_profiles.py:96`) already
models device, model, backend, precision, adapter scope, batch settings and
reserved peaks; `MemoryProfileStore` (`:282`) already does atomic,
schema-versioned, bounded persistence. Store one record **per probed batch
size** and fit across them.

**Interfaces:**
- `fit_batch_curve(records) -> tuple[int, int]` — `(base_bytes, slope_bytes)`
  from `(batch_size, reserved_peak)` pairs of records sharing one
  `ProfileIdentity`. With a single record the slope is the per-item cost implied
  by that point (never zero).
- `select_batch(records, *, usable_bytes: int, maximum: int, safety_fraction: float = 0.8) -> int` —
  requirement for candidate *n* is `max(fit(n), every observed peak at batch <= n)`:
  a **conservative envelope that can never predict below an observed sample**.
  Never exceeds the largest batch observed without OOM. Returns `0` when no
  candidate fits, including for an empty record set.
  `usable_bytes` is **raw free device bytes**; `safety_fraction` is applied
  inside this function and **nowhere else**, so safety cannot be double-applied.
  Every caller passes raw free memory. Assert this in the docstring and pin it
  with a test.
- `profile_store_path(scope: str) -> Path` — canonical location,
  `get_data_dir() / "memory_profiles" / f"{scope}.json"` via
  `hydra_suite.paths`, never `Path(__file__).parents[N]`. `MemoryProfileStore`
  has no production caller today; this plan is its first, so it defines the
  convention and every later caller reuses this function.
- `merge_records(existing, incoming) -> tuple[MemoryMeasurement, ...]` —
  **replace, do not append**: a record is keyed by
  `(identity, settings.batch_size)` and the newest observation wins. Without
  this, repeated `HYDRA_SAM3_FORCE_PROBE=1` runs grow the store until
  `MAX_PROFILE_RECORDS` truncation silently drops good data.
- Reuse unchanged: `recommend_batch_size` stays the single-record extrapolator
  and owns the MPS/CUDA/host pool distinction; `select_batch` shares that logic
  rather than duplicating it.

`ProbePlan.__post_init__` (`:132`) mandates that an unknown-profile probe begin
at batch size one; the candidate ladder `(1, 2, 4, 8)` complies.

- [ ] **Step 1: Write the failing tests**

```python
def test_fit_batch_curve_recovers_a_known_line():
    base, slope = fit_batch_curve(_records({1: 8_000, 2: 12_000, 4: 20_000}))
    assert abs(base - 4_000) < 200 and abs(slope - 4_000) < 200


def test_single_record_never_yields_a_zero_slope():
    _base, slope = fit_batch_curve(_records({1: 8_000}))
    assert slope > 0, "a zero slope would make every batch size look free"


def test_selection_never_predicts_below_an_observed_peak():
    # A superlinear jump at 2 must not be smoothed away by the linear fit.
    assert select_batch(_records({1: 8 * GiB, 2: 20 * GiB}),
                        usable_bytes=24 * GiB, maximum=8) == 1


def test_selection_never_exceeds_the_largest_observed_batch():
    assert select_batch(_records({1: 1 * GiB, 2: 2 * GiB}),
                        usable_bytes=512 * GiB, maximum=64) == 2


def test_selection_returns_zero_when_nothing_fits():
    assert select_batch((), usable_bytes=24 * GiB, maximum=8) == 0
    assert select_batch(_records({1: 40 * GiB}), usable_bytes=24 * GiB, maximum=8) == 0


def test_safety_fraction_is_applied_exactly_once():
    # usable_bytes is RAW free memory; 20 GiB at 0.8 admits a 16 GiB peak.
    assert select_batch(_records({1: 16 * GiB}), usable_bytes=20 * GiB, maximum=1) == 1
    assert select_batch(_records({1: 16 * GiB}), usable_bytes=19 * GiB, maximum=1) == 0


def test_merge_replaces_a_record_for_the_same_batch_size():
    merged = merge_records(_records({1: 1_000}), _records({1: 2_000}))
    assert len(merged) == 1 and merged[0].accelerator_reserved_peak_bytes == 2_000


def test_store_round_trips_one_record_per_batch(tmp_path):
    store = MemoryProfileStore(tmp_path / "sam3.json")
    store.save(_records({1: 1_000, 2: 2_000}))
    assert {r.settings.batch_size for r in store.load()} == {1, 2}
```

- [ ] **Step 2: Run them and watch them fail.**
- [ ] **Step 3: Implement.** A corrupt store must degrade to "measure again"
  (`load()` returning nothing), never to a crash or a wrong number — confirm
  `MemoryProfileStore.load` already does this and add a test if it does not.
- [ ] **Step 4: Tests pass.**
- [ ] **Step 5: Commit** — `feat(runtime): batch-curve fit and conservative selection`.

---

## Task 2: SAM3 workload fingerprint

**Files:**
- Create: `src/hydra_suite/training/sam3_lora/autobatch.py` (fingerprint half)
- Test: `tests/test_sam3_autobatch.py`

**The gap:** a `ProfileIdentity` built from the spec alone cannot distinguish
the things that actually drive the measurement. `base_model` defaults to the
literal string `"sam3"` (`detectkit/config/training.py:296`), so it identifies
no upstream revision at all, and a measurement taken on a sparse dataset would
be reused to size a dense one.

**Interface:** `sam3_workload_fingerprint(spec, *, cuda_device, dataset) -> ProfileIdentity`

Every field below is memory-driving and **must** participate in the key; a
change in any one must miss the cache:

| `ProfileIdentity` field | Source |
|---|---|
| `operation` | `"sam3_lora_train"` |
| `model_identity` | the sidecar-env package hash (below) **+** the resolved checkpoint file's `(size, mtime_ns)` — never the bare `base_model` string |
| `backend` | sidecar env name + the sidecar-env package hash (below) |
| `device_identity` | the **observed physical device** from `initial.cuda_device` (name + total VRAM + UUID-derived model class), not the requested `"cuda"` string. `device_identity` has **no existing producer** — `ProfileIdentity` is unused in production today — so this task defines the convention (`f"{name}|{total_vram_bytes}"`, never the UUID itself: a UUID would make the profile non-portable between two identical cards) and every later producer reuses one shared helper |
| `precision` | `params.mixed_precision` |
| `task` | tile size (`imgsz`), tile overlap, `object_tile_fraction` |
| `tiling_mode` | sliced vs whole-image |
| `adapter_scope` / `adapter_rank` | LoRA scope + `rank`/`alpha` |

### The sidecar-env package hash (how the parent computes a key it cannot import)

`sam3` and the CUDA-matched `torch` live in the **sidecar conda env**, not in
the parent's env — the parent cannot `import sam3` at all, and its own
`torch.__version__` is the wrong answer. But §3c step 2 must read the cache
*before* any child runs. So:

- `resolve_sam3_env` returns an env **name** (`env.py:21`), not a prefix. Add
  `sam3_env_prefix(env_name) -> Path`, resolving the prefix from
  `conda env list --json` (parent-side, no import, no child of the sidecar).
- `sam3_env_package_hash(prefix) -> str` = a stable hash of the **sorted
  filenames of `<prefix>/conda-meta/*.json`**. Those filenames are
  `name-version-build` strings, so the hash changes whenever any package in the
  env — `sam3`, `torch`, the CUDA runtime — is upgraded, and it is a pure
  directory listing: filesystem-only, milliseconds, no subprocess.
- Cache the hash in-process keyed on `(prefix, conda-meta mtime_ns)`. Only if
  the prefix cannot be resolved, fall back to one
  `conda run -n <env> python -c "..."` version query (and record that the key is
  degraded).

Plus, folded into `task` (the only free-text field with room) as a stable
sub-hash — the **dataset density profile**:
- maximum active instances per tile (the densest tile the probe will use),
- the p95 instances per tile,
- `num_negatives` and the negative-prompt composition (count + hash of the
  prompt pool), since negative queries are per-tile device state.

- [ ] **Step 1: Write the failing cross-rejection tests** — one per driver. Each
  asserts that changing that property alone produces a different key:

```python
@pytest.mark.parametrize("mutate", [
    _change_sidecar_env_packages,   # a new conda-meta entry (sam3, torch, CUDA)
    _change_checkpoint_bytes,
    _change_sidecar_env_name,
    _change_physical_gpu,
    _change_precision,
    _change_tile_px,
    _change_lora_rank,
    _change_max_instances_per_tile,
    _change_negative_prompt_pool,
    _change_num_negatives,
])
def test_every_memory_driver_misses_the_cache(mutate):
    base = sam3_workload_fingerprint(_spec(), cuda_device=_dev(), dataset=_dataset())
    assert mutate(sam3_workload_fingerprint) != base


def test_the_same_workload_hits_the_cache():
    assert _fingerprint() == _fingerprint()


def test_a_denser_dataset_does_not_reuse_a_sparse_measurement(tmp_path):
    sparse = _fingerprint(max_instances=3)
    dense = _fingerprint(max_instances=40)
    store = MemoryProfileStore(tmp_path / "sam3.json")
    store.save(_records({1: 8 * GiB}, identity=sparse))
    assert _records_for(store, dense) == ()
```

- [ ] **Step 2-4: Fail, implement, pass.**
- [ ] **Step 5: Commit** — `feat(training): workload fingerprint for SAM3 memory profiles`.

---

## Task 3: SAM3 parent-orchestrated probe phase

This is the heart of the plan. It fixes the original ordering bug and the
probe-admission and cache-ordering gaps.

**The ordering bug:** `train.py:144` runs `assess_preflight`, writes
`spec.json`, and builds immutable `WorkLimits` **before** launching the child.
`preflight.py:644` turns `-1` into `max(1, int(params.batch)) == 1`. A
child-side choice of batch 2-8 would run under limits, a host estimate, and an
accelerator estimate all admitted for batch 1. And `cli.py:727` (validation)
reads `params.batch` directly, so a local `batch_size` in `run_training` would
not propagate.

**Files:**
- Modify: `src/hydra_suite/training/sam3_lora/autobatch.py` (probe half)
- Modify: `src/hydra_suite/training/sam3_lora/train.py` (`train_sam3_lora`)
- Modify: `src/hydra_suite/training/sam3_lora/cli.py` (add `--probe`)
- Modify: `src/hydra_suite/training/sam3_lora/preflight.py` (probe admission)
- Modify: `src/hydra_suite/training/contracts.py:218` (`Sam3LoraParams.batch`)
- Modify: `src/hydra_suite/detectkit/config/training.py:650`
- Test: `tests/test_sam3_autobatch.py`, `tests/test_sam3_train_probe_phase.py`

### 3a. Probe admission policy (do not gate a measurement on the guess)

`assess_probe_preflight(spec, *, batch)` is a **new, separate** entry point in
`preflight.py`. It differs from `assess_preflight` in exactly two ways:

1. **Device gate:** it does NOT use `_MEASURED_BF16_DEVICE_PEAK_BYTES` (12 GiB
   at `preflight.py:75`) or `_EXTRA_BATCH_DEVICE_BYTES`. Those are the numbers
   the probe exists to replace; gating on them means hardware whose real
   batch-1 peak fits can be refused before it is ever measured. Instead it
   requires a **hard floor** — enough free VRAM to hold the model, its LoRA
   state, and one tile (an under-estimate by construction, computed from
   checkpoint size and tile geometry, not from an inherited envelope) — and
   otherwise lets the probe's own OOM handling be the gate. An OOM inside a
   contained probe child is a *measurement*, not a failure.
2. **Host gate:** unchanged in form but evaluated **per candidate**, since
   `preflight.py:644` shows host estimates scale with `batch_size` (decoded
   tiles, transformed tiles, collated images, dense masks).

Every other admission concern (leases, run-dir writability, dataset validity,
publish policy) is shared with `assess_preflight` — factor, don't fork.

### 3b. One fresh sidecar per candidate

Candidates 2-8 must not run under batch-1 containment. Launch **one contained
child per candidate**, each with:
- `assess_probe_preflight(spec, batch=candidate)` → its own `WorkLimits`,
- the same `CUDA_VISIBLE_DEVICES` UUID pin as training,
- the same `ContainmentPlan`/`SupervisedSidecar` machinery.

**How the child learns its candidate:** `spec.json` carries exactly one `batch`,
so the candidate is passed on the command line —
`cli --spec ... --run-dir ... --probe --probe-batch N`. The child writes
`run_dir/probe_records/batch_<N>.json`. Do not mutate `spec.json` per candidate;
the spec is rewritten once, in step 7, with the resolved value.

A child that exits on a classified memory-pressure kind is recorded as "this
candidate does not fit" and **stops the ladder** — exactly as an in-process
`OutOfMemoryError` would. A `assess_probe_preflight` **host refusal** at
candidate N likewise stops the ladder rather than skipping it: host demand is
monotone in batch, so no larger candidate can be admitted. Model reload per
candidate costs seconds and buys correct containment; take that trade.

### 3c. Flow in `train_sam3_lora`

0. **Observe the physical device first.** `_probe_cuda_device` already exists as
   a standalone observer (it is imported this way at
   `ultralytics_supervisor.py:55`), so call it before any preflight. Every path
   needs the UUID: the fingerprint needs it in step 2, and the explicit and
   cached paths need it for the `CUDA_VISIBLE_DEVICES` pin in step 7.
1. If `params.batch > 0` → `provenance="explicit"`, jump to step 7.
2. Compute the fingerprint (Task 2) and read the store. Records present and
   `HYDRA_SAM3_FORCE_PROBE != "1"` → `provenance="cached"`, jump to step 5.
3. `assess_probe_preflight(batch=1)`; refuse the run if even the hard floor
   fails.
4. Run the candidate ladder (3b), collecting one `MemoryMeasurement` per
   surviving candidate. **Zero survivors → refuse, write nothing.**
5. **Validate, then merge, then select** — in that order:
   - validate the records (identity matches the fingerprint, peaks positive and
     monotone non-decreasing in batch, `allocated <= reserved`); a record that
     fails validation is discarded, not stored;
   - `merge_records` into the store. This happens **before** selection and
     **independently of it**, because a measurement is a fact about this
     hardware and workload, while a selection also depends on how much VRAM
     happens to be free right now. Caching the fact and refusing the run is the
     correct pair: the next attempt on a quieter GPU reuses the measurement
     rather than re-probing. The only never-cache case is the fail-closed one in
     step 4;
   - `resolved = select_batch(records, usable_bytes=<raw free VRAM>, maximum=..., safety_fraction=0.8)`.
6. `resolved == 0` → `_result(success=False, failure_kind=ExitKind.HOST_ADMISSION_REFUSAL.value)`
   with a message naming the measured requirement and the free bytes. Do not
   launch.
7. Write the resolved batch into the spec **and** the provenance artifact (3d),
   run the final `assess_preflight` at `resolved`, build
   `_memory_limits`/`build_limited_launch`, launch training.

The training child is unchanged: `cli.py:567` and `:727` both read a positive
`params.batch`, so training and validation agree by construction.

### 3d. Where provenance lives (schema-legal)

`_SidecarSpec` does `Sam3LoraParams(**sam3_data)` (`cli.py:79`), so **any extra
key inside `sam3_params` raises `TypeError` in the child**. Therefore:

- `spec.json["sam3_params"]["batch"]` = the resolved positive integer. That is
  the only mutation inside `sam3_params`.
- Provenance goes in a **new top-level, output-only block**,
  `spec.json["batch_resolution"] = {"requested": -1, "resolved": 2,
  "provenance": "measured", "fingerprint": "<key>", "measured_reserved_bytes":
  ..., "free_bytes": ..., "resolved_at_unix_ns": ...}`. `_SidecarSpec` reads
  fields by `.get`, so an unknown top-level key is inert.
- The same block is written to `run_dir/batch_resolution.json` as the durable
  artifact, and echoed into `resource_preflight.json`.
- **Not forgeable:** the parent writes this block unconditionally on every run,
  after resolution, overwriting anything present. The DetectKit plan schema
  must reject `batch_resolution` as an unknown key on input (the plan loader
  already rejects unknown keys — extend its test).

### 3e. Probe fidelity

- Build model + loss + **optimizer** (the optimizer at `cli.py:573` is part of
  the peak — lazy Adam state and the first step).
- Run **at least two full steps through `optimizer.step()`** per candidate.
- Use the **densest tiles** (maximum instances per tile) — mask memory scales
  with active instances, so the median tile understates the peak. This is the
  same density the fingerprint records.
- `reset_peak_memory_stats()` before; record `max_memory_reserved()` and
  `max_memory_allocated()` after; `empty_cache()` at teardown.
- Zero survivors → typed `ProbeFailedError`, non-zero exit, no records.

### 3f. Cancellation and teardown

- Check `should_cancel()` before each candidate and between probe and final
  preflight.
- Tear the probe child down through the same supervisor path as training;
  handle `WorkloadStillOwnedError` exactly as `train.py:254`/`:336`/`:344` do.
- On cancellation: **merge nothing, write no `batch_resolution`, launch
  nothing**, and return the same cancelled-result shape training uses.

**Interfaces:**
- `probe_candidates(spec) -> tuple[int, ...]`
- `run_probe(spec, run_dir, *, step_fn=None) -> tuple[MemoryMeasurement, ...]` —
  `step_fn(batch_size)` injectable so the module is testable without `sam3` or a
  GPU.
- `resolve_batch(spec, records, *, usable_bytes: int, maximum: int) -> tuple[int, str]`
  → `(batch, provenance)`. `usable_bytes` is **raw free device bytes**; the
  safety fraction is applied only inside `select_batch`.

- [ ] **Step 1: Write the failing tests**

```python
def test_probe_stops_at_the_first_oom_and_never_exceeds_it():
    seen = []

    def step(batch):
        seen.append(batch)
        if batch >= 4:
            raise torch.cuda.OutOfMemoryError("out of memory")

    records = autobatch.run_probe(_spec(), _run_dir(), step_fn=step)
    assert seen == [1, 2, 4], "must stop probing after the first OOM"
    assert max(r.settings.batch_size for r in records) == 2


def test_a_probe_that_ooms_at_batch_one_fails_closed(tmp_path):
    def step(_batch):
        raise torch.cuda.OutOfMemoryError("out of memory")

    with pytest.raises(autobatch.ProbeFailedError):
        autobatch.run_probe(_spec(), tmp_path, step_fn=step)
    assert not list(tmp_path.glob("*.json")), "a doomed config must not be cached"


def test_each_candidate_is_admitted_and_contained_at_its_own_batch():
    train_sam3_lora(_spec(batch=-1), run_dir)
    assert recorded_probe_preflight_batches == [1, 2, 4]
    assert [limits.batch for limits in recorded_launch_limits] == [1, 2, 4]


def test_probe_admission_does_not_use_the_fallback_device_envelope(monkeypatch):
    monkeypatch.setattr(pf, "_MEASURED_BF16_DEVICE_PEAK_BYTES", 10_000 * GiB)
    assert pf.assess_probe_preflight(_spec(), batch=1).admitted


def test_a_successful_probe_is_cached_even_when_selection_refuses(tmp_path):
    # Measurements are facts; free memory is transient.
    result = train_sam3_lora(_spec(batch=-1), run_dir, free_bytes=1 * GiB)
    assert not result["success"]
    assert MemoryProfileStore(_store_path()).load(), "the measurement must survive"
    assert not launched


def test_cancellation_merges_nothing_and_launches_nothing(tmp_path):
    train_sam3_lora(_spec(batch=-1), run_dir, should_cancel=_cancel_after_first_candidate)
    assert not MemoryProfileStore(_store_path()).load()
    assert not launched
    assert not (run_dir / "batch_resolution.json").exists()


def test_explicit_batch_is_honoured_without_probing():
    batch, provenance = autobatch.resolve_batch(
        _spec(batch=4), records=(), usable_bytes=24 * GiB, maximum=8
    )
    assert (batch, provenance) == (4, "explicit")
```

```python
# tests/test_sam3_train_probe_phase.py — the ordering the audit demanded
def test_resolved_batch_is_written_before_limits_are_built(monkeypatch):
    train_sam3_lora(_spec(batch=-1), run_dir)
    assert order.index("probe") < order.index("final_preflight") \
        < order.index("build_limited_launch")
    spec_on_disk = json.loads((run_dir / "spec.json").read_text())
    assert spec_on_disk["sam3_params"]["batch"] == 2
    assert set(spec_on_disk["sam3_params"]) == _SAM3_PARAM_FIELDS, \
        "extra keys in sam3_params would raise in the child"
    assert spec_on_disk["batch_resolution"]["requested"] == -1
    assert spec_on_disk["batch_resolution"]["provenance"] == "measured"
    assert json.loads((run_dir / "batch_resolution.json").read_text())["resolved"] == 2


def test_the_child_can_load_the_rewritten_spec():
    # Guards 3d directly: Sam3LoraParams(**sam3_data) must not raise.
    cli._load_spec(run_dir / "spec.json")


def test_final_preflight_sees_the_resolved_batch_not_minus_one():
    assert recorded_final_preflight_batches == [2]


def test_training_and_validation_use_the_same_resolved_batch():
    ...  # cli.py:567 (train) and cli.py:727 (validate) both read 2


def test_a_plan_may_not_supply_batch_resolution():
    with pytest.raises(TrainingPlanError):
        load_plan({"batch_resolution": {"resolved": 64}})
```

- [ ] **Step 2: Run and watch fail.**
- [ ] **Step 3: Implement `assess_probe_preflight`** (3a), factoring the shared
  admission concerns rather than forking them.
- [ ] **Step 4: Implement the probe half of `autobatch.py` + `--probe` in `cli.py`** (3e).
- [ ] **Step 5: Implement the parent phase in `train.py`** (3b, 3c, 3f).
- [ ] **Step 6: Implement the provenance block** (3d) and reject it on plan input.
- [ ] **Step 7: Log prominently** — at the same prominence as the existing shape
  banner:
  `auto batch: 2 (measured 16.0 GiB reserved at batch 2, 67% of 24 GiB free; provenance=measured)`.
- [ ] **Step 8: Allow `-1`** in `Sam3LoraParams` and the plan validator; still
  reject `0` and `< -1`. Update the stale `contracts.py:217` comment to point at
  the probe as the authority.
- [ ] **Step 9: Tests pass**, plus `python -m pytest tests/ -k "sam3" -q`.
- [ ] **Step 10: Commit** — `feat(training): parent-orchestrated measured auto batch for SAM3`.

---

## Task 4: The measured preflight formula

**Files:**
- Modify: `src/hydra_suite/training/sam3_lora/preflight.py` (`:75-83`, and the
  `training_device_peak` computation at `:721`)
- Test: `tests/test_sam3_preflight.py`

**The ambiguity to remove.** Today:

```python
training_device_peak = (
    _MEASURED_BF16_DEVICE_PEAK_BYTES * precision_multiplier          # base envelope
    + max(0, batch_size - 1) * _EXTRA_BATCH_DEVICE_BYTES * precision_multiplier
    + max(0, lora_training_state - default_lora_training_state)      # LoRA adjustment
    + dense_masks_device                                             # masks
)
```

A `MemoryMeasurement` is the **complete observed reserved peak** — it already
contains the base, the per-batch increment, the LoRA state, the optimizer state
and the dense masks. So the measured path **replaces the entire expression**,
not one term. Retaining any of the four addends would double-count.

**Specified formula** — when validated records exist for this fingerprint:

```python
base, slope = fit_batch_curve(records)
measured_peak = max(
    base + slope * batch_size,                       # fitted
    *(r.accelerator_reserved_peak_bytes             # never below an observation
      for r in records if r.settings.batch_size <= batch_size),
)
training_device_peak = int(measured_peak)
```

with these rules:
- **Headroom is applied in the admission comparison, once.** A measured peak is
  a raw envelope with no margin; admitting at `free >= measured_peak` exactly
  would be a safety *regression* against today's padded fallback, in a gate
  whose stated philosophy (`preflight.py:78-83`) is to refuse a marginal run
  rather than OOM after minutes of setup. So the measured path admits only when
  `free_bytes * MEASURED_SAFETY_FRACTION >= training_device_peak`, using the
  **same shared constant `MEASURED_SAFETY_FRACTION = 0.8`** that `select_batch`
  uses — defined once in `memory_profiles.py` and imported by both. Because the
  headroom lives in the comparison and not in the peak, the estimate assertions
  below stay exact. This matters most on the **explicit-batch path with a cached
  record**, which never goes through `select_batch` and would otherwise have no
  margin at all.
- **An explicit batch above the largest observed-OK batch is not extrapolated
  into an admission.** `select_batch` refuses to extrapolate; preflight must not
  quietly do what selection refuses. When `batch_size` exceeds
  `max(r.settings.batch_size for r in records)`, still compute the fitted peak,
  but stamp the diagnostic provenance `"extrapolated"` and require the full
  fallback headroom (`_MEASURED_BF16_DEVICE_PEAK_BYTES`'s ~53% margin) rather
  than 0.8 — the fit is unvalidated past the last observation.
- **No `precision_multiplier`.** The fingerprint includes `precision`, so a
  record only matches a run of the same precision. A precision change misses the
  cache and falls back. Never scale a measurement by a guessed multiplier.
- **No `_DEVICE_STEADY_BYTES`, no `_EXTRA_BATCH_DEVICE_BYTES`, no LoRA
  adjustment, no `dense_masks_device`** on the measured path — all are inside
  the observation.
- `validation_device_peak = training_device_peak`, as today.
- Host estimates are **unchanged**; only the device peak becomes measured.
- Fall back to the existing expression verbatim when no record matches.

- [ ] **Step 1: Write the failing tests — assert the exact formula, not a bound**

```python
def test_measured_peak_replaces_the_whole_estimate_stack(monkeypatch, tmp_path):
    records = _records({1: 8 * GiB, 2: 12 * GiB})   # base 4, slope 4
    monkeypatch.setattr(pf, "_stored_records", lambda *_a, **_k: records)
    decision = _decision(_spec(tmp_path, batch=2))
    assert decision.budget.accelerator_peak_bytes == 12 * GiB, \
        "no base envelope, per-batch term, LoRA adjustment or mask term may be added"


def test_measured_peak_is_never_below_an_observation(monkeypatch, tmp_path):
    records = _records({1: 8 * GiB, 2: 20 * GiB})   # superlinear
    monkeypatch.setattr(pf, "_stored_records", lambda *_a, **_k: records)
    assert _decision(_spec(tmp_path, batch=2)).budget.accelerator_peak_bytes == 20 * GiB


def test_a_precision_change_misses_the_cache_and_falls_back(monkeypatch, tmp_path):
    monkeypatch.setattr(pf, "_stored_records", lambda *_a, **_k: _records({1: 8 * GiB}, precision="bf16"))
    decision = _decision(_spec(tmp_path, batch=1, precision="fp32"))
    assert decision.budget.accelerator_peak_bytes == _legacy_expression(batch=1, precision="fp32")


def test_fallback_matches_the_current_expression_exactly(monkeypatch, tmp_path):
    monkeypatch.setattr(pf, "_stored_records", lambda *_a, **_k: ())
    assert _decision(_spec(tmp_path, batch=2)).budget.accelerator_peak_bytes \
        == _legacy_expression(batch=2, precision="bf16")


def test_the_measured_path_still_keeps_headroom(monkeypatch, tmp_path):
    monkeypatch.setattr(pf, "_stored_records", lambda *_a, **_k: _records({1: 16 * GiB}))
    assert _decision(_spec(tmp_path, batch=1), free_bytes=20 * GiB).admitted
    assert not _decision(_spec(tmp_path, batch=1), free_bytes=19 * GiB).admitted


def test_an_explicit_batch_beyond_the_observations_is_marked_extrapolated(monkeypatch, tmp_path):
    monkeypatch.setattr(pf, "_stored_records", lambda *_a, **_k: _records({1: 8 * GiB}))
    decision = _decision(_spec(tmp_path, batch=8))
    assert decision.budget.device_peak_provenance == "extrapolated"


def test_preflight_never_probes(monkeypatch, tmp_path):
    monkeypatch.setattr(autobatch, "run_probe", _explode)
    _decision(_spec(tmp_path, batch=2))  # must not raise
```

- [ ] **Step 2-4: Fail, implement, pass.** Preflight stays offline and fast:
  read the store only, never probe.
- [ ] **Step 5: Record provenance.** The refusal/diagnostic text and
  `resource_preflight.json` must say whether the device estimate came from a
  measurement (and which fingerprint) or from the fallback constants.
- [ ] **Step 6: Commit** — `fix(training): estimate the device peak from measurement when one exists`.

---

## Task 5: YOLO pre-launch batch resolution

**Files:**
- Create: `src/hydra_suite/training/yolo_autobatch.py`
- Modify: `src/hydra_suite/training/ultralytics_supervisor.py` (`:65-70`, `:319-341`)
- Test: `tests/test_yolo_autobatch.py`, `tests/test_ultralytics_supervisor.py`,
  `tests/test_detectkit_training_cli.py`

**Why not "just pass `batch=-1` through".** Revision 2 proposed leaving `-1` in
the launch command and accounting pressure with a "ceiling at `fraction=0.60`".
Reading the installed Ultralytics 8.4.138 shows that is not implementable:

- it profiles `[1, 2, 4, 8, 16]` or `[1, …, 64]` depending on total VRAM, then
  **extrapolates from a linear fit** and only rejects the result outside
  `1..1024` (`ultralytics/utils/autobatch.py:99,125`) — there is no static
  ceiling the supervisor can infer cheaply, and the range is version-dependent;
- on CPU/MPS it **returns the default without measuring** (`:81`), as it does
  when `cudnn.benchmark` is on;
- and with `-1` left in the command the parent **never learns the selected
  value**, which contradicts the global requirement to persist the resolved
  batch and its provenance.

**So YOLO gets the same shape as SAM3: resolve before launch.**

`resolve_yolo_batch(spec, *, log_cb, should_cancel) -> tuple[int, str]`:
1. `params.batch > 0` → `(batch, "explicit")`, no child.
2. Otherwise launch a short **contained resolution child** in the training
   sidecar env that loads the model, calls Ultralytics'
   `check_train_batch_size`/`autobatch` with the run's `imgsz`, `fraction`, and
   `max_num_obj`, and **writes `run_dir/batch_resolution.json`**. The parent
   reads that file; a missing or unparseable file is a resolution failure.
   **Not stdout:** the supervisor drops child output under pressure — it reports
   `dropped_output_lines` and `output_error` in `_containment_diagnostic`
   (`train.py:100ff`) — so a dropped line would silently degrade a resolved 48
   into a fallback 16 with no trace. A file is non-lossy and matches what the
   SAM3 probe already does with its records.
3. **Bounded, version-independent stand-in:** clamp whatever the child reports
   to `[1, YOLO_MAX_AUTO_BATCH]` (a repo constant, default 64 — the largest size
   Ultralytics actually profiles rather than extrapolates). A clamp is recorded
   in the provenance string (`"ultralytics_autobatch_clamped"`).
4. **Non-CUDA devices:** Ultralytics does not measure there, so do not pretend.
   Return `(TrainingHyperParams.batch default, "default_non_cuda")` and say so
   in the log.
5. Resolution-child failure, cancellation, or a missing/unparseable
   `batch_resolution.json` → fall back to
   the explicit default with `provenance="fallback"`; never propagate `-1`.

The supervisor then receives a **positive** batch, so `PressureSettings` keeps
its `>= 1` invariant untouched, `_command_with_pressure` keeps rewriting
`batch=` from `settings.batch_size`, and the OOM-retry halving ladder works
unchanged from a real starting point. Fix `_estimate_host_bytes` (`:65-70`) to
size from the **resolved** batch — with the resolution phase in front of it,
`max(1, int(params.batch))` is no longer reachable with `-1`.

Write `{"requested": -1, "resolved": 24, "provenance": "..."}` to the run dir
using the **same `batch_resolution.json` schema as SAM3** (Task 3d).

- [ ] **Step 1: Write the failing tests**

```python
def test_resolution_runs_before_launch_and_the_command_has_a_positive_batch():
    commands = _capture_commands(_spec(batch=-1), child_writes=24)
    assert "batch=-1" not in commands[0] and "batch=24" in commands[0]


def test_the_resolved_batch_is_persisted_with_its_provenance(tmp_path):
    _run(_spec(batch=-1), tmp_path, child_writes=24)
    payload = json.loads((tmp_path / "batch_resolution.json").read_text())
    assert payload == {"requested": -1, "resolved": 24,
                       "provenance": "ultralytics_autobatch", **_ANY_TIMESTAMP}


def test_an_absurd_report_is_clamped_not_trusted():
    assert _resolve(child_writes=1024) == (64, "ultralytics_autobatch_clamped")


def test_non_cuda_returns_the_default_and_says_so():
    assert _resolve(device="mps") == (16, "default_non_cuda")


def test_a_resolution_failure_falls_back_and_never_propagates_minus_one():
    batch, provenance = _resolve(child_fails=True)
    assert batch >= 1 and provenance == "fallback"


def test_cancellation_during_resolution_launches_no_training():
    _run(_spec(batch=-1), should_cancel=_immediately)
    assert not launched


def test_an_explicit_batch_is_never_rewritten():
    assert "batch=16" in _capture_commands(_spec(batch=16))[0]


def test_a_memory_pressure_retry_halves_from_the_resolved_batch():
    commands = _capture_commands(_spec(batch=-1), child_writes=24, oom_on_attempt=0)
    assert "batch=24" in commands[0] and "batch=12" in commands[1]


def test_host_estimate_uses_the_resolved_batch():
    assert _estimate_host_bytes(_spec(batch=24)) > _estimate_host_bytes(_spec(batch=1))
```

Plus, in `tests/test_detectkit_training_cli.py`: a plan omitting
`training.batch` resolves to `-1` at the plan level, and one setting `batch: 16`
still resolves to 16.

- [ ] **Step 2-4: Fail, implement, pass.**
- [ ] **Step 5: Document** in `docs/runbooks/detectkit-headless-training.md`:
  `-1` means "let Ultralytics measure before launch", the resolved value lands
  in `batch_resolution.json`, and a fixed batch is the way to get
  byte-reproducible runs.
- [ ] **Step 6: Commit** — `feat(detectkit): resolve YOLO auto batch before launch`.

**Ruling if you find one:** changing the DetectKit default from 16 to `-1`
alters training dynamics for existing plans that omit `batch`. If any
equivalence fixture or golden test depends on the current default, DO NOT change
the default — land the resolution phase (a correctness improvement regardless)
and make `-1` a documented opt-in. Record the decision in the ledger.

---

## Task 6: Expose auto batch in the SAM3 training panel

**Files:**
- Modify: `src/hydra_suite/detectkit/gui/panels/sam3_training_panel.py:391`
- Test: extend the panel's existing test module

`self.batch_spin.setRange(1, 256)` makes `-1` unreachable from the GUI, so
without this the feature ships headless-only.

- [ ] **Step 1: Add an "Auto (measure)" checkbox** next to the Batch spin. When
  checked, the spin is disabled and the panel emits `batch = -1`; unchecked, the
  spin's value is emitted and its range stays `1..256`.
- [ ] **Step 2: Round-trip test** — loading a plan with `batch: -1` checks the
  box; `batch: 8` unchecks it and shows 8.
- [ ] **Step 3: Commit** — `feat(detectkit): expose SAM3 auto batch in the training panel`.

---

## Dropped: measured probe for inference batch sizing

The original Task 5 (`src/hydra_suite/utils/batch_optimizer.py`) is **removed
from this plan**, not deferred silently:

- `BatchOptimizer` has **zero production callers** — a repo-wide grep finds it
  only in its own module and in tests (`tests/test_batch_optimizer.py`;
  `tests/test_tracking_worker_helpers.py` stubs it;
  `tests/test_legacy_batching_removal.py` names it as the one sanctioned legacy
  exception). The live pipeline consumes an explicit detection batch at
  `core/inference/pipeline.py:222`.
- Its API receives no loaded model, executor, or representative frames, so it
  cannot run "real inference batches" without violating layer boundaries. Doing
  it properly means a callback from the model-owning runtime seam, plus defined
  TensorRT/CoreML behaviour and separate MPS measurement semantics (unified
  memory — the CUDA-only `max_memory_reserved` API does not apply).

**Follow-up for the ledger:** decide whether `BatchOptimizer` is deleted or
redesigned around a runtime-seam callback. Out of scope here.

---

## Verification

- [ ] `python -m pytest tests/ -k "sam3 or batch or preflight or memory or ultralytics" -q`
- [ ] `make lint-moderate`
- [ ] `make format-check`
- [ ] `make docs-check` — Task 5 edits a runbook.
- [ ] **SAM3 GPU acceptance (coordinate first — see Global Constraints):** run
  the probe end-to-end and confirm (a) the selected batch trains without OOM,
  (b) the profile store is written and a second run skips the probe,
  (c) `HYDRA_SAM3_FORCE_PROBE=1` re-probes and **replaces** rather than appends,
  (d) a spec whose probe OOMs at batch 1 refuses before launching,
  (e) a probe that succeeds but cannot fit right now still caches its
  measurement, (f) cancelling mid-probe leaves no store entry, no
  `batch_resolution.json`, and no training launch. Record the measured
  `(base, slope)` in the commit message — that number is the point of the whole
  exercise.
- [ ] **YOLO CUDA acceptance on `mehek`:** a real DetectKit YOLO plan with
  `training.batch: -1` resolves to a positive batch **before** launch, the
  launch command contains that number (never `-1`), `batch_resolution.json`
  records it with its provenance, and the run completes.
- [ ] Confirm the run's `spec.json`, `batch_resolution.json`, and logs all state
  the same resolved batch and provenance, and that SAM3 training and validation
  report the same number.
- [ ] **Final branch-diff review:** `git diff main...codex/measured-auto-batch`
  read end to end before merge — no formatting churn, no unrelated cleanup, no
  measurement baked into a source constant.
- [ ] **Merge back** into the primary checkout only after the branch is green
  (`--no-ff`), preserving every existing user change in the primary checkout.
- [ ] **Re-run the test and lint gates in the primary checkout** after the
  merge; a green worktree is not evidence about main.
- [ ] **Only then** remove the worktree and branch:
  `git worktree remove .worktrees/auto-batch && git worktree prune`.
- [ ] **Update this plan's `**Status:**` header** to
  `Shipped — merged to main (<sha>)` and `git mv` it into
  `docs/superpowers/plans/done/` in the merge commit, per CLAUDE.md.

## Known reference points

- SAM3 at batch 1, rank 16, 1008 px, 206 adapters: **7.82-7.83 GiB reserved**,
  measured identically on an RTX 6000 Ada (48 GB) and an RTX 4090 (24 GB).
  A correct probe should reproduce this base within a few hundred MiB — but that
  figure came from forward+backward accounting; a probe that includes the
  optimizer step may legitimately read higher, and higher is the number to
  trust.
- `preflight.py:75` currently pads that measurement to a 12 GiB envelope
  (~53% headroom). Once the measured formula in Task 4 is live, that padding
  applies only to the fallback path.
- `contracts.py:217` carries the comment "batch 2 OOMs at 1008 px on a 47 GB
  card". That claim predates the `perflib_compat` change and the streaming
  dataloader, and is **unverified** against the current code. The probe is what
  should settle it — if batch 2 now fits, update the comment; if it still OOMs,
  the probe will discover that safely and the comment stands.
