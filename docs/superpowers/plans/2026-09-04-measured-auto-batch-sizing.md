# Measured auto batch sizing implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development
> (recommended) or superpowers:executing-plans to implement this plan task-by-task.
> Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace inherited/analytic batch-memory constants with numbers this
codebase measures on the machine it is actually running on, and expose that
measurement as `batch: -1` auto sizing for SAM3 and YOLO.

**Architecture:** One shared, hardware-keyed measurement cache plus one probe
per training role. The probe runs a few real forward+backward steps at
increasing batch sizes, records `torch.cuda.max_memory_reserved()`, backs off
on OOM, fits a linear model (`bytes = base + slope * batch`), and both (a)
selects a batch size and (b) hands preflight a measured slope instead of a
guessed constant.

**Tech Stack:** PyTorch CUDA memory stats, Ultralytics `autobatch`, existing
`hydra_suite.runtime.resource_budget` / `training.sam3_lora.preflight`.

**Spec:** No separate design doc. The motivating evidence is in
`docs/superpowers/specs/2026-09-04-sam3-spike-vs-build-consolidation.md`
(inherited-constant class of bug) and commit `7a95cf25`, which replaced one
such constant with a measurement.

---

## Global Constraints

- **Never bake a measurement into a source constant.** A measurement is valid
  for one (GPU model, config) pair. `_MEASURED_BF16_DEVICE_PEAK_BYTES = 29 GiB`
  was inherited from a different machine and configuration, was 3.7x the real
  figure, and silently excluded every card below ~32 GiB. Cache measurements
  keyed on hardware; treat any in-source constant as a fallback only.
- **Measure `max_memory_reserved`, not `max_memory_allocated`.** The caching
  allocator does not return blocks to the driver between steps, so *reserved*
  is what the device must actually have free.
- **A probe must never leave the process in a worse state than it found it.**
  Call `torch.cuda.empty_cache()` and `reset_peak_memory_stats()` between
  probe points, and restore model/optimizer state (or build a throwaway) so
  probing cannot perturb the run that follows.
- **Auto sizing proposes; it does not silently change training dynamics.**
  Batch size changes gradient noise and therefore the trained model. Every
  auto-selected value must be logged prominently, recorded in the run's
  metadata/spec, and never changed mid-run.
- **Preflight stays offline and millisecond-fast.** It may READ a cached
  measurement; it must not run a probe or a network call.
- **GPU etiquette:** `courtship.taild08eb9.ts.net` may be running SAM3
  training (unit `sam3-4090`) and `mehek.taild08eb9.ts.net` is shared. Check
  `systemctl --user is-active sam3-4090` and `nvidia-smi` before any GPU work,
  never kill a process that is not yours, and prefer short probes.
- **Isolation:** work in a git worktree branched from local HEAD
  (`git worktree add .worktrees/auto-batch -b feat/measured-auto-batch HEAD`).
  Never `git stash`, never `git clean -fdx`, never `rm -rf`.
- Run tests with `conda activate hydra-mps` and `KMP_DUPLICATE_LIB_OK=TRUE`.
  Most of this plan is testable on CPU with fakes; GPU work is called out
  explicitly per task.

---

## File Structure

- **Create** `src/hydra_suite/runtime/memory_probe.py` — hardware-keyed
  measurement cache + the shared linear-fit helper. Runtime layer: importable
  by training and app layers, imports nothing from them.
- **Create** `src/hydra_suite/training/sam3_lora/autobatch.py` — the SAM3
  probe (runs inside the sidecar env; may import `torch`, never Qt).
- **Modify** `src/hydra_suite/training/sam3_lora/preflight.py` — consume a
  cached measured slope in place of `_EXTRA_BATCH_DEVICE_BYTES`.
- **Modify** `src/hydra_suite/training/sam3_lora/cli.py` — resolve `batch: -1`
  before the training loop; log the chosen value.
- **Modify** `src/hydra_suite/training/contracts.py` and
  `src/hydra_suite/detectkit/config/training.py` — allow `sam3.batch = -1`.
- **Modify** `src/hydra_suite/utils/batch_optimizer.py` — measured probe path
  for inference, heuristic retained as fallback.
- **Tests**: `tests/test_memory_probe.py`, `tests/test_sam3_autobatch.py`,
  `tests/test_batch_optimizer_measured.py`, plus additions to
  `tests/test_sam3_preflight.py`.

---

## Task 1: Hardware-keyed measurement cache

**Files:**
- Create: `src/hydra_suite/runtime/memory_probe.py`
- Test: `tests/test_memory_probe.py`

**Interfaces:**
- Produces:
  - `@dataclass(frozen=True) MemoryMeasurement(base_bytes: int, slope_bytes: int, max_batch: int, measured_at: str, samples: tuple[tuple[int, int], ...])`
  - `probe_key(**parts: object) -> str` — stable hash of the hardware +
    config identity (GPU name, total VRAM, torch version, plus caller-supplied
    parts such as `imgsz`, `rank`, `tile_px`).
  - `load_measurement(key: str) -> MemoryMeasurement | None`
  - `store_measurement(key: str, measurement: MemoryMeasurement) -> None`
  - `fit_linear(samples: Sequence[tuple[int, int]]) -> tuple[int, int]` —
    returns `(base_bytes, slope_bytes)`; with a single sample the slope is the
    per-item cost implied by that point (never zero).

Cache location: `get_data_dir() / "memory_probes" / f"{key}.json"` via
`hydra_suite.paths` — never `Path(__file__).parents[N]`.

- [ ] **Step 1: Write the failing tests**

```python
def test_probe_key_is_stable_and_hardware_sensitive():
    a = memory_probe.probe_key(gpu="RTX 4090", vram=24, imgsz=1008)
    b = memory_probe.probe_key(gpu="RTX 4090", vram=24, imgsz=1008)
    c = memory_probe.probe_key(gpu="RTX 6000 Ada", vram=48, imgsz=1008)
    assert a == b and a != c


def test_fit_linear_recovers_a_known_line():
    base, slope = memory_probe.fit_linear([(1, 8_000), (2, 12_000), (4, 20_000)])
    assert abs(base - 4_000) < 200
    assert abs(slope - 4_000) < 200


def test_single_sample_never_yields_a_zero_slope():
    _base, slope = memory_probe.fit_linear([(1, 8_000)])
    assert slope > 0, "a zero slope would make every batch size look free"


def test_store_then_load_round_trips(tmp_path, monkeypatch):
    monkeypatch.setattr(memory_probe, "_cache_dir", lambda: tmp_path)
    m = memory_probe.MemoryMeasurement(1, 2, 4, "2026-09-04T00:00:00", ((1, 1),))
    memory_probe.store_measurement("k", m)
    assert memory_probe.load_measurement("k") == m


def test_load_returns_none_for_a_corrupt_or_missing_entry(tmp_path, monkeypatch):
    monkeypatch.setattr(memory_probe, "_cache_dir", lambda: tmp_path)
    assert memory_probe.load_measurement("absent") is None
    (tmp_path / "bad.json").write_text("{not json")
    assert memory_probe.load_measurement("bad") is None
```

- [ ] **Step 2: Run them and watch them fail** —
  `python -m pytest tests/test_memory_probe.py -v`. Expected: import error.
- [ ] **Step 3: Implement the module.** A corrupt cache entry must return
  `None` (re-measure), never raise — a stale cache must degrade to "measure
  again", never to a crash or, worse, a wrong number.
- [ ] **Step 4: Tests pass.**
- [ ] **Step 5: Commit** — `feat(runtime): hardware-keyed memory measurement cache`.

---

## Task 2: SAM3 measured autobatch

**Files:**
- Create: `src/hydra_suite/training/sam3_lora/autobatch.py`
- Modify: `src/hydra_suite/training/sam3_lora/cli.py` (in `run_training`,
  after the model and loss are built, before the epoch loop)
- Modify: `src/hydra_suite/training/contracts.py:218` (`Sam3LoraParams.batch`)
- Modify: `src/hydra_suite/detectkit/config/training.py:651`
  (`sam3.batch must be positive` → allow `-1`)
- Test: `tests/test_sam3_autobatch.py`

**Interfaces:**
- Consumes: `memory_probe.{probe_key, load_measurement, store_measurement, fit_linear, MemoryMeasurement}`
- Produces: `measure_sam3_batch(step_fn, *, candidates=(1, 2, 4, 8), device, fraction=0.8) -> MemoryMeasurement`
  and `resolve_batch_size(params, *, device, probe) -> int`.

`step_fn(batch_size: int) -> None` performs one real forward + backward at
that batch size; the probe owns all memory bookkeeping around it. Injecting it
keeps this module testable without `sam3` or a GPU.

Behaviour:
- Candidates ascending; stop at the first `torch.cuda.OutOfMemoryError`
  (catch it, `empty_cache()`, and treat the previous size as the maximum).
- Record `(batch, max_memory_reserved)` per successful candidate.
- Select the largest candidate whose fitted requirement fits
  `fraction * total_vram`; **never return more than the largest candidate
  actually measured without OOM.**
- `params.batch > 0` short-circuits: no probe, honour the user.
- Cache hit short-circuits too, unless `HYDRA_SAM3_FORCE_PROBE=1`.

- [ ] **Step 1: Write the failing tests**

```python
def test_probe_stops_at_the_first_oom_and_never_exceeds_it():
    seen = []

    def step(batch):
        seen.append(batch)
        if batch >= 4:
            raise torch.cuda.OutOfMemoryError("out of memory")

    m = autobatch.measure_sam3_batch(
        step, candidates=(1, 2, 4, 8), device=_fake_device(total=24 * 2**30)
    )
    assert seen == [1, 2, 4], "must stop probing after the first OOM"
    assert m.max_batch == 2


def test_selected_batch_respects_the_memory_fraction():
    # 10 GiB base + 6 GiB per item on a 24 GiB card at fraction 0.8 (19.2 GiB)
    # admits batch 1 only.
    m = autobatch.MemoryMeasurement(
        base_bytes=4 * 2**30, slope_bytes=6 * 2**30, max_batch=8,
        measured_at="", samples=((1, 10 * 2**30),),
    )
    assert autobatch.select_batch(m, total_bytes=24 * 2**30, fraction=0.8) == 2


def test_explicit_batch_is_honoured_without_probing():
    called = []
    params = SimpleNamespace(batch=4)
    got = autobatch.resolve_batch_size(
        params, device=None, probe=lambda: called.append(1)
    )
    assert got == 4 and not called


def test_oom_is_recovered_not_propagated():
    def step(batch):
        raise torch.cuda.OutOfMemoryError("out of memory")

    m = autobatch.measure_sam3_batch(
        step, candidates=(1, 2), device=_fake_device(total=24 * 2**30)
    )
    assert m.max_batch == 1, "batch 1 is the floor even if it OOMs; refuse elsewhere"
```

- [ ] **Step 2: Run and watch fail.**
- [ ] **Step 3: Implement `autobatch.py`.**
- [ ] **Step 4: Wire into `cli.run_training`.** Resolve before the epoch loop.
  Log at the same prominence as the existing shape banner, e.g.
  `emit_log(f"auto batch: {n} (measured {gib:.2f} GiB at batch {n}, {pct}% of {total:.0f} GiB)")`,
  and record the resolved value into the run's spec/metadata so the trained
  artifact carries the batch it was actually trained at.
- [ ] **Step 5: Allow `-1`** in `Sam3LoraParams` and the plan validator; the
  validator must still reject `0` and `< -1`.
- [ ] **Step 6: Tests pass**, plus `python -m pytest tests/ -k "sam3" -q`.
- [ ] **Step 7: Commit** — `feat(training): measured auto batch sizing for SAM3`.

---

## Task 3: Feed the measurement into preflight

**Files:**
- Modify: `src/hydra_suite/training/sam3_lora/preflight.py:82` and the
  `training_device_peak` computation (~`:722`)
- Test: `tests/test_sam3_preflight.py`

`_EXTRA_BATCH_DEVICE_BYTES = 18 * GiB` is a guess of the same class as the
29 GiB constant already replaced. Preflight must prefer a cached
`MemoryMeasurement` for this hardware and fall back to the constant only when
none exists.

- [ ] **Step 1: Write the failing tests**

```python
def test_preflight_prefers_a_cached_measurement_over_the_constant(monkeypatch):
    measured = MemoryMeasurement(
        base_bytes=8 * 2**30, slope_bytes=2 * 2**30, max_batch=4,
        measured_at="", samples=(),
    )
    monkeypatch.setattr(pf, "_cached_measurement", lambda *_a, **_k: measured)
    decision = _decision(_spec(tmp_path, batch=2))
    # 8 + 2*2 = 12 GiB, far below the 29+ GiB the constants would predict.
    assert decision.budget.accelerator_peak_bytes < 16 * 2**30


def test_preflight_falls_back_to_the_constant_without_a_measurement(monkeypatch):
    monkeypatch.setattr(pf, "_cached_measurement", lambda *_a, **_k: None)
    decision = _decision(_spec(tmp_path, batch=2))
    assert decision.budget.accelerator_peak_bytes >= pf._EXTRA_BATCH_DEVICE_BYTES
```

- [ ] **Step 2-4: Fail, implement, pass.** Preflight must remain offline and
  fast: read the cache file only, never probe.
- [ ] **Step 5: Record provenance.** The refusal/diagnostic text must say
  whether the estimate came from a measurement or a fallback constant, so a
  future reader can tell which numbers are trustworthy.
- [ ] **Step 6: Commit** — `fix(training): estimate from measurement when one exists`.

---

## Task 4: Default YOLO plans to Ultralytics autobatch

**Files:**
- Modify: `src/hydra_suite/detectkit/config/training.py` (~`:716`,
  `training.batch must be -1 or a positive integer` — already permits `-1`)
- Modify: wherever `TrainingHyperParams.batch` (`contracts.py:60`) is
  defaulted for DetectKit plans
- Test: `tests/test_detectkit_training_cli.py`

Ultralytics already implements a profiling autobatch
(`ultralytics.utils.autobatch.autobatch`, `fraction=0.60`), used when
`batch=-1`. Our plans pass an explicit batch, so it never runs.

- [ ] **Step 1: Write the failing test** — a plan omitting `training.batch`
  resolves to `-1`, and one setting `batch: 16` still resolves to 16.
- [ ] **Step 2-4: Fail, implement, pass.**
- [ ] **Step 5: Document** the change in `docs/runbooks/detectkit-headless-training.md`,
  including that `-1` means "let Ultralytics measure" and that a fixed batch is
  the way to get byte-reproducible runs.
- [ ] **Step 6: Commit** — `feat(detectkit): default YOLO training to measured autobatch`.

**Ruling if you find one:** changing the default alters training dynamics for
existing plans that omit `batch`. If any equivalence fixture or golden test
depends on the current default, DO NOT change the default — add `-1` as a
documented option instead and record the decision in the ledger.

---

## Task 5: Measured probe for inference batch sizing

**Files:**
- Modify: `src/hydra_suite/utils/batch_optimizer.py`
  (`_auto_batch_size` at `:158`, `estimate_batch_size` at `:81`)
- Test: `tests/test_batch_optimizer_measured.py`

Today `_auto_batch_size` uses a hardcoded per-model MB table
(`{"yolo26n": 20, "yolo26s": 50, ...}`), `frame_memory_mb * 2.5`, a 0.7
memory fraction and a 0.8 safety factor. Nothing is measured. Add a measured
path that runs a few real inference batches once per (device, model, frame
size), caches via Task 1, and falls back to the existing heuristic when
measurement is impossible (CPU, no CUDA/MPS, probe failure).

- [ ] **Step 1: Write the failing tests** — a cached measurement is used in
  preference to the table; a probe failure falls back to today's number
  (assert the exact current value for a known input, so the fallback is
  pinned); CPU still returns 1.
- [ ] **Step 2-4: Fail, implement, pass.**
- [ ] **Step 5: Commit** — `feat(utils): measure inference batch memory instead of estimating it`.

---

## Verification

- [ ] `python -m pytest tests/ -k "sam3 or batch or preflight or memory" -q`
- [ ] `make lint-moderate`
- [ ] **GPU acceptance (coordinate first — see Global Constraints):** on a box
  with a free GPU, run the SAM3 probe end-to-end and confirm (a) the selected
  batch trains without OOM, (b) the cache file is written and a second run
  skips the probe, (c) `HYDRA_SAM3_FORCE_PROBE=1` re-probes. Record the
  measured `(base, slope)` in the commit message — that number is the point of
  the whole exercise.
- [ ] Confirm a run's logs state the resolved batch size and whether it came
  from measurement, cache, or an explicit setting.

## Known reference points

- SAM3 at batch 1, rank 16, 1008 px, 206 adapters: **7.82-7.83 GiB reserved**,
  measured identically on an RTX 6000 Ada (48 GB) and an RTX 4090 (24 GB).
  A correct probe should reproduce this base within a few hundred MiB.
- `contracts.py:217` carries the comment "batch 2 OOMs at 1008 px on a 47 GB
  card". That claim predates the `perflib_compat` change and the streaming
  dataloader, and is **unverified** against the current code. The probe is
  what should settle it — if batch 2 now fits, update that comment; if it
  still OOMs, the probe will discover that safely and the comment stands.
