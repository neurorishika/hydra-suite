# Legacy Detector Batching Vestige Removal Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Delete the "Legacy Detector Batching" UI group box and its three config keys, remove the dead two-phase detection machinery they were built for, and leave `detection_batch_size` as the single detection-batching value in the product.

**Architecture:** The legacy two-phase YOLO detection path is already dead — `InferenceRunner.run_batch_pass` replaced it and `_run_batched_detection_phase` has no callers. What survives is a control panel wired to a corpse, plus one genuinely live coupling: on TensorRT/CoreML runtimes the box's "Frame batch" spin is what actually sets `tensorrt_max_batch_size`. This plan re-sources that from `detection_batch_size`, then deletes the box, the keys, and the orphaned modules.

**Tech Stack:** Python 3, PySide6 (Qt), pytest.

## Prerequisite

**This plan MUST NOT start until `docs/superpowers/plans/2026-07-16-bgsub-inference-stage.md` has fully landed on `main`.** Verify before Task 1:

```bash
grep -rn "create_detector" src/
```
Expected: no output. If `create_detector` still exists in `src/`, the bg-sub plan has not finished — stop.

The two plans are complementary and touch disjoint concerns. The only shared file is
`core/tracking/optimization/optimizer_workers.py`, on disjoint lines: the bg-sub plan's Task 12
migrates `create_detector`/`detect_objects` (lines 367, 372) and explicitly leaves
`detect_objects_batched` (line 441) and `BatchOptimizer` (line 403) alone. Those two are this
plan's business.

## Global Constraints

- **Dependency direction:** App layers (trackerkit) may import Core/Runtime/Data/Utils; never the reverse. `core/inference` must NEVER import from `core/detectors`.
- **`core/detectors` is NOT being retired.** Per the bg-sub plan's Task 13, `_direct_obb_runtime.py` and `_runtime_artifacts.py` remain live dependencies of `core/inference` (`core/inference/runtime_artifacts.py:241`). This plan removes the *batching vestige*, not the detectors package.
- **Behavior preservation:** every change here is a no-op for a user on default config, except where explicitly called out in Task 6 (Manual batch mode for the cache builder) and Task 4 (users who had unchecked "GPU batching"). No other behavior may change.
- **Formatting:** run `make format` before committing (autopep8 → black → isort). Pre-commit hooks run black/ruff/flake8/isort automatically.
- **Tests:** `python -m pytest tests/ -m "not benchmark"`. Single file: `python -m pytest tests/test_<name>.py -v`.
- **Legacy policy:** never import from `legacy/` in `src/` or `tests/`.
- **Old configs must keep opening.** Project JSONs on disk carry `enable_yolo_batching`, `yolo_batch_size_mode`, `yolo_manual_batch_size`. After this plan those keys are unknown. Loading must ignore them silently, never raise. Task 7 tests this.

---

## Background: what is actually true today

Established by tracing the code on 2026-07-16. Later tasks depend on these facts; re-verify any that
look stale before acting on them.

| Claim | Evidence |
|---|---|
| The two-phase detection path is dead | `_run_batched_detection_phase` (`worker.py:675`) has no callers; `InferenceRunner.run_batch_pass` runs instead (`worker.py:1185`) |
| `BatchOptimizer` has one live caller | `optimizer_workers.py:403` (`DetectionCacheBuilderWorker`). The other, `detection_phase.py:318`, is unreachable |
| The box's label is wrong about Preview | `worker.py:799` excludes `preview_mode`; `BatchOptimizer` never appears in `preview_worker.py` |
| The box's label is wrong about Benchmark | `benchmarking.py:678` reads `spin_detection_batch_size`, the *new* key |
| The box's label is wrong about live tracking | `worker.py:804` reads `enable_yolo_batching` on real tracking runs |
| `enable_yolo_batching`'s only surviving effect is the prefetcher | It gates `use_batched_detection` (`worker.py:798-808`), whose live readers are `:981` (profiler metadata), `:1734` (`use_prefetcher`), `:3882` (status string) |
| **`spin_yolo_batch_size` sets the TensorRT engine batch** | `config.py:1638-1642` and `config.py:2131-2135`: `tensorrt_max_batch_size` = `spin_yolo_batch_size.value()` when `_runtime_requires_fixed_yolo_batch(compute_runtime)` |
| **`spin_tensorrt_batch` is dead** | It is only consulted when `_runtime_requires_fixed_yolo_batch()` is False, which (per `session.py:1011-1015`) means the runtime is not `tensorrt` — i.e. only when TensorRT is off and the value is irrelevant. It is also `setVisible(False)` at `detection_panel.py:971` |
| `chk_enable_tensorrt` is dead | `setVisible(False)` at `detection_panel.py:970`; never read to write config. `enable_tensorrt` is derived from the runtime (`config.py:1637`, `config.py:2200` via `legacy_detection_runtime_fields`) |
| `FramePrefetcherBackward` is dead | Only `utils/__init__.py:8,27` and `tests/test_frame_prefetcher.py`; no `src/` caller |
| `FramePrefetcher` is LIVE — do not delete | `worker.py:412`; serves the Phase 2 tracking loop when `use_prefetcher` is True |

---

## File Structure

| File | Change |
|---|---|
| `src/hydra_suite/core/tracking/ingest/detection_phase.py` | delete — orphaned two-phase detection |
| `src/hydra_suite/core/tracking/worker.py` | delete `_run_batched_detection_phase` (675-702); drop `enable_yolo_batching` term (798-808) |
| `src/hydra_suite/utils/batch_optimizer.py` | delete (conditional — see Task 1) |
| `src/hydra_suite/utils/frame_prefetcher.py` | delete `FramePrefetcherBackward` only |
| `src/hydra_suite/utils/__init__.py` | drop `FramePrefetcherBackward` export |
| `src/hydra_suite/trackerkit/gui/panels/detection_panel.py` | delete the group box, its 5 widgets, 4 handlers, `_sync_batch_policy_controls` |
| `src/hydra_suite/trackerkit/gui/orchestrators/config.py` | re-source `tensorrt_max_batch_size`; drop 3 keys from load/save/inject |
| `src/hydra_suite/trackerkit/cli_config.py` | drop 3 keys; re-source `TENSORRT_MAX_BATCH_SIZE` default |
| `src/hydra_suite/resources/configs/default.json` | drop 3 keys |
| `src/hydra_suite/resources/configs/ooceraea_biroi.json` | drop 3 keys |
| `tests/test_legacy_batching_removal.py` | create — characterization + regression |
| `tests/test_batch_optimizer.py` | delete (conditional — see Task 1) |
| `tests/test_detection_phase_nvdec_fallback.py` | delete |
| `tests/test_frame_prefetcher.py` | drop `TestFramePrefetcherBackward` |
| `tests/test_tracking_worker_helpers.py` | drop `BatchOptimizer` stub (conditional) |

---

## Task 1: Confirm `DetectionCacheBuilderWorker` stays

After the bg-sub plan, `DetectionCacheBuilderWorker` is the last legacy-YOLO consumer: it calls
`detect_objects_batched` (`optimizer_workers.py:441`) and `BatchOptimizer` (`:403`), while
`InferenceRunner` also produces detection caches. That raises the obvious question of whether it is
redundant.

**It is almost certainly not, and the codebase says so.** `optimizer_workers.py:40-44` records:

> The detection-cache builder writes via the legacy `DetectionCache` API (`mode="w"`/`add_frame`/
> `save`) and the scorer reads it via the same legacy API. The new `InferenceRunner`
> `DetectionCacheHandle` is not a drop-in replacement (different lifecycle/constructor), so bind the
> legacy class directly.

So the substitution has already been evaluated and rejected, and there are three coupled consumers of
the legacy API — the builder, the optimizer's scorer, and `TrackingPreviewWorker` (`:503`, which
reads the cache back with `mode="r"` to render preview frames). Replacing the builder means migrating
all three.

What it does, for context: the Parameter Helper runs Bayesian optimization over *tracking* params
(Kalman noise, assignment distance). Detection is invariant to those, so it is run once and cached,
then replayed through tracking for each candidate parameter set. The worker is deliberately
"Phase-1-only" — no Kalman, pose, CSV, or interpolation.

This task exists to confirm that reasoning still holds post-bg-sub and record it, not to re-litigate
it. It produces a written decision, not code.

**Files:**
- Read only. Produces: `docs/superpowers/plans/notes/2026-07-16-cache-builder-decision.md`

**Interfaces:**
- Consumes: nothing
- Produces: a decision recorded as either `KEEP` or `DELETE`, consumed by Tasks 7 and 8

- [ ] **Step 1: Find who spawns the worker**

```bash
grep -rn "DetectionCacheBuilderWorker" src/ tests/
sed -n '3415,3440p' src/hydra_suite/trackerkit/gui/orchestrators/config.py
sed -n '1015,1060p' src/hydra_suite/trackerkit/gui/main_window.py
```

Establish: which UI action reaches it, and what consumes the cache it writes.

- [ ] **Step 2: Verify the legacy-API coupling still holds**

```bash
sed -n '38,46p' src/hydra_suite/core/tracking/optimization/optimizer_workers.py
grep -rn "DetectionCache(" src/hydra_suite/core/tracking/optimization/
grep -rn "DetectionCacheHandle" src/hydra_suite/core/inference/
```

Confirm: the builder and `TrackingPreviewWorker` both construct the legacy `DetectionCache`
directly, and `DetectionCacheHandle` still has a different constructor/lifecycle. If the bg-sub work
converged the two APIs, this premise has changed — say so in the note.

- [ ] **Step 3: Apply the decision rule**

- **`KEEP`** unless the two cache APIs have converged such that `InferenceRunner` can serve the builder, the scorer, AND `TrackingPreviewWorker` with no lifecycle change.
- **`DELETE`** only on that positive evidence.

When uncertain, choose `KEEP`. A wrong `DELETE` removes a working feature; a wrong `KEEP` leaves one
file alive for another cycle. The costs are not symmetric.

Either way, **this plan does not delete the worker** — see Task 7 Step 4. A `DELETE` finding becomes
recorded follow-up work, not action here.

- [ ] **Step 4: Record the decision**

Create `docs/superpowers/plans/notes/2026-07-16-cache-builder-decision.md` with:

```markdown
# DetectionCacheBuilderWorker decision

**Decision:** KEEP | DELETE

**Spawned from:** <file:line of the UI action>
**Cache consumed by:** <what reads the cache>
**InferenceRunner equivalent exists:** yes | no — <evidence>

**Reasoning:** <2-4 sentences>

**Consequences for this plan:**
- Task 7 deletes utils/batch_optimizer.py: yes | no
- Task 7 deletes tests/test_batch_optimizer.py: yes | no
- Task 7 drops the BatchOptimizer stub in tests/test_tracking_worker_helpers.py:81: yes | no
```

- [ ] **Step 5: Commit**

```bash
git add docs/superpowers/plans/notes/2026-07-16-cache-builder-decision.md
git commit -m "docs: record DetectionCacheBuilderWorker keep/delete decision

After the bg-sub migration it is the last legacy-YOLO consumer, while
InferenceRunner already owns detection caching. Recorded so Tasks 7-8 have
a single referenceable answer."
```

---

## Task 2: Delete the dead two-phase detection machinery

`run_batched_detection_phase` is reachable only via `TrackingWorker._run_batched_detection_phase`,
which nothing calls. Both go. This is provably behavior-preserving: unreachable code cannot run.

**Files:**
- Delete: `src/hydra_suite/core/tracking/ingest/detection_phase.py`
- Delete: `tests/test_detection_phase_nvdec_fallback.py`
- Modify: `src/hydra_suite/core/tracking/worker.py:675-702`

**Interfaces:**
- Consumes: nothing from Task 1
- Produces: `worker.py` no longer references `detection_phase`

- [ ] **Step 1: Prove there are no callers**

```bash
grep -rn "_run_batched_detection_phase\|run_batched_detection_phase\|detection_phase" src/ tests/
```

Expected: only the definition at `worker.py:675`, the import inside it at `worker.py:686-688`, the
module itself, and `tests/test_detection_phase_nvdec_fallback.py`.

**If anything else appears, STOP** — a caller exists and this plan's premise is wrong for that path.
Record what you found and escalate rather than deleting.

Note: `data/dataset_generation.py:528` defines its own local `_run_batched_detection` — a different
function with a similar name. It is unrelated. Do not touch it.

- [ ] **Step 2: Delete the method**

In `src/hydra_suite/core/tracking/worker.py`, delete lines 675-702 in their entirety — the whole
`_run_batched_detection_phase` method, from its `def` through the closing paren of its `return`.

- [ ] **Step 3: Delete the module and its test**

```bash
git rm src/hydra_suite/core/tracking/ingest/detection_phase.py
git rm tests/test_detection_phase_nvdec_fallback.py
```

- [ ] **Step 4: Verify the package still imports**

```bash
python -c "import hydra_suite.core.tracking.worker; print('ok')"
grep -rn "detection_phase" src/ tests/
```
Expected: `ok`; second command produces no output.

- [ ] **Step 5: Run the full suite**

```bash
python -m pytest tests/ -m "not benchmark" -q
```
Expected: no new failures.

- [ ] **Step 6: Commit**

```bash
make format
git add -A src/hydra_suite/core/tracking/ tests/
git commit -m "chore(tracking): delete the orphaned two-phase detection path

InferenceRunner.run_batch_pass replaced the legacy batched YOLO prepass, but
detection_phase.py and TrackingWorker._run_batched_detection_phase were left
behind with no callers. Unreachable since the inference redesign."
```

---

## Task 3: Re-source `tensorrt_max_batch_size` from `detection_batch_size`

**This is the one genuinely load-bearing coupling in the box.** On TensorRT/CoreML runtimes,
`tensorrt_max_batch_size` is read from `spin_yolo_batch_size` — the "Frame batch" widget inside a box
labelled "does not affect live tracking speed". It feeds `TENSORRT_MAX_BATCH_SIZE`, which flows to
`TENSORRT_BUILD_BATCH_SIZE` (`tracking.py:4089`, `tracking_cache.py:189`) and the engine-cache key.

Re-sourcing it from `detection_batch_size` is both the unblocking move and the semantically correct
one: a TensorRT engine's max batch should match the detection batch it will actually be fed.

**Files:**
- Modify: `src/hydra_suite/trackerkit/gui/orchestrators/config.py:1638-1642, 2131-2135`
- Modify: `src/hydra_suite/trackerkit/cli_config.py:626-631`
- Test: `tests/test_legacy_batching_removal.py` (create)

**Interfaces:**
- Consumes: nothing from Tasks 1-2
- Produces: `tensorrt_max_batch_size` sourced from `spin_detection_batch_size` on fixed runtimes; `spin_yolo_batch_size` no longer read by config persistence

- [ ] **Step 1: Write the failing test**

Create `tests/test_legacy_batching_removal.py`:

```python
"""Regression tests for the legacy detector-batching vestige removal.

The 'Legacy Detector Batching' group box claimed to affect only cache build /
preview / benchmark. It was wrong in three directions, and its 'Frame batch'
spin was in fact the TensorRT engine batch control on fixed runtimes. These
tests pin the corrected wiring.
"""

def test_trt_batch_follows_detection_batch_on_fixed_runtime():
    """On TensorRT/CoreML the engine's max batch must equal the detection
    batch it will actually be fed -- previously it came from the legacy
    'Frame batch' spin inside a box that claimed not to affect tracking."""
    from hydra_suite.trackerkit.gui.orchestrators.config import (
        resolve_tensorrt_max_batch_size,
    )

    assert (
        resolve_tensorrt_max_batch_size(
            detection_batch_size=8, fixed_runtime=True
        )
        == 8
    )


def test_trt_batch_is_one_on_non_fixed_runtime():
    """When the runtime is not TensorRT/CoreML, TensorRT is off and the value
    is inert. Pin it to a stable 1 rather than a stale widget value."""
    from hydra_suite.trackerkit.gui.orchestrators.config import (
        resolve_tensorrt_max_batch_size,
    )

    assert (
        resolve_tensorrt_max_batch_size(
            detection_batch_size=8, fixed_runtime=False
        )
        == 1
    )


def test_trt_batch_clamps_to_at_least_one():
    from hydra_suite.trackerkit.gui.orchestrators.config import (
        resolve_tensorrt_max_batch_size,
    )

    assert (
        resolve_tensorrt_max_batch_size(
            detection_batch_size=0, fixed_runtime=True
        )
        == 1
    )
```

- [ ] **Step 2: Run test to verify it fails**

```bash
python -m pytest tests/test_legacy_batching_removal.py -v
```
Expected: FAIL — `ImportError: cannot import name 'resolve_tensorrt_max_batch_size'`.

- [ ] **Step 3: Add the helper**

Both call sites duplicate the same conditional today. Extract it once. Add to
`src/hydra_suite/trackerkit/gui/orchestrators/config.py`, at module level above the class:

```python
def resolve_tensorrt_max_batch_size(
    *, detection_batch_size: int, fixed_runtime: bool
) -> int:
    """Return the TensorRT engine's max batch size.

    On a fixed runtime (TensorRT, or gpu_fast-on-CoreML) this must match the
    detection batch the engine will actually be fed, so it tracks
    detection_batch_size -- the single detection-batching value in the UI.

    On any other runtime TensorRT is off (enable_tensorrt is derived from the
    runtime; see cli_config.legacy_detection_runtime_fields) and this value is
    inert, so it is pinned to 1 rather than carrying a stale widget value into
    the engine-cache key.

    Previously the fixed-runtime branch read the legacy 'Frame batch' spin from
    the 'Legacy Detector Batching' box -- a control whose own help text claimed
    it did not affect live tracking.
    """
    if not fixed_runtime:
        return 1
    return max(1, int(detection_batch_size or 1))
```

- [ ] **Step 4: Run test to verify it passes**

```bash
python -m pytest tests/test_legacy_batching_removal.py -v
```
Expected: 3 passed.

- [ ] **Step 5: Use the helper at both call sites**

In `src/hydra_suite/trackerkit/gui/orchestrators/config.py`, replace lines 1638-1642:

```python
                "tensorrt_max_batch_size": (
                    self._panels.detection.spin_yolo_batch_size.value()
                    if self._mw._runtime_requires_fixed_yolo_batch(compute_runtime)
                    else self._panels.detection.spin_tensorrt_batch.value()
                ),
```

with:

```python
                "tensorrt_max_batch_size": resolve_tensorrt_max_batch_size(
                    detection_batch_size=self._panels.detection.spin_detection_batch_size.value(),
                    fixed_runtime=self._mw._runtime_requires_fixed_yolo_batch(
                        compute_runtime
                    ),
                ),
```

Then replace lines 2131-2135:

```python
        trt_batch_size = (
            self._panels.detection.spin_yolo_batch_size.value()
            if self._mw._runtime_requires_fixed_yolo_batch(compute_runtime)
            else self._panels.detection.spin_tensorrt_batch.value()
        )
```

with:

```python
        trt_batch_size = resolve_tensorrt_max_batch_size(
            detection_batch_size=self._panels.detection.spin_detection_batch_size.value(),
            fixed_runtime=self._mw._runtime_requires_fixed_yolo_batch(compute_runtime),
        )
```

- [ ] **Step 6: Re-source the CLI default**

In `src/hydra_suite/trackerkit/cli_config.py`, replace lines 626-631:

```python
        "TENSORRT_MAX_BATCH_SIZE": int(
            _cfg_get(
                cfg,
                "tensorrt_max_batch_size",
                default=max(1, int(advanced["yolo_manual_batch_size"])),
            )
        ),
```

with:

```python
        # Defaults to the detection batch the engine will actually be fed.
        # Previously defaulted to yolo_manual_batch_size, a legacy-detector key.
        "TENSORRT_MAX_BATCH_SIZE": int(
            _cfg_get(
                cfg,
                "tensorrt_max_batch_size",
                default=max(1, int(_cfg_get(cfg, "detection_batch_size", default=1) or 1)),
            )
        ),
```

- [ ] **Step 7: Verify no config site still reads the legacy spin**

```bash
grep -rn "spin_yolo_batch_size\|spin_tensorrt_batch" src/hydra_suite/trackerkit/gui/orchestrators/
```
Expected: no output.

- [ ] **Step 8: Run the suite**

```bash
python -m pytest tests/ -m "not benchmark" -q
```
Expected: no new failures.

- [ ] **Step 9: Commit**

```bash
make format
git add tests/test_legacy_batching_removal.py src/hydra_suite/trackerkit/gui/orchestrators/config.py src/hydra_suite/trackerkit/cli_config.py
git commit -m "fix(trackerkit): TensorRT engine batch follows detection_batch_size

On TensorRT/CoreML runtimes tensorrt_max_batch_size was read from
spin_yolo_batch_size -- the 'Frame batch' spin inside the 'Legacy Detector
Batching' box, whose own help text said it did not affect live tracking. It
flows to TENSORRT_BUILD_BATCH_SIZE and the engine-cache key, so it did.

The engine's max batch should match the detection batch it is fed, which is
detection_batch_size. On non-fixed runtimes TensorRT is off and the value is
inert; pin it to 1 rather than carry a stale widget value into the cache key.

Extracts the duplicated conditional into resolve_tensorrt_max_batch_size."
```

---

## Task 4: Drop `enable_yolo_batching` from the `use_batched_detection` conjunction

`use_batched_detection` (`worker.py:798-808`) is a six-term conjunction. Only one term is the config
key being deleted. Removing that term leaves the other five intact and still driving the prefetcher,
profiler metadata, and status text correctly.

Since `enable_yolo_batching` defaulted to `True`, this is an exact no-op for any user on defaults.
Users who explicitly unchecked "GPU batching" lose the frame prefetcher on non-realtime OBB runs —
which they were only getting via a control mislabelled as a batching toggle.

**Files:**
- Modify: `src/hydra_suite/core/tracking/worker.py:798-808`
- Test: `tests/test_legacy_batching_removal.py` (append)

**Interfaces:**
- Consumes: Task 2's deletion (no code dependency)
- Produces: `worker.py` no longer reads `enable_yolo_batching`

- [ ] **Step 1: Read the current conjunction and its readers**

```bash
sed -n '795,815p' src/hydra_suite/core/tracking/worker.py
sed -n '1729,1740p' src/hydra_suite/core/tracking/worker.py
```

Confirm the shape before editing. The readers are `:981` (profiler `batched_detection=`), `:1734`
(`use_prefetcher`), `:3882` (status string). None of them are removed by this task.

- [ ] **Step 2: Write the characterization test**

Append to `tests/test_legacy_batching_removal.py`:

```python
def test_worker_module_no_longer_reads_enable_yolo_batching():
    """The key is gone from config; worker.py must not read it back.

    A stale advanced_config.get('enable_yolo_batching') would silently read
    False for every project saved after this change, flipping the prefetcher
    on for everyone -- the exact regression this plan avoids.
    """
    from pathlib import Path

    import hydra_suite.core.tracking.worker as worker_mod

    source = Path(worker_mod.__file__).read_text()
    assert "enable_yolo_batching" not in source


def test_no_src_module_reads_the_removed_batching_keys():
    """Guard against a re-introduction anywhere in src/.

    utils/batch_optimizer.py is the one sanctioned exception: it still reads
    all three keys via .get() with defaults, and survives this plan because
    DetectionCacheBuilderWorker is its live caller. With the keys gone from
    config it falls back to auto GPU-memory sizing -- see Task 7. It dies with
    core/detectors' YOLO half, not here.
    """
    from pathlib import Path

    import hydra_suite

    root = Path(hydra_suite.__file__).parent
    allowed = {root / "utils" / "batch_optimizer.py"}
    removed = ("enable_yolo_batching", "yolo_batch_size_mode", "yolo_manual_batch_size")
    offenders = []
    for path in root.rglob("*.py"):
        if path in allowed:
            continue
        text = path.read_text()
        for key in removed:
            if key in text:
                offenders.append(f"{path}: {key}")
    assert offenders == [], f"removed keys still referenced: {offenders}"
```

- [ ] **Step 3: Run test to verify it fails**

```bash
python -m pytest tests/test_legacy_batching_removal.py -k enable_yolo_batching -v
```
Expected: FAIL — the string is still present in `worker.py`.

Note `test_no_src_module_reads_the_removed_batching_keys` will also fail here and stays failing until
Task 7. That is expected and intended: it is the plan's completion gate.

- [ ] **Step 4: Drop the term**

In `src/hydra_suite/core/tracking/worker.py`, replace lines 798-808:

```python
        use_batched_detection = (
            not self.preview_mode  # Not preview mode
            and not self.backward_mode  # Not backward mode (uses cache)
            and detection_method == "yolo_obb"  # Only YOLO benefits from batching
            and not realtime_tracking_mode_requested
            and advanced_config.get(
                "enable_yolo_batching", True
            )  # Batching enabled in config
            and self.detection_cache_path
            is not None  # Need cache path for two-phase approach
        )
```

with:

```python
        # NOTE: this no longer selects a detection architecture -- the legacy
        # two-phase prepass it once gated is gone, and InferenceRunner owns the
        # batch pass. It survives as the condition for the Phase 2 frame
        # prefetcher (see use_prefetcher below), profiler metadata, and the
        # status string. The old `enable_yolo_batching` term was dropped with
        # the legacy batching config; it defaulted to True, so this is
        # unchanged for any config that did not explicitly disable it.
        use_batched_detection = (
            not self.preview_mode  # Not preview mode
            and not self.backward_mode  # Not backward mode (uses cache)
            and detection_method == "yolo_obb"  # Only YOLO benefits from batching
            and not realtime_tracking_mode_requested
            and self.detection_cache_path
            is not None  # Need cache path for two-phase approach
        )
```

Leave `advanced_config` (assigned at `:790`) alone — other code reads it.

- [ ] **Step 5: Run tests to verify the worker test passes**

```bash
python -m pytest tests/test_legacy_batching_removal.py -k enable_yolo_batching -v
python -m pytest tests/ -m "not benchmark" -q
```
Expected: `test_worker_module_no_longer_reads_enable_yolo_batching` PASSES; no new failures
elsewhere. `test_no_src_module_reads_the_removed_batching_keys` still fails — expected until Task 7.

- [ ] **Step 6: Commit**

```bash
make format
git add src/hydra_suite/core/tracking/worker.py tests/test_legacy_batching_removal.py
git commit -m "refactor(tracking): drop enable_yolo_batching from use_batched_detection

The flag once selected a detection architecture (two-phase batched vs
frame-by-frame). That path is gone -- InferenceRunner owns the batch pass --
so its only surviving effect was inverting the Phase 2 frame prefetcher via a
checkbox labelled 'GPU batching'.

Only that one term of the six-term conjunction is removed; the rest still
drive the prefetcher, profiler metadata, and status text. The key defaulted
to True, so behavior is unchanged for any config that did not disable it."
```

---

## Task 5: Remove config/CLI persistence of the group-box widgets

> **Reordered during execution (2026-07-17):** this task (config/CLI consumer removal) now runs
> BEFORE the panel-widget deletion (Task 6). Reason: the orchestrator reads the widgets — including a
> TensorRT loader block that the original plan missed — so deleting the widgets first would leave the
> orchestrator referencing deleted attributes (an AttributeError on project load, and a broken
> intermediate commit). Removing every consumer first keeps each commit coherent. The augmented
> TensorRT-loader removal (Step 2 below) was also added then.

Remove every orchestrator and CLI reference to the group-box widgets. After this task, the ONLY code
mentioning `chk_enable_yolo_batching`, `combo_yolo_batch_mode`, `spin_yolo_batch_size`,
`chk_enable_tensorrt`, `spin_tensorrt_batch`, and their labels is the panel construction in
`detection_panel.py` (deleted in Task 6). `tensorrt_max_batch_size` stays as a saved config key — it
is written via `resolve_tensorrt_max_batch_size` (Task 3); only the load-into-a-widget is removed.

**Files:**
- Modify: `src/hydra_suite/trackerkit/gui/orchestrators/config.py` (load blocks, save block, inject block)
- Modify: `src/hydra_suite/trackerkit/cli_config.py` (the three-key translation block)

**Interfaces:**
- Consumes: Task 3's `resolve_tensorrt_max_batch_size` (the save site already uses it; this task removes the remaining direct widget reads/writes)
- Produces: no orchestrator or CLI code reads/writes `enable_yolo_batching` / `yolo_batch_size_mode` / `yolo_manual_batch_size`, and no orchestrator code reads/writes the group-box widgets. Only `detection_panel.py` still names them.

**Locating targets:** the line numbers below are approximate (a bg-sub migration shifted them). Find
each block by content — grep for `spin_yolo_batch_size`, `spin_tensorrt_batch`, `enable_yolo_batching`
in `config.py` and `cli_config.py`.

- [ ] **Step 1: Remove the YOLO-batching load block**

In `src/hydra_suite/trackerkit/gui/orchestrators/config.py`, delete the `# YOLO Batching settings`
block — the three statements setting `chk_enable_yolo_batching.setChecked(...)`,
`combo_yolo_batch_mode.setCurrentIndex(...)`, and `spin_yolo_batch_size.setValue(...)` (was ~532-542,
now ~555-565). Keep the `# Live Detection Batching` block that follows.

- [ ] **Step 2: Remove the orphaned TensorRT loader block (added in reorder)**

Immediately above the YOLO-batching load block is a TensorRT loader that pushes the kept
`tensorrt_max_batch_size` value into widgets Task 6 deletes. Delete this whole block (the comment plus
all three widget calls — `spin_tensorrt_batch.setValue`, `spin_tensorrt_batch.setEnabled`,
`lbl_tensorrt_batch.setEnabled`, was ~537-553):

```python
        # TensorRT batch size is still configurable (runtime-derived usage).
        self._panels.detection.spin_tensorrt_batch.setValue(
            get_cfg("tensorrt_max_batch_size", default=16)
        )
        self._panels.detection.spin_tensorrt_batch.setEnabled(
            bool(
                legacy_detection_runtime_fields(self._mw._selected_compute_runtime())[
                    "enable_tensorrt"
                ]
            )
        )
        self._panels.detection.lbl_tensorrt_batch.setEnabled(
            bool(
                legacy_detection_runtime_fields(self._mw._selected_compute_runtime())[
                    "enable_tensorrt"
                ]
            )
        )
```

This is a UI-display loader only; `tensorrt_max_batch_size` is still saved via
`resolve_tensorrt_max_batch_size` and still consumed by the engine. If removing this block leaves
`legacy_detection_runtime_fields` or `get_cfg` unused in the surrounding method, leave them — they are
used elsewhere in the same method; verify with a grep before removing any import.

- [ ] **Step 3: Remove the save block**

Delete these entries from the `cfg.update({...})` (was ~1643-1651):

```python
                # YOLO Batching
                "enable_yolo_batching": self._panels.detection.chk_enable_yolo_batching.isChecked(),
                "yolo_batch_size_mode": (
                    "auto"
                    if self._panels.detection.combo_yolo_batch_mode.currentIndex() == 0
                    else "manual"
                ),
                "yolo_manual_batch_size": self._panels.detection.spin_yolo_batch_size.value(),
```

Also update the now-stale comment above the `enable_tensorrt` cfg entry from:

```python
                # TensorRT (legacy detector: cache build / preview / benchmark only)
```

to:

```python
                # TensorRT: derived from the selected runtime, retained for
                # legacy config round-tripping and the engine-cache key.
```

- [ ] **Step 4: Remove the advanced_config injection**

Delete the three `advanced_config[...]` assignments for the YOLO batching keys (was ~2064-2076):

```python
        # YOLO Batching settings from UI (overrides advanced_config defaults)
        advanced_config = self._mw.advanced_config.copy()
        advanced_config["enable_yolo_batching"] = (
            self._panels.detection.chk_enable_yolo_batching.isChecked()
        )
        advanced_config["yolo_batch_size_mode"] = (
            "auto"
            if self._panels.detection.combo_yolo_batch_mode.currentIndex() == 0
            else "manual"
        )
        advanced_config["yolo_manual_batch_size"] = (
            self._panels.detection.spin_yolo_batch_size.value()
        )
```

**Keep the `advanced_config = self._mw.advanced_config.copy()` line** — the assignments below it
(`detection_batch_size`, `yolo_seq_individual_batch_size`, `video_show_pose`, …) still need it.
The result should read:

```python
        advanced_config = self._mw.advanced_config.copy()
        advanced_config["detection_batch_size"] = (
            self._panels.detection.spin_detection_batch_size.value()
        )
```

- [ ] **Step 5: Remove the CLI translation block**

In `src/hydra_suite/trackerkit/cli_config.py`, delete the three `advanced[...]` assignments (was
~310-330) for `enable_yolo_batching`, `yolo_batch_size_mode`, and `yolo_manual_batch_size`. Keep
`advanced["yolo_seq_individual_batch_size"]` immediately after them.

- [ ] **Step 6: Verify**

```bash
source /Users/neurorishika/miniforge3/etc/profile.d/conda.sh && conda activate hydra-mps
# 1) the three keys are gone from the trackerkit code paths (batch_optimizer.py is allowed elsewhere)
grep -rn "enable_yolo_batching\|yolo_batch_size_mode\|yolo_manual_batch_size" src/hydra_suite/trackerkit/
# 2) the orchestrator no longer reads the widgets this task handles
grep -rn "spin_tensorrt_batch\|lbl_tensorrt_batch\|chk_enable_yolo_batching\|combo_yolo_batch_mode\|spin_yolo_batch_size" src/hydra_suite/trackerkit/gui/orchestrators/
# 3) modules still import
python -c "import hydra_suite.trackerkit.gui.orchestrators.config; import hydra_suite.trackerkit.cli_config; print('ok')"
```
Expected: (1) no output; (2) no output; (3) `ok`.

Then a targeted test run (NOT the full suite — it hangs in this env; see below):

```bash
python -m pytest tests/test_legacy_batching_removal.py tests/test_trackerkit_headless_tracking.py \
  --timeout=90 --timeout-method=thread -p no:cacheprovider -q
```
Expected: the `resolve_tensorrt_max_batch_size` tests pass; `test_no_src_module_reads_the_removed_batching_keys` still fails (Task 7 gate). No NEW failures beyond the known pre-existing ones.

- [ ] **Step 7: Commit**

```bash
make format
git add src/hydra_suite/trackerkit/gui/orchestrators/config.py src/hydra_suite/trackerkit/cli_config.py
git commit -m "chore(trackerkit): drop group-box config persistence

Removes every orchestrator/CLI read and write of enable_yolo_batching /
yolo_batch_size_mode / yolo_manual_batch_size, plus the TensorRT loader block
that pushed tensorrt_max_batch_size into the (now-unread) spin_tensorrt_batch
widget. tensorrt_max_batch_size is still saved via resolve_tensorrt_max_batch_size
and consumed by the engine; only the load-into-a-widget is gone. Old project
configs carrying the three keys still load -- unknown keys are ignored.

After this, only detection_panel.py references the group-box widgets, so the
panel deletion that follows is a clean, self-contained removal."
```

---

## Task 6: Delete the group box from the detection panel

> **Reordered during execution (2026-07-17):** now runs AFTER Task 5 (config/CLI consumer removal).
> Because Task 5 removed every orchestrator reference to these widgets, the Step 1 grep below now
> finds hits only inside `detection_panel.py` — the panel deletion is self-contained.

Delete the box, its widgets, its four handlers, and `_sync_batch_policy_controls`.

**Files:**
- Modify: `src/hydra_suite/trackerkit/gui/panels/detection_panel.py` (group-box construction ~840-982; handlers ~1287-1379)

**Interfaces:**
- Consumes: Task 5's config/CLI removal (no external code references these widgets)
- Produces: `detection_panel` exposes no `chk_enable_yolo_batching`, `combo_yolo_batch_mode`, `spin_yolo_batch_size`, `chk_enable_tensorrt`, `spin_tensorrt_batch`, `lbl_yolo_batch_mode`, `lbl_yolo_batch_size`, `lbl_tensorrt_batch`, `g_gpu_accel`

- [ ] **Step 1: Enumerate every reference before deleting**

```bash
grep -rn "g_gpu_accel\|chk_enable_yolo_batching\|combo_yolo_batch_mode\|spin_yolo_batch_size\|chk_enable_tensorrt\|spin_tensorrt_batch\|lbl_yolo_batch_mode\|lbl_yolo_batch_size\|lbl_tensorrt_batch\|_sync_batch_policy_controls\|_on_yolo_batching_toggled\|_on_yolo_batch_mode_changed\|_on_yolo_manual_batch_size_changed\|_on_tensorrt_toggled" src/ tests/
```

After Task 5 every hit must be inside `detection_panel.py` (group-box construction or the handlers
below). **If a hit appears in any OTHER file, STOP and record it** — an unmapped consumer exists that
Task 5 should have cleared.

Also confirm `benchmarking.py` reads `spin_detection_batch_size` (the surviving control), not
`spin_yolo_batch_size`.

- [ ] **Step 2: Delete the group-box construction block**

In `src/hydra_suite/trackerkit/gui/panels/detection_panel.py`, delete the block from the `# ====`
comment banner introducing "Legacy Detector Batching" through the final `self._sync_batch_policy_controls()`
call (was ~840-982). Locate it by the banner and the trailing sync call, not by line number.

That block ends immediately before:

```python
        # Add pages to stack
        self.stack_detection.addWidget(page_bg)
```

Which must remain.

- [ ] **Step 3: Replace the deleted `_sync_batch_policy_controls()` call**

The deleted block's final line called `self._sync_batch_policy_controls()`, which ended by calling
`self._sync_live_detection_batch_controls()`. That second call is still needed — it initializes the
surviving Frame-batch-size control. Immediately before `# Add pages to stack`, add:

```python
        self._sync_live_detection_batch_controls()
```

- [ ] **Step 4: Delete the four dead handlers and the sync method**

Delete these methods entirely from `detection_panel.py` (locate by name):

- `_on_yolo_batching_toggled`
- `_on_yolo_manual_batch_size_changed`
- `_on_yolo_batch_mode_changed`
- `_on_tensorrt_toggled`
- `_sync_batch_policy_controls`

Keep `_on_detection_batch_size_changed` and `_sync_live_detection_batch_controls` — both serve the
surviving control.

- [ ] **Step 5: Find and fix orphaned callers of `_sync_batch_policy_controls`**

```bash
grep -rn "_sync_batch_policy_controls" src/ tests/
```

Any remaining caller must become `_sync_live_detection_batch_controls`, which is what the deleted
method delegated to for the surviving control. Fix each hit.

- [ ] **Step 6: Verify the panel constructs**

```bash
source /Users/neurorishika/miniforge3/etc/profile.d/conda.sh && conda activate hydra-mps
python -c "import hydra_suite.trackerkit.gui.panels.detection_panel; print('ok')"
# targeted, NOT the full suite (it hangs in this env)
python -m pytest tests/test_trackerkit_preview_worker.py tests/test_trackerkit_benchmarking.py \
  --timeout=90 --timeout-method=thread -p no:cacheprovider -q
```
Expected: `ok`; no NEW failures beyond the known pre-existing ones.

- [ ] **Step 7: Launch the GUI and confirm the box is gone** (manual — controller/human)

```bash
trackerkit
```

Navigate to the detection settings, YOLO page. Confirm: the "Legacy Detector Batching" box is absent;
"Live Detection Batching" with its "Frame batch size" spin remains and is interactive; no traceback in
the console. Close the app.

- [ ] **Step 8: Commit**

```bash
make format
git add src/hydra_suite/trackerkit/gui/panels/detection_panel.py
git commit -m "feat(trackerkit): delete the Legacy Detector Batching group box

The box claimed to affect only cache build / preview / benchmark. It was
wrong three ways: preview is excluded at worker.py:799, benchmark reads
detection_batch_size, and it did affect live tracking. Its 'GPU batching'
checkbox had become an inverted frame-prefetcher toggle, and its TensorRT
controls were invisible (setVisible(False)) and never read.

detection_batch_size in 'Live Detection Batching' is now the single
detection-batching control."
```

---

## Task 7: Delete the remaining dead modules and config defaults

**Files:**
- Modify: `src/hydra_suite/resources/configs/default.json:56-58`
- Modify: `src/hydra_suite/resources/configs/ooceraea_biroi.json:70-72`
- Modify: `src/hydra_suite/utils/frame_prefetcher.py:420-478`
- Modify: `src/hydra_suite/utils/__init__.py:8,27`
- Modify: `tests/test_frame_prefetcher.py`
- Conditional on Task 1's decision: `src/hydra_suite/utils/batch_optimizer.py`, `tests/test_batch_optimizer.py`, `tests/test_tracking_worker_helpers.py:80-81`

**Interfaces:**
- Consumes: Task 1's decision; Tasks 4-6's removals
- Produces: `test_no_src_module_reads_the_removed_batching_keys` passes

- [ ] **Step 1: Remove the keys from both bundled configs**

From `src/hydra_suite/resources/configs/default.json`, delete these three lines:

```json
  "enable_yolo_batching": true,
  "yolo_batch_size_mode": "auto",
  "yolo_manual_batch_size": 16,
```

From `src/hydra_suite/resources/configs/ooceraea_biroi.json`, delete these three lines:

```json
  "enable_yolo_batching": true,
  "yolo_batch_size_mode": "manual",
  "yolo_manual_batch_size": 16,
```

Keep `enable_tensorrt` and `tensorrt_max_batch_size` in both — still live per Task 3.

- [ ] **Step 2: Verify both files are still valid JSON**

```bash
python -c "import json; json.load(open('src/hydra_suite/resources/configs/default.json')); print('default ok')"
python -c "import json; json.load(open('src/hydra_suite/resources/configs/ooceraea_biroi.json')); print('biroi ok')"
```
Expected: `default ok`, `biroi ok`. A trailing-comma error here is the likeliest slip.

- [ ] **Step 3: Delete `FramePrefetcherBackward`**

**Delete only `FramePrefetcherBackward`.** `FramePrefetcher`, `SparseFramePrefetcher`, and
`SequentialScanPrefetcher` are all live (`worker.py:412`, `crops_worker.py:920,927`). Confirm first:

```bash
grep -rn "FramePrefetcherBackward" src/ tests/
```
Expected: only `utils/frame_prefetcher.py`, `utils/__init__.py:8,27`, and `tests/test_frame_prefetcher.py`.

In `src/hydra_suite/utils/frame_prefetcher.py`, delete lines 420-478 — the entire
`class FramePrefetcherBackward(FramePrefetcher)`.

In `src/hydra_suite/utils/__init__.py`, change line 8 from:

```python
from .frame_prefetcher import FramePrefetcher, FramePrefetcherBackward
```

to:

```python
from .frame_prefetcher import FramePrefetcher
```

and delete `"FramePrefetcherBackward",` from `__all__` (line 27).

In `tests/test_frame_prefetcher.py`, delete the entire `class TestFramePrefetcherBackward` (starts
line 252) and remove `FramePrefetcherBackward` from the import at lines 19-23.

- [ ] **Step 4: Apply Task 1's decision on `BatchOptimizer`**

Read `docs/superpowers/plans/notes/2026-07-16-cache-builder-decision.md`.

**If the decision was `DELETE`:** deleting `DetectionCacheBuilderWorker` is out of scope for this
plan — it is a feature removal, not a batching cleanup. Record it as follow-up work and treat
`BatchOptimizer` as `KEEP` here:

```bash
cat >> docs/superpowers/plans/notes/2026-07-16-cache-builder-decision.md <<'EOF'

## Follow-up

DetectionCacheBuilderWorker was found redundant, but removing it is a feature
deletion rather than part of the batching-vestige cleanup. It needs its own
plan. utils/batch_optimizer.py stays alive until then as its only remaining
consumer.
EOF
```

**If the decision was `KEEP`:** `utils/batch_optimizer.py` stays. It has one live caller
(`optimizer_workers.py:403`) and will die with `core/detectors`' YOLO half.

**In both branches, `utils/batch_optimizer.py` and `tests/test_batch_optimizer.py` are NOT deleted by
this plan.**

Note the consequence, which is intended: with the three keys gone from config, `BatchOptimizer`
falls back to its own defaults — `enable_yolo_batching=True` (`batch_optimizer.py:118`) and
`yolo_batch_size_mode="auto"` (`:152`). The cache builder therefore gets GPU-memory-derived sizing.
Users who had set Manual lose that for cache builds only. This is the one intended behavior change in
the plan.

- [ ] **Step 5: Run the completion gate**

```bash
python -m pytest tests/test_legacy_batching_removal.py -v
```
Expected: all pass, including `test_no_src_module_reads_the_removed_batching_keys`.

If it still fails, its assertion message names the offending `file: key`. Fix each and re-run.

- [ ] **Step 6: Run the full suite**

```bash
python -m pytest tests/ -m "not benchmark" -q
```
Expected: no new failures.

- [ ] **Step 7: Commit**

```bash
make format
git add -A src/hydra_suite/resources/configs/ src/hydra_suite/utils/ tests/ docs/superpowers/plans/notes/
git commit -m "chore: drop legacy batching defaults and the dead backward prefetcher

Removes the three legacy batching keys from the bundled configs, and deletes
FramePrefetcherBackward, which had no caller in src/ (the forward
FramePrefetcher, SparseFramePrefetcher, and SequentialScanPrefetcher are all
live and untouched).

utils/batch_optimizer.py survives: DetectionCacheBuilderWorker still uses it.
With the config keys gone it falls back to auto GPU-memory sizing for cache
builds."
```

---

## Task 8: End-to-end verification

Deletion-heavy changes in a 19k-line worker are exactly where unit tests give false confidence. This
task drives the real app.

**Files:**
- No source changes. Produces a verification record.

**Interfaces:**
- Consumes: Tasks 2-7
- Produces: confirmation that tracking behavior is unchanged

- [ ] **Step 1: Confirm no reference survives anywhere**

```bash
grep -rn "detection_phase\|FramePrefetcherBackward\|_sync_batch_policy_controls\|g_gpu_accel\|chk_enable_yolo_batching\|combo_yolo_batch_mode\|spin_yolo_batch_size\|chk_enable_tensorrt\|spin_tensorrt_batch" src/ tests/
```
Expected: no output.

```bash
grep -rn "enable_yolo_batching\|yolo_batch_size_mode\|yolo_manual_batch_size" src/ tests/
```
Expected: hits **only** in `src/hydra_suite/utils/batch_optimizer.py` and
`tests/test_batch_optimizer.py`, plus the sanctioned-exception comment in
`tests/test_legacy_batching_removal.py`. Anything else is a leak — fix it before proceeding.

- [ ] **Step 2: Confirm every kit still imports**

```bash
python -c "import hydra_suite"
for kit in trackerkit posekit classkit refinekit detectkit filterkit; do
  python -c "import hydra_suite.$kit" || echo "FAILED: $kit"
done
```
Expected: no ImportError, no `FAILED` lines.

- [ ] **Step 3: Confirm an old project config still opens**

This is the backward-compatibility gate. Build a config carrying the removed keys and load it:

```bash
python - <<'EOF'
import json, tempfile, pathlib
from hydra_suite.trackerkit.cli_config import legacy_detection_runtime_fields

# A config as saved by the previous release, carrying the three dead keys.
cfg = {
    "compute_runtime": "cpu",
    "enable_yolo_batching": False,
    "yolo_batch_size_mode": "manual",
    "yolo_manual_batch_size": 32,
    "detection_batch_size": 4,
    "enable_tensorrt": False,
    "tensorrt_max_batch_size": 16,
}
p = pathlib.Path(tempfile.mkdtemp()) / "old_project.json"
p.write_text(json.dumps(cfg))
loaded = json.loads(p.read_text())
assert loaded["enable_yolo_batching"] is False  # still on disk, simply unread
print("legacy keys present on disk and ignored: ok")
print("runtime fields still derive:", legacy_detection_runtime_fields("cpu"))
EOF
```
Expected: both lines print; no exception.

- [ ] **Step 4: Load a real pre-existing project in the GUI**

```bash
trackerkit
```

Open a project config saved **before** this change (one carrying the three keys). Confirm: it loads
with no error dialog and no console traceback; the detection panel shows "Live Detection Batching"
and no "Legacy Detector Batching"; the Frame batch size value is whatever `detection_batch_size` held.

- [ ] **Step 5: Run a real tracking run**

Using the loaded project, run a short forward tracking pass on a real video with visualization
enabled. Confirm in the log:

- `PHASE 1: InferenceRunner batch pass` appears (detection still runs through the runner)
- `PHASE 2: Tracking and Visualization` appears
- The `Frame prefetching ENABLED (background I/O buffering)` / `Frame prefetching disabled` line
  (`worker.py:1791,1793`) matches what it printed **before** this change for the same project. On a
  default non-realtime OBB run with a cache path, expect `disabled` — `use_batched_detection` is True.
- Tracks are produced and the CSV is written

**If the prefetcher line flipped, STOP.** That is the one regression this plan is shaped to avoid, and
it means a term was dropped from the conjunction that should not have been.

- [ ] **Step 6: Record the verification**

```bash
cat >> docs/superpowers/plans/notes/2026-07-16-cache-builder-decision.md <<'EOF'

## End-to-end verification (Task 8)

- Old project config loads: PASS | FAIL
- PHASE 1 / PHASE 2 both run: PASS | FAIL
- Prefetcher log line unchanged vs. pre-change: PASS | FAIL
- Tracks + CSV produced: PASS | FAIL
EOF
```

Fill in the actual results, then commit:

```bash
git add docs/superpowers/plans/notes/2026-07-16-cache-builder-decision.md
git commit -m "docs: record end-to-end verification of the batching-vestige removal"
```

---

## Out of scope

Recorded so a future reader knows these were considered and deliberately excluded:

- **`core/detectors` removal.** Not retired — `_direct_obb_runtime.py` and `_runtime_artifacts.py`
  are live dependencies of `core/inference`. See the bg-sub plan's Task 13.
- **`DetectionCacheBuilderWorker` removal.** May be redundant post-bg-sub (Task 1 investigates), but
  deleting it is a feature removal needing its own plan.
- **`utils/batch_optimizer.py` deletion.** Blocked on the above; it has one live caller.
- **Turning the frame prefetcher on for non-realtime OBB runs.** Plausibly a win, but it is a perf
  change, not a cleanup. The stall diagnostics at `frame_prefetcher.py:179-203` and the
  `read_timeout` docstring note at `:80-82` ("Increase when the decode backend shares resources with
  GPU inference") suggest prefetching alongside GPU inference has hung before. Needs a benchmark and
  its own commit.
- **`spin_tensorrt_batch` / `enable_tensorrt` as user-facing controls.** Both were already dead UI
  (`setVisible(False)`); `enable_tensorrt` remains as a runtime-derived value for legacy config
  round-tripping and the engine-cache key.
