# Plan A — Task 8 Report

## Task
Make `bench_classify` receive a resolved compute-runtime string via the tier helper (mechanical — `bench_classify` itself was already correct).

## What I implemented

1. Added a failing test `test_run_target_benchmark_resolves_tier_before_bench_classify` in
   `tests/test_trackerkit_benchmarking.py` (inserted after
   `test_bench_headtail_uses_classifier_backend_not_legacy_detector`, before
   `test_collect_active_targets_includes_sequential_crop_settings`).

   Adapted from the brief's snippet because:
   - `BenchmarkGeometry` is a plain dataclass requiring all 11 geometry fields
     (`frame_width`, `frame_height`, `resize_factor`, `effective_frame_width`,
     `effective_frame_height`, `reference_body_size`, `reference_aspect_ratio`,
     `padding_fraction`, `canonical_crop_width`, `canonical_crop_height`) — the
     brief's 3-kwarg constructor call doesn't match the current dataclass.
   - `run_target_benchmark`'s `warmup`/`iterations` params are keyword-only
     (`*` in the signature) — the brief's bare positional call
     `run_target_benchmark(target, geometry, "gpu_fast", 1)` would raise
     `TypeError`. Added `warmup=0, iterations=1`.
   - Strengthened the assertion beyond the brief's loose membership check
     (`seen_runtimes[0] in {"cpu", "cuda", "mps", "tensorrt", "coreml"}`),
     which is too weak to catch the actual bug: `_normalize_runtime("gpu_fast")`
     silently collapses to `"cpu"`, and `"cpu"` is itself a member of that set,
     so the brief's test as written passes even against the buggy code (verified
     this directly — see Step 2 below). Instead the test now computes the
     platform-correct expected value via `resolve_compute_runtime("gpu_fast",
     detect_platform(), stage="cnn")` and asserts exact equality, plus asserts
     the resolved runtime isn't `"cpu"` when an accelerator is present. This
     reproduces the RED state and proves the fix.

2. Fixed the dispatch site in `run_target_benchmark` (`src/hydra_suite/trackerkit/benchmarking.py`,
   `classify` branch, ~line 1511-1519):

   ```python
   if target.pipeline == "classify":
       platform = detect_platform()
       resolved_runtime = resolve_compute_runtime(runtime, platform, stage="cnn")
       return bench_classify(
           target.model_path,
           resolved_runtime,
           warmup,
           iterations,
           batch_size,
           crop_size,
       )
   ```

   Previously this branch passed `normalized_runtime` (computed via
   `_normalize_runtime(runtime)` at the top of the function), which silently
   collapses tier strings `"gpu"`/`"gpu_fast"` to `"cpu"` because
   `_normalize_runtime` doesn't recognize the 3-value tier vocabulary
   (`cpu`/`gpu`/`gpu_fast`) that `collect_active_targets` now hands
   `run_target_benchmark`.

   `"cnn"` is a canonical stage name already present in
   `src/hydra_suite/runtime/resolver.py::STAGES = ("obb", "head_tail", "cnn",
   "yolo_pose", "sleap_pose")` — no resolver change needed.

   Scope note: `normalized_runtime` is still used for the `headtail` branch
   (line ~1489), which was NOT touched — that branch's dispatch appears to have
   the same collapse bug (`bench_headtail`'s own body internally calls
   `resolve_compute_runtime(tier, platform, stage="head_tail")` expecting a raw
   tier, but the dispatch site feeds it `normalized_runtime` = the collapsed
   `"cpu"`/etc. string, not the tier). This is out of scope for Task 8 (brief
   explicitly restricts scope to the `bench_classify` dispatch site only) and is
   flagged here for whoever owns that follow-up.

## `artifact_available` verification (not assumed)

Read `src/hydra_suite/core/identity/classification/cnn.py`,
`CNNIdentityBackend.__init__` (lines 349-366):

```python
def __init__(self, config, model_path=None, compute_runtime="cpu"):
    from hydra_suite.core.identity.classification.backend import ClassifierBackend
    ...
    self._backend = ClassifierBackend(resolved_path, compute_runtime=compute_runtime)
```

`CNNIdentityBackend` is a thin wrapper that directly instantiates
`ClassifierBackend` — **the exact same backend class** used by
`bench_headtail` (`src/hydra_suite/trackerkit/benchmarking.py:1127`,
`from hydra_suite.core.identity.classification.backend import ClassifierBackend`).
All inference (`predict_batch`, `predict_batch_cuda`) is delegated straight
through to `self._backend`. There is no separate/parallel export path for CNN
classification — it shares `ClassifierBackend`'s real ONNX+TensorRT-EP loading
machinery verbatim.

Conclusion: this matches Task 7's finding for head-tail (real production
gpu_fast export mechanism exists), NOT Task 6's finding for YOLO pose (no real
export artifact). Used the **default** `artifact_available` (no override) —
`resolve_compute_runtime(runtime, platform, stage="cnn")` — consistent with
`bench_headtail`'s own internal call (`resolve_compute_runtime(tier, platform,
stage="head_tail")`, no override, line 1114).

## Test results

Environment: `hydra-mps` conda env (Apple Silicon, `has_cuda=False, has_mps=True`).

Step 2 (RED, before fix):
```
$ python -m pytest tests/test_trackerkit_benchmarking.py -k resolves_tier_before_bench_classify -v
...
FAILED tests/test_trackerkit_benchmarking.py::test_run_target_benchmark_resolves_tier_before_bench_classify
AssertionError: assert ['cpu'] == ['coreml']
1 failed, 27 deselected in 3.55s
```

Step 4 (GREEN, after fix):
```
$ python -m pytest tests/test_trackerkit_benchmarking.py -k resolves_tier_before_bench_classify -v
tests/test_trackerkit_benchmarking.py::test_run_target_benchmark_resolves_tier_before_bench_classify PASSED [100%]
1 passed, 27 deselected in 3.48s
```

Full regression file:
```
$ python -m pytest tests/test_trackerkit_benchmarking.py -v
... (28 tests)
============================== 28 passed in 3.85s ==============================
```

Output is pristine — no warnings, no skips, no errors.

## Files changed

- `src/hydra_suite/trackerkit/benchmarking.py` — `run_target_benchmark`, `classify` dispatch branch (~4 lines changed)
- `tests/test_trackerkit_benchmarking.py` — new test added (~59 lines)

## Self-review

- [x] Implemented the brief's intent (resolve tier before dispatching to `bench_classify`), adapted mechanically to current dataclass fields and keyword-only params.
- [x] Verified (not assumed) the `artifact_available` question by reading `CNNIdentityBackend.__init__` directly — confirmed it shares `ClassifierBackend` with head-tail, so default (no override) is correct.
- [x] Ran pytest with `hydra-mps` env activated; real PASS/FAIL output captured above (RED then GREEN).
- [x] `tests/test_trackerkit_benchmarking.py` is fully green (28/28).
- [x] Test output is pristine.

## Concerns

- Confirmed but out-of-scope: the `headtail` dispatch branch (line ~1489) appears to have the same tier-collapse bug (passes `normalized_runtime` where `bench_headtail` expects a raw tier for its own internal `resolve_compute_runtime` call). Not fixed here per explicit task scope restriction ("Don't touch `bench_obb`/`bench_sequential`/`bench_pose`/`bench_headtail` themselves — only the `bench_classify` dispatch site in `run_target_benchmark` is this task's scope"). Flagging for the plan owner since it affects `bench_headtail`'s dispatch, not its body.
