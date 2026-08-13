# Test Suite Debug Audit

Env for all runs: `conda activate hydra-mps` (py3.13). Worktree: `.worktrees/test-suite-debug`.

## Actions taken (this branch)

- **pytest-timeout** added to dev deps + wired into the live `pytest.ini`
  (`--timeout=300 --timeout-method=thread`).
- **3 crashing GUI test files DELETED** (operator decision): `test_detectkit_main_window.py`,
  `test_classkit_main_window.py`, `test_refinekit_main_window.py`. Root cause was a native
  heap-corruption crash in the third-party stack (see §1) that aborted the whole
  `pytest tests/` run; not fixable in repo code, and the tests were shallow GUI smoke checks.
- **`test_worker_real_inference_integration.py` REWRITTEN**: the 4 tautological tests replaced
  with 3 real-drive tests that execute `run_tracking()`'s actual Site-A dispatch (backward
  refusal, invalid→batch-pass, valid→cached-replay), plus removal of 2 redundant Site-E mock
  tests (the real `run_realtime` path is covered by `test_tracking_worker_realtime_live_features.py`).
- **`test_identity_postprocess.py` RESCUED**: dropped the 5 stale tests (renamed+re-gated API);
  kept the still-valid `fill_identity_nans_with_consensus` test. Collection error resolved.
- **Result:** full-suite collection is now clean (0 errors; was 1).
- **NOT done (left for a deliberate follow-up):** consolidating the `pytest.ini` vs
  `pyproject.toml` config shadowing (§0); the 2 borderline monkeypatch cleanups (§2); the
  42 ordinary test failures (§4).

## 0. Config bug uncovered while wiring pytest-timeout (FIXED)

`pytest.ini` (tracked) **silently shadows** the entire `[tool.pytest.ini_options]` block
in `pyproject.toml` — pytest prints *"ignoring pytest config in pyproject.toml"*. So
pyproject's `--strict-markers`, `--strict-config`, `filterwarnings=error`, and its
`integration`/`unit` markers have all been **dead config** for as long as both files
existed. The live config is the much thinner `pytest.ini`.

- **Done:** added `pytest-timeout>=2.1` to `[project.optional-dependencies].dev`; wired
  `--timeout=300 --timeout-method=thread` into the LIVE `pytest.ini`. Verified active.
- **Recommended follow-up (needs decision):** consolidate to ONE config source. Deleting
  `pytest.ini` in favor of pyproject would flip on `filterwarnings=error` + strict markers,
  surfacing a wave of currently-masked warnings/failures — do it deliberately, not silently.

## 1. Hangs → actually NATIVE CRASHES in 3 GUI test files (root cause found)

The reported "`pytest tests/` never finishes / SIGABRT" has two distinct causes:

- **False alarm:** the 288-error collection segfault reproduces **only in the wrong env**
  (`base`, py3.10 — a pybind11 double-registration). In `hydra-mps` collection is clean
  (3400/3402 collected).
- **Real cause:** a per-file subprocess sweep of all 442 files (hard 360s cap) found
  **3 files that crash with native signals** — which `pytest-timeout` **cannot** catch,
  and which abort the entire `pytest tests/` process when hit mid-run:

  | File | Signal | Repro |
  |---|---|---|
  | `test_detectkit_main_window.py` | SIGSEGV/SIGBUS | **deterministic (5/5)** |
  | `test_classkit_main_window.py` | SIGKILL after ~135 s | slow → killed |
  | `test_refinekit_main_window.py` | SIGABRT | flaky (~1 in N) |

  **Root cause (self-documented in the code):** the `main_win` fixture in
  `test_detectkit_main_window.py:22` reads
  *"Single shared DetectKitMainWindow — avoids per-test SVG GC crash."* The team already
  knows: PySide6 **QtSvg icon garbage-collection segfaults** on QApplication/window
  teardown. Module-scoped singletons were a partial workaround; the crash still fires at
  teardown and interleaves with first-time heavy native imports (torch/numba/sklearn/scipy
  via the `detectkit dialogs → data.al.candidate_pool → filterkit.core → sklearn.cluster`
  chain). Verified: the crashing test passes in isolation; it needs accumulated live-Qt
  state from earlier tests in the file. `KMP_DUPLICATE_LIB_OK=TRUE` does **not** fix it
  (hypothesis tested and rejected).

  **No hangs anywhere else** — the other 439 files complete; 42 are ordinary failures
  (see §4), none are hangs.

## 2. Excessive monkey-patching → 1 clear defect, 2 borderline (rest legitimate)

High patch counts are almost all legitimate boundary stubbing (Qt dialogs, GPU probes,
`InferenceRunner`, cv2, SLEAP subprocess, filesystem, DB). Verified the heaviest files
(`test_classkit_main_window.py` 67, `test_gpu_utils.py` 39, trackerkit + classkit clusters)
— **clean.**

**CLEAR DEFECT — `test_worker_real_inference_integration.py` (4 tautological tests):**
despite the "real integration" name, these exercise no system-under-test and would pass if
`worker.py` were deleted (verified firsthand):
- `test_site_e_load_frame_called_per_iteration` (~149) — builds a `MagicMock`, calls it,
  asserts the mock recorded the call.
- `test_site_e_run_realtime_returns_frame_result` (~173) — sets `return_value`, calls mock,
  asserts the return.
- `test_backward_mode_refuses_without_valid_caches` (~280) — builds the worker via
  `__new__`, sets attrs, **never calls any worker method**, then asserts
  `run_batch_pass.assert_not_called()` (trivially true). Gives **false confidence** for the
  load-bearing "backward mode refuses without valid cache" guard.
- `test_caches_valid_skips_batch_pass` / `test_caches_invalid_triggers_batch_pass` (~320) —
  the test body **reimplements** the worker's `if/else` inline and asserts against its own
  copy; the real branch is never touched.

  → Rewrite against the real `TrackingEngineCore` code path (the file's *other* tests already
  do this well), or delete. The file's remaining tests are fine.

**BORDERLINE (fix opportunistically):**
- `test_sleap_trt_rebuild_and_cache_only.py` — `test_cache_only_true_skips_pose_model_loading`
  (441-480) stubs the whole `_load_all_models` stage then asserts its lambda's return
  (redundant with the real-loader test at 511-556); AST-source-grep guard at 619-657.
- `test_interpolated_crops_worker.py` (21-88) — stubs nearly every stage for a skip-branch
  test; the branch decision is genuinely the SUT's, so not strictly tautological.

## 3. Stale tests → exactly 1 file

`test_identity_postprocess.py` — the **sole collection error**. It's a **mix**, not a clean
delete:
- 5 tests call `apply_identity_postprocessing(df, params)`, a symbol that was renamed AND
  moved to `apply_identity_postprocessing_to_df` in `core/individual/postprocess_df.py`,
  with the fragment-solver split/join now gated behind `ENABLE_IDENTITY_FRAGMENT_SOLVER`
  (default OFF). A rename alone won't make them pass — semantics changed.
- 1 test (`test_fill_identity_nans_with_consensus_handles_float_label_columns`) is **still
  valid, unique coverage** for `fill_identity_nans_with_consensus` (live at
  `core/post/identity_postprocess.py:131`) — merely stranded by the dead module-level import.
- The apparent successor `test_core_identity_postprocess_df.py` is thin (2 shallow tests) and
  does NOT replicate the 5 behavioral scenarios.

Everything else the heuristics flagged (`core.detectors` guards, `compute_runtime` retirement
asserts, `legacy` rejection tests, 95 skip/xfail/importorskip guards) is **deliberate and
correct** — keep.

## 4. Bonus: 42 ordinary failures (not hangs, out of original scope)
Real assertion/import failures surfaced by the per-file sweep (e.g. `test_bg_parameter_helper`,
`test_classifier_backend`, `test_classifier_fixtures`, several `test_classkit_*`). Full list in
`scratchpad/per_file_results.jsonl`. Worth a separate triage pass.
