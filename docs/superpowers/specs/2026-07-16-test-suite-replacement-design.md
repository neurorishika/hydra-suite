# Replacing the Test Suite with Golden Equivalence Harnesses

**Date:** 2026-07-16
**Status:** Approved design, pending implementation plan

## Problem

The test suite has stopped functioning as a feedback loop. Concretely, as measured on
2026-07-16:

- **245 test files, ~59k lines, 2315 tests.** 56 files contain two tests or fewer —
  the signature of one-file-per-bugfix accretion. Median file has 6 tests. Growth is
  accelerating: 30 new test files in May 2026, 29 in the first half of July.
- **91 of 245 files depend on mocks or monkeypatching; 28 bypass real imports** via
  `tests/helpers/module_loader.py`, and 7 stub `sys.modules` outright. Tests that stub
  the import graph verify that the mock behaves, not that the code does. They pin
  *structure*, which is precisely what the Simplification Sprint is about to change.
- **The suite does not terminate.** A full run parks indefinitely at ~15% with live
  pytest processes. A suite that cannot finish cannot gate anything.
- **`tests/test_postproc_equivalence.py` reports "16 passed" while asserting nothing.**
  Its golden CSVs were never committed — `.gitignore:140` carries a blanket `*.csv`.
  `available_cases()` (`tests/helpers/postproc_runner.py:222`) enumerates *directories*
  rather than checking for goldens, and `_read_csv` returns an empty DataFrame for a
  missing path. It therefore compares empty to empty at 1e-9 tolerance and goes green in
  4 seconds. This is the best-designed test in the repo, and it has been silently vacuous.

Meanwhile the equivalence and benchmark harnesses in `tools/` carry the real signal about
whether the pipeline works.

A note on diagnosis, recorded because it will recur: a large share of the apparent
breakage is an **environment artifact**, not a test problem. Under the `base` conda env,
`torch` fails on a circular import inside `torch/_library`, cascading into **124 collection
errors**. Under `hydra-mps` the same tree yields **1**. Always confirm the env before
reading a failure count.

## Non-goals

- Rebuilding equivalent coverage. Coverage is not the goal; trustworthy signal is.
- Refactoring `tools/equivalence/` beyond what golden mode requires.
- Any change to CLI entry points or inter-kit APIs.

## Key insight

**In a git repo, deleting tests is not lossy.** All 245 files remain in history,
recoverable by path forever. The real cost of deletion is losing the *knowledge of which
ones were worth keeping* — so the work is to write that knowledge down first, then delete
freely.

## The structural problem with the harness as it stands

`tools/equivalence/` is **not** a golden-baseline harness. It is a **differential A/B
between two source trees**: it points `PYTHONPATH` at a legacy worktree and at `src/`,
runs both, and diffs. It answers exactly one question — *"does new match legacy?"* — and
has no stored notion of correct output.

This makes it a **migration tool**, not a regression net. It has been doing its job well
enough to be mistaken for one. Its oracle depends on the legacy tree remaining runnable.

Current state of that dependency: `legacy` exists as a local **and** `origin` branch, plus
a `legacy/main` tag. The oracle is durably archived — this is not urgent. The worktree the
harness expects (`.worktrees/inference-pipeline-redesign`) is gone, but is one
`git worktree add` from the tag. The real deadline is **environment drift**: `legacy` can
only produce goldens while a conda env exists that can still run it. Months of slack, not
days.

Three further limits, which shape what the spine must cover:

1. **It is not a gate.** `run_matrix.sh:116` swallows exit codes
   (`python compare.py "$1" "$2" || true`), so the matrix always exits 0. Pass/fail is a
   human reading whether EQUIVALENCE numbers sit near DETERMINISM numbers.
2. **It cannot see where the sprint is happening.** It enters at `run_tracking_cli`
   (`tools/equivalence/runner.py:294`), covering detect → headtail → pose → identity →
   track → CSV. It never touches any GUI kit, `training`, `integrations`, `data`, or
   `paths`. All four Simplification Sprint slices are GUI-layer work.
3. **It is a release ritual, not a dev loop.** 384 MB of weights plus clips fetched from
   GitHub Release `equiv-fixtures-v2`, seven clips × three runs × minutes each, and no CI
   exists (`docs-pages.yml` is the only workflow).

What survives from this: equivalence harnesses are excellent **oracles** and poor
**localizers**. They report that output drifted, not which of forty changed files drifted
it.

## Approach

Freeze, delete, rebuild a thin spine.

Considered and rejected:

- **Pure deletion, harnesses only.** Defensible for the pipeline, but leaves the GUI kits
  at zero coverage exactly as Slice 4 splits a 19k-line `main_window.py`, and leaves
  `git bisect` over minutes-per-run as the only debugging tool when equivalence goes red.
- **Quarantine and earn back.** Quarantined tests are never promoted; they rot in a new
  directory while everyone feels responsible about them. Deletion with extra steps.

## Design

### 1. Golden freeze

`tools/equivalence/compare.py` already exits 0/1 on a real verdict — `pos_p99 <= 0.5px`,
`theta_mean <= 0.05 rad`, `unmatched == 0` (`compare.py:190-201`) — with ID-agnostic
Hungarian matching, so track renumbering does not raise false alarms. **Golden mode
requires no new comparison logic**: it is the same call with the first argument pointed at
a frozen directory instead of a live legacy run.

1. `git worktree add .worktrees/legacy legacy` to restore the tree the harness expects.
2. Run `runner.py` once per clip with `PYTHONPATH` at `.worktrees/legacy/src`. Commit the
   `forward`/`final` CSVs to `tools/equivalence/goldens/<clip>/<kind>.csv`, alongside the
   `meta.json` recording device, torch version, and legacy commit.
3. Negate the `.gitignore:140` `*.csv` rule for that path. **Goldens belong in git, not in
   the Release with the clips.** The entire value of a golden is that an intentional
   behavior change appears as a reviewable diff in a PR. A golden you cannot see change is
   a golden that changes silently — which is exactly how the postproc goldens failed.
4. A `make equivalence` target running new-vs-golden that **propagates the exit code**,
   replacing the `|| true` at `run_matrix.sh:116`.

### 2. Device nondeterminism: CPU-only goldens

Runs are not bit-exact — that is why the harness runs `new_a` vs `new_b` to measure a
DETERMINISM noise floor. Today legacy and new both execute on the same box, so the floor
is shared. A frozen golden breaks that symmetry: frozen on one device, compared on
another, MPS-vs-CUDA divergence becomes indistinguishable from a real regression.

**Decision: freeze goldens on CPU.** GPU runs compare against the CPU golden with a
tolerance set above the measured cross-device floor. One golden set, runnable anywhere
including CI. Accepted cost: GPU-specific regressions smaller than the cross-device floor
are invisible.

Two consequences this design commits to:

- **Tolerance must be measured, not guessed.** `PARITY_AUDIT.md` currently records no
  determinism numbers at all, so today's `--pos-atol 0.5` default is a number someone
  chose. A one-time determinism probe (`new_a` vs `new_b`, per device, plus CPU-vs-GPU)
  records real numbers into `PARITY_AUDIT.md`, and tolerances derive from those.
- **CI shape.** Seven clips × 500 frames on CPU is too slow per-PR. Full matrix runs
  **nightly**; a **single-clip smoke** runs per-PR.

### 3. Deletion scope

Delete effectively all of `tests/` — 245 files, ~59k lines.

The exception is `test_postproc_equivalence.py`, `tests/helpers/postproc_runner.py`, and
`tests/fixtures/postproc/`. That file is already the thing being built: golden-based,
CPU-only, no mocks, no weights, 4 seconds. It does not need deleting; it needs its goldens
committed (per §1.3). It is the template, not a casualty.

Everything else goes, and returns from git history by path if a specific test proves
missed.

### 4. Rebuild: demand-driven only

Nothing is rebuilt upfront. The spine grows on demand, and exactly two demands qualify:

1. **A failure the harness localized to a module** — write the test that pins it, at the
   module.
2. **A sprint slice that needs a net** — Slice 4's `main_window.py` split gets GUI smoke
   tests written *as* that slice happens, not before.

Anything not tracing to one of those two does not get written. This rule is what prevents
silent regrowth to 245 files. The expected landing zone is 20–40 files, but that number is
an **outcome, not a target**.

Every rebuilt test meets the `test_postproc_equivalence.py` bar: no mocks of the code
under test, no `module_loader` import bypass, fast, and asserting against something real.

### 5. Bugs to fix regardless

- **`.gitignore:140`** — one line; it is what made the best test in the repo vacuous.
- **The hang** — see below.

## Step zero: identify the hang before deleting

Deleting the suite would make the hang disappear without explaining it. That is the one
place where bulk deletion can destroy signal:

- If a **bad test** deadlocks, deleting it is the correct fix and costs nothing.
- If a **deadlock in `src/`** is merely exposed by a test, deleting the test hides a real
  bug in shipping code — and the equivalence harness would never catch it, being
  GUI/threading-shaped and outside `run_tracking_cli`.

Fifteen minutes of bisection to classify it, before deletion.

## Risks

| Risk | Mitigation |
|---|---|
| Goldens frozen from a legacy tree that is itself wrong | Legacy is the shipped, trusted release. Freezing "known-good" ≠ "provably correct" — this is explicitly a regression net, not a correctness proof. |
| GPU-only regressions below the cross-device floor go unseen | Accepted, per §2. Nightly GPU runs against the CPU golden still catch anything larger. |
| Demand-driven rebuild never happens; GUI ends up uncovered | The two triggers in §4 are the plan; Slice 4 cannot proceed without its net. |
| Env drift makes `legacy` unrunnable before goldens are frozen | Freeze first. It is step one for this reason. |
| `*.csv` gitignore hole recurs for the new goldens | A test asserting goldens exist and are non-empty — the check whose absence caused the original failure. |

## Success criteria

- `make equivalence` exits non-zero on a real regression and zero otherwise, with no human
  reading numbers.
- Golden CSVs are committed, non-empty, and diff reviewably in PRs.
- `PARITY_AUDIT.md` records measured determinism numbers; every tolerance traces to one.
- `pytest` terminates.
- No test in the tree passes while asserting nothing.
