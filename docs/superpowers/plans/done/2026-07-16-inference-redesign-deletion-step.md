# Task Brief: Complete the deletion step of the inference pipeline redesign

> **SUPERSEDED (2026-07-19).** This brief predates the four `2026-07-17-detector-retirement-*` plans and
> the pose golden-rule work, and its assumptions (e.g. which files were live, its parent spec's deletion
> list) are stale. The `core/detectors` retirement was completed by those plans instead: Plan 1 (foundation),
> Plan 2 (consumer migration + class_ids), Plan 3 (benchmarking replacement), Plan 4 (package deletion),
> plus the pose runtime golden rule. `core/detectors` is deleted; `core/inference` is the sole pipeline.
> Kept for history only — do NOT execute.

**For:** a fresh agent with no context on this codebase
**Parent spec:** `docs/superpowers/specs/2026-04-26-inference-pipeline-redesign.md`
(the deletion list is the fenced block under *"Explicitly deleted after new pipeline is verified working"*, and the
migration table immediately above it is equally load-bearing — read both)

## Background

The inference pipeline redesign replaced the old detection/pose/identity code with a new
`core/inference/` package (`stages/`, `cache/`, `pipeline.py`, `runner.py`, `result.py`).
The new architecture is live and merged. The spec's **final step — deleting the superseded
modules — was never performed.**

A prior audit established:

- The new pipeline is fully in production; nothing imports the old `core.tracking.{detection_phase,precompute,pose_pipeline}` module *paths*.
- **But** several deletion targets were *relocated* rather than deleted, and are still imported
  under their new paths. The spec's own final-scan greps use the pre-relocation paths, so they
  **pass vacuously** and would not catch this. Do not trust them.

Current state of the spec's 20-file deletion list (verified 2026-07-16):

| File (spec path) | Actual location today |
|---|---|
| `core/tracking/precompute.py` | **already gone** — nothing to do |
| `core/tracking/detection_phase.py` | relocated → `core/tracking/ingest/detection_phase.py` |
| `core/tracking/pose_pipeline.py` | relocated → `core/tracking/pose/pose_pipeline.py` |
| `core/tracking/live_features.py` | relocated → `core/tracking/features/live_features.py` |
| `core/tracking/cnn_features.py` | relocated → `core/tracking/features/cnn_features.py` |
| `core/tracking/tag_features.py` | relocated → `core/tracking/features/tag_features.py` |
| `core/tracking/evidence_emitter.py` | relocated → `core/tracking/identity/evidence_emitter.py` |
| the other 13 | still at their original spec paths |

Known live importers (non-exhaustive, verify yourself):
`core/tracking/worker.py:686`, `trackerkit/gui/workers/crops_worker.py:670`,
`trackerkit/gui/workers/preview_worker.py:1548`.

## Your objective

For each file on the deletion list, **prove** it is genuinely dead, then delete it.
Where it is *not* dead, do **not** delete it — report it instead.

**The deliverable is a correct answer, not an empty directory.** "These 6 are dead and now
deleted; these 4 are load-bearing and here's what still needs to migrate first" is a complete
success. Deleting something still in use is a failure, and so is silently skipping a file.

## Method

Work **one file at a time**. Do not batch-delete.

### Step 1 — Establish the real name and path

Relocations mean the spec path is not the import path. For each target, find where it lives now
and what symbols it exports. Search for the *symbols*, not just the module path — a module can be
reached via a package `__init__.py` re-export without its own path appearing anywhere.

### Step 2 — Prove deadness

A file is dead only if **all** of these hold. Check each explicitly:

1. No module in `src/` imports it — by module path, by symbol, or via any `__init__.py` re-export chain.
2. No test in `tests/` imports it. *(A test importing it does not by itself make it alive — see Step 3.)*
3. It is not reachable by dynamic dispatch: `importlib`, `__import__`, string module names,
   entry-points in `pyproject.toml`, or Qt signal/slot wiring by name. Grep for the bare module
   basename across the whole repo, including non-Python files.
4. Nothing in `legacy/` matters here — `legacy/` is excluded from tests and is not a live importer.
   But `src/` importing `legacy/` **is** a violation worth reporting if you find it.

Record the evidence per file. If any check fails, the file is **alive** → Step 4.

### Step 3 — Judge test-only importers carefully

If a file's only importers are its own tests, that is a **decision point, not an automatic delete.**
Ask: does the test cover behavior that the new `core/inference/` path also implements?

- **Yes, covered by new tests** → delete both the module and its now-obsolete test. Name the
  replacement test in your report.
- **No equivalent coverage exists** → deleting the test silently removes coverage. Stop and report
  it; the coverage gap must be filled first.

### Step 4 — Handle live files honestly

If a file is alive, it is **out of scope for deletion**. Do not migrate its callers to make it dead
— that is a separate, much larger piece of work with its own risk. Instead report: the file, its
live importers with `file:line`, and what would have to happen before it could be removed.

The migration table in the spec (just above the deletion list) already describes the intended
rewiring for many of these — cite the relevant row so the follow-up work is actionable.

### Step 5 — Delete, incrementally

For each proven-dead file:

- Use `git rm` so history follows.
- Remove its re-exports from the owning `__init__.py`. The spec calls for backwards-compat shims
  in some `__init__.py` files "for one cycle" — check whether that cycle has elapsed; if unsure,
  leave the shim and report it rather than guessing.
- **After each deletion**, run `make pytest`. A failure means the file was not dead — restore it
  (`git checkout`) and treat it as a Step 4 case. Do not "fix" the test to make the deletion stick.
- Commit per logical group, not one giant commit.

## Verification (do not skip; do not reuse the spec's greps)

The spec's final-scan greps reference pre-relocation paths and pass vacuously. Write fresh ones
against the **current** paths. Then:

```bash
make pytest          # must pass
make lint            # must pass
make dead-code       # should surface fewer findings than before, not more
```

Also confirm the app still imports cleanly — a deleted module can break a GUI import path that
tests never exercise:

```bash
python -c "import hydra_suite.trackerkit.gui.main_window, hydra_suite.posekit.gui.main_window"
```

Per `CLAUDE.md`, run `make format` before committing.

## Report back

- **Deleted:** file → the evidence that proved it dead.
- **Kept:** file → live importers (`file:line`) → what must migrate first (cite the spec's
  migration-table row).
- **Judgment calls:** anything where you deleted a test, left a compat shim, or were less than certain.
- Whether the parent spec can now be considered complete. If files remain, it **cannot** — say so
  plainly, and say what's left.

## Guardrails

- Do not weaken, skip, or delete a test to make a deletion pass.
- Do not migrate callers to manufacture deadness.
- If the evidence is ambiguous, keep the file and report it. Under-deleting is cheap to fix later;
  deleting something load-bearing is not.
