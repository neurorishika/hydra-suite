# Repository Working Agreement

These instructions apply to all work in this repository. `CLAUDE.md` contains
additional architecture and operational background; consult it when the task
touches an unfamiliar subsystem, runtime, equivalence testing, or release docs.

## Required fix workflow

Every code or product fix must follow this sequence. Do not skip directly to
editing the primary checkout.

1. Inspect the primary checkout and preserve all existing user changes.
2. Create a dedicated git worktree under `.worktrees/`, branched from the
   current local `HEAD`, not from `origin/main`:

   ```bash
   git worktree add .worktrees/<slug> -b codex/<slug> HEAD
   ```

   Local `main` is commonly ahead of the remote. Perform implementation,
   formatting, test execution, and commits inside this worktree.
3. Reproduce or characterize the problem and add a regression test. Extend the
   suite to cover important neighboring behavior, edge cases, and invariants;
   avoid tests that only mirror the implementation.
4. Implement the smallest coherent fix. For multi-step work, divide changes by
   subsystem and commit each completed subsystem separately. Each commit must
   be understandable, reviewable, and leave the worktree coherent. Do not
   squash these commits during merge-back unless the user explicitly asks.
5. Run focused tests while iterating, then the appropriate broader regression,
   formatting, lint, documentation, equivalence, and platform checks for the
   risk of the change. Investigate all failures. Never call a fix verified when
   a required check failed, crashed, was skipped, or could not run; report that
   limitation explicitly.
6. Review the complete branch diff against its merge base for correctness,
   regressions, unintended files, architecture violations, test quality,
   formatting, and documentation. Resolve every actionable finding and repeat
   all affected checks. Review occurs after implementation is complete and
   before merge-back.
7. Merge the reviewed worktree branch into the primary checkout only after the
   implementation, tests, verification, and review are complete. Preserve the
   piecemeal subsystem commits. Confirm the resulting integrated diff and rerun
   the appropriate verification from the primary checkout.
8. Do not remove the worktree or delete its branch until merge-back and
   post-merge verification have succeeded. Then remove it cleanly with
   `git worktree remove .worktrees/<slug>` and run `git worktree prune`.

If the primary checkout changes while the fix is underway, reconcile those
changes safely in the worktree, repeat affected review and verification, and
only then merge. Never discard or overwrite unrelated user work.

Administrative-only edits to repository agent instructions are not product
fixes and may be made in the primary checkout when explicitly requested.

## Environments and verification

- Use the `hydra-mps` conda environment for local tests and equivalence checks
  on this Apple Silicon machine. Environment names are `hydra`, `hydra-mps`,
  and `hydra-cuda`.
- Use the `hydra-cuda` environment on
  `rutalab@mehek.taild08eb9.ts.net` when a change requires CUDA verification.
- Before any heavy equivalence, training, or inference run, inspect and stop
  only dead or stale SLEAP/Hydra processes. Never interfere with an unrelated
  process or an actively running SLEAP/Hydra job.
- Pytest is configured by the repository, tests live in `tests/`, and
  benchmarks are excluded by default. Useful commands include:

  ```bash
  python -m pytest tests/test_<name>.py
  python -m pytest tests/test_<name>.py::test_fn
  make pytest
  make test-cov
  ```

- Run verification in proportion to risk. A normal fix should include focused
  regression tests plus the nearest subsystem suite. Cross-cutting or pipeline
  changes require broader tests. Tracking-pipeline or performance-sensitive
  changes may require the equivalence harness in `tools/equivalence/`; read
  `tools/equivalence/README.md` and the detailed guidance in `CLAUDE.md` first.
- Equivalence results are trustworthy only when output CSV row counts are
  non-empty. Pose/SLEAP clips require an activated conda environment. Compare
  both `_forward.csv` and `_tracking_final.csv`; the expected performance ratio
  is at most `PERF_TOLERANCE` (default `1.25`). Risky pipeline changes require
  MPS and CUDA checks unless the user explicitly narrows platform scope.
- Before committing, use the repository quality commands appropriate to the
  change:

  ```bash
  make format
  make format-check
  make lint
  make commit-prep
  make lint-moderate
  make docs-check
  ```

  For large changes, also consider `make audit`. Run `make docs-build` or
  `make docs-check` when documentation or public behavior changes.

## Architecture and implementation rules

### Dependency direction

Dependency flow is:

`App layers -> Core / Runtime / Data / Training / Utils`

- App packages (`trackerkit`, `posekit`, `classkit`, `refinekit`, `filterkit`,
  and `detectkit`) may import lower layers.
- Core, Runtime, Data, Training, and Utils must not import from app packages.
- Integrations may import Core, Runtime, Data, and Utils, but not app packages.
- The shared `widgets/` layer may be imported by app packages and must not
  import from them.
- Keep the Data layer reusable from both GUIs and scripts.

### Shared abstractions and GUI structure

- Keep `MainWindow` classes as thin coordinators: instantiate panels, connect
  signals, and delegate to typed configuration and focused services. Do not add
  business logic to them.
- A class approaching roughly 500 lines should trigger extraction into focused
  modules. Workers, dialogs, configuration schemas, and business logic belong
  in separate files.
- Use `BaseWorker` from `hydra_suite.widgets.workers` for background tasks,
  `BaseDialog` from `hydra_suite.widgets.dialogs` for modal dialogs, and
  `WelcomePage` from `hydra_suite.widgets.welcome_page` for kit welcome pages.
- Store kit runtime state in typed dataclasses under `<kit>/config/schemas.py`
  with `to_dict`/`from_dict`; do not scatter configuration as ad hoc widget or
  `MainWindow` attributes.
- Follow the standard kit layout: `app.py`, a thin `gui/main_window.py`, focused
  `gui/panels/`, `gui/dialogs/`, optional `gui/widgets/`, and
  `config/schemas.py`.
- When fixing a shared pattern in one kit, inspect sibling kits for the same
  defect. Reuse shared worker, dialog, and configuration abstractions rather
  than copying boilerplate.

### Runtime, paths, and legacy code

- Runtime Gen-2 has one stored tier: `cpu`, `gpu`, or `gpu_fast`. Resolve it
  through `hydra_suite.runtime.resolver.RuntimeResolver`; inference consumers
  use `ResolvedBackend.backend` and `.device` rather than inventing new runtime
  strings.
- ONNX consumers obtain providers through
  `hydra_suite.runtime.onnx_providers.execution_providers_for`.
- Resolve bundled and user-writable paths through `hydra_suite.paths`. Never
  navigate to the repository root using `Path(__file__).parents[N]`. Bundled
  assets use `importlib.resources`; writable data uses `platformdirs` and the
  supported `HYDRA_DATA_DIR` / `HYDRA_CONFIG_DIR` overrides.
- Never import from `legacy/` in `src/` or `tests/`. Superseded code remains in
  `legacy/` for one release cycle and is excluded from normal checks.
- Preserve public CLI entry points and inter-kit APIs during the active
  simplification work unless the task explicitly authorizes a breaking change.
- Use the terminology `posekit` for the CLI and `hydra_suite.posekit` for the
  package; do not reintroduce legacy names.

## Active refactoring context

The repository is in a simplification sprint described by
`docs/superpowers/specs/2026-04-04-codebase-simplification-design.md`:

1. Migrate QThread workers to shared `BaseWorker`.
2. Move each kit to typed configuration schemas.
3. Migrate dialogs to `BaseDialog` and split dialog monoliths.
4. Decompose the TrackerKit `MainWindow` into focused modules.

Before changing a kit GUI, check whether the required shared abstraction
already exists or is planned. Do not add new boilerplate that conflicts with
this direction.

## Documentation lifecycle

When branch work is merged to `main`, move its completed plan from
`docs/superpowers/plans/` and its completed design spec from
`docs/superpowers/specs/` into the corresponding `done/` directories as part
of the same integration. Use `git mv` and do not rewrite the documents except
to update a stale status header to `Shipped — merged to main (<sha>)`.

Leave a document active when its checklist is incomplete, the design is
deferred or superseded, or no corresponding branch has merged.

## Output and safety conventions

- Do not use the `artifact-design` skill or create hosted Artifacts. Present
  visual results as repository/scratch files or inline output.
- Preserve unrelated dirty-worktree changes and avoid destructive git or file
  operations. Never use reset/checkout cleanup to erase user work.
- Prefer repository-native abstractions and focused changes over duplicated or
  speculative infrastructure.
