# Project Guidelines

## Start Here

- Treat `CLAUDE.md` as the high-level architecture and refactoring brief.
- Link to existing docs instead of duplicating them:
  - `../CLAUDE.md`
  - `../docs/developer-guide/architecture.md`
  - `../docs/developer-guide/contributing.md`
  - `../docs/developer-guide/code-health.md`
  - `../docs/getting-started/environments.md`
  - `../docs/developer-guide/runtime-integration.md`
  - `PRE_COMMIT_GUIDE.md`
  - `CODE_QUALITY_GUIDE.md`

## Python Environments

- Run repository Python, pytest, and CLI commands inside a HYDRA conda environment, not the system interpreter.
- Prefer the currently active project environment when it already has the needed dependencies.
- If no environment is active, choose an available environment in this order:
  - repo-local `.conda` via `conda run -p ./.conda ...`
  - Apple Silicon macOS: `hydra-mps`, then `hydra`
  - NVIDIA systems: `hydra-cuda`, then `hydra`
- When terminal state is uncertain, prefer `conda run -n <env> ...` or `conda run -p ./.conda ...` for one-off commands.
- Install missing dependencies into the selected HYDRA environment, not globally: `make install`, `make install-mps`, `make install-cuda`, `make install-dev`, `make docs-install`.

## Build And Test

- Use the Makefile as the source of truth for routine commands.
- Format with `make format`.
- Lint with `make lint`; use `make lint-strict` only when explicitly needed.
- Run tests with `make pytest` or a focused pytest command in the selected environment.
- Run coverage with `make test-cov` or `make test-cov-html` after `make install-dev`.
- Run docs verification with `make docs-build` or `make docs-check` when changing docs, config semantics, runtime behavior, or user-visible workflows.
- Run `make audit` for broad health checks on larger changes.
- The docs currently mention `make commit-prep`, but that target is not in the Makefile; use `make format` unless the alias is added.

## Architecture

- Preserve the dependency direction: app kits may import shared/core layers, but `core`, `runtime`, `data`, `training`, and `utils` must not import from app-layer packages.
- Keep shared UI code in `hydra_suite.widgets`; do not add app-layer imports there.
- Use `hydra_suite.paths` for asset, config, and data locations. Do not derive repository paths from `__file__`.
- Never import from `legacy/` into `src/` or `tests/`.
- Keep GUI coordinators thin. In GUI work, prefer existing abstractions:
  - `src/hydra_suite/widgets/workers.py` for `BaseWorker`
  - `src/hydra_suite/widgets/dialogs.py` for `BaseDialog`
  - `<kit>/config/schemas.py` for typed runtime state
- During the simplification sprint, avoid adding new business logic back into large `main_window.py` files when an orchestrator, panel, worker, dialog, or config schema is the correct home.

## Conventions

- Match existing code style and keep edits minimal; avoid unrelated refactors.
- Keep public CLI entry points and cross-kit APIs stable unless the task explicitly requires changing them.
- When you fix a shared pattern in one kit, check whether the same abstraction or cleanup should be applied consistently across other kits.
- Update nearby documentation when behavior, runtime support, path semantics, or configuration persistence changes.
- Prefer moderate-lint-compatible code paths and use `make lint` as the default validation gate.
