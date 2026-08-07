# Code Health & Dependency Auditing

This page documents the integrated toolchain for diagnosing orphaned code, mapping
module dependencies, and enforcing type safety. All tools are available after a
standard `make docs-install` (or by installing the environments, which now include
`vulture`, `pylint`, and `mypy`).

---

## Quick Reference

| Goal | Command |
|------|---------|
| Find dead/orphaned code | `make dead-code` |
| Whitelist false-positives | `make dead-code-whitelist` |
| Visual dependency graph | `make dep-graph` |
| Text dependency map | `make dep-graph-text` |
| Static type checking | `make type-check` |
| Full audit (all of the above + coverage) | `make audit` |

---

## Dead Code Detection — `vulture`

[Vulture](https://github.com/jendrikseipp/vulture) scans Python source for unused
functions, classes, variables, and imports.

```bash
make dead-code                # prints items with ≥80% confidence of being unused
make dead-code-whitelist      # writes vulture_whitelist.py for false-positive review
```

### Interpreting results

- **100% confidence**: safe to delete (private helpers, unreachable branches).
- **60–80% confidence**: review manually; may be called via `getattr`, signals, or
  Qt dynamic dispatch.
- **Attributes on QThread/QObject subclasses**: always whitelist — Qt calls them
  dynamically.

### Whitelist workflow

```bash
make dead-code-whitelist          # generates vulture_whitelist.py
# open vulture_whitelist.py, REMOVE entries for genuinely dead code
vulture src/hydra_suite vulture_whitelist.py   # re-run with whitelist
```

Commit `vulture_whitelist.py` to preserve the false-positive decisions.

---

## Dependency Graph — `pydeps` & `pyreverse`

### Visual SVG graph

```bash
make dep-graph        # writes hydra_suite.svg
```

Opens cleanly in any browser. `--max-bacon=4` limits transitive depth; increase to
`6` or remove the flag to see the full graph.

### Text / DOT map

```bash
make dep-graph-text   # writes .audit/packages_hydra_suite.dot
```

Useful for diffing architecture changes in CI without requiring a display.
Convert to PNG offline:

```bash
dot -Tpng .audit/packages_hydra_suite.dot -o dep_map.png
```

### Reading the graph

- **Clusters** = top-level packages (`core`, `gui`, `data`, `posekit`, …).
- **Thick edges** = many imports across modules — candidates for interface extraction.
- **Isolated nodes** = modules with no importers — strong orphan signal.

---

## Static Type Checking — `mypy`

```bash
make type-check
```

Runs `mypy` in lenient mode (`--ignore-missing-imports`, no strict untyped-def
enforcement) so it integrates cleanly with the mixed-annotation codebase.

To tighten incrementally:

```bash
mypy src/hydra_suite/core/individual/ --strict
```

Type errors in `legacy/` are excluded via `pyproject.toml`
`[tool.mypy]` configuration.

---

## Unused Imports — `ruff`

`ruff` (already part of the lint pipeline) handles unused imports separately from
`vulture`:

```bash
ruff check src/hydra_suite --select F401   # unused imports
ruff check src/hydra_suite --select F811   # redefined before use
ruff check --fix --select F401 src/hydra_suite  # auto-remove safe ones
```

This integrates with `make lint-autofix`.

---

## Coverage-Guided Orphan Analysis

Test coverage is the highest-confidence dead-code signal. Zero-coverage modules that
are also flagged by `vulture` are the safest candidates for removal.

```bash
make test-cov-html    # generates htmlcov/index.html
```

Look for modules with **0% coverage** in the HTML report. Cross-reference with
`make dead-code` output. If a module has 0% coverage AND `vulture` flags its public
API as unused, it can be deleted or moved to `legacy/`.

---

## Full Audit

```bash
make audit
```

Runs: `dead-code` → `dep-graph` → `type-check` → `test-cov` in sequence. Artifacts
are written to:

- `hydra_suite.svg` — visual dependency graph
- `htmlcov/index.html` — coverage report


---

## Recommended Cadence

| Frequency | Action |
|-----------|--------|
| Every PR | `make lint-autofix` + `make format` |
| Weekly | `make dead-code` — review new entries |
| Monthly | `make audit` — full health snapshot |
| Before major release | `make dep-graph` — verify no unintended cross-layer deps |
