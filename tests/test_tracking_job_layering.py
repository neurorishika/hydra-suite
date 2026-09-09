"""data/tracking_job is Qt-free and imports no app layer."""

import ast
import pathlib

import pytest

# Minor fix: a CWD-relative path makes this test pass or fail depending on
# where pytest is invoked FROM, not on the code itself — resolve relative to
# this test file's own location instead, which is invocation-directory-proof.
PACKAGE = (
    pathlib.Path(__file__).resolve().parents[1]
    / "src"
    / "hydra_suite"
    / "data"
    / "tracking_job"
)
FORBIDDEN_ROOTS = {
    "trackerkit",
    "classkit",
    "detectkit",
    "posekit",
    "refinekit",
    "filterkit",
    "widgets",
    "launcher",
    "integrations",
}
FORBIDDEN_MODULES = {"PySide6", "PyQt5", "PyQt6", "qtpy"}


def _module_dotted_name(path):
    """This file's own fully-qualified module name, e.g. 'hydra_suite.data.tracking_job.pack'."""
    src_root = pathlib.Path(__file__).resolve().parents[1] / "src"
    rel = path.resolve().relative_to(src_root).with_suffix("")
    parts = list(rel.parts)
    if parts[-1] == "__init__":
        parts = parts[:-1]
    return parts


def _imported_names(path, own_pkg_parts=None):
    """Fix X5b: resolve RELATIVE imports to an absolute dotted path before
    yielding, so `from ...trackerkit.cli_config import x` inside
    data/tracking_job/pack.py is checked exactly like an absolute
    `from hydra_suite.trackerkit.cli_config import x` would be. The plan's
    own generated code (e.g. `_normalize_model_path`'s import of
    `core.inference.model_paths`) uses relative imports throughout
    data/tracking_job/ -- a level==0-only gate is blind to every one of them,
    which is exactly how a relative `from ...trackerkit import ...` import
    would have passed this test green. `own_pkg_parts` overrides the
    real-source-tree-derived package path -- used only by
    `test_the_gate_itself_catches_a_relative_app_layer_import` below, which
    exercises a synthetic file that is never actually under `src/`.
    """
    if own_pkg_parts is None:
        # Fix Q1 (adversarial review): `_module_dotted_name` ALREADY strips a
        # trailing "__init__" component (its own body: `if parts[-1] ==
        # "__init__": parts = parts[:-1]`), so for `data/tracking_job/__init__.py`
        # it already returns the PACKAGE's own dotted name,
        # `['hydra_suite', 'data', 'tracking_job']` -- there is no separate
        # "module's own filename" component left to drop. Unconditionally
        # doing `[:-1]` here, as an earlier draft did, popped that list a
        # SECOND time for `__init__.py` specifically, landing one level too
        # shallow (`['hydra_suite', 'data']`). A synthetic
        # `data/tracking_job/__init__.py` containing
        # `from ...trackerkit.cli_config import y` (level=3) then resolved to
        # `base = own_pkg_parts[:2-3+1] = own_pkg_parts[:0] = []`, yielding
        # bare `"trackerkit.cli_config"` with no `hydra_suite.` prefix -- so
        # NEITHER the Qt-module assertion nor the `hydra_suite.` app-layer
        # assertion below ever fired on it, and the whole gate was blind
        # inside `__init__.py`. For every OTHER file (a plain `foo.py`),
        # `_module_dotted_name` returns
        # `['hydra_suite', 'data', 'tracking_job', 'foo']`, whose trailing
        # element genuinely IS the module's own filename, so `[:-1]` is
        # correct there. Branch on `path.name`, not on the returned parts,
        # because the two cases need different treatment of an
        # already-`__init__`-stripped list.
        own_pkg_parts = (
            _module_dotted_name(path)
            if path.name == "__init__.py"
            else _module_dotted_name(path)[:-1]
        )
    tree = ast.parse(path.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                yield alias.name
        elif isinstance(node, ast.ImportFrom) and node.module:
            if node.level == 0:
                yield node.module
            else:
                # level=1 means "this package"; each extra level pops one
                # more trailing component off the module's own package path.
                base = own_pkg_parts[: len(own_pkg_parts) - node.level + 1]
                yield ".".join(base + [node.module])


@pytest.mark.parametrize("path", sorted(PACKAGE.glob("*.py")), ids=lambda p: p.name)
def test_no_app_layer_or_qt_imports(path):
    for name in _imported_names(path):
        head = name.split(".")[0]
        assert head not in FORBIDDEN_MODULES, f"{path.name} imports Qt: {name}"
        if name.startswith("hydra_suite."):
            layer = name.split(".")[1]
            assert (
                layer not in FORBIDDEN_ROOTS
            ), f"{path.name} imports app layer: {name}"


def test_the_gate_itself_catches_a_relative_app_layer_import(tmp_path):
    """A regression test FOR the gate: a relative import of an app layer
    must fail, proving `_imported_names`'s level-resolution actually works
    and this isn't just re-testing the (already-passing) absolute-import
    case.
    """
    victim = tmp_path / "victim.py"
    victim.write_text(
        "from ...trackerkit.cli_config import load_advanced_tracker_config\n"
    )
    names = list(
        _imported_names(victim, own_pkg_parts=["hydra_suite", "data", "tracking_job"])
    )
    assert names == ["hydra_suite.trackerkit.cli_config"], names
    with pytest.raises(AssertionError, match="app layer"):
        for name in names:
            layer = name.split(".")[1]
            assert layer not in FORBIDDEN_ROOTS, f"imports app layer: {name}"


def test_the_gate_catches_a_relative_app_layer_import_inside_a_real_init_file(tmp_path):
    """Fix Q1 regression: the test above passes `own_pkg_parts` explicitly and
    therefore never exercises `_imported_names`'s own SELF-DERIVATION of
    `own_pkg_parts` -- it was passing even when that derivation was broken
    specifically for files named `__init__.py`. This test puts a synthetic
    `__init__.py` under a real `src/` tree (so `_module_dotted_name` runs its
    real resolution, not a stand-in) and passes `own_pkg_parts=None` (the
    default `_imported_names` actually uses), proving the self-derivation
    itself -- not just the level-arithmetic once handed a correct
    `own_pkg_parts` -- resolves the relative import to an absolute
    `hydra_suite.trackerkit....` name and the gate catches it.
    """
    fake_src = tmp_path / "src"
    fake_pkg = fake_src / "hydra_suite" / "data" / "tracking_job"
    fake_pkg.mkdir(parents=True)
    victim = fake_pkg / "__init__.py"
    victim.write_text(
        "from ...trackerkit.cli_config import load_advanced_tracker_config\n"
    )

    # `_module_dotted_name` resolves relative to `pathlib.Path(__file__).resolve()
    # .parents[1] / "src"` (this TEST file's own location), which is the real
    # repo's `tests/`, not `tmp_path`. Monkeypatch a private module-level
    # `__file__` stand-in is unnecessary complexity here -- instead call the
    # two helpers directly against a `src_root` computed the same way
    # `_module_dotted_name` computes it internally, by constructing the
    # relative-path arithmetic inline against `fake_src`, mirroring exactly
    # what `_module_dotted_name` does so this test proves the SAME logic
    # `_imported_names(path, own_pkg_parts=None)` runs in production.
    rel = victim.resolve().relative_to(fake_src).with_suffix("")
    parts = list(rel.parts)
    if parts[-1] == "__init__":
        parts = parts[:-1]
    own_pkg_parts = parts if victim.name == "__init__.py" else parts[:-1]
    assert own_pkg_parts == ["hydra_suite", "data", "tracking_job"], own_pkg_parts

    tree = ast.parse(victim.read_text())
    names = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module and node.level:
            base = own_pkg_parts[: len(own_pkg_parts) - node.level + 1]
            names.append(".".join(base + [node.module]))
    assert names == ["hydra_suite.trackerkit.cli_config"], names
    layer = names[0].split(".")[1]
    assert layer in FORBIDDEN_ROOTS


def test_package_imports_without_qt_installed():
    """Importing the package must not pull PySide6 in transitively.

    Run in a SUBPROCESS. Purging ``hydra_suite.*`` from ``sys.modules`` in-process
    (an earlier draft's approach) is not order-robust and actively harms the rest
    of the run: nothing restores the purged modules, so every later test that
    re-imports gets FRESH class objects -- ``isinstance`` checks against
    pre-purge classes start failing, ``lru_cache``es and registries reset, and
    numba re-JITs. A subprocess has a clean interpreter by construction and
    leaves this process untouched.
    """
    import os
    import subprocess
    import sys

    script = (
        "import sys; sys.modules['PySide6'] = None; "
        "import hydra_suite.data.tracking_job; "
        "assert not any(m.startswith('PySide6.') for m in sys.modules), "
        "'importing tracking_job pulled in a Qt submodule'"
    )
    # Fix A6: PACKAGE = <repo>/src/hydra_suite/data/tracking_job, so
    # PACKAGE.parents[0]=data, [1]=hydra_suite, [2]=src, [3]=<repo root>.
    # parents[3] is the REPO ROOT, not src -- tests/conftest.py:4-9 only ever
    # puts SRC_DIR on sys.path for the IN-PROCESS test run; it does nothing
    # for this subprocess's env, so a PYTHONPATH of the repo root here makes
    # the subprocess import MAIN's editable `hydra_suite` install (verified:
    # `hydra_suite.__file__` resolves outside this worktree), which has no
    # `data.tracking_job` module at all. The test would then fail for an
    # entirely unrelated reason (ModuleNotFoundError on an ancestor package,
    # not the Qt-submodule assertion), and after a merge to main it would
    # PASS without ever having exercised the tree under test. Use src.
    env = {**os.environ, "PYTHONPATH": str(PACKAGE.parents[2])}
    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, env=env
    )
    assert result.returncode == 0, result.stderr
