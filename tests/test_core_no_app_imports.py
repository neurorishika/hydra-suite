"""core/ must never import from any app-layer package."""

import ast
import pathlib

APP_PACKAGES = (
    "hydra_suite.trackerkit",
    "hydra_suite.posekit",
    "hydra_suite.classkit",
    "hydra_suite.refinekit",
    "hydra_suite.detectkit",
    "hydra_suite.filterkit",
    "hydra_suite.integrations",
    "PySide6",
    "PyQt5",
    "PyQt6",
)
CORE_ROOT = pathlib.Path(__file__).resolve().parents[1] / "src" / "hydra_suite" / "core"

# Pre-existing, load-bearing violations tracked as tech debt (not introduced by this work).
# The SLEAP pose backend lazily imports the integrations service bridge; retiring this is
# its own effort (see the core/ Qt-in-core + integrations-boundary cleanup). Allow-listed so
# this guard still fails on any NEW core->app/integrations import.
KNOWN_VIOLATIONS = {
    ("identity/pose/backends/sleap.py", "hydra_suite.integrations.sleap.service"),
}


def _imports(path):
    tree = ast.parse(path.read_text(), filename=str(path))
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            yield node.module
        elif isinstance(node, ast.Import):
            for alias in node.names:
                yield alias.name


def test_core_has_no_app_layer_imports():
    offenders = []
    found_known = set()
    for py in CORE_ROOT.rglob("*.py"):
        relpath = str(py.relative_to(CORE_ROOT))
        for mod in _imports(py):
            if mod.startswith(APP_PACKAGES):
                key = (relpath, mod)
                if key in KNOWN_VIOLATIONS:
                    found_known.add(key)
                else:
                    offenders.append(f"{relpath} -> {mod}")
    assert not offenders, "core/ imports app layers:\n" + "\n".join(offenders)

    stale = KNOWN_VIOLATIONS - found_known
    assert not stale, (
        "KNOWN_VIOLATIONS allowlist entries no longer present in core/ — "
        "delete these stale entries:\n" + "\n".join(f"{p} -> {m}" for p, m in stale)
    )


def test_model_paths_importable_from_core():
    from hydra_suite.core.inference import model_paths

    assert hasattr(model_paths, "resolve_model_path")
    assert hasattr(model_paths, "get_yolo_model_metadata")
