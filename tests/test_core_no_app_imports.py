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
)
CORE_ROOT = pathlib.Path(__file__).resolve().parents[1] / "src" / "hydra_suite" / "core"


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
    for py in CORE_ROOT.rglob("*.py"):
        for mod in _imports(py):
            if mod.startswith(APP_PACKAGES):
                offenders.append(f"{py.relative_to(CORE_ROOT)} -> {mod}")
    assert not offenders, "core/ imports app layers:\n" + "\n".join(offenders)


def test_model_paths_importable_from_core():
    from hydra_suite.core.inference import model_paths

    assert hasattr(model_paths, "resolve_model_path")
    assert hasattr(model_paths, "get_yolo_model_metadata")
