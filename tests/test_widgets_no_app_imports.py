"""The shared ``widgets`` layer must not import from any app layer."""

from __future__ import annotations

import ast
from pathlib import Path

APP_LAYERS = (
    "trackerkit",
    "posekit",
    "classkit",
    "refinekit",
    "detectkit",
    "filterkit",
    "launcher",
)


def _imported_modules(tree: ast.AST, package: str) -> list[str]:
    names: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            if node.level:  # relative: resolve against hydra_suite.widgets
                parts = package.split(".")[: len(package.split(".")) - node.level + 1]
                module = ".".join(parts + ([module] if module else []))
            names.append(module)
    return names


def test_widgets_import_no_app_layer():
    root = Path(__file__).resolve().parents[1] / "src" / "hydra_suite" / "widgets"
    files = list(root.rglob("*.py"))
    assert files
    for path in files:
        tree = ast.parse(path.read_text())
        for name in _imported_modules(tree, "hydra_suite.widgets"):
            for app in APP_LAYERS:
                assert not (
                    name == f"hydra_suite.{app}"
                    or name.startswith(f"hydra_suite.{app}.")
                    or name == app
                    or name.startswith(f"{app}.")
                ), (path, name)
