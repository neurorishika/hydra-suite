"""Leaf-purity AST checks for hydra_suite.core.identity.pose.vitpose.

`vitpose/` (including the `training/` subpackage) must never import from
`hydra_suite` — it is a standalone leaf. `vitpose/__init__.py` must also not
eagerly import the `training` subpackage, so pure-inference importers never
load the training loop.
"""

import ast
from pathlib import Path

VITPOSE = Path("src/hydra_suite/core/identity/pose/vitpose")


def test_training_subpackage_is_leaf_pure():
    offenders = []
    for py in VITPOSE.rglob("*.py"):
        tree = ast.parse(py.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                if any(a.name.split(".")[0] == "hydra_suite" for a in node.names):
                    offenders.append(str(py))
            elif isinstance(node, ast.ImportFrom):
                if (node.module or "").split(".")[
                    0
                ] == "hydra_suite" and node.level == 0:
                    offenders.append(str(py))
    assert not offenders, f"leaf-impure files: {sorted(set(offenders))}"


def test_init_does_not_eager_import_training():
    src = (VITPOSE / "__init__.py").read_text(encoding="utf-8")
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and (node.module or "").startswith(
            "training"
        ):
            raise AssertionError("vitpose/__init__.py must not import training/")
        if (
            isinstance(node, ast.ImportFrom)
            and node.level > 0
            and (node.module or "") == "training"
        ):
            raise AssertionError("vitpose/__init__.py must not import training/")
