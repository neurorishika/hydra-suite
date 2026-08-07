# tests/test_core_qtfree_slice2.py
import importlib
import pathlib
import subprocess

CORE = pathlib.Path(__file__).resolve().parents[1] / "src" / "hydra_suite" / "core"


def test_core_has_no_qt_imports():
    # Real Qt IMPORT statements only. A bare substring scan for
    # "Signal|Slot|QThread" yields false positives on core identifiers
    # (IdentitySlotLockLabel, "Slot %d" log strings, a QThread docstring
    # mention) that are not Qt usage — so guard on import statements, which
    # is what "core must not depend on Qt" actually means. Mirrors the
    # existing AST guard in tests/test_core_no_app_imports.py.
    out = subprocess.run(
        [
            "grep",
            "-rnE",
            r"^\s*(from|import)\s+PySide6|^\s*(from|import)\s+PyQt",
            str(CORE),
        ],
        capture_output=True,
        text=True,
    )
    assert out.stdout.strip() == "", f"Qt imports leaked into core/:\n{out.stdout}"


def test_new_core_modules_import_without_qt():
    for mod in (
        "hydra_suite.core.tracking.session",
        "hydra_suite.core.tracking.errors",
        "hydra_suite.core.post.merge",
        "hydra_suite.core.post.pose_merge",
        "hydra_suite.core.post.rich_export",
        "hydra_suite.core.post.interpolated_crops",
        "hydra_suite.core.individual.postprocess_df",
    ):
        importlib.import_module(mod)
