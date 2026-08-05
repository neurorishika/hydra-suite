import subprocess
from pathlib import Path


def test_core_has_no_qt_imports():
    root = Path(__file__).resolve().parents[1] / "src" / "hydra_suite" / "core"
    proc = subprocess.run(
        [
            "grep",
            "-rnE",
            "from PySide6|import PySide6|from PyQt|import PyQt",
            str(root),
        ],
        capture_output=True,
        text=True,
    )
    # grep exit code 1 == no matches (the required Qt-free state).
    assert proc.returncode == 1, f"Real Qt imports found in core/:\n{proc.stdout}"


def test_media_and_dataset_export_import_clean():
    import hydra_suite.core.post.dataset_export  # noqa: F401
    import hydra_suite.core.post.media_export  # noqa: F401
