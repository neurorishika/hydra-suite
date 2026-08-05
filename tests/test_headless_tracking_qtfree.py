"""Executable definition of done: the CLI tracks with PySide6 blocked from import."""

from __future__ import annotations

import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
FLY_CLIP = REPO / "tools/equivalence/fixtures/clips/fly_obb.mp4"
FLY_CONFIG = REPO / "tools/equivalence/fixtures/configs/fly_obb.json"

# A sys.meta_path finder that raises ImportError on ANY PySide6 import. Injected
# at interpreter start so even a lazy `import PySide6` deep in the CLI path fails.
_BLOCKER_PREAMBLE = textwrap.dedent("""
    import sys

    class _BlockPySide6:
        def find_spec(self, name, path=None, target=None):
            if name == "PySide6" or name.startswith("PySide6."):
                raise ImportError(f"PySide6 import blocked by Qt-free guard: {name}")
            return None

    sys.meta_path.insert(0, _BlockPySide6())
    """)


def test_headless_tracking_imports_with_pyside6_blocked():
    """Importing the CLI runtime path must not require PySide6."""
    script = _BLOCKER_PREAMBLE + textwrap.dedent("""
        import hydra_suite.trackerkit.headless_tracking  # noqa: F401
        import hydra_suite.trackerkit.cli  # noqa: F401
        import PySide6  # this line MUST raise, proving the blocker is live
        raise SystemExit("PySide6 was importable - blocker not active")
        """)
    proc = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        cwd=str(REPO),
    )
    # The final `import PySide6` raises ImportError -> nonzero exit with that text;
    # the two CLI imports above it must have SUCCEEDED (no traceback naming them).
    assert "PySide6 import blocked by Qt-free guard" in proc.stderr, proc.stderr
    assert "headless_tracking" not in proc.stderr, proc.stderr
    assert "trackerkit/cli" not in proc.stderr, proc.stderr


@pytest.mark.skipif(
    not (FLY_CLIP.exists() and FLY_CONFIG.exists()),
    reason="fly_obb fixture not present (run tools/equivalence/fixtures/fetch_fixtures.sh)",
)
def test_cli_tracks_to_completion_with_pyside6_blocked(tmp_path):
    """THE executable DoD: trackerkit track completes + writes a non-empty CSV, no PySide6."""
    clip = tmp_path / "fly_obb.mp4"
    clip.write_bytes(FLY_CLIP.read_bytes())  # copy so outputs land in tmp_path

    script = _BLOCKER_PREAMBLE + textwrap.dedent(f"""
        from hydra_suite.trackerkit.cli import run_tracking_cli
        code = run_tracking_cli([{str(clip)!r}], config_path={str(FLY_CONFIG)!r})
        raise SystemExit(int(code))
        """)
    proc = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        cwd=str(REPO),
    )
    assert proc.returncode == 0, f"CLI failed under blocked PySide6:\n{proc.stderr}"

    # Direct path writes <clip>_tracking_forward_processed.csv next to the clip.
    csvs = list(tmp_path.glob("*_forward_processed.csv")) + list(
        tmp_path.glob("*_final.csv")
    )
    assert csvs, f"no output CSV produced; stderr:\n{proc.stderr}"
    rows = sum(1 for _ in csvs[0].open())
    assert rows > 1, f"CSV {csvs[0]} has only {rows} line(s) (header-only or empty)"
