"""Every source file must parse under the MINIMUM supported Python.

`pyproject.toml` declares `requires-python = ">=3.11"`, but multi-line
expressions inside f-strings (PEP 701) are valid only from 3.12. Such a file
imports fine on a 3.13 dev machine and fails outright on 3.11 -- and the first
place it surfaced here was the docs CI job (Python 3.11), where griffe could not
parse `core/tracking/worker.py` and mkdocstrings reported the far more
confusing `hydra_suite.core.tracking.worker could not be found`.

`ast.parse(..., feature_version=(3, 11))` does NOT reject PEP 701 (verified),
so this scans for the syntax directly.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCANNED_DIRS = ("src", "tests", "tools")

# An f-string prefix whose `{` opens an expression that continues on the next
# line: `f"... {` with nothing after the brace. Valid from 3.12, a SyntaxError
# on 3.11.
_MULTILINE_FSTRING_EXPR = re.compile(
    r"""(?<![A-Za-z0-9_])[fF][rRbB]?["'][^"']*\{\s*$"""
)


def _python_files() -> list[Path]:
    files: list[Path] = []
    for directory in SCANNED_DIRS:
        root = REPO_ROOT / directory
        if root.is_dir():
            files.extend(
                p
                for p in root.rglob("*.py")
                if "__pycache__" not in p.parts and ".worktrees" not in p.parts
            )
    return sorted(files)


def test_no_multiline_fstring_expressions() -> None:
    """PEP 701 multi-line f-string expressions break Python 3.11."""
    offenders: list[str] = []
    for path in _python_files():
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except (OSError, UnicodeDecodeError):
            continue
        for number, line in enumerate(lines, start=1):
            if _MULTILINE_FSTRING_EXPR.search(line):
                rel = path.relative_to(REPO_ROOT)
                offenders.append(f"{rel}:{number}: {line.strip()}")

    assert not offenders, (
        "Multi-line f-string expressions (PEP 701) are valid only on Python "
        ">= 3.12, but pyproject.toml declares requires-python = '>=3.11'. "
        "These files will not even parse on 3.11, which breaks the docs CI job "
        "(it runs 3.11) with a misleading 'could not be found' from "
        "mkdocstrings. Put the whole expression on one line:\n  "
        + "\n  ".join(offenders)
    )


def test_the_guard_actually_detects_the_pattern() -> None:
    """The regex must match the real shape, or the test above is vacuous."""
    assert _MULTILINE_FSTRING_EXPR.search('    f"Frame {')
    assert _MULTILINE_FSTRING_EXPR.search("    f'Frame {   ")
    # ... and must not fire on ordinary, single-line f-strings.
    assert not _MULTILINE_FSTRING_EXPR.search('    f"Frame {n} of {total}"')
    assert not _MULTILINE_FSTRING_EXPR.search('    logger.info("plain {")')
    # A bare `f` ending an ordinary word must not be read as an f-prefix
    # (this exact false positive fired on `"perf": {` first time round).
    assert not _MULTILINE_FSTRING_EXPR.search('        "perf": {')


@pytest.mark.parametrize("directory", SCANNED_DIRS)
def test_scanner_sees_files(directory: str) -> None:
    """Guard against the scan silently covering nothing."""
    root = REPO_ROOT / directory
    if not root.is_dir():
        pytest.skip(f"{directory}/ not present")
    assert any(p.parent == root or root in p.parents for p in _python_files())
