"""Cross-platform helpers for invoking the ``conda`` CLI via subprocess.

On Windows, ``conda`` is a batch script (``condabin\\conda.bat``), not a native
``.exe``. ``subprocess`` can only launch batch/cmd scripts when ``shell=True``
(``CreateProcess`` otherwise only resolves a bare command name to ``.exe``), so
a plain ``subprocess.run(["conda", ...])`` fails with ``FileNotFoundError`` on
Windows even when ``condabin`` is correctly on ``PATH``. On POSIX, ``conda`` is
a regular executable/script and needs no such treatment.
"""

import platform
import subprocess
from typing import Any, Dict, List


def conda_subprocess_kwargs() -> Dict[str, Any]:
    """Extra subprocess kwargs needed to run a ``conda`` command on this platform."""
    return {"shell": True} if platform.system() == "Windows" else {}


def run_conda(args: List[str], **kwargs: Any) -> "subprocess.CompletedProcess[str]":
    """``subprocess.run`` wrapper that works with conda's Windows batch-script entry point."""
    return subprocess.run(args, **conda_subprocess_kwargs(), **kwargs)


def popen_conda(args: List[str], **kwargs: Any) -> "subprocess.Popen[str]":
    """``subprocess.Popen`` wrapper that works with conda's Windows batch-script entry point."""
    return subprocess.Popen(args, **conda_subprocess_kwargs(), **kwargs)
