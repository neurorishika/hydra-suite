#!/usr/bin/env python
"""Repair the duplicate OpenMP runtime in the macOS (MPS) conda environment.

The problem
-----------
pip's torch wheel vendors its own ``libomp.dylib``. Its ``LC_ID_DYLIB`` is the
absolute path ``/opt/llvm-openmp/lib/libomp.dylib``, whereas conda's copy (from
``llvm-openmp``) identifies as ``@rpath/libomp.dylib``. Because the two install
names differ, dyld cannot recognise them as the same library and maps both.
OpenMP then aborts the process::

    OMP: Error #15: Initializing libomp.dylib, but found libomp.dylib already
    initialized.

In practice this kills any process that loads both torch and cv2 -- which is
most of this package. It is load-order sensitive (``import cv2, torch`` happens
to survive; ``import torch, cv2`` aborts), so it surfaces intermittently and
looks like an unrelated crash.

The fix
-------
Point torch's copy at conda's, so exactly one OpenMP runtime maps. The original
is kept alongside as ``libomp.dylib.orig-backup``.

Why not the alternatives
------------------------
* ``KMP_DUPLICATE_LIB_OK=TRUE`` -- LLVM documents this as possibly producing
  *silently incorrect results*. For a package doing sub-pixel keypoint work that
  is the one failure mode we cannot accept.
* "import cv2 before torch" -- does not work. isort sorts third-party ``import
  torch`` ahead of first-party ``from hydra_suite...``, so torch wins the race
  in any module importing both.
* ``DYLD_LIBRARY_PATH`` -- does not override an absolute ``LC_ID_DYLIB``.

Linux needs none of this: ``configure-cuda-ort`` already writes an ``activate.d``
hook that puts ``$CONDA_PREFIX/lib`` on ``LD_LIBRARY_PATH``, which resolves the
analogous (but distinct) libstdc++ conflict there.

Idempotent. Run via ``make configure-mps-libs``; ``install-mps`` invokes it, so a
rebuilt environment is repaired automatically.
"""

from __future__ import annotations

import os
import platform
import re
import shutil
import subprocess
import sys
from pathlib import Path

BACKUP_SUFFIX = ".orig-backup"


def _dylib_id_versions(path: Path) -> tuple[str, str] | None:
    """Return (compatibility_version, current_version) from LC_ID_DYLIB.

    Returns None if otool is unavailable or the load command is unreadable --
    callers treat that as "cannot verify", not as "compatible".
    """
    try:
        out = subprocess.run(
            ["otool", "-l", str(path)],
            capture_output=True,
            text=True,
            check=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError):
        return None

    block = out.split("LC_ID_DYLIB", 1)
    if len(block) < 2:
        return None
    compat = re.search(r"compatibility version ([\d.]+)", block[1])
    current = re.search(r"current version ([\d.]+)", block[1])
    if not compat or not current:
        return None
    return compat.group(1), current.group(1)


def _find_torch_libomp(prefix: Path) -> Path | None:
    hits = sorted(prefix.glob("lib/python3.*/site-packages/torch/lib/libomp.dylib"))
    return hits[0] if hits else None


def configure(prefix: Path) -> int:
    conda_omp = prefix / "lib" / "libomp.dylib"
    torch_omp = _find_torch_libomp(prefix)

    if torch_omp is None:
        print("  torch does not vendor libomp.dylib here — nothing to do")
        return 0
    if not conda_omp.exists():
        print(f"  SKIP: no conda libomp at {conda_omp}")
        print("        (install llvm-openmp, or leave torch's vendored copy alone)")
        return 0

    if torch_omp.is_symlink() and torch_omp.resolve() == conda_omp.resolve():
        print("  already pointed at the conda libomp — nothing to do")
        return 0

    # Only relink when the two libraries agree on their ABI. A future torch
    # needing a newer libomp than conda ships must NOT be silently downgraded.
    conda_v = _dylib_id_versions(conda_omp)
    torch_v = _dylib_id_versions(torch_omp)
    if conda_v is None or torch_v is None:
        print("  SKIP: could not read LC_ID_DYLIB from both libraries")
        return 0
    if conda_v != torch_v:
        print(f"  SKIP: libomp ABI differs — conda {conda_v} vs torch {torch_v}")
        print("        Relinking could break torch. Leaving both in place;")
        print("        expect OMP: Error #15 until the versions reconverge.")
        return 0

    backup = torch_omp.with_name(torch_omp.name + BACKUP_SUFFIX)
    if not backup.exists():
        shutil.copy2(torch_omp, backup)
        print(f"  backed up torch's libomp -> {backup.name}")

    torch_omp.unlink()
    torch_omp.symlink_to(conda_omp)
    print(f"  linked {torch_omp} -> {conda_omp}")
    print(f"  one OpenMP runtime now maps (compat {conda_v[0]})")
    return 0


def verify(prefix: Path) -> int:
    """Prove the fix worked in the order that actually aborts."""
    py = prefix / "bin" / "python"
    if not py.exists():
        return 0
    code = "import torch, cv2"  # the aborting order, pre-fix
    r = subprocess.run([str(py), "-c", code], capture_output=True, text=True)
    if r.returncode == 0:
        print("  verified: `import torch, cv2` succeeds")
        return 0
    print("  WARNING: `import torch, cv2` still fails:")
    print("    " + (r.stderr.strip().splitlines() or ["<no output>"])[-1])
    return 1


def main() -> int:
    if platform.system() != "Darwin":
        print("configure-mps-libs: not macOS — nothing to do")
        print("  (Linux's libstdc++ conflict is handled by configure-cuda-ort)")
        return 0

    prefix_str = os.environ.get("CONDA_PREFIX")
    if not prefix_str:
        print("ERROR: activate the MPS conda env first (conda activate hydra-mps)")
        return 1

    print(f"configure-mps-libs: {prefix_str}")
    prefix = Path(prefix_str)
    rc = configure(prefix)
    if rc:
        return rc
    return verify(prefix)


if __name__ == "__main__":
    sys.exit(main())
