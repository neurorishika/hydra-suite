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

Linux and Windows need none of this: their wheels do not ship conflicting
libomp install names. Shared-library mismatches there (e.g. libstdc++) surface
in ``hydra doctor``'s import checks.

Idempotent. ``install.py`` runs it on the mps tier (``python -m
hydra_suite.runtime.macos_libomp``); ``make configure-mps-libs`` wraps that. It
acts on ``sys.prefix``, so run it with the target env's interpreter.
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


def _site_packages(prefix: Path) -> list[Path]:
    return sorted(prefix.glob("lib/python3.*/site-packages"))


def find_libomp_copies(prefix: Path) -> list[Path]:
    """Every real (non-symlink) libomp.dylib in the env: conda's, then wheels'."""
    hits: list[Path] = []
    conda_omp = prefix / "lib" / "libomp.dylib"
    if conda_omp.exists() and not conda_omp.is_symlink():
        hits.append(conda_omp)
    for site in _site_packages(prefix):
        for path in sorted(site.glob("**/libomp.dylib")):
            if not path.is_symlink():
                hits.append(path)
    return hits


def _canonical(copies: list[Path], prefix: Path) -> Path:
    """conda's copy wins (other conda libs link it); else torch's."""
    conda_omp = prefix / "lib" / "libomp.dylib"
    if conda_omp in copies:
        return conda_omp
    for path in copies:
        if "/torch/" in str(path):
            return path
    return copies[0]


def repair_dangling_links(prefix: Path) -> None:
    """A previous run replaced wheel copies with symlinks. A later reinstall
    (torch bump) or a removed conda libomp can leave those links dangling, and
    a dangling libomp makes every import fail. Restore the backup or drop it."""
    candidates = [prefix / "lib" / "libomp.dylib"]
    for site in _site_packages(prefix):
        candidates += list(site.glob("**/libomp.dylib"))
    for path in candidates:
        if not (path.is_symlink() and not path.exists()):
            continue
        backup = path.with_name(path.name + BACKUP_SUFFIX)
        path.unlink()
        if backup.exists():
            shutil.copy2(backup, path)
            print(f"  restored dangling {path} from its backup")
        else:
            print(f"  removed dangling {path}")


def configure(prefix: Path) -> int:
    repair_dangling_links(prefix)
    copies = find_libomp_copies(prefix)
    if len(copies) <= 1:
        print(f"  {len(copies)} libomp copy in the env -- nothing to do")
        return 0
    canonical = _canonical(copies, prefix)
    canonical_v = _dylib_id_versions(canonical)
    if canonical_v is None:
        print(f"  SKIP: could not read LC_ID_DYLIB from {canonical}")
        return 0
    for path in copies:
        if path == canonical:
            continue
        # Only relink when the two libraries agree on their ABI. A wheel needing
        # a newer libomp than the canonical copy must NOT be silently downgraded.
        version = _dylib_id_versions(path)
        if version != canonical_v:
            print(f"  SKIP {path}: libomp ABI {version} != canonical {canonical_v}")
            print("        Leaving it in place; expect OMP: Error #15 if both load.")
            continue
        backup = path.with_name(path.name + BACKUP_SUFFIX)
        if not backup.exists():
            shutil.copy2(path, backup)
        path.unlink()
        path.symlink_to(canonical)
        print(f"  linked {path} -> {canonical}")
    return 0


#: Import orders that load two OpenMP runtimes if the env has the defect.
IMPORT_ORDERS = (
    "import torch, cv2",
    "import cv2, torch",
    "import torch, sklearn.cluster, numba",
    "import sklearn.cluster, torch",
)


def verify(python: str) -> int:
    """Prove one OpenMP runtime maps, in the orders that actually abort."""
    failed = 0
    for code in IMPORT_ORDERS:
        env = {k: v for k, v in os.environ.items() if k != "KMP_DUPLICATE_LIB_OK"}
        r = subprocess.run(
            [python, "-c", code], capture_output=True, text=True, env=env
        )
        if r.returncode == 0:
            print(f"  verified: `{code}` succeeds")
            continue
        failed = 1
        print(f"  WARNING: `{code}` fails:")
        print("    " + (r.stderr.strip().splitlines() or ["<no output>"])[-1])
    return failed


def main() -> int:
    if platform.system() != "Darwin":
        print("macos_libomp: not macOS -- nothing to do")
        return 0
    prefix = Path(sys.prefix)
    print(f"macos_libomp: {prefix}")
    rc = configure(prefix)
    if rc:
        return rc
    return verify(sys.executable)


if __name__ == "__main__":
    sys.exit(main())
