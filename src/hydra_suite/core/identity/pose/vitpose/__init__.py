"""Native-PyTorch ViTPose. Standalone leaf: imports nothing from hydra_suite.

Import order matters. cv2 must be imported before torch on this platform:
conda ships libomp.dylib (@rpath) and torch vendors its own
(/opt/llvm-openmp/lib/libomp.dylib). Loading torch first then cv2 aborts the
process with `OMP: Error #15`. cv2-first is stable. Do not "fix" this with
KMP_DUPLICATE_LIB_OK=TRUE: LLVM documents that as possibly producing silently
incorrect results, which defeats the point of a numerical-parity module.
"""

import cv2 as _cv2  # noqa: F401  (must precede torch; see module docstring)
import torch as _torch  # noqa: F401
