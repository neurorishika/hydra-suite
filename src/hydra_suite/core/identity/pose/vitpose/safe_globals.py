"""Shared `torch.load(weights_only=True)` allowlist for third-party ViTPose
checkpoints.

mmpose checkpoints (including the collaborator's externally trained ones)
carry numpy scalars in their ``meta`` dict, which PyTorch's default
`weights_only=True` unpickler rejects. This module allowlists exactly those
numpy primitives -- never switch any of these loaders to `weights_only=False`;
that would unpickle arbitrary objects from a downloaded third-party binary,
which is precisely what the allowlist exists to avoid.

Reference/validated implementation: `tools/vitpose/external_ckpt/model.py`.
Two details here are load-bearing and were each established the hard way:

- numpy 2.x renamed `numpy.core` to `numpy._core`, so the plain-callable form
  of `add_safe_globals` does not match the name pickled in these checkpoints.
  The `(callable, "pickled.name")` tuple form is required, and BOTH spellings
  must be registered: real (older-numpy) mmpose checkpoints pickle
  `numpy.core.multiarray.scalar`, while a checkpoint written under this numpy
  (2.x, e.g. in tests) pickles `numpy._core.multiarray.scalar`.
- The `numpy.dtypes.*DType` entries are not dead code. Removing them broke
  loading of the real ~1GB collaborator checkpoints (confirmed empirically on
  the probe branch, then reverted) -- keep them.

This allowlist's safety depends on numpy's own object-dtype hardening:
`multiarray.scalar` together with `np.dtype` would be a remote-code-execution
gadget on sufficiently old numpy, where `scalar` given an object dtype ran
`pickle.loads` on its payload. numpy 2.x blocks that path, which is what makes
allowlisting these primitives (rather than falling back to `weights_only=False`)
safe here.
"""

from __future__ import annotations

import numpy as np
import numpy.core.multiarray as _np_multiarray
import torch

_SAFE_GLOBALS = [
    (_np_multiarray.scalar, "numpy.core.multiarray.scalar"),
    (_np_multiarray.scalar, "numpy._core.multiarray.scalar"),
    (np.dtype, "numpy.dtype"),
]
for _name in ("Float64DType", "Float32DType", "Int64DType", "Int32DType", "BoolDType"):
    _dtype = getattr(np.dtypes, _name, None)
    if _dtype is not None:
        _SAFE_GLOBALS.append(_dtype)


def ensure_numpy_safe_globals() -> None:
    """Allowlist the numpy primitives mmpose checkpoints pickle into `meta`.

    Idempotent (`torch.serialization.add_safe_globals` is idempotent); call at
    the start of every production loader that does a
    `torch.load(..., weights_only=True)` on a user-supplied or externally
    trained checkpoint.
    """
    torch.serialization.add_safe_globals(_SAFE_GLOBALS)
