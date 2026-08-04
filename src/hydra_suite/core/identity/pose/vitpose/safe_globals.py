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
safe -- but nothing in this repo's dependency floor (`pyproject.toml` pins
`numpy>=1.24`) guarantees numpy 2.x is actually installed. `ensure_numpy_safe_globals`
therefore checks the installed numpy version itself and refuses to install the
allowlist below 2.0, failing closed with an actionable error rather than silently
installing a known RCE gadget on an old numpy. This is a targeted runtime guard,
not a `pyproject.toml` floor bump: raising the floor project-wide would bind
ClassKit, DetectKit, and every other kit to a constraint only this loader needs.
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
# numpy.dtypes (the module holding the concrete *DType classes, e.g.
# Float64DType) was added in numpy 1.25 -- `np.dtypes` does not exist at all
# on older numpy, so this is guarded with `hasattr` rather than assumed, or
# module import itself would raise AttributeError on numpy<1.25 before
# `ensure_numpy_safe_globals`'s version guard ever gets a chance to run and
# fail closed with an actionable message. In practice this list is only ever
# consulted (via `add_safe_globals`) once that guard has already passed numpy
# >=2.0, at which point `numpy.dtypes` is always present -- the `hasattr`
# check here exists purely so an old-numpy install fails at the guard, not at
# import time.
if hasattr(np, "dtypes"):
    for _name in (
        "Float64DType",
        "Float32DType",
        "Int64DType",
        "Int32DType",
        "BoolDType",
    ):
        _dtype = getattr(np.dtypes, _name, None)
        if _dtype is not None:
            _SAFE_GLOBALS.append(_dtype)


def _numpy_version_tuple() -> tuple[int, ...]:
    parts: list[int] = []
    for chunk in np.__version__.split(".")[:3]:
        digits = ""
        for ch in chunk:
            if ch.isdigit():
                digits += ch
            else:
                break
        parts.append(int(digits) if digits else 0)
    return tuple(parts)


def ensure_numpy_safe_globals() -> None:
    """Allowlist the numpy primitives mmpose checkpoints pickle into `meta`.

    Idempotent (`torch.serialization.add_safe_globals` is idempotent); call at
    the start of every production loader that does a
    `torch.load(..., weights_only=True)` on a user-supplied or externally
    trained checkpoint.

    Fails closed on numpy < 2.0: allowlisting `numpy.core.multiarray.scalar`
    together with `np.dtype` is only safe because numpy 2.x hardened
    object-dtype scalars against the `pickle.loads` gadget those two globals
    otherwise form. Raises rather than degrading silently -- loading these
    checkpoints becoming unavailable on old numpy is the correct posture.
    """
    version = _numpy_version_tuple()
    if version < (2, 0):
        raise RuntimeError(
            f"ViTPose checkpoint loading requires numpy>=2.0 (installed: "
            f"{np.__version__}). Loading these checkpoints needs allowlisting "
            "numpy.core.multiarray.scalar for torch.load(weights_only=True), "
            "which is only safe on numpy>=2.0 (numpy 2.x hardened object-dtype "
            "scalars against the pickle.loads gadget that allowlist would "
            "otherwise open on older numpy). Upgrade numpy to 2.0 or later to "
            "load ViTPose checkpoints."
        )
    torch.serialization.add_safe_globals(_SAFE_GLOBALS)
