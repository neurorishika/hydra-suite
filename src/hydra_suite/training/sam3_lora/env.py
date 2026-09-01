"""Sidecar conda environment resolution for SAM3 LoRA training.

Meta's ``sam3`` pins ``numpy<2``, which cannot coexist with the numpy 2.x
runtimes in ``hydra-mps``/``hydra-cuda``. Training therefore runs in a
dedicated conda environment, launched as a subprocess (see
``docs/superpowers/specs/2026-09-01-sam3-training-sidecar-env-design.md``).

Pure string/dict construction only: no subprocess, no ``sam3`` import, no Qt.
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional

DEFAULT_SAM3_ENV = "hydra-sam3"

_SAM3_ENV_VAR = "HYDRA_SAM3_ENV"


def resolve_sam3_env(configured: Optional[str] = None) -> str:
    """Resolve which conda env to run SAM3 training in.

    Precedence: an explicit non-empty ``configured`` value, then the
    ``HYDRA_SAM3_ENV`` environment variable, then ``DEFAULT_SAM3_ENV``.
    """
    if configured:
        return configured
    env_value = os.environ.get(_SAM3_ENV_VAR)
    if env_value:
        return env_value
    return DEFAULT_SAM3_ENV


def sam3_env_command(env: str, module_args: List[str]) -> List[str]:
    """Build the ``conda run`` command line to invoke a module in ``env``."""
    return ["conda", "run", "-n", env, "python", "-m", *module_args]


def sam3_env_environ() -> Dict[str, str]:
    """Environment variable overrides required by the sidecar child process.

    ``KMP_DUPLICATE_LIB_OK=TRUE`` is required: without it a bare ``import
    torch`` aborts with ``OMP Error #15`` (double-linked libomp), observed
    while building the mac env; ``tools/equivalence/run_matrix.sh`` sets the
    same variable for the same reason.
    """
    return {"KMP_DUPLICATE_LIB_OK": "TRUE"}
