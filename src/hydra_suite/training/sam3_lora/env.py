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
    """Build the ``conda run`` command line to invoke a module in ``env``.

    Two flags, for two different buffers:

    ``--no-capture-output`` stops ``conda run`` itself from swallowing the
    child's streams. Without it conda buffers stdout and stderr in full and
    releases them only when the child EXITS -- so a training run emits
    nothing for its entire duration, then dumps everything at once. A 6-hour
    run was killed after printing not one progress line, and a run that dies
    without flushing loses its diagnostics entirely. ``-u`` alone does not
    help: it unbuffers PYTHON, while the capture happens a level above it.

    ``-u`` (unbuffered stdout/stderr) still matters: the launcher parses the
    child's stdout line by line for progress, and ordinary ``sam3``/``torch``
    prints are not flushed like this package's own sentinel records are.
    """
    return [
        "conda",
        "run",
        "-n",
        env,
        "--no-capture-output",
        "python",
        "-u",
        "-m",
        *module_args,
    ]


def sam3_env_environ() -> Dict[str, str]:
    """Environment variable overrides required by the sidecar child process.

    ``KMP_DUPLICATE_LIB_OK=TRUE`` is required: without it a bare ``import
    torch`` aborts with ``OMP Error #15`` (double-linked libomp), observed
    while building the mac env; ``tools/equivalence/run_matrix.sh`` sets the
    same variable for the same reason.

    ``PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`` removes caching-
    allocator fragmentation, which MEASURED AT 45% OF PEAK VRAM on a full
    10-epoch run. Two matched runs, same commit, same corpus (1871 instances,
    486 train tiles, 1766px), same rank/precision/312-adapter surface, batch 1:

        default allocator (mehek, RTX 6000 Ada)
            reserved 7.60 -> 8.89 -> 10.22 -> 11.59 -> 12.99 GiB,
            last new max at step 320 of 2430.
            Growth 2->2430: +35% reserved but only +2% allocated.

        expandable_segments (courtship, RTX 4090)
            reserved 6.70 -> ... -> 7.125 GiB,
            last new max at step 302, then FLAT for 2128 steps across
            eight epoch boundaries.
            Growth: +6.4% reserved, +6.2% allocated -- the two move
            TOGETHER, which is real transient demand, not fragmentation.

    Delta 12.99 - 7.125 = 5.87 GiB (cross-box; same code/data/config).

    This is set here, rather than left to the caller's shell, because the
    sidecar is launched through ``conda run`` and must get it regardless of
    who invoked the launcher.

    It also makes measurement possible at all: under the default allocator a
    30-step probe under-read the full-run peak by 41%, so no short probe could
    bound it. Under this flag a 30-60 step probe under-reads by only 2.8%,
    which is what makes measured auto batch sizing tractable. A probe taken
    WITHOUT this flag does not transfer to a run WITH it, or vice versa --
    treat the allocator config as part of any VRAM measurement's identity.

    No downside observed across 2466 logged steps: zero allocator warnings,
    ~3.09 s/step, normal loss descent, zero skipped steps, exit 0. Not
    measured: a matched same-box throughput baseline, so a few-percent
    step-time cost is not excluded.
    """
    return {
        "KMP_DUPLICATE_LIB_OK": "TRUE",
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
    }
