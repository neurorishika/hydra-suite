"""Structured availability for SAM3 LoRA training.

Same discipline as core/inference/semantic/checkpoints.py: never import the
heavy packages in this process, never download. The GUI disables the action
with `reason` rather than failing at click time.

Training runs in a dedicated sidecar conda env (see
``docs/superpowers/specs/2026-09-01-sam3-training-sidecar-env-design.md``)
because ``sam3`` pins ``numpy<2`` and cannot coexist with the numpy 2.x
runtimes in ``hydra-mps``/``hydra-cuda``. The probe therefore asks "can the
sidecar env import what training needs", not "is sam3 importable here" -- it
runs a short script in the child via ``run_conda`` and parses the child's
JSON output. Never import ``sam3`` at module scope, or anywhere in this
process.
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from hydra_suite.training.sam3_lora.env import DEFAULT_SAM3_ENV
from hydra_suite.utils.conda_utils import run_conda

# `iopath` is a transitive import of `sam3.train.loss`, not a direct one of
# ours. It is probed anyway: without it the probe would report the role usable
# and training would then die on a bare ModuleNotFoundError partway in.
TRAINING_PACKAGES = (
    "sam3",
    "torch",
    "torchmetrics",
    "scipy",
    "einops",
    "decord",
    "iopath",
)

_ENV_RECIPE_HINT = (
    "See the hydra-sam3 env recipe in "
    "docs/superpowers/specs/2026-09-01-sam3-training-sidecar-env-design.md"
)
INSTALL_HINTS = {
    "sam3": _ENV_RECIPE_HINT,
}
DEFAULT_INSTALL_HINT = _ENV_RECIPE_HINT

DEFAULT_PROBE_TIMEOUT_S = 30.0

_PROBE_SCRIPT_PATH = Path(__file__).with_name("_probe_script.py")


@dataclass(frozen=True)
class Sam3TrainingAvailability:
    usable: bool
    reason: str = ""


def _checkpoint_present(cache_dir: Optional[Path] = None) -> bool:  # seam for tests
    from hydra_suite.core.inference.semantic.checkpoints import checkpoint_path

    return checkpoint_path("sam3", cache_dir).exists()


def _run_probe(
    env: str, timeout: float
) -> "subprocess.CompletedProcess[str]":  # seam for tests
    import os

    from hydra_suite.training.sam3_lora.env import sam3_env_environ

    # Deliberately NOT sam3_env_command/`-m` -- see _probe_script.py docstring
    # for why importing hydra_suite package chains here would misreport.
    command = ["conda", "run", "-n", env, "python", str(_PROBE_SCRIPT_PATH)]
    child_environ = {**os.environ, **sam3_env_environ()}
    return run_conda(
        command,
        env=child_environ,
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def probe_sam3_training_availability(
    cache_dir: Optional[Path] = None,
    env: Optional[str] = None,
    timeout: float = DEFAULT_PROBE_TIMEOUT_S,
) -> Sam3TrainingAvailability:
    """Probe whether the sidecar conda env can run SAM3 LoRA training.

    Runs a short script inside ``env`` (default: ``DEFAULT_SAM3_ENV``) that
    checks ``TRAINING_PACKAGES`` are importable there, and parses its JSON
    result. The reason always carries the child's real failure text when
    unusable, so the user knows exactly what to fix.
    """
    target_env = env or DEFAULT_SAM3_ENV

    try:
        result = _run_probe(target_env, timeout)
    except FileNotFoundError:
        return Sam3TrainingAvailability(
            False,
            "conda was not found on PATH. Install conda/miniconda, or ensure "
            "it is on PATH, before training in the sidecar env.",
        )
    except subprocess.TimeoutExpired:
        return Sam3TrainingAvailability(
            False,
            f"Probing the {target_env!r} conda env timed out after "
            f"{timeout:g}s. The env may be broken or hanging on import; "
            "check it manually with "
            f"`conda run -n {target_env} python -c 'import sam3'`.",
        )

    if result.returncode != 0:
        stderr_tail = (result.stderr or "").strip().splitlines()
        tail = "\n".join(stderr_tail[-20:]) if stderr_tail else "(no output)"
        reason = (
            f"The {target_env!r} conda env does not exist or failed to run "
            f"the availability probe. Child output:\n{tail}"
        )
        return Sam3TrainingAvailability(False, reason)

    stdout = (result.stdout or "").strip()
    try:
        payload: Any = json.loads(stdout.splitlines()[-1]) if stdout else {}
    except (json.JSONDecodeError, IndexError):
        return Sam3TrainingAvailability(
            False,
            f"The {target_env!r} conda env produced output that could not be "
            f"parsed as the expected JSON probe result: {stdout[:500]!r}",
        )

    missing = payload.get("missing")
    if missing:
        pkg = (
            missing[0].get("package", "?")
            if isinstance(missing[0], dict)
            else missing[0]
        )
        detail = missing[0].get("error", "") if isinstance(missing[0], dict) else ""
        hint = INSTALL_HINTS.get(pkg, DEFAULT_INSTALL_HINT)
        detail_suffix = f" ({detail})" if detail else ""
        return Sam3TrainingAvailability(
            False,
            f"Python package {pkg!r} is not importable in the {target_env!r} "
            f"env{detail_suffix}. {hint}",
        )

    if not payload.get("ok", False):
        error = payload.get("error", "unknown error")
        return Sam3TrainingAvailability(
            False,
            f"The {target_env!r} conda env reported it cannot run SAM3 "
            f"training: {error}",
        )

    if not payload.get("cuda_available", False):
        return Sam3TrainingAvailability(
            False,
            "The sidecar has no CUDA device. SAM3 training is admitted only "
            "on CUDA BF16 hardware; CPU and MPS training are disabled.",
        )
    capability = payload.get("cuda_compute_capability")
    try:
        major, minor = int(capability[0]), int(capability[1])
    except (IndexError, TypeError, ValueError):
        return Sam3TrainingAvailability(
            False, "The sidecar could not report its CUDA compute capability."
        )
    if major < 8:
        return Sam3TrainingAvailability(
            False,
            "SAM3 training requires CUDA BF16 and compute capability >= 8.0; "
            f"the sidecar reports {major}.{minor}. FP32 fallback is disabled.",
        )
    if not payload.get("cuda_bf16_supported", False):
        return Sam3TrainingAvailability(
            False,
            "The sidecar CUDA runtime reports that BF16 operations are not "
            "supported. SAM3 training has no safe FP32 fallback.",
        )

    if not _checkpoint_present(cache_dir):
        return Sam3TrainingAvailability(
            False,
            "The SAM3 base checkpoint has not been downloaded yet. Run a "
            "semantic escalation once to fetch it, or accept the licence at "
            "https://huggingface.co/facebook/sam3 and run `hf auth login`.",
        )

    return Sam3TrainingAvailability(True)
