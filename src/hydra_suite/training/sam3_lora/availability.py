"""Structured availability for SAM3 LoRA training.

Same discipline as core/inference/semantic/checkpoints.py: never import the
heavy packages, never download. The GUI disables the action with `reason`
rather than failing at click time.
"""

from __future__ import annotations

import importlib.util
from dataclasses import dataclass
from pathlib import Path
from typing import Any

TRAINING_PACKAGES = ("sam3", "torch", "torchmetrics", "scipy", "einops", "decord")

INSTALL_HINTS = {
    "sam3": "pip install git+https://github.com/facebookresearch/sam3.git",
}
DEFAULT_INSTALL_HINT = "pip install 'hydra-suite[sam3-train]'"


@dataclass(frozen=True)
class Sam3TrainingAvailability:
    usable: bool
    reason: str = ""


def _find_spec(name: str) -> Any:  # seam for tests
    return importlib.util.find_spec(name)


def _checkpoint_present(cache_dir: Path | None = None) -> bool:  # seam for tests
    from hydra_suite.core.inference.semantic.checkpoints import checkpoint_path

    return checkpoint_path("sam3", cache_dir).exists()


def probe_sam3_training_availability(
    cache_dir: Path | None = None,
) -> Sam3TrainingAvailability:
    for pkg in TRAINING_PACKAGES:
        if _find_spec(pkg) is None:
            hint = INSTALL_HINTS.get(pkg, DEFAULT_INSTALL_HINT)
            return Sam3TrainingAvailability(
                False, f"Python package {pkg!r} is missing. Install it with: {hint}"
            )
    if not _checkpoint_present(cache_dir):
        return Sam3TrainingAvailability(
            False,
            "The SAM3 base checkpoint has not been downloaded yet. Run a "
            "semantic escalation once to fetch it, or accept the licence at "
            "https://huggingface.co/facebook/sam3 and run `hf auth login`.",
        )
    return Sam3TrainingAvailability(True)
