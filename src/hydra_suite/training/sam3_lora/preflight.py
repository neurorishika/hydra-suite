"""Preflight refusal checks for SAM3 LoRA training.

Runs before any weights load, any ``sam3`` import, or any GPU allocation, so
a doomed run is caught in milliseconds rather than after an hour of GPU time.
Every environment probe is a module-level seam (``_cuda_free_gb``,
``_free_disk_gb``, ``_instance_count``) so tests can monkeypatch them without
a GPU or a real dataset.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

# Measured: ~29 GB at batch 1 on the spike; batch 2 OOMed on a 47 GB card.
# REFUSE must sit ABOVE the measured requirement, or the 24-29 GB band passes
# preflight and then OOMs -- the exact failure preflight exists to prevent.
REFUSE_BELOW_GB = 32.0
WARN_BELOW_GB = 40.0
MIN_TRAIN_INSTANCES = 20  # matches calibration.py's existing floor
REQUIRED_DISK_GB = 8.0  # merged artifact is ~3.2 GB; merge needs base resident


def _cuda_free_gb() -> float | None:  # seam for tests
    try:
        import torch
    except ImportError:
        return None
    if not torch.cuda.is_available():
        return None
    free_bytes, _total_bytes = torch.cuda.mem_get_info()
    return free_bytes / (1024**3)


def _free_disk_gb(path: str) -> float:  # seam for tests
    usage = shutil.disk_usage(Path(path).expanduser().resolve().anchor or "/")
    return usage.free / (1024**3)


def _instance_count(source_datasets: Any) -> int:  # seam for tests
    # Real implementation would count annotated instances across the spec's
    # source datasets; tests always monkeypatch this seam, so a conservative
    # zero here never masks a missing patch.
    return 0


def _bf16_capability_warning() -> str | None:
    try:
        import torch
    except ImportError:
        return None
    if not torch.cuda.is_available():
        return None
    major, _minor = torch.cuda.get_device_capability()
    if major < 8:
        return (
            "GPU compute capability is below 8.0; bf16 autocast is not "
            "supported and training will fall back to fp32 (slower)."
        )
    return None


def preflight(spec: Any) -> list[str]:
    """Return refusal reasons for `spec`; empty list means OK to train.

    Order matters only for readability -- callers join all reasons, they do
    not short-circuit on the first one.
    """
    reasons: list[str] = []

    free_gb = _cuda_free_gb()
    if free_gb is None:
        reasons.append("No CUDA device is available; SAM3 LoRA training requires CUDA.")
    elif free_gb < REFUSE_BELOW_GB:
        reasons.append(
            f"Only {free_gb:.1f} GB free VRAM; SAM3 LoRA training requires at "
            f"least {REFUSE_BELOW_GB:.0f} GB free."
        )

    sam3_params = spec.sam3_params
    prompt = (sam3_params.prompt if sam3_params is not None else "") or ""
    if not prompt.strip():
        reasons.append("Prompt is empty; SAM3 requires a text prompt to train against.")

    n_instances = _instance_count(spec.source_datasets)
    if n_instances < MIN_TRAIN_INSTANCES:
        reasons.append(
            f"Only {n_instances} labeled instances found; at least "
            f"{MIN_TRAIN_INSTANCES} are required to train."
        )

    disk_gb = _free_disk_gb(spec.derived_dataset_dir)
    if disk_gb < REQUIRED_DISK_GB:
        reasons.append(
            f"Only {disk_gb:.1f} GB free disk space; at least "
            f"{REQUIRED_DISK_GB:.0f} GB is required for the merged artifact."
        )

    if sam3_params is None or not sam3_params.label_quality_acknowledged:
        reasons.append(
            "Label quality has not been acknowledged; affirm the training "
            "labels are good before SAM3 learns from them."
        )

    if spec.resume_from:
        reasons.append(
            "resume_from is set, but SAM3 LoRA training does not checkpoint "
            "optimiser state; resuming is not supported."
        )

    return reasons


def preflight_warnings(spec: Any) -> list[str]:
    """Return non-blocking warnings for `spec` (does not affect refusal)."""
    warnings: list[str] = []

    free_gb = _cuda_free_gb()
    if free_gb is not None and free_gb < WARN_BELOW_GB:
        warnings.append(
            f"Only {free_gb:.1f} GB free VRAM; runs below "
            f"{WARN_BELOW_GB:.0f} GB may be tight."
        )

    sam3_params = spec.sam3_params
    if sam3_params is not None and sam3_params.mixed_precision == "bf16":
        bf16_warning = _bf16_capability_warning()
        if bf16_warning:
            warnings.append(bf16_warning)

    return warnings
