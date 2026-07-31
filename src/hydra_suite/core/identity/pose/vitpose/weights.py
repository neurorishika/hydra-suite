"""Checkpoint loading with strict-key assertions."""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn


class CheckpointKeyError(RuntimeError):
    """Raised when checkpoint keys do not match the model."""


def load_checkpoint(model: nn.Module, path: Path, strict: bool = True) -> None:
    # weights_only=True is mandatory: these checkpoints come from a third-party
    # re-host, and the default (False) unpickles arbitrary objects.
    blob = torch.load(path, map_location="cpu", weights_only=True)
    state = blob["state_dict"] if "state_dict" in blob else blob
    missing, unexpected = model.load_state_dict(state, strict=False)
    if strict and (missing or unexpected):
        raise CheckpointKeyError(
            f"strict load failed for {path.name}\n"
            f"  missing ({len(missing)}): {sorted(missing)[:10]}\n"
            f"  unexpected ({len(unexpected)}): {sorted(unexpected)[:10]}"
        )
