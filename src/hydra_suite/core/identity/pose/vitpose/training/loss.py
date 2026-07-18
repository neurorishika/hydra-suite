from __future__ import annotations

import torch
import torch.nn as nn


class JointsMSELoss(nn.Module):
    """Per-joint heatmap MSE with optional per-joint visibility weighting.
    Mirrors mmpose JointsMSELoss(use_target_weight=True)."""

    def __init__(self, use_target_weight: bool = True) -> None:
        super().__init__()
        self.criterion = nn.MSELoss(reduction="mean")
        self.use_target_weight = use_target_weight

    def forward(
        self, output: torch.Tensor, target: torch.Tensor, target_weight: torch.Tensor
    ) -> torch.Tensor:
        b, k = output.shape[0], output.shape[1]
        pred = output.reshape(b, k, -1)
        gt = target.reshape(b, k, -1)
        loss = output.new_zeros(())
        for j in range(k):
            pj, gj = pred[:, j], gt[:, j]
            if self.use_target_weight:
                w = target_weight[:, j]  # (B,1)
                loss = loss + 0.5 * self.criterion(pj * w, gj * w)
            else:
                loss = loss + 0.5 * self.criterion(pj, gj)
        return loss / k
