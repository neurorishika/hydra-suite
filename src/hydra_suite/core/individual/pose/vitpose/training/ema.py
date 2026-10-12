"""Exponential moving average of model weights (as in mmpose / RTMPose).

The averaged copy is smoother than the live weights late in training and
usually validates better; it is what gets validated and saved.
"""

from __future__ import annotations

import copy
import math

import torch


class ModelEMA:
    def __init__(self, model: torch.nn.Module, decay: float = 0.9998) -> None:
        self.module = copy.deepcopy(model).eval()
        for p in self.module.parameters():
            p.requires_grad_(False)
        self.decay = float(decay)
        self.updates = 0

    @torch.no_grad()
    def update(self, model: torch.nn.Module) -> None:
        self.updates += 1
        # Warm up the decay so early (random-ish) weights fade quickly.
        d = self.decay * (1.0 - math.exp(-self.updates / 2000.0))
        live = model.state_dict()
        for name, value in self.module.state_dict().items():
            if value.dtype.is_floating_point:
                value.mul_(d).add_(live[name].detach(), alpha=1.0 - d)
            else:
                value.copy_(live[name])

    def state_dict(self) -> dict:
        return {"module": self.module.state_dict(), "updates": self.updates}

    def load_state_dict(self, state: dict) -> None:
        self.module.load_state_dict(state["module"])
        self.updates = int(state.get("updates", 0))
