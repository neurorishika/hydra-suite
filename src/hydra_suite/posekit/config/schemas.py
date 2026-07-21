"""Runtime configuration schema for PoseKit."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class PoseKitConfig:
    """User-configurable preferences for the PoseKit labeling application.

    Project data (project object, image_paths) is passed at construction
    time and does not belong here.
    """

    mode: str = "frame"  # 'frame' or 'keypoint' progression
    show_predictions: bool = True
    show_pred_conf: bool = False
    sleap_env_path: str = ""
    autosave_delay_ms: int = 3000
    runtime_tier: str = "gpu"  # 'cpu' | 'gpu' | 'gpu_fast'

    def to_dict(self) -> dict[str, Any]:
        """Serialize all user preferences to a JSON-compatible dictionary for persistence."""
        return {
            "mode": self.mode,
            "show_predictions": self.show_predictions,
            "show_pred_conf": self.show_pred_conf,
            "sleap_env_path": self.sleap_env_path,
            "autosave_delay_ms": self.autosave_delay_ms,
            "runtime_tier": self.runtime_tier,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PoseKitConfig":
        """Reconstruct a PoseKitConfig from a dictionary produced by ``to_dict``.

        ``runtime_tier`` is the sole runtime knob (Runtime Gen-2). Old settings
        that only stored legacy ``compute_runtime`` / ``pred_runtime`` strings
        are migrated by the one-shot config-migration script; here a missing
        tier simply defaults to ``"gpu"``.
        """
        runtime_tier = str(data.get("runtime_tier", "")).strip() or "gpu"
        return cls(
            mode=data.get("mode", "frame"),
            show_predictions=data.get("show_predictions", True),
            show_pred_conf=data.get("show_pred_conf", False),
            sleap_env_path=data.get("sleap_env_path", ""),
            autosave_delay_ms=data.get("autosave_delay_ms", 3000),
            runtime_tier=runtime_tier,
        )
