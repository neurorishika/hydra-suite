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
    frame_mode: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Serialize all user preferences to a JSON-compatible dictionary for persistence."""
        return {
            "mode": self.mode,
            "show_predictions": self.show_predictions,
            "show_pred_conf": self.show_pred_conf,
            "sleap_env_path": self.sleap_env_path,
            "autosave_delay_ms": self.autosave_delay_ms,
            "frame_mode": self.frame_mode,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PoseKitConfig":
        """Reconstruct a PoseKitConfig from a dictionary produced by ``to_dict``."""
        return cls(
            mode=data.get("mode", "frame"),
            show_predictions=data.get("show_predictions", True),
            show_pred_conf=data.get("show_pred_conf", False),
            sleap_env_path=data.get("sleap_env_path", ""),
            autosave_delay_ms=data.get("autosave_delay_ms", 3000),
            frame_mode=data.get("frame_mode", False),
        )
