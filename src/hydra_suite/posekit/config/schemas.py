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

        Legacy settings that store ``compute_runtime`` / ``pred_runtime`` instead
        of ``runtime_tier`` are migrated automatically.
        """
        runtime_tier = data.get("runtime_tier", "")
        if not runtime_tier:
            # Migrate from legacy canonical-runtime strings stored under older keys.
            try:
                from hydra_suite.posekit.gui.runtimes import canonical_runtime_to_tier

                legacy = str(
                    data.get("compute_runtime", data.get("pred_runtime", ""))
                ).strip()
                runtime_tier = canonical_runtime_to_tier(legacy) if legacy else "gpu"
            except Exception:
                runtime_tier = "gpu"
        return cls(
            mode=data.get("mode", "frame"),
            show_predictions=data.get("show_predictions", True),
            show_pred_conf=data.get("show_pred_conf", False),
            sleap_env_path=data.get("sleap_env_path", ""),
            autosave_delay_ms=data.get("autosave_delay_ms", 3000),
            runtime_tier=runtime_tier,
        )
