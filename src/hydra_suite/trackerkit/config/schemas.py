"""Runtime configuration schema for the MAT tracker."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from hydra_suite.core.inference.config import migrate_runtime_to_tier
from hydra_suite.trackerkit.engine_params import n_arenas_from_shapes


@dataclass
class TrackerConfig:
    """Session-meaningful state for the MAT tracking application.

    Only persistent, user-configurable fields live here.
    Ephemeral runtime state (ROI masks, playback position, session
    counters, etc.) stays on MainWindow.
    """

    # --- Input ---
    current_video_path: str = ""
    batch_videos: list = field(default_factory=list)

    # --- ROI ---
    roi_shapes: list = field(default_factory=list)
    roi_current_mode: str = "circle"  # 'circle' or 'polygon'
    roi_current_zone_type: str = "include"  # 'include' or 'exclude'

    # --- Arenas ---
    # One shared animal count per arena; MAX_TARGETS is derived
    # (n_arenas * animals_per_arena), never entered directly.
    animals_per_arena: int = 1

    # --- Runtime ---
    runtime_tier: str = "gpu"

    # --- Debug ---
    debug_mode: bool = False

    # --- Active-learning dataset export ---
    dataset_export_levels: list = field(
        default_factory=lambda: ["polygon", "obb", "aabb"]
    )
    dataset_dedup_method: str = "phash"
    dataset_dedup_threshold: int = 8
    dataset_class_names: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-compatible dict.

        ``animals_per_arena`` is only emitted once ``roi_shapes`` actually
        encodes more than one arena (``n_arenas_from_shapes(self.roi_shapes)
        > 1``) -- the same gate ``ConfigOrchestrator.build_config_dict``
        applies to the GUI's own (separate) config dict. Keeping the two
        consistent means this dataclass can never become a "loaded gun": if
        it were ever serialized directly into an engine-params config
        (bypassing the GUI glue), a single-arena project still wouldn't
        carry an `animals_per_arena` override that could defeat
        ``build_engine_params``'s fallback-to-`max_targets` safety net.
        """
        d = {
            "current_video_path": self.current_video_path,
            "batch_videos": list(self.batch_videos),
            "roi_shapes": list(self.roi_shapes),
            "roi_current_mode": self.roi_current_mode,
            "roi_current_zone_type": self.roi_current_zone_type,
            "runtime_tier": self.runtime_tier,
            "debug_mode": self.debug_mode,
            "dataset_export_levels": list(self.dataset_export_levels),
            "dataset_dedup_method": self.dataset_dedup_method,
            "dataset_dedup_threshold": self.dataset_dedup_threshold,
            "dataset_class_names": self.dataset_class_names,
        }
        if n_arenas_from_shapes(self.roi_shapes) > 1:
            d["animals_per_arena"] = int(self.animals_per_arena)
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TrackerConfig:
        """Deserialize from a dict produced by ``to_dict``."""
        raw_tier = data.get("runtime_tier")
        if raw_tier is None:
            legacy = set()
            for key in ("compute_runtime", "headtail_runtime", "cnn_runtime"):
                v = data.get(key)
                if v:
                    legacy.add(str(v))
            raw_tier = migrate_runtime_to_tier(legacy) if legacy else "gpu"
        return cls(
            current_video_path=data.get("current_video_path", ""),
            batch_videos=list(data.get("batch_videos", [])),
            roi_shapes=list(data.get("roi_shapes", [])),
            roi_current_mode=data.get("roi_current_mode", "circle"),
            roi_current_zone_type=data.get("roi_current_zone_type", "include"),
            animals_per_arena=int(data.get("animals_per_arena", 1)),
            runtime_tier=str(raw_tier),
            debug_mode=bool(data.get("debug_mode", False)),
            dataset_export_levels=list(
                data.get("dataset_export_levels", ["polygon", "obb", "aabb"])
            ),
            dataset_dedup_method=str(data.get("dataset_dedup_method", "phash")),
            dataset_dedup_threshold=int(data.get("dataset_dedup_threshold", 8)),
            dataset_class_names=str(data.get("dataset_class_names", "")),
        )
