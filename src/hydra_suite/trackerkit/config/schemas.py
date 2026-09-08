"""Runtime configuration schema for the MAT tracker."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from hydra_suite.core.inference.config import (
    DEFAULT_CALIBRATION_BUDGET_SECONDS,
    migrate_runtime_to_tier,
)
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

    # --- Batch fan-out (session state only; NEVER emitted into the per-video
    # engine config, so cache keys and child invocations are unchanged) ---
    batch_parallel: bool = False
    batch_parallel_jobs: int = 0  # 0 = one per selected GPU (or 1 without GPUs)
    batch_parallel_gpus: str = "auto"

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

    # --- Inference throughput autotuner ---
    # This is intentionally distinct from the semantic tracking autotuner.
    # Existing projects retain their configured execution settings until they
    # explicitly opt in.
    apply_tuned_inference: bool = False
    inference_autotune_manual_fields: list[str] = field(default_factory=list)
    inference_autotune_budget_seconds: float = DEFAULT_CALIBRATION_BUDGET_SECONDS

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
            "batch_parallel": bool(self.batch_parallel),
            "batch_parallel_jobs": int(self.batch_parallel_jobs),
            "batch_parallel_gpus": str(self.batch_parallel_gpus),
            "roi_shapes": list(self.roi_shapes),
            "roi_current_mode": self.roi_current_mode,
            "roi_current_zone_type": self.roi_current_zone_type,
            "runtime_tier": self.runtime_tier,
            "apply_tuned_inference": bool(self.apply_tuned_inference),
            "inference_autotune_manual_fields": list(
                self.inference_autotune_manual_fields
            ),
            "inference_autotune_budget_seconds": float(
                self.inference_autotune_budget_seconds
            ),
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
        raw_manual_fields = data.get("inference_autotune_manual_fields", []) or []
        if isinstance(raw_manual_fields, str):
            raw_manual_fields = [
                value.strip() for value in raw_manual_fields.split(",") if value.strip()
            ]
        elif not isinstance(raw_manual_fields, (list, tuple, set)):
            raw_manual_fields = []
        return cls(
            current_video_path=data.get("current_video_path", ""),
            batch_videos=list(data.get("batch_videos", [])),
            batch_parallel=bool(data.get("batch_parallel", False)),
            batch_parallel_jobs=int(data.get("batch_parallel_jobs", 0)),
            batch_parallel_gpus=str(data.get("batch_parallel_gpus", "auto")),
            roi_shapes=list(data.get("roi_shapes", [])),
            roi_current_mode=data.get("roi_current_mode", "circle"),
            roi_current_zone_type=data.get("roi_current_zone_type", "include"),
            animals_per_arena=int(data.get("animals_per_arena", 1)),
            runtime_tier=str(raw_tier),
            apply_tuned_inference=bool(
                data.get(
                    "apply_tuned_inference",
                    str(data.get("inference_autotune_mode", "off")).strip().lower()
                    in {"automatic", "record"},
                )
            ),
            inference_autotune_manual_fields=[
                str(value).strip() for value in raw_manual_fields if str(value).strip()
            ],
            inference_autotune_budget_seconds=float(
                data.get(
                    "inference_autotune_budget_seconds",
                    DEFAULT_CALIBRATION_BUDGET_SECONDS,
                )
            ),
            debug_mode=bool(data.get("debug_mode", False)),
            dataset_export_levels=list(
                data.get("dataset_export_levels", ["polygon", "obb", "aabb"])
            ),
            dataset_dedup_method=str(data.get("dataset_dedup_method", "phash")),
            dataset_dedup_threshold=int(data.get("dataset_dedup_threshold", 8)),
            dataset_class_names=str(data.get("dataset_class_names", "")),
        )
