"""Core data types and interfaces for pose runtime backends."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Protocol, Sequence, Tuple

import numpy as np


@dataclass
class PoseResult:
    """Canonical pose output for one crop."""

    keypoints: Optional[np.ndarray]
    mean_conf: float
    valid_fraction: float
    num_valid: int
    num_keypoints: int


@dataclass
class PoseRuntimeConfig:
    """Configuration for pose runtime backend selection."""

    backend_family: str  # yolo | sleap
    runtime_flavor: str = "auto"  # native | onnx | tensorrt | auto
    device: str = "auto"  # auto | cpu | cuda | mps
    batch_size: int = 4
    model_path: str = ""
    exported_model_path: str = ""
    out_root: str = "."
    min_valid_conf: float = 0.2
    yolo_conf: float = 1e-4
    yolo_iou: float = 0.7
    yolo_max_det: int = 1
    yolo_batch: int = 4
    sleap_env: str = "sleap"
    sleap_device: str = "auto"
    sleap_batch: int = 4
    sleap_max_instances: int = 1
    sleap_export_input_hw: Optional[Tuple[int, int]] = None
    # --- vitpose-specific (flat, mirrors yolo_/sleap_) ---
    vitpose_batch: int = 4
    vitpose_variant: str = "auto"  # "auto" = infer from checkpoint
    vitpose_num_keypoints: int = 0  # 0 = infer from checkpoint
    keypoint_names: List[str] = field(default_factory=list)
    skeleton_edges: List[Tuple[int, int]] = field(default_factory=list)


@dataclass
class RuntimeMetrics:
    """Timing metrics for pose runtime lifecycle."""

    startup_ms: float = 0.0
    warmup_ms: float = 0.0
    closed_ms: float = 0.0


class PoseInferenceBackend(Protocol):
    """Protocol for all runtime backends."""

    output_keypoint_names: List[str]

    @property
    def preferred_input_size(self) -> int:
        """Preferred max input dimension for pre-resize (0 = no preference)."""
        ...

    def warmup(self) -> None:
        """Warm runtime (optional no-op for unsupported backends)."""

    def predict_batch(self, crops: Sequence[np.ndarray]) -> List[PoseResult]:
        """Run inference for a list of crops."""

    def close(self) -> None:
        """Release backend resources."""
