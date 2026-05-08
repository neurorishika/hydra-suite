"""Pose inference subsystem: backends, types, utilities, and quality assessment."""

from hydra_suite.core.identity.pose.api import (
    build_runtime_config,
    create_pose_backend_from_config,
)
from hydra_suite.core.identity.pose.backends.sleap import (
    SleapServiceBackend,
    auto_export_sleap_model,
)
from hydra_suite.core.identity.pose.backends.yolo import (
    YoloNativeBackend,
    auto_export_yolo_model,
)
from hydra_suite.core.identity.pose.types import (
    PoseInferenceBackend,
    PoseResult,
    PoseRuntimeConfig,
    RuntimeMetrics,
)

__all__ = [
    "PoseResult",
    "PoseRuntimeConfig",
    "PoseInferenceBackend",
    "RuntimeMetrics",
    "YoloNativeBackend",
    "SleapServiceBackend",
    "auto_export_yolo_model",
    "auto_export_sleap_model",
    "build_runtime_config",
    "create_pose_backend_from_config",
]
