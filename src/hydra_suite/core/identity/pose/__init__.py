"""Pose inference subsystem: backends, types, utilities, and quality assessment.

Correction 25 (Task 18a): Backend symbols and factory functions are re-exported
with try/except guards so this package remains importable after those legacy
modules are removed in Task 18.  The stable type symbols (PoseInferenceBackend,
PoseResult, PoseRuntimeConfig, RuntimeMetrics) keep direct imports.

New callers should use hydra_suite.core.inference.api.build_runtime_config and
hydra_suite.core.inference.api.create_pose_backend_from_config.
"""

from hydra_suite.core.identity.pose.types import (
    PoseInferenceBackend,
    PoseResult,
    PoseRuntimeConfig,
    RuntimeMetrics,
)

try:
    from hydra_suite.core.identity.pose.api import (  # noqa: F401
        build_runtime_config,
        create_pose_backend_from_config,
    )
except ImportError:
    build_runtime_config = None  # type: ignore[assignment]
    create_pose_backend_from_config = None  # type: ignore[assignment]

try:
    from hydra_suite.core.identity.pose.backends.sleap import (  # noqa: F401
        SleapServiceBackend,
        auto_export_sleap_model,
    )
except ImportError:
    SleapServiceBackend = None  # type: ignore[assignment,misc]
    auto_export_sleap_model = None  # type: ignore[assignment]

try:
    from hydra_suite.core.identity.pose.backends.yolo import (  # noqa: F401
        YoloNativeBackend,
        auto_export_yolo_model,
    )
except ImportError:
    YoloNativeBackend = None  # type: ignore[assignment,misc]
    auto_export_yolo_model = None  # type: ignore[assignment]

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
