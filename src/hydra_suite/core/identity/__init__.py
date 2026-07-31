"""Identity and individual-level analysis.

Sub-packages:
    pose/           — Pose inference backends, types, utilities, quality
    classification/ — Per-detection classifiers (AprilTag, CNN, head-tail)
    properties/     — Properties caching and CSV export aggregation
    dataset/        — Crop generation and video export
"""

# Classification
from .classification.apriltag import AprilTagConfig, AprilTagDetector
from .classification.cnn import (
    ClassPrediction,
    CNNIdentityBackend,
    CNNIdentityCache,
    CNNIdentityConfig,
    TrackCNNHistory,
    apply_cnn_identity_cost,
)
from .classification.headtail import HeadTailAnalyzer

# Dataset generation
from .dataset.generator import IndividualDatasetGenerator
from .dataset.oriented_video import OrientedTrackVideoExporter

# Geometry
from .geometry import (
    ellipse_axes_from_area,
    ellipse_to_obb_corners,
    resolve_directed_angle,
)
from .pose.api import create_pose_backend_from_config

# Pose
from .pose.types import (
    PoseInferenceBackend,
    PoseResult,
    PoseRuntimeConfig,
    RuntimeMetrics,
)

# Properties
from .properties.cache import IndividualPropertiesCache

__all__ = [
    # Dataset
    "IndividualDatasetGenerator",
    "OrientedTrackVideoExporter",
    # Classification
    "AprilTagConfig",
    "AprilTagDetector",
    "CNNIdentityConfig",
    "CNNIdentityBackend",
    "CNNIdentityCache",
    "ClassPrediction",
    "TrackCNNHistory",
    "apply_cnn_identity_cost",
    "HeadTailAnalyzer",
    # Geometry
    "ellipse_to_obb_corners",
    "ellipse_axes_from_area",
    "resolve_directed_angle",
    # Pose
    "PoseResult",
    "PoseRuntimeConfig",
    "PoseInferenceBackend",
    "RuntimeMetrics",
    "create_pose_backend_from_config",
    # Properties
    "IndividualPropertiesCache",
]
