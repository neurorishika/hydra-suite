"""Per-detection identity classifiers: AprilTag, CNN, and head-tail direction."""

from hydra_suite.core.individual.classification.apriltag import (
    AprilTagConfig,
    AprilTagDetector,
)
from hydra_suite.core.individual.classification.backend import (
    ClassifierBackend,
    ClassifierMetadata,
)
from hydra_suite.core.individual.classification.cnn import (
    ClassPrediction,
    CNNIdentityBackend,
    CNNIdentityCache,
    CNNIdentityConfig,
    TrackCNNHistory,
    apply_cnn_identity_cost,
)
from hydra_suite.core.individual.classification.errors import (
    ClassifierConfigError,
    ClassifierError,
    ClassifierFormatError,
    ClassifierRuntimeError,
    HeadTailFormatError,
)
from hydra_suite.core.individual.classification.headtail import HeadTailAnalyzer

__all__ = [
    "AprilTagConfig",
    "AprilTagDetector",
    "ClassifierBackend",
    "ClassifierConfigError",
    "ClassifierError",
    "ClassifierFormatError",
    "ClassifierMetadata",
    "ClassifierRuntimeError",
    "CNNIdentityBackend",
    "CNNIdentityCache",
    "CNNIdentityConfig",
    "ClassPrediction",
    "HeadTailAnalyzer",
    "HeadTailFormatError",
    "TrackCNNHistory",
    "apply_cnn_identity_cost",
]
