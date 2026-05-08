"""Per-detection identity classifiers: AprilTag, CNN, and head-tail direction.

Correction 25 (Task 18a): CNN-related and HeadTail symbols are re-exported with
try/except guards so this package remains importable after those legacy modules
are removed in Task 18.  The stable symbols (AprilTag*, ClassifierBackend,
classifier errors) keep direct imports.
"""

from hydra_suite.core.identity.classification.apriltag import (
    AprilTagConfig,
    AprilTagDetector,
)
from hydra_suite.core.identity.classification.backend import (
    ClassifierBackend,
    ClassifierMetadata,
)
from hydra_suite.core.identity.classification.errors import (
    ClassifierConfigError,
    ClassifierError,
    ClassifierFormatError,
    ClassifierRuntimeError,
    HeadTailFormatError,
)

try:
    from hydra_suite.core.identity.classification.cnn import (  # noqa: F401
        ClassPrediction,
        CNNIdentityBackend,
        CNNIdentityCache,
        CNNIdentityConfig,
        TrackCNNHistory,
        apply_cnn_identity_cost,
    )
except ImportError:
    ClassPrediction = None  # type: ignore[assignment,misc]
    CNNIdentityBackend = None  # type: ignore[assignment,misc]
    CNNIdentityCache = None  # type: ignore[assignment,misc]
    CNNIdentityConfig = None  # type: ignore[assignment,misc]
    TrackCNNHistory = None  # type: ignore[assignment,misc]
    apply_cnn_identity_cost = None  # type: ignore[assignment]

try:
    from hydra_suite.core.identity.classification.headtail import (  # noqa: F401
        HeadTailAnalyzer,
    )
except ImportError:
    HeadTailAnalyzer = None  # type: ignore[assignment,misc]

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
