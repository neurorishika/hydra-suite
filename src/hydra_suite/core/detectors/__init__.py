"""Object detection engines.

Correction 25 (Task 18a): DetectionFilter, create_detector and YOLOOBBDetector
are re-exported with try/except guards so this package remains importable after
those legacy modules are removed in Task 18.  New callers should use
hydra_suite.core.inference.api.apply_detection_filter instead.
"""

from .bg_detector import ObjectDetector

try:
    from .detection_filter import DetectionFilter  # noqa: F401
except ImportError:
    DetectionFilter = None  # type: ignore[assignment,misc]

try:
    from .factory import create_detector  # noqa: F401
except ImportError:
    create_detector = None  # type: ignore[assignment]

try:
    from .yolo_detector import YOLOOBBDetector  # noqa: F401
except ImportError:
    YOLOOBBDetector = None  # type: ignore[assignment,misc]

__all__ = ["ObjectDetector", "YOLOOBBDetector", "create_detector", "DetectionFilter"]
