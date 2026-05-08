"""Data I/O and dataset preparation utilities.

Correction 25 (Task 18a): DetectionCache is re-exported with a try/except guard
so this package remains importable after the legacy detection_cache module is
removed in Task 18.  New callers should use
hydra_suite.core.inference.cache.DetectionCacheHandle instead.
"""

from .csv_writer import CSVWriterThread
from .dataset_generation import FrameQualityScorer, export_dataset
from .dataset_merge import (
    detect_dataset_layout,
    get_dataset_class_name,
    merge_datasets,
    rewrite_labels_to_single_class,
    update_dataset_class_name,
    validate_labels,
)

try:
    from .detection_cache import DetectionCache  # noqa: F401
except ImportError:
    DetectionCache = None  # type: ignore[assignment,misc]

__all__ = [
    "CSVWriterThread",
    "DetectionCache",
    "FrameQualityScorer",
    "detect_dataset_layout",
    "export_dataset",
    "get_dataset_class_name",
    "merge_datasets",
    "rewrite_labels_to_single_class",
    "update_dataset_class_name",
    "validate_labels",
]
