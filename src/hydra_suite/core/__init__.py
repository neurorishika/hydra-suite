"""
Core tracking algorithms and components for the HYDRA Suite.

This package contains the tracking worker and supporting components for
multi-object tracking including Kalman filters, background models,
object detection, and track assignment.
"""

from .assigners.hungarian import TrackAssigner
from .background.model import BackgroundModel
from .filters.kalman import KalmanFilterManager
from .identity.dataset.generator import IndividualDatasetGenerator
from .post.processing import (
    interpolate_trajectories,
    process_trajectories,
    process_trajectories_from_csv,
    resolve_trajectories,
)
from .tracking.worker import TrackingWorker

__all__ = [
    "TrackingWorker",
    "KalmanFilterManager",
    "BackgroundModel",
    "TrackAssigner",
    "process_trajectories",
    "process_trajectories_from_csv",
    "resolve_trajectories",
    "interpolate_trajectories",
    "IndividualDatasetGenerator",
]
