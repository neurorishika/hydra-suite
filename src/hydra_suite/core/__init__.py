"""
Core tracking algorithms and components for the HYDRA Suite.

This package contains the tracking worker and supporting components for
multi-object tracking including Kalman filters, background models,
object detection, and track assignment.

WHY THESE ARE LAZY: importing any leaf under ``hydra_suite.core`` -- say
``core.inference.shape_prior`` -- executes this module first. Eagerly
importing ``TrackingEngineCore`` here dragged in the whole tracking stack,
which reaches ``data`` -> ``data.al`` -> ``filterkit.core`` -> ``sklearn``.
That is fine in the full ``hydra-cuda`` environment and fatal in the slim
``sam3-lora`` sidecar, where a SAM3 training run died at epoch 0 on
``ModuleNotFoundError: No module named 'sklearn'`` merely for importing a
pure-geometry helper. The names below stay importable exactly as before;
they are just resolved on first attribute access instead of at import time.
"""

from importlib import import_module

_LAZY = {
    "TrackAssigner": ".assigners.hungarian",
    "BackgroundModel": ".background.model",
    "KalmanFilterManager": ".filters.kalman",
    "IndividualDatasetGenerator": ".individual.dataset.generator",
    "interpolate_trajectories": ".post.processing",
    "process_trajectories": ".post.processing",
    "process_trajectories_from_csv": ".post.processing",
    "resolve_trajectories": ".post.processing",
    "TrackingEngineCore": ".tracking.worker",
}


def __getattr__(name: str):
    """PEP 562 lazy re-export, mirroring hydra_suite/training/__init__.py."""
    module = _LAZY.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    return getattr(import_module(module, __name__), name)


def __dir__() -> list[str]:
    return sorted(__all__)


__all__ = [
    "TrackingEngineCore",
    "KalmanFilterManager",
    "BackgroundModel",
    "TrackAssigner",
    "process_trajectories",
    "process_trajectories_from_csv",
    "resolve_trajectories",
    "interpolate_trajectories",
    "IndividualDatasetGenerator",
]
