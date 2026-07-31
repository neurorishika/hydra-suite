"""
HYDRA Suite Package

Holistic YOLO-based Detection Recognition and Analysis Suite.
A comprehensive solution for tracking multiple animals in video recordings using computer vision techniques.
The system combines background subtraction, Kalman filtering, and Hungarian algorithm for robust multi-object tracking.

Key Features:
- Real-time multi-animal tracking
- Circular Region of Interest (ROI) definition for arena-based experiments
- Kalman filter-based motion prediction and state estimation
- Hungarian algorithm for optimal target-detection assignment
- Background subtraction using "lightest pixel" method
- Orientation tracking with anti-flip mechanisms
- CSV data export with trajectory information
- Real-time preview and parameter adjustment
- Configurable tracking parameters with persistence
"""

from importlib import import_module

from hydra_suite._warning_filters import install_warning_filters

install_warning_filters()

try:
    from importlib.metadata import version as _version

    __version__ = _version("hydra-suite")
except Exception:
    __version__ = "1.0.0"  # Fallback for editable installs without metadata

__author__ = "Rishika Mohanta"
__email__ = "neurorishika@gmail.com"

__all__ = [
    "main",
    "parse_arguments",
    "setup_logging",
    "TrackingWorker",
    "MainWindow",
]


def main(*args, **kwargs):
    """Lazy proxy for the TrackerKit entrypoint."""
    return import_module("hydra_suite.trackerkit.app").main(*args, **kwargs)


def parse_arguments(*args, **kwargs):
    """Lazy proxy for TrackerKit CLI argument parsing."""
    return import_module("hydra_suite.trackerkit.app").parse_arguments(*args, **kwargs)


def setup_logging(*args, **kwargs):
    """Lazy proxy for TrackerKit logging setup."""
    return import_module("hydra_suite.trackerkit.app").setup_logging(*args, **kwargs)


def __getattr__(name: str):
    """Lazy-load heavyweight symbols for backwards compatibility."""
    if name == "TrackingWorker":
        return import_module(
            "hydra_suite.trackerkit.gui.workers.tracking_worker"
        ).TrackingWorker
    if name == "MainWindow":
        return import_module("hydra_suite.trackerkit.gui.main_window").MainWindow
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
