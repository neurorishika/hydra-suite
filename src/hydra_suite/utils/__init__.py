"""
Utility modules for the HYDRA Suite.

This package contains various utility functions and classes for image processing,
CSV writing, ROI handling, video processing, GPU acceleration, and other common operations.
"""

from .frame_prefetcher import FramePrefetcher
from .geometry import fit_circle_to_points, wrap_angle_degs
from .gpu_utils import (
    CUDA_AVAILABLE,
    GPU_AVAILABLE,
    MPS_AVAILABLE,
    NUMBA_AVAILABLE,
    get_device_info,
    log_device_info,
)
from .image_processing import apply_image_adjustments, stabilize_lighting
from .video_encoder import VideoEncoder, probe_video_backend

__all__ = [
    "apply_image_adjustments",
    "stabilize_lighting",
    "fit_circle_to_points",
    "wrap_angle_degs",
    "FramePrefetcher",
    "CUDA_AVAILABLE",
    "MPS_AVAILABLE",
    "GPU_AVAILABLE",
    "NUMBA_AVAILABLE",
    "get_device_info",
    "log_device_info",
    "VideoEncoder",
    "probe_video_backend",
]
