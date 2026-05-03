"""Active learning core: frame sources, candidate pool, signals, acquisition."""

from .candidate_pool import CandidatePoolConfig, build_candidate_pool
from .frame_source import (
    DetectKitProjectSource,
    FrameRef,
    FrameSource,
    ImageFolderFrameSource,
    VideoFrameSource,
)

__all__ = [
    "CandidatePoolConfig",
    "DetectKitProjectSource",
    "FrameRef",
    "FrameSource",
    "ImageFolderFrameSource",
    "VideoFrameSource",
    "build_candidate_pool",
]
