"""Active learning core: frame sources, candidate pool, signals, acquisition."""

from .candidate_pool import CandidatePoolConfig, build_candidate_pool
from .frame_source import (
    DetectKitProjectSource,
    FrameRef,
    FrameSource,
    ImageFolderFrameSource,
    VideoFrameSource,
)
from .signals import ALSignals, score_count_deviation, score_crowd, score_uncertainty

__all__ = [
    "ALSignals",
    "CandidatePoolConfig",
    "DetectKitProjectSource",
    "FrameRef",
    "FrameSource",
    "ImageFolderFrameSource",
    "VideoFrameSource",
    "build_candidate_pool",
    "score_count_deviation",
    "score_crowd",
    "score_uncertainty",
]
