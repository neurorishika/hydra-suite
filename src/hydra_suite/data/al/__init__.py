"""Active learning core: frame sources, candidate pool, signals, acquisition."""

from .acquisition import PRESETS, AcquisitionWeights, select
from .candidate_pool import CandidatePoolConfig, build_candidate_pool
from .frame_source import (
    DetectKitProjectSource,
    FrameRef,
    FrameSource,
    ImageFolderFrameSource,
    VideoFrameSource,
)
from .signals import (
    ALSignals,
    score_count_deviation,
    score_crowd,
    score_nms_instability,
    score_uncertainty,
)

__all__ = [
    "AcquisitionWeights",
    "ALSignals",
    "CandidatePoolConfig",
    "DetectKitProjectSource",
    "FrameRef",
    "FrameSource",
    "ImageFolderFrameSource",
    "PRESETS",
    "VideoFrameSource",
    "build_candidate_pool",
    "score_count_deviation",
    "score_crowd",
    "score_nms_instability",
    "score_uncertainty",
    "select",
]
