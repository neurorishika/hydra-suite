"""Active learning core: frame sources, candidate pool, signals, acquisition."""

from .frame_source import (
    DetectKitProjectSource,
    FrameRef,
    FrameSource,
    ImageFolderFrameSource,
    VideoFrameSource,
)

__all__ = [
    "DetectKitProjectSource",
    "FrameRef",
    "FrameSource",
    "ImageFolderFrameSource",
    "VideoFrameSource",
]
