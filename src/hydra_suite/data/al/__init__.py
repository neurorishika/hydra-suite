"""Active learning core: frame sources, candidate pool, signals, acquisition."""

from .frame_source import FrameRef, FrameSource, VideoFrameSource

__all__ = ["FrameRef", "FrameSource", "VideoFrameSource"]
