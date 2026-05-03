"""Frame-source adapters for active learning pipelines."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Protocol

import cv2
import numpy as np


@dataclass(frozen=True)
class FrameRef:
    """Reference to one candidate frame within a source."""

    source_id: str
    frame_id: int
    path: str | None = None


class FrameSource(Protocol):
    """Stream of FrameRefs with random-access read."""

    def __iter__(self) -> Iterator[FrameRef]: ...  # noqa: E704

    def read(self, ref: FrameRef) -> np.ndarray | None: ...  # noqa: E704

    def length(self) -> int: ...  # noqa: E704


class VideoFrameSource:
    """FrameSource backed by a video file."""

    def __init__(self, video_path: str, stride: int = 1) -> None:
        if stride < 1:
            raise ValueError("stride must be >= 1")
        self._video_path = video_path
        self._stride = stride
        self._source_id = f"video:{Path(video_path).name}"
        cap = cv2.VideoCapture(video_path)
        try:
            self._n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        finally:
            cap.release()

    def __iter__(self) -> Iterator[FrameRef]:
        for fid in range(0, self._n_frames, self._stride):
            yield FrameRef(source_id=self._source_id, frame_id=fid, path=None)

    def read(self, ref: FrameRef) -> np.ndarray | None:
        cap = cv2.VideoCapture(self._video_path)
        try:
            cap.set(cv2.CAP_PROP_POS_FRAMES, ref.frame_id)
            ok, frame = cap.read()
            return frame if ok else None
        finally:
            cap.release()

    def length(self) -> int:
        return self._n_frames
