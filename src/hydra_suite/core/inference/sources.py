"""FrameSource abstraction for the inference pipeline.

A :class:`FrameSource` is an iterator over ``(frame_index, frame)`` pairs in
strict ascending frame-index order.  The CPU implementation wraps cv2 and
mirrors *exactly* the inline ``_frame_source`` generator that was previously
embedded in :meth:`~hydra_suite.core.inference.runner.InferenceRunner.run_batch_pass`.

The abstraction exists so Task 14 can drop in an NVDEC-backed reader without
touching ``run_batch_pass`` or ``Pipeline.run``.

Contract
--------
* Iteration yields ``(int, numpy.ndarray)`` pairs where the int is the
  absolute video frame index and the ndarray is an HWC uint8 BGR image (on
  the CPU path).  NVDEC implementations may yield a CUDA tensor instead of
  a numpy array.
* Frame indices are ascending and contiguous within the requested range.
  The range ``[start_frame, end_frame]`` is *inclusive* on both ends.
* ``frame_count`` returns the number of frames in the requested range
  (``end_frame - start_frame + 1`` after clamping to the video length).
  It is available immediately after construction, before iteration begins.
* ``close()`` releases underlying resources.  It is safe to call more than
  once.  Using the source as a context manager (``with CpuFrameReader(...) as
  src:``) calls ``close()`` on exit.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Iterator

import numpy as np


class FrameSource(ABC):
    """Abstract base for frame providers consumed by :class:`~hydra_suite.core.inference.pipeline.Pipeline`.

    Subclasses yield ``(frame_index, frame)`` pairs in ascending index order
    for the range ``[start_frame, end_frame]`` supplied at construction time.
    """

    @abstractmethod
    def __iter__(self) -> Iterator[tuple[int, np.ndarray]]:
        """Yield ``(frame_index, frame)`` pairs in ascending order."""
        ...

    @property
    @abstractmethod
    def frame_count(self) -> int:
        """Total number of frames in the requested range."""
        ...

    @property
    @abstractmethod
    def start_frame(self) -> int:
        """First frame index (inclusive, 0-based, after clamping to video bounds)."""
        ...

    @property
    @abstractmethod
    def end_frame(self) -> int:
        """Last frame index (inclusive, 0-based, after clamping to video bounds)."""
        ...

    @abstractmethod
    def close(self) -> None:
        """Release underlying resources."""
        ...

    # Context-manager support so callers can write ``with CpuFrameReader(...) as s``.
    def __enter__(self) -> "FrameSource":
        return self

    def __exit__(self, *_) -> None:
        self.close()


class CpuFrameReader(FrameSource):
    """cv2.VideoCapture-backed :class:`FrameSource`.

    Mirrors the semantics of the inline ``_frame_source`` generator that was
    previously embedded in :meth:`~hydra_suite.core.inference.runner.InferenceRunner.run_batch_pass`:

    * Seeks to ``start_frame`` before iteration (``cap.set(CAP_PROP_POS_FRAMES)``).
    * Yields ``(idx, frame)`` for ``idx`` in ``[start_frame, end_frame]`` in
      ascending order, stopping early if ``cap.read()`` returns ``False``.
    * ``end_frame`` is clamped to ``video_total - 1`` so callers may pass an
      arbitrarily large sentinel (e.g. ``sys.maxsize``) without overshooting.
    * ``start_frame`` is clamped to ``max(0, start_frame)``.
    * Frames are ``numpy.ndarray`` HWC uint8 BGR (raw cv2 output, no colour
      conversion) — identical to what ``cap.read()`` returns.

    Parameters
    ----------
    video_path:
        Path to the video file.
    start_frame:
        First frame index (inclusive, 0-based).  Defaults to 0.
    end_frame:
        Last frame index (inclusive, 0-based).  ``None`` means the last frame
        of the video (``video_total - 1``).
    """

    def __init__(
        self,
        video_path: str | Path,
        start_frame: int = 0,
        end_frame: int | None = None,
    ) -> None:
        import cv2

        self._video_path = Path(video_path)
        cap = cv2.VideoCapture(str(self._video_path))
        if not cap.isOpened():
            raise IOError(f"Cannot open video: {self._video_path}")
        self._cap = cap
        self._cv2 = cv2

        video_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if end_frame is None:
            end_frame = video_total - 1
        self._start_frame = max(0, int(start_frame))
        self._end_frame = min(video_total - 1, int(end_frame))
        self._frame_count = max(0, self._end_frame - self._start_frame + 1)

        # Seek to the start position upfront (mirrors run_batch_pass).
        if self._start_frame > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, self._start_frame)

    @property
    def frame_count(self) -> int:
        return self._frame_count

    @property
    def start_frame(self) -> int:
        return self._start_frame

    @property
    def end_frame(self) -> int:
        return self._end_frame

    def __iter__(self) -> Iterator[tuple[int, np.ndarray]]:
        """Yield ``(frame_index, frame)`` pairs from ``start_frame`` to ``end_frame``.

        Stops early on a failed ``cap.read()`` (e.g. corrupted/truncated video),
        exactly as the legacy inline generator did.
        """
        cap = self._cap
        if cap is None:
            return
        idx = self._start_frame
        end = self._end_frame
        while idx <= end:
            ret, frame = cap.read()
            if not ret:
                break
            yield idx, frame
            idx += 1

    def close(self) -> None:
        """Release the cv2 VideoCapture handle."""
        if self._cap is not None:
            self._cap.release()
            self._cap = None  # type: ignore[assignment]
