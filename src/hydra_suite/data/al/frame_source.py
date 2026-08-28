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
    """FrameSource backed by a video file.

    `read()` reuses a single lazily-opened `cv2.VideoCapture` across calls
    instead of reopening the container per frame. When the requested frame is
    exactly the next one after the last frame actually read, it calls
    `cap.read()` directly (no seek) -- the common case for a caller scanning
    frames in ascending order. Any other request (out-of-order, skipped, or
    the very first read of a non-zero frame) falls back to
    `cap.set(CAP_PROP_POS_FRAMES, ...)` first, so random access remains
    correct. Call `close()` (or use as a context manager) to release the
    capture once done.
    """

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
        # The read-side capture is opened lazily on first `read()` -- not
        # every VideoFrameSource is ever read from (e.g. `__iter__`-only
        # probing), so there is no reason to hold a second open handle from
        # construction onward.
        self._cap: cv2.VideoCapture | None = None
        self._last_read_index: int | None = None

    def __iter__(self) -> Iterator[FrameRef]:
        for fid in range(0, self._n_frames, self._stride):
            yield FrameRef(source_id=self._source_id, frame_id=fid, path=None)

    def read(self, ref: FrameRef) -> np.ndarray | None:
        if self._cap is None:
            self._cap = cv2.VideoCapture(self._video_path)
            self._last_read_index = None

        is_next_sequential = (
            ref.frame_id == 0
            if self._last_read_index is None
            else ref.frame_id == self._last_read_index + 1
        )
        if not is_next_sequential:
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, ref.frame_id)
        ok, frame = self._cap.read()
        if ok:
            self._last_read_index = ref.frame_id
        return frame if ok else None

    def length(self) -> int:
        return self._n_frames

    def close(self) -> None:
        """Release the underlying capture, if open. Safe to call more than once."""
        if self._cap is not None:
            self._cap.release()
            self._cap = None
            self._last_read_index = None

    def __enter__(self) -> "VideoFrameSource":
        return self

    def __exit__(self, *_exc_info) -> None:
        self.close()


_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


class ImageFolderFrameSource:
    """FrameSource backed by a directory of image files."""

    def __init__(self, folder: str) -> None:
        self._folder = Path(folder)
        self._paths: list[Path] = sorted(
            p
            for p in self._folder.iterdir()
            if p.is_file() and p.suffix.lower() in _IMAGE_EXTS
        )
        self._source_id = f"folder:{self._folder.name}"

    def __iter__(self) -> Iterator[FrameRef]:
        for idx, p in enumerate(self._paths):
            yield FrameRef(source_id=self._source_id, frame_id=idx, path=str(p))

    def read(self, ref: FrameRef) -> np.ndarray | None:
        if ref.path is None:
            return None
        return cv2.imread(ref.path)

    def length(self) -> int:
        return len(self._paths)


class DetectKitProjectSource:
    """FrameSource backed by all sources in a DetectKitProject.

    Iterates `<source_path>/images/` for each entry in `project.sources`. Project
    sources without an `images/` subdirectory are silently skipped. When
    `only_unlabeled=True`, images with a non-empty `<source_path>/labels/<stem>.txt`
    label file are skipped.
    """

    def __init__(self, project, only_unlabeled: bool = True) -> None:
        self._only_unlabeled = only_unlabeled
        self._items: list[tuple[str, Path]] = []
        for src in getattr(project, "sources", []):
            root = Path(src.path)
            images_dir = root / "images"
            labels_dir = root / "labels"
            if not images_dir.is_dir():
                continue
            for img_path in sorted(images_dir.iterdir()):
                if img_path.suffix.lower() not in _IMAGE_EXTS:
                    continue
                if only_unlabeled:
                    label_path = labels_dir / (img_path.stem + ".txt")
                    if label_path.is_file() and label_path.stat().st_size > 0:
                        continue
                self._items.append((f"project:{src.name}", img_path))

    def __iter__(self) -> Iterator[FrameRef]:
        for idx, (sid, p) in enumerate(self._items):
            yield FrameRef(source_id=sid, frame_id=idx, path=str(p))

    def read(self, ref: FrameRef) -> np.ndarray | None:
        if ref.path is None:
            return None
        return cv2.imread(ref.path)

    def length(self) -> int:
        return len(self._items)
