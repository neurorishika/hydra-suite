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
        img = cv2.imread(ref.path)
        return img if img is not None else None

    def length(self) -> int:
        return len(self._paths)


class DetectKitProjectSource:
    """FrameSource backed by all sources in a DetectKitProject.

    `only_unlabeled=True` skips images that have a corresponding non-empty `.txt`
    label file in the source's `labels/` directory.
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
