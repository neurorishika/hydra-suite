"""Arena layout: the static slot<->arena mapping and detection->arena lookup.

An arena is a labelled ROI region. Arena membership is a *static* property --
of a track slot for its whole life, and of a detection via its centroid -- so
independent per-arena tracking needs no control-flow change, only this label.

Qt-free and app-layer-free by the Core dependency rule.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np


@dataclass
class ArenaLayout:
    """Slot<->arena mapping plus the frame-space arena label image.

    Slots are laid out in contiguous per-arena blocks: with 3 arenas of 2
    animals, slots 0-1 belong to arena 0, 2-3 to arena 1, 4-5 to arena 2.
    """

    n_arenas: int
    animals_per_arena: int
    label_image: np.ndarray | None = field(default=None, repr=False, compare=False)
    _resize_cache: dict = field(default_factory=dict, repr=False, compare=False)

    @property
    def max_targets(self) -> int:
        return int(self.n_arenas) * int(self.animals_per_arena)

    @property
    def is_single_arena(self) -> bool:
        return int(self.n_arenas) <= 1

    @property
    def slot_arena(self) -> np.ndarray:
        """(max_targets,) int32 arena id per track slot."""
        return np.repeat(
            np.arange(self.n_arenas, dtype=np.int32), self.animals_per_arena
        )

    def label_image_for_size(self, width: int, height: int) -> np.ndarray | None:
        """Label image at (width, height), nearest-neighbour resized and cached.

        INTER_NEAREST is mandatory: any interpolating resize would blend
        neighbouring arena ids and invent labels at arena boundaries.
        """
        if self.label_image is None:
            return None
        if self.label_image.shape[:2] == (height, width):
            return self.label_image
        key = (width, height)
        cached = self._resize_cache.get(key)
        if cached is None:
            cached = cv2.resize(
                self.label_image, (width, height), interpolation=cv2.INTER_NEAREST
            )
            self._resize_cache[key] = cached
        return cached

    def arena_of_points(
        self, xy: np.ndarray, frame_size: tuple[int, int] | None = None
    ) -> np.ndarray:
        """Arena id per point; -1 for points outside every arena.

        Without a label image every point is arena 0, so single-arena runs take
        an identical path to today's.

        `frame_size`, if given, is `(width, height)` of the frame `xy` is
        expressed in -- e.g. after `RESIZE_FACTOR` has scaled the tracking
        frame relative to the label image's native resolution. The label
        image is resized (nearest-neighbour, cached) to match before lookup.
        When omitted, `xy` is assumed to already be in the label image's
        native resolution (today's behaviour, unchanged).
        """
        xy = np.asarray(xy, dtype=np.float32).reshape(-1, 2)
        if xy.shape[0] == 0:
            return np.zeros(0, dtype=np.int32)
        if self.label_image is None:
            return np.zeros(xy.shape[0], dtype=np.int32)
        if frame_size is not None:
            w, h = frame_size
            labels = self.label_image_for_size(w, h)
        else:
            h, w = self.label_image.shape[:2]
            labels = self.label_image
        cx = np.clip(xy[:, 0].astype(np.int32), 0, w - 1)
        cy = np.clip(xy[:, 1].astype(np.int32), 0, h - 1)
        return labels[cy, cx].astype(np.int32) - 1
