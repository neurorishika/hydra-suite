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


def arena_layout_from_params(params) -> ArenaLayout:
    """Build the layout the way every consumer of engine params must build it.

    One constructor, so a new consumer cannot silently disagree with the live
    tracking path about how many arenas there are or which label image defines
    them -- which is exactly how the parameter optimizer ended up simulating
    unrestricted cross-arena tracking while the real run was gated.
    """
    return ArenaLayout(
        n_arenas=int(params.get("N_ARENAS", 1)),
        animals_per_arena=int(params.get("ANIMALS_PER_ARENA", params["MAX_TARGETS"])),
        label_image=params.get("ARENA_LABELS"),
    )


def check_slot_arena_covers_all_slots(layout: ArenaLayout, n_slots: int) -> None:
    """Raise unless the layout labels exactly ``n_slots`` track slots.

    ``raise``, not ``assert``: a mismatch leaves the numba cost kernel ungated
    (``_arena_arrays`` fails open on any length mismatch) while the identity
    overlay/respawn gates -- which fail open only on a SHORT array -- stay
    active, i.e. a half-gated cost matrix. ``assert`` vanishes under ``-O``.
    """
    if int(layout.slot_arena.shape[0]) != int(n_slots):
        raise RuntimeError(
            "arena_layout.slot_arena must have exactly one entry per "
            "Kalman track slot (N == n_arenas * animals_per_arena) -- a "
            "mismatch here would leave the numba cost kernel ungated "
            "while the identity overlay/respawn gates stay active "
            "(half-gated cost matrix)."
        )


def tracking_frame_size(params, base_w: int, base_h: int):
    """``(width, height)`` of the frame detections are expressed in.

    Mirrors ``worker.py``'s cached-detection fallback: the capture's native
    size scaled by ``RESIZE_FACTOR``. Returns ``None`` when the base size is
    unusable (e.g. the video could not be opened), which makes
    ``arena_of_points`` fall back to the label image's native resolution
    rather than resizing to a degenerate size.
    """
    resize_f = float(params.get("RESIZE_FACTOR", 1.0))
    if int(base_w) <= 0 or int(base_h) <= 0:
        return None
    return (max(1, int(int(base_w) * resize_f)), max(1, int(int(base_h) * resize_f)))


def arena_ids_for_meas(layout: ArenaLayout, meas, frame_size=None):
    """Arena id per detection for a ``[x, y, theta]`` measurement list.

    Returns ``None`` for a single-arena layout: that is what makes single-arena
    callers take the assigner's original, ungated path STRUCTURALLY (see
    ``_arena_arrays``) rather than merely arithmetically.
    """
    if layout.is_single_arena:
        return None
    if len(meas) == 0:
        return np.zeros(0, dtype=np.int32)
    xy = np.asarray([[m[0], m[1]] for m in meas], dtype=np.float32)
    return layout.arena_of_points(xy, frame_size=frame_size)
