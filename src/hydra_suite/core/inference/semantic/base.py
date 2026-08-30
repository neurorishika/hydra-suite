"""The SemanticLabeler seam: a prompt in, instance polygons out.

Qt-free and backend-free by construction -- every test in this subsystem
runs against a fake labeler, and the SAM3 weights are needed only by
``sam3.py``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np


@dataclass(frozen=True)
class SemanticInstance:
    """One segmented instance.

    ``polygon_px`` is in the coordinate space of the image handed to
    ``label_image`` -- TILE-LOCAL under tiled inference. ``tiling.py``
    offsets it to frame space; nothing else may assume frame space.
    """

    polygon_px: np.ndarray  # (P, 2) float32
    confidence: float


@runtime_checkable
class SemanticLabeler(Protocol):
    """A model that turns (image, noun phrase) into instance polygons."""

    @property
    def name(self) -> str:
        """Short identifier for provenance, e.g. ``"sam3"``."""

    def label_image(
        self,
        image_bgr: np.ndarray,
        prompt: str,
        *,
        confidence_threshold: float = 0.0,
        max_instances: int = 0,
    ) -> list[SemanticInstance]:
        """Segment every instance matching *prompt*.

        ``max_instances=0`` means unlimited. Implementations return
        instances sorted by descending confidence.
        """
