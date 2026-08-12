"""Per-checkpoint ViTPose input geometry.

The model input size used to be a process-wide constant pinned to COCO's human
portrait aspect (192x256, 0.75). Our animals arrive from OBB tracking roughly
square, so that aspect spent about a quarter of the pixel budget on padding.
This value object makes the geometry a property of the checkpoint instead.

The heatmap is DERIVED, never stored: ClassicHead is two stride-2 transposed
convolutions applied to the patch grid, so its output is always
image / 16 * 4 == image / 4. Keeping it as a second constant only created a
pair that had to agree by hand.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

PATCH_SIZE = 16
HEATMAP_DIVISOR = 4
SIZE_MULTIPLE = 32


@dataclass(frozen=True)
class PoseGeometry:
    """Model input geometry as (width, height).

    Dimensions must be positive multiples of 32: patch-16 embedding needs 16,
    and 32 matches the snapping convention already used for classifier input
    sizes while keeping the heatmap dimension divisible by 8.
    """

    image_size_wh: tuple[int, int]

    def __post_init__(self) -> None:
        wh = tuple(int(v) for v in self.image_size_wh)
        if len(wh) != 2:
            raise ValueError(
                f"image_size_wh must have two entries (width, height); got {wh!r}"
            )
        for value, name in ((wh[0], "width"), (wh[1], "height")):
            if value <= 0:
                raise ValueError(f"{name} must be positive; got {value}")
            if value % SIZE_MULTIPLE:
                raise ValueError(
                    f"{name} must be a multiple of {SIZE_MULTIPLE}; got {value}"
                )
        object.__setattr__(self, "image_size_wh", wh)

    @property
    def heatmap_size_wh(self) -> tuple[int, int]:
        w, h = self.image_size_wh
        return (w // HEATMAP_DIVISOR, h // HEATMAP_DIVISOR)

    @property
    def patch_grid_hw(self) -> tuple[int, int]:
        w, h = self.image_size_wh
        return (h // PATCH_SIZE, w // PATCH_SIZE)

    @property
    def num_tokens(self) -> int:
        """Patch count plus the MAE cls slot upstream keeps in pos_embed."""
        gh, gw = self.patch_grid_hw
        return gh * gw + 1

    @property
    def aspect(self) -> float:
        w, h = self.image_size_wh
        return w / h

    def to_hw(self) -> list[int]:
        """[H, W] -- height first, matching the classifier stack's `input_size`."""
        w, h = self.image_size_wh
        return [h, w]

    @classmethod
    def from_hw(cls, hw: Sequence[int]) -> "PoseGeometry":
        values = [int(v) for v in hw]
        if len(values) != 2:
            raise ValueError(f"input_size must have two entries [H, W]; got {hw!r}")
        h, w = values
        return cls((w, h))


DEFAULT_GEOMETRY = PoseGeometry((192, 256))
