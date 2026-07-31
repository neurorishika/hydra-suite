"""Region-source abstraction for the OBB stage (phase C)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Affine:
    """Maps region-local pixel coords -> frame coords: p_frame = p_region * scale + offset."""

    offset: tuple[float, float] = (0.0, 0.0)
    scale: tuple[float, float] = (1.0, 1.0)

    @property
    def is_translate_only(self) -> bool:
        return self.scale == (1.0, 1.0)


Affine.IDENTITY = Affine()
