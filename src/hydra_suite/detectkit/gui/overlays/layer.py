"""A DetectKit canvas overlay layer, as a value object.

Before this existed, a layer's semantics -- whether it is class-filtered,
which colour policy it uses, whether its labels carry confidence, what it
stacks above -- were encoded in WHICH of three set_* methods you called,
and adding a layer meant editing five places in canvas.py. Two of the four
findings an adversarial review raised against the third layer were
layer-bookkeeping defects of exactly that kind.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from typing import TYPE_CHECKING, Optional

from PySide6.QtGui import QColor

if TYPE_CHECKING:
    from PySide6.QtCore import Qt

    from hydra_suite.utils.geometry_levels import GeometryLevel


class ColourPolicy(Enum):
    PER_CLASS = auto()  # index the palette by class_id
    FIXED = auto()  # one hue for the whole layer


class LabelMode(Enum):
    NAME_AND_CLASS_ID = auto()  # "ant (0)"
    NAME_AND_CONFIDENCE = auto()  # "ant (0.42)", or bare "ant" when absent


class Emphasis(Enum):
    UNREVIEWED = auto()  # hatch the native level; keep its own pen style


@dataclass(frozen=True)
class LayerStyle:
    pen_style: "Qt.PenStyle"
    brush_style: "Qt.BrushStyle"
    fill_alpha: int  # 0-255; ignored when brush_style is NoBrush


@dataclass(frozen=True)
class OverlayLayer:
    key: str
    detections: list[dict]
    native_level: "GeometryLevel"
    class_names: list[str] | dict[int, str] | None
    colour_policy: ColourPolicy
    fixed_colour: Optional[QColor] = None
    z: int = 0
    class_filtered: bool = True
    label_mode: LabelMode = LabelMode.NAME_AND_CLASS_ID
    emphasis: Optional[Emphasis] = None
    derive_levels: bool = True
    style: Optional[LayerStyle] = None

    def __post_init__(self) -> None:
        if self.colour_policy is ColourPolicy.FIXED and self.fixed_colour is None:
            raise ValueError("ColourPolicy.FIXED requires fixed_colour")
        if (
            self.colour_policy is ColourPolicy.PER_CLASS
            and self.fixed_colour is not None
        ):
            raise ValueError("fixed_colour is meaningless under ColourPolicy.PER_CLASS")
        if self.style is not None and self.derive_levels:
            raise ValueError(
                "an explicit style applies to the native level only; "
                "set derive_levels=False"
            )
