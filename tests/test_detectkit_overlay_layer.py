"""Invariants of the OverlayLayer value object.

The point of the value object is that a layer's semantics are STATED, not
implied by which of three set_* methods a caller happened to pick. These
tests pin the combinations that are contradictory, so a provider cannot
construct one.
"""

from __future__ import annotations

import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytest.importorskip("PySide6")

from PySide6.QtCore import Qt  # noqa: E402
from PySide6.QtGui import QColor  # noqa: E402

from hydra_suite.detectkit.gui.overlays import (  # noqa: E402
    ColourPolicy,
    LabelMode,
    LayerStyle,
    OverlayLayer,
)
from hydra_suite.utils.geometry_levels import GeometryLevel  # noqa: E402

_DET = [{"class_id": 0, "polygon_px": [(0, 0), (10, 0), (10, 10), (0, 10)]}]


def _layer(**kw):
    base = dict(
        key="gt",
        detections=_DET,
        native_level=GeometryLevel.POLYGON,
        class_names=["ant"],
        colour_policy=ColourPolicy.PER_CLASS,
    )
    base.update(kw)
    return OverlayLayer(**base)


def test_fixed_colour_policy_requires_a_colour():
    with pytest.raises(ValueError, match="fixed_colour"):
        _layer(colour_policy=ColourPolicy.FIXED)


def test_per_class_policy_rejects_a_fixed_colour():
    """Supplying both would leave which one wins to the renderer."""
    with pytest.raises(ValueError, match="fixed_colour"):
        _layer(colour_policy=ColourPolicy.PER_CLASS, fixed_colour=QColor("red"))


def test_an_explicit_style_forbids_level_derivation():
    """A single LayerStyle cannot describe three derived levels, each of
    which has its own pen and brush."""
    style = LayerStyle(Qt.PenStyle.SolidLine, Qt.BrushStyle.SolidPattern, 65)
    with pytest.raises(ValueError, match="derive_levels"):
        _layer(style=style, derive_levels=True)


def test_defaults_describe_the_ground_truth_layer():
    layer = _layer()
    assert layer.derive_levels is True
    assert layer.style is None
    assert layer.class_filtered is True
    assert layer.label_mode is LabelMode.NAME_AND_CLASS_ID
    assert layer.emphasis is None
    assert layer.z == 0


def test_the_layer_is_frozen():
    layer = _layer()
    with pytest.raises(Exception):  # noqa: B017
        layer.key = "other"
