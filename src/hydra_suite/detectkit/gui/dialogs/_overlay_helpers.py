"""The two-layer GT/prediction overlay both calibration dialogs draw."""

from __future__ import annotations

from PySide6.QtCore import Qt

from hydra_suite.detectkit.gui.overlays import (
    ColourPolicy,
    LabelMode,
    LayerStyle,
    OverlayLayer,
)
from hydra_suite.utils.geometry_levels import GeometryLevel


def dialog_gt_layer(detections, class_names) -> OverlayLayer:
    return OverlayLayer(
        key="gt",
        detections=detections,
        native_level=GeometryLevel.AABB,
        class_names=class_names,
        colour_policy=ColourPolicy.PER_CLASS,
        derive_levels=False,
        style=LayerStyle(Qt.PenStyle.SolidLine, Qt.BrushStyle.SolidPattern, 65),
        label_mode=LabelMode.NAME_AND_CLASS_ID,
        z=0,
    )


def dialog_pred_layer(detections, class_names) -> OverlayLayer:
    return OverlayLayer(
        key="pred",
        detections=detections,
        native_level=GeometryLevel.AABB,
        class_names=class_names,
        colour_policy=ColourPolicy.PER_CLASS,
        derive_levels=False,
        style=LayerStyle(Qt.PenStyle.DashLine, Qt.BrushStyle.SolidPattern, 55),
        label_mode=LabelMode.NAME_AND_CONFIDENCE,
        z=20,
    )
