"""Tests for the Reference Scale preview: box-geometry selection + wiring."""

from __future__ import annotations

import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")

from hydra_suite.core.canonicalization.geometry import CanonicalGeometry
from hydra_suite.trackerkit.gui.panels.reference_scale_preview import (
    select_boxes,
    select_palette,
)


def test_select_boxes_current_only_without_filters():
    boxes = select_boxes(
        reference_body_px=80.0, reference_aspect_ratio=2.0, canonical_margin=1.3
    )
    assert [cat for cat, _geom in boxes] == ["current"]
    expected = CanonicalGeometry.from_reference(80.0, 2.0, 1.3)
    assert boxes[0][1] == expected


def test_select_boxes_includes_size_extremes_when_size_filter_set():
    boxes = select_boxes(
        reference_body_px=80.0,
        reference_aspect_ratio=2.0,
        canonical_margin=1.3,
        size_filter=(0.5, 1.5),
    )
    categories = [cat for cat, _geom in boxes]
    assert categories == ["current", "size", "size"]
    _cat, min_geom = boxes[1]
    _cat, max_geom = boxes[2]
    assert min_geom.canvas_w < boxes[0][1].canvas_w < max_geom.canvas_w


def test_select_boxes_includes_aspect_extremes_when_aspect_filter_set():
    boxes = select_boxes(
        reference_body_px=80.0,
        reference_aspect_ratio=2.0,
        canonical_margin=1.3,
        aspect_filter=(0.75, 1.5),
    )
    categories = [cat for cat, _geom in boxes]
    assert categories == ["current", "aspect", "aspect"]


def test_select_boxes_combines_both_filters():
    boxes = select_boxes(
        reference_body_px=80.0,
        reference_aspect_ratio=2.0,
        canonical_margin=1.3,
        size_filter=(0.5, 1.5),
        aspect_filter=(0.75, 1.5),
    )
    categories = [cat for cat, _geom in boxes]
    assert categories == ["current", "size", "size", "aspect", "aspect"]


def test_select_boxes_clamps_degenerate_inputs():
    # aspect_ratio multiplier can push the extreme below 1.0 -- from_reference
    # clamps aspect_ratio >= 1.0 internally; selection should not crash.
    boxes = select_boxes(
        reference_body_px=80.0,
        reference_aspect_ratio=1.0,
        canonical_margin=1.3,
        aspect_filter=(0.1, 1.0),
    )
    assert len(boxes) == 3
    for _cat, geom in boxes:
        assert geom.aspect_ratio >= 1.0


def test_select_palette_switches_on_luminance_threshold():
    light_bg_palette = select_palette(200.0)
    dark_bg_palette = select_palette(30.0)
    assert light_bg_palette != dark_bg_palette
    assert set(light_bg_palette) == {"current", "size", "aspect"}
    assert set(dark_bg_palette) == {"current", "size", "aspect"}
    # Light backgrounds need darker, more saturated colors for contrast.
    assert sum(light_bg_palette["current"]) < sum(dark_bg_palette["current"])


def test_select_palette_never_uses_yellow_hue():
    for luminance in (0.0, 64.0, 127.9, 128.0, 200.0, 255.0):
        for color in select_palette(luminance).values():
            r, g, b = color
            # Yellow is high R+G with low B; reject anything close to that.
            assert not (r > 180 and g > 180 and b < 100)


def test_reference_scale_preview_widget_placeholder_before_detections():
    from PySide6.QtWidgets import QApplication

    from hydra_suite.trackerkit.gui.panels.reference_scale_preview import (
        ReferenceScalePreviewWidget,
    )

    QApplication.instance() or QApplication([])
    widget = ReferenceScalePreviewWidget()
    assert widget.pixmap() is None or widget.pixmap().isNull()
    assert "Test Detection" in widget.text()


def test_reference_scale_preview_widget_renders_pixmap_for_detections():
    import numpy as np
    from PySide6.QtWidgets import QApplication

    from hydra_suite.trackerkit.gui.panels.reference_scale_preview import (
        ReferenceScalePreviewWidget,
    )

    QApplication.instance() or QApplication([])
    widget = ReferenceScalePreviewWidget()
    frame = np.zeros((300, 300, 3), dtype=np.uint8)
    detections = [(150.0, 150.0, 60.0, 30.0, 0.0)]
    widget.set_detections(frame, 1.0, detections)
    widget.set_box_params(
        reference_body_px=40.0,
        reference_aspect_ratio=2.0,
        canonical_margin=1.3,
        size_filter=(0.5, 1.5),
        aspect_filter=(0.75, 1.5),
    )
    assert widget.text() == ""
    assert not widget.pixmap().isNull()
    assert not widget._legend_label.isHidden()
    legend_text = widget._legend_label.text()
    assert "reference box" in legend_text
    assert "size filter range" in legend_text
    assert "aspect filter range" in legend_text
    # Only one detection in this run -- clicking would have nothing to switch
    # to, so the "click for another" hint must stay hidden.
    assert widget._hint_label.isHidden()


def test_reference_scale_preview_widget_shows_resample_hint_with_multiple_detections():
    import numpy as np
    from PySide6.QtWidgets import QApplication

    from hydra_suite.trackerkit.gui.panels.reference_scale_preview import (
        ReferenceScalePreviewWidget,
    )

    QApplication.instance() or QApplication([])
    widget = ReferenceScalePreviewWidget()
    frame = np.zeros((300, 300, 3), dtype=np.uint8)
    detections = [
        (150.0, 150.0, 60.0, 30.0, 0.0),
        (80.0, 80.0, 60.0, 30.0, 10.0),
    ]
    widget.set_detections(frame, 1.0, detections)
    widget.set_box_params(
        reference_body_px=40.0, reference_aspect_ratio=2.0, canonical_margin=1.3
    )
    assert not widget._hint_label.isHidden()
    assert "click" in widget._hint_label.text().lower()


def test_reference_scale_preview_widget_legend_omits_inactive_filters():
    import numpy as np
    from PySide6.QtWidgets import QApplication

    from hydra_suite.trackerkit.gui.panels.reference_scale_preview import (
        ReferenceScalePreviewWidget,
    )

    QApplication.instance() or QApplication([])
    widget = ReferenceScalePreviewWidget()
    frame = np.zeros((300, 300, 3), dtype=np.uint8)
    widget.set_detections(frame, 1.0, [(150.0, 150.0, 60.0, 30.0, 0.0)])
    widget.set_box_params(
        reference_body_px=40.0, reference_aspect_ratio=2.0, canonical_margin=1.3
    )
    legend_text = widget._legend_label.text()
    assert "reference box" in legend_text
    assert "size filter range" not in legend_text
    assert "aspect filter range" not in legend_text


def test_reference_scale_preview_widget_empty_detections_shows_message():
    import numpy as np
    from PySide6.QtWidgets import QApplication

    from hydra_suite.trackerkit.gui.panels.reference_scale_preview import (
        ReferenceScalePreviewWidget,
    )

    QApplication.instance() or QApplication([])
    widget = ReferenceScalePreviewWidget()
    frame = np.zeros((300, 300, 3), dtype=np.uint8)
    widget.set_detections(frame, 1.0, [])
    assert "No detections" in widget.text()


def test_detection_panel_has_reference_scale_preview_widget():
    from PySide6.QtWidgets import QApplication

    from hydra_suite.trackerkit.gui.main_window import MainWindow
    from hydra_suite.trackerkit.gui.panels.reference_scale_preview import (
        ReferenceScalePreviewWidget,
    )

    QApplication.instance() or QApplication([])
    w = MainWindow()
    try:
        panel = w._detection_panel
        assert isinstance(panel.reference_scale_preview, ReferenceScalePreviewWidget)
    finally:
        w.close()
