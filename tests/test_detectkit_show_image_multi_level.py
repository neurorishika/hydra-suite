"""Regression: show_image must render every geometry level, not just native."""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

pytest.importorskip("PySide6")


def test_show_image_drives_the_overlay_providers():
    """show_image must not know how any single layer is built; it builds
    the frame context and asks each provider."""
    import inspect

    from hydra_suite.detectkit.gui.main_window import MainWindow

    source = inspect.getsource(MainWindow.show_image)
    assert "_refresh_overlays" in source
    assert "set_gt_detections" not in source
    assert "clear_escalation_detections" not in source

    refresh = inspect.getsource(MainWindow._refresh_overlays)
    assert "PROVIDERS" in refresh
    assert "remove_layer" in refresh
    assert "set_layer" in refresh


def test_frame_context_retains_the_geometry_kind_that_built_the_cache(monkeypatch):
    """A later model-history edit must not relabel cached segment contours."""
    from types import SimpleNamespace

    import numpy as np
    from PySide6.QtWidgets import QApplication

    from hydra_suite.detectkit.gui.main_window import DetectKitMainWindow

    QApplication.instance() or QApplication([])
    window = DetectKitMainWindow()
    window._project = SimpleNamespace()
    window._current_source_path = "/source"
    window._current_image_path = "/source/images/frame.png"
    assert window._canvas.set_image_array(np.zeros((20, 20, 3), dtype=np.uint8))
    window._dataset_prediction_signature = ("source", "model")
    window._dataset_prediction_inference_kind = "segment_direct"
    window._dataset_prediction_cache = SimpleNamespace(
        read_frame=lambda index: [
            {
                "class_id": 0,
                "confidence": 0.8,
                "polygon_px": [(0, 0), (5, 0), (5, 5)],
            }
        ]
    )
    window._dataset_prediction_paths = SimpleNamespace(index_of=lambda path: 0)
    monkeypatch.setattr(
        window._tools_panel,
        "get_overlay_settings",
        lambda: SimpleNamespace(confidence_threshold=0.5),
    )
    monkeypatch.setattr(
        window,
        "_dataset_signature",
        lambda settings: ("source", "model"),
    )

    context = window._frame_context()
    window.deleteLater()

    assert context is not None
    assert context.prediction_inference_kind == "segment_direct"
    assert len(context.predictions) == 1
