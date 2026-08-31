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
