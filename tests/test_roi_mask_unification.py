"""``engine_params.build_roi_mask`` must reproduce the GUI's live ROI raster.

The GUI rasterizes ``roi_shapes`` into ``self._mw.roi_mask`` in
``gui/orchestrators/session.py::_generate_combined_roi_mask`` (include=255 then
exclude=0, over ``np.zeros((height, width), np.uint8)``). Task 6 routes the
CLI/shared param path's ROI through the Qt-free ``build_roi_mask`` while the
live GUI keeps passing ``self._mw.roi_mask``. For GUI and CLI to stay
byte-identical, those two rasterizers must agree. No gate fixture carries an
ROI (all ``roi_shapes == []``), so the oracle/characterization cannot catch a
divergence -- this focused test does, by driving BOTH the REAL GUI rasterizer
(via an offscreen ``MainWindow``) and the REAL shared ``build_roi_mask`` on a
synthetic ROI (one include circle + one exclude polygon) and asserting
byte-equality. Invoking the real GUI method (rather than an inline copy) means
a future edit to EITHER rasterizer that breaks parity fails this test.
"""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np  # noqa: E402
import pytest  # noqa: E402

pytest.importorskip("PySide6")

from PySide6.QtWidgets import QApplication  # noqa: E402

from hydra_suite.trackerkit.engine_params import build_roi_mask  # noqa: E402
from hydra_suite.trackerkit.gui.main_window import MainWindow  # noqa: E402

WIDTH, HEIGHT = 200, 160
SYNTHETIC_ROI = [
    {"type": "circle", "mode": "include", "params": [100, 80, 60]},
    {
        "type": "polygon",
        "mode": "exclude",
        "params": [[90, 70], [140, 72], [130, 120], [85, 110]],
    },
]


@pytest.fixture(scope="module")
def qapp() -> QApplication:
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


@pytest.fixture
def main_window(monkeypatch, qapp):
    monkeypatch.setattr(MainWindow, "_save_advanced_config", lambda self: None)
    monkeypatch.setattr(MainWindow, "_load_advanced_config", lambda self: {})
    window = MainWindow()
    try:
        yield window
    finally:
        window.close()


def _gui_rasterize(main_window, roi_shapes, height, width):
    """Drive the REAL GUI rasterizer and return the mask it produces."""
    main_window.roi_shapes = list(roi_shapes)
    # _generate_combined_roi_mask(height, width) writes self._mw.roi_mask.
    main_window._session_orch._generate_combined_roi_mask(height, width)
    return main_window.roi_mask


def test_build_roi_mask_matches_gui_rasterization(main_window):
    reference = _gui_rasterize(main_window, SYNTHETIC_ROI, HEIGHT, WIDTH)
    shared = build_roi_mask(SYNTHETIC_ROI, WIDTH, HEIGHT)

    assert shared is not None and reference is not None
    assert shared.dtype == reference.dtype == np.uint8
    assert shared.shape == reference.shape == (HEIGHT, WIDTH)
    assert set(np.unique(shared)).issubset({0, 255})
    assert np.array_equal(shared, reference)


def test_build_roi_mask_empty_shapes_matches_gui_none(main_window):
    # Both the GUI rasterizer and build_roi_mask yield None for no shapes.
    assert _gui_rasterize(main_window, [], HEIGHT, WIDTH) is None
    assert build_roi_mask([], WIDTH, HEIGHT) is None
    assert build_roi_mask(None, WIDTH, HEIGHT) is None
