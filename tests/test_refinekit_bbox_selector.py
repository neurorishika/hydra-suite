"""Focused ROI dialog regressions for RefineKit."""

from __future__ import annotations

import sys

import numpy as np
import pytest
from PySide6.QtWidgets import QApplication

from hydra_suite.refinekit.gui.dialogs import bbox_selector as bbox_selector_module


@pytest.fixture
def qapp() -> QApplication:
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    return app


def test_bbox_selector_dialog_shows_decoded_frame(
    qapp: QApplication,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _FakeCapture:
        def __init__(self, _path: str) -> None:
            self._frame = np.zeros((40, 60, 3), dtype=np.uint8)
            self._frame[:, :] = (0, 0, 255)

        def get(self, prop: int) -> float:
            if prop == bbox_selector_module.cv2.CAP_PROP_FRAME_COUNT:
                return 10.0
            if prop == bbox_selector_module.cv2.CAP_PROP_FRAME_WIDTH:
                return 60.0
            if prop == bbox_selector_module.cv2.CAP_PROP_FRAME_HEIGHT:
                return 40.0
            return 0.0

        def set(self, _prop: int, _value: float) -> None:
            return None

        def read(self):
            return True, self._frame.copy()

        def release(self) -> None:
            return None

    monkeypatch.setattr(bbox_selector_module.cv2, "VideoCapture", _FakeCapture)

    dialog = bbox_selector_module.BboxSelectorDialog("dummy.mp4", 3)
    dialog.show()
    qapp.processEvents()

    assert dialog._canvas is not None
    image = dialog._canvas.grab().toImage()
    color = image.pixelColor(10, 10)

    assert color.red() > 200
    assert color.green() < 40
    assert color.blue() < 40
