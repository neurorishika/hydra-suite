"""Tests for the PoseKit source browser panel's Frame Mode checkbox."""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import sys

import pytest

pytest.importorskip("PySide6")

from PySide6.QtWidgets import QApplication  # noqa: E402

from hydra_suite.posekit.gui.panels.source_browser_panel import (  # noqa: E402
    PoseSourceBrowserPanel,
)


@pytest.fixture()
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    return app


def test_source_browser_panel_has_frame_mode_checkbox_above_labeling_list(qapp):
    panel = PoseSourceBrowserPanel()

    assert panel.chk_frame_mode.text() == "Frame Mode"
    assert not panel.chk_frame_mode.isChecked()
    assert (
        panel.chk_frame_mode.toolTip()
        == "Frame Mode: sampling and labeling operations act on entire frames "
        "(all detected individuals together), not single crops. Required if "
        "you're building a dataset for bottom-up multi-animal pose models."
    )
    # The checkbox is the first widget in the panel's layout, above the
    # "Labeling Set" label.
    layout = panel.layout()
    assert layout.itemAt(0).widget() is panel.chk_frame_mode
