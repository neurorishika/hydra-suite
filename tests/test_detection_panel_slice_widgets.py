import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytest.importorskip("PySide6")
from PySide6.QtWidgets import QApplication

_app = QApplication.instance() or QApplication([])

# DetectionPanel takes constructor args in this codebase; construct it the same
# way tests do — via a MainWindow — to avoid guessing its signature. Reuse the
# persistence test's helper.
from tests.test_main_window_config_persistence import _make_main_window


def test_slice_widgets_exist_with_defaults(monkeypatch):
    window = _make_main_window(monkeypatch)
    panel = window._detection_panel
    assert hasattr(panel, "chk_slice_enabled")
    assert panel.chk_slice_enabled.isChecked() is False
    assert hasattr(panel, "combo_slice_geometry")
    items = [
        panel.combo_slice_geometry.itemText(i)
        for i in range(panel.combo_slice_geometry.count())
    ]
    assert items == ["auto_model", "auto_object", "custom"]
    window.close()


def test_slice_widgets_hidden_in_sequential_mode(monkeypatch):
    window = _make_main_window(monkeypatch)
    panel = window._detection_panel
    # Sequential mode = obb-mode combo index 1; _on_yolo_mode_changed drives all
    # direct-only row visibility (detection_panel.py:1837).
    panel.combo_yolo_obb_mode.setCurrentIndex(1)
    panel._on_yolo_mode_changed(1)
    assert panel.chk_slice_enabled.isVisibleTo(panel) is False
    window.close()
