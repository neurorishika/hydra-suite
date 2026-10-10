"""S6: TrackerKit's SAHI block fits the panel and collapses when off.

The YOLO group lives in a narrow side panel (a scroll area). The shared
widget's bottom preview layout must not widen that panel (no sideways
scrolling) in any geometry mode with Advanced open, and SAHI off must leave
only the Enable checkbox (plus no panel-owned SAHI rows).
"""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

pytest.importorskip("PySide6")

from PySide6.QtWidgets import QApplication, QLayout, QScrollArea  # noqa: E402

_TOOL = Path(__file__).resolve().parents[1] / "tools" / "sahi_widget_gallery.py"
# Tolerance for the SAHI-on vs SAHI-off content width (px).
WIDTH_TOLERANCE = 8


@pytest.fixture(scope="module", autouse=True)
def _qapp():
    yield QApplication.instance() or QApplication([])


def _gallery():
    spec = importlib.util.spec_from_file_location("sahi_widget_gallery", _TOOL)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def panel(monkeypatch, tmp_path):
    """TrackerKit at the gallery's size: YOLO direct, a two-profile sidecar."""
    gallery = _gallery()
    for key, sub in (("HYDRA_DATA_DIR", "data"), ("HYDRA_CONFIG_DIR", "config")):
        (tmp_path / sub).mkdir()
        monkeypatch.setenv(key, str(tmp_path / sub))
    from hydra_suite.trackerkit.gui.main_window import MainWindow

    monkeypatch.setattr(MainWindow, "_save_advanced_config", lambda self: None)
    monkeypatch.setattr(MainWindow, "_load_advanced_config", lambda self: {})
    model = gallery._seed_trackerkit_model(tmp_path)
    window = MainWindow()
    window.resize(1500, 1000)
    window._show_workspace()
    panel = window._detection_panel
    window.tabs.setCurrentWidget(panel)
    panel.combo_detection_method.setCurrentIndex(1)  # YOLO
    panel._refresh_yolo_model_combo(preferred_model_path=str(model))
    window._set_yolo_model_selection(str(model))
    panel.combo_yolo_obb_mode.setCurrentIndex(0)  # Direct
    panel.apply_slice_meta_for_model(str(model))
    panel.set_slice_preview_frame_size(2448, 2048)
    window.show()
    yield panel
    window.close()
    window.deleteLater()


def _settle(widget) -> None:
    app = QApplication.instance()
    for _ in range(3):
        app.processEvents()
    layouts = widget.findChildren(QLayout)
    for layout in layouts:
        layout.invalidate()
    for layout in reversed(layouts):
        layout.activate()
    for _ in range(3):
        app.processEvents()


def _content(panel):
    return panel.findChild(QScrollArea).widget()


def _content_min_width(panel) -> int:
    content = _content(panel)
    _settle(content)
    return content.minimumSizeHint().width()


def test_sahi_block_does_not_widen_the_panel(panel):
    panel.chk_slice_enabled.setChecked(False)
    off_width = _content_min_width(panel)
    off_yolo = panel.yolo_group.minimumSizeHint().width()
    panel.chk_slice_enabled.setChecked(True)
    panel.slice_settings.btn_slice_advanced.setChecked(True)
    widths = {}
    for profile in ("balanced", "fast"):
        panel.combo_slice_profile.setCurrentIndex(
            panel.combo_slice_profile.findData(profile)
        )
        for mode in ("auto_model", "auto_object", "custom"):
            combo = panel.combo_slice_geometry
            combo.setCurrentIndex(combo.findData(mode))
            widths[(profile, mode)] = _content_min_width(panel)
    assert max(widths.values()) <= off_width + WIDTH_TOLERANCE, (off_width, widths)
    # Sanity: the YOLO group did grow beyond its SAHI-off width only within
    # what the other groups already need.
    assert panel.yolo_group.minimumSizeHint().width() <= off_width, off_yolo


def test_sahi_adds_no_sideways_scroll_at_the_default_window_size(panel):
    """The direct check: SAHI on (Advanced open) scrolls sideways no further
    than SAHI off. (At 1500 px the Reference Scale group's three buttons
    already overflow the viewport by a few px; that is not SAHI's.)"""
    scroll = panel.findChild(QScrollArea)
    panel.chk_slice_enabled.setChecked(False)
    _settle(_content(panel))
    off = scroll.horizontalScrollBar().maximum()
    panel.chk_slice_enabled.setChecked(True)
    panel.slice_settings.btn_slice_advanced.setChecked(True)
    _settle(_content(panel))
    assert scroll.horizontalScrollBar().maximum() <= off


def test_bottom_preview_is_below_the_controls_and_visible_when_on(panel):
    panel.chk_slice_enabled.setChecked(True)
    widget = panel.slice_settings
    _settle(_content(panel))
    assert widget.capabilities.preview_position == "bottom"
    assert widget.preview.isVisibleTo(panel)
    assert widget.preview.geometry().top() > widget._controls.geometry().bottom()
    assert widget.preview.height() >= widget.preview.MIN_BOTTOM_HEIGHT


def test_sahi_off_hides_every_sahi_row_including_panel_owned(panel):
    widget = panel.slice_settings
    panel.chk_slice_enabled.setChecked(True)
    widget.btn_slice_advanced.setChecked(True)
    assert panel.lbl_slice_batch_admission.isVisibleTo(panel)
    assert panel.lbl_slice_profile_status.isVisibleTo(panel)
    assert panel.combo_slice_profile.isVisibleTo(panel)
    on_height = _content(panel).sizeHint().height()

    panel.chk_slice_enabled.setChecked(False)
    assert panel.chk_slice_enabled.isVisibleTo(panel)
    for hidden in (
        panel.combo_slice_profile,
        panel.combo_slice_geometry,
        panel.spin_slice_overlap,
        panel.spin_slice_tile_batch,
        widget.btn_slice_advanced,
        widget.preview,
        panel.lbl_slice_batch_admission,
        panel.lbl_slice_profile_status,
    ):
        assert not hidden.isVisibleTo(panel), hidden
    _settle(_content(panel))
    assert _content(panel).sizeHint().height() < on_height - 200

    panel.chk_slice_enabled.setChecked(True)
    assert widget.btn_slice_advanced.isChecked()
    assert panel.lbl_slice_batch_admission.isVisibleTo(panel)
    assert panel.lbl_slice_profile_status.isVisibleTo(panel)
    assert widget.preview.isVisibleTo(panel)


def test_focus_leaves_the_profile_combo_when_sahi_is_turned_off(panel):
    """Keyboard focus never stays on a control the SAHI-off collapse hid
    (Down on a hidden profile combo would silently switch profiles)."""
    window = panel.window()
    window.activateWindow()
    panel.chk_slice_enabled.setChecked(True)
    _settle(_content(panel))
    panel.combo_slice_profile.setFocus()
    _settle(_content(panel))
    assert window.focusWidget() is panel.combo_slice_profile
    panel.chk_slice_enabled.setChecked(False)  # programmatic: no click focus
    _settle(_content(panel))
    assert window.focusWidget() is panel.chk_slice_enabled
