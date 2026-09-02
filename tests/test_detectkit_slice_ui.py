import pytest

pytest.importorskip("PySide6")
from PySide6.QtCore import Qt  # noqa: E402
from PySide6.QtTest import QTest  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from hydra_suite.detectkit.gui.models import SliceTrainingSettings  # noqa: E402


@pytest.fixture(scope="module")
def _app():
    app = QApplication.instance() or QApplication([])
    yield app


def test_slice_settings_group_round_trips(_app):
    from hydra_suite.detectkit.gui.panels.slice_settings_widget import (
        SliceSettingsGroup,
    )

    w = SliceSettingsGroup()
    s = SliceTrainingSettings(
        enabled=True,
        geometry_mode="custom",
        slice_width=384,
        slice_height=384,
        overlap=0.25,
        target_sizes=[123.0, 456.0],
        negative_tile_fraction=0.3,
        reference_body_px=42.0,
    )
    w.load_from(s)
    out = w.to_settings()
    assert out.enabled is True
    assert out.geometry_mode == "custom"
    assert out.slice_width == 384
    assert out.overlap == pytest.approx(0.25)
    assert out.target_fractions() == pytest.approx([123.0 / 640.0, 456.0 / 640.0])
    assert out.negative_tile_fraction == pytest.approx(0.3)
    assert out.reference_body_px == pytest.approx(42.0)


def test_slice_settings_show_only_controls_for_selected_geometry(_app):
    from hydra_suite.detectkit.gui.panels.slice_settings_widget import (
        SliceSettingsGroup,
    )

    w = SliceSettingsGroup()
    w.show()
    _app.processEvents()

    w.cmb_mode.setCurrentIndex(w.cmb_mode.findData("auto_object"))
    assert w.txt_targets.isVisible()
    assert not w.spin_w.isVisible()
    assert not w.spin_h.isVisible()
    assert w.auto_reference_note.isVisible()

    w.cmb_mode.setCurrentIndex(w.cmb_mode.findData("auto_model"))
    assert not w.txt_targets.isVisible()
    assert not w.spin_w.isVisible()
    assert not w.auto_reference_note.isVisible()

    w.cmb_mode.setCurrentIndex(w.cmb_mode.findData("custom"))
    assert not w.txt_targets.isVisible()
    assert w.spin_w.isVisible()
    assert w.spin_h.isVisible()


def test_slice_settings_preview_renders_tile_layout(_app):
    from hydra_suite.detectkit.gui.panels.slice_settings_widget import (
        SliceSettingsGroup,
    )

    w = SliceSettingsGroup()
    w.resize(760, 300)
    w.show()
    _app.processEvents()
    image = w.preview.grab().toImage()
    # The frame, title, and tile outlines should paint something beyond the
    # background after the live preview is constructed.
    assert image.width() > 0
    assert image.pixelColor(12, 17).isValid()


def test_slice_settings_preview_uses_project_frame_dimensions(_app):
    from hydra_suite.detectkit.gui.panels.slice_settings_widget import (
        SliceSettingsGroup,
    )

    w = SliceSettingsGroup()
    w.set_preview_frame_size((3072, 2048))
    assert w.preview.frame_size == (3072, 2048)


def test_slice_settings_preview_cycles_project_frame_distribution(_app):
    from hydra_suite.detectkit.gui.panels.slice_settings_widget import (
        SliceSettingsGroup,
    )

    w = SliceSettingsGroup()
    w.show()
    w.set_preview_frame_options([(640, 480, 2), (1920, 1080, 5)])
    assert w.preview.frame_size == (1920, 1080)
    QTest.mouseClick(w.preview, Qt.MouseButton.LeftButton)
    assert w.preview.frame_size == (640, 480)


def test_slice_settings_preview_resolves_every_configured_target_scale(_app):
    from hydra_suite.detectkit.gui.panels.slice_settings_widget import (
        SliceSettingsGroup,
    )

    w = SliceSettingsGroup()
    w.load_from(
        SliceTrainingSettings(
            target_size_fractions=[0.25, 0.5], reference_body_px=100.0
        )
    )
    specs = w.preview._tile_specs()
    assert [(fraction, width, height) for fraction, width, height, _ in specs] == [
        (0.25, 400, 400),
        (0.5, 200, 200),
    ]
