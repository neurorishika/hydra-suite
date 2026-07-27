import pytest

pytest.importorskip("PySide6")
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
    assert out.target_sizes == [123.0, 456.0]
    assert out.negative_tile_fraction == pytest.approx(0.3)
    assert out.reference_body_px == pytest.approx(42.0)
