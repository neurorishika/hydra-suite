"""F1: DetectKit SAHI settings are fraction-only; nothing converts by 640."""

import pytest

pytest.importorskip("PySide6")

from hydra_suite.detectkit.gui.models import (  # noqa: E402
    InferenceRunSettings,
    SliceTrainingSettings,
)


@pytest.fixture(scope="module", autouse=True)
def _qapp():
    # Repo convention (no pytest-qt): see tests/test_detectkit_review_bar.py.
    from PySide6.QtWidgets import QApplication

    yield QApplication.instance() or QApplication([])


@pytest.fixture
def dialog():
    from hydra_suite.detectkit.gui.dialogs.inference_settings import (
        InferenceSettingsDialog,
    )

    defaults = InferenceRunSettings(
        device="cpu",
        confidence_threshold=0.25,
        slice_settings=SliceTrainingSettings(),
    )
    return InferenceSettingsDialog(defaults, defaults, model_input_size=1024)


def test_dialog_round_trips_a_fraction_without_640(dialog):
    dialog.chk_sliced.setChecked(True)
    dialog.spin_object_fraction.setValue(0.12)
    sliced = dialog.settings().slice_settings
    assert sliced.target_size_fractions == [0.12]
    assert sliced.object_tile_fraction == 0.12
    assert sliced.target_fractions() == [0.12]


def test_dialog_shows_pixels_at_the_real_input_size(dialog):
    dialog.spin_object_fraction.setValue(0.1)
    assert "102.4" in dialog.lbl_scale_px.text()  # 0.1 x 1024


def test_dialog_loads_a_legacy_pixel_project_as_its_fraction(dialog):
    legacy = SliceTrainingSettings.from_dict({"enabled": True, "target_sizes": [96.0]})
    dialog.load_from(
        InferenceRunSettings(
            device="cpu", confidence_threshold=0.25, slice_settings=legacy
        )
    )
    assert dialog.spin_object_fraction.value() == pytest.approx(0.15)


def test_training_widget_emits_no_640_pixel_list():
    from hydra_suite.detectkit.gui.panels.slice_settings_widget import (
        SliceSettingsGroup,
    )

    group = SliceSettingsGroup()
    group.set_model_input_size(1024)
    group.txt_targets.setText("0.1, 0.2")
    settings = group.to_settings()
    assert settings.target_size_fractions == [0.1, 0.2]
    # untouched default, ignored whenever fractions are present
    assert settings.target_sizes == SliceTrainingSettings().target_sizes
    assert settings.target_sizes_for(1024) == [102.4, 204.8]
