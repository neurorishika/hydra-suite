"""DetectKit <-> shared-widget adapters (S4 Task 16)."""

from __future__ import annotations

import os
from dataclasses import fields

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

pytest.importorskip("PySide6")

from hydra_suite.detectkit.gui.models import SliceTrainingSettings  # noqa: E402
from hydra_suite.detectkit.gui.panels.slice_settings_adapter import (  # noqa: E402
    sam3_tiling_to_spec,
    settings_to_spec,
    spec_to_sam3_tiling,
    spec_to_settings,
)
from hydra_suite.training.contracts import Sam3LoraParams  # noqa: E402

# Exactly what SliceSettingsGroup.to_sam3_tiling() returned before S4.
SAM3_TILING_KEYS = {
    "geometry_mode",
    "object_tile_fraction",
    "object_tile_fractions",
    "full_frame_mix",
    "slice_width",
    "slice_height",
    "tile_overlap",
    "keep_empty_tiles",
    "min_area_ratio",
}


@pytest.fixture(scope="module", autouse=True)
def _qapp():
    # Repo convention (no pytest-qt): see tests/test_detectkit_review_bar.py.
    from PySide6.QtWidgets import QApplication

    yield QApplication.instance() or QApplication([])


_SETTINGS_CASES = [
    SliceTrainingSettings(),
    SliceTrainingSettings(
        enabled=True,
        geometry_mode="custom",
        slice_width=384,
        slice_height=512,
        overlap=0.35,
        min_area_ratio=0.4,
        negative_tile_fraction=0.3,
        target_size_fractions=[0.05, 0.12, 0.2],
        full_frame_mix=False,
        merge_threshold=0.65,
        balance_multiscale_loss=False,
        balance_multiscale_loss_power=0.8,
        reference_body_px=42.5,
    ),
    # Legacy pixel-only project: fractions resolve at the stated 640 anchor.
    SliceTrainingSettings(
        enabled=True, geometry_mode="auto_model", target_sizes=[96.0, 192.0]
    ),
]


def _expected_round_trip(s: SliceTrainingSettings) -> SliceTrainingSettings:
    """What the pre-S4 widget's load_from -> to_settings produced."""
    from statistics import median

    fractions = s.target_fractions() or SliceTrainingSettings().target_fractions()
    return SliceTrainingSettings(
        enabled=s.enabled,
        geometry_mode=s.geometry_mode,
        object_tile_fraction=float(median(fractions)),
        reference_body_px=s.reference_body_px,
        slice_width=s.slice_width,
        slice_height=s.slice_height,
        overlap=s.overlap,
        min_area_ratio=s.min_area_ratio,
        negative_tile_fraction=s.negative_tile_fraction,
        target_size_fractions=list(fractions),
        full_frame_mix=s.full_frame_mix,
        merge_threshold=s.merge_threshold,
        balance_multiscale_loss=s.balance_multiscale_loss,
        balance_multiscale_loss_power=s.balance_multiscale_loss_power,
    )


@pytest.mark.parametrize("settings", _SETTINGS_CASES)
def test_settings_round_trip_through_the_adapter(settings):
    spec, extras = settings_to_spec(settings)
    out = spec_to_settings(spec, extras)
    expected = _expected_round_trip(settings)
    for f in fields(SliceTrainingSettings):
        assert getattr(out, f.name) == pytest.approx(getattr(expected, f.name)), f.name


@pytest.mark.parametrize("settings", _SETTINGS_CASES)
def test_settings_round_trip_through_the_widget(settings):
    from hydra_suite.detectkit.gui.panels.slice_settings_widget import (
        SliceSettingsGroup,
    )

    group = SliceSettingsGroup()
    group.load_from(settings)
    out = group.to_settings()
    expected = _expected_round_trip(settings)
    for f in fields(SliceTrainingSettings):
        assert getattr(out, f.name) == pytest.approx(getattr(expected, f.name)), f.name


def test_unknown_geometry_reads_as_fit_to_animal_size():
    spec, _ = settings_to_spec(SliceTrainingSettings(geometry_mode="bogus"))
    assert spec.geometry_mode == "auto_object"


_SAM3_CASES = [
    Sam3LoraParams(),
    Sam3LoraParams(
        geometry_mode="custom",
        object_tile_fraction=0.0725,
        object_tile_fractions=(0.03, 0.07),
        full_frame_mix=True,
        slice_width=900,
        slice_height=700,
        tile_overlap=0.95,  # above the shared 0.9 ceiling; SAM3 allows [0, 1)
        keep_empty_tiles=False,
        min_area_ratio=0.4,
    ),
]


def _sam3_kwargs(p: Sam3LoraParams) -> dict:
    return {key: getattr(p, key) for key in SAM3_TILING_KEYS}


@pytest.mark.parametrize("params", _SAM3_CASES)
def test_sam3_tiling_round_trips_every_key(params):
    kwargs = _sam3_kwargs(params)
    spec, extras = sam3_tiling_to_spec(kwargs)
    out = spec_to_sam3_tiling(spec, extras)
    assert set(out) == SAM3_TILING_KEYS
    for key in SAM3_TILING_KEYS:
        assert out[key] == pytest.approx(kwargs[key]), key
    assert spec.fragment_policy == "crowd"


def test_sam3_tiling_keys_are_contract_fields():
    spec, extras = sam3_tiling_to_spec(_sam3_kwargs(Sam3LoraParams()))
    emitted = set(spec_to_sam3_tiling(spec, extras))
    contract = {f.name for f in fields(Sam3LoraParams)}
    assert emitted == contract & SAM3_TILING_KEYS


@pytest.mark.parametrize("params", _SAM3_CASES)
def test_sam3_panel_round_trips_its_tiling(params):
    from hydra_suite.detectkit.gui.panels.sam3_training_panel import Sam3TrainingPanel

    panel = Sam3TrainingPanel()
    panel.set_params(params)
    got = panel.params()
    for key in SAM3_TILING_KEYS:
        assert getattr(got, key) == pytest.approx(getattr(params, key)), key


def test_sam3_overlap_spin_is_bounded_by_its_contract():
    """F5: SAM3 overlap used to accept 1.0, which its builder rejects."""
    from hydra_suite.detectkit.gui.panels.sam3_training_panel import Sam3TrainingPanel

    panel = Sam3TrainingPanel()
    assert panel.slice_group.spin_slice_overlap.maximum() < 1.0
    panel.slice_group.spin_slice_overlap.setValue(1.0)
    assert panel.slice_group.to_sam3_tiling()["tile_overlap"] < 1.0


def test_yolo_overlap_spin_is_the_shared_ceiling():
    from hydra_suite.detectkit.gui.panels.slice_settings_widget import (
        SliceSettingsGroup,
    )
    from hydra_suite.utils.tiling_spec import OVERLAP_MAX

    assert SliceSettingsGroup().spin_slice_overlap.maximum() == OVERLAP_MAX


def test_group_keeps_legacy_attribute_aliases():
    from hydra_suite.detectkit.gui.panels.slice_settings_widget import (
        SliceSettingsGroup,
    )

    group = SliceSettingsGroup()
    assert group.cmb_mode is group.combo_slice_geometry
    assert group.txt_targets is group.txt_slice_scales
    assert group.spin_w is group.spin_slice_tile_w
    assert group.spin_overlap is group.spin_slice_overlap
    assert group.chk_full is group.chk_slice_full_frame_mix


def test_inference_dialog_hosts_the_shared_widget():
    from hydra_suite.detectkit.gui.dialogs.inference_settings import (
        InferenceSettingsDialog,
    )
    from hydra_suite.detectkit.gui.models import InferenceRunSettings
    from hydra_suite.widgets.slice_settings import SliceSettingsWidget

    defaults = InferenceRunSettings(
        device="cpu",
        confidence_threshold=0.25,
        slice_settings=SliceTrainingSettings(
            enabled=True,
            geometry_mode="auto_object",
            target_size_fractions=[0.1],
            reference_body_px=50.0,
            overlap=0.3,
            merge_threshold=0.6,
        ),
    )
    dialog = InferenceSettingsDialog(defaults, defaults, model_input_size=1024)
    widgets = dialog.findChildren(SliceSettingsWidget)
    assert [w.role for w in widgets] == ["infer_yolo"]
    assert dialog.combo_geometry is widgets[0].combo_slice_geometry
    out = dialog.settings().slice_settings
    assert (out.enabled, out.geometry_mode, out.object_tile_fraction) == (
        True,
        "auto_object",
        pytest.approx(0.1),
    )
    assert out.reference_body_px == pytest.approx(50.0)
    assert out.overlap == pytest.approx(0.3)
    assert out.merge_threshold == pytest.approx(0.6)
    assert "500" in widgets[0].lbl_slice_tile_size.text()


def test_clamping_a_saved_value_logs_one_warning(caplog):
    import logging

    caplog.set_level(logging.WARNING)
    spec, _ = settings_to_spec(SliceTrainingSettings(overlap=0.95))
    assert spec.overlap == pytest.approx(0.9)
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1
    message = warnings[0].getMessage()
    assert "overlap" in message and "0.95" in message and "0.9" in message


def test_in_range_values_log_nothing(caplog):
    import logging

    caplog.set_level(logging.WARNING)
    settings_to_spec(SliceTrainingSettings(overlap=0.3))
    sam3_tiling_to_spec(_sam3_kwargs(Sam3LoraParams()))
    assert not [r for r in caplog.records if r.levelno == logging.WARNING]


def test_sam3_oversized_tile_is_clamped_with_a_warning(caplog):
    import logging

    caplog.set_level(logging.WARNING)
    spec, _ = sam3_tiling_to_spec(
        {**_sam3_kwargs(Sam3LoraParams()), "slice_width": 100000}
    )
    assert spec.slice_width == 8192
    assert any("slice_width" in r.getMessage() for r in caplog.records)
