"""The SAHI slice-settings widget is shared by the YOLO and SAM3 paths.

Fail-first: every test here raises `AttributeError`/`TypeError` before the
shared widget grew its `backend="sam3"` mode and the SAM3 panel started
using it (the panel had `_object_tile_fractions` as a dead attribute no
widget ever set, so multi-scale SAM3 was unreachable from the GUI).
"""

from __future__ import annotations

import os
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

pytest.importorskip("PySide6")

from PySide6.QtWidgets import QApplication  # noqa: E402


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance() or QApplication(sys.argv)
    yield app


def _panel():
    from hydra_suite.detectkit.gui.panels.sam3_training_panel import Sam3TrainingPanel

    return Sam3TrainingPanel()


def test_sam3_panel_round_trips_a_scale_set_into_params(qapp):
    """A multi-scale set typed into the shared widget reaches Sam3LoraParams."""
    from hydra_suite.training.contracts import Sam3LoraParams

    panel = _panel()
    # Defaults are untouched: empty set, scalar fallback, no full-frame arm.
    assert panel.params().object_tile_fractions == ()
    assert panel.params().full_frame_mix is False
    assert panel.params().object_tile_fraction == Sam3LoraParams().object_tile_fraction

    panel.slice_group.txt_targets.setText("0.05, 0.12, 0.2")
    panel.slice_group.chk_full.setChecked(True)
    got = panel.params()
    assert got.object_tile_fractions == (0.05, 0.12, 0.2)
    assert got.full_frame_mix is True
    # The scalar is NOT the median of the set.
    assert got.object_tile_fraction == Sam3LoraParams().object_tile_fraction

    # And a spec loaded from JSON/CLI shows up in the widget and survives.
    panel.set_params(
        Sam3LoraParams(object_tile_fractions=(0.03, 0.07), full_frame_mix=False)
    )
    assert panel.slice_group.txt_targets.text() == "0.03, 0.07"
    assert panel.params().object_tile_fractions == (0.03, 0.07)
    assert panel.params().full_frame_mix is False


def test_sam3_mode_emits_no_640_derived_pixel_list(qapp):
    """SAM3 runs at 1008px; a 640-anchored size list would be a 1.575x shift."""
    from hydra_suite.detectkit.gui.panels.slice_settings_widget import (
        SliceSettingsGroup,
    )

    widget = SliceSettingsGroup(backend="sam3")
    widget.txt_targets.setText("0.5")
    emitted = widget.to_sam3_tiling()
    assert "target_sizes" not in emitted
    assert 320.0 not in emitted.values()
    assert emitted["object_tile_fractions"] == (0.5,)

    # The YOLO accessors are unreachable in SAM3 mode, so the two independent
    # `full_frame_mix` fields can never be cross-assigned.
    with pytest.raises(RuntimeError):
        widget.to_settings()


def test_hidden_sam3_controls_have_no_contract_field(qapp):
    """Hidden-in-SAM3 controls are absent from what the panel can emit."""
    from dataclasses import fields

    from hydra_suite.detectkit.gui.panels.slice_settings_widget import (
        SliceSettingsGroup,
    )
    from hydra_suite.training.contracts import Sam3LoraParams

    widget = SliceSettingsGroup(backend="sam3")
    emitted = set(widget.to_sam3_tiling())
    contract = {f.name for f in fields(Sam3LoraParams)}
    assert emitted <= contract

    for name in (
        "negative_tile_fraction",
        "merge_threshold",
        "balance_multiscale_loss",
        "balance_multiscale_loss_power",
    ):
        assert name not in contract
        assert name not in emitted

    widget.show()
    assert widget.spin_neg.isVisible() is False
    assert widget.spin_merge.isVisible() is False
    assert widget.chk_balance_loss.isVisible() is False
    assert widget.spin_balance_power.isVisible() is False
    assert widget.chk_enabled.isVisible() is False
    # Controls that DO reach the SAM3 builder stay visible.
    assert widget.spin_min_area.isVisible() is True
    assert widget.chk_keep_empty.isVisible() is True
    assert widget.spin_overlap.isVisible() is True
    widget.hide()


def test_min_area_ratio_is_live_in_both_modes_with_mode_specific_wording(qapp):
    """D3: same number, opposite consequence -- presentation differs, not behaviour."""
    from hydra_suite.detectkit.gui.panels.slice_settings_widget import (
        SliceSettingsGroup,
    )
    from hydra_suite.training.contracts import Sam3LoraParams

    sam3 = SliceSettingsGroup(backend="sam3")
    yolo = SliceSettingsGroup()
    assert "is_crowd" in sam3.spin_min_area.toolTip()
    assert "is_crowd" not in yolo.spin_min_area.toolTip()
    # A bare widget holds no defaults of its own (the R4 duplicated-defaults
    # defect): the panel is the only place SAM3 defaults live, via set_params.
    panel = _panel()
    assert panel.params().min_area_ratio == Sam3LoraParams().min_area_ratio

    panel.set_params(Sam3LoraParams(min_area_ratio=0.4))
    assert panel.params().min_area_ratio == pytest.approx(0.4)


def test_yolo_mode_is_the_default_and_keeps_its_pixel_list(qapp):
    """Characterization guard: the YOLO contract is untouched by the sharing."""
    from hydra_suite.detectkit.gui.panels.slice_settings_widget import (
        SliceSettingsGroup,
    )

    widget = SliceSettingsGroup()
    assert widget.backend == "yolo"
    widget.txt_targets.setText("0.5")
    settings = widget.to_settings()
    assert settings.target_sizes == [320.0]
    assert settings.target_size_fractions == [0.5]
    with pytest.raises(RuntimeError):
        widget.to_sam3_tiling()


def test_sam3_widget_never_falls_back_to_the_yolo_default_scale_set(qapp):
    """An empty box means single-scale for SAM3, not the YOLO four-scale set."""
    from hydra_suite.detectkit.gui.panels.slice_settings_widget import (
        SliceSettingsGroup,
    )

    widget = SliceSettingsGroup(backend="sam3")
    widget.txt_targets.setText("")
    assert widget.to_sam3_tiling()["object_tile_fractions"] == ()
    widget.txt_targets.setText("not a number")
    assert widget.to_sam3_tiling()["object_tile_fractions"] == ()
