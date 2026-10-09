"""The shared SAHI settings widget (S4 Task 15, plan Revision 1 decisions 22-33)."""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import Qt  # noqa: E402
from PySide6.QtWidgets import (  # noqa: E402
    QAbstractButton,
    QAbstractSpinBox,
    QApplication,
    QComboBox,
    QLineEdit,
)

from hydra_suite.utils.tiling_spec import (  # noqa: E402
    FRACTION_MAX,
    FRACTION_MIN,
    OVERLAP_MAX,
    TilingSpec,
)
from hydra_suite.widgets.slice_settings import (  # noqa: E402
    ROLES,
    SliceSettingsWidget,
    SliceWidgetCapabilities,
)


@pytest.fixture(scope="module", autouse=True)
def _qapp():
    # Repo convention (no pytest-qt): see tests/test_detectkit_review_bar.py.
    yield QApplication.instance() or QApplication([])


def test_unknown_role_is_rejected():
    with pytest.raises(ValueError):
        SliceSettingsWidget(role="tracker")


@pytest.mark.parametrize("role", ROLES)
def test_every_role_round_trips_its_spec(role):
    w = SliceSettingsWidget(role=role)
    spec = TilingSpec(
        enabled=True,
        geometry_mode="auto_object",
        object_tile_fractions=(0.1,),
        reference_body_px=40.0,
        overlap=0.3,
    )
    w.set_spec(spec)
    got = w.spec()
    assert got.object_tile_fractions == (0.1,)
    assert got.reference_body_px == pytest.approx(40.0)
    if role != "escalate_sam2":
        assert got.overlap == pytest.approx(0.3)
    else:
        # SAM2 owner tiles always overlap by the host's constant.
        assert got.overlap == pytest.approx(0.5)


@pytest.mark.parametrize("role", ROLES)
def test_fields_the_widget_does_not_show_round_trip_untouched(role):
    w = SliceSettingsWidget(role=role)
    spec = TilingSpec(
        enabled=True,
        geometry_mode="auto_object",
        object_tile_fractions=(0.2,),
        fragment_policy="mask",
        merge_policy="nms",
        merge_metric="iou",
    )
    w.set_spec(spec)
    got = w.spec()
    assert (got.fragment_policy, got.merge_policy, got.merge_metric) == (
        "mask",
        "nms",
        "iou",
    )


def test_geometry_combo_shows_labels_but_stores_enums():
    w = SliceSettingsWidget(role="infer_yolo")
    combo = w.combo_slice_geometry
    assert [combo.itemData(i) for i in range(combo.count())] == [
        "auto_model",
        "auto_object",
        "custom",
    ]
    assert [combo.itemText(i) for i in range(combo.count())] == [
        "Use model input size",
        "Fit to animal size",
        "Custom tile size",
    ]
    for i in range(combo.count()):
        assert combo.itemData(i) in combo.itemData(i, Qt.ItemDataRole.ToolTipRole)


def test_ranges_come_from_the_contract():
    """F5: SAM3 overlap used to accept 1.0; its own contract is [0, 1)."""
    for role in ("infer_yolo", "train_yolo", "escalate_sam3"):
        w = SliceSettingsWidget(role=role)
        assert w.spin_slice_overlap.maximum() == OVERLAP_MAX
    sam3 = SliceSettingsWidget(role="train_sam3")
    assert sam3.spin_slice_overlap.maximum() == pytest.approx(0.99)
    w = SliceSettingsWidget(role="infer_yolo")
    assert (
        w.spin_slice_object_fraction.minimum(),
        w.spin_slice_object_fraction.maximum(),
    ) == (FRACTION_MIN, FRACTION_MAX)
    assert w.spin_slice_tile_w.maximum() == 8192
    for role in ("escalate_sam3", "escalate_sam2"):
        esc = SliceSettingsWidget(role=role)
        assert esc.spin_slice_object_fraction.minimum() == 0.0
        assert esc.spin_slice_object_fraction.maximum() == FRACTION_MAX


def test_sam3_overlap_above_the_shared_ceiling_survives_through_extras():
    w = SliceSettingsWidget(role="train_sam3")
    w.set_spec(
        TilingSpec(enabled=True, geometry_mode="auto_object", overlap=0.9),
        extras={"tile_overlap": 0.95},
    )
    assert w.spin_slice_overlap.value() == pytest.approx(0.95)
    assert w.extras()["tile_overlap"] == pytest.approx(0.95)
    assert w.spec().overlap == pytest.approx(OVERLAP_MAX)


def test_tile_size_editable_only_in_custom():
    w = SliceSettingsWidget(role="infer_yolo")
    w.set_spec(
        TilingSpec(
            enabled=True,
            geometry_mode="auto_object",
            object_tile_fractions=(0.1,),
            reference_body_px=50.0,
        )
    )
    # Constrained, not hidden: the row stays on screen, disabled.
    assert not w.spin_slice_tile_w.isEnabled()
    assert not w.spin_slice_tile_w.isHidden()
    assert w.source_badge("tile_size") == "derived"
    assert "500" in w.lbl_slice_tile_size.text()  # 50 px / 0.1
    w.combo_slice_geometry.setCurrentIndex(w.combo_slice_geometry.findData("custom"))
    assert w.spin_slice_tile_w.isEnabled()
    assert w.source_badge("tile_size") == "user"


def test_disabling_tiling_disables_the_rest():
    w = SliceSettingsWidget(role="infer_yolo")
    w.set_spec(TilingSpec(enabled=False))
    assert not w.combo_slice_geometry.isEnabled()
    assert not w.spin_slice_overlap.isEnabled()
    w.chk_slice_enabled.setChecked(True)
    assert w.combo_slice_geometry.isEnabled()
    assert w.spin_slice_overlap.isEnabled()


def test_body_is_read_only_until_override():
    w = SliceSettingsWidget(role="escalate_sam3")
    w.set_reference_body(82.2, "dataset")
    assert not w.spin_slice_body.isEnabled()
    assert w.source_badge("reference_body_px") == "dataset"
    assert "dataset" in w.lbl_slice_body_badge.text()
    w.chk_slice_body_override.setChecked(True)
    assert w.spin_slice_body.isEnabled()
    assert w.source_badge("reference_body_px") == "override"
    w.spin_slice_body.setValue(120.0)
    w.chk_slice_body_override.setChecked(False)
    # Dropping the override restores the derived value and its source.
    assert w.spin_slice_body.value() == pytest.approx(82.2)
    assert w.source_badge("reference_body_px") == "dataset"


def test_unknown_body_is_editable_without_override():
    """I6: the last link of the body chain is the user; 0 must stay typeable."""
    for role in ("escalate_sam3", "escalate_sam2", "infer_yolo"):
        w = SliceSettingsWidget(role=role)
        w.set_spec(TilingSpec(enabled=True, geometry_mode="auto_object"))
        w.set_reference_body(0.0, "dataset")
        assert w.spin_slice_body.isEnabled(), role
        # Unknown is the user's to enter: badged "user", nothing to override.
        assert w.source_badge("reference_body_px") == "user"
        assert w.chk_slice_body_override.isHidden()
        assert "unknown" in w.spin_slice_body.specialValueText()
        w.spin_slice_body.setValue(40.0)
        assert w.spin_slice_body.isEnabled(), role  # never locks mid-edit
        assert w.source_badge("reference_body_px") == "user"


def test_body_display_only_without_override_capability():
    w = SliceSettingsWidget(
        role="infer_yolo", capabilities=SliceWidgetCapabilities(body_override=False)
    )
    w.set_spec(TilingSpec(enabled=True, geometry_mode="auto_object"))
    w.set_reference_body(60.0, "stamped")
    assert not w.spin_slice_body.isEnabled()
    assert w.chk_slice_body_override.isHidden()


def test_display_only_body_stays_read_only_even_when_unknown():
    """A host that owns the body elsewhere (TrackerKit: the model's stamp or
    profile) shows it as a badge only; an unknown 0 must not become an
    editable control that is bound to nothing."""
    w = SliceSettingsWidget(
        role="infer_yolo",
        capabilities=SliceWidgetCapabilities(
            body_override=False, body_display_only=True
        ),
    )
    w.chk_slice_enabled.setChecked(True)
    w.combo_slice_geometry.setCurrentIndex(
        w.combo_slice_geometry.findData("auto_object")
    )
    w.set_reference_body(0.0, "default")
    assert not w.spin_slice_body.isEnabled()
    assert w.chk_slice_body_override.isHidden()
    # The host's source is kept, not rewritten to "user" (it is not the user's).
    assert w.source_badge("reference_body_px") == "default"
    w.set_reference_body(75.0, "stamped")
    assert not w.spin_slice_body.isEnabled()
    assert w.source_badge("reference_body_px") == "stamped"
    # Default capabilities keep I6: unknown stays typeable.
    assert SliceWidgetCapabilities().body_display_only is False


def test_merge_threshold_row_can_be_left_to_the_profile():
    """TrackerKit's merge settings are profile-owned (S3 deviation 14): its
    infer_yolo widget has no merge-threshold row, so nothing editable is
    bound to slice_merge_threshold. Default keeps DetectKit's row."""
    caps = SliceWidgetCapabilities(
        advanced_merge=False, merge_threshold_row=False, execution_knobs=True
    )
    w = SliceSettingsWidget(role="infer_yolo", capabilities=caps)
    w.set_advanced_expanded(True)
    assert w.spin_slice_merge.isHidden()
    assert "merge_threshold" not in w.extras()
    assert not w.spin_slice_tile_batch.isHidden()
    default = SliceSettingsWidget(
        role="infer_yolo", capabilities=SliceWidgetCapabilities(advanced_merge=False)
    )
    default.set_advanced_expanded(True)
    assert not default.spin_slice_merge.isHidden()
    assert "merge_threshold" in default.extras()


def test_overlap_at_or_above_the_whole_animal_minimum_is_left_alone():
    """F7 (decision 22): max(scale) + margin is a MINIMUM, never a nudge down."""
    w = SliceSettingsWidget(role="train_yolo")
    w.set_spec(
        TilingSpec(
            enabled=True,
            geometry_mode="auto_object",
            object_tile_fractions=(0.05, 0.15),
            overlap=0.25,
        )
    )
    assert w.spin_slice_overlap.value() == pytest.approx(0.25)
    assert w.spin_slice_overlap.isEnabled()
    assert w.lbl_slice_overlap_minimum.text() == "≥ whole-animal minimum (0.20)"
    assert w.btn_slice_overlap_raise.isHidden()


def test_overlap_below_the_minimum_warns_and_offers_a_raise():
    w = SliceSettingsWidget(role="train_yolo")
    w.set_spec(
        TilingSpec(
            enabled=True,
            geometry_mode="auto_object",
            object_tile_fractions=(0.05, 0.15),
            overlap=0.1,
        )
    )
    assert w.spin_slice_overlap.value() == pytest.approx(0.1)  # never applied
    assert w.lbl_slice_overlap_minimum.text() == "Below whole-animal minimum (0.20)"
    assert not w.btn_slice_overlap_raise.isHidden()
    assert w.btn_slice_overlap_raise.isEnabled()
    assert w.btn_slice_overlap_raise.text() == "Raise to 0.20"
    hits = []
    w.field_changed.connect(hits.append)
    w.btn_slice_overlap_raise.click()
    assert w.spin_slice_overlap.value() == pytest.approx(0.2)
    assert "overlap" in hits
    assert w.btn_slice_overlap_raise.isHidden()
    assert w.lbl_slice_overlap_minimum.text().startswith("≥")


def test_overlap_minimum_follows_the_scales():
    w = SliceSettingsWidget(role="infer_yolo")
    w.set_spec(
        TilingSpec(enabled=True, geometry_mode="auto_object", overlap=0.2),
    )
    w.spin_slice_object_fraction.setValue(0.4)
    assert "0.45" in w.lbl_slice_overlap_minimum.text()
    assert w.lbl_slice_overlap_minimum.text().startswith("Below")
    assert w.spin_slice_overlap.value() == pytest.approx(0.2)  # untouched


def test_overlap_raise_target_rounds_up_to_the_spin_precision():
    """0.055 + 0.05 = 0.105 must not round DOWN below the minimum at 2 dp."""
    w = SliceSettingsWidget(role="escalate_sam3")
    w.set_spec(
        TilingSpec(
            enabled=True,
            geometry_mode="auto_object",
            object_tile_fractions=(0.055,),
            overlap=0.1,
        )
    )
    assert w.btn_slice_overlap_raise.text() == "Raise to 0.11"
    w.btn_slice_overlap_raise.click()
    assert w.spin_slice_overlap.value() == pytest.approx(0.11)
    assert w.btn_slice_overlap_raise.isHidden()


def test_full_frame_escalation_has_no_overlap_minimum():
    w = SliceSettingsWidget(role="escalate_sam3")
    w.set_spec(TilingSpec(enabled=False, overlap=0.1))
    assert w.lbl_slice_overlap_minimum.text() == ""
    assert w.btn_slice_overlap_raise.isHidden()


def test_sam2_overlap_is_a_disabled_constant():
    w = SliceSettingsWidget(role="escalate_sam2")
    assert not w.spin_slice_overlap.isEnabled()
    assert w.spin_slice_overlap.value() == pytest.approx(0.5)
    assert w.lbl_slice_overlap_minimum.text() == "fixed"
    assert w.btn_slice_overlap_raise.isHidden()


def test_field_changed_only_on_user_edits():
    w = SliceSettingsWidget(role="infer_yolo")
    hits: list[str] = []
    w.field_changed.connect(hits.append)
    w.set_spec(TilingSpec(enabled=True, overlap=0.3))
    assert hits == []
    w.spin_slice_overlap.setValue(0.4)  # the user path (signals not blocked)
    assert hits == ["overlap"]


def test_escalation_full_frame_round_trip():
    w = SliceSettingsWidget(role="escalate_sam3")
    w.set_spec(TilingSpec(enabled=False, reference_body_px=50.0))
    assert w.spin_slice_object_fraction.value() == 0.0
    assert "full frame" in w.spin_slice_object_fraction.specialValueText()
    got = w.spec()
    assert got.enabled is False and got.object_tile_fractions == ()
    w.spin_slice_object_fraction.setValue(0.08)
    got = w.spec()
    assert got.enabled is True and got.object_tile_fractions == (0.08,)


def test_tile_label_formatter_is_the_hosts():
    fmt = lambda tile, body, frac: (  # noqa: E731
        (f"T{tile}", "tip") if tile else ("off", "")
    )
    w = SliceSettingsWidget(
        role="escalate_sam3",
        capabilities=SliceWidgetCapabilities(tile_label_formatter=fmt),
    )
    w.set_reference_body(82.2, "dataset")
    w.spin_slice_object_fraction.setValue(0.05)
    assert w.lbl_slice_tile_size.text() == "T1644"
    w.spin_slice_object_fraction.setValue(0.0)
    assert w.lbl_slice_tile_size.text() == "off"


def test_saved_nmm_shows_greedy_nmm_and_keeps_the_raw_value():
    """F6: nmm is the greedy_nmm code path; the cache hash needs the raw value."""
    w = SliceSettingsWidget(role="infer_yolo")
    w.set_spec(
        TilingSpec(enabled=True, merge_policy="greedy_nmm"),
        extras={"merge_policy_raw": "nmm"},
    )
    assert w.combo_slice_merge_policy.currentData() == "greedy_nmm"
    assert "NMM" in w.combo_slice_merge_policy.currentText()
    assert w.extras()["merge_policy_raw"] == "nmm"
    assert w.spec().merge_policy == "greedy_nmm"


def test_advanced_is_collapsed_by_default():
    w = SliceSettingsWidget(role="train_yolo")
    w.show()
    QApplication.processEvents()
    assert not w.btn_slice_advanced.isChecked()
    assert not w.spin_slice_min_area.isVisible()
    w.set_advanced_expanded(True)
    assert w.spin_slice_min_area.isVisible()
    w.hide()


def test_per_role_extras():
    expected = {
        "train_yolo": {
            "negative_tile_fraction",
            "full_frame_mix",
            "balance_multiscale_loss",
            "balance_multiscale_loss_power",
            "min_area_ratio",
        },
        "train_sam3": {
            "object_tile_fraction",
            "keep_empty_tiles",
            "full_frame_mix",
            "min_area_ratio",
            "tile_overlap",
        },
        "infer_yolo": {"merge_threshold"},
        "escalate_sam3": {"seam_margin_px", "merge_iou"},
        "escalate_sam2": set(),
    }
    for role, keys in expected.items():
        assert set(SliceSettingsWidget(role=role).extras()) == keys, role


def test_sam3_scalar_is_preserved_after_editing_the_set():
    w = SliceSettingsWidget(role="train_sam3")
    w.set_spec(
        TilingSpec(enabled=True, geometry_mode="auto_object"),
        extras={"object_tile_fraction": 0.055},
    )
    w.txt_slice_scales.setText("0.05, 0.2")
    assert w.extras()["object_tile_fraction"] == pytest.approx(0.055)
    assert w.spec().object_tile_fractions == (0.05, 0.2)


def test_px_display_uses_the_real_model_input():
    w = SliceSettingsWidget(role="infer_yolo")
    w.set_model_input_size(1024)
    w.spin_slice_object_fraction.setValue(0.1)
    assert "102.4" in w.lbl_slice_scale_px.text()


@pytest.mark.parametrize("role", ROLES)
def test_every_enabled_control_has_a_tooltip(role):
    w = SliceSettingsWidget(role=role)
    w.set_advanced_expanded(True)
    for cls in (QAbstractButton, QAbstractSpinBox, QComboBox, QLineEdit):
        for child in w.findChildren(cls):
            if isinstance(child.parent(), (QAbstractSpinBox, QComboBox)):
                continue  # internal line edits of spin boxes / combos
            assert child.toolTip().strip(), child.objectName() or type(child).__name__


def test_combo_items_have_tooltips():
    w = SliceSettingsWidget(role="infer_yolo")
    for combo in w.findChildren(QComboBox):
        for i in range(combo.count()):
            assert combo.itemData(i, Qt.ItemDataRole.ToolTipRole), combo.objectName()


def test_override_appears_only_for_a_derived_body():
    w = SliceSettingsWidget(role="escalate_sam3")
    w.set_reference_body(50.0, "user")
    assert w.chk_slice_body_override.isHidden()
    w.set_reference_body(50.0, "project")
    assert "project" in w.lbl_slice_body_badge.toolTip().lower()
    for source in ("project", "dataset", "stamped", "profile"):
        w.set_reference_body(50.0, source)
        assert not w.chk_slice_body_override.isHidden(), source
        assert w.source_badge("reference_body_px") == source
        assert not w.spin_slice_body.isEnabled()


def test_overlap_minimum_is_marked_capped_at_the_ceiling():
    """max scale >= 0.85: the true minimum is unreachable; say so."""
    w = SliceSettingsWidget(role="infer_yolo")
    w.set_spec(
        TilingSpec(
            enabled=True,
            geometry_mode="auto_object",
            object_tile_fractions=(0.88,),
            overlap=0.2,
        )
    )
    assert w.lbl_slice_overlap_minimum.text() == (
        "Below whole-animal minimum (0.90, capped)"
    )
    w.btn_slice_overlap_raise.click()
    assert w.spin_slice_overlap.value() == pytest.approx(0.9)
    assert w.lbl_slice_overlap_minimum.text() == (
        "≥ whole-animal minimum (0.90, capped)"
    )


def test_overlap_minimum_does_not_spam_warnings(caplog):
    import logging

    from hydra_suite.utils.tiling_spec import reset_warnings

    reset_warnings()
    caplog.set_level(logging.WARNING)
    w = SliceSettingsWidget(role="train_yolo")
    w.set_spec(TilingSpec(enabled=True, geometry_mode="auto_object", overlap=0.2))
    for _ in range(10):
        w.txt_slice_scales.setText("0.05, 0.95")  # above FRACTION_MAX
        w.refresh()
    assert len([r for r in caplog.records if r.levelno >= logging.WARNING]) <= 1


def test_overlap_warning_hidden_while_tiling_is_off():
    w = SliceSettingsWidget(role="train_yolo")
    w.set_spec(
        TilingSpec(
            enabled=False,
            geometry_mode="auto_object",
            object_tile_fractions=(0.05, 0.15),
            overlap=0.1,
        )
    )
    assert "Below" not in w.lbl_slice_overlap_minimum.text()
    assert w.btn_slice_overlap_raise.isHidden()
    w.chk_slice_enabled.setChecked(True)
    assert w.lbl_slice_overlap_minimum.text().startswith("Below")
    assert not w.btn_slice_overlap_raise.isHidden()
