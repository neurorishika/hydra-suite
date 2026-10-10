"""The shared SAHI settings widget (S4 Task 15, plan Revision 1 decisions 22-33)."""

from __future__ import annotations

import os
from dataclasses import replace

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


def test_disabling_tiling_hides_the_rest():
    """S6: SAHI off collapses the block to its Enable checkbox (was: the
    rows stayed visible but disabled)."""
    w = SliceSettingsWidget(role="infer_yolo")
    w.set_spec(TilingSpec(enabled=False))
    assert not w.combo_slice_geometry.isVisibleTo(w)
    assert not w.spin_slice_overlap.isVisibleTo(w)
    assert w.chk_slice_enabled.isVisibleTo(w)
    w.chk_slice_enabled.setChecked(True)
    assert w.combo_slice_geometry.isVisibleTo(w)
    assert w.combo_slice_geometry.isEnabled()
    assert w.spin_slice_overlap.isVisibleTo(w)
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
    w.chk_slice_enabled.setChecked(True)  # S6: SAHI off hides every row
    w.set_advanced_expanded(True)
    assert w.spin_slice_merge.isHidden()
    assert "merge_threshold" not in w.extras()
    assert not w.spin_slice_tile_batch.isHidden()
    default = SliceSettingsWidget(
        role="infer_yolo", capabilities=SliceWidgetCapabilities(advanced_merge=False)
    )
    default.chk_slice_enabled.setChecked(True)
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
    w.chk_slice_enabled.setChecked(True)  # S6: SAHI off hides every row
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


# ---------------------------------------------- S4b follow-ups (coordinator)


def _infer_widget(mode: str, *, body: float, tile=(0, 0), overlap=0.05):
    w = SliceSettingsWidget(role="infer_yolo")
    w.set_model_input_size(640)
    w.set_spec(
        TilingSpec(
            enabled=True,
            geometry_mode=mode,
            object_tile_fractions=(0.4,),  # irrelevant outside auto_object
            slice_width=tile[0],
            slice_height=tile[1],
            overlap=overlap,
        )
    )
    w.set_reference_body(body, "stamped")
    return w


def test_custom_minimum_uses_body_over_the_smaller_custom_side():
    """Custom tiles: the animal's share of a tile is body / min(W, H), not
    the (disabled, unused) object scale."""
    w = _infer_widget("custom", body=48.0, tile=(480, 400))
    # 48 / 400 = 0.12 (+0.05 margin) = 0.17 -- not 0.4 + 0.05.
    assert w.lbl_slice_overlap_minimum.text() == "Below whole-animal minimum (0.17)"
    assert w.btn_slice_overlap_raise.text() == "Raise to 0.17"
    w.spin_slice_tile_h.setValue(800)  # now min side 480: 0.1 + 0.05
    assert "(0.15)" in w.lbl_slice_overlap_minimum.text()


def test_custom_zero_side_means_model_input():
    w = _infer_widget("custom", body=64.0, tile=(0, 0))
    assert "(0.15)" in w.lbl_slice_overlap_minimum.text()  # 64/640 + 0.05


def test_custom_without_a_body_shows_no_minimum():
    w = _infer_widget("custom", body=0.0, tile=(480, 400))
    assert w.lbl_slice_overlap_minimum.text() == ""
    assert w.btn_slice_overlap_raise.isHidden()


def test_auto_model_minimum_uses_body_over_the_model_input():
    w = _infer_widget("auto_model", body=96.0)
    assert w.lbl_slice_overlap_minimum.text() == "Below whole-animal minimum (0.20)"
    w = _infer_widget("auto_model", body=0.0)
    assert w.lbl_slice_overlap_minimum.text() == ""
    assert w.btn_slice_overlap_raise.isHidden()


def test_auto_object_minimum_still_uses_the_scale():
    w = _infer_widget("auto_object", body=48.0)
    assert "(0.45)" in w.lbl_slice_overlap_minimum.text()


def test_custom_minimum_is_capped_for_a_tile_smaller_than_the_animal():
    w = _infer_widget("custom", body=500.0, tile=(400, 400))
    assert w.lbl_slice_overlap_minimum.text() == (
        "Below whole-animal minimum (0.90, capped)"
    )


def test_tile_size_badge_follows_the_host_source_until_a_user_edit():
    w = _infer_widget("custom", body=48.0, tile=(1024, 800))
    assert w.source_badge("tile_size") == "user"
    w.set_source("tile_size", "profile")
    assert w.source_badge("tile_size") == "profile"
    assert w.lbl_slice_tile_badge.text() == "profile"
    w.spin_slice_tile_w.setValue(900)  # the user path
    assert w.source_badge("tile_size") == "user"
    assert w.lbl_slice_tile_badge.text() == "user"
    w.set_source("tile_size", "stamped")
    w.combo_slice_geometry.setCurrentIndex(
        w.combo_slice_geometry.findData("auto_model")
    )
    assert w.source_badge("tile_size") == "derived"  # not custom: derived
    w.combo_slice_geometry.setCurrentIndex(w.combo_slice_geometry.findData("custom"))
    assert w.source_badge("tile_size") == "user"  # the mode change was an edit


def _badge_words_in_src() -> set[str]:
    """String literals passed as a badge source anywhere in src."""
    import ast
    from pathlib import Path

    import hydra_suite

    words: set[str] = set()
    root = Path(hydra_suite.__file__).parent
    for path in root.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            name = getattr(node.func, "attr", getattr(node.func, "id", ""))
            if name in ("set_reference_body", "set_source", "set_badge"):
                if len(node.args) >= 2 and isinstance(node.args[1], ast.Constant):
                    words.add(node.args[1].value)
            for kw in node.keywords:
                if kw.arg == "body_px_source" and isinstance(kw.value, ast.Constant):
                    if isinstance(kw.value.value, str) and kw.value.value:
                        words.add(kw.value.value)
    return words


def test_every_badge_word_in_use_has_a_description():
    from hydra_suite.trackerkit.gui.panels.detection_panel import (
        SLICE_SOURCE_BY_RESOLUTION,
    )
    from hydra_suite.widgets.slice_settings_parts import SOURCE_DESCRIPTIONS

    used = _badge_words_in_src() | set(SLICE_SOURCE_BY_RESOLUTION.values())
    # Sources reached through variables: tiling_resolve.resolve_body_px's
    # chain, the SAM3 dialog's _body_source(), TrackerKit's no-sidecar body.
    used |= {"user", "override", "dataset", "stamped", "default", "project"}
    used |= {"profile", "derived", "config"}
    missing = sorted(word for word in used if word not in SOURCE_DESCRIPTIONS)
    assert not missing, missing
    for word, text in SOURCE_DESCRIPTIONS.items():
        assert len(text.split()) >= 3, word


def test_preview_labels_the_fallback_frame_clearly():
    w = SliceSettingsWidget(role="infer_yolo")
    title, *_rest = w.preview.caption_texts()
    assert "example" in title and "1920" in title
    w.set_preview_frame_size((2048, 1536))
    title, tile_line, *_rest = w.preview.caption_texts()
    assert title.startswith("Tile layout on a 2048 × 1536 image")
    assert "example" not in title and "example" not in tile_line


def test_preview_caption_elides_at_word_boundaries():
    from PySide6.QtGui import QFont, QFontMetrics

    from hydra_suite.widgets.tile_layout_preview import elide_at_word

    metrics = QFontMetrics(QFont())
    text = "15 tiles · 480 × 480 px · example frame uses the body size"
    full = elide_at_word(text, metrics, 10_000)
    assert full == text
    for width in (60, 120, 200, 260):
        out = elide_at_word(text, metrics, width)
        assert metrics.horizontalAdvance(out) <= width
        if out != text:
            assert out.endswith("…")
            stem = out[:-1].rstrip()
            # Only whole words survive: the stem is a word-prefix of the text.
            assert text.startswith(stem)
            assert stem == "" or text[len(stem)] == " "


def test_profile_row_hides_while_tiling_is_off_but_stays_programmable():
    """S6 (was review MAJOR-1's "picker stays enabled while off"): the
    profile row hides with the rest while SAHI is off -- the user path is
    now "tick Enable, then pick" -- but the combo itself stays enabled, so a
    host (or a restore) can still drive it programmatically."""
    w = SliceSettingsWidget(role="infer_yolo")
    w.combo_slice_profile.addItems(["Training geometry", "Fast scan"])
    w.set_profile_row_visible(True)
    w.chk_slice_enabled.setChecked(False)
    assert not w.combo_slice_profile.isVisibleTo(w)
    assert w.combo_slice_profile.isEnabled()
    seen = []
    w.combo_slice_profile.currentIndexChanged.connect(seen.append)
    w.combo_slice_profile.setCurrentIndex(1)
    assert seen == [1]
    w.chk_slice_enabled.setChecked(True)
    assert w.combo_slice_profile.isVisibleTo(w)
    assert w.combo_slice_profile.currentIndex() == 1


def test_measured_overlap_below_the_minimum_is_info_not_a_warning():
    """Review MINOR-2 (decision 22/22a): an overlap a profile or the stamp
    set is never nudged -- muted info naming its source, no Raise button."""
    w = _infer_widget("custom", body=48.0, tile=(1024, 800), overlap=0.1)
    w.set_source("overlap", "profile", note="profile 'Fast scan'")
    label = w.lbl_slice_overlap_minimum
    assert label.text() == (
        "below whole-animal minimum (0.11) — set by profile 'Fast scan'"
    )
    assert "e0943a" not in label.styleSheet()
    assert w.btn_slice_overlap_raise.isHidden()
    # A user edit makes it the user's value again: warning + Raise.
    w.spin_slice_overlap.setValue(0.05)
    assert label.text() == "Below whole-animal minimum (0.11)"
    assert not w.btn_slice_overlap_raise.isHidden()
    assert w.source_badge("overlap") == "user"


def test_stamped_overlap_note_defaults_to_the_source_description():
    w = _infer_widget("custom", body=48.0, tile=(1024, 800), overlap=0.1)
    w.set_source("overlap", "stamped")
    assert w.lbl_slice_overlap_minimum.text().endswith("— set by stamped on the model")
    assert w.btn_slice_overlap_raise.isHidden()


def test_unknown_body_shows_no_source_badge():
    """Review MINOR-3: a 0 (unknown) body has no source to claim."""
    w = SliceSettingsWidget(
        role="infer_yolo",
        capabilities=SliceWidgetCapabilities(
            body_override=False, body_display_only=True
        ),
    )
    w.set_reference_body(0.0, "default")
    assert w.lbl_slice_body_badge.text() == ""
    w.set_reference_body(48.0, "stamped")
    assert w.lbl_slice_body_badge.text() == "stamped"
    typable = SliceSettingsWidget(role="escalate_sam3")
    typable.set_reference_body(0.0, "dataset")
    assert typable.lbl_slice_body_badge.text() == ""
    typable.spin_slice_body.setValue(40.0)
    assert typable.lbl_slice_body_badge.text() == "user"


def test_widget_owns_no_profile_status_label():
    """Review NIT: hosts own the profile status prose (TrackerKit's row)."""
    w = SliceSettingsWidget(role="infer_yolo")
    assert not hasattr(w, "lbl_slice_profile_status")


def test_profile_overlap_claim_cleared_by_geometry_or_scale_edits():
    """S4b re-check MINOR-A: a measured overlap describes its own geometry."""
    for edit in ("geometry", "tile", "fraction"):
        w = _infer_widget("custom", body=48.0, tile=(1024, 800), overlap=0.1)
        w.set_source("overlap", "profile", note="profile 'Fast scan'")
        assert w.btn_slice_overlap_raise.isHidden()
        if edit == "geometry":
            w.combo_slice_geometry.setCurrentIndex(
                w.combo_slice_geometry.findData("auto_object")
            )
        elif edit == "tile":
            w.spin_slice_tile_w.setValue(512)
        else:
            w.combo_slice_geometry.setCurrentIndex(
                w.combo_slice_geometry.findData("auto_object")
            )
            w.set_source("overlap", "profile", note="profile 'Fast scan'")
            w.spin_slice_object_fraction.setValue(0.33)
        assert w.source_badge("overlap") == "user", edit
        assert "set by profile" not in w.lbl_slice_overlap_minimum.text(), edit


# ------------------------------------------------- S6: preview position


def _settle(widget) -> None:
    """Drop every cached layout minimum, then re-measure leaf-first."""
    from PySide6.QtWidgets import QLayout

    app = QApplication.instance()
    for _ in range(3):
        app.processEvents()
    layouts = widget.findChildren(QLayout)
    for layout in layouts:
        layout.invalidate()
    for layout in reversed(layouts):
        layout.activate()
    if widget.layout() is not None:
        widget.layout().invalidate()
        widget.layout().activate()


def _bottom_widget(**caps):
    return SliceSettingsWidget(
        role="infer_yolo",
        capabilities=SliceWidgetCapabilities(preview_position="bottom", **caps),
    )


def test_preview_position_defaults_to_side():
    w = SliceSettingsWidget(role="infer_yolo")
    assert w.capabilities.preview_position == "side"
    w.set_spec(TilingSpec(enabled=True))
    w.resize(900, 500)
    w.show()
    _settle(w)
    assert w.preview.isVisibleTo(w)
    assert w.preview.geometry().left() > w._controls.geometry().right()
    w.hide()


def test_unknown_preview_position_is_rejected():
    with pytest.raises(ValueError):
        SliceSettingsWidget(
            role="infer_yolo",
            capabilities=SliceWidgetCapabilities(preview_position="left"),
        )


def test_bottom_preview_sits_below_the_controls_and_spans_them():
    w = _bottom_widget()
    w.set_spec(TilingSpec(enabled=True))
    w.resize(520, 900)
    w.show()
    _settle(w)
    assert w.preview.isVisibleTo(w)
    assert w.preview.geometry().top() > w._controls.geometry().bottom()
    # Scales to the available width (not a fixed 290 px box).
    assert w.preview.width() >= w.width() - 40
    w.hide()


def test_bottom_preview_height_follows_width_and_frame_aspect():
    w = _bottom_widget()
    preview = w.preview
    assert preview.hasHeightForWidth()
    preview.set_frame_size((2000, 1000))
    narrow, wide = preview.heightForWidth(300), preview.heightForWidth(500)
    assert narrow < wide
    preview.set_frame_size((1000, 2000))  # portrait: taller, but capped
    assert preview.heightForWidth(500) > wide
    assert preview.heightForWidth(5000) <= preview.MAX_BOTTOM_HEIGHT
    assert preview.heightForWidth(10) >= preview.MIN_BOTTOM_HEIGHT


def test_side_preview_keeps_its_fixed_box():
    w = SliceSettingsWidget(role="train_yolo")
    assert not w.preview.hasHeightForWidth()
    assert w.preview.minimumWidth() == 290


def test_bottom_layout_is_much_narrower_than_side_in_every_mode():
    """The SAHI block must not force its host wide: the bottom layout's
    minimum width stays compact in every geometry mode with Advanced open and
    the longest derived texts showing (a profile-set overlap below the
    minimum, a derived tile size)."""
    side = SliceSettingsWidget(
        role="infer_yolo",
        capabilities=SliceWidgetCapabilities(execution_knobs=True),
    )
    bottom = _bottom_widget(execution_knobs=True)
    for w in (side, bottom):
        w.set_model_input_size(1024)
        w.set_advanced_expanded(True)
    widths = {}
    for mode in ("auto_model", "auto_object", "custom"):
        for w, name in ((side, "side"), (bottom, "bottom")):
            w.set_spec(
                TilingSpec(
                    enabled=True,
                    geometry_mode=mode,
                    object_tile_fractions=(0.1,),
                    slice_width=1024,
                    slice_height=800,
                    overlap=0.05,
                )
            )
            w.set_reference_body(48.0, "stamped")
            w.set_source("overlap", "profile", note="profile 'Balanced scan'")
            w.show()
            _settle(w)
            widths[(name, mode)] = w.minimumSizeHint().width()
            w.hide()
        assert widths[("bottom", mode)] <= 460, widths
        assert widths[("bottom", mode)] < widths[("side", mode)] - 250, widths


def test_bottom_derived_labels_keep_full_text_and_elide_when_narrow():
    w = _bottom_widget()
    w.set_spec(
        TilingSpec(
            enabled=True,
            geometry_mode="custom",
            slice_width=1024,
            slice_height=800,
            overlap=0.05,
        )
    )
    w.set_reference_body(48.0, "stamped")
    w.set_source("overlap", "profile", note="profile 'Balanced scan'")
    label = w.lbl_slice_overlap_minimum
    assert label.text().startswith("below whole-animal minimum")
    assert "profile 'Balanced scan'" in label.text()
    # A short minimum: the full sentence never sets the host's width.
    assert label.minimumSizeHint().width() < 120


# ------------------------------------------------- S6: hide when off


def _shown_rows(w) -> set[str]:
    return {key for key, widgets in w._row_widgets.items() if not widgets[0].isHidden()}


@pytest.mark.parametrize("role", ["infer_yolo", "train_yolo"])
@pytest.mark.parametrize("position", ["side", "bottom"])
def test_sahi_off_leaves_only_the_enable_checkbox(role, position):
    w = SliceSettingsWidget(
        role=role,
        capabilities=SliceWidgetCapabilities(
            preview_position=position, execution_knobs=True, full_frame_pass=True
        ),
    )
    w.set_profile_row_visible(True)
    w.set_advanced_expanded(True)
    w.set_spec(TilingSpec(enabled=True, geometry_mode="auto_object"))
    on_rows = _shown_rows(w)
    assert {"enabled", "overlap", "advanced"} <= on_rows
    assert not w.preview.isHidden()
    w.chk_slice_enabled.setChecked(False)  # the user path
    assert _shown_rows(w) == {"enabled"}
    assert w.preview.isHidden()
    assert w.btn_slice_advanced.isHidden()
    w.chk_slice_enabled.setChecked(True)
    assert _shown_rows(w) == on_rows
    assert not w.preview.isHidden()
    assert w.btn_slice_advanced.isChecked()  # Advanced stays expanded


def test_sahi_off_via_set_spec_also_collapses():
    w = SliceSettingsWidget(role="infer_yolo")
    w.set_spec(TilingSpec(enabled=False))
    assert _shown_rows(w) == {"enabled"}
    assert w.preview.isHidden()


@pytest.mark.parametrize("role", ["train_sam3", "escalate_sam3", "escalate_sam2"])
def test_roles_without_the_checkbox_never_collapse(role):
    w = SliceSettingsWidget(role=role)
    rows = _shown_rows(w)
    assert "enabled" not in rows
    assert {"tile", "overlap"} <= rows


@pytest.mark.parametrize("role", ["infer_yolo", "train_yolo"])
def test_hide_show_round_trip_preserves_every_value_and_emits_only_enabled(role):
    w = SliceSettingsWidget(
        role=role,
        capabilities=SliceWidgetCapabilities(
            preview_position="bottom", execution_knobs=True, full_frame_pass=True
        ),
    )
    w.set_model_input_size(1024)
    spec = TilingSpec(
        enabled=True,
        geometry_mode="custom",
        object_tile_fractions=(0.1, 0.2) if role == "train_yolo" else (0.1,),
        reference_body_px=48.0,
        slice_width=1024,
        slice_height=800,
        overlap=0.3,
        merge_threshold=0.45,
    )
    w.set_spec(spec, extras={"tile_batch_size": 7, "memory_budget_mib": 99})
    w.set_advanced_expanded(True)
    before_spec, before_extras = w.spec(), w.extras()
    emitted = []
    w.field_changed.connect(emitted.append)
    w.chk_slice_enabled.setChecked(False)
    off_spec, off_extras = w.spec(), w.extras()
    assert off_spec == replace(before_spec, enabled=False)
    assert off_extras == before_extras
    w.chk_slice_enabled.setChecked(True)
    assert emitted == ["enabled", "enabled"]
    assert w.spec() == before_spec
    assert w.extras() == before_extras


# ------------------------------------------- S7: compact (paired) layout


def _compact_widget(**caps):
    caps.setdefault("execution_knobs", True)
    return SliceSettingsWidget(
        role="infer_yolo",
        capabilities=SliceWidgetCapabilities(
            preview_position="bottom", layout="compact", **caps
        ),
    )


def _grid_cell(w, widget):
    """(row, column, row span, column span) of ``widget`` in the controls grid."""
    return w._grid.getItemPosition(w._grid.indexOf(widget))


def _row_label(w, key):
    return w._rows[key][0]


def test_layout_defaults_to_rows_and_rejects_unknown():
    assert SliceSettingsWidget(role="infer_yolo").capabilities.layout == "rows"
    with pytest.raises(ValueError):
        SliceSettingsWidget(
            role="infer_yolo",
            capabilities=SliceWidgetCapabilities(layout="grid"),
        )


def test_compact_layout_pairs_controls_on_one_grid_row():
    w = _compact_widget()
    w.set_profile_row_visible(True)
    w.set_advanced_expanded(True)
    w.set_spec(TilingSpec(enabled=True, geometry_mode="auto_object"))
    for left, right in (
        ("profile", "mode"),
        ("object_fraction", "body"),
        ("tile", "overlap"),
        ("tile_batch", "memory"),
    ):
        left_row, left_col, *_ = _grid_cell(w, _row_label(w, left))
        right_row, right_col, *_ = _grid_cell(w, _row_label(w, right))
        assert left_row == right_row, (left, right)
        assert left_col == 0 and right_col == 2, (left, right)
    # Distinct pairs sit on distinct rows, in reading order.
    rows = [
        _grid_cell(w, _row_label(w, key))[0]
        for key in ("profile", "object_fraction", "tile", "tile_batch")
    ]
    assert rows == sorted(set(rows))


def test_compact_layout_moves_a_lone_partner_to_the_left():
    """No profile row: the tile strategy stands alone at the left edge."""
    w = _compact_widget()
    w.set_spec(TilingSpec(enabled=True, geometry_mode="auto_object"))
    w.set_profile_row_visible(False)
    row, col, _rs, _cs = _grid_cell(w, _row_label(w, "mode"))
    assert col == 0
    assert _grid_cell(w, w.combo_slice_geometry)[1] == 1
    assert _grid_cell(w, _row_label(w, "object_fraction"))[0] > row
    w.set_profile_row_visible(True)
    assert _grid_cell(w, _row_label(w, "mode"))[1] == 2
    assert _grid_cell(w, _row_label(w, "profile"))[1] == 0


def test_compact_layout_keeps_constrained_fields_visible_but_disabled():
    w = _compact_widget()
    w.set_reference_body(48.0, "stamped")
    w.set_spec(TilingSpec(enabled=True, geometry_mode="auto_model"))
    for control in (w.spin_slice_object_fraction, w.spin_slice_tile_w):
        assert not control.isHidden() and control.isVisibleTo(w)
        assert not control.isEnabled()
    w.chk_slice_enabled.setChecked(False)
    assert _shown_rows(w) == {"enabled"}
    assert not w.lbl_slice_summary.isVisibleTo(w)


def _summary(w) -> str:
    return w.lbl_slice_summary.text()


def test_compact_summary_line_per_mode():
    w = _compact_widget()
    w.set_model_input_size(1024)
    w.set_reference_body(48.0, "stamped")
    w.set_spec(
        TilingSpec(
            enabled=True,
            geometry_mode="auto_object",
            object_tile_fractions=(0.1,),
            reference_body_px=48.0,
            overlap=0.2,
        )
    )
    w.set_reference_body(48.0, "stamped")
    text = _summary(w)
    assert text.startswith("→ 480 × 480 px")
    assert "≈102 px at 1024" in text
    # The overlap note is its own label on the same line (its colour kept).
    assert w.lbl_slice_overlap_minimum.text() == "≥ whole-animal minimum (0.15)"
    assert _grid_cell(w, w._summary_row)[3] == 4
    assert w.lbl_slice_overlap_minimum.parentWidget() is w._summary_row
    assert w.btn_slice_overlap_raise.parentWidget() is w._summary_row
    full = w.lbl_slice_summary.toolTip()
    assert "480 × 480 px" in full and "whole-animal minimum" in full

    w.combo_slice_geometry.setCurrentIndex(w.combo_slice_geometry.findData("custom"))
    w.spin_slice_tile_w.setValue(1024)
    w.spin_slice_tile_h.setValue(800)
    assert _summary(w).startswith("→ 1024 × 800 px")
    assert "px at 1024" not in _summary(w)  # object scale unused

    w.combo_slice_geometry.setCurrentIndex(
        w.combo_slice_geometry.findData("auto_model")
    )
    assert _summary(w).startswith("→ 1024 × 1024 px (model input)")


def test_compact_below_minimum_warning_keeps_colour_and_raise_on_the_line():
    w = _compact_widget()
    w.set_model_input_size(1024)
    w.set_spec(
        TilingSpec(
            enabled=True,
            geometry_mode="auto_object",
            object_tile_fractions=(0.1,),
            overlap=0.05,
        )
    )
    w.set_reference_body(48.0, "stamped")
    label = w.lbl_slice_overlap_minimum
    assert label.text().startswith("Below whole-animal minimum")
    assert "#e0943a" in label.styleSheet()
    assert not w.btn_slice_overlap_raise.isHidden()
    # The warning never elides: the muted summary gives way first.
    assert label.minimumWidth() >= label.fontMetrics().horizontalAdvance(label.text())
    # A profile-set overlap stays muted info, no Raise.
    w.set_source("overlap", "profile", note="profile 'Fast scan'")
    assert "set by profile 'Fast scan'" in label.text()
    assert "#8f969e" in label.styleSheet()
    assert w.btn_slice_overlap_raise.isHidden()


def test_compact_badges_stay_with_their_fields():
    w = _compact_widget()
    w.set_reference_body(48.0, "profile")
    w.set_spec(TilingSpec(enabled=True, geometry_mode="auto_object"))
    w.set_reference_body(48.0, "profile")
    assert w.source_badge("reference_body_px") == "profile"
    assert w.lbl_slice_body_badge.text() == "profile"
    assert w.lbl_slice_tile_badge.text() == "derived"
    body_row = _grid_cell(w, _row_label(w, "body"))[0]
    tile_row = _grid_cell(w, _row_label(w, "tile"))[0]
    assert w.lbl_slice_body_badge.parentWidget() is w._rows["body"][1]
    assert w.lbl_slice_tile_badge.parentWidget() is w._rows["tile"][1]
    assert body_row != tile_row


def test_compact_advanced_note_sits_under_the_advanced_pair():
    from PySide6.QtWidgets import QLabel

    w = _compact_widget()
    note = QLabel("Up to 16 tiles/call")
    w.set_advanced_note(note)
    w.set_spec(TilingSpec(enabled=True))
    w.set_advanced_expanded(True)
    note_row, note_col, _rs, note_span = _grid_cell(w, note)
    assert note_row > _grid_cell(w, _row_label(w, "tile_batch"))[0]
    assert (note_col, note_span) == (0, 4)


def test_compact_values_and_signals_match_the_rows_layout():
    spec = TilingSpec(
        enabled=True,
        geometry_mode="custom",
        object_tile_fractions=(0.1,),
        reference_body_px=48.0,
        slice_width=1024,
        slice_height=800,
        overlap=0.3,
    )
    extras = {"tile_batch_size": 7, "memory_budget_mib": 99}
    rows = _bottom_widget(execution_knobs=True)
    compact = _compact_widget()
    for w in (rows, compact):
        w.set_spec(spec, extras=extras)
    assert compact.spec() == rows.spec()
    assert compact.extras() == rows.extras()
    emitted = []
    compact.field_changed.connect(emitted.append)
    compact.spin_slice_overlap.setValue(0.4)
    compact.spin_slice_memory_budget.setValue(64)
    assert emitted == ["overlap", "memory_budget_mib"]


@pytest.mark.parametrize("mode", ["auto_model", "auto_object", "custom"])
def test_compact_preview_is_shorter_with_two_caption_lines(mode):
    w = _compact_widget()
    w.set_spec(TilingSpec(enabled=True, geometry_mode=mode))
    preview = w.preview
    preview.set_frame_size((2448, 2048))
    assert preview.heightForWidth(480) <= 260
    assert len(preview.caption_lines()) <= 2
    # The rows layout keeps its three lines (body note on its own line).
    rows = _bottom_widget()
    rows.set_spec(TilingSpec(enabled=True, geometry_mode="auto_object"))
    rows.preview.set_frame_size((2448, 2048))
    assert len(rows.preview.caption_lines()) == 3
