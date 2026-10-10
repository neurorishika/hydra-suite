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
    combo = panel.combo_slice_geometry
    # The enum is the item data (what config.py persists); the text is the
    # shared widget's human label.
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
    assert combo.currentData() == "auto_model"
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


def test_slice_geometry_disabled_while_sahi_off_in_direct_mode(monkeypatch):
    """SAHI inputs are pointless while sliced inference is off: the geometry
    picker is DISABLED (shared widget: constrained = disabled, never hidden)
    until the SAHI checkbox is checked."""
    window = _make_main_window(monkeypatch)
    panel = window._detection_panel
    assert panel.combo_yolo_obb_mode.currentIndex() == 0  # Direct
    panel.chk_slice_enabled.setChecked(False)
    panel._on_yolo_mode_changed(0)
    assert panel.combo_slice_geometry.isVisibleTo(panel) is True
    assert panel.combo_slice_geometry.isEnabled() is False
    # The checkbox itself stays usable so the user can turn SAHI on.
    assert panel.chk_slice_enabled.isVisibleTo(panel) is True
    assert panel.chk_slice_enabled.isEnabled() is True

    panel.chk_slice_enabled.setChecked(True)
    assert panel.combo_slice_geometry.isVisibleTo(panel) is True
    assert panel.combo_slice_geometry.isEnabled() is True
    window.close()


def test_sequential_model_rows_hidden_in_direct_mode(monkeypatch):
    """Sequential selectors (and their row labels) must not be visible when the
    mode is Direct — previously only the combos were hidden, leaving orphaned
    'Seq detect model'/'Seq crop OBB model' labels in the form."""
    window = _make_main_window(monkeypatch)
    panel = window._detection_panel
    assert panel.combo_yolo_obb_mode.currentIndex() == 0  # Direct
    panel._on_yolo_mode_changed(0)
    assert panel.row_seq_detect.isHidden() is True
    assert panel.row_seq_crop.isHidden() is True
    assert panel.seq_detect_model_row_widget.isVisibleTo(panel) is False
    assert panel.seq_crop_obb_model_row_widget.isVisibleTo(panel) is False
    assert panel.yolo_seq_advanced.isVisibleTo(panel) is False
    assert panel.row_direct_model.isVisibleTo(panel) is True
    window.close()


def test_direct_task_combo_is_hidden_state_holder(monkeypatch):
    """The direct-model task is no longer a user-facing control: the combo is
    hidden and the read-only label reflects it."""
    window = _make_main_window(monkeypatch)
    panel = window._detection_panel
    assert panel.combo_yolo_direct_task.isHidden() is True
    assert panel.lbl_direct_task_inferred.text() == "OBB (native)"
    assert panel.spin_yolo_fixed_angle.isHidden() is True
    # Programmatic task changes (e.g. config load) still drive the label.
    panel.combo_yolo_direct_task.setCurrentIndex(1)  # Detect
    assert panel.lbl_direct_task_inferred.text() == "Detect (fixed angle)"
    assert panel.spin_yolo_fixed_angle.isHidden() is False
    window.close()


def _seed_direct_model(monkeypatch, tmp_path, *, task=None) -> None:
    """Seed a stub direct-OBB model (optionally with a registry-recorded task)."""
    data_dir = tmp_path / "hydra-data"
    monkeypatch.setenv("HYDRA_DATA_DIR", str(data_dir))
    models_root = data_dir / "models"
    obb_dir = models_root / "obb"
    obb_dir.mkdir(parents=True, exist_ok=True)
    (obb_dir / "direct_stub.pt").write_text("stub model", encoding="utf-8")
    entry = {
        "task_family": "obb",
        "usage_role": "obb_direct",
        "size": "26s",
        "species": "ant",
        "model_info": "direct_stub",
    }
    if task:
        entry["task"] = task
    registry = {
        "schema_version": 2,
        "entries": {"obb/direct_stub.pt": entry},
    }
    (models_root / "model_registry.json").write_text(
        __import__("json").dumps(registry), encoding="utf-8"
    )


def test_direct_task_inferred_from_registry(monkeypatch, tmp_path):
    """The direct model task is auto-inferred from the checkpoint/registry — a
    registry-recorded 'detect' task must select Detect (fixed angle) and reveal
    the fixed-angle input without any user input."""
    _seed_direct_model(monkeypatch, tmp_path, task="detect")
    window = _make_main_window(monkeypatch)
    panel = window._detection_panel
    # No event loop runs in tests, so the deferred kick hasn't fired yet.
    assert panel._task_kick_scheduled is True
    panel._run_scheduled_task_inference()
    assert panel.combo_yolo_direct_task.currentIndex() == 1
    assert panel.lbl_direct_task_inferred.text() == "Detect (fixed angle)"
    assert panel.spin_yolo_fixed_angle.isHidden() is False
    window.close()


def test_direct_task_inference_defers_worker_until_event_loop(
    monkeypatch,
    tmp_path,
):
    """The checkpoint-task read must not spawn a thread during construction
    (tests / startup): the kick is deferred to the event loop. An unreadable
    stub leaves the default task untouched."""
    _seed_direct_model(monkeypatch, tmp_path)
    window = _make_main_window(monkeypatch)
    panel = window._detection_panel
    assert panel._task_worker is None
    assert panel._task_kick_scheduled is True
    panel._run_scheduled_task_inference()
    assert panel._task_kick_scheduled is False
    assert panel.combo_yolo_direct_task.currentIndex() == 0
    assert panel.lbl_direct_task_inferred.text() == "OBB (native)"
    window.close()


def _select_geometry(panel, mode: str) -> None:
    combo = panel.combo_slice_geometry
    combo.setCurrentIndex(combo.findData(mode))
    assert combo.currentData() == mode


def test_slice_params_controls_follow_geometry_mode(monkeypatch):
    """custom mode enables tile W/H; auto_object enables the object fraction;
    auto_model enables neither (only the shared tile overlap). Fields the mode
    does not use are disabled, not hidden."""
    window = _make_main_window(monkeypatch)
    panel = window._detection_panel
    panel.chk_slice_enabled.setChecked(True)

    _select_geometry(panel, "custom")
    assert panel.spin_slice_tile_w.isEnabled() is True
    assert panel.spin_slice_tile_h.isEnabled() is True
    assert panel.spin_slice_object_fraction.isEnabled() is False
    assert panel.spin_slice_overlap.isEnabled() is True

    _select_geometry(panel, "auto_object")
    assert panel.spin_slice_tile_w.isEnabled() is False
    assert panel.spin_slice_tile_h.isEnabled() is False
    assert panel.spin_slice_object_fraction.isEnabled() is True

    _select_geometry(panel, "auto_model")
    assert panel.spin_slice_tile_w.isEnabled() is False
    assert panel.spin_slice_tile_h.isEnabled() is False
    assert panel.spin_slice_object_fraction.isEnabled() is False
    assert panel.spin_slice_overlap.isEnabled() is True
    for spin in (
        panel.spin_slice_tile_w,
        panel.spin_slice_tile_h,
        panel.spin_slice_object_fraction,
        panel.spin_slice_overlap,
    ):
        assert spin.isHidden() is False
    window.close()


def test_slice_params_disabled_while_sahi_off(monkeypatch):
    window = _make_main_window(monkeypatch)
    panel = window._detection_panel
    assert panel.chk_slice_enabled.isChecked() is False
    for control in (
        panel.combo_slice_geometry,
        panel.spin_slice_overlap,
        panel.spin_slice_tile_batch,
        panel.spin_slice_memory_budget,
    ):
        assert control.isEnabled() is False
    panel.chk_slice_enabled.setChecked(True)
    assert panel.spin_slice_overlap.isEnabled() is True
    assert panel.spin_slice_tile_batch.isEnabled() is True
    window.close()


def test_slice_params_sync_advanced_config(monkeypatch):
    """The SAHI parameter spins read from and write back to advanced_config
    (the keys engine_params.py consumes at run time)."""
    window = _make_main_window(monkeypatch, advanced_config={"slice_overlap": 0.3})
    panel = window._detection_panel
    assert panel.spin_slice_overlap.value() == pytest.approx(0.3)
    assert panel.spin_slice_tile_w.value() == 0

    panel.spin_slice_tile_w.setValue(640)
    panel.spin_slice_tile_h.setValue(480)
    panel.spin_slice_overlap.setValue(0.25)
    panel.spin_slice_object_fraction.setValue(0.2)
    adv = window.advanced_config
    assert adv["slice_width"] == 640
    assert adv["slice_height"] == 480
    assert adv["slice_overlap"] == pytest.approx(0.25)
    assert adv["slice_object_tile_fraction"] == pytest.approx(0.2)
    window.close()


def test_slice_tile_execution_controls_sync_and_explain_admission(monkeypatch):
    """Requested tile chunks persist separately from geometry and are honest
    about runtime memory admission rather than promising a speedup."""
    window = _make_main_window(
        monkeypatch,
        advanced_config={"slice_tile_batch_size": 12, "slice_memory_budget_mib": 96},
    )
    panel = window._detection_panel
    panel.chk_slice_enabled.setChecked(True)

    assert panel.spin_slice_tile_batch.value() == 12
    assert panel.spin_slice_memory_budget.value() == 96
    assert "admitted" in panel.lbl_slice_batch_admission.text().lower()
    assert "not always faster" in panel.spin_slice_tile_batch.toolTip()

    panel.spin_slice_tile_batch.setValue(7)
    panel.spin_slice_memory_budget.setValue(48)
    assert window.advanced_config["slice_tile_batch_size"] == 7
    assert window.advanced_config["slice_memory_budget_mib"] == 48
    assert "7 tiles/call" in panel.lbl_slice_batch_admission.text()
    assert "48 MiB" in panel.lbl_slice_batch_admission.text()
    # The retired "Auto" tile-batch checkbox is gone: Tiles / call is always
    # an explicit requested maximum, and the label always says so.
    assert not hasattr(panel, "chk_slice_tile_batch_autotune")
    assert panel.spin_slice_tile_batch.isEnabled() is True
    window.close()


def _seed_seq_crop_model(monkeypatch, tmp_path, *, name, training_params=None) -> str:
    """Seed a stub sequential crop-OBB model; returns its registry key."""
    data_dir = tmp_path / "hydra-data"
    monkeypatch.setenv("HYDRA_DATA_DIR", str(data_dir))
    models_root = data_dir / "models"
    crop_dir = models_root / "obb" / "cropped"
    crop_dir.mkdir(parents=True, exist_ok=True)
    (crop_dir / f"{name}.pt").write_text("stub model", encoding="utf-8")
    entry = {
        "task_family": "obb",
        "usage_role": "seq_crop_obb",
        "size": "26s",
        "species": "ant",
        "model_info": name,
    }
    if training_params:
        entry["training_params"] = dict(training_params)
    registry = {
        "schema_version": 2,
        "entries": {f"obb/cropped/{name}.pt": entry},
    }
    (models_root / "model_registry.json").write_text(
        __import__("json").dumps(registry), encoding="utf-8"
    )
    return f"obb/cropped/{name}.pt"


def test_seq_crop_imgsz_autoset_from_checkpoint_fallback(
    monkeypatch,
    tmp_path,
):
    """A crop-OBB model without training metadata gets its stage-2 imgsz from
    the checkpoint read; the value is cached into the registry."""
    key = _seed_seq_crop_model(monkeypatch, tmp_path, name="crop_nometa")
    window = _make_main_window(monkeypatch)
    panel = window._detection_panel
    assert panel.spin_yolo_seq_stage2_imgsz.value() == 160  # default untouched

    panel._apply_seq_crop_imgsz(key, 256)
    assert panel.spin_yolo_seq_stage2_imgsz.value() == 256

    # The registry now carries training_params.imgsz so future selections are
    # instant and the checkpoint fallback defers to it.
    from hydra_suite.core.inference.model_paths import get_yolo_model_metadata

    meta = get_yolo_model_metadata(key) or {}
    assert meta["training_params"]["imgsz"] == 256
    window.close()


def test_seq_crop_imgsz_defers_to_training_recorded_value(
    monkeypatch,
    tmp_path,
):
    """A DetectKit-published training_params.imgsz is authoritative: the
    checkpoint read must not clobber it."""
    from PySide6.QtCore import Qt

    key = _seed_seq_crop_model(
        monkeypatch,
        tmp_path,
        name="crop_trained",
        training_params={"imgsz": 128},
    )
    window = _make_main_window(monkeypatch)
    panel = window._detection_panel
    # Selecting the model applies its training defaults (stage-2 imgsz 128).
    combo = panel.combo_yolo_crop_obb_model
    idx = combo.findData(key, Qt.UserRole)
    assert idx >= 0
    combo.setCurrentIndex(idx)
    assert panel.spin_yolo_seq_stage2_imgsz.value() == 128
    # The checkpoint fallback defers to the training-recorded value.
    panel._apply_seq_crop_imgsz(key, 512)
    assert panel.spin_yolo_seq_stage2_imgsz.value() == 128
    window.close()


def test_seq_advanced_compact_two_column_grid(monkeypatch):
    """The sequential advanced settings use a compact 2-column grid, not a
    tall single-column form."""
    from PySide6.QtWidgets import QGridLayout

    window = _make_main_window(monkeypatch)
    panel = window._detection_panel
    layout = panel.yolo_seq_advanced.findChild(QGridLayout)
    assert isinstance(layout, QGridLayout)
    assert layout is not None and layout.columnCount() >= 4  # label|field × 2
    window.close()


def test_seq_advanced_autoset_fields_immutable_when_model_trained(
    monkeypatch, tmp_path
):
    """Knobs auto-derived from the model's training are disabled (immutable);
    the runtime-only knobs stay editable."""
    from PySide6.QtCore import Qt

    key = _seed_seq_crop_model(
        monkeypatch,
        tmp_path,
        name="trained_crop",
        training_params={
            "imgsz": 128,
            "crop_pad_ratio": 0.1,
            "min_crop_size_px": 64,
            "enforce_square": True,
        },
    )
    window = _make_main_window(monkeypatch)
    panel = window._detection_panel
    combo = panel.combo_yolo_crop_obb_model
    idx = combo.findData(key, Qt.UserRole)
    assert idx >= 0
    combo.setCurrentIndex(idx)

    assert panel.spin_yolo_seq_crop_pad.isEnabled() is False
    assert panel.spin_yolo_seq_min_crop_px.isEnabled() is False
    assert panel.chk_yolo_seq_square_crop.isEnabled() is False
    assert panel.spin_yolo_seq_stage2_imgsz.isEnabled() is False
    # Runtime knobs remain editable.
    assert panel.spin_yolo_seq_detect_conf.isEnabled() is True
    assert panel.spin_yolo_seq_individual_batch_size.isEnabled() is True
    assert panel.chk_yolo_seq_stage2_pow2_pad.isEnabled() is True
    window.close()


def test_seq_advanced_editable_without_training_metadata(monkeypatch, tmp_path):
    """A model with no training metadata leaves all knobs editable (manual
    fallback), and the checkpoint-imgsz fallback locks only imgsz."""
    from PySide6.QtCore import Qt

    key = _seed_seq_crop_model(monkeypatch, tmp_path, name="bare_crop")
    window = _make_main_window(monkeypatch)
    panel = window._detection_panel
    combo = panel.combo_yolo_crop_obb_model
    idx = combo.findData(key, Qt.UserRole)
    assert idx >= 0
    combo.setCurrentIndex(idx)

    assert panel.spin_yolo_seq_crop_pad.isEnabled() is True
    assert panel.spin_yolo_seq_min_crop_px.isEnabled() is True
    assert panel.chk_yolo_seq_square_crop.isEnabled() is True
    assert panel.spin_yolo_seq_stage2_imgsz.isEnabled() is True

    # Checkpoint fallback sets + locks stage-2 imgsz only.
    panel._apply_seq_crop_imgsz(key, 256)
    assert panel.spin_yolo_seq_stage2_imgsz.value() == 256
    assert panel.spin_yolo_seq_stage2_imgsz.isEnabled() is False
    assert panel.spin_yolo_seq_crop_pad.isEnabled() is True
    window.close()


def test_seq_advanced_editable_when_no_model_selected(monkeypatch):
    window = _make_main_window(monkeypatch)
    panel = window._detection_panel
    assert panel.spin_yolo_seq_crop_pad.isEnabled() is True
    assert panel.spin_yolo_seq_min_crop_px.isEnabled() is True
    assert panel.chk_yolo_seq_square_crop.isEnabled() is True
    assert panel.spin_yolo_seq_stage2_imgsz.isEnabled() is True
    window.close()


def _visible_cells(layout, panel):
    owners: dict[tuple[int, int], list[str]] = {}
    for index in range(layout.count()):
        item = layout.itemAt(index)
        widget = item.widget()
        if widget is None or widget.isHidden():
            continue
        row, col, rows, cols = layout.getItemPosition(index)
        name = widget.objectName() or type(widget).__name__
        for r in range(row, row + rows):
            for c in range(col, col + cols):
                owners.setdefault((r, c), []).append(name)
    return {cell: names for cell, names in owners.items() if len(names) > 1}


def test_no_two_visible_yolo_items_share_a_grid_cell(monkeypatch, tmp_path):
    """Decision 30: the SAHI profile combo used to be painted UNDER the
    Classes field (both at f_yolo (7, 1)), so it could not be reached."""
    from tests.test_trackerkit_slice_meta_prefill import _write_sidecar_with_profile

    window = _make_main_window(monkeypatch)
    panel = window._detection_panel
    model_path = tmp_path / "model.pt"
    model_path.write_text("stub model", encoding="utf-8")
    _write_sidecar_with_profile(model_path)
    panel.apply_slice_meta_for_model(str(model_path))
    layout = panel.yolo_group.layout()
    for obb_mode in (0, 1):
        panel.combo_yolo_obb_mode.setCurrentIndex(obb_mode)
        panel._on_yolo_mode_changed(obb_mode)
        for advanced in (False, True):
            panel.slice_settings.btn_slice_advanced.setChecked(advanced)
            assert _visible_cells(layout, panel) == {}, (obb_mode, advanced)
    # The profile picker is reachable in direct mode.
    panel.combo_yolo_obb_mode.setCurrentIndex(0)
    panel._on_yolo_mode_changed(0)
    assert panel.combo_slice_profile.isVisibleTo(panel) is True
    window.close()


def test_fresh_session_keeps_engine_overlap_and_writes_no_key(monkeypatch):
    """Decision 32 / F7: a fraction of 0.4 with no saved overlap still runs
    the engine default 0.2 -- the whole-animal minimum is only SUGGESTED."""
    window = _make_main_window(
        monkeypatch, advanced_config={"slice_object_tile_fraction": 0.4}
    )
    panel = window._detection_panel
    panel.chk_slice_enabled.setChecked(True)
    _select_geometry(panel, "auto_object")
    params = window.get_parameters_dict()
    assert params["SLICE_OVERLAP"] == pytest.approx(0.2)
    assert params["SLICE_OBJECT_TILE_FRACTION"] == pytest.approx(0.4)
    assert "slice_overlap" not in window.advanced_config
    assert panel.spin_slice_overlap.value() == pytest.approx(0.2)
    # The suggestion is offered, not applied.
    raise_button = panel.slice_settings.btn_slice_overlap_raise
    assert raise_button.isVisibleTo(panel) is True
    assert "slice_overlap" not in window.advanced_config
    window.close()


_SPIN_KEYS = (
    ("slice_overlap", "spin_slice_overlap", 0.2),
    ("slice_object_tile_fraction", "spin_slice_object_fraction", 0.15),
    ("slice_width", "spin_slice_tile_w", 0),
    ("slice_height", "spin_slice_tile_h", 0),
    ("slice_tile_batch_size", "spin_slice_tile_batch", 16),
    ("slice_memory_budget_mib", "spin_slice_memory_budget", 256),
)


def _assert_advanced_matches_spins(window, where: str) -> None:
    panel = window._detection_panel
    for key, attr, default in _SPIN_KEYS:
        spin = getattr(panel, attr)
        expected = window.advanced_config.get(key, default)
        decimals = spin.decimals() if hasattr(spin, "decimals") else 0
        assert spin.value() == pytest.approx(round(float(expected), decimals)), (
            where,
            key,
        )


def test_advanced_slice_keys_equal_spins_after_load_profile_and_geometry(
    monkeypatch, tmp_path
):
    """Decision 32: every advanced_config slice key equals its spin after a
    config load, a profile apply and a geometry change."""
    import json

    from tests.test_main_window_config_persistence import (
        _seed_trackerkit_model_repository,
        _select_first_model_with_suffix,
    )
    from tests.test_trackerkit_sahi_widget_persistence import TWO_PROFILE_SIDECAR

    models_root = _seed_trackerkit_model_repository(tmp_path, monkeypatch)
    model_path = models_root / "obb" / "direct_keep.pt"
    (model_path.parent / (model_path.name + ".slice_meta.json")).write_text(
        json.dumps(TWO_PROFILE_SIDECAR), encoding="utf-8"
    )
    seeded = {"slice_object_tile_fraction": 0.3, "slice_tile_batch_size": 9}
    window = _make_main_window(monkeypatch, advanced_config=seeded)
    panel = window._detection_panel
    _assert_advanced_matches_spins(window, "construction")
    panel.combo_detection_method.setCurrentIndex(1)
    _select_first_model_with_suffix(panel.combo_yolo_model, "direct_keep.pt")
    _assert_advanced_matches_spins(window, "primary profile applied")

    panel.combo_slice_profile.setCurrentIndex(
        panel.combo_slice_profile.findData("fast")
    )
    assert panel.combo_slice_geometry.currentData() == "custom"
    _assert_advanced_matches_spins(window, "profile switch")

    _select_geometry(panel, "auto_object")
    panel.spin_slice_overlap.setValue(0.35)
    _assert_advanced_matches_spins(window, "geometry change + edit")

    config_path = tmp_path / "session.json"
    assert window.save_config(preset_mode=True, preset_path=str(config_path))
    window.close()
    reloaded = _make_main_window(monkeypatch, advanced_config=seeded)
    reloaded._load_config_from_file(str(config_path), preset_mode=True)
    assert reloaded._detection_panel.combo_slice_geometry.currentData() == (
        "auto_object"
    )
    _assert_advanced_matches_spins(reloaded, "config load")
    reloaded.close()


def test_advanced_shows_only_bound_trackerkit_rows(monkeypatch):
    """Decision 25: no merge rows (profile-owned) and no full-frame pass;
    Advanced holds exactly the execution knobs, with their admission note."""
    window = _make_main_window(monkeypatch)
    panel = window._detection_panel
    widget = panel.slice_settings
    panel.chk_slice_enabled.setChecked(True)
    assert panel.lbl_slice_batch_admission.isVisibleTo(panel) is False
    widget.btn_slice_advanced.setChecked(True)
    for hidden in (
        widget.spin_slice_merge,
        widget.combo_slice_merge_policy,
        widget.combo_slice_merge_metric,
        widget.chk_slice_full_frame_pass,
    ):
        assert hidden.isVisibleTo(panel) is False
    assert panel.spin_slice_tile_batch.isVisibleTo(panel) is True
    assert panel.spin_slice_memory_budget.isVisibleTo(panel) is True
    assert panel.lbl_slice_batch_admission.isVisibleTo(panel) is True
    # The body size is display-only, never editable (no config key to bind).
    assert widget.spin_slice_body.isEnabled() is False
    window.close()


def test_profile_custom_tile_is_badged_profile_until_the_user_edits(
    monkeypatch, tmp_path
):
    import json

    from tests.test_trackerkit_sahi_widget_persistence import TWO_PROFILE_SIDECAR

    window = _make_main_window(monkeypatch)
    panel = window._detection_panel
    widget = panel.slice_settings
    model_path = tmp_path / "model.pt"
    model_path.write_text("stub model", encoding="utf-8")
    sidecar = model_path.parent / (model_path.name + ".slice_meta.json")
    sidecar.write_text(json.dumps(TWO_PROFILE_SIDECAR), encoding="utf-8")
    panel.apply_slice_meta_for_model(str(model_path))
    panel.combo_slice_profile.setCurrentIndex(
        panel.combo_slice_profile.findData("fast")
    )
    assert panel.combo_slice_geometry.currentData() == "custom"
    assert widget.source_badge("tile_size") == "profile"
    panel.spin_slice_tile_w.setValue(900)
    assert widget.source_badge("tile_size") == "user"

    # Training geometry that is itself custom: the stamp supplied the size.
    custom_training = {
        "schema_version": 2,
        "training_geometry": {
            "geometry_mode": "custom",
            "slice_width": 768,
            "slice_height": 768,
        },
        "primary_profile_id": "",
        "profiles": [],
    }
    other = tmp_path / "other.pt"
    other.write_text("stub model", encoding="utf-8")
    (tmp_path / "other.pt.slice_meta.json").write_text(
        json.dumps(custom_training), encoding="utf-8"
    )
    panel.apply_slice_meta_for_model(str(other))
    assert panel.combo_slice_geometry.currentData() == "custom"
    assert panel.spin_slice_tile_w.value() == 768
    assert widget.source_badge("tile_size") == "stamped"
    window.close()


def test_slice_preview_uses_the_loaded_video_frame_size(monkeypatch):
    window = _make_main_window(monkeypatch)
    panel = window._detection_panel
    preview = panel.slice_settings.preview
    assert preview.frame_size == (1920, 1080)  # no video: labelled fallback
    panel.set_slice_preview_frame_size(2448, 2048)
    assert preview.frame_size == (2448, 2048)
    panel.set_slice_preview_frame_size(None, None)
    assert preview.frame_size == (1920, 1080)
    window.close()


def test_picking_a_profile_while_sahi_is_off_applies_it_and_enables_sahi(
    monkeypatch, tmp_path
):
    """Review MAJOR-1 (main behaviour, probe A26 -> A27)."""
    import json

    from tests.test_trackerkit_sahi_widget_persistence import TWO_PROFILE_SIDECAR

    window = _make_main_window(monkeypatch)
    panel = window._detection_panel
    model_path = tmp_path / "model.pt"
    model_path.write_text("stub model", encoding="utf-8")
    (tmp_path / "model.pt.slice_meta.json").write_text(
        json.dumps(TWO_PROFILE_SIDECAR), encoding="utf-8"
    )
    panel.apply_slice_meta_for_model(str(model_path))
    panel.chk_slice_enabled.setChecked(False)
    assert panel.combo_slice_profile.isEnabled() is True
    assert panel.combo_slice_profile.isVisibleTo(panel) is True
    panel.combo_slice_profile.setCurrentIndex(
        panel.combo_slice_profile.findData("fast")
    )
    params = window.get_parameters_dict()
    assert params["SLICE_ENABLED"] is True
    assert panel.chk_slice_enabled.isChecked() is True
    assert window.advanced_config["slice_profile_id"] == "fast"
    assert not panel.slice_profile_status_text().startswith("Custom")
    window.close()


def test_profile_overlap_below_minimum_is_not_offered_a_raise(monkeypatch, tmp_path):
    """Review MINOR-2: 'Fast scan' sets overlap 0.1 (< 0.11 for 48 px on an
    800 px side): shown as info naming the profile, no Raise until edited."""
    import json

    from tests.test_trackerkit_sahi_widget_persistence import TWO_PROFILE_SIDECAR

    window = _make_main_window(monkeypatch)
    panel = window._detection_panel
    widget = panel.slice_settings
    model_path = tmp_path / "model.pt"
    model_path.write_text("stub model", encoding="utf-8")
    (tmp_path / "model.pt.slice_meta.json").write_text(
        json.dumps(TWO_PROFILE_SIDECAR), encoding="utf-8"
    )
    panel.apply_slice_meta_for_model(str(model_path))
    panel.combo_slice_profile.setCurrentIndex(
        panel.combo_slice_profile.findData("fast")
    )
    widget.set_reference_body(48.0, "profile")
    text = widget.lbl_slice_overlap_minimum.text()
    assert "set by profile 'Fast scan'" in text, text
    assert widget.btn_slice_overlap_raise.isVisibleTo(panel) is False
    panel.spin_slice_overlap.setValue(0.05)
    assert widget.btn_slice_overlap_raise.isVisibleTo(panel) is True
    window.close()
