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


def test_slice_geometry_hidden_while_sahi_off_in_direct_mode(monkeypatch):
    """SAHI inputs are pointless while sliced inference is off: the geometry
    picker must be hidden until the SAHI checkbox is checked."""
    window = _make_main_window(monkeypatch)
    panel = window._detection_panel
    assert panel.combo_yolo_obb_mode.currentIndex() == 0  # Direct
    panel.chk_slice_enabled.setChecked(False)
    panel._on_yolo_mode_changed(0)
    assert panel.row_slice_geometry.isHidden() is True
    assert panel.combo_slice_geometry.isVisibleTo(panel) is False
    # The checkbox itself stays visible so the user can turn SAHI on.
    assert panel.chk_slice_enabled.isVisibleTo(panel) is True

    panel.chk_slice_enabled.setChecked(True)
    panel._on_slice_toggled(True)
    assert panel.row_slice_geometry.isHidden() is False
    assert panel.combo_slice_geometry.isVisibleTo(panel) is True
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


def test_slice_params_controls_follow_geometry_mode(monkeypatch):
    """custom mode reveals tile W/H; auto_object reveals the object fraction;
    auto_model reveals neither (only the shared tile overlap)."""
    window = _make_main_window(monkeypatch)
    panel = window._detection_panel
    panel.chk_slice_enabled.setChecked(True)
    panel._on_slice_toggled(True)

    panel.combo_slice_geometry.setCurrentText("custom")
    panel._on_slice_geometry_changed(panel.combo_slice_geometry.currentIndex())
    assert panel.spin_slice_tile_w.isHidden() is False
    assert panel.spin_slice_tile_h.isHidden() is False
    assert panel.spin_slice_object_fraction.isHidden() is True
    assert panel.spin_slice_overlap.isHidden() is False

    panel.combo_slice_geometry.setCurrentText("auto_object")
    panel._on_slice_geometry_changed(panel.combo_slice_geometry.currentIndex())
    assert panel.spin_slice_tile_w.isHidden() is True
    assert panel.spin_slice_tile_h.isHidden() is True
    assert panel.spin_slice_object_fraction.isHidden() is False

    panel.combo_slice_geometry.setCurrentText("auto_model")
    panel._on_slice_geometry_changed(panel.combo_slice_geometry.currentIndex())
    assert panel.spin_slice_tile_w.isHidden() is True
    assert panel.spin_slice_tile_h.isHidden() is True
    assert panel.spin_slice_object_fraction.isHidden() is True
    assert panel.spin_slice_overlap.isHidden() is False
    window.close()


def test_slice_params_row_hidden_while_sahi_off(monkeypatch):
    window = _make_main_window(monkeypatch)
    panel = window._detection_panel
    assert panel.chk_slice_enabled.isChecked() is False
    assert panel.row_slice_params.isHidden() is True
    panel.chk_slice_enabled.setChecked(True)
    panel._on_slice_toggled(True)
    assert panel.row_slice_params.isHidden() is False
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
