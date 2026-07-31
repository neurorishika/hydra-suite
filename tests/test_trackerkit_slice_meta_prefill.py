from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytest.importorskip("PySide6")
from PySide6.QtWidgets import QApplication

_app = QApplication.instance() or QApplication([])

from tests.test_main_window_config_persistence import _make_main_window


def _make_panel_with_sidecar(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Construct the detection panel via a real MainWindow (mirrors
    tests/test_detection_panel_slice_widgets.py) and a bare model file that a
    test can attach a .slice_meta.json sidecar to.

    Returns (panel, main_window, model_path).
    """
    data_dir = tmp_path / "hydra-data"
    monkeypatch.setenv("HYDRA_DATA_DIR", str(data_dir))
    models_root = data_dir / "models"
    models_root.mkdir(parents=True, exist_ok=True)
    (models_root / "model_registry.json").write_text(
        json.dumps({"schema_version": 2, "entries": {}}, indent=2),
        encoding="utf-8",
    )
    window = _make_main_window(monkeypatch)
    panel = window._detection_panel
    model_path = tmp_path / "model.pt"
    model_path.write_text("stub model", encoding="utf-8")
    return panel, window, model_path


def test_selecting_model_with_sidecar_prefills(tmp_path, monkeypatch):
    panel, mw, model_path = _make_panel_with_sidecar(tmp_path, monkeypatch)
    (model_path.parent / (model_path.name + ".slice_meta.json")).write_text(
        json.dumps(
            {
                "geometry_mode": "auto_object",
                "overlap": 0.25,
                "reference_body_px": 560.0,
                "target_sizes": [200.0, 300.0, 400.0],
                "imgsz": 640,
            }
        ),
        encoding="utf-8",
    )
    ref_before = panel.spin_reference_body_size.value()

    panel.apply_slice_meta_for_model(str(model_path))

    assert panel.chk_slice_enabled.isChecked() is True
    assert panel.combo_slice_geometry.currentText() == "auto_object"
    assert mw.advanced_config["slice_overlap"] == 0.25
    assert abs(mw.advanced_config["slice_object_tile_fraction"] - 300.0 / 640.0) < 1e-6
    assert mw.advanced_config["slice_trained_body_px"] == 560.0
    # REFERENCE_BODY_SIZE must be left untouched.
    assert panel.spin_reference_body_size.value() == ref_before
    window = mw
    window.close()


def test_selecting_model_without_sidecar_is_noop(tmp_path, monkeypatch):
    panel, mw, model_path = _make_panel_with_sidecar(tmp_path, monkeypatch)
    enabled_before = panel.chk_slice_enabled.isChecked()
    adv_before = dict(mw.advanced_config)

    panel.apply_slice_meta_for_model(str(model_path))  # no sidecar written

    assert panel.chk_slice_enabled.isChecked() == enabled_before
    assert mw.advanced_config == adv_before
    mw.close()
