"""TrackerKit SAHI GUI SEQUENCES golden (S4b review MINOR-1).

Complements ``test_trackerkit_sahi_widget_persistence`` (static grid) with
the stateful paths the grid never walks: model switches (sidecar -> sidecar,
sidecar -> none -> back), selecting the custom-tile profile, a profile's
confidence applied after startup, a sequential -> direct round trip, and
opening/closing Advanced. Captured on MAIN's src (the pre-adoption panel,
@4617d692) with ``PYTHONPATH=<main>/src`` and ``--update-golden``; the
branch must reproduce every step.

Each step records the params/saved/advanced slice state (as in the grid
golden) plus the panel's SAHI control values, read tree-agnostically.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")

from PySide6.QtWidgets import QApplication  # noqa: E402

_app = QApplication.instance() or QApplication([])

from tests.test_main_window_config_persistence import (  # noqa: E402
    _make_main_window,
    _seed_trackerkit_model_repository,
    _select_first_model_with_suffix,
)

GOLDEN = Path(__file__).parent / "goldens" / "trackerkit_sahi_sequences.json"

SIDECAR_A = {
    "schema_version": 2,
    "training_geometry": {"geometry_mode": "auto_model", "imgsz": 640},
    "primary_profile_id": "balanced",
    "profiles": [
        {
            "id": "balanced",
            "name": "Balanced",
            "settings": {
                "enabled": True,
                "geometry_mode": "auto_object",
                "object_tile_fraction": 0.4,
                "overlap": 0.25,
                "trained_body_px": 75.0,
                "confidence_threshold": 0.42,
                "merge_policy": "nmm",
                "merge_metric": "iou",
                "merge_threshold": 0.6,
            },
        },
        {
            "id": "fast",
            "name": "Fast scan",
            "settings": {
                "geometry_mode": "custom",
                "slice_width": 1024,
                "slice_height": 800,
                "overlap": 0.1,
            },
        },
    ],
}

# Same profile ids as A, different values: a stale id carried across a
# model switch would silently apply B's "fast" instead of B's primary.
SIDECAR_B = {
    "schema_version": 2,
    "training_geometry": {"geometry_mode": "auto_model", "imgsz": 640},
    "primary_profile_id": "balanced",
    "profiles": [
        {
            "id": "balanced",
            "name": "B balanced",
            "settings": {
                "enabled": True,
                "geometry_mode": "auto_object",
                "object_tile_fraction": 0.2,
                "overlap": 0.3,
                "trained_body_px": 50.0,
            },
        },
        {
            "id": "fast",
            "name": "B fast",
            "settings": {
                "geometry_mode": "custom",
                "slice_width": 512,
                "slice_height": 384,
                "overlap": 0.05,
                "confidence_threshold": 0.33,
            },
        },
    ],
}


def _write_sidecar(model_path: Path, meta: dict) -> None:
    (model_path.parent / (model_path.name + ".slice_meta.json")).write_text(
        json.dumps(meta), encoding="utf-8"
    )


def _geometry_value(combo) -> str:
    data = combo.currentData()
    return str(data) if data else combo.currentText()


def _select_profile(panel, profile_id: str) -> None:
    index = panel.combo_slice_profile.findData(profile_id)
    assert index >= 0, profile_id
    panel.combo_slice_profile.setCurrentIndex(index)


def _toggle_advanced(panel) -> None:
    """Branch-only control (main has no Advanced section): open, then close."""
    widget = getattr(panel, "slice_settings", None)
    if widget is not None:
        widget.btn_slice_advanced.setChecked(True)
        widget.btn_slice_advanced.setChecked(False)


def _capture(window, tmp_path: Path, label: str) -> dict:
    panel = window._detection_panel
    advanced = {
        key: value
        for key, value in window.advanced_config.items()
        if key.startswith(("slice_", "_slice_"))
    }
    params = window.get_parameters_dict()
    slice_params = {k: params[k] for k in params if k.startswith("SLICE_")}
    slice_params["YOLO_CONFIDENCE_THRESHOLD"] = params["YOLO_CONFIDENCE_THRESHOLD"]
    path = tmp_path / "step.json"
    assert window.save_config(preset_mode=True, preset_path=str(path))
    saved = json.loads(path.read_text(encoding="utf-8"))
    controls = {
        "enabled": panel.chk_slice_enabled.isChecked(),
        "geometry": _geometry_value(panel.combo_slice_geometry),
        "profile": panel.combo_slice_profile.currentData(),
        "overlap": panel.spin_slice_overlap.value(),
        "fraction": panel.spin_slice_object_fraction.value(),
        "tile_w": panel.spin_slice_tile_w.value(),
        "tile_h": panel.spin_slice_tile_h.value(),
        "tile_batch": panel.spin_slice_tile_batch.value(),
        "memory": panel.spin_slice_memory_budget.value(),
        "confidence": panel.spin_yolo_confidence.value(),
        "obb_mode": panel.combo_yolo_obb_mode.currentIndex(),
    }
    return json.loads(
        json.dumps(
            {
                "label": label,
                "params": slice_params,
                "saved": {k: v for k, v in saved.items() if k.startswith("slice_")},
                "advanced": advanced,
                "controls": controls,
            },
            sort_keys=True,
        )
    )


def _start(tmp_path, monkeypatch, *, remove_sidecar: dict | None):
    models_root = _seed_trackerkit_model_repository(tmp_path, monkeypatch)
    monkeypatch.setenv("HYDRA_CONFIG_DIR", str(tmp_path / "hydra-config"))
    _write_sidecar(models_root / "obb" / "direct_keep.pt", SIDECAR_A)
    if remove_sidecar is not None:
        _write_sidecar(models_root / "obb" / "direct_remove.pt", remove_sidecar)
    window = _make_main_window(monkeypatch)
    panel = window._detection_panel
    panel.combo_detection_method.setCurrentIndex(1)  # YOLO
    _select_first_model_with_suffix(panel.combo_yolo_model, "direct_keep.pt")
    panel.combo_yolo_obb_mode.setCurrentIndex(0)  # Direct
    return window, panel


def _seq_model_switch_sidecars(tmp_path, monkeypatch) -> list[dict]:
    window, panel = _start(tmp_path, monkeypatch, remove_sidecar=SIDECAR_B)
    steps = [_capture(window, tmp_path, "keep: primary applied at selection")]
    _select_profile(panel, "fast")
    steps.append(_capture(window, tmp_path, "keep: fast (custom tile)"))
    _select_first_model_with_suffix(panel.combo_yolo_model, "direct_remove.pt")
    steps.append(_capture(window, tmp_path, "switch to remove (sidecar B)"))
    _select_profile(panel, "fast")
    steps.append(_capture(window, tmp_path, "remove: B fast (claims confidence)"))
    _select_first_model_with_suffix(panel.combo_yolo_model, "direct_keep.pt")
    steps.append(_capture(window, tmp_path, "back to keep (sidecar A)"))
    window.close()
    return steps


def _seq_model_switch_no_sidecar(tmp_path, monkeypatch) -> list[dict]:
    window, panel = _start(tmp_path, monkeypatch, remove_sidecar=None)
    _select_profile(panel, "fast")
    steps = [_capture(window, tmp_path, "keep: fast")]
    panel.spin_slice_tile_h.setValue(640)
    steps.append(_capture(window, tmp_path, "keep: custom edit H=640"))
    _select_first_model_with_suffix(panel.combo_yolo_model, "direct_remove.pt")
    steps.append(_capture(window, tmp_path, "switch to remove (no sidecar)"))
    _select_first_model_with_suffix(panel.combo_yolo_model, "direct_keep.pt")
    steps.append(_capture(window, tmp_path, "back to keep"))
    window.close()
    return steps


def _seq_profiles_confidence_mode_advanced(tmp_path, monkeypatch) -> list[dict]:
    window, panel = _start(tmp_path, monkeypatch, remove_sidecar=None)
    panel.spin_yolo_confidence.setValue(0.55)
    steps = [_capture(window, tmp_path, "user confidence 0.55 (marks custom)")]
    _select_profile(panel, "fast")
    steps.append(_capture(window, tmp_path, "fast: W/H reach the spins"))
    _select_profile(panel, "balanced")
    steps.append(_capture(window, tmp_path, "balanced: claimed confidence 0.42"))
    panel.combo_yolo_obb_mode.setCurrentIndex(1)
    steps.append(_capture(window, tmp_path, "sequential"))
    panel.combo_yolo_obb_mode.setCurrentIndex(0)
    steps.append(_capture(window, tmp_path, "back to direct (no custom mark)"))
    _toggle_advanced(panel)
    steps.append(_capture(window, tmp_path, "advanced open/close (no-op)"))
    panel.chk_slice_enabled.setChecked(False)
    steps.append(_capture(window, tmp_path, "SAHI off"))
    _select_profile(panel, "fast")
    steps.append(_capture(window, tmp_path, "pick fast while SAHI off"))
    window.close()
    return steps


SEQUENCES = {
    "model_switch_sidecars": _seq_model_switch_sidecars,
    "model_switch_no_sidecar": _seq_model_switch_no_sidecar,
    "profiles_confidence_mode_advanced": _seq_profiles_confidence_mode_advanced,
}


@pytest.mark.parametrize("name", sorted(SEQUENCES))
def test_trackerkit_sahi_sequence_matches_golden(request, tmp_path, monkeypatch, name):
    got = SEQUENCES[name](tmp_path, monkeypatch)
    if request.config.getoption("--update-golden", default=False):
        golden = json.loads(GOLDEN.read_text()) if GOLDEN.exists() else {}
        golden[name] = got
        GOLDEN.parent.mkdir(parents=True, exist_ok=True)
        GOLDEN.write_text(json.dumps(golden, indent=1, sort_keys=True) + "\n")
        pytest.skip("golden updated")
    expected = json.loads(GOLDEN.read_text())[name]
    assert [step["label"] for step in got] == [step["label"] for step in expected]
    for got_step, want_step in zip(got, expected):
        assert got_step == want_step, got_step["label"]


def test_sequence_golden_covers_every_sequence():
    assert sorted(json.loads(GOLDEN.read_text())) == sorted(SEQUENCES)
