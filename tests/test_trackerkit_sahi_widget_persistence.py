"""TrackerKit SAHI persistence golden (S4b, shared-widget adoption guard).

Captured on the UNMODIFIED tree (feat/sahi-unify-s4a @ffafadd7) before the
detection panel adopted the shared SAHI widget. For a grid of GUI states --
SAHI on/off x the three geometry modes x (no sidecar | a two-profile sidecar)
x object fraction (seeded into advanced_config, no saved overlap) -- it records,
for a FRESH session and for the same session SAVED and RELOADED:

* every ``SLICE_*`` key (+ ``YOLO_CONFIDENCE_THRESHOLD``) of
  ``get_parameters_dict()`` -- what tracking actually runs with;
* every ``slice_*`` key of the saved config JSON (incl. the
  ``slice_profile_settings`` snapshot, and ``slice_geometry_mode`` as the enum);
* every ``slice_*`` / ``_slice_*`` key of ``advanced_config`` -- including key
  ABSENCE (a fresh session must not gain a ``slice_overlap`` key).

Any diff is a TrackerKit output change. Regenerate (only on purpose) with
``pytest tests/test_trackerkit_sahi_widget_persistence.py --update-golden``.
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

GOLDEN = Path(__file__).parent / "goldens" / "trackerkit_sahi_widget_persistence.json"

MODES = ("auto_model", "auto_object", "custom")
FRACTIONS = (0.15, 0.4, 0.125)

TWO_PROFILE_SIDECAR = {
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


def select_geometry(combo, mode: str) -> None:
    """Select ``mode`` by enum whether it is item data (new) or text (old)."""
    for index in range(combo.count()):
        if combo.itemData(index) == mode or combo.itemText(index) == mode:
            combo.setCurrentIndex(index)
            return
    raise AssertionError(f"geometry mode {mode!r} not in combo")


def _normalize(value):
    return json.loads(json.dumps(value, sort_keys=True))


def _capture(window, save_path: Path) -> dict:
    advanced = {
        key: value
        for key, value in window.advanced_config.items()
        if key.startswith(("slice_", "_slice_"))
    }
    params = window.get_parameters_dict()
    slice_params = {k: params[k] for k in params if k.startswith("SLICE_")}
    slice_params["YOLO_CONFIDENCE_THRESHOLD"] = params["YOLO_CONFIDENCE_THRESHOLD"]
    assert window.save_config(preset_mode=True, preset_path=str(save_path))
    saved = json.loads(save_path.read_text(encoding="utf-8"))
    saved_slice = {k: v for k, v in saved.items() if k.startswith("slice_")}
    return _normalize(
        {"params": slice_params, "saved": saved_slice, "advanced": advanced}
    )


def _case_id(sahi: bool, mode: str, sidecar: bool, fraction: float) -> str:
    return (
        f"sahi_{'on' if sahi else 'off'}-{mode}-"
        f"{'sidecar' if sidecar else 'nosidecar'}-f{fraction:g}"
    )


CASES = [
    (sahi, mode, sidecar, fraction)
    for sahi in (False, True)
    for mode in MODES
    for sidecar in (False, True)
    for fraction in FRACTIONS
]


def _run_case(tmp_path, monkeypatch, sahi, mode, sidecar, fraction) -> dict:
    models_root = _seed_trackerkit_model_repository(tmp_path, monkeypatch)
    monkeypatch.setenv("HYDRA_CONFIG_DIR", str(tmp_path / "hydra-config"))
    model_path = models_root / "obb" / "direct_keep.pt"
    if sidecar:
        (model_path.parent / (model_path.name + ".slice_meta.json")).write_text(
            json.dumps(TWO_PROFILE_SIDECAR), encoding="utf-8"
        )
    seeded = {"slice_object_tile_fraction": fraction}

    window = _make_main_window(monkeypatch, advanced_config=seeded)
    panel = window._detection_panel
    panel.combo_detection_method.setCurrentIndex(1)  # YOLO
    _select_first_model_with_suffix(panel.combo_yolo_model, "direct_keep.pt")
    panel.combo_yolo_obb_mode.setCurrentIndex(0)  # Direct
    panel.chk_slice_enabled.setChecked(sahi)
    select_geometry(panel.combo_slice_geometry, mode)
    config_path = tmp_path / "session.json"
    fresh = _capture(window, config_path)
    window.close()

    reloaded = _make_main_window(monkeypatch, advanced_config=seeded)
    reloaded._load_config_from_file(str(config_path), preset_mode=True)
    loaded = _capture(reloaded, tmp_path / "session_resaved.json")
    reloaded.close()
    return {"fresh": fresh, "loaded": loaded}


@pytest.mark.parametrize(
    "sahi,mode,sidecar,fraction",
    CASES,
    ids=[_case_id(*case) for case in CASES],
)
def test_trackerkit_sahi_state_matches_golden(
    request, tmp_path, monkeypatch, sahi, mode, sidecar, fraction
):
    case_id = _case_id(sahi, mode, sidecar, fraction)
    got = _run_case(tmp_path, monkeypatch, sahi, mode, sidecar, fraction)
    if request.config.getoption("--update-golden", default=False):
        golden = json.loads(GOLDEN.read_text()) if GOLDEN.exists() else {}
        golden[case_id] = got
        GOLDEN.parent.mkdir(parents=True, exist_ok=True)
        GOLDEN.write_text(json.dumps(golden, indent=1, sort_keys=True) + "\n")
        pytest.skip("golden updated")
    expected = json.loads(GOLDEN.read_text())[case_id]
    assert got == expected


def test_golden_covers_every_case():
    golden = json.loads(GOLDEN.read_text())
    assert sorted(golden) == sorted(_case_id(*case) for case in CASES)
