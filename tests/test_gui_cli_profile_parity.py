"""A calibrated profile must produce identical params in the GUI and the CLI.

This is the hole that let the headless path ignore profiles for a whole
release: every golden and oracle covered DEFAULT SAHI settings, where the
GUI's in-memory advanced_config and the CLI's machine-global
advanced_config.json happen to agree.
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

from hydra_suite.trackerkit.engine_params import (  # noqa: E402
    RuntimeContext,
    build_engine_params,
)
from tests.test_trackerkit_slice_meta_prefill import (  # noqa: E402
    _make_panel_with_sidecar,
)

PROFILE_SETTINGS = {
    "enabled": True,
    "geometry_mode": "auto_object",
    "slice_width": 704,
    "slice_height": 512,
    "overlap": 0.31,
    "object_tile_fraction": 0.11,
    "trained_body_px": 560.0,
    "confidence_threshold": 0.42,
    "merge_policy": "nmm",
    "merge_metric": "iou",
    "merge_threshold": 0.6,
    "merge_backend": "cv2",
}

SLICE_KEYS = (
    "SLICE_ENABLED",
    "SLICE_GEOMETRY_MODE",
    "SLICE_OVERLAP",
    "SLICE_OBJECT_TILE_FRACTION",
    "SLICE_WIDTH",
    "SLICE_HEIGHT",
    "SLICE_TRAINED_BODY_PX",
    "SLICE_MERGE_POLICY",
    "SLICE_MERGE_METRIC",
    "SLICE_MERGE_THRESHOLD",
    "SLICE_MERGE_BACKEND",
    "YOLO_CONFIDENCE_THRESHOLD",
)


def _select_profiled_model(tmp_path, monkeypatch):
    panel, window, raw_model_path = _make_panel_with_sidecar(tmp_path, monkeypatch)

    # `_make_panel_with_sidecar` writes its stub model under `tmp_path`, outside
    # the models repository the combo actually scans
    # (`get_models_root_directory()`, recursive). `_set_yolo_model_selection`
    # only accepts a path already present as combo item data, so a model
    # outside that tree is silently dropped and `build_config_dict()` would
    # see an empty path (review C4). Copy the stub into the repository and
    # refresh the combo so the selection is real.
    import shutil

    from hydra_suite.core.inference.model_paths import get_models_root_directory

    models_root = Path(get_models_root_directory())
    models_root.mkdir(parents=True, exist_ok=True)
    model_path = models_root / raw_model_path.name
    shutil.copy2(str(raw_model_path), str(model_path))

    (model_path.parent / (model_path.name + ".slice_meta.json")).write_text(
        json.dumps(
            {
                "schema_version": 2,
                "training_geometry": {"geometry_mode": "auto_model", "imgsz": 640},
                "primary_profile_id": "balanced",
                "profiles": [
                    {
                        "id": "balanced",
                        "name": "Balanced",
                        "settings": PROFILE_SETTINGS,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    # Refresh the combo so it discovers the newly-copied model, then select it
    # through the real selector -- build_config_dict reads the path from the
    # selector, not from apply_slice_meta_for_model's argument.
    panel._refresh_yolo_model_combo(preferred_model_path=str(model_path))
    window._set_yolo_model_selection(str(model_path))
    panel.apply_slice_meta_for_model(str(model_path))
    return panel, window, model_path


def test_gui_and_builder_agree_on_a_profiled_model(tmp_path, monkeypatch):
    panel, window, model_path = _select_profiled_model(tmp_path, monkeypatch)

    gui_params = window.get_parameters_dict()
    cfg = window._config_orch.build_config_dict()
    assert cfg["yolo_obb_direct_model_path"], "model must be selected in the config"

    cli_params = build_engine_params(
        cfg,
        runtime=RuntimeContext(
            fps=float(gui_params.get("FPS", 30.0)),
            total_frames=None,
            frame_width=None,
            frame_height=None,
        ),
    )
    for key in SLICE_KEYS:
        assert cli_params[key] == gui_params[key], key
    # Guard against a vacuous pass: the profile must genuinely be in effect.
    assert gui_params["SLICE_OVERLAP"] == 0.31
    assert gui_params["SLICE_MERGE_POLICY"] == "nmm"


def test_overlay_is_a_no_op_inside_the_gui(tmp_path, monkeypatch):
    """get_parameters_dict() routes through build_engine_params too (C1).

    The GUI's advanced_config already holds the panel-applied profile values,
    so the overlay must recompute exactly those -- if it disagrees, selecting a
    profile in the GUI would change params behind the user's back.
    """
    panel, window, model_path = _select_profiled_model(tmp_path, monkeypatch)
    cfg = window._config_orch.build_config_dict()
    runtime = RuntimeContext(
        fps=30.0, total_frames=None, frame_width=None, frame_height=None
    )

    with_overlay = build_engine_params(
        cfg, runtime=runtime, advanced_config=dict(window.advanced_config)
    )
    without_model = build_engine_params(
        dict(cfg, yolo_obb_direct_model_path=""),
        runtime=runtime,
        advanced_config=dict(window.advanced_config),
    )
    for key in SLICE_KEYS:
        if key in ("SLICE_ENABLED", "SLICE_GEOMETRY_MODE"):
            continue  # config-owned; unaffected by the model path
        assert with_overlay[key] == without_model[key], key
