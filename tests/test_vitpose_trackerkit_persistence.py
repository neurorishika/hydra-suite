from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytest.importorskip("PySide6")

from PySide6.QtWidgets import QApplication

from hydra_suite.trackerkit.gui.main_window import MainWindow


@pytest.fixture(scope="module")
def qapp() -> QApplication:
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def _make_main_window(
    monkeypatch: pytest.MonkeyPatch,
    advanced_config: dict[str, object] | None = None,
) -> MainWindow:
    monkeypatch.setattr(MainWindow, "_save_advanced_config", lambda self: None)
    monkeypatch.setattr(
        MainWindow,
        "_load_advanced_config",
        lambda self: dict(advanced_config or {}),
    )
    return MainWindow()


def test_vitpose_model_selection_roundtrips_through_save_and_load(
    monkeypatch: pytest.MonkeyPatch,
    qapp: QApplication,
    tmp_path: Path,
) -> None:
    """A selected ViTPose checkpoint must survive a save -> reload cycle.

    Regression test: the config save block previously wrote
    ``pose_yolo_model_dir``/``pose_sleap_model_dir`` but not
    ``pose_vitpose_model_dir``, and the load block never routed a saved
    value back into the vitpose slot, so a chosen ViTPose checkpoint was
    silently dropped on save/reload while YOLO and SLEAP round-tripped fine.
    """
    window = _make_main_window(monkeypatch)
    selected_model = "ViTPose/fake_vitpose_model.pth"
    window._set_pose_model_path_for_backend(selected_model, backend="vitpose")
    assert window._pose_model_path_for_backend("vitpose") == selected_model

    config_path = tmp_path / "vitpose_roundtrip.json"
    assert window.save_config(preset_mode=True, preset_path=str(config_path))
    saved_cfg = json.loads(config_path.read_text(encoding="utf-8"))
    assert saved_cfg["pose_vitpose_model_dir"] == selected_model
    window.close()

    reloaded_window = _make_main_window(monkeypatch)
    reloaded_window._load_config_from_file(str(config_path), preset_mode=True)

    assert reloaded_window._pose_model_path_for_backend("vitpose") == selected_model
    reloaded_window.close()
