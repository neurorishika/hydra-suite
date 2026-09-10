"""build_config_dict() -- the GUI's real save path -- emits portable paths.

Spec §6.5's characterization test: the GUI must relativize model paths on
its actual save path (``ConfigOrchestrator.build_config_dict``), not merely
via the ``make_model_path_relative`` helper in isolation (that is what
``tests/test_config_model_path_portability.py`` already proves)."""

import os
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytest.importorskip("PySide6")

from PySide6.QtWidgets import QApplication  # noqa: E402

from hydra_suite.trackerkit.gui.main_window import MainWindow  # noqa: E402


@pytest.fixture(scope="module")
def qapp() -> QApplication:
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


@pytest.fixture
def main_window(monkeypatch, qapp):
    monkeypatch.setattr(MainWindow, "_save_advanced_config", lambda self: None)
    monkeypatch.setattr(MainWindow, "_load_advanced_config", lambda self: {})
    window = MainWindow()
    try:
        yield window
    finally:
        window.close()


def test_build_config_dict_relativizes_color_tag_and_cnn_paths(
    main_window, tmp_path, monkeypatch
):
    models = tmp_path / "models"
    (models / "classification" / "identity").mkdir(parents=True)
    tag = models / "classification" / "identity" / "tag.pth"
    clf = models / "classification" / "identity" / "clf.multihead.json"
    tag.write_bytes(b"t")
    clf.write_text("{}")
    monkeypatch.setenv("HYDRA_MODELS_DIR", str(models))

    # Fix W8: `main_window._panels` does not exist -- `_panels_bundle()`
    # (main_window.py:979) builds a FRESH SimpleNamespace each call. The
    # persistent attribute is `ConfigOrchestrator._panels`
    # (orchestrators/config.py:165), reached via `main_window._config_orch`.
    orch = main_window._config_orch
    orch._panels.identity.line_color_tag_model.setText(str(tag))
    monkeypatch.setattr(
        main_window,
        "_identity_config",
        lambda: {"cnn_classifiers": [{"model_path": str(clf), "batch_size": 8}]},
    )

    cfg = orch.build_config_dict(preset_mode=False)

    assert cfg["color_tag_model_path"] == "classification/identity/tag.pth"
    assert cfg["cnn_classifiers"][0]["model_path"] == (
        "classification/identity/clf.multihead.json"
    )
    assert not Path(cfg["color_tag_model_path"]).is_absolute()
    assert not Path(cfg["cnn_classifiers"][0]["model_path"]).is_absolute()
