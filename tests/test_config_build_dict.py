"""build_config_dict() returns the same dict save_config would write, without touching disk."""

import json
import os

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
def qtbot_config_stub(monkeypatch, qapp):
    """A real, offscreen MainWindow's ConfigOrchestrator (no disk/network I/O).

    Mirrors the offscreen-MainWindow convention in
    tests/test_vitpose_trackerkit_persistence.py: construct a real MainWindow
    under QT_QPA_PLATFORM=offscreen with the advanced-config disk hooks
    stubbed out, so the widget tree is real and purity assertions on
    build_config_dict() are meaningful.
    """
    monkeypatch.setattr(MainWindow, "_save_advanced_config", lambda self: None)
    monkeypatch.setattr(MainWindow, "_load_advanced_config", lambda self: {})
    window = MainWindow()
    try:
        yield window._config_orch
    finally:
        window.close()


def test_build_config_dict_is_pure(monkeypatch, qtbot_config_stub):
    orch = qtbot_config_stub  # a constructed ConfigOrchestrator with real panels (offscreen)
    called = {"atomic_write": 0, "resolve_path": 0}
    monkeypatch.setattr(
        orch,
        "_atomic_json_write",
        lambda *a, **k: called.__setitem__("atomic_write", called["atomic_write"] + 1)
        or (True, None),
    )
    monkeypatch.setattr(
        orch,
        "_resolve_config_save_path",
        lambda *a, **k: called.__setitem__("resolve_path", called["resolve_path"] + 1)
        or None,
    )

    cfg = orch.build_config_dict()

    assert isinstance(cfg, dict)
    assert "detection_method" in cfg
    assert "enable_pose_extractor" in cfg
    assert called["atomic_write"] == 0
    assert called["resolve_path"] == 0


def test_max_bridge_gap_fragment_veto_color_tag_round_trip(qtbot_config_stub, tmp_path):
    """The 4 params-only leaks (Task 3) must persist into config and round-trip.

    MAX_BRIDGE_GAP_FRAMES / FRAGMENT_SPATIAL_VETO_THRESHOLD / COLOR_TAG_MODEL_PATH /
    COLOR_TAG_CONFIDENCE were previously read straight from widgets into engine
    params without ever being persisted to config, so the CLI/shared builder
    couldn't see them. This test drives non-default widget values through
    build_config_dict() -> disk -> _load_config_from_file() and asserts both the
    serialized snake_case fields and the widget read-back survive the round trip.
    """
    orch = qtbot_config_stub

    # Set widgets to NON-default values.
    orch._panels.postprocess.spin_max_bridge_gap_frames.setValue(77)
    orch._panels.postprocess.spin_fragment_spatial_veto_threshold.setValue(0.42)
    orch._panels.identity.line_color_tag_model.setText("models/color_tag.pt")
    orch._panels.identity.spin_color_tag_conf.setValue(0.91)

    cfg = orch.build_config_dict()

    assert cfg["max_bridge_gap_frames"] == 77
    assert cfg["fragment_spatial_veto_threshold"] == pytest.approx(0.42)
    assert cfg["color_tag_model_path"] == "models/color_tag.pt"
    assert cfg["color_tag_confidence"] == pytest.approx(0.91)

    # Reset widgets to defaults so the load path is provably responsible for
    # restoring the non-default values, not leftover widget state.
    orch._panels.postprocess.spin_max_bridge_gap_frames.setValue(30)
    orch._panels.postprocess.spin_fragment_spatial_veto_threshold.setValue(0.05)
    orch._panels.identity.line_color_tag_model.setText("")
    orch._panels.identity.spin_color_tag_conf.setValue(0.5)

    config_path = tmp_path / "roundtrip_config.json"
    config_path.write_text(json.dumps(cfg))
    orch._load_config_from_file(str(config_path))

    assert orch._panels.postprocess.spin_max_bridge_gap_frames.value() == 77
    assert orch._panels.postprocess.spin_fragment_spatial_veto_threshold.value() == (
        pytest.approx(0.42)
    )
    assert orch._panels.identity.line_color_tag_model.text() == "models/color_tag.pt"
    assert orch._panels.identity.spin_color_tag_conf.value() == pytest.approx(0.91)


def test_max_bridge_gap_fragment_veto_color_tag_defaults(qtbot_config_stub):
    """Defaults must match today's widget-constructed defaults exactly."""
    orch = qtbot_config_stub
    cfg = orch.build_config_dict()
    assert cfg["max_bridge_gap_frames"] == 30
    assert cfg["fragment_spatial_veto_threshold"] == pytest.approx(0.05)
    assert cfg["color_tag_model_path"] == ""
    assert cfg["color_tag_confidence"] == pytest.approx(0.5)
