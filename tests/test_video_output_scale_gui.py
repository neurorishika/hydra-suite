"""GUI persistence of ``video_output_scale`` (post-processing panel spinbox)."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")

from PySide6.QtWidgets import QApplication  # noqa: E402

from tests.test_main_window_config_persistence import _make_main_window  # noqa: E402


@pytest.fixture(scope="module")
def qapp() -> QApplication:
    return QApplication.instance() or QApplication([])


def test_spinbox_shape_and_default(monkeypatch, qapp):
    window = _make_main_window(monkeypatch)
    spin = window._postprocess_panel.spin_video_output_scale
    assert spin.value() == pytest.approx(0.5)
    assert (spin.minimum(), spin.maximum()) == (pytest.approx(0.1), pytest.approx(1.0))
    assert spin.singleStep() == pytest.approx(0.05)
    assert spin.decimals() == 2
    window.close()


def test_scale_round_trips_through_save_and_load(monkeypatch, qapp, tmp_path: Path):
    window = _make_main_window(monkeypatch)
    window._postprocess_panel.spin_video_output_scale.setValue(0.35)
    assert window._config_orch.build_config_dict()["video_output_scale"] == (
        pytest.approx(0.35)
    )
    path = tmp_path / "scale.json"
    assert window.save_config(preset_mode=True, preset_path=str(path))
    saved = json.loads(path.read_text(encoding="utf-8"))
    assert saved["video_output_scale"] == pytest.approx(0.35)
    window.close()

    reloaded = _make_main_window(monkeypatch)
    reloaded._postprocess_panel.spin_video_output_scale.setValue(0.9)
    reloaded._load_config_from_file(str(path), preset_mode=True)
    assert reloaded._postprocess_panel.spin_video_output_scale.value() == (
        pytest.approx(0.35)
    )

    # A config written before the knob existed loads as the 0.5 default.
    saved.pop("video_output_scale")
    old = tmp_path / "old.json"
    old.write_text(json.dumps(saved), encoding="utf-8")
    reloaded._postprocess_panel.spin_video_output_scale.setValue(0.9)
    reloaded._load_config_from_file(str(old), preset_mode=True)
    assert reloaded._postprocess_panel.spin_video_output_scale.value() == (
        pytest.approx(0.5)
    )
    reloaded.close()
