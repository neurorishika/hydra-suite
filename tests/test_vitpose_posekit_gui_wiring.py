"""ViTPose GUI wiring: selectable in the combo, mapped by _pred_backend, and
the vitpose_pose resolver stage is used for its runtime flavor."""

from __future__ import annotations

import importlib
from types import SimpleNamespace


def _mw_module():
    import os

    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    return importlib.import_module("hydra_suite.posekit.gui.main_window")


class _Combo:
    def __init__(self, text: str):
        self._t = text

    def currentText(self) -> str:
        return self._t


def test_pred_backend_maps_vitpose():
    mw = _mw_module()
    self_ns = SimpleNamespace(combo_pred_backend=_Combo("ViTPose"))
    assert mw.MainWindow._pred_backend(self_ns) == "vitpose"
    self_ns2 = SimpleNamespace(combo_pred_backend=_Combo("SLEAP"))
    assert mw.MainWindow._pred_backend(self_ns2) == "sleap"
    self_ns3 = SimpleNamespace(combo_pred_backend=_Combo("YOLO"))
    assert mw.MainWindow._pred_backend(self_ns3) == "yolo"


def test_pred_runtime_flavor_uses_vitpose_stage(monkeypatch):
    mw = _mw_module()
    captured = {}

    class _Resolved:
        backend = "torch"
        device = "cpu"

    class _Resolver:
        def __init__(self, tier, platform):
            pass

        def resolve(self, stage):
            captured["stage"] = stage
            return _Resolved()

    monkeypatch.setattr("hydra_suite.runtime.resolver.RuntimeResolver", _Resolver)
    self_ns = SimpleNamespace(
        _selected_tier=lambda: "cpu",
        _pred_backend=lambda: "vitpose",
    )
    flavor = mw.MainWindow._pred_runtime_flavor(self_ns)
    assert captured["stage"] == "vitpose_pose"
    assert flavor == "cpu"


def test_combo_and_widget_present_in_source():
    import hydra_suite.posekit.gui.main_window as mw

    with open(mw.__file__, "r", encoding="utf-8") as fh:
        text = fh.read()
    assert '"ViTPose"' in text  # combo item
    assert "vitpose_pred_widget" in text  # dedicated settings widget
    assert "pred_vitpose_edit" in text  # checkpoint line edit
