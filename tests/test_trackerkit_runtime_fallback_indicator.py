"""Task 7 (Phase 3): the GPU-Fast fallback hint (spec §5.4) is populated when
GPU-Fast is selected and cleared otherwise.

Exercises `SessionOrchestrator._update_runtime_fallback_hint` against light
duck-typed fakes so the check needs neither a QApplication nor the full
trackerkit MainWindow (constructing real Qt widgets without a QApplication
aborts the process).
"""

from hydra_suite.trackerkit.gui.orchestrators.session import SessionOrchestrator


class _FakeCombo:
    def __init__(self, tier: str) -> None:
        self._tier = tier

    def currentData(self):
        return self._tier


class _FakeLabel:
    def __init__(self) -> None:
        self.text = ""
        self.visible = False

    def setText(self, text: str) -> None:
        self.text = text

    def setVisible(self, flag: bool) -> None:
        self.visible = bool(flag)


class _FakePanel:
    def __init__(self, tier: str) -> None:
        self.combo_runtime_tier = _FakeCombo(tier)
        self.lbl_runtime_fallback = _FakeLabel()


class _FakeMW:
    def __init__(self, tier: str) -> None:
        self._setup_panel = _FakePanel(tier)


def _hint_for(tier: str) -> _FakeLabel:
    orch = SessionOrchestrator.__new__(SessionOrchestrator)
    orch._mw = _FakeMW(tier)
    orch._update_runtime_fallback_hint()
    return orch._mw._setup_panel.lbl_runtime_fallback


def test_gpu_fast_shows_fallback_hint():
    lbl = _hint_for("gpu_fast")
    assert lbl.visible is True
    assert "GPU-Fast" in lbl.text
    assert ("TensorRT" in lbl.text) or ("CoreML" in lbl.text)


def test_gpu_tier_clears_hint():
    lbl = _hint_for("gpu")
    assert lbl.text == ""
    assert lbl.visible is False


def test_cpu_tier_clears_hint():
    lbl = _hint_for("cpu")
    assert lbl.text == ""
    assert lbl.visible is False


def test_missing_panel_is_safe():
    orch = SessionOrchestrator.__new__(SessionOrchestrator)

    class _MW:
        pass

    orch._mw = _MW()
    # Should not raise when there is no setup panel / label yet.
    orch._update_runtime_fallback_hint()
