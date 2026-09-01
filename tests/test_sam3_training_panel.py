"""The panel owns the knobs, the acknowledgement, and disabled-with-reason."""

from __future__ import annotations

import os
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

pytest.importorskip("PySide6")

from PySide6.QtWidgets import QApplication  # noqa: E402


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance() or QApplication(sys.argv)
    yield app


def test_params_round_trip(qapp):
    from hydra_suite.detectkit.gui.panels.sam3_training_panel import Sam3TrainingPanel
    from hydra_suite.training.contracts import Sam3LoraParams

    panel = Sam3TrainingPanel()
    got = panel.params()
    assert got.epochs == 10 and got.rank == 16

    panel.set_params(Sam3LoraParams(prompt="beetle", epochs=3, rank=8))
    back = panel.params()
    assert back.prompt == "beetle" and back.epochs == 3 and back.rank == 8


def test_training_is_blocked_until_labels_are_acknowledged(qapp):
    from hydra_suite.detectkit.gui.panels.sam3_training_panel import Sam3TrainingPanel

    panel = Sam3TrainingPanel()
    # Provenance does not survive a review, so the user must affirm the labels
    # are good before SAM3 learns them -- including its own accepted output.
    assert panel.acknowledged() is False
    assert panel.params().label_quality_acknowledged is False
    panel.chk_ack.setChecked(True)
    assert panel.acknowledged() is True
    assert panel.params().label_quality_acknowledged is True


def test_unavailable_backend_disables_with_a_reason(qapp, monkeypatch):
    import hydra_suite.detectkit.gui.panels.sam3_training_panel as mod

    monkeypatch.setattr(
        mod,
        "probe_sam3_training_availability",
        lambda: mod.Sam3TrainingAvailability(False, "package 'sam3' is missing"),
    )
    panel = mod.Sam3TrainingPanel()
    assert "sam3" in panel.unavailable_reason()
    assert not panel.isEnabled()
