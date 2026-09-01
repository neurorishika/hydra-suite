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
        lambda **kwargs: mod.Sam3TrainingAvailability(
            False, "package 'sam3' is missing"
        ),
    )
    panel = mod.Sam3TrainingPanel()
    # The probe is never spawned at construction time (it's a subprocess and
    # must not block the GUI thread on every panel creation) -- only on an
    # explicit check.
    assert panel.unavailable_reason() == ""
    assert panel.isEnabled()

    panel.check_availability()
    assert "sam3" in panel.unavailable_reason()
    # The body (hyperparameters, ack) is disabled, but the env row itself
    # stays interactive so the user can fix the env name and re-check.
    assert not panel._body.isEnabled()
    assert panel.env_edit.isEnabled()
    assert panel.isEnabled()


def test_available_backend_enables_body(qapp, monkeypatch):
    import hydra_suite.detectkit.gui.panels.sam3_training_panel as mod

    monkeypatch.setattr(
        mod,
        "probe_sam3_training_availability",
        lambda **kwargs: mod.Sam3TrainingAvailability(True),
    )
    panel = mod.Sam3TrainingPanel()
    panel.check_availability()
    assert panel.unavailable_reason() == ""
    assert panel._body.isEnabled()


def test_env_name_round_trips_through_params(qapp):
    from hydra_suite.detectkit.gui.panels.sam3_training_panel import Sam3TrainingPanel
    from hydra_suite.training.contracts import Sam3LoraParams

    panel = Sam3TrainingPanel()
    assert panel.params().env_name == "hydra-sam3"

    panel.env_edit.setText("my-custom-env")
    assert panel.params().env_name == "my-custom-env"

    panel.set_params(Sam3LoraParams(env_name="another-env"))
    assert panel.env_edit.text() == "another-env"
    assert panel.params().env_name == "another-env"


def test_show_event_probes_asynchronously_without_blocking(qapp, monkeypatch):
    """The on-show probe must run off the GUI thread: showEvent() must
    return immediately, and the widget must not be enabled/disabled until
    the background worker's result is delivered."""
    import hydra_suite.detectkit.gui.panels.sam3_training_panel as mod

    calls = {"n": 0}

    def fake_probe(env=None, timeout=None, cache_dir=None):
        calls["n"] += 1
        assert env == mod.DEFAULT_SAM3_ENV
        return mod.Sam3TrainingAvailability(False, "package 'sam3' is missing")

    monkeypatch.setattr(mod, "probe_sam3_training_availability", fake_probe)
    panel = mod.Sam3TrainingPanel()
    panel.show()  # triggers showEvent -> _start_async_probe

    # showEvent must not have called the probe on the GUI thread directly;
    # it must have handed the work to the worker thread.
    assert panel._probe_worker is not None
    assert "Checking" in panel.env_status_label.text()

    # The result only lands once the worker thread finishes AND the queued
    # signal is delivered by the event loop.
    assert panel._probe_worker.wait(2000)
    qapp.processEvents()

    assert calls["n"] == 1
    assert "sam3" in panel.unavailable_reason()
    assert not panel._body.isEnabled()

    panel.close()


def test_destroyed_panel_does_not_crash_on_late_probe_result(qapp, monkeypatch):
    """A queued result delivered after the panel is destroyed must be a
    silent no-op, not a crash against a dead C++ widget."""
    import hydra_suite.detectkit.gui.panels.sam3_training_panel as mod

    monkeypatch.setattr(
        mod,
        "probe_sam3_training_availability",
        lambda **kwargs: mod.Sam3TrainingAvailability(True),
    )
    panel = mod.Sam3TrainingPanel()
    panel.show()
    worker = panel._probe_worker
    assert worker is not None
    assert worker.wait(2000)

    # Destroy the panel BEFORE the queued `result` signal is delivered.
    panel.close()
    panel.deleteLater()
    qapp.processEvents()  # deliver deleteLater + the queued probe result

    # No exception raised above is the assertion.


def test_check_availability_uses_the_env_edit_value(qapp, monkeypatch):
    import hydra_suite.detectkit.gui.panels.sam3_training_panel as mod

    seen = {}

    def fake_probe(env=None, timeout=None, cache_dir=None):
        seen["env"] = env
        return mod.Sam3TrainingAvailability(True)

    monkeypatch.setattr(mod, "probe_sam3_training_availability", fake_probe)
    panel = mod.Sam3TrainingPanel()
    panel.env_edit.setText("hydra-sam3-custom")
    panel.check_availability()
    assert seen["env"] == "hydra-sam3-custom"
