"""A published model must be selectable, resolvable and prefill its geometry."""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

pytest.importorskip("PySide6")

import sys  # noqa: E402

from PySide6.QtWidgets import QApplication  # noqa: E402


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance() or QApplication(sys.argv)
    yield app


def test_combo_lists_published_models(qapp, monkeypatch):
    from hydra_suite.detectkit.gui.dialogs import semantic_escalation_dialog as d

    monkeypatch.setattr(d, "available_models", lambda: ["sam3", "run123"])
    dlg = d.SemanticEscalationDialog(sources=[], reference_body_px=55.0)
    items = [dlg._variant.itemText(i) for i in range(dlg._variant.count())]
    assert "run123" in items


def test_selecting_a_published_model_prefills_prompt_and_fraction(
    qapp, monkeypatch, tmp_path
):
    from hydra_suite.detectkit.gui.dialogs import semantic_escalation_dialog as d

    monkeypatch.setattr(d, "available_models", lambda: ["sam3", "run123"])
    monkeypatch.setattr(
        d,
        "sidecar_for",
        lambda k: {
            "prompt": "ant with color patch",
            "object_tile_fraction": 0.055,
            "train_tile_px": 1007,
        },
    )
    dlg = d.SemanticEscalationDialog(sources=[], reference_body_px=55.0)
    dlg.prefill_from_sidecar("run123")
    assert dlg.prompt() == "ant with color patch"
    assert abs(dlg.tile_fraction() - 0.055) < 1e-9


def test_prefill_is_a_default_not_a_lock(qapp, monkeypatch):
    # REFERENCE_BODY_SIZE precedent: a measured value is sacrosanct, a derived
    # one is a starting point the user may override.
    from hydra_suite.detectkit.gui.dialogs import semantic_escalation_dialog as d

    monkeypatch.setattr(d, "available_models", lambda: ["sam3", "run123"])
    monkeypatch.setattr(
        d, "sidecar_for", lambda k: {"prompt": "x", "object_tile_fraction": 0.05}
    )
    dlg = d.SemanticEscalationDialog(sources=[], reference_body_px=55.0)
    dlg.prefill_from_sidecar("run123")
    assert dlg._prompt.isEnabled()
    assert dlg._prompt.isReadOnly() is False


def test_job_resolves_the_selected_key_to_a_checkpoint(monkeypatch, tmp_path):
    from hydra_suite.detectkit.jobs import semantic_escalation as job

    seen = {}

    class FakeLabeler:
        @classmethod
        def from_variant(cls, variant="sam3", **kw):
            seen["variant"] = variant
            seen["checkpoint"] = kw.get("checkpoint")
            return cls()

    monkeypatch.setattr(job, "Sam3SemanticLabeler", FakeLabeler, raising=False)
    monkeypatch.setattr(
        job, "resolve_checkpoint", lambda k, **kw: tmp_path / f"{k}.pt", raising=False
    )
    ck = job.labeler_checkpoint_for("run123")
    assert str(ck).endswith("run123.pt")
