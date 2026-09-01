"""The dialog must build a spec from the panel, skip the merge, and gate on ack."""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

pytest.importorskip("PySide6")


def test_mode_combo_offers_semantic():
    # Without a combo item the plan key is unreachable and no user can ever
    # start a SAM3 run, however complete the plumbing behind it is.
    from hydra_suite.detectkit.gui.dialogs.training_dialog import (
        _SELECTION_DESCRIPTIONS,
    )

    assert ("semantic", "segment") in _SELECTION_DESCRIPTIONS


def test_selection_map_has_the_semantic_entry():
    from hydra_suite.detectkit.gui.dialogs.training_dialog import _SELECTION_ROLE_MAP

    # "segment" is the existing task vocabulary for polygon-level output;
    # inventing "polygon" would fork it.
    assert _SELECTION_ROLE_MAP[("semantic", "segment")] == ("semantic_sam3",)


def test_spec_carries_the_panel_params(monkeypatch):
    from hydra_suite.detectkit.gui.dialogs import training_dialog as td
    from hydra_suite.training.contracts import Sam3LoraParams, TrainingRole

    params = Sam3LoraParams(prompt="ant", epochs=4, label_quality_acknowledged=True)
    spec = td.TrainingDialog._sam3_spec_for(
        source_path="/tmp/src", params=params, derived_dir="/tmp/derived", seed=7
    )
    assert spec.role is TrainingRole.SEMANTIC_SAM3
    assert spec.sam3_params.prompt == "ant"
    assert spec.seed == 7
    # ONE raw source, not a merged OBB dataset.
    assert len(spec.source_datasets) == 1
    assert spec.source_datasets[0].path == "/tmp/src"


def test_unacknowledged_labels_block_the_run():
    from hydra_suite.detectkit.gui.dialogs import training_dialog as td
    from hydra_suite.training.contracts import Sam3LoraParams

    with pytest.raises(ValueError, match="acknowledge"):
        td.TrainingDialog._sam3_spec_for(
            source_path="/tmp/src",
            params=Sam3LoraParams(prompt="ant"),
            derived_dir="/tmp/derived",
            seed=7,
        )


def test_run_path_carries_sam3_params_and_is_reachable(tmp_path, monkeypatch):
    """C1 regression: the START button's actual code path, not just the builder.

    Before the fix, `_start_training` required a non-empty ultralytics base
    model for EVERY role (including semantic_sam3, which uses none) and
    built a bare `TrainingRunSpec(...)` that never set `sam3_params=`. Either
    bug alone aborts the whole multi-role run or reaches `preflight` with no
    prompt/acknowledgement. This drives `_start_training` itself and inspects
    the spec the worker was actually handed.
    """
    from PySide6.QtWidgets import QApplication

    QApplication.instance() or QApplication([])

    from hydra_suite.detectkit.gui.dialogs import training_dialog as td
    from hydra_suite.detectkit.gui.models import DetectKitProject, OBBSource
    from hydra_suite.training.contracts import TrainingRole

    proj = DetectKitProject(project_dir=tmp_path, class_names=["ant"])
    src_dir = tmp_path / "ds1"
    src_dir.mkdir()
    proj.sources = [OBBSource(path=str(src_dir), name="ds1")]

    dlg = td.TrainingDialog(proj)
    dlg.chk_role_segment_direct.setChecked(False)  # default plan; not under test
    dlg.chk_semantic_sam3.setChecked(True)
    dlg.sam3_panel.prompt_edit.setText("ant")
    dlg.sam3_panel.chk_ack.setChecked(True)

    # Skip the real dataset build (no orchestrator/filesystem dataset here):
    # only the spec-construction path in _start_training is under test.
    monkeypatch.setattr(dlg, "_build_role_datasets", lambda silent=True: True)
    dlg.role_dataset_dirs = {
        TrainingRole.SEMANTIC_SAM3.value: str(tmp_path / "derived")
    }
    monkeypatch.setattr(dlg, "_get_orchestrator", lambda: object())
    monkeypatch.setattr(dlg, "_write_to_project", lambda: None)

    captured = {}

    class _FakeWorker:
        def __init__(self, orchestrator, role_entries):
            captured["role_entries"] = role_entries
            self.log_signal = _Signal()
            self.role_started = _Signal()
            self.role_finished = _Signal()
            self.progress_signal = _Signal()
            self.done_signal = _Signal()
            self.finished = _Signal()

        def isRunning(self):
            return False

        def start(self):
            pass

    class _Signal:
        def connect(self, *_a, **_k):
            pass

    monkeypatch.setattr(td, "_TrainingWorker", _FakeWorker)

    dlg._start_training()

    entries = captured.get("role_entries")
    assert entries, "no role_entries reached the worker -- run path is unreachable"
    sam3_entries = [e for e in entries if e["role"] is TrainingRole.SEMANTIC_SAM3]
    assert len(sam3_entries) == 1
    spec = sam3_entries[0]["spec"]
    assert spec.sam3_params is not None
    assert spec.sam3_params.prompt == "ant"
    assert spec.sam3_params.label_quality_acknowledged is True
