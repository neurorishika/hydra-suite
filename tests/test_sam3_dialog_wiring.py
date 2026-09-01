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
