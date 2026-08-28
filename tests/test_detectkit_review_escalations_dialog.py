"""Tests for DetectKit ReviewEscalationsDialog."""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import Qt  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance() or QApplication(sys.argv)
    yield app


def _make_pending_source(tmp_path, name="orig"):
    from hydra_suite.detectkit.gui.models import OBBSource, PendingEscalation

    source_root = tmp_path / name
    (source_root / "labels").mkdir(parents=True)
    (source_root / "images").mkdir(parents=True)
    (source_root / "classes.txt").write_text("ant\n", encoding="utf-8")

    staged_root = tmp_path / f"{name}-staged"
    (staged_root / "labels").mkdir(parents=True)
    (staged_root / "labels" / "a.txt").write_text(
        "0 0.1 0.1 0.2 0.1 0.2 0.2 0.1 0.2 0.1 0.1\n", encoding="utf-8"
    )
    (staged_root / "classes.txt").write_text("ant\n", encoding="utf-8")

    return OBBSource(
        path=str(source_root),
        name=name,
        level="obb",
        pending_escalation=PendingEscalation(
            staged_path=str(staged_root),
            target_level="polygon",
            sam2_variant="sam2.1-hiera-base_plus",
            created_at="2026-08-27T00:00:00",
        ),
    )


def test_review_escalations_dialog_is_base_dialog(qapp, tmp_path):
    from hydra_suite.detectkit.gui.dialogs.review_escalations_dialog import (
        ReviewEscalationsDialog,
    )
    from hydra_suite.widgets.dialogs import BaseDialog

    src = _make_pending_source(tmp_path)
    dlg = ReviewEscalationsDialog([src])
    assert isinstance(dlg, BaseDialog)
    assert dlg._list.count() == 1


def test_review_escalations_dialog_accept_checked_promotes_source(qapp, tmp_path):
    from hydra_suite.detectkit.gui.dialogs.review_escalations_dialog import (
        ReviewEscalationsDialog,
    )

    src = _make_pending_source(tmp_path)
    dlg = ReviewEscalationsDialog([src])
    dlg._list.item(0).setCheckState(Qt.Checked)

    dlg._apply_checked(accept=True)

    assert dlg.accepted_names == ["orig"]
    assert dlg._list.count() == 0
    assert src.pending_escalation is None
    assert src.level == "polygon"
    assert src.reviewed is False
    assert (Path(src.path) / "labels" / "a.txt").exists()


def test_review_escalations_dialog_reject_checked_discards_staging(qapp, tmp_path):
    from hydra_suite.detectkit.gui.dialogs.review_escalations_dialog import (
        ReviewEscalationsDialog,
    )

    src = _make_pending_source(tmp_path)
    staged_path = src.pending_escalation.staged_path
    dlg = ReviewEscalationsDialog([src])
    dlg._list.item(0).setCheckState(Qt.Checked)

    dlg._apply_checked(accept=False)

    assert dlg.rejected_names == ["orig"]
    assert dlg._list.count() == 0
    assert src.pending_escalation is None
    assert src.level == "obb"
    assert not Path(staged_path).exists()


def test_review_escalations_dialog_skips_unchecked_rows(qapp, tmp_path, monkeypatch):
    from hydra_suite.detectkit.gui.dialogs.review_escalations_dialog import (
        ReviewEscalationsDialog,
    )

    # With no rows checked, _apply_checked shows a blocking QMessageBox.information
    # to tell the user nothing was selected -- under QT_QPA_PLATFORM=offscreen that
    # still opens a real (invisible) event loop and hangs the test process waiting
    # for a click that will never come, so it must be monkeypatched out here (this
    # repo has hit exactly this class of hang before -- see CLAUDE.md's "main
    # whole-suite blockers" note on modal-dialog hangs).
    monkeypatch.setattr(
        "hydra_suite.detectkit.gui.dialogs.review_escalations_dialog.QMessageBox.information",
        lambda *args, **kwargs: None,
    )

    src = _make_pending_source(tmp_path)
    dlg = ReviewEscalationsDialog([src])
    dlg._list.item(0).setCheckState(Qt.Unchecked)

    dlg._apply_checked(accept=True)

    assert dlg.accepted_names == []
    assert dlg._list.count() == 1
    assert src.pending_escalation is not None
