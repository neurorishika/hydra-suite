"""Tests for PoseKit's SmartSelectDialog Frame Mode support."""

from __future__ import annotations

import gc
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytest.importorskip("PySide6")

from PySide6.QtWidgets import QApplication  # noqa: E402

from hydra_suite.posekit.gui.dialogs.exploration import SmartSelectDialog  # noqa: E402


@pytest.fixture()
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    return app


@pytest.fixture(autouse=True)
def cleanup_qt_widgets(qapp):
    yield
    for widget in list(qapp.topLevelWidgets()):
        widget.close()
        widget.deleteLater()
    qapp.processEvents()
    gc.collect()


def _make_dialog(qapp, frame_mode, image_paths, source_ids=None, out_root=None):
    project = SimpleNamespace(
        enhance_enabled=False, out_root=out_root or Path("/tmp/_unused_out_root")
    )
    return SmartSelectDialog(
        None,
        project,
        image_paths,
        lambda p: False,
        frame_mode=frame_mode,
        source_ids=source_ids or [None] * len(image_paths),
    )


def test_smart_select_dialog_frame_mode_disables_stratified_controls(qapp):
    dialog = _make_dialog(
        qapp, True, [Path("did10000.jpg"), Path("did10001.jpg")], ["src_a", "src_a"]
    )

    assert dialog.min_per_spin.isEnabled() is False
    assert dialog.strategy_combo.isEnabled() is False
    assert dialog.min_per_spin.toolTip() == (
        "Not used in Frame Mode — frame selection uses greedy "
        "cluster-coverage instead of per-cluster quotas."
    )


def test_smart_select_dialog_individual_mode_controls_stay_enabled(qapp):
    dialog = _make_dialog(qapp, False, [Path("a.png"), Path("b.png")])

    assert dialog.min_per_spin.isEnabled() is True
    assert dialog.strategy_combo.isEnabled() is True


def test_smart_select_dialog_preview_renders_one_line_per_frame(qapp, tmp_path):
    dialog = _make_dialog(
        qapp,
        True,
        [Path("did10000.jpg"), Path("did10001.jpg"), Path("did20000.jpg")],
        ["src_a", "src_a", "src_a"],
        out_root=tmp_path,
    )
    dialog._eligible_indices = [0, 1, 2]
    # Non-degenerate, distinguishable embeddings: idx0/idx2 identical
    # (frame (src_a, 1) is internally similar), idx1 orthogonal to both
    # (a distinct, separable cluster) -- avoids NaN cosine distances that
    # all-zero vectors would produce.
    dialog._emb = np.array(
        [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]],
        dtype=np.float32,
    )
    dialog.n_spin.setValue(1)
    dialog.k_spin.setValue(2)

    dialog._preview()

    text = dialog.preview.toPlainText()
    assert text.startswith("[frame 1] covers clusters")
    assert "2 instances" in text
    assert sorted(dialog.selected_indices) == [0, 1]
