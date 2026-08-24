"""Tracking refuses to start while any two arenas overlap."""

import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


def _circle(cx, cy, r, arena_id):
    return {
        "type": "circle",
        "params": (cx, cy, r),
        "mode": "include",
        "arena_id": arena_id,
    }


@pytest.fixture(scope="module")
def window():
    from PySide6.QtWidgets import QApplication

    from hydra_suite.trackerkit.gui.main_window import MainWindow

    QApplication.instance() or QApplication([])
    return MainWindow()


def test_gate_passes_with_no_overlap(window, monkeypatch):
    window.roi_shapes = [_circle(50, 50, 20, 0), _circle(300, 300, 20, 1)]
    window.arena_panel.set_frame_size(400, 400)
    window.arena_panel.set_shapes(window.roi_shapes)
    monkeypatch.setattr("PySide6.QtWidgets.QMessageBox.warning", lambda *a, **k: None)
    assert window._tracking_orch._validate_arena_overlaps("Forward") is True


def test_gate_blocks_on_overlap(window, monkeypatch):
    window.roi_shapes = [_circle(100, 100, 50, 0), _circle(130, 100, 50, 1)]
    window.arena_panel.set_frame_size(400, 400)
    window.arena_panel.set_shapes(window.roi_shapes)
    monkeypatch.setattr("PySide6.QtWidgets.QMessageBox.warning", lambda *a, **k: None)
    assert window._tracking_orch._validate_arena_overlaps("Forward") is False


def test_gate_passes_for_a_single_arena(window, monkeypatch):
    """One arena cannot overlap itself; plain-ROI users must not be gated."""
    window.roi_shapes = [_circle(100, 100, 50, 0), _circle(130, 100, 50, 0)]
    window.arena_panel.set_frame_size(400, 400)
    window.arena_panel.set_shapes(window.roi_shapes)
    monkeypatch.setattr("PySide6.QtWidgets.QMessageBox.warning", lambda *a, **k: None)
    assert window._tracking_orch._validate_arena_overlaps("Forward") is True


def test_gate_passes_with_no_arenas_at_all(window, monkeypatch):
    """No ROI means the whole video is used -- nothing to conflict."""
    window.roi_shapes = []
    window.arena_panel.set_shapes([])
    monkeypatch.setattr("PySide6.QtWidgets.QMessageBox.warning", lambda *a, **k: None)
    assert window._tracking_orch._validate_arena_overlaps("Forward") is True


def test_gate_dialog_title_distinguishes_overlap_from_disconnected(window, monkeypatch):
    """Finding 3 (fix wave 20): the refusal dialog title must match the
    ACTUAL reason -- "Overlapping Arenas" for a real overlap, "Disconnected
    Arena Regions" for a non-contiguous single arena. Previously both cases
    were hardcoded to "Overlapping Arenas"."""
    titles: list[str] = []
    monkeypatch.setattr(
        "PySide6.QtWidgets.QMessageBox.warning",
        lambda parent, title, text: titles.append(title),
    )

    # Overlap case.
    window.roi_shapes = [_circle(100, 100, 50, 0), _circle(130, 100, 50, 1)]
    window.arena_panel.set_frame_size(400, 400)
    window.arena_panel.set_shapes(window.roi_shapes)
    assert window._tracking_orch._validate_arena_overlaps("Forward") is False
    assert titles[-1] == "Forward: Overlapping Arenas"

    # Disconnected case: one arena made of two circles that don't touch.
    window.roi_shapes = [_circle(50, 50, 10, 0), _circle(350, 350, 10, 0)]
    window.arena_panel.set_frame_size(400, 400)
    window.arena_panel.set_shapes(window.roi_shapes)
    assert window._tracking_orch._validate_arena_overlaps("Forward") is False
    assert titles[-1] == "Forward: Disconnected Arena Regions"
