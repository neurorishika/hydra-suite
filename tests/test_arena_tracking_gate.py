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
