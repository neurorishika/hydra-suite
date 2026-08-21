"""Task 8: ``SessionOrchestrator.current_arena_id`` / ``start_new_arena()`` and
the arena-id stamping ``finish_roi_selection`` does on every newly added
shape.

One arena is often several shapes (an include circle plus an exclude hole
punched in it), so a new shape must join the CURRENT arena unless the user
explicitly starts a new one -- shape count is never arena count. Every shape
(excludes included) carries an ``arena_id`` so ``start_new_arena`` can find
the next free id by scanning ALL shapes, not just includes.

Drives the real ``SessionOrchestrator`` (not a reimplementation) against a
``MagicMock`` stand-in for ``MainWindow`` -- a real Qt ``MainWindow`` is not
actually required here (an earlier draft of this file wrongly claimed one
was blocked by a pre-existing ``SyntaxError`` in
``trackerkit/gui/dialogs/train_yolo_dialog.py``; that was an artifact of
running pytest under the wrong interpreter -- see
``tests/test_arena_gui_frame_dimensions.py``'s docstring for the correction).
The ``MagicMock`` is used here because ``SessionOrchestrator`` needs many
more attributes than ``ConfigOrchestrator`` does, and none of this file's
assertions depend on real widget behavior. A real (offscreen)
``QApplication`` backs the ``QTimer.singleShot`` calls
``finish_roi_selection`` makes at the end of a successful shape add.
"""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from unittest.mock import MagicMock

import pytest

pytest.importorskip("PySide6")

from PySide6.QtWidgets import QApplication  # noqa: E402

from hydra_suite.trackerkit.gui.orchestrators.session import (  # noqa: E402
    SessionOrchestrator,
)


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def _make_orch(qapp, roi_shapes=None):
    mw = MagicMock()
    mw.roi_shapes = roi_shapes if roi_shapes is not None else []
    mw.roi_base_frame.height.return_value = 240
    mw.roi_base_frame.width.return_value = 320
    mw.roi_current_mode = "circle"
    mw.roi_fitted_circle = (10.0, 10.0, 5.0)
    mw.roi_points = []
    return SessionOrchestrator(mw, config=MagicMock(), panels=MagicMock())


def test_current_arena_id_starts_at_zero(qapp):
    orch = _make_orch(qapp)
    assert orch.current_arena_id == 0


def test_start_new_arena_on_empty_shapes_stays_zero(qapp):
    orch = _make_orch(qapp)
    assert orch.start_new_arena() == 0
    assert orch.current_arena_id == 0


def test_start_new_arena_advances_past_highest_used_id(qapp):
    orch = _make_orch(
        qapp,
        roi_shapes=[
            {"type": "circle", "params": [1, 1, 1], "mode": "include", "arena_id": 0},
            {"type": "circle", "params": [2, 2, 1], "mode": "include", "arena_id": 2},
        ],
    )
    assert orch.start_new_arena() == 3
    assert orch.current_arena_id == 3


def test_start_new_arena_counts_exclude_only_arenas():
    """An exclude hole belonging to arena 3 means arena 3 is in use --
    start_new_arena must scan ALL shapes, not just includes."""
    mw = MagicMock()
    mw.roi_shapes = [
        {"type": "circle", "params": [1, 1, 1], "mode": "exclude", "arena_id": 3},
    ]
    orch = SessionOrchestrator(mw, config=MagicMock(), panels=MagicMock())
    assert orch.start_new_arena() == 4


def test_finish_roi_selection_stamps_current_arena_id_on_new_shape(qapp):
    orch = _make_orch(qapp)
    orch.current_arena_id = 5
    orch.finish_roi_selection()
    assert orch._mw.roi_shapes[-1]["arena_id"] == 5


def test_finish_roi_selection_stamps_shapes_after_new_arena(qapp):
    """New Arena, then Add Shape: the freshly-added shape carries the NEW id,
    not the one it had before start_new_arena() was pressed."""
    orch = _make_orch(
        qapp,
        roi_shapes=[
            {"type": "circle", "params": [1, 1, 1], "mode": "include", "arena_id": 0},
        ],
    )
    new_id = orch.start_new_arena()
    orch.finish_roi_selection()
    assert orch._mw.roi_shapes[-1]["arena_id"] == new_id
    assert new_id == 1


def test_finish_roi_selection_stamps_polygon_shapes_too(qapp):
    """The polygon append site (session.py's second roi_shapes.append) must
    stamp arena_id exactly like the circle site does."""
    orch = _make_orch(qapp)
    orch._mw.roi_current_mode = "polygon"
    orch._mw.roi_points = [(0, 0), (10, 0), (10, 10)]
    orch.current_arena_id = 7
    orch.finish_roi_selection()
    assert orch._mw.roi_shapes[-1]["type"] == "polygon"
    assert orch._mw.roi_shapes[-1]["arena_id"] == 7


def test_finish_roi_selection_without_new_arena_joins_current_arena(qapp):
    """Two shapes added back-to-back without pressing New Arena both join
    arena 0 -- an include circle plus its exclude hole are ONE arena."""
    orch = _make_orch(qapp)
    orch.finish_roi_selection()
    orch._mw.roi_fitted_circle = (20.0, 20.0, 5.0)  # start a second shape
    orch.finish_roi_selection()
    arena_ids = [s["arena_id"] for s in orch._mw.roi_shapes]
    assert arena_ids == [0, 0]


def test_undo_last_roi_shape_is_scoped_to_the_current_arena(qapp):
    """Undo (Remove Last Zone) must never remove a DIFFERENT arena's shape
    just because it happens to be the list's last entry overall."""
    shapes = [
        {"type": "circle", "params": [1, 1, 1], "mode": "include", "arena_id": 0},
        {"type": "circle", "params": [2, 2, 2], "mode": "include", "arena_id": 0},
        {"type": "circle", "params": [3, 3, 3], "mode": "include", "arena_id": 1},
    ]
    orch = _make_orch(qapp, roi_shapes=list(shapes))
    # Current arena is arena 0 (NOT the list's last shape's arena, which is 1).
    orch.current_arena_id = 0
    orch.undo_last_roi_shape()

    remaining = orch._mw.roi_shapes
    # Arena 0's own last shape (index 1) was removed.
    assert remaining == [shapes[0], shapes[2]]
    # Arena 1's shape is completely untouched, even though it was the list's
    # actual last element before the undo.
    assert shapes[2] in remaining
