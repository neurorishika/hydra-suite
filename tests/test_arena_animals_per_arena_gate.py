"""Task 8: the ``animals_per_arena`` override must stay silent for
single-arena users.

``build_engine_params`` (``trackerkit/engine_params.py``) derives
``animals_per_arena`` as ``_cfg_get(cfg, "animals_per_arena",
default=max_targets)`` -- when the GUI's saved config dict carries no
``animals_per_arena`` key at all, it falls back to the legacy ``max_targets``
value, reproducing today's ``MAX_TARGETS`` exactly. ``ConfigOrchestrator``
must therefore only ever write ``cfg["animals_per_arena"]`` once more than
one arena is actually in use (``n_arenas_from_shapes(roi_shapes) > 1``) --
writing it unconditionally (as a first draft of this task did) would collapse
``MAX_TARGETS`` for every existing single-arena user down to the new
control's default of 1, the exact silent-regression class the task-8 brief's
blocker warning called out ("a silently *undersized* slot count").

This file tests ``n_arenas_from_shapes`` itself -- the pure gating primitive
``ConfigOrchestrator.build_config_dict`` calls directly
(``if n_arenas_from_shapes(self._mw.roi_shapes) > 1:``). The call site in
``build_config_dict`` is a single line that consumes this function verbatim;
it is verified by source-level mutation (see the task-8 report) rather than
an automated test here, since exercising the full method needs dozens of
live panel widgets that cannot be constructed without a real Qt
``MainWindow`` -- which hits an unrelated, pre-existing ``SyntaxError`` in
``trackerkit/gui/dialogs/train_yolo_dialog.py`` on this box's Python 3.10 (see
``tests/test_arena_gui_frame_dimensions.py``).
"""

from __future__ import annotations

from hydra_suite.trackerkit.engine_params import n_arenas_from_shapes


def test_n_arenas_from_shapes_empty_is_one():
    assert n_arenas_from_shapes([]) == 1
    assert n_arenas_from_shapes(None) == 1


def test_n_arenas_from_shapes_legacy_shapes_no_arena_id_is_one():
    shapes = [
        {"type": "circle", "params": [1, 1, 1], "mode": "include"},
        {"type": "circle", "params": [2, 2, 1], "mode": "exclude"},
    ]
    assert n_arenas_from_shapes(shapes) == 1


def test_n_arenas_from_shapes_counts_distinct_include_arena_ids():
    shapes = [
        {"type": "circle", "params": [1, 1, 1], "mode": "include", "arena_id": 0},
        {"type": "circle", "params": [2, 2, 1], "mode": "include", "arena_id": 1},
        {"type": "circle", "params": [3, 3, 1], "mode": "include", "arena_id": 2},
    ]
    assert n_arenas_from_shapes(shapes) == 3


def test_n_arenas_from_shapes_ignores_exclude_only_arena():
    """An exclude-only "arena" renders nothing, so it isn't counted here --
    unlike start_new_arena's id-allocation, which DOES count it to avoid
    colliding ids. The two rules are deliberately different."""
    shapes = [
        {"type": "circle", "params": [1, 1, 1], "mode": "include", "arena_id": 0},
        {"type": "circle", "params": [2, 2, 1], "mode": "exclude", "arena_id": 5},
    ]
    assert n_arenas_from_shapes(shapes) == 1
