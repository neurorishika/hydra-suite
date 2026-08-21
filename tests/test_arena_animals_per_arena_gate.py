"""Task 8: the ``animals_per_arena`` override must stay silent for
single-arena users.

``build_engine_params`` (``trackerkit/engine_params.py``) derives
``animals_per_arena`` as ``_cfg_get(cfg, "animals_per_arena",
default=max_targets)`` -- when the GUI's saved config dict carries no
``animals_per_arena`` key at all, it falls back to the legacy ``max_targets``
value, reproducing today's ``MAX_TARGETS`` exactly. ``ConfigOrchestrator``
must therefore only ever write ``cfg["animals_per_arena"]`` once more than
one arena is actually in use (``n_arenas_from_shapes(roi_shapes) > 1``) --
writing it unconditionally (as an earlier draft of this task did, before
"Animals per arena" was unified onto the single ``spin_max_targets`` control
in fix round 1 / C1) would collapse ``MAX_TARGETS`` for every existing
single-arena user down to ``TrackerConfig.animals_per_arena``'s bare
dataclass default of 1, the exact silent-regression class the task-8 brief's
blocker warning called out ("a silently *undersized* slot count").

This file first tests ``n_arenas_from_shapes`` itself -- the pure gating
primitive both ``ConfigOrchestrator.build_config_dict`` (the GUI's own ad-hoc
config dict) and ``TrackerConfig.to_dict`` (the typed schema, fix round 1 /
M5) call directly. It then drives the real, offscreen ``MainWindow`` +
``ConfigOrchestrator.build_config_dict`` (fix round 1 / I3) to prove the
call site itself -- not just the primitive it calls -- actually gates the
write. (An earlier draft of this file claimed a real ``MainWindow`` could
not be constructed here due to a pre-existing ``SyntaxError`` in
``trackerkit/gui/dialogs/train_yolo_dialog.py``; that was wrong -- an
artifact of running under the wrong interpreter, see
``tests/test_arena_gui_frame_dimensions.py``'s docstring for the
correction.)
"""

from __future__ import annotations

import json
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

pytest.importorskip("PySide6")

from PySide6.QtWidgets import QApplication  # noqa: E402

from hydra_suite.trackerkit.config.schemas import TrackerConfig  # noqa: E402
from hydra_suite.trackerkit.engine_params import n_arenas_from_shapes  # noqa: E402
from hydra_suite.trackerkit.gui.main_window import MainWindow  # noqa: E402

_TWO_ARENA_SHAPES = [
    {"type": "circle", "params": [1, 1, 1], "mode": "include", "arena_id": 0},
    {"type": "circle", "params": [2, 2, 1], "mode": "include", "arena_id": 1},
]


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


# ---------------------------------------------------------------------------
# I3 (fix round 1): the gate's CALL SITE in build_config_dict, not just the
# n_arenas_from_shapes primitive it calls. Reviewer's mutation, that call
# site alone: removing the `if n_arenas_from_shapes(...) > 1:` guard (writing
# cfg["animals_per_arena"] unconditionally) left all 32 pre-round-1 arena
# tests passing, because none of them exercised the real, panel-heavy
# build_config_dict end to end.
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def qapp() -> QApplication:
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


@pytest.fixture
def main_window(monkeypatch, qapp):
    """A real, offscreen ``MainWindow`` (same pattern as
    ``tests/test_config_build_dict.py``/``tests/test_gui_cli_param_
    equivalence.py``): the advanced-config disk hooks are stubbed, but the
    widget tree -- including ``spin_max_targets`` -- is real."""
    monkeypatch.setattr(MainWindow, "_save_advanced_config", lambda self: None)
    monkeypatch.setattr(MainWindow, "_load_advanced_config", lambda self: {})
    window = MainWindow()
    try:
        yield window
    finally:
        window.close()


def test_build_config_dict_omits_animals_per_arena_for_single_arena(main_window):
    main_window.roi_shapes = []
    main_window._setup_panel.spin_max_targets.setValue(20)

    cfg = main_window._config_orch.build_config_dict()

    assert cfg["max_targets"] == 20
    assert "animals_per_arena" not in cfg


def test_build_config_dict_emits_animals_per_arena_for_multi_arena(main_window):
    main_window.roi_shapes = list(_TWO_ARENA_SHAPES)
    main_window._setup_panel.spin_max_targets.setValue(20)

    cfg = main_window._config_orch.build_config_dict()

    assert cfg["max_targets"] == 20
    assert cfg["animals_per_arena"] == 20


# ---------------------------------------------------------------------------
# M5 (fix round 1): TrackerConfig.to_dict() must apply the SAME gate, so the
# typed schema can never become a "loaded gun" if ever serialized directly
# into an engine-params config, bypassing build_config_dict's own gate.
# ---------------------------------------------------------------------------


def test_tracker_config_to_dict_omits_animals_per_arena_for_single_arena():
    cfg = TrackerConfig(animals_per_arena=6)  # roi_shapes=[] -> single arena
    assert "animals_per_arena" not in cfg.to_dict()


def test_tracker_config_to_dict_emits_animals_per_arena_for_multi_arena():
    cfg = TrackerConfig(animals_per_arena=6, roi_shapes=_TWO_ARENA_SHAPES)
    assert cfg.to_dict()["animals_per_arena"] == 6


# ---------------------------------------------------------------------------
# M6 (fix round 1): confirm the save/load asymmetry the reviewer flagged
# ("a single-arena user who sets the count to 7 has it dropped on save and
# reset to 1 on load") dissolves under C1's single control. With ONE control
# (spin_max_targets, relabeled "Animals per arena"), the value always
# round-trips via the pre-existing, unconditional cfg["max_targets"] --
# there is no longer a second, gated `animals_per_arena` key a single-arena
# user's value could get dropped into or reset from.
# ---------------------------------------------------------------------------


def test_single_arena_animal_count_round_trips_through_save_and_load(
    main_window, tmp_path
):
    main_window.roi_shapes = []  # single arena
    main_window._setup_panel.spin_max_targets.setValue(7)

    cfg = main_window._config_orch.build_config_dict()
    assert cfg["max_targets"] == 7
    assert "animals_per_arena" not in cfg  # single-arena: gate stays closed

    config_path = tmp_path / "roundtrip.json"
    config_path.write_text(json.dumps(cfg))

    fresh = MainWindow()
    try:
        fresh._config_orch._load_config_from_file(str(config_path))
        assert fresh._setup_panel.spin_max_targets.value() == 7
    finally:
        fresh.close()


# ---------------------------------------------------------------------------
# Fix round 2, Finding 1 (Important): the solver auto-pick
# (_resolve_solver_flags, called from build_config_dict for
# enable_greedy_assignment/enable_spatial_optimization) must see the DERIVED
# total slot count (n_arenas * animals_per_arena), not the per-arena spinbox
# value directly -- otherwise a many-arena session with few animals per
# arena silently auto-picks the small-N solver branch while the engine
# actually allocates a large total (exactly the N > 50 Python-cost-matrix-
# fallback regime a many-arena session is likely to hit).
# ---------------------------------------------------------------------------


def _make_n_arena_shapes(n_arenas):
    return [
        {"type": "circle", "params": [i, i, 1], "mode": "include", "arena_id": i}
        for i in range(n_arenas)
    ]


def _clear_solver_overrides(main_window):
    """The shipped default.json preset (loaded on every MainWindow startup)
    carries explicit `enable_greedy_assignment`/`enable_spatial_optimization:
    false` -- a "saved override" that always wins over auto-pick. Clearing
    it here is what actually exercises the auto-pick branch these tests are
    about (pre-existing app behavior, unrelated to this fix)."""
    main_window._tracking_panel._enable_greedy_override = None
    main_window._tracking_panel._enable_spatial_override = None


def test_solver_flags_use_derived_total_not_per_arena_count(main_window):
    """10 arenas x 6 animals/arena = 60 total (>= the 50-target autopick
    threshold) must resolve the SAME solver flags as a single-arena config
    whose max_targets is entered directly as 60 -- not the per-arena
    spinbox value of 6, which alone would stay under the threshold."""
    _clear_solver_overrides(main_window)
    main_window.roi_shapes = _make_n_arena_shapes(10)
    main_window._setup_panel.spin_max_targets.setValue(
        6
    )  # per-arena; derived total = 60

    cfg = main_window._config_orch.build_config_dict()

    assert cfg["animals_per_arena"] == 6
    assert cfg["enable_greedy_assignment"] is True
    assert cfg["enable_spatial_optimization"] is True

    # Equivalent single-arena config with the same DERIVED total entered directly.
    main_window.roi_shapes = []
    main_window._setup_panel.spin_max_targets.setValue(60)
    _clear_solver_overrides(main_window)
    equivalent_cfg = main_window._config_orch.build_config_dict()

    assert equivalent_cfg["enable_greedy_assignment"] == cfg["enable_greedy_assignment"]
    assert (
        equivalent_cfg["enable_spatial_optimization"]
        == cfg["enable_spatial_optimization"]
    )


def test_solver_flags_stay_false_below_threshold_for_multi_arena(main_window):
    """10 arenas x 4 animals/arena = 40 total (< the 50-target threshold)
    must NOT auto-pick greedy/spatial -- confirms the derived-total fix
    doesn't just always return True, only when the real total crosses the
    threshold."""
    _clear_solver_overrides(main_window)
    main_window.roi_shapes = _make_n_arena_shapes(10)
    main_window._setup_panel.spin_max_targets.setValue(4)  # derived total = 40

    cfg = main_window._config_orch.build_config_dict()

    assert cfg["enable_greedy_assignment"] is False
    assert cfg["enable_spatial_optimization"] is False
