"""Pins the Task-8 blocker fix: the GUI must supply REAL frame dimensions to
``RuntimeContext`` so ``build_arena_labels`` can actually rasterize arenas.

Before the fix, ``ConfigOrchestrator._gui_runtime_context`` (in
``trackerkit/gui/orchestrators/config.py``) hardcoded
``frame_width=None, frame_height=None``. ``build_arena_labels``
(``trackerkit/engine_params.py``) short-circuits to ``(None, 1)`` whenever
either dimension is falsy -- so every GUI run silently produced
``N_ARENAS=1``, ``ARENA_LABELS=None`` regardless of how many arenas were
drawn in the ROI shapes. There is no single-arena regression to catch this
because ``animals_per_arena`` defaults to the legacy ``max_targets``, so the
bug is invisible unless a multi-arena project is actually run.

This test builds a ``ConfigOrchestrator`` against a bare-bones fake
``MainWindow``/``panels`` stand-in rather than a real Qt ``MainWindow`` --
``ConfigOrchestrator`` itself has no Qt widget construction in its
constructor, so plain Python doubles that only provide the attributes
``_gui_runtime_context``/``get_parameters_dict`` actually read are enough,
and it keeps this file's fixtures minimal and fast. (An earlier draft of
this file claimed a real ``MainWindow`` could not be constructed here at all
due to a pre-existing ``SyntaxError`` in
``trackerkit/gui/dialogs/train_yolo_dialog.py`` -- that claim was wrong: it
was an artifact of running pytest under the wrong interpreter (base conda,
Python 3.10) instead of this project's ``hydra-mps`` environment (Python
3.13, where the file's PEP 701 f-string is valid). Under
``conda run -n hydra-mps env PYTHONPATH=$PWD/src QT_QPA_PLATFORM=offscreen
python -m pytest`` a real ``MainWindow`` constructs fine, as
``tests/test_gui_cli_param_equivalence.py`` already demonstrates. The fakes
here are a legitimate lightweight choice for a narrowly-scoped unit test,
not a workaround for a real blocker.)
"""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest

pytest.importorskip("PySide6")

from hydra_suite.trackerkit.gui.orchestrators.config import (  # noqa: E402
    ConfigOrchestrator,
)

TWO_ARENA_SHAPES = [
    {"type": "circle", "params": [25, 25, 15], "mode": "include", "arena_id": 0},
    {"type": "circle", "params": [75, 25, 15], "mode": "include", "arena_id": 1},
]


class _FakeSpin:
    def __init__(self, value=0):
        self._value = value

    def value(self):
        return self._value


class _FakeSetupPanel:
    def __init__(self):
        self.spin_start_frame = _FakeSpin(0)
        self.spin_end_frame = _FakeSpin(99)


class _FakeIdentityPanel:
    def _get_selected_yolo_headtail_model_path(self):
        return ""


class _FakePanels:
    def __init__(self):
        self.setup = _FakeSetupPanel()
        self.identity = _FakeIdentityPanel()


class _FakeMainWindow:
    """Just enough surface for ``_gui_runtime_context`` to run."""

    def __init__(self, *, video_width, video_height, roi_mask=None):
        self.current_video_path = ""
        self.video_total_frames = 100
        self.video_width = video_width
        self.video_height = video_height
        self.roi_mask = roi_mask
        self._individual_dataset_run_id = ""
        self.current_individual_properties_cache_path = ""
        self.advanced_config = {}


def _make_orch(*, video_width, video_height, roi_mask=None):
    mw = _FakeMainWindow(
        video_width=video_width, video_height=video_height, roi_mask=roi_mask
    )
    panels = _FakePanels()
    return ConfigOrchestrator(mw, config={}, panels=panels)


def test_gui_runtime_context_supplies_real_frame_dimensions():
    """The blocker fix: RuntimeContext.frame_width/height must be the live
    video dimensions, not the hardcoded ``None`` that silently disabled
    ``build_arena_labels`` for every GUI run."""
    orch = _make_orch(video_width=320, video_height=240)
    ctx = orch._gui_runtime_context({"roi_shapes": TWO_ARENA_SHAPES, "fps": 30.0})
    assert (ctx.frame_width, ctx.frame_height) == (320, 240)


def test_gui_runtime_context_drives_real_arena_rasterization():
    """End-to-end: with real dimensions supplied, build_engine_params must
    actually rasterize N_ARENAS > 1 and a non-None ARENA_LABELS for a
    multi-arena config -- not the (None, 1) short-circuit."""
    from hydra_suite.trackerkit.engine_params import build_engine_params

    orch = _make_orch(video_width=320, video_height=240)
    config = {
        "roi_shapes": TWO_ARENA_SHAPES,
        "fps": 30.0,
        "animals_per_arena": 3,
        "detection_method": "background_subtraction",
    }
    params = build_engine_params(
        config, runtime=orch._gui_runtime_context(config), advanced_config={}
    )
    assert params["N_ARENAS"] == 2
    assert params["ARENA_LABELS"] is not None
    assert isinstance(params["ARENA_LABELS"], np.ndarray)
    assert params["ARENA_LABELS"].shape == (240, 320)
    assert params["MAX_TARGETS"] == 2 * 3


def test_stale_roi_mask_now_rasterizes_with_real_dimensions():
    """Deliberate side effect of the blocker fix (fix round 1, I4): once a
    video is loaded, ``self._mw.roi_mask`` being ``None`` while
    ``roi_shapes`` is populated (e.g. a config loaded when its saved video
    path didn't exist yet, followed by the user manually loading a video
    later -- ``_setup_video_file`` never re-derives ``roi_mask`` in that
    case) now correctly rasterizes ``ROI_MASK`` via
    ``build_engine_params``'s own internal ``build_roi_mask`` fallback,
    which reads the same real ``runtime.frame_width``/``frame_height`` the
    blocker fix supplies. Before this fix round that fallback always got
    ``None, None`` too (dead), so ``ROI_MASK`` stayed ``None`` forever in
    this state -- silently disabling ROI gating. This is a genuine GUI/CLI
    parity fix (the CLI has always fed this fallback real dimensions); see
    ``ConfigOrchestrator._gui_runtime_context``'s docstring for the full
    trace. Single-arena shapes here -- this must NOT also flip on arena
    gating (N_ARENAS stays 1)."""
    from hydra_suite.trackerkit.engine_params import build_engine_params, build_roi_mask

    single_arena_shapes = [
        {"type": "circle", "params": [25, 25, 15], "mode": "include"},
    ]
    orch = _make_orch(video_width=320, video_height=240, roi_mask=None)
    config = {
        "roi_shapes": single_arena_shapes,
        "fps": 30.0,
        "detection_method": "background_subtraction",
    }
    params = build_engine_params(
        config, runtime=orch._gui_runtime_context(config), advanced_config={}
    )
    assert params["N_ARENAS"] == 1
    assert params["ROI_MASK"] is not None
    expected = build_roi_mask(single_arena_shapes, 320, 240)
    assert np.array_equal(params["ROI_MASK"], expected)


# ---------------------------------------------------------------------------
# I2 (fix round 1): the PRODUCER half of the blocker fix -- session.py's
# SessionOrchestrator._init_video_player writing self._mw.video_width/height
# -- had no test of its own. Every test above sets those attributes directly
# on a fake MainWindow, which only pins the CONSUMER read in
# ConfigOrchestrator._gui_runtime_context. Reviewer's mutation, that
# producer site alone: deleting both `self._mw.video_width = width` /
# `self._mw.video_height = height` lines left all 18 pre-round-1 tests
# passing. This test drives the real SessionOrchestrator._init_video_player
# against a tiny real cv2-written video file, so it fails if those two
# lines are removed.
# ---------------------------------------------------------------------------


class _FakeValueSpin:
    """Minimal numeric spinbox double: real ``.value()``, no-op the rest.

    ``_init_video_player`` -> ``_update_range_info``/``_sync_trail_history_
    bounds`` do real arithmetic on several spinboxes' ``.value()`` (e.g.
    ``end - start + 1``, ``num_frames / fps``) -- a bare ``MagicMock``'s
    auto-vivified ``.value()`` return is not a number, so those call sites
    would raise ``TypeError``. This fake is the smallest thing that behaves
    correctly there while no-op'ing everything else the method calls.
    """

    def __init__(self, value):
        self._value = value

    def value(self):
        return self._value

    def setValue(self, v):
        self._value = v

    def maximum(self):
        return 999999

    def setMaximum(self, v):
        pass

    def setRange(self, lo, hi):
        pass

    def setEnabled(self, v):
        pass

    def blockSignals(self, b):
        pass


def _write_tiny_video(path, *, width, height, n_frames=3, fps=10.0):
    import cv2

    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    for _ in range(n_frames):
        writer.write(frame)
    writer.release()


def test_init_video_player_writes_real_frame_dimensions_onto_main_window(tmp_path):
    from unittest.mock import MagicMock

    from PySide6.QtWidgets import QApplication

    from hydra_suite.trackerkit.gui.orchestrators.session import SessionOrchestrator

    # _init_video_player ends with QTimer.singleShot(0, ...), which needs a
    # live QCoreApplication instance to schedule against safely.
    app = QApplication.instance()
    if app is None:
        app = QApplication([])

    video_path = tmp_path / "tiny.mp4"
    _write_tiny_video(video_path, width=64, height=48)

    mw = MagicMock()
    mw.video_cap = None
    mw.playback_timer = None
    # Read via getattr(..., 0) > 0 in _sync_trail_history_bounds -- must be
    # a real int, not an auto-vivified MagicMock (which does not support >).
    mw.video_total_frames = 0

    setup = MagicMock()
    setup.spin_traj_hist = _FakeValueSpin(30)
    setup.spin_start_frame = _FakeValueSpin(0)
    setup.spin_end_frame = _FakeValueSpin(0)
    setup.spin_fps = _FakeValueSpin(10.0)
    panels = MagicMock()
    panels.setup = setup

    orch = SessionOrchestrator(mw, config=MagicMock(), panels=panels)
    try:
        orch._init_video_player(str(video_path))
        assert mw.video_width == 64
        assert mw.video_height == 48
    finally:
        # Release the real cv2.VideoCapture _init_video_player opened
        # (stored on a MagicMock, so nothing else would ever release it)
        # before interpreter teardown.
        cap = mw.video_cap
        if cap is not None and hasattr(cap, "release"):
            cap.release()


# ---------------------------------------------------------------------------
# Fix wave 8, round 2, finding 2: loading a saved config replaces roi_shapes
# wholesale (``self._mw.roi_shapes = cfg.get("roi_shapes", [])`` in
# ``ConfigOrchestrator._load_config_individual_analysis``), which must also
# flip the arena panel's ``made_via_grid`` to False -- otherwise reopening
# the grid dialog to "modify" arenas loaded from a config file would
# silently regenerate/replace them from a stale grid definition that has
# nothing to do with what was actually loaded. Driven against a REAL
# ArenaPanel (not a MagicMock stand-in for ``panels.arena``), per the
# reviewer's note that a mocked arena panel hides broken wiring. Everything
# else this large method touches (identity/tracking/dataset/setup panel
# widgets) is stubbed with a permissive fake since none of it is under
# test here.
# ---------------------------------------------------------------------------


class _AnyWidget:
    """A widget double that accepts any call/attribute and stays chainable.

    ``findText`` must return an int (compared with ``>= 0``/``max(0, ...)``
    in the loader), ``isChecked``/``currentText`` must return the right
    type -- everything else (including arbitrarily nested sub-widgets, e.g.
    ``identity.g_identity.setChecked(...)``) is a harmless no-op that stays
    an ``_AnyWidget`` all the way down, and is also directly callable so a
    chain like ``.setChecked(...)`` resolves to nothing.
    """

    def __init__(self):
        self.__dict__["_children"] = {}

    def findText(self, *a, **k):
        return 0

    def isChecked(self, *a, **k):
        return False

    def currentText(self, *a, **k):
        return ""

    def __call__(self, *a, **k):
        return None

    def __getattr__(self, name):
        children = self.__dict__["_children"]
        if name not in children:
            children[name] = _AnyWidget()
        return children[name]


class _AutoPanel:
    """Any attribute access yields a permissive widget double (cached per
    name, so repeated access -- e.g. set then read -- sees the same
    object)."""

    def __init__(self):
        self.__dict__["_children"] = {}

    def __getattr__(self, name):
        children = self.__dict__["_children"]
        if name not in children:
            children[name] = _AnyWidget()
        return children[name]


def _make_get_cfg(cfg):
    def get_cfg(*keys, default=None):
        for key in keys:
            if key in cfg:
                return cfg[key]
        return default

    return get_cfg


def test_config_load_marks_hand_drawn_even_if_previously_grid_generated(tmp_path):
    from unittest.mock import MagicMock

    from hydra_suite.trackerkit.gui.orchestrators.config import ConfigOrchestrator
    from hydra_suite.trackerkit.gui.panels.arena_panel import ArenaPanel

    video_path = tmp_path / "tiny.mp4"
    _write_tiny_video(video_path, width=64, height=48)

    loaded_shapes = [
        {"type": "circle", "params": [10, 10, 5], "mode": "include", "arena_id": 0},
    ]
    cfg = {"roi_shapes": loaded_shapes, "file_path": str(video_path)}

    mw = MagicMock()
    mw.advanced_config = {}

    arena_panel = ArenaPanel()
    arena_panel.set_frame_size(64, 48)
    # Pretend the PREVIOUS in-session arena set was grid-generated, so the
    # config load below has something stale to clear.
    arena_panel.mark_grid_generated({"first_arena_id": 0})
    assert arena_panel.made_via_grid is True

    panels = _AutoPanel()
    panels.arena = arena_panel

    orch = ConfigOrchestrator(mw, config={}, panels=panels)
    orch._load_config_individual_analysis(cfg, _make_get_cfg(cfg))

    assert mw.roi_shapes == loaded_shapes
    assert arena_panel.made_via_grid is False
