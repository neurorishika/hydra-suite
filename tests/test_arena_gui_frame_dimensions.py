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
``MainWindow``/``panels`` stand-in -- deliberately NOT a real Qt
``MainWindow`` (importing that in this environment hits an unrelated,
pre-existing ``SyntaxError`` in ``trackerkit/gui/dialogs/train_yolo_dialog.py``
that breaks collection of several GUI test files on this box's Python 3.10;
see the module docstring of ``tests/test_get_parameters_dict_characterization.py``
and the task-8 brief). ``ConfigOrchestrator`` itself has no Qt widget
construction in its constructor, so it is safe to instantiate directly with
plain Python doubles that only provide the attributes
``_gui_runtime_context``/``get_parameters_dict`` actually read.
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


def _make_orch(*, video_width, video_height):
    mw = _FakeMainWindow(video_width=video_width, video_height=video_height)
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
