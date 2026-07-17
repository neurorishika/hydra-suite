from __future__ import annotations

import math
import os
import sys
from unittest.mock import MagicMock

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytest.importorskip("PySide6")

from PySide6.QtCore import QPoint, QPointF, Qt  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from hydra_suite.core.background.optimizer import _suggest_trial_params  # noqa: E402
from hydra_suite.trackerkit.gui.dialogs.bg_parameter_helper import (  # noqa: E402
    BgParameterHelperDialog,
)
from hydra_suite.trackerkit.gui.dialogs.parameter_helper import (  # noqa: E402
    ParameterHelperDialog,
)


@pytest.fixture(scope="module")
def qapp() -> QApplication:
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    return app


def test_bg_parameter_helper_exposes_full_bg_tuning_surface(qapp: QApplication) -> None:
    dialog = BgParameterHelperDialog(
        video_path="/tmp/video.mp4",
        current_params={
            "MAX_TARGETS": 5,
            "RESIZE_FACTOR": 0.5,
            "START_FRAME": 10,
            "END_FRAME": 100,
            "DARK_ON_LIGHT_BACKGROUND": True,
        },
    )

    tuning_config = dialog.get_tuning_config()

    expected_keys = {
        "BRIGHTNESS",
        "CONTRAST",
        "GAMMA",
        "DARK_ON_LIGHT_BACKGROUND",
        "BACKGROUND_PRIME_FRAMES",
        "ENABLE_ADAPTIVE_BACKGROUND",
        "BACKGROUND_LEARNING_RATE",
        "ENABLE_LIGHTING_STABILIZATION",
        "LIGHTING_SMOOTH_FACTOR",
        "LIGHTING_MEDIAN_WINDOW",
        "THRESHOLD_VALUE",
        "MORPH_KERNEL_SIZE",
        "MIN_CONTOUR_AREA",
        "MAX_CONTOUR_MULTIPLIER",
        "ENABLE_SIZE_FILTERING",
        "MIN_OBJECT_SIZE",
        "MAX_OBJECT_SIZE",
        "ENABLE_ADDITIONAL_DILATION",
        "DILATION_KERNEL_SIZE",
        "DILATION_ITERATIONS",
        "ENABLE_CONSERVATIVE_SPLIT",
        "CONSERVATIVE_KERNEL_SIZE",
        "CONSERVATIVE_ERODE_ITER",
    }

    assert expected_keys.issubset(tuning_config.keys())
    assert tuning_config["THRESHOLD_VALUE"] is True
    assert tuning_config["MORPH_KERNEL_SIZE"] is True
    assert tuning_config["BRIGHTNESS"] is False
    assert tuning_config["ENABLE_SIZE_FILTERING"] is False

    dialog.close()


class _StubWheelEvent:
    def __init__(self, global_pos: QPoint, delta: int = 120) -> None:
        self._global_pos = global_pos
        self._delta = delta
        self.accepted = False
        self.ignored = False

    def modifiers(self):
        return Qt.ControlModifier

    def angleDelta(self):
        return QPoint(0, self._delta)

    def globalPosition(self):
        return QPointF(self._global_pos)

    def accept(self) -> None:
        self.accepted = True

    def ignore(self) -> None:
        self.ignored = True


def test_bg_parameter_helper_preview_auto_fits_first_frame(
    qapp: QApplication,
) -> None:
    dialog = BgParameterHelperDialog(
        video_path="/tmp/video.mp4",
        current_params={
            "MAX_TARGETS": 5,
            "RESIZE_FACTOR": 1.0,
            "START_FRAME": 0,
            "END_FRAME": 100,
        },
    )
    dialog.resize(1200, 640)
    dialog.show()
    qapp.processEvents()

    frame = np.zeros((1800, 2400, 3), dtype=np.uint8)
    dialog._on_preview_frame_received(0, frame)
    qapp.processEvents()
    qapp.processEvents()

    assert dialog._prev_zoom_slider.value() < 100

    dialog.close()


def test_bg_parameter_helper_preview_frame_receive_renders_once(
    qapp: QApplication,
) -> None:
    dialog = BgParameterHelperDialog(
        video_path="/tmp/video.mp4",
        current_params={
            "MAX_TARGETS": 5,
            "RESIZE_FACTOR": 1.0,
            "START_FRAME": 0,
            "END_FRAME": 100,
        },
    )

    render_spy = MagicMock()
    dialog._display_preview_frame = render_spy

    dialog._on_preview_frame_received(0, np.zeros((32, 32, 3), dtype=np.uint8))

    assert render_spy.call_count == 1

    dialog.close()


def test_bg_parameter_helper_slider_scrub_does_not_render_until_release(
    qapp: QApplication,
) -> None:
    dialog = BgParameterHelperDialog(
        video_path="/tmp/video.mp4",
        current_params={
            "MAX_TARGETS": 5,
            "RESIZE_FACTOR": 1.0,
            "START_FRAME": 0,
            "END_FRAME": 100,
        },
    )
    dialog._prev_frames = [
        np.zeros((16, 16, 3), dtype=np.uint8),
        np.zeros((16, 16, 3), dtype=np.uint8),
        np.zeros((16, 16, 3), dtype=np.uint8),
    ]

    render_spy = MagicMock()
    dialog._display_preview_frame = render_spy

    dialog._on_frame_slider_moved(2)

    assert render_spy.call_count == 0
    assert dialog._frame_label.text() == "3/3"
    assert dialog._frame_slider.hasTracking() is (not sys.platform.startswith("linux"))

    dialog.close()


def test_bg_parameter_helper_wheel_zoom_keeps_cursor_anchor(
    qapp: QApplication,
) -> None:
    dialog = BgParameterHelperDialog(
        video_path="/tmp/video.mp4",
        current_params={
            "MAX_TARGETS": 5,
            "RESIZE_FACTOR": 1.0,
            "START_FRAME": 0,
            "END_FRAME": 100,
        },
    )
    dialog.resize(1200, 640)
    dialog.show()
    qapp.processEvents()

    frame = np.zeros((1200, 1600, 3), dtype=np.uint8)
    dialog._prev_frames = [frame]
    dialog._prev_current_idx = 0
    dialog._prev_zoom_slider.setValue(250)
    qapp.processEvents()

    hbar = dialog._prev_scroll.horizontalScrollBar()
    vbar = dialog._prev_scroll.verticalScrollBar()
    hbar.setValue(hbar.maximum() // 2)
    vbar.setValue(vbar.maximum() // 2)
    previous = (hbar.value(), vbar.value())

    viewport_center = dialog._prev_scroll.viewport().rect().center()
    event = _StubWheelEvent(dialog._prev_scroll.viewport().mapToGlobal(viewport_center))
    dialog._on_prev_wheel(event)
    qapp.processEvents()

    assert event.accepted is True
    assert (hbar.value(), vbar.value()) != previous

    dialog.close()


def test_tracking_parameter_helper_preview_auto_fits_first_frame(
    qapp: QApplication,
) -> None:
    dialog = ParameterHelperDialog(
        video_path="/tmp/video.mp4",
        detection_cache_path="/tmp/cache.npz",
        start_frame=0,
        end_frame=100,
        current_params={
            "REFERENCE_BODY_SIZE": 40.0,
            "RESIZE_FACTOR": 1.0,
            "YOLO_CONFIDENCE_THRESHOLD": 0.5,
        },
    )
    dialog.resize(1200, 740)
    dialog.show()
    qapp.processEvents()

    frame = np.zeros((1800, 2400, 3), dtype=np.uint8)
    dialog._on_preview_frame_received(frame)
    qapp.processEvents()
    qapp.processEvents()

    assert dialog._prev_zoom_slider.value() < 100

    dialog.close()


def test_tracking_parameter_helper_coalesces_preview_frame_renders(
    qapp: QApplication,
) -> None:
    dialog = ParameterHelperDialog(
        video_path="/tmp/video.mp4",
        detection_cache_path="/tmp/cache.npz",
        start_frame=0,
        end_frame=100,
        current_params={
            "REFERENCE_BODY_SIZE": 40.0,
            "RESIZE_FACTOR": 1.0,
            "YOLO_CONFIDENCE_THRESHOLD": 0.5,
        },
    )

    render_spy = MagicMock()
    dialog._display_preview_frame = render_spy

    frame_a = np.zeros((24, 24, 3), dtype=np.uint8)
    frame_b = np.ones((24, 24, 3), dtype=np.uint8)
    dialog._on_preview_frame_received(frame_a)
    dialog._on_preview_frame_received(frame_b)

    assert render_spy.call_count == 0

    qapp.processEvents()

    assert render_spy.call_count == 1
    assert render_spy.call_args[0][0] is frame_b

    dialog.close()


class _FakeTrial:
    def __init__(self) -> None:
        self.values = {
            "ENABLE_ADAPTIVE_BACKGROUND": False,
            "BACKGROUND_LEARNING_RATE": 0.02,
            "ENABLE_LIGHTING_STABILIZATION": False,
            "LIGHTING_SMOOTH_FACTOR": 0.91,
            "LIGHTING_MEDIAN_HALF": 3,
            "ENABLE_SIZE_FILTERING": False,
            "MIN_OBJECT_SIZE_MULTIPLIER": 0.8,
            "MAX_OBJECT_SIZE_MULTIPLIER": 4.2,
            "ENABLE_ADDITIONAL_DILATION": False,
            "DILATION_KERNEL_HALF": 3,
            "DILATION_ITERATIONS": 4,
            "ENABLE_CONSERVATIVE_SPLIT": False,
            "CONSERVATIVE_KERNEL_HALF": 2,
            "CONSERVATIVE_ERODE_ITER": 6,
        }
        self.calls: list[str] = []

    def suggest_int(self, name: str, _low: int, _high: int) -> int:
        self.calls.append(name)
        return int(self.values[name])

    def suggest_float(
        self,
        name: str,
        _low: float,
        _high: float,
        log: bool = False,
    ) -> float:
        self.calls.append(f"{name}|log={log}")
        return float(self.values[name])

    def suggest_categorical(self, name: str, _choices: list[object]) -> object:
        self.calls.append(name)
        return self.values[name]


def test_suggest_trial_params_samples_subparams_even_when_enable_flags_disable_them() -> (
    None
):
    trial = _FakeTrial()
    params = {
        "REFERENCE_BODY_SIZE": 40.0,
        "RESIZE_FACTOR": 0.5,
        "BACKGROUND_LEARNING_RATE": 0.001,
        "LIGHTING_SMOOTH_FACTOR": 0.95,
        "LIGHTING_MEDIAN_WINDOW": 5,
        "MIN_OBJECT_SIZE": 100,
        "MAX_OBJECT_SIZE": 500,
        "ENABLE_ADDITIONAL_DILATION": True,
        "DILATION_KERNEL_SIZE": 3,
        "DILATION_ITERATIONS": 1,
        "ENABLE_CONSERVATIVE_SPLIT": True,
        "CONSERVATIVE_KERNEL_SIZE": 3,
        "CONSERVATIVE_ERODE_ITER": 1,
        "ENABLE_ADAPTIVE_BACKGROUND": True,
        "ENABLE_LIGHTING_STABILIZATION": True,
        "ENABLE_SIZE_FILTERING": True,
        "THRESHOLD_VALUE": 50,
        "MORPH_KERNEL_SIZE": 5,
        "MIN_CONTOUR_AREA": 50,
    }
    tune = {
        "ENABLE_ADAPTIVE_BACKGROUND": True,
        "BACKGROUND_LEARNING_RATE": True,
        "ENABLE_LIGHTING_STABILIZATION": True,
        "LIGHTING_SMOOTH_FACTOR": True,
        "LIGHTING_MEDIAN_WINDOW": True,
        "ENABLE_SIZE_FILTERING": True,
        "MIN_OBJECT_SIZE": True,
        "MAX_OBJECT_SIZE": True,
        "ENABLE_ADDITIONAL_DILATION": True,
        "DILATION_KERNEL_SIZE": True,
        "DILATION_ITERATIONS": True,
        "ENABLE_CONSERVATIVE_SPLIT": True,
        "CONSERVATIVE_KERNEL_SIZE": True,
        "CONSERVATIVE_ERODE_ITER": True,
    }

    trial_params = _suggest_trial_params(trial, tune, params, total_frames=240)

    expected_body_area = (
        math.pi
        * (params["REFERENCE_BODY_SIZE"] / 2.0) ** 2
        * (params["RESIZE_FACTOR"] ** 2)
    )

    assert trial_params["ENABLE_ADAPTIVE_BACKGROUND"] is False
    assert trial_params["BACKGROUND_LEARNING_RATE"] == pytest.approx(0.02)
    assert trial_params["ENABLE_LIGHTING_STABILIZATION"] is False
    assert trial_params["LIGHTING_SMOOTH_FACTOR"] == pytest.approx(0.91)
    assert trial_params["LIGHTING_MEDIAN_WINDOW"] == 7
    assert trial_params["ENABLE_SIZE_FILTERING"] is False
    assert trial_params["MIN_OBJECT_SIZE"] == int(round(0.8 * expected_body_area))
    assert trial_params["MAX_OBJECT_SIZE"] == int(round(4.2 * expected_body_area))
    assert trial_params["ENABLE_ADDITIONAL_DILATION"] is False
    assert trial_params["DILATION_KERNEL_SIZE"] == 7
    assert trial_params["DILATION_ITERATIONS"] == 4
    assert trial_params["ENABLE_CONSERVATIVE_SPLIT"] is False
    assert trial_params["CONSERVATIVE_KERNEL_SIZE"] == 5
    assert trial_params["CONSERVATIVE_ERODE_ITER"] == 6

    assert "BACKGROUND_LEARNING_RATE|log=True" in trial.calls
    assert "MIN_OBJECT_SIZE_MULTIPLIER|log=False" in trial.calls
    assert "BACKGROUND_LEARNING_RATE|log=True" in trial.calls
    assert "MIN_OBJECT_SIZE_MULTIPLIER|log=False" in trial.calls
    assert "BACKGROUND_LEARNING_RATE|log=True" in trial.calls
    assert "MIN_OBJECT_SIZE_MULTIPLIER|log=False" in trial.calls
    assert "BACKGROUND_LEARNING_RATE|log=True" in trial.calls
    assert "MIN_OBJECT_SIZE_MULTIPLIER|log=False" in trial.calls
