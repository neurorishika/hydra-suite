from __future__ import annotations

import os
import sys
from types import SimpleNamespace

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytest.importorskip("PySide6")

from PySide6.QtWidgets import QApplication  # noqa: E402

from hydra_suite.core.tracking.optimization.optimizer import (  # noqa: E402
    _PARAM_RANGES,
    OptimizationResult,
)
from hydra_suite.trackerkit.gui.autotune_contract import (  # noqa: E402
    TRACKING_AUTOTUNE_CANDIDATE_KEYS,
    AutotuneCandidateApplicationError,
    applicable_candidate_params,
    apply_tracking_autotune_candidate,
)
from hydra_suite.trackerkit.gui.dialogs.parameter_helper import (  # noqa: E402
    ParameterHelperDialog,
)
from hydra_suite.trackerkit.gui.orchestrators.config import (  # noqa: E402
    ConfigOrchestrator,
)


class _Spin:
    def __init__(self, minimum: float, maximum: float, value: float = 0.0) -> None:
        self._minimum = minimum
        self._maximum = maximum
        self._value = value

    def minimum(self) -> float:
        return self._minimum

    def maximum(self) -> float:
        return self._maximum

    def setValue(self, value: float) -> None:
        self._value = value

    def value(self) -> float:
        return self._value


def _panels(*, fps: float = 120.0) -> SimpleNamespace:
    return SimpleNamespace(
        setup=SimpleNamespace(spin_fps=_Spin(1.0, 240.0, fps)),
        detection=SimpleNamespace(
            spin_yolo_confidence=_Spin(0.01, 1.0),
            spin_yolo_iou=_Spin(0.01, 1.0),
        ),
        tracking=SimpleNamespace(
            spin_max_dist=_Spin(0.1, 20.0),
            spin_Wp=_Spin(0.0, 10.0),
            spin_Wo=_Spin(0.0, 10.0),
            spin_Wa=_Spin(0.0, 2.0),
            spin_Wasp=_Spin(0.0, 10.0),
            spin_kalman_noise=_Spin(0.0, 1.0),
            spin_kalman_meas=_Spin(0.0, 1.0),
            spin_kalman_damping=_Spin(0.5, 0.999),
            spin_kalman_longitudinal_noise=_Spin(0.1, 20.0),
            spin_kalman_initial_velocity_retention=_Spin(0.0, 1.0),
            spin_kalman_maturity_age=_Spin(0.001, 8.0),
            spin_lost_thresh=_Spin(0.001, 40.0),
        ),
    )


def test_autotune_contract_covers_every_core_tunable() -> None:
    assert set(TRACKING_AUTOTUNE_CANDIDATE_KEYS) == set(_PARAM_RANGES)
    assert applicable_candidate_params(
        {
            "KALMAN_INITIAL_VELOCITY_RETENTION": 0.6,
            "CORE_ONLY_PARAMETER": 2.0,
            "MAX_DISTANCE_THRESHOLD": 100.0,
        }
    ) == {"KALMAN_INITIAL_VELOCITY_RETENTION": 0.6}


def test_orchestrator_applies_candidate_velocity_and_converts_frame_units() -> None:
    panels = _panels(fps=120.0)

    orchestrator = ConfigOrchestrator(
        main_window=object(), config=object(), panels=panels
    )
    orchestrator._apply_optimized_params(
        {
            "W_AREA": 1.75,
            "KALMAN_INITIAL_VELOCITY_RETENTION": 0.65,
            "KALMAN_MATURITY_AGE": 18,
            "LOST_THRESHOLD_FRAMES": 24,
        }
    )

    assert panels.tracking.spin_Wa.value() == 1.75
    assert panels.tracking.spin_kalman_initial_velocity_retention.value() == 0.65
    assert panels.tracking.spin_kalman_maturity_age.value() == pytest.approx(0.15)
    assert panels.tracking.spin_lost_thresh.value() == pytest.approx(0.2)


def test_apply_candidate_rejects_out_of_range_value_without_partial_write() -> None:
    panels = _panels()
    with pytest.raises(AutotuneCandidateApplicationError, match="W_AREA"):
        apply_tracking_autotune_candidate(
            {
                "KALMAN_INITIAL_VELOCITY_RETENTION": 0.9,
                "W_AREA": 2.01,
            },
            panels,
        )

    assert panels.tracking.spin_kalman_initial_velocity_retention.value() == 0.0
    assert panels.tracking.spin_Wa.value() == 0.0


def test_apply_candidate_rejects_non_numeric_value_without_partial_write() -> None:
    panels = _panels()
    with pytest.raises(AutotuneCandidateApplicationError, match="not numeric"):
        apply_tracking_autotune_candidate(
            {
                "W_POSITION": 2.0,
                "LOST_THRESHOLD_FRAMES": None,
            },
            panels,
        )

    assert panels.tracking.spin_Wp.value() == 0.0


@pytest.fixture(scope="module")
def qapp() -> QApplication:
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    return app


def test_state_key_changes_for_objective_and_search_configuration(
    qapp: QApplication, tmp_path
) -> None:
    cache_path = tmp_path / "optimizer-cache"
    cache_path.mkdir()
    dialog = ParameterHelperDialog(
        video_path="/tmp/video.mp4",
        detection_cache_path=str(cache_path),
        start_frame=10,
        end_frame=100,
        current_params={
            "REFERENCE_BODY_SIZE": 40.0,
            "YOLO_CONFIDENCE_THRESHOLD": 0.25,
        },
    )
    try:
        baseline = dialog._compute_state_key()
        dialog.spin_w_coverage.setValue(dialog.spin_w_coverage.value() + 0.05)
        assert dialog._compute_state_key() != baseline

        dialog.spin_w_coverage.setValue(dialog.spin_w_coverage.value() - 0.05)
        baseline = dialog._compute_state_key()
        dialog.cb_kalman_init_vel.setChecked(not dialog.cb_kalman_init_vel.isChecked())
        assert dialog._compute_state_key() != baseline

        dialog.cb_kalman_init_vel.setChecked(not dialog.cb_kalman_init_vel.isChecked())
        baseline = dialog._compute_state_key()
        dialog.spin_trials.setValue(dialog.spin_trials.value() + 1)
        assert dialog._compute_state_key() != baseline

        baseline = dialog._compute_state_key()
        (cache_path / "detection.npz").write_bytes(b"rebuilt cache")
        assert dialog._compute_state_key() != baseline
    finally:
        dialog.close()


def test_selected_params_contain_only_applyable_candidate_overrides(
    qapp: QApplication, tmp_path
) -> None:
    dialog = ParameterHelperDialog(
        video_path="/tmp/video.mp4",
        detection_cache_path=str(tmp_path / "detection.npz"),
        start_frame=0,
        end_frame=10,
        current_params={"KALMAN_MAX_VELOCITY_MULTIPLIER": 2.0},
    )
    try:
        dialog.results = [
            OptimizationResult(
                params={
                    "KALMAN_INITIAL_VELOCITY_RETENTION": 0.4,
                    "KALMAN_YOUNG_GATE_MULTIPLIER": 3.0,
                    "MAX_DISTANCE_THRESHOLD": 100.0,
                },
                score=0.1,
                trial_number=1,
            )
        ]
        dialog._selected_row_to_apply = 0

        assert dialog.get_selected_params() == {
            "KALMAN_INITIAL_VELOCITY_RETENTION": 0.4
        }
    finally:
        dialog.close()
