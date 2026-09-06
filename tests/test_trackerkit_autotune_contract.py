from __future__ import annotations

import os
import sys
from types import SimpleNamespace

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytest.importorskip("PySide6")

from PySide6.QtWidgets import QApplication, QDoubleSpinBox, QLabel  # noqa: E402

from hydra_suite.core.tracking.optimization.optimizer import (  # noqa: E402
    _PARAM_RANGES,
    OptimizationResult,
    TrackingOptimizerCore,
)
from hydra_suite.core.tracking.optimization.parameter_contract import (  # noqa: E402
    tracking_autotune_parameter,
)
from hydra_suite.core.tracking.optimization.production_replay import (  # noqa: E402
    ProductionReplayEvaluator,
)
from hydra_suite.trackerkit.config.schemas import TrackerConfig  # noqa: E402
from hydra_suite.trackerkit.engine_params import (  # noqa: E402
    RuntimeContext,
    build_engine_params,
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
from hydra_suite.trackerkit.gui.orchestrators import (  # noqa: E402
    config as config_module,
)
from hydra_suite.trackerkit.gui.orchestrators.config import (  # noqa: E402
    ConfigOrchestrator,
)
from hydra_suite.trackerkit.gui.panels.detection_panel import (  # noqa: E402
    DetectionPanel,
)
from hydra_suite.trackerkit.gui.panels.tracking_panel import TrackingPanel  # noqa: E402


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


class _TrackingPanelHost:
    """Minimal MainWindow surface needed to construct a real TrackingPanel."""

    def _set_compact_scroll_layout(self, _layout) -> None:
        pass

    def _set_compact_section_widget(self, _widget) -> None:
        pass

    def _create_help_label(self, text: str, **_kwargs) -> QLabel:
        return QLabel(text)

    def _remember_collapsible_state(self, *_args) -> None:
        pass

    def _open_parameter_helper(self) -> None:
        pass


class _DetectionPanelHost(_TrackingPanelHost):
    """Minimal MainWindow surface needed to construct a real DetectionPanel."""

    def __init__(self) -> None:
        self.advanced_config: dict[str, object] = {}

    def _open_bg_parameter_helper(self) -> None:
        pass

    def _gpu_fast_obb_is_coreml_only(self) -> bool:
        return False

    def _update_obb_mode_warning(self) -> None:
        pass

    def _auto_set_body_size_from_detection(self) -> None:
        pass

    def _auto_set_aspect_ratio_from_detection(self) -> None:
        pass

    def _auto_set_margin_from_detection(self) -> None:
        pass


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


def _qt_spin(
    minimum: float, maximum: float, decimals: int, value: float = 0.0
) -> QDoubleSpinBox:
    spin = QDoubleSpinBox()
    spin.setRange(minimum, maximum)
    spin.setDecimals(decimals)
    spin.setValue(value)
    return spin


def _qt_panels(*, fps: float) -> SimpleNamespace:
    """Real QDoubleSpinBoxes with TrackerKit's production precisions."""

    return SimpleNamespace(
        setup=SimpleNamespace(spin_fps=_qt_spin(1.0, 240.0, 2, fps)),
        detection=SimpleNamespace(
            spin_yolo_confidence=_qt_spin(0.01, 1.0, 2),
            spin_yolo_iou=_qt_spin(0.01, 1.0, 2),
        ),
        tracking=SimpleNamespace(
            spin_max_dist=_qt_spin(0.1, 20.0, 2),
            spin_Wp=_qt_spin(0.0, 10.0, 2),
            spin_Wo=_qt_spin(0.0, 10.0, 2),
            spin_Wa=_qt_spin(0.0, 2.0, 4),
            spin_Wasp=_qt_spin(0.0, 10.0, 2),
            spin_kalman_noise=_qt_spin(0.0, 1.0, 4),
            spin_kalman_meas=_qt_spin(0.0, 1.0, 4),
            spin_kalman_damping=_qt_spin(0.5, 0.999, 3),
            spin_kalman_longitudinal_noise=_qt_spin(0.1, 20.0, 1),
            spin_kalman_initial_velocity_retention=_qt_spin(0.0, 1.0, 2),
            spin_kalman_maturity_age=_qt_spin(0.001, 8.0, 4),
            spin_lost_thresh=_qt_spin(0.001, 40.0, 4),
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


def test_orchestrator_normalizes_production_cache_member_path(
    monkeypatch, tmp_path
) -> None:
    cache_member = tmp_path / "production-cache" / "detection.npz"
    main_window = SimpleNamespace(current_detection_cache_path=str(cache_member))
    panels = SimpleNamespace(
        setup=SimpleNamespace(csv_line=SimpleNamespace(text=lambda: ""))
    )
    orchestrator = ConfigOrchestrator(
        main_window=main_window,
        config=object(),
        panels=panels,
    )
    monkeypatch.setattr(
        config_module,
        "detection_cache_dir_covers_range",
        lambda *_args, **_kwargs: True,
    )

    path, already_valid = orchestrator._find_or_plan_optimizer_cache_path(
        "video.mp4", {}, 0, 10
    )

    assert already_valid is True
    assert path == str(cache_member.parent)


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


def test_apply_candidate_uses_real_qt_precision_before_write(
    qapp: QApplication,
) -> None:
    panels = _qt_panels(fps=240.0)

    apply_tracking_autotune_candidate(
        {
            "YOLO_CONFIDENCE_THRESHOLD": 0.995,
            "YOLO_IOU_THRESHOLD": 0.125,
            "W_AREA": 0.12345678,
            "KALMAN_DAMPING": 0.91234,
            "KALMAN_LONGITUDINAL_NOISE_MULTIPLIER": 5.44,
        },
        panels,
    )

    # These are actual QDoubleSpinBox round trips, not mock rounding. In
    # particular Qt represents 0.995 as 0.99 at two decimals.
    assert panels.detection.spin_yolo_confidence.value() == 0.99
    assert panels.detection.spin_yolo_iou.value() == 0.13
    assert panels.tracking.spin_Wa.value() == 0.1235
    assert panels.tracking.spin_kalman_damping.value() == 0.912
    assert panels.tracking.spin_kalman_longitudinal_noise.value() == 5.4


def test_longitudinal_candidate_replay_matches_applied_engine_params(tmp_path) -> None:
    """Candidate replay must use the same long-only Kalman change as the GUI."""

    base_params = {
        "MAX_TARGETS": 1,
        "REFERENCE_BODY_SIZE": 12.0,
        "RESIZE_FACTOR": 0.5,
        "MAX_DISTANCE_MULTIPLIER": 3.0,
        "MAX_DISTANCE_THRESHOLD": 18.0,
        "KALMAN_LONGITUDINAL_NOISE_MULTIPLIER": 5.0,
        "KALMAN_LATERAL_NOISE_MULTIPLIER": 0.2,
        "KALMAN_ANISOTROPY_RATIO": 25.0,
    }
    candidate = {
        "MAX_DISTANCE_MULTIPLIER": 1.234,
        "MAX_DISTANCE_THRESHOLD": 999.0,
        "KALMAN_LONGITUDINAL_NOISE_MULTIPLIER": 7.04,
        "KALMAN_ANISOTROPY_RATIO": 999.0,
    }
    optimizer = TrackingOptimizerCore(
        "clip.mp4",
        str(tmp_path / "cache"),
        0,
        1,
        base_params,
        {
            "MAX_DISTANCE_MULTIPLIER": True,
            "KALMAN_LONGITUDINAL_NOISE_MULTIPLIER": True,
        },
    )

    lightweight_params = optimizer._candidate_evaluation_params(candidate)
    effective_keys = (
        "KALMAN_LONGITUDINAL_NOISE_MULTIPLIER",
        "KALMAN_LATERAL_NOISE_MULTIPLIER",
        "KALMAN_ANISOTROPY_RATIO",
    )
    expected = {
        "KALMAN_LONGITUDINAL_NOISE_MULTIPLIER": 7.0,
        "KALMAN_LATERAL_NOISE_MULTIPLIER": 0.2,
        "KALMAN_ANISOTROPY_RATIO": 35.0,
    }
    assert {key: lightweight_params[key] for key in effective_keys} == expected
    assert lightweight_params["MAX_DISTANCE_THRESHOLD"] == pytest.approx(7.38)

    captured: dict[str, object] = {}

    class _CapturingEngine:
        def __init__(self, _video_path, **kwargs) -> None:
            self.kwargs = kwargs

        def set_parameters(self, params) -> None:
            captured.update(params)

        def run_tracking(self) -> None:
            self.kwargs["on_finished"](True, [], [])

    replay = ProductionReplayEvaluator(
        "clip.mp4",
        str(tmp_path / "cache"),
        0,
        1,
        engine_factory=_CapturingEngine,
    )
    assert replay.run(lightweight_params).success
    assert {key: captured[key] for key in effective_keys} == expected

    panels = _panels()
    panels.tracking._kalman_lateral_noise_multiplier = 0.2
    apply_tracking_autotune_candidate(candidate, panels)
    # Applying the row does not mutate the hidden lateral setting. The next
    # production engine build must reproduce both replay's ratio and axes.
    assert panels.tracking._kalman_lateral_noise_multiplier == 0.2
    rebuilt = build_engine_params(
        {
            "frame_width": 640,
            "frame_height": 480,
            "reference_body_size": base_params["REFERENCE_BODY_SIZE"],
            "max_assignment_distance_multiplier": panels.tracking.spin_max_dist.value(),
            "kalman_longitudinal_noise_multiplier": panels.tracking.spin_kalman_longitudinal_noise.value(),
            "kalman_lateral_noise_multiplier": panels.tracking._kalman_lateral_noise_multiplier,
        },
        runtime=RuntimeContext(
            fps=30.0, total_frames=2, frame_width=640, frame_height=480
        ),
    )
    assert {key: rebuilt[key] for key in effective_keys} == expected


def test_real_tracking_panel_matches_shared_precision_contract(
    qapp: QApplication,
) -> None:
    """Exercise production TrackerKit controls, not only precision-matched mocks."""

    panel = TrackingPanel(_TrackingPanelHost(), TrackerConfig())
    try:
        widgets = {
            "MAX_DISTANCE_MULTIPLIER": "spin_max_dist",
            "W_POSITION": "spin_Wp",
            "W_ORIENTATION": "spin_Wo",
            "W_AREA": "spin_Wa",
            "W_ASPECT": "spin_Wasp",
            "KALMAN_NOISE_COVARIANCE": "spin_kalman_noise",
            "KALMAN_MEASUREMENT_NOISE_COVARIANCE": "spin_kalman_meas",
            "KALMAN_DAMPING": "spin_kalman_damping",
            "KALMAN_LONGITUDINAL_NOISE_MULTIPLIER": "spin_kalman_longitudinal_noise",
            "KALMAN_INITIAL_VELOCITY_RETENTION": "spin_kalman_initial_velocity_retention",
            "KALMAN_MATURITY_AGE": "spin_kalman_maturity_age",
            "LOST_THRESHOLD_FRAMES": "spin_lost_thresh",
        }
        for key, widget_name in widgets.items():
            assert (
                getattr(panel, widget_name).decimals()
                == tracking_autotune_parameter(key).widget_decimals
            )

        tooltip = panel.spin_kalman_longitudinal_noise.toolTip()
        assert "lateral multiplier stays fixed" in tooltip
        assert "anisotropy changes" in tooltip
        assert "locked" not in tooltip

        apply_tracking_autotune_candidate(
            {"W_POSITION": 0.995, "W_AREA": 0.12345678},
            SimpleNamespace(tracking=panel),
        )
        assert panel.spin_Wp.value() == 0.99
        assert panel.spin_Wa.value() == 0.1235
    finally:
        panel.close()


def test_kalman_autotune_copy_explains_retained_lateral_semantics(
    qapp: QApplication, tmp_path
) -> None:
    dialog = ParameterHelperDialog(
        video_path="/tmp/video.mp4",
        detection_cache_path=str(tmp_path / "cache"),
        start_frame=0,
        end_frame=10,
        current_params={
            "KALMAN_LONGITUDINAL_NOISE_MULTIPLIER": 5.0,
            "KALMAN_LATERAL_NOISE_MULTIPLIER": 0.1,
            "KALMAN_ANISOTROPY_RATIO": 50.0,
        },
    )
    try:
        tooltip = dialog.cb_kalman_long_noise.toolTip()
        assert "lateral multiplier stays fixed" in tooltip
        assert "anisotropy" in tooltip
        assert "Lateral noise is derived automatically" not in tooltip
    finally:
        dialog.close()


def test_real_detection_panel_matches_shared_threshold_precision_contract(
    qapp: QApplication,
) -> None:
    panel = DetectionPanel(_DetectionPanelHost(), TrackerConfig())
    try:
        assert (
            panel.spin_yolo_confidence.decimals()
            == tracking_autotune_parameter("YOLO_CONFIDENCE_THRESHOLD").widget_decimals
        )
        assert (
            panel.spin_yolo_iou.decimals()
            == tracking_autotune_parameter("YOLO_IOU_THRESHOLD").widget_decimals
        )
    finally:
        panel.close()


@pytest.mark.parametrize(
    ("fps", "maturity_frames", "lost_frames"),
    [(30.0, 25, 19), (120.0, 18, 84), (240.0, 7, 9_600)],
)
def test_lifecycle_candidate_round_trips_qt_seconds_to_engine_frames(
    qapp: QApplication,
    fps: float,
    maturity_frames: int,
    lost_frames: int,
) -> None:
    panels = _qt_panels(fps=fps)

    apply_tracking_autotune_candidate(
        {
            "KALMAN_MATURITY_AGE": maturity_frames,
            "LOST_THRESHOLD_FRAMES": lost_frames,
        },
        panels,
    )

    assert (
        round(panels.tracking.spin_kalman_maturity_age.value() * fps) == maturity_frames
    )
    assert round(panels.tracking.spin_lost_thresh.value() * fps) == lost_frames


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

        baseline = dialog._compute_state_key()
        dialog.base_params["MAX_TARGETS"] = 4
        assert dialog._compute_state_key() != baseline

        dialog.base_params["MAX_TARGETS"] = 1
        dialog.base_params["ASSOCIATION_STAGE1_MOTION_GATE_MULTIPLIER"] = 1.0
        baseline = dialog._compute_state_key()
        dialog.base_params["ASSOCIATION_STAGE1_MOTION_GATE_MULTIPLIER"] = 1.5
        assert dialog._compute_state_key() != baseline
    finally:
        dialog.close()


def test_state_key_hashes_ndarray_content_and_fixed_replay_settings(
    qapp: QApplication, tmp_path
) -> None:
    cache_path = tmp_path / "optimizer-cache"
    cache_path.mkdir()
    roi_mask = np.array([[0, 1], [1, 0]], dtype=np.uint8)
    dialog = ParameterHelperDialog(
        video_path="/tmp/video.mp4",
        detection_cache_path=str(cache_path / "detection.npz"),
        start_frame=0,
        end_frame=20,
        current_params={
            "MAX_TARGETS": 2,
            "ROI_MASK": roi_mask,
            "ENABLE_CONFIDENCE_DENSITY_MAP": True,
            "DENSITY_TEMPORAL_SIGMA": 2.0,
            "ENABLE_POSE_EXTRACTOR": False,
        },
    )
    try:
        baseline = dialog._compute_state_key()
        roi_mask[0, 0] = 1
        assert dialog._compute_state_key() != baseline

        baseline = dialog._compute_state_key()
        dialog.base_params["DENSITY_TEMPORAL_SIGMA"] = 3.0
        assert dialog._compute_state_key() != baseline
    finally:
        dialog.close()


def test_state_persists_across_fresh_dialog_without_hashing_its_sidecar(
    qapp: QApplication, tmp_path
) -> None:
    cache_path = tmp_path / "optimizer-cache"
    cache_path.mkdir()
    (cache_path / "detection.npz").write_bytes(b"raw cache contents")
    current_params = {"MAX_TARGETS": 2, "REFERENCE_BODY_SIZE": 40.0}
    dialog = ParameterHelperDialog(
        video_path="/tmp/video.mp4",
        detection_cache_path=str(cache_path),
        start_frame=0,
        end_frame=20,
        current_params=current_params,
    )
    try:
        dialog.results = [
            OptimizationResult(params={"W_POSITION": 1.25}, score=0.2, trial_number=1)
        ]
        key_before_save = dialog._compute_state_key()
        dialog._save_state()
        assert dialog._compute_state_key() == key_before_save

        restored = ParameterHelperDialog(
            video_path="/tmp/video.mp4",
            detection_cache_path=str(cache_path / "detection.npz"),
            start_frame=0,
            end_frame=20,
            current_params=current_params,
        )
        try:
            assert len(restored.results) == 1
            assert restored.results[0].params == {"W_POSITION": 1.25}
            assert "Restored 1 cached result" in restored.status_label.text()
        finally:
            restored.close()
    finally:
        dialog.close()


def test_background_source_does_not_offer_inert_yolo_dimensions(
    qapp: QApplication, tmp_path
) -> None:
    dialog = ParameterHelperDialog(
        video_path="/tmp/video.mp4",
        detection_cache_path=str(tmp_path / "cache"),
        start_frame=0,
        end_frame=10,
        current_params={"DETECTION_METHOD": "background_subtraction"},
    )
    try:
        assert dialog.cb_conf.parent() is None
        assert dialog.cb_iou.parent() is None
        assert dialog.get_tuning_config()["YOLO_CONFIDENCE_THRESHOLD"] is False
        assert dialog.get_tuning_config()["YOLO_IOU_THRESHOLD"] is False
    finally:
        dialog.close()


def test_yolo_replay_disables_unfaithful_threshold_dimensions(
    qapp: QApplication, tmp_path
) -> None:
    dialog = ParameterHelperDialog(
        video_path="/tmp/video.mp4",
        detection_cache_path=str(tmp_path / "cache"),
        start_frame=0,
        end_frame=10,
        current_params={
            "DETECTION_METHOD": "yolo_obb",
            "CNN_CLASSIFIERS": [{"model_path": "identity.pt"}],
        },
    )
    try:
        assert dialog.cb_conf.isEnabled() is False
        assert dialog.cb_iou.isEnabled() is False
        assert dialog.cb_conf.isChecked() is False
        assert "cannot faithfully" in dialog.cb_conf.toolTip()
        assert dialog.get_tuning_config()["YOLO_CONFIDENCE_THRESHOLD"] is False
    finally:
        dialog.close()


class _RunningWorker:
    def __init__(self, *, stop_finishes: bool = True) -> None:
        self.stop_calls = 0
        self.wait_calls: list[int] = []
        self.running = True
        self.stop_finishes = stop_finishes

    def isRunning(self) -> bool:
        return self.running

    def stop(self) -> None:
        self.stop_calls += 1

    def wait(self, timeout_ms: int) -> bool:
        self.wait_calls.append(timeout_ms)
        if self.stop_finishes:
            self.running = False
        return self.stop_finishes


def test_terminal_dialog_paths_stop_workers_with_bounded_wait(
    qapp: QApplication, tmp_path
) -> None:
    dialog = ParameterHelperDialog(
        video_path="/tmp/video.mp4",
        detection_cache_path=str(tmp_path / "cache"),
        start_frame=0,
        end_frame=10,
        current_params={},
    )
    optimizer = _RunningWorker()
    preview = _RunningWorker()
    dialog.optimizer = optimizer
    dialog.preview_worker = preview

    dialog.reject()

    assert optimizer.stop_calls == preview.stop_calls == 1
    assert (
        optimizer.wait_calls == preview.wait_calls == [dialog._WORKER_SHUTDOWN_WAIT_MS]
    )

    dialog.close()


@pytest.mark.parametrize("terminal_action", ["reject", "done"])
def test_noncooperative_worker_keeps_terminal_dialog_open_until_safe(
    qapp: QApplication, tmp_path, terminal_action: str
) -> None:
    """A failed bounded stop must not hide a dialog that still owns a worker."""

    dialog = ParameterHelperDialog(
        video_path="/tmp/video.mp4",
        detection_cache_path=str(tmp_path / "cache"),
        start_frame=0,
        end_frame=10,
        current_params={},
    )
    optimizer = _RunningWorker(stop_finishes=False)
    dialog.optimizer = optimizer
    dialog.show()
    qapp.processEvents()

    (
        getattr(dialog, terminal_action)(0)
        if terminal_action == "done"
        else getattr(dialog, terminal_action)()
    )

    assert dialog.isVisible()
    assert dialog._terminal_shutdown_started is False
    assert optimizer.stop_calls >= 1
    assert optimizer.wait_calls
    assert set(optimizer.wait_calls) == {dialog._WORKER_SHUTDOWN_WAIT_MS}

    # A worker that finishes after the initial bounded wait must still be able
    # to update the dialog: the failed terminal request did not suppress its
    # eventual completion callback.
    dialog.on_finished()
    assert "Search finished" in dialog.status_label.text()

    optimizer.stop_finishes = True
    (
        getattr(dialog, terminal_action)(0)
        if terminal_action == "done"
        else getattr(dialog, terminal_action)()
    )

    assert dialog._terminal_shutdown_started is True
    assert dialog.isVisible() is False


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
