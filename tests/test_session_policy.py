"""Pure policy predicates agree with the GUI widget-reading methods across all fixture configs."""

import json
import os
import pathlib

import pytest

from hydra_suite.core.tracking import session_policy as sp

FIXTURES = sorted(
    (
        pathlib.Path(__file__).resolve().parents[1]
        / "tools"
        / "equivalence"
        / "fixtures"
        / "configs"
    ).glob("*.json")
)

PREDICATES = [
    "is_individual_pipeline_enabled",
    "is_pose_inference_enabled",
    "is_headtail_compute_enabled",
    "should_export_final_canonical_images",
    "should_export_final_media_videos",
    "should_run_interpolated_postpass",
    "is_pose_export_enabled",
]


@pytest.mark.parametrize("cfg_path", FIXTURES, ids=lambda p: p.stem)
def test_predicates_callable_and_boolean(cfg_path):
    cfg = json.loads(cfg_path.read_text())
    for name in PREDICATES:
        assert isinstance(getattr(sp, name)(cfg), bool), name
    assert sp.workflow_mode_key(cfg) in ("realtime", "non_realtime")


def _base_config(**overrides):
    """Base config for unit-level postpass trigger tests."""
    cfg = {
        "detection_method": "yolo_obb",
        "individual_interpolate_occlusions": True,
        "enable_individual_dataset": False,
        "enable_pose_extractor": False,
        "final_media_export_videos_enabled": False,
        "cnn_classifiers": [],
        "use_apriltags": False,
        "enable_headtail_orientation": False,
        "yolo_headtail_model_path": "",
    }
    cfg.update(overrides)
    return cfg


def test_postpass_triggers_on_cnn_classifiers_alone():
    cfg = _base_config(cnn_classifiers=[{"model_path": "x.pt", "label": "id"}])
    assert sp.should_run_interpolated_postpass(cfg) is True


def test_postpass_triggers_on_apriltags_alone():
    cfg = _base_config(use_apriltags=True)
    assert sp.should_run_interpolated_postpass(cfg) is True


def test_postpass_triggers_on_headtail_alone():
    cfg = _base_config(
        enable_headtail_orientation=True,
        yolo_headtail_model_path="/tmp/headtail.pt",
    )
    assert sp.should_run_interpolated_postpass(cfg) is True


def test_postpass_false_when_nothing_enabled():
    cfg = _base_config()
    assert sp.should_run_interpolated_postpass(cfg) is False


def test_fly_obb_expected_values():
    cfg = json.loads((FIXTURES[0].parent / "fly_obb.json").read_text())
    # fly_obb: yolo_obb detection, no pose extractor, no identity.
    assert sp.is_individual_pipeline_enabled(cfg) is True
    assert sp.is_pose_export_enabled(cfg) is False
    assert sp.is_pose_inference_enabled(cfg) is False
    # NOTE: the fly_obb fixture has "realtime_tracking_mode": true (verified in
    # tools/equivalence/fixtures/configs/fly_obb.json), so the correct expected
    # value is "realtime", not "non_realtime" as the task brief's test literal
    # claimed. workflow_mode_key is a straight pass-through of that flag.
    assert sp.workflow_mode_key(cfg) == "realtime"


# Independent, value-pinned oracle for the 5 predicates that delegate through the
# *full* build_config_dict() (is_pose_export_enabled, is_pose_inference_enabled,
# is_headtail_compute_enabled, should_export_final_media_videos,
# should_run_interpolated_postpass). The offscreen-MainWindow agreement test below
# compares sp.fn(build_config_dict()) against a GUI method that IS sp.fn(build_config_dict())
# for these 5 -- that only proves determinism, not independent correctness. This table
# was derived by reading each fixture's config semantics directly (not by running the
# pure predicates), so it locks the key mapping against future regressions.
#
# should_export_final_media_videos is False for every current fixture (none enable
# final_media_export_videos_enabled), so its True branch is exercised only by unit-level
# logic elsewhere, not by this fixture table.
#
# NOTE: After Task 5 (postpass-trigger completeness fix), postpass now also triggers
# on CNN classifiers, AprilTags, and head-tail compute being enabled (not just on the
# three export modes). The oracle has been updated to reflect this.
ORACLE = {
    "ant_pose_headtail": {
        "is_pose_export_enabled": True,
        "is_pose_inference_enabled": True,
        "is_headtail_compute_enabled": True,
        "should_export_final_media_videos": False,
        "should_run_interpolated_postpass": True,
    },
    "ant_cnn_identity": {
        "is_pose_export_enabled": True,
        "is_pose_inference_enabled": True,
        "is_headtail_compute_enabled": True,
        "should_export_final_media_videos": False,
        "should_run_interpolated_postpass": True,
    },
    "ant_cnn_identity_marked": {
        "is_pose_export_enabled": True,
        "is_pose_inference_enabled": True,
        "is_headtail_compute_enabled": True,
        "should_export_final_media_videos": False,
        "should_run_interpolated_postpass": True,
    },
    # Identical to "ant_cnn_identity" except "enable_tracklet_relinking":
    # true (Task 8) -- none of these 5 predicates read that key, so the
    # oracle is the same.
    "ant_cnn_identity_relink": {
        "is_pose_export_enabled": True,
        "is_pose_inference_enabled": True,
        "is_headtail_compute_enabled": True,
        "should_export_final_media_videos": False,
        "should_run_interpolated_postpass": True,
    },
    "ant_obb_sleap": {
        "is_pose_export_enabled": False,
        "is_pose_inference_enabled": False,
        "is_headtail_compute_enabled": True,
        "should_export_final_media_videos": False,
        "should_run_interpolated_postpass": True,
    },
    "ant_obb_sequential": {
        "is_pose_export_enabled": False,
        "is_pose_inference_enabled": False,
        "is_headtail_compute_enabled": True,
        "should_export_final_media_videos": False,
        "should_run_interpolated_postpass": True,
    },
    "emi_obb_identity": {
        "is_pose_export_enabled": False,
        "is_pose_inference_enabled": False,
        "is_headtail_compute_enabled": True,
        "should_export_final_media_videos": False,
        "should_run_interpolated_postpass": True,
    },
    "fly_obb": {
        "is_pose_export_enabled": False,
        "is_pose_inference_enabled": False,
        "is_headtail_compute_enabled": False,
        "should_export_final_media_videos": False,
        "should_run_interpolated_postpass": False,
    },
    "worm_bgsub": {
        "is_pose_export_enabled": False,
        "is_pose_inference_enabled": False,
        "is_headtail_compute_enabled": False,
        "should_export_final_media_videos": False,
        "should_run_interpolated_postpass": False,
    },
    "worm_bgsub_scaled": {
        "is_pose_export_enabled": False,
        "is_pose_inference_enabled": False,
        "is_headtail_compute_enabled": False,
        "should_export_final_media_videos": False,
        "should_run_interpolated_postpass": False,
    },
}


@pytest.mark.parametrize("cfg_path", FIXTURES, ids=lambda p: p.stem)
def test_full_dict_predicates_match_pinned_oracle(cfg_path):
    """Independent oracle for the 5 predicates that delegate via build_config_dict().

    Loads the raw fixture JSON directly (no MainWindow involved) and checks each
    predicate against a hand-derived expected value, so a bug shared between the
    predicate and any GUI delegate would still be caught here.
    """
    cfg = json.loads(cfg_path.read_text())
    expected = ORACLE[cfg_path.stem]
    for name, expected_value in expected.items():
        assert getattr(sp, name)(cfg) is expected_value, f"{name} for {cfg_path.stem}"


def test_is_pose_inference_enabled_ignores_stale_inactive_backend_slot():
    """A stale/leftover model path in a non-active backend slot must not enable pose.

    build_config_dict() writes a per-backend slot for every backend (pose_yolo_model_dir,
    pose_sleap_model_dir, pose_vitpose_model_dir) independently of which backend is
    active; only pose_model_dir reflects the active backend's resolved path. A saved
    config can therefore have a populated slot for an inactive backend while the active
    backend's own slot (and pose_model_dir) is empty -- that must read as pose inference
    disabled, not enabled.
    """
    cfg = {
        "detection_method": "yolo_obb",
        "enable_pose_extractor": True,
        "pose_model_type": "sleap",
        "pose_model_dir": "",
        "pose_sleap_model_dir": "",
        "pose_yolo_model_dir": "YOLO-pose/x.pt",
    }
    assert sp.is_pose_inference_enabled(cfg) is False


def test_is_pose_inference_enabled_true_when_active_backend_model_set():
    cfg = {
        "detection_method": "yolo_obb",
        "enable_pose_extractor": True,
        "pose_model_type": "sleap",
        "pose_model_dir": "pose/SLEAP/model",
        "pose_sleap_model_dir": "",
        "pose_yolo_model_dir": "YOLO-pose/x.pt",
    }
    assert sp.is_pose_inference_enabled(cfg) is True


def test_build_trajectory_colors_pinned_values():
    assert sp.build_trajectory_colors(3) == [
        (102, 179, 92),
        (14, 106, 71),
        (188, 20, 102),
    ]


def test_build_trajectory_colors_does_not_leak_global_seed():
    import numpy as np

    state_before = np.random.get_state()
    sp.build_trajectory_colors(5)
    state_after = np.random.get_state()
    # The saved/restored guard must leave the global RNG state exactly as found.
    assert state_before[0] == state_after[0]
    assert (state_before[1] == state_after[1]).all()


os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

try:
    import PySide6  # noqa: F401

    _HAS_QT = True
except ImportError:
    _HAS_QT = False


if _HAS_QT:
    from PySide6.QtWidgets import QApplication

    from hydra_suite.trackerkit.gui.main_window import MainWindow

    @pytest.fixture(scope="module")
    def qapp() -> QApplication:
        app = QApplication.instance()
        if app is None:
            app = QApplication([])
        return app

    @pytest.fixture
    def qtbot_config_stub(monkeypatch, qapp):
        """A real, offscreen MainWindow (no disk/network I/O), per test_config_build_dict.py."""
        monkeypatch.setattr(MainWindow, "_save_advanced_config", lambda self: None)
        monkeypatch.setattr(MainWindow, "_load_advanced_config", lambda self: {})
        window = MainWindow()
        try:
            yield window
        finally:
            window.close()

    GUI_METHOD_MAP = {
        "is_individual_pipeline_enabled": "_is_individual_pipeline_enabled",
        "is_pose_inference_enabled": "_is_pose_inference_enabled",
        "is_headtail_compute_enabled": "_is_headtail_compute_enabled",
        "should_export_final_canonical_images": "_should_export_final_canonical_images",
        "should_export_final_media_videos": "_should_export_final_media_videos",
        "should_run_interpolated_postpass": "_should_run_interpolated_postpass",
        "is_pose_export_enabled": "_is_pose_export_enabled",
        "workflow_mode_key": "_workflow_mode_key",
    }

    @pytest.mark.parametrize("cfg_path", FIXTURES, ids=lambda p: p.stem)
    def test_predicates_agree_with_gui(qtbot_config_stub, cfg_path):
        window = qtbot_config_stub
        window._config_orch._load_config_from_file(str(cfg_path))
        cfg = window._config_orch.build_config_dict()

        for pure_name, gui_name in GUI_METHOD_MAP.items():
            pure_fn = getattr(sp, pure_name)
            if gui_name == "_is_pose_export_enabled":
                gui_value = window._is_pose_export_enabled()
            else:
                gui_value = getattr(window._session_orch, gui_name)()
            assert (
                pure_fn(cfg) == gui_value
            ), f"{pure_name} disagrees with {gui_name} for {cfg_path.stem}"
