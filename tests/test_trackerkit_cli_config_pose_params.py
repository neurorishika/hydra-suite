from __future__ import annotations

import json
from pathlib import Path

from hydra_suite.trackerkit.cli_config import (
    TrackerCliVideoProbe,
    load_tracker_cli_session,
)

FIXTURES_DIR = (
    Path(__file__).resolve().parents[1]
    / "tools"
    / "equivalence"
    / "fixtures"
    / "configs"
)


def _load_fixture_config(name: str) -> dict:
    with open(FIXTURES_DIR / f"{name}.json") as f:
        return json.load(f)


def _build_params(config_data: dict, tmp_path: Path) -> dict:
    session = load_tracker_cli_session(
        str(tmp_path / "video.mp4"),
        config_data=config_data,
        video_probe=TrackerCliVideoProbe(
            fps=30.0, total_frames=100, width=640, height=480
        ),
    )
    return session.params


def test_ant_cnn_identity_config_enables_pose_extractor_and_direction(tmp_path):
    # Root-cause fixture for Task B3: ant_cnn_identity's saved config engages
    # pose-directed heading (worker.py:1667-1720), which the CLI previously
    # never emitted because ENABLE_POSE_EXTRACTOR defaulted False. The bridge
    # runs pose here (gui/orchestrators/config.py:2436-2460); the CLI must
    # match.
    config = _load_fixture_config("ant_cnn_identity")
    params = _build_params(config, tmp_path)

    assert params["ENABLE_POSE_EXTRACTOR"] is True
    assert params["POSE_MODEL_TYPE"] == "sleap"
    assert params["POSE_MODEL_DIR"]
    assert str(params["POSE_MODEL_DIR"]).strip() != ""
    assert params["POSE_DIRECTION_ANTERIOR_KEYPOINTS"] == [
        "left_antenna_tip",
        "right_antenna_tip",
        "left_antenna_elbow",
        "right_antenna_elbow",
        "clypeus",
        "neck",
    ]
    assert params["POSE_DIRECTION_POSTERIOR_KEYPOINTS"] == [
        "petiole_post_petiole",
        "tip_of_gaster",
    ]
    assert params["POSE_IGNORE_KEYPOINTS"] == []
    assert params["POSE_MIN_KPT_CONF_VALID"] == config["pose_min_kpt_conf_valid"]
    assert params["POSE_BATCH_SIZE"] == config["pose_batch_size"]
    assert params["POSE_SLEAP_ENV"] == config["pose_sleap_env"]
    assert params["POSE_SLEAP_BATCH"] == config["pose_sleap_batch"]
    assert params["POSE_SLEAP_MAX_INSTANCES"] == config["pose_sleap_max_instances"]
    assert params["POSE_EXPORTED_MODEL_PATH"] == ""


def test_fly_obb_and_worm_bgsub_derive_pose_extractor_falsy(tmp_path):
    # Two of the five protected clips: neither has "enable_pose_extractor"
    # set, so pose must stay structurally off, keeping direct-path tracking
    # byte-identical to the bridge (pose no-ops on both sides).
    for name in ("fly_obb", "worm_bgsub"):
        config = _load_fixture_config(name)
        params = _build_params(config, tmp_path)
        assert params["ENABLE_POSE_EXTRACTOR"] is False, name
        assert params["POSE_DIRECTION_ANTERIOR_KEYPOINTS"] == [], name
        assert params["POSE_DIRECTION_POSTERIOR_KEYPOINTS"] == [], name


def test_emi_and_sequential_configs_derive_pose_extractor_falsy(tmp_path):
    # The other three protected clips: "pose_model_dir" is populated (a
    # SLEAP model is configured) but "enable_pose_extractor" is False in the
    # saved config, so the bridge's is_pose_export_enabled gate keeps pose
    # off for these too.
    for name in ("emi_obb_identity", "ant_obb_sleap", "ant_obb_sequential"):
        config = _load_fixture_config(name)
        assert config.get("enable_pose_extractor") is False, name
        params = _build_params(config, tmp_path)
        assert params["ENABLE_POSE_EXTRACTOR"] is False, name


def test_pose_skeleton_file_passes_through_from_config(tmp_path):
    # The equivalence runner writes an absolute --skeleton path directly into
    # the config's "pose_skeleton_file" field before either the bridge or the
    # CLI ever sees it (tools/equivalence/runner.py:152-153); the CLI just
    # needs to read it back verbatim.
    config = dict(_load_fixture_config("ant_cnn_identity"))
    config["pose_skeleton_file"] = "/tmp/some_skeleton.json"
    params = _build_params(config, tmp_path)
    assert params["POSE_SKELETON_FILE"] == "/tmp/some_skeleton.json"
