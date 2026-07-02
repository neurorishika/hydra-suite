"""Tests for PoseKitConfig schema."""


def test_posekit_config_defaults():
    from hydra_suite.posekit.config.schemas import PoseKitConfig

    cfg = PoseKitConfig()
    assert cfg.mode == "frame"
    assert cfg.show_predictions is True
    assert cfg.show_pred_conf is False
    assert cfg.sleap_env_path == ""
    assert cfg.autosave_delay_ms == 3000


def test_posekit_config_round_trip():
    from hydra_suite.posekit.config.schemas import PoseKitConfig

    cfg = PoseKitConfig(
        mode="keypoint", sleap_env_path="/opt/sleap", autosave_delay_ms=5000
    )
    restored = PoseKitConfig.from_dict(cfg.to_dict())
    assert restored.mode == "keypoint"
    assert restored.sleap_env_path == "/opt/sleap"
    assert restored.autosave_delay_ms == 5000


def test_posekit_config_defaults_frame_mode_off():
    from hydra_suite.posekit.config.schemas import PoseKitConfig

    cfg = PoseKitConfig()
    assert cfg.frame_mode is False


def test_posekit_config_round_trip_frame_mode():
    from hydra_suite.posekit.config.schemas import PoseKitConfig

    cfg = PoseKitConfig(frame_mode=True)
    restored = PoseKitConfig.from_dict(cfg.to_dict())
    assert restored.frame_mode is True
