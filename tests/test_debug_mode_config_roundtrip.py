from hydra_suite.trackerkit.config.schemas import TrackerConfig


def test_debug_mode_defaults_false_and_roundtrips():
    cfg = TrackerConfig()
    assert cfg.debug_mode is False
    cfg.debug_mode = True
    restored = TrackerConfig.from_dict(cfg.to_dict())
    assert restored.debug_mode is True
