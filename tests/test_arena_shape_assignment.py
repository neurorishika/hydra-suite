from hydra_suite.trackerkit.config.schemas import TrackerConfig


def test_animals_per_arena_defaults_to_one():
    assert TrackerConfig().animals_per_arena == 1


def test_animals_per_arena_round_trips():
    cfg = TrackerConfig(animals_per_arena=6)
    assert TrackerConfig.from_dict(cfg.to_dict()).animals_per_arena == 6


def test_legacy_config_without_the_key_loads():
    cfg = TrackerConfig.from_dict({"roi_shapes": [], "current_video_path": ""})
    assert cfg.animals_per_arena == 1


def test_shapes_round_trip_their_arena_id():
    shapes = [
        {"type": "circle", "params": [10, 10, 5], "mode": "include", "arena_id": 0},
        {"type": "circle", "params": [40, 10, 5], "mode": "include", "arena_id": 1},
    ]
    cfg = TrackerConfig(roi_shapes=shapes)
    loaded = TrackerConfig.from_dict(cfg.to_dict())
    assert [s["arena_id"] for s in loaded.roi_shapes] == [0, 1]
