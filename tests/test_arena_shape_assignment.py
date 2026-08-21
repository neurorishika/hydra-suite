from hydra_suite.trackerkit.config.schemas import TrackerConfig


def test_animals_per_arena_defaults_to_one():
    assert TrackerConfig().animals_per_arena == 1


_TWO_ARENA_SHAPES = [
    {"type": "circle", "params": [10, 10, 5], "mode": "include", "arena_id": 0},
    {"type": "circle", "params": [40, 10, 5], "mode": "include", "arena_id": 1},
]


def test_animals_per_arena_round_trips():
    """Only meaningful once >1 arena is in use -- see
    test_animals_per_arena_is_gated_for_single_arena for the single-arena
    side of this same rule (to_dict() must not emit the key then)."""
    cfg = TrackerConfig(animals_per_arena=6, roi_shapes=_TWO_ARENA_SHAPES)
    assert TrackerConfig.from_dict(cfg.to_dict()).animals_per_arena == 6


def test_animals_per_arena_is_gated_for_single_arena():
    """to_dict() must not emit `animals_per_arena` for a single-arena
    project -- the same gate ConfigOrchestrator.build_config_dict applies to
    the GUI's own config dict (task-8 fix round 1, M5). Otherwise this
    dataclass would be a loaded gun if ever serialized directly into an
    engine-params config: a stray non-default `animals_per_arena` on an
    otherwise single-arena project would defeat build_engine_params'
    fallback-to-`max_targets` safety net."""
    cfg = TrackerConfig(animals_per_arena=6)  # roi_shapes=[] -> single arena
    assert "animals_per_arena" not in cfg.to_dict()
    assert TrackerConfig.from_dict(cfg.to_dict()).animals_per_arena == 1


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
