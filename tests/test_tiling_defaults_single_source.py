"""F4/F8: one defaults table feeds every SAHI dataclass default."""

from hydra_suite.core.inference.config import SliceConfig, _slice_config_from_params
from hydra_suite.core.inference.semantic.tiling import SEMANTIC_TILE_FRACTION_SEED
from hydra_suite.detectkit.config.training import SliceTrainingConfig
from hydra_suite.detectkit.gui.models import SliceTrainingSettings
from hydra_suite.training.contracts import Sam3LoraParams
from hydra_suite.utils.tiling_spec import BACKEND_DEFAULTS, DEFAULT_OVERLAP


def test_inference_defaults_come_from_the_table():
    assert (
        SliceConfig().object_tile_fraction
        == BACKEND_DEFAULTS["yolo_infer"].object_tile_fractions[0]
        == 0.15
    )
    cfg = _slice_config_from_params({}, "SLICE_", reference_body_px=0.0)
    assert cfg.object_tile_fraction == 0.15
    assert cfg.overlap_width_ratio == DEFAULT_OVERLAP == 0.2


def test_training_scale_set_defaults_come_from_one_table():
    """F4: GUI and headless resolve the same set, both from the table."""
    expected = list(BACKEND_DEFAULTS["yolo_train"].object_tile_fractions)
    assert SliceTrainingSettings().target_fractions() == expected
    assert list(SliceTrainingConfig().target_fractions()) == expected
    assert list(SliceTrainingConfig().target_size_fractions) == expected
    # derived, same values
    assert SliceTrainingSettings().target_sizes == [32.0, 64.0, 96.0, 128.0]


def test_constructor_pixels_still_win_over_the_default():
    """Plan review M1: a non-empty fractions default would have ignored these."""
    assert SliceTrainingSettings(target_sizes=[200.0]).target_fractions() == [
        200.0 / 640.0
    ]


def test_sam3_defaults_come_from_the_table():
    assert Sam3LoraParams.__dataclass_fields__["object_tile_fraction"].default == 0.055
    assert SEMANTIC_TILE_FRACTION_SEED == 0.05  # escalation seed, deviation 15


def test_legacy_project_without_fractions_keeps_its_scales():
    """Review Focus 2: an old project (pixel target_sizes, no fractions key)."""
    old = SliceTrainingSettings.from_dict({"target_sizes": [64.0, 128.0]})
    assert old.target_fractions() == [0.1, 0.2]
