"""Contract tests for the crop-padding retirement (spec 2026-08-18)."""

from pathlib import Path

from hydra_suite.core.individual.classification.apriltag import AprilTagConfig


def test_apriltag_config_reads_its_own_key_default_zero():
    cfg = AprilTagConfig.from_params({})
    assert cfg.padding_fraction == 0.0


def test_apriltag_config_ignores_individual_crop_padding():
    cfg = AprilTagConfig.from_params({"INDIVIDUAL_CROP_PADDING": 0.5})
    assert cfg.padding_fraction == 0.0


def test_apriltag_config_honours_apriltag_crop_padding():
    cfg = AprilTagConfig.from_params({"APRILTAG_CROP_PADDING": 0.25})
    assert cfg.padding_fraction == 0.25


REPO = Path(__file__).resolve().parent.parent
FIXTURE = REPO / "tools" / "equivalence" / "fixtures" / "configs" / "fly_obb.json"


def _params(cfg_overrides):
    """Build engine params from a minimal config, the Qt-free way.

    Mirrors tests/test_get_parameters_dict_characterization.py:262-282.
    """
    from hydra_suite.trackerkit import cli_config
    from hydra_suite.trackerkit.engine_params import RuntimeContext, build_engine_params

    cfg = cli_config.load_tracker_cli_config(str(FIXTURE))
    cfg.update(cfg_overrides)
    rt = RuntimeContext(fps=100.0, total_frames=500, frame_width=640, frame_height=480)
    return build_engine_params(cfg, runtime=rt)


def test_engine_params_emit_apriltag_crop_padding():
    assert _params({"apriltag_crop_padding": 0.2})["APRILTAG_CROP_PADDING"] == 0.2
