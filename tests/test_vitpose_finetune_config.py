import pytest

from hydra_suite.core.identity.pose.vitpose.training.config import (
    RunConfig,
    validate_run_config,
)


def _good(**over):
    d = dict(
        init_checkpoint="/tmp/x.pth",
        variant="B",
        num_keypoints=6,
        dataset_dir="/tmp/ds",
        output_dir="/tmp/run",
        device="cpu",
        epochs=40,
        batch_size=16,
        lr=5e-4,
        weight_decay=0.1,
        drop_path=0.1,
        sigma=2.0,
        grad_clip=1.0,
        val_fraction=0.2,
        seed=0,
        resume_from=None,
    )
    d.update(over)
    return d


def test_valid_config_roundtrips(tmp_path):
    cfg = validate_run_config(_good())
    assert cfg.variant == "B" and cfg.num_keypoints == 6
    p = tmp_path / "run.json"
    cfg.to_json(p)
    assert RunConfig.from_json(p).num_keypoints == 6


@pytest.mark.parametrize(
    "over",
    [
        {"variant": "b"},  # lowercase rejected
        {"variant": "X"},  # unknown
        {"num_keypoints": 0},  # non-positive
        {"epochs": 0},  # non-positive
        {"val_fraction": 1.5},  # out of range
        {"unknown_key": 1},  # unknown key
    ],
)
def test_bad_config_rejected(over):
    with pytest.raises(ValueError):
        validate_run_config(_good(**over))
