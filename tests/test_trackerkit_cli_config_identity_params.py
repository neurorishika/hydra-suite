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


def test_cnn_identity_config_enables_individual_pipeline_and_cnn_classifiers(tmp_path):
    config = _load_fixture_config("ant_cnn_identity")
    params = _build_params(config, tmp_path)

    assert params["ENABLE_IDENTITY_ANALYSIS"]
    assert params["ENABLE_INDIVIDUAL_PIPELINE"]
    assert isinstance(params["CNN_CLASSIFIERS"], list)
    assert len(params["CNN_CLASSIFIERS"]) > 0
    # Each configured classifier keeps its config fields (label, confidence, ...).
    first = params["CNN_CLASSIFIERS"][0]
    assert first.get("label")
    assert "model_path" in first


def test_worm_bgsub_config_leaves_individual_pipeline_disabled(tmp_path):
    config = _load_fixture_config("worm_bgsub")
    params = _build_params(config, tmp_path)

    assert not params.get("ENABLE_IDENTITY_ANALYSIS")
    assert not params.get("ENABLE_INDIVIDUAL_PIPELINE")
    assert not params.get("CNN_CLASSIFIERS")


def test_fly_obb_config_has_no_cnn_classifiers_even_with_individual_pipeline_on(
    tmp_path,
):
    # fly_obb runs YOLO OBB (individual pipeline "on" per the detection-method
    # gate) but has no configured CNN classifiers -- CNN_CLASSIFIERS must stay
    # empty, matching the bridge's behavior for this config.
    config = _load_fixture_config("fly_obb")
    params = _build_params(config, tmp_path)

    assert params["ENABLE_IDENTITY_ANALYSIS"]
    assert params["ENABLE_INDIVIDUAL_PIPELINE"]
    assert not params.get("CNN_CLASSIFIERS")
