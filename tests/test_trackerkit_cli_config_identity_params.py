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


def test_ant_cnn_identity_config_enables_identity_in_tracking_block(tmp_path):
    # This is the root-cause fixture for Task B2: ant_cnn_identity's saved
    # config engages the Bayesian identity-cost term in the Hungarian
    # assigner. Every key in the identity-in-tracking block (bridge
    # gui/orchestrators/config.py:2400-2436) must be derived from the config,
    # not left at the CLI's previous permissive defaults.
    config = _load_fixture_config("ant_cnn_identity")
    params = _build_params(config, tmp_path)

    assert params["ENABLE_IDENTITY_IN_TRACKING"] is True
    assert params["ENABLE_IDENTITY_ONLINE_DECODER"] is True
    assert params["ASSOCIATION_IDENTITY_HINT_SCALE"] == config["identity_weight"]
    assert params["ASSOCIATION_IDENTITY_HINT_SCALE"] == 0.05
    assert params["IDENTITY_COMMIT_THRESHOLD"] == config["identity_commit_threshold"]
    assert params["IDENTITY_DISPLAY_THRESHOLD"] == config["identity_display_threshold"]
    assert (
        params["IDENTITY_TRANSITION_EPSILON"] == config["identity_transition_epsilon"]
    )
    assert params["IDENTITY_UNKNOWN_PRIOR"] == config["identity_unknown_prior"]
    assert params["IDENTITY_REJOIN_THRESHOLD"] == config["identity_rejoin_threshold"]
    assert params["IDENTITY_SWAP_ENABLED"] == config["enable_identity_swap_correction"]
    assert params["IDENTITY_SWAP_MIN_FRAMES"] == config["identity_swap_min_frames"]
    assert params["IDENTITY_POSTPROCESS_MODE"] == config["identity_postprocess_mode"]
    assert params["ENABLE_IDENTITY_FRAGMENT_SOLVER"] is True
    assert params["APRILTAG_FAMILY"] == config["apriltag_family"]
    assert params["APRILTAG_DECIMATE"] == config["apriltag_decimate"]
    assert params["CNN_CLASSIFIER_WINDOW"] == 10
    assert params["IDENTITY_METHOD"] == "cnn_classifier"
    # Advanced-config-only knobs (no config-file key) keep the bridge's
    # advanced-config-driven defaults.
    assert params["IDENTITY_SWAP_CONF_MARGIN"] == 0.2
    assert params["IDENTITY_REJOIN_VELOCITY_BUDGET"] == 1.5
    assert params["IDENTITY_REJOIN_DIST_FLOOR"] is None


def test_worm_bgsub_and_fly_obb_derive_inert_identity_in_tracking_block(tmp_path):
    # Both protected-clip fixtures must derive the bridge-equivalent inert
    # values so the equivalence gate stays byte-identical: the online
    # decoder either stays off outright (fly_obb: master gate off) or its
    # effect is neutralized by a zero hint-scale weight elsewhere. Here we
    # only assert the two configs that have NO nonzero identity_weight and
    # rely on the master/decoder checkboxes to be off.
    fly_config = _load_fixture_config("fly_obb")
    fly_params = _build_params(fly_config, tmp_path)
    assert fly_params["ENABLE_IDENTITY_IN_TRACKING"] is False
    assert fly_params["ENABLE_IDENTITY_ONLINE_DECODER"] is False

    worm_config = _load_fixture_config("worm_bgsub")
    worm_params = _build_params(worm_config, tmp_path)
    assert worm_params["ENABLE_IDENTITY_IN_TRACKING"] is True
    assert worm_params["ENABLE_IDENTITY_ONLINE_DECODER"] is False
    assert worm_params["ASSOCIATION_IDENTITY_HINT_SCALE"] == 1.0


def test_emi_and_sleap_and_sequential_configs_zero_the_identity_hint_scale(tmp_path):
    # These three protected-gate clips DO enable the online decoder in their
    # saved configs but pin identity_weight to 0.0, which makes
    # _apply_bayesian_identity_cost and the identity-first slot-rejoining
    # gate both no-ops (hungarian.py:239, worker.py:2899-2910) -- so the
    # decoder engaging is harmless and byte-identity is preserved.
    for name in ("emi_obb_identity", "ant_obb_sleap", "ant_obb_sequential"):
        config = _load_fixture_config(name)
        params = _build_params(config, tmp_path)
        assert params["ENABLE_IDENTITY_ONLINE_DECODER"] is True, name
        assert params["ASSOCIATION_IDENTITY_HINT_SCALE"] == 0.0, name
