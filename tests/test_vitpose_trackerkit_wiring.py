"""Regression guard: the trackerkit GUI is wired for the ViTPose pose backend.

Trackerkit's ViTPose GUI wiring (the pose-backend combo item, the backend-key
resolution, and the ``models/pose/ViTPose`` model directory) pre-existed in the
tracking app; the full-integration branch's Tasks 1-6 make the pose factory it
targets functional. These tests pin that wiring so a later refactor cannot
silently drop ViTPose from the live tracking path.

Source-level assertions are used for the Qt panels to avoid needing a live
``QApplication`` in headless CI; the config-builder test exercises real
behavior through the exact params the trackerkit orchestrator emits.
"""

from pathlib import Path

from hydra_suite.trackerkit.gui import model_utils


def test_pose_models_directory_vitpose():
    # get_pose_models_directory("vitpose") must resolve to the ViTPose repo dir.
    d = str(model_utils.get_pose_models_directory("vitpose"))
    assert d.endswith("ViTPose")


def test_identity_panel_lists_vitpose():
    # The trackerkit pose-backend combo must offer ViTPose alongside YOLO/SLEAP.
    import hydra_suite.trackerkit.gui.panels.identity_panel as ip

    text = Path(ip.__file__).read_text(encoding="utf-8")
    assert "combo_pose_model_type" in text
    assert '"ViTPose"' in text


def test_config_orchestrator_handles_vitpose_backend():
    # The config orchestrator must have a vitpose backend-key branch (model
    # directory selection + path resolution).
    import hydra_suite.trackerkit.gui.orchestrators.config as cfg

    text = Path(cfg.__file__).read_text(encoding="utf-8")
    assert '"vitpose"' in text


def test_build_from_params_matches_trackerkit_emission(tmp_path):
    # Mirror the exact params the trackerkit config orchestrator emits for a
    # ViTPose selection: POSE_MODEL_TYPE="vitpose" and the model path under
    # POSE_MODEL_DIR (NOT POSE_VITPOSE_MODEL_PATH). Confirm the inference-config
    # builder honors that emission shape and produces a vitpose PoseConfig.
    from hydra_suite.core.inference.config import build_inference_config_from_params

    p = tmp_path / "best.pt"
    p.write_bytes(b"x")
    config = build_inference_config_from_params(
        {
            "ENABLE_POSE_EXTRACTOR": True,
            "POSE_MODEL_TYPE": "vitpose",
            "POSE_MODEL_DIR": str(p),
        }
    )
    assert config.pose is not None
    assert config.pose.backend == "vitpose"
    assert config.pose.vitpose is not None
    assert config.pose.vitpose.model_path == str(p)
