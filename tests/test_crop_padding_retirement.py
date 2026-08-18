"""Contract tests for the crop-padding retirement (spec 2026-08-18)."""

from pathlib import Path

import numpy as np

from hydra_suite.core.individual.classification.apriltag import AprilTagConfig


def _corners(cx, cy, w, h):
    return np.array(
        [
            [cx - w / 2, cy - h / 2],
            [cx + w / 2, cy - h / 2],
            [cx + w / 2, cy + h / 2],
            [cx - w / 2, cy + h / 2],
        ],
        dtype=np.float32,
    )


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


def test_zero_padding_is_the_exact_obb_extent():
    from hydra_suite.core.tracking.pose.pose_pipeline import _expand_obb_to_aabb

    corners = _corners(100.0, 80.0, 40.0, 20.0)
    x0, y0, x1, y1 = _expand_obb_to_aabb(corners, 0.0, 480, 640)
    assert (x0, y0, x1, y1) == (80, 70, 120, 90)


def test_negative_padding_shrinks_symmetrically():
    from hydra_suite.core.tracking.pose.pose_pipeline import _expand_obb_to_aabb

    corners = _corners(100.0, 80.0, 40.0, 20.0)
    x0, y0, x1, y1 = _expand_obb_to_aabb(corners, -0.25, 480, 640)
    # pad = -0.25 * max(40, 20) = -10 on every side
    assert (x0, y0, x1, y1) == (90, 80, 110, 80)


def test_aabb_helpers_agree_with_the_live_path():
    from hydra_suite.core.inference.stages.crops import extract_aabb_crops
    from hydra_suite.core.tracking.pose.pose_pipeline import _expand_obb_to_aabb

    class _StubOBB:
        num_detections = 1
        corners = np.stack([_corners(100.0, 80.0, 40.0, 20.0)])

    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    for pad in (0.0, 0.1, 0.3):
        live = extract_aabb_crops(frame, _StubOBB(), padding=pad)[0]
        x0, y0, x1, y1 = _expand_obb_to_aabb(
            _StubOBB.corners[0], pad, frame.shape[0], frame.shape[1]
        )
        assert live.shape[:2] == (y1 - y0, x1 - x0), f"mismatch at padding={pad}"
