"""Test performance-control parameters for the equivalence runner.

These controls enable detector, crop-backend, and pipeline-depth sweeps on real
fixture videos without mutating their checked-in configs.
"""

import importlib.util
import json
import sys
from pathlib import Path

# Import build_config from tools/equivalence/runner.py via importlib
_RUNNER_PATH = (
    Path(__file__).resolve().parents[1] / "tools" / "equivalence" / "runner.py"
)
_spec = importlib.util.spec_from_file_location("equiv_runner", _RUNNER_PATH)
equiv_runner = importlib.util.module_from_spec(_spec)
sys.modules["equiv_runner"] = equiv_runner
_spec.loader.exec_module(equiv_runner)


def test_build_config_sets_detection_batch_size_when_provided(tmp_path):
    """When detection_batch_size is provided, detection_batch_size is set in config."""
    # Create a minimal valid orig-config
    orig_config = tmp_path / "orig_config.json"
    orig_config.write_text(json.dumps({"file_path": "", "csv_path": ""}))

    # Call build_config with detection_batch_size=8
    video_link = tmp_path / "test_video.mp4"
    out_cfg = equiv_runner.build_config(
        str(orig_config),
        video_link=video_link,
        outdir=tmp_path,
        runtime="config",
        detection_batch_size=8,
    )

    # Verify the output config contains YOLO_BATCH_SIZE=8
    with open(out_cfg) as fh:
        cfg = json.load(fh)

    assert "detection_batch_size" in cfg, "detection_batch_size not found in config"
    assert (
        cfg["detection_batch_size"] == 8
    ), f"Expected detection_batch_size=8, got {cfg['detection_batch_size']}"


def test_build_config_omits_detection_batch_size_when_not_provided(tmp_path):
    """When detection_batch_size is not provided, detection_batch_size is not set."""
    # Create a minimal valid orig-config
    orig_config = tmp_path / "orig_config.json"
    orig_config.write_text(json.dumps({"file_path": "", "csv_path": ""}))

    # Call build_config without detection_batch_size (None)
    video_link = tmp_path / "test_video.mp4"
    out_cfg = equiv_runner.build_config(
        str(orig_config),
        video_link=video_link,
        outdir=tmp_path,
        runtime="config",
        detection_batch_size=None,
    )

    # Verify the output config does NOT contain YOLO_BATCH_SIZE
    with open(out_cfg) as fh:
        cfg = json.load(fh)

    assert (
        "detection_batch_size" not in cfg
    ), "detection_batch_size should not be set when detection_batch_size is None"


def test_build_config_sets_individual_stage_batch_sizes(tmp_path):
    orig_config = tmp_path / "orig_config.json"
    orig_config.write_text(
        json.dumps(
            {
                "file_path": "",
                "csv_path": "",
                "cnn_classifiers": [
                    {"label": "color", "batch_size": 64},
                    {"label": "mark", "batch_size": 32},
                ],
            }
        )
    )

    out_cfg = equiv_runner.build_config(
        str(orig_config),
        video_link=tmp_path / "test_video.mp4",
        outdir=tmp_path,
        runtime="config",
        headtail_batch_size=16,
        cnn_batch_size=24,
        pose_batch_size=8,
        pipeline_depth=1,
    )

    with open(out_cfg) as fh:
        cfg = json.load(fh)

    assert cfg["headtail_batch_size"] == 16
    assert [item["batch_size"] for item in cfg["cnn_classifiers"]] == [24, 24]
    assert cfg["pose_batch_size"] == 8
    assert cfg["pipeline_depth"] == 1


def test_benchmark_controls_records_resolved_fixture_settings():
    controls = equiv_runner.benchmark_controls(
        {
            "runtime_tier": "gpu",
            "detection_batch_size": 4,
            "headtail_batch_size": 25,
            "cnn_classifiers": [
                {"label": "color", "batch_size": 25},
                {"label": "mark", "batch_size": 16},
            ],
            "pose_batch_size": 8,
            "pipeline_depth": 2,
            "start_frame": 0,
            "end_frame": 499,
        }
    )

    assert controls == {
        "runtime_tier": "gpu",
        "detection_batch_size": 4,
        "headtail_batch_size": 25,
        "cnn_batch_sizes": [
            {"label": "color", "batch_size": 25},
            {"label": "mark", "batch_size": 16},
        ],
        "pose_batch_size": 8,
        "pipeline_depth": 2,
        "start_frame": 0,
        "end_frame": 499,
    }
