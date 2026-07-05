"""Test detection_batch_size parameter for equivalence runner.

This module tests the --detection-batch-size CLI flag and build_config() parameter
that enable testing TensorRT dynamic-batch OBB detection on real video.
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


def test_build_config_sets_yolo_batch_size_when_provided(tmp_path):
    """When detection_batch_size is provided, YOLO_BATCH_SIZE is set in config."""
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

    assert "YOLO_BATCH_SIZE" in cfg, "YOLO_BATCH_SIZE not found in config"
    assert (
        cfg["YOLO_BATCH_SIZE"] == 8
    ), f"Expected YOLO_BATCH_SIZE=8, got {cfg['YOLO_BATCH_SIZE']}"


def test_build_config_omits_yolo_batch_size_when_not_provided(tmp_path):
    """When detection_batch_size is not provided, YOLO_BATCH_SIZE is not set."""
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
        "YOLO_BATCH_SIZE" not in cfg
    ), "YOLO_BATCH_SIZE should not be set when detection_batch_size is None"
