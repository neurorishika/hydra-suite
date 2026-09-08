"""The headless equivalent of the Calibrate button."""

import pytest

from hydra_suite.trackerkit.app import build_parser


def test_track_exposes_the_apply_flag_and_not_the_old_modes():
    parser = build_parser()
    args = parser.parse_args(["track", "--video", "v.mp4", "--apply-tuned-inference"])
    assert args.apply_tuned_inference is True
    with pytest.raises(SystemExit):
        parser.parse_args(
            ["track", "--video", "v.mp4", "--inference-autotune", "record"]
        )


def test_no_apply_flag_disables():
    parser = build_parser()
    args = parser.parse_args(
        ["track", "--video", "v.mp4", "--no-apply-tuned-inference"]
    )
    assert args.apply_tuned_inference is False


def test_calibrate_subcommand_exists_with_a_budget():
    parser = build_parser()
    args = parser.parse_args(
        ["calibrate", "--video", "v.mp4", "--budget-seconds", "600"]
    )
    assert args.command == "calibrate"
    assert args.budget_seconds == 600.0


def test_calibrate_refuses_fanout_flags():
    """Concurrent calibration on one box measures contention, not throughput."""
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["calibrate", "--video", "v.mp4", "--gpus", "auto"])
