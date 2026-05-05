"""Tests for TrackerKit CLI argument parsing."""

from hydra_suite.trackerkit.app import parse_arguments


def test_parse_arguments_defaults_to_gui_mode():
    args = parse_arguments([])

    assert args.command is None
    assert args.log_level == "INFO"


def test_parse_arguments_track_command_accepts_basic_batch_flags():
    args = parse_arguments(
        [
            "track",
            "first.mp4",
            "second.mp4",
            "--config",
            "shared.json",
            "--keystone-override",
        ]
    )

    assert args.command == "track"
    assert args.videos == ["first.mp4", "second.mp4"]
    assert args.config == "shared.json"
    assert args.keystone_override is True
