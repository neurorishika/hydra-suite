"""Tests for TrackerKit CLI argument parsing."""

from pathlib import Path

import pytest

from hydra_suite.trackerkit.app import (
    load_video_list,
    parse_arguments,
    resolve_track_video_inputs,
)


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


def test_parse_arguments_track_command_accepts_video_list_file():
    args = parse_arguments(
        ["track", "--video-list", "batch.txt", "--config", "shared.json"]
    )

    assert args.command == "track"
    assert args.videos == []
    assert args.video_list == "batch.txt"
    assert args.config == "shared.json"


def test_parse_arguments_track_command_rejects_mixed_inputs():
    with pytest.raises(SystemExit):
        parse_arguments(["track", "video.mp4", "--video-list", "batch.txt"])


def test_load_video_list_skips_missing_non_keystone_entries(tmp_path: Path):
    first_video = tmp_path / "first.mp4"
    third_video = tmp_path / "third.mp4"
    first_video.write_text("", encoding="utf-8")
    third_video.write_text("", encoding="utf-8")
    batch_file = tmp_path / "batch.txt"
    batch_file.write_text(
        f"{first_video}\n{tmp_path / 'missing.mp4'}\n{third_video}\n",
        encoding="utf-8",
    )

    resolved = load_video_list(str(batch_file))

    assert resolved == [str(first_video), str(third_video)]


def test_load_video_list_rejects_missing_keystone(tmp_path: Path):
    second_video = tmp_path / "second.mp4"
    second_video.write_text("", encoding="utf-8")
    batch_file = tmp_path / "batch.txt"
    batch_file.write_text(
        f"{tmp_path / 'missing.mp4'}\n{second_video}\n",
        encoding="utf-8",
    )

    with pytest.raises(FileNotFoundError, match="keystone video"):
        load_video_list(str(batch_file))


def test_resolve_track_video_inputs_prefers_video_list_file(tmp_path: Path):
    first_video = tmp_path / "first.mp4"
    first_video.write_text("", encoding="utf-8")
    batch_file = tmp_path / "batch.txt"
    batch_file.write_text(f"{first_video}\n", encoding="utf-8")

    resolved = resolve_track_video_inputs([], str(batch_file))

    assert resolved == [str(first_video)]
