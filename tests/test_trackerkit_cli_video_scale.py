"""``trackerkit track --video-scale`` -> config ``video_output_scale``."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from hydra_suite.trackerkit import cli as tracker_cli
from hydra_suite.trackerkit.app import build_parser
from hydra_suite.trackerkit.batch_plan import plan_batch_jobs


def _mk_video(directory: Path, name: str, cfg: dict | None = None) -> str:
    video = directory / f"{name}.mp4"
    video.write_bytes(b"\x00")
    if cfg is not None:
        (directory / f"{name}_config.json").write_text(json.dumps(cfg))
    return str(video)


def test_flag_parses_a_float_and_defaults_to_none():
    parser = build_parser()
    assert parser.parse_args(["track", "v.mp4"]).video_scale is None
    args = parser.parse_args(["track", "v.mp4", "--video-scale", "0.25"])
    assert args.video_scale == pytest.approx(0.25)


@pytest.mark.parametrize("bad", ["0", "0.05", "1.5", "-1", "abc", "nan"])
def test_out_of_range_flag_is_rejected_at_parse_time(bad, capsys):
    with pytest.raises(SystemExit):
        build_parser().parse_args(["track", "v.mp4", "--video-scale", bad])
    assert "video-scale" in capsys.readouterr().err


def test_flag_overrides_every_video_config(tmp_path):
    a = _mk_video(tmp_path, "a", {"k": "a", "video_output_scale": 0.8})
    b = _mk_video(tmp_path, "b")
    specs = plan_batch_jobs([a, b], video_scale=0.3)
    assert [s.config["video_output_scale"] for s in specs] == [0.3, 0.3]


def test_no_flag_leaves_config_untouched(tmp_path):
    a = _mk_video(tmp_path, "a", {"k": "a"})
    (spec,) = plan_batch_jobs([a])
    assert "video_output_scale" not in spec.config  # renderer defaults to 0.5


def test_invalid_config_value_fails_at_plan_time_not_after_tracking(tmp_path):
    a = _mk_video(tmp_path, "a", {"video_output_scale": 2.0})
    with pytest.raises(ValueError, match="video_output_scale"):
        plan_batch_jobs([a])


def test_run_tracking_cli_threads_the_flag_to_the_planner(tmp_path, monkeypatch):
    a = _mk_video(tmp_path, "a", {})
    seen = {}

    def _fake_plan(videos, **kwargs):
        seen.update(kwargs)
        return []

    monkeypatch.setattr(tracker_cli, "plan_batch_jobs", _fake_plan)
    with pytest.raises(ValueError):  # no specs -> "No videos were resolved"
        tracker_cli.run_tracking_cli([a], video_scale=0.4)
    assert seen["video_scale"] == pytest.approx(0.4)


def test_job_pack_accepts_the_flag():
    args = build_parser().parse_args(
        ["job", "pack", "jobdir", "v.mp4", "--video-scale", "0.75"]
    )
    assert args.video_scale == pytest.approx(0.75)
