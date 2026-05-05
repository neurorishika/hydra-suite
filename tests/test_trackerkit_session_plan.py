"""Tests for TrackerKit batch/config planning helpers."""

from __future__ import annotations

from pathlib import Path

from hydra_suite.trackerkit.session_plan import (
    build_batch_video_plan,
    get_video_config_path,
    resolve_video_plan,
)


def _touch(path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{}", encoding="utf-8")
    return str(path)


def test_resolve_video_plan_prefers_own_sidecar_when_not_overridden(tmp_path: Path):
    video_path = tmp_path / "clip.mp4"
    video_path.write_text("", encoding="utf-8")
    _touch(Path(get_video_config_path(str(video_path)) or ""))

    plan = resolve_video_plan(str(video_path), keystone_override=False)

    assert plan.video_path == str(video_path)
    assert plan.has_own_config is True
    assert plan.use_keystone_baseline is False
    assert plan.config_path == get_video_config_path(str(video_path))


def test_resolve_video_plan_uses_keystone_when_overridden(tmp_path: Path):
    first_video = tmp_path / "first.mp4"
    second_video = tmp_path / "second.mp4"
    first_video.write_text("", encoding="utf-8")
    second_video.write_text("", encoding="utf-8")
    keystone_config = _touch(Path(get_video_config_path(str(first_video)) or ""))
    _touch(Path(get_video_config_path(str(second_video)) or ""))

    plan = resolve_video_plan(
        str(second_video),
        keystone_config_path=keystone_config,
        keystone_override=True,
    )

    assert plan.has_own_config is False
    assert plan.use_keystone_baseline is True
    assert plan.config_path == keystone_config


def test_build_batch_video_plan_uses_keystone_for_videos_without_sidecars(
    tmp_path: Path,
):
    first_video = tmp_path / "first.mp4"
    second_video = tmp_path / "second.mp4"
    first_video.write_text("", encoding="utf-8")
    second_video.write_text("", encoding="utf-8")
    first_config = _touch(Path(get_video_config_path(str(first_video)) or ""))

    plan = build_batch_video_plan([str(first_video), str(second_video)])

    assert len(plan) == 2
    assert plan[0].config_path == first_config
    assert plan[0].use_keystone_baseline is False
    assert plan[1].config_path == first_config
    assert plan[1].use_keystone_baseline is True


def test_build_batch_video_plan_prefers_per_video_configs_by_default(tmp_path: Path):
    first_video = tmp_path / "first.mp4"
    second_video = tmp_path / "second.mp4"
    first_video.write_text("", encoding="utf-8")
    second_video.write_text("", encoding="utf-8")
    first_config = _touch(Path(get_video_config_path(str(first_video)) or ""))
    second_config = _touch(Path(get_video_config_path(str(second_video)) or ""))

    plan = build_batch_video_plan([str(first_video), str(second_video)])

    assert plan[0].config_path == first_config
    assert plan[1].config_path == second_config
    assert plan[1].use_keystone_baseline is False


def test_build_batch_video_plan_uses_explicit_config_as_keystone(tmp_path: Path):
    first_video = tmp_path / "first.mp4"
    second_video = tmp_path / "second.mp4"
    third_video = tmp_path / "third.mp4"
    first_video.write_text("", encoding="utf-8")
    second_video.write_text("", encoding="utf-8")
    third_video.write_text("", encoding="utf-8")
    explicit_config = _touch(tmp_path / "shared_config.json")
    _touch(Path(get_video_config_path(str(third_video)) or ""))

    plan = build_batch_video_plan(
        [str(first_video), str(second_video), str(third_video)],
        explicit_config_path=explicit_config,
    )

    assert plan[0].config_path == explicit_config
    assert plan[1].config_path == explicit_config
    assert plan[1].use_keystone_baseline is True
    assert plan[2].config_path == explicit_config
    assert plan[2].use_keystone_baseline is True


def test_build_batch_video_plan_does_not_force_override_for_single_video(
    tmp_path: Path,
):
    video_path = tmp_path / "single.mp4"
    video_path.write_text("", encoding="utf-8")
    explicit_config = _touch(tmp_path / "shared_config.json")

    plan = build_batch_video_plan(
        [str(video_path)],
        explicit_config_path=explicit_config,
    )

    assert len(plan) == 1
    assert plan[0].config_path == explicit_config
    assert plan[0].use_keystone_baseline is False
