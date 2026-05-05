"""Shared planning helpers for TrackerKit config-driven runs."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class TrackerVideoPlan:
    """Resolved config source for one tracker video run."""

    video_path: str
    config_path: str | None
    has_own_config: bool
    use_keystone_baseline: bool


def get_video_config_path(video_path: str | None) -> str | None:
    """Return the sidecar config path for *video_path*."""
    if not video_path:
        return None
    video_dir = os.path.dirname(video_path)
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    return os.path.join(video_dir, f"{video_name}_config.json")


def _existing_config_path(config_path: str | None) -> str | None:
    if not config_path:
        return None
    return config_path if os.path.isfile(config_path) else None


def resolve_video_plan(
    video_path: str,
    *,
    keystone_config_path: str | None = None,
    keystone_override: bool = False,
) -> TrackerVideoPlan:
    """Resolve which config should drive *video_path*.

    When ``keystone_override`` is enabled, the caller should preserve the
    keystone state for every non-keystone video, regardless of whether that
    video has its own sidecar config.
    """

    own_config_path = _existing_config_path(get_video_config_path(video_path))
    has_own_config = own_config_path is not None

    if has_own_config and not keystone_override:
        return TrackerVideoPlan(
            video_path=video_path,
            config_path=own_config_path,
            has_own_config=True,
            use_keystone_baseline=False,
        )

    return TrackerVideoPlan(
        video_path=video_path,
        config_path=_existing_config_path(keystone_config_path),
        has_own_config=False,
        use_keystone_baseline=True,
    )


def build_batch_video_plan(
    video_paths: Sequence[str],
    *,
    explicit_config_path: str | None = None,
    keystone_override: bool = False,
) -> list[TrackerVideoPlan]:
    """Resolve config precedence for a batch of videos.

    Rules:
    - first video uses ``explicit_config_path`` when provided
    - an explicit config on a multi-video batch implicitly enables keystone override
    - otherwise the first video uses its own sidecar config when present
    - later videos use their own sidecar config unless keystone override is on
    - later videos without their own config inherit the keystone baseline
    """

    videos = [str(path).strip() for path in video_paths if str(path).strip()]
    if not videos:
        return []

    explicit_path = _existing_config_path(explicit_config_path)
    effective_keystone_override = bool(
        keystone_override or (explicit_path is not None and len(videos) > 1)
    )
    first_video = videos[0]
    first_own_config = _existing_config_path(get_video_config_path(first_video))
    first_has_own = first_own_config is not None
    first_config_path = explicit_path or first_own_config

    plan = [
        TrackerVideoPlan(
            video_path=first_video,
            config_path=first_config_path,
            has_own_config=bool(first_has_own and explicit_path is None),
            use_keystone_baseline=False,
        )
    ]

    keystone_config_path = first_config_path
    for video_path in videos[1:]:
        plan.append(
            resolve_video_plan(
                video_path,
                keystone_config_path=keystone_config_path,
                keystone_override=effective_keystone_override,
            )
        )
    return plan
