"""One planner that resolves the effective config for every video in a batch.

Both the in-process sequential loop (``cli.run_tracking_cli``) and the
process-per-GPU fan-out (``batch_fanout``) consume this, so the config a child
receives is -- by construction -- the config the sequential loop would have
used. Qt-free.
"""

from __future__ import annotations

import os
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Sequence

from hydra_suite.trackerkit.cli_config import (
    _default_output_paths,
    apply_sahi_profile_override,
    load_tracker_cli_config,
)
from hydra_suite.trackerkit.session_plan import build_batch_video_plan


class BatchPlanError(ValueError):
    """A batch cannot run: duplicate inputs or colliding outputs."""


@dataclass(frozen=True)
class BatchJobSpec:
    index: int  # 1-based; 1 is the keystone
    video_path: str
    config_path: str | None  # the sidecar/explicit file this config came from
    config: dict[str, Any]  # effective config; deep-copied per job
    provenance: str  # "own-sidecar" | "explicit" | "keystone-baseline"


def _reject_collisions(video_paths: Sequence[str]) -> None:
    seen_real: dict[str, str] = {}
    seen_raw_csv: dict[str, str] = {}
    for video in video_paths:
        real = os.path.realpath(video)
        if real in seen_real:
            raise BatchPlanError(f"video listed twice: {video} and {seen_real[real]}")
        seen_real[real] = video
        raw_csv, _ = _default_output_paths(video)
        raw_key = os.path.realpath(raw_csv)
        if raw_key in seen_raw_csv:
            raise BatchPlanError(
                f"two videos would write the same output {raw_csv}: "
                f"{video} and {seen_raw_csv[raw_key]}"
            )
        seen_raw_csv[raw_key] = video


def plan_batch_jobs(
    video_paths: Sequence[str],
    *,
    explicit_config_path: str | None = None,
    keystone_override: bool = False,
    sahi_profile: str | None = None,
) -> list[BatchJobSpec]:
    """Resolve one ``BatchJobSpec`` per video with today's keystone rules.

    Semantics are exactly the pre-refactor ``cli.run_tracking_cli`` loop:
    video 1's resolved config (pre-SAHI-override when it came from a file)
    becomes the keystone baseline; a video with no config of its own inherits
    the baseline; ``sahi_profile`` is applied to every video.
    """
    videos = [str(v).strip() for v in video_paths if str(v).strip()]
    if not videos:
        raise BatchPlanError("at least one video path is required")
    _reject_collisions(videos)

    plan = build_batch_video_plan(
        videos,
        explicit_config_path=explicit_config_path,
        keystone_override=keystone_override,
    )
    specs: list[BatchJobSpec] = []
    baseline: dict[str, Any] | None = None
    for index, item in enumerate(plan, start=1):
        # ``inherits`` is the pre-refactor loop's predicate for "this video has
        # no config file at all, so it takes the keystone's resolved dict".
        inherits = item.use_keystone_baseline and item.config_path is None
        if inherits:
            cfg: dict[str, Any] = deepcopy(baseline or {})
        else:
            cfg = load_tracker_cli_config(item.config_path)
        # Provenance labels WHERE the config came from, which is not the same
        # question as whether the baseline dict was reused: a later video under
        # keystone override loads the keystone's own file, and that is still
        # keystone provenance.
        if item.use_keystone_baseline:
            provenance = "keystone-baseline"
        elif (
            explicit_config_path
            and item.config_path
            and os.path.realpath(item.config_path)
            == os.path.realpath(explicit_config_path)
        ):
            provenance = "explicit"
        else:
            provenance = "own-sidecar"
        if sahi_profile:
            cfg = apply_sahi_profile_override(cfg, sahi_profile)
        cfg = deepcopy(dict(cfg))
        if index == 1:
            baseline = (
                deepcopy(load_tracker_cli_config(item.config_path))
                if item.config_path
                else deepcopy(cfg)
            )
        specs.append(
            BatchJobSpec(
                index=index,
                video_path=item.video_path,
                config_path=item.config_path if not inherits else None,
                config=cfg,
                provenance=provenance,
            )
        )
    return specs
