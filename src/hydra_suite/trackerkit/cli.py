"""Minimal TrackerKit CLI runner for config-driven tracking sessions (Qt-free)."""

from __future__ import annotations

import json
import logging
import tempfile
from copy import deepcopy
from pathlib import Path
from typing import Any, Sequence

from hydra_suite.trackerkit.cli_config import (
    load_tracker_cli_config,
    load_tracker_cli_session,
)
from hydra_suite.trackerkit.headless_tracking import run_headless_tracking_session
from hydra_suite.trackerkit.session_plan import build_batch_video_plan

logger = logging.getLogger(__name__)


def run_tracking_cli(
    video_paths: Sequence[str],
    *,
    config_path: str | None = None,
    keystone_override: bool = False,
) -> int:
    """Run one or more TrackerKit sessions from the CLI (direct Qt-free path)."""

    videos = [str(path).strip() for path in video_paths if str(path).strip()]
    if not videos:
        raise ValueError("At least one video path is required.")

    for video_path in videos:
        if not Path(video_path).is_file():
            raise FileNotFoundError(f"Video not found: {video_path}")
    if config_path and not Path(config_path).is_file():
        raise FileNotFoundError(f"Config not found: {config_path}")

    plan = build_batch_video_plan(
        videos,
        explicit_config_path=config_path,
        keystone_override=keystone_override,
    )
    if not plan:
        raise ValueError("No videos were resolved for tracking.")

    exit_code = 0
    with tempfile.TemporaryDirectory(prefix="trackerkit-cli-") as tmpdir:
        tmpdir_path = Path(tmpdir)
        baseline_config_data: dict[str, Any] | None = None

        for index, item in enumerate(plan, start=1):
            logger.info(
                "Tracker CLI: preparing video %s/%s: %s",
                index,
                len(plan),
                item.video_path,
            )
            effective_config_data = None
            if item.use_keystone_baseline and item.config_path is None:
                effective_config_data = baseline_config_data or {}
            session = load_tracker_cli_session(
                item.video_path,
                config_path=(
                    item.config_path if effective_config_data is None else None
                ),
                config_data=effective_config_data,
            )

            if index == 1:
                baseline_config_data = (
                    deepcopy(load_tracker_cli_config(item.config_path))
                    if item.config_path
                    else deepcopy(session.config)
                )

            # Persist the resolved keystone baseline for provenance/debugging; the
            # direct path consumes ``session`` directly and needs no config file.
            if item.use_keystone_baseline and item.config_path is None:
                keystone_dump = tmpdir_path / f"keystone_config_{index}.json"
                with open(keystone_dump, "w", encoding="utf-8") as handle:
                    json.dump(baseline_config_data or {}, handle, indent=2)

            result = run_headless_tracking_session(session)

            if result.get("success"):
                summary = " | ".join(result.get("lines", []))
                logger.info("Tracker CLI completed: %s", summary)
            else:
                error_message = result.get("error") or "Tracker session failed."
                logger.error(
                    "Tracker CLI failed for %s: %s",
                    item.video_path,
                    error_message,
                )
                exit_code = 1
                break

    return exit_code
