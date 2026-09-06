"""Minimal TrackerKit CLI runner for config-driven tracking sessions (Qt-free)."""

from __future__ import annotations

import json
import logging
import tempfile
from pathlib import Path
from typing import Sequence

from hydra_suite.trackerkit.batch_plan import BatchJobSpec, plan_batch_jobs
from hydra_suite.trackerkit.cli_config import load_tracker_cli_session
from hydra_suite.trackerkit.headless_tracking import run_headless_tracking_session

logger = logging.getLogger(__name__)


def run_tracking_cli(
    video_paths: Sequence[str],
    *,
    config_path: str | None = None,
    keystone_override: bool = False,
    sahi_profile: str | None = None,
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

    specs = plan_batch_jobs(
        videos,
        explicit_config_path=config_path,
        keystone_override=keystone_override,
        sahi_profile=sahi_profile,
    )
    if not specs:
        raise ValueError("No videos were resolved for tracking.")
    return _run_sequential(specs)


def _run_sequential(specs: Sequence[BatchJobSpec]) -> int:
    """The in-process path: one session after another on this process."""
    exit_code = 0
    with tempfile.TemporaryDirectory(prefix="trackerkit-cli-") as tmpdir:
        tmpdir_path = Path(tmpdir)
        for spec in specs:
            logger.info(
                "Tracker CLI: preparing video %s/%s: %s",
                spec.index,
                len(specs),
                spec.video_path,
            )
            session = load_tracker_cli_session(
                spec.video_path,
                config_path=spec.config_path,
                config_data=spec.config,
            )
            # Persist the resolved keystone baseline for provenance/debugging;
            # the direct path consumes ``session`` directly and needs no config
            # file. Dump the OVERRIDDEN config (``session.config``), not the
            # pre-override baseline, so the provenance file names the profile
            # that actually ran. ``config_path is None`` narrows this to the
            # videos that truly inherited the baseline dict -- a later video
            # that loads the keystone's own file is keystone provenance too,
            # but it has a file of its own to point at.
            if spec.provenance == "keystone-baseline" and spec.config_path is None:
                keystone_dump = tmpdir_path / f"keystone_config_{spec.index}.json"
                with open(keystone_dump, "w", encoding="utf-8") as handle:
                    json.dump(session.config or {}, handle, indent=2)

            result = run_headless_tracking_session(session)

            if result.get("success"):
                summary = " | ".join(result.get("lines", []))
                logger.info("Tracker CLI completed: %s", summary)
            else:
                error_message = result.get("error") or "Tracker session failed."
                logger.error(
                    "Tracker CLI failed for %s: %s",
                    spec.video_path,
                    error_message,
                )
                exit_code = 1
                break

    return exit_code
