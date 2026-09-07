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


# The per-video side-output keys a borrowed config carries over from the
# keystone. NOT consumed by the tracking engine: ``build_engine_params`` never
# reads them and ``load_tracker_cli_session`` derives the raw/final CSV paths
# from the video path itself, so rewriting them cannot change tracking output.
# ``video_output_path`` IS consumed, by ``CoreTrackingSession._run_annotated_video``.
_SIDE_OUTPUT_KEYS = ("file_path", "csv_path", "video_output_path")


def _retarget_side_outputs(cfg: dict[str, Any], video_path: str) -> dict[str, Any]:
    """Point a BORROWED config's side-output paths at *video_path*.

    A config borrowed from ANOTHER video holds that video's ABSOLUTE output
    paths. Left alone, every video in a keystone-inherited batch renders its
    annotated overlay into the keystone's one mp4 -- N concurrent writers, one
    corrupt file, and no overlay for any other video. The pre-refactor
    sequential loop had the same latent bug; the GUI's own sequential batch did
    not, because it re-derived the path per video.

    "Borrowed" is the whole justification, so it is also the whole scope: see
    the call site for the two provenances that qualify. A config the user
    named for THIS video -- its own sidecar, or a ``--config`` on a
    single-video run -- is not borrowed from anyone, and rewriting its paths
    would discard the render location the user asked for.

    Only keys that are ALREADY present are rewritten: inventing
    ``video_output_path`` would make a config that never rendered a video start
    rendering one.
    """
    if not cfg:
        return cfg
    csv_path, video_output_path = _default_output_paths(video_path)
    values = {
        "file_path": video_path,
        "csv_path": csv_path,
        "video_output_path": video_output_path,
    }
    for key in _SIDE_OUTPUT_KEYS:
        if key in cfg:
            cfg[key] = values[key]
    return cfg


def _reject_render_collisions(specs: Sequence[BatchJobSpec]) -> None:
    """Belt and braces: two videos must never render into one annotated mp4.

    Only jobs that will actually render are compared -- the same predicate
    ``_run_annotated_video`` uses -- so a stale, disabled path cannot block a
    run that would never have written it.
    """
    seen: dict[str, str] = {}
    for spec in specs:
        path = str(spec.config.get("video_output_path", "") or "").strip()
        if not path or not spec.config.get("video_output_enabled", False):
            continue
        key = os.path.realpath(path)
        if key in seen:
            raise BatchPlanError(
                f"two videos would render the same annotated video {path}: "
                f"{spec.video_path} and {seen[key]}"
            )
        seen[key] = spec.video_path


def plan_batch_jobs(
    video_paths: Sequence[str],
    *,
    explicit_config_path: str | None = None,
    keystone_override: bool = False,
    sahi_profile: str | None = None,
) -> list[BatchJobSpec]:
    """Resolve one ``BatchJobSpec`` per video with today's keystone rules.

    Semantics are exactly the pre-refactor ``cli.run_tracking_cli`` loop:
    video 1's resolved config becomes the keystone baseline; a video with no
    config of its own inherits the baseline; ``sahi_profile`` is applied to
    every video. The one deliberate divergence is that a config a video
    BORROWED from another gets its side-output paths retargeted -- see
    ``_retarget_side_outputs`` and the guard at its call site.
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
        # Retarget only a config this video BORROWED from another video:
        #  * "keystone-baseline" -- always borrowed, by definition;
        #  * "explicit" on a MULTI-video batch -- one --config shared by every
        #    video, so at most one of them can own its paths.
        # An "explicit" config on a SINGLE video is not borrowed: the user
        # named that config for that video, and rewriting its paths would (a)
        # throw away the render location they asked for and (b) make the
        # fan-out child's re-plan of its own job config -- which arrives as
        # ``track <video> --config job_N.json``, i.e. explicit, len(plan) == 1
        # -- overwrite the parent's decision, so an own-sidecar batch's
        # renders diverge from sequential while its CSVs stay identical.
        if provenance == "keystone-baseline" or (
            provenance == "explicit" and len(plan) > 1
        ):
            cfg = _retarget_side_outputs(cfg, item.video_path)
        if index == 1:
            # The keystone's own resolved dict IS the baseline; a video that
            # inherits it necessarily has no config file of its own, which can
            # only happen when the keystone had none either (build_batch_video_plan
            # hands every later video the keystone's path when there is one).
            baseline = deepcopy(cfg)
        specs.append(
            BatchJobSpec(
                index=index,
                video_path=item.video_path,
                config_path=item.config_path,
                config=cfg,
                provenance=provenance,
            )
        )
    _reject_render_collisions(specs)
    return specs
