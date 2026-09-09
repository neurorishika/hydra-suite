"""Structural output discovery and origin mapping for `job pull`.

The contract is structural, not name-based: an output is anything under
`videos/` that is not a manifest video and not in that video's
`pushed_siblings`. A future artifact type is therefore pulled with no code
change.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Iterable

from .manifest import JobManifest, JobVideo, TrackingJobError

_IGNORED_BASENAMES = {".DS_Store"}


def _is_ignored(relpath: str) -> bool:
    name = PurePosixPath(relpath).name
    return name in _IGNORED_BASENAMES or name.startswith(".DS_Store")


def discover_outputs(manifest: JobManifest, listing: Iterable[str]) -> list[str]:
    """Job-relative output paths, given a flat listing of everything under `videos/`."""
    video_paths = {video.job_path for video in manifest.videos}
    pushed = {sibling for video in manifest.videos for sibling in video.pushed_siblings}
    outputs: list[str] = []
    for relpath in listing:
        parts = PurePosixPath(relpath).parts
        if not parts or parts[0] != "videos":
            continue
        if relpath in video_paths or relpath in pushed:
            continue
        if _is_ignored(relpath):
            continue
        outputs.append(relpath)
    return outputs


def _owning_video(manifest: JobManifest, relpath: str) -> JobVideo:
    first = PurePosixPath(relpath).parts[1]  # after "videos/"
    best, best_len = None, -1
    for video in manifest.videos:
        stem = PurePosixPath(video.job_path).stem
        if first.startswith(stem) or first == f".inference_cache_{stem}":
            if len(stem) > best_len:
                best, best_len = video, len(stem)
    if best is None:
        raise TrackingJobError(
            f"pulled artifact {relpath} matches no video in the job; refusing to "
            "guess a destination",
            code=5,
        )
    return best


@dataclass(frozen=True)
class PullDestination:
    job_relpath: str
    destination: str
    redirected: bool


def map_outputs_to_origins(
    manifest: JobManifest, outputs: Iterable[str]
) -> list[PullDestination]:
    """Map job-relative output paths to their absolute destinations beside the origin."""
    mapped: list[PullDestination] = []
    for relpath in outputs:
        if _is_ignored(relpath):
            continue
        video = _owning_video(manifest, relpath)
        if relpath in video.redirected_outputs:
            mapped.append(
                PullDestination(
                    job_relpath=relpath,
                    destination=video.redirected_outputs[relpath],
                    redirected=True,
                )
            )
            continue
        origin_dir = PurePosixPath(video.origin_path).parent
        video_dir = PurePosixPath(video.job_path).parent
        rel_to_video_dir = PurePosixPath(relpath).relative_to(video_dir)
        destination = str(origin_dir / rel_to_video_dir)
        mapped.append(
            PullDestination(
                job_relpath=relpath, destination=destination, redirected=False
            )
        )
    return mapped


def plan_pull(
    manifest: JobManifest, listing: Iterable[str], *, include_caches: bool = True
) -> list[PullDestination]:
    """Discover outputs and map them to origins, optionally excluding cache trees."""
    outputs = discover_outputs(manifest, listing)
    if not include_caches:
        outputs = [p for p in outputs if ".inference_cache_" not in p]
    return map_outputs_to_origins(manifest, outputs)
