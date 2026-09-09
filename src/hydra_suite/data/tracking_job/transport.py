"""rsync/ssh transport for portable tracking jobs.

`push_job` syncs a packed job directory to a compute box; `pull_job` syncs
outputs back beside their origin videos. The push input list is derived
*from the manifest*, never from an exclude list, so a re-push after a local
pull can never clobber remote outputs with stale local copies -- `--delete`
is never passed.
"""

from __future__ import annotations

import datetime
import posixpath
import shlex
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Callable

from .manifest import JobManifest, TrackingJobError
from .outputs import PullDestination, plan_pull
from .references import _sha256
from .shared_roots import load_shared_roots, resolve_shared

Runner = Callable[..., subprocess.CompletedProcess]

_RUNNING_CHECK_SLACK_SECONDS = 1.0


@dataclass(frozen=True)
class RemoteTarget:
    host: str
    path: str


def parse_remote(text: str) -> RemoteTarget:
    """Parse ``user@host:/absolute/path`` into a `RemoteTarget`."""
    if not text or ":" not in text:
        raise TrackingJobError(f"malformed remote target: {text!r}")
    host, _, path = text.partition(":")
    if not host or not path or not path.startswith("/"):
        raise TrackingJobError(
            f"malformed remote target: {text!r} (expected host:/absolute/path)"
        )
    return RemoteTarget(host=host, path=path)


def _as_remote(remote: "RemoteTarget | str") -> RemoteTarget:
    return remote if isinstance(remote, RemoteTarget) else parse_remote(remote)


def build_push_input_list(manifest: JobManifest) -> list[str]:
    """Job-relative paths to push, derived entirely from the manifest.

    Never an exclude list: outputs (``*_tracking*``, ``.inference_cache_*``,
    ``logs/``) simply never appear here because nothing in the manifest
    names them.
    """
    entries: set[str] = {"hydra_job.json", "run.sh", "videos.txt"}

    config_snapshot = manifest.config_snapshot or {}
    advanced_config = config_snapshot.get("advanced_config")
    if advanced_config:
        entries.add(advanced_config)
    for skeleton in config_snapshot.get("skeletons", []):
        entries.add(skeleton)
    entries.add("config/presets/.seeded")
    entries.add("config/skeletons/.seeded")

    entries.add("models/model_registry.json")
    for model in manifest.models:
        if model.files:
            # A directory model: rsync --files-from does NOT recurse into a
            # listed directory, so `key` itself (a directory path) is never
            # a transferable list entry -- only its enumerated member files.
            for member in model.files:
                entries.add(f"models/{member}")
        else:
            entries.add(f"models/{model.key}")
        for sidecar in model.sidecars:
            entries.add(f"models/{sidecar}")

    for video in manifest.videos:
        if video.shared is None:
            entries.add(video.job_path)
        for sibling in video.pushed_siblings:
            entries.add(sibling)

    return sorted(entries)


def build_rsync_argv(
    source: str,
    destination: str,
    *,
    files_from: str,
    extra: tuple[str, ...] = (),
) -> list[str]:
    return [
        "rsync",
        "-a",
        "--partial",
        "--info=progress2",
        f"--files-from={files_from}",
        *extra,
        source,
        destination,
    ]


def _write_list_file(entries: list[str]) -> str:
    handle = tempfile.NamedTemporaryFile(
        mode="w", suffix=".txt", delete=False, encoding="utf-8"
    )
    try:
        for entry in entries:
            handle.write(entry + "\n")
    finally:
        handle.close()
    return handle.name


def push_job(
    job_dir: Path,
    remote: "RemoteTarget | str",
    *,
    runner: Runner = subprocess.run,
) -> None:
    target = _as_remote(remote)
    job_dir = Path(job_dir)

    # Fix V-minor: rsync -a only creates the FINAL path component of the
    # destination; if the parent chain (e.g. "jobs/") does not yet exist on
    # a fresh remote, rsync fails before a single file transfers.
    parent = posixpath.dirname(target.path.rstrip("/")) or "/"
    mkdir_argv = ["ssh", target.host, f"mkdir -p {shlex.quote(parent)}"]
    mkdir_result = runner(mkdir_argv, capture_output=True, text=True)
    if mkdir_result.returncode != 0:
        raise TrackingJobError(
            "failed to create remote parent directory: "
            f"{' '.join(mkdir_argv)}\n{mkdir_result.stderr}",
            code=4,
        )

    manifest = JobManifest.read(job_dir / "hydra_job.json")
    entries = build_push_input_list(manifest)
    list_path = _write_list_file(entries)
    try:
        argv = build_rsync_argv(
            f"{job_dir}/",
            f"{target.host}:{target.path}/",
            files_from=list_path,
            extra=("--copy-links",),
        )
        result = runner(argv, capture_output=True, text=True)
        if result.returncode != 0:
            raise TrackingJobError(
                f"push failed: {' '.join(argv)}\n{result.stderr}", code=4
            )
    finally:
        try:
            Path(list_path).unlink()
        except OSError:
            pass


def remote_video_listing(remote: "RemoteTarget | str", *, runner: Runner) -> list[str]:
    target = _as_remote(remote)
    argv = [
        "ssh",
        target.host,
        f"cd {shlex.quote(target.path)} && find videos -type f -o -type l",
    ]
    result = runner(argv, capture_output=True, text=True)
    if result.returncode != 0:
        raise TrackingJobError(
            f"failed to list remote videos/: {' '.join(argv)}\n{result.stderr}",
            code=4,
        )
    return [line for line in (result.stdout or "").splitlines() if line]


def _last_runs_jsonl_timestamp(path: Path) -> float | None:
    """Parse the LAST line of ``runs.jsonl`` and return its `finished_at`.

    Never raises -- missing file, empty file, and an unparseable/missing
    `finished_at` on the last line all return ``None``.
    """
    import json

    try:
        with open(path, encoding="utf-8") as handle:
            lines = [line.strip() for line in handle if line.strip()]
    except OSError:
        return None
    if not lines:
        return None
    try:
        entry = json.loads(lines[-1])
        finished_at = entry["finished_at"]
        return datetime.datetime.fromisoformat(finished_at).timestamp()
    except (ValueError, KeyError, TypeError):
        return None


@dataclass(frozen=True)
class PullReport:
    pulled: list[PullDestination] = field(default_factory=list)
    planned: list[PullDestination] = field(default_factory=list)
    skipped: list[tuple[str, str]] = field(default_factory=list)
    dry_run: bool = False

    @property
    def ok(self) -> bool:
        return not self.skipped


def _fetch_logs(remote: RemoteTarget, job_dir: Path, *, runner: Runner) -> None:
    job_dir.mkdir(parents=True, exist_ok=True)
    remote_logs_path = posixpath.join(remote.path, "logs")
    logs_argv = ["rsync", "-a", f"{remote_logs_path}/", str(job_dir / "logs") + "/"]
    result = runner(logs_argv, capture_output=True, text=True)
    if result.returncode not in (0, 23):  # 23 = "no logs yet", tolerated
        raise TrackingJobError(
            f"failed to fetch logs/: {' '.join(logs_argv)}\n{result.stderr}", code=4
        )


def _check_not_running(job_dir: Path, *, force: bool) -> None:
    run_log = job_dir / "logs" / "run.log"
    runs_jsonl = job_dir / "logs" / "runs.jsonl"
    if not run_log.is_file() or force:
        return
    if not runs_jsonl.is_file():
        raise TrackingJobError(
            "the remote run has not finished yet (logs/run.log exists but "
            "logs/runs.jsonl has no entries) -- refusing to pull a partial "
            "result; pass --force to pull anyway",
            code=5,
        )
    last_entry_ts = _last_runs_jsonl_timestamp(runs_jsonl)
    if (
        last_entry_ts is None
        or last_entry_ts < run_log.stat().st_mtime - _RUNNING_CHECK_SLACK_SECONDS
    ):
        raise TrackingJobError(
            "logs/run.log was modified after the last logs/runs.jsonl entry "
            "-- the remote run looks still in progress; refusing to pull a "
            "partial result; pass --force to pull anyway",
            code=5,
        )


def _classify_skips(
    manifest: JobManifest, planned: list[PullDestination]
) -> tuple[list[PullDestination], list[tuple[str, str]]]:
    """Split `planned` into transferable entries and (relpath, reason) skips."""
    videos_by_job_path = {video.job_path: video for video in manifest.videos}
    shared_table: dict[str, str] | None = None
    keep: list[PullDestination] = []
    skipped: list[tuple[str, str]] = []
    for entry in planned:
        video = _owning_video_for(manifest, entry.job_relpath, videos_by_job_path)
        if video is None:
            keep.append(entry)
            continue
        if video.shared is not None:
            if shared_table is None:
                shared_table = load_shared_roots()
            alias = video.shared.get("alias", "")
            relpath = video.shared.get("relpath", "")
            try:
                resolved = resolve_shared(alias, relpath, shared_table)
            except TrackingJobError:
                skipped.append(
                    (
                        entry.job_relpath,
                        f"shared mount unavailable for alias '{alias}'",
                    )
                )
                continue
            if not resolved.exists():
                skipped.append(
                    (
                        entry.job_relpath,
                        f"shared mount unavailable for alias '{alias}': {resolved}",
                    )
                )
                continue
        else:
            origin_dir = Path(video.origin_path).parent
            if not origin_dir.is_dir():
                skipped.append(
                    (
                        entry.job_relpath,
                        f"origin directory missing for {entry.job_relpath}: "
                        f"{video.origin_path}",
                    )
                )
                continue
        keep.append(entry)
    return keep, skipped


def _owning_video_for(manifest: JobManifest, relpath: str, videos_by_job_path: dict):
    from pathlib import PurePosixPath

    parts = PurePosixPath(relpath).parts
    if len(parts) < 2:
        return None
    stem_candidate = parts[1]
    best = None
    best_len = -1
    for video in manifest.videos:
        stem = PurePosixPath(video.job_path).stem
        if (
            stem_candidate.startswith(stem)
            or stem_candidate == f".inference_cache_{stem}"
        ):
            if len(stem) > best_len:
                best, best_len = video, len(stem)
    return best


def pull_job(
    remote: "RemoteTarget | str",
    job_dir: Path,
    *,
    include_caches: bool = True,
    overwrite: bool = False,
    dry_run: bool = False,
    force: bool = False,
    runner: Runner,
) -> PullReport:
    target = _as_remote(remote)
    job_dir = Path(job_dir)

    # Fix Y5: step order is pinned. logs/ fetch -> running-check -> ONLY
    # THEN read the local manifest. Reversing this makes the running-check
    # dead on the "never pulled before" input shape.
    _fetch_logs(target, job_dir, runner=runner)
    _check_not_running(job_dir, force=force)

    manifest = JobManifest.read(job_dir / "hydra_job.json")
    listing = remote_video_listing(target, runner=runner)
    planned = plan_pull(manifest, listing, include_caches=include_caches)
    keep, skipped = _classify_skips(manifest, planned)

    if dry_run:
        return PullReport(pulled=[], planned=planned, skipped=skipped, dry_run=True)

    if not keep:
        pulled_at = datetime.datetime.now(datetime.timezone.utc).isoformat(
            timespec="microseconds"
        )
        _merge_pull_history(manifest, job_dir, pulled_at=pulled_at, files=0, bytes_=0)
        return PullReport(pulled=[], planned=planned, skipped=skipped, dry_run=False)

    list_path = _write_list_file([entry.job_relpath for entry in keep])
    try:
        argv = build_rsync_argv(
            f"{target.host}:{target.path}/", f"{job_dir}/", files_from=list_path
        )
        result = runner(argv, capture_output=True, text=True)
        if result.returncode != 0:
            raise TrackingJobError(
                f"failed to fetch outputs: {' '.join(argv)}\n{result.stderr}", code=4
            )
    finally:
        try:
            Path(list_path).unlink()
        except OSError:
            pass

    # Collision detection (fix M12): sha256-compare every pulled file
    # against its destination before touching any destination.
    collisions: list[str] = []
    for entry in keep:
        source = job_dir / entry.job_relpath
        destination = Path(entry.destination)
        if destination.exists() and destination.is_file() and source.is_file():
            if _sha256(source) != _sha256(destination):
                collisions.append(entry.job_relpath)
    if collisions and not overwrite:
        listed = "\n".join(f"  - {relpath}" for relpath in collisions)
        raise TrackingJobError(
            "destination collision(s) -- pass --overwrite to replace:\n" + listed,
            code=5,
        )

    pulled: list[PullDestination] = []
    total_bytes = 0
    for entry in keep:
        source = job_dir / entry.job_relpath
        destination = Path(entry.destination)
        if not source.exists():
            continue
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            destination.unlink()
        try:
            import os

            os.link(source, destination)
        except OSError:
            shutil.copy2(source, destination)
        if destination.is_file():
            total_bytes += destination.stat().st_size
        pulled.append(entry)

    pulled_at = datetime.datetime.now(datetime.timezone.utc).isoformat(
        timespec="microseconds"
    )
    _merge_pull_history(
        manifest, job_dir, pulled_at=pulled_at, files=len(pulled), bytes_=total_bytes
    )

    return PullReport(pulled=pulled, planned=planned, skipped=skipped, dry_run=False)


def _merge_pull_history(
    manifest: JobManifest, job_dir: Path, *, pulled_at: str, files: int, bytes_: int
) -> None:
    """Merge a pull-history entry, keyed on `pulled_at` (fix: idempotent retry)."""
    existing = {entry.get("pulled_at") for entry in manifest.pull_history}
    if pulled_at in existing:
        return
    entry: dict[str, Any] = {"pulled_at": pulled_at, "files": files, "bytes": bytes_}
    updated = replace(manifest, pull_history=list(manifest.pull_history) + [entry])
    updated.write(job_dir / "hydra_job.json")
