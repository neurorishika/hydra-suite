"""Host preflight for a packed tracking job.

Runs every check (never short-circuits), materializes shared-root videos as
symlinks, and writes ``logs/preflight.json`` -- the report a compute box's
``run.sh`` (and its own ``set -euo pipefail``) rely on to abort BEFORE
``track`` ever touches a possibly-truncated or wrong video.
"""

from __future__ import annotations

import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ...core.inference import content_id
from ...paths import get_platform_config_dir
from .manifest import JobManifest, TrackingJobError
from .shared_roots import load_shared_roots, resolve_shared
from .verify import verify_job

_HOST_CFG_VAR = "HYDRA_HOST_CONFIG_DIR"

# fix: shared videos' bytes are not counted against the video-size disk
# estimate (they live on the mount, not on this filesystem), but caches
# derived from them still land locally, so the 1.5x safety margin is applied
# to the full total (spec §9.4 check 8).
_DISK_SAFETY_FACTOR = 1.5


@dataclass(frozen=True)
class PreflightResult:
    ok: bool
    checks: list[dict[str, Any]] = field(default_factory=list)


class _CheckList:
    """Accumulates `{"name", "ok", "detail"}` entries; never short-circuits."""

    def __init__(self) -> None:
        self._checks: list[dict[str, Any]] = []

    def record(self, name: str, ok: bool, detail: str = "") -> None:
        self._checks.append({"name": name, "ok": bool(ok), "detail": detail})

    @property
    def checks(self) -> list[dict[str, Any]]:
        return list(self._checks)

    @property
    def ok(self) -> bool:
        return all(c["ok"] for c in self._checks)


def _resolve_host_shared_roots_table() -> dict[str, str]:
    """Fix X1/X1b: branch on PRESENCE of HYDRA_HOST_CONFIG_DIR, not truthiness.

    run.sh does ``export HYDRA_HOST_CONFIG_DIR="${HYDRA_CONFIG_DIR:-}"`` --
    on a host using the platformdirs default that exports the EMPTY STRING,
    which must resolve via ``get_platform_config_dir()`` (never
    ``HYDRA_CONFIG_DIR``, which by then points at the job's own redirected
    config/ snapshot, which never has a shared_roots.json).
    """
    if _HOST_CFG_VAR in os.environ:
        host_cfg = os.environ[_HOST_CFG_VAR]
        if host_cfg:
            # (a) run.sh, host had a HYDRA_CONFIG_DIR override at launch.
            return load_shared_roots(Path(host_cfg) / "shared_roots.json")
        # (b) run.sh, host used the platformdirs default -- HYDRA_CONFIG_DIR
        # has already been redirected to the job's own config/, so it must
        # NOT be consulted here.
        return load_shared_roots(get_platform_config_dir() / "shared_roots.json")
    # (c) invoked directly, never through run.sh -- no redirection has
    # happened, so HYDRA_CONFIG_DIR (or its own platformdirs default) is
    # correct as-is.
    return load_shared_roots()


def materialize_shared_videos(
    manifest: JobManifest, job_dir: Path, table: dict[str, str]
) -> None:
    """Symlink every shared video's `videos/<basename>` at the resolved mount
    path. Replaces an existing SYMLINK, never a regular file (fix B6)."""
    job_dir = Path(job_dir)
    for video in manifest.videos:
        if video.shared is None:
            continue
        alias = video.shared["alias"]
        relpath = video.shared["relpath"]
        source = resolve_shared(alias, relpath, table)
        link_path = job_dir / "videos" / Path(video.job_path).name
        link_path.parent.mkdir(parents=True, exist_ok=True)
        if link_path.is_symlink() or link_path.exists():
            if link_path.is_symlink():
                link_path.unlink()
            elif not link_path.exists():
                pass
            else:
                # A regular file sits where the symlink should go -- never
                # replace it (fix B6).
                continue
        try:
            link_path.symlink_to(source)
        except FileExistsError:
            continue


def _check_manifest(
    checks: _CheckList, job_dir: Path, *, fast: bool
) -> JobManifest | None:
    manifest_path = job_dir / "hydra_job.json"
    if not manifest_path.is_file():
        checks.record("manifest", False, f"missing manifest: {manifest_path}")
        return None
    try:
        manifest = JobManifest.read(manifest_path)
    except TrackingJobError as exc:
        checks.record("manifest", False, f"manifest invalid: {exc}")
        return None

    problems = verify_job(job_dir, fast=fast)
    if problems:
        checks.record("manifest", False, "; ".join(problems))
        return manifest
    checks.record("manifest", True)
    return manifest


def _check_version(checks: _CheckList, manifest: JobManifest) -> None:
    import hydra_suite

    installed = str(getattr(hydra_suite, "__version__", "0"))
    required = str(manifest.requirements.get("min_hydra_suite_version", "0"))

    def _parse(v: str) -> tuple[int, ...]:
        parts: list[int] = []
        for part in v.split("."):
            digits = "".join(ch for ch in part if ch.isdigit())
            parts.append(int(digits) if digits else 0)
        return tuple(parts)

    if _parse(installed) < _parse(required):
        checks.record(
            "version",
            False,
            f"hydra_suite {installed} < required {required}",
        )
        return
    checks.record("version", True)


def _check_conda_envs(
    checks: _CheckList, manifest: JobManifest, conda_envs: tuple[str, ...] | None
) -> None:
    required = list(manifest.requirements.get("conda_envs", []) or [])
    available = set(conda_envs or ())
    missing = [env for env in required if env not in available]
    if missing:
        checks.record(
            "conda_envs",
            False,
            f"missing conda env(s): {', '.join(missing)}",
        )
        return
    checks.record("conda_envs", True)


def _check_runtime_tier(
    checks: _CheckList,
    manifest: JobManifest,
    available_tiers: tuple[str, ...] | None,
    allow_tier_fallback: bool,
) -> None:
    if available_tiers is None:
        return
    requested = str(manifest.requirements.get("runtime_tier", "gpu"))
    if requested in available_tiers:
        checks.record("runtime_tier", True)
        return
    fallback = available_tiers[0] if available_tiers else None
    if allow_tier_fallback:
        checks.record(
            "runtime_tier",
            True,
            f"requested tier {requested!r} unavailable; falling back to {fallback!r}",
        )
        return
    checks.record(
        "runtime_tier",
        False,
        f"requested tier {requested!r} not in available tiers "
        f"{list(available_tiers)}; would fall back to {fallback!r} "
        "(pass allow_tier_fallback=True to accept)",
    )


def _check_shared_roots(
    checks: _CheckList,
    manifest: JobManifest,
    job_dir: Path,
    shared_root_overrides: dict[str, str] | None,
) -> None:
    shared_videos = [v for v in manifest.videos if v.shared is not None]
    if not shared_videos:
        return

    table = dict(_resolve_host_shared_roots_table())
    table.update(shared_root_overrides or {})

    problems: list[str] = []
    for video in shared_videos:
        alias = video.shared["alias"]
        relpath = video.shared["relpath"]
        if alias not in table:
            known = ", ".join(sorted(table)) or "(none configured)"
            problems.append(
                f"video {video.job_path}: unknown shared-root alias {alias!r}; "
                f"known aliases: {known}"
            )
            continue
        source = Path(table[alias]) / relpath
        if not source.is_file():
            problems.append(
                f"video {video.job_path}: shared source missing: "
                f"{video.job_path} (expected at {source})"
            )
            continue
        actual_size = os.path.getsize(source)
        if actual_size != video.size_bytes:
            problems.append(
                f"video {video.job_path}: shared source size ({actual_size}) != "
                f"manifest size_bytes ({video.size_bytes}) at {source}"
            )
        actual_signature = content_id.video_signature(source)
        if actual_signature != video.signature:
            problems.append(
                f"video {video.job_path}: shared source signature mismatch at "
                f"{source} (expected {video.signature!r}, got {actual_signature!r})"
            )

    if problems:
        checks.record("shared_roots", False, "; ".join(problems))
        return

    materialize_shared_videos(manifest, job_dir, table)
    checks.record("shared_roots", True)


def _check_video_signature(
    checks: _CheckList, manifest: JobManifest, job_dir: Path
) -> None:
    problems: list[str] = []
    for video in manifest.videos:
        target = job_dir / video.job_path
        resolved = target.resolve() if target.exists() else target
        actual_signature = content_id.video_signature(resolved)
        if not actual_signature or actual_signature != video.signature:
            problems.append(
                f"video {video.job_path}: content signature mismatch "
                f"(expected {video.signature!r}, got {actual_signature!r})"
            )
    if problems:
        checks.record("video_signature", False, "; ".join(problems))
        return
    checks.record("video_signature", True)


def _check_disk(checks: _CheckList, manifest: JobManifest, job_dir: Path) -> None:
    total_bytes = sum(v.size_bytes for v in manifest.videos)
    required_bytes = int(total_bytes * _DISK_SAFETY_FACTOR)
    videos_dir = job_dir / "videos"
    videos_dir.mkdir(parents=True, exist_ok=True)
    usage = shutil.disk_usage(str(videos_dir))
    if usage.free < required_bytes:
        checks.record(
            "disk",
            False,
            f"free space {usage.free} bytes < required {required_bytes} bytes "
            f"({_DISK_SAFETY_FACTOR}x {total_bytes} bytes of video)",
        )
        return
    checks.record("disk", True)


def preflight_job(
    job_dir,
    *,
    shared_root_overrides: dict[str, str] | None = None,
    fast: bool = False,
    allow_tier_fallback: bool = False,
    available_tiers: tuple[str, ...] | None = None,
    conda_envs: tuple[str, ...] | None = None,
) -> PreflightResult:
    """Run every check (spec §9.4), never short-circuiting. Writes
    ``logs/preflight.json``."""
    from ..project_bundle import write_json_atomic

    job_dir = Path(job_dir)
    checks = _CheckList()

    manifest = _check_manifest(checks, job_dir, fast=fast)
    if manifest is None:
        result = PreflightResult(ok=checks.ok, checks=checks.checks)
        write_json_atomic(
            job_dir / "logs" / "preflight.json",
            {"ok": result.ok, "checks": result.checks},
        )
        return result

    _check_version(checks, manifest)
    _check_conda_envs(checks, manifest, conda_envs)
    _check_runtime_tier(checks, manifest, available_tiers, allow_tier_fallback)
    # `models` re-runs verify_job's own hashing under its own check name
    # (spec §9.4 check 5), independent of the "manifest" check above so a
    # tampered model is reported under BOTH names when relevant.
    model_problems = [
        p for p in verify_job(job_dir, fast=fast) if p.startswith("model ")
    ]
    if model_problems:
        checks.record("models", False, "; ".join(model_problems))
    else:
        checks.record("models", True)
    _check_shared_roots(checks, manifest, job_dir, shared_root_overrides)
    _check_video_signature(checks, manifest, job_dir)
    _check_disk(checks, manifest, job_dir)

    result = PreflightResult(ok=checks.ok, checks=checks.checks)
    write_json_atomic(
        job_dir / "logs" / "preflight.json",
        {"ok": result.ok, "checks": result.checks},
    )
    return result
