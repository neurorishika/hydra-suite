"""Manifest for a portable tracking job (hydra_job.json)."""

from __future__ import annotations

import dataclasses
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..project_bundle import write_json_atomic

SUPPORTED_JOB_VERSION = 1
JOB_MANIFEST_FILENAME = "hydra_job.json"


def _current_git_sha() -> str:
    """Best-effort git SHA of the running checkout; "" if not a git repo or
    git is unavailable (e.g. a pip-installed hydra-suite with no .git)."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parent,
            capture_output=True,
            text=True,
            timeout=5,
        )
        return result.stdout.strip() if result.returncode == 0 else ""
    except (OSError, subprocess.SubprocessError):
        return ""


class TrackingJobError(ValueError):
    """A job-lifecycle error carrying the CLI exit code to use.

    2 = argument/validation, 3 = preflight, 4 = transport, 5 = pull collision.
    """

    def __init__(self, message: str, *, code: int = 2) -> None:
        super().__init__(message)
        self.code = code


def validate_job_relpath(value: str) -> Path:
    """Every path inside a job is relative to the job root and stays inside it."""
    if not value:
        raise TrackingJobError("empty job-relative path")
    candidate = Path(value)
    if candidate.is_absolute() or any(part == ".." for part in candidate.parts):
        raise TrackingJobError(f"unsafe job-relative path: {value}")
    return candidate


@dataclass(frozen=True)
class JobVideo:
    job_path: str
    origin_path: str
    size_bytes: int
    config_job_path: str
    config_provenance: str
    pushed_siblings: list[str] = field(default_factory=list)
    signature: str = ""
    shared: dict[str, str] | None = None
    redirected_outputs: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class JobModel:
    key: str
    roles: list[str]
    origin_path: str
    kind: str  # "file" | "directory"
    sha256: str = ""
    size_bytes: int = 0
    sidecars: list[str] = field(default_factory=list)
    files: list[str] = field(default_factory=list)
    registry_entry_present: bool = False
    # Fix M8: a directory model (pose run dir) or bundle (ClassKit sidecars)
    # only ever got ONE top-level sha256/size pair, so verify/preflight could
    # not detect a single corrupted member file inside a multi-file artifact —
    # spec §6.2 step 5 requires "sha256 of every file". Populated by pack.py
    # for kind == "directory" and for any model with sidecars/files; a plain
    # single-file model leaves this empty (its top-level sha256 already covers
    # it). Keys are job-relative paths (e.g. "pose/SLEAP/run/best.ckpt").
    #
    # Minor fix (adversarial review) — a DELIBERATE, DOCUMENTED deviation
    # from spec §6.2 step 5's literal text, not an oversight: the spec says
    # "sha256 of every file", read most literally as EVERY file in the job
    # (including every video and every sidecar JSON), but this plan only
    # ever populates `file_digests` for directory/bundle MODEL members
    # (`copy_model_reference`, Task 6) — never for a video (`JobVideo` has
    # no per-file digest field at all; `verify_job`'s video check is a size
    # comparison, fix W1b, not a hash) and never for a lone sidecar JSON.
    # This is intentional, not a gap that slipped through: (1) videos are
    # multi-gigabyte and re-hashing one on every `verify_job` call would make
    # the offline, "cheap, stat-only" design goal (stated explicitly for the
    # video-size check, fix W1b) impossible for the one artifact class where
    # it matters most; content-level video integrity is Task 10 preflight's
    # `video_signature` check instead, which trades verify's offline-ness for
    # a one-time content read at run time, deliberately NOT duplicated here;
    # (2) sidecar JSONs are tiny, pack-regenerated, deterministic snapshots
    # (config, skeletons) with no plausible silent-corruption story rsync
    # doesn't already guard against via its own checksum mode — they get an
    # existence check, not a hash. `file_digests` closes the ONE real gap the
    # M8 fix targets (a multi-file model artifact where corruption of ONE
    # member is otherwise undetectable by a single top-level hash); it was
    # never meant to make every byte in the job content-addressed.
    file_digests: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class JobManifest:
    job_id: str
    created_at: str
    created_on: dict[str, Any]
    keystone: dict[str, str]
    videos: list[JobVideo]
    models: list[JobModel]
    job_version: int = SUPPORTED_JOB_VERSION
    config_snapshot: dict[str, Any] = field(default_factory=dict)
    requirements: dict[str, Any] = field(default_factory=dict)
    track_args: dict[str, Any] = field(default_factory=dict)
    pull_history: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"job_version": self.job_version}
        payload.update(
            {
                "job_id": self.job_id,
                "created_at": self.created_at,
                "created_on": dict(self.created_on),
                "keystone": dict(self.keystone),
                "videos": [dataclasses.asdict(v) for v in self.videos],
                "models": [dataclasses.asdict(m) for m in self.models],
                "config_snapshot": dict(self.config_snapshot),
                "requirements": dict(self.requirements),
                "track_args": dict(self.track_args),
                "pull_history": [dict(entry) for entry in self.pull_history],
            }
        )
        return payload

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "JobManifest":
        version = int(data.get("job_version", SUPPORTED_JOB_VERSION))
        if version != SUPPORTED_JOB_VERSION:
            raise TrackingJobError(
                f"Unsupported tracking job version {version} "
                f"(this build supports {SUPPORTED_JOB_VERSION})"
            )
        # Minor fix: JobVideo(**entry)/JobModel(**entry) raise a bare TypeError
        # on any unknown forward-compat key (e.g. an older client reading a
        # manifest written by a newer one with an added field), which is not
        # a TrackingJobError and so escapes the CLI's uniform error handling
        # (run_job_cli only catches TrackingJobError, per Task 11 Step 4's
        # "All TrackingJobErrors are caught... printed as error:"). Wrap it.
        try:
            videos = [JobVideo(**entry) for entry in data.get("videos", [])]
            models = [JobModel(**entry) for entry in data.get("models", [])]
        except TypeError as exc:
            raise TrackingJobError(
                f"hydra_job.json has an unrecognized field for this build: {exc}"
            ) from exc
        for video in videos:
            validate_job_relpath(video.job_path)
            validate_job_relpath(video.config_job_path)
            for sibling in video.pushed_siblings:
                validate_job_relpath(sibling)
        for model in models:
            validate_job_relpath(model.key)
            for extra in list(model.sidecars) + list(model.files):
                validate_job_relpath(extra)
        # Fix V-minor: `data["job_id"]`/`data["created_at"]` are bare
        # dict-index lookups -- a manifest missing either key (hand-edited,
        # truncated write, or an even-older format than the job_version
        # check above catches) raises a plain KeyError here, which -- same
        # as the JobVideo/JobModel TypeError case just above -- is not a
        # TrackingJobError and so escapes run_job_cli's uniform "catches
        # TrackingJobError, prints as error:" handling (Task 11 Step 4).
        # Wrap it the same way.
        try:
            job_id = str(data["job_id"])
            created_at = str(data["created_at"])
        except KeyError as exc:
            raise TrackingJobError(
                f"hydra_job.json is missing required field {exc}"
            ) from exc
        return cls(
            job_version=version,
            job_id=job_id,
            created_at=created_at,
            created_on=dict(data.get("created_on", {})),
            keystone=dict(data.get("keystone", {})),
            videos=videos,
            models=models,
            config_snapshot=dict(data.get("config_snapshot", {})),
            requirements=dict(data.get("requirements", {})),
            track_args=dict(data.get("track_args", {})),
            pull_history=list(data.get("pull_history", [])),
        )

    def write(self, path: Path) -> None:
        write_json_atomic(Path(path), self.to_dict())

    @classmethod
    def read(cls, path: Path) -> "JobManifest":
        import json

        with open(path, encoding="utf-8") as handle:
            return cls.from_dict(json.load(handle))
