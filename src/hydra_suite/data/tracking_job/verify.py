"""Offline, mount-agnostic integrity check for a packed job directory."""

from __future__ import annotations

import json
import os
from pathlib import Path

from .manifest import JobManifest, TrackingJobError, validate_job_relpath
from .references import _sha256

ABSOLUTE_PATH_FORBIDDEN_KEYS = (
    "file_path",
    "csv_path",
    "video_output_path",
    "pose_skeleton_file",
    "yolo_obb_direct_model_path",
    "yolo_detect_model_path",
    "yolo_crop_obb_model_path",
    "yolo_headtail_model_path",
    "pose_model_dir",
    "pose_yolo_model_dir",
    "pose_sleap_model_dir",
    "pose_vitpose_model_dir",
    "color_tag_model_path",
    # Fix M2: the legacy alias engine_params.py:815 falls back to when the
    # mode-specific key is empty. Must be forbidden too, or a legacy config
    # ships an absolute path through this back door undetected.
    "yolo_model_path",
)


def _is_forbidden_absolute(value: object) -> bool:
    text = str(value or "").strip()
    if not text:
        return False
    if Path(text).is_absolute():
        return True
    return any(part == ".." for part in Path(text).parts)


def verify_job(job_dir, *, fast: bool = False) -> list[str]:
    """Return every problem found; empty means valid. Never short-circuits."""
    job_dir = Path(job_dir)
    problems: list[str] = []

    manifest_path = job_dir / "hydra_job.json"
    if not manifest_path.is_file():
        return [f"missing manifest: {manifest_path}"]
    try:
        manifest = JobManifest.read(manifest_path)
    except TrackingJobError as exc:
        return [f"manifest invalid: {exc}"]

    if manifest.job_version != 1:
        problems.append(f"unsupported job_version: {manifest.job_version}")

    # --- models ---
    for model in manifest.models:
        try:
            validate_job_relpath(model.key)
        except TrackingJobError as exc:
            problems.append(f"model {model.key}: {exc}")
            continue

        if model.kind == "directory":
            # No top-level digest exists. Integrity is the per-member check below,
            # which is why file_digests MUST be populated (fix B4).
            if not model.file_digests:
                problems.append(
                    f"model {model.key}: directory model has no file_digests; "
                    "repack with a build that populates them"
                )
        else:
            target = job_dir / "models" / model.key
            if not target.exists():
                problems.append(f"model {model.key}: missing at {target}")
            else:
                if not fast:
                    actual_size = os.path.getsize(target)
                    if actual_size != model.size_bytes:
                        problems.append(
                            f"model {model.key}: size on disk ({actual_size}) != "
                            f"manifest size_bytes ({model.size_bytes})"
                        )
                    actual_sha = _sha256(target)
                    if actual_sha != model.sha256:
                        problems.append(
                            f"model {model.key}: sha256 mismatch "
                            f"(expected {model.sha256}, got {actual_sha})"
                        )

        # For every model with a non-empty file_digests (directory models and
        # bundles -- fix M8), every listed job-relative member path exists AND
        # its sha256 matches.
        for relpath, expected_sha in model.file_digests.items():
            member = job_dir / "models" / relpath
            if not member.is_file():
                problems.append(f"model {model.key}: member missing: {relpath}")
                continue
            if not fast:
                actual_sha = _sha256(member)
                if actual_sha != expected_sha:
                    problems.append(
                        f"model {model.key}: member {relpath} sha256 mismatch "
                        f"(expected {expected_sha}, got {actual_sha})"
                    )

        for sidecar in model.sidecars:
            if not (job_dir / "models" / sidecar).exists():
                problems.append(f"model {model.key}: sidecar missing: {sidecar}")

    # --- videos ---
    video_by_job_path = {video.job_path: video for video in manifest.videos}
    for video in manifest.videos:
        try:
            validate_job_relpath(video.job_path)
            validate_job_relpath(video.config_job_path)
        except TrackingJobError as exc:
            problems.append(f"video {video.job_path}: {exc}")
            continue

        if video.shared is None:
            target = job_dir / video.job_path
            if not target.exists():
                problems.append(f"video {video.job_path}: missing at {target}")
            else:
                actual_size = os.path.getsize(target)
                if actual_size != video.size_bytes:
                    problems.append(
                        f"video {video.job_path}: size on disk ({actual_size}) != "
                        f"manifest size_bytes ({video.size_bytes})"
                    )

        config_target = job_dir / video.config_job_path
        if not config_target.is_file():
            problems.append(
                f"video {video.job_path}: config missing: {video.config_job_path}"
            )
            continue

        try:
            sidecar = json.loads(config_target.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            problems.append(f"video {video.job_path}: cannot read sidecar: {exc}")
            continue

        for key in ABSOLUTE_PATH_FORBIDDEN_KEYS:
            if key in sidecar and _is_forbidden_absolute(sidecar[key]):
                problems.append(
                    f"video {video.job_path}: sidecar key {key!r} is not job-relative: "
                    f"{sidecar[key]!r}"
                )

        entries = sidecar.get("cnn_classifiers")
        if isinstance(entries, list):
            for index, entry in enumerate(entries):
                if not isinstance(entry, dict):
                    continue
                value = entry.get("model_path")
                if _is_forbidden_absolute(value):
                    problems.append(
                        f"video {video.job_path}: cnn_classifiers[{index}].model_path "
                        f"is not job-relative: {value!r}"
                    )

        skeleton = str(sidecar.get("pose_skeleton_file", "") or "").strip()
        if skeleton:
            if not (job_dir / skeleton).exists():
                problems.append(
                    f"video {video.job_path}: referenced skeleton missing: {skeleton}"
                )

        for sibling in video.pushed_siblings:
            try:
                validate_job_relpath(sibling)
            except TrackingJobError as exc:
                problems.append(f"video {video.job_path}: {exc}")

    # --- videos.txt ---
    videos_txt = job_dir / "videos.txt"
    if not videos_txt.is_file():
        problems.append("missing videos.txt")
    else:
        lines = [
            line for line in videos_txt.read_text(encoding="utf-8").splitlines() if line
        ]
        if not lines or lines[0] != manifest.keystone.get("video"):
            problems.append(
                f"videos.txt keystone mismatch: expected "
                f"{manifest.keystone.get('video')!r}, got {lines[0] if lines else None!r}"
            )
        for line in lines:
            video = video_by_job_path.get(line)
            if video is None:
                problems.append(f"videos.txt: unknown video {line!r}")
                continue
            if video.shared is not None:
                # A shared video has no file under videos/ at pack time (spec
                # §6.7): it is materialized on the remote by preflight.
                continue
            target = job_dir / line
            if not target.exists():
                problems.append(f"videos.txt: {line} does not exist on disk")

    return problems
