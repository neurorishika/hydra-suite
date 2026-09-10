"""Pack a job directory: copy → rewrite → runner → verify."""

from __future__ import annotations

import copy
import dataclasses
import datetime
import os
import platform
import shutil
import socket
import stat
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from ...core.inference import content_id
from ...core.tracking.session_policy import is_pose_inference_enabled
from ..project_bundle import write_json_atomic
from .manifest import JobManifest, JobModel, JobVideo, TrackingJobError
from .references import PlannedModel, copy_model_reference, write_registry_subset
from .runner import render_run_sh
from .shared_roots import match_shared_root
from .verify import verify_job

# The FULL role -> config-key mapping. Every role iter_model_references
# (Task 3) can yield MUST appear here or _rewrite_config silently leaves an
# absolute path in the sidecar for that role.
ROLE_TO_CONFIG_KEY: dict[str, "str | tuple[str, ...]"] = {
    "YOLO_OBB_DIRECT_MODEL_PATH": "yolo_obb_direct_model_path",
    "YOLO_DETECT_MODEL_PATH": "yolo_detect_model_path",
    "YOLO_CROP_OBB_MODEL_PATH": "yolo_crop_obb_model_path",
    "YOLO_HEADTAIL_MODEL_PATH": "yolo_headtail_model_path",
    # POSE_MODEL_DIR fans out to every backend-specific alias engine_params.py
    # also populates from the same directory, plus the legacy singular bridge.
    # Only the ACTIVE backend's alias (plus the legacy bridge) is ever
    # populated for a given video -- see _scalar_model_keys.
    "POSE_MODEL_DIR": (
        "pose_model_dir",
        "pose_yolo_model_dir",
        "pose_sleap_model_dir",
        "pose_vitpose_model_dir",
    ),
    # CNN_CLASSIFIERS is handled separately: it rewrites cnn_classifiers[].model_path
    # entries in place, not a single scalar config key.
}
# Legacy alias: engine_params.py:814-822 falls back to yolo_model_path when
# the mode-specific key is empty -- MODE-DEPENDENTLY: direct mode reads
# yolo_obb_direct_model_path then yolo_model_path, sequential mode reads
# yolo_crop_obb_model_path then yolo_model_path.
LEGACY_ALIAS_CONFIG_KEYS = ("yolo_model_path",)

# The model-path subset of verify.ABSOLUTE_PATH_FORBIDDEN_KEYS: every key that
# names a model, as opposed to a video/CSV/skeleton path. An inactive role's
# key in this subset is blanked to "" rather than left holding a stale
# absolute path (fix X6).
_MODEL_PATH_KEYS = (
    "yolo_obb_direct_model_path",
    "yolo_detect_model_path",
    "yolo_crop_obb_model_path",
    "yolo_headtail_model_path",
    "pose_model_dir",
    "pose_yolo_model_dir",
    "pose_sleap_model_dir",
    "pose_vitpose_model_dir",
    "yolo_model_path",
)

# Host-local pack outputs that must never be reimplemented as `job clean`'s
# job (fix X8): a stray file directly under videos/ that no OLD manifest
# entry accounts for blocks a --force repack unless force_discard_outputs is
# also passed.
_KEEP_ALWAYS = {"logs", "hydra_job.json"}


@dataclass(frozen=True)
class PlannedVideo:
    """One video to pack, already resolved by the app layer."""

    video_path: str
    config: dict[str, Any]
    config_provenance: str
    planned_models: list[PlannedModel]
    skeleton_path: str
    # Fix B5: per-entry map from a CNN_CLASSIFIERS source path (normalized via
    # _normalize_model_path) to the job key it was assigned. Built by the
    # caller (job_cli.py), which is the only code that walked
    # iter_model_references and saw the per-entry association.
    cnn_model_keys: dict[str, str] = field(default_factory=dict)


def _normalize_model_path(value: object) -> str:
    """Canonical lookup form for a CNN classifier entry's model_path."""
    from ...core.inference.model_paths import resolve_model_path

    raw = str(value or "").strip()
    if not raw:
        return ""
    return str(Path(str(resolve_model_path(raw))).expanduser().resolve())


def _default_output_paths(video_source_path: str) -> tuple[str, str]:
    """Mirrors trackerkit/cli_config.py:270-276's own body (three-line pure
    function of the source path) -- inlined here rather than imported, since
    that module lives in the trackerkit app layer and pack.py is Data."""
    video = Path(video_source_path)
    base = video.with_suffix("")
    return (
        str(base.parent / f"{base.name}_tracking.csv"),
        str(base.parent / f"{base.name}_tracking.mp4"),
    )


def _rewrite_config(
    config,
    *,
    video_source_path,
    video_basename,
    model_keys,
    cnn_model_keys,
    skeleton_job_path,
):
    """Make one video's config job-relative. Returns (config, redirected)."""
    out = copy.deepcopy(dict(config))
    stem = Path(video_basename).stem
    out["file_path"] = f"videos/{video_basename}"
    redirected: dict[str, str] = {}
    default_csv, default_video_out = _default_output_paths(video_source_path)
    default_paths = {"csv_path": default_csv, "video_output_path": default_video_out}
    for key, default_suffix in (
        ("csv_path", "_tracking.csv"),
        ("video_output_path", "_tracking.mp4"),
    ):
        original = str(out.get(key, "") or "")
        if not original:
            continue
        default_name = f"{stem}{default_suffix}"
        is_default_location = str(Path(original).resolve()) == str(
            Path(default_paths[key]).resolve()
        )
        if is_default_location:
            out[key] = f"videos/{default_name}"
        else:
            job_relative = f"videos/{stem}_{Path(original).name}"
            out[key] = job_relative
            redirected[job_relative] = str(Path(original).resolve())
    for config_key, job_key in model_keys.items():
        out[config_key] = job_key
    # Fix B5: cnn_classifiers[].model_path is a LIST of dicts, not a scalar
    # key, so it cannot go through model_keys.
    entries = out.get("cnn_classifiers")
    if isinstance(entries, list) and cnn_model_keys:
        rewritten = []
        for entry in entries:
            entry = dict(entry)
            lookup = _normalize_model_path(entry.get("model_path", ""))
            if lookup and lookup in cnn_model_keys:
                entry["model_path"] = cnn_model_keys[lookup]
            rewritten.append(entry)
        out["cnn_classifiers"] = rewritten
    # Fix B2: unconditional on pose_skeleton_file being non-empty, NOT gated
    # on pose being enabled.
    if skeleton_job_path:
        out["pose_skeleton_file"] = skeleton_job_path
    return out, redirected


def _scalar_model_keys(
    config: dict[str, Any], planned_models: list[PlannedModel]
) -> dict[str, str]:
    """Fix M2/V6: role -> config-key rewrite table, active-backend-only for
    the POSE_MODEL_DIR tuple role."""
    role_to_job_key: dict[str, str] = {}
    for model in planned_models:
        if model.role == "CNN_CLASSIFIERS":
            continue
        role_to_job_key[model.role] = model.key

    model_keys: dict[str, str] = {}
    for role, config_key in ROLE_TO_CONFIG_KEY.items():
        if role == "POSE_MODEL_DIR":
            continue
        if role in role_to_job_key:
            model_keys[config_key] = role_to_job_key[role]

    if "POSE_MODEL_DIR" in role_to_job_key:
        pose_job_key = role_to_job_key["POSE_MODEL_DIR"]
        raw_type = str(config.get("pose_model_type", "yolo")).strip().lower()
        if raw_type not in {"yolo", "sleap", "vitpose"}:
            raw_type = "yolo"
        model_keys[f"pose_{raw_type}_model_dir"] = pose_job_key
        model_keys["pose_model_dir"] = pose_job_key

    # WHICH JOB KEY IT TAKES (fix B5): yolo_model_path takes the same job key
    # as whichever OBB role is live for this video.
    obb_role = (
        "YOLO_OBB_DIRECT_MODEL_PATH"
        if str(config.get("yolo_obb_mode", "direct")).strip().lower() != "sequential"
        else "YOLO_CROP_OBB_MODEL_PATH"
    )
    if obb_role in role_to_job_key:
        for alias in LEGACY_ALIAS_CONFIG_KEYS:
            model_keys[alias] = role_to_job_key[obb_role]

    return model_keys


def _blank_inactive_model_keys(out: dict[str, Any], model_keys: dict[str, str]) -> None:
    """Fix X6/Q2: an inactive role's model-path key, and the dead
    color_tag_model_path key, are always blanked -- never left holding a
    stale absolute source-machine path."""
    for key in _MODEL_PATH_KEYS:
        if key not in model_keys and key in out:
            out[key] = ""
    if "color_tag_model_path" in out:
        out["color_tag_model_path"] = ""


def _sleap_conda_env(cfg: dict[str, Any]) -> str:
    """Fix Y8: mirror engine_params.py:995-998's own defaulting verbatim."""
    raw_sleap_env = str(cfg.get("pose_sleap_env", "") or "").strip()
    if not raw_sleap_env or raw_sleap_env.lower().startswith("no sleap envs"):
        raw_sleap_env = "sleap"
    return raw_sleap_env


def _resolve_skeleton_job_paths(
    job_dir: Path, planned_videos: list[PlannedVideo]
) -> dict[str, str]:
    """Copy every distinct skeleton into config/skeletons/, deduped by
    basename+content; raise on a genuine basename collision (minor fix)."""
    skeletons_dir = job_dir / "config" / "skeletons"
    skeletons_dir.mkdir(parents=True, exist_ok=True)
    job_paths: dict[str, str] = {}  # resolved source path -> job-relative path
    target_sources: dict[str, str] = {}  # job-relative path -> resolved source path
    for planned in planned_videos:
        if not planned.skeleton_path:
            continue
        source = Path(planned.skeleton_path).expanduser()
        resolved = str(source.resolve())
        if resolved in job_paths:
            continue
        name = source.name
        target_rel = f"config/skeletons/{name}"
        if target_rel in target_sources and target_sources[target_rel] != resolved:
            if content_id.file_content_id(
                target_sources[target_rel]
            ) != content_id.file_content_id(resolved):
                raise TrackingJobError(
                    f"skeleton basename collision: {target_rel!r} would be overwritten "
                    f"by both {target_sources[target_rel]!r} and {resolved!r} "
                    f"(different content)",
                    code=2,
                )
            job_paths[resolved] = target_rel
            continue
        target_sources[target_rel] = resolved
        job_paths[resolved] = target_rel
        destination = job_dir / target_rel
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
    return job_paths


def _check_basename_collisions(planned_videos: list[PlannedVideo]) -> None:
    by_basename: dict[str, dict[str, str]] = {}
    for planned in planned_videos:
        basename = Path(planned.video_path).name
        resolved = str(Path(planned.video_path).resolve())
        by_basename.setdefault(basename, {})[resolved] = planned.video_path
    for basename, origins in by_basename.items():
        if len(origins) > 1:
            names = ", ".join(sorted(origins.values()))
            raise TrackingJobError(
                f"two videos would collide at videos/{basename}: {names}", code=2
            )


def _prepack_force_cleanup(
    job_dir: Path, *, force: bool, force_discard_outputs: bool
) -> JobManifest | None:
    """Fix V-minor/X8: refuse a non-empty job_dir unless force=True; when
    forced, remove only manifest-known pack artifacts and refuse to discard
    anything else (pulled outputs) unless force_discard_outputs=True too."""
    if not job_dir.exists() or not any(job_dir.iterdir()):
        return None
    if not force:
        raise TrackingJobError(
            f"{job_dir} is not empty; pass force=True to re-pack into it", code=2
        )

    old_manifest: JobManifest | None = None
    manifest_path = job_dir / "hydra_job.json"
    if manifest_path.is_file():
        old_manifest = JobManifest.read(manifest_path)

    if old_manifest is not None:
        # Step 1: remove only manifest-KNOWN pack artifacts, tolerating a
        # shared video (which has no local file at all).
        for video in old_manifest.videos:
            for relpath in (video.job_path, video.config_job_path):
                (job_dir / relpath).unlink(missing_ok=True)

        # Step 2: refuse if videos/ contains anything unaccounted for, unless
        # force_discard_outputs=True.
        videos_dir = job_dir / "videos"
        if videos_dir.exists():
            stray = [
                path.relative_to(job_dir).as_posix()
                for path in sorted(videos_dir.rglob("*"))
                if path.is_file()
            ]
            if stray:
                if force_discard_outputs:
                    for relpath in stray:
                        (job_dir / relpath).unlink(missing_ok=True)
                    for sub in sorted(videos_dir.glob("**/*"), reverse=True):
                        if sub.is_dir() and not any(sub.iterdir()):
                            sub.rmdir()
                else:
                    names = ", ".join(stray)
                    raise TrackingJobError(
                        f"{videos_dir} contains files the previous pack does not "
                        f"account for (pass force_discard_outputs=True to remove "
                        f"them): {names}",
                        code=2,
                    )

    # models/ and config/ hold only pack-owned, deterministically-regenerable
    # artifacts -- no pull-time output is ever written into either.
    for name in ("models", "config"):
        shutil.rmtree(job_dir / name, ignore_errors=True)

    return old_manifest


def pack_job(
    job_dir,
    planned_videos: list[PlannedVideo],
    *,
    registry_entries: Iterable[tuple[str, dict]],
    advanced_config_path: str,
    advanced_config_fallback: dict[str, Any] | None = None,
    track_args: dict[str, Any],
    shared_table: dict[str, str],
    copy_videos: bool = False,
    shared_mode: str = "auto",
    job_name: str | None = None,
    force: bool = False,
    force_discard_outputs: bool = False,
) -> JobManifest:
    # Fix Q4: materialize ONCE -- registry_entries may be a one-shot
    # generator, consumed both by write_registry_subset and by the
    # registry_entry_present computation below.
    registry_entries = list(registry_entries)

    job_dir = Path(job_dir)
    old_manifest = _prepack_force_cleanup(
        job_dir, force=force, force_discard_outputs=force_discard_outputs
    )

    job_dir.mkdir(parents=True, exist_ok=True)
    videos_dir = job_dir / "videos"
    models_dir = job_dir / "models"
    config_dir = job_dir / "config"
    presets_dir = config_dir / "presets"
    skeletons_dir = config_dir / "skeletons"
    for directory in (videos_dir, models_dir, config_dir, presets_dir, skeletons_dir):
        directory.mkdir(parents=True, exist_ok=True)

    _check_basename_collisions(planned_videos)

    # Fix X3a / skeleton-mismatch guard: run BEFORE any copy/write.
    for planned in planned_videos:
        basename = Path(planned.video_path).name
        raw_skeleton = str(planned.config.get("pose_skeleton_file", "") or "").strip()
        if raw_skeleton and not planned.skeleton_path:
            raise TrackingJobError(
                f"{basename}: config sets pose_skeleton_file ({raw_skeleton!r}) but "
                f"PlannedVideo.skeleton_path is empty; the caller must resolve and "
                f"stamp skeleton_path to match"
            )
        if (
            is_pose_inference_enabled(planned.config)
            and not str(planned.config.get("pose_skeleton_file", "") or "").strip()
        ):
            raise TrackingJobError(
                f"{basename}: pose inference is enabled but pose_skeleton_file is "
                f"empty; pack cannot produce a job that will fail at remote model load"
            )

    # --- copy models (dedup by key across ALL videos) ---
    unique_models: dict[str, PlannedModel] = {}
    roles_by_key: dict[str, set[str]] = {}
    for planned in planned_videos:
        for model in planned.planned_models:
            roles_by_key.setdefault(model.key, set()).add(model.role)
            unique_models.setdefault(model.key, model)

    registry_keys = {key for key, _ in registry_entries}
    job_models: list[JobModel] = []
    for key, planned_model in unique_models.items():
        job_model = copy_model_reference(planned_model, models_dir)
        job_model = dataclasses.replace(
            job_model,
            roles=sorted(roles_by_key[key]),
            registry_entry_present=key in registry_keys,
        )
        job_models.append(job_model)

    write_registry_subset(
        set(unique_models), registry_entries, models_dir / "model_registry.json"
    )

    # --- advanced config (fix V4/X5a) ---
    source = Path(advanced_config_path)
    if source.exists():
        shutil.copy2(source, config_dir / "advanced_config.json")
    elif advanced_config_fallback is not None:
        write_json_atomic(config_dir / "advanced_config.json", advanced_config_fallback)
    else:
        raise TrackingJobError(
            f"advanced_config_path {source} does not exist and no "
            f"advanced_config_fallback was supplied; the caller must resolve "
            f"load_advanced_tracker_config() itself (pack.py never imports "
            f"trackerkit) before calling pack_job"
        )

    # --- skeletons ---
    skeleton_job_paths = _resolve_skeleton_job_paths(job_dir, planned_videos)

    # --- .seeded markers ---
    (presets_dir / ".seeded").touch()
    (skeletons_dir / ".seeded").touch()

    # --- per-video: resolve shared/symlink/copy, rewrite config, write sidecar ---
    job_videos: list[JobVideo] = []
    for planned in planned_videos:
        origin_path = str(Path(planned.video_path).resolve())
        basename = Path(planned.video_path).name
        stem = Path(basename).stem
        job_path = f"videos/{basename}"

        # Fix W1: signature and size are computed unconditionally, regardless
        # of which materialization branch this video takes.
        signature = content_id.video_signature(planned.video_path)
        size_bytes = os.path.getsize(planned.video_path)

        match = None
        if shared_mode != "no-shared":
            match = match_shared_root(planned.video_path, shared_table)
        if shared_mode == "shared-only" and match is None:
            raise TrackingJobError(
                f"{planned.video_path} is not under any configured shared root "
                f"(shared_mode=shared-only)",
                code=3,
            )

        shared_entry: dict[str, str] | None = None
        if match is not None and shared_mode != "no-shared":
            alias, relpath = match
            shared_entry = {"alias": alias, "relpath": relpath}
        else:
            target = videos_dir / basename
            if copy_videos:
                shutil.copy2(Path(planned.video_path), target)
            else:
                os.symlink(origin_path, target)

        model_keys = _scalar_model_keys(planned.config, planned.planned_models)
        skeleton_job_path = skeleton_job_paths.get(
            (
                str(Path(planned.skeleton_path).expanduser().resolve())
                if planned.skeleton_path
                else ""
            ),
            "",
        )
        rewritten, redirected = _rewrite_config(
            planned.config,
            video_source_path=origin_path,
            video_basename=basename,
            model_keys=model_keys,
            cnn_model_keys=planned.cnn_model_keys,
            skeleton_job_path=skeleton_job_path,
        )
        _blank_inactive_model_keys(rewritten, model_keys)

        config_job_path = f"videos/{stem}_config.json"
        write_json_atomic(job_dir / config_job_path, rewritten)

        job_videos.append(
            JobVideo(
                job_path=job_path,
                origin_path=origin_path,
                size_bytes=size_bytes,
                config_job_path=config_job_path,
                config_provenance=planned.config_provenance,
                pushed_siblings=[config_job_path],
                signature=signature,
                shared=shared_entry,
                redirected_outputs=redirected,
            )
        )

    # --- videos.txt: keystone first ---
    keystone_video = job_videos[0]
    lines = [keystone_video.job_path] + [
        v.job_path for v in job_videos[1:] if v.job_path != keystone_video.job_path
    ]
    (job_dir / "videos.txt").write_text("\n".join(lines) + ("\n" if lines else ""))

    # --- requirements ---
    conda_envs: list[str] = []
    for planned in planned_videos:
        cfg = planned.config
        if (
            is_pose_inference_enabled(cfg)
            and str(cfg.get("pose_model_type", "")).strip().lower() == "sleap"
        ):
            env = _sleap_conda_env(cfg)
            if env not in conda_envs:
                conda_envs.append(env)

    import hydra_suite

    requirements = {
        "min_hydra_suite_version": getattr(hydra_suite, "__version__", "1.0.0"),
        "conda_envs": conda_envs,
        "runtime_tier": track_args.get("runtime_tier", "gpu"),
    }

    # --- run.sh ---
    run_sh_path = job_dir / "run.sh"
    run_sh_path.write_text(render_run_sh())
    mode = run_sh_path.stat().st_mode
    run_sh_path.chmod(mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)

    # --- config snapshot ---
    config_snapshot = {
        "advanced_config": "config/advanced_config.json",
        "skeletons": sorted(set(skeleton_job_paths.values())),
    }

    job_id = old_manifest.job_id if old_manifest is not None else uuid.uuid4().hex
    pull_history = list(old_manifest.pull_history) if old_manifest is not None else []
    created_at = datetime.datetime.now(datetime.timezone.utc).strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )
    created_on = {
        "hostname": socket.gethostname(),
        "platform": platform.system(),
        "job_name": job_name,
    }

    manifest = JobManifest(
        job_id=job_id,
        created_at=created_at,
        created_on=created_on,
        keystone={"video": keystone_video.job_path},
        videos=job_videos,
        models=job_models,
        config_snapshot=config_snapshot,
        requirements=requirements,
        track_args=dict(track_args),
        pull_history=pull_history,
    )
    manifest.write(job_dir / "hydra_job.json")

    problems = verify_job(job_dir)
    if problems:
        raise TrackingJobError(
            "pack_job produced an invalid job:\n" + "\n".join(problems)
        )

    return manifest
