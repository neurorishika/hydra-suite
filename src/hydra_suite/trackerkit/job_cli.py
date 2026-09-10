"""App-layer bridge for ``trackerkit job ...``.

``data/tracking_job`` must never import ``trackerkit`` or ``classkit`` (Core/
Data never import an app layer), so THIS module does the planning, the
engine-parameter resolution and the ClassKit multi-head bundle discovery,
then hands the results down to ``pack_job``/``push_job``/``pull_job``/
``preflight_job``/``verify_job`` as plain data.
"""

from __future__ import annotations

import json
import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any

from hydra_suite.core.inference.model_paths import (
    make_model_path_relative,
    make_pose_model_path_relative,
    resolve_model_path,
)
from hydra_suite.data.tracking_job.manifest import JobManifest, TrackingJobError
from hydra_suite.data.tracking_job.pack import (
    PlannedVideo,
    _normalize_model_path,
    pack_job,
)
from hydra_suite.data.tracking_job.preflight import preflight_job
from hydra_suite.data.tracking_job.references import PlannedModel, external_key_for
from hydra_suite.data.tracking_job.shared_roots import load_shared_roots, save_alias
from hydra_suite.data.tracking_job.transport import parse_remote, pull_job, push_job
from hydra_suite.data.tracking_job.verify import verify_job

_RECORD_RUN_FIELDS = (
    "started_at",
    "finished_at",
    "hostname",
    "exit_code",
    "hydra_suite_version",
    "git_sha",
    "argv",
    "host_advanced_config_used",
)


def _multihead_bundle_artifacts(manifest_path: str) -> list[str]:
    """Every factor-model head a ``.multihead.json`` manifest references,
    resolved the same way core resolves them at load time
    (``core/individual/classification/backend.py:494-497``: ``base =
    manifest_path.parent``; ``path = (base / entry["path"]).resolve()``).

    Fix A1: ``discover_multihead_model_bundle`` (``classkit/model_bundle.py``)
    is ``.bundle.json``-only and returns ``None`` for a bare ``.multihead.json``
    primary path -- production identity classifiers use this format, so
    packing them without this helper ships the manifest alone and fails at
    remote load time with no pack-time error.
    """
    manifest = Path(manifest_path)
    try:
        data = json.loads(manifest.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise TrackingJobError(
            f"cannot read multihead manifest {manifest_path!r}: {exc}",
            code=2,
        ) from exc
    base = manifest.parent
    heads: list[str] = []
    for entry in data.get("factor_models", []):
        try:
            entry_path = entry["path"]
        except (KeyError, TypeError) as exc:
            raise TrackingJobError(
                f"multihead manifest {manifest_path!r} has a factor_models "
                f"entry with no 'path': {entry!r}",
                code=2,
            ) from exc
        if Path(entry_path).name != entry_path or Path(entry_path).is_absolute():
            raise TrackingJobError(
                f"multihead manifest {manifest_path!r} has a non-relative "
                f"factor_models path (expected a bare basename): {entry_path!r}",
                code=2,
            )
        head_path = (base / entry_path).resolve()
        if not head_path.exists():
            raise TrackingJobError(
                f"multihead manifest {manifest_path!r} references a missing "
                f"head model: {head_path}",
                code=2,
            )
        heads.append(str(head_path))
    return heads


def _plan_video_models(session_config: dict[str, Any], params: dict[str, Any]):
    """Build ``(planned_models, cnn_model_keys)`` for one video's resolved
    engine params. Bundle discovery is restricted to ``CNN_CLASSIFIERS`` refs
    only -- see fix (minor) in Task 11 Step 4 item 5."""
    from hydra_suite.classkit.model_bundle import discover_multihead_model_bundle
    from hydra_suite.trackerkit.engine_params import iter_model_references

    planned_models: list[PlannedModel] = []
    cnn_model_keys: dict[str, str] = {}

    for ref in iter_model_references(params):
        if ref.kind == "directory":
            key = make_pose_model_path_relative(ref.path)
        else:
            key = make_model_path_relative(ref.path)
        key = str(key)
        if os.path.isabs(key):
            key = external_key_for(ref.path)

        bundle_artifacts: list[str] = []
        if ref.role == "CNN_CLASSIFIERS":
            if str(ref.path).lower().endswith(".multihead.json"):
                bundle_artifacts = _multihead_bundle_artifacts(ref.path)
            else:
                bundle = discover_multihead_model_bundle(ref.path)
                if bundle:
                    selected = str(Path(ref.path).expanduser().resolve())
                    bundle_artifacts = [
                        artifact
                        for artifact in bundle.get("artifact_paths", [])
                        if str(Path(artifact).resolve()) != selected
                    ]
            cnn_model_keys[_normalize_model_path(ref.path)] = key

        planned_models.append(
            PlannedModel(
                role=ref.role,
                source_path=ref.path,
                kind=ref.kind,
                key=key,
                bundle_artifacts=bundle_artifacts,
            )
        )

    return planned_models, cnn_model_keys


def _export_warnings(config: dict[str, Any]) -> list[str]:
    """Spec §6.6: warn when an export stage is enabled -- the CLI leaves the
    output-dir keys at ``None`` (``cli_config.py:293-295``), so those exports
    produce nothing on the remote."""
    warnings: list[str] = []
    if config.get("enable_dataset_generation"):
        warnings.append(
            "enable_dataset_generation is set, but DATASET_OUTPUT_DIR is never "
            "populated by the CLI -- dataset export will produce nothing on "
            "the remote (see docs follow-up §17.1)"
        )
    if config.get("final_media_export_videos_enabled"):
        warnings.append(
            "final_media_export_videos_enabled is set, but "
            "FINAL_MEDIA_EXPORT_VIDEO_OUTPUT_DIR is never populated by the CLI "
            "-- final media export will produce nothing on the remote (see "
            "docs follow-up §17.1)"
        )
    if config.get("enable_individual_dataset"):
        warnings.append(
            "enable_individual_dataset is set, but INDIVIDUAL_DATASET_OUTPUT_DIR "
            "is never populated by the CLI -- individual dataset export will "
            "produce nothing on the remote (see docs follow-up §17.1)"
        )
    return warnings


def _cmd_pack(args) -> int:
    from hydra_suite.paths import get_advanced_config_path
    from hydra_suite.trackerkit.app import resolve_track_video_inputs
    from hydra_suite.trackerkit.batch_plan import plan_batch_jobs
    from hydra_suite.trackerkit.cli_config import (
        load_advanced_tracker_config,
        load_tracker_cli_session,
    )
    from hydra_suite.training.model_publish import iter_registry_entries

    try:
        videos = [
            os.path.abspath(v)
            for v in resolve_track_video_inputs(args.videos, args.video_list)
        ]
    except ValueError as exc:
        raise TrackingJobError(str(exc), code=2) from exc

    specs = plan_batch_jobs(
        videos,
        explicit_config_path=args.config,
        keystone_override=bool(args.keystone_override),
        sahi_profile=args.sahi_profile,
        apply_tuned_inference=args.apply_tuned_inference,
        inference_autotune_manual=list(args.inference_autotune_manual or []),
    )

    planned_videos: list[PlannedVideo] = []
    export_warnings: list[str] = []
    keystone_runtime_tier = "gpu"
    for index, spec in enumerate(specs):
        session = load_tracker_cli_session(spec.video_path, config_data=spec.config)
        params = session.params
        planned_models, cnn_model_keys = _plan_video_models(session.config, params)

        raw_skeleton = str(session.config.get("pose_skeleton_file", "") or "").strip()
        skeleton_path = str(resolve_model_path(raw_skeleton)) if raw_skeleton else ""

        planned_videos.append(
            PlannedVideo(
                video_path=spec.video_path,
                config=session.config,
                config_provenance=spec.provenance,
                planned_models=planned_models,
                skeleton_path=skeleton_path,
                cnn_model_keys=cnn_model_keys,
            )
        )
        export_warnings.extend(_export_warnings(session.config))
        if index == 0:
            keystone_runtime_tier = str(
                session.config.get("runtime_tier", "gpu") or "gpu"
            )

    shared_mode = "auto"
    if getattr(args, "no_shared", False):
        shared_mode = "no-shared"
    elif getattr(args, "shared_only", False):
        shared_mode = "shared-only"

    track_args = {
        "keystone_override": bool(args.keystone_override),
        "sahi_profile": args.sahi_profile,
        "apply_tuned_inference": args.apply_tuned_inference,
        "inference_autotune_manual": list(args.inference_autotune_manual or []),
        "runtime_tier": keystone_runtime_tier,
    }

    manifest = pack_job(
        args.job_dir,
        planned_videos,
        registry_entries=list(iter_registry_entries()),
        advanced_config_path=str(get_advanced_config_path()),
        advanced_config_fallback=load_advanced_tracker_config(),
        track_args=track_args,
        shared_table=load_shared_roots(),
        copy_videos=bool(getattr(args, "copy_videos", False)),
        shared_mode=shared_mode,
        job_name=getattr(args, "job_name", None),
        force=bool(args.force),
        force_discard_outputs=bool(args.discard_outputs),
    )

    for message in export_warnings:
        print(f"warning: {message}", file=sys.stderr)
    print(f"packed job {manifest.job_id} -> {args.job_dir}")
    return 0


def _cmd_verify(args) -> int:
    problems = verify_job(args.job_dir, fast=bool(getattr(args, "fast", False)))
    if problems:
        for problem in problems:
            print(problem)
        return 1
    print("ok")
    return 0


def _cmd_shared_root(args) -> int:
    sub = getattr(args, "shared_root_command", None)
    if sub == "add":
        save_alias(args.alias, args.path)
        print(f"added shared-root alias {args.alias!r} -> {args.path}")
        return 0
    if sub == "list":
        table = load_shared_roots()
        if not table:
            print("(no shared roots configured)")
            return 0
        for alias, path in sorted(table.items()):
            print(f"{alias}\t{path}")
        return 0
    raise TrackingJobError("job shared-root requires a sub-command: add|list", code=2)


def _parse_shared_root_overrides(values: list[str] | None) -> dict[str, str]:
    overrides: dict[str, str] = {}
    for raw in values or []:
        alias, sep, path = str(raw).partition("=")
        if not sep or not alias or not path:
            raise TrackingJobError(
                f"malformed --shared-root {raw!r} (expected ALIAS=PATH)", code=2
            )
        overrides[alias] = path
    return overrides


def _cmd_preflight(args) -> int:
    overrides = _parse_shared_root_overrides(getattr(args, "shared_root", None))
    result = preflight_job(
        args.job_dir,
        shared_root_overrides=overrides,
        fast=bool(getattr(args, "fast", False)),
        allow_tier_fallback=bool(getattr(args, "allow_tier_fallback", False)),
    )
    for check in result.checks:
        status = "ok" if check["ok"] else "FAIL"
        detail = f" -- {check['detail']}" if check.get("detail") else ""
        print(f"[{status}] {check['name']}{detail}")
    return 0 if result.ok else 3


def _cmd_push(args) -> int:
    job_dir = Path(args.job_dir)
    target = parse_remote(args.target)
    push_job(job_dir, target, runner=subprocess.run)

    bootstrap = _bootstrap_prefix(getattr(args, "remote_bootstrap", "") or "")
    remote_cmd = (
        f"{bootstrap}cd {shlex.quote(target.path)} && "
        f"{_remote_trackerkit_invocation()} job verify ."
    )
    result = subprocess.run(
        ["ssh", target.host, remote_cmd], capture_output=True, text=True
    )
    if result.returncode != 0:
        raise TrackingJobError(
            "push succeeded but the remote job failed job verify:\n"
            f"{result.stdout}\n{result.stderr}",
            code=4,
        )
    print(f"pushed {job_dir} -> {args.target}, remote verify ok")
    return 0


def _cmd_pull(args) -> int:
    report = pull_job(
        args.target,
        Path(args.job_dir),
        include_caches=not bool(args.no_caches),
        overwrite=bool(args.overwrite),
        dry_run=bool(args.dry_run),
        force=bool(args.force),
        runner=subprocess.run,
    )
    for entry in report.pulled:
        print(f"pulled: {entry.job_relpath} -> {entry.destination}")
    for relpath, reason in report.skipped:
        print(f"skipped: {relpath} ({reason})")
    return 0 if report.ok else 1


def _cmd_status(args) -> int:
    if ":" in args.target and not Path(args.target).exists():
        target = parse_remote(args.target)
        bootstrap = _bootstrap_prefix(getattr(args, "remote_bootstrap", "") or "")
        remote_cmd = (
            f"{bootstrap}cd {shlex.quote(target.path)} && "
            "cat hydra_job.json && echo '---' && "
            "tail -n 5 logs/runs.jsonl 2>/dev/null || true"
        )
        result = subprocess.run(
            ["ssh", target.host, remote_cmd], capture_output=True, text=True
        )
        print(result.stdout)
        if result.returncode != 0:
            print(result.stderr, file=sys.stderr)
            return 4
        return 0

    job_dir = Path(args.target)
    manifest = JobManifest.read(job_dir / "hydra_job.json")
    print(f"job_id: {manifest.job_id}")
    print(f"created_at: {manifest.created_at}")
    print(f"videos: {len(manifest.videos)}")
    runs_jsonl = job_dir / "logs" / "runs.jsonl"
    if runs_jsonl.is_file():
        lines = [
            line
            for line in runs_jsonl.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        for line in lines[-5:]:
            print(line)
    return 0


def _local_trackerkit_invocation() -> str:
    """``HYDRA_JOB_TRACKERKIT``'s default for a LOCAL ``job run``: the
    interpreter running THIS process, invoking the module entry point --
    never a bare ``trackerkit`` resolved fresh from PATH, which in a dev
    worktree resolves the MAIN editable install instead of this checkout
    (fix V3)."""
    return f"{shlex.quote(sys.executable)} -m hydra_suite.trackerkit.app"


def _remote_trackerkit_invocation() -> str:
    """The invocation to use INSIDE an ssh command on the compute box.

    This must NOT be the local ``sys.executable``: that is an absolute path on
    THIS machine (e.g. a macOS conda prefix) which does not exist on the
    remote, so ssh fails with ``No such file or directory`` -- caught in
    practice when the post-push remote ``job verify`` tried to run the Mac's
    interpreter on firebrat. ``--remote-bootstrap`` is what puts the right
    environment on PATH remotely (verified: a bare ssh, and even ``bash -lc``,
    cannot find ``trackerkit`` without it), so a plain name is correct here and
    ``HYDRA_JOB_TRACKERKIT`` remains available as the per-host override.
    """
    return "trackerkit"


def _preflight_flag_string(
    shared_overrides: dict[str, str], allow_tier_fallback: bool
) -> str:
    """The `job preflight` flags implied by this invocation, as one string.

    Used two ways: appended to the ssh-chained remote `job preflight` call, and
    exported as ``HYDRA_JOB_PREFLIGHT_ARGS`` so ``run.sh``'s own self-preflight
    sees the same overrides on a later hand-run.
    """
    flags = "".join(
        f" --shared-root {shlex.quote(alias)}={shlex.quote(root)}"
        for alias, root in shared_overrides.items()
    )
    if allow_tier_fallback:
        flags += " --allow-tier-fallback"
    return flags


def _bootstrap_prefix(remote_bootstrap: str) -> str:
    """Fix A2b: a non-empty bootstrap gets a trailing ``; `` so it cannot
    short-circuit the following ``&&`` chain by being read as its LHS."""
    text = remote_bootstrap.rstrip()
    return f"{text}; " if text else ""


def _job_env_with_trackerkit() -> dict[str, str]:
    """Fix V3: the local ``job run``/``job calibrate`` branch must itself set
    ``HYDRA_JOB_TRACKERKIT``, or ``run.sh``'s PATH-independence is dead code
    on every dev machine (a bare ``trackerkit`` resolves MAIN's editable
    install, not this worktree)."""
    env = dict(os.environ)
    env.setdefault("HYDRA_JOB_TRACKERKIT", _local_trackerkit_invocation())
    return env


def _run_track_args(job_dir: Path) -> list[str]:
    manifest = JobManifest.read(job_dir / "hydra_job.json")
    track_args = manifest.track_args or {}
    passthrough: list[str] = []
    if track_args.get("keystone_override"):
        passthrough.append("--keystone-override")
    sahi_profile = track_args.get("sahi_profile")
    if sahi_profile:
        passthrough += ["--sahi-profile", str(sahi_profile)]
    apply_tuned = track_args.get("apply_tuned_inference")
    if apply_tuned is True:
        passthrough.append("--apply-tuned-inference")
    elif apply_tuned is False:
        passthrough.append("--no-apply-tuned-inference")
    for field in track_args.get("inference_autotune_manual", []) or []:
        passthrough += ["--inference-autotune-manual", str(field)]
    return passthrough


def _cmd_run(args) -> int:
    target = args.target
    passthrough = _run_track_args(Path(target)) if _is_local(target) else None
    shared_overrides = _parse_shared_root_overrides(getattr(args, "shared_root", None))
    allow_tier_fallback = bool(getattr(args, "allow_tier_fallback", False))

    if getattr(args, "calibrate", False):
        calibrate_exit = _run_calibrate(target, args)
        if calibrate_exit != 0:
            return calibrate_exit

    if _is_local(target):
        job_dir = Path(target)
        passthrough = _run_track_args(job_dir)
        result = preflight_job(
            job_dir,
            shared_root_overrides=shared_overrides,
            allow_tier_fallback=allow_tier_fallback,
        )
        if not result.ok:
            raise TrackingJobError(
                f"preflight failed for {job_dir}; see logs/preflight.json", code=3
            )
        env = _job_env_with_trackerkit()
        env["HYDRA_JOB_SKIP_PREFLIGHT"] = "1"
        # Also record the overrides so the run.sh left behind is re-runnable BY
        # HAND with the same result. Without this a later `./run.sh` aborts on
        # "unknown shared-root alias" for a non-persisted alias, even though
        # `job run` had just succeeded -- and the spec calls run.sh "the
        # executable contract".
        preflight_args = _preflight_flag_string(shared_overrides, allow_tier_fallback)
        if preflight_args:
            env["HYDRA_JOB_PREFLIGHT_ARGS"] = preflight_args.strip()
        run_argv = ["./run.sh", *passthrough]
        if getattr(args, "detach", False):
            with open(os.devnull, "wb") as devnull:
                subprocess.Popen(
                    run_argv,
                    cwd=str(job_dir),
                    env=env,
                    stdout=devnull,
                    stderr=devnull,
                    stdin=subprocess.DEVNULL,
                    start_new_session=True,
                )
            print(f"detached: {job_dir}")
            return 0
        result_proc = subprocess.run(run_argv, cwd=str(job_dir), check=False, env=env)
        return result_proc.returncode

    remote = parse_remote(target)
    bootstrap = _bootstrap_prefix(getattr(args, "remote_bootstrap", "") or "")
    preflight_flags = _preflight_flag_string(shared_overrides, allow_tier_fallback)
    passthrough = list(passthrough or [])
    tk = _remote_trackerkit_invocation()
    run_tail = f"./run.sh {shlex.join(passthrough)}".rstrip()
    if getattr(args, "detach", False):
        run_tail = f"nohup {run_tail} >/dev/null 2>&1 & disown"
    remote_cmd = (
        f"{bootstrap}cd {shlex.quote(remote.path)} && "
        f"{tk} job preflight .{preflight_flags} && "
        f"HYDRA_JOB_SKIP_PREFLIGHT=1 "
        f"HYDRA_JOB_PREFLIGHT_ARGS={shlex.quote(preflight_flags.strip())} "
        f"{run_tail}"
    )
    result = subprocess.run(
        ["ssh", remote.host, remote_cmd], check=False, stdin=subprocess.DEVNULL
    )
    return result.returncode


def _is_local(target: str) -> bool:
    return Path(target).exists() or (":" not in target)


def _run_calibrate(target: str, args) -> int:
    if _is_local(target):
        job_dir = Path(target)
        manifest = JobManifest.read(job_dir / "hydra_job.json")
        keystone_relpath = manifest.keystone.get("video", "")
        video = manifest.videos[0]
        for candidate in manifest.videos:
            if candidate.job_path == keystone_relpath:
                video = candidate
                break
        env = _job_env_with_trackerkit()
        env["HYDRA_MODELS_DIR"] = str(job_dir / "models")
        env["HYDRA_CONFIG_DIR"] = str(job_dir / "config")
        manual_fields = list(manifest.track_args.get("inference_autotune_manual", []))
        argv = [
            sys.executable,
            "-m",
            "hydra_suite.trackerkit.app",
            "calibrate",
            "--video",
            video.job_path,
            "--config",
            video.config_job_path,
            "--budget-seconds",
            str(args.budget_seconds),
        ]
        for field in manual_fields:
            argv += ["--inference-autotune-manual", str(field)]
        result = subprocess.run(argv, cwd=str(job_dir), env=env, check=False)
        return result.returncode

    remote = parse_remote(target)
    bootstrap = _bootstrap_prefix(getattr(args, "remote_bootstrap", "") or "")
    tk = _remote_trackerkit_invocation()
    remote_cmd = f"{bootstrap}cd {shlex.quote(remote.path)} && {tk} job calibrate ."
    result = subprocess.run(
        ["ssh", remote.host, remote_cmd], check=False, stdin=subprocess.DEVNULL
    )
    return result.returncode


def _cmd_calibrate(args) -> int:
    return _run_calibrate(args.target, args)


def _cmd_record_run(args) -> int:
    import datetime
    import platform
    import socket

    import hydra_suite
    from hydra_suite.data.tracking_job.manifest import _current_git_sha

    job_dir = Path.cwd()
    logs_dir = job_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    finished_at = datetime.datetime.now(datetime.timezone.utc).isoformat(
        timespec="microseconds"
    )
    entry = {
        "started_at": args.started,
        "finished_at": finished_at,
        "hostname": socket.gethostname(),
        "platform": platform.system(),
        "exit_code": int(args.exit_code),
        "hydra_suite_version": getattr(hydra_suite, "__version__", "1.0.0"),
        "git_sha": _current_git_sha(),
        "argv": list(getattr(args, "argv", []) or []),
        "host_advanced_config_used": bool(
            os.environ.get("HYDRA_JOB_HOST_ADVANCED_CONFIG_USED")
        ),
    }
    with open(logs_dir / "runs.jsonl", "a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry) + "\n")
    return 0


_DISPATCH = {
    "pack": _cmd_pack,
    "verify": _cmd_verify,
    "shared-root": _cmd_shared_root,
    "push": _cmd_push,
    "pull": _cmd_pull,
    "preflight": _cmd_preflight,
    "run": _cmd_run,
    "calibrate": _cmd_calibrate,
    "status": _cmd_status,
    "_record-run": _cmd_record_run,
}


def _dispatch(args) -> int:
    job_command = getattr(args, "job_command", None)
    handler = _DISPATCH.get(job_command)
    if handler is None:
        raise TrackingJobError(
            f"job requires a sub-command (one of: {', '.join(sorted(_DISPATCH))}); "
            f"got {job_command!r}",
            code=2,
        )
    return handler(args)


def run_job_cli(args) -> int:
    """Entry point invoked from ``trackerkit.app.main``'s ``job`` branch."""
    try:
        return _dispatch(args)
    except TrackingJobError as exc:
        print(f"error: {exc}")
        return exc.code
