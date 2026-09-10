"""Packing a job: rewrites, videos, sidecars, runner, manifest."""

import json
import os
import stat
from pathlib import Path

import pytest

from hydra_suite.data.tracking_job.manifest import JobManifest, TrackingJobError
from hydra_suite.data.tracking_job.pack import PlannedVideo, pack_job
from hydra_suite.data.tracking_job.references import PlannedModel
from tests.helpers.tracking_job import _planned


def _pack(tmp_path, staging, planned=None, **kwargs):
    # Fix M1: shared_table used to be hard-coded AND forwarded via **kwargs,
    # so any caller passing shared_table= (several tests below do) raised
    # "TypeError: got multiple values for argument 'shared_table'". Pop the
    # default out of kwargs instead of hard-coding the keyword argument.
    kwargs.setdefault("shared_table", {})
    return pack_job(
        tmp_path / "job",
        [planned or _planned(staging)],
        registry_entries=[
            ("obb/x.pt", {"species": "ant", "source_path": "/host/a.pt"})
        ],
        advanced_config_path=str(staging["advanced"]),
        track_args={"video_list": "videos.txt"},
        **kwargs,
    )


def test_job_layout_is_created(tmp_path, staging):
    _pack(tmp_path, staging)
    job = tmp_path / "job"
    assert (job / "hydra_job.json").is_file()
    assert (job / "run.sh").is_file()
    assert (job / "videos.txt").is_file()
    assert (job / "models" / "obb" / "x.pt").is_file()
    assert (job / "models" / "model_registry.json").is_file()
    assert (job / "config" / "advanced_config.json").is_file()
    assert (job / "config" / "presets" / ".seeded").is_file()
    assert (job / "config" / "skeletons" / ".seeded").is_file()
    assert (job / "videos" / "colony_config.json").is_file()


def test_run_sh_is_executable(tmp_path, staging):
    _pack(tmp_path, staging)
    mode = (tmp_path / "job" / "run.sh").stat().st_mode
    assert mode & stat.S_IXUSR


# Fix Y1 (round-7): the round-6 `eval "$TRACKERKIT track --video-list
# videos.txt $(printf '%q ' "$@")"` form is broken with ZERO passthrough
# args -- `printf '%q ' "$@"` on an empty "$@" still emits one `''` token,
# and `eval` re-parses that as a real, bogus empty positional, so
# `parse_arguments` sees `track --video-list videos.txt ''` and raises
# ("use either explicit video paths or --video-list, not both") before a
# single frame runs. This must be a REAL executed test -- monkeypatching
# `subprocess.run` (as the `job run` unit tests do) never invokes the actual
# bash template, so it cannot see this bug class at all.
def test_run_sh_track_invocation_has_no_stray_empty_positional_with_zero_args(
    tmp_path, staging
):
    import subprocess
    import sys

    _pack(tmp_path, staging)
    job = tmp_path / "job"

    # An argv-dumping stub standing in for `trackerkit`: writes its argv
    # (one JSON list per invocation) to argv_dump.jsonl and always exits 0,
    # so run.sh's own preflight/track/`_record-run` calls all "succeed"
    # without needing a real engine.
    stub = tmp_path / "trackerkit_stub.py"
    stub.write_text(
        "import json, sys\n"
        "with open(sys.argv[0] + '.dump', 'a') as f:\n"
        "    f.write(json.dumps(sys.argv[1:]) + '\\n')\n"
        "sys.exit(0)\n"
    )
    dump = Path(str(stub) + ".dump")

    env = dict(os.environ)
    env["HYDRA_JOB_TRACKERKIT"] = f"{sys.executable} {stub}"
    env["HYDRA_JOB_SKIP_PREFLIGHT"] = "1"  # isolate the `track` + `_record-run` calls

    result = subprocess.run(
        ["bash", str(job / "run.sh")], cwd=job, env=env, capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr

    lines = dump.read_text().strip().splitlines()
    assert len(lines) == 2, lines  # track, then job _record-run
    track_argv = json.loads(lines[0])
    assert track_argv == ["track", "--video-list", "videos.txt"], track_argv
    assert "" not in track_argv, "run.sh must never pass an empty positional to track"


def test_video_is_symlinked_by_default(tmp_path, staging):
    _pack(tmp_path, staging)
    link = tmp_path / "job" / "videos" / "colony.mp4"
    assert link.is_symlink()
    assert os.path.realpath(link) == os.path.realpath(staging["video"])


def test_copy_videos_makes_a_real_file(tmp_path, staging):
    _pack(tmp_path, staging, copy_videos=True)
    real = tmp_path / "job" / "videos" / "colony.mp4"
    assert real.is_file() and not real.is_symlink()


def test_sidecar_has_no_absolute_paths(tmp_path, staging):
    _pack(tmp_path, staging)
    sidecar = json.loads(
        (tmp_path / "job" / "videos" / "colony_config.json").read_text()
    )
    assert sidecar["file_path"] == "videos/colony.mp4"
    assert sidecar["csv_path"] == "videos/colony_tracking.csv"
    assert sidecar["video_output_path"] == "videos/colony_tracking.mp4"
    assert sidecar["pose_skeleton_file"] == "config/skeletons/ant.json"
    assert sidecar["yolo_obb_direct_model_path"] == "obb/x.pt"


def test_skeleton_is_snapshotted(tmp_path, staging):
    _pack(tmp_path, staging)
    assert (
        tmp_path / "job" / "config" / "skeletons" / "ant.json"
    ).read_text() == '{"nodes": []}'


def test_redirected_output_is_recorded(tmp_path, staging, tmp_path_factory):
    elsewhere = tmp_path / "renders" / "custom.mp4"
    planned = _planned(staging, config={"video_output_path": str(elsewhere)})
    manifest = _pack(tmp_path, staging, planned=planned)
    sidecar = json.loads(
        (tmp_path / "job" / "videos" / "colony_config.json").read_text()
    )
    assert sidecar["video_output_path"] == "videos/colony_custom.mp4"
    assert manifest.videos[0].redirected_outputs == {
        "videos/colony_custom.mp4": str(elsewhere)
    }


def test_redirected_output_with_the_default_basename_in_a_different_directory_is_still_recorded(
    tmp_path, staging
):
    """Fix W10: /renders/colony_tracking.mp4 has the DEFAULT basename
    (colony_tracking.mp4 -- what _default_output_paths would compute for
    colony.mp4), so a basename-only redirect check wrongly treats it as
    already-in-the-default-location and drops it with NO
    redirected_outputs entry -- pull would then place it beside the video
    instead of restoring it to /renders. The fix compares the full
    RESOLVED path against _default_output_paths(video_source_path), which
    correctly sees this as redirected because the DIRECTORY differs."""
    elsewhere = tmp_path / "renders" / "colony_tracking.mp4"
    planned = _planned(staging, config={"video_output_path": str(elsewhere)})
    manifest = _pack(tmp_path, staging, planned=planned)
    sidecar = json.loads(
        (tmp_path / "job" / "videos" / "colony_config.json").read_text()
    )
    assert sidecar["video_output_path"] == "videos/colony_colony_tracking.mp4"
    assert manifest.videos[0].redirected_outputs == {
        "videos/colony_colony_tracking.mp4": str(elsewhere)
    }


def test_videos_txt_is_job_relative_and_keystone_first(tmp_path, staging):
    _pack(tmp_path, staging)
    lines = (tmp_path / "job" / "videos.txt").read_text().splitlines()
    assert lines == ["videos/colony.mp4"]


def test_pushed_siblings_lists_the_sidecar(tmp_path, staging):
    manifest = _pack(tmp_path, staging)
    assert manifest.videos[0].pushed_siblings == ["videos/colony_config.json"]


def test_origin_path_is_recorded_for_pull(tmp_path, staging):
    manifest = _pack(tmp_path, staging)
    assert manifest.videos[0].origin_path == str(staging["video"])


def test_registry_subset_is_written_with_null_source(tmp_path, staging):
    _pack(tmp_path, staging)
    payload = json.loads(
        (tmp_path / "job" / "models" / "model_registry.json").read_text()
    )
    assert set(payload["entries"]) == {"obb/x.pt"}
    assert payload["entries"]["obb/x.pt"]["source_path"] is None


def test_basename_collision_across_directories_is_rejected(tmp_path, staging):
    other_dir = tmp_path / "other"
    other_dir.mkdir()
    twin = other_dir / "colony.mp4"
    twin.write_bytes(b"\x00" * 512)
    second = _planned(staging)
    second = PlannedVideo(
        video_path=str(twin),
        config={"file_path": str(twin)},
        config_provenance="keystone-baseline",
        planned_models=[],
        skeleton_path="",
    )
    with pytest.raises(TrackingJobError) as excinfo:
        pack_job(
            tmp_path / "job",
            [_planned(staging), second],
            registry_entries=[],
            advanced_config_path=str(staging["advanced"]),
            track_args={},
            shared_table={},
        )
    message = str(excinfo.value)
    assert str(staging["video"]) in message and str(twin) in message


def test_shared_video_is_referenced_not_copied(tmp_path, staging):
    table = {"labnas": str(staging["video"].parent)}
    manifest = _pack(tmp_path, staging, shared_table=table)
    entry = manifest.videos[0]
    assert entry.shared == {"alias": "labnas", "relpath": "colony.mp4"}
    assert entry.signature
    assert not (tmp_path / "job" / "videos" / "colony.mp4").exists()


def test_shared_only_rejects_an_off_share_video(tmp_path, staging):
    with pytest.raises(TrackingJobError):
        _pack(tmp_path, staging, shared_table={}, shared_mode="shared-only")


def test_no_shared_disables_matching(tmp_path, staging):
    table = {"labnas": str(staging["video"].parent)}
    manifest = _pack(tmp_path, staging, shared_table=table, shared_mode="no-shared")
    assert manifest.videos[0].shared is None
    assert (tmp_path / "job" / "videos" / "colony.mp4").is_symlink()


def test_pack_self_verifies_and_manifest_reads_back(tmp_path, staging):
    _pack(tmp_path, staging)
    manifest = JobManifest.read(tmp_path / "job" / "hydra_job.json")
    assert manifest.job_version == 1
    assert manifest.keystone["video"] == "videos/colony.mp4"


def test_pack_dedupes_a_model_shared_by_two_videos(tmp_path, staging):
    videos_dir = staging["video"].parent
    second_video = videos_dir / "second.mp4"
    second_video.write_bytes(b"\x00" * 1024)
    first = _planned(staging)
    second = _planned(
        staging,
        video_path=str(second_video),
        config={"file_path": str(second_video)},
    )
    manifest = pack_job(
        tmp_path / "job",
        [first, second],
        registry_entries=[
            ("obb/x.pt", {"species": "ant", "source_path": "/host/a.pt"})
        ],
        advanced_config_path=str(staging["advanced"]),
        track_args={},
        shared_table={},
    )
    matching = [m for m in manifest.models if m.key == "obb/x.pt"]
    assert len(matching) == 1


def test_pack_into_a_nonempty_job_dir_without_force_refuses(tmp_path, staging):
    _pack(tmp_path, staging)
    with pytest.raises(TrackingJobError):
        _pack(tmp_path, staging)


def test_pack_with_force_clears_stale_artifacts(tmp_path, staging):
    videos_dir = staging["video"].parent
    second_video = videos_dir / "second.mp4"
    second_video.write_bytes(b"\x00" * 1024)
    first = _planned(staging)
    second = _planned(
        staging,
        video_path=str(second_video),
        config={"file_path": str(second_video)},
    )
    pack_job(
        tmp_path / "job",
        [first, second],
        registry_entries=[
            ("obb/x.pt", {"species": "ant", "source_path": "/host/a.pt"})
        ],
        advanced_config_path=str(staging["advanced"]),
        track_args={},
        shared_table={},
    )
    from hydra_suite.data.tracking_job.verify import verify_job

    pack_job(
        tmp_path / "job",
        [first],
        registry_entries=[
            ("obb/x.pt", {"species": "ant", "source_path": "/host/a.pt"})
        ],
        advanced_config_path=str(staging["advanced"]),
        track_args={},
        shared_table={},
        force=True,
    )
    assert not (tmp_path / "job" / "videos" / "second.mp4").exists()
    assert not (tmp_path / "job" / "videos" / "second_config.json").exists()
    assert verify_job(tmp_path / "job") == []


def test_pack_with_force_preserves_pulled_outputs(tmp_path, staging):
    _pack(tmp_path, staging)
    job = tmp_path / "job"
    (job / "videos" / "colony_tracking_final.csv").write_text("a,b\n1,2\n")
    cache_dir = job / "videos" / ".inference_cache_colony"
    cache_dir.mkdir()
    (cache_dir / "detection.npz").write_bytes(b"\x00")
    with pytest.raises(TrackingJobError) as excinfo:
        _pack(tmp_path, staging, force=True)
    message = str(excinfo.value)
    assert "colony_tracking_final.csv" in message
    assert "detection.npz" in message
    assert (job / "videos" / "colony_tracking_final.csv").is_file()
    assert (cache_dir / "detection.npz").is_file()


def test_pack_with_force_discard_outputs_removes_them(tmp_path, staging):
    _pack(tmp_path, staging)
    job = tmp_path / "job"
    (job / "videos" / "colony_tracking_final.csv").write_text("a,b\n1,2\n")
    cache_dir = job / "videos" / ".inference_cache_colony"
    cache_dir.mkdir()
    (cache_dir / "detection.npz").write_bytes(b"\x00")
    _pack(tmp_path, staging, force=True, force_discard_outputs=True)
    assert not (job / "videos" / "colony_tracking_final.csv").exists()
    assert not cache_dir.exists()


def test_pack_with_force_carries_pull_history_forward(tmp_path, staging):
    _pack(tmp_path, staging)
    job = tmp_path / "job"
    manifest = JobManifest.read(job / "hydra_job.json")
    entry = {"pulled_at": "2026-01-01T00:00:00Z", "note": "hand-written"}
    updated = JobManifest(
        job_version=manifest.job_version,
        job_id=manifest.job_id,
        created_at=manifest.created_at,
        created_on=manifest.created_on,
        keystone=manifest.keystone,
        videos=manifest.videos,
        models=manifest.models,
        config_snapshot=manifest.config_snapshot,
        requirements=manifest.requirements,
        track_args=manifest.track_args,
        pull_history=[entry],
    )
    updated.write(job / "hydra_job.json")
    _pack(tmp_path, staging, force=True)
    new_manifest = JobManifest.read(job / "hydra_job.json")
    assert entry in new_manifest.pull_history
    assert new_manifest.job_id == manifest.job_id


def test_pack_with_force_tolerates_a_shared_video_with_no_local_file(tmp_path, staging):
    table = {"labnas": str(staging["video"].parent)}
    _pack(tmp_path, staging, shared_table=table)
    from hydra_suite.data.tracking_job.verify import verify_job

    _pack(tmp_path, staging, shared_table=table, force=True)
    assert verify_job(tmp_path / "job") == []


def test_skeleton_path_config_mismatch_raises(tmp_path, staging):
    planned = PlannedVideo(
        video_path=str(staging["video"]),
        config={
            "file_path": str(staging["video"]),
            "pose_skeleton_file": "/host/some/skeleton.json",
        },
        config_provenance="own-sidecar",
        planned_models=[],
        skeleton_path="",
    )
    with pytest.raises(TrackingJobError) as excinfo:
        pack_job(
            tmp_path / "job",
            [planned],
            registry_entries=[],
            advanced_config_path=str(staging["advanced"]),
            track_args={},
            shared_table={},
        )
    message = str(excinfo.value)
    assert "colony.mp4" in message


def test_pack_pose_enabled_no_skeleton_raises(tmp_path, staging):
    planned = _planned(
        staging,
        config={
            "detection_method": "yolo_obb",
            "enable_pose_extractor": True,
            "pose_model_dir": "pose/SLEAP/run",
            "pose_model_type": "sleap",
            "pose_skeleton_file": "",
        },
    )
    with pytest.raises(TrackingJobError) as excinfo:
        pack_job(
            tmp_path / "job",
            [planned],
            registry_entries=[],
            advanced_config_path=str(staging["advanced"]),
            track_args={},
            shared_table={},
        )
    assert "colony.mp4" in str(excinfo.value)
    assert not (tmp_path / "job" / "hydra_job.json").exists()


def test_pack_blanks_color_tag_model_path_even_when_out_of_root(tmp_path, staging):
    from hydra_suite.data.tracking_job.verify import verify_job

    planned = _planned(
        staging,
        config={"color_tag_model_path": "/some/out/of/root/color.pt"},
    )
    manifest = _pack(tmp_path, staging, planned=planned)
    sidecar = json.loads(
        (tmp_path / "job" / "videos" / "colony_config.json").read_text()
    )
    assert sidecar["color_tag_model_path"] == ""
    assert verify_job(tmp_path / "job") == []
    assert manifest is not None


def test_pack_blanks_an_inactive_role_absolute_path(
    tmp_path, staging, tmp_path_factory
):
    from hydra_suite.data.tracking_job.verify import verify_job

    stale = tmp_path / "stale_sequential.pt"
    stale.write_bytes(b"s")
    planned = _planned(
        staging,
        config={
            "yolo_obb_mode": "direct",
            "yolo_crop_obb_model_path": str(stale),
        },
    )
    manifest = _pack(tmp_path, staging, planned=planned)
    sidecar = json.loads(
        (tmp_path / "job" / "videos" / "colony_config.json").read_text()
    )
    assert sidecar["yolo_crop_obb_model_path"] == ""
    assert verify_job(tmp_path / "job") == []
    assert manifest is not None


def test_pack_blanks_an_inactive_pose_backend_path(tmp_path, staging):
    from hydra_suite.data.tracking_job.verify import verify_job

    stale = tmp_path / "stale_sleap"
    stale.mkdir()
    (stale / "best.ckpt").write_bytes(b"c")
    planned = _planned(
        staging,
        config={
            "pose_model_type": "yolo",
            "pose_sleap_model_dir": str(stale),
        },
    )
    manifest = _pack(tmp_path, staging, planned=planned)
    sidecar = json.loads(
        (tmp_path / "job" / "videos" / "colony_config.json").read_text()
    )
    assert sidecar["pose_sleap_model_dir"] == ""
    assert verify_job(tmp_path / "job") == []
    assert manifest is not None


def test_pack_blanks_yolo_model_path_alias_under_bgsub(tmp_path, staging):
    from hydra_suite.data.tracking_job.verify import verify_job

    stale = tmp_path / "stale_yolo.pt"
    stale.write_bytes(b"s")
    planned = _planned(
        staging,
        config={
            "detection_method": "bgsub",
            "yolo_obb_direct_model_path": "",
            "yolo_model_path": str(stale),
        },
        planned_models=[],
    )
    manifest = _pack(tmp_path, staging, planned=planned)
    sidecar = json.loads(
        (tmp_path / "job" / "videos" / "colony_config.json").read_text()
    )
    assert sidecar["yolo_model_path"] == ""
    assert verify_job(tmp_path / "job") == []
    assert manifest is not None


def test_pack_rewrites_cnn_classifiers_model_path(tmp_path, staging, monkeypatch):
    from hydra_suite.data.tracking_job.verify import verify_job

    head = staging["models"] / "classification" / "identity" / "head_a.pth"
    head.parent.mkdir(parents=True)
    head.write_bytes(b"h")
    # `_normalize_model_path` -> `resolve_model_path` resolves a
    # models-root-relative config value against HYDRA_MODELS_DIR; pin it to
    # this fixture's own models root so the lookup is deterministic here
    # regardless of what's configured on the machine running the test.
    monkeypatch.setenv("HYDRA_MODELS_DIR", str(staging["models"]))
    planned = _planned(
        staging,
        config={
            "cnn_classifiers": [
                {"model_path": str(head), "species": "ant"},
            ]
        },
        planned_models=[
            PlannedModel(
                role="YOLO_OBB_DIRECT_MODEL_PATH",
                source_path=str(staging["models"] / "obb" / "x.pt"),
                kind="file",
                key="obb/x.pt",
            ),
            PlannedModel(
                role="CNN_CLASSIFIERS",
                source_path=str(head),
                kind="file",
                key="classification/identity/head_a.pth",
            ),
        ],
        cnn_model_keys={
            str(head.expanduser().resolve()): "classification/identity/head_a.pth"
        },
    )
    manifest = _pack(tmp_path, staging, planned=planned)
    sidecar = json.loads(
        (tmp_path / "job" / "videos" / "colony_config.json").read_text()
    )
    assert (
        sidecar["cnn_classifiers"][0]["model_path"]
        == "classification/identity/head_a.pth"
    )
    assert (
        sidecar["cnn_classifiers"][0]["species"] == "ant"
    )  # non-path keys survive untouched
    assert verify_job(tmp_path / "job") == []
    assert manifest is not None


def test_conda_envs_defaults_to_sleap_when_key_is_absent(tmp_path, staging):
    planned = _planned(
        staging,
        config={
            "detection_method": "yolo_obb",
            "enable_pose_extractor": True,
            "pose_model_dir": "pose/SLEAP/run",
            "pose_model_type": "sleap",
            "pose_skeleton_file": str(staging["skeleton"]),
            # pose_sleap_env genuinely ABSENT -- not set to "", the shape a
            # real staging config that never touched the SLEAP-env combo box
            # takes.
        },
        planned_models=[
            PlannedModel(
                role="YOLO_OBB_DIRECT_MODEL_PATH",
                source_path=str(staging["models"] / "obb" / "x.pt"),
                kind="file",
                key="obb/x.pt",
            ),
            PlannedModel(
                role="POSE_MODEL_DIR",
                source_path=str(staging["models"] / "obb"),  # any real directory
                kind="directory",
                key="pose/SLEAP/run",
            ),
        ],
    )
    manifest = _pack(tmp_path, staging, planned=planned)
    assert manifest.requirements["conda_envs"] == ["sleap"]


def test_conda_envs_defaults_to_sleap_for_the_placeholder_value(tmp_path, staging):
    planned = _planned(
        staging,
        config={
            "detection_method": "yolo_obb",
            "enable_pose_extractor": True,
            "pose_model_dir": "pose/SLEAP/run",
            "pose_model_type": "sleap",
            "pose_skeleton_file": str(staging["skeleton"]),
            "pose_sleap_env": "no sleap envs found",
        },
        planned_models=[
            PlannedModel(
                role="YOLO_OBB_DIRECT_MODEL_PATH",
                source_path=str(staging["models"] / "obb" / "x.pt"),
                kind="file",
                key="obb/x.pt",
            ),
            PlannedModel(
                role="POSE_MODEL_DIR",
                source_path=str(staging["models"] / "obb"),
                kind="directory",
                key="pose/SLEAP/run",
            ),
        ],
    )
    manifest = _pack(tmp_path, staging, planned=planned)
    assert manifest.requirements["conda_envs"] == ["sleap"]


def test_pack_job_synthesizes_advanced_config_when_the_host_has_none(tmp_path, staging):
    planned = _planned(staging)
    manifest = pack_job(
        tmp_path / "job",
        [planned],
        registry_entries=[
            ("obb/x.pt", {"species": "ant", "source_path": "/host/a.pt"})
        ],
        advanced_config_path=str(tmp_path / "does_not_exist.json"),
        advanced_config_fallback={"adv": "fallback"},
        track_args={},
        shared_table={},
    )
    payload = json.loads(
        (tmp_path / "job" / "config" / "advanced_config.json").read_text()
    )
    assert payload == {"adv": "fallback"}
    assert manifest is not None


def test_pack_py_module_has_no_trackerkit_import():
    import pathlib

    from tests.test_tracking_job_layering import _imported_names

    pack_py = (
        pathlib.Path(__file__).resolve().parents[1]
        / "src"
        / "hydra_suite"
        / "data"
        / "tracking_job"
        / "pack.py"
    )
    for name in _imported_names(pack_py):
        assert "trackerkit" not in name


def test_active_roles_derivation_never_imports_trackerkit():
    import pathlib

    from tests.test_tracking_job_layering import _imported_names

    pack_py = (
        pathlib.Path(__file__).resolve().parents[1]
        / "src"
        / "hydra_suite"
        / "data"
        / "tracking_job"
        / "pack.py"
    )
    for name in _imported_names(pack_py):
        assert "engine_params" not in name
        assert "iter_model_references" not in name


def test_pack_job_accepts_a_registry_entries_generator(tmp_path, staging):
    entries = (
        e for e in [("obb/x.pt", {"species": "ant", "source_path": "/host/a.pt"})]
    )
    manifest = pack_job(
        tmp_path / "job",
        [_planned(staging)],
        registry_entries=entries,
        advanced_config_path=str(staging["advanced"]),
        track_args={"video_list": "videos.txt"},
        shared_table={},
    )
    payload = json.loads(
        (tmp_path / "job" / "models" / "model_registry.json").read_text()
    )
    assert "obb/x.pt" in payload["entries"]
    matching = [m for m in manifest.models if m.key == "obb/x.pt"]
    assert matching and matching[0].registry_entry_present is True


def test_run_sh_self_preflight_forwards_shared_root_overrides():
    """A hand-run ./run.sh must be able to resolve a NON-persisted alias.

    Regression: the self-preflight was flag-less, so `./run.sh` on a box whose
    shared-root table lacks the alias aborted with "unknown shared-root alias
    ...; known aliases: (none configured)" even when `trackerkit job run
    --shared-root ...` had just succeeded against the same box. The spec calls
    run.sh "the executable contract" that anyone can ssh in and run by hand, so
    the two paths must agree. Verified end to end on firebrat.
    """
    from hydra_suite.data.tracking_job.runner import render_run_sh

    script = render_run_sh()
    assert "HYDRA_JOB_PREFLIGHT_ARGS" in script
    # The overrides must reach the self-preflight, not just be read into a var.
    assert 'eval "PF_ARGS=(${HYDRA_JOB_PREFLIGHT_ARGS:-})"' in script
    assert 'job preflight . ${PF_ARGS[@]+"${PF_ARGS[@]}"}' in script
    # ...and it must stay safe under `set -u` when unset (the ${x[@]+...} guard).
    assert "${PF_ARGS[@]+" in script


def test_job_run_exports_the_overrides_for_a_later_hand_run():
    """`job run --shared-root ...` leaves a run.sh a human can re-run."""
    from hydra_suite.trackerkit.job_cli import _preflight_flag_string

    flags = _preflight_flag_string({"labnas": "/mnt/lab"}, False)
    assert "--shared-root labnas=/mnt/lab" in flags
    assert "--allow-tier-fallback" not in flags

    flags = _preflight_flag_string({}, True)
    assert flags.strip() == "--allow-tier-fallback"

    # A path with spaces must survive shell word-splitting on the far side.
    flags = _preflight_flag_string({"lab": "/mnt/my lab"}, False)
    assert "'/mnt/my lab'" in flags or '"/mnt/my lab"' in flags
