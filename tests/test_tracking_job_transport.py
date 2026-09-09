"""Transport builds exact argv and never invents file lists."""

import shutil
from pathlib import Path

import pytest

from hydra_suite.data.tracking_job.manifest import (
    JobManifest,
    JobModel,
    JobVideo,
    TrackingJobError,
)
from hydra_suite.data.tracking_job.transport import (
    build_push_input_list,
    build_rsync_argv,
    parse_remote,
)


def _manifest(shared=None):
    return JobManifest(
        job_id="j",
        created_at="t",
        created_on={},
        keystone={"video": "videos/a.mp4", "config": "videos/a_config.json"},
        videos=[
            JobVideo(
                job_path="videos/a.mp4",
                origin_path="/data/a.mp4",
                size_bytes=1,
                config_job_path="videos/a_config.json",
                config_provenance="own-sidecar",
                pushed_siblings=["videos/a_config.json"],
                shared=shared,
            )
        ],
        models=[
            JobModel(
                key="obb/x.pt",
                roles=["R"],
                origin_path="/h/x.pt",
                kind="file",
                sha256="ab",
                size_bytes=1,
                sidecars=["obb/x.pt.slice_meta.json"],
            ),
            # Minor fix (round-7): a directory model, so build_push_input_list's
            # models-root-relative `files[]` prefixing is actually exercised --
            # without one, a bug that emitted `files[]` entries WITHOUT the
            # `models/` prefix (rsync would then look for them at the wrong
            # path relative to the push source root and silently omit them)
            # had no test able to catch it.
            JobModel(
                key="pose/SLEAP/run",
                roles=["POSE_MODEL_DIR"],
                origin_path="/h/pose/SLEAP/run",
                kind="directory",
                files=[
                    "pose/SLEAP/run/best.ckpt",
                    "pose/SLEAP/run/training_config.json",
                ],
                file_digests={
                    "pose/SLEAP/run/best.ckpt": "cd",
                    "pose/SLEAP/run/training_config.json": "ef",
                },
            ),
        ],
        config_snapshot={
            "advanced_config": "config/advanced_config.json",
            "skeletons": ["config/skeletons/ant.json"],
        },
    )


def test_parse_remote_splits_host_and_path():
    target = parse_remote("rutalab@firebrat:/home/rutalab/jobs/j1")
    assert target.host == "rutalab@firebrat"
    assert target.path == "/home/rutalab/jobs/j1"


@pytest.mark.parametrize("bad", ["nohost", "host:relative/path", ""])
def test_parse_remote_rejects_malformed_targets(bad):
    with pytest.raises(TrackingJobError):
        parse_remote(bad)


def test_push_list_is_inputs_only():
    entries = set(build_push_input_list(_manifest()))
    assert entries == {
        "hydra_job.json",
        "run.sh",
        "videos.txt",
        "config/advanced_config.json",
        "config/skeletons/ant.json",
        "config/presets/.seeded",
        "config/skeletons/.seeded",
        "models/model_registry.json",
        "models/obb/x.pt",
        "models/obb/x.pt.slice_meta.json",
        # Minor fix (round-7): the directory model's `files[]` -- these are
        # already models-root-relative on JobModel (verified: fix M8's
        # `file_digests["pose/SLEAP/run/best.ckpt"]` example uses the same
        # convention), so build_push_input_list must prefix them with
        # "models/" exactly like every other model entry, never emit them
        # bare (which would make rsync look in the wrong place entirely).
        "models/pose/SLEAP/run/best.ckpt",
        "models/pose/SLEAP/run/training_config.json",
        "videos/a.mp4",
        "videos/a_config.json",
    }


def test_push_list_never_contains_an_output():
    """A re-push after a local pull must not overwrite remote outputs."""
    entries = build_push_input_list(_manifest())
    assert not any("_tracking" in e or ".inference_cache_" in e for e in entries)
    assert not any(e.startswith("logs/") for e in entries)


def test_a_shared_video_is_excluded_from_the_push_list():
    entries = build_push_input_list(
        _manifest(shared={"alias": "labnas", "relpath": "a.mp4"})
    )
    assert "videos/a.mp4" not in entries
    assert "videos/a_config.json" in entries


def test_rsync_argv_is_exact():
    argv = build_rsync_argv("/job/", "host:/remote/", files_from="/tmp/list.txt")
    assert argv == [
        "rsync",
        "-a",
        "--partial",
        "--info=progress2",
        "--files-from=/tmp/list.txt",
        "/job/",
        "host:/remote/",
    ]


def test_push_argv_dereferences_symlinks():
    argv = build_rsync_argv(
        "/job/", "host:/remote/", files_from="/tmp/l.txt", extra=("--copy-links",)
    )
    assert "--copy-links" in argv


def test_rsync_argv_never_deletes():
    argv = build_rsync_argv("/job/", "host:/remote/", files_from="/tmp/l.txt")
    assert "--delete" not in argv


def test_push_reports_the_exact_command_on_failure(tmp_path):
    """Fix V-minor: push_job runs an `ssh ... mkdir -p` remote-parent check
    (see the V-minor "creates the remote parent directory" fix above) BEFORE
    the rsync transfer -- a runner that fails EVERY call would raise on the
    mkdir step, never reaching the actual transfer, and this test would then
    be proving nothing about the transfer-failure path it's named for (the
    same trap the fix-note above documents for the presence check). The fake
    runner here succeeds on any ssh/mkdir call and fails ONLY the rsync
    transfer, so this test actually exercises what its name claims."""
    import subprocess

    from hydra_suite.data.tracking_job.transport import push_job

    def selective_runner(argv, **kwargs):
        if argv and argv[0] == "rsync":
            return subprocess.CompletedProcess(
                argv, 23, stdout="", stderr="rsync: boom"
            )
        return subprocess.CompletedProcess(argv, 0, stdout="", stderr="")

    manifest = _manifest()
    (tmp_path / "hydra_job.json").write_text("{}")
    manifest.write(tmp_path / "hydra_job.json")
    with pytest.raises(TrackingJobError) as excinfo:
        push_job(tmp_path, "host:/remote/j", runner=selective_runner)
    message = str(excinfo.value)
    assert "rsync: boom" in message and "rsync" in message
    assert excinfo.value.code == 4


# Fix W3a: BOTH tests below had two bugs that made them pass vacuously.
# (1) They wrote run.log under the REMOTE fixture path
#     (tmp_path/"remote"/"logs"), but the running-check in pull_job reads
#     job_dir/"logs"/"run.log" where job_dir is the DESTINATION -- the
#     directory pull_job fetches logs/ INTO, not the fake remote source
#     the test constructed. Since nothing ever wrote to
#     tmp_path/"dest"/"logs"/"run.log", the "no runs.jsonl at all" guard's
#     own precondition (run_log.is_file()) was False, so the guard
#     self-skipped via its own "no run.log -> nothing to check" rule --
#     the test's pytest.raises(TrackingJobError) only passed because of
#     whatever OTHER exception the AssertionError runner below produced,
#     never because the running-check itself fired.
# (2) `runner=lambda *a, **k: (_ for _ in ()).throw(AssertionError(...))`
#     fires on EVERY runner call, including the logs/ rsync fetch that
#     MUST happen before the running-check can even run (pull_job fetches
#     logs/ first, per the Task 9 prose above: "fetch logs/ first
#     (including runs.jsonl), then check the run actually finished").
#     That rsync call itself raised AssertionError before the
#     running-check ever executed, so the "raises TrackingJobError"
#     assertion below was again satisfied by an entirely different,
#     unintended code path.
#
# Fixed: the fixture writes DIRECTLY to dest/logs/{run.log,runs.jsonl} (the
# post-fetch state pull_job's own check reads), and the runner returns a
# real `subprocess.CompletedProcess(argv, 0, stdout="", stderr="")` for the
# logs/ rsync call -- mirroring test_push_reports_the_exact_command_on_failure's
# `selective_runner` shape above, just always returning success instead of
# failing selectively -- so the fetch step itself is a harmless no-op and
# the running-check is the ONLY thing that can raise.
import subprocess


def test_pull_refuses_a_job_that_looks_still_running(tmp_path):
    """Fix A5/W3a: run.log newer than the last runs.jsonl entry (or
    runs.jsonl absent while run.log exists) means the run has not finished."""
    from hydra_suite.data.tracking_job.manifest import TrackingJobError
    from hydra_suite.data.tracking_job.transport import pull_job

    dest_logs = tmp_path / "dest" / "logs"
    dest_logs.mkdir(parents=True)
    (dest_logs / "run.log").write_text("still going...\n")
    # No runs.jsonl at all -- the run has not exited yet.

    def logs_only_runner(argv, **kwargs):
        return subprocess.CompletedProcess(argv, 0, stdout="", stderr="")

    with pytest.raises(TrackingJobError) as excinfo:
        pull_job(
            "host:/remote",
            tmp_path / "dest",
            include_caches=True,
            overwrite=False,
            dry_run=False,
            runner=logs_only_runner,
        )
    assert excinfo.value.code == 5


def test_pull_force_bypasses_the_running_check(tmp_path):
    """--force must still allow pulling a genuinely in-progress job on
    purpose (e.g. to inspect partial CSVs while a long run is ongoing).

    Fix Y5 (round-7): the round-6 version of this test only asserted a
    forbidden SUBSTRING was absent from whatever exception (if any) came
    out, catching `Exception` broadly -- so a completely different failure
    (e.g. a bare `FileNotFoundError` from `pull_job` reading a local
    manifest that this fixture never created) would ALSO satisfy the
    assertion, without --force having been exercised at all. Fixed: the
    fixture supplies a real, minimal local manifest (matching fix Y5's
    pinned step order, where the manifest is read only AFTER the
    running-check) so a force=True, dry_run=True pull can run to actual
    completion, and the test asserts NO exception is raised and a
    PullReport comes back -- proving the bypass, not merely the absence of
    one particular error string.
    """
    from hydra_suite.data.tracking_job.manifest import JobManifest
    from hydra_suite.data.tracking_job.transport import PullReport, pull_job

    dest = tmp_path / "dest"
    dest_logs = dest / "logs"
    dest_logs.mkdir(parents=True)
    (dest_logs / "run.log").write_text("still going...\n")
    # A minimal, valid, zero-video local manifest -- enough for pull_job to
    # get past the (post-running-check) manifest read and plan_pull with an
    # empty video set, all the way to the dry-run report.
    manifest = JobManifest(
        job_id="j",
        created_at="t",
        created_on={},
        keystone={},
        videos=[],
        models=[],
    )
    manifest.write(dest / "hydra_job.json")

    def succeeding_runner(argv, **kwargs):
        return subprocess.CompletedProcess(argv, 0, stdout="", stderr="")

    report = pull_job(
        "host:/remote",
        dest,
        include_caches=True,
        overwrite=False,
        dry_run=True,
        force=True,
        runner=succeeding_runner,
    )
    assert isinstance(report, PullReport)


def test_push_creates_the_remote_parent_directory(tmp_path):
    """Fix V-minor: the ssh mkdir -p call must happen before rsync, and it
    must name the PARENT of the remote job path, not the job path itself
    (rsync -a already owns creating the final path component)."""
    from hydra_suite.data.tracking_job.transport import push_job

    calls = []

    def recording_runner(argv, **kwargs):
        calls.append(list(argv))
        return subprocess.CompletedProcess(argv, 0, stdout="", stderr="")

    manifest = _manifest()
    manifest.write(tmp_path / "hydra_job.json")
    push_job(tmp_path, "host:/remote/jobs/j1", runner=recording_runner)

    assert len(calls) == 2
    mkdir_call, rsync_call = calls
    assert mkdir_call[0] == "ssh"
    assert mkdir_call[1] == "host"
    assert "mkdir -p" in mkdir_call[2]
    assert "jobs" in mkdir_call[2] and "j1" not in mkdir_call[2]
    assert rsync_call[0] == "rsync"


def test_pull_logs_fetch_transfers_both_run_log_and_runs_jsonl(tmp_path):
    """Fix Y4: the logs/ fetch must be a plain recursive rsync, not a
    build_rsync_argv(..., files_from=...) call -- otherwise files rsync
    never enumerated ahead of time (like a future artifact type) would
    silently never arrive."""
    import datetime

    from hydra_suite.data.tracking_job.manifest import JobManifest
    from hydra_suite.data.tracking_job.transport import pull_job

    remote_fixture_logs = tmp_path / "remote_fixture" / "logs"
    remote_fixture_logs.mkdir(parents=True)
    (remote_fixture_logs / "run.log").write_text("done\n")
    finished_at = datetime.datetime.now(datetime.timezone.utc).isoformat(
        timespec="microseconds"
    )
    (remote_fixture_logs / "runs.jsonl").write_text(
        '{"finished_at": "%s"}\n' % finished_at
    )

    dest = tmp_path / "dest"
    manifest = JobManifest(
        job_id="j",
        created_at="t",
        created_on={},
        keystone={},
        videos=[],
        models=[],
    )
    manifest.write(dest / "hydra_job.json") if dest.exists() else None

    def copying_runner(argv, **kwargs):
        if (
            argv[0] == "rsync"
            and argv[1] == "-a"
            and str(argv[-1]).rstrip("/").endswith("logs")
        ):
            destination = Path(argv[-1])
            destination.mkdir(parents=True, exist_ok=True)
            for item in remote_fixture_logs.iterdir():
                shutil.copy2(item, destination / item.name)
            return subprocess.CompletedProcess(argv, 0, stdout="", stderr="")
        return subprocess.CompletedProcess(argv, 0, stdout="", stderr="")

    dest.mkdir(parents=True, exist_ok=True)
    manifest.write(dest / "hydra_job.json")

    report = pull_job(
        "host:/remote",
        dest,
        include_caches=True,
        overwrite=False,
        dry_run=True,
        runner=copying_runner,
    )
    assert (dest / "logs" / "run.log").is_file()
    assert (dest / "logs" / "runs.jsonl").is_file()
    assert report.dry_run is True
