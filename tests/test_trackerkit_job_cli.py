"""The job subcommand group: parsing, rejection, dispatch."""

import pytest

from hydra_suite.core.inference.config import DEFAULT_CALIBRATION_BUDGET_SECONDS
from hydra_suite.trackerkit.app import build_parser, parse_arguments


def test_job_pack_parses_like_track():
    args = parse_arguments(["job", "pack", "/tmp/j", "--video-list", "list.txt"])
    assert args.command == "job"
    assert args.job_command == "pack"
    assert args.video_list == "list.txt"


def test_job_pack_accepts_explicit_videos():
    args = parse_arguments(["job", "pack", "/tmp/j", "a.mp4", "b.mp4"])
    assert args.videos == ["a.mp4", "b.mp4"]


def test_job_pack_rejects_both_videos_and_video_list():
    with pytest.raises(SystemExit):
        parse_arguments(["job", "pack", "/tmp/j", "a.mp4", "--video-list", "l.txt"])


def test_job_pack_rejects_neither():
    with pytest.raises(SystemExit):
        parse_arguments(["job", "pack", "/tmp/j"])


@pytest.mark.parametrize("flag", ["--gpus", "--jobs", "--threads-per-job"])
def test_job_pack_rejects_compute_box_flags(flag):
    """They describe the compute box, not the experiment (calibrate precedent)."""
    with pytest.raises(SystemExit):
        parse_arguments(["job", "pack", "/tmp/j", "a.mp4", flag, "2"])


@pytest.mark.parametrize("flag", ["--sahi-profile", "--inference-autotune-manual"])
def test_job_run_rejects_experiment_flags(flag):
    """Those are fixed at pack time and always forwarded from track_args."""
    with pytest.raises(SystemExit):
        parse_arguments(["job", "run", "/tmp/j", flag, "x"])


def test_job_run_accepts_compute_box_flags():
    args = parse_arguments(["job", "run", "/tmp/j", "--gpus", "auto", "--jobs", "2"])
    assert args.gpus == "auto"
    assert args.jobs == 2


def test_job_pack_rejects_conflicting_shared_modes():
    with pytest.raises(SystemExit):
        parse_arguments(
            ["job", "pack", "/tmp/j", "a.mp4", "--no-shared", "--shared-only"]
        )


def test_every_documented_subcommand_exists():
    parser = build_parser()
    job_action = next(
        a for a in parser._subparsers._group_actions if "job" in a.choices
    )
    assert set(job_action.choices["job"]._subparsers._group_actions[0].choices) >= {
        "pack",
        "verify",
        "shared-root",
        "push",
        "preflight",
        "run",
        "calibrate",
        "status",
        "pull",
    }


def test_shared_root_add_parses_alias_and_path():
    args = parse_arguments(["job", "shared-root", "add", "labnas", "/Volumes/lab"])
    assert args.alias == "labnas"
    assert args.path == "/Volumes/lab"


def test_pull_flags_parse():
    args = parse_arguments(
        [
            "job",
            "pull",
            "host:/r/j",
            "/tmp/j",
            "--dry-run",
            "--no-caches",
            "--overwrite",
        ]
    )
    assert args.dry_run and args.no_caches and args.overwrite
    assert args.force is False  # fix A5's still-running guard is ON by default


def test_pull_force_flag_parses():
    args = parse_arguments(["job", "pull", "host:/r/j", "/tmp/j", "--force"])
    assert args.force is True


def test_shared_root_override_parses_as_alias_equals_path():
    args = parse_arguments(
        ["job", "preflight", "/tmp/j", "--shared-root", "labnas=/mnt/lab"]
    )
    assert args.shared_root == ["labnas=/mnt/lab"]


def test_exit_codes_are_mapped(monkeypatch):
    """TrackingJobError.code becomes the process exit code."""
    from hydra_suite.data.tracking_job.manifest import TrackingJobError
    from hydra_suite.trackerkit import job_cli

    def boom(_args):
        raise TrackingJobError("nope", code=3)

    monkeypatch.setattr(job_cli, "_dispatch", boom)
    assert job_cli.run_job_cli(object()) == 3


def test_record_run_parser_is_hidden_but_callable():
    parser = build_parser()
    assert "_record-run" not in parser.format_help()
    args = build_parser().parse_args(
        [
            "job",
            "_record-run",
            "--started",
            "x",
            "--exit-code",
            "0",
            "--",
            "--gpus",
            "auto",
        ]
    )
    assert args.job_command == "_record-run"


def test_job_status_parses():
    args = parse_arguments(["job", "status", "/tmp/j"])
    assert args.job_command == "status"


def test_job_run_calibrate_flag_parses():
    args = parse_arguments(["job", "run", "/tmp/j", "--calibrate"])
    assert args.calibrate is True


def test_job_calibrate_accepts_a_remote_target():
    args = parse_arguments(["job", "calibrate", "host:/remote/j"])
    assert args.target == "host:/remote/j"


def test_job_run_budget_seconds_parses():
    args = parse_arguments(["job", "run", "/tmp/j", "--budget-seconds", "3600"])
    assert args.budget_seconds == 3600


def test_job_run_remote_bootstrap_defaults_to_empty():
    """Fix A2b: default must be '' -- an empty bootstrap fails loudly against a
    box where trackerkit is not on a bare ssh PATH, rather than silently
    mis-scheduling. See the firebrat value documented in fix A2b."""
    args = parse_arguments(["job", "run", "host:/remote/j"])
    assert args.remote_bootstrap == ""


def test_job_run_remote_bootstrap_parses():
    args = parse_arguments(
        [
            "job",
            "run",
            "host:/remote/j",
            "--remote-bootstrap",
            "source ~/miniforge3/etc/profile.d/conda.sh && conda activate hydra-cuda",
        ]
    )
    assert "conda activate hydra-cuda" in args.remote_bootstrap


def test_job_calibrate_budget_seconds_defaults_when_omitted():
    args = parse_arguments(["job", "calibrate", "/tmp/j"])
    assert args.budget_seconds == DEFAULT_CALIBRATION_BUDGET_SECONDS


def test_job_run_forwards_shared_root_and_sets_skip_preflight(monkeypatch, tmp_path):
    from hydra_suite.data.tracking_job.manifest import JobManifest, JobVideo
    from hydra_suite.trackerkit import job_cli

    job_dir = tmp_path / "job"
    job_dir.mkdir()
    manifest = JobManifest(
        job_id="abc",
        created_at="2026-01-01T00:00:00Z",
        created_on={},
        keystone={"video": "videos/a.mp4"},
        videos=[
            JobVideo(
                job_path="videos/a.mp4",
                origin_path="/tmp/a.mp4",
                size_bytes=1,
                config_job_path="videos/a_config.json",
                config_provenance="explicit",
            )
        ],
        models=[],
    )
    manifest.write(job_dir / "hydra_job.json")

    captured = {}

    def fake_preflight(job_dir_arg, **kwargs):
        captured["preflight_kwargs"] = kwargs
        from hydra_suite.data.tracking_job.preflight import PreflightResult

        return PreflightResult(ok=True, checks=[])

    def fake_run(argv, **kwargs):
        captured["env"] = kwargs.get("env")
        import subprocess as _subprocess

        return _subprocess.CompletedProcess(argv, 0)

    monkeypatch.setattr(job_cli, "preflight_job", fake_preflight)
    monkeypatch.setattr(job_cli.subprocess, "run", fake_run)

    args = parse_arguments(
        [
            "job",
            "run",
            str(job_dir),
            "--shared-root",
            "labnas=/mnt/lab",
            "--allow-tier-fallback",
        ]
    )
    job_cli._cmd_run(args)

    assert captured["preflight_kwargs"]["shared_root_overrides"] == {
        "labnas": "/mnt/lab"
    }
    assert captured["preflight_kwargs"]["allow_tier_fallback"] is True
    assert captured["env"]["HYDRA_JOB_SKIP_PREFLIGHT"] == "1"


def test_job_run_local_sets_hydra_job_trackerkit(monkeypatch, tmp_path):
    import sys

    from hydra_suite.data.tracking_job.manifest import JobManifest, JobVideo
    from hydra_suite.data.tracking_job.preflight import PreflightResult
    from hydra_suite.trackerkit import job_cli

    job_dir = tmp_path / "job2"
    job_dir.mkdir()
    manifest = JobManifest(
        job_id="abc",
        created_at="2026-01-01T00:00:00Z",
        created_on={},
        keystone={"video": "videos/a.mp4"},
        videos=[
            JobVideo(
                job_path="videos/a.mp4",
                origin_path="/tmp/a.mp4",
                size_bytes=1,
                config_job_path="videos/a_config.json",
                config_provenance="explicit",
            )
        ],
        models=[],
    )
    manifest.write(job_dir / "hydra_job.json")

    monkeypatch.setattr(
        job_cli, "preflight_job", lambda *a, **k: PreflightResult(True, [])
    )

    captured = {}

    def fake_run(argv, **kwargs):
        captured["env"] = kwargs.get("env")
        import subprocess as _subprocess

        return _subprocess.CompletedProcess(argv, 0)

    monkeypatch.setattr(job_cli.subprocess, "run", fake_run)

    args = parse_arguments(["job", "run", str(job_dir)])
    job_cli._cmd_run(args)

    env = captured["env"]
    assert sys.executable in env["HYDRA_JOB_TRACKERKIT"]
    assert "hydra_suite.trackerkit.app" in env["HYDRA_JOB_TRACKERKIT"]


def test_job_run_remote_command_includes_shared_root_flag_and_skip_preflight(
    monkeypatch,
):
    from hydra_suite.trackerkit import job_cli

    captured = {}

    def fake_run(argv, **kwargs):
        captured["argv"] = argv
        import subprocess as _subprocess

        return _subprocess.CompletedProcess(argv, 0)

    monkeypatch.setattr(job_cli.subprocess, "run", fake_run)

    args = parse_arguments(
        [
            "job",
            "run",
            "host:/remote/j",
            "--shared-root",
            "labnas=/mnt/lab",
        ]
    )
    job_cli._cmd_run(args)

    remote_cmd = captured["argv"][-1]
    assert "--shared-root labnas=/mnt/lab" in remote_cmd
    assert "HYDRA_JOB_SKIP_PREFLIGHT=1" in remote_cmd
    assert remote_cmd.index("--shared-root labnas=/mnt/lab") < remote_cmd.index(
        "HYDRA_JOB_SKIP_PREFLIGHT=1"
    )
