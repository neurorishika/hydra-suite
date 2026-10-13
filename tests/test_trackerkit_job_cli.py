"""The job subcommand group: parsing, rejection, dispatch."""

import argparse

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


@pytest.mark.parametrize(
    "flag", ["--gpus", "--jobs", "--threads-per-job", "--cpu-budget"]
)
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
    args = parse_arguments(["job", "run", "/tmp/j", "--cpu-budget", "8"])
    assert args.cpu_budget == 8


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


def test_shared_root_remove_is_registered():
    """Spec section 13 lists `shared-root add | remove | list`; remove was missing.

    Found in practice trying to undo a test alias: `job shared-root remove` did
    not exist, so an alias could be created but never deleted through the CLI.
    """
    args = parse_arguments(["job", "shared-root", "remove", "labnas"])
    assert args.job_command == "shared-root"
    assert args.shared_root_command == "remove"
    assert args.alias == "labnas"


def test_shared_root_remove_preserves_other_aliases(tmp_path, monkeypatch):
    from hydra_suite.data.tracking_job.shared_roots import (
        load_shared_roots,
        save_shared_roots,
    )

    monkeypatch.setenv("HYDRA_CONFIG_DIR", str(tmp_path))
    save_shared_roots({"keep": "/mnt/keep", "drop": "/mnt/drop"})

    from hydra_suite.trackerkit import job_cli

    ns = argparse.Namespace(
        job_command="shared-root", shared_root_command="remove", alias="drop"
    )
    assert job_cli._cmd_shared_root(ns) == 0
    assert load_shared_roots() == {"keep": "/mnt/keep"}


def test_shared_root_remove_unknown_alias_names_the_known_ones(tmp_path, monkeypatch):
    from hydra_suite.data.tracking_job.manifest import TrackingJobError
    from hydra_suite.data.tracking_job.shared_roots import save_shared_roots

    monkeypatch.setenv("HYDRA_CONFIG_DIR", str(tmp_path))
    save_shared_roots({"keep": "/mnt/keep"})

    from hydra_suite.trackerkit import job_cli

    ns = argparse.Namespace(
        job_command="shared-root", shared_root_command="remove", alias="ghost"
    )
    with pytest.raises(TrackingJobError) as excinfo:
        job_cli._cmd_shared_root(ns)
    assert "ghost" in str(excinfo.value) and "keep" in str(excinfo.value)


_FANOUT_ARGV = ["--gpus", "0,1", "--jobs", "2", "--threads-per-job", "3"]
_FANOUT_ARGV += ["--cpu-budget", "8"]


def _write_min_job(job_dir):
    from hydra_suite.data.tracking_job.manifest import JobManifest, JobVideo

    job_dir.mkdir()
    JobManifest(
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
    ).write(job_dir / "hydra_job.json")


def test_fanout_flags_skip_unset_values():
    from hydra_suite.trackerkit import job_cli

    assert job_cli._fanout_flags(parse_arguments(["job", "run", "/tmp/j"])) == []
    args = parse_arguments(["job", "run", "/tmp/j", *_FANOUT_ARGV])
    assert job_cli._fanout_flags(args) == _FANOUT_ARGV


def test_job_run_local_forwards_fanout_flags(monkeypatch, tmp_path):
    from hydra_suite.data.tracking_job.preflight import PreflightResult
    from hydra_suite.trackerkit import job_cli

    job_dir = tmp_path / "job3"
    _write_min_job(job_dir)
    monkeypatch.setattr(
        job_cli, "preflight_job", lambda *a, **k: PreflightResult(True, [])
    )
    captured = {}

    def fake_run(argv, **kwargs):
        captured["argv"] = argv
        import subprocess as _subprocess

        return _subprocess.CompletedProcess(argv, 0)

    monkeypatch.setattr(job_cli.subprocess, "run", fake_run)
    job_cli._cmd_run(parse_arguments(["job", "run", str(job_dir), *_FANOUT_ARGV]))
    assert captured["argv"][0] == "./run.sh"
    assert captured["argv"][-len(_FANOUT_ARGV) :] == _FANOUT_ARGV


def test_job_run_local_detach_forwards_fanout_flags(monkeypatch, tmp_path):
    from hydra_suite.data.tracking_job.preflight import PreflightResult
    from hydra_suite.trackerkit import job_cli

    job_dir = tmp_path / "job4"
    _write_min_job(job_dir)
    monkeypatch.setattr(
        job_cli, "preflight_job", lambda *a, **k: PreflightResult(True, [])
    )
    captured = {}
    monkeypatch.setattr(
        job_cli.subprocess,
        "Popen",
        lambda argv, **kw: captured.setdefault("argv", argv),
    )
    job_cli._cmd_run(
        parse_arguments(["job", "run", str(job_dir), "--detach", *_FANOUT_ARGV])
    )
    assert captured["argv"][-len(_FANOUT_ARGV) :] == _FANOUT_ARGV


@pytest.mark.parametrize("detach", [False, True])
def test_job_run_remote_forwards_fanout_flags(monkeypatch, detach):
    import shlex

    from hydra_suite.trackerkit import job_cli

    captured = {}

    def fake_run(argv, **kwargs):
        captured["argv"] = argv
        import subprocess as _subprocess

        return _subprocess.CompletedProcess(argv, 0)

    monkeypatch.setattr(job_cli.subprocess, "run", fake_run)
    extra = ["--detach"] if detach else []
    job_cli._cmd_run(
        parse_arguments(["job", "run", "host:/remote/j", *extra, *_FANOUT_ARGV])
    )
    remote_cmd = captured["argv"][-1]
    assert f"./run.sh {shlex.join(_FANOUT_ARGV)}" in remote_cmd


def test_job_run_rejects_non_positive_cpu_budget():
    with pytest.raises(SystemExit):
        parse_arguments(["job", "run", "/tmp/j", "--cpu-budget", "0"])


@pytest.mark.parametrize("flag", ["--jobs", "--threads-per-job"])
def test_job_run_rejects_non_positive_fanout_counts(flag, capsys):
    with pytest.raises(SystemExit):
        parse_arguments(["job", "run", "/tmp/j", flag, "0"])
    assert "job run" in capsys.readouterr().err


def test_job_calibrate_accepts_and_validates_cpu_budget(capsys):
    args = parse_arguments(["job", "calibrate", "/tmp/j", "--cpu-budget", "6"])
    assert args.cpu_budget == 6
    with pytest.raises(SystemExit):
        parse_arguments(["job", "calibrate", "/tmp/j", "--cpu-budget", "0"])
    assert "job calibrate" in capsys.readouterr().err


@pytest.mark.parametrize("command", ["run", "calibrate"])
def test_local_calibrate_leg_forwards_cpu_budget(monkeypatch, tmp_path, command):
    from hydra_suite.trackerkit import job_cli

    job_dir = tmp_path / f"job_{command}"
    _write_min_job(job_dir)
    captured = []

    def fake_run(argv, **kwargs):
        captured.append(argv)
        import subprocess as _subprocess

        return _subprocess.CompletedProcess(argv, 1)  # stop after calibrate

    monkeypatch.setattr(job_cli.subprocess, "run", fake_run)
    argv = ["job", command, str(job_dir), "--cpu-budget", "5"]
    if command == "run":
        argv.append("--calibrate")
    args = parse_arguments(argv)
    (job_cli._cmd_run if command == "run" else job_cli._cmd_calibrate)(args)
    calibrate_argv = captured[0]
    assert "calibrate" in calibrate_argv
    assert calibrate_argv[-2:] == ["--cpu-budget", "5"]


def test_local_calibrate_leg_omits_cpu_budget_when_unset(monkeypatch, tmp_path):
    from hydra_suite.trackerkit import job_cli

    job_dir = tmp_path / "job_plain"
    _write_min_job(job_dir)
    captured = []

    def fake_run(argv, **kwargs):
        captured.append(argv)
        import subprocess as _subprocess

        return _subprocess.CompletedProcess(argv, 1)

    monkeypatch.setattr(job_cli.subprocess, "run", fake_run)
    job_cli._cmd_calibrate(parse_arguments(["job", "calibrate", str(job_dir)]))
    assert "--cpu-budget" not in captured[0]


def test_remote_calibrate_leg_forwards_cpu_budget(monkeypatch):
    from hydra_suite.trackerkit import job_cli

    captured = []

    def fake_run(argv, **kwargs):
        captured.append(argv)
        import subprocess as _subprocess

        return _subprocess.CompletedProcess(argv, 1)

    monkeypatch.setattr(job_cli.subprocess, "run", fake_run)
    job_cli._cmd_run(
        parse_arguments(
            ["job", "run", "host:/remote/j", "--calibrate", "--cpu-budget", "7"]
        )
    )
    remote_cmd = captured[0][-1]
    assert "job calibrate . --cpu-budget 7" in remote_cmd
