"""Preflight fails loudly, reporting every problem at once."""

import json

from hydra_suite.data.tracking_job.preflight import preflight_job


def _names(result):
    return {c["name"] for c in result.checks if not c["ok"]}


def test_a_good_job_passes(packed_job):
    result = preflight_job(
        packed_job, available_tiers=("cpu", "gpu"), conda_envs=("sleap",)
    )
    assert result.ok, result.checks


def test_a_missing_conda_env_fails_naming_it(packed_job_needing_sleap):
    result = preflight_job(
        packed_job_needing_sleap, available_tiers=("gpu",), conda_envs=()
    )
    assert not result.ok
    detail = " ".join(c["detail"] for c in result.checks if not c["ok"])
    assert "sleap" in detail


def test_an_unavailable_tier_fails_unless_fallback_is_allowed(packed_job_gpu_tier):
    strict = preflight_job(packed_job_gpu_tier, available_tiers=("cpu",), conda_envs=())
    assert not strict.ok
    assert "runtime_tier" in _names(strict)
    lenient = preflight_job(
        packed_job_gpu_tier,
        available_tiers=("cpu",),
        conda_envs=(),
        allow_tier_fallback=True,
    )
    assert "runtime_tier" not in _names(lenient)


def test_a_tampered_model_fails_the_hash_check(packed_job):
    (packed_job / "models" / "obb" / "x.pt").write_bytes(b"tampered")
    result = preflight_job(packed_job, available_tiers=("cpu",), conda_envs=())
    assert not result.ok
    assert "models" in _names(result)


def test_fast_skips_hashing(packed_job):
    """Fix B-minor: `fast` is only real if it reaches verify_job.

    preflight's FIRST check is `verify_job(job_dir)`, which hashes every model
    and every file_digests member. Calling it without threading `fast` through
    would defeat `fast` entirely -- preflight would skip its own hash check
    while verify quietly did the same work and reported the same tamper under a
    different check name. `preflight_job` must call
    `verify_job(job_dir, fast=fast)` (Task 7 signature), and this test asserts
    the tamper is invisible under BOTH names.
    """
    (packed_job / "models" / "obb" / "x.pt").write_bytes(b"tampered")
    result = preflight_job(
        packed_job, available_tiers=("cpu",), conda_envs=(), fast=True
    )
    failed = _names(result)
    assert "models" not in failed
    assert "verify" not in failed


def test_all_failures_are_reported_not_just_the_first(packed_job_gpu_tier):
    (packed_job_gpu_tier / "models" / "obb" / "x.pt").write_bytes(b"tampered")
    result = preflight_job(packed_job_gpu_tier, available_tiers=("cpu",), conda_envs=())
    assert len(_names(result)) >= 2


def test_a_result_file_is_written(packed_job):
    preflight_job(packed_job, available_tiers=("cpu",), conda_envs=())
    assert (packed_job / "logs" / "preflight.json").is_file()


def test_a_packed_shared_job_verifies_clean(packed_job_shared):
    """Fix V2: a shared video has no file under videos/ at pack time (spec
    §6.7) -- verify_job must exempt it from both the per-model job_path
    existence/size check and the videos.txt line-existence check, or
    pack_job's own self-verify (which every packed_job* fixture, including
    this one, goes through) raises for every shared job -- meaning THIS
    FIXTURE would raise at setup and every shared test in this file would
    error before its body ever ran."""
    from hydra_suite.data.tracking_job.verify import verify_job

    assert verify_job(packed_job_shared) == []


def test_shared_video_is_materialized_as_a_symlink(packed_job_shared, tmp_path):
    # Fix B9: shared.relpath is "colony.mp4" (see packed_job_shared), so the
    # mount copy goes at "<mount>/colony.mp4".
    mount = tmp_path / "mnt" / "lab"
    mount.mkdir(parents=True)
    source = mount / "colony.mp4"
    source.write_bytes(b"\x00" * 2048)
    result = preflight_job(
        packed_job_shared,
        shared_root_overrides={"labnas": str(mount)},
        available_tiers=("cpu",),
        conda_envs=(),
    )
    assert result.ok, result.checks
    link = packed_job_shared / "videos" / "colony.mp4"
    assert link.is_symlink()
    assert link.resolve() == source.resolve()


def test_an_unknown_alias_fails_listing_known_aliases(packed_job_shared):
    result = preflight_job(packed_job_shared, shared_root_overrides={}, conda_envs=())
    assert not result.ok
    detail = " ".join(c["detail"] for c in result.checks if not c["ok"])
    assert "labnas" in detail


def test_a_re_encoded_shared_video_fails_the_signature_check(
    packed_job_shared, tmp_path
):
    """Fix B9: the file MUST EXIST at the resolved path with DIFFERENT bytes.

    Writing it at the wrong path made this pass because the file was missing,
    which is the *previous* test's failure mode -- the signature check itself
    was never exercised and could have been deleted with the suite still green.
    """
    mount = tmp_path / "mnt" / "lab"
    mount.mkdir(parents=True)
    source = mount / "colony.mp4"
    # Fix (minor): SAME size as the staging fixture's video (2048 bytes,
    # b"\x00" * 2048) -- using 4096 here would also trip a size_bytes
    # mismatch, so a failure message containing "signature" would not
    # actually prove the SIGNATURE check (vs. a size check) is what fired.
    # Same size, different content isolates the content-signature check.
    source.write_bytes(b"\xff" * 2048)  # exists; re-encoded => different content
    assert source.is_file()
    result = preflight_job(
        packed_job_shared, shared_root_overrides={"labnas": str(mount)}, conda_envs=()
    )
    assert not result.ok
    assert "shared_roots" in _names(result)
    detail = " ".join(c["detail"] for c in result.checks if not c["ok"])
    assert (
        "signature" in detail.lower()
    ), "must fail on the SIGNATURE, not on a missing file"


def test_a_missing_shared_file_fails_with_both_paths(packed_job_shared, tmp_path):
    mount = tmp_path / "mnt" / "lab"
    mount.mkdir(parents=True)  # empty: "<mount>/colony.mp4" does not exist
    result = preflight_job(
        packed_job_shared, shared_root_overrides={"labnas": str(mount)}, conda_envs=()
    )
    assert not result.ok
    detail = " ".join(c["detail"] for c in result.checks if not c["ok"])
    assert "colony.mp4" in detail and str(mount) in detail


def test_a_truncated_non_shared_video_fails_the_signature_check(packed_job):
    """Fix W1c: this is the actual gap -- a NON-shared, pushed video that
    rsync --partial truncated must fail preflight even though it exists and
    even before verify_job's cheaper size_bytes check would also catch it,
    proving the two checks are independent layers, not one masquerading as
    the other."""
    video = packed_job / "videos" / "colony.mp4"
    real = video.resolve()
    real.write_bytes(real.read_bytes()[:100])
    result = preflight_job(packed_job, available_tiers=("cpu",), conda_envs=())
    assert not result.ok
    assert "video_signature" in _names(result)


def test_a_missing_video_fails_preflight(packed_job):
    """Fix W1d/load_video_list gap: load_video_list only WARNs on a missing
    video and silently continues with a subset batch (trackerkit/app.py,
    the `missing = [...]` block). preflight is what turns that into a hard,
    named failure before track ever runs."""
    (packed_job / "videos" / "colony.mp4").unlink()
    result = preflight_job(packed_job, available_tiers=("cpu",), conda_envs=())
    assert not result.ok


def test_insufficient_disk_fails(packed_job, monkeypatch):
    import shutil

    monkeypatch.setattr(
        shutil, "disk_usage", lambda _p: shutil._ntuple_diskusage(1, 1, 0)
    )
    result = preflight_job(packed_job, available_tiers=("cpu",), conda_envs=())
    assert "disk" in _names(result)


# --- Fix X1b: presence-vs-truthiness of HYDRA_HOST_CONFIG_DIR, the three
# distinct real invocation shapes. ---


def test_preflight_reads_the_host_shared_roots_table_not_the_job_snapshot(
    packed_job_shared, tmp_path, monkeypatch
):
    job_cfg = tmp_path / "job_snapshot_config"
    job_cfg.mkdir()
    host_cfg = tmp_path / "real_host_config"
    host_cfg.mkdir()
    (host_cfg / "shared_roots.json").write_text(
        json.dumps({"labnas": str(tmp_path / "unused")})
    )
    monkeypatch.setenv("HYDRA_CONFIG_DIR", str(job_cfg))
    monkeypatch.setenv("HYDRA_HOST_CONFIG_DIR", str(host_cfg))

    mount = tmp_path / "mnt" / "lab"
    mount.mkdir(parents=True)
    (mount / "colony.mp4").write_bytes(b"\x00" * 2048)

    result = preflight_job(
        packed_job_shared,
        shared_root_overrides={"labnas": str(mount)},
        available_tiers=("cpu",),
        conda_envs=(),
    )
    assert "shared_roots" not in _names(result)


def test_preflight_run_sh_empty_host_config_dir_reads_platformdirs_default(
    packed_job_shared, tmp_path, monkeypatch
):
    job_cfg = tmp_path / "job_snapshot" / "config"
    job_cfg.mkdir(parents=True)
    monkeypatch.setenv("HYDRA_CONFIG_DIR", str(job_cfg))
    # run.sh's own signal: HYDRA_HOST_CONFIG_DIR present, but empty.
    monkeypatch.setenv("HYDRA_HOST_CONFIG_DIR", "")

    platformdirs_default = tmp_path / "platformdirs_default"
    platformdirs_default.mkdir()
    mount = tmp_path / "mnt" / "lab"
    mount.mkdir(parents=True)
    (mount / "colony.mp4").write_bytes(b"\x00" * 2048)
    (platformdirs_default / "shared_roots.json").write_text(
        json.dumps({"labnas": str(mount)})
    )

    import hydra_suite.data.tracking_job.preflight as preflight_module

    monkeypatch.setattr(
        preflight_module, "get_platform_config_dir", lambda: platformdirs_default
    )

    result = preflight_job(packed_job_shared, available_tiers=("cpu",), conda_envs=())
    assert "shared_roots" not in _names(result)


def test_preflight_direct_invocation_no_host_config_dir_reads_config_dir(
    packed_job_shared, tmp_path, monkeypatch
):
    monkeypatch.delenv("HYDRA_HOST_CONFIG_DIR", raising=False)
    cfg = tmp_path / "direct_config"
    cfg.mkdir()
    mount = tmp_path / "mnt" / "lab"
    mount.mkdir(parents=True)
    (mount / "colony.mp4").write_bytes(b"\x00" * 2048)
    (cfg / "shared_roots.json").write_text(json.dumps({"labnas": str(mount)}))
    monkeypatch.setenv("HYDRA_CONFIG_DIR", str(cfg))

    result = preflight_job(packed_job_shared, available_tiers=("cpu",), conda_envs=())
    assert "shared_roots" not in _names(result)
