"""verify_job is an offline, mount-agnostic integrity check."""

import json

from hydra_suite.data.tracking_job.verify import verify_job


def test_a_freshly_packed_job_verifies(packed_job):
    assert verify_job(packed_job) == []


def test_a_corrupted_model_is_detected(packed_job):
    (packed_job / "models" / "obb" / "x.pt").write_bytes(b"tampered")
    problems = verify_job(packed_job)
    assert any("obb/x.pt" in p for p in problems)


def test_a_missing_model_is_detected(packed_job):
    (packed_job / "models" / "obb" / "x.pt").unlink()
    assert any("obb/x.pt" in p for p in verify_job(packed_job))


def test_an_absolute_path_in_a_sidecar_is_detected(packed_job):
    sidecar = packed_job / "videos" / "colony_config.json"
    payload = json.loads(sidecar.read_text())
    payload["csv_path"] = "/Users/someone/out.csv"
    sidecar.write_text(json.dumps(payload))
    assert any("csv_path" in p for p in verify_job(packed_job))


def test_a_missing_skeleton_is_detected(packed_job):
    (packed_job / "config" / "skeletons" / "ant.json").unlink()
    assert any("skeleton" in p.lower() for p in verify_job(packed_job))


def test_videos_txt_keystone_mismatch_is_detected(packed_job):
    (packed_job / "videos.txt").write_text("videos/other.mp4\n")
    assert verify_job(packed_job)


def test_a_truncated_video_is_detected(packed_job):
    """Fix W1b: rsync --partial leaves a truncated file at the destination
    path on interruption -- it EXISTS, so the pre-fix existence-only check
    passed. Truncating in place (same path, fewer bytes) reproduces exactly
    that failure mode without needing an actual push."""
    video = packed_job / "videos" / "colony.mp4"
    real = video.resolve()  # colony.mp4 is a symlink by default (copy_videos=False)
    real.write_bytes(real.read_bytes()[:100])
    problems = verify_job(packed_job)
    assert any("colony.mp4" in p and "size" in p.lower() for p in problems)


def test_all_problems_are_reported_not_just_the_first(packed_job):
    (packed_job / "models" / "obb" / "x.pt").unlink()
    (packed_job / "config" / "skeletons" / "ant.json").unlink()
    assert len(verify_job(packed_job)) >= 2
