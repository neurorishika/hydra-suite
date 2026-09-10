"""Job manifest: versioned, atomic, traversal-safe."""

import json

import pytest

from hydra_suite.data.tracking_job.manifest import (
    SUPPORTED_JOB_VERSION,
    JobManifest,
    JobModel,
    JobVideo,
    TrackingJobError,
    validate_job_relpath,
)


def _manifest():
    return JobManifest(
        job_id="2026-09-09T14-03-12_test",
        created_at="2026-09-09T14:03:12Z",
        created_on={"hostname": "mbp", "platform": "darwin"},
        keystone={"video": "videos/a.mp4", "config": "videos/a_config.json"},
        videos=[
            JobVideo(
                job_path="videos/a.mp4",
                origin_path="/Volumes/lab/a.mp4",
                size_bytes=10,
                config_job_path="videos/a_config.json",
                config_provenance="own-sidecar",
                pushed_siblings=["videos/a_config.json"],
            )
        ],
        models=[
            JobModel(
                key="obb/x.pt",
                roles=["YOLO_OBB_DIRECT_MODEL_PATH"],
                origin_path="/host/models/obb/x.pt",
                kind="file",
                sha256="ab",
                size_bytes=3,
            )
        ],
    )


def test_round_trip_is_lossless(tmp_path):
    path = tmp_path / "hydra_job.json"
    original = _manifest()
    original.write(path)
    assert JobManifest.read(path).to_dict() == original.to_dict()


def test_job_version_is_emitted_first_and_defaults_to_one():
    payload = _manifest().to_dict()
    assert next(iter(payload)) == "job_version"
    assert payload["job_version"] == SUPPORTED_JOB_VERSION


def test_unsupported_version_raises(tmp_path):
    path = tmp_path / "hydra_job.json"
    payload = _manifest().to_dict()
    payload["job_version"] = 99
    path.write_text(json.dumps(payload))
    with pytest.raises(TrackingJobError) as excinfo:
        JobManifest.read(path)
    assert "99" in str(excinfo.value)


def test_write_is_atomic_leaving_no_temp_files(tmp_path):
    path = tmp_path / "hydra_job.json"
    _manifest().write(path)
    assert [p.name for p in tmp_path.iterdir()] == ["hydra_job.json"]


@pytest.mark.parametrize(
    "bad", ["/abs/path", "../escape", "videos/../../etc/passwd", ""]
)
def test_validate_job_relpath_rejects_unsafe(bad):
    with pytest.raises(TrackingJobError):
        validate_job_relpath(bad)


@pytest.mark.parametrize(
    "good", ["videos/a.mp4", "models/obb/x.pt", "config/skeletons/s.json"]
)
def test_validate_job_relpath_accepts_safe(good):
    assert str(validate_job_relpath(good)) == good


def test_manifest_read_validates_every_relpath(tmp_path):
    path = tmp_path / "hydra_job.json"
    payload = _manifest().to_dict()
    payload["videos"][0]["job_path"] = "/etc/passwd"
    path.write_text(json.dumps(payload))
    with pytest.raises(TrackingJobError):
        JobManifest.read(path)


def test_error_carries_a_code():
    err = TrackingJobError("boom", code=3)
    assert err.code == 3
