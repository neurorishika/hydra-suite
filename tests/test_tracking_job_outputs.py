"""Output discovery is structural; origin mapping puts artifacts back."""

import pytest

from hydra_suite.data.tracking_job.manifest import JobManifest, JobVideo
from hydra_suite.data.tracking_job.outputs import (
    discover_outputs,
    map_outputs_to_origins,
    plan_pull,
)


def _manifest(**video_overrides):
    video = JobVideo(
        job_path="videos/colony.mp4",
        origin_path="/Volumes/lab/2026-09/colony.mp4",
        size_bytes=10,
        config_job_path="videos/colony_config.json",
        config_provenance="own-sidecar",
        pushed_siblings=["videos/colony_config.json"],
        **video_overrides,
    )
    return JobManifest(
        job_id="j",
        created_at="t",
        created_on={},
        keystone={"video": "videos/colony.mp4", "config": "videos/colony_config.json"},
        videos=[video],
        models=[],
    )


# Every artifact the pipeline is known to produce today (spec section 10).
# Fix X4: the original list used invented names ("colony_tracking.csv",
# "colony_tracking_with_individual.csv") that headless_tracking.py never
# writes. Verified real names: with enable_backward_tracking (the fixtures'
# default), headless_tracking.py:221-224 writes raw `<stem>_tracking_forward
# .csv`/`<stem>_tracking_backward.csv`; cli_config.py:325 names the
# post-processing intermediate `<stem>_tracking_forward_processed.csv`; the
# Debug-mode terminal files are `<stem>_tracking_final.csv` (bare) and
# `<stem>_tracking_final_with_individual.csv` (rich export, RICH_EXPORT_SUFFIX
# appended per core/tracking/session.py -- also the name run_matrix.sh:320
# compares); the User-mode terminal file is `<stem>_tracks.csv`
# (core/tracking/session.py:812, `user_tracks_path`). Both terminal-file
# families are listed since a job's mode (User/Debug) is a config choice, not
# a fixed pipeline output.
KNOWN_ARTIFACTS = [
    "videos/colony_tracking_forward.csv",
    "videos/colony_tracking_backward.csv",
    "videos/colony_tracking_forward_processed.csv",
    "videos/colony_tracking_final.csv",
    "videos/colony_tracking_final_with_individual.csv",
    "videos/colony_tracks.csv",
    "videos/colony_tracking.mp4",
    "videos/colony_logs/run.log",
    "videos/.inference_cache_colony/detection.npz",
    "videos/.inference_cache_colony/opt/trial_0.npz",
    "videos/colony_datasets/active_learning/labels.json",
    "videos/colony_datasets/oriented_videos/a.mp4",
    "videos/colony_datasets/individual_crops/0001.png",
]


def test_every_known_artifact_is_discovered():
    listing = ["videos/colony.mp4", "videos/colony_config.json", *KNOWN_ARTIFACTS]
    assert sorted(discover_outputs(_manifest(), listing)) == sorted(KNOWN_ARTIFACTS)


def test_the_video_itself_is_never_an_output():
    assert discover_outputs(_manifest(), ["videos/colony.mp4"]) == []


def test_pushed_siblings_are_never_outputs():
    assert discover_outputs(_manifest(), ["videos/colony_config.json"]) == []


def test_an_unknown_future_artifact_is_still_discovered():
    """Structural discovery means new artifact types need no code change."""
    listing = ["videos/colony.mp4", "videos/colony_somethingnew/report.html"]
    assert discover_outputs(_manifest(), listing) == [
        "videos/colony_somethingnew/report.html"
    ]


def test_outputs_map_beside_the_origin_video():
    outputs = [
        "videos/colony_tracking.csv",
        "videos/.inference_cache_colony/detection.npz",
    ]
    mapped = {
        d.job_relpath: d.destination
        for d in map_outputs_to_origins(_manifest(), outputs)
    }
    assert (
        mapped["videos/colony_tracking.csv"]
        == "/Volumes/lab/2026-09/colony_tracking.csv"
    )
    assert (
        mapped["videos/.inference_cache_colony/detection.npz"]
        == "/Volumes/lab/2026-09/.inference_cache_colony/detection.npz"
    )


def test_a_redirected_output_goes_back_to_its_recorded_absolute_path():
    manifest = _manifest(
        redirected_outputs={"videos/colony_custom.mp4": "/Volumes/renders/custom.mp4"}
    )
    mapped = map_outputs_to_origins(manifest, ["videos/colony_custom.mp4"])
    assert mapped[0].destination == "/Volumes/renders/custom.mp4"
    assert mapped[0].redirected is True


def test_a_shared_video_maps_outputs_to_the_local_mount_of_the_origin():
    """Outputs land beside the original on the share, via origin_path."""
    manifest = _manifest(shared={"alias": "labnas", "relpath": "2026-09/colony.mp4"})
    mapped = map_outputs_to_origins(manifest, ["videos/colony_tracking.csv"])
    assert mapped[0].destination == "/Volumes/lab/2026-09/colony_tracking.csv"


def test_no_caches_excludes_the_inference_cache_tree():
    listing = ["videos/colony.mp4", *KNOWN_ARTIFACTS]
    planned = [
        d.job_relpath for d in plan_pull(_manifest(), listing, include_caches=False)
    ]
    assert not any(".inference_cache_" in p for p in planned)
    assert "videos/colony_tracking_final.csv" in planned


def test_caches_are_included_by_default():
    listing = ["videos/colony.mp4", *KNOWN_ARTIFACTS]
    planned = [d.job_relpath for d in plan_pull(_manifest(), listing)]
    assert "videos/.inference_cache_colony/detection.npz" in planned


def test_an_output_outside_videos_is_ignored():
    listing = ["logs/run.log", "hydra_job.json", "videos/colony_tracking.csv"]
    assert discover_outputs(_manifest(), listing) == ["videos/colony_tracking.csv"]


def test_multiple_videos_route_to_their_own_origins():
    a = JobVideo(
        job_path="videos/a.mp4",
        origin_path="/data/one/a.mp4",
        size_bytes=1,
        config_job_path="videos/a_config.json",
        config_provenance="own-sidecar",
        pushed_siblings=["videos/a_config.json"],
    )
    b = JobVideo(
        job_path="videos/b.mp4",
        origin_path="/data/two/b.mp4",
        size_bytes=1,
        config_job_path="videos/b_config.json",
        config_provenance="keystone-baseline",
        pushed_siblings=["videos/b_config.json"],
    )
    manifest = JobManifest(
        job_id="j",
        created_at="t",
        created_on={},
        keystone={"video": "videos/a.mp4", "config": "videos/a_config.json"},
        videos=[a, b],
        models=[],
    )
    mapped = {
        d.job_relpath: d.destination
        for d in map_outputs_to_origins(
            manifest, ["videos/a_tracking.csv", "videos/b_tracking.csv"]
        )
    }
    assert mapped["videos/a_tracking.csv"] == "/data/one/a_tracking.csv"
    assert mapped["videos/b_tracking.csv"] == "/data/two/b_tracking.csv"


def test_an_unattributable_output_raises():
    """An artifact matching no video's stem must not be silently dropped."""
    from hydra_suite.data.tracking_job.manifest import TrackingJobError

    with pytest.raises(TrackingJobError):
        map_outputs_to_origins(_manifest(), ["videos/unrelated_thing.csv"])


def test_a_stray_ds_store_is_ignored_not_raised():
    listing = ["videos/colony.mp4", "videos/.DS_Store"]
    assert discover_outputs(_manifest(), listing) == []
