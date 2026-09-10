"""Shared-root mount table: alias in, host path out."""

import pytest

from hydra_suite.data.tracking_job import shared_roots
from hydra_suite.data.tracking_job.manifest import TrackingJobError


@pytest.fixture()
def config_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("HYDRA_CONFIG_DIR", str(tmp_path))
    return tmp_path


def test_missing_table_is_an_empty_mapping(config_dir):
    assert shared_roots.load_shared_roots() == {}


def test_save_then_load_round_trip(config_dir):
    shared_roots.save_shared_roots({"labnas": "/Volumes/lab"})
    assert shared_roots.load_shared_roots() == {"labnas": "/Volumes/lab"}


def test_match_returns_alias_and_relpath(config_dir):
    table = {"labnas": "/Volumes/lab"}
    assert shared_roots.match_shared_root("/Volumes/lab/2026-09/a.mp4", table) == (
        "labnas",
        "2026-09/a.mp4",
    )


def test_longest_root_wins(config_dir):
    table = {"lab": "/Volumes/lab", "project": "/Volumes/lab/2026-09"}
    alias, rel = shared_roots.match_shared_root("/Volumes/lab/2026-09/a.mp4", table)
    assert alias == "project"
    assert rel == "a.mp4"


def test_no_match_returns_none(config_dir):
    assert (
        shared_roots.match_shared_root("/Users/me/a.mp4", {"lab": "/Volumes/lab"})
        is None
    )


def test_partial_path_component_is_not_a_match(config_dir):
    """/Volumes/lab must not match /Volumes/labour."""
    assert (
        shared_roots.match_shared_root("/Volumes/labour/a.mp4", {"lab": "/Volumes/lab"})
        is None
    )


def test_resolve_unknown_alias_lists_the_known_ones(config_dir):
    with pytest.raises(TrackingJobError) as excinfo:
        shared_roots.resolve_shared("nope", "a.mp4", {"labnas": "/Volumes/lab"})
    message = str(excinfo.value)
    assert "nope" in message and "labnas" in message


def test_resolve_returns_the_host_path(config_dir):
    resolved = shared_roots.resolve_shared(
        "labnas", "2026-09/a.mp4", {"labnas": "/mnt/lab"}
    )
    assert str(resolved) == "/mnt/lab/2026-09/a.mp4"


def test_resolve_rejects_a_traversing_relpath(config_dir):
    with pytest.raises(TrackingJobError):
        shared_roots.resolve_shared(
            "labnas", "../../etc/passwd", {"labnas": "/mnt/lab"}
        )
