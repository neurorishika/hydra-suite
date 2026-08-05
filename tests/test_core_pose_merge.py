import pandas as pd

from hydra_suite.core.post.pose_merge import (
    PoseSourceState,
    check_pose_export_sources,
    resolve_current_tag_cache_path,
)


def test_check_sources_empty_state_reports_nothing():
    (
        has_other,
        cache_path,
        cache_ok,
        interp_path,
        interp_ok,
        interp_mem,
        interp_mem_ok,
    ) = check_pose_export_sources(PoseSourceState())
    assert has_other is False
    assert cache_ok is False
    assert interp_ok is False
    assert interp_mem_ok is False


def test_check_sources_detects_in_memory_pose_df():
    state = PoseSourceState(interpolated_pose_df=pd.DataFrame({"X": [1.0]}))
    result = check_pose_export_sources(state)
    assert result[6] is True  # interp_mem_available


def test_resolve_tag_cache_returns_empty_without_apriltags():
    assert resolve_current_tag_cache_path({"USE_APRILTAGS": False}, "/nope.npz") == ""
