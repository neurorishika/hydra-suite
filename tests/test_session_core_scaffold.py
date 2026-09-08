import pytest

from hydra_suite.core.tracking.session import SessionCallbacks, TrackingSessionCore


def test_callbacks_defaults_are_silent_noops():
    cb = SessionCallbacks()
    assert cb.progress(50, "half") is None
    assert cb.status("working") is None
    assert cb.warning("Title", "Message") is None
    assert cb.stage_changed("merge") is None
    assert cb.should_stop() is False


def test_core_constructs_keyword_only_and_stores_state():
    core = TrackingSessionCore(
        video_path="/v.mp4",
        config={"enable_postprocessing": True},
        params={"FPS": 30.0},
        paths={"raw_csv_path": "/out.csv"},
    )
    assert core.video_path == "/v.mp4"
    assert core.config["enable_postprocessing"] is True
    assert core.params["FPS"] == 30.0
    assert core.paths["raw_csv_path"] == "/out.csv"
    assert isinstance(core.callbacks, SessionCallbacks)


def test_core_requires_keyword_arguments():
    with pytest.raises(TypeError):
        TrackingSessionCore("/v.mp4", {}, {}, {})  # positional not allowed


def test_pose_state_defaults_to_the_derived_production_cache_dir():
    """No override in ``paths`` -> unchanged production behaviour (S8)."""
    from hydra_suite.utils.video_artifacts import build_inference_cache_dir

    core = TrackingSessionCore(
        video_path="/v.mp4",
        config={},
        params={},
        paths={"raw_csv_path": "/out.csv"},
    )
    assert core.pose_state.inference_cache_dir == str(
        build_inference_cache_dir("/v.mp4")
    )


def test_pose_state_honors_an_explicit_inference_cache_dir_override(tmp_path):
    """A caller-supplied cache dir (e.g. a trial's private cache) wins (S8)."""
    private_cache = tmp_path / "run-root" / "inference-cache"
    core = TrackingSessionCore(
        video_path="/v.mp4",
        config={},
        params={},
        paths={
            "raw_csv_path": "/out.csv",
            "inference_cache_dir": str(private_cache),
        },
    )
    assert core.pose_state.inference_cache_dir == str(private_cache)
