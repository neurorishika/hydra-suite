from hydra_suite.utils import video_artifacts as va


def test_dead_builders_removed():
    for name in (
        "build_apriltag_cache_path",
        "find_existing_apriltag_cache_path",
        "build_classify_cache_path",
        "find_existing_classify_cache_path",
        "build_legacy_detection_cache_path",
    ):
        assert not hasattr(va, name), f"{name} should be deleted"
