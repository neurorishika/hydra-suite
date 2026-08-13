from hydra_suite.utils import video_artifacts as va


def test_inference_cache_dir_next_to_video(tmp_path):
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"x")
    assert va.build_inference_cache_dir(video) == tmp_path / ".inference_cache_clip"


def test_detection_cache_path_is_modern_detection_npz(tmp_path):
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"x")
    p = va.build_detection_cache_path(video, "modelXYZ")
    assert p == tmp_path / ".inference_cache_clip" / "detection.npz"


def test_props_caches_land_alongside_detection_cache(tmp_path):
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"x")
    det = va.build_detection_cache_path(video, "m")
    pose = va.build_individual_properties_cache_path(
        video, "pid", 0, 10, detection_cache_path=str(det)
    )
    assert pose.parent == tmp_path / ".inference_cache_clip"
