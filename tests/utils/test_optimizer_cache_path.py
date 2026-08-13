from hydra_suite.utils import video_artifacts as va


def test_optimizer_cache_under_inference_cache_opt(tmp_path):
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"x")
    p = va.build_optimizer_detection_cache_path(video, "modelA", 100)
    assert p == tmp_path / ".inference_cache_clip" / "opt"
    assert "_caches" not in str(p) and "r100" not in str(p)
