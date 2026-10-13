import pytest

from hydra_suite.runtime import cpu_budget as cb


@pytest.fixture
def cap(monkeypatch):
    def _set(n):
        monkeypatch.setattr(cb, "_cap", n)

    return _set


def test_session_options_none_when_uncapped():
    from hydra_suite.runtime.onnx_providers import session_options_for

    assert session_options_for(None) is None


def test_session_options_sets_intra_op():
    pytest.importorskip("onnxruntime")
    from hydra_suite.runtime.onnx_providers import session_options_for

    assert session_options_for(3).intra_op_num_threads == 3


def test_crop_warp_threads_respects_cap(cap, monkeypatch):
    from hydra_suite.core.inference.stages import crops

    monkeypatch.delenv("HYDRA_CROP_WARP_THREADS", raising=False)
    cap(None)
    uncapped = crops._crop_warp_threads()
    cap(2)
    assert crops._crop_warp_threads() == min(uncapped, 2)
    monkeypatch.setenv("HYDRA_CROP_WARP_THREADS", "6")
    assert crops._crop_warp_threads() == 2


def test_apriltag_threads_capped(cap):
    from hydra_suite.runtime.cpu_budget import apriltag_thread_count

    cap(None)
    assert apriltag_thread_count({"APRILTAG_THREADS": 4}) == 4
    assert apriltag_thread_count({}) == 4
    cap(2)
    assert apriltag_thread_count({"APRILTAG_THREADS": 4}) == 2
    assert apriltag_thread_count({}) == 2


def test_encoder_thread_count_capped(cap, tmp_path):
    pytest.importorskip("av")
    import numpy as np

    from hydra_suite.utils.video_encoder import VideoEncoder

    cap(2)
    enc = VideoEncoder(
        tmp_path / "x.mp4", fps=10, width=64, height=64, backend="pyav_software"
    )
    try:
        enc.write(np.zeros((64, 64, 3), np.uint8))
        assert enc._stream.thread_count == 2
    finally:
        enc.release()


def test_encoder_thread_count_untouched_when_uncapped(cap, tmp_path):
    pytest.importorskip("av")
    from hydra_suite.utils.video_encoder import VideoEncoder

    cap(None)
    enc = VideoEncoder(
        tmp_path / "y.mp4", fps=10, width=64, height=64, backend="pyav_software"
    )
    try:
        assert enc._stream.thread_count == 0  # PyAV default (codec decides)
    finally:
        enc.release()


def test_cv2_capture_opens_fewer_threads_when_capped():
    """OPENCV_FFMPEG_THREADS (set by the cap) shrinks cv2's decoder pool."""
    import subprocess
    import sys
    from pathlib import Path

    psutil = pytest.importorskip("psutil")  # noqa: F841
    clip = (
        Path(__file__).resolve().parents[1]
        / "tools/equivalence/fixtures/clips/fly_obb.mp4"
    )
    if not clip.exists():
        pytest.skip("fixture clip not present")
    code = (
        "import os,sys,cv2,psutil\n"
        "if sys.argv[2]=='capped': os.environ['OPENCV_FFMPEG_THREADS']='2'\n"
        "c=cv2.VideoCapture(sys.argv[1],cv2.CAP_FFMPEG); c.read()\n"
        "print(psutil.Process().num_threads())\n"
    )

    def run(mode):
        out = subprocess.run(
            [sys.executable, "-c", code, str(clip), mode],
            capture_output=True,
            text=True,
            check=True,
        )
        return int(out.stdout.strip().splitlines()[-1])

    assert run("capped") < run("uncapped")
