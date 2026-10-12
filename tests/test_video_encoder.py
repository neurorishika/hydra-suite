import sys
from unittest import mock

import numpy as np
import pytest

# ── probe_video_backend ────────────────────────────────────────────────────────


def test_probe_returns_known_backend():
    """probe_video_backend() always returns one of the four known backend strings."""
    import hydra_suite.utils.video_encoder as ve

    ve._BACKEND_CACHE = None
    backend = ve.probe_video_backend()
    assert backend in ("nvenc", "videotoolbox", "pyav_software", "opencv")


def test_probe_result_is_cached():
    """Calling probe_video_backend() twice returns the same value."""
    import hydra_suite.utils.video_encoder as ve

    ve._BACKEND_CACHE = None
    b1 = ve.probe_video_backend()
    b2 = ve.probe_video_backend()
    assert b1 == b2


def test_probe_falls_back_to_opencv_when_av_unavailable():
    """_probe_backend() returns 'opencv' when av cannot be imported."""
    import hydra_suite.utils.video_encoder as ve

    saved = sys.modules.get("av")
    try:
        sys.modules["av"] = None  # type: ignore[assignment]
        ve._BACKEND_CACHE = None
        result = ve._probe_backend()
        assert result == "opencv"
    finally:
        ve._BACKEND_CACHE = None  # prevent cache state leaking to other tests
        if saved is None:
            sys.modules.pop("av", None)
        else:
            sys.modules["av"] = saved


# ── VideoEncoder write + release ──────────────────────────────────────────────


def test_video_encoder_opencv_creates_nonempty_file(tmp_path):
    """VideoEncoder(backend='opencv') writes a valid, non-empty MP4."""
    from hydra_suite.utils.video_encoder import VideoEncoder

    path = tmp_path / "out.mp4"
    enc = VideoEncoder(path, fps=30.0, width=64, height=64, backend="opencv")
    frame = np.zeros((64, 64, 3), dtype=np.uint8)
    enc.write(frame)
    enc.write(frame)
    enc.release()
    assert path.exists()
    assert path.stat().st_size > 100


def test_video_encoder_context_manager(tmp_path):
    """Context manager calls release() on __exit__ and the file is written."""
    from hydra_suite.utils.video_encoder import VideoEncoder

    path = tmp_path / "ctx.mp4"
    with VideoEncoder(path, fps=25.0, width=32, height=32, backend="opencv") as enc:
        enc.write(np.zeros((32, 32, 3), dtype=np.uint8))
    assert path.exists()


def test_video_encoder_double_release_is_safe(tmp_path):
    """Calling release() twice must not raise."""
    from hydra_suite.utils.video_encoder import VideoEncoder

    path = tmp_path / "safe.mp4"
    enc = VideoEncoder(path, fps=10.0, width=16, height=16, backend="opencv")
    enc.release()
    enc.release()  # must not raise


def test_video_encoder_auto_backend_writes_file(tmp_path):
    """VideoEncoder with auto backend selection produces a readable file."""
    from hydra_suite.utils.video_encoder import VideoEncoder

    path = tmp_path / "auto.mp4"
    with VideoEncoder(path, fps=10.0, width=64, height=64) as enc:
        for _ in range(5):
            enc.write(np.zeros((64, 64, 3), dtype=np.uint8))
    assert path.exists()
    assert path.stat().st_size > 0


def test_video_encoder_pyav_write_uses_bgr_input(monkeypatch: pytest.MonkeyPatch):
    """Odd-sized frames (no I420 fast path) ingest BGR directly, no channel reversal."""
    import hydra_suite.utils.video_encoder as ve

    class _FakeFrame:
        def reformat(self, *, format):
            assert format == "yuv420p"
            return self

    calls = []

    class _FakeVideoFrame:
        @staticmethod
        def from_ndarray(arr, format):
            calls.append((arr.shape, arr.flags.c_contiguous, format))
            return _FakeFrame()

    monkeypatch.setitem(sys.modules, "av", mock.Mock(VideoFrame=_FakeVideoFrame))

    enc = ve.VideoEncoder.__new__(ve.VideoEncoder)
    enc._stream = mock.Mock()
    enc._stream.encode.return_value = [object()]
    enc._container = mock.Mock()
    enc._cv_writer = None

    frame = np.zeros((33, 33, 3), dtype=np.uint8)
    enc.write(frame)

    assert calls == [((33, 33, 3), True, "bgr24")]
    enc._container.mux.assert_called_once()


# ── NVENC session cap ─────────────────────────────────────────────────────────


def test_nvenc_cap_exceeded_falls_back_to_software(tmp_path):
    """When _NVENC_ACTIVE >= _NVENC_MAX, VideoEncoder silently downgrades from nvenc."""
    import hydra_suite.utils.video_encoder as ve

    saved_max, saved_active = ve._NVENC_MAX, ve._NVENC_ACTIVE
    try:
        ve._NVENC_MAX = 0  # cap at zero: any nvenc request must fall back
        ve._NVENC_ACTIVE = 0
        enc = ve.VideoEncoder(
            tmp_path / "capped.mp4",
            fps=10.0,
            width=16,
            height=16,
            backend="nvenc",
        )
        # Must have been downgraded; backend is no longer "nvenc"
        assert enc._backend != "nvenc"
        enc.release()
    finally:
        ve._NVENC_MAX = saved_max
        ve._NVENC_ACTIVE = saved_active


def test_nvenc_session_counter_decrements_on_release(tmp_path):
    """Releasing an NVENC encoder decrements _NVENC_ACTIVE."""
    import hydra_suite.utils.video_encoder as ve

    saved_cache = ve._BACKEND_CACHE
    saved_max, saved_active = ve._NVENC_MAX, ve._NVENC_ACTIVE
    try:
        ve._NVENC_MAX = 10
        ve._NVENC_ACTIVE = 0

        with mock.patch.object(ve.VideoEncoder, "_open", lambda self: None):
            enc = ve.VideoEncoder.__new__(ve.VideoEncoder)
            enc._backend = "nvenc"
            enc._used_nvenc = False
            enc._container = None
            enc._stream = None
            enc._cv_writer = None
            # Manually simulate what _open does for nvenc
            ve._NVENC_ACTIVE += 1
            enc._used_nvenc = True

            before = ve._NVENC_ACTIVE
            enc.release()
            assert ve._NVENC_ACTIVE == before - 1
    finally:
        ve._BACKEND_CACHE = saved_cache
        ve._NVENC_MAX = saved_max
        ve._NVENC_ACTIVE = saved_active


def test_try_encode_probe_frame_meets_nvenc_minimum(monkeypatch: pytest.MonkeyPatch):
    """The probe clip must be large enough for NVENC to open.

    NVENC rejects tiny frames in avcodec_open2 (measured on an RTX 6000 Ada:
    h264_nvenc needs > 144 px, hevc_nvenc > 128 px), so a too-small probe
    reports NVENC unavailable on every NVIDIA box and silently drops the
    whole render to libx264 (~10x slower on 4512x4512 frames).
    """
    import hydra_suite.utils.video_encoder as ve

    seen = {}

    class _Stream:
        def encode(self, frame=None):
            if frame is not None:
                seen["frame"] = (frame.width, frame.height)
            return []

    class _Container:
        def add_stream(self, codec_name, rate):
            stream = _Stream()
            seen["stream"] = stream
            return stream

        def mux(self, pkt):
            pass

        def close(self):
            pass

    import av

    monkeypatch.setattr(av, "open", lambda *a, **k: _Container())
    assert ve._try_encode("h264_nvenc") is True
    width, height = seen["stream"].width, seen["stream"].height
    assert (width, height) == seen["frame"]
    assert min(width, height) >= 256
    assert width % 16 == 0 and height % 16 == 0


# ── PyAV frame hand-off ───────────────────────────────────────────────────────


def _round_trip_mean_bgr(tmp_path, width, height, bgr):
    av = pytest.importorskip("av")
    from hydra_suite.utils.video_encoder import VideoEncoder

    path = tmp_path / "rt.mp4"
    frame = np.empty((height, width, 3), dtype=np.uint8)
    frame[:] = bgr
    with VideoEncoder(
        path, fps=10.0, width=width, height=height, backend="pyav_software"
    ) as enc:
        assert enc._backend == "pyav_software"
        for _ in range(3):
            enc.write(frame)
    with av.open(str(path)) as container:
        decoded = [f.to_ndarray(format="bgr24") for f in container.decode(video=0)]
    assert len(decoded) == 3
    return decoded[-1].reshape(-1, 3).mean(axis=0)


@pytest.mark.parametrize("width,height", [(64, 48), (65, 49)])
def test_pyav_write_preserves_bgr_colour(tmp_path, width, height):
    """Both hand-off paths (even I420 fast path, odd fallback) keep BGR order."""
    pytest.importorskip("av")
    import av

    if "libx264" not in av.codecs_available:
        pytest.skip("libx264 not available")
    mean = _round_trip_mean_bgr(tmp_path, width, height, (200, 50, 10))
    assert np.allclose(mean, (200, 50, 10), atol=6), mean


def test_pyav_write_even_frame_skips_bgr24_conversion(tmp_path, monkeypatch):
    """Even-sized frames must not go through PyAV's bgr24 ingest + swscale.

    That path copies the frame row by row under the GIL (~67 ms per 4512x4512
    frame) and was the render bottleneck once NVENC was in use; cv2's I420
    conversion plus a plane memcpy is ~7 ms.
    """
    av = pytest.importorskip("av")
    if "libx264" not in av.codecs_available:
        pytest.skip("libx264 not available")
    from hydra_suite.utils.video_encoder import VideoEncoder

    formats = []
    real_cls = av.VideoFrame

    class _SpyVideoFrame:
        # av.VideoFrame is an immutable extension type, so spy via a proxy.
        def __call__(self, *args, **kwargs):
            return real_cls(*args, **kwargs)

        def __getattr__(self, name):
            return getattr(real_cls, name)

        @staticmethod
        def from_ndarray(arr, format, **kw):
            formats.append(format)
            return real_cls.from_ndarray(arr, format=format, **kw)

    monkeypatch.setattr(av, "VideoFrame", _SpyVideoFrame())
    with VideoEncoder(
        tmp_path / "fast.mp4", fps=10.0, width=64, height=48, backend="pyav_software"
    ) as enc:
        enc.write(np.zeros((48, 64, 3), dtype=np.uint8))
    assert "bgr24" not in formats


@pytest.mark.parametrize(
    "fps,expected",
    [
        (29.97002997002997, "30000/1001"),
        (59.94005994005994, "60000/1001"),
        (25.0, "25"),
        (100.0, "100"),
        (12.5, "25/2"),
    ],
)
def test_encoder_rate_keeps_fractional_fps(fps, expected):
    from fractions import Fraction

    from hydra_suite.utils.video_encoder import _encoder_rate

    assert _encoder_rate(fps) == Fraction(expected)
    assert _encoder_rate(Fraction(30000, 1001)) == Fraction(30000, 1001)
